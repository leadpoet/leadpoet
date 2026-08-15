#!/usr/bin/env python3
"""Run the disposable PostgreSQL/PostgREST boundary for physical parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (  # noqa: E402
    ProductionParityError,
    file_sha256,
    migration_delta,
    validate_archive,
    validate_snapshot_manifest,
    verify_contract_checkout,
)
from leadpoet_canonical.production_parity_epoch_authority import (  # noqa: E402
    ProductionParityEpochAuthorityError,
    install_base as epoch_authority_install_base,
    validate_installed as validate_installed_epoch_authority,
)


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


class DatabaseHostError(RuntimeError):
    pass


TABLE_RE = re.compile(r"^research_lab_[a-z0-9_]{1,96}$")


def _run(
    command: Sequence[str],
    *,
    timeout: int,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require(result: subprocess.CompletedProcess[bytes], *, stage: str) -> bytes:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()[-800:]
        raise DatabaseHostError(f"{stage} failed: {detail}")
    return result.stdout


def _secret(client: Any, secret_id: str) -> dict[str, str]:
    value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    if not isinstance(value, str):
        raise DatabaseHostError("database secret has no string value")
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise DatabaseHostError("database secret is not JSON") from exc
    required = {"AUTHENTICATOR_PASSWORD", "JWT_SECRET", "POSTGRES_PASSWORD"}
    if not isinstance(parsed, Mapping) or required != set(parsed):
        raise DatabaseHostError("database secret fields differ")
    result = {key: str(parsed[key]) for key in required}
    if any(len(value) < 24 or "\n" in value for value in result.values()):
        raise DatabaseHostError("database secret value is invalid")
    return result


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def _resource_names(run_id: str) -> dict[str, str]:
    prefix = f"leadpoet-parity-{run_id}"
    return {
        "network": prefix,
        "postgres": prefix + "-postgres",
        "postgrest": prefix + "-postgrest",
    }


def _docker_absent(kind: str, name: str) -> bool:
    command = ["docker", kind, "inspect", name]
    return _run(command, timeout=20).returncode != 0


def _cleanup(names: Mapping[str, str], runtime_root: Path) -> dict[str, Any]:
    for name in (names["postgrest"], names["postgres"]):
        _run(["docker", "rm", "-f", name], timeout=60)
    _run(["docker", "network", "rm", names["network"]], timeout=30)
    remaining = [
        f"container:{name}"
        for name in (names["postgrest"], names["postgres"])
        if not _docker_absent("container", name)
    ]
    if not _docker_absent("network", names["network"]):
        remaining.append("network:" + names["network"])
    shutil.rmtree(runtime_root, ignore_errors=True)
    if remaining:
        raise DatabaseHostError("database resources remain: " + ",".join(remaining))
    return {"resources_removed": True, "runtime_root_removed": not runtime_root.exists()}


def _postgres_sql(
    *,
    container: str,
    database_name: str,
    sql: str,
    timeout: int = 120,
) -> str:
    return _require(
        _run(
            [
                "docker",
                "exec",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                database_name,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                sql,
            ],
            timeout=timeout,
        ),
        stage="testnet epoch authority database verification",
    ).decode("utf-8", "replace").strip()


def _authority_toc_entries(
    listing: str, *, tables: Sequence[str]
) -> list[str]:
    selected: dict[str, str] = {}
    expected = set(tables)
    for raw_line in listing.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        match = re.match(
            r"^\d+;\s+\d+\s+\d+\s+TABLE DATA\s+public\s+([^\s]+)\s+.*$",
            line,
        )
        if match is None:
            continue
        table = match.group(1)
        if table in expected:
            if table in selected:
                raise DatabaseHostError(
                    "testnet epoch authority dump has duplicate table data"
                )
            selected[table] = raw_line
    missing = sorted(expected - set(selected))
    if missing:
        raise DatabaseHostError(
            "testnet epoch authority dump is missing table data: "
            + ",".join(missing)
        )
    return [raw for raw in listing.splitlines() if raw.strip() and any(
        raw == selected[table] for table in selected
    )]


def _overlay_epoch_authority(
    *,
    container: str,
    database_name: str,
    runtime_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    tables = [str(value) for value in authority.get("database_tables", [])]
    if (
        not tables
        or len(tables) != len(set(tables))
        or any(not TABLE_RE.fullmatch(table) for table in tables)
    ):
        raise DatabaseHostError("testnet epoch authority table scope is invalid")
    quoted_values = ",".join("'" + table + "'" for table in sorted(tables))
    closure_sql = f"""
WITH RECURSIVE authority_tables(oid) AS (
    SELECT c.oid
    FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname IN ({quoted_values})
), dependent_tables(oid) AS (
    SELECT oid FROM authority_tables
    UNION
    SELECT constraint_row.conrelid
    FROM pg_catalog.pg_constraint constraint_row
    JOIN dependent_tables parent ON parent.oid = constraint_row.confrelid
    WHERE constraint_row.contype = 'f'
)
SELECT COALESCE(json_agg(c.relname ORDER BY c.relname), '[]'::json)::text
FROM dependent_tables dependency
JOIN pg_catalog.pg_class c ON c.oid = dependency.oid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public';
"""
    try:
        closure = json.loads(
            _postgres_sql(
                container=container,
                database_name=database_name,
                sql=closure_sql,
            )
        )
    except ValueError as exc:
        raise DatabaseHostError(
            "testnet epoch authority dependency closure is invalid"
        ) from exc
    if not isinstance(closure, list):
        raise DatabaseHostError(
            "testnet epoch authority dependency closure is invalid"
        )
    missing = sorted(set(tables) - set(closure))
    if missing:
        raise DatabaseHostError(
            "testnet epoch authority table scope differs from FK closure: "
            f"missing={','.join(missing)}"
        )
    candidate_only_dependents = sorted(set(closure) - set(tables))
    listing = _require(
        _run(
            [
                "docker",
                "exec",
                container,
                "pg_restore",
                "--list",
                "/parity-authority/authority.dump",
            ],
            timeout=120,
        ),
        stage="testnet epoch authority archive listing",
    ).decode("utf-8", "replace")
    use_list = _authority_toc_entries(listing, tables=tables)
    use_list_path = runtime_root / "authority.list"
    _write_private(use_list_path, "\n".join(use_list) + "\n")
    qualified = ",".join(f'public."{table}"' for table in sorted(tables))
    _postgres_sql(
        container=container,
        database_name=database_name,
        sql=f"TRUNCATE TABLE {qualified} CASCADE;",
    )
    _require(
        _run(
            [
                "docker",
                "exec",
                container,
                "pg_restore",
                "-U",
                "postgres",
                "-d",
                database_name,
                "--data-only",
                "--disable-triggers",
                "--single-transaction",
                "--exit-on-error",
                "--use-list=/parity-runtime/authority.list",
                "/parity-authority/authority.dump",
            ],
            timeout=900,
        ),
        stage="testnet epoch authority restore",
    )
    observed_counts: dict[str, int] = {}
    for table in sorted(tables):
        value = _postgres_sql(
            container=container,
            database_name=database_name,
            sql=f'SELECT COUNT(*) FROM public."{table}";',
        )
        try:
            observed_counts[table] = int(value)
        except ValueError as exc:
            raise DatabaseHostError(
                f"testnet epoch authority row count is invalid: {table}"
            ) from exc
    expected_counts = {
        str(key): int(value)
        for key, value in dict(authority.get("database_row_counts") or {}).items()
    }
    if observed_counts != expected_counts:
        raise DatabaseHostError("testnet epoch authority row counts differ")
    candidate_only_counts: dict[str, int] = {}
    for table in candidate_only_dependents:
        value = _postgres_sql(
            container=container,
            database_name=database_name,
            sql=f'SELECT COUNT(*) FROM public."{table}";',
        )
        try:
            candidate_only_counts[table] = int(value)
        except ValueError as exc:
            raise DatabaseHostError(
                f"testnet epoch authority dependent row count is invalid: {table}"
            ) from exc
    if any(candidate_only_counts.values()):
        raise DatabaseHostError(
            "candidate-only epoch authority dependents were not cleared"
        )
    state_raw = _postgres_sql(
        container=container,
        database_name=database_name,
        sql="""
SELECT row_to_json(state_row)::text
FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 state_row;
""",
    )
    try:
        state = json.loads(state_raw)
    except ValueError as exc:
        raise DatabaseHostError("testnet epoch authority state is invalid") from exc
    if (
        not isinstance(state, Mapping)
        or state.get("lifecycle_state") != "stateful_active"
        or state.get("mapping_hash") != authority.get("mapping_hash")
        or state.get("network_genesis_hash")
        != authority.get("network_genesis_hash")
        or state.get("netuid") != authority.get("netuid")
    ):
        raise DatabaseHostError("testnet epoch authority state differs")
    cutover_count = _postgres_sql(
        container=container,
        database_name=database_name,
        sql=(
            "SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 "
            f"WHERE mapping_hash = '{authority['mapping_hash']}' "
            f"AND network_genesis_hash = '{authority['network_genesis_hash']}' "
            f"AND netuid = {int(authority['netuid'])};"
        ),
    )
    if cutover_count != "1":
        raise DatabaseHostError("testnet epoch cutover authority row differs")
    use_list_path.unlink(missing_ok=True)
    return {
        "mapping_hash": authority["mapping_hash"],
        "network_genesis_hash": authority["network_genesis_hash"],
        "netuid": authority["netuid"],
        "table_count": len(tables),
        "row_counts": observed_counts,
        "candidate_only_dependents": candidate_only_counts,
        "ceremony_evidence_hash": authority["ceremony_evidence_hash"],
    }


def start(
    *,
    run_id: str,
    candidate_sha: str,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    epoch_authority_root: Path,
    database_secret_id: str,
    region: str,
    database_domain: str,
    postgres_image: str,
    postgrest_image: str,
) -> dict[str, Any]:
    if not RUN_RE.fullmatch(run_id) or not SHA_RE.fullmatch(candidate_sha):
        raise DatabaseHostError("database run identity is invalid")
    if not DOMAIN_RE.fullmatch(database_domain):
        raise DatabaseHostError("database TLS domain is invalid")
    for image in (postgres_image, postgrest_image):
        if not IMAGE_RE.fullmatch(image):
            raise DatabaseHostError("database boundary image is not digest-pinned")
    expected_authority_root = epoch_authority_install_base(run_id)
    try:
        if epoch_authority_root.resolve() != expected_authority_root.resolve():
            raise DatabaseHostError("testnet epoch authority path differs")
        epoch_authority = validate_installed_epoch_authority(
            epoch_authority_root
        )
    except (OSError, ProductionParityEpochAuthorityError) as exc:
        raise DatabaseHostError(
            "installed testnet epoch authority is invalid"
        ) from exc
    contract = verify_contract_checkout(
        ROOT, json.loads(contract_path.read_text(encoding="utf-8"))
    )
    manifest = validate_snapshot_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    validate_archive(archive_path, manifest)
    if contract["candidate_sha"] != candidate_sha:
        raise DatabaseHostError("database candidate differs from the contract")
    delta = migration_delta(
        snapshot_migrations=manifest["migrations"],
        candidate_migrations=contract["migrations"],
    )
    names = _resource_names(run_id)
    runtime_root = Path("/run/leadpoet-production-parity") / run_id
    if runtime_root.exists():
        raise DatabaseHostError("database run directory already exists")
    runtime_root.mkdir(parents=True, mode=0o700)
    secrets_doc = _secret(
        boto3.client("secretsmanager", region_name=region), database_secret_id
    )
    postgres_env = runtime_root / "postgres.env"
    postgrest_env = runtime_root / "postgrest.env"
    database_name = "leadpoet_parity_" + run_id.replace("-", "_")
    _write_private(
        postgres_env,
        f"POSTGRES_DB={database_name}\nPOSTGRES_PASSWORD={secrets_doc['POSTGRES_PASSWORD']}\n",
    )
    _write_private(
        postgrest_env,
        "\n".join(
            [
                "PGRST_DB_ANON_ROLE=anon",
                "PGRST_DB_SCHEMAS=public",
                (
                    "PGRST_DB_URI=postgres://authenticator:"
                    + secrets_doc["AUTHENTICATOR_PASSWORD"]
                    + f"@{names['postgres']}:5432/{database_name}"
                ),
                "PGRST_JWT_SECRET=" + secrets_doc["JWT_SECRET"],
                "",
            ]
        ),
    )
    service_key = _jwt_for_role(secrets_doc["JWT_SECRET"], "service_role")
    labels = [
        "--label",
        f"leadpoet.parity.run={run_id}",
        "--label",
        f"leadpoet.candidate.sha={candidate_sha}",
    ]
    try:
        _require(
            _run(["docker", "network", "create", names["network"]], timeout=30),
            stage="database network creation",
        )
        _require(
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    names["postgres"],
                    "--network",
                    names["network"],
                    "--env-file",
                    str(postgres_env),
                    "-v",
                    f"{runtime_root}:/parity-runtime",
                    "-v",
                    f"{epoch_authority_root}:/parity-authority:ro",
                    *labels,
                    postgres_image,
                    "postgres",
                    "-c",
                    "max_connections=300",
                ],
                timeout=60,
            ),
            stage="database PostgreSQL start",
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if _run(
                ["docker", "exec", names["postgres"], "pg_isready", "-U", "postgres"],
                timeout=10,
            ).returncode == 0:
                break
            time.sleep(2)
        else:
            raise DatabaseHostError("database PostgreSQL did not become ready")
        with archive_path.open("rb") as archive:
            restore = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    names["postgres"],
                    "pg_restore",
                    "-U",
                    "postgres",
                    "-d",
                    database_name,
                    "--clean",
                    "--if-exists",
                    "--no-owner",
                    "--no-acl",
                    "--exit-on-error",
                ],
                cwd=ROOT,
                stdin=archive,
                capture_output=True,
                check=False,
                timeout=1800,
            )
        _require(restore, stage="production snapshot restore")
        for migration in delta:
            path = ROOT / str(migration["path"])
            if not path.is_file() or file_sha256(path) != migration["sha256"]:
                raise DatabaseHostError(f"candidate migration bytes differ: {migration['path']}")
            _require(
                _run(
                    [
                        "docker",
                        "exec",
                        "-i",
                        names["postgres"],
                        "psql",
                        "-U",
                        "postgres",
                        "-d",
                        database_name,
                        "-v",
                        "ON_ERROR_STOP=1",
                    ],
                    timeout=600,
                    stdin=path.read_bytes(),
                ),
                stage=f"candidate migration {migration['path']}",
            )
        epoch_authority_evidence = _overlay_epoch_authority(
            container=names["postgres"],
            database_name=database_name,
            runtime_root=runtime_root,
            authority=epoch_authority,
        )
        auth_password = secrets_doc["AUTHENTICATOR_PASSWORD"].replace("'", "''")
        role_sql = f"""
DO $$ BEGIN CREATE ROLE authenticator LOGIN PASSWORD '{auth_password}';
EXCEPTION WHEN duplicate_object THEN ALTER ROLE authenticator WITH LOGIN PASSWORD '{auth_password}'; END $$;
DO $$ BEGIN CREATE ROLE anon NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE service_role NOLOGIN BYPASSRLS; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
GRANT anon, service_role TO authenticator;
GRANT USAGE ON SCHEMA public TO anon, service_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO service_role;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO service_role;
"""
        _require(
            _run(
                [
                    "docker",
                    "exec",
                    "-i",
                    names["postgres"],
                    "psql",
                    "-U",
                    "postgres",
                    "-d",
                    database_name,
                    "-v",
                    "ON_ERROR_STOP=1",
                ],
                timeout=120,
                stdin=role_sql.encode("utf-8"),
            ),
            stage="PostgREST role provisioning",
        )
        _require(
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    names["postgrest"],
                    "--network",
                    names["network"],
                    "-p",
                    "3000:3000",
                    "--env-file",
                    str(postgrest_env),
                    *labels,
                    postgrest_image,
                ],
                timeout=60,
            ),
            stage="PostgREST start",
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                request = Request(
                    "http://127.0.0.1:3000/",
                    headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                )
                with urlopen(request, timeout=10) as response:
                    if 200 <= int(response.status) < 300:
                        break
            except Exception:
                time.sleep(3)
        else:
            raise DatabaseHostError("database PostgREST endpoint did not become ready")
        postgres_env.unlink(missing_ok=True)
        postgrest_env.unlink(missing_ok=True)
        return {
            "candidate_sha": candidate_sha,
            "run_id": run_id,
            "manifest_hash": manifest["manifest_hash"],
            "migration_delta_count": len(delta),
            "epoch_authority": epoch_authority_evidence,
            "database_domain": database_domain,
            "resource_names": names,
            "status": "ready",
        }
    except Exception:
        _cleanup(names, runtime_root)
        raise


def stop(*, run_id: str) -> dict[str, Any]:
    if not RUN_RE.fullmatch(run_id):
        raise DatabaseHostError("database run identity is invalid")
    return _cleanup(
        _resource_names(run_id),
        Path("/run/leadpoet-production-parity") / run_id,
    )


def _jwt_for_role(secret: str, role: str) -> str:
    import base64
    import hashlib
    import hmac

    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    now = int(time.time())
    header = encode(b'{"alg":"HS256","typ":"JWT"}')
    payload = encode(
        json.dumps(
            {
                "aud": "authenticated",
                "exp": now + 86400 * 30,
                "iat": now - 5,
                "iss": "leadpoet-production-parity",
                "role": role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = encode(hmac.new(secret.encode("ascii"), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--run-id", required=True)
    start_parser.add_argument("--candidate-sha", required=True)
    start_parser.add_argument("--contract", type=Path, required=True)
    start_parser.add_argument("--manifest", type=Path, required=True)
    start_parser.add_argument("--archive", type=Path, required=True)
    start_parser.add_argument("--epoch-authority-root", type=Path, required=True)
    start_parser.add_argument("--database-secret-id", required=True)
    start_parser.add_argument("--region", required=True)
    start_parser.add_argument("--database-domain", required=True)
    start_parser.add_argument("--postgres-image", required=True)
    start_parser.add_argument("--postgrest-image", required=True)
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            result = start(
                run_id=args.run_id,
                candidate_sha=str(args.candidate_sha).lower(),
                contract_path=args.contract,
                manifest_path=args.manifest,
                archive_path=args.archive,
                epoch_authority_root=args.epoch_authority_root,
                database_secret_id=args.database_secret_id,
                region=args.region,
                database_domain=args.database_domain,
                postgres_image=args.postgres_image,
                postgrest_image=args.postgrest_image,
            )
        else:
            result = stop(run_id=args.run_id)
    except (
        OSError,
        ValueError,
        BotoCoreError,
        ClientError,
        ProductionParityError,
        DatabaseHostError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
