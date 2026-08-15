#!/usr/bin/env python3
"""Build one immutable testnet epoch-authority artifact from a frozen ceremony."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover  # noqa: E402
from leadpoet_canonical.production_parity import canonical_bytes  # noqa: E402
from leadpoet_canonical.production_parity_epoch_authority import (  # noqa: E402
    CEREMONY_SCHEMA_VERSION,
    REQUIRED_NONEMPTY_TABLES,
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    TABLE_RE,
    ProductionParityEpochAuthorityError,
    sha256_bytes,
    validate_archive,
)


DEFAULT_DSN_ENV = "LEADPOET_PARITY_TESTNET_AUTHORITY_DSN"


def _run(
    command: Sequence[str], *, env: Mapping[str, str], timeout: int
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env),
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require(result: subprocess.CompletedProcess[bytes], *, stage: str) -> bytes:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace")[-800:]
        raise ProductionParityEpochAuthorityError(f"{stage} failed: {detail.strip()}")
    return result.stdout


def _database_env(dsn_env: str) -> dict[str, str]:
    if dsn_env in {"PGAPPNAME", "PGDATABASE", "PGOPTIONS"}:
        raise ProductionParityEpochAuthorityError(
            "testnet authority DSN environment name is reserved"
        )
    dsn = os.environ.get(dsn_env, "").strip()
    if not dsn or "\n" in dsn:
        raise ProductionParityEpochAuthorityError(
            "testnet authority database credential is unavailable"
        )
    env = os.environ.copy()
    env["PGDATABASE"] = dsn
    env["PGAPPNAME"] = "leadpoet-production-parity-authority-builder"
    env.pop(dsn_env, None)
    return env


def _psql_json(sql: str, *, env: Mapping[str, str]) -> Any:
    raw = _require(
        _run(
            [
                "psql",
                "-X",
                "-A",
                "-t",
                "-v",
                "ON_ERROR_STOP=1",
                "--no-password",
                "-c",
                "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY; "
                + sql
                + "; COMMIT;",
            ],
            env=env,
            timeout=180,
        ),
        stage="testnet authority read-only query",
    ).decode("utf-8", "strict").strip()
    lines = [line for line in raw.splitlines() if line not in {"BEGIN", "COMMIT"}]
    try:
        return json.loads("\n".join(lines))
    except ValueError as exc:
        raise ProductionParityEpochAuthorityError(
            "testnet authority query result is invalid"
        ) from exc


def _authority_tables(*, env: Mapping[str, str]) -> list[str]:
    roots = ",".join("'" + table + "'" for table in sorted(REQUIRED_TABLES))
    value = _psql_json(
        f"""
WITH RECURSIVE roots(oid) AS (
    SELECT c.oid
    FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname IN ({roots})
), dependency_closure(oid) AS (
    SELECT oid FROM roots
    UNION
    SELECT fk.conrelid
    FROM pg_catalog.pg_constraint fk
    JOIN dependency_closure parent ON parent.oid = fk.confrelid
    WHERE fk.contype = 'f'
)
SELECT COALESCE(json_agg(c.relname ORDER BY c.relname), '[]'::json)
FROM dependency_closure dependency
JOIN pg_catalog.pg_class c ON c.oid = dependency.oid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
""",
        env=env,
    )
    if (
        not isinstance(value, list)
        or len(value) != len(set(value))
        or any(not TABLE_RE.fullmatch(str(table or "")) for table in value)
        or not REQUIRED_TABLES.issubset(set(value))
    ):
        raise ProductionParityEpochAuthorityError(
            "testnet authority database table closure is invalid"
        )
    return [str(table) for table in value]


def _row_counts(tables: Sequence[str], *, env: Mapping[str, str]) -> dict[str, int]:
    selects = " UNION ALL ".join(
        f'SELECT \'{table}\'::text AS table_name, COUNT(*)::bigint AS row_count '
        f'FROM public."{table}"'
        for table in tables
    )
    value = _psql_json(
        "SELECT COALESCE(json_object_agg(table_name, row_count), '{}'::json) "
        f"FROM ({selects}) rows",
        env=env,
    )
    if not isinstance(value, Mapping) or set(value) != set(tables):
        raise ProductionParityEpochAuthorityError(
            "testnet authority database row counts are invalid"
        )
    counts: dict[str, int] = {}
    for table, raw_count in value.items():
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count < 0
        ):
            raise ProductionParityEpochAuthorityError(
                "testnet authority database row count is invalid"
            )
        counts[str(table)] = raw_count
    if any(counts.get(table, 0) <= 0 for table in REQUIRED_NONEMPTY_TABLES):
        raise ProductionParityEpochAuthorityError(
            "testnet authority database is incomplete"
        )
    return counts


def _cutover_state(*, env: Mapping[str, str]) -> dict[str, Any]:
    value = _psql_json(
        "SELECT row_to_json(state_row) FROM "
        "public.research_lab_stateful_subnet_epoch_cutover_state_v1 state_row",
        env=env,
    )
    if not isinstance(value, dict):
        raise ProductionParityEpochAuthorityError(
            "testnet authority cutover state is invalid"
        )
    return value


def _fingerprint(
    *, tables: Sequence[str], env: Mapping[str, str]
) -> tuple[dict[str, int], dict[str, Any], str]:
    counts = _row_counts(tables, env=env)
    state = _cutover_state(env=env)
    fingerprint = sha256_bytes(
        canonical_bytes({"database_row_counts": counts, "cutover_state": state})
    )
    return counts, state, fingerprint


def _validate_state(state: Mapping[str, Any], cutover: SubnetEpochCutover) -> None:
    if (
        state.get("lifecycle_state") != "stateful_active"
        or state.get("mapping_hash") != cutover.mapping_hash
        or state.get("network_genesis_hash") != cutover.network_genesis_hash
        or state.get("netuid") != cutover.netuid
        or state.get("last_legacy_epoch_id") != cutover.last_legacy_epoch_id
        or state.get("first_settlement_epoch_id")
        != cutover.first_settlement_epoch_id
    ):
        raise ProductionParityEpochAuthorityError(
            "testnet authority cutover state differs from the manifest"
        )


def _dump_tables(
    path: Path, *, tables: Sequence[str], env: Mapping[str, str]
) -> None:
    command = [
        "pg_dump",
        "--data-only",
        "--format=custom",
        "--no-owner",
        "--no-acl",
        "--serializable-deferrable",
        "--file",
        str(path),
    ]
    for table in tables:
        command.extend(["--table", f"public.{table}"])
    _require(
        _run(command, env=env, timeout=900),
        stage="testnet authority database dump",
    )
    listing = _require(
        _run(["pg_restore", "--list", str(path)], env=env, timeout=120),
        stage="testnet authority database dump listing",
    ).decode("utf-8", "strict")
    observed = []
    for line in listing.splitlines():
        match = re.match(
            r"^\d+;\s+\d+\s+\d+\s+TABLE DATA\s+public\s+([^\s]+)\s+.*$",
            line.strip(),
        )
        if match is not None:
            observed.append(match.group(1))
    if len(observed) != len(set(observed)) or set(observed) != set(tables):
        raise ProductionParityEpochAuthorityError(
            "testnet authority database dump table inventory differs"
        )


def _archive(files: Mapping[str, bytes]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(files):
            value = files[name]
            member = tarfile.TarInfo(name=name)
            member.size = len(value)
            member.mode = 0o600
            member.mtime = 0
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, BytesIO(value))
    return output.getvalue()


def build(*, cutover_path: Path, output: Path, dsn_env: str) -> dict[str, Any]:
    if output.exists():
        raise ProductionParityEpochAuthorityError(
            "epoch authority output already exists"
        )
    try:
        cutover_document = json.loads(cutover_path.read_text(encoding="utf-8"))
        cutover = SubnetEpochCutover.from_mapping(cutover_document)
    except (OSError, ValueError, TypeError) as exc:
        raise ProductionParityEpochAuthorityError(
            "testnet cutover manifest is invalid"
        ) from exc
    if cutover_document != cutover.to_dict():
        raise ProductionParityEpochAuthorityError(
            "testnet cutover manifest is not canonical"
        )
    env = _database_env(dsn_env)
    tables = _authority_tables(env=env)
    before_counts, before_state, before_fingerprint = _fingerprint(
        tables=tables, env=env
    )
    _validate_state(before_state, cutover)
    with tempfile.TemporaryDirectory(prefix="leadpoet-parity-authority-") as temp:
        dump_path = Path(temp) / "authority.dump"
        _dump_tables(dump_path, tables=tables, env=env)
        dump_bytes = dump_path.read_bytes()
    after_counts, after_state, after_fingerprint = _fingerprint(
        tables=tables, env=env
    )
    if (
        before_counts != after_counts
        or before_state != after_state
        or before_fingerprint != after_fingerprint
    ):
        raise ProductionParityEpochAuthorityError(
            "testnet authority changed during capture"
        )
    cutover_bytes = (
        json.dumps(cutover.to_dict(), sort_keys=True, indent=2) + "\n"
    ).encode("ascii")
    ceremony = {
        "schema_version": CEREMONY_SCHEMA_VERSION,
        "network": "test",
        "netuid": cutover.netuid,
        "network_genesis_hash": cutover.network_genesis_hash,
        "mapping_hash": cutover.mapping_hash,
        "cutover_manifest_hash": sha256_bytes(cutover_bytes),
        "database_fingerprint_hash": before_fingerprint,
        "authority_dump_hash": sha256_bytes(dump_bytes),
        "table_count": len(tables),
        "row_count": sum(before_counts.values()),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "network": "test",
        "netuid": cutover.netuid,
        "network_genesis_hash": cutover.network_genesis_hash,
        "mapping_hash": cutover.mapping_hash,
        "database_tables": list(tables),
        "database_row_counts": before_counts,
        "files": {
            "stateful-epoch-cutover.json": sha256_bytes(cutover_bytes),
            "authority.dump": sha256_bytes(dump_bytes),
        },
        "ceremony_evidence": ceremony,
        "ceremony_evidence_hash": sha256_bytes(canonical_bytes(ceremony)),
    }
    files = {
        "authority.dump": dump_bytes,
        "manifest.json": (
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        ).encode("ascii"),
        "stateful-epoch-cutover.json": cutover_bytes,
    }
    payload = _archive(files)
    validate_archive(
        payload,
        {
            "netuid": cutover.netuid,
            "mapping_hash": cutover.mapping_hash,
            "network_genesis_hash": cutover.network_genesis_hash,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o600)
    return {
        "artifact_sha256": sha256_bytes(payload),
        "ceremony_evidence_hash": manifest["ceremony_evidence_hash"],
        "database_fingerprint_hash": before_fingerprint,
        "mapping_hash": cutover.mapping_hash,
        "network_genesis_hash": cutover.network_genesis_hash,
        "netuid": cutover.netuid,
        "table_count": len(tables),
        "row_count": sum(before_counts.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutover", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-url-env", default=DEFAULT_DSN_ENV)
    args = parser.parse_args(argv)
    try:
        result = build(
            cutover_path=args.cutover,
            output=args.output,
            dsn_env=str(args.database_url_env),
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        subprocess.TimeoutExpired,
        ProductionParityEpochAuthorityError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
