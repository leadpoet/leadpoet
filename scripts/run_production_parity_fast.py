#!/usr/bin/env python3
"""Run the bounded post-push parity lane against a real production snapshot."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tee.supabase_schema_preflight_v2 import (  # noqa: E402
    verify_required_supabase_v2_schema,
)
from gateway.tee.supabase_source_v2 import (  # noqa: E402
    SUPABASE_WEIGHT_SOURCE_ORIGIN,
    SupabaseSourceReaderV2,
)
from leadpoet_canonical.production_parity import (  # noqa: E402
    ProductionParityError,
    StageLedger,
    required_oracle_stage_ids,
    sha256_json,
    validate_contract,
    validate_historical_oracle,
    validate_snapshot_manifest,
    verify_contract_checkout,
)
from scripts.production_parity_snapshot import (  # noqa: E402
    restore_snapshot,
    verify_snapshot,
)


CRITICAL_STAGES = (
    "candidate-contract",
    "production-snapshot",
    "snapshot-restore-and-migrations",
    "production-data-shape",
    "production-weight-input-scale",
    "supabase-schema-and-rpc",
    "exact-n-minus-one-launchers",
    "protected-v2-workflows",
    "canonical-bundle-generation",
    "primary-auditor-bundle-equality",
    "sign-finalize-readback",
    "cleanup",
)
PINNED_IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")


class _CloneSupabaseProvider:
    """Strictly adapt measured production-origin reads to the disposable clone."""

    def __init__(self, *, clone_url: str, service_role_key: str) -> None:
        self.clone_url = clone_url.rstrip("/")
        self.service_role_key = service_role_key
        self.pages: list[dict[str, Any]] = []

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        parsed = urlsplit(str(request.get("url") or ""))
        production = urlsplit(SUPABASE_WEIGHT_SOURCE_ORIGIN)
        if (
            request.get("provider_id") != "supabase"
            or request.get("method") != "GET"
            or parsed.scheme != "https"
            or parsed.hostname != production.hostname
            or not parsed.path.startswith("/rest/v1/")
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ProductionParityError(
                "clone adapter rejected a non-measured Supabase read"
            )
        clone = urlsplit(self.clone_url)
        clone_path = parsed.path[len("/rest/v1") :]
        if not clone_path.startswith("/"):
            raise ProductionParityError(
                "clone adapter could not resolve the measured PostgREST path"
            )
        local_url = urlunsplit(
            (clone.scheme, clone.netloc, clone_path, parsed.query, "")
        )
        headers = {
            str(name): str(value)
            for name, value in dict(request.get("headers") or {}).items()
        }
        headers.update(
            {
                "Authorization": f"Bearer {self.service_role_key}",
                "apikey": self.service_role_key,
            }
        )
        outbound = Request(local_url, headers=headers, method="GET")
        try:
            with urlopen(
                outbound,
                timeout=max(1, int(request.get("timeout_ms") or 0) // 1000),
            ) as response:
                body = response.read()
                status = int(response.status)
        except HTTPError as exc:
            body = exc.read()
            status = int(exc.code)
        response_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        request_artifact_hash = sha256_json(
            {
                "schema_version": "leadpoet.production_parity_clone_request.v1",
                "logical_operation_id": request.get("logical_operation_id"),
                "method": "GET",
                "production_path_and_query": urlunsplit(
                    ("", "", parsed.path, parsed.query, "")
                ),
                "range": headers.get("range"),
            }
        )
        self.pages.append(
            {
                "http_status": status,
                "response_bytes": len(body),
                "response_hash": response_hash,
                "request_artifact_hash": request_artifact_hash,
            }
        )
        attempt = {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "response_hash": response_hash,
            "request_artifact_hash": request_artifact_hash,
            "response_artifact_hash": response_hash,
            "adapter": "strict-production-origin-to-disposable-clone",
        }
        return {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": attempt,
        }


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionParityError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise ProductionParityError(f"{description} must be an object")
    return value


def _run(
    command: Sequence[str],
    *,
    timeout: int,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env) if env is not None else None,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require_success(result: subprocess.CompletedProcess[str], *, stage: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise ProductionParityError(f"{stage} failed: {detail}")
    return result.stdout


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _service_role_token(secret: str) -> str:
    now = int(time.time())
    header = _b64url(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64url(
        json.dumps(
            {
                "aud": "authenticated",
                "exp": now + 3600,
                "iat": now - 5,
                "iss": "leadpoet-production-parity",
                "role": "service_role",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(secret.encode("ascii"), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


class _DockerDatabase:
    def __init__(
        self,
        *,
        candidate_sha: str,
        postgres_image: str,
        postgrest_image: str,
    ) -> None:
        for field_name, image in (
            ("postgres image", postgres_image),
            ("PostgREST image", postgrest_image),
        ):
            if not PINNED_IMAGE_RE.fullmatch(image):
                raise ProductionParityError(f"{field_name} must be digest-pinned")
        suffix = candidate_sha[:10] + "-" + secrets.token_hex(3)
        self.network = f"leadpoet-parity-{suffix}"
        self.postgres = f"leadpoet-parity-postgres-{suffix}"
        self.postgrest = f"leadpoet-parity-postgrest-{suffix}"
        self.database = f"leadpoet_parity_{candidate_sha[:12]}"
        self.password = secrets.token_urlsafe(24)
        self.jwt_secret = secrets.token_urlsafe(48)
        self.authenticator_password = secrets.token_urlsafe(24)
        self.postgres_image = postgres_image
        self.postgrest_image = postgrest_image
        self.target_dsn = ""
        self.supabase_url = ""

    def start(self) -> None:
        _require_success(
            _run(["docker", "network", "create", self.network], timeout=30),
            stage="parity Docker network creation",
        )
        _require_success(
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.postgres,
                    "--network",
                    self.network,
                    "-p",
                    "127.0.0.1::5432",
                    "-e",
                    f"POSTGRES_PASSWORD={self.password}",
                    "-e",
                    f"POSTGRES_DB={self.database}",
                    self.postgres_image,
                ],
                timeout=60,
            ),
            stage="parity PostgreSQL start",
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ready = _run(
                ["docker", "exec", self.postgres, "pg_isready", "-U", "postgres"],
                timeout=10,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise ProductionParityError("parity PostgreSQL did not become ready")
        port = _require_success(
            _run(["docker", "port", self.postgres, "5432/tcp"], timeout=10),
            stage="parity PostgreSQL port discovery",
        ).strip().rsplit(":", 1)[-1]
        self.target_dsn = (
            f"postgresql://postgres:{self.password}@127.0.0.1:{port}/{self.database}"
        )

    def _psql(self, sql: str, *, timeout: int = 120) -> str:
        return _require_success(
            _run(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.postgres,
                    "psql",
                    "-X",
                    "-U",
                    "postgres",
                    "-d",
                    self.database,
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-A",
                    "-t",
                ],
                timeout=timeout,
                stdin=sql,
            ),
            stage="parity PostgreSQL probe",
        )

    def start_postgrest(self) -> tuple[str, str]:
        role_sql = f"""
DO $$ BEGIN
  CREATE ROLE authenticator LOGIN PASSWORD '{self.authenticator_password}';
EXCEPTION WHEN duplicate_object THEN
  ALTER ROLE authenticator WITH LOGIN PASSWORD '{self.authenticator_password}';
END $$;
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
        self._psql(role_sql)
        db_uri = (
            f"postgres://authenticator:{self.authenticator_password}@"
            f"{self.postgres}:5432/{self.database}"
        )
        _require_success(
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.postgrest,
                    "--network",
                    self.network,
                    "-p",
                    "127.0.0.1::3000",
                    "-e",
                    f"PGRST_DB_URI={db_uri}",
                    "-e",
                    "PGRST_DB_SCHEMAS=public",
                    "-e",
                    "PGRST_DB_ANON_ROLE=anon",
                    "-e",
                    f"PGRST_JWT_SECRET={self.jwt_secret}",
                    self.postgrest_image,
                ],
                timeout=60,
            ),
            stage="parity PostgREST start",
        )
        port = _require_success(
            _run(["docker", "port", self.postgrest, "3000/tcp"], timeout=10),
            stage="parity PostgREST port discovery",
        ).strip().rsplit(":", 1)[-1]
        self.supabase_url = f"http://127.0.0.1:{port}"
        token = _service_role_token(self.jwt_secret)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{self.supabase_url}/", timeout=3) as response:
                    if 200 <= int(response.status) < 300:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise ProductionParityError("parity PostgREST did not become ready")
        return self.supabase_url, token

    def shape_evidence(
        self,
        *,
        service_role_key: str,
        expected_shape: Mapping[str, Any],
    ) -> dict[str, Any]:
        restored_raw = self._psql(
            """
SELECT json_build_object(
  'relation_count', COUNT(*),
  'total_relation_bytes', COALESCE(SUM(pg_total_relation_size(c.oid)), 0),
  'largest_relation_bytes', COALESCE(MAX(pg_total_relation_size(c.oid)), 0)
)::text
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'm')
  AND n.nspname = 'public';
"""
        ).strip()
        restored = json.loads(restored_raw)
        expected_relations = int(expected_shape.get("relation_count") or 0)
        restored_relations = int(restored.get("relation_count") or 0)
        if expected_relations <= 0 or restored_relations < expected_relations:
            raise ProductionParityError(
                "restored relation inventory lost production relations"
            )
        expected_total = int(expected_shape.get("total_relation_bytes") or 0)
        restored_total = int(restored.get("total_relation_bytes") or 0)
        if expected_total <= 0 or restored_total <= 0:
            raise ProductionParityError("restored production data shape is empty")
        size_ratio = restored_total / expected_total
        if not 0.5 <= size_ratio <= 2.0:
            raise ProductionParityError(
                "restored relation size differs materially from the production snapshot"
            )
        largest_raw = self._psql(
            """
SELECT json_build_object(
  'relation', c.relname,
  'relation_bytes', pg_total_relation_size(c.oid),
  'estimated_rows', GREATEST(c.reltuples::bigint, 0)
)::text
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY pg_total_relation_size(c.oid) DESC, c.relname
LIMIT 1;
"""
        ).strip()
        largest = json.loads(largest_raw)
        relation = str(largest["relation"])
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", relation):
            raise ProductionParityError("largest production relation name is unsafe")
        request = Request(
            f"{self.supabase_url}/{relation}?select=*&limit=1000",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {service_role_key}",
                "apikey": service_role_key,
            },
        )
        with urlopen(request, timeout=60) as response:
            payload = response.read()
            status = int(response.status)
        if status != 200:
            raise ProductionParityError("production-shaped PostgREST read failed")
        return {
            "captured_relation_count": expected_relations,
            "restored_relation_count": restored_relations,
            "candidate_relation_delta": restored_relations - expected_relations,
            "captured_total_relation_bytes": expected_total,
            "restored_total_relation_bytes": restored_total,
            "restored_to_captured_size_ratio": round(size_ratio, 6),
            **largest,
            "postgrest_status": status,
            "postgrest_page_bytes": len(payload),
            "postgrest_page_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }

    def weight_input_scale_evidence(
        self,
        *,
        service_role_key: str,
    ) -> dict[str, Any]:
        scope_raw = self._psql(
            """
SELECT json_build_object(
  'netuid', netuid,
  'start_epoch', MIN(epoch_id),
  'end_epoch', MAX(epoch_id),
  'expected_rows', COUNT(*)
)::text
FROM public.research_lab_finalized_allocation_epochs_v2
GROUP BY netuid
ORDER BY COUNT(*) DESC, netuid
LIMIT 1;
"""
        ).strip()
        if not scope_raw:
            raise ProductionParityError(
                "production snapshot has no finalized allocation authority history"
            )
        scope = json.loads(scope_raw)
        expected_rows = int(scope.get("expected_rows") or 0)
        if expected_rows <= 0:
            raise ProductionParityError(
                "production finalized allocation authority history is empty"
            )
        adapter = _CloneSupabaseProvider(
            clone_url=self.supabase_url,
            service_role_key=service_role_key,
        )
        attempts: list[dict[str, Any]] = []
        artifacts: list[str] = []
        reader = SupabaseSourceReaderV2(
            execute_provider=adapter,
            retry_policy_hash="sha256:" + "a" * 64,
            sleep=lambda _seconds: None,
        )
        rows = reader.read(
            policy_id="finalized_allocation_authorities",
            parameters={
                "netuid": int(scope["netuid"]),
                "start_epoch": int(scope["start_epoch"]),
                "end_epoch": int(scope["end_epoch"]),
            },
            job_id="production-parity-weight-history",
            purpose="research_lab.legacy_finalized_allocation.v2",
            record_transport=lambda attempt: attempts.append(dict(attempt)),
            record_artifact=artifacts.append,
        )
        if len(rows) != expected_rows:
            raise ProductionParityError(
                "candidate measured source did not reproduce the complete production history"
            )
        page_bytes = [int(item["response_bytes"]) for item in adapter.pages]
        if not page_bytes or len(attempts) != len(adapter.pages):
            raise ProductionParityError(
                "candidate measured source did not produce complete page evidence"
            )
        return {
            **scope,
            "read_rows": len(rows),
            "page_count": len(adapter.pages),
            "total_response_bytes": sum(page_bytes),
            "max_page_bytes": max(page_bytes),
            "response_hashes": [item["response_hash"] for item in adapter.pages],
            "artifact_count": len(artifacts),
            "adapter": "strict-production-origin-to-disposable-clone",
        }

    def cleanup(self) -> dict[str, Any]:
        for container in (self.postgrest, self.postgres):
            _run(["docker", "rm", "-f", container], timeout=30)
        _run(["docker", "network", "rm", self.network], timeout=30)
        remaining: list[str] = []
        for resource_type, resource_name in (
            ("container", self.postgrest),
            ("container", self.postgres),
            ("network", self.network),
        ):
            command = (
                ["docker", "container", "inspect", resource_name]
                if resource_type == "container"
                else ["docker", "network", "inspect", resource_name]
            )
            if _run(command, timeout=10).returncode == 0:
                remaining.append(f"{resource_type}:{resource_name}")
        if remaining:
            raise ProductionParityError(
                "parity resources remain after cleanup: " + ",".join(remaining)
            )
        return {
            "containers_removed": [self.postgres, self.postgrest],
            "network_removed": self.network,
        }


def _rehearsal_evidence_path(candidate_sha: str) -> Path:
    return Path("/tmp") / f"leadpoet-restart-rehearsal-{candidate_sha}-prepush.json"


def _run_rehearsal(*, base_sha: str, candidate_sha: str) -> dict[str, Any]:
    evidence_path = _rehearsal_evidence_path(candidate_sha)
    evidence_path.unlink(missing_ok=True)
    result = _run(
        [
            sys.executable,
            "scripts/run_local_restart_rehearsal.py",
            "--from-sha",
            base_sha,
            "--candidate-sha",
            candidate_sha,
            "--transition",
            "forward",
            "--profile",
            "prepush",
        ],
        timeout=600,
    )
    _require_success(result, stage="candidate-derived N-1 rehearsal")
    evidence = _load_json(evidence_path, description="joined restart rehearsal evidence")
    if (
        evidence.get("status") != "passed"
        or evidence.get("release_sha") != candidate_sha
        or evidence.get("from_sha") != base_sha
        or evidence.get("profile") != "prepush"
    ):
        raise ProductionParityError("joined restart rehearsal identity differs")
    return evidence


def _run_database_lane(
    *,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    production_host: str,
    postgres_image: str,
    postgrest_image: str,
) -> dict[str, Any]:
    contract = validate_contract(_load_json(contract_path, description="parity contract"))
    database = _DockerDatabase(
        candidate_sha=contract["candidate_sha"],
        postgres_image=postgres_image,
        postgrest_image=postgrest_image,
    )
    result: dict[str, Any] = {}
    cleanup: dict[str, Any] | None = None
    try:
        database.start()
        restore = restore_snapshot(
            root=ROOT,
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            target_dsn=database.target_dsn,
            production_host=production_host,
        )
        supabase_url, service_role_key = database.start_postgrest()
        schema = verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": supabase_url,
                "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
                "BITTENSOR_NETUID": os.environ.get("BITTENSOR_NETUID", "71"),
            },
            timeout_seconds=20,
        )
        manifest = validate_snapshot_manifest(
            _load_json(manifest_path, description="snapshot manifest")
        )
        shape = database.shape_evidence(
            service_role_key=service_role_key,
            expected_shape=manifest["database"],
        )
        weight_input_scale = database.weight_input_scale_evidence(
            service_role_key=service_role_key
        )
        result = {
            "restore": restore,
            "schema": schema,
            "shape": shape,
            "weight_input_scale": weight_input_scale,
        }
    finally:
        cleanup = database.cleanup()
    return {**result, "cleanup": cleanup}


def run_fast_lane(
    *,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    ledger_path: Path,
    production_host: str,
    postgres_image: str,
    postgrest_image: str,
) -> dict[str, Any]:
    contract = verify_contract_checkout(
        ROOT, _load_json(contract_path, description="parity contract")
    )
    manifest = validate_snapshot_manifest(
        _load_json(manifest_path, description="snapshot manifest")
    )
    oracle = validate_historical_oracle(
        _load_json(
            ROOT / "tests/restart_rehearsal/fixtures/august_9_known_good_v2.json",
            description="historical production oracle",
        )
    )
    oracle_stages = set(required_oracle_stage_ids(oracle, lane="fast"))
    if (
        sha256_json(oracle) != contract["historical_oracle_hash"]
        or not oracle_stages.issubset(CRITICAL_STAGES)
    ):
        raise ProductionParityError(
            "fast lane does not cover the historical production behavior oracle"
        )
    ledger = StageLedger(
        lane="fast",
        candidate_sha=contract["candidate_sha"],
        contract_hash=contract["contract_hash"],
        snapshot_hash=manifest["manifest_hash"],
        critical_stage_ids=CRITICAL_STAGES,
    )
    ledger.record(
        "candidate-contract",
        status="passed",
        duration_seconds=0,
        evidence={
            "base_sha": contract["base_sha"],
            "risk": contract["risk"],
            "source_count": len(contract["source_commitments"]),
        },
    )
    snapshot_started = time.monotonic()
    try:
        snapshot_evidence = verify_snapshot(
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            expected_production_host=production_host,
        )
        for label, ancestor, descendant in (
            (
                "source",
                manifest["source_sha"],
                manifest["capture_sha"],
            ),
            (
                "capture",
                manifest["capture_sha"],
                contract["candidate_sha"],
            ),
        ):
            ancestry = _run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    descendant,
                ],
                timeout=30,
            )
            if ancestry.returncode != 0:
                raise ProductionParityError(
                    f"production snapshot {label} lineage differs from the candidate"
                )
        snapshot_evidence["source_is_candidate_ancestor"] = True
        snapshot_evidence["capture_is_candidate_ancestor"] = True
    except Exception as exc:
        ledger.record(
            "production-snapshot",
            status="failed",
            duration_seconds=time.monotonic() - snapshot_started,
            reason=f"{type(exc).__name__}: {exc}",
        )
        snapshot_evidence = None
    else:
        ledger.record(
            "production-snapshot",
            status="passed",
            duration_seconds=time.monotonic() - snapshot_started,
            evidence=snapshot_evidence,
        )

    results: dict[str, Any] = {}
    failures: dict[str, Exception] = {}
    actions: dict[str, Callable[[], dict[str, Any]]] = {
        "rehearsal": lambda: _run_rehearsal(
            base_sha=contract["base_sha"], candidate_sha=contract["candidate_sha"]
        ),
    }
    if snapshot_evidence is not None:
        actions["database"] = lambda: _run_database_lane(
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            production_host=production_host,
            postgres_image=postgres_image,
            postgrest_image=postgrest_image,
        )
    started = {name: time.monotonic() for name in actions}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(actions)) as pool:
        futures = {pool.submit(action): name for name, action in actions.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                failures[name] = exc

    database_duration = time.monotonic() - started.get("database", time.monotonic())
    if "database" in results:
        database = results["database"]
        ledger.record(
            "snapshot-restore-and-migrations",
            status="passed",
            duration_seconds=database_duration,
            evidence=database["restore"],
        )
        ledger.record(
            "production-data-shape",
            status="passed",
            duration_seconds=0,
            evidence=database["shape"],
        )
        ledger.record(
            "production-weight-input-scale",
            status="passed",
            duration_seconds=0,
            evidence=database["weight_input_scale"],
        )
        ledger.record(
            "supabase-schema-and-rpc",
            status="passed",
            duration_seconds=0,
            evidence=database["schema"],
        )
    else:
        reason = (
            f"{type(failures['database']).__name__}: {failures['database']}"
            if "database" in failures
            else "production snapshot was unavailable"
        )
        for stage in (
            "snapshot-restore-and-migrations",
            "production-data-shape",
            "production-weight-input-scale",
            "supabase-schema-and-rpc",
        ):
            ledger.record(stage, status="failed", duration_seconds=database_duration, reason=reason)

    rehearsal_duration = time.monotonic() - started["rehearsal"]
    if "rehearsal" in results:
        rehearsal = results["rehearsal"]
        if rehearsal.get("behavior_contract_hash") != contract["behavior_contract_hash"]:
            failures["rehearsal"] = ProductionParityError(
                "rehearsal behavior contract differs from parity contract"
            )
        else:
            common = {
                "evidence_schema": rehearsal.get("schema_version"),
                "pcr0": rehearsal.get("pcr0"),
                "bundle_hash": rehearsal.get("bundle_hash"),
            }
            ledger.record(
                "exact-n-minus-one-launchers",
                status="passed",
                duration_seconds=rehearsal_duration,
                evidence={**common, "restart_invariants": rehearsal["restart_invariants"]},
            )
            ledger.record(
                "protected-v2-workflows",
                status="passed",
                duration_seconds=0,
                evidence={
                    "behavior_contract_hash": rehearsal["behavior_contract_hash"],
                    "invariants": rehearsal["behavioral_invariants"],
                },
            )
            ledger.record(
                "canonical-bundle-generation",
                status="passed",
                duration_seconds=0,
                evidence={**common, "receipt_ancestry": rehearsal["receipt_ancestry"]},
            )
            ledger.record(
                "primary-auditor-bundle-equality",
                status="passed",
                duration_seconds=0,
                evidence={
                    "canonical_vector": rehearsal["canonical_vector"],
                    "auditor": rehearsal["auditor"],
                },
            )
            ledger.record(
                "sign-finalize-readback",
                status="passed",
                duration_seconds=0,
                evidence={
                    "signed_extrinsic": rehearsal["signed_extrinsic"],
                    "finalization": rehearsal["finalization"],
                    "reveal": rehearsal["reveal"],
                },
            )
    if "rehearsal" in failures:
        reason = f"{type(failures['rehearsal']).__name__}: {failures['rehearsal']}"
        for stage in (
            "exact-n-minus-one-launchers",
            "protected-v2-workflows",
            "canonical-bundle-generation",
            "primary-auditor-bundle-equality",
            "sign-finalize-readback",
        ):
            if not any(item["stage_id"] == stage for item in ledger.stages):
                ledger.record(stage, status="failed", duration_seconds=rehearsal_duration, reason=reason)

    cleanup = results.get("database", {}).get("cleanup")
    cleanup_ok = snapshot_evidence is not None and cleanup is not None
    ledger.record(
        "cleanup",
        status="passed" if cleanup_ok else "failed",
        duration_seconds=0,
        evidence=dict(cleanup or {}),
        reason="" if cleanup_ok else "database lane did not prove run-scoped cleanup",
    )
    final = ledger.finalize()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(final, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-archive", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--production-db-host", required=True)
    parser.add_argument("--postgres-image", required=True)
    parser.add_argument("--postgrest-image", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_fast_lane(
            contract_path=args.contract,
            manifest_path=args.snapshot_manifest,
            archive_path=args.snapshot_archive,
            ledger_path=args.ledger,
            production_host=args.production_db_host,
            postgres_image=args.postgres_image,
            postgrest_image=args.postgrest_image,
        )
    except (OSError, ValueError, ProductionParityError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
