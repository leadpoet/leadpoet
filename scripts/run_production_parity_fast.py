#!/usr/bin/env python3
"""Run the bounded post-push parity lane against a real production snapshot."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
    urlopen,
)

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tee.supabase_schema_preflight_v2 import (  # noqa: E402
    REQUIRED_SUPABASE_V2_SCHEMA,
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
    DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS,
    DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    restore_schema_only_source_add_acl_contract,
    restore_snapshot,
    verify_snapshot,
)
from scripts.materialize_production_parity_secrets import (  # noqa: E402
    SecretMaterializationError,
    _parse_environment_document,
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
CHAIN_REALIZED_ACTIVATION_TABLE = "research_lab_chain_realized_settlement_activation_v1"
CHAIN_REALIZED_ACTIVATION_SCHEMA_VERSION = (
    "leadpoet.research_lab_chain_realized_settlement_activation.v1"
)
CHAIN_REALIZED_ACTIVATION_COLUMNS = (
    "netuid",
    "schema_version",
    "first_epoch_id",
    "source_bundle_hash",
    "source_bundle_epoch_id",
    "source_finalized_block",
)
# The candidate controller keeps the authoritative prepush wall-clock cap at
# 600 seconds.  Its parent must leave enough time for the controller's bounded
# worker cancellation, evidence normalization, local unwind, and sanitized
# failure handoff; otherwise the parent wins the deadline race and destroys the
# exact stage diagnostics produced at the inner limit.
FAST_REHEARSAL_INNER_TIMEOUT_SECONDS = 600
FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS = 120
FAST_REHEARSAL_TIMEOUT_SECONDS = (
    FAST_REHEARSAL_INNER_TIMEOUT_SECONDS
    + FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS
)
FAST_SCHEMA_PREFLIGHT_TIMEOUT_SECONDS = 20
# Fast validates every schema table plus OpenAPI and two contract RPCs. The
# activation value is validated from the strict live read instead of another
# disposable-clone data request.
FAST_SCHEMA_PREFLIGHT_NETWORK_PROBE_COUNT = len(REQUIRED_SUPABASE_V2_SCHEMA) + 3
FAST_SCHEMA_PREFLIGHT_HEADROOM_SECONDS = (
    FAST_SCHEMA_PREFLIGHT_NETWORK_PROBE_COUNT
    * FAST_SCHEMA_PREFLIGHT_TIMEOUT_SECONDS
)
FAST_CANDIDATE_MIGRATION_HEADROOM_COUNT = 2
FAST_CANDIDATE_MIGRATION_HEADROOM_SECONDS = (
    FAST_CANDIDATE_MIGRATION_HEADROOM_COUNT
    * DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS
)
FAST_DOCKER_STARTUP_AND_CLEANUP_HEADROOM_SECONDS = 420
# All live production rows are read through the measured two-row pagination
# policy.  Enforce one aggregate wall-clock bound here instead of pretending
# that every policy retry timeout can fit inside the workflow independently.
FAST_PRODUCTION_DATA_READ_TIMEOUT_SECONDS = 480
FAST_PRODUCTION_DATA_READ_REQUEST_TIMEOUT_SECONDS = 5
FAST_JOB_SETUP_HEADROOM_SECONDS = 300
FAST_JOB_FIXED_TIMEOUT_SECONDS = (
    2 * DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS
    + FAST_REHEARSAL_TIMEOUT_SECONDS
    + FAST_SCHEMA_PREFLIGHT_HEADROOM_SECONDS
    + FAST_DOCKER_STARTUP_AND_CLEANUP_HEADROOM_SECONDS
    + FAST_PRODUCTION_DATA_READ_TIMEOUT_SECONDS
    + FAST_JOB_SETUP_HEADROOM_SECONDS
)
FAST_JOB_MINIMUM_TIMEOUT_SECONDS = (
    FAST_JOB_FIXED_TIMEOUT_SECONDS + FAST_CANDIDATE_MIGRATION_HEADROOM_SECONDS
)
FAST_JOB_OUTER_TIMEOUT_SECONDS = 110 * 60
FAST_AWS_ROLE_DURATION_SECONDS = 7740
SAFE_REHEARSAL_ERROR_TYPES = frozenset(
    {
        "AssertionError",
        "CalledProcessError",
        "FileNotFoundError",
        "OSError",
        "PermissionError",
        "RehearsalTimeBudgetExceeded",
        "RuntimeError",
        "SystemExit",
        "TimeoutExpired",
        "TypeError",
        "ValueError",
    }
)
SAFE_WORKFLOW_PROJECTION_ERROR_TYPES = SAFE_REHEARSAL_ERROR_TYPES | {
    "None",
    "OtherError",
    "TimeoutError",
}
SAFE_WORKFLOW_STAGE_KINDS = frozenset(
    {
        "allocation",
        "behavior",
        "boundary",
        "concurrency",
        "diagnostic",
        "epoch",
        "fault",
        "input",
        "source_identity",
        "unknown",
        "validation",
    }
)
SAFE_REHEARSAL_STAGE_PATTERNS = (
    r"python37-finalization",
    r"drand-artifact-[0-9a-f]{12}",
    r"fixture-seed-[0-9a-f]{12}",
    r"fixture-orchestration",
    r"fixture-cleanup",
    r"(?:gateway|validator)-(?:forward|rollback)-[1-3]",
    r"workflow-(?:prepush|release)",
    r"evidence-join-(?:prepush|release)",
    r"time-budget",
)
SAFE_PREPUSH_PHASE_ORDER = (
    "exact-image-build",
    "source-snapshot",
    "runtime-prefix",
    "fixture-preparation",
    "workflow-runtime",
    "validator-runtime",
    "gateway-runtime",
)
SAFE_PREPUSH_PHASES = frozenset(SAFE_PREPUSH_PHASE_ORDER)
SAFE_PREPUSH_PHASE_STATUSES = frozenset({"failed", "passed", "started"})
SAFE_PREPUSH_PHASE_DURATION_MAX_SECONDS = 600.0
SAFE_PREPUSH_PHASE_MARKER_MAX = len(SAFE_PREPUSH_PHASES) * 2
SAFE_PREPUSH_PHASE_MARKER_RE = re.compile(
    r"^REHEARSAL_PREPUSH_PHASE "
    r"phase=([a-z-]+) status=([a-z]+) "
    r"duration_seconds=([0-9]+(?:\.[0-9]{1,3})?)$",
    re.MULTILINE,
)
SAFE_IMAGE_BUILD_PHASE_ORDER = (
    "system-packages",
    "python-dependencies",
    "scoring-wheelhouses",
    "external-artifacts",
    "image-finalization",
)
SAFE_IMAGE_BUILD_PHASES = frozenset(SAFE_IMAGE_BUILD_PHASE_ORDER)
SAFE_IMAGE_BUILD_PHASE_STATUSES = frozenset({"failed", "passed", "started"})
SAFE_IMAGE_BUILD_PHASE_MARKER_MAX = len(SAFE_IMAGE_BUILD_PHASE_ORDER) * 2
SAFE_IMAGE_BUILD_PHASE_MARKER_RE = re.compile(
    r"^#[0-9]+ [0-9]+(?:\.[0-9]+)? "
    r"REHEARSAL_IMAGE_BUILD_PHASE "
    r"phase=([a-z-]+) status=([a-z]+)$",
    re.MULTILINE,
)
SECRET_LIKE_DIAGNOSTIC_RE = re.compile(
    r"(?i)(?:authorization|bearer|api[_-]?key|secret|password|token|"
    r"credential|private[_ -]?key|-----begin|://[^/\s]+:[^@\s]+@)"
)


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


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class _StandalonePostgrestSchemaOpener:
    """Map Supabase API paths only onto one loopback PostgREST origin."""

    def __init__(self, origin: str) -> None:
        parsed = urlsplit(str(origin or "").rstrip("/"))
        try:
            port = parsed.port
        except ValueError as exc:
            raise ProductionParityError(
                "standalone PostgREST origin is invalid"
            ) from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ProductionParityError(
                "standalone PostgREST origin is not exact loopback"
            )
        self._origin = parsed
        self._opener = build_opener(ProxyHandler({}), _NoRedirect)

    def __call__(self, request: Request, *, timeout: float):
        if not isinstance(request, Request):
            raise ProductionParityError(
                "standalone PostgREST schema request is invalid"
            )
        parsed = urlsplit(request.full_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ProductionParityError(
                "standalone PostgREST schema request is invalid"
            ) from exc
        if (
            parsed.scheme != self._origin.scheme
            or parsed.hostname != self._origin.hostname
            or port != self._origin.port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or (parsed.path != "/rest/v1" and not parsed.path.startswith("/rest/v1/"))
        ):
            raise ProductionParityError(
                "standalone PostgREST schema request escaped the clone"
            )
        standalone_path = parsed.path[len("/rest/v1") :] or "/"
        outbound = Request(
            urlunsplit(
                (
                    self._origin.scheme,
                    self._origin.netloc,
                    standalone_path,
                    parsed.query,
                    "",
                )
            ),
            data=request.data,
            headers=dict(request.header_items()),
            method=request.get_method(),
        )
        return self._opener.open(outbound, timeout=timeout)


class _ProductionReadOnlySupabaseProvider:
    """Execute only candidate-generated GETs against the exact live origin."""

    adapter_name = "strict-read-only-production-postgrest"

    def __init__(
        self,
        *,
        origin: str,
        service_role_key: str,
        deadline_monotonic: float | None = None,
    ) -> None:
        normalized = str(origin or "").strip().rstrip("/")
        if normalized != SUPABASE_WEIGHT_SOURCE_ORIGIN:
            raise ProductionParityError(
                "production PostgREST origin differs from the measured origin"
            )
        if not str(service_role_key or "").strip():
            raise ProductionParityError(
                "production PostgREST read credential is missing"
            )
        self.origin = normalized
        self.service_role_key = str(service_role_key)
        if deadline_monotonic is not None and deadline_monotonic <= 0:
            raise ProductionParityError(
                "production PostgREST read deadline is invalid"
            )
        self.deadline_monotonic = deadline_monotonic
        self.pages: list[dict[str, Any]] = []
        self._opener = build_opener(ProxyHandler({}), _NoRedirect)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        url = str(request.get("url") or "")
        parsed = urlsplit(url)
        production = urlsplit(self.origin)
        try:
            body = base64.b64decode(str(request.get("body_b64") or ""), validate=True)
        except ValueError as exc:
            raise ProductionParityError(
                "production read adapter rejected malformed request bytes"
            ) from exc
        if (
            request.get("provider_id") != "supabase"
            or request.get("method") != "GET"
            or body
            or parsed.scheme != "https"
            or parsed.hostname != production.hostname
            or parsed.port not in (None, 443)
            or not parsed.path.startswith("/rest/v1/")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ProductionParityError(
                "production read adapter rejected a non-GET or foreign request"
            )
        headers = {
            str(name): str(value)
            for name, value in dict(request.get("headers") or {}).items()
            if str(name).lower() not in {"authorization", "apikey"}
        }
        headers.update(
            {
                "Authorization": f"Bearer {self.service_role_key}",
                "apikey": self.service_role_key,
            }
        )
        outbound = Request(url, headers=headers, method="GET")
        requested_timeout = max(1, int(request.get("timeout_ms") or 0) // 1000)
        timeout = min(
            requested_timeout,
            FAST_PRODUCTION_DATA_READ_REQUEST_TIMEOUT_SECONDS,
        )
        if self.deadline_monotonic is not None:
            remaining = self.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise ProductionParityError(
                    "production PostgREST read deadline expired"
                )
            timeout = min(float(timeout), remaining)
        try:
            with self._opener.open(
                outbound,
                timeout=timeout,
            ) as response:
                response_body = response.read()
                status = int(response.status)
        except HTTPError as exc:
            response_body = exc.read()
            status = int(exc.code)
        response_hash = "sha256:" + hashlib.sha256(response_body).hexdigest()
        request_artifact_hash = sha256_json(
            {
                "schema_version": "leadpoet.production_parity_live_read.v1",
                "logical_operation_id": request.get("logical_operation_id"),
                "method": "GET",
                "path_and_query": urlunsplit(("", "", parsed.path, parsed.query, "")),
                "range": headers.get("range"),
            }
        )
        self.pages.append(
            {
                "http_status": status,
                "response_bytes": len(response_body),
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
            "adapter": self.adapter_name,
        }
        return {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "body_b64": base64.b64encode(response_body).decode("ascii"),
            "transport_attempt": attempt,
        }


def _load_production_supabase_read(
    *, region: str, secret_id: str, deadline_monotonic: float
) -> _ProductionReadOnlySupabaseProvider:
    try:
        raw = (
            boto3.client("secretsmanager", region_name=region)
            .get_secret_value(SecretId=secret_id)
            .get("SecretString")
        )
        values = _parse_environment_document(
            str(raw or ""), field="production gateway environment"
        )
    except (BotoCoreError, ClientError, SecretMaterializationError) as exc:
        raise ProductionParityError(
            "production PostgREST read configuration is unavailable"
        ) from exc
    return _ProductionReadOnlySupabaseProvider(
        origin=str(values.get("SUPABASE_URL") or ""),
        service_role_key=str(values.get("SUPABASE_SERVICE_ROLE_KEY") or ""),
        deadline_monotonic=deadline_monotonic,
    )


def _validated_chain_realized_activation(
    value: Any, *, expected_netuid: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        CHAIN_REALIZED_ACTIVATION_COLUMNS
    ):
        raise ProductionParityError(
            "production chain-realized activation row is invalid"
        )
    integer_fields = (
        "netuid",
        "first_epoch_id",
        "source_bundle_epoch_id",
        "source_finalized_block",
    )
    if (
        any(type(value.get(name)) is not int for name in integer_fields)
        or type(value.get("schema_version")) is not str
        or type(value.get("source_bundle_hash")) is not str
    ):
        raise ProductionParityError(
            "production chain-realized activation row is invalid"
        )
    row = {name: value[name] for name in CHAIN_REALIZED_ACTIVATION_COLUMNS}
    if (
        type(expected_netuid) is not int
        or expected_netuid <= 0
        or row["netuid"] != expected_netuid
        or row["schema_version"] != CHAIN_REALIZED_ACTIVATION_SCHEMA_VERSION
        or row["first_epoch_id"] < 0
        or row["source_bundle_epoch_id"] != row["first_epoch_id"]
        or row["source_finalized_block"] < 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["source_bundle_hash"])) is None
    ):
        raise ProductionParityError(
            "production chain-realized activation row is invalid"
        )
    return row


def _read_production_chain_realized_activation(
    *, provider: _ProductionReadOnlySupabaseProvider, netuid: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(netuid) is not int or netuid <= 0:
        raise ProductionParityError(
            "production chain-realized activation netuid is invalid"
        )
    query = urlencode(
        {
            "select": ",".join(CHAIN_REALIZED_ACTIVATION_COLUMNS),
            "netuid": f"eq.{netuid}",
            "limit": "2",
        }
    )
    result = provider(
        {
            "provider_id": "supabase",
            "method": "GET",
            "url": (
                f"{SUPABASE_WEIGHT_SOURCE_ORIGIN}/rest/v1/"
                f"{CHAIN_REALIZED_ACTIVATION_TABLE}?{query}"
            ),
            "headers": {"Accept": "application/json"},
            "body_b64": "",
            "timeout_ms": 20_000,
            "logical_operation_id": ("production-parity-chain-realized-activation-v1"),
        }
    )
    attempt = result.get("transport_attempt") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or result.get("terminal_status") != "authenticated_response"
        or type(result.get("http_status")) is not int
        or result.get("http_status") != 200
        or not isinstance(attempt, Mapping)
        or attempt.get("terminal_status") != "authenticated_response"
        or type(attempt.get("http_status")) is not int
        or attempt.get("http_status") != 200
        or attempt.get("adapter") != _ProductionReadOnlySupabaseProvider.adapter_name
        or not isinstance(result.get("body_b64"), str)
    ):
        raise ProductionParityError("production chain-realized activation read failed")
    response_hash = str(attempt.get("response_hash") or "")
    request_hash = str(attempt.get("request_artifact_hash") or "")
    if any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        for value in (response_hash, request_hash)
    ):
        raise ProductionParityError(
            "production chain-realized activation evidence is invalid"
        )
    try:
        encoded = base64.b64decode(str(result.get("body_b64") or ""), validate=True)
        if not encoded or len(encoded) > 4096:
            raise ValueError("activation response size is invalid")
        if (
            "sha256:" + hashlib.sha256(encoded).hexdigest() != response_hash
            or attempt.get("response_artifact_hash") != response_hash
        ):
            raise ValueError("activation response hash differs")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            decoded: dict[str, Any] = {}
            for key, value in pairs:
                if key in decoded:
                    raise ValueError("activation response repeats a field")
                decoded[key] = value
            return decoded

        rows = json.loads(encoded.decode("utf-8"), object_pairs_hook=unique_object)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ProductionParityError(
            "production chain-realized activation response is invalid"
        ) from exc
    if not isinstance(rows, list) or len(rows) != 1:
        raise ProductionParityError(
            "production chain-realized activation is missing or ambiguous"
        )
    row = _validated_chain_realized_activation(rows[0], expected_netuid=netuid)
    return row, {
        "netuid": row["netuid"],
        "first_epoch_id": row["first_epoch_id"],
        "source_bundle_epoch_id": row["source_bundle_epoch_id"],
        "source_finalized_block": row["source_finalized_block"],
        "source_bundle_hash": row["source_bundle_hash"],
        "schema_version_hash": "sha256:"
        + hashlib.sha256(row["schema_version"].encode("utf-8")).hexdigest(),
        "activation_row_hash": sha256_json(row),
        "response_hash": response_hash,
        "request_artifact_hash": request_hash,
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


def _fast_job_minimum_timeout_seconds(candidate_migration_count: int) -> int:
    if (
        type(candidate_migration_count) is not int
        or candidate_migration_count < 0
    ):
        raise ProductionParityError("candidate migration count is invalid")
    return (
        FAST_JOB_FIXED_TIMEOUT_SECONDS
        + candidate_migration_count
        * DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS
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
        postgres_publish: str = "127.0.0.1::5432",
        postgrest_publish: str = "127.0.0.1::3000",
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
        self.postgres_volume = f"leadpoet-parity-pgdata-{suffix}"
        self.database = f"leadpoet_parity_{candidate_sha[:12]}"
        self.password = secrets.token_urlsafe(24)
        self.jwt_secret = secrets.token_urlsafe(48)
        self.authenticator_password = secrets.token_urlsafe(24)
        self.postgres_image = postgres_image
        self.postgrest_image = postgrest_image
        self.postgres_publish = str(postgres_publish)
        self.postgrest_publish = str(postgrest_publish)
        self.target_dsn = ""
        self.supabase_url = ""

    def start(self) -> None:
        _require_success(
            _run(
                ["docker", "volume", "create", self.postgres_volume],
                timeout=30,
            ),
            stage="parity PostgreSQL volume creation",
        )
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
                    "--mount",
                    (
                        "type=volume,source="
                        f"{self.postgres_volume},target=/var/lib/postgresql/data"
                    ),
                    "-p",
                    self.postgres_publish,
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
            # The image initializes through a socket-only temporary postmaster.
            # TCP readiness proves the final postmaster has replaced it.
            ready = _run(
                [
                    "docker",
                    "exec",
                    self.postgres,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-p",
                    "5432",
                    "-U",
                    "postgres",
                    "-d",
                    self.database,
                ],
                timeout=10,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise ProductionParityError("parity PostgreSQL did not become ready")
        port = (
            _require_success(
                _run(["docker", "port", self.postgres, "5432/tcp"], timeout=10),
                stage="parity PostgreSQL port discovery",
            )
            .strip()
            .rsplit(":", 1)[-1]
        )
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

    def prepare_snapshot_restore(self) -> dict[str, bool]:
        """Install and verify the Supabase objects referenced by the public dump."""
        self._psql(
            """
DO $$ BEGIN CREATE ROLE anon NOLOGIN INHERIT; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE authenticated NOLOGIN INHERIT; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE service_role NOLOGIN INHERIT BYPASSRLS; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
ALTER ROLE anon WITH NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE authenticated WITH NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE service_role WITH NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
CREATE OR REPLACE FUNCTION auth.role()
RETURNS text
LANGUAGE sql
STABLE
AS $$
  SELECT COALESCE(
    NULLIF(current_setting('request.jwt.claim.role', true), ''),
    NULLIF(current_setting('request.jwt.claims', true), '')::jsonb ->> 'role',
    current_user::text
  )
$$;
CREATE OR REPLACE FUNCTION auth.jwt()
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
  SELECT COALESCE(
    NULLIF(current_setting('request.jwt.claim', true), ''),
    NULLIF(current_setting('request.jwt.claims', true), '')
  )::jsonb
$$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA auth TO anon, authenticated, service_role;
"""
        )
        raw = self._psql(
            """
SELECT json_build_object(
  'anon_role', EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'anon'
      AND NOT rolcanlogin AND rolinherit AND NOT rolsuper
      AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
      AND NOT rolbypassrls
  ),
  'authenticated_role', EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'authenticated'
      AND NOT rolcanlogin AND rolinherit AND NOT rolsuper
      AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
      AND NOT rolbypassrls
  ),
  'service_role', EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'service_role'
      AND NOT rolcanlogin AND rolinherit AND NOT rolsuper
      AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
      AND rolbypassrls
  ),
  'auth_schema', to_regnamespace('auth') IS NOT NULL,
  'extensions_schema', to_regnamespace('extensions') IS NOT NULL,
  'pgcrypto_extension', EXISTS (
    SELECT 1
    FROM pg_extension AS e
    JOIN pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'pgcrypto' AND n.nspname = 'extensions'
  ),
  'auth_role_function', to_regprocedure('auth.role()') IS NOT NULL,
  'auth_jwt_function', to_regprocedure('auth.jwt()') IS NOT NULL
)::text;
"""
        ).strip()
        try:
            evidence = json.loads(raw)
        except ValueError as exc:
            raise ProductionParityError(
                "snapshot restore prerequisite readback is invalid"
            ) from exc
        expected = {
            "anon_role",
            "authenticated_role",
            "service_role",
            "auth_schema",
            "extensions_schema",
            "pgcrypto_extension",
            "auth_role_function",
            "auth_jwt_function",
        }
        if (
            not isinstance(evidence, dict)
            or set(evidence) != expected
            or any(evidence.get(name) is not True for name in expected)
        ):
            raise ProductionParityError("snapshot restore prerequisites did not verify")
        return {name: True for name in sorted(expected)}

    def verify_snapshot_restore(self) -> dict[str, bool]:
        raw = self._psql(
            """
SELECT json_build_object(
  'deterministic_uuid_repeatable',
  public.research_lab_deterministic_uuid('production-parity-restore-probe') IS NOT NULL
  AND public.research_lab_deterministic_uuid('production-parity-restore-probe')
      = public.research_lab_deterministic_uuid('production-parity-restore-probe')
)::text;
"""
        ).strip()
        try:
            evidence = json.loads(raw)
        except ValueError as exc:
            raise ProductionParityError(
                "snapshot restore contract readback is invalid"
            ) from exc
        if evidence != {"deterministic_uuid_repeatable": True}:
            raise ProductionParityError("snapshot restore contract did not verify")
        return {"deterministic_uuid_repeatable": True}

    def start_postgrest(self) -> tuple[str, str]:
        self.prepare_snapshot_restore()
        role_sql = f"""
DO $$ BEGIN
  CREATE ROLE authenticator LOGIN PASSWORD '{self.authenticator_password}';
EXCEPTION WHEN duplicate_object THEN
  ALTER ROLE authenticator WITH LOGIN PASSWORD '{self.authenticator_password}';
END $$;
GRANT anon, authenticated, service_role TO authenticator;
DO $roles$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
  ) THEN
    EXECUTE 'GRANT lab_arena_service TO authenticator';
  END IF;
END
$roles$;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA extensions TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO service_role;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;
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
                    self.postgrest_publish,
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
        port = (
            _require_success(
                _run(["docker", "port", self.postgrest, "3000/tcp"], timeout=10),
                stage="parity PostgREST port discovery",
            )
            .strip()
            .rsplit(":", 1)[-1]
        )
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
        capture_mode: str,
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
        if capture_mode == "schema-only":
            return {
                "capture_mode": capture_mode,
                "captured_relation_count": expected_relations,
                "restored_relation_count": restored_relations,
                "candidate_relation_delta": restored_relations - expected_relations,
                "captured_total_relation_bytes": expected_total,
                "captured_largest_relation_bytes": int(
                    expected_shape.get("largest_relation_bytes") or 0
                ),
                "restored_schema_bytes": restored_total,
                "live_rows_copied": 0,
            }
        if capture_mode != "full":
            raise ProductionParityError("snapshot capture mode is unsupported")
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
            "capture_mode": capture_mode,
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
        scope: Mapping[str, Any] | None = None,
        provider: Any | None = None,
    ) -> dict[str, Any]:
        if scope is None:
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
        else:
            scope = dict(scope)
        expected_rows = int(scope.get("expected_rows") or 0)
        if expected_rows <= 0:
            raise ProductionParityError(
                "production finalized allocation authority history is empty"
            )
        adapter = provider or _CloneSupabaseProvider(
            clone_url=self.supabase_url, service_role_key=service_role_key
        )
        page_offset = len(adapter.pages)
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
        measured_pages = adapter.pages[page_offset:]
        page_bytes = [int(item["response_bytes"]) for item in measured_pages]
        if not page_bytes or len(attempts) != len(measured_pages):
            raise ProductionParityError(
                "candidate measured source did not produce complete page evidence"
            )
        return {
            **scope,
            "read_rows": len(rows),
            "page_count": len(measured_pages),
            "total_response_bytes": sum(page_bytes),
            "max_page_bytes": max(page_bytes),
            "response_hashes": [item["response_hash"] for item in measured_pages],
            "artifact_count": len(artifacts),
            "adapter": getattr(
                adapter,
                "adapter_name",
                "strict-production-origin-to-disposable-clone",
            ),
        }

    def cleanup(self) -> dict[str, Any]:
        for container in (self.postgrest, self.postgres):
            _run(["docker", "rm", "-f", container], timeout=30)
        _run(["docker", "network", "rm", self.network], timeout=30)
        _run(["docker", "volume", "rm", self.postgres_volume], timeout=30)
        remaining: list[str] = []
        for resource_type, resource_name in (
            ("container", self.postgrest),
            ("container", self.postgres),
            ("network", self.network),
            ("volume", self.postgres_volume),
        ):
            if resource_type == "container":
                command = [
                    "docker",
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"name=^/{resource_name}$",
                ]
            elif resource_type == "network":
                command = [
                    "docker",
                    "network",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{resource_name}$",
                ]
            else:
                command = [
                    "docker",
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{resource_name}$",
                ]
            probe = _run(command, timeout=10)
            if probe.returncode != 0:
                raise ProductionParityError(
                    f"parity {resource_type} cleanup verification failed"
                )
            if probe.stdout.strip():
                remaining.append(f"{resource_type}:{resource_name}")
        if remaining:
            raise ProductionParityError(
                "parity resources remain after cleanup: " + ",".join(remaining)
            )
        return {
            "containers_removed": [self.postgres, self.postgrest],
            "network_removed": self.network,
            "volume_removed": self.postgres_volume,
        }


def _rehearsal_evidence_path(candidate_sha: str) -> Path:
    return Path("/tmp") / f"leadpoet-restart-rehearsal-{candidate_sha}-prepush.json"


def _safe_rehearsal_stage(stage: str) -> bool:
    return any(
        re.fullmatch(pattern, stage) for pattern in SAFE_REHEARSAL_STAGE_PATTERNS
    )


def _rehearsal_error_category(line: str) -> str | None:
    lowered = line.lower()
    categories = (
        ("resource_oom", ("out of memory", "oom", "cannot allocate memory")),
        ("resource_disk", ("no space left", "disk", "free bytes", "storage")),
        ("permission", ("permission denied", "operation not permitted")),
        ("timeout", ("timed out", "timeout", "exceeded its")),
        ("database_shutting_down", ("database system is shutting down",)),
        ("connection_refused", ("connection refused",)),
        ("postgrest", ("postgrest",)),
        ("schema", ("schema",)),
        ("gateway_launcher", ("gateway launcher failed",)),
        ("validator_launcher", ("validator launcher failed",)),
        ("docker_unavailable", ("docker daemon", "docker/containerd")),
    )
    for category, markers in categories:
        if any(marker in lowered for marker in markers):
            return category
    return None


def _prepush_phase_diagnostics(*streams: str) -> list[dict[str, Any]]:
    """Project a fixed phase lifecycle without retaining either raw stream."""

    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    start_counts = {phase: 0 for phase in SAFE_PREPUSH_PHASE_ORDER}
    terminal_counts = {phase: 0 for phase in SAFE_PREPUSH_PHASE_ORDER}
    for stream in streams:
        for match in SAFE_PREPUSH_PHASE_MARKER_RE.finditer(stream):
            phase = match.group(1)
            status = match.group(2)
            duration = float(match.group(3))
            if (
                phase not in SAFE_PREPUSH_PHASES
                or status not in SAFE_PREPUSH_PHASE_STATUSES
                or duration > SAFE_PREPUSH_PHASE_DURATION_MAX_SECONDS
            ):
                continue
            projected = {
                "marker": "prepush_phase",
                "phase": phase,
                "status": status,
                "duration_seconds": round(duration, 3),
            }
            if status == "started":
                if duration != 0.0:
                    continue
                if start_counts[phase] == 0:
                    starts[phase] = projected
                start_counts[phase] = min(2, start_counts[phase] + 1)
            else:
                if terminal_counts[phase] == 0:
                    terminals[phase] = projected
                terminal_counts[phase] = min(2, terminal_counts[phase] + 1)
    diagnostics: list[dict[str, Any]] = []
    for phase in SAFE_PREPUSH_PHASE_ORDER:
        if start_counts[phase] != 1:
            continue
        diagnostics.append(starts[phase])
        if terminal_counts[phase] == 1:
            diagnostics.append(terminals[phase])
        if len(diagnostics) >= SAFE_PREPUSH_PHASE_MARKER_MAX:
            return diagnostics[:SAFE_PREPUSH_PHASE_MARKER_MAX]
    return diagnostics


def _image_build_failure_diagnostics(
    stderr_text: str,
    *,
    exact_image_build_failed: bool,
) -> list[dict[str, Any]]:
    """Classify one failed cold build from its fixed BuildKit lifecycle."""

    if not exact_image_build_failed:
        return []
    current_index = 0
    active_phase: str | None = None
    failed_phase: str | None = None
    marker_count = 0
    invalid = False
    for match in SAFE_IMAGE_BUILD_PHASE_MARKER_RE.finditer(stderr_text):
        marker_count += 1
        if marker_count > SAFE_IMAGE_BUILD_PHASE_MARKER_MAX:
            invalid = True
            break
        phase = match.group(1)
        status = match.group(2)
        if (
            phase not in SAFE_IMAGE_BUILD_PHASES
            or status not in SAFE_IMAGE_BUILD_PHASE_STATUSES
            or failed_phase is not None
            or current_index >= len(SAFE_IMAGE_BUILD_PHASE_ORDER)
        ):
            invalid = True
            continue
        expected_phase = SAFE_IMAGE_BUILD_PHASE_ORDER[current_index]
        if active_phase is None:
            if phase != expected_phase or status != "started":
                invalid = True
                continue
            active_phase = phase
            continue
        if phase != active_phase or status == "started":
            invalid = True
            continue
        if status == "failed":
            failed_phase = phase
        else:
            current_index += 1
        active_phase = None

    if invalid or active_phase is not None:
        phase = "unknown"
        category = "unlocalized"
    elif failed_phase is not None:
        phase = failed_phase
        category = "build_command_failed"
    elif current_index == len(SAFE_IMAGE_BUILD_PHASE_ORDER):
        phase = "image-export-load"
        category = "build_export_or_load_failed"
    else:
        # Cached layers do not replay stdout. A partial or marker-free stream
        # therefore cannot safely distinguish a command, fetch, or export fault.
        phase = "unknown"
        category = "unlocalized"
    return [
        {
            "marker": "image_build_failure",
            "phase": phase,
            "category": category,
        }
    ]


def _rehearsal_output_diagnostics(output_tail: str) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    contract_kinds = {
        "adapter",
        "aws",
        "ctr",
        "curl",
        "docker",
        "getconf",
        "nitro",
        "nsenter",
        "pip",
        "python",
        "python-inline",
        "python-module",
        "sudo",
        "systemctl",
    }
    for raw_line in output_tail.splitlines():
        if len(diagnostics) >= 32:
            break
        line = raw_line.strip()
        if not line or len(line) > 2048 or SECRET_LIKE_DIAGNOSTIC_RE.search(line):
            continue
        projected: dict[str, Any] | None = None
        match = re.fullmatch(
            r"REHEARSAL_FAILURE_DIAGNOSTICS "
            r"component=(gateway|validator|workflow) status=([0-9]{1,3})",
            line,
        )
        if match and 0 <= int(match.group(2)) <= 255:
            projected = {
                "marker": "component_failure",
                "component": match.group(1),
                "status": int(match.group(2)),
            }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_EVIDENCE_NORMALIZATION_FAILED "
                r"phase=(container|host) "
                r"category=(permission|not_found|resource|unknown) "
                r"status=([0-9]{1,3})",
                line,
            )
            if match and 1 <= int(match.group(3)) <= 255:
                projected = {
                    "marker": "evidence_normalization_failure",
                    "phase": match.group(1),
                    "category": match.group(2),
                    "status": int(match.group(3)),
                }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_FIXTURE_GENERATION_FAILED "
                r"category=(permission|not_found|resource|unknown) "
                r"status=([0-9]{1,3})",
                line,
            )
            if match and 1 <= int(match.group(2)) <= 255:
                projected = {
                    "marker": "fixture_generation_failure",
                    "category": match.group(1),
                    "status": int(match.group(2)),
                }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_WORKFLOW_FAILURE_SUMMARY "
                r"failed=([0-9]{1,3}) unexercised=([0-9]{1,3}) "
                r"emitted=([0-9]{1,2}) truncated=([01])",
                line,
            )
            if (
                match
                and int(match.group(1)) <= 512
                and int(match.group(2)) <= 512
                and int(match.group(3)) <= 16
            ):
                projected = {
                    "marker": "workflow_failure_summary",
                    "failed_count": int(match.group(1)),
                    "unexercised_count": int(match.group(2)),
                    "emitted_count": int(match.group(3)),
                    "truncated": match.group(4) == "1",
                }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_WORKFLOW_STAGE_RESULT "
                r"status=(failed|unexercised) "
                r"stage_kind=([a-z_]+) "
                r"stage_id_sha256=([0-9a-f]{64}) "
                r"error_type=([A-Za-z0-9_]+)",
                line,
            )
            if (
                match
                and match.group(2) in SAFE_WORKFLOW_STAGE_KINDS
                and match.group(4) in SAFE_WORKFLOW_PROJECTION_ERROR_TYPES
            ):
                projected = {
                    "marker": "workflow_stage_result",
                    "status": match.group(1),
                    "stage_kind": match.group(2),
                    "stage_id_sha256": match.group(3),
                    "error_type": match.group(4),
                }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_WORKFLOW_DIAGNOSTIC_UNAVAILABLE "
                r"category=(permission|not_found|resource|unknown) "
                r"status=([0-9]{1,3})",
                line,
            )
            if match and 1 <= int(match.group(2)) <= 255:
                projected = {
                    "marker": "workflow_diagnostic_unavailable",
                    "category": match.group(1),
                    "status": int(match.group(2)),
                }
        if projected is None:
            match = re.fullmatch(
                r"REHEARSAL_HTTP_DIAGNOSTIC "
                r"endpoint=(/research-lab/status|/attest) "
                r"status=(curl_failed|[1-5][0-9]{2})",
                line,
            )
            if match:
                projected = {
                    "marker": "http",
                    "endpoint": match.group(1),
                    "status": match.group(2),
                }
        if projected is None and line.startswith("REHEARSAL_STAGE_FAILED_CONTINUING "):
            stage_match = re.search(r"(?:^| )stage=([a-z0-9-]+)(?: |$)", line)
            error_match = re.search(r"(?:^| )error_type=([A-Za-z0-9_]+)(?: |$)", line)
            duration_match = re.search(
                r"(?:^| )duration_seconds=([0-9]+(?:\.[0-9]+)?)(?: |$)",
                line,
            )
            if (
                stage_match
                and _safe_rehearsal_stage(stage_match.group(1))
                and error_match
                and error_match.group(1) in SAFE_REHEARSAL_ERROR_TYPES
            ):
                projected = {
                    "marker": "stage_failure",
                    "stage": stage_match.group(1),
                    "error_type": error_match.group(1),
                }
                if duration_match and float(duration_match.group(1)) <= 3600:
                    projected["duration_seconds"] = round(
                        float(duration_match.group(1)), 3
                    )
        if projected is None:
            match = re.match(r"REHEARSAL CONTRACT ERROR \[([a-z-]+)\]:", line)
            if match and match.group(1) in contract_kinds:
                projected = {
                    "marker": "contract_error",
                    "kind": match.group(1),
                }
        if projected is None and line.startswith("ERROR:"):
            projected = {"marker": "error"}
            category = _rehearsal_error_category(line)
            if category is not None:
                projected["category"] = category
        if projected is not None and projected not in diagnostics:
            diagnostics.append(projected)
    return diagnostics


def _sanitize_workflow_failure_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or type(value.get("available")) is not bool:
        return None
    if value["available"] is False:
        if (
            value.get("category")
            not in {"permission", "not_found", "resource", "unknown"}
            or type(value.get("status")) is not int
            or not 1 <= value["status"] <= 255
        ):
            return None
        return {
            "available": False,
            "category": value["category"],
            "status": value["status"],
        }
    counts = (
        value.get("failed_count"),
        value.get("unexercised_count"),
        value.get("emitted_count"),
    )
    raw_stages = value.get("stages")
    if (
        any(type(item) is not int or not 0 <= item <= 512 for item in counts)
        or type(value.get("truncated")) is not bool
        or not isinstance(raw_stages, list)
        or len(raw_stages) > 16
        or value["emitted_count"] != len(raw_stages)
    ):
        return None
    stages: list[dict[str, Any]] = []
    for item in raw_stages:
        if (
            not isinstance(item, Mapping)
            or item.get("status") not in {"failed", "unexercised"}
            or item.get("stage_kind") not in SAFE_WORKFLOW_STAGE_KINDS
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("stage_id_sha256") or ""),
            )
            is None
            or item.get("error_type")
            not in SAFE_WORKFLOW_PROJECTION_ERROR_TYPES
        ):
            return None
        stages.append(
            {
                "status": item["status"],
                "stage_kind": item["stage_kind"],
                "stage_id_sha256": item["stage_id_sha256"],
                "error_type": item["error_type"],
            }
        )
    return {
        "available": True,
        "failed_count": value["failed_count"],
        "unexercised_count": value["unexercised_count"],
        "emitted_count": value["emitted_count"],
        "truncated": value["truncated"],
        "stages": stages,
    }


def _rehearsal_failure_diagnostics(
    result: subprocess.CompletedProcess[str], *, candidate_sha: str
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "returncode": int(result.returncode),
        "failure_summary_available": False,
    }
    stdout_text = _timeout_stream_text(result.stdout)
    stderr_text = _timeout_stream_text(result.stderr)
    output_tail = "\n".join((stdout_text[-8192:], stderr_text[-8192:]))
    phase_diagnostics = _prepush_phase_diagnostics(stderr_text)
    exact_image_build_failed = any(
        item.get("phase") == "exact-image-build"
        and item.get("status") == "failed"
        for item in phase_diagnostics
    )
    output_diagnostics = [
        # The controller reserves stderr for its phase lifecycle. Stdout may
        # contain arbitrary action output and is never phase authority.
        *phase_diagnostics,
        *_image_build_failure_diagnostics(
            stderr_text,
            exact_image_build_failed=exact_image_build_failed,
        ),
        *_rehearsal_output_diagnostics(output_tail),
    ]
    if output_diagnostics:
        projection["output_markers"] = output_diagnostics
    # The rehearsal emits this fixed-format authority marker to stderr before
    # outer cleanup. Cleanup can exceed the bounded diagnostic tail, so locate
    # the marker in the captured stderr while keeping projected output bounded.
    matches = re.findall(
        r"(?:^|\n)REHEARSAL_BATCH_FAILURE_EVIDENCE ([^\s]+)",
        stderr_text,
    )
    if not matches or re.fullmatch(r"[0-9a-f]{40}", candidate_sha) is None:
        return projection
    try:
        durable_root = Path(matches[-1]).resolve(strict=True)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        expected_prefix = f"leadpoet-rehearsal-failure-{candidate_sha[:12]}-full-path-"
        if (
            durable_root.parent != temp_root
            or not durable_root.name.startswith(expected_prefix)
            or not durable_root.is_dir()
        ):
            return projection
        summary_path = (durable_root / "failure-summary.json").resolve(strict=True)
        if summary_path.parent != durable_root or not summary_path.is_file():
            return projection
        if summary_path.stat().st_size > 131_072:
            return projection
        encoded = summary_path.read_bytes()
        if not encoded or len(encoded) > 131_072:
            return projection
        document = json.loads(encoded.decode("utf-8"))
        if (
            not isinstance(document, Mapping)
            or document.get("candidate_sha") != candidate_sha
            or document.get("status") != "failed"
            or not isinstance(document.get("stages"), list)
            or len(document["stages"]) > 64
        ):
            return projection
        stages: list[dict[str, Any]] = []
        for item in document["stages"]:
            if not isinstance(item, Mapping) or item.get("status") not in {
                "failed",
                "unexercised",
            }:
                continue
            stage = str(item.get("stage") or "")
            if not _safe_rehearsal_stage(stage):
                continue
            sanitized: dict[str, Any] = {
                "stage": stage,
                "status": item["status"],
            }
            error_type = str(item.get("error_type") or "")
            if error_type in SAFE_REHEARSAL_ERROR_TYPES:
                sanitized["error_type"] = error_type
            returncode = item.get("returncode")
            if type(returncode) is int and -255 <= returncode <= 255:
                sanitized["returncode"] = returncode
            duration = item.get("duration_seconds")
            if type(duration) in {int, float} and 0 <= float(duration) <= 3600:
                sanitized["duration_seconds"] = round(float(duration), 3)
            fixture = item.get("fixture_generation_diagnostic")
            if (
                isinstance(fixture, Mapping)
                and fixture.get("category")
                in {"permission", "not_found", "resource", "unknown"}
                and type(fixture.get("status")) is int
                and 1 <= fixture["status"] <= 255
            ):
                sanitized["fixture_generation_diagnostic"] = {
                    "category": fixture["category"],
                    "status": fixture["status"],
                }
            normalization = item.get("evidence_normalization_diagnostics")
            if isinstance(normalization, list) and len(normalization) <= 2:
                safe_normalization = []
                for diagnostic in normalization:
                    if (
                        not isinstance(diagnostic, Mapping)
                        or diagnostic.get("phase") not in {"container", "host"}
                        or diagnostic.get("category")
                        not in {"permission", "not_found", "resource", "unknown"}
                        or type(diagnostic.get("status")) is not int
                        or not 1 <= diagnostic["status"] <= 255
                    ):
                        safe_normalization = []
                        break
                    safe_normalization.append(
                        {
                            "phase": diagnostic["phase"],
                            "category": diagnostic["category"],
                            "status": diagnostic["status"],
                        }
                    )
                if safe_normalization:
                    sanitized["evidence_normalization_diagnostics"] = (
                        safe_normalization
                    )
            workflow = _sanitize_workflow_failure_projection(
                item.get("workflow_failure_projection")
            )
            if workflow is not None:
                sanitized["workflow_failure_projection"] = workflow
            stages.append(sanitized)
        projection.update(
            {
                "failure_summary_available": True,
                "failure_summary_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
                "failed_stage_count": sum(
                    item["status"] == "failed" for item in stages
                ),
                "unexercised_stage_count": sum(
                    item["status"] == "unexercised" for item in stages
                ),
                "stages": stages,
            }
        )
    except (OSError, TypeError, ValueError, UnicodeDecodeError):
        return projection
    return projection


def _timeout_stream_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return ""


def _rehearsal_timeout_diagnostics(
    exc: subprocess.TimeoutExpired, *, candidate_sha: str
) -> dict[str, Any]:
    """Project only the existing secret-safe child diagnostic contract."""

    result = subprocess.CompletedProcess(
        args=(),
        returncode=124,
        stdout=_timeout_stream_text(exc.stdout),
        stderr=_timeout_stream_text(exc.stderr),
    )
    projection = _rehearsal_failure_diagnostics(
        result, candidate_sha=candidate_sha
    )
    projection["parent_watchdog_timeout_seconds"] = int(exc.timeout)
    return projection


def _run_rehearsal(*, base_sha: str, candidate_sha: str) -> dict[str, Any]:
    evidence_path = _rehearsal_evidence_path(candidate_sha)
    evidence_path.unlink(missing_ok=True)
    try:
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
            timeout=FAST_REHEARSAL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        diagnostics = _rehearsal_timeout_diagnostics(
            exc, candidate_sha=candidate_sha
        )
        raise ProductionParityError(
            "candidate-derived N-1 rehearsal parent watchdog timed out: "
            + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        ) from None
    if result.returncode != 0:
        diagnostics = _rehearsal_failure_diagnostics(
            result, candidate_sha=candidate_sha
        )
        raise ProductionParityError(
            "candidate-derived N-1 rehearsal failed: "
            + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        )
    evidence = _load_json(
        evidence_path, description="joined restart rehearsal evidence"
    )
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
    region: str,
    production_gateway_secret_id: str,
) -> dict[str, Any]:
    contract = validate_contract(
        _load_json(contract_path, description="parity contract")
    )
    database = _DockerDatabase(
        candidate_sha=contract["candidate_sha"],
        postgres_image=postgres_image,
        postgrest_image=postgrest_image,
    )
    result: dict[str, Any] = {}
    cleanup: dict[str, Any] | None = None
    primary_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        database.start()
        prerequisites = database.prepare_snapshot_restore()
        restore = restore_snapshot(
            root=ROOT,
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            target_dsn=database.target_dsn,
            production_host=production_host,
        )
        restore_contract = database.verify_snapshot_restore()
        try:
            netuid = int(os.environ.get("BITTENSOR_NETUID", "71"))
        except (TypeError, ValueError) as exc:
            raise ProductionParityError(
                "production parity BITTENSOR_NETUID is invalid"
            ) from exc
        production_read_started = time.monotonic()
        live_provider = _load_production_supabase_read(
            region=region,
            secret_id=production_gateway_secret_id,
            deadline_monotonic=(
                production_read_started + FAST_PRODUCTION_DATA_READ_TIMEOUT_SECONDS
            ),
        )
        activation_row, activation_live_evidence = (
            _read_production_chain_realized_activation(
                provider=live_provider,
                netuid=netuid,
            )
        )
        supabase_url, service_role_key = database.start_postgrest()
        manifest = validate_snapshot_manifest(
            _load_json(manifest_path, description="snapshot manifest")
        )
        if manifest["capture_mode"] != "schema-only":
            raise ProductionParityError(
                "fast database lane requires a schema-only snapshot"
            )
        source_add_acl = restore_schema_only_source_add_acl_contract(
            target_dsn=database.target_dsn,
            production_host=production_host,
            candidate_migrations=contract["migrations"],
        )
        schema = verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": supabase_url,
                "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
                "BITTENSOR_NETUID": os.environ.get("BITTENSOR_NETUID", "71"),
            },
            opener=_StandalonePostgrestSchemaOpener(supabase_url),
            timeout_seconds=FAST_SCHEMA_PREFLIGHT_TIMEOUT_SECONDS,
            chain_realized_activation_authority=activation_row,
        )
        shape = database.shape_evidence(
            service_role_key=service_role_key,
            expected_shape=manifest["database"],
            capture_mode=manifest["capture_mode"],
        )
        weight_input_scale = database.weight_input_scale_evidence(
            service_role_key=service_role_key,
            scope=manifest["database"]["weight_history_scope"],
            provider=live_provider,
        )
        weight_input_scale["bounded_live_read_seconds"] = round(
            time.monotonic() - production_read_started,
            6,
        )
        weight_input_scale["live_read_timeout_seconds"] = (
            FAST_PRODUCTION_DATA_READ_TIMEOUT_SECONDS
        )
        result = {
            "restore": {
                **restore,
                "clone_prerequisites": prerequisites,
                "clone_restore_contract": restore_contract,
                "schema_only_source_add_acl": source_add_acl,
                "production_activation_authority": activation_live_evidence,
            },
            "schema": schema,
            "shape": shape,
            "weight_input_scale": weight_input_scale,
        }
    except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive
        primary_error = exc
    finally:
        try:
            cleanup = database.cleanup()
        except Exception as exc:  # noqa: BLE001 - preserve both failure causes
            cleanup_error = exc
    return {
        "result": result or None,
        "primary_error": primary_error,
        "cleanup": cleanup,
        "cleanup_error": cleanup_error,
    }


def run_fast_lane(
    *,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    ledger_path: Path,
    production_host: str,
    postgres_image: str,
    postgrest_image: str,
    region: str,
    production_gateway_secret_id: str,
) -> dict[str, Any]:
    contract = verify_contract_checkout(
        ROOT, _load_json(contract_path, description="parity contract")
    )
    manifest = validate_snapshot_manifest(
        _load_json(manifest_path, description="snapshot manifest")
    )
    if manifest["capture_mode"] != "schema-only":
        raise ProductionParityError("fast lane requires a bounded schema-only snapshot")
    oracle = validate_historical_oracle(
        _load_json(
            ROOT / "tests/restart_rehearsal/fixtures/august_9_known_good_v2.json",
            description="historical production oracle",
        )
    )
    oracle_stages = set(required_oracle_stage_ids(oracle, lane="fast"))
    if sha256_json(oracle) != contract[
        "historical_oracle_hash"
    ] or not oracle_stages.issubset(CRITICAL_STAGES):
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
        candidate_migration_count = len(snapshot_evidence["migration_delta"])
        required_timeout_seconds = _fast_job_minimum_timeout_seconds(
            candidate_migration_count
        )
        if required_timeout_seconds >= FAST_JOB_OUTER_TIMEOUT_SECONDS:
            raise ProductionParityError(
                "production parity Fast workflow timeout does not exceed its "
                "configured serial inner bounds"
            )
        snapshot_evidence["source_is_candidate_ancestor"] = True
        snapshot_evidence["capture_is_candidate_ancestor"] = True
        snapshot_evidence["candidate_migration_count"] = candidate_migration_count
        snapshot_evidence["minimum_workflow_budget_seconds"] = (
            required_timeout_seconds
        )
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
    started: dict[str, float] = {}
    durations: dict[str, float] = {}
    if snapshot_evidence is not None:
        started["database"] = time.monotonic()
        try:
            results["database"] = _run_database_lane(
                contract_path=contract_path,
                manifest_path=manifest_path,
                archive_path=archive_path,
                production_host=production_host,
                postgres_image=postgres_image,
                postgrest_image=postgrest_image,
                region=region,
                production_gateway_secret_id=production_gateway_secret_id,
            )
        except Exception as exc:
            failures["database"] = exc
        finally:
            durations["database"] = time.monotonic() - started["database"]

    # The database lane owns its cleanup in a finally block. Keep its
    # containers off the bounded hosted runner before the N-1 rehearsal starts,
    # while still exercising the independent rehearsal after a database failure.
    started["rehearsal"] = time.monotonic()
    try:
        results["rehearsal"] = _run_rehearsal(
            base_sha=contract["base_sha"], candidate_sha=contract["candidate_sha"]
        )
    except Exception as exc:
        failures["rehearsal"] = exc
    finally:
        durations["rehearsal"] = time.monotonic() - started["rehearsal"]

    database_duration = durations.get("database", 0.0)
    database_outcome = results.get("database")
    database = (
        database_outcome.get("result")
        if isinstance(database_outcome, Mapping)
        else None
    )
    database_primary_error = (
        database_outcome.get("primary_error")
        if isinstance(database_outcome, Mapping)
        else None
    )
    if isinstance(database, Mapping) and database_primary_error is None:
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
            f"{type(database_primary_error).__name__}: {database_primary_error}"
            if isinstance(database_primary_error, Exception)
            else (
                f"{type(failures['database']).__name__}: {failures['database']}"
                if "database" in failures
                else "production snapshot was unavailable"
            )
        )
        for stage in (
            "snapshot-restore-and-migrations",
            "production-data-shape",
            "production-weight-input-scale",
            "supabase-schema-and-rpc",
        ):
            ledger.record(
                stage,
                status="failed",
                duration_seconds=database_duration,
                reason=reason,
            )

    rehearsal_duration = durations["rehearsal"]
    if "rehearsal" in results:
        rehearsal = results["rehearsal"]
        if (
            rehearsal.get("behavior_contract_hash")
            != contract["behavior_contract_hash"]
        ):
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
                evidence={
                    **common,
                    "restart_invariants": rehearsal["restart_invariants"],
                },
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
                ledger.record(
                    stage,
                    status="failed",
                    duration_seconds=rehearsal_duration,
                    reason=reason,
                )

    cleanup = (
        database_outcome.get("cleanup")
        if isinstance(database_outcome, Mapping)
        else None
    )
    cleanup_error = (
        database_outcome.get("cleanup_error")
        if isinstance(database_outcome, Mapping)
        else None
    )
    cleanup_ok = cleanup is not None and cleanup_error is None
    ledger.record(
        "cleanup",
        status="passed" if cleanup_ok else "failed",
        duration_seconds=0,
        evidence=dict(cleanup or {}),
        reason=(
            ""
            if cleanup_ok
            else (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
                if isinstance(cleanup_error, Exception)
                else "database lane did not prove run-scoped cleanup"
            )
        ),
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
    parser.add_argument("--region", required=True)
    parser.add_argument("--production-gateway-secret-id", required=True)
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
            region=args.region,
            production_gateway_secret_id=args.production_gateway_secret_id,
        )
    except (
        OSError,
        ValueError,
        ProductionParityError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
