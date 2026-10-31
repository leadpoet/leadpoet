"""Arena service: daily benchmark, execution, scoring, and publication."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
import secrets
import threading
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from lab_arena import telemetry, runtime_version, reviewer_runtime, training_input, training_observations, training_coverage, training_eligibility, training_reader, baseline_reasoning
from lab_arena import code_review, code_review_policy, company_judgments, contact_policy, contact_evidence, integrity, intent_details_policy, icp_disclosure, judgment_cache, provider_observations, quality_policy, reasoning_capture, signal_judgments, trajectory, validator_events
from lab_arena import broker as broker_module, chain as chain_module, contracts, credentials as credentials_module, operations, public_dashboard, rewards, scoring, scorer_image_access as scorer_image_access_module, signing, source_bundle, source_disclosure, submission_rate_limit, submission_similarity, verify, weight_state
from leadpoet_verifier.identity.normalization import normalize_url
from leadpoet_canonical.arena_weights import (
    SLOT_ELIGIBILITY_UNTIL_REPLACED,
    accepted_weight_state_body,
    validate_accepted_weight_state,
    verify_accepted_weight_state_signature,
)
from lab_arena.contracts import ArenaContractError, ArenaSignatureError
from lab_arena.output import MAX_OUTPUT_BYTES, OutputInvalid, validate_output_document
from lab_arena.owner_admission import OwnerAdmissionError, resolve_finalized_owner
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    ArenaStoreUnavailable,
    CURRENT_CONFIGURATION_FIELDS,
    PUBLIC_CONFIGURATION_COLUMN,
    PUBLIC_CONFIGURATION_FIELDS,
    PUBLIC_RESULTS_CONFIGURATION_FIELDS,
    hash_lease_token,
    safe_store_error_metadata,
)
from gateway.utils.hotkey_roles import (
    ValidatorIneligible, permitted_validator_uid, validator_uid,
)

logger = logging.getLogger(__name__)

_CompanyEvidenceReads = Dict[Tuple[str, str, int], Dict[str, Any]]

# Final observation uploads already have two 60-second I/O attempts. Allow
# their exact lease to finish after expiry without extending execution authority.
_OBSERVATION_FINALIZATION_GRACE = timedelta(seconds=180)


class _PublicResultEvidenceReads(dict):
    """One validation call's evidence and bounded raw reads; never shared."""

    MAX_IDENTITIES = 128
    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self) -> None:
        super().__init__()
        self.runs: Dict[str, Optional[Dict[str, Any]]] = {}
        self.submissions: Dict[str, Optional[Dict[str, Any]]] = {}
        self.signal_uses: Dict[str, List[Dict[str, Any]]] = {}
        self.signals: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
        self.companies: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
        self.exclusions: Dict[str, List[str]] = {}
        self.objects: Dict[Tuple[str, int], bytes] = {}
        self._bytes = 0

    def _read(self, rows: Dict, identity: Any, load: Callable[[], Any]) -> Any:
        if identity in rows:
            return copy.deepcopy(rows[identity])
        value = load()
        if len(rows) < self.MAX_IDENTITIES and self._bytes < self.MAX_BYTES:
            try:
                size = len(json.dumps(value, separators=(",", ":")).encode("utf-8"))
            except (TypeError, ValueError, OverflowError):
                return value  # Leave malformed values on their original validation path.
            if self._bytes + size <= self.MAX_BYTES:
                rows[identity] = copy.deepcopy(value)
                self._bytes += size
        return value

    def seed(self, rows: Dict, values: Mapping) -> None:
        for identity, value in values.items():
            self._read(rows, identity, lambda value=value: value)

    def company(self, store: ArenaStore, cache_key: str, authority_slot: int) -> Optional[Dict[str, Any]]:
        if type(cache_key) is not str or type(authority_slot) is not int:
            return store.get_company_judgment(cache_key, authority_slot)
        return self._read(self.companies, (cache_key, authority_slot),
                          lambda: store.get_company_judgment(cache_key, authority_slot))

    def run(self, store: ArenaStore, run_id: str) -> Optional[Dict[str, Any]]:
        return self._read(self.runs, run_id, lambda: store.get_run(run_id))

    def submission(self, store: ArenaStore, submission_id: str) -> Optional[Dict[str, Any]]:
        return self._read(self.submissions, submission_id, lambda: store.get_submission(submission_id))

    def uses(self, store: ArenaStore, run_id: str) -> List[Dict[str, Any]]:
        return self._read(self.signal_uses, run_id, lambda: store.get_signal_judgment_uses(run_id))

    def signal(self, store: ArenaStore, cache_key: str, authority_slot: int) -> Optional[Dict[str, Any]]:
        if type(cache_key) is not str or type(authority_slot) is not int:
            return store.get_signal_judgment(cache_key, authority_slot)
        return self._read(self.signals, (cache_key, authority_slot),
                          lambda: store.get_signal_judgment(cache_key, authority_slot))

    def authority_exclusions(self, store: ArenaStore, run_id: str) -> List[str]:
        return self._read(self.exclusions, run_id, lambda: store.signal_authority_exclusions(run_id))

    def object(self, objects: ObjectStore, ref: str, max_bytes: int) -> bytes:
        if type(ref) is not str or type(max_bytes) is not int:
            return objects.get_bounded(ref, max_bytes)
        identity = (ref, max_bytes)
        if identity in self.objects:
            return self.objects[identity]
        value = objects.get_bounded(ref, max_bytes)
        if (len(self.objects) < self.MAX_IDENTITIES
                and self._bytes + len(value) <= self.MAX_BYTES):
            self.objects[identity] = value
            self._bytes += len(value)
        return value

_DEEPLINE_BILLING_BATCH_LIMIT = 96
_DEEPLINE_BILLING_WORKERS = 8
_DEEPLINE_BILLING_PAGE_LIMIT = 20  # Candidate RPC contract.
_DEEPLINE_BILLING_ABANDONED_SLOTS = 4
_DEEPLINE_BILLING_EXECUTE_SLOTS = 1
_DEEPLINE_BILLING_SCORE_SLOTS = 88
_DEEPLINE_BILLING_GENERAL_SLOTS = 3

# Provider error codes that mean "the call was declined before or instead of
# doing upstream work" versus "the call may have happened and we cannot tell"
# versus "our own machinery failed". Telemetry only; dispatch never reads it.
_TELEMETRY_UNCERTAIN_CODES = frozenset({"call_uncertain"})
_TELEMETRY_FAILED_CODES = frozenset({"broker_unavailable", "provider_unavailable"})


_TELEMETRY_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}\.[a-z][a-z0-9_]{0,31}$")
_TELEMETRY_ERROR_CODES = operations.ERROR_CODES | frozenset({
    "broker_unavailable", "budget_refused", "call_refused", "call_uncertain",
    "capability_unavailable",
    "invalid_request", "lease_stale", "miner_credentials_unavailable",
    "miner_provider_not_configured", "model_not_allowed",
    "provider_request_refused", "provider_unavailable",
})


def _telemetry_provider_operation(operation_id: str) -> Tuple[str, str]:
    """Reduce a provider/operation pair to the frozen telemetry vocabulary.

    The operation table, not the broker summary or the operation prefix, owns
    provider attribution. An Exa compatibility operation therefore belongs to
    Deepline while retaining its registered Exa operation name.
    """

    operation = operations.get_operation(operation_id)
    if operation is None or operation.provider not in contracts.PROVIDERS:
        return "unknown", "unknown"
    if _TELEMETRY_OPERATION_RE.fullmatch(operation_id):
        return operation.provider, operation_id
    return operation.provider, "unknown"


def _telemetry_tool(operation_id: str, parameters: Mapping[str, Any]) -> str:
    """Name the Deepline tool behind one provider call, or nothing.

    Every Deepline call leaves as ``deepline.execute``, so the operation id
    alone cannot say which upstream tool answered. This mirrors the runner's
    refusal-guard resolution: a compatibility route carries its pinned tool,
    the native execute route names its tool in the admitted frame. The result
    is checked against the frozen tool table, so a frame that names something
    outside it degrades to "no tool" rather than exporting free text.
    """

    operation = operations.get_operation(operation_id)
    if operation is None or operation.provider != "deepline":
        return telemetry.NO_TOOL
    if operation_id == "deepline.execute":
        tool = str(parameters.get("tool") or "")
    else:
        tool = str(operation.deepline_tool or "")
    return tool if tool in operations.DEEPLINE_TOOLS else telemetry.NO_TOOL


def _provider_telemetry_fields(
    call: Mapping[str, Any], http_status: Any, attempts: int
) -> Dict[str, Any]:
    """Project one broker call summary into bounded telemetry fields.

    Mirrors the existing ``_provider_attempt_summary`` projection in the
    broker: vocabulary members and bounded integers only, never an identity,
    a body, a URL, or a credential.
    """

    reported_error = call.get("error_code")
    error_code = (
        reported_error
        if isinstance(reported_error, str) and reported_error in _TELEMETRY_ERROR_CODES
        else ""
    )
    if not reported_error:
        if call.get("outcome") == "uncertain":
            outcome = "uncertain"
        elif type(http_status) is int and (http_status == 429 or 500 <= http_status <= 599):
            # A settled charge does not make the response delivered to the
            # worker successful. Some adapters return an HTTP failure without
            # an error_code in the accounting summary.
            outcome = "failed"
        elif type(http_status) is int and 400 <= http_status <= 499:
            outcome = "refused"
        else:
            outcome = "ok"
    elif error_code in _TELEMETRY_UNCERTAIN_CODES or call.get("outcome") == "uncertain":
        outcome = "uncertain"
    elif error_code in _TELEMETRY_FAILED_CODES:
        outcome = "failed"
    else:
        outcome = "refused"
    operation_id = call.get("operation_id")
    provider, operation = _telemetry_provider_operation(
        operation_id if isinstance(operation_id, str) else ""
    )
    provider_status = call.get("provider_status")
    observed_status = call.get("observed_provider_http_status")
    if (
        provider_status is None
        and type(observed_status) is int
        and 100 <= observed_status <= 599
    ):
        provider_status = observed_status
    error_class = call.get("error_class")
    cost = call.get("actual_microusd")
    funding_source = call.get("funding_source")
    return {
        "provider": provider,
        "operation": operation,
        "outcome": outcome,
        "funding_source": (
            funding_source
            if funding_source in ("host", "miner_key")
            else telemetry.FUNDING_UNKNOWN
        ),
        "error_code": error_code or telemetry.NO_ERROR,
        **({"error_type": error_class} if (
            isinstance(error_class, str)
            and error_class in (broker_module._SAFE_EXCEPTION_CLASSES | {"Exception"})
        ) else {}),
        "http_status": max(0, min(int(http_status), 999)) if isinstance(http_status, int) and not isinstance(http_status, bool) else 0,
        "provider_status": max(0, min(int(provider_status), 999)) if isinstance(provider_status, int) and not isinstance(provider_status, bool) else 0,
        "attempts": max(1, min(int(attempts), 4)),
        "cost_microusd": max(0, min(int(cost), 1_000_000_000)) if isinstance(cost, int) and not isinstance(cost, bool) else 0,
    }


def _telemetry_run_identity(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Project only database-owned run identity into private host telemetry."""

    return {
        "arena.run_id": run.get("run_id"),
        "arena.round_id": run.get("round_id"),
        "arena.submission_id": run.get("submission_id"),
        "arena.runner_hotkey": run.get("runner_hotkey"),
        "arena.stage": run.get("stage"),
        "arena.icp_position": run.get("icp_position"),
        "arena.attempt": run.get("attempt"),
    }

_TRAJECTORY_WARNING_LOCK = threading.Lock()
_TRAJECTORY_WARNING_LAST: Dict[str, float] = {}


def _warn_trajectory_failure(phase: str, exc: BaseException) -> None:
    """Emit a bounded warning without provider content or exception text."""

    now = time.monotonic()
    with _TRAJECTORY_WARNING_LOCK:
        if now - _TRAJECTORY_WARNING_LAST.get(phase, float("-inf")) < 30.0:
            return
        _TRAJECTORY_WARNING_LAST[phase] = now
    logger.warning(
        "arena trajectory persistence failed phase=%s error_class=%s",
        phase[:32],
        type(exc).__name__[:64],
    )

MODES = ("off", "shadow", "live")
HOT_ROUND_TTL_SECONDS = 2.0
TERMINAL_STATUSES = ("published", "cancelled")
JUDGE_CONTINUATION_EVALUATION_DATE = "2026-10-14"
JUDGE_CONTINUATION_SHADOW_CREATED_AT = datetime(2026, 10, 13, 16, 30, tzinfo=timezone.utc)


def _judge_continuation_enabled(row: Mapping[str, Any]) -> bool:
    if str(row.get("evaluation_date") or "") >= JUDGE_CONTINUATION_EVALUATION_DATE:
        return True
    if ((row.get("configuration_doc") or {}).get("mode") != "shadow"
        or row.get("rewards_enabled") is not False):
        return False
    try:
        return _parse_iso(str(row["created_at"])) >= JUDGE_CONTINUATION_SHADOW_CREATED_AT
    except (KeyError, TypeError, ValueError):
        return False
ACTIVE_ROUND_STATUSES = tuple(
    status for status in contracts.ROUND_STATUSES if status not in TERMINAL_STATUSES
)
SOURCE_UPLOAD_EXPIRES_SECONDS = 900
DEFAULT_BASELINE_SOURCE_URL = "https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz"
FULL_THINKING_BASELINE_URL_RE = re.compile(
    r"https://github\.com/leadpoet/leadpoet-sales-agent/archive/[0-9a-f]{40}\.tar\.gz"
)
DEFAULT_EXECUTION_CAP_MICROUSD = 80_000_000
DEFAULT_COST_PER_COMPANY_MICROUSD = contracts.PER_ICP_QUALIFIED_PAIR_CAP_MICROUSD
DEFAULT_EXECUTION_ICP_CAP_MICROUSD = contracts.PER_ICP_EXECUTION_CAP_MICROUSD
DEFAULT_STAGE_MINUTES = {
    "benchmark": 30,
    "stage_1": 240,
    "stage_1_scoring": 390,
    "stage_2": 180,
    "final_scoring": 390,
}
CANCEL_REASONS = {
    "benchmark_leak": "benchmark_leaked_before_cutoff",
    "benchmark_invalid": "benchmark_data_invalid",
    "capacity": "runner_capacity",
    "scoring": "scoring_window_closed",
    "scoring_incomplete": "scoring_incomplete",
    "publication": "publication_sanitizer_failed",
    "operator": "operator",
}


def _gateway_validator_authorizer(
    hotkey: str, *, network_name: str, netuid: int, metagraph: Any
) -> Tuple[bool, Optional[str]]:
    """Check existing-work identity without rechecking benchmark stake."""

    try:
        validator_uid(
            metagraph, hotkey, netuid=netuid,
            network_name=chain_module.normalize_network_name(network_name),
            require_stake=False,
        )
    except ValidatorIneligible as exc:
        return (False, None) if str(exc) == "runner_hotkey_unregistered" else (True, "miner")
    return True, "validator"


class ServiceError(RuntimeError):
    """A request or transition failed closed."""

    def __init__(self, code: str, status: int = 400, *, source_path: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.source_path = source_path


# ---------------------------------------------------------------------------
# Object store
# ---------------------------------------------------------------------------


class ObjectStore(Protocol):
    def put(self, ref: str, data: bytes) -> None: ...

    def get(self, ref: str) -> bytes: ...

    def get_bounded(self, ref: str, max_bytes: int) -> bytes: ...

    def get_optional_bounded(self, ref: str, max_bytes: int) -> Optional[bytes]: ...

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]: ...


class _PublicResultInFlightReads:
    """Share only overlapping public-result object reads, never completed reads."""

    def __init__(self, source: ObjectStore) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._pending: Dict[Tuple[str, str, Optional[int]], Future[bytes]] = {}

    def _read(self, key: Tuple[str, str, Optional[int]], fetch: Callable[[], bytes]) -> bytes:
        with self._lock:
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                self._pending[key] = pending
        if not owner:
            return pending.result()
        try:
            data = fetch()
        except BaseException as exc:
            with self._lock:
                del self._pending[key]
            pending.set_exception(exc)
            raise
        with self._lock:
            del self._pending[key]
        pending.set_result(data)
        return data

    def get(self, ref: str) -> bytes:
        return self._read(("get", ref, None), lambda: self.source.get(ref))

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        return self._read(
            ("get_bounded", ref, max_bytes),
            lambda: self.source.get_bounded(ref, max_bytes),
        )


class LocalObjectStore:
    """Directory-backed object store for tests and local runs; refs are write-once."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref: str) -> Path:
        if not ref or ref.startswith("/") or ".." in Path(ref).parts:
            raise ArenaContractError("object ref is invalid")
        return self._root / ref

    def put(self, ref: str, data: bytes) -> None:
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish complete bytes atomically and never replace an existing ref.
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".arena-")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(bytes(data))
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != bytes(data):
                    raise ArenaContractError("object ref %s already holds different bytes" % ref)
        finally:
            os.unlink(temporary)

    def get(self, ref: str) -> bytes:
        path = self._path(ref)
        if not path.exists():
            raise ArenaContractError("object store has no object at %s" % ref)
        return path.read_bytes()

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        data = self.get(ref)
        if len(data) > max_bytes:
            raise ArenaContractError("object exceeds source size limit")
        return data

    def get_optional_bounded(self, ref: str, max_bytes: int) -> Optional[bytes]:
        path = self._path(ref)
        if not path.exists():
            return None
        return self.get_bounded(ref, max_bytes)

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]:
        raise ServiceError("source_upload_not_configured", 503)


class S3ObjectStore:
    """Versioned, delete-denied Arena bucket (section 3.1); boto3 imported lazily."""

    def __init__(
        self,
        bucket: str,
        *,
        client: Any = None,
        region_name: Optional[str] = None,
        prefix: str = "",
    ) -> None:
        self._key_prefix = self._validate_key_prefix(prefix)
        if client is None:
            import boto3  # noqa: WPS433
            from botocore.config import Config

            client = boto3.client("s3", region_name=region_name, config=Config(signature_version="s3v4"))
        self._client = client
        self._bucket = bucket

    @staticmethod
    def _validate_key_prefix(value: str) -> str:
        if not isinstance(value, str):
            raise ArenaContractError("object key prefix is invalid")
        if value == "":
            return value
        segments = value.split("/")
        if (
            value != value.strip()
            or any(
                segment in ("", ".", "..") or segment != segment.strip()
                for segment in segments
            )
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ArenaContractError("object key prefix is invalid")
        return value

    def _key(self, ref: str) -> str:
        if not self._key_prefix:
            return ref
        return "%s/%s" % (self._key_prefix, ref)

    def put(self, ref: str, data: bytes) -> None:
        payload = bytes(data)
        key = self._key(ref)
        for attempt in range(2):
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=payload,
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
                return
            except Exception as exc:
                response = getattr(exc, "response", {})
                error = response.get("Error", {}) if isinstance(response, Mapping) else {}
                metadata = response.get("ResponseMetadata", {}) if isinstance(response, Mapping) else {}
                code = str(error.get("Code") or "") if isinstance(error, Mapping) else ""
                status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
                conditional_conflict = code in ("ConditionalRequestConflict", "409") or status == 409
                already_exists = code in ("PreconditionFailed", "412") or status == 412
                if conditional_conflict and attempt == 0:
                    continue
                if not (conditional_conflict or already_exists):
                    raise
                try:
                    existing = self.get(ref)
                except Exception:
                    raise exc
                if existing != payload:
                    raise ArenaContractError("object ref %s already holds different bytes" % ref) from exc
                return
        raise ArenaContractError("object ref %s could not be written safely" % ref)

    def get(self, ref: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=self._key(ref))
        return response["Body"].read()

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        key = self._key(ref)
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body = response["Body"]
        try:
            # GET carries the size of the same object version as its body.
            # Check before reading; the bounded read also covers absent or
            # understated lengths without a separate HEAD round trip.
            if int(response.get("ContentLength") or 0) > int(max_bytes):
                raise ArenaContractError("object exceeds source size limit")
            data = body.read(int(max_bytes) + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if len(data) > int(max_bytes):
            raise ArenaContractError("object exceeds source size limit")
        return data

    def get_optional_bounded(self, ref: str, max_bytes: int) -> Optional[bytes]:
        try:
            return self.get_bounded(ref, max_bytes)
        except Exception as exc:
            response = getattr(exc, "response", {})
            error = response.get("Error", {}) if isinstance(response, Mapping) else {}
            metadata = response.get("ResponseMetadata", {}) if isinstance(response, Mapping) else {}
            code = str(error.get("Code") or "") if isinstance(error, Mapping) else ""
            status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
            if code in {"NoSuchKey", "404", "NotFound"} and status in {None, 404}:
                return None
            raise

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]:
        key = self._key(ref)
        headers = {
            "content-type": str(content_type),
            "content-length": str(int(size_bytes)),
            "if-none-match": "*",
        }
        params = {
            "Bucket": self._bucket,
            "Key": key,
            "ContentType": str(content_type),
            "ContentLength": int(size_bytes),
            "IfNoneMatch": "*",
        }
        if source_content_md5 is not None:
            checksum = contracts.validate_source_content_md5(source_content_md5)
            params["ContentMD5"] = checksum
            headers["content-md5"] = checksum
        url = self._client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=int(expires_seconds),
            HttpMethod="PUT",
        )
        return {"upload_url": url, "upload_headers": headers, "expires_in_seconds": int(expires_seconds)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ChainReads(Protocol):
    def finalized_head(self) -> Any: ...

    def metagraph(self, finalized: bool = True) -> Any: ...

    def current_settlement_epoch(self) -> int: ...

    def accepted_weight_epoch_scope(self) -> Mapping[str, Any]: ...

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]: ...

    def uid_for_hotkey(self, hotkey: str) -> Optional[int]: ...

@dataclass
class RoundDefaults:
    benchmark_icp_count: int = contracts.DEFAULT_BENCHMARK_ICP_COUNT
    promotion_margin: float = contracts.DEFAULT_PROMOTION_MARGIN
    execution_cap_microusd: int = DEFAULT_EXECUTION_CAP_MICROUSD
    cost_per_company_microusd: int = DEFAULT_COST_PER_COMPANY_MICROUSD
    execution_icp_cap_microusd: int = DEFAULT_EXECUTION_ICP_CAP_MICROUSD
    per_icp_cost_policy: bool = False
    # Production opts new rounds into the flat allowance. Existing callers and
    # frozen rounds keep their explicit historical policy.
    flat_per_icp_cost_policy: bool = False
    scoring_cap_microusd: int = 50_000_000
    runner_hotkeys: Tuple[str, ...] = ()
    # The service freezes its public execution ceiling into each new round.
    # RUNNER_SLOT_CEILING is only the public maximum; a runner declaration can
    # never raise this authority cap.
    runner_slot_ceiling: int = contracts.RUNNER_SLOT_CEILING
    parallel_twenty_icp_execution: bool = False
    execution_sequence_from: Optional[str] = None
    checkpoint_deadline_enabled: bool = True
    baseline_hotkey: str = ""
    baseline_source_url: str = DEFAULT_BASELINE_SOURCE_URL
    stage_minutes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_STAGE_MINUTES))
    max_challengers: int = contracts.DEFAULT_MAX_CHALLENGERS  # admitted challengers per round, at most MAX_CHALLENGERS
    # The trusted scorer is resolved once and copied into each round. A service
    # restart therefore cannot change the scorer midway through that round.
    scorer_image_digest: str = "sha256:" + "0" * 64
    scorer_image_reference: str = ""
    # Automatic daily rounds: the UTC hour of each day's submission cutoff, or
    # None to leave round creation to the operator (``lab_arena_admin.py create``).
    daily_cutoff_hour_utc: Optional[int] = None
    # A new round's cutoff lies at least this far ahead so miners can submit.
    integrity_from: Optional[str] = None
    contacts_from: Optional[str] = None
    company_quality_from: Optional[str] = None
    company_judgment_from: Optional[str] = None
    signal_judgment_from: Optional[str] = None
    intent_details_from: Optional[str] = None
    # Opt-in for future intake windows only; existing rounds keep their policy.
    reasoning_capture_from: Optional[str] = None
    training_minimum_icp_coverage_percent: int = training_coverage.NEW_ROUND_PERCENT
    training_maximum_missing_reasoning_percent: int = training_coverage.NEW_MISSING_REASONING_PERCENT
    # Operator declaration for the tested baseline archive. The fetched
    # archive must match before baseline admission in a full-thinking round.
    reasoning_baseline_source_sha256: Optional[str] = None
    reasoning_baseline_source_url: Optional[str] = None
    # Frozen only into newly created integrity rounds. Existing rounds keep
    # their original policy, including after a gateway restart.
    normalize_intent_scale: bool = True
    fit_fp_receipt_authority: bool = True
    benchmark_disclosure_from: Optional[str] = None
    min_submission_hours: int = 6
    # The king's pool as a percent of total emissions (LAB_ARENA_POOL_PERCENT).
    # Announced in every round configuration and carried by every reward basis,
    # so a change applies from the next round and never rewrites a published one.
    pool_percent: int = contracts.LAB_ARENA_POOL_PERCENT
    # Frozen into each round. A later environment change cannot activate an
    # older round that was intentionally published without rewards.
    rewards_enabled: bool = False


@dataclass
class ServiceConfig:
    mode: str
    store: ArenaStore
    object_store: ObjectStore
    # Reward signing is downstream of competition publication. Production
    # supplies a lazy factory; tests may supply a signer directly.
    signer: Optional[signing.ArenaSigner]
    chain: ChainReads
    verify_signature: Callable[[str, str, str], bool]
    daily_icp_source: Callable[..., Mapping[str, Any]]
    banned_hotkeys_source: Callable[[], Iterable[str]]
    broker_factory: Callable[["ArenaService", Mapping[str, Any]], broker_module.Broker]
    defaults: RoundDefaults = field(default_factory=RoundDefaults)
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    network_name: str = "finney"
    netuid: int = 71
    # Optional process-local ownership boundary for isolated one-round hosts.
    # It is deliberately not persisted in a round configuration or schema.
    pinned_round_id: Optional[str] = None
    baseline_source_fetcher: Optional[Callable[[str, int], bytes]] = None
    deepline_catalog_source: Optional[Callable[..., Mapping[str, Any]]] = None
    reasoning_registry_source: Optional[Callable[[], Mapping[str, Any]]] = None
    reward_signer_factory: Optional[Callable[[], signing.ArenaSigner]] = None
    credential_manager: Optional[credentials_module.CredentialManager] = None
    code_reviewer: Optional[Any] = None
    # Only the host publishes accepted source. Miner code never gets GitHub access.
    baseline_promoter_factory: Optional[Callable[[], Any]] = None
    # Supplies plain accepted economic inputs. It must not return receipts,
    # ancestry, release identity, or a preconstructed weight vector.
    accepted_burn_hotkey: str = ""
    # Identity-only test seam for existing leases. New claims and capacity
    # always require the finalized permit/stake snapshot, never this override.
    validator_authorizer: Optional[
        Callable[[str], Tuple[bool, Optional[str]]]
    ] = None
    # Returns temporary access only for the scorer image frozen into a run's
    # round. Generic public registries do not configure this ECR-only path.
    scorer_image_access: Optional[
        Callable[[str, str], Mapping[str, Any]]
    ] = None
    # Observational uploads use a separate bounded client in production.
    training_object_store: Optional[ObjectStore] = None

    def __post_init__(self) -> None:
        if (
            type(self.defaults.benchmark_icp_count) is not int
            or not 2 <= self.defaults.benchmark_icp_count <= contracts.MAX_BENCHMARK_ICP_COUNT
        ):
            raise ServiceError("benchmark_icp_count_invalid", 500)
        if (
            isinstance(self.defaults.promotion_margin, bool)
            or not isinstance(self.defaults.promotion_margin, (int, float))
            or not 0 <= self.defaults.promotion_margin <= 100
        ):
            raise ServiceError("promotion_margin_invalid", 500)
        if (
            isinstance(self.defaults.runner_slot_ceiling, bool)
            or not isinstance(self.defaults.runner_slot_ceiling, int)
            or not 1
            <= self.defaults.runner_slot_ceiling
            <= contracts.RUNNER_SLOT_CEILING
        ):
            raise ServiceError("runner_slot_ceiling_invalid", 500)
        if not isinstance(self.defaults.parallel_twenty_icp_execution, bool):
            raise ServiceError("parallel_twenty_icp_execution_invalid", 500)
        if self.defaults.execution_sequence_from is not None:
            try:
                activation = _parse_iso(self.defaults.execution_sequence_from)
                if activation.tzinfo is None:
                    raise ValueError("missing timezone")
            except (TypeError, ValueError) as exc:
                raise ServiceError("execution_sequence_activation_invalid", 500) from exc
        if self.defaults.integrity_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.integrity_from.replace("Z", "+00:00"))
                if activation.tzinfo is None:
                    raise ValueError("missing timezone")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("integrity_activation_invalid", 500) from exc
        if self.defaults.contacts_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.contacts_from.replace("Z", "+00:00"))
                if activation.tzinfo is None or self.defaults.integrity_from is None:
                    raise ValueError("contact activation requires timezone and integrity activation")
                integrity_activation = datetime.fromisoformat(self.defaults.integrity_from.replace("Z", "+00:00"))
                # Contacts are announced at intake open; integrity activates
                # at the cutoff one day later. Every contact round needs both.
                if activation + timedelta(days=1) < integrity_activation:
                    raise ValueError("contact activation precedes integrity activation")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("contact_activation_invalid", 500) from exc
        if self.defaults.company_judgment_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.company_judgment_from.replace("Z", "+00:00"))
                if self.defaults.integrity_from is None or activation.tzinfo is None:
                    raise ValueError("company judgments require integrity activation")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("company_judgment_activation_invalid", 500) from exc
        if self.defaults.signal_judgment_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.signal_judgment_from.replace("Z", "+00:00"))
                if self.defaults.company_judgment_from is None or activation.tzinfo is None:
                    raise ValueError("signal judgments require company judgments")
                company_activation = datetime.fromisoformat(self.defaults.company_judgment_from.replace("Z", "+00:00"))
                if activation < company_activation:
                    raise ValueError("signal judgments precede company judgments")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("signal_judgment_activation_invalid", 500) from exc
        if self.defaults.company_quality_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.company_quality_from.replace("Z", "+00:00"))
                if activation.tzinfo is None or self.defaults.integrity_from is None:
                    raise ValueError("company quality requires timezone and integrity activation")
                integrity_activation = datetime.fromisoformat(self.defaults.integrity_from.replace("Z", "+00:00"))
                if activation + timedelta(days=1) < integrity_activation:
                    raise ValueError("company quality precedes integrity activation")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("company_quality_activation_invalid", 500) from exc
        if self.defaults.intent_details_from is not None:
            try:
                activation = datetime.fromisoformat(
                    self.defaults.intent_details_from.replace("Z", "+00:00")
                )
                if activation.tzinfo is None or self.defaults.integrity_from is None:
                    raise ValueError(
                        "intent details requires timezone and integrity activation"
                    )
                integrity_activation = datetime.fromisoformat(
                    self.defaults.integrity_from.replace("Z", "+00:00")
                )
                if activation + timedelta(days=1) < integrity_activation:
                    raise ValueError("intent details precedes integrity activation")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("intent_details_activation_invalid", 500) from exc
        if self.defaults.benchmark_disclosure_from is not None:
            try:
                icp_disclosure.parse_activation(
                    self.defaults.benchmark_disclosure_from
                )
            except icp_disclosure.IcpDisclosureError as exc:
                raise ServiceError(
                    "benchmark_disclosure_activation_invalid", 500
                ) from exc
        if self.defaults.reasoning_capture_from is not None:
            try:
                _activation_time(self.defaults.reasoning_capture_from)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ServiceError("reasoning_capture_activation_invalid", 500) from exc
        if self.mode not in MODES:
            raise ServiceError("mode_invalid", 500)
        if self.mode == "off":
            raise ServiceError("mode_off", 500)
        try:
            self.network_name = chain_module.normalize_network_name(self.network_name)
        except Exception as exc:
            raise ServiceError("network_name_invalid", 500) from exc
        if isinstance(self.netuid, bool) or not isinstance(self.netuid, int) or self.netuid < 1:
            raise ServiceError("netuid_invalid", 500)
        if self.pinned_round_id is not None and (
            not isinstance(self.pinned_round_id, str)
            or not self.pinned_round_id
            or self.pinned_round_id != self.pinned_round_id.strip()
        ):
            raise ServiceError("pinned_round_id_invalid", 500)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    timestamp_format = (
        "%Y-%m-%dT%H:%M:%S.%f%z" if "." in value else "%Y-%m-%dT%H:%M:%S%z"
    )
    return datetime.strptime(value, timestamp_format).astimezone(timezone.utc)


def _activation_time(value: str) -> datetime:
    """Read a timezone-aware operator activation timestamp."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("missing timezone")
    return moment


def _partial_baseline_transition_deadline(
    round_row: Mapping[str, Any], *, stage: int,
    execution_incomplete: bool, judge_incomplete: bool,
) -> Optional[datetime]:
    """Use the same proof deadlines as SQL for an incomplete stage-one baseline."""

    configuration = round_row.get("configuration_doc") or {}
    if (
        stage != 1
        or configuration.get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
        or configuration.get("integrity_policy") != "arena_integrity_v1"
        or configuration.get("sourcing_cost_eligibility_policy")
            not in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
    ):
        return None
    schedule = configuration["schedule"]
    deadlines = []
    if execution_incomplete:
        deadlines.append(_parse_iso(schedule["stage_1_close"]))
    if judge_incomplete:
        deadlines.append(_parse_iso(schedule["stage_1_scoring_close"]))
    return max(deadlines) if deadlines else None


def round_id_for_cutoff(cutoff: datetime) -> str:
    return "arena-%s" % cutoff.astimezone(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class _PublicationReads:
    """Validated reads shared only within one publication attempt."""

    service: ArenaService
    round_row: Mapping[str, Any]
    runs: Sequence[Mapping[str, Any]]
    icps: Optional[Sequence[Mapping[str, Any]]] = None
    judges_by_stage: Dict[int, Dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    outputs_by_ref: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    raw_outputs_by_ref: Dict[str, bytes] = field(default_factory=dict)
    qualified_by_submission: Dict[str, Dict[int, int]] = field(default_factory=dict)
    costs_by_position: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)

    def evaluation_icps(self) -> Sequence[Mapping[str, Any]]:
        if self.icps is None:
            self.icps = self.service.evaluation_icps(str(self.round_row["round_id"]))
        return self.icps

    def scoring_outputs(self, stage: int) -> Dict[str, Mapping[str, Any]]:
        if stage not in self.judges_by_stage:
            self.judges_by_stage[stage] = self.service._scoring_outputs(
                str(self.round_row["round_id"]), stage, round_row=self.round_row
            )
        return self.judges_by_stage[stage]

    def output(self, ref: str) -> Dict[str, Any]:
        if ref not in self.outputs_by_ref:
            raw = self.service._objects.get_bounded(ref, MAX_OUTPUT_BYTES)
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise OutputInvalid("accepted output is not valid JSON") from exc
            self.outputs_by_ref[ref] = validate_output_document(document)
            # Retain the exact bytes already verified during this publication
            # attempt. Training export can revalidate them if a later read
            # finds the immutable object temporarily unavailable.
            self.raw_outputs_by_ref[ref] = raw
        return self.outputs_by_ref[ref]

    def qualified_counts(self, submission_id: str, positions: Sequence[int]) -> Dict[int, int]:
        cached = self.qualified_by_submission.setdefault(submission_id, {})
        missing = sorted(set(positions) - cached.keys())
        if missing:
            counted = self.service._qualified_company_counts(
                self.round_row, submission_id, self.runs,
                positions=missing, publication_reads=self,
            )
            cached.update({position: counted.get(position, 0) for position in missing})
        return {position: cached[position] for position in positions}

    def per_icp_cost(self, submission_id: str, position: int) -> Dict[str, Any]:
        key = (submission_id, position)
        if key not in self.costs_by_position:
            qualified = self.qualified_counts(submission_id, (position,))[position]
            self.costs_by_position[key] = self.service._per_icp_cost_eligibility(
                self.round_row, submission_id, position, self.runs,
                qualified=qualified,
            )
        return self.costs_by_position[key]


class _PublicRunLookup(Mapping):
    """Resolve cross-submission scoring sources without scanning the round."""

    def __init__(self, rows: Mapping[str, Any], store: ArenaStore) -> None:
        self._rows = dict(rows)
        self._store = store

    def get(self, key: Any, default: Any = None) -> Any:
        run_id = str(key)
        if run_id not in self._rows:
            self._rows[run_id] = self._store.get_run(run_id)
        row = self._rows[run_id]
        return default if row is None else row

    def __getitem__(self, key: Any) -> Any:
        row = self.get(key)
        if row is None:
            raise KeyError(key)
        return row

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class ArenaService:
    def __init__(self, config: ServiceConfig) -> None:
        self._config = config
        self._store = config.store
        self._objects = config.object_store
        self._public_result_reads = _PublicResultInFlightReads(self._objects)
        self._signer = config.signer
        self._signer_lock = threading.Lock()
        self._clock = config.clock
        self._lock = threading.RLock()
        self._hot_round_lock = threading.Lock()
        self._hot_rounds: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._submission_request_limiter = (
            submission_rate_limit.SubmissionRequestLimiter()
        )
        self._scorer_policy = scoring.build_scorer_policy()
        self._brokers: Dict[Tuple[str, str], broker_module.Broker] = {}
        self._openrouter_reconciliation_after: Dict[Tuple[str, str], int] = {}
        self._deepline_reconciliation_after: Dict[Tuple[str, str], int] = {}
        self._deepline_priority_reconciliation_after: Dict[Tuple[str, str], int] = {}
        self._deepline_score_reconciliation_after: Dict[Tuple[str, str], int] = {}
        self._active_deepline_reconciliations: set[str] = set()
        self._abandoned_billing_run_after: Dict[str, str] = {}
        self._closed_provider_reconciliation_after = 0
        self._completed_scores_lock = threading.Lock()
        self._completed_scores_cache: Dict[str, Dict[str, Any]] = {}
        self._completed_scores_refresh_slots = threading.BoundedSemaphore(2)

    # -- accessors -------------------------------------------------------------

    @property
    def config(self) -> ServiceConfig:
        return self._config

    @property
    def store(self) -> ArenaStore:
        return self._store

    @property
    def scorer_policy(self) -> Dict[str, Any]:
        return dict(self._scorer_policy)

    def signing_key_document(self) -> Dict[str, Any]:
        return signing.signing_key_document(self._reward_signer().public_key_der)

    def now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    def _sign(self, document: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
        return signing.sign_document(self._reward_signer(), document, hash_field=hash_field)

    def _reward_signer(self) -> signing.ArenaSigner:
        """Create the downstream reward signer only when activation needs it."""

        with self._signer_lock:
            if self._signer is None:
                factory = self._config.reward_signer_factory
                if factory is None:
                    raise ServiceError("reward_signer_unavailable", 503)
                self._signer = factory()
            return self._signer

    def _round(self, round_id: str) -> Dict[str, Any]:
        self._require_round_ownership(round_id)
        row = self._store.get_round(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        return self._require_round_mode(row)

    def _round_from_context(
        self, round_id: str, round_context: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Reuse only the exact round already validated for this response."""

        if round_context is not None and round_context.get("round_id") == round_id:
            return round_context
        return self._round(round_id)

    def _public_results_round(self, round_id: str) -> Dict[str, Any]:
        """Keep the published read small; other statuses need the full row."""

        read_published = getattr(self._store, "get_published_results_round", None)
        if read_published is None:
            return self._round(round_id)
        self._require_round_ownership(round_id)
        row = read_published(round_id)
        if row is None:
            # Completed, unpublished submissions use the whole frozen round
            # when their score cache refreshes.
            return self._round(round_id)
        configuration = {
            key: json.loads(row["cfg_" + key])
            for key in PUBLIC_RESULTS_CONFIGURATION_FIELDS
            if row.get("cfg_" + key) is not None
        }
        return self._require_round_mode({**row, "configuration_doc": configuration})

    def _public_submissions_round(self, round_id: str) -> Dict[str, Any]:
        read_public = getattr(self._store, "get_public_submissions_round", None)
        if read_public is None:
            return self._round(round_id)
        self._require_round_ownership(round_id)
        row = read_public(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        configuration = dict(row.get("configuration_doc") or {})
        # The public configuration omits the scorer policy. Keep the existing
        # adapter validation, including absent keys versus explicit JSON null.
        if row.get("cfg_scorer_policy") is not None:
            configuration["scorer_policy"] = json.loads(row["cfg_scorer_policy"])
        return self._require_round_mode({**row, "configuration_doc": configuration})

    def _pinned_round_id(self) -> Optional[str]:
        return getattr(getattr(self, "_config", None), "pinned_round_id", None)

    def _chain_scope(self) -> Tuple[str, int]:
        """Return this process's chain pair, with the legacy production default."""

        config = getattr(self, "_config", None)
        return (
            str(getattr(config, "network_name", "finney")),
            int(getattr(config, "netuid", 71)),
        )

    @staticmethod
    def _require_identifier(value: Any, kind: str) -> None:
        if (
            not isinstance(value, str) or not value or len(value) > 200
            or any(char in ",.()\"'" or ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ServiceError("%s_id_invalid" % kind, 400)

    def _require_round_ownership(self, round_id: str) -> None:
        self._require_identifier(round_id, "round")
        pinned_round_id = self._pinned_round_id()
        if pinned_round_id is not None and round_id != pinned_round_id:
            raise ServiceError("round_scope_mismatch", 409)

    def _pinned_round(self) -> Optional[Dict[str, Any]]:
        pinned_round_id = self._pinned_round_id()
        if pinned_round_id is None:
            return None
        row = self._store.get_round(pinned_round_id)
        if row is None:
            return None
        return self._require_round_mode(row)

    def _require_round_mode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse a round owned by another Arena mode or chain scope."""

        configuration = row.get("configuration_doc") or {}
        try:
            policy_enabled = integrity.enabled(configuration)
            contacts_enabled = contact_policy.enabled(configuration)
            quality_enabled = quality_policy.enabled(configuration)
            if (configuration.get("company_judgment_policy") != (configuration.get("scorer_policy") or {}).get("company_judgment_policy")
                or (configuration.get("company_judgment_policy") and (quality_enabled or contacts_enabled or not policy_enabled))):
                raise ValueError("company judgment scorer policy mismatch")
            company_judgments.enabled(configuration)
            intent_details_enabled = intent_details_policy.enabled(configuration)
            reasoning_capture.enabled(configuration)
            scorer_quality = quality_policy.scorer_enabled(configuration.get("scorer_policy") or {})
            scorer_intent_details = intent_details_policy.scorer_enabled(
                configuration.get("scorer_policy") or {}
            )
        except ValueError as exc:
            raise ServiceError("unsupported_integrity_policy", 409) from exc
        adapter = configuration.get("scorer_policy", {}).get("scoring_adapter_version")
        if contacts_enabled != (adapter == contact_policy.SCORING_ADAPTER) or policy_enabled != contact_policy.integrity_adapter(adapter):
            raise ServiceError("integrity_scorer_policy_mismatch", 409)
        if quality_enabled != scorer_quality or (quality_enabled and not policy_enabled):
            raise ServiceError("company_quality_scorer_policy_mismatch", 409)
        if intent_details_enabled != scorer_intent_details or (
            intent_details_enabled and not policy_enabled
        ):
            raise ServiceError("intent_details_scorer_policy_mismatch", 409)
        try:
            icp_disclosure.configured_policy(row)
        except icp_disclosure.IcpDisclosureError as exc:
            raise ServiceError("benchmark_disclosure_policy_invalid", 503) from exc
        if configuration.get("mode") != self._config.mode:
            raise ServiceError("round_mode_mismatch", 409)
        # Rows created before schema 189 had no explicit chain pair. They are
        # permanently interpreted as the original Finney/netuid 71 scope.
        network_name = configuration.get("network_name", "finney")
        netuid = configuration.get("netuid", 71)
        if (network_name, netuid) != self._chain_scope():
            raise ServiceError("round_network_mismatch", 409)
        return row

    def startup_checks(self) -> Dict[str, Any]:
        """Check competition database and object-store dependencies.

        Checks the database role, every Arena table and function grant, the
        object store. Reward signing and epoch cutover are lazy downstream
        dependencies and cannot block service startup or publication.
        """

        identity = self._store.require_service_role()
        if getattr(self._config, "deepline_catalog_source", None) is not None:
            try:
                self._store.deepline_catalog_schema()
                self._store.deepline_response_schema()
            except ArenaStoreError as exc:
                raise ServiceError("deepline_catalog_schema_unavailable", 500) from exc
        try:
            schema = self._store._transport.rpc("lab_arena_schema_version_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("function_unavailable:lab_arena_schema_version_v1", 500) from exc
        expected_schema = "leadpoet.lab_arena.schema_version.v1"
        schema_version = schema.get("version") if isinstance(schema, Mapping) else None
        if (
            not isinstance(schema, Mapping)
            or schema.get("schema_version") != expected_schema
            or schema_version != 197
        ):
            raise ServiceError("arena_schema_version_invalid", 500)
        for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
            try:
                self._store._transport.select(
                    table,
                    limit=1,
                    columns="icp_set_date" if table == "lab_arena_rounds" else "*",
                )
            except ArenaStoreError as exc:
                raise ServiceError("table_unavailable:%s" % table, 500) from exc
        try:
            self._store.code_review_schema()
        except ArenaStoreError as exc:
            raise ServiceError("code_review_schema_unavailable", 500) from exc
        try:
            self._store.submission_similarity_schema()
        except ArenaStoreError as exc:
            raise ServiceError("submission_similarity_schema_unavailable", 500) from exc
        try:
            self._store.validator_scoring_authority_schema()
        except ArenaStoreError as exc:
            raise ServiceError(
                "validator_scoring_authority_schema_unavailable", 500
            ) from exc
        champion_funding_schema = getattr(
            self._store, "champion_funding_schema", None
        )
        if callable(champion_funding_schema):
            try:
                champion_funding_schema()
            except ArenaStoreError as exc:
                raise ServiceError(
                    "champion_funding_schema_unavailable", 500
                ) from exc
        # Every service function must exist and be granted: a missing round is the
        # expected structured failure; a permission or undefined-function error is not.
        for function, params in (
            (
                "lab_arena_commit_round_v2",
                {
                    "p_round_id": "arena-0000-00-00",
                    "p_participants": [],
                    "p_benchmark_ref": "probe",
                    "p_evaluation_date": "2000-01-02",
                    "p_icp_set_date": "2000-01-01",
                    "p_scorer_image_digest": "sha256:" + "0" * 64,
                    "p_scorer_image_reference": "probe@sha256:" + "0" * 64,
                },
            ),
            ("lab_arena_expire_leases", {"p_round_id": "arena-0000-00-00"}),
            ("lab_arena_close_stage", {"p_round_id": "arena-0000-00-00", "p_stage": 1}),
            ("lab_arena_cancel_round", {"p_round_id": "arena-0000-00-00", "p_reason": "startup-probe"}),
        ):
            try:
                self._store._transport.rpc(function, params)
            except ArenaStoreError as exc:
                if "lab_arena_round_missing" not in str(exc):
                    raise ServiceError("function_unavailable:%s" % function, 500) from exc
        # The aggregate cost RPC was added after the legacy schema-version
        # marker. Probe its service grant and its fail-closed missing-row path
        # explicitly so an old database cannot start the new budget service.
        cost_function = "lab_arena_submission_costs"
        try:
            self._store._transport.rpc(
                cost_function,
                {"p_submission_id": "__arena_budget_probe__"},
            )
        except ArenaStoreError as exc:
            if "lab_arena_submission_missing" not in str(exc):
                raise ServiceError(
                    "function_unavailable:%s" % cost_function, 500
                ) from exc
        else:
            raise ServiceError("function_probe_invalid:%s" % cost_function, 500)
        try:
            self._store.successful_call_cost_schema()
        except ArenaStoreError as exc:
            raise ServiceError(
                "successful_call_cost_schema_unavailable", 500
            ) from exc
        if (getattr(self._config.defaults, "per_icp_cost_policy", False)
                or getattr(self._config.defaults, "flat_per_icp_cost_policy", False)):
            try:
                self._store.per_icp_cost_schema()
            except ArenaStoreError as exc:
                raise ServiceError("per_icp_cost_schema_unavailable", 500) from exc
            try:
                self._store.infrastructure_retry_cost_schema()
            except ArenaStoreError as exc:
                raise ServiceError("infrastructure_retry_cost_schema_unavailable", 500) from exc
        try:
            self._store.submission_replacement_schema()
        except ArenaStoreError as exc:
            raise ServiceError("submission_replacement_schema_unavailable", 500) from exc
        try:
            self._store.dynamic_benchmark_schema()
        except ArenaStoreError as exc:
            raise ServiceError("dynamic_benchmark_schema_unavailable", 500) from exc
        if (
            getattr(
                self._config.defaults, "parallel_twenty_icp_execution", False
            )
            is True
            or getattr(
                self._config.defaults, "execution_sequence_from", None
            ) is not None
        ):
            try:
                parallel_schema = self._store.parallel_execution_schema(
                    self._config.defaults.runner_slot_ceiling
                )
            except ArenaStoreError as exc:
                raise ServiceError(
                    "parallel_execution_schema_unavailable", 500
                ) from exc
            if int(parallel_schema["max_parallel_icps"]) < int(
                self._config.defaults.runner_slot_ceiling
            ):
                raise ServiceError(
                    "parallel_execution_schema_unavailable", 500
                )
        today = int(self.now().strftime("%Y%m%d"))
        source = self._config.daily_icp_source(set_id=today, active_at=self.now())
        if not isinstance(source, Mapping) or source.get("status") not in (
            "ready",
            "unavailable",
        ):
            raise ServiceError("daily_icp_source_invalid", 500)
        probe_ref = "arena/_startup/object-store.json"
        probe_bytes = contracts.canonical_json({"probe": "lab_arena_object_store_v1"}).encode("utf-8")
        try:
            self._objects.put(probe_ref, probe_bytes)
            if self._objects.get(probe_ref) != probe_bytes:
                raise ServiceError("object_store_mismatch", 500)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("object_store_unavailable", 500) from exc
        current = self.current_round()
        if getattr(getattr(self._config, "defaults", None), "integrity_from", None) or (current and integrity.enabled(current.get("configuration_doc") or {})):
            self._require_integrity_schema()
        if self._config.defaults.contacts_from or (current and contact_policy.enabled(current.get("configuration_doc") or {})):
            self._require_contact_schema()
        if (getattr(self._config.defaults, "company_quality_from", None)
            or getattr(self._config.defaults, "company_judgment_from", None)
            or (current and company_judgments.enabled(current.get("configuration_doc") or {}))):
            self._require_company_quality_schema()
        if (getattr(self._config.defaults, "company_judgment_from", None)
            or (current and (current.get("configuration_doc") or {}).get("company_judgment_policy"))):
            try:
                self._store._transport.rpc("lab_arena_commit_round_v4", {
                    "p_round_id": "arena-0000-00-00", "p_participants": [], "p_benchmark_ref": "probe",
                    "p_evaluation_date": "2000-01-02", "p_icp_set_date": "2000-01-01",
                    "p_scorer_image_digest": "sha256:" + "0" * 64,
                    "p_scorer_image_reference": "probe@sha256:" + "0" * 64,
                    "p_deepline_catalog": None, "p_company_judgment_policy": company_judgments.POLICY,
                })
            except ArenaStoreError as exc:
                if "lab_arena_round_missing" not in str(exc):
                    raise ServiceError("company_judgment_schema_unavailable", 503) from exc
        return {
            "database_identity": identity,
            "schema_version": int(schema_version),
            "scoring_adapter_version": self._scorer_policy["scoring_adapter_version"],
            "current_round": current["round_id"] if current else None,
        }

    # -- round creation (section 5.1) ----------------------------------------

    def _require_integrity_schema(self) -> None:
        try:
            result = self._store._transport.rpc("lab_arena_integrity_schema_v1", {})
            promotion = self._store._transport.rpc(
                "lab_arena_twenty_icp_promotion_schema_v1", {}
            )
        except ArenaStoreError as exc:
            raise ServiceError("integrity_schema_unavailable", 503) from exc
        if not isinstance(result, Mapping) or result.get("schema_version") != "leadpoet.lab_arena.integrity_schema.v1" or result.get("version") != 213:
            raise ServiceError("integrity_schema_invalid", 503)
        if (
            not isinstance(promotion, Mapping)
            or promotion.get("schema_version")
            != "leadpoet.lab_arena.twenty_icp_promotion_schema.v1"
            or promotion.get("version") != 251
        ):
            raise ServiceError("integrity_schema_invalid", 503)
        try:
            baseline_cost = self._store._transport.rpc(
                "lab_arena_baseline_cost_eligibility_schema_v1", {}
            )
        except ArenaStoreError as exc:
            raise ServiceError(
                "baseline_cost_eligibility_schema_unavailable", 503
            ) from exc
        if (
            not isinstance(baseline_cost, Mapping)
            or baseline_cost.get("schema_version")
            != "leadpoet.lab_arena.baseline_cost_eligibility_schema.v1"
            or baseline_cost.get("version") != 254
            or baseline_cost.get("policy") != "zero_baseline_cost_ineligible_v1"
        ):
            raise ServiceError("baseline_cost_eligibility_schema_invalid", 503)

    def _require_contact_schema(self) -> None:
        try:
            result = self._store._transport.rpc("lab_arena_contact_schema_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("contact_schema_unavailable", 503) from exc
        if not isinstance(result, Mapping) or result.get("schema_version") != "leadpoet.lab_arena.contact_schema.v1" or result.get("version") != 215:
            raise ServiceError("contact_schema_invalid", 503)

    def _require_company_quality_schema(self) -> None:
        try:
            result = self._store._transport.rpc("lab_arena_company_quality_schema_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("company_quality_schema_unavailable", 503) from exc
        if not isinstance(result, Mapping) or result.get("schema_version") != "leadpoet.lab_arena.company_quality_schema.v1" or result.get("version") != 1:
            raise ServiceError("company_quality_schema_invalid", 503)

    def build_schedule(self, cutoff: datetime) -> Dict[str, str]:
        """Build the round's cutoff and absolute timeout budget.

        ``submission_cutoff`` is the only minimum start time.  The stage start
        fields mark nominal capacity-budget boundaries.  The close fields are
        deadlines for incomplete work.  Ready phases advance without waiting
        for a nominal boundary.
        """

        minutes = self._config.defaults.stage_minutes
        cutoff = cutoff.astimezone(timezone.utc)
        benchmark_deadline = cutoff + timedelta(minutes=minutes["benchmark"])
        stage_1_start = benchmark_deadline + timedelta(seconds=1)
        stage_1_close = stage_1_start + timedelta(minutes=minutes["stage_1"])
        stage_1_scoring_close = stage_1_close + timedelta(minutes=minutes["stage_1_scoring"])
        stage_2_start = stage_1_scoring_close + timedelta(seconds=1)
        stage_2_close = stage_2_start + timedelta(minutes=minutes["stage_2"])
        final_scoring_close = stage_2_close + timedelta(minutes=minutes["final_scoring"])
        return {
            "submission_open": _iso(cutoff - timedelta(days=1)),
            "submission_cutoff": _iso(cutoff),
            "benchmark_deadline": _iso(benchmark_deadline),
            "stage_1_start": _iso(stage_1_start),
            "stage_1_close": _iso(stage_1_close),
            "stage_1_scoring_close": _iso(stage_1_scoring_close),
            "stage_2_start": _iso(stage_2_start),
            "stage_2_close": _iso(stage_2_close),
            "final_scoring_close": _iso(final_scoring_close),
            "publication_deadline": _iso(final_scoring_close + timedelta(seconds=1)),
        }

    def runner_settings(self) -> Tuple[List[str], List[str]]:
        banned = sorted(set(str(item) for item in self._config.banned_hotkeys_source()))
        runners = sorted(set(self._config.defaults.runner_hotkeys))
        for hotkey in runners:
            if hotkey in banned:
                raise ServiceError("runner_banned", 500)
        snapshot = self._benchmark_snapshot()
        eligible = []
        for hotkey in runners:
            try:
                self._benchmark_validator_uid(snapshot, hotkey)
            except ServiceError as exc:
                if exc.status != 403:
                    raise
            else:
                eligible.append(hotkey)
        if not eligible:
            raise ServiceError("daily_runner_capacity_insufficient", 503)
        # Require an eligible planned validator for new rounds. Runtime claims
        # still check each validator independently. This set grants no authority.
        return eligible, banned

    def _freeze_deepline_catalog(self, configuration: Dict[str, Any]) -> None:
        source = getattr(self._config, "deepline_catalog_source", None)
        if source is None or configuration.get("deepline_catalog") is not None:
            return
        if configuration.get("sourcing_cost_eligibility_policy") not in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES:
            return
        from lab_arena.deepline_catalog import validate_catalog
        configuration["deepline_catalog"] = validate_catalog(source(
            allow_people=configuration.get("contact_policy") == "contacts_v1",
        ))
        configuration["call_quotas"] = {**configuration["call_quotas"], "deepline": 0}
        configuration["scoring_call_quotas"] = {**configuration["scoring_call_quotas"], "deepline": 0}

    def _existing_round_configuration_for_create(self, row: Dict[str, Any]) -> Dict[str, Any]:
        configuration = dict(self._require_round_mode(row).get("configuration_doc") or {})
        if configuration.get("reasoning_activation_status") == "not_ready":
            raise ServiceError(
                "reasoning_capture_activation_not_ready:%s" % (
                    configuration.get("reasoning_activation_reason") or "unknown"
                ), 503,
            )
        return configuration

    def create_round(self, cutoff: datetime, *, round_id: Optional[str] = None) -> Dict[str, Any]:
        defaults = self._config.defaults
        round_id = round_id or round_id_for_cutoff(cutoff)
        self._require_round_ownership(round_id)
        # Frozen rounds do not depend on today's registry or provider catalog.
        existing = self._store.get_round(round_id)
        if existing is not None:
            return self._existing_round_configuration_for_create(existing)
        runner_hotkeys, banned_hotkeys = self.runner_settings()
        checkpoint_policy = contracts.DEFAULT_CHECKPOINT_DEADLINE_POLICY
        checkpoint_wall, checkpoint_lease = contracts.CHECKPOINT_DEADLINE_PROFILES[
            checkpoint_policy
        ]
        document = {
            "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
            "round_id": round_id,
            "mode": self._config.mode,
            "network_name": self._config.network_name,
            "netuid": self._config.netuid,
            "rewards_enabled": bool(defaults.rewards_enabled and self._config.mode == "live"),
            "schedule": self.build_schedule(cutoff),
            "stage_1_icp_count": (defaults.benchmark_icp_count + 1) // 2,
            "stage_2_icp_count": defaults.benchmark_icp_count // 2,
            "promotion_margin": defaults.promotion_margin,
            "finalist_count": contracts.FINALIST_COUNT,
            "max_challengers": int(defaults.max_challengers),
            "runner_slot_ceiling": int(defaults.runner_slot_ceiling),
            "max_attempts_per_assignment": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
            "lease_ttl_seconds": (
                checkpoint_lease
                if defaults.checkpoint_deadline_enabled else contracts.LEASE_TTL_SECONDS
            ),
            "companies_per_icp": 5,
            "providers": list(contracts.PROVIDERS),
            "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
            "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
            "icp_wall_clock_seconds": (
                checkpoint_wall
                if defaults.checkpoint_deadline_enabled else contracts.ICP_WALL_CLOCK_SECONDS
            ),
            "scoring_wall_clock_seconds": contracts.SCORING_WALL_CLOCK_SECONDS,
            "scorer_policy": self._scorer_policy,
            "execution_cap_microusd": defaults.execution_cap_microusd,
            "cost_per_company_microusd": (
                contracts.PER_ICP_EXECUTION_CAP_MICROUSD
                if defaults.flat_per_icp_cost_policy else defaults.cost_per_company_microusd
            ),
            "sourcing_cost_eligibility_policy": (
                contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
                if defaults.flat_per_icp_cost_policy
                else contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
                if defaults.per_icp_cost_policy
                else contracts.SUCCESSFUL_CALLS_COST_POLICY
            ),
            "scoring_cap_microusd": defaults.scoring_cap_microusd,
            "scorer_image_digest": defaults.scorer_image_digest,
            "scorer_image_reference": defaults.scorer_image_reference,
            "baseline_hotkey": defaults.baseline_hotkey,
            "baseline_source_url": defaults.baseline_source_url,
            "runner_hotkeys": runner_hotkeys,
            "banned_hotkeys": banned_hotkeys,
            "reward_constants": rewards.reward_constants_document(int(defaults.pool_percent)),
        }
        baseline_first = (
            defaults.execution_sequence_from is not None
            and cutoff >= _parse_iso(defaults.execution_sequence_from)
        )
        if baseline_first:
            document["execution_sequence_policy"] = (
                contracts.BASELINE_SCORED_FIRST_POLICY
            )
        elif defaults.parallel_twenty_icp_execution:
            document["parallel_twenty_icp_execution"] = True
        if defaults.per_icp_cost_policy or defaults.flat_per_icp_cost_policy:
            document["execution_icp_cap_microusd"] = (
                defaults.execution_icp_cap_microusd
            )
            document["infrastructure_retry_cost_policy"] = (
                contracts.INFRASTRUCTURE_RETRY_COST_POLICY
            )
        if defaults.checkpoint_deadline_enabled:
            document["checkpoint_deadline_policy"] = checkpoint_policy
        if defaults.integrity_from is not None and cutoff >= datetime.fromisoformat(defaults.integrity_from.replace("Z", "+00:00")):
            self._require_integrity_schema()
            document["integrity_policy"] = integrity.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(scoring_adapter_version=integrity.SCORING_ADAPTER)
        if (
            defaults.contacts_from is not None
            and _parse_iso(document["schedule"]["submission_open"])
            >= datetime.fromisoformat(defaults.contacts_from.replace("Z", "+00:00"))
        ):
            if not integrity.enabled(document):
                raise ServiceError("contact_policy_requires_integrity", 503)
            self._require_contact_schema()
            document["contact_policy"] = contact_policy.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(scoring_adapter_version=contact_policy.SCORING_ADAPTER)
        if defaults.company_quality_from is not None and _parse_iso(document["schedule"]["submission_open"]) >= datetime.fromisoformat(defaults.company_quality_from.replace("Z", "+00:00")):
            if not integrity.enabled(document):
                raise ServiceError("company_quality_requires_integrity", 503)
            self._require_company_quality_schema()
            document["company_quality_policy"] = quality_policy.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(
                scoring_adapter_version=document["scorer_policy"]["scoring_adapter_version"],
                company_quality=True,
            )
        if (
            defaults.intent_details_from is not None
            and _parse_iso(document["schedule"]["submission_open"])
            >= datetime.fromisoformat(
                defaults.intent_details_from.replace("Z", "+00:00")
            )
        ):
            if not integrity.enabled(document):
                raise ServiceError("intent_details_policy_requires_integrity", 503)
            document["intent_details_policy"] = intent_details_policy.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(
                scoring_adapter_version=document["scorer_policy"][
                    "scoring_adapter_version"
                ],
                company_quality=quality_policy.enabled(document),
                intent_details=True,
                provider_observation_handoff=True,
            )
        self._activate_company_judgments(document)
        if (defaults.signal_judgment_from is not None
            and document.get("company_judgment_policy") == company_judgments.POLICY
            and _parse_iso(document["schedule"]["submission_open"])
            >= datetime.fromisoformat(defaults.signal_judgment_from.replace("Z", "+00:00"))):
            document["scorer_policy"] = dict(document["scorer_policy"],
                signal_judgment_policy="signal_judgment_v1")
        if defaults.normalize_intent_scale and integrity.enabled(document):
            document["scorer_policy"] = dict(document["scorer_policy"], env_bindings={
                **document["scorer_policy"]["env_bindings"],
                contracts.SCORE_NORMALIZATION_BINDING:
                    contracts.AVAILABLE_INTENT_CAP_NORMALIZATION,
            })
        if defaults.fit_fp_receipt_authority and integrity.enabled(document):
            document["scorer_policy"] = dict(document["scorer_policy"], env_bindings={
                **document["scorer_policy"]["env_bindings"],
                contracts.FIT_FP_CLASSIFICATION_BINDING:
                    contracts.FIT_RECEIPT_AUTHORITY,
            })
        if (
            defaults.benchmark_disclosure_from is not None
            and cutoff >= icp_disclosure.parse_activation(
                defaults.benchmark_disclosure_from
            )
        ):
            document["benchmark_disclosure_policy"] = (
                icp_disclosure.CUTOFF_PUBLIC_POLICY
            )
        if (defaults.reasoning_capture_from is not None
            and _parse_iso(document["schedule"]["submission_open"])
            >= _activation_time(defaults.reasoning_capture_from)):
            source = getattr(self._config, "reasoning_registry_source", None)
            baseline_digest = defaults.reasoning_baseline_source_sha256
            unready_reason = None
            try:
                registry = reasoning_capture.validate_registry(
                    source() if source is not None else reasoning_capture.load_registry()
                )
                if not registry.get("approved_routes") or not isinstance(registry.get("baseline_route"), Mapping):
                    unready_reason = "registry_unavailable"
            except Exception as exc:
                registry = None
                unready_reason = "registry_unavailable"
                logger.warning("arena_full_thinking_registry_unavailable round_id=%s error=%s",
                               round_id, type(exc).__name__)
            if unready_reason is None and (
                not isinstance(baseline_digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", baseline_digest) is None
                or not isinstance(defaults.reasoning_baseline_source_url, str)
                or FULL_THINKING_BASELINE_URL_RE.fullmatch(defaults.reasoning_baseline_source_url) is None
            ):
                unready_reason = "baseline_unready"
            if unready_reason is None and not defaults.flat_per_icp_cost_policy:
                unready_reason = "flat_policy_required"
            if unready_reason is None:
                document["reasoning_capture_policy"] = reasoning_capture.POLICY
                document[training_coverage.SETTING] = defaults.training_minimum_icp_coverage_percent
                document[training_coverage.MISSING_REASONING_SETTING] = defaults.training_maximum_missing_reasoning_percent
                document["reasoning_route_registry"] = registry
                document["reasoning_baseline_source_sha256"] = baseline_digest
                document["baseline_source_url"] = defaults.reasoning_baseline_source_url
                per_icp_cap = contracts.FULL_THINKING_PER_ICP_CAP_MICROUSD
                document["execution_icp_cap_microusd"] = per_icp_cap
                document["cost_per_company_microusd"] = per_icp_cap
                document["execution_cap_microusd"] = (
                    per_icp_cap * contracts.benchmark_icp_count(document)
                )
            else:
                logger.warning("arena_full_thinking_not_ready round_id=%s reason=%s",
                               round_id, unready_reason)
                raise ServiceError(
                    "reasoning_capture_activation_not_ready:%s" % unready_reason, 503,
                )
        self._freeze_deepline_catalog(document)
        configuration = contracts.validate_round_configuration(document)
        result = self._store.create_round(round_id, configuration)
        if result.get("status") not in ("created", "existing"):
            raise ServiceError("round_create_failed", 500)
        if result.get("status") == "existing":
            return self._existing_round_configuration_for_create(self._round(round_id))
        return configuration

    def ensure_daily_round(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Create the next daily round when no round is open for submissions.

        Rounds overlap: the day's round runs its benchmark while the next
        round is already open, so miners can always submit. Every signed
        request names its round, and the driver advances every round that is
        not published or cancelled. The new round's cutoff is the next
        configured UTC hour at least ``min_submission_hours`` ahead; a date
        whose round already exists (published or cancelled that day) moves to
        the next day, because a round id is its cutoff date. Idempotent: a
        second call finds the round it created.
        """

        defaults = self._config.defaults
        # A round made after a late startup can have tomorrow's submission
        # opening. It must block another creation without being advertised to
        # miners or accepting work before that opening.
        open_round = self.open_round(include_future=True)
        if open_round is not None:
            return {"status": "existing", "round_id": open_round["round_id"], "round_status": open_round["status"]}
        if defaults.daily_cutoff_hour_utc is None:
            return {"status": "disabled"}
        hour = int(defaults.daily_cutoff_hour_utc)
        if not 0 <= hour <= 23:
            raise ServiceError("daily_cutoff_hour_invalid", 500)
        moment = (now or self.now()).astimezone(timezone.utc)
        earliest = moment + timedelta(hours=max(0, int(defaults.min_submission_hours)))
        cutoff = earliest.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cutoff < earliest:
            cutoff += timedelta(days=1)
        for _ in range(14):
            if self._store.get_round(round_id_for_cutoff(cutoff)) is None:
                created = self.create_round(cutoff)
                return {"status": "created", "round_id": created["round_id"], "cutoff": _iso(cutoff)}
            cutoff += timedelta(days=1)
        raise ServiceError("daily_round_dates_exhausted", 500)

    def current_round(self) -> Optional[Dict[str, Any]]:
        """The newest round that is not published or cancelled (operator status)."""

        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and row["status"] not in TERMINAL_STATUSES else None
        # Scan ids and statuses only; a full row can be large at hundreds of participants.
        network_name, netuid = self._chain_scope()
        for row in self._store.list_rounds(
            statuses=ACTIVE_ROUND_STATUSES,
            mode=self._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=20,
            columns="round_id,status,created_at,configuration_doc",
        ):
            if row["status"] not in TERMINAL_STATUSES and (row.get("configuration_doc") or {}).get("mode") == self._config.mode:
                return self._round(row["round_id"])
        return None

    @staticmethod
    def _public_current_round(row: Mapping[str, Any]) -> Dict[str, Any]:
        if "configuration_doc" in row:
            return dict(row)
        # Text casts preserve explicit JSON null while SQL NULL means absent.
        configuration = {
            key: json.loads(row["cfg_" + key])
            for key in CURRENT_CONFIGURATION_FIELDS + ("baseline_hotkey",)
            if row.get("cfg_" + key) is not None
        }
        return {**row, "configuration_doc": configuration}

    def _frozen_unfinished_judgment(self, row: Mapping[str, Any]) -> bool:
        """Prove an old score is unfinished using only compact run authority."""
        if row["status"] != "scored":
            return False
        # An absent or invalid evaluation day is not proof that this round
        # predates the continuation rollout.
        try:
            self._evaluation_day(row)
        except ServiceError:
            return False
        if _judge_continuation_enabled(row):
            return False
        deadline = ((row.get("configuration_doc") or {}).get("schedule") or {}).get(
            "publication_deadline")
        try:
            if not deadline or self.now() < _parse_iso(deadline):
                return False
        except (TypeError, ValueError):
            return False
        round_id = str(row["round_id"])
        runs = self._store.list_runs(
            round_id,
            columns="run_id,kind,stage,status,scored_run_id,per_icp_score",
        )
        if any(run.get("status") in ("pending", "leased", "submitted") for run in runs):
            return False
        accepted_judges = {
            str(run.get("scored_run_id")) for run in runs
            if run.get("kind") == "score" and run.get("status") == "accepted"
        }
        return any(
            run.get("per_icp_score") is None
            or str(run.get("run_id")) not in accepted_judges
            for run in runs
            if run.get("kind") == "execute" and run.get("stage") in (1, 2)
            and run.get("status") == "accepted"
        )

    def _round_needs_active_poll(
        self, row: Mapping[str, Any], *, public_only: bool = False,
    ) -> bool:
        """Keep billing active while omitting frozen judges from worker polling."""

        try:
            frozen = self._frozen_unfinished_judgment(row)
        except (ArenaStoreError, ArenaStoreUnavailable, ServiceError):
            return True
        if not frozen:
            return True
        if public_only:
            return False
        round_id = str(row["round_id"])
        # Only the existing bounded reconciliation reads may keep this round
        # in the driver. Closed-round billing does not select scored rounds.
        try:
            if (self._store.list_openrouter_cost_reconciliations(round_id, limit=1)
                or self._store.list_deepline_cost_reconciliations(round_id, limit=1)):
                return True
        except (ArenaStoreError, ArenaStoreUnavailable):
            return True
        return False

    def active_rounds(self, *, public_only: bool = False) -> List[Dict[str, Any]]:
        """Every active round, oldest first, with its public pinned output policy.

        Rounds overlap (one open for submissions while the previous one runs),
        so the driver advances each of them on every tick.
        """

        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            if (row is None or row["status"] in TERMINAL_STATUSES
                or not self._round_needs_active_poll(row, public_only=public_only)):
                return []
            return [{
                "round_id": row["round_id"],
                "status": row["status"],
                "schedule": dict((row.get("configuration_doc") or {}).get("schedule") or {}),
                **self._public_round_output_policy(row.get("configuration_doc") or {}),
            }]

        network_name, netuid = self._chain_scope()
        rows: List[Dict[str, Any]] = []
        offset = 0
        page_size = 20
        while True:
            page = self._store.list_rounds(
                statuses=ACTIVE_ROUND_STATUSES,
                mode=self._config.mode,
                network_name=network_name,
                netuid=netuid,
                limit=page_size,
                offset=offset,
                columns=(
                    "round_id,status,created_at,evaluation_date,rewards_enabled,"
                    + (PUBLIC_CONFIGURATION_COLUMN if public_only else "configuration_doc")
                ),
            )
            if public_only:
                page = [self._public_current_round(row) for row in page]
            rows.extend(
                row
                for row in page
                if row["status"] not in TERMINAL_STATUSES
                and (row.get("configuration_doc") or {}).get("mode") == self._config.mode
                and self._round_needs_active_poll(row, public_only=public_only)
            )
            if len(page) < page_size:
                break
            offset += page_size
        return [
            {
                "round_id": row["round_id"],
                "status": row["status"],
                "schedule": dict((row.get("configuration_doc") or {}).get("schedule") or {}),
                **self._public_round_output_policy(row.get("configuration_doc") or {}),
            }
            for row in reversed(rows)
        ]

    def open_round(self, *, include_future: bool = False) -> Optional[Dict[str, Any]]:
        """Find the active intake round, or an existing future open row for the driver."""

        now = self.now()
        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and (
                self._round_pending_submission(row, now) if include_future
                else self._round_accepts_submissions(row, now)
            ) else None
        network_name, netuid = self._chain_scope()
        for row in self._store.list_rounds(
            status="open",
            mode=self._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=20,
            columns="round_id,status,created_at,configuration_doc",
        ):
            if ((self._round_pending_submission(row, now) if include_future
                 else self._round_accepts_submissions(row, now))
                    and (row.get("configuration_doc") or {}).get("mode") == self._config.mode):
                return self._round(row["round_id"])
        return None

    @staticmethod
    def _round_pending_submission(row: Mapping[str, Any], now: datetime) -> bool:
        if row.get("status") != "open":
            return False
        schedule = (row.get("configuration_doc") or {}).get("schedule") or {}
        try:
            return now < _parse_iso(schedule["submission_cutoff"])
        except (KeyError, TypeError, ValueError):
            # An invalid open row needs repair, never a second daily row.
            return True

    @staticmethod
    def _round_accepts_submissions(row: Mapping[str, Any], now: datetime) -> bool:
        if row.get("status") != "open":
            return False
        schedule = (row.get("configuration_doc") or {}).get("schedule") or row.get("schedule") or {}
        try:
            return (_parse_iso(schedule["submission_open"]) <= now
                    < _parse_iso(schedule["submission_cutoff"]))
        except (KeyError, TypeError, ValueError):
            return False

    def _due_daily_round_opening(self, round_id: str) -> bool:
        """Identify only the missing canonical round near its UTC opening."""

        if self._pinned_round_id() is not None:
            return False
        hour = self._config.defaults.daily_cutoff_hour_utc
        if hour is None:
            return False
        now = self.now().astimezone(timezone.utc)
        opening = now.replace(hour=int(hour), minute=0, second=0, microsecond=0)
        if now < opening:
            opening -= timedelta(days=1)
        # This only describes the bounded scheduler cutover, not a permanent
        # alternative admission path if daily creation fails. The window spans
        # the opening in both directions: miners racing the cutoff name the
        # next canonical round moments before it is created, and a terminal
        # 404 there loses a submission that a short retry would have placed.
        next_opening = opening + timedelta(days=1)
        grace = timedelta(seconds=90)
        return (
            (opening <= now < opening + grace
             and round_id == round_id_for_cutoff(opening + timedelta(days=1)))
            or (next_opening - grace <= now < next_opening
                and round_id == round_id_for_cutoff(next_opening + timedelta(days=1)))
        )

    def _hot_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """One round's row for runner-facing handlers, cached for a few seconds.

        Claims and completions arrive by the thousand per stage; the SQL
        functions remain the authority for status, so a briefly stale row
        only yields a structured refusal.
        """

        self._require_round_ownership(round_id)
        now = time.monotonic()
        with self._hot_round_lock:
            cached = self._hot_rounds.get(round_id)
            if cached is not None and now - cached[0] < HOT_ROUND_TTL_SECONDS:
                return cached[1]
        row = self._store.get_round(round_id)
        if row is None:
            return None
        row = self._require_round_mode(row)
        with self._hot_round_lock:
            self._hot_rounds[round_id] = (now, row)
        return row

    def _request_round(self, envelope: Any, *, scope: str, hot: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate a signed request and resolve the round its envelope names.

        Rounds overlap, so the envelope, not "the current round", says which
        round a submission, claim, or completion belongs to. An unknown round
        is refused before any banned-list or status check.
        """

        validated = self.validate_request(envelope, scope=scope, round_id=None)
        round_id = str(validated["round_id"])
        self._require_round_ownership(round_id)
        round_row = self._hot_round(round_id) if hot else self._store.get_round(round_id)
        if round_row is None:
            if (scope == contracts.SCOPE_SUBMISSION_PRESIGN
                    and self._due_daily_round_opening(round_id)):
                raise ServiceError("round_opening", 503)
            raise ServiceError("round_unknown", 404)
        round_row = self._require_round_mode(round_row)
        if validated["hotkey"] in self._banned_hotkeys(round_row):
            raise ServiceError("hotkey_banned", 403)
        return validated, round_row

    def latest_published_round(self, *, public_only: bool = False) -> Optional[Dict[str, Any]]:
        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and row["status"] == "published" else None
        network_name, netuid = self._chain_scope()
        rows = self._store.list_rounds(
            status="published", mode=self._config.mode,
            network_name=network_name, netuid=netuid,
            # The store applies mode/network filters and newest-first ordering.
            # Current-round polling needs one summary, not every large result.
            limit=1,
            columns=(
                "round_id,status,published_at,"
                + (PUBLIC_CONFIGURATION_COLUMN if public_only else "configuration_doc")
                + ",king_outcome,king_hotkey,effective_reward_epoch,reward_basis_hash"
            ),
        )
        if public_only:
            rows = [self._public_current_round(row) for row in rows]
        return next(
            (row for row in rows if (row.get("configuration_doc") or {}).get("mode") == self._config.mode),
            None,
        )

    # -- signed requests ------------------------------------------------------

    def validate_request(self, envelope: Any, *, scope: str, round_id: Optional[str]) -> Dict[str, Any]:
        try:
            validated = contracts.validate_signed_request(envelope, expected_scope=scope, now=int(self.now().timestamp()), verify_signature=self._config.verify_signature, expected_round_id=round_id)
        except ArenaSignatureError:
            raise ServiceError("signature_invalid", 401)
        except ArenaContractError as exc:
            raise ServiceError("request_invalid:%s" % str(exc)[:80], 400)
        if round_id is not None and validated["hotkey"] in self._banned_hotkeys(self._round(round_id)):
            raise ServiceError("hotkey_banned", 403)
        return validated

    @staticmethod
    def _banned_hotkeys(round_row: Mapping[str, Any]) -> set:
        """Return the plain banned-hotkey list stored with the round."""

        return set(str(item) for item in (round_row.get("configuration_doc") or {}).get("banned_hotkeys") or [])

    # -- submissions (sections 6, 7, 14.2) -------------------------------------

    def handle_submission_credit_retry(
        self, submission_id: str, envelope: Any,
    ) -> Dict[str, Any]:
        """Request a fresh bounded attempt after the owner replenishes credit."""
        validated, round_row = self._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_CREDIT_RETRY
        )
        body = contracts.validate_submission_credit_retry_body(validated["body"])
        if body["submission_id"] != submission_id:
            raise ServiceError("credit_retry_request_invalid", 400)
        self._enforce_submission_request_limit(validated["hotkey"])
        row = self._store.get_submission(submission_id)
        if (
            row is None or row.get("round_id") != round_row["round_id"]
            or row.get("miner_hotkey") != validated["hotkey"]
        ):
            raise ServiceError("credit_retry_owner_required", 403)
        # A nonempty failed output is eligible only as an authenticated partial
        # company judgment. Reprove its immutable source bytes and evidence
        # before SQL admits the owner-requested retry under the existing gates.
        partial_sources = (
            self._store.list_runs(round_row["round_id"], kind="score", submission_id=submission_id)
            if company_judgments.enabled(round_row.get("configuration_doc") or {}) else ()
        )
        for source in self._current_score_runs(round_row, partial_sources):
            if (source.get("status") != "failed"
                or source.get("terminal_cause") != "credential_error"
                or not source.get("output_ref")
                or not (source.get("result_doc") or {}).get("company_judgment_completion_request_hash")):
                continue
            try:
                lease = company_judgments.validate_lease_context(
                    (source.get("claim_response") or {}).get("company_judgment_cache") or {},
                )
                bound = False
                seen = set()
                for miss in lease["misses"]:
                    identity = (miss["cache_key"], miss["authority_slot"])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    cached = self._store.get_company_judgment(*identity)
                    if cached is None or cached.get("source_score_run_id") != source["run_id"]:
                        continue
                    self._company_judgment_evidence(
                        cache_key=miss["cache_key"], company_input_hash=miss["company_input_hash"],
                        authority_slot=miss["authority_slot"],
                    )
                    bound = True
                if not bound:
                    raise scoring.ScoringError("partial credit retry source is unbound")
            except (company_judgments.CompanyJudgmentError, scoring.ScoringError) as exc:
                raise ServiceError("credit_retry_source_unavailable", 503) from exc
        if not company_judgments.enabled(round_row.get("configuration_doc") or {}):
            for source in self._current_score_runs(round_row, self._store.list_runs(round_row["round_id"], kind="score", submission_id=submission_id)):
                if (source.get("status") == "failed" and source.get("terminal_cause") == "credential_error"
                    and source.get("output_ref") and (source.get("result_doc") or {}).get("legacy_partial_checkpoint")):
                    try:
                        self._validated_legacy_partial(source, round_row)
                    except (scoring.ScoringError, ServiceError, ValueError, OutputInvalid) as exc:
                        raise ServiceError("credit_retry_source_unavailable", 503) from exc
        # SQL repeats authorization and checks the deadline, proof, budget
        # ledger and attempt limit under locks. A signature is not cost proof.
        receipt = self._store.retry_credit_failures(
            round_row["round_id"], submission_id, validated["hotkey"],
            contracts.request_bytes_hash(validated),
        )
        try:
            return contracts.validate_credit_retry_result(receipt)
        except ArenaContractError as exc:
            raise ServiceError("credit_retry_unavailable", 503) from exc

    def submission_status(self, submission_id: str) -> Dict[str, Any]:
        self._require_identifier(submission_id, "submission")
        row = self._store.get_submission(submission_id)
        if row is None:
            raise ServiceError("submission_missing", 404)
        round_row = self._round(str(row.get("round_id") or ""))
        return {
            "submission_id": submission_id,
            "status": row["status"],
            "rejection_rule": row.get("rejection_rule"),
            "replaces_submission_id": row.get("replaces_submission_id"),
            "replaced_by_submission_id": row.get("replaced_by_submission_id"),
            "code_review": self._public_code_review(row, round_row=round_row, now=self.now()),
        }

    @staticmethod
    def _public_code_review(
        row: Mapping[str, Any], *, round_row: Optional[Mapping[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        return public_dashboard.code_review_summary(row, round_row=round_row, now=now)

    @staticmethod
    def _is_daily_baseline(row: Mapping[str, Any], round_row: Mapping[str, Any]) -> bool:
        return bool(
            row.get("is_king")
            and row.get("submission_id") == "baseline-" + str(round_row["round_id"]).removeprefix("arena-")
            and row.get("miner_hotkey") == (round_row.get("configuration_doc") or {}).get("baseline_hotkey")
        )

    def _require_code_review(self, submission_id: str, round_row: Mapping[str, Any]) -> None:
        row = self._store.get_submission(submission_id)
        if row is None:
            raise ServiceError("submission_missing", 404)
        if not self._is_daily_baseline(row, round_row):
            if row.get("code_review_status") != "passed":
                raise ServiceError("code_review_required", 409)
            # A frozen entrant must be allowed to finish after a restart or
            # review-policy change. Recheck its current proof at publication;
            # do not turn a data-eligibility gap into a stuck execution lease.
            if row.get("status") != "frozen" and not training_eligibility.assess_source(
                row, enabled=self._full_thinking_required(round_row),
            )["eligible"]:
                raise ServiceError("code_review_required", 409)

    def review_pending_submissions(
        self, *, should_continue: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Called by a separate worker, never in the competition scheduler."""
        reviewer = self._config.code_reviewer
        if reviewer is None:
            raise ServiceError("code_review_unavailable", 503)
        candidates = []
        for active in self.active_rounds():
            round_row = self._round(active["round_id"])
            # Review only after source replacements close. The database repeats
            # this check under its admission lock before reserving review spend.
            if round_row["status"] == "open" and self.now() < self._submission_replacement_cutoff(round_row):
                continue
            for status in ("accepted", "frozen"):
                for row in self._store.list_submissions(round_row["round_id"], status=status):
                    if self._is_daily_baseline(row, round_row) or row.get("code_review_status") in ("passed", "rejected"):
                        continue
                    candidates.append((row, round_row))
        def review_if_active(candidate) -> Optional[Dict[str, Any]]:
            # A queued task has not reserved review spend yet. TERM or the
            # restart guard can defer it while the four active calls finish.
            if should_continue is not None and not should_continue():
                return None
            row, round_row = candidate
            review_in_round = getattr(reviewer, "review_with_frozen_round", None)
            if callable(review_in_round):
                return review_in_round(row, round_row)
            return reviewer.review(row)

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="arena-code-review") as pool:
            results = list(pool.map(review_if_active, candidates))
        reviewed = sum(
            isinstance(result, Mapping)
            and result.get("status") in ("passed", "rejected", "error", "ok")
            for result in results
        )
        return {"reviewed": reviewed}

    def submission_similarity_references(self, row: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Select earlier admitted sources privately, with the frozen champion.

        Admission time, not upload-reservation time, determines precedence.
        Replaced sources and the miner's own entry never block a replacement.
        No reference identity or source is returned through the miner API.
        """
        round_row = self._round(str(row["round_id"]))
        order = lambda item: (
            datetime.fromisoformat(str(item.get("accepted_at") or item["created_at"]).replace("Z", "+00:00")),
            str(item["submission_id"]),
        )
        references = []
        rows = self._store.list_submissions(str(row["round_id"]))
        for other in rows:
            if other["submission_id"] == row["submission_id"]:
                continue
            baseline = self._is_daily_baseline(other, round_row)
            if not baseline and (
                other.get("miner_hotkey") == row.get("miner_hotkey")
                or other.get("status") not in ("accepted", "frozen")
                or other.get("code_review_status") == "rejected"
                or order(other) >= order(row)
            ):
                continue
            if baseline and other.get("status") not in ("accepted", "frozen"):
                raise ServiceError("similarity_reference_unavailable", 503)
            references.append({
                "row": other,
                "source_public": source_disclosure.disclosure_status(
                    other, self.now(), round_row=round_row,
                )["available"],
            })
        champion = self._store.submission_similarity_champion(str(row["round_id"]))
        if champion.get("status") != "ready":
            raise ServiceError("similarity_reference_unavailable", 503)
        champion_id = champion.get("submission_id")
        if champion_id:
            champion_row = self._store.get_submission(str(champion_id))
            if not champion_row:
                raise ServiceError("similarity_reference_unavailable", 503)
            # A pinned service may review the historical champion without
            # admitting requests or work for that champion's previous round.
            champion_round = self._store.get_round(str(champion_row["round_id"]))
            if champion_round is None:
                raise ServiceError("similarity_reference_unavailable", 503)
            champion_round = self._require_round_mode(champion_round)
            references.append({
                "row": champion_row,
                "source_public": source_disclosure.disclosure_status(
                    champion_row, self.now(), round_row=champion_round,
                )["available"],
            })
        else:
            # Bootstrap champion: use only the public source URL the
            # existing baseline path uses, without freezing it early.
            fetcher = self._config.baseline_source_fetcher
            if fetcher is None:
                raise ServiceError("similarity_reference_unavailable", 503)
            config = round_row.get("configuration_doc") or {}
            source_url = (
                DEFAULT_BASELINE_SOURCE_URL if config.get("mode") == "live"
                else config.get("baseline_source_url") or self.defaults.baseline_source_url
            )
            payload = bytes(fetcher(source_url, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
            references.append({
                "row": {"source_size_bytes": len(payload)},
                "payload": payload,
                "source_public": source_url == DEFAULT_BASELINE_SOURCE_URL,
            })
        if len(references) > contracts.MAX_CHALLENGERS + 1:
            raise ServiceError("similarity_reference_limit", 503)
        return references

    @staticmethod
    def _submission_replacement_cutoff(round_row: Mapping[str, Any]) -> datetime:
        schedule = (round_row.get("configuration_doc") or {}).get("schedule") or {}
        return _parse_iso(schedule["submission_cutoff"]) - timedelta(hours=1)

    def _require_submission_window(self, round_row: Mapping[str, Any]) -> None:
        if round_row["status"] != "open":
            raise ServiceError("submission_window_closed", 409)
        schedule = (round_row.get("configuration_doc") or {}).get("schedule") or {}
        now = self.now()
        if now < _parse_iso(schedule["submission_open"]) or now >= _parse_iso(schedule["submission_cutoff"]):
            raise ServiceError("submission_window_closed", 409)

    def _enforce_submission_request_limit(self, hotkey: str) -> None:
        # Some narrow unit tests construct the service without __init__. The
        # fallback is test-only; production always creates the limiter above.
        limiter = getattr(self, "_submission_request_limiter", None)
        if limiter is None:
            limiter = submission_rate_limit.SubmissionRequestLimiter()
            self._submission_request_limiter = limiter
        decision = limiter.check(hotkey)
        if not decision.allowed:
            raise ServiceError("submission_rate_limited", 429)

    def handle_submission_presign(self, envelope: Any) -> Dict[str, Any]:
        """Reserve one private source upload for a signed miner request."""

        validated, round_row = self._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_PRESIGN
        )
        self._require_submission_window(round_row)
        round_id = round_row["round_id"]
        configuration = round_row.get("configuration_doc") or {}
        owner_admission = None
        if integrity.enabled(configuration):
            try:
                owner_admission = resolve_finalized_owner(
                    self._config.chain, validated["hotkey"]
                )
            except OwnerAdmissionError as exc:
                status = 403 if exc.code == "hotkey_unregistered" else 503
                raise ServiceError(exc.code, status) from exc
        elif self._config.chain.uid_for_hotkey(validated["hotkey"]) is None:
            raise ServiceError("hotkey_unregistered", 403)
        if validated["hotkey"] == (round_row.get("configuration_doc") or {}).get(
            "baseline_hotkey"
        ):
            raise ServiceError("baseline_hotkey_reserved", 403)
        body = contracts.validate_submission_presign_body(validated["body"])
        self._enforce_submission_request_limit(validated["hotkey"])
        # The submission id is the only server-assigned source identity.
        submission_id = "sub-%s" % secrets.token_hex(16)
        source_ref = "arena/%s/sources/%s.tar.gz" % (round_id, submission_id)
        document = {
            "source_ref": source_ref,
            "source_size_bytes": body["source_size_bytes"],
            "consent": dict(body["consent"]),
        }
        if body.get("source_content_md5") is not None:
            document["source_content_md5"] = body["source_content_md5"]
        try:
            registration_args = (
                round_id, submission_id, validated["hotkey"], document
            )
            if owner_admission is None:
                registration = self._store.register_submission(*registration_args)
            else:
                registration = self._store.register_submission(
                    *registration_args, owner_admission=owner_admission
                )
        except ArenaStoreError as exc:
            if "lab_arena_submission_owner_changed" in str(exc):
                raise ServiceError("submission_owner_changed", 409) from exc
            if "lab_arena_submission_replacement_ineligible" in str(exc):
                raise ServiceError("submission_replacement_ineligible", 409) from exc
            if "lab_arena_submission_conflict" in str(exc):
                raise ServiceError("submission_conflict", 409) from exc
            raise
        if registration.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        if registration.get("status") == "replacement_closed":
            raise ServiceError("submission_replacement_closed", 409)
        if registration.get("status") == "replacement_limit_reached":
            raise ServiceError("submission_replacement_limit_reached", 409)
        if registration.get("status") not in ("registered", "existing"):
            raise ServiceError("submission_registration_failed", 500)
        submission_id = str(registration.get("submission_id") or submission_id)
        source_ref = str(registration.get("source_ref") or source_ref)
        if registration.get("submission_status") in ("accepted", "frozen"):
            # Historical accepted rows predate persisted transport checksums.
            # Verify their bytes before treating a same-size upload as a retry.
            checksum = body.get("source_content_md5")
            if checksum is not None:
                self._validate_uploaded_source({
                    "source_ref": source_ref,
                    "source_size_bytes": body["source_size_bytes"],
                    "submission_doc": {"source_content_md5": checksum},
                })
        try:
            upload_arguments = {
                "size_bytes": int(body["source_size_bytes"]),
                "content_type": source_bundle.SOURCE_CONTENT_TYPE,
                "expires_seconds": SOURCE_UPLOAD_EXPIRES_SECONDS,
            }
            if body.get("source_content_md5") is not None:
                upload_arguments["source_content_md5"] = body["source_content_md5"]
            upload = self._objects.presign_put(source_ref, **upload_arguments)
        except Exception as exc:
            raise ServiceError("source_upload_unavailable", 503) from exc
        return {
            "status": "upload_ready",
            "submission_id": submission_id,
            "source_ref": source_ref,
            "upload_url": str(upload["upload_url"]),
            "upload_headers": dict(upload["upload_headers"]),
            "expires_in_seconds": int(upload["expires_in_seconds"]),
        }

    def _validate_uploaded_source(
        self,
        row: Mapping[str, Any],
        *,
        forbidden_values: Sequence[str] = (),
        require_license: bool = False,
    ) -> bytes:
        expected_size = int(row.get("source_size_bytes") or 0)
        source_ref = str(row.get("source_ref") or "")
        try:
            payload = self._objects.get_bounded(
                source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
        except Exception as exc:
            raise ServiceError("source_upload_unavailable", 409) from exc
        if len(payload) != expected_size:
            raise ServiceError("submission_rejected:source_size_mismatch", 400)
        checksum = (row.get("submission_doc") or {}).get("source_content_md5")
        if checksum is not None:
            actual = base64.b64encode(
                hashlib.md5(payload, usedforsecurity=False).digest()
            ).decode("ascii")
            if not hmac.compare_digest(actual, checksum):
                raise ServiceError("submission_rejected:source_checksum_mismatch", 400)
        try:
            source_bundle.validate_source_archive(
                payload,
                forbidden_values=forbidden_values,
                require_license=require_license,
            )
        except source_bundle.SourceBundleError as exc:
            path = exc.path or ""
            for value in forbidden_values:
                if value:
                    path = path.replace(value, "[REDACTED]")
            raise ServiceError(
                "submission_rejected:%s" % exc.code, 400,
                source_path=path[:source_bundle.MAX_SOURCE_PATH_BYTES],
            ) from exc
        return payload

    def handle_submission_finalize(
        self, submission_id: str, envelope: Any
    ) -> Dict[str, Any]:
        """Verify uploaded bytes and admit the source under the same signed owner."""

        validated, round_row = self._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_FINALIZE
        )
        self._require_submission_window(round_row)
        body = contracts.validate_submission_finalize_body(validated["body"])
        if body["submission_id"] != submission_id:
            raise ServiceError("submission_id_mismatch", 400)
        row = self._store.get_submission(submission_id)
        if (
            row is None
            or row.get("round_id") != round_row["round_id"]
            or row.get("miner_hotkey") != validated["hotkey"]
        ):
            raise ServiceError("submission_missing", 404)
        for field in ("source_ref", "source_size_bytes"):
            if body[field] != row.get(field):
                raise ServiceError("submission_transport_mismatch", 409)
        if row.get("status") == "accepted":
            stored_credentials = {
                provider: self._store.get_submission_credential(
                    submission_id, validated["hotkey"], provider
                )
                for provider in credentials_module.RUNTIME_PROVIDERS
            }
            if all(
                stored_credentials[provider] is not None
                for provider in credentials_module.REQUIRED_RUNTIME_PROVIDERS
            ):
                if (
                    "scrapingdog_api_key" in body["credentials"]
                    and stored_credentials["scrapingdog"] is None
                ):
                    raise ServiceError("submission_credentials_immutable", 409)
                return {"status": "accepted", "submission_id": submission_id}
            raise ServiceError("submission_credentials_missing", 409)
        if row.get("status") != "uploading":
            if row.get("rejection_rule") == "source_replaced":
                raise ServiceError("submission_superseded", 409)
            raise ServiceError("submission_not_uploading", 409)
        if row.get("replaces_submission_id") and self.now() >= self._submission_replacement_cutoff(round_row):
            raise ServiceError("submission_replacement_closed", 409)
        self._enforce_submission_request_limit(validated["hotkey"])
        try:
            payload = self._validate_uploaded_source(
                row,
                forbidden_values=tuple(body["credentials"].values()),
                require_license=True,
            )
            identity = submission_similarity.inspect_archive(payload)
        except code_review.CodeReviewError as exc:
            self._store.update_submission(
                str(round_row["round_id"]), submission_id, "uploading", "rejected",
                {"rejection_rule": exc.code},
            )
            raise ServiceError("submission_rejected:%s" % exc.code, 400) from exc
        except ServiceError as exc:
            if exc.code.startswith("submission_rejected:"):
                self._store.update_submission(
                    str(round_row["round_id"]),
                    submission_id,
                    "uploading",
                    "rejected",
                    {"rejection_rule": exc.code.split(":", 1)[1]},
                )
            raise
        manager = self._config.credential_manager
        if manager is None:
            raise ServiceError("credential_validation_unavailable", 503)
        try:
            encrypted_credentials = manager.validate_and_encrypt(
                body["credentials"],
                submission_id=submission_id,
                miner_hotkey=validated["hotkey"],
            )
        except credentials_module.CredentialError as exc:
            if exc.retryable:
                raise ServiceError(exc.code, 503) from exc
            self._store.update_submission(
                str(round_row["round_id"]),
                submission_id,
                "uploading",
                "rejected",
                {"rejection_rule": exc.code},
            )
            raise ServiceError("submission_rejected:%s" % exc.code, 400) from exc
        try:
            result = self._store.accept_submission_source_with_credentials(
                str(round_row["round_id"]),
                submission_id,
                validated["hotkey"],
                encrypted_credentials,
                identity.archive_sha256,
                identity.normalized_sha256,
            )
        except ArenaStoreError as exc:
            if "lab_arena_source_digest_conflict" in str(exc):
                raise ServiceError("submission_source_immutable", 409) from exc
            if "lab_arena_submission_credentials_immutable" in str(exc):
                raise ServiceError("submission_credentials_immutable", 409) from exc
            if "lab_arena_submission_replacement_closed" in str(exc):
                raise ServiceError("submission_replacement_closed", 409) from exc
            if "lab_arena_submission_window_closed" in str(exc):
                raise ServiceError("submission_window_closed", 409) from exc
            if "lab_arena_submission_replacement_ineligible" in str(exc):
                raise ServiceError("submission_replacement_ineligible", 409) from exc
            if any(code in str(exc) for code in (
                "lab_arena_submission_replacement_stale",
                "lab_arena_submission_not_uploading",
            )):
                raise ServiceError("submission_superseded", 409) from exc
            if "lab_arena_round_full" not in str(exc):
                raise
            self._store.update_submission(
                str(round_row["round_id"]), submission_id, "uploading", "rejected",
                {"rejection_rule": "capacity.round_full"},
            )
            raise ServiceError("submission_rejected:capacity.round_full", 409) from exc
        if result.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        if result.get("status") == "replacement_closed":
            raise ServiceError("submission_replacement_closed", 409)
        if result.get("status") == "rejected_duplicate":
            raise ServiceError("submission_rejected:duplicate_submission", 409)
        if result.get("status") not in ("ok", "existing"):
            raise ServiceError("submission_finalize_failed", 500)
        return {"status": "accepted", "submission_id": submission_id}

    def admit_uploaded_submissions(self, round_id: str, *, final: bool = False) -> Dict[str, Any]:
        """At cutoff, reject source slots that a miner did not finalize."""

        round_row = self._round(round_id)
        if round_row["status"] != "open":
            return {"status": "stale", "round_status": round_row["status"]}
        outcomes: Dict[str, Any] = {"status": "ok", "accepted": 0, "rejected": 0, "deferred": 0, "remaining": 0}
        pending = self._store.list_submissions(round_id, status="uploading")
        if not final:
            outcomes["remaining"] = len(pending)
            return outcomes
        for row in pending:
            result = self._store.update_submission(
                round_id,
                str(row["submission_id"]),
                "uploading",
                "rejected",
                {"rejection_rule": "source_upload_incomplete"},
            )
            outcomes["rejected" if result.get("status") == "ok" else "deferred"] += 1
        return outcomes

    @staticmethod
    def _participant(row: Mapping[str, Any], *, is_king: bool) -> Dict[str, Any]:
        """The small frozen participant record used for leases and publication."""

        return {
            "submission_id": row["submission_id"],
            "miner_hotkey": row["miner_hotkey"],
            "source_ref": row["source_ref"],
            "source_size_bytes": int(row["source_size_bytes"]),
            "is_king": bool(is_king),
        }

    # -- participant freeze and benchmark (sections 7.1, 8) --------------------

    def _verify_full_thinking_baseline_source(
        self, configuration: Mapping[str, Any], baseline: Mapping[str, Any]
    ) -> None:
        """Check the selected stored baseline against this round's frozen archive."""

        if configuration.get("reasoning_capture_policy") != reasoning_capture.POLICY:
            return
        source_ref = baseline.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref:
            raise ServiceError("baseline_source_not_ready", 503)
        try:
            payload = self._objects.get_bounded(
                source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
        except Exception as exc:
            raise ServiceError("baseline_source_not_ready", 503) from exc
        actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if actual_digest != configuration.get("reasoning_baseline_source_sha256"):
            raise ServiceError("reasoning_baseline_source_mismatch", 503)

    def _freeze_full_thinking_baseline(
        self, round_id: str, round_row: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Freeze the exact promoted source and its original capture contract."""

        if round_row.get("champion_funding_frozen") is True:
            return {"status": "existing"}
        configuration = round_row.get("configuration_doc") or {}
        source_url = str(configuration["baseline_source_url"])
        source_sha256 = str(configuration["reasoning_baseline_source_sha256"])
        network_name, netuid = self._chain_scope()
        promoted = self._store.latest_promoted_champion(
            network_name=network_name, netuid=netuid
        )
        promotion_round_id = winner_id = commit = None
        if promoted is not None:
            promotion_round_id = str(promoted.get("round_id") or "")
            winner_id = str(promoted.get("winner_submission_id") or "")
            commit = str(promoted.get("commit") or "")
            if (not promotion_round_id or not winner_id
                    or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
                raise ServiceError("baseline_promotion_invalid", 500)
            source_url = (
                "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
                + commit + ".tar.gz"
            )
            fetcher = self._config.baseline_source_fetcher
            if fetcher is None:
                raise ServiceError("baseline_source_fetcher_missing", 500)
            try:
                payload = bytes(fetcher(source_url, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
                source_bundle.validate_source_archive(payload)
            except source_bundle.SourceBundleError as exc:
                raise ServiceError("baseline_source_invalid:%s" % exc.code, 500) from exc
            except Exception as exc:
                raise ServiceError("baseline_source_not_ready", 503) from exc
            source_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
        result = self._store.freeze_full_thinking_baseline(
            round_id,
            expected_promotion_round_id=promotion_round_id,
            expected_winner_submission_id=winner_id,
            expected_commit=commit,
            expected_source_url=str(configuration["baseline_source_url"]),
            expected_source_sha256=str(configuration["reasoning_baseline_source_sha256"]),
            source_url=source_url,
            source_sha256=source_sha256,
        )
        if result.get("status") == "promotion_pending":
            raise ServiceError("baseline_promotion_pending", 503)
        if result.get("status") == "stale":
            raise ServiceError("baseline_source_not_ready", 503)
        if result.get("status") not in ("frozen", "existing"):
            raise ServiceError("champion_funding_freeze_failed", 503)
        return result

    def _initial_baseline(self, round_row: Mapping[str, Any]) -> Dict[str, Any]:
        """Download and freeze the public baseline through the source path."""

        round_id = str(round_row["round_id"])
        configuration = round_row.get("configuration_doc") or {}
        hotkey = str(configuration.get("baseline_hotkey") or "").strip()
        source_url = str(
            configuration.get("baseline_source_url")
            or self._config.defaults.baseline_source_url
            or ""
        ).strip()
        if not hotkey:
            raise ServiceError("baseline_hotkey_missing", 500)
        submission_id = "baseline-%s" % round_id.removeprefix("arena-")
        source_ref = "arena/%s/sources/%s.tar.gz" % (round_id, submission_id)
        row = self._store.get_submission(submission_id)
        reused = row is not None
        if row is None:
            fetcher = self._config.baseline_source_fetcher
            if fetcher is None:
                raise ServiceError("baseline_source_fetcher_missing", 500)
            selected_source_url = source_url
            source_observation = "configured_shadow_source"
            try:
                payload = self._objects.get_bounded(
                    source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
                )
                source_observation = "stored_object:%s" % source_ref
            except Exception:
                try:
                    if str(configuration.get("mode") or self._config.mode) == "live":
                        if self._pending_promotion_blocks():
                            raise ServiceError("baseline_promotion_pending", 503)
                        if configuration.get("reasoning_capture_policy") != reasoning_capture.POLICY:
                            selected_source_url = DEFAULT_BASELINE_SOURCE_URL
                    source_observation = (
                        selected_source_url
                        if selected_source_url == DEFAULT_BASELINE_SOURCE_URL
                        else "configured_shadow_source"
                    )
                    payload = bytes(
                        fetcher(
                            selected_source_url,
                            source_bundle.MAX_SOURCE_ARCHIVE_BYTES,
                        )
                    )
                    facts = source_bundle.validate_source_archive(payload)
                    if configuration.get("reasoning_capture_policy") == reasoning_capture.POLICY:
                        actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                        if actual_digest != configuration.get("reasoning_baseline_source_sha256"):
                            raise ServiceError("reasoning_baseline_source_mismatch", 503)
                    self._objects.put(source_ref, payload)
                except ServiceError:
                    raise
                except source_bundle.SourceBundleError as exc:
                    raise ServiceError(
                        "baseline_source_invalid:%s" % exc.code, 500
                    ) from exc
                except Exception as exc:
                    raise ServiceError("baseline_source_not_ready", 503) from exc
            else:
                try:
                    facts = source_bundle.validate_source_archive(payload)
                except source_bundle.SourceBundleError as exc:
                    raise ServiceError(
                        "baseline_source_invalid:%s" % exc.code, 500
                    ) from exc
            if configuration.get("reasoning_capture_policy") == reasoning_capture.POLICY:
                actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                if actual_digest != configuration.get("reasoning_baseline_source_sha256"):
                    raise ServiceError("reasoning_baseline_source_mismatch", 503)
            logger.info(
                "arena_baseline_source_frozen round_id=%s source=%s source_commit=%s",
                round_id,
                source_observation,
                source_bundle.source_archive_commit(payload) or "unavailable",
            )
            result = self._store.register_submission(
                round_id,
                submission_id,
                hotkey,
                {
                    "source_ref": source_ref,
                    "source_size_bytes": facts["source_size_bytes"],
                    "consent": {"public_rerun": True},
                    "is_king": True,
                },
            )
            if result.get("status") not in ("registered", "existing"):
                raise ServiceError("baseline_registration_failed", 500)
            row = self._store.get_submission(submission_id)
        if (
            row is None
            or row.get("round_id") != round_id
            or row.get("miner_hotkey") != hotkey
            or not row.get("is_king")
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        if reused:
            self._verify_full_thinking_baseline_source(configuration, row)
        status = str(row.get("status") or "")
        if status == "uploading":
            try:
                self._validate_uploaded_source(row)
            except ServiceError as exc:
                if exc.code == "source_upload_unavailable":
                    raise ServiceError("baseline_source_not_ready", 503) from exc
                raise ServiceError("baseline_source_invalid", 500) from exc
            result = self._store.update_submission(
                round_id, submission_id, "uploading", "accepted"
            )
            if result.get("status") not in ("ok", "stale"):
                raise ServiceError("baseline_source_not_ready", 503)
            row = self._store.get_submission(submission_id)
            status = str((row or {}).get("status") or "")
        if status not in ("accepted", "frozen"):
            raise ServiceError("baseline_source_rejected", 500)
        return dict(row)

    def freeze_participants(self, round_id: str) -> List[Dict[str, Any]]:
        round_row = self._round(round_id)
        # The daily baseline source is service-owned, but its execution funding
        # belongs to the miner whose completed promotion established the
        # champion. Freeze that identity before any baseline submission or run
        # can be created. The RPC is idempotent, so recovery keeps the same
        # original submission even if repository ownership changes later.
        if (self._full_thinking_required(round_row)
                and (round_row.get("configuration_doc") or {}).get("mode") == "live"):
            funding = self._freeze_full_thinking_baseline(round_id, round_row)
        else:
            funding = self._store.freeze_champion_funding(round_id)
        if funding.get("status") not in ("frozen", "existing"):
            raise ServiceError("champion_funding_freeze_failed", 503)
        round_row = self._round(round_id)
        if round_row.get("champion_funding_frozen") is not True:
            raise ServiceError("champion_funding_freeze_failed", 503)
        participants: List[Dict[str, Any]] = []
        frozen = self._store.list_submissions(round_id, status="frozen")
        accepted = self._store.list_submissions(round_id, status="accepted")
        unresolved = [
            row for row in accepted
            if not self._is_daily_baseline(row, round_row)
            and row.get("code_review_status") not in ("passed", "rejected")
            and (
                row.get("code_review_status") == "reviewing"
                or row.get("code_review_status") == "pending"
                or code_review_policy.review_retry_available(row)
            )
        ]
        if unresolved:
            deadline = (round_row["configuration_doc"].get("schedule") or {}).get("benchmark_deadline")
            if not deadline or self.now() < _parse_iso(deadline):
                raise ServiceError("code_review_pending", 409)
        frozen_kings = [row for row in frozen if row.get("is_king")]
        if len(frozen_kings) > 1:
            raise ServiceError("baseline_submission_invalid", 500)
        baseline_id = "baseline-%s" % round_id.removeprefix("arena-")
        baseline_hotkey = str(
            (round_row.get("configuration_doc") or {}).get("baseline_hotkey") or ""
        )
        if frozen_kings:
            baseline = frozen_kings[0]
        else:
            baseline = self._initial_baseline(round_row)
            if baseline.get("status") == "accepted":
                accepted = self._store.list_submissions(round_id, status="accepted")
        if (
            baseline.get("submission_id") != baseline_id
            or baseline.get("miner_hotkey") != baseline_hotkey
            or not baseline.get("is_king")
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        if frozen_kings:
            self._verify_full_thinking_baseline_source(
                round_row["configuration_doc"], baseline
            )
        if any(
            row.get("is_king") and row.get("submission_id") != baseline_id
            for row in accepted
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        cap = int(round_row["configuration_doc"].get("max_challengers") or contracts.MAX_CHALLENGERS)
        # Freeze order (acceptance order) decides who enters when the cap binds.
        participants.extend(self._participant(row, is_king=bool(row.get("is_king"))) for row in frozen)
        frozen_count = sum(1 for row in frozen if not row.get("is_king"))
        for row in accepted:
            is_king = row["submission_id"] == baseline_id
            if not is_king and (row.get("code_review_status") != "passed"
                    or not training_eligibility.assess_source(
                        row, enabled=self._full_thinking_required(round_row),
                    )["eligible"]):
                rule = "code_review_rejected" if row.get("code_review_status") == "rejected" else "code_review_incomplete"
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": rule})
                continue
            if not is_king and frozen_count >= cap:
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": "capacity.round_full"})
                continue
            if not is_king:
                frozen_count += 1
            result = self._store.update_submission(round_id, row["submission_id"], "accepted", "frozen", {"is_king": True} if is_king else {})
            if result.get("status") in ("ok", "stale"):
                participants.append(self._participant(row, is_king=is_king))
        if sum(1 for participant in participants if participant.get("is_king")) != 1:
            raise ServiceError("baseline_submission_invalid", 500)
        return participants

    def _activate_company_judgments(self, document: Dict[str, Any]) -> bool:
        """Add only cache semantics to a future, not-yet-committed round."""
        activation = self._config.defaults.company_judgment_from
        if (activation is None
            or _parse_iso(document["schedule"]["submission_open"]) < datetime.fromisoformat(activation.replace("Z", "+00:00"))):
            return False
        # Other grading modes own their existing cache semantics. A legacy
        # cache activation must not prevent their future rounds from opening.
        if contact_policy.enabled(document) or quality_policy.enabled(document):
            return False
        if not integrity.enabled(document):
            raise ServiceError("company_judgments_require_legacy_company_integrity", 503)
        self._require_company_quality_schema()
        document["company_judgment_policy"] = company_judgments.POLICY
        document["scorer_policy"] = dict(document["scorer_policy"], company_judgment_policy=company_judgments.POLICY)
        return True

    def commit_benchmark(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "open":
            return {"status": "existing", "round_status": round_row["status"]}
        scorer_image = {
            "scorer_image_digest": self._config.defaults.scorer_image_digest,
            "scorer_image_reference": self._config.defaults.scorer_image_reference,
        }
        refreshed_configuration = {
            **dict(round_row.get("configuration_doc") or {}),
            **scorer_image,
        }
        activate_company_cache = self._activate_company_judgments(refreshed_configuration)
        if refreshed_configuration.get("mode") == "live" and (
            "execution_cap_microusd" not in refreshed_configuration
            or "cost_per_company_microusd" not in refreshed_configuration
        ):
            raise ServiceError("round_cost_policy_missing", 500)
        try:
            contracts.validate_round_configuration(refreshed_configuration)
        except ArenaContractError as exc:
            raise ServiceError("scorer_image_invalid", 500) from exc
        started = self.now()
        schedule = round_row["configuration_doc"]["schedule"]
        if started < _parse_iso(schedule["submission_cutoff"]):
            return {"status": "waiting", "round_status": "open"}
        submission_open = _parse_iso(schedule["submission_open"])
        icp_set_date = submission_open.astimezone(timezone.utc).date().isoformat()
        set_id = int(icp_set_date.replace("-", ""))
        source = self._config.daily_icp_source(set_id=set_id, active_at=started)
        if not isinstance(source, Mapping) or source.get("status") not in (
            "ready",
            "unavailable",
        ):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        if source.get("status") == "unavailable":
            if started >= _parse_iso(schedule["benchmark_deadline"]):
                self._store.cancel_round(
                    round_id, CANCEL_REASONS["benchmark_invalid"]
                )
                return {
                    "status": "cancelled",
                    "reason": CANCEL_REASONS["benchmark_invalid"],
                }
            return {
                "status": "retry",
                "reason": "daily_icp_set_not_ready",
                "set_id": set_id,
            }
        raw_icps = source.get("icps")
        try:
            source_set_id = int(source.get("set_id") or 0)
        except (TypeError, ValueError):
            source_set_id = 0
        if source_set_id != set_id or not isinstance(raw_icps, list):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        icps = [dict(icp) for icp in raw_icps if isinstance(icp, Mapping)]
        icp_ids = [str(icp.get("icp_id") or "").strip() for icp in icps]
        contact_bank_valid = True
        if contact_policy.enabled(round_row["configuration_doc"]):
            try:
                for icp in icps:
                    contact_policy.validate_icp(icp)
            except (ValueError, TypeError):
                contact_bank_valid = False
        if (
            len(icps) != contracts.benchmark_icp_count(round_row.get("configuration_doc"))
            or len(icps) != len(raw_icps)
            or any(not icp_id for icp_id in icp_ids)
            or len(set(icp_ids)) != len(icp_ids)
            or not contact_bank_valid
        ):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        try:
            participants = self.freeze_participants(round_id)
        except ServiceError as exc:
            if exc.code in ("baseline_source_not_ready", "baseline_promotion_pending", "code_review_pending"):
                return {"status": "retry", "reason": exc.code, "set_id": set_id}
            raise
        evaluation_date = _parse_iso(
            schedule["submission_cutoff"]
        ).astimezone(timezone.utc).date().isoformat()
        self._freeze_deepline_catalog(refreshed_configuration)
        contracts.validate_round_configuration(refreshed_configuration)
        benchmark_ref = "arena/%s/benchmark.json" % round_id
        self._objects.put(benchmark_ref, contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": round_id, "icps": icps}).encode("utf-8"))
        transition = self._store.commit_round_v2(
            round_id,
            participants=participants,
            benchmark_ref=benchmark_ref,
            evaluation_date=evaluation_date,
            icp_set_date=icp_set_date,
            scorer_image_digest=scorer_image["scorer_image_digest"],
            scorer_image_reference=scorer_image["scorer_image_reference"],
            **({"company_judgment_policy": company_judgments.POLICY} if activate_company_cache else {}),
            **({"deepline_catalog": refreshed_configuration["deepline_catalog"]}
               if refreshed_configuration.get("deepline_catalog") is not None else {}),
        )
        return {"status": transition.get("status"), "participants": len(participants)}

    def benchmark_icps(self, round_id: str) -> List[Dict[str, Any]]:
        return self._benchmark_icps_from_row(round_id, self._round(round_id))

    def _benchmark_icps_from_row(
        self, round_id: str, round_row: Mapping[str, Any],
        *, get_object: Optional[Callable[[str], bytes]] = None,
    ) -> List[Dict[str, Any]]:
        ref = round_row.get("benchmark_ref")
        if not ref:
            raise ServiceError("benchmark_not_committed", 409)
        document = json.loads((get_object or self._objects.get)(ref).decode("utf-8"))
        if not isinstance(document, Mapping) or set(document) != {"schema_version", "round_id", "icps"}:
            raise ServiceError("benchmark_data_invalid", 500)
        if document.get("schema_version") != "leadpoet.lab_arena.benchmark.v1" or document.get("round_id") != round_id:
            raise ServiceError("benchmark_data_invalid", 500)
        icps = list(document["icps"])
        if len(icps) != contracts.benchmark_icp_count(round_row.get("configuration_doc")):
            raise ServiceError("benchmark_data_invalid", 500)
        return icps

    def evaluation_icps(self, round_id: str) -> List[Dict[str, Any]]:
        return self.benchmark_icps(round_id)

    # -- stages (sections 2, 9) ----------------------------------------------

    def open_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if stage not in (1, 2):
            raise ServiceError("stage_invalid", 400)
        participants = list(round_row.get("participants") or [])
        configuration = round_row.get("configuration_doc") or {}
        baseline_first = (
            configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        )
        if baseline_first:
            participants = [
                participant
                for participant in participants
                if bool(participant.get("is_king")) == (stage == 1)
            ]
        elif stage == 2 and round_row.get("icp_set_date") is None:
            finalists = set(str(item) for item in (round_row.get("finalists") or []))
            participants = [participant for participant in participants if participant["submission_id"] in finalists or participant.get("is_king")]
        self.benchmark_icps(round_id)
        positions = list(
            contracts.execution_positions(
                stage, configuration.get("execution_sequence_policy"), configuration
            )
        )
        rows = [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"]} for p in participants]
        if configuration.get(
            "parallel_twenty_icp_execution"
        ):
            if stage == 1:
                return self._store.open_parallel_execution(round_id, rows)
            return self._store.activate_preexecuted_stage2(round_id)
        return self._store.open_stage(round_id, stage, rows, positions)

    def stage_is_complete(self, round_id: str, stage: int) -> bool:
        round_row = self._round(round_id)
        configuration = round_row.get("configuration_doc") or {}
        parallel_twenty = bool(
            configuration.get("parallel_twenty_icp_execution")
        )
        runs = self._store.list_runs(
            round_id,
            stage=None if parallel_twenty and stage == 1 else stage,
            kind="execute",
        )
        if stage == 2 and configuration.get("execution_sequence_policy") == contracts.BASELINE_SCORED_FIRST_POLICY:
            original = self._store.list_runs(round_id, stage=1, kind="execute")
            runs += [run for run in original if self._is_baseline_recovery(run, original, round_row)]
        if (
            not runs
            and stage == 2
            and configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        ):
            return not any(
                not participant.get("is_king")
                for participant in round_row.get("participants") or []
            )
        return bool(runs) and all(run["status"] in ("accepted", "failed") for run in runs)

    def close_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        parallel_twenty = bool(
            (round_row.get("configuration_doc") or {}).get(
                "parallel_twenty_icp_execution"
            )
        )
        closed = (
            self._store.close_parallel_execution(round_id)
            if parallel_twenty and stage == 1
            else self._store.close_stage(round_id, stage)
        )
        if closed.get("status") != "closed":
            return closed
        return self.commit_scoring_plan(round_id, stage)

    def commit_scoring_plan(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        self.benchmark_icps(round_id)
        admitted = {p["submission_id"] for p in round_row.get("participants") or []}
        runs = [run for run in self._store.list_runs(
            round_id, stage=stage, kind="execute",
            columns="*,execution_attempt:lab_arena_execution_attempt_count",
        ) if run["submission_id"] in admitted]
        plan = scoring.build_scoring_plan(
            round_id=round_id,
            stage=stage,
            runs=runs,
            execution_sequence_policy=(round_row.get("configuration_doc") or {}).get(
                "execution_sequence_policy"
            ),
            configuration=round_row.get("configuration_doc"),
        )
        status = "stage%d_closed" % stage
        result = self._store.transition_round(round_id, status, status, {"stage%d_scoring_plan_doc" % stage: plan})
        return {"status": result.get("status"), "work_items": len(plan["work_items"])}

    def _load_scoring_plan(self, round_row: Mapping[str, Any], stage: int) -> Dict[str, Any]:
        plan = round_row.get("stage%d_scoring_plan_doc" % stage)
        if not plan:
            raise ServiceError("scoring_plan_missing", 409)
        try:
            validated = contracts.validate_scoring_plan(plan, round_row.get("configuration_doc"))
        except ArenaContractError as exc:
            raise ServiceError("scoring_plan_invalid", 500) from exc
        if validated["round_id"] != round_row["round_id"] or int(validated["stage"]) != stage:
            raise ServiceError("scoring_plan_invalid", 500)
        return validated

    def _outputs_by_run(self, round_id: str, stage: int) -> Dict[str, List[Dict[str, Any]]]:
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
            outputs[str(run["run_id"])] = list(document["companies"])
        return outputs

    # -- validator scoring (sections 10 and 16 as revised) ----------------------

    def open_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Turn the committed plan into scoring assignments validators claim."""

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_closed" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        plan = self._load_scoring_plan(round_row, stage)
        items = self._scoring_work_items(round_row, stage, plan)
        company_quality_cache = company_judgments.enabled(round_row.get("configuration_doc") or {})
        integrity_cache = integrity.enabled(round_row.get("configuration_doc") or {}) and not company_quality_cache
        result = self._store.open_scoring(
            round_id,
            stage,
            items,
            integrity_cache=integrity_cache,
            **({"company_quality_cache": True} if company_quality_cache else {}),
        )
        return {"status": result.get("status"), "round_status": result.get("round_status"), "assignments": result.get("assignments"), "work_items": len(plan["work_items"])}

    def _scoring_work_items(self, round_row: Mapping[str, Any], stage: int, plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Build ordinary frozen scoring inputs, including cache authority checks."""
        round_id = str(round_row["round_id"])
        for submission_id in {item["submission_id"] for item in plan["work_items"]}:
            self._require_code_review(submission_id, round_row)
        accepted = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            accepted[str(run["run_id"])] = run
        items: List[Dict[str, Any]] = []
        company_quality_cache = company_judgments.enabled(
            round_row.get("configuration_doc") or {}
        )
        integrity_cache = (
            integrity.enabled(round_row.get("configuration_doc") or {})
            and not company_quality_cache
        )
        icps = (
            self.evaluation_icps(round_id)
            if integrity_cache or company_quality_cache else []
        )
        configuration = round_row.get("configuration_doc") or {}
        policy = configuration.get("scorer_policy") or {}
        for item in plan["work_items"]:
            submission_id = item["submission_id"]
            run = accepted.get(str(item["scored_run_id"]))
            if run is None or run["submission_id"] != submission_id or int(run["icp_position"]) != int(item["icp_position"]) or run.get("output_ref") != item["output_ref"]:
                raise ServiceError("scoring_plan_run_mismatch", 500)
            work_item = {
                "scored_run_id": run["run_id"],
                "submission_id": run["submission_id"],
                "icp_position": int(run["icp_position"]),
                "output_ref": run["output_ref"],
            }
            if integrity_cache or company_quality_cache:
                output = json.loads(
                    self._objects.get_bounded(
                        str(run["output_ref"]), MAX_OUTPUT_BYTES
                    ).decode("utf-8")
                )
                companies = validate_output_document(output, expected_schema_version=contact_policy.output_schema(configuration))["companies"]
                position = int(run["icp_position"])
                scoring_input = scoring.build_scoring_input(
                    scored_run_id=str(run["run_id"]),
                    icp=integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(configuration)),
                    companies=companies,
                    policy=policy,
                    evaluation_date=str(round_row.get("evaluation_date") or ""),
                    contact_source_evidence=(contact_evidence.resolve_sources(self._store, run, companies) if contact_policy.enabled(configuration) else None),
                    provider_observations=(
                        provider_observations.resolve_observations(
                            self._store, run, companies,
                            str(round_row.get("evaluation_date") or ""),
                        )
                        if (
                            intent_details_policy.enabled(configuration)
                            and provider_observations.enabled(policy)
                        ) else None
                    ),
                )
                if company_quality_cache:
                    work_item["company_judgment_refs"] = (
                        company_judgments.build_company_scopes(
                            scoring_input=scoring_input,
                            round_id=round_id,
                            network_name=str(
                                round_row.get("arena_network_name") or ""
                            ),
                            netuid=int(round_row.get("arena_netuid") or 0),
                            scorer_image_digest=str(
                                configuration.get("scorer_image_digest") or ""
                            ),
                            scorer_image_reference=str(
                                configuration.get("scorer_image_reference") or ""
                            ),
                            integrity_policy=str(
                                configuration.get("integrity_policy") or ""
                            ),
                            company_quality_policy=str(
                                configuration.get("company_quality_policy") or ""
                            ),
                            company_judgment_policy=str(configuration.get("company_judgment_policy") or ""),
                        )
                    )
                else:
                    cache_scope = judgment_cache.build_cache_scope(
                        scoring_input=scoring_input,
                        round_id=round_id,
                        network_name=str(round_row.get("arena_network_name") or ""),
                        netuid=int(round_row.get("arena_netuid") or 0),
                        scorer_image_digest=str(configuration.get("scorer_image_digest") or ""),
                        scorer_image_reference=str(configuration.get("scorer_image_reference") or ""),
                        integrity_policy=str(configuration.get("integrity_policy") or ""),
                    )
                    work_item.update({
                        "judgment_cache_key": cache_scope["cache_key"],
                        "judgment_scope_doc": cache_scope,
                        "judgment_input_hash": cache_scope["scoring_input_hash"],
                    })
            items.append(work_item)
        if integrity_cache:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            participant_by_submission = {
                str(participant["submission_id"]): participant
                for participant in round_row.get("participants") or []
            }
            for item in items:
                participant = participant_by_submission.get(
                    str(item["submission_id"])
                ) or {}
                while True:
                    cache_key = str(item["judgment_cache_key"])
                    cached = self._store.get_judgment_cache(cache_key)
                    if cached is None:
                        grouped.setdefault(cache_key, []).append(item)
                        break
                    try:
                        judgment_cache.validate_evidence_snapshot(
                            cached.get("evidence_doc") or {},
                            cache_key=cache_key,
                            evidence_hash=str(cached.get("evidence_hash") or ""),
                        )
                    except judgment_cache.JudgmentCacheError as exc:
                        raise ServiceError("judgment_cache_invalid", 500) from exc
                    evidence = cached.get("evidence_doc") or {}
                    excluded = set(evidence["runner_authority_exclusions"])
                    if bool(participant.get("is_king")) or str(
                        participant.get("miner_hotkey") or ""
                    ) not in excluded:
                        item["reuse_cache_key"] = cache_key
                        break
                    try:
                        partitioned = judgment_cache.partition_cache_scope(
                            item["judgment_scope_doc"],
                            incompatible_hotkeys=list(excluded),
                        )
                    except judgment_cache.JudgmentCacheError as exc:
                        raise ServiceError("judgment_cache_authority_invalid", 500) from exc
                    item.update({
                        "judgment_cache_key": partitioned["cache_key"],
                        "judgment_scope_doc": partitioned,
                    })
            for cache_key, pending in grouped.items():
                group_miners = sorted({
                    str((participant_by_submission.get(str(item["submission_id"])) or {}).get("miner_hotkey") or "")
                    for item in pending
                } - {""})
                for index, item in enumerate(sorted(pending, key=lambda row: str(row["scored_run_id"]))):
                    item["judgment_group_leader"] = index == 0
                    item["judgment_group_miner_hotkeys"] = group_miners
        return items

    @staticmethod
    def _is_baseline_recovery(run: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> bool:
        configuration = row.get("configuration_doc") or {}
        baselines = [p for p in row.get("participants") or [] if p.get("is_king")]
        if (configuration.get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
            or configuration.get("sourcing_cost_eligibility_policy") not in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
            or configuration.get("integrity_policy") != integrity.POLICY
            or quality_policy.enabled(configuration)
            or type(configuration.get("max_attempts_per_assignment")) is not int
            or configuration["max_attempts_per_assignment"] < 1
            or len(baselines) != 1 or run.get("stage") != 1
            or run.get("round_id") != row.get("round_id")
            or run.get("submission_id") != baselines[0].get("submission_id")
            or run.get("miner_hotkey") != baselines[0].get("miner_hotkey")):
            return False
        assignment = "%s:%s:1:%s%s" % (row["round_id"], run["submission_id"], run["icp_position"],
            ":score" if run.get("kind") == "score" else "")
        if (run.get("assignment_id") != assignment
            or run.get("run_id") != assignment + ":" + str(run.get("attempt"))
            or run.get("champion_restart_required")):
            return False
        if run.get("kind") == "score":
            source = next((r for r in runs if r.get("run_id") == run.get("scored_run_id")), {})
            if (source.get("kind") == "execute" and source.get("status") == "accepted"
                and source.get("terminal_cause") == "accepted" and bool(source.get("output_ref"))
                and all(source.get(k) == run.get(k) for k in ("round_id", "submission_id", "miner_hotkey", "stage", "icp_position"))
                and ArenaService._is_baseline_recovery(source, runs, row)):
                return True
            if run.get("attempt") != 3:
                return False
        attempt = run.get("attempt")
        if (run.get("kind") not in ("execute", "score") or type(attempt) is not int
            or (run.get("kind") == "execute"
                and attempt <= configuration["max_attempts_per_assignment"])):
            return False
        parents = [r for r in runs if r.get("assignment_id") == assignment
            and type(r.get("attempt")) is int and 1 <= r["attempt"] < attempt]
        causes = {"provider_error", "lease_expired"} if run.get("kind") == "execute" else {"judge_error", "judge_timeout", "lease_expired"}
        return (len(parents) == attempt - 1
            and {r["attempt"] for r in parents} == set(range(1, attempt)) and all(
            r.get("status") == "failed" and r.get("terminal_cause") in causes
            and (run.get("kind") != "execute" or r.get("output_ref") is None)
            and (r.get("terminal_cause") == "lease_expired"
                or (isinstance(r.get("result_doc"), Mapping)
                    and r["result_doc"].get("terminal_status") == r.get("terminal_cause")))
            and not r.get("champion_restart_required")
            and r.get("run_id") == assignment + ":" + str(r["attempt"])
            and all(r.get(k) == run.get(k) for k in ("round_id", "submission_id", "miner_hotkey", "stage", "icp_position", "kind", "scored_run_id"))
            for r in parents))

    def _baseline_recovery_plan(self, row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], judgments: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        """Verify complete original baseline membership before any late score write."""
        baseline = next((p for p in row.get("participants") or [] if p.get("is_king")), {})
        positions = set(contracts.execution_positions(1, contracts.BASELINE_SCORED_FIRST_POLICY, row["configuration_doc"]))
        accepted = [r for r in runs if r.get("submission_id") == baseline.get("submission_id") and r.get("stage") == 1 and r.get("status") == "accepted"]
        if len(accepted) != len(positions) or {r.get("icp_position") for r in accepted} != positions:
            return None
        chosen = self._select_scoring_outputs(judgments, round_row=row)
        for run in accepted:
            assignment = "%s:%s:1:%s" % (row["round_id"], baseline["submission_id"], run["icp_position"])
            judge = chosen.get(run["run_id"], {})
            if (run.get("round_id") != row["round_id"] or run.get("miner_hotkey") != baseline.get("miner_hotkey")
                or run.get("terminal_cause") != "accepted" or not run.get("output_ref")
                or run.get("assignment_id") != assignment or run.get("run_id") != assignment + ":" + str(run["attempt"])
                or judge.get("assignment_id") != assignment + ":score"
                or judge.get("run_id") != assignment + ":score:" + str(judge.get("attempt"))
                or not judge.get("output_ref") or judge.get("terminal_cause") != "accepted"
                or not self._same_scoring_membership(judge, run, round_id=str(row["round_id"]))):
                return None
        return scoring.build_scoring_plan(round_id=str(row["round_id"]), stage=1, runs=list(accepted),
            execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY, configuration=row["configuration_doc"])

    def _recover_baseline(self, row: Mapping[str, Any]) -> None:
        if ((row.get("configuration_doc") or {}).get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
            or quality_policy.enabled(row.get("configuration_doc") or {})):
            return
        round_id = str(row["round_id"])
        self._store.recover_baseline(round_id)
        runs = self._store.list_runs(round_id, stage=1, kind="execute")
        judges = self._store.list_runs(round_id, stage=1, kind="score")
        pending = [run for run in runs if run.get("status") == "accepted"
            and self._is_baseline_recovery(run, runs, row)
            and not any(j.get("scored_run_id") == run["run_id"] for j in judges)]
        if pending:
            plan = {"work_items": [{"scored_run_id": r["run_id"], "submission_id": r["submission_id"],
                "icp_position": r["icp_position"], "output_ref": r["output_ref"]} for r in pending]}
            self._store.recover_baseline(round_id, self._scoring_work_items(row, 1, plan))

    def _score_recovered_baseline(self, row: Mapping[str, Any]) -> None:
        round_id = str(row["round_id"])
        runs = self._store.list_runs(round_id, stage=1, kind="execute")
        judges = self._store.list_runs(round_id, stage=1, kind="score")
        if (not self._score_generation_floor(row)
            and not (_judge_continuation_enabled(row)
                and (row.get("configuration_doc") or {}).get("execution_sequence_policy")
                    == contracts.BASELINE_SCORED_FIRST_POLICY)
            and not any(self._is_baseline_recovery(r, runs + judges, row) for r in runs + judges)):
            return
        plan = self._baseline_recovery_plan(row, runs, judges)
        if plan is None:
            return  # Miner scores persist; final authority remains held by the baseline gate.
        outputs = self._outputs_by_run(round_id, 1)
        chosen = self._select_scoring_outputs(judges, round_row=row)
        icps = self.evaluation_icps(round_id)
        policy = row["configuration_doc"]["scorer_policy"]
        breakdowns = {item["scored_run_id"]: self._verified_breakdowns(chosen[item["scored_run_id"]],
            icp=icps[item["icp_position"]], companies=outputs[item["scored_run_id"]], policy=policy)
            for item in plan["work_items"]}
        scores = scoring.build_stage_scores(plan=plan, policy=policy, icps_by_position=dict(enumerate(icps)),
            outputs_by_run=outputs, breakdowns_by_item=breakdowns, configuration=row["configuration_doc"])
        recorded = self._record_current_run_scores(row, 1, scoring.run_scores_for_store(scores, runs))
        if recorded.get("status") != "ok":
            raise ServiceError("baseline_recovery_scores_not_recorded", 500)

    def _scoring_runs_for_stage(self, round_id: str, stage: int) -> List[Dict[str, Any]]:
        row = self._round(round_id)
        runs = self._store.list_runs(round_id, stage=stage, kind="score")
        if stage == 2:
            if (row.get("configuration_doc") or {}).get("execution_sequence_policy") == contracts.BASELINE_SCORED_FIRST_POLICY:
                original = self._store.list_runs(round_id, stage=1)
                # Ordinary first attempts also continue while miners execute.
                # Keep the same scoring-floor fence as every other judgment.
                runs += [r for r in original if self._is_baseline_recovery(r, original, row)
                         or (r.get("kind") == "score" and (
                             _judge_continuation_enabled(row)
                             or self._score_generation_floor(row) > 0))]
        return self._current_score_runs(row, runs)

    def scoring_is_complete(self, round_id: str, stage: int) -> bool:
        runs = self._scoring_runs_for_stage(round_id, stage)
        return all(run["status"] in ("accepted", "failed") for run in runs)

    def close_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        return self._store.close_scoring(round_id, stage)

    @staticmethod
    def _score_generation_floor(row: Mapping[str, Any]) -> int:
        floor = row.get("scoring_generation_floor", 0)
        if type(floor) is not int or floor < 0:
            raise ServiceError("scoring_generation_floor_invalid", 500)
        return floor

    @classmethod
    def _current_score_runs(cls, row, runs):
        """Keep historical judgments readable without treating them as current."""
        floor = cls._score_generation_floor(row)
        return [run for run in runs
                if int(run.get("scoring_generation_floor") or 0) == floor]

    def _record_current_run_scores(self, row, stage, scores):
        """Bind score writes to the source round's exact policy revision."""
        return self._store.record_run_scores(str(row["round_id"]), stage, scores,
            scoring_generation_floor=self._score_generation_floor(row))

    @staticmethod
    def _select_scoring_outputs(
        runs: Sequence[Mapping[str, Any]], *,
        round_row: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Select current-policy authority, retaining accepted retry precedence."""
        if round_row is not None:
            runs = ArenaService._current_score_runs(round_row, runs)

        def rank(item: Mapping[str, Any]) -> Tuple[bool, int, int]:
            return (
                item["status"] == "accepted",
                int(item.get("stage_generation") or 0),
                int(item.get("attempt") or 0),
            )

        chosen: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            current = chosen.get(run["scored_run_id"])
            if current is None or rank(run) > rank(current):
                chosen[run["scored_run_id"]] = dict(run)
        return chosen

    def _scoring_outputs(self, round_id: str, stage: int, *, round_row=None) -> Dict[str, Dict[str, Any]]:
        """The score run that counts for each scored execution run."""

        return self._select_scoring_outputs(
            self._store.list_runs(round_id, stage=stage, kind="score"),
            round_row=round_row if round_row is not None else self._round(round_id),
        )

    def _company_judgment_evidence(
        self,
        *,
        cache_key: str,
        company_input_hash: str,
        authority_slot: int,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read and fully bind one immutable company judgment variant."""

        reads = evidence_reads if isinstance(evidence_reads, _PublicResultEvidenceReads) else None
        # Public results and publication supply this call-local map. Keep invalid
        # argument types on the original validation path (bool/float slots
        # must not alias an already validated integer slot).
        identity = (
            (cache_key, company_input_hash, authority_slot)
            if type(cache_key) is str and type(company_input_hash) is str
            and type(authority_slot) is int else None
        )
        if evidence_reads is not None and identity is not None and identity in evidence_reads:
            return copy.deepcopy(evidence_reads[identity])
        cached = (reads.company(self._store, cache_key, authority_slot) if reads is not None
                  else self._store.get_company_judgment(cache_key, authority_slot))
        if cached is None:
            raise scoring.ScoringError(
                "accepted company judgment cache entry is missing"
            )
        if cached.get("company_input_hash") != company_input_hash:
            raise scoring.ScoringError(
                "accepted company judgment input binding is invalid"
            )
        try:
            company_judgments.validate_company_ref({
                "schema_version": company_judgments.COMPANY_REF_SCHEMA_VERSION,
                "company_index": 0,
                "cache_key": cache_key,
                "company_input_hash": company_input_hash,
                "scope_doc": cached.get("scope_doc") or {},
            })
            evidence = company_judgments.validate_evidence_snapshot(
                cached.get("evidence_doc") or {},
                cache_key=cache_key,
                company_input_hash=company_input_hash,
                authority_slot=authority_slot,
                evidence_hash=str(cached.get("evidence_hash") or ""),
            )
            company_judgments.validate_raw_judgment_mode(evidence["raw_judgment"], cached.get("scope_doc") or {})
        except company_judgments.CompanyJudgmentError as exc:
            raise scoring.ScoringError(
                "accepted company judgment evidence is invalid"
            ) from exc
        source_id = str(evidence["source_score_run_id"])
        source = reads.run(self._store, source_id) if reads is not None else self._store.get_run(source_id)
        if (
            source is None
            or source.get("kind") != "score"
            or source.get("status") not in ("accepted", "failed")
            or source.get("runner_hotkey") != evidence["source_runner_hotkey"]
            or source.get("scored_run_id") != evidence["source_scored_run_id"]
            or source.get("output_ref") != evidence["source_output_ref"]
            or source.get("claim_request_id")
            != evidence["source_claim_request_id"]
            or source.get("claim_request_hash")
            != evidence["source_claim_request_hash"]
            or int(source.get("lease_generation") or 0)
            != evidence["source_lease_generation"]
            or (
                (reads.authority_exclusions(self._store, str(source["run_id"]))
                 if reads is not None else self._store.signal_authority_exclusions(str(source["run_id"])))
                if ((self._round_from_context(str(source["round_id"]), round_context) or {}).get("configuration_doc") or {}).get("scorer_policy", {}).get("signal_judgment_policy") == signal_judgments.POLICY
                else (source.get("claim_response") or {}).get("runner_authority_exclusions")
            ) != evidence["runner_authority_exclusions"]
            or cached.get("source_score_run_id")
            != evidence["source_score_run_id"]
            or cached.get("source_scored_run_id")
            != evidence["source_scored_run_id"]
            or cached.get("source_runner_hotkey")
            != evidence["source_runner_hotkey"]
        ):
            raise scoring.ScoringError(
                "accepted company judgment authority is invalid"
            )
        try:
            get_object = (lambda ref, limit: reads.object(self._objects, ref, limit)) if reads is not None else self._objects.get_bounded
            source_output = json.loads(
                get_object(
                    str(evidence["source_output_ref"]),
                    scoring.MAX_SCORING_OUTPUT_BYTES,
                ).decode("utf-8")
            )
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise scoring.ScoringError(
                "accepted company judgment source output is unavailable"
            ) from exc
        if contracts.document_hash(source_output) != evidence["source_output_hash"]:
            raise scoring.ScoringError(
                "accepted company judgment source output hash is invalid"
            )
        try:
            validated_output = scoring.validate_scoring_output_document(
                source_output
            )
        except scoring.ScoringError as exc:
            raise scoring.ScoringError(
                "accepted company judgment source output is invalid"
            ) from exc
        if source.get("status") == "failed":
            result = source.get("result_doc") or {}
            terminal = result.get("terminal_status")
            if (terminal not in ("judge_error", "judge_timeout", "credential_error")
                or (source.get("terminal_cause") != terminal
                    and not (terminal == "judge_timeout" and source.get("terminal_cause") == "judge_error"))
                or result.get("company_judgment_completion_request_hash")
                != evidence["source_completion_request_hash"]
                or "failure" not in validated_output
                or (terminal != "credential_error"
                    and validated_output["failure"] != terminal)):
                raise scoring.ScoringError("partial company judgment completion is invalid")
            try:
                company_judgments.validate_new_company_judgments(
                    validated_output.get("company_judgments") or [],
                    lease_context=(source.get("claim_response") or {}).get("company_judgment_cache") or {},
                    require_complete=False,
                )
            except company_judgments.CompanyJudgmentError as exc:
                raise scoring.ScoringError("partial company judgment source lease is invalid") from exc
        elif "failure" in validated_output:
            raise scoring.ScoringError("accepted company judgment contains a failure")
        matching_rows = [
            row for row in validated_output.get("company_judgments") or []
            if row.get("cache_key") == cache_key
            and row.get("company_input_hash") == company_input_hash
            and row.get("authority_slot") == authority_slot
        ]
        if (
            validated_output.get("scored_run_id")
            != evidence["source_scored_run_id"]
            or len(matching_rows) != 1
            or matching_rows[0].get("raw_judgment")
            != evidence["raw_judgment"]
        ):
            raise scoring.ScoringError(
                "accepted company judgment source output is not bound"
            )
        if evidence_reads is not None and identity is not None:
            # A consumer cannot alter the successful proof used by a later
            # consumer. Failures never populate the map.
            evidence_reads[identity] = copy.deepcopy(evidence)
        return evidence

    def _legacy_partial_input(self, run, round_row):
        """Reconstruct the immutable full-group input, never a sparse subset."""
        configuration = round_row.get("configuration_doc") or {}
        policy = configuration.get("scorer_policy") or {}
        if (not integrity.enabled(configuration) or company_judgments.enabled(configuration)
            or company_judgments.enabled(policy)):
            raise scoring.ScoringError("legacy checkpoint policy is invalid")
        executed = self._store.get_run(str(run.get("scored_run_id") or ""))
        if (executed is None or executed.get("kind") != "execute"
            or executed.get("status") != "accepted" or not executed.get("output_ref")
            or any(executed.get(key) != run.get(key) for key in
                   ("round_id", "submission_id", "miner_hotkey", "stage", "icp_position"))):
            raise scoring.ScoringError("legacy checkpoint execution is invalid")
        output = validate_output_document(
            json.loads(self._objects.get_bounded(str(executed["output_ref"]), MAX_OUTPUT_BYTES)),
            expected_schema_version=contact_policy.output_schema(configuration),
        )
        companies = output["companies"]
        icp = integrity.agent_visible_icp(
            self.evaluation_icps(str(round_row["round_id"]))[int(run["icp_position"])],
            contacts_required=contact_policy.enabled(configuration),
        )
        document = scoring.build_scoring_input(
            scored_run_id=str(run["scored_run_id"]), icp=icp,
            companies=companies, policy=policy,
            evaluation_date=str(round_row.get("evaluation_date") or ""),
            contact_source_evidence=(contact_evidence.resolve_sources(self._store, executed, companies)
                                     if contact_policy.enabled(configuration) else None),
            provider_observations=(provider_observations.resolve_observations(
                self._store, executed, companies, str(round_row.get("evaluation_date") or ""))
                if intent_details_policy.enabled(configuration) and provider_observations.enabled(policy) else None),
        )
        return document, contracts.document_hash(output)

    @staticmethod
    def _legacy_partial_receipt(run, round_row, scoring_input, execution_hash, output, completion_hash):
        configuration = round_row.get("configuration_doc") or {}
        return {
            "schema_version": "leadpoet.lab_arena.legacy_partial.v1",
            **{key: run.get(key) for key in ("round_id", "submission_id", "miner_hotkey", "stage", "icp_position", "scored_run_id")},
            "source_score_run_id": run["run_id"],
            "source_runner_hotkey": run.get("runner_hotkey"),
            "source_claim_request_id": run.get("claim_request_id"),
            "source_claim_request_hash": run.get("claim_request_hash"),
            "source_lease_generation": int(run.get("lease_generation") or 0),
            "runner_authority_exclusions": (run.get("claim_response") or {}).get("runner_authority_exclusions"),
            "source_output_ref": "arena/%s/scores/items/%s.json" % (round_row["round_id"], run["run_id"]),
            "source_output_hash": contracts.document_hash(output),
            "scoring_input_hash": contracts.document_hash(scoring_input),
            "execution_output_hash": execution_hash,
            "scorer_image_digest": configuration.get("scorer_image_digest"),
            "scorer_image_reference": configuration.get("scorer_image_reference"),
            "scorer_policy": configuration.get("scorer_policy"),
            "evaluation_date": str(round_row.get("evaluation_date") or ""),
            "terminal_count": len(output.get("terminal_breakdowns") or []),
            "source_completion_request_hash": completion_hash,
        }

    def _validated_legacy_partial(self, source, round_row):
        receipt = (source.get("result_doc") or {}).get("legacy_partial_checkpoint")
        if (source.get("kind") != "score" or source.get("status") != "failed"
            or source.get("terminal_cause") not in ("judge_error", "judge_timeout", "credential_error")
            or (source.get("result_doc") or {}).get("terminal_status") != source.get("terminal_cause")
            or not isinstance(receipt, dict) or not source.get("output_ref")):
            raise scoring.ScoringError("legacy checkpoint source is not committed")
        completion_hash = receipt.get("source_completion_request_hash")
        if not isinstance(completion_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", completion_hash):
            raise scoring.ScoringError("legacy checkpoint completion is unbound")
        self._require_validator_authority(str(source.get("runner_hotkey") or ""))
        submission = self._store.get_submission(str(source.get("submission_id") or "")) or {}
        exclusions = receipt.get("runner_authority_exclusions")
        if (source.get("runner_hotkey") in self._banned_hotkeys(round_row)
            or not isinstance(exclusions, list)
            or (not submission.get("is_king") and source.get("miner_hotkey") in exclusions)):
            raise scoring.ScoringError("legacy checkpoint authority is invalid")
        output = scoring.validate_scoring_output_document(json.loads(
            self._objects.get_bounded(str(source["output_ref"]), scoring.MAX_SCORING_OUTPUT_BYTES)))
        document, execution_hash = self._legacy_partial_input(source, round_row)
        expected = self._legacy_partial_receipt(source, round_row, document, execution_hash, output, completion_hash)
        if (receipt != expected or source["output_ref"] != expected["source_output_ref"]
            or output.get("scored_run_id") != source.get("scored_run_id")
            or "failure" not in output or not output.get("terminal_breakdowns")
            or (source.get("terminal_cause") != "credential_error" and output["failure"] != source["terminal_cause"])):
            raise scoring.ScoringError("legacy checkpoint input or artifact changed")
        return scoring.validate_terminal_breakdowns_for_work_item(
            output["terminal_breakdowns"], icp=document["icp"], companies=document["companies"], policy=document["scorer_policy"],
        )

    def _prior_legacy_terminal_breakdowns(self, run, round_row):
        sources = self._store.list_runs(str(round_row["round_id"]), kind="score", submission_id=str(run["submission_id"]))
        sources = [source for source in self._current_score_runs(round_row, sources)
                   if source.get("scored_run_id") == run.get("scored_run_id")
                   and source.get("assignment_id") == run.get("assignment_id")
                   and int(source.get("attempt") or 0) < int(run.get("attempt") or 0)
                   and (source.get("result_doc") or {}).get("legacy_partial_checkpoint")]
        if not sources:
            return []
        source = max(sources, key=lambda item: int(item.get("attempt") or 0))
        return self._validated_legacy_partial(source, round_row)

    def _validated_company_judgment_lease(
        self,
        context: Mapping[str, Any],
        *,
        recipient_miner_hotkey: str,
        baseline: bool,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            lease = company_judgments.validate_lease_context(context)
        except company_judgments.CompanyJudgmentError as exc:
            raise scoring.ScoringError(
                "company judgment lease context is invalid"
            ) from exc
        for hit in lease["hits"]:
            evidence = self._company_judgment_evidence(
                cache_key=hit["cache_key"],
                company_input_hash=hit["company_input_hash"],
                authority_slot=hit["authority_slot"],
                evidence_reads=evidence_reads,
                round_context=round_context,
            )
            if evidence != hit["evidence_doc"]:
                raise scoring.ScoringError(
                    "company judgment lease evidence changed"
                )
            if (
                not baseline
                and recipient_miner_hotkey
                in evidence["runner_authority_exclusions"]
            ):
                raise scoring.ScoringError(
                    "company judgment lease authority is incompatible"
                )
        return lease

    def _company_quality_breakdowns(
        self,
        run: Mapping[str, Any],
        *,
        icp: Mapping[str, Any],
        companies: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Rebuild list context from immutable raw company evidence."""

        reads = evidence_reads if isinstance(evidence_reads, _PublicResultEvidenceReads) else None
        submission_id = str(run["submission_id"])
        submission = (reads.submission(self._store, submission_id)
                      if reads is not None else self._store.get_submission(submission_id)) or {}
        context = self._validated_company_judgment_lease(
            (run.get("claim_response") or {}).get("company_judgment_cache") or {},
            recipient_miner_hotkey=str(run.get("miner_hotkey") or ""),
            baseline=bool(submission.get("is_king")),
            evidence_reads=evidence_reads,
            round_context=round_context,
        )
        try:
            get_object = (lambda ref, limit: reads.object(self._objects, ref, limit)) if reads is not None else self._objects.get_bounded
            document = json.loads(
                get_object(
                    str(run["output_ref"]), scoring.MAX_SCORING_OUTPUT_BYTES
                ).decode("utf-8")
            )
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise scoring.ScoringError("scoring output is not valid JSON") from exc
        output = scoring.validate_scoring_output_document(document)
        if output["scored_run_id"] != run["scored_run_id"]:
            raise scoring.ScoringError("scoring output names the wrong execution run")
        try:
            company_judgments.validate_new_company_judgments(
                output.get("company_judgments") or [],
                lease_context=context,
                require_complete=quality_policy.scorer_enabled(policy),
            )
        except company_judgments.CompanyJudgmentError as exc:
            raise scoring.ScoringError(
                "scoring output company judgments are invalid"
            ) from exc

        rows = sorted(
            context["hits"] + context["misses"],
            key=lambda item: item["company_index"],
        )
        raw_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
        raw = []
        supplied_keys = {item["cache_key"] for item in output.get("company_judgments") or []}
        hit_keys = {item["cache_key"] for item in context["hits"]}
        for item in rows:
            identity = (item["cache_key"], item["authority_slot"])
            if (not quality_policy.scorer_enabled(policy) and item["cache_key"] not in supplied_keys | hit_keys):
                # Reconstruct this accepted lease, even if a later assignment
                # subsequently judged an intrinsic row this lease skipped.
                raw.append(None)
                continue
            if identity not in raw_by_key:
                evidence = self._company_judgment_evidence(
                    cache_key=item["cache_key"],
                    company_input_hash=item["company_input_hash"],
                    authority_slot=item["authority_slot"],
                    evidence_reads=evidence_reads,
                    round_context=round_context,
                )
                if (
                    not bool(submission.get("is_king"))
                    and str(run.get("miner_hotkey") or "")
                    in evidence["runner_authority_exclusions"]
                ):
                    raise scoring.ScoringError(
                        "accepted company judgment authority is incompatible"
                    )
                raw_by_key[identity] = evidence["raw_judgment"]
            raw.append(raw_by_key[identity])
        consumed_indexes: set[int] = set()
        recomputed = scoring.project_company_judgments(
            list(companies)[:len(rows)],
            raw,
            contacts_required=contact_policy.scorer_enabled(policy),
            company_quality=quality_policy.scorer_enabled(policy),
            scored_indexes=scoring.verify.bucket_skip(icp, companies, max_scored_companies=int(policy["max_scored_companies"]))[0],
            consumed_indexes=consumed_indexes,
        )
        if not quality_policy.scorer_enabled(policy):
            try:
                stored_indexes = company_judgments.validated_consumed_indexes(run.get("result_doc") or {}, company_count=len(rows))
            except company_judgments.CompanyJudgmentError as exc:
                raise scoring.ScoringError("company judgment consumed indexes are invalid") from exc
            if stored_indexes != sorted(consumed_indexes):
                raise scoring.ScoringError("company judgment consumed indexes differ from replay")
        if output["breakdowns"] != recomputed:
            raise scoring.ScoringError(
                "scoring output company context does not match the gateway"
            )
        return scoring.validate_breakdowns_for_item(
            recomputed,
            icp=icp,
            companies=companies,
            max_scored_companies=int(policy["max_scored_companies"]),
            integrity_policy=True,
            contacts_required=contact_policy.scorer_enabled(policy),
            company_quality=quality_policy.scorer_enabled(policy),
        )

    def _verified_breakdowns(
        self,
        run: Mapping[str, Any],
        *,
        icp: Mapping[str, Any],
        companies: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        cached_rows: Optional[Mapping[str, Any]] = None,
        source_rows: Optional[Mapping[str, Any]] = None,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if run.get("company_judgment_refs") is not None:
            return self._company_quality_breakdowns(
                run, icp=icp, companies=companies, policy=policy,
                evidence_reads=evidence_reads,
                round_context=round_context,
            )
        cache_key = str(run.get("judgment_cache_key") or "")
        if cache_key:
            cached = (
                cached_rows.get(cache_key)
                if cached_rows is not None and cache_key in cached_rows
                else self._store.get_judgment_cache(cache_key)
            )
            if cached is None:
                raise scoring.ScoringError("accepted judgment cache entry is missing")
            try:
                evidence = judgment_cache.validate_evidence_snapshot(
                    cached.get("evidence_doc") or {},
                    cache_key=cache_key,
                    evidence_hash=str(cached.get("evidence_hash") or ""),
                )
            except judgment_cache.JudgmentCacheError as exc:
                raise scoring.ScoringError("accepted judgment cache evidence is invalid") from exc
            source_id = str(evidence["source_score_run_id"])
            source = (
                source_rows.get(source_id)
                if source_rows is not None and source_id in source_rows
                else self._store.get_run(source_id)
            )
            if (
                source is None
                or source.get("kind") != "score"
                or source.get("status") != "accepted"
                or source.get("runner_hotkey") != evidence["source_runner_hotkey"]
                or source.get("scored_run_id") != evidence["source_scored_run_id"]
                or cached.get("source_score_run_id") != evidence["source_score_run_id"]
                or cached.get("scoring_input_hash") != evidence["scoring_input_hash"]
                or run.get("judgment_cache_source_run_id") != evidence["source_score_run_id"]
            ):
                raise scoring.ScoringError("accepted judgment cache authority is invalid")
            return scoring.validate_breakdowns_for_item(
                evidence["breakdowns"],
                icp=icp,
                companies=companies,
                max_scored_companies=int(policy["max_scored_companies"]),
                integrity_policy=contact_policy.integrity_adapter(policy.get("scoring_adapter_version")),
                contacts_required=contact_policy.scorer_enabled(policy),
            )
        try:
            document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise scoring.ScoringError("scoring output is not valid JSON") from exc
        output = scoring.validate_scoring_output_document(document)
        if output["scored_run_id"] != run["scored_run_id"]:
            raise scoring.ScoringError("scoring output names the wrong execution run")
        if "breakdowns" not in output:
            raise scoring.ScoringError("accepted scoring output contains a failure")
        return scoring.validate_breakdowns_for_item(
            output["breakdowns"], icp=icp, companies=companies,
            max_scored_companies=int(policy["max_scored_companies"]),
            integrity_policy=contact_policy.integrity_adapter(policy.get("scoring_adapter_version")),
            contacts_required=contact_policy.scorer_enabled(policy),
        )

    def score_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Assemble the stage bundle from configured-runner scoring results."""

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_judged" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        policy = contracts.validate_scorer_policy(round_row["configuration_doc"]["scorer_policy"])
        plan = self._load_scoring_plan(round_row, stage)
        icps = self.evaluation_icps(round_id)
        outputs = self._outputs_by_run(round_id, stage)
        chosen = self._scoring_outputs(round_id, stage, round_row=round_row)
        baseline_ids = {
            str(participant["submission_id"])
            for participant in round_row.get("participants") or []
            if participant.get("is_king")
        }
        if len(baseline_ids) != 1:
            raise ServiceError("baseline_submission_invalid", 500)
        baseline_execution_incomplete = any(
            item["submission_id"] in baseline_ids
            for item in plan.get("incomplete_rows") or []
        )
        failed_items: Dict[str, str] = {}
        incomplete_items: Dict[str, str] = {}
        terminal_credit_items: set[str] = set()
        for item in plan["work_items"]:
            scored_run_id = item["scored_run_id"]
            run = chosen.get(scored_run_id)
            if run is None:
                incomplete_items[scored_run_id] = "stage_closed"
                continue
            if run["status"] == "accepted":
                continue
            cause = str(run.get("terminal_cause") or "")
            if run["status"] in ("pending", "leased", "submitted"):
                incomplete_items[scored_run_id] = "stage_closed"
                continue
            if (
                run["status"] != "failed"
                or cause not in contracts.TERMINAL_CAUSES
                or cause == "accepted"
            ):
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            if cause in (contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed", "credential_error"}):
                incomplete_items[scored_run_id] = cause
                if (_judge_continuation_enabled(round_row)
                    and self._terminal_miner_credit_judgment(scored_run_id)):
                    terminal_credit_items.add(scored_run_id)
                continue
            # Only an explicit terminal score failure receives a frozen zero.
            failed_items[scored_run_id] = cause
        final_deadline = _parse_iso(round_row["configuration_doc"]["schedule"]["publication_deadline"])
        isolate_incomplete = (
            stage == 2 and self.now() >= final_deadline
            and not any(item["submission_id"] in baseline_ids
                and item["scored_run_id"] in incomplete_items for item in plan["work_items"])
            and not self._evaluation_draining(round_row)
        )
        if not isolate_incomplete and incomplete_items.keys() - terminal_credit_items and (
            stage != 1 or
            round_row["configuration_doc"].get("execution_sequence_policy")
                != contracts.BASELINE_SCORED_FIRST_POLICY
            or self.now() >= _parse_iso(
                round_row["configuration_doc"]["schedule"]["publication_deadline"])
        ):
            return {**self._hold_incomplete_evaluation(
                round_row, reason="judge_incomplete"),
                "remaining": len(incomplete_items)}
        baseline_judge_incomplete = any(
            item["submission_id"] in baseline_ids
            and incomplete_items.get(item["scored_run_id"]) == "stage_closed"
            for item in plan["work_items"]
        )
        breakdowns_by_item: Dict[str, List[Dict[str, Any]]] = {}
        judge_executions = 0
        for item in plan["work_items"]:
            scored_run_id = item["scored_run_id"]
            if scored_run_id in failed_items or scored_run_id in incomplete_items:
                continue
            run = chosen.get(scored_run_id)
            icp = icps[int(item["icp_position"])]
            companies = outputs[scored_run_id]
            try:
                breakdowns_by_item[scored_run_id] = self._verified_breakdowns(
                    run, icp=icp, companies=companies, policy=policy
                )
            except scoring.ScoringError:
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            judge_executions += 1
        if failed_items or incomplete_items:
            plan = {
                **plan,
                "work_items": [
                    item for item in plan["work_items"]
                    if item["scored_run_id"] not in failed_items
                    and item["scored_run_id"] not in incomplete_items
                ],
                "incomplete_rows": list(plan.get("incomplete_rows") or []) + [
                    {"submission_id": item["submission_id"],
                     "icp_position": item["icp_position"],
                     "cause": incomplete_items[item["scored_run_id"]]}
                    for item in plan["work_items"]
                    if item["scored_run_id"] in incomplete_items
                ],
                "zero_rows": list(plan["zero_rows"]) + [
                    {
                        "submission_id": item["submission_id"],
                        "icp_position": item["icp_position"],
                        "cause": failed_items[item["scored_run_id"]],
                    }
                    for item in plan["work_items"]
                    if item["scored_run_id"] in failed_items
                ],
            }
        stage_scores = scoring.build_stage_scores(
            plan=plan,
            policy=policy,
            icps_by_position=dict(enumerate(icps)),
            outputs_by_run=outputs,
            breakdowns_by_item=breakdowns_by_item,
            configuration=round_row.get("configuration_doc"),
        )
        runs = self._store.list_runs(round_id, stage=stage, kind="execute")
        recorded = self._record_current_run_scores(round_row, stage, scoring.run_scores_for_store(stage_scores, runs))
        if recorded.get("status") != "ok":
            # The per-run scores are part of the published result; a write the
            # database refused must stop the stage, never pass silently.
            raise ServiceError("scores_not_recorded:%s" % str(recorded.get("status") or "unknown")[:40], 500)
        deadline = _partial_baseline_transition_deadline(
            round_row, stage=stage,
            execution_incomplete=baseline_execution_incomplete,
            judge_incomplete=baseline_judge_incomplete,
        )
        if deadline is not None and self.now() < deadline:
            return {
                "status": "retry", "round_status": round_row["status"],
                "reason": "partial_baseline_deadline_pending",
                "judge_executions": judge_executions,
            }
        if stage == 1:
            if (
                (round_row.get("configuration_doc") or {}).get(
                    "execution_sequence_policy"
                )
                == contracts.BASELINE_SCORED_FIRST_POLICY
            ):
                finalists = sorted(
                    str(participant["submission_id"])
                    for participant in round_row.get("participants") or []
                    if not participant.get("is_king")
                )
            else:
                ranking = verify.stage1_ranking(
                    self._score_entries_from_runs(round_row, contracts.stage_positions(1, round_row.get("configuration_doc")), "stage1_score")
                )
                finalists = verify.select_finalists(ranking)
            transition = self._store.transition_round(
                round_id,
                "stage1_judged",
                "stage1_scored",
                {"finalists": finalists},
            )
            return {
                "status": transition.get("status"),
                "judge_executions": judge_executions,
                "ineligible_submissions": [],
                "finalists": finalists,
            }
        # Continuation judgments are materialized by publish in the scored
        # phase. Its guarded writer selects only missing projections.
        if not _judge_continuation_enabled(round_row):
            self._score_recovered_baseline(round_row)
        transition = self._store.transition_round(round_id, "stage2_judged", "scored", {})
        return {
            "status": transition.get("status"),
            "judge_executions": judge_executions,
            "ineligible_submissions": [],
        }

    def _score_entries_from_runs(
        self,
        round_row: Mapping[str, Any],
        positions: Sequence[int],
        score_key: str,
        *, publication_reads: Optional[_PublicationReads] = None,
    ) -> List[Dict[str, Any]]:
        """Derive one score per participant from write-once run scores."""

        wanted = set(int(position) for position in positions)
        all_runs = (
            publication_reads.runs if publication_reads is not None
            else self._store.list_runs(str(round_row["round_id"]), kind="execute")
        )
        selected: Dict[Tuple[str, int], Mapping[str, Any]] = {}
        for run in all_runs:
            position = int(run["icp_position"])
            if position not in wanted or run.get("per_icp_score") is None:
                continue
            key = (str(run["submission_id"]), position)
            current = selected.get(key)
            if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
                selected[key] = run
        entries = []
        for participant in round_row.get("participants") or []:
            submission_id = str(participant["submission_id"])
            rows = [selected.get((submission_id, position)) for position in sorted(wanted)]
            if any(row is None for row in rows):
                if score_key == "final_score":
                    entries.append({
                        "submission_id": submission_id,
                        "hotkey": str(participant["miner_hotkey"]),
                        "is_king": bool(participant.get("is_king")),
                        "final_score": None,
                        "execution_incomplete": True,
                    })
                continue
            values = [float(row["per_icp_score"]) for row in rows if row is not None]
            if (
                round_row.get("configuration_doc", {}).get(
                    "sourcing_cost_eligibility_policy"
                )
                in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
            ):
                qualified_by_position = (
                    publication_reads.qualified_counts(submission_id, sorted(wanted))
                    if publication_reads is not None else self._qualified_company_counts(
                        round_row, submission_id, all_runs, positions=sorted(wanted)
                    )
                )
                adjusted = []
                for position, value in zip(sorted(wanted), values):
                    cost = (
                        publication_reads.per_icp_cost(submission_id, position)
                        if publication_reads is not None else self._per_icp_cost_eligibility(
                            round_row, submission_id, position, all_runs,
                            qualified=qualified_by_position.get(position, 0),
                        )
                    )
                    adjusted.append(value if cost["eligible"] else 0.0)
                values = adjusted
            entry = {
                "submission_id": submission_id,
                score_key: verify.stage_score(values, len(wanted)),
                "is_king": bool(participant.get("is_king")),
            }
            if score_key == "final_score":
                entry["hotkey"] = str(participant["miner_hotkey"])
                if not any(row.get("terminal_cause") == "accepted" for row in rows if row is not None):
                    entry[score_key] = None
            entries.append(entry)
        return entries

    @staticmethod
    def _selected_accepted_execution_runs(
        runs: Sequence[Mapping[str, Any]], submission_id: str
    ) -> Dict[int, Mapping[str, Any]]:
        """Select one accepted attempt per ICP, as the scoring plan does."""

        selected: Dict[int, Mapping[str, Any]] = {}
        for run in runs:
            if (
                str(run.get("submission_id") or "") != submission_id
                or run.get("status") != "accepted"
            ):
                continue
            position = int(run.get("icp_position") or 0)
            current = selected.get(position)
            if current is None or int(run.get("attempt") or 0) > int(
                current.get("attempt") or 0
            ):
                selected[position] = run
        return selected

    def _returned_company_count(
        self,
        submission_id: str,
        runs: Sequence[Mapping[str, Any]],
        *, positions: Optional[Sequence[int]] = None,
    ) -> int:
        """Count unique validated company domains within each configured ICP."""

        return sum(self._returned_company_counts(
            submission_id, runs, positions=positions
        ).values())

    def _returned_company_counts(
        self, submission_id: str, runs: Sequence[Mapping[str, Any]],
        *, positions: Optional[Sequence[int]] = None,
        publication_reads: Optional[_PublicationReads] = None,
        evidence_reads: Optional[_PublicResultEvidenceReads] = None,
    ) -> Dict[int, int]:
        selected = self._selected_accepted_execution_runs(runs, submission_id)
        counts: Dict[int, int] = {}
        wanted = set(selected if positions is None else positions)
        for position in sorted(wanted):
            run = selected.get(position)
            if run is None:
                continue
            output_ref = str(run.get("output_ref") or "")
            if not output_ref:
                raise OutputInvalid("accepted output has no object reference")
            if publication_reads is not None:
                output = publication_reads.output(output_ref)
            else:
                raw = (evidence_reads.object(self._objects, output_ref, MAX_OUTPUT_BYTES)
                       if evidence_reads is not None else self._objects.get_bounded(output_ref, MAX_OUTPUT_BYTES))
                try:
                    document = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise OutputInvalid("accepted output is not valid JSON") from exc
                output = validate_output_document(document)
            domains = set()
            for company in output["companies"]:
                domains.add(
                    normalize_url(str(company["company_website"]))
                    .domain.registrable_domain
                )
            # The same company in different ICPs is a different returned slot.
            counts[position] = len(domains)
        return counts

    def _per_icp_cost_eligibility(
        self,
        round_row: Mapping[str, Any],
        submission_id: str,
        position: int,
        runs: Sequence[Mapping[str, Any]],
        qualified: Optional[int] = None,
    ) -> Dict[str, Any]:
        if qualified is None:
            qualified = self._qualified_company_count(
                round_row, submission_id, runs, positions=(position,)
            )
        result = self._store.icp_cost_eligibility(
            round_id=str(round_row["round_id"]),
            submission_id=submission_id,
            icp_position=position,
            qualified_company_count=qualified,
        )
        result["qualified_company_count"] = qualified
        return result

    def _submission_per_icp_cost_eligibility(
        self,
        round_row: Mapping[str, Any],
        submission_id: str,
        runs: Sequence[Mapping[str, Any]],
        positions: Sequence[int],
        *, publication_reads: Optional[_PublicationReads] = None,
        evidence_reads: Optional[_PublicResultEvidenceReads] = None,
    ) -> Dict[str, Any]:
        wanted = sorted(set(int(value) for value in positions))
        qualified = (
            publication_reads.qualified_counts(submission_id, wanted)
            if publication_reads is not None else self._qualified_company_counts(
                round_row, submission_id, runs, positions=wanted,
                **({"evidence_reads": evidence_reads} if evidence_reads is not None else {}),
            )
        )
        returned = self._returned_company_counts(
            submission_id, runs, positions=wanted,
            publication_reads=publication_reads,
            **({"evidence_reads": evidence_reads} if evidence_reads is not None else {}),
        )
        rows = []
        for position in wanted:
            row = (
                dict(publication_reads.per_icp_cost(submission_id, position))
                if publication_reads is not None else self._per_icp_cost_eligibility(
                    round_row, submission_id, position, runs,
                    qualified=qualified.get(position, 0),
                )
            )
            row["returned_company_count"] = returned.get(position, 0)
            rows.append(row)
        hard_reason = next(
            (
                str(row["eligibility_reason"])
                for row in rows
                if row["eligibility_reason"]
                in ("provider_calls_inflight", "provider_cost_uncertain")
            ),
            None,
        )
        costs = self._store.submission_costs(submission_id)
        execution = self._cost_kind_summary(costs, "execute", include_successful=True)
        judge = self._cost_kind_summary(costs, "score", include_successful=True)
        if judge["inflight_calls"]:
            hard_reason = "provider_calls_inflight"
        summary = {
            "sourcing_cost_eligibility_policy": (
                round_row["configuration_doc"]["sourcing_cost_eligibility_policy"]
            ),
            "execution_icp_cap_microusd": int(
                round_row["configuration_doc"]["execution_icp_cap_microusd"]
            ),
            "cost_per_company_cap_microusd": int(
                round_row["configuration_doc"]["cost_per_company_microusd"]
            ),
            "returned_company_count": sum(
                int(row["returned_company_count"]) for row in rows
            ),
            "qualified_company_count": sum(
                int(row["qualified_company_count"]) for row in rows
            ),
            "eligible_icp_count": sum(bool(row["eligible"]) for row in rows),
            "competition_sourcing_microusd": sum(
                int(row["competition_sourcing_microusd"]) for row in rows
            ),
            "per_icp": rows,
            "execution": execution,
            "judge": judge,
        }
        return {
            "cost_summary": summary,
            "eligible": hard_reason is None,
            "eligibility_reason": hard_reason or "eligible",
        }

    def _qualified_company_count(
        self, round_row: Mapping[str, Any], submission_id: str,
        runs: Sequence[Mapping[str, Any]], *, positions: Optional[Sequence[int]] = None,
    ) -> int:
        """Count unique qualified entities from accepted, output-bound judgments."""
        return sum(self._qualified_company_counts(
            round_row, submission_id, runs, positions=positions
        ).values())

    def _qualified_company_counts(
        self, round_row: Mapping[str, Any], submission_id: str,
        runs: Sequence[Mapping[str, Any]], *, positions: Optional[Sequence[int]] = None,
        publication_reads: Optional[_PublicationReads] = None,
        evidence_reads: Optional[_PublicResultEvidenceReads] = None,
    ) -> Dict[int, int]:
        """Count qualified entities per ICP in one judgment/object pass."""
        # Keep immutable proof reuse within this participant qualification call,
        # rather than retaining company evidence across the whole publication.
        if publication_reads is not None and evidence_reads is None:
            evidence_reads = _PublicResultEvidenceReads()
        policy = round_row["configuration_doc"]["scorer_policy"]
        round_id = str(round_row["round_id"])
        icps = (
            publication_reads.evaluation_icps() if publication_reads is not None
            else self.evaluation_icps(round_id)
        )
        wanted = set(range(len(icps)) if positions is None else positions)
        selected = self._selected_accepted_execution_runs(runs, submission_id)
        judges = {}
        judged_stages = {
            int(run.get("stage") or 0)
            for position, run in selected.items()
            if position in wanted
        }
        for stage in judged_stages:
            judges.update(
                publication_reads.scoring_outputs(stage)
                if publication_reads is not None else self._scoring_outputs(round_id, stage, round_row=round_row)
            )
        counts: Dict[int, int] = {}
        for position in sorted(wanted):
            execution = selected.get(position)
            if execution is None:
                continue
            judge = judges.get(str(execution["run_id"]))
            if (
                judge is not None and judge.get("status") == "failed"
                and judge.get("terminal_cause") in contracts.TERMINAL_CAUSES
                and judge.get("terminal_cause") != "accepted"
                and execution.get("per_icp_score") == 0
                and execution.get("qualification_doc") == {"companies": []}
            ):
                # Only the persisted zero from the closed scoring stage can
                # establish zero qualified companies without an accepted judge.
                counts[position] = 0
                continue
            if judge is None or judge.get("status") != "accepted":
                raise scoring.ScoringError("qualified count requires accepted judgment")
            if publication_reads is not None:
                output = publication_reads.output(str(execution["output_ref"]))
            else:
                raw = (evidence_reads.object(self._objects, execution["output_ref"], MAX_OUTPUT_BYTES)
                       if evidence_reads is not None else self._objects.get_bounded(execution["output_ref"], MAX_OUTPUT_BYTES))
                document = json.loads(raw.decode("utf-8"))
                output = validate_output_document(document)
            companies = output["companies"]
            rows = self._verified_breakdowns(
                judge, icp=icps[position], companies=companies, policy=policy,
                **({"evidence_reads": evidence_reads, "round_context": round_row}
                   if evidence_reads is not None else {}),
            )
            scored_indexes, _ = verify.bucket_skip(icps[position], verify.slice_first_n(companies, verify.icp_company_goal(icps[position])))
            identities = set()
            for original_index, breakdown in zip(scored_indexes, rows):
                if breakdown.get("company_index") != original_index:
                    raise scoring.ScoringError("integrity breakdown company index mismatch")
                if not isinstance(breakdown.get("company_qualified"), bool) or not isinstance(breakdown.get("duplicate_company"), bool):
                    raise scoring.ScoringError("integrity qualification receipt missing")
                key = breakdown.get("company_identity_key")
                if not isinstance(key, str) or not key:
                    raise scoring.ScoringError("integrity company identity missing")
                if breakdown["company_qualified"] and not breakdown["duplicate_company"]:
                    if float(breakdown.get("final_score") or 0) <= 0:
                        raise scoring.ScoringError("qualified company cannot carry zero score")
                    identities.add(key)
            counts[position] = len(identities)
        return counts

    @staticmethod
    def _cost_kind_summary(
        costs: Mapping[str, Any], kind: str, *, include_successful: bool = False
    ) -> Dict[str, Any]:
        providers = []
        totals = {
            "settled_microusd": 0,
            "reserved_or_uncertain_microusd": 0,
            "conservative_microusd": 0,
            "inflight_calls": 0,
            "uncertain_calls": 0,
            "refused_calls": 0,
            "call_count": 0,
        }
        counter_keys = (
            "settled_microusd",
            "reserved_or_uncertain_microusd",
            "inflight_calls",
            "uncertain_calls",
            "refused_calls",
            "call_count",
        )
        policy_counter_keys = (
            "successful_microusd",
            "successful_calls",
            "success_unresolved_microusd",
            "success_unresolved_calls",
        )
        if include_successful:
            counter_keys += policy_counter_keys
            totals.update({key: 0 for key in policy_counter_keys})
        for raw in costs["providers"]:
            if raw["kind"] != kind:
                continue
            row = {"provider": raw["provider"]}
            for key in counter_keys:
                if key not in raw:
                    raise ServiceError("successful_call_cost_schema_unavailable", 500)
                value = int(raw[key])
                row[key] = value
                totals[key] += value
            row["conservative_microusd"] = (
                row["settled_microusd"]
                + row["reserved_or_uncertain_microusd"]
            )
            providers.append(row)
        providers.sort(key=lambda row: row["provider"])
        totals["conservative_microusd"] = (
            totals["settled_microusd"]
            + totals["reserved_or_uncertain_microusd"]
        )
        return {**totals, "providers": providers}

    def _submission_cost_eligibility(
        self,
        round_row: Mapping[str, Any],
        submission_id: str,
        runs: Sequence[Mapping[str, Any]],
        *, positions: Optional[Sequence[int]] = None,
        publication_reads: Optional[_PublicationReads] = None,
        evidence_reads: Optional[_PublicResultEvidenceReads] = None,
    ) -> Dict[str, Any]:
        """Build final cost reporting without changing the quality score."""

        configuration = round_row.get("configuration_doc") or {}
        if (
            configuration.get("sourcing_cost_eligibility_policy")
            in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
        ):
            return self._submission_per_icp_cost_eligibility(
                round_row,
                submission_id,
                runs,
                tuple(
                    range(contracts.benchmark_icp_count(round_row.get("configuration_doc")))
                    if positions is None else positions
                ),
                publication_reads=publication_reads,
                **({"evidence_reads": evidence_reads} if evidence_reads is not None else {}),
            )
        if "cost_per_company_microusd" not in configuration:
            return {
                "cost_summary": None,
                "eligible": True,
                "eligibility_reason": "historical_round",
            }
        try:
            returned = self._returned_company_count(submission_id, runs)
            qualified = self._qualified_company_count(round_row, submission_id, runs, positions=positions) if integrity.enabled(configuration) else returned
        except (ArenaContractError, OutputInvalid, TypeError, ValueError, scoring.ScoringError):
            return {
                "cost_summary": None,
                "eligible": False,
                "eligibility_reason": "stored_output_invalid",
            }
        successful_call_policy = (
            configuration.get("sourcing_cost_eligibility_policy")
            == contracts.SUCCESSFUL_CALLS_COST_POLICY
        )
        costs = self._store.submission_costs(submission_id)
        execution = self._cost_kind_summary(
            costs, "execute", include_successful=successful_call_policy
        )
        judge = self._cost_kind_summary(
            costs, "score", include_successful=successful_call_policy
        )
        required_policy_counters = (
            "successful_microusd",
            "successful_calls",
            "success_unresolved_microusd",
            "success_unresolved_calls",
        )
        if successful_call_policy and any(
            key not in execution for key in required_policy_counters
        ):
            raise ServiceError("successful_call_cost_schema_unavailable", 500)
        execution_cap = int(configuration["execution_cap_microusd"])
        per_company_cap = int(configuration["cost_per_company_microusd"])
        eligibility_cap = min(execution_cap, per_company_cap * qualified)
        summary = {
            "returned_company_count": returned,
            "execution_cap_microusd": execution_cap,
            "cost_per_company_cap_microusd": per_company_cap,
            "eligibility_cap_microusd": eligibility_cap,
            "execution": execution,
            # Judge spend is reported, but it does not enter sourcing eligibility.
            "judge": judge,
        }
        if successful_call_policy:
            summary.update(
                {
                    "sourcing_cost_eligibility_policy": (
                        contracts.SUCCESSFUL_CALLS_COST_POLICY
                    ),
                    "competition_sourcing_microusd": (
                        execution["successful_microusd"]
                        + execution["success_unresolved_microusd"]
                    ),
                }
            )
        if integrity.enabled(configuration):
            summary["qualified_company_count"] = qualified
            selected = self._selected_accepted_execution_runs(runs, submission_id)
            if not any(position in selected for position in (positions if positions is not None else range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))))):
                return {"cost_summary": summary, "eligible": False, "eligibility_reason": "stored_output_invalid"}
        if execution["inflight_calls"] or judge["inflight_calls"]:
            return {
                "cost_summary": summary,
                "eligible": False,
                "eligibility_reason": "provider_calls_inflight",
            }
        if successful_call_policy and execution["success_unresolved_calls"]:
            return {
                "cost_summary": summary,
                "eligible": False,
                "eligibility_reason": "provider_cost_uncertain",
            }
        if (
            not successful_call_policy
            and (execution["uncertain_calls"] or judge["uncertain_calls"])
        ):
            return {
                "cost_summary": summary,
                "eligible": False,
                "eligibility_reason": "provider_cost_uncertain",
            }
        efficiency_cost = (
            execution["successful_microusd"]
            + execution["success_unresolved_microusd"]
            if successful_call_policy
            else execution["conservative_microusd"]
        )
        if efficiency_cost > execution_cap:
            reason = "execution_cap_exceeded"
        elif efficiency_cost > per_company_cap * qualified:
            reason = "cost_per_company_exceeded"
        else:
            reason = "eligible"
        return {
            "cost_summary": summary,
            "eligible": reason == "eligible",
            "eligibility_reason": reason,
        }

    def _judgment_matches_execution(
        self, row: Mapping[str, Any], judge: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> bool:
        """Check the judgment authority used by this round's frozen scorer."""

        round_id = str(row["round_id"])
        configuration = row.get("configuration_doc") or {}
        if (int(judge.get("scoring_generation_floor") or 0) != self._score_generation_floor(row)
            or judge.get("status") != "accepted"
            or judge.get("terminal_cause") != "accepted"
            or not judge.get("output_ref")
            or not self._same_scoring_membership(judge, source, round_id=round_id)
            or judge.get("miner_hotkey") != source.get("miner_hotkey")):
            return False
        if company_judgments.enabled(configuration):
            refs = judge.get("company_judgment_refs")
            if not isinstance(refs, list):
                return False
            try:
                scopes = [company_judgments.validate_company_ref(ref)["scope_doc"]
                          for ref in refs]
                if not refs:
                    raw = self._objects.get_bounded(
                        str(source.get("output_ref") or ""), MAX_OUTPUT_BYTES)
                    output = validate_output_document(json.loads(raw),
                        expected_schema_version=contact_policy.output_schema(configuration))
                    if output["companies"]:
                        return False
            except (company_judgments.CompanyJudgmentError, OutputInvalid,
                    TypeError, ValueError, UnicodeDecodeError):
                return False
        elif integrity.enabled(configuration):
            scope = judge.get("judgment_scope_doc")
            if not isinstance(scope, Mapping):
                return False
            scopes = [scope]
        else:
            # Legacy scoring assignments have no judgment-cache scope. Their
            # accepted output and execution membership are the final authority.
            return True
        return all(scope.get("round_id") == round_id and all(
            scope.get(key) == configuration.get(key)
            for key in ("scorer_image_digest", "scorer_image_reference")
        ) for scope in scopes)

    def _terminal_miner_credit_judgment(self, source_id: str) -> bool:
        """Only the database's bound credit proof can end an unmeasured judge."""
        return self._store.miner_credit_judgment_terminal(source_id)

    def _unfinished_judgments(self, row: Mapping[str, Any]) -> bool:
        """Unmeasured work waits unless its miner funding is proved terminal."""

        round_id = str(row["round_id"])
        admitted = {p["submission_id"] for p in row.get("participants") or []}
        sources = [run for run in self._store.list_runs(round_id, kind="execute")
                   if run.get("stage") in (1, 2) and run.get("status") == "accepted"
                   and run.get("submission_id") in admitted]
        score_runs = self._store.list_runs(round_id, kind="score")
        judges = self._select_scoring_outputs(score_runs, round_row=row)
        for source in sources:
            judge = judges.get(str(source["run_id"]))
            if judge is not None and judge.get("status") == "accepted" and not self._judgment_matches_execution(row, judge, source):
                raise ServiceError("accepted_judgment_binding_invalid", 500)
            if source.get("per_icp_score") is None or judge is None or judge.get("status") != "accepted":
                if (_judge_continuation_enabled(row)
                    and self._terminal_miner_credit_judgment(str(source["run_id"]))):
                    continue
                return True
        return False

    def _record_recovered_judgment_scores(self, row: Mapping[str, Any]) -> None:
        """Record verified decisions that arrived after a stage was scored."""

        round_id = str(row["round_id"])
        admitted = {p["submission_id"] for p in row.get("participants") or []}
        configuration = row["configuration_doc"]
        policy = contracts.validate_scorer_policy(configuration["scorer_policy"])
        icps = self.evaluation_icps(round_id)
        for stage in (1, 2):
            sources = self._store.list_runs(round_id, stage=stage, kind="execute")
            score_runs = self._store.list_runs(round_id, stage=stage, kind="score")
            judges = self._select_scoring_outputs(score_runs, round_row=row)
            ready = [source for source in sources
                if source.get("submission_id") in admitted
                and source.get("status") == "accepted"
                and source.get("per_icp_score") is None
                and judges.get(str(source["run_id"]), {}).get("status") == "accepted"]
            if not ready:
                continue
            outputs = self._outputs_by_run(round_id, stage)
            rows = []
            for source in ready:
                judge = judges[str(source["run_id"])]
                if not self._judgment_matches_execution(row, judge, source):
                    raise ServiceError("recovered_judgment_binding_invalid", 500)
                position = int(source["icp_position"])
                companies = outputs[str(source["run_id"])]
                breakdowns = self._verified_breakdowns(
                    judge, icp=icps[position], companies=companies, policy=policy)
                rows.append(verify.scored_row(str(source["submission_id"]),
                    position, str(source["run_id"]), icps[position], companies,
                    breakdowns, policy))
            records = scoring.run_scores_for_store({
                "rows": rows,
                **({"integrity_policy": "arena_integrity_v1"}
                    if configuration.get("integrity_policy") == "arena_integrity_v1" else {}),
            }, ready)
            result = self._record_current_run_scores(row, stage, records)
            if result.get("status") != "ok":
                raise ServiceError("recovered_judgment_scores_not_recorded", 500)

    # -- publication and downstream reward activation -------------------------

    @staticmethod
    def _selected_execution_runs(runs, submission_id):
        selected = {}
        for run in runs:
            if str(run.get("submission_id")) != submission_id or run.get("kind") != "execute":
                continue
            position = int(run["icp_position"])
            current = selected.get(position)
            rank = lambda item: (item.get("status") == "accepted", int(item.get("attempt") or 0))
            if current is None or rank(run) > rank(current):
                selected[position] = run
        return selected

    def _execution_infrastructure_failures(self, reads: _PublicationReads) -> Dict[str, str]:
        """Recognize saved empty outputs whose last model call could not run.

        Only publication uses this derived check. Original accepted runs, native
        receipts, costs, and judgments stay unchanged; other work can continue.
        """
        failures: Dict[str, str] = {}
        for participant in reads.round_row.get("participants") or []:
            submission_id = str(participant["submission_id"])
            selected = self._selected_execution_runs(reads.runs, submission_id)
            for run in selected.values():
                if run.get("status") == "accepted":
                    if reads.output(str(run["output_ref"]))["companies"] and run.get("per_icp_score") is not None:
                        continue
                elif run.get("status") != "failed" or run.get("terminal_cause") not in (
                    "provider_error", "stage_closed"):
                    continue
                try:
                    unavailable = self._native_execution_route_unavailable(run)
                except (ArenaStoreError, broker_module.BrokerError, KeyError, TypeError,
                        ValueError, UnicodeDecodeError):
                    if run.get("status") == "accepted":
                        raise
                    unavailable = False
                if unavailable:
                    failures[str(run["run_id"])] = "openrouter_zdr_route_unavailable"
        return failures

    def _native_execution_route_unavailable(self, run) -> bool:
        issued = self._store.latest_execution_model_reservation(str(run["run_id"]))
        if issued is None:
            return False
        document = issued.get("entry_doc") or {}
        sequence = document.get("action_sequence")
        operation = issued.get("operation_id")
        if (type(sequence) is not int or sequence < 0
            or operation not in ("openrouter.chat", "openrouter.responses")
            or not isinstance(document.get("request_hash"), str)
            or not issued.get("call_identity")):
            raise ArenaStoreError("execution model reservation is invalid")
        ledger = self._store.list_ledger(
            run_id=str(run["run_id"]), call_identity=issued["call_identity"], limit=20,
        )
        reservations = [row for row in ledger if row["entry_kind"] == "reservation"]
        dispatches = [row for row in ledger if row["entry_kind"] == "dispatch"]
        settlements = [row for row in ledger if row["entry_kind"] == "settlement"]
        if len(ledger) >= 20 or any(len(rows) != 1 for rows in (reservations, dispatches, settlements)):
            raise ArenaStoreError("execution model call lifecycle is incomplete")
        reservation, dispatch, terminal = reservations[0], dispatches[0], settlements[0]
        expected = {"run_id": run["run_id"], "call_identity": issued["call_identity"],
                    "operation_id": operation, "provider": "openrouter",
                    "funding_source": issued.get("funding_source")}
        if (any(row.get(key) != value for row in ledger for key, value in expected.items())
            or not reservation["entry_id"] < dispatch["entry_id"] < terminal["entry_id"]
            or max(row["entry_id"] for row in ledger) != terminal["entry_id"]
            or reservation["entry_id"] != issued["entry_id"]
            or reservation.get("entry_doc") != document):
            raise ArenaStoreError("execution model call binding is invalid")
        native = terminal.get("terminal_response")
        if not isinstance(native, Mapping):
            raise ArenaStoreError("execution model call receipt is missing")
        status, _headers, body = broker_module._decode_terminal(native)
        if broker_module._openrouter_data_policy_route_unavailable(status, body):
            if native.get("call_succeeded") is not False:
                raise ArenaStoreError("execution model call receipt contradicts its outcome")
            return True
        return False

    def _empty_execution_infrastructure_failure(self, reads: _PublicationReads) -> bool:
        return bool(self._execution_infrastructure_failures(reads))

    @staticmethod
    def _full_thinking_required(round_row: Mapping[str, Any]) -> bool:
        return ((round_row.get("configuration_doc") or {}).get("reasoning_capture_policy")
                == reasoning_capture.POLICY)

    def _submission_training_eligibility(
        self, round_row: Mapping[str, Any], submission_id: str,
        runs: Sequence[Mapping[str, Any]],
        *, publication_reads: Optional[_PublicationReads] = None,
    ) -> Dict[str, Any]:
        """Read persisted evidence only. A capture failure cannot cancel a round."""
        try:
            submission = self._store.get_submission(submission_id)
            if not submission:
                return {"eligible": False, "reasons": ["submission_evidence_missing"]}
            baseline = self._is_daily_baseline(submission, round_row)
            if (baseline and baseline_reasoning.FIELD in (round_row.get("configuration_doc") or {})
                    and not reasoning_capture.enabled(baseline_reasoning.for_run(
                        round_row, dict(submission, kind="execute")))):
                return {"eligible": False, "reasons": ["baseline_reasoning_not_required"],
                        "exemption": "champion_baseline"}
            source = (training_eligibility.assess_source(submission, enabled=True)
                      if not baseline else {"eligible": True, "reasons": [],
                                            "exemption": "organizer_baseline"})
            selected = self._selected_accepted_execution_runs(runs, submission_id)
            positions = set(range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))))
            if set(selected) - positions:
                return {"eligible": False, "reasons": ["selected_execution_coverage_incomplete"],
                        "source_review": source}
            run_ids = [str(selected[position]["run_id"]) for position in sorted(selected)]
            unavailable_outputs = {}
            unavailable_artifacts = {}
            unavailable_exports = {}

            def export_example(run_id):
                output_ref = "arena/%s/outputs/%s.json" % (round_row["round_id"], run_id)
                fallback = {}
                if publication_reads is not None and output_ref in publication_reads.outputs_by_ref:
                    cached = publication_reads.raw_outputs_by_ref.get(output_ref)
                    if cached is not None:
                        fallback[output_ref] = cached
                read_view = training_reader.AvailabilityReadView(self._objects, fallback)
                # Failed reconstruction contributes no usable data. Keep the
                # failure local to this ICP and continue checking the others.
                try:
                    example = training_reader.export_run(self._store, read_view, run_id)
                except Exception as exc:
                    unavailable_exports[run_id] = type(exc).__name__[:64]
                    return None
                transient_artifacts = {kind: ref for kind, ref in read_view.artifact_refs.items()
                                       if read_view.transient_failures.get(ref) == "transient"}
                if transient_artifacts:
                    unavailable_artifacts[run_id] = transient_artifacts
                if read_view.unavailable in ({output_ref: "missing"},
                                             {output_ref: "transient"}):
                    unavailable_outputs[run_id] = output_ref
                return example

            class Exports(Mapping):
                """Load one exact example on demand; never retain its graph."""

                def __iter__(self):
                    return iter(run_ids)

                def __len__(self):
                    return len(run_ids)

                def __getitem__(self, run_id):
                    if run_id not in run_ids:
                        raise KeyError(run_id)
                    return export_example(run_id)

                def get(self, run_id, default=None):
                    # Mapping.get would swallow an exporter's KeyError and
                    # misclassify corrupt evidence as a missing-example gap.
                    return self[run_id] if run_id in run_ids else default

            capture = training_eligibility.assess_capture(
                submission, Exports(), run_ids, enabled=True,
                expected_run_count=len(positions),
                minimum_coverage_percent=training_coverage.percent_from_configuration(
                    round_row.get("configuration_doc") or {}),
                maximum_missing_reasoning_percent=(
                    training_coverage.missing_reasoning_percent_from_configuration(
                        round_row.get("configuration_doc") or {})),
                unavailable_exports=unavailable_exports,
                unavailable_outputs=unavailable_outputs,
                unavailable_artifacts=unavailable_artifacts,
            )
            if positions - set(selected):
                capture["counts"]["missing_icp_positions"] = sorted(positions - set(selected))
            return {"eligible": source["eligible"] and capture["eligible"],
                    "reasons": sorted(set(source["reasons"] + capture["reasons"])),
                    "source_review": source, "capture": capture}
        except Exception as exc:
            # Keep diagnostics useful without exposing stored data, object refs,
            # provider exception text or tracebacks. Identity hashes are bounded.
            round_id = round_row.get("round_id")
            logger.warning(
                "arena_training_evidence_check_failed round_id_hash=%s submission_id_hash=%s error_class=%s",
                contracts.hash_bytes(round_id.encode("utf-8", "replace")) if isinstance(round_id, str) else "unavailable",
                contracts.hash_bytes(submission_id.encode("utf-8", "replace")) if isinstance(submission_id, str) else "unavailable",
                type(exc).__name__[:64],
            )
            # The data check affects eligibility only, not publication availability.
            return {"eligible": False, "reasons": ["training_evidence_check_failed"]}

    def publish(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "scored":
            return {"status": "stale", "round_status": round_row["status"]}
        evaluation_date = str(round_row.get("evaluation_date") or "")
        if _judge_continuation_enabled(round_row):
            self._record_recovered_judgment_scores(round_row)
        unfinished_judgments = bool(evaluation_date and self._unfinished_judgments(round_row))
        configuration = round_row.get("configuration_doc") or {}
        # An infrastructure zero is not a measured baseline. Keep the round
        # recoverable until its frozen final deadline, without comparing miners
        # or creating a new reward authority against unfinished baseline work.
        if not self._baseline_fully_succeeded(round_row):
            return self._hold_incomplete_baseline(round_row)
        baseline_first = (
            configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        )
        publication_reads = _PublicationReads(
            self, round_row, self._store.list_runs(round_id, kind="execute")
        )
        try:
            infrastructure_failures = self._execution_infrastructure_failures(publication_reads)
        except (ArenaStoreError, broker_module.BrokerError, OutputInvalid, KeyError,
                TypeError, ValueError, UnicodeDecodeError):
            return self._hold_incomplete_evaluation(
                round_row, reason="execution_evidence_unavailable",
            )
        baseline_ids = {str(p["submission_id"]) for p in round_row.get("participants") or [] if p.get("is_king")}
        if any(str(run.get("submission_id")) in baseline_ids and str(run.get("run_id")) in infrastructure_failures
               for run in publication_reads.runs):
            return self._hold_incomplete_baseline(round_row)
        incomplete = self._participant_incomplete_reasons(round_row, publication_reads, infrastructure_failures)
        if unfinished_judgments or incomplete:
            if (self.now() < _parse_iso(configuration["schedule"]["publication_deadline"])
                or self._evaluation_draining(round_row)):
                return {"status": "retry", "round_status": round_row["status"],
                    "reason": "judge_incomplete" if unfinished_judgments else "execution_infrastructure_incomplete"}
            if not incomplete:
                # An accepted mismatched judgment is an integrity failure, not
                # an unfinished participant that publication can omit.
                return self._hold_incomplete_evaluation(round_row, reason="scoring_evidence_invalid")
        stage1_ranking = verify.stage1_ranking(
            self._score_entries_from_runs(
                round_row,
                (
                    range(contracts.benchmark_icp_count(round_row.get("configuration_doc")))
                    if baseline_first
                    else contracts.stage_positions(1, configuration)
                ),
                "stage1_score",
                publication_reads=publication_reads,
            )
        )
        finalists = list(round_row.get("finalists") or [])
        final_entries = self._score_entries_from_runs(
            round_row, range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))), "final_score",
            publication_reads=publication_reads,
        )
        for entry in final_entries:
            if str(entry["submission_id"]) in incomplete:
                entry["final_score"] = None
                entry["execution_incomplete"] = True
        eligibility = {
            str(entry["submission_id"]): ({
                "cost_summary": None,
                "eligible": False,
                "eligibility_reason": "execution_incomplete",
                **({"evaluation": "not_evaluated",
                    "evaluation_status": "provider_blocked" if incomplete[str(entry["submission_id"])] == "openrouter_zdr_route_unavailable" else "evaluation_incomplete",
                    "evaluation_reason": incomplete[str(entry["submission_id"])]}
                    if str(entry["submission_id"]) in incomplete else {}),
            } if entry.get("execution_incomplete") else self._submission_cost_eligibility(
                round_row,
                str(entry["submission_id"]),
                publication_reads.runs,
                positions=range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))),
                publication_reads=publication_reads,
            ))
            for entry in final_entries
        }
        effective_final_entries = [dict(entry) for entry in final_entries]
        king_entry = next(entry for entry in effective_final_entries if entry["is_king"])
        if eligibility[str(king_entry["submission_id"])]["eligibility_reason"] not in (
            "eligible", "historical_round", "cost_per_company_exceeded",
            "execution_cap_exceeded",
        ):
            return self._hold_incomplete_baseline(round_row)
        if (
            king_entry["final_score"] is not None
            and integrity.enabled(round_row.get("configuration_doc") or {})
            and eligibility[str(king_entry["submission_id"])]["eligibility_reason"]
            in ("cost_per_company_exceeded", "execution_cap_exceeded")
        ):
            # Retain the accepted per-ICP scores in the run ledger, but apply
            # the same final cost consequence to the baseline that challengers
            # face. Other fail-closed cost states do not invent a zero score.
            king_entry["final_score"] = 0.0
        if self._full_thinking_required(round_row):
            # Keep full source and capture proof for every cost-eligible
            # challenger that can cross the frozen baseline margin. Other
            # training evidence stays explicitly unassessed.
            qualifier_audit = (
                integrity.enabled(configuration)
                and configuration.get("sourcing_cost_eligibility_policy")
                in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
            )
            for entry in effective_final_entries:
                submission_id = str(entry["submission_id"])
                if qualifier_audit and (
                    entry["is_king"] or not eligibility[submission_id]["eligible"]
                    or verify.king_decision([entry], king_entry, configuration)["outcome"] != "crowned"
                ):
                    eligibility[submission_id]["training_data_eligibility"] = {
                        "status": "not_assessed", "eligible": None, "reasons": []}
                    continue
                # Refresh before actual assessments, not skipped rows. The
                # final compare-and-swap still protects every publication.
                current = self._round(round_id)
                if current["status"] != "scored":
                    return {"status": "stale", "round_status": current["status"]}
                training = self._submission_training_eligibility(
                    round_row, submission_id, publication_reads.runs,
                    publication_reads=publication_reads,
                )
                eligibility[submission_id]["training_data_eligibility"] = training
                if not training["eligible"]:
                    if eligibility[submission_id]["eligible"]:
                        eligibility[submission_id]["eligibility_reason"] = "training_data_incomplete"
                    eligibility[submission_id]["eligible"] = False
            # The organizer baseline remains a measured comparison, never a
            # contender. Its missing training data must not halt publication or
            # prevent a verified challenger from competing against its score.
        decision = verify.king_decision(
            [
                entry
                for entry in effective_final_entries
                if not entry["is_king"]
                and eligibility[str(entry["submission_id"])]["eligible"]
            ],
            king_entry,
            configuration=configuration,
        )
        if self._full_thinking_required(round_row) and qualifier_audit and decision["outcome"] == "crowned":
            # A crowned round still needs the organizer's original training
            # result: reward activation reads its published eligibility.
            current = self._round(round_id)
            if current["status"] != "scored":
                return {"status": "stale", "round_status": current["status"]}
            baseline_id = str(king_entry["submission_id"])
            training = self._submission_training_eligibility(
                round_row, baseline_id, publication_reads.runs,
                publication_reads=publication_reads,
            )
            eligibility[baseline_id]["training_data_eligibility"] = training
            if not training["eligible"]:
                if eligibility[baseline_id]["eligible"]:
                    eligibility[baseline_id]["eligibility_reason"] = "training_data_incomplete"
                eligibility[baseline_id]["eligible"] = False
        published_at = _iso(self.now())
        final_ranking = verify.final_ranking(effective_final_entries)
        for row in final_ranking:
            row.update(eligibility[str(row["submission_id"])])
        publication = {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "is_baseline": bool(p.get("is_king"))} for p in round_row.get("participants") or []],
            "stage1_ranking": stage1_ranking,
            "finalists": finalists,
            "final_ranking": final_ranking,
            "king_decision": decision,
            "published_at": published_at,
        }
        contracts.check_strict_document(publication, contracts.PUBLICATION_LIMITS)
        transition = self._store.transition_round(round_id, "scored", "published", {
            "publication_doc": publication,
            "published_at": published_at,
        })
        return {
            "status": transition.get("status"),
            "king_outcome": decision["outcome"],
            "king_hotkey": str(decision.get("king_hotkey") or ""),
        }

    def promote_pending_baselines(self) -> Dict[str, Any]:
        """Finish accepted winners oldest-first before a new baseline or reward.

        Git and PostgreSQL cannot share a transaction. Persist the prepared Git
        update first, push both branches atomically, then mark completion. A
        lost response is reconciled against those same branch heads on retry.
        """

        if self._config.mode != "live":
            return {"status": "disabled", "promoted": 0}
        promoted = 0
        network_name, netuid = self._chain_scope()
        for row in self._pending_promotion_rows(
            pinned_round_id=self._config.pinned_round_id,
            network_name=network_name, netuid=netuid,
        ):
            result = self.promote_baseline(str(row["round_id"]))
            if result.get("status") in ("superseded", "waiting_for_baseline"):
                continue
            if result.get("status") not in ("promoted", "existing"):
                return {"status": result.get("status", "pending"), "promoted": promoted}
            promoted += int(result.get("status") == "promoted")
        return {"status": "ok", "promoted": promoted}

    @staticmethod
    def _evaluation_day(row: Mapping[str, Any]) -> date:
        value = row.get("evaluation_date")
        if not isinstance(value, str) or len(value) != 10:
            raise ServiceError("round_evaluation_date_invalid", 500)
        try:
            day = date.fromisoformat(value)
        except ValueError as exc:
            raise ServiceError("round_evaluation_date_invalid", 500) from exc
        if day.isoformat() != value:
            raise ServiceError("round_evaluation_date_invalid", 500)
        return day

    def _latest_published_day(self) -> Optional[date]:
        """Find the newest completed live day in this chain scope."""

        network_name, netuid = self._chain_scope()
        latest = self._store.latest_published_day(
            network_name=network_name, netuid=netuid
        )
        return self._evaluation_day(latest) if latest is not None else None

    def _superseded_by_published_day(self, row: Mapping[str, Any]) -> bool:
        latest = self._latest_published_day()
        return latest is not None and latest > self._evaluation_day(row)

    def _pending_promotion_rows(
        self, *, pinned_round_id: Optional[str] = None,
        network_name: str, netuid: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = self._store.pending_promotions(
                pinned_round_id=pinned_round_id,
                network_name=network_name, netuid=netuid,
                limit=100, offset=offset,
            )
            rows.extend(page)
            if len(page) < 100:
                return rows
            offset += 100

    def _pending_promotion_blocks(self) -> bool:
        network_name, netuid = self._chain_scope()
        latest = self._latest_published_day()
        for item in self._pending_promotion_rows(
            network_name=network_name, netuid=netuid
        ):
            row = self._store.get_round(str(item["round_id"]))
            if row is None:
                raise ServiceError("promotion_round_missing", 500)
            if not self._baseline_final_ready(row):
                continue
            if latest is None or latest <= self._evaluation_day(row):
                return True
        return False

    def promote_baseline(self, round_id: str) -> Dict[str, Any]:
        """Publish only the stored, scored winner, without executing its source."""

        row = self._round(round_id)
        if row.get("baseline_promoted_at"):
            return {"status": "existing"}
        configuration = row.get("configuration_doc") or {}
        decision = (row.get("publication_doc") or {}).get("king_decision") or {}
        if (
            row.get("status") != "published"
            or configuration.get("mode") != "live"
            or not row.get("promotion_required")
            or decision.get("outcome") != "crowned"
        ):
            return {"status": "not_required"}
        if self._superseded_by_published_day(row):
            return {"status": "superseded"}
        if not self._baseline_final_ready(row):
            return {"status": "waiting_for_baseline"}
        factory = self._config.baseline_promoter_factory
        if factory is None:
            raise ServiceError("baseline_promoter_unavailable", 503)
        submission_id = str(decision.get("winner_submission_id") or "")
        submission = self._store.get_submission(submission_id)
        if (
            not submission
            or submission.get("round_id") != round_id
            or submission.get("miner_hotkey") != decision.get("king_hotkey")
            or submission.get("is_king")
            or submission.get("status") != "frozen"
        ):
            raise ServiceError("promotion_winner_invalid", 500)
        # Public Git branches use the same completed-evaluation publication
        # gate as the dashboard. There is no additional upload-age delay.
        from lab_arena.source_disclosure import disclosure_status
        disclosure = disclosure_status(submission, self.now(), round_row=row)
        if not disclosure["available"]:
            return {"status": "source_private", "available_at": disclosure["available_at"]}
        payload = self._objects.get_bounded(
            str(submission["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        )
        if len(payload) != int(submission["source_size_bytes"]):
            raise ServiceError("promotion_source_size_mismatch", 500)
        promoter = factory()
        plan = row.get("promotion_doc")
        if plan is None:
            proposed = promoter.prepare(
                payload, round_id=round_id, submission_id=submission_id,
                timestamp=str(row["published_at"]),
            )
            prepared = self._store.prepare_promotion(round_id, proposed)
            if prepared.get("status") not in ("prepared", "existing"):
                return prepared
            # Another worker can win the initial prepare. Its stored plan is
            # authoritative; never silently replace it with a new Git base.
            row = self._store.get_round(round_id) or {}
            plan = row.get("promotion_doc")
        if not isinstance(plan, dict):
            raise ServiceError("promotion_plan_missing", 500)
        if self._superseded_by_published_day(row):
            return {"status": "superseded"}
        commit = promoter.publish(
            payload, plan=plan, round_id=round_id, submission_id=submission_id
        )
        if commit != plan["commit"]:
            raise ServiceError("promotion_commit_mismatch", 500)
        return self._store.complete_promotion(round_id, plan)

    def activate_pending_rewards(self) -> Dict[str, Any]:
        """Activate eligible live rounds oldest-first after publication.

        Publication never calls this method. A transient chain, cutover, KMS,
        or database failure leaves the compact competition result published
        and the next driver tick retries the same oldest round.
        """

        if self._config.mode != "live":
            return {"status": "disabled", "activated": 0}
        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            rows = [row] if row is not None and row["status"] == "published" else []
        else:
            network_name, netuid = self._chain_scope()
            rows = list(
                reversed(
                    self._store.list_rounds(
                        status="published", mode="live",
                        network_name=network_name,
                        netuid=netuid,
                        limit=200
                    )
                )
            )
        pending = [
            row for row in rows
            if (row.get("configuration_doc") or {}).get("mode") == "live"
            and (row.get("configuration_doc") or {}).get("rewards_enabled") is True
            and not row.get("reward_activated_at")
        ]
        activated = 0
        for row in pending:
            result = self.activate_reward(str(row["round_id"]))
            if result.get("status") in ("superseded", "waiting_for_baseline"):
                continue
            if result.get("status") not in ("activated", "existing"):
                return {"status": str(result.get("status") or "stale"), "activated": activated}
            activated += int(result.get("status") == "activated")
        return {"status": "ok", "activated": activated}

    def _usable_reward_bases(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return activated miner/no-winner bases, never organizer baselines."""

        usable: List[Dict[str, Any]] = []
        for row in rows:
            configuration = row.get("configuration_doc")
            if not isinstance(configuration, Mapping) or configuration.get("mode") != "live":
                continue
            document = row.get("reward_basis_doc")
            try:
                basis = rewards.validate_reward_basis(document)
            except (ArenaContractError, TypeError, ValueError) as exc:
                raise ServiceError("reward_history_invalid", 500) from exc
            baseline_hotkey = str(configuration.get("baseline_hotkey") or "")
            if basis["king_outcome"] != "no_king" and (
                not baseline_hotkey or basis["king_hotkey"] == baseline_hotkey
            ):
                continue
            usable.append(basis)
        return usable

    @staticmethod
    def _latest_miner_basis(
        bases: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        latest = max(
            (dict(basis) for basis in bases),
            key=lambda basis: int(basis["effective_reward_epoch"]),
            default=None,
        )
        # Nonpaying signed bases are explicit continuity barriers. Looking
        # past one would revive a champion that a later activation revoked.
        if (
            latest is None
            or latest.get("king_outcome") not in rewards.PAYING_KING_OUTCOMES
            or not str(latest.get("king_hotkey") or "")
        ):
            return None
        return latest

    def _baseline_fully_succeeded(self, row: Mapping[str, Any]) -> bool:
        """Whether all configured baseline executions and judgments succeeded."""

        baselines = [participant for participant in row.get("participants") or []
                     if participant.get("is_king")]
        if len(baselines) != 1 or not baselines[0].get("submission_id"):
            return False
        baseline_id = str(baselines[0]["submission_id"])
        baseline_hotkey = str(baselines[0].get("miner_hotkey") or "")
        round_id = str(row["round_id"])
        policy = (row.get("configuration_doc") or {}).get(
            "execution_sequence_policy"
        )
        if policy == contracts.BASELINE_SCORED_FIRST_POLICY:
            expected = {
                (1, position)
                for position in contracts.execution_positions(1, policy, row.get("configuration_doc"))
            }
        else:
            expected = {
                (stage, position)
                for stage in (1, 2)
                for position in contracts.execution_positions(stage, policy, row.get("configuration_doc"))
            }
        accepted_executions: Dict[Tuple[int, int], Mapping[str, Any]] = {}
        for run in self._store.list_runs(
            str(row["round_id"]), kind="execute"
        ):
            if (
                str(run.get("submission_id") or "") != baseline_id
                or run.get("round_id") != round_id
                or run.get("miner_hotkey") != baseline_hotkey
                or run.get("status") != "accepted"
                or run.get("terminal_cause") != "accepted"
                or run.get("per_icp_score") is None
                or not run.get("output_ref")
                or not isinstance(run.get("result_doc"), Mapping)
            ):
                continue
            key = (int(run.get("stage") or 0), int(run.get("icp_position") or 0))
            if key not in expected:
                continue
            if key in accepted_executions:
                return False
            accepted_executions[key] = run
        if set(accepted_executions) != expected:
            return False
        judgments = self._current_score_runs(row, self._store.list_runs(round_id, kind="score"))
        return all(any(
            judge.get("status") == "accepted"
            and judge.get("terminal_cause") == "accepted"
            and judge.get("round_id") == round_id
            and judge.get("submission_id") == baseline_id
            and judge.get("miner_hotkey") == baseline_hotkey
            and judge.get("stage") == execution.get("stage")
            and judge.get("icp_position") == execution.get("icp_position")
            and judge.get("scored_run_id") == execution.get("run_id")
            and bool(judge.get("output_ref"))
            and isinstance(judge.get("result_doc"), Mapping)
            for judge in judgments
        ) for execution in accepted_executions.values())

    def _baseline_final_ready(self, row: Mapping[str, Any]) -> bool:
        if not self._baseline_fully_succeeded(row):
            return False
        baseline_id = next(str(p["submission_id"]) for p in row["participants"] if p.get("is_king"))
        eligibility = self._submission_cost_eligibility(
            row, baseline_id, self._store.list_runs(str(row["round_id"]), kind="execute"),
        )
        return eligibility["eligibility_reason"] in (
            "eligible", "historical_round", "cost_per_company_exceeded",
            "execution_cap_exceeded",
        )

    def _participant_incomplete_reasons(self, row, reads, blocked):
        """Describe gaps without changing accepted scores or their provenance."""
        count = contracts.benchmark_icp_count(row.get("configuration_doc"))
        scores = self._select_scoring_outputs(
            self._store.list_runs(str(row["round_id"]), kind="score"), round_row=row)
        reasons = {}
        confirmed_model_positions = {(str(run.get("submission_id")), int(run["icp_position"]))
            for run in reads.runs if run.get("terminal_cause") in contracts.MODEL_CAUSED_TERMINAL_CAUSES}
        def remember(submission_id, reason):
            # A missing score is the fallback; concrete execution/judge evidence
            # takes precedence regardless of ICP iteration order.
            if submission_id not in reasons or reasons[submission_id] == "execution_score_missing":
                reasons[submission_id] = reason

        for participant in row.get("participants") or []:
            submission_id = str(participant["submission_id"])
            if participant.get("is_king"):
                continue
            selected = self._selected_execution_runs(reads.runs, submission_id)
            for position in range(count):
                run = selected.get(position)
                if run is None:
                    remember(submission_id, "execution_missing")
                    continue
                if run.get("status") != "accepted" and scores.get(str(run["run_id"]), {}).get("status") == "accepted":
                    raise ServiceError("accepted_judgment_source_invalid", 500)
                prior_model_failure = (submission_id, position) in confirmed_model_positions
                if str(run["run_id"]) in blocked and (run.get("status") == "accepted" or not prior_model_failure or run.get("per_icp_score") is None):
                    reasons[submission_id] = blocked[str(run["run_id"])]
                    continue
                if (run.get("status") == "failed" and (not prior_model_failure or run.get("per_icp_score") is None)
                    and run.get("terminal_cause") in (contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed"})):
                    remember(submission_id, str(run["terminal_cause"]))
                    continue
                if run.get("per_icp_score") is not None:
                    continue
                if (run.get("status") == "failed"
                    and run.get("terminal_cause") in contracts.MODEL_CAUSED_TERMINAL_CAUSES):
                    try:
                        result = contracts.validate_run_result(run.get("result_doc"))
                    except ArenaContractError as exc:
                        raise ServiceError("execution_terminal_evidence_invalid", 500) from exc
                    if (result["terminal_status"] != run["terminal_cause"]
                        or run.get("output_ref") is not None
                        or run.get("qualification_doc") is not None):
                        raise ServiceError("execution_terminal_evidence_invalid", 500)
                    remember(submission_id, "execution_score_missing")
                    continue
                if run.get("status") == "accepted":
                    judge = scores.get(str(run["run_id"]))
                    if judge is not None and judge.get("status") == "accepted":
                        # Accepted but unmaterialized or invalid decisions must
                        # pass score recovery; they are never omission evidence.
                        raise ServiceError("accepted_judgment_score_missing", 500)
                    cause = str((judge or {}).get("terminal_cause") or ("judge_pending" if judge else "judge_missing"))
                    remember(submission_id, cause)
                else:
                    remember(submission_id, str(run.get("terminal_cause") or "execution_incomplete"))
        return reasons

    def _evaluation_draining(self, row: Mapping[str, Any]) -> bool:
        runs = self._store.list_runs(str(row["round_id"]))
        if any(run.get("status") in ("leased", "submitted") for run in runs):
            return True
        return any(int(provider.get("inflight_calls", 0)) > 0
            for participant in row.get("participants") or []
            for provider in self._store.submission_costs(str(participant["submission_id"]))["providers"])

    def _hold_incomplete_baseline(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return self._hold_incomplete_evaluation(row, reason="baseline_incomplete")

    def _hold_incomplete_evaluation(
        self, row: Mapping[str, Any], *, reason: str
    ) -> Dict[str, Any]:
        """Drain unfinished evaluation at its frozen final deadline."""
        deadline = _parse_iso(row["configuration_doc"]["schedule"]["publication_deadline"])
        if self.now() >= deadline:
            runs = self._store.list_runs(str(row["round_id"]))
            active = any(run.get("status") in ("leased", "submitted") for run in runs)
            # Cancellation must not terminate a still-dispatched provider call.
            # Known failed fees remain durable and do not prolong this gate.
            for participant in row.get("participants") or []:
                costs = self._store.submission_costs(str(participant["submission_id"]))
                active = active or any(int(provider.get("inflight_calls", 0)) > 0
                                       for provider in costs["providers"])
            if not active:
                return self._store.cancel_round(str(row["round_id"]), CANCEL_REASONS["scoring_incomplete"])
        return {"status": "retry", "round_status": row["status"],
                "reason": reason}

    def activate_reward(self, round_id: str) -> Dict[str, Any]:
        """Sign and atomically activate one already-published live result."""

        row = self._round(round_id)
        if row.get("reward_activated_at"):
            return {"status": "existing", "effective_reward_epoch": row.get("effective_reward_epoch")}
        configuration = row.get("configuration_doc") or {}
        if row.get("status") != "published":
            return {"status": "stale", "round_status": row.get("status")}
        if configuration.get("mode") != "live" or configuration.get("rewards_enabled") is not True:
            return {"status": "disabled"}
        if self._superseded_by_published_day(row):
            return {"status": "superseded"}
        if not self._baseline_final_ready(row):
            return {"status": "waiting_for_baseline"}
        # A later no-winner round must not activate while an earlier accepted
        # baseline in the same chain scope is still unpublished.
        network_name, netuid = self._chain_scope()
        if self._pending_promotion_blocks():
            return {"status": "waiting_for_promotion"}
        publication = row.get("publication_doc") or {}
        decision = publication.get("king_decision") or {}
        if (
            decision.get("outcome") == "crowned"
            and not row.get("baseline_promoted_at")
        ):
            return {"status": "waiting_for_promotion"}
        prior = self._store.published_reward_bases(
            mode="live", network_name=network_name, netuid=netuid, limit=200
        )
        maximum_epoch = max((int(item["effective_reward_epoch"]) for item in prior if item.get("effective_reward_epoch") is not None), default=-1)
        effective_epoch = max(int(self._config.chain.current_settlement_epoch()) + 1, maximum_epoch + 1)
        usable = self._usable_reward_bases(prior)
        previous = self._latest_miner_basis(usable)
        baseline_hotkey = str(configuration.get("baseline_hotkey") or "")
        if previous is not None and previous["king_hotkey"] == baseline_hotkey:
            previous = None
        daily_hotkey = (
            str(decision.get("king_hotkey") or "")
            if decision.get("outcome") == "crowned"
            else ""
        )
        # Old pending publications can name the organizer baseline. Treat them
        # as no-winner days so migration cannot turn that baseline into a payee.
        if not daily_hotkey or daily_hotkey == baseline_hotkey:
            daily_hotkey = ""
        if daily_hotkey:
            king_hotkey = daily_hotkey
            king_outcome = (
                "defended"
                if previous is not None and previous["king_hotkey"] == daily_hotkey
                else "crowned"
            )
        elif previous is not None:
            king_hotkey = str(previous["king_hotkey"])
            king_outcome = "defended"
        else:
            king_hotkey = ""
            king_outcome = "no_king"
        previous_start = (
            int(previous["king_start_epoch"])
            if previous is not None and king_outcome == "defended"
            else None
        )
        champion_reward_factor_ppm = rewards.FULL_CHAMPION_REWARD_FACTOR_PPM
        fallback_providers = row.get("champion_fallback_providers") or []
        if (
            not isinstance(fallback_providers, (list, tuple))
            or any(provider not in contracts.PROVIDERS for provider in fallback_providers)
            or len(set(fallback_providers)) != len(fallback_providers)
        ):
            raise ServiceError("champion_funding_state_invalid", 500)
        # A fresh winner always proves a full-share result, including when the
        # hotkey happens to match the incumbent. Without a fresh winner, start
        # from the prior signed factor. This preserves a half-share penalty over
        # failed or legacy days instead of repeatedly halving or silently
        # restoring it. A complete miner-funded baseline restores the full
        # factor; any organizer-account fallback latches the fixed half factor.
        if not daily_hotkey and previous is not None:
            champion_reward_factor_ppm = int(
                previous.get(
                    "champion_reward_factor_ppm",
                    rewards.FULL_CHAMPION_REWARD_FACTOR_PPM,
                )
            )
            snapshot_matches = (
                row.get("champion_funding_frozen") is True
                and str(row.get("champion_hotkey") or "") == king_hotkey
            )
            if snapshot_matches and fallback_providers:
                champion_reward_factor_ppm = (
                    rewards.FALLBACK_CHAMPION_REWARD_FACTOR_PPM
                )
            elif snapshot_matches and self._baseline_fully_succeeded(row):
                champion_reward_factor_ppm = (
                    rewards.FULL_CHAMPION_REWARD_FACTOR_PPM
                )
        slot_policy = rewards.reward_slot_policy_document(
            self._evaluation_day(row), effective_reward_epoch=effective_epoch,
        )
        reward_slots = self._store.reward_slot_snapshot(round_id, slot_policy)
        basis = self._sign(
            rewards.reward_basis_document(
                round_id=round_id,
                published_at=str(publication["published_at"]),
                finalized_epoch=effective_epoch - 1,
                king_hotkey=king_hotkey,
                king_outcome=king_outcome,
                previous_king_start_epoch=previous_start,
                reward_constants=configuration["reward_constants"],
                champion_reward_factor_ppm=champion_reward_factor_ppm,
                slot_policy=slot_policy,
                reward_slots=reward_slots,
            ),
            "reward_basis_hash",
        )
        return self._store.activate_reward(round_id, basis, self.signing_key_document())

    # -- runner handlers (section 14.3) ----------------------------------------

    def _lease_token(self, validated: Mapping[str, Any]) -> str:
        return contracts.document_hash({"lease": validated["request_id"], "signature": validated["signature"]})[7:]

    def _original_claim_for_recovery(self, validated: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Bind fresh claim authority to one older, unchanged signed request."""

        body = validated["body"]
        if "recover_claim" not in body:
            return None
        original = body["recover_claim"]
        if set(body) != {"recover_claim"} or not isinstance(original, Mapping):
            raise ServiceError("claim_recovery_invalid", 400)
        timestamp = original.get("timestamp")
        if (type(timestamp) is not int or timestamp > validated["timestamp"]
            or original.get("request_id") == validated["request_id"]
            or original.get("hotkey") != validated["hotkey"]):
            raise ServiceError("claim_recovery_invalid", 400)
        try:
            claim = contracts.validate_signed_request(
                original, expected_scope=contracts.SCOPE_CLAIM,
                expected_round_id=validated["round_id"], now=timestamp,
                verify_signature=self._config.verify_signature,
            )
        except (ArenaContractError, ArenaSignatureError, TypeError, ValueError):
            raise ServiceError("claim_recovery_invalid", 400) from None
        if "recover_claim" in claim["body"]:
            raise ServiceError("claim_recovery_invalid", 400)
        return claim

    def _recovered_claim_expiry(
        self, *, response: Mapping[str, Any], claim: Mapping[str, Any],
        token: str, round_id: str,
    ) -> Optional[str]:
        """Return the current expiry only for this exact owned, live lease."""

        run = self._store.get_run(str(response.get("run_id") or ""))
        current_round = self._store.get_round(round_id)
        if run is None or current_round is None:
            raise ServiceError("claim_recovery_unavailable", 503)
        if (run.get("round_id") != round_id
            or run.get("claim_request_id") != claim["request_id"]
            or run.get("claim_request_hash") != contracts.request_bytes_hash(claim)):
            raise ServiceError("claim_recovery_invalid", 503)
        try:
            raw_expiry = run.get("lease_expires_at")
            expiry = raw_expiry if isinstance(raw_expiry, datetime) else datetime.fromisoformat(
                str(raw_expiry).replace("Z", "+00:00")
            )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("claim_recovery_unavailable", 503) from None
        generation_current = run.get("stage_generation") == current_round.get("stage_generation")
        # SQL permits a bounded score continuation across a round generation.
        # Its later lease-bound RPCs remain the final authority for that case.
        continued_score = (
            run.get("kind") == "score" and _judge_continuation_enabled(current_round)
            and int(run.get("scoring_generation_floor") or 0) == self._score_generation_floor(current_round)
            and ((run.get("stage") == 1 and current_round.get("status") in (
                "stage1_scoring", "stage1_judged", "stage1_scored", "stage2",
                "stage2_closed", "stage2_scoring", "stage2_judged", "scored",
            )) or (run.get("stage") == 2 and current_round.get("status") in (
                "stage2_scoring", "stage2_judged", "scored",
            )))
        )
        live = (
            run.get("status") == "leased"
            and run.get("runner_hotkey") == claim["hotkey"]
            and hmac.compare_digest(
                str(run.get("lease_token_hash") or ""), hash_lease_token(token)
            )
            and run.get("lease_generation") == response.get("lease_generation")
            and run.get("stage_generation") == response.get("stage_generation")
            and (generation_current or continued_score)
            and (run.get("kind") != "score"
                 or int(run.get("scoring_generation_floor") or 0) == self._score_generation_floor(current_round))
            and current_round.get("status") not in TERMINAL_STATUSES
            and expiry.astimezone(timezone.utc) > self.now()
        )
        return expiry.astimezone(timezone.utc).isoformat() if live else None

    def _finalize_claim_recovery(
        self, lease: Dict[str, Any], original: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if original is None:
            return lease
        current_expiry = self._recovered_claim_expiry(
            response=lease, claim=original, token=lease["lease_token"],
            round_id=lease["round_id"],
        )
        if current_expiry is None:
            return {"status": "no_work", "claim_recovery": "resolved",
                    "claim_recovery_request_id": original["request_id"]}
        # A provider heartbeat may have renewed the durable lease after its
        # cached claim response. Report that verified deadline; never renew it.
        return dict(lease, lease_expires_at=current_expiry)

    def _require_validator_authority(self, hotkey: str) -> None:
        """Require validator identity for existing work, without a stake gate."""

        authorizer = getattr(self._config, "validator_authorizer", None)
        if authorizer is None:
            network_name, netuid = self._chain_scope()
            authorizer = lambda candidate: _gateway_validator_authorizer(
                candidate,
                network_name=network_name,
                netuid=netuid,
                metagraph=self._config.chain.metagraph(finalized=True),
            )
        try:
            registered, role = authorizer(hotkey)
        except Exception as exc:
            raise ServiceError("runner_validator_authority_unavailable", 503) from exc
        if not registered:
            raise ServiceError("runner_hotkey_unregistered", 403)
        if role != "validator":
            raise ServiceError("runner_validator_required", 403)

    def _benchmark_snapshot(self) -> chain_module.MetagraphSnapshot:
        try:
            return self._config.chain.metagraph(finalized=True)
        except Exception as exc:
            raise ServiceError("runner_benchmark_eligibility_unavailable", 503) from exc

    def _benchmark_validator_uid(self, snapshot: Any, hotkey: str) -> int:
        try:
            network_name, netuid = self._chain_scope()
            uid = validator_uid(
                snapshot, hotkey, netuid=netuid,
                network_name=chain_module.normalize_network_name(network_name),
            )
            # Bind miner self-dealing exclusions to the same authorized snapshot.
            owners = tuple(snapshot.coldkeys)
            if len(owners) != len(snapshot.hotkeys) or any(
                not isinstance(owner, str) or not owner for owner in owners
            ):
                raise ValueError("metagraph coldkeys are invalid")
            return uid
        except ValidatorIneligible as exc:
            raise ServiceError(str(exc), 403) from exc
        except Exception as exc:
            raise ServiceError("runner_benchmark_eligibility_unavailable", 503) from exc

    def handle_claim(self, envelope: Any) -> Dict[str, Any]:
        validated, round_row = self._request_round(envelope, scope=contracts.SCOPE_CLAIM, hot=True)
        round_id = round_row["round_id"]
        original = self._original_claim_for_recovery(validated)
        claim = original or validated
        if round_row["status"] in TERMINAL_STATUSES:
            if original is not None:
                return {"status": "no_work", "claim_recovery": "resolved",
                        "claim_recovery_request_id": original["request_id"]}
            raise ServiceError("round_ended", 409)
        body = claim["body"]
        declared = body.get("declared_parallelism")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
            raise ServiceError("declared_parallelism_invalid", 400)
        configuration = round_row["configuration_doc"]
        required_output_schema = contact_policy.output_schema(configuration)
        if (
            required_output_schema
            == intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
        ):
            advertised_output_schemas = body.get("output_schema_versions")
            if (
                not isinstance(advertised_output_schemas, list)
                or any(
                    not isinstance(schema, str)
                    for schema in advertised_output_schemas
                )
                or required_output_schema not in advertised_output_schemas
            ):
                raise ServiceError(
                    "validator_output_schema_upgrade_required", 409
                )
        snapshot = self._benchmark_snapshot()
        proxy_execution_required = (
            configuration.get("parallel_twenty_icp_execution") is True
            or configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        )
        if (
            proxy_execution_required
            and body.get("proxy_execution_version")
            != contracts.PROXY_EXECUTION_VERSION
        ):
            raise ServiceError("validator_proxy_execution_upgrade_required", 409)
        checkpoint_policy = configuration.get("checkpoint_deadline_policy")
        if checkpoint_policy in contracts.CHECKPOINT_DEADLINE_PROFILES:
            advertised_policies = body.get("checkpoint_deadline_policies")
            supports_checkpoint = body.get("checkpoint_deadline_policy") == checkpoint_policy
            if isinstance(advertised_policies, list) and all(
                isinstance(policy, str) for policy in advertised_policies
            ):
                supports_checkpoint = (
                    supports_checkpoint or checkpoint_policy in advertised_policies
                )
            if not supports_checkpoint:
                raise ServiceError("validator_checkpoint_upgrade_required", 409)
        response = None
        try:
            uid = self._benchmark_validator_uid(snapshot, validated["hotkey"])
        except ServiceError as exc:
            if exc.code != "runner_stake_below_minimum":
                raise
            # A lost response must not strand a lease issued before stake fell.
            # This exact signed-request lookup cannot allocate or extend work.
            response = self._store.recover_claim_response(
                round_id=round_id, runner_hotkey=validated["hotkey"],
                request_id=claim["request_id"],
                request_hash=contracts.request_bytes_hash(claim),
            )
            if response is None:
                raise
        else:
            excluded = chain_module.hotkeys_owned_by_coldkey(snapshot, snapshot.coldkeys[uid])
            if integrity.enabled(configuration):
                runner_coldkey = snapshot.coldkeys[uid]
                for submission in self._store.list_submissions(
                    round_id,
                    status="frozen",
                    columns="miner_hotkey,owner_coldkey,is_king",
                ):
                    if (
                        not bool(submission.get("is_king"))
                        and submission.get("owner_coldkey") == runner_coldkey
                    ):
                        excluded.append(str(submission.get("miner_hotkey") or ""))
                excluded = sorted(set(excluded) - {""})
        token = self._lease_token(claim)
        if response is None:
            response = self._store.claim_assignment(
                round_id=round_id, runner_hotkey=validated["hotkey"], declared_parallelism=declared, slot_ceiling=int(configuration["runner_slot_ceiling"]),
                excluded_miner_hotkeys=excluded, request_id=claim["request_id"], request_hash=contracts.request_bytes_hash(claim), lease_token_hash=hash_lease_token(token),
                lease_ttl_seconds=int(configuration["lease_ttl_seconds"]),
            )
        if response.get("status") != "leased":
            return (dict(response, claim_recovery="unresolved",
                         claim_recovery_request_id=original["request_id"])
                    if original is not None else response)
        if original is not None and not self._recovered_claim_expiry(
            response=response, claim=claim, token=token, round_id=round_id,
        ):
            return {"status": "no_work", "claim_recovery": "resolved",
                    "claim_recovery_request_id": original["request_id"]}
        self._require_code_review(str(response["submission_id"]), round_row)
        icps = self.evaluation_icps(round_id) if integrity.enabled(configuration) else self.benchmark_icps(round_id)
        position = int(response["icp_position"])
        if not 0 <= position < len(icps):
            raise ServiceError("benchmark_data_invalid", 500)
        lease_icp = integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(configuration)) if integrity.enabled(configuration) else dict(icps[position])
        # Generation can be enabled before round activation. Only the frozen
        # round contract can request v2 output, including for legacy rounds.
        if not contact_policy.enabled(configuration):
            lease_icp.pop("contact_policy", None)
        lease = dict(response, icp=lease_icp, lease_token=token, round_id=round_id,
                     claim_request_hash=contracts.request_bytes_hash(claim),
                     evaluation_date=str(round_row.get("evaluation_date") or ""))
        if original is not None:
            lease["claim_recovery"] = "resolved"
            lease["claim_recovery_request_id"] = original["request_id"]
        # Describes the gateway serving this lease, not the frozen scorer image.
        # No admission, accounting or completion decision depends on this value.
        lease["gateway_source_commit"] = runtime_version.SOURCE_METADATA["validator_source_commit"]
        # Reader support is independent of the frozen scoring/capture policy.
        # Older gateways omit this field, so new runners use legacy failure tags.
        lease[training_observations.OPTIONAL_FAILURE_TAG_CAPABILITY] = True
        lease["icp_wall_clock_seconds"] = int(configuration.get(
            "icp_wall_clock_seconds", contracts.ICP_WALL_CLOCK_SECONDS
        ))
        lease["scoring_wall_clock_seconds"] = int(configuration.get(
            "scoring_wall_clock_seconds", contracts.SCORING_WALL_CLOCK_SECONDS
        ))
        lease["lease_ttl_seconds"] = int(configuration["lease_ttl_seconds"])
        capture_configuration = baseline_reasoning.for_run(round_row, response)
        if response.get("kind") == "execute" and reasoning_capture.enabled(capture_configuration):
            # The worker's request validator receives the frozen round policy
            # through the gateway lease, never through a miner operation frame.
            lease["reasoning_capture_policy"] = reasoning_capture.POLICY
        if configuration.get("deepline_catalog") is not None:
            lease["deepline_catalog"] = configuration["deepline_catalog"]
        checkpoint_policy = configuration.get("checkpoint_deadline_policy")
        if checkpoint_policy in contracts.CHECKPOINT_DEADLINE_PROFILES:
            lease["checkpoint_deadline_policy"] = checkpoint_policy
        if integrity.enabled(configuration):
            lease["integrity_policy"] = integrity.POLICY
        if contact_policy.enabled(configuration):
            lease["contact_policy"] = contact_policy.POLICY
        if quality_policy.enabled(configuration):
            lease["company_quality_policy"] = quality_policy.POLICY
        if configuration.get("company_judgment_policy"):
            lease["company_judgment_policy"] = company_judgments.POLICY
        if intent_details_policy.enabled(configuration):
            lease["intent_details_policy"] = intent_details_policy.POLICY
        lease.update({
            "image_digest": configuration["scorer_image_digest"],
            "image_reference": configuration["scorer_image_reference"],
        })
        if response.get("kind") == "score":
            # A scoring assignment: the validator runs the pinned judge image on
            # the scored output with the signed scorer policy.
            current_score_run = self._store.get_run(str(response["run_id"]))
            if current_score_run is None:
                raise ServiceError("run_missing", 404)
            if current_score_run.get("lease_generation") != response.get("lease_generation"):
                raise ServiceError("reviewer_runtime_lease_changed", 409)
            try:
                binding = reviewer_runtime.lease_binding(current_score_run, configuration)
                lease.update(reviewer_runtime.selected_image(current_score_run, configuration))
            except reviewer_runtime.ReviewerRuntimeError as exc:
                raise ServiceError("reviewer_runtime_invalid", 503) from exc
            if binding is not None:
                lease["reviewer_runtime_doc"] = binding
            if company_judgments.enabled(configuration):
                submission = self._store.get_submission(
                    str(response["submission_id"])
                ) or {}
                try:
                    self._validated_company_judgment_lease(
                        response.get("company_judgment_cache") or {},
                        recipient_miner_hotkey=str(
                            response.get("miner_hotkey") or ""
                        ),
                        baseline=bool(submission.get("is_king")),
                    )
                except scoring.ScoringError as exc:
                    raise ServiceError(
                        "company_judgment_cache_invalid", 500
                    ) from exc
            scored = self._store.get_run(str(response.get("scored_run_id") or ""))
            if scored is None or not scored.get("output_ref"):
                raise ServiceError("scored_run_missing", 500)
            output = json.loads(self._objects.get(scored["output_ref"]).decode("utf-8"))
            lease.update({"scored_output": output, "scorer_policy": configuration["scorer_policy"]})
            if contact_policy.enabled(configuration):
                companies = validate_output_document(
                    output,
                    expected_schema_version=contact_policy.output_schema(
                        configuration
                    ),
                )["companies"]
                lease["contact_source_evidence"] = contact_evidence.resolve_sources(self._store, scored, companies)
            if (
                intent_details_policy.enabled(configuration)
                and provider_observations.enabled(configuration["scorer_policy"])
            ):
                companies = validate_output_document(
                    output,
                    expected_schema_version=contact_policy.output_schema(
                        configuration
                    ),
                )["companies"]
                lease["icp"] = provider_observations.scoring_icp(
                    lease["icp"],
                    configuration["scorer_policy"],
                    provider_observations.resolve_observations(
                        self._store, scored, companies,
                        str(round_row.get("evaluation_date") or ""),
                    ),
                )
            if not company_judgments.enabled(configuration) and integrity.enabled(configuration):
                try:
                    prior = self._prior_legacy_terminal_breakdowns(current_score_run, round_row)
                    if prior:
                        lease["prior_terminal_breakdowns"] = prior
                except (scoring.ScoringError, ServiceError, ValueError, OutputInvalid) as exc:
                    raise ServiceError("legacy_checkpoint_invalid", 503) from exc
            return self._finalize_claim_recovery(lease, original)
        # An execution uses the participant's private source archive under the
        # same trusted Python image as every other agent.
        participant = next((p for p in round_row.get("participants") or [] if p["submission_id"] == response.get("submission_id")), None)
        if participant is None:
            raise ServiceError("participant_missing", 500)
        if any(
            participant.get(field) in (None, "")
            for field in ("source_ref", "source_size_bytes")
        ):
            raise ServiceError("participant_source_missing", 500)
        lease.update(
            {
                "source_ref": participant["source_ref"],
                "source_size_bytes": int(participant["source_size_bytes"]),
                # Availability only: neither ciphertext nor the provider key
                # belongs in a validator lease or the model environment.
                "scrapingdog_configured": bool(participant.get("is_king")) or (
                    self._store.get_submission_credential(
                        str(participant["submission_id"]),
                        str(participant["miner_hotkey"]),
                        "scrapingdog",
                    ) is not None
                ),
            }
        )
        if configuration.get("parallel_twenty_icp_execution") is True:
            lease["parallel_twenty_icp_execution"] = True
        if (
            configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        ):
            lease["execution_sequence_policy"] = (
                contracts.BASELINE_SCORED_FIRST_POLICY
            )
        if (
            configuration.get("sourcing_cost_eligibility_policy")
            in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
        ):
            lease["sourcing_cost_eligibility_policy"] = (
                configuration["sourcing_cost_eligibility_policy"]
            )
        return self._finalize_claim_recovery(lease, original)

    def _run_context(self, run_id: str, lease_token: str) -> Tuple[Dict[str, Any], broker_module.RunContext]:
        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        judgment_cache_scope = None
        row = self._hot_round(str(run["round_id"]))
        if row is None:
            raise ServiceError("round_missing", 404)
        configuration = row.get("configuration_doc") or {}
        if run.get("kind") == "score":
            # Only gateway-owned frozen configuration can select a reusable
            # provider judgment. Run/miner identity is deliberately excluded.
            judgment_cache_scope = {
                "round_id": str(row["round_id"]),
                "evaluation_date": str(row.get("evaluation_date") or ""),
                "scorer_image_digest": configuration.get("scorer_image_digest"),
                "scorer_image_reference": configuration.get("scorer_image_reference"),
                "scorer_policy": configuration.get("scorer_policy"),
            }
        capture_configuration = baseline_reasoning.for_run(row, run)
        return run, broker_module.RunContext(
            run_id=run_id, assignment_id=run["assignment_id"],
            attempt=int(run["attempt"]), icp_position=int(run["icp_position"]),
            lease_token_hash=hash_lease_token(lease_token),
            miner_hotkey=run["miner_hotkey"], submission_id=run["submission_id"],
            stage=int(run["stage"]), kind=str(run.get("kind") or "execute"),
            round_id=str(run.get("round_id") or ""),
            judgment_cache_scope=judgment_cache_scope,
            deepline_catalog=configuration.get("deepline_catalog"),
            reasoning_capture_policy=capture_configuration.get("reasoning_capture_policy"),
            reasoning_route_registry=capture_configuration.get("reasoning_route_registry"),
        )

    def handle_signal_judgment(
        self, run_id: str, lease_token: str,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Resolve one signal under the current score lease and frozen input."""

        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        if (
            run.get("kind") != "score"
            or run.get("status") != "leased"
            or run.get("lease_token_hash") != hash_lease_token(lease_token)
        ):
            raise ServiceError("signal_judgment_lease_invalid", 403)
        round_row = self._hot_round(str(run["round_id"]))
        if round_row is None:
            raise ServiceError("round_missing", 404)
        configuration = round_row.get("configuration_doc") or {}
        policy = configuration.get("scorer_policy") or {}
        if policy.get("signal_judgment_policy") != signal_judgments.POLICY:
            raise ServiceError("signal_judgment_policy_unavailable", 409)
        if not isinstance(request, Mapping):
            raise ServiceError("signal_judgment_request_invalid", 400)
        operation = request.get("operation")
        company_index = request.get("company_index")
        signal_index = request.get("signal_index")
        if operation not in ("claim", "complete") or type(company_index) is not int or type(signal_index) is not int:
            raise ServiceError("signal_judgment_request_invalid", 400)
        executed = self._store.get_run(str(run.get("scored_run_id") or ""))
        if executed is None or executed.get("status") != "accepted" or not executed.get("output_ref"):
            raise ServiceError("scored_run_missing", 500)
        try:
            output = json.loads(self._objects.get_bounded(
                str(executed["output_ref"]), MAX_OUTPUT_BYTES
            ).decode("utf-8"))
            companies = validate_output_document(
                output, expected_schema_version=contact_policy.output_schema(configuration)
            )["companies"]
            icp = self.evaluation_icps(str(run["round_id"]))[int(run["icp_position"])]
            scoring_input = scoring.build_scoring_input(
                scored_run_id=str(executed["run_id"]),
                icp=integrity.agent_visible_icp(
                    icp, contacts_required=contact_policy.enabled(configuration)
                ),
                companies=companies, policy=policy,
                evaluation_date=str(round_row.get("evaluation_date") or ""),
                contact_source_evidence=(
                    contact_evidence.resolve_sources(self._store, executed, companies)
                    if contact_policy.enabled(configuration) else None
                ),
                provider_observations=(
                    provider_observations.resolve_observations(
                        self._store, executed, companies,
                        str(round_row.get("evaluation_date") or ""),
                    ) if intent_details_policy.enabled(configuration)
                    and provider_observations.enabled(policy) else None
                ),
            )
            scope = signal_judgments.build_scope(
                scoring_input=scoring_input, company_index=company_index,
                signal_index=signal_index,
                verified_company_identity=request.get("verified_company_identity"),
                verified_funding_stage_context=request.get("verified_funding_stage_context"),
                verified_source_pages=request.get("verified_source_pages"),
                round_id=str(run["round_id"]),
                network_name=str(round_row.get("arena_network_name") or ""),
                netuid=int(round_row.get("arena_netuid") or 0),
                scorer_image_digest=str(configuration.get("scorer_image_digest") or ""),
                scorer_image_reference=str(configuration.get("scorer_image_reference") or ""),
            )
        except (ValueError, TypeError, KeyError, IndexError, OutputInvalid,
                scoring.ScoringError, signal_judgments.SignalJudgmentError,
                company_judgments.CompanyJudgmentError) as exc:
            raise ServiceError("signal_judgment_input_invalid", 400) from exc
        lease_hash = hash_lease_token(lease_token)
        if operation == "claim":
            result = self._store.claim_signal_judgment(
                run_id=run_id, lease_token_hash=lease_hash,
                company_index=company_index, signal_index=signal_index,
                scope=scope,
            )
            if result.get("status") in ("busy", "pending"):
                return {"status": result["status"]}
            try:
                reference = signal_judgments.validate_reference(result.get("reference"), scope)
                if result.get("status") == "hit":
                    judgment = signal_judgments.validate_judgment(result.get("judgment"), scope)
                    source = self._store.get_signal_judgment(reference["cache_key"], reference["authority_slot"])
                    if (source is None or source.get("signal_input_hash") != reference["signal_input_hash"]
                        or source.get("scope_doc") != scope
                        or source.get("judgment_hash") != contracts.document_hash(judgment)
                        or source.get("judgment_doc") != judgment):
                        raise signal_judgments.SignalJudgmentError("accepted signal evidence changed")
                    source_run = self._store.get_run(str(source.get("source_score_run_id") or ""))
                    exclusions = source.get("runner_authority_exclusions")
                    same_current_lease = (
                        source.get("source_score_run_id") == run_id
                        and source_run is not None
                        and source_run.get("run_id") == run_id
                        and source_run.get("status") == "leased"
                        and source_run.get("lease_token_hash") == lease_hash
                        and source_run.get("claim_request_hash") == run.get("claim_request_hash")
                        and source_run.get("lease_generation") == run.get("lease_generation")
                        and source_run.get("runner_hotkey") == run.get("runner_hotkey")
                    )
                    if (source_run is None or source_run.get("kind") != "score"
                        or source_run.get("status") not in ("leased", "accepted", "failed")
                        or source_run.get("round_id") != run.get("round_id")
                        or source_run.get("scored_run_id") != source.get("source_scored_run_id")
                        or source_run.get("runner_hotkey") != source.get("source_runner_hotkey")
                        or source_run.get("claim_request_hash") != source.get("source_claim_request_hash")
                        or source_run.get("lease_generation") != source.get("source_lease_generation")
                        or not isinstance(exclusions, list)
                        or source.get("source_runner_hotkey") not in exclusions
                        or (run.get("miner_hotkey") in exclusions and not same_current_lease)):
                        raise signal_judgments.SignalJudgmentError("accepted signal source is invalid")
                    return {"status": "hit", "reference": reference, "judgment": judgment}
                if result.get("status") == "miss":
                    return {"status": "miss", "reference": reference}
            except signal_judgments.SignalJudgmentError as exc:
                raise ServiceError("signal_judgment_cache_invalid", 500) from exc
            raise ServiceError("signal_judgment_claim_invalid", 500)
        try:
            reference = signal_judgments.validate_reference(request.get("reference"), scope)
            judgment = (signal_judgments.validate_judgment(request["judgment"], scope)
                        if request.get("judgment") is not None else None)
        except signal_judgments.SignalJudgmentError as exc:
            raise ServiceError("signal_judgment_result_invalid", 400) from exc
        result = self._store.complete_signal_judgment(
            run_id=run_id, lease_token_hash=lease_hash,
            company_index=company_index, signal_index=signal_index,
            scope=scope, reference=reference, judgment=judgment,
            judgment_hash=contracts.document_hash(judgment) if judgment is not None else "",
        )
        if result.get("status") not in ("stored", "existing", "released", "pending"):
            raise ServiceError("signal_judgment_completion_invalid", 500)
        return {"status": result["status"], "reference": reference}

    def handle_quota_snapshot(
        self,
        run_id: str,
        lease_token: str,
        *,
        include_sourcing_cost: bool = False,
    ) -> Dict[str, Any]:
        """Return counters for one live lease without changing run state."""

        try:
            run = self._store.get_run(run_id)
            if run is None:
                raise ArenaStoreError("run missing")
            self._require_round_ownership(str(run.get("round_id") or ""))
            if include_sourcing_cost:
                return self._store.run_quota_snapshot(
                    run_id,
                    hash_lease_token(lease_token),
                    include_sourcing_cost=True,
                )
            return self._store.run_quota_snapshot(
                run_id, hash_lease_token(lease_token)
            )
        except Exception:
            # A caller must not distinguish a missing run, stale lease, schema
            # failure, or database failure through this passive read.
            raise ServiceError("quota_unavailable", 503) from None

    def handle_trajectory(
        self, run_id: str, lease_token: str, document: Any
    ) -> Dict[str, Any]:
        """Append sanitized events while the exact run lease is active."""

        try:
            events = trajectory.validate_batch(document, secrets=(lease_token,))
        except trajectory.TrajectoryError as exc:
            raise ServiceError("trajectory_invalid:%s" % str(exc)[:80], 400) from None
        if any(not item["kind"].startswith("runtime.") for item in events):
            raise ServiceError("trajectory_invalid:runtime events required", 400)
        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        if run.get("reviewer_runtime_doc") is not None:
            row = self._round(str(run["round_id"]))
            try:
                binding = reviewer_runtime.lease_binding(run, row.get("configuration_doc") or {})
            except reviewer_runtime.ReviewerRuntimeError as exc:
                raise ServiceError("reviewer_runtime_invalid", 503) from exc
            for event in events:
                if event["kind"] == "runtime.started":
                    # Host-owned selection audit also covers older validators.
                    # This records the lease, not proof of image execution.
                    event["content"]["reviewer_runtime_selection"] = {
                        key: binding[key] for key in reviewer_runtime.AUDIT_FIELDS
                    }
        try:
            result = self._store.append_trajectory_events(
                run_id, hash_lease_token(lease_token), events
            )
        except ArenaStoreError:
            raise ServiceError("trajectory_unavailable", 503) from None
        if result.get("status") == "stale":
            raise ServiceError("lease_stale", 409)
        if (
            result.get("status") != "accepted"
            or result.get("accepted") != len(events)
        ):
            raise ServiceError("trajectory_unavailable", 503)
        inserted = result.get("inserted")
        existing = result.get("existing")
        if (
            type(inserted) is int and type(existing) is int
            and 0 < inserted <= len(events)
            and 0 <= existing
            and inserted + existing == len(events)
        ):
            kinds = {event["kind"] for event in events}
            if kinds & {"runtime.started", "runtime.finished", "runtime.error"}:
                # The RPC proves that at least one event was newly persisted.
                # A lifecycle flag describes the accepted batch, which may
                # also contain replayed events; it is not a transition count.
                failed_finish = any(
                    event["kind"] == "runtime.finished"
                    and type(status := event["content"].get("status")) is str
                    and status in contracts.TERMINAL_CAUSES
                    and status != "accepted"
                    for event in events
                )
                telemetry.record_runtime_upload(
                    inserted_count=inserted,
                    replayed_count=existing,
                    has_started="runtime.started" in kinds,
                    has_finished="runtime.finished" in kinds,
                    has_error="runtime.error" in kinds or failed_finish,
                    run_identity=_telemetry_run_identity(run),
                )
        return result

    def _training_capture_run(
        self, run_id: str, lease_token: str, *,
        allow_observation_finalization: bool = False,
    ) -> Dict[str, Any]:
        """Authorize capture; only final observations have a bounded grace."""
        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        expected_hash = str(run.get("lease_token_hash") or "")
        if not expected_hash or not hmac.compare_digest(expected_hash, hash_lease_token(lease_token)):
            raise ServiceError("lease_invalid", 401)
        expired_attempt = (
            allow_observation_finalization
            and run.get("status") == "failed"
            and run.get("terminal_cause") == "lease_expired"
        )
        if run.get("status") != "leased" and not expired_attempt:
            raise ServiceError("lease_inactive", 409)
        if allow_observation_finalization:
            try:
                training_observations.object_ref(
                    run_id, run.get("lease_generation"), run.get("claim_request_hash"),
                )
            except (ValueError, TypeError):
                raise ServiceError("lease_invalid", 401) from None
        try:
            raw_expiry = run.get("lease_expires_at")
            expiry = raw_expiry if isinstance(raw_expiry, datetime) else datetime.fromisoformat(str(raw_expiry).replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("lease_invalid", 401) from None
        expiry = expiry.astimezone(timezone.utc)
        now = self.now()
        if expiry <= now:
            if (not allow_observation_finalization
                    or now >= expiry + _OBSERVATION_FINALIZATION_GRACE):
                raise ServiceError("lease_expired", 409)
        elif expired_attempt:
            raise ServiceError("lease_inactive", 409)
        return run

    def handle_training_input(
        self, run_id: str, lease_token: str, document: Any,
    ) -> Dict[str, Any]:
        """Store the final runner input only for this live, owned run lease."""
        run = self._training_capture_run(run_id, lease_token)
        try:
            snapshot, compressed = training_input.build_snapshot(
                run_id, str(run.get("kind") or "execute"), document,
                lease_generation=run.get("lease_generation"),
                claim_request_hash=run.get("claim_request_hash"),
                secrets=(lease_token,),
            )
        except (ValueError, TypeError, RecursionError):
            raise ServiceError("training_input_invalid", 400) from None
        objects = self._config.training_object_store
        if objects is None:
            raise ServiceError("training_input_unavailable", 503)
        try:
            objects.put(training_input.object_ref(run_id, snapshot["lease_generation"]), compressed)
        except ArenaContractError:
            raise ServiceError("training_input_conflict", 409) from None
        except Exception:
            raise ServiceError("training_input_unavailable", 503) from None
        return {
            "status": "stored", "schema_version": training_input.SCHEMA_VERSION,
            "ref": training_input.object_ref(run_id, snapshot["lease_generation"]),
            "lease_generation": snapshot["lease_generation"],
            "claim_request_hash": snapshot["claim_request_hash"],
            "input_sha256": snapshot["input_sha256"],
            "privacy": snapshot["privacy"],
        }

    def handle_training_observations(self, run_id: str, lease_token: str, document: Any) -> Dict[str, Any]:
        run = self._training_capture_run(run_id, lease_token, allow_observation_finalization=True)
        try:
            snapshot, compressed = training_observations.build_snapshot(
                run_id, str(run.get("kind") or "execute"), document,
                lease_generation=run.get("lease_generation"),
                claim_request_hash=run.get("claim_request_hash"), secrets=(lease_token,),
            )
        except (ValueError, TypeError, RecursionError):
            raise ServiceError("training_observations_invalid", 400) from None
        objects = self._config.training_object_store
        if objects is None:
            raise ServiceError("training_observations_unavailable", 503)
        ref = training_observations.object_ref(run_id, snapshot["lease_generation"], snapshot["claim_request_hash"])
        try:
            objects.put(ref, compressed)
        except ArenaContractError:
            raise ServiceError("training_observations_conflict", 409) from None
        except Exception:
            raise ServiceError("training_observations_unavailable", 503) from None
        return {"status": "stored", "schema_version": snapshot["schema_version"],
                "ref": ref, "lease_generation": snapshot["lease_generation"],
                "claim_request_hash": snapshot["claim_request_hash"],
                "payload_sha256": snapshot["payload_sha256"]}

    def verify_training_observations(self, run_id: str, lease_token: str, payload_sha256: str) -> str:
        """Confirm only the exact final snapshot under this owned lease identity."""

        run = self._training_capture_run(run_id, lease_token, allow_observation_finalization=True)
        if not isinstance(payload_sha256, str) or not contracts.SHA256_RE.fullmatch(payload_sha256):
            raise ServiceError("training_observations_invalid", 400)
        objects = self._config.training_object_store
        if objects is None:
            raise ServiceError("training_observations_unavailable", 503)
        try:
            snapshot = training_observations.read_snapshot(
                objects, run_id, str(run.get("kind") or "execute"),
                run.get("lease_generation"), run.get("claim_request_hash"),
            )
        except training_observations.ObservationError:
            raise ServiceError("training_observations_unavailable", 503) from None
        if not hmac.compare_digest(snapshot["payload_sha256"], payload_sha256):
            raise ServiceError("training_observations_digest_mismatch", 412)
        return snapshot["payload_sha256"]

    def handle_training_observation_segment(self, run_id: str, lease_token: str, document: Any) -> Dict[str, Any]:
        run = self._training_capture_run(run_id, lease_token, allow_observation_finalization=True)
        if not isinstance(document, Mapping):
            raise ServiceError("training_observation_segment_invalid", 400)
        try:
            segment, compressed = training_observations.build_segment(
                run_id, str(run.get("kind") or "execute"), document["payload"],
                lease_generation=run.get("lease_generation"),
                claim_request_hash=run.get("claim_request_hash"),
                number=document["number"], previous_sha256=document["previous_sha256"],
                cumulative_bytes=document["cumulative_bytes"],
                secrets=(lease_token,),
            )
            if set(document) != {"number", "previous_sha256", "cumulative_bytes", "payload"}:
                raise ValueError("segment fields invalid")
        except (KeyError, ValueError, TypeError, RecursionError):
            raise ServiceError("training_observation_segment_invalid", 400) from None
        objects = self._config.training_object_store
        if objects is None:
            raise ServiceError("training_observation_segment_unavailable", 503)
        if segment["number"]:
            try:
                previous = training_observations.read_segment(
                    objects, run_id, segment["kind"], segment["lease_generation"],
                    segment["claim_request_hash"], segment["number"] - 1)
            except Exception:
                raise ServiceError("training_observation_segment_unavailable", 503) from None
            if (previous is None or previous["segment_sha256"] != segment["previous_sha256"]
                    or segment["cumulative_bytes"] != previous["cumulative_bytes"]
                    + len(contracts.canonical_json(segment["payload"]).encode("utf-8"))
                    or any(segment["payload"][start] != previous["payload"][start]
                           + len(previous["payload"][rows]) for start, rows in (
                               ("observation_from", "observations"),
                               ("receipt_from", "response_receipts"),
                               ("agent_from", "agent_observations")))
                    or any(segment["payload"][name] < previous["payload"][name]
                           for name in ("observation_count", "local_refusal_count",
                                        "response_receipt_count", "agent_observation_count"))):
                raise ServiceError("training_observation_segment_conflict", 409)
        elif segment["cumulative_bytes"] != len(contracts.canonical_json(segment["payload"]).encode("utf-8")):
            raise ServiceError("training_observation_segment_invalid", 400)
        ref = training_observations.segment_ref(
            run_id, segment["lease_generation"], segment["claim_request_hash"], segment["number"])
        try:
            objects.put(ref, compressed)
        except ArenaContractError:
            raise ServiceError("training_observation_segment_conflict", 409) from None
        except Exception:
            raise ServiceError("training_observation_segment_unavailable", 503) from None
        return {"status": "stored", "number": segment["number"],
                "segment_sha256": segment["segment_sha256"]}

    def handle_source(self, run_id: str, lease_token: str) -> bytes:
        """Return source bytes only to the runner that holds the active lease."""

        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        if run.get("kind") != "execute":
            raise ServiceError("run_source_unavailable", 409)
        expected_token_hash = str(run.get("lease_token_hash") or "")
        if not expected_token_hash or not hmac.compare_digest(
            expected_token_hash, hash_lease_token(lease_token)
        ):
            raise ServiceError("lease_invalid", 401)
        if run.get("status") != "leased":
            raise ServiceError("lease_inactive", 409)
        raw_expiry = run.get("lease_expires_at")
        try:
            if isinstance(raw_expiry, datetime):
                expiry = raw_expiry
            else:
                encoded_expiry = str(raw_expiry).replace("Z", "+00:00")
                try:
                    expiry = datetime.fromisoformat(encoded_expiry)
                except ValueError:
                    # Python 3.9 accepts only three or six fractional digits
                    # here, but PostgreSQL can serialize any width from one
                    # through six. strptime accepts the full PostgreSQL range.
                    expiry = datetime.strptime(
                        encoded_expiry, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("lease_invalid", 401)
        if expiry.astimezone(timezone.utc) <= self.now():
            raise ServiceError("lease_expired", 409)
        submission = self._store.get_submission(str(run.get("submission_id") or ""))
        if submission is None or submission.get("status") != "frozen":
            raise ServiceError("run_source_unavailable", 409)
        source_ref = str(submission.get("source_ref") or "")
        expected_size = int(submission.get("source_size_bytes") or 0)
        try:
            payload = self._objects.get_bounded(
                source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
        except Exception as exc:
            raise ServiceError("run_source_unavailable", 503) from exc
        if len(payload) != expected_size:
            raise ServiceError("run_source_integrity_failed", 500)
        expected_digest = submission.get("source_archive_sha256")
        if expected_digest and not hmac.compare_digest(
            str(expected_digest), "sha256:" + hashlib.sha256(payload).hexdigest()
        ):
            raise ServiceError("run_source_integrity_failed", 500)
        return payload

    def handle_scorer_image_access(
        self, run_id: str, lease_token: str
    ) -> Mapping[str, Any]:
        """Return ECR blob capabilities only to the active lease's validator."""

        self._require_identifier(run_id, "run")
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        round_id = str(run.get("round_id") or "")
        self._require_round_ownership(round_id)
        expected_token_hash = str(run.get("lease_token_hash") or "")
        if not expected_token_hash or not hmac.compare_digest(
            expected_token_hash, hash_lease_token(lease_token)
        ):
            raise ServiceError("lease_invalid", 401)
        if run.get("status") != "leased":
            raise ServiceError("lease_inactive", 409)
        raw_expiry = run.get("lease_expires_at")
        try:
            if isinstance(raw_expiry, datetime):
                expiry = raw_expiry
            else:
                encoded_expiry = str(raw_expiry).replace("Z", "+00:00")
                try:
                    expiry = datetime.fromisoformat(encoded_expiry)
                except ValueError:
                    expiry = datetime.strptime(
                        encoded_expiry, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("lease_invalid", 401) from None
        if expiry.astimezone(timezone.utc) <= self.now():
            raise ServiceError("lease_expired", 409)
        round_row = self._round(round_id)
        if round_row.get("status") in TERMINAL_STATUSES:
            raise ServiceError("round_ended", 409)
        runner_hotkey = str(run.get("runner_hotkey") or "")
        try:
            contracts.require_hotkey(runner_hotkey)
        except ArenaContractError:
            raise ServiceError("lease_invalid", 401) from None
        self._require_validator_authority(runner_hotkey)
        provider = self._config.scorer_image_access
        if provider is None:
            raise ServiceError("scorer_image_access_unsupported", 409)
        configuration = round_row.get("configuration_doc") or {}
        try:
            selected = reviewer_runtime.selected_image(run, configuration)
        except reviewer_runtime.ReviewerRuntimeError as exc:
            raise ServiceError("reviewer_runtime_invalid", 503) from exc
        image_reference = selected["image_reference"]
        image_digest = selected["image_digest"]
        access = None
        failed = False
        try:
            access = provider(image_reference, image_digest)
        except scorer_image_access_module.ScorerImageAccessError:
            failed = True
        except Exception:
            failed = True
        if failed:
            # Raise outside the handler. Thus, an SDK exception containing a
            # signed URL is not retained as this public error's context.
            raise ServiceError("scorer_image_access_unavailable", 503) from None
        return access

    def _broker_for(self, round_id: str) -> broker_module.Broker:
        # Provider handlers already resolved this hot snapshot. Match the
        # factory's policy/routing inputs, without hashing its large catalogue
        # or retaining a prior policy's allowlist after an operator revision.
        row = self._hot_round(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        configuration = row.get("configuration_doc") or {}
        identity = contracts.document_hash({key: configuration.get(key) for key in (
            "scorer_policy", "lease_ttl_seconds", "baseline_source_url",
            "mode", "rewards_enabled",
        )})
        key = (round_id, identity)
        with self._lock:
            broker = self._brokers.get(key)
            if broker is None:
                broker = self._config.broker_factory(self, row)
                self._brokers[key] = broker
            return broker

    def _reconcile_openrouter_cost(
        self, round_id: str, *, run_id: str = ""
    ) -> Dict[str, Any]:
        """Perform at most one exact billing GET without holding service locks."""

        cursor_key = (round_id, run_id)
        with self._lock:
            after_entry_id = self._openrouter_reconciliation_after.get(
                cursor_key, 0
            )
        items = self._store.list_openrouter_cost_reconciliations(
            round_id,
            run_id=run_id,
            after_entry_id=after_entry_id,
            limit=1,
        )
        if not items:
            return {"status": "none"}
        candidate = items[0]
        with self._lock:
            self._openrouter_reconciliation_after[cursor_key] = int(
                candidate.get("uncertain_entry_id") or 0
            )
        result = self._broker_for(round_id).reconcile_openrouter_cost(candidate)
        return {
            "status": str(result.get("status") or "unavailable"),
            "run_status": str(candidate.get("run_status") or ""),
            "lease_expires_at": str(candidate.get("lease_expires_at") or ""),
        }

    def reconcile_closed_provider_costs(self) -> Dict[str, Any]:
        """Check a closed judge round without changing published results."""
        network_name, netuid = self._chain_scope()
        candidate = self._store.next_closed_provider_reconciliation(
            mode=self._config.mode, network_name=network_name, netuid=netuid,
            round_id=self._pinned_round_id() or "",
            after_entry_id=self._closed_provider_reconciliation_after,
        )
        if candidate.get("status") == "none":
            return {"status": "none"}
        if candidate.get("status") != "ok" or candidate.get("provider") not in {
            "deepline", "openrouter",
        }:
            raise ServiceError("closed_provider_cost_candidate_invalid", 500)
        self._closed_provider_reconciliation_after = int(candidate["uncertain_entry_id"])
        if candidate["provider"] == "deepline":
            # V1/V3 SQL alone decides which published score liabilities are
            # still billable. The active batch already uses those exact lists
            # and never replays a provider operation or rewrites publication.
            return self._reconcile_active_deepline_cost(str(candidate["round_id"]))
        return self._reconcile_openrouter_cost(
            str(candidate["round_id"]), run_id=str(candidate["run_id"]),
        )

    def _reconcile_deepline_cost(
        self,
        round_id: str,
        *,
        run_id: str = "",
        successful_execute_only: bool = False,
    ) -> Dict[str, Any]:
        """Perform one bounded billing check without holding service locks."""
        items = self._deepline_cost_candidates(
            round_id, run_id=run_id,
            successful_execute_only=successful_execute_only, limit=1,
        )
        if not items:
            return {"status": "none"}
        candidate = items[0]
        result = self._broker_for(round_id).reconcile_deepline_cost(candidate)
        return {
            "status": str(result.get("status") or "unavailable"),
            "run_status": str(candidate.get("run_status") or ""),
            "lease_expires_at": str(candidate.get("lease_expires_at") or ""),
        }

    def _deepline_cost_candidates(
        self, round_id: str, *, run_id: str = "",
        successful_execute_only: bool = False, score_only: bool = False,
        limit: int = 1,
        lookup_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Reserve a bounded cursor page before its billing reads begin."""
        cursor_key = (round_id, run_id)
        if successful_execute_only and score_only:
            raise ValueError("Deepline candidate lanes are mutually exclusive")
        cursors = (self._deepline_priority_reconciliation_after
                   if successful_execute_only else
                   self._deepline_score_reconciliation_after
                   if score_only else self._deepline_reconciliation_after)
        with self._lock:
            after_entry_id = cursors.get(cursor_key, 0)
        if lookup_context is not None:
            # Report the exact cursor sent to the RPC, not a second racy read.
            lookup_context["cursor_entry_id"] = after_entry_id
        options = ({"successful_execute_only": True} if successful_execute_only else
                   {"score_only": True} if score_only else {})
        items = self._store.list_deepline_cost_reconciliations(
            round_id,
            run_id=run_id,
            after_entry_id=after_entry_id,
            limit=limit,
            **options,
        )
        if not items:
            return []
        # The RPC's JSON aggregate sorts by entry ID, including a wrapped
        # page. Restore cursor order before reserving the end of that page.
        items = sorted(items, key=lambda item: (
            int(item.get("uncertain_entry_id") or 0) <= after_entry_id,
            int(item.get("uncertain_entry_id") or 0),
        ))
        with self._lock:
            cursors[cursor_key] = int(items[-1].get("uncertain_entry_id") or 0)
        return items

    def _reconcile_active_deepline_cost(self, round_id: str) -> Dict[str, Any]:
        """Check at most 96 bills in twelve waves of eight five-second reads.

        Abandoned, successful execute, score, and general lanes rotate
        independently. Selection does not authorize expiry or provider replay.
        """
        with self._lock:
            if round_id in self._active_deepline_reconciliations:
                return {"status": "none"}
            self._active_deepline_reconciliations.add(round_id)
            after_run = self._abandoned_billing_run_after.get(round_id, "")
        try:
            unique: Dict[str, Dict[str, Any]] = {}
            logger = logging.getLogger(__name__)
            try:
                with telemetry.stage("reconcile_deepline_billing_lookup") as observed:
                    abandoned = sorted(self._store.list_abandoned_billing_runs(round_id))
                    observed.count = len(abandoned)
                    observed.idle = not abandoned
            except Exception as exc:
                logger.warning("Arena abandoned billing lookup failed: %s", type(exc).__name__)
                abandoned = []

            def select(target: int, **options: Any) -> None:
                # Overlapping lanes may need more than one cursor page. Cap
                # selection reads even when a small candidate set wraps.
                seen_in_lane: set[str] = set()
                for _ in range(6):
                    missing = min(target - len(unique),
                                  _DEEPLINE_BILLING_BATCH_LIMIT - len(unique))
                    if missing <= 0:
                        return
                    page_limit = min(_DEEPLINE_BILLING_PAGE_LIMIT, missing)
                    lookup_context: Dict[str, Any] = {
                        "round_id": round_id,
                        "lane": ("execute" if options.get("successful_execute_only") else
                                 "score" if options.get("score_only") else "general"),
                        "run_scoped": bool(options.get("run_id")),
                    }
                    try:
                        with telemetry.stage("reconcile_deepline_billing_lookup",
                                             billing_lookup=lookup_context) as observed:
                            items = self._deepline_cost_candidates(
                                round_id, limit=page_limit,
                                lookup_context=lookup_context,
                                **options,
                            )
                            observed.count = len(items)
                            observed.idle = not items
                    except Exception as exc:
                        logger.warning("Arena billing candidate lookup failed: %s", type(exc).__name__)
                        return
                    if not items:
                        return
                    identities = {str(item["call_identity"]) for item in items}
                    if identities <= seen_in_lane:
                        return
                    seen_in_lane.update(identities)
                    for item in items:
                        unique.setdefault(str(item["call_identity"]), item)
                    if len(items) < page_limit:
                        return

            ordered = [run for run in abandoned if run > after_run]
            ordered += [run for run in abandoned if run <= after_run]
            for run_id in ordered[:_DEEPLINE_BILLING_ABANDONED_SLOTS]:
                select(len(unique) + 1, run_id=run_id)
                with self._lock:
                    self._abandoned_billing_run_after[round_id] = run_id
            select(len(unique) + _DEEPLINE_BILLING_EXECUTE_SLOTS,
                   successful_execute_only=True)
            select(len(unique) + _DEEPLINE_BILLING_GENERAL_SLOTS)
            select(len(unique) + _DEEPLINE_BILLING_SCORE_SLOTS, score_only=True)
            # Unused reserved slots return first to score, then to general.
            select(_DEEPLINE_BILLING_BATCH_LIMIT, score_only=True)
            select(_DEEPLINE_BILLING_BATCH_LIMIT)
            if not unique:
                return {"status": "none"}
            broker = self._broker_for(round_id)

            def reconcile(candidate: Mapping[str, Any]) -> Dict[str, Any]:
                try:
                    with telemetry.stage("reconcile_deepline_cost"):
                        return broker.reconcile_deepline_cost(candidate)
                except Exception as exc:
                    # A failed read/write leaves the original liability intact.
                    # Independent candidates must still make progress.
                    logger.warning(
                        "Arena delayed Deepline billing check failed: %s",
                        type(exc).__name__,
                    )
                    return {"status": "unavailable"}

            with telemetry.stage("reconcile_active_deepline_costs") as observed:
                with ThreadPoolExecutor(max_workers=_DEEPLINE_BILLING_WORKERS,
                                        thread_name_prefix="arena-billing") as pool:
                    results = list(pool.map(reconcile, unique.values()))
                settled = sum(result.get("status") == "settled" for result in results)
                observed.count = settled
                observed.idle = settled == 0
            status = "settled" if settled else str(results[0].get("status") or "unavailable")
            return {"status": status,
                    "checked": len(results), "settled": settled}
        finally:
            with self._lock:
                self._active_deepline_reconciliations.discard(round_id)

    def handle_provider(
        self,
        run_id: str,
        lease_token: str,
        frame: Any,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(frame, Mapping) or set(frame) != {"operation_id", "parameters", "timeout_ms", "action_sequence"}:
            raise ServiceError("frame_invalid", 400)
        limits = (contracts.RESPONSES_PROVIDER_FRAME_LIMITS
                  if frame["operation_id"] == "openrouter.responses" else contracts.PROVIDER_FRAME_LIMITS)
        # Decode only a bounded candidate before resolving its existing lease.
        # The frozen execute policy, never a caller hint, admits larger history.
        candidate_limits = limits
        if frame["operation_id"] in ("openrouter.chat", "openrouter.responses"):
            candidate_limits = (contracts.FULL_THINKING_RESPONSES_PROVIDER_FRAME_LIMITS
                if frame["operation_id"] == "openrouter.responses" else contracts.FULL_THINKING_PROVIDER_FRAME_LIMITS)
        contracts.check_strict_document(frame, candidate_limits)
        run, context = self._run_context(run_id, lease_token)
        if not (getattr(context, "kind", "execute") == "execute"
                and getattr(context, "reasoning_capture_policy", None) == reasoning_capture.POLICY):
            contracts.check_strict_document(frame, limits)
        broker = self._broker_for(run["round_id"])
        operation_id = str(frame["operation_id"])
        operation = operations.get_operation(operation_id)
        # The lease hash is derived from the supplied token in _run_context.
        # Only a match to the canonical run row can attach run attribution;
        # broker refusal and host failure still belong to that attempt.
        run_identity = (
            _telemetry_run_identity(run)
            if context.lease_token_hash == hash_lease_token(lease_token)
            and context.lease_token_hash == run.get("lease_token_hash")
            else None
        )

        def persist_provider_event(document: Mapping[str, Any]) -> None:
            # A fixed event UUID makes one replay safe when the first RPC
            # committed and only its response was lost. Contract, cap, and
            # stale-lease denials are not transport failures and are not retried.
            for attempt in range(2):
                try:
                    persisted = self._store.append_trajectory_events(
                        run_id, context.lease_token_hash, [document]
                    )
                    if (
                        persisted.get("status") != "accepted"
                        or persisted.get("accepted") != 1
                    ):
                        raise ArenaStoreError(
                            "provider trajectory was not accepted"
                        )
                    return
                except ArenaStoreUnavailable:
                    if attempt:
                        raise

        try:
            request_content = trajectory.provider_request_content(frame)
            if operation is not None:
                request_content["provider"] = operation.provider
            persist_provider_event(
                trajectory.event("provider.request", request_content)
            )
        except Exception as exc:
            # Diagnostics must never change provider dispatch, billing, retry,
            # or the result returned to the worker.
            _warn_trajectory_failure("provider_request", exc)
        started = time.monotonic()
        try:
            result = broker.execute(
                context,
                operation_id=operation_id,
                parameters=frame["parameters"],
                action_sequence=frame["action_sequence"],
                timeout_ms=int(frame["timeout_ms"]),
                cancel_requested=cancel_requested,
            )
        except Exception as exc:
            store_error = {}
            try:
                if isinstance(exc, ArenaStoreError):
                    store_error = safe_store_error_metadata(getattr(exc, "store_error", None))
            except Exception:
                pass  # A diagnostic must not replace the original exception.
            try:
                error_content = {
                    "operation_id": operation_id,
                    "provider": operation.provider if operation is not None else "unknown",
                    "action_sequence": frame["action_sequence"],
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "error_class": type(exc).__name__,
                }
                if store_error:
                    error_content["store_error"] = store_error
                attempts = getattr(exc, "_arena_trajectory_attempts", ())
                if isinstance(attempts, (list, tuple)):
                    error_content["provider_attempts"] = [
                        dict(item)
                        for item in attempts[:4]
                        if isinstance(item, Mapping)
                    ]
                persist_provider_event(
                    trajectory.event("provider.error", error_content)
                )
            except Exception as diagnostic_exc:
                _warn_trajectory_failure("provider_error", diagnostic_exc)
            raised_provider, raised_operation = _telemetry_provider_operation(operation_id)
            telemetry.record_provider(
                raised_provider,
                raised_operation,
                "failed",
                error_type=type(exc).__name__,
                store_error=store_error or None,
                tool=_telemetry_tool(operation_id, frame["parameters"]),
                duration_ms=(time.monotonic() - started) * 1000.0,
                run_identity=run_identity,
            )
            raise
        document = result.to_document()
        call = result.call if isinstance(result.call, Mapping) else {}
        try:
            setup_diagnostics = broker_module._safe_setup_diagnostics(getattr(result, "setup_diagnostics", None))
        except Exception:
            setup_diagnostics = {}  # Optional observation cannot replace a broker result.
        if call.get("cached") is not True and call.get("idempotent") is not True:
            # The call summary, not the frame, owns the provider half of this
            # span. Resolving the tool from the same operation id keeps the
            # two consistent, so a degraded operation can never pair a
            # Deepline tool with an "unknown" provider and lose the span.
            provider_fields = _provider_telemetry_fields(
                call, result.status, len(result.attempt_trace) or 1,
            )
            if setup_diagnostics:
                provider_fields["error_type"] = setup_diagnostics["error_class"]
                if setup_diagnostics.get("store_error"):
                    provider_fields["store_error"] = setup_diagnostics["store_error"]
            telemetry.record_provider(
                tool=_telemetry_tool(
                    call.get("operation_id") if isinstance(call.get("operation_id"), str) else "",
                    frame["parameters"],
                ),
                duration_ms=(time.monotonic() - started) * 1000.0,
                run_identity=run_identity,
                **provider_fields,
            )
        try:
            response_content = trajectory.provider_response_content(
                document,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            # The admitted frame is the stable correlation source. Early
            # broker errors may not have a call identity or repeat these fields.
            response_content["operation_id"] = operation_id
            response_content["action_sequence"] = frame["action_sequence"]
            if len(result.attempt_trace) > 1:
                response_content["provider_attempts"] = [
                    dict(item) for item in result.attempt_trace[:4]
                ]
            response_event = trajectory.event("provider.response", response_content)
            if setup_diagnostics:
                try:
                    response_event = trajectory.validate_event({
                        **response_event,
                        "content": {**response_event["content"], "setup_diagnostics": setup_diagnostics},
                    })
                except trajectory.TrajectoryError:
                    pass  # Optional diagnostics must preserve the original event at its cap.
            persist_provider_event(response_event)
        except Exception as exc:
            _warn_trajectory_failure("provider_response", exc)
        return document

    def handle_complete(self, envelope: Any) -> Dict[str, Any]:
        validated, round_row = self._request_round(envelope, scope=contracts.SCOPE_COMPLETE, hot=True)
        round_id = round_row["round_id"]
        body = validated["body"]
        run_id = str(body.get("run_id") or "")
        try:
            run_result = contracts.validate_run_result(body.get("result"))
        except ArenaContractError as exc:
            raise ServiceError("run_result_invalid:%s" % str(exc)[:80], 400)
        self._require_validator_authority(validated["hotkey"])
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        if str(run.get("round_id") or "") != round_id:
            raise ServiceError("run_round_mismatch", 400)
        if run.get("runner_hotkey") != validated["hotkey"]:
            raise ServiceError("run_runner_mismatch", 403)
        try:
            reviewer_runtime.validate_completion_audit(
                run, round_row.get("configuration_doc") or {}, run_result,
            )
        except reviewer_runtime.ReviewerRuntimeError as exc:
            raise ServiceError("reviewer_runtime_audit_invalid", 400) from exc
        kind = str(run.get("kind") or "execute")
        terminal_status = run_result["terminal_status"]
        if (
            kind == "execute"
            and terminal_status == "budget_exhausted"
            and (round_row.get("configuration_doc") or {}).get(
                "sourcing_cost_eligibility_policy"
            )
            in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
            and not any(
                entry.get("entry_kind") == "refusal"
                and (entry.get("entry_doc") or {}).get("reason")
                in contracts.PER_ICP_POLICY_BUDGET_STOP_REASONS
                for entry in self._store.list_ledger(
                    run_id=run_id, entry_kind="refusal"
                )
            )
        ):
            raise ServiceError("run_result_budget_unproved", 400)
        champion_restart_required = (
            kind == "execute" and run.get("champion_restart_required") is True
        )
        if champion_restart_required:
            # The broker has durably retired a credential for this ICP. Ignore
            # partial output even if the model swallowed its provider error.
            terminal_status = "provider_error"
            run_result = dict(run_result, terminal_status=terminal_status)
        if kind == "score" and terminal_status not in contracts.SCORE_TERMINAL_CAUSES:
            raise ServiceError("run_result_cause_kind_mismatch", 400)
        if kind == "execute" and terminal_status in ("judge_error", "judge_timeout"):
            raise ServiceError("run_result_cause_kind_mismatch", 400)
        lease_token = self._lease_token_for_run(validated, run)
        # A retired champion credential must release the ICP immediately. Its
        # uncertain ledger entry stays eligible for the background exact-cost
        # reconciler after this attempt becomes failed.
        output_ref = ""
        judgment_evidence = None
        judgment_evidence_hash = ""
        company_judgment_evidence = None
        output_hash = ""
        completion_request_hash = ""
        company_quality_run = (
            company_judgments.enabled(round_row.get("configuration_doc") or {})
            and run.get("company_judgment_refs") is not None
        )
        if company_quality_run:
            company_judgment_evidence = []
            completion_request_hash = contracts.request_bytes_hash(validated)
        partial_quality_output = (
            kind == "score" and company_quality_run
            and terminal_status in ("judge_error", "judge_timeout", "credential_error")
            and body.get("output") is not None
        )
        partial_legacy_output = (
            kind == "score" and not company_quality_run
            and integrity.enabled(round_row.get("configuration_doc") or {})
            and terminal_status in ("judge_error", "judge_timeout", "credential_error")
            and isinstance(body.get("output"), dict) and bool(body["output"].get("terminal_breakdowns"))
        )
        if kind == "score" and (terminal_status == "accepted" or partial_quality_output or partial_legacy_output):
            try:
                output = scoring.validate_scoring_output_document(body.get("output"))
            except scoring.ScoringError:
                raise ServiceError("output_invalid", 400)
            if (output["scored_run_id"] != run.get("scored_run_id")
                or (terminal_status == "accepted" and "failure" in output)
                or (partial_quality_output and (
                    "failure" not in output
                    or not output.get("company_judgments")
                    or (terminal_status != "credential_error"
                        and output["failure"] != terminal_status)
                ))):
                raise ServiceError("output_invalid", 400)
            if terminal_status == "accepted":
                policy = contracts.validate_scorer_policy(
                    round_row["configuration_doc"]["scorer_policy"])
                executed = self._store.get_run(str(run["scored_run_id"]))
                if executed is None or not executed.get("output_ref"):
                    raise ServiceError("scored_run_missing", 500)
                companies = validate_output_document(
                    json.loads(self._objects.get_bounded(executed["output_ref"], MAX_OUTPUT_BYTES)),
                    expected_schema_version=contact_policy.output_schema(
                        round_row["configuration_doc"]
                    ),
                )["companies"]
                try:
                    scoring.validate_breakdowns_for_item(
                        output["breakdowns"], icp=self.evaluation_icps(round_id)[int(run["icp_position"])],
                        companies=companies,
                        max_scored_companies=int(policy["max_scored_companies"]),
                        integrity_policy=contact_policy.integrity_adapter(
                            policy["scoring_adapter_version"]),
                        contacts_required=contact_policy.enabled(
                            round_row["configuration_doc"]
                        ),
                        company_quality=quality_policy.enabled(round_row["configuration_doc"]),
                    )
                except scoring.ScoringError as exc:
                    raise ServiceError("score_invalid", 400) from exc
            if partial_legacy_output:
                try:
                    document, execution_hash = self._legacy_partial_input(run, round_row)
                    if ("failure" not in output or "company_judgments" in output
                        or (terminal_status != "credential_error" and output["failure"] != terminal_status)):
                        raise scoring.ScoringError("legacy partial output is invalid")
                    output["terminal_breakdowns"] = scoring.validate_terminal_breakdowns_for_work_item(
                        output["terminal_breakdowns"], icp=document["icp"], companies=document["companies"], policy=document["scorer_policy"],
                    )
                    run_result["legacy_partial_checkpoint"] = self._legacy_partial_receipt(
                        run, round_row, document, execution_hash, output, contracts.request_bytes_hash(validated),
                    )
                except (scoring.ScoringError, ValueError, OutputInvalid) as exc:
                    raise ServiceError("legacy_checkpoint_output_invalid", 400) from exc
            output_ref = "arena/%s/scores/items/%s.json" % (round_id, run_id)
            output_bytes = contracts.canonical_json(output).encode("utf-8")
            self._objects.put(output_ref, output_bytes)
            output_hash = contracts.document_hash(output)
            if company_quality_run:
                context = (run.get("claim_response") or {}).get(
                    "company_judgment_cache"
                ) or {}
                try:
                    submission = self._store.get_submission(
                        str(run["submission_id"])
                    ) or {}
                    validated_context = self._validated_company_judgment_lease(
                        context,
                        recipient_miner_hotkey=str(
                            run.get("miner_hotkey") or ""
                        ),
                        baseline=bool(submission.get("is_king")),
                    )
                    new_judgments = (
                        company_judgments.validate_new_company_judgments(
                            output.get("company_judgments") or [],
                            lease_context=validated_context,
                            require_complete=not partial_quality_output and quality_policy.enabled(round_row["configuration_doc"]),
                        )
                    )
                    if not partial_quality_output:
                        new_by_key = {
                            item["cache_key"]: item["raw_judgment"]
                            for item in new_judgments
                        }
                        hit_by_key = {
                            item["cache_key"]: item["evidence_doc"]["raw_judgment"]
                            for item in validated_context["hits"]
                        }
                        lease_rows = sorted(
                            validated_context["hits"]
                            + validated_context["misses"],
                            key=lambda item: item["company_index"],
                        )
                        raw_rows = [
                            (
                                hit_by_key[item["cache_key"]]
                                if item["cache_key"] in hit_by_key
                                else new_by_key.get(item["cache_key"])
                            )
                            for item in lease_rows
                        ]
                        consumed_indexes: set[int] = set()
                        expected_breakdowns = scoring.project_company_judgments(
                            companies[:len(lease_rows)],
                            raw_rows,
                            contacts_required=contact_policy.enabled(round_row["configuration_doc"]),
                            company_quality=quality_policy.enabled(round_row["configuration_doc"]),
                            scored_indexes=scoring.verify.bucket_skip(
                                self.evaluation_icps(round_id)[int(run["icp_position"])], companies,
                                max_scored_companies=int(policy["max_scored_companies"]),
                            )[0],
                            consumed_indexes=consumed_indexes,
                        )
                        if output["breakdowns"] != expected_breakdowns:
                            raise company_judgments.CompanyJudgmentError(
                                "company judgment destination context mismatch"
                            )
                        if not quality_policy.enabled(round_row["configuration_doc"]):
                            # Never trust a runner's supplied authority metadata.
                            run_result["company_judgment_consumed_indexes"] = sorted(consumed_indexes)
                    refs = [
                        company_judgments.validate_company_ref(ref)
                        for ref in run.get("company_judgment_refs") or []
                    ]
                    for new_judgment in new_judgments:
                        company_ref = next(
                            (
                                ref for ref in refs
                                if ref["cache_key"]
                                == new_judgment["cache_key"]
                                and ref["company_index"]
                                == new_judgment["company_index"]
                            ),
                            None,
                        )
                        if company_ref is None:
                            raise company_judgments.CompanyJudgmentError(
                                "company judgment ref is missing"
                            )
                        evidence = company_judgments.build_evidence_snapshot(
                            new_judgment=new_judgment,
                            company_ref=company_ref,
                            source_score_run_id=run_id,
                            source_scored_run_id=str(run["scored_run_id"]),
                            source_output_ref=output_ref,
                            source_output_hash=output_hash,
                            source_runner_hotkey=str(validated["hotkey"]),
                            source_claim_request_id=str(
                                run.get("claim_request_id") or ""
                            ),
                            source_claim_request_hash=str(
                                run.get("claim_request_hash") or ""
                            ),
                            source_lease_generation=int(
                                run.get("lease_generation") or 0
                            ),
                            source_completion_request_hash=(
                                completion_request_hash
                            ),
                            runner_authority_exclusions=(
                                self._store.signal_authority_exclusions(run_id)
                                if (round_row.get("configuration_doc") or {}).get("scorer_policy", {}).get("signal_judgment_policy") == signal_judgments.POLICY
                                else (run.get("claim_response") or {}).get("runner_authority_exclusions")
                            ),
                        )
                        company_judgment_evidence.append({
                            "cache_key": evidence["cache_key"],
                            "company_input_hash": evidence[
                                "company_input_hash"
                            ],
                            "authority_slot": evidence["authority_slot"],
                            "evidence_hash": contracts.document_hash(evidence),
                            "evidence_doc": evidence,
                        })
                except (company_judgments.CompanyJudgmentError, scoring.ScoringError) as exc:
                    raise ServiceError(
                        "company_judgment_output_invalid", 400
                    ) from exc
            elif run.get("judgment_cache_key") and not partial_legacy_output:
                scope = dict(run.get("judgment_scope_doc") or {})
                scope["cache_key"] = str(run["judgment_cache_key"])
                try:
                    judgment_evidence = judgment_cache.build_evidence_snapshot(
                        output=output,
                        cache_scope=scope,
                        source_score_run_id=run_id,
                        source_scored_run_id=str(run["scored_run_id"]),
                        source_output_ref=output_ref,
                        source_runner_hotkey=str(validated["hotkey"]),
                        runner_authority_exclusions=(
                            run.get("claim_response") or {}
                        ).get("runner_authority_exclusions"),
                    )
                except judgment_cache.JudgmentCacheError as exc:
                    raise ServiceError("judgment_cache_invalid", 500) from exc
                judgment_evidence_hash = contracts.document_hash(judgment_evidence)
        elif terminal_status == "accepted":
            try:
                output = validate_output_document(
                    body.get("output"),
                    expected_schema_version=contact_policy.output_schema(round_row["configuration_doc"]),
                    require_intent_dates=not integrity.enabled(
                        round_row["configuration_doc"]
                    ),
                )
            except OutputInvalid:
                raise ServiceError("output_invalid", 400)
            output_ref = "arena/%s/outputs/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        result = self._store.complete_attempt(
            run_id=run_id, lease_token_hash=hash_lease_token(lease_token), result=run_result, terminal_cause=terminal_status,
            output_ref=output_ref, judgment_evidence=judgment_evidence,
            judgment_evidence_hash=judgment_evidence_hash,
            company_judgment_evidence=company_judgment_evidence,
            output_hash=output_hash if company_quality_run else "",
            completion_request_hash=completion_request_hash,
        )
        # The completion RPC can also report a stale lease, open accounting,
        # or an idempotent replay. Only a newly committed terminal contributes
        # one run outcome to telemetry.
        expected_status = "accepted" if terminal_status == "accepted" else "failed"
        if result.get("status") == expected_status and result.get("idempotent") is False:
            telemetry.record_run(
                kind, terminal_status, run_identity=_telemetry_run_identity(run)
            )
        return result

    def _lease_token_for_run(self, validated: Mapping[str, Any], run: Mapping[str, Any]) -> str:
        token = validated["body"].get("lease_token")
        if not isinstance(token, str) or hash_lease_token(token) != run.get("lease_token_hash"):
            raise ServiceError("lease_token_invalid", 403)
        return token

    # -- daily driver (section 14.4) -------------------------------------------

    def advance_round(self, round_id: str) -> Dict[str, Any]:
        """One idempotent compare-and-set step for the round's current state."""

        # Every driver step may change the round's status; the runner-facing
        # cache must never serve a row from before this process's own transition.
        self._invalidate_hot_round()
        try:
            row = self._round(round_id)
            if row["status"] not in ("open",) + TERMINAL_STATUSES:
                # A bounded billing batch runs outside service locks. Claims
                # for other submissions stay available while charges post.
                with telemetry.stage("advance_billing_read") as observed:
                    deepline_billing = self._reconcile_active_deepline_cost(round_id)
                    billing = self._reconcile_openrouter_cost(round_id)
                    observed.count = int(deepline_billing.get("checked") or 0) + int(
                        billing.get("status") != "none"
                    )
                if (
                    billing["status"] not in ("none", "settled")
                    and billing["run_status"] == "leased"
                ):
                    try:
                        lease_active = self.now() < _parse_iso(
                            billing["lease_expires_at"]
                        )
                    except (TypeError, ValueError):
                        lease_active = False
                    if lease_active and not (
                        row["status"] in ("stage1", "stage2")
                        and row.get("configuration_doc", {}).get("execution_sequence_policy")
                            == contracts.BASELINE_SCORED_FIRST_POLICY
                        and self.now() >= _parse_iso(
                            row["configuration_doc"]["schedule"][
                                "stage_%s_close" % row["status"][-1]
                            ]
                        )
                    ):
                        # At the execution cutoff, normal SQL closure owns the
                        # fixed drain bound even if billing renews this lease.
                        # Keep this live billing lease open without starving
                        # unrelated overdue attempts of their normal retry.
                        # The expiry RPC preserves live cost-bearing leases;
                        # an operator hold must still stop round mutations.
                        if (
                            row["status"] in (
                                "stage1", "stage2",
                                "stage1_scoring", "stage2_scoring",
                            )
                            and not self._store.operator_hold_active()
                        ):
                            with telemetry.stage("advance_expire_leases"):
                                self._store.expire_leases(round_id)
                        return {
                            "status": "retry",
                            "round_status": row["status"],
                            "reason": "provider_billing_pending",
                        }
            if row["status"] == "open":
                # At cutoff, reject source uploads that were not finalized
                # before participant freeze.
                final = self.now() >= _parse_iso(row["configuration_doc"]["schedule"]["submission_cutoff"])
                with telemetry.stage("advance_admission") as observed:
                    admission = self.admit_uploaded_submissions(round_id, final=final)
                    observed.count = int(admission.get("remaining") or 0)
                if final and int(admission.get("remaining") or 0) > 0:
                    return {
                        "status": "retry",
                        "round_status": "open",
                        "remaining_admissions": int(admission["remaining"]),
                    }
            # Freeze the open round's bank, scorer, and participants at cutoff
            # even during a claim hold. Subsequent execution/scoring progress
            # waits; migration 371 also guards old or racing driver processes.
            if row["status"] not in ("open",) + TERMINAL_STATUSES and self._store.operator_hold_active():
                return {"status": "paused", "round_status": row["status"]}
            return self._advance_round_locked(round_id)
        finally:
            self._invalidate_hot_round()

    def _invalidate_hot_round(self) -> None:
        with self._hot_round_lock:
            self._hot_rounds.clear()

    def _advance_round_locked(self, round_id: str) -> Dict[str, Any]:
        # The wait for the service lock is timed on its own, so a driver
        # blocked behind another thread's transition is distinguishable from
        # one doing slow work. The lock itself stays a plain `with`.
        lock_start_ns = time.time_ns()
        lock_start = time.monotonic()
        with self._lock:
            telemetry.record(
                "advance_lock_wait",
                "ok",
                duration_ms=(time.monotonic() - lock_start) * 1000.0,
                start_ns=lock_start_ns,
            )
            round_row = self._round(round_id)
            status = round_row["status"]
            schedule = round_row["configuration_doc"]["schedule"]
            now = self.now()
            if (
                status not in TERMINAL_STATUSES
                and now < _parse_iso(schedule["submission_cutoff"])
            ):
                return {"status": "waiting", "round_status": status}
            if status == "open":
                with telemetry.stage("advance_commit_benchmark"):
                    return self.commit_benchmark(round_id)
            if status == "committed":
                with telemetry.stage("advance_open_stage"):
                    return self.open_stage(round_id, 1)
            if (status in ("stage1_scoring", "stage1_judged")
                and round_row["configuration_doc"].get("execution_sequence_policy")
                    == contracts.BASELINE_SCORED_FIRST_POLICY
                and _judge_continuation_enabled(round_row)):
                # The database checks every accepted baseline execution under
                # the round lock. A pre-migration database returns stale and
                # continues its existing sequential path.
                with telemetry.stage("advance_open_stage"):
                    opened = self.open_stage(round_id, 2)
                if opened.get("status") == "ok":
                    return opened
            if status in ("stage2", "stage2_closed", "stage2_scoring"):
                self._recover_baseline(round_row)
            if status in ("stage1", "stage2"):
                stage = int(status[-1])
                with telemetry.stage("advance_expire_leases"):
                    self._store.expire_leases(round_id)
                if now >= _parse_iso(schedule["stage_%d_close" % stage]) or self.stage_is_complete(round_id, stage):
                    with telemetry.stage("advance_close_stage"):
                        return self.close_stage(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_closed", "stage2_closed"):
                stage = int(status[5])
                if not round_row.get("stage%d_scoring_plan_doc" % stage):
                    with telemetry.stage("advance_commit_scoring_plan"):
                        return self.commit_scoring_plan(round_id, stage)
                with telemetry.stage("advance_open_scoring"):
                    return self.open_scoring(round_id, stage)
            if status in ("stage1_scoring", "stage2_scoring"):
                stage = int(status[5])
                with telemetry.stage("advance_expire_leases"):
                    self._store.expire_leases(round_id)
                with telemetry.stage("advance_list_scoring_runs") as observed:
                    scoring_runs = self._scoring_runs_for_stage(round_id, stage)
                    observed.count = len(scoring_runs)
                window = schedule["stage_1_scoring_close" if stage == 1 else "final_scoring_close"]
                if stage == 2 and _judge_continuation_enabled(round_row) and now < _parse_iso(window):
                    # Provider settlement or schema activation can delay the
                    # queue tick. Keep eligible failed work recoverable until
                    # the frozen execution claim boundary.
                    original = self._store.list_runs(round_id, stage=1, kind="execute")
                    if any(run.get("status") == "failed" and run["attempt"] < 2147483647
                        and self._is_baseline_recovery(
                        {**run, "attempt": run["attempt"] + 1,
                         "run_id": run["assignment_id"] + ":" + str(run["attempt"] + 1)},
                        original, round_row) and not any(
                            later.get("assignment_id") == run.get("assignment_id")
                            and later.get("attempt", 0) > run["attempt"] for later in original)
                        for run in original):
                        return {"status": "waiting", "round_status": status,
                                "reason": "baseline_infrastructure_recovery"}
                if now >= _parse_iso(window) or all(
                    run["status"] in ("accepted", "failed")
                    for run in scoring_runs
                ):
                    with telemetry.stage("advance_close_scoring"):
                        return self.close_scoring(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_judged", "stage2_judged"):
                stage = int(status[5])
                window = schedule["stage_1_scoring_close" if stage == 1 else "final_scoring_close"]
                with telemetry.stage("advance_expire_leases"):
                    self._store.expire_leases(round_id)
                try:
                    with telemetry.stage("advance_score_stage"):
                        return self.score_stage(round_id, stage)
                except scoring.ScoringError:
                    if now >= _parse_iso(window) + timedelta(hours=2):
                        return self._store.cancel_round(round_id, CANCEL_REASONS["scoring"])
                    return {"status": "retry", "round_status": status}
            if status == "stage1_scored":
                with telemetry.stage("advance_open_stage"):
                    return self.open_stage(round_id, 2)
            if status == "scored":
                if _judge_continuation_enabled(round_row):
                    with telemetry.stage("advance_expire_leases"):
                        self._store.expire_leases(round_id)
                if self._frozen_unfinished_judgment(round_row):
                    return {"status": "waiting", "round_status": status,
                            "reason": "frozen_judgment_incomplete"}
                try:
                    with telemetry.stage("advance_publish"):
                        return self.publish(round_id)
                except ServiceError as exc:
                    if exc.code == "publication_sanitizer_failed" and now >= _parse_iso(schedule["publication_deadline"]) + timedelta(hours=14):
                        return self._store.cancel_round(round_id, CANCEL_REASONS["publication"])
                    raise
            return {"status": "terminal", "round_status": status}

    def cancel(self, round_id: str, reason: str) -> Dict[str, Any]:
        if reason not in CANCEL_REASONS.values():
            raise ServiceError("cancel_reason_invalid", 400)
        self._round(round_id)
        self._invalidate_hot_round()
        return self._store.cancel_round(round_id, reason)

    # -- public reads (section 14.1) -------------------------------------------

    def public_history(self, *, cursor: Optional[str] = None, limit: int = 25,
                       day: Optional[str] = None, hotkey: str = "") -> Dict[str, Any]:
        return public_dashboard.history_snapshot(self, cursor=cursor, limit=limit, day=day, hotkey=hotkey)

    def public_competition(self) -> Dict[str, Any]:
        return public_dashboard.competition_snapshot(self)

    def completed_submission_scores(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Read completed model results without publishing or advancing a round.

        Use the same accepted judgments, arithmetic and cost eligibility as
        final publication. Failed/retrying judges wait for the normal closed
        stage to record their outcome. Never turn unfinished work into a zero.
        A bounded background refresh keeps public reads responsive even when
        many models need object-store and judgment verification. Completed
        models appear as each check finishes; costs are rechecked on refresh.
        Cancelled-round values stay provisional in the public projection.
        """
        configuration = row.get("configuration_doc") or {}
        round_id = str(row["round_id"])
        generation = int(row.get("status_generation") or 0)
        status = row.get("status")
        # Compact and full reads share these fields. Normal closure preserves
        # the private configuration and scoring plans under the DB write-once
        # guards. Cancellation may retain the same verified measurements;
        # a changed scoring floor or configuration never reuses them.
        identity = contracts.canonical_json({
            "participants": row.get("participants") or [],
            "scoring_generation_floor": self._score_generation_floor(row),
            "configuration": {key: configuration[key] for key in PUBLIC_CONFIGURATION_FIELDS
                              if key in configuration},
        })
        worker = None
        with self._completed_scores_lock:
            cached = self._completed_scores_cache.get(round_id)
            # A delayed request must not roll back a newer entry or evict its
            # carried scores, including when the old row is no longer eligible.
            if cached is not None and cached["generation"] > generation:
                return {}
            if (status not in {
                "stage1_scoring", "stage1_judged", "stage1_scored", "stage2",
                "stage2_closed", "stage2_scoring", "stage2_judged", "scored",
                "cancelled",
            } or configuration.get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
                or configuration.get("sourcing_cost_eligibility_policy") not in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES):
                self._completed_scores_cache.pop(round_id, None)
                return {}
            if (cached is None or cached["generation"] != generation
                    or cached["status"] != status or cached["identity"] != identity):
                carry = bool(
                    cached is not None and generation == cached["generation"] + 1
                    and cached["identity"] == identity
                    and (cached["status"], status) in {
                        ("stage1_scoring", "stage1_judged"),
                        ("stage1_judged", "stage1_scored"),
                        ("stage2_scoring", "stage2_judged"),
                        ("stage2_judged", "scored"),
                        ("stage1_scoring", "cancelled"),
                        ("stage1_judged", "cancelled"),
                        ("stage1_scored", "cancelled"),
                        ("stage2", "cancelled"),
                        ("stage2_closed", "cancelled"),
                        ("stage2_scoring", "cancelled"),
                        ("stage2_judged", "cancelled"),
                        ("scored", "cancelled"),
                    }
                )
                cached = {"generation": generation, "status": status, "identity": identity,
                          "completed": dict(cached["completed"]) if carry else {},
                          "refreshed_at": 0.0, "refreshing": False}
                self._completed_scores_cache[round_id] = cached
                # Public summaries enumerate a bounded recent history.
                if len(self._completed_scores_cache) > 8:
                    oldest = next(iter(self._completed_scores_cache))
                    if oldest != round_id:
                        self._completed_scores_cache.pop(oldest)
            if (not cached["refreshing"]
                    and (not cached["refreshed_at"] or time.monotonic() - cached["refreshed_at"] >= 15)
                    and self._completed_scores_refresh_slots.acquire(blocking=False)):
                cached["refreshing"] = True
                worker = threading.Thread(
                    target=self._refresh_completed_submission_scores,
                    args=(dict(row), cached), daemon=True,
                    name="arena-completed-scores",
                )
            result = dict(cached["completed"])
        if worker is not None:
            try:
                worker.start()
            except Exception:
                with self._completed_scores_lock:
                    cached["refreshing"] = False
                self._completed_scores_refresh_slots.release()
                raise
        return result

    def _refresh_completed_submission_scores(self, row: Mapping[str, Any], entry: Dict[str, Any]) -> None:
        round_id = str(row["round_id"])

        def current() -> bool:
            with self._completed_scores_lock:
                return self._completed_scores_cache.get(round_id) is entry

        def record(submission_id: str, result: Optional[Dict[str, Any]]) -> None:
            with self._completed_scores_lock:
                if self._completed_scores_cache.get(round_id) is entry:
                    if result is None:
                        entry["completed"].pop(submission_id, None)
                    else:
                        entry["completed"][submission_id] = result

        results = {}
        generation_matches = True
        try:
            # The compact public SELECT has no private scoring plans. Fetch
            # them only after a cache miss has acquired a refresh slot.
            if ("stage1_scoring_plan_doc" not in row
                    and "stage2_scoring_plan_doc" not in row):
                row = self._round(round_id)
                generation_matches = int(row.get("status_generation") or 0) == entry["generation"]
            if not generation_matches or not current():
                return
            results = self._read_completed_submission_scores(row, on_completed=record, should_continue=current)
        except Exception as exc:
            logger.warning("Completed model score refresh unavailable: round=%s type=%s", round_id, type(exc).__name__)
            results = {}
        finally:
            with self._completed_scores_lock:
                if self._completed_scores_cache.get(round_id) is entry:
                    entry["completed"] = results
                    entry["refreshed_at"] = time.monotonic() if generation_matches else 0.0
                    entry["refreshing"] = False
            self._completed_scores_refresh_slots.release()

    def _read_completed_submission_scores(
        self, row: Mapping[str, Any], *,
        on_completed: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        round_id = str(row["round_id"])
        configuration = row["configuration_doc"]
        positions = set(range(contracts.benchmark_icp_count(configuration)))
        execution_runs = self._store.list_runs(round_id, kind="execute")
        judgment_runs = self._store.list_runs(round_id, kind="score")
        judges = self._select_scoring_outputs(judgment_runs, round_row=row)
        loaded_rows = {str(run["run_id"]): run for run in execution_runs + judgment_runs}
        completed = {}
        icps = None
        participants = []
        for participant in row.get("participants") or []:
            submission_id = str(participant["submission_id"])
            stage = 1 if participant.get("is_king") else 2
            selected = {}
            for run in execution_runs:
                if run["submission_id"] != submission_id or run["stage"] != stage:
                    continue
                position = int(run["icp_position"])
                current = selected.get(position)
                if current is None or (run["status"] == "accepted", run["attempt"]) > (current["status"] == "accepted", current["attempt"]):
                    selected[position] = run
            stored = set(selected) == positions and all(
                run.get("per_icp_score") is not None for run in selected.values()
            )
            participants.append((participant, stage, selected, stored))
        # Release persisted scores before assembling unpublished evidence. Sort
        # stably so peers in each group retain their configured order.
        participants.sort(key=lambda item: not item[3])
        for participant, stage, selected, stored in participants:
            if should_continue is not None and not should_continue():
                break
            submission_id = str(participant["submission_id"])
            plan = (self._baseline_recovery_plan(row, execution_runs, judgment_runs)
                    if stage == 1 and configuration.get("execution_sequence_policy") == contracts.BASELINE_SCORED_FIRST_POLICY
                    else row.get("stage%d_scoring_plan_doc" % stage))
            if not plan:
                if on_completed is not None:
                    on_completed(submission_id, None)
                continue
            if set(selected) != positions or any(run["status"] not in {"accepted", "failed"} for run in selected.values()):
                if on_completed is not None:
                    on_completed(submission_id, None)
                continue
            items = [item for item in plan["work_items"] if item["submission_id"] == submission_id]
            zeros = [item for item in plan["zero_rows"] if item["submission_id"] == submission_id]
            closed = row["status"] in (
                {"stage1_judged", "stage1_scored", "stage2", "stage2_closed", "stage2_scoring", "stage2_judged", "scored"}
                if stage == 1 else {"stage2_judged", "scored"}
            )
            unfinished = [judges.get(item["scored_run_id"], {}) for item in items
                          if judges.get(item["scored_run_id"], {}).get("status") != "accepted"]
            if unfinished and (not stored or not closed or any(
                judge.get("status") != "failed"
                or judge.get("terminal_cause") not in contracts.TERMINAL_CAUSES
                or judge.get("terminal_cause") == "accepted" for judge in unfinished
            )):
                if on_completed is not None:
                    on_completed(submission_id, None)
                continue
            try:
                runs = list(selected.values())
                for item in items:
                    run_id = item["scored_run_id"]
                    run = selected[int(item["icp_position"])]
                    if run["run_id"] != run_id or (
                        judges[run_id]["status"] == "accepted"
                        and not self._same_scoring_membership(judges[run_id], run, round_id=round_id)
                    ):
                        raise scoring.ScoringError("completed score membership differs")
                # Proof reuse belongs to this model in this refresh only.
                # Keep pending peers out of the bounded raw-read maps.
                evidence_reads = _PublicResultEvidenceReads()
                self._prefetch_public_result_evidence(
                    evidence_reads, round_id=round_id, execution_runs=runs,
                    score_runs=[judges[item["scored_run_id"]] for item in items],
                    public_run_ids={str(run["run_id"]) for run in runs},
                    source_rows=loaded_rows, round_context=row,
                    include_signal_attribution=False,
                )
                if not stored:
                    if icps is None:
                        icps = self.evaluation_icps(round_id)
                    outputs = {}
                    breakdowns = {}
                    for item in items:
                        run_id = item["scored_run_id"]
                        run = selected[int(item["icp_position"])]
                        output = validate_output_document(json.loads(evidence_reads.object(self._objects, run["output_ref"], MAX_OUTPUT_BYTES).decode("utf-8")))
                        outputs[run_id] = output["companies"]
                        breakdowns[run_id] = self._verified_breakdowns(
                            judges[run_id], icp=icps[int(item["icp_position"])],
                            companies=output["companies"], policy=configuration["scorer_policy"],
                            evidence_reads=evidence_reads, round_context=row,
                        )
                    scores = scoring.build_stage_scores(
                        plan={**plan, "work_items": items, "zero_rows": zeros,
                              "incomplete_rows": [item for item in plan.get("incomplete_rows") or []
                                                  if item["submission_id"] == submission_id]},
                        policy=configuration["scorer_policy"],
                        icps_by_position=dict(enumerate(icps)), outputs_by_run=outputs,
                        breakdowns_by_item=breakdowns, configuration=configuration,
                    )
                    records = {r["run_id"]: r for r in scoring.run_scores_for_store(scores, runs)}
                    if set(records) != {run["run_id"] for run in runs}:
                        if on_completed is not None:
                            on_completed(submission_id, None)
                        continue
                    # Match PostgreSQL NUMERIC(12, 6), which final publication
                    # reads back. Preserve any already recorded per-ICP value.
                    for record in records.values():
                        record["per_icp_score"] = float(Decimal(str(record["per_icp_score"])).quantize(
                            Decimal("0.000001"), rounding=ROUND_HALF_UP,
                        ))
                    runs = [run if run.get("per_icp_score") is not None else
                            {**run, **records[run["run_id"]]} for run in runs]
                eligibility = self._submission_cost_eligibility(
                    row, submission_id, runs, positions=sorted(positions),
                    evidence_reads=evidence_reads,
                )
                if not eligibility["eligible"]:
                    if on_completed is not None:
                        on_completed(submission_id, None)
                    continue  # Billing in flight/uncertain is not a completed score.
                eligible_positions = {int(item["icp_position"]) for item in eligibility["cost_summary"]["per_icp"] if item["eligible"]}
                value = verify.stage_score([
                    float(run["per_icp_score"]) if int(run["icp_position"]) in eligible_positions else 0.0
                    for run in sorted(runs, key=lambda item: item["icp_position"])
                ], len(positions))
                if not any(run.get("terminal_cause") == "accepted" for run in runs):
                    if on_completed is not None:
                        on_completed(submission_id, None)
                    continue
                completed[submission_id] = {
                    "submission_id": submission_id, "final_score": value,
                    **eligibility, "execution_runs": runs,
                }
                if on_completed is not None:
                    on_completed(submission_id, completed[submission_id])
            except Exception as exc:
                # Invalid evidence or a transient object-store failure for one
                # model must not hide the rest of the competition snapshot.
                logger.warning("Completed model score unavailable: round=%s submission=%s type=%s", round_id, submission_id, type(exc).__name__)
                if on_completed is not None:
                    on_completed(submission_id, None)
        return completed

    def public_submissions(self, round_id: str) -> Dict[str, Any]:
        return public_dashboard.submissions_snapshot(self, round_id)

    def public_submission_code(self, submission_id: str) -> Dict[str, Any]:
        self._require_identifier(submission_id, "submission")
        submission = self._store.get_submission(submission_id)
        if submission is None:
            raise ServiceError("submission_missing", 404)
        row = self._round(str(submission.get("round_id") or ""))
        try:
            return source_disclosure.public_source_code(
                self._objects, submission, self.now(), round_row=row
            )
        except source_disclosure.SourceDisclosureError as exc:
            raise ServiceError(exc.code, exc.status) from exc

    def public_current(self) -> Dict[str, Any]:
        active = self.active_rounds(public_only=True)
        current = active[-1] if active else None
        now = self.now()
        open_round = next((row for row in reversed(active)
                           if self._round_accepts_submissions(row, now)), None)
        running = [row for row in active if row["status"] != "open"]
        published = self.latest_published_round(public_only=True)
        published_round = (
            {
                "round_id": published["round_id"],
                "status": published["status"],
                "published_at": published.get("published_at"),
            }
            if published is not None
            else None
        )
        epoch = None
        try:
            epoch = int(self._config.chain.current_settlement_epoch())
        except Exception:
            epoch = None
        eligibility = None
        week = None
        governing = None
        if epoch is not None:
            governing = self.public_reward_basis(epoch, public_only=True)
            if governing is None:
                eligibility = False
            else:
                # A previously accepted epoch state is immutable. Report its
                # signed eligibility rule during a rolling upgrade, rather
                # than the rule that a fresh state would receive.
                if "slot_policy" in governing:
                    network_name, netuid = self._chain_scope()
                    existing = self._store.get_weight_state(network_name, netuid, epoch)
                    if existing is not None:
                        # This is the informational projection of trusted
                        # storage, as for the governing basis above. Signature
                        # verification remains mandatory in public_weight_state.
                        state = validate_accepted_weight_state(existing.get("state_doc"))
                        if not rewards.reward_basis_matches_policy_projection(state["reward_basis"], governing, epoch):
                            raise ServiceError("accepted_weight_state_reward_conflict", 409)
                        governing = state["reward_basis"]
                        eligibility = rewards.epoch_eligible(
                            governing, epoch,
                            until_replaced=state.get("slot_eligibility") == SLOT_ELIGIBILITY_UNTIL_REPLACED,
                        )
                    else:
                        # No accepted state has chosen a rule for this epoch.
                        eligibility = None
                else:
                    eligibility = rewards.epoch_eligible(governing, epoch)
                if eligibility and "slot_policy" not in governing:
                    week = rewards.reward_week_index(epoch, int(governing["king_start_epoch"]))
        elif self._config.mode == "live":
            network_name, netuid = self._chain_scope()
            rows = self._store.published_reward_bases(
                mode="live", network_name=network_name, netuid=netuid, limit=200,
                public_only=True,
            )
            rows = [self._public_current_round(row) for row in rows]
            bases = self._usable_reward_bases(rows)
            if bases:
                governing = max(
                    bases, key=lambda basis: int(basis["effective_reward_epoch"])
                )
        return {
            "mode": self._config.mode,
            "round": dict(current) if current else None,
            # Rounds overlap: miners submit to the open round while runners work the running ones.
            "open_round": dict(open_round) if open_round else None,
            "running_rounds": [dict(row) for row in running],
            "published_round": published_round,
            "king": {"hotkey": governing.get("king_hotkey"), "outcome": governing.get("king_outcome"), "round_id": governing.get("round_id"), "king_start_epoch": governing.get("king_start_epoch")} if governing else None,
            "reward_week_index": week,
            "epoch_eligible": eligibility,
            "reward_slot_policy": governing.get("slot_policy") if governing else None,
            "reward_slots": governing.get("reward_slots") if governing else None,
            "current_epoch": epoch,
        }

    def public_reward_basis(self, epoch: int, *, public_only: bool = False) -> Optional[Dict[str, Any]]:
        if self._config.mode != "live":
            return None
        network_name, netuid = self._chain_scope()
        rows = self._store.published_reward_bases(
            mode="live", network_name=network_name, netuid=netuid, limit=200,
            **({"public_only": True} if public_only else {}),
        )
        if public_only:
            rows = [self._public_current_round(row) for row in rows]
        return rewards.governing_reward_basis(
            self._usable_reward_bases(rows), int(epoch)
        )

    def _project_reward_basis(self, basis: Mapping[str, Any], epoch: int) -> Dict[str, Any]:
        """Sign new policy over verified source achievements, without history writes."""
        if (int(epoch) < rewards.REWARD_SLOT_POLICY_OCT23_EFFECTIVE_EPOCH
                or basis.get("schema_version") != contracts.REWARD_BASIS_V2_SCHEMA_VERSION):
            return basis
        policy = rewards.reward_slot_policy_document(effective_reward_epoch=int(epoch))
        if basis["slot_policy"] == policy:
            return basis
        # The old signed document is immutable provenance. Verify it before
        # signing a policy projection, so a changed database row cannot acquire
        # reward authority merely by carrying a valid content hash.
        signer = self._reward_signer()
        try:
            signing.verify_document_signature(
                basis, public_key_der=signer.public_key_der,
                hash_field="reward_basis_hash",
                expected_public_key_hash=signer.public_key_hash,
            )
        except (TypeError, ValueError) as exc:
            raise ServiceError("reward_history_signature_invalid", 500) from exc
        try:
            projected = rewards.reward_basis_policy_projection_document(basis, policy)
        except (TypeError, ValueError) as exc:
            raise ServiceError("reward_policy_projection_invalid", 503) from exc
        return self._sign(projected, "reward_basis_hash")

    def _validated_existing_weight_state(
        self, existing: Mapping[str, Any], basis: Mapping[str, Any], *,
        network: str, netuid: int, epoch: int,
        expected_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            state = validate_accepted_weight_state(existing.get("state_doc"))
            signer = self._reward_signer()
            verify_accepted_weight_state_signature(
                state, public_key_der=signer.public_key_der,
                expected_public_key_hash=signer.public_key_hash,
            )
        except (TypeError, ValueError) as exc:
            raise ServiceError("accepted_weight_state_invalid", 500) from exc
        if (state["network"], state["netuid"], state["epoch"]) != (network, netuid, epoch):
            raise ServiceError("accepted_weight_state_scope_conflict", 409)
        if not rewards.reward_basis_matches_policy_projection(state["reward_basis"], basis, epoch):
            raise ServiceError("accepted_weight_state_reward_conflict", 409)
        if expected_state is not None:
            # ECDSA bytes may differ; every other accepted body field must match.
            def comparable(value):
                body = accepted_weight_state_body(value)
                body["reward_basis"] = {
                    key: item for key, item in body["reward_basis"].items() if key != "signature"
                }
                return body
            if comparable(state) != comparable(expected_state):
                raise ServiceError("accepted_weight_state_conflict", 409)
        return state

    def public_weight_state(self, epoch: int) -> Dict[str, Any]:
        """Return or atomically publish one accepted state for an Arena epoch."""

        if self._config.mode != "live":
            return {"state": None, "lookup_ok": True}
        network, netuid = self._chain_scope()
        requested_epoch = int(epoch)
        basis = self.public_reward_basis(requested_epoch)
        if basis is None:
            return {"state": None, "lookup_ok": True}
        existing = self._store.get_weight_state(network, netuid, requested_epoch)
        if existing is not None:
            return {"state": self._validated_existing_weight_state(
                existing, basis, network=network, netuid=netuid, epoch=requested_epoch,
            ), "lookup_ok": True}
        try:
            epoch_scope = dict(self._config.chain.accepted_weight_epoch_scope())
            if int(epoch_scope.get("epoch", -1)) != requested_epoch:
                raise ServiceError("accepted_weight_epoch_not_current", 409)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("accepted_weight_epoch_scope_unavailable", 503) from exc
        basis = self._project_reward_basis(basis, requested_epoch)
        try:
            state = weight_state.build_accepted_weight_state(
                self._reward_signer(), network=network,
                genesis_hash=epoch_scope["genesis_hash"], netuid=netuid,
                epoch=requested_epoch,
                valid_from_block=epoch_scope["valid_from_block"],
                valid_until_block=epoch_scope["valid_until_block"],
                reward_basis=basis,
                burn_hotkey=self._config.accepted_burn_hotkey,
                issued_at=str(basis["published_at"]),
            )
            stored = self._store.publish_weight_state(
                network, netuid, requested_epoch, state["state_hash"], state
            )
        except ArenaStoreError as exc:
            # Independently signed projections have the same basis body hash,
            # but their nested ECDSA signatures can differ. Keep the first
            # accepted bytes, and accept only a fully verified equivalent basis.
            existing = self._store.get_weight_state(network, netuid, requested_epoch)
            if existing is not None:
                return {"state": self._validated_existing_weight_state(
                    existing, basis, network=network, netuid=netuid, epoch=requested_epoch,
                    expected_state=state,
                ), "lookup_ok": True}
            raise ServiceError("accepted_weight_state_conflict", 409) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError("accepted_weight_state_conflict", 409) from exc
        return {"state": validate_accepted_weight_state(stored["state"]), "lookup_ok": True}

    def handle_validator_events(self, envelope: Any) -> Dict[str, Any]:
        """Accept private observations from a finalized permit holder without a lease."""
        if len(contracts.canonical_json(envelope).encode("utf-8")) > validator_events.MAX_REQUEST_BYTES:
            raise ServiceError("validator_events_too_large", 413)
        validated = self.validate_request(
            envelope, scope=contracts.SCOPE_VALIDATOR_EVENTS, round_id=None
        )
        network, netuid = self._chain_scope()
        body = validated["body"]
        if (validated["round_id"] != "validator-events"
                or set(body) != {"network", "netuid", "events"}
                or body.get("network") != network
                or isinstance(body.get("netuid"), bool)
                or not isinstance(body.get("netuid"), int)
                or body["netuid"] != netuid):
            raise ServiceError("validator_events_scope_invalid", 400)
        try:
            events = validator_events.validate_events(body["events"])
            for item in events:
                observed = datetime.fromisoformat(item["occurred_at"].replace("Z", "+00:00"))
                if not -300 <= (self.now() - observed).total_seconds() <= 2 * 86400:
                    raise validator_events.ValidatorEventError("validator event time invalid")
        except validator_events.ValidatorEventError:
            raise ServiceError("validator_events_invalid", 400) from None
        try:
            permitted_validator_uid(self._config.chain.metagraph(finalized=True),
                                    validated["hotkey"], netuid=netuid)
        except ValidatorIneligible as exc:
            raise ServiceError(str(exc), 403) from None
        except Exception:
            raise ServiceError("validator_snapshot_unavailable", 503) from None
        try:
            result = self._store.append_validator_events(
                validated["hotkey"], network, netuid, events,
                runtime_version.SOURCE_METADATA["validator_source_commit"],
            )
        except ArenaStoreError:
            raise ServiceError("validator_events_unavailable", 503) from None
        if result.get("status") == "owner_required":
            raise ServiceError("validator_events_owner_required", 403)
        if result.get("status") == "correlation_invalid":
            raise ServiceError("validator_events_correlation_invalid", 400)
        if result.get("status") == "rate_limited":
            raise ServiceError("validator_events_rate_limited", 429)
        if result.get("status") != "accepted":
            raise ServiceError("validator_events_unavailable", 503)
        return result

    def handle_weight_state(self, envelope: Any) -> Dict[str, Any]:
        """Authorize a finalized permit holder before returning accepted state."""

        validated = self.validate_request(
            envelope, scope=contracts.SCOPE_WEIGHT_STATE, round_id=None
        )
        network, netuid = self._chain_scope()
        body = validated["body"]
        if (
            validated["round_id"] != "weight-state"
            or set(body) != {"epoch", "network", "netuid"}
        ):
            raise ServiceError("weight_state_request_invalid", 400)
        epoch = body.get("epoch")
        requested_netuid = body.get("netuid")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or body.get("network") != network
            or isinstance(requested_netuid, bool)
            or not isinstance(requested_netuid, int)
            or requested_netuid != netuid
        ):
            raise ServiceError("weight_state_scope_mismatch", 400)
        try:
            metagraph = self._config.chain.metagraph(finalized=True)
            permitted_validator_uid(
                metagraph, validated["hotkey"], netuid=netuid
            )
        except ValidatorIneligible as exc:
            raise ServiceError(str(exc), 403) from None
        except Exception:
            raise ServiceError("validator_snapshot_unavailable", 503) from None
        return self.public_weight_state(epoch)

    def record_chain_outcome(self, document: Any) -> Dict[str, Any]:
        """Record an authenticated validator observation, never settlement authority."""

        # Authenticate every retry before SQL compares finalized proof fields.
        # A recovered observation may have a fresh signature without changing
        # its request ID; SQL preserves the first authenticated receipt.
        validated = weight_state.validate_chain_outcome(document, now=self.now())
        network, netuid = self._chain_scope()
        if validated["network"] != network or int(validated["netuid"]) != netuid:
            raise ServiceError("chain_outcome_scope_mismatch", 409)
        if not self._config.verify_signature(
            validated["validator_hotkey"], validated["signature"],
            weight_state.chain_outcome_message(validated),
        ):
            raise ServiceError("chain_outcome_signature_invalid", 401)
        state = self._store.get_weight_state(network, netuid, int(validated["epoch"]))
        if state is None or state.get("state_hash") != validated["state_hash"]:
            raise ServiceError("chain_outcome_state_unknown", 409)
        try:
            return self._store.record_chain_outcome(
                network=network, netuid=netuid, epoch=int(validated["epoch"]),
                validator_hotkey=validated["validator_hotkey"],
                request_id=validated["request_id"],
                extrinsic_hash=validated["extrinsic_hash"], outcome_doc=validated,
            )
        except ArenaStoreError as exc:
            raise ServiceError("chain_outcome_conflict", 409) from exc

    def public_chain_outcomes(self, epoch: int) -> Dict[str, Any]:
        network, netuid = self._chain_scope()
        rows = self._store.list_chain_outcomes(network, netuid, int(epoch))
        return {"outcomes": [dict(row["outcome_doc"]) for row in rows], "lookup_ok": True}

    @staticmethod
    def _public_round_output_policy(configuration: Mapping[str, Any]) -> Dict[str, Any]:
        """Expose the output contract and replacement deadline, never ICPs."""

        cutoff = (configuration.get("schedule") or {}).get("submission_cutoff")
        return {
            "max_replacement_attempts": 1,
            "benchmark_icp_count": contracts.benchmark_icp_count(configuration),
            "promotion_margin": contracts.promotion_margin(configuration),
            **({"submission_replacement_cutoff": (_parse_iso(cutoff) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")} if cutoff else {}),
            "output_schema_version": contact_policy.output_schema(configuration),
            **({"integrity_policy": integrity.POLICY} if integrity.enabled(configuration) else {}),
            **({"contact_policy": contact_policy.POLICY} if contact_policy.enabled(configuration) else {}),
            **({"company_quality_policy": quality_policy.POLICY} if quality_policy.enabled(configuration) else {}),
            **({"company_judgment_policy": company_judgments.POLICY} if configuration.get("company_judgment_policy") else {}),
            **({"intent_details_policy": intent_details_policy.POLICY} if intent_details_policy.enabled(configuration) else {}),
            **({"reasoning_capture_policy": configuration["reasoning_capture_policy"],
                training_coverage.SETTING: training_coverage.percent_from_configuration(configuration),
                training_coverage.MISSING_REASONING_SETTING: training_coverage.missing_reasoning_percent_from_configuration(configuration)}
               if reasoning_capture.enabled(configuration) else {}),
            **({"baseline_reasoning_capture_policy": baseline_reasoning.validate(
                configuration[baseline_reasoning.FIELD]).get("reasoning_capture_policy")}
               if baseline_reasoning.FIELD in configuration else {}),
            **({"reasoning_activation_status": configuration["reasoning_activation_status"],
                "reasoning_activation_reason": configuration["reasoning_activation_reason"]}
               if configuration.get("reasoning_activation_status") == "not_ready" else {}),
        }

    def public_round(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        configuration = row.get("configuration_doc") or {}
        participants = None
        if row["status"] != "open":
            participants = [
                {
                    "submission_id": participant["submission_id"],
                    "miner_hotkey": participant["miner_hotkey"],
                    "is_baseline": bool(participant.get("is_king")),
                }
                for participant in (row.get("participants") or [])
            ]
        view = {
            "round_id": round_id,
            "status": row["status"],
            "schedule": dict(configuration.get("schedule") or {}),
            **self._public_round_output_policy(configuration),
            "participants": participants,
            "finalists": row.get("finalists") if row["status"] == "published" else None,
            "publication": row.get("publication_doc"), "king_outcome": row.get("king_outcome"), "king_hotkey": row.get("king_hotkey"),
            "effective_reward_epoch": row.get("effective_reward_epoch"), "cancel_reason": row.get("cancel_reason"),
            "final_ranking": None,
        }
        if row["status"] == "published":
            publication = row.get("publication_doc") or {}
            view.update({"final_ranking": publication.get("final_ranking"), "king_decision": publication.get("king_decision")})
        return view

    def _public_icp_disclosure(self, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if icp_disclosure.disclosure_metadata(row) is None:
            return None
        baselines = [p for p in row.get("participants") or [] if p.get("is_king") is True]
        runs = (
            self._store.list_runs(
                str(row["round_id"]),
                submission_id=str(baselines[0]["submission_id"]),
                kind="execute",
            )
            if len(baselines) == 1
            else []
        )
        return icp_disclosure.baseline_disclosure(row, runs, self.now())

    def public_benchmark(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        disclosure = self._public_icp_disclosure(row)
        if disclosure is None:
            raise ServiceError("benchmark_not_public", 403)
        icps = self.benchmark_icps(round_id)
        result = {
            "round_id": round_id,
            "icps": [
                {**(integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(row.get("configuration_doc") or {})) if integrity.enabled(row.get("configuration_doc") or {}) else icps[position]), "icp_position": position,
                 "baseline_score": disclosure["baseline_scores"].get(position)}
                for position in disclosure["public_positions"]
            ],
            "icp_set_date": disclosure["icp_set_date"],
            "public_at": disclosure["public_at"],
            "public_icp_count": contracts.benchmark_icp_count(row.get("configuration_doc")),
            "benchmark_icp_count": contracts.benchmark_icp_count(row.get("configuration_doc")),
            "private_icp_count": 0,
            "disclosure_policy": disclosure["disclosure_policy"],
        }
        return result

    @staticmethod
    def _same_scoring_membership(
        score_run: Mapping[str, Any],
        execution_run: Mapping[str, Any],
        *,
        round_id: str,
        completed_company: bool = False,
        completed_signal: bool = False,
    ) -> bool:
        """Bind one accepted judgment to its accepted execution without leaking ids."""

        score_stage = score_run.get("stage")
        execution_stage = execution_run.get("stage")
        score_position = score_run.get("icp_position")
        execution_position = execution_run.get("icp_position")
        return (
            score_run.get("kind") == "score"
            and (score_run.get("status") == "accepted"
                 or (completed_company and score_run.get("status") == "failed")
                 or (completed_signal and score_run.get("status") in ("leased", "failed")))
            and execution_run.get("kind") == "execute"
            and execution_run.get("status") == "accepted"
            and type(score_stage) is int
            and type(execution_stage) is int
            and type(score_position) is int
            and type(execution_position) is int
            and str(score_run.get("round_id") or "") == round_id
            and str(execution_run.get("round_id") or "") == round_id
            and str(score_run.get("scored_run_id") or "")
            == str(execution_run.get("run_id") or "")
            and str(score_run.get("submission_id") or "")
            == str(execution_run.get("submission_id") or "")
            and score_stage == execution_stage
            and score_position == execution_position
        )

    def _validated_attribution_source(
        self,
        *,
        source_score_run_id: str,
        source_scored_run_id: str,
        source_runner_hotkey: str,
        round_id: str,
        runs_by_id: Mapping[str, Mapping[str, Any]],
        company_evidence: Optional[Mapping[str, Any]] = None,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Return the judge hotkey only for a complete stored source binding."""

        source_score = runs_by_id.get(source_score_run_id)
        source_execution = runs_by_id.get(source_scored_run_id)
        completed_company = False
        if company_evidence is not None and (source_score or {}).get("status") == "failed":
            evidence = self._company_judgment_evidence(
                cache_key=str(company_evidence["cache_key"]),
                company_input_hash=str(company_evidence["company_input_hash"]),
                authority_slot=int(company_evidence["authority_slot"]),
                evidence_reads=evidence_reads,
                round_context=round_context,
            )
            if evidence != company_evidence:
                raise scoring.ScoringError("company scoring attribution evidence differs")
            completed_company = True
        if (
            source_score is None
            or source_execution is None
            or source_score.get("runner_hotkey") != source_runner_hotkey
            or not self._same_scoring_membership(
                source_score, source_execution, round_id=round_id,
                completed_company=completed_company,
            )
        ):
            raise scoring.ScoringError("scoring attribution source is invalid")
        try:
            contracts.require_hotkey(source_runner_hotkey)
        except ArenaContractError as exc:
            raise scoring.ScoringError(
                "scoring attribution hotkey is invalid"
            ) from exc
        return source_runner_hotkey

    def _add_signal_attribution(
        self,
        attributed: Dict[str, bool],
        *,
        source_ids: Optional[set[str]],
        company_source_run_id: str,
        company_indexes: set[int],
        current_score_run_id: str,
        round_id: str,
        runs_by_id: Mapping[str, Mapping[str, Any]],
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
    ) -> None:
        """Credit immutable original signal sources used by this company."""

        reads = evidence_reads if isinstance(evidence_reads, _PublicResultEvidenceReads) else None
        uses = (reads.uses(self._store, company_source_run_id) if reads is not None
                else self._store.get_signal_judgment_uses(company_source_run_id))
        seen: set[tuple[int, int]] = set()
        for use in uses:
            if use.get("company_index") not in company_indexes:
                continue
            signal_index = use.get("signal_index")
            use_position = (use.get("company_index"), signal_index)
            if (use.get("run_id") != company_source_run_id
                or type(signal_index) is not int or use_position in seen):
                raise scoring.ScoringError("signal attribution use is invalid")
            seen.add(use_position)
            cache_key = str(use.get("cache_key") or "")
            authority_slot = use.get("authority_slot")
            if type(authority_slot) is not int or authority_slot < 0:
                raise scoring.ScoringError("signal attribution slot is invalid")
            row = (reads.signal(self._store, cache_key, authority_slot) if reads is not None
                   else self._store.get_signal_judgment(cache_key, authority_slot))
            if row is None or any(row.get(field) != use.get(field) for field in (
                "cache_key", "authority_slot", "signal_input_hash",
                "source_score_run_id", "judgment_hash",
            )):
                raise scoring.ScoringError("signal attribution evidence is missing")
            scope = row.get("scope_doc")
            if not isinstance(scope, Mapping) or scope.get("round_id") != round_id:
                raise scoring.ScoringError("signal attribution scope is invalid")
            if (scope.get("cache_key") != cache_key
                or contracts.document_hash({key: value for key, value in scope.items() if key != "cache_key"}) != cache_key
                or scope.get("signal_input_hash") != row.get("signal_input_hash")
                or row.get("judgment_hash") != contracts.document_hash(row.get("judgment_doc"))):
                raise scoring.ScoringError("signal attribution hash is invalid")
            try:
                signal_judgments.validate_judgment(row.get("judgment_doc"), scope)
            except signal_judgments.SignalJudgmentError as exc:
                raise scoring.ScoringError("signal attribution judgment is invalid") from exc
            source_id = str(row.get("source_score_run_id") or "")
            source = runs_by_id.get(source_id)
            execution = runs_by_id.get(str(row.get("source_scored_run_id") or ""))
            if (source is None or execution is None
                or source.get("runner_hotkey") != row.get("source_runner_hotkey")
                or source.get("claim_request_hash") != row.get("source_claim_request_hash")
                or int(source.get("lease_generation") or 0) != row.get("source_lease_generation")
                or not self._same_scoring_membership(
                    source, execution, round_id=round_id, completed_signal=True
                )):
                raise scoring.ScoringError("signal attribution source is invalid")
            hotkey = str(row["source_runner_hotkey"])
            exclusions = row.get("runner_authority_exclusions")
            if (not isinstance(exclusions, list) or hotkey not in exclusions
                or len(exclusions) != len(set(exclusions))):
                raise scoring.ScoringError("signal attribution exclusions are invalid")
            try:
                contracts.require_hotkey(hotkey)
            except ArenaContractError as exc:
                raise scoring.ScoringError("signal attribution hotkey is invalid") from exc
            if source_ids is not None:
                source_ids.add(source_id)
            attributed[hotkey] = attributed.get(hotkey, False) or source_id != current_score_run_id

    def _score_run_attribution(
        self,
        score_run: Mapping[str, Any],
        execution_run: Mapping[str, Any],
        *,
        round_id: str,
        runs_by_id: Mapping[str, Mapping[str, Any]],
        source_ids: Optional[set[str]] = None,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Resolve actual judgment authorities for one accepted score row."""

        if not self._same_scoring_membership(
            score_run, execution_run, round_id=round_id
        ):
            raise scoring.ScoringError("scoring attribution membership is invalid")
        score_run_id = str(score_run["run_id"])
        attributed: Dict[str, bool] = {}
        signal_policy = (
            hasattr(self._store, "get_signal_judgment_uses")
            and ((self._round_from_context(round_id, round_context).get("configuration_doc") or {})
                 .get("scorer_policy") or {}).get("signal_judgment_policy")
            == signal_judgments.POLICY
        )

        if score_run.get("company_judgment_refs") is not None:
            try:
                lease = company_judgments.validate_lease_context(
                    (score_run.get("claim_response") or {}).get(
                        "company_judgment_cache"
                    ) or {}
                )
            except company_judgments.CompanyJudgmentError as exc:
                raise scoring.ScoringError(
                    "company scoring attribution lease is invalid"
                ) from exc
            refs = {
                int(ref["company_index"]): ref
                for ref in (
                    company_judgments.validate_company_ref(item)
                    for item in score_run.get("company_judgment_refs") or []
                )
            }
            lease_rows = sorted(
                lease["hits"] + lease["misses"],
                key=lambda item: item["company_index"],
            )
            if len(refs) != len(lease_rows):
                raise scoring.ScoringError(
                    "company scoring attribution refs are incomplete"
                )
            legacy_cache = any(ref["scope_doc"].get("company_judgment_policy") == company_judgments.POLICY for ref in refs.values())
            consumed = None
            if legacy_cache:
                try:
                    consumed = set(company_judgments.validated_consumed_indexes(score_run.get("result_doc") or {}, company_count=len(lease_rows)))
                except company_judgments.CompanyJudgmentError as exc:
                    raise scoring.ScoringError("company scoring attribution consumed indexes are invalid") from exc
            for item in lease_rows:
                ref = refs.get(int(item["company_index"]))
                if (
                    ref is None
                    or ref["cache_key"] != item["cache_key"]
                    or ref["company_input_hash"] != item["company_input_hash"]
                ):
                    raise scoring.ScoringError(
                        "company scoring attribution ref is invalid"
                    )
                if consumed is not None and item["company_index"] not in consumed:
                    continue
                evidence = item.get("evidence_doc")
                if (ref["scope_doc"].get("company_judgment_policy") == company_judgments.POLICY
                    and evidence is None):
                    evidence = self._company_judgment_evidence(
                        cache_key=item["cache_key"], company_input_hash=item["company_input_hash"],
                        authority_slot=item["authority_slot"],
                        evidence_reads=evidence_reads,
                        round_context=round_context,
                    )
                    # The proof already binds the stored source. Each consumer
                    # must still own it, including response-local proof hits.
                    if evidence["source_score_run_id"] != score_run_id:
                        raise scoring.ScoringError("consumed company judgment authority is missing or foreign")
                if evidence is None:
                    hotkey = self._validated_attribution_source(
                        source_score_run_id=score_run_id,
                        source_scored_run_id=str(score_run["scored_run_id"]),
                        source_runner_hotkey=str(
                            score_run.get("runner_hotkey") or ""
                        ),
                        round_id=round_id,
                        runs_by_id=runs_by_id,
                        evidence_reads=evidence_reads,
                        round_context=round_context,
                    )
                    if source_ids is not None:
                        source_ids.add(score_run_id)
                    attributed[hotkey] = attributed.get(hotkey, False)
                    if signal_policy:
                        self._add_signal_attribution(
                            attributed, source_ids=source_ids,
                            company_source_run_id=score_run_id,
                            company_indexes={int(item["company_index"])},
                            current_score_run_id=score_run_id, round_id=round_id,
                            runs_by_id=runs_by_id,
                            evidence_reads=evidence_reads,
                        )
                    continue
                hotkey = self._validated_attribution_source(
                    source_score_run_id=str(evidence["source_score_run_id"]),
                    source_scored_run_id=str(evidence["source_scored_run_id"]),
                    source_runner_hotkey=str(evidence["source_runner_hotkey"]),
                    round_id=round_id,
                    runs_by_id=runs_by_id,
                    company_evidence=evidence,
                    evidence_reads=evidence_reads,
                    round_context=round_context,
                )
                if source_ids is not None:
                    source_ids.add(str(evidence["source_score_run_id"]))
                reused = str(evidence["source_score_run_id"]) != score_run_id
                attributed[hotkey] = attributed.get(hotkey, False) or reused
                if signal_policy:
                    source_score = runs_by_id.get(str(evidence["source_score_run_id"])) or {}
                    source_indexes = {
                        int(source_ref["company_index"])
                        for source_ref in source_score.get("company_judgment_refs") or []
                        if isinstance(source_ref, Mapping)
                        and source_ref.get("cache_key") == evidence["cache_key"]
                        and source_ref.get("company_input_hash") == evidence["company_input_hash"]
                        and type(source_ref.get("company_index")) is int
                    }
                    if not source_indexes:
                        raise scoring.ScoringError("signal attribution company source is missing")
                    self._add_signal_attribution(
                        attributed, source_ids=source_ids,
                        company_source_run_id=str(evidence["source_score_run_id"]),
                        company_indexes=source_indexes,
                        current_score_run_id=score_run_id, round_id=round_id,
                        runs_by_id=runs_by_id,
                        evidence_reads=evidence_reads,
                    )
            return attributed

        cache_key = str(score_run.get("judgment_cache_key") or "")
        if cache_key:
            source_score_run_id = str(
                score_run.get("judgment_cache_source_run_id") or ""
            )
            source_score = runs_by_id.get(source_score_run_id) or {}
            if (
                not source_score_run_id
                or not source_score.get("scored_run_id")
                or source_score.get("judgment_cache_key") != cache_key
                or source_score.get("judgment_cache_source_run_id")
                != source_score_run_id
            ):
                raise scoring.ScoringError(
                    "scoring attribution cache authority is invalid"
                )
            if source_score_run_id != score_run_id:
                cached_result = score_run.get("result_doc") or {}
                if (
                    cached_result.get("schema_version")
                    != "leadpoet.lab_arena.cached_run_result.v1"
                    or cached_result.get("terminal_status") != "accepted"
                    or cached_result.get("cache_key") != cache_key
                    or cached_result.get("source_score_run_id")
                    != source_score_run_id
                ):
                    raise scoring.ScoringError(
                        "scoring attribution cached result is invalid"
                    )
            hotkey = self._validated_attribution_source(
                source_score_run_id=source_score_run_id,
                source_scored_run_id=str(source_score["scored_run_id"]),
                source_runner_hotkey=str(source_score.get("runner_hotkey") or ""),
                round_id=round_id,
                runs_by_id=runs_by_id,
                evidence_reads=evidence_reads,
                round_context=round_context,
            )
            if source_ids is not None:
                source_ids.add(source_score_run_id)
            attributed[hotkey] = source_score_run_id != score_run_id
            return attributed

        hotkey = self._validated_attribution_source(
            source_score_run_id=score_run_id,
            source_scored_run_id=str(score_run["scored_run_id"]),
            source_runner_hotkey=str(score_run.get("runner_hotkey") or ""),
            round_id=round_id,
            runs_by_id=runs_by_id,
        )
        if source_ids is not None:
            source_ids.add(score_run_id)
        attributed[hotkey] = False
        return attributed

    def _prefetch_public_result_evidence(
        self, reads: _PublicResultEvidenceReads, *, round_id: str,
        execution_runs, score_runs, public_run_ids, source_rows, round_context,
        include_signal_attribution: bool = True,
    ) -> None:
        """Seed bounded raw reads only; all authority validators still run below."""
        executions = {str(run["run_id"]): run for run in execution_runs}
        judgments = self._select_scoring_outputs(score_runs, round_row=round_context)
        pairs = set()
        source_ids = set(executions) | {str(run["run_id"]) for run in score_runs}
        signal_companies: Dict[str, set[int]] = {}
        signal_policy = (((round_context.get("configuration_doc") or {}).get("scorer_policy") or {})
                         .get("signal_judgment_policy") == signal_judgments.POLICY)
        for execution_id, judge in judgments.items():
            if (execution_id not in public_run_ids or judge.get("status") != "accepted"
                    or execution_id not in executions
                    or not self._same_scoring_membership(judge, executions[execution_id], round_id=round_id)
                    or judge.get("company_judgment_refs") is None):
                continue
            try:
                lease = company_judgments.validate_lease_context(
                    (judge.get("claim_response") or {}).get("company_judgment_cache") or {}
                )
            except company_judgments.CompanyJudgmentError:
                continue  # The unchanged consumer verifier rejects malformed leases.
            for item in lease["hits"] + lease["misses"]:
                pairs.add((item["cache_key"], item["authority_slot"]))
                evidence = item.get("evidence_doc")
                if evidence is not None:
                    source_ids.add(str(evidence["source_score_run_id"]))
                if not signal_policy or not include_signal_attribution:
                    continue
                source_id = str(evidence["source_score_run_id"]) if evidence is not None else str(judge["run_id"])
                if evidence is None:
                    indexes = {item["company_index"]}
                else:
                    source = source_rows.get(source_id) or {}
                    indexes = {
                        ref["company_index"] for ref in source.get("company_judgment_refs") or []
                        if isinstance(ref, Mapping) and type(ref.get("company_index")) is int
                        and ref.get("cache_key") == item["cache_key"]
                        and ref.get("company_input_hash") == item["company_input_hash"]
                    }
                if indexes:
                    signal_companies.setdefault(source_id, set()).update(indexes)
        reads.seed(reads.runs, source_rows if include_signal_attribution else {
            run_id: source_rows[run_id] for run_id in source_ids if run_id in source_rows
        })
        company_reader = getattr(self._store, "get_company_judgment_batch", None)
        if pairs and company_reader is not None:
            rows = company_reader(sorted(pairs))
            reads.seed(reads.companies, {pair: rows.get(pair) for pair in pairs})
        uses_reader = getattr(self._store, "get_signal_judgment_uses_batch", None)
        if signal_companies and uses_reader is not None:
            reads.seed(reads.signal_uses, uses_reader(sorted(signal_companies)))
        signal_reader = getattr(self._store, "get_signal_judgment_batch", None)
        if signal_companies and signal_reader is not None:
            signal_pairs = set()
            for run_id, indexes in signal_companies.items():
                for use in reads.uses(self._store, run_id):
                    key, slot = use.get("cache_key"), use.get("authority_slot")
                    if (use.get("company_index") in indexes and type(key) is str
                            and key and type(slot) is int and 0 <= slot <= 2147483647):
                        signal_pairs.add((key, slot))
            if signal_pairs:
                rows = signal_reader(sorted(signal_pairs))
                reads.seed(reads.signals, {pair: rows.get(pair) for pair in signal_pairs})

    def _public_result_authority_reads(
        self,
        round_id: str,
        execution_runs: Sequence[Mapping[str, Any]],
        score_runs: Sequence[Mapping[str, Any]],
        public_run_ids: set[str], *, round_context=None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prefetch only accepted, referenced authorities for this response."""

        executions = {str(run["run_id"]): run for run in execution_runs}
        judgments = self._select_scoring_outputs(score_runs, round_row=self._round_from_context(round_id, round_context))
        selected = [
            judge for run_id, judge in judgments.items()
            if judge.get("status") == "accepted"
            and run_id in executions
            and self._same_scoring_membership(judge, executions[run_id], round_id=round_id)
        ]
        keys = {
            str(judge["judgment_cache_key"])
            for judge in selected
            if str(judge.get("scored_run_id") or "") in public_run_ids
            and judge.get("judgment_cache_key")
            and judge.get("company_judgment_refs") is None
        }
        cache_reader = getattr(self._store, "get_judgment_caches", None)
        try:
            caches = cache_reader(sorted(keys)) if keys and cache_reader is not None else {}
        except ArenaStoreUnavailable:
            raise
        except ArenaStoreError as exc:
            raise ServiceError("public_result_unavailable", 503) from exc
        cached_rows = {key: caches.get(key) for key in keys} if cache_reader is not None else {}

        local_rows = {
            str(run["run_id"]): run
            for run in list(execution_runs) + list(score_runs)
        }
        source_ids: set[str] = set()
        for judge in selected:
            cache_source = str(judge.get("judgment_cache_source_run_id") or "")
            if cache_source:
                source_ids.add(cache_source)
            if judge.get("company_judgment_refs") is not None:
                try:
                    lease = company_judgments.validate_lease_context(
                        (judge.get("claim_response") or {}).get("company_judgment_cache") or {}
                    )
                except company_judgments.CompanyJudgmentError:
                    continue  # The existing attribution verifier marks this invalid.
                for item in lease["hits"] + lease["misses"]:
                    evidence = item.get("evidence_doc")
                    if isinstance(evidence, Mapping) and evidence.get("source_score_run_id"):
                        source_ids.add(str(evidence["source_score_run_id"]))

        run_reader = getattr(self._store, "get_runs", None)
        if run_reader is None:
            return cached_rows, local_rows
        missing_scores = source_ids - local_rows.keys()
        try:
            fetched_scores = run_reader(sorted(missing_scores)) if missing_scores else {}
        except ArenaStoreUnavailable:
            raise
        except ArenaStoreError as exc:
            raise ServiceError("public_result_unavailable", 503) from exc
        source_rows = {run_id: fetched_scores.get(run_id) for run_id in missing_scores}
        source_rows.update(local_rows)
        source_execution_ids = {
            str(source["scored_run_id"])
            for source_id in source_ids
            if (source := source_rows.get(source_id)) is not None
            and source.get("scored_run_id")
        }
        missing_executions = source_execution_ids - source_rows.keys()
        try:
            fetched_executions = run_reader(sorted(missing_executions)) if missing_executions else {}
        except ArenaStoreUnavailable:
            raise
        except ArenaStoreError as exc:
            raise ServiceError("public_result_unavailable", 503) from exc
        source_rows.update({run_id: fetched_executions.get(run_id) for run_id in missing_executions})
        return cached_rows, source_rows

    def _public_scoring_attribution(
        self,
        round_id: str,
        submission_id: str,
        execution_runs: Sequence[Mapping[str, Any]],
        source_execution_runs: Sequence[Mapping[str, Any]],
        public_positions: set[int],
        score_runs: Optional[Sequence[Mapping[str, Any]]] = None,
        source_rows: Optional[Mapping[str, Any]] = None,
        evidence_reads: Optional[_CompanyEvidenceReads] = None,
        round_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build aggregate and disclosed per-ICP judge attribution."""

        # A score run belongs to the submission of its scored execution run.
        # Cross-submission cache sources are resolved by id below.
        if score_runs is None:
            score_runs = (
                self._store.list_runs(round_id, kind="score", submission_id=submission_id)
                if execution_runs else []
            )
        judgments = self._select_scoring_outputs(score_runs, round_row=self._round_from_context(round_id, round_context))
        runs_by_id = _PublicRunLookup(
            {
                **(source_rows or {}),
                **{
                    str(run["run_id"]): run
                    for run in list(source_execution_runs) + list(score_runs)
                    if run.get("run_id")
                },
            }, self._store,
        )
        by_position: Dict[int, Dict[str, bool]] = {}
        unattributed_positions = set()
        versions_by_position: Dict[int, set] = {}
        for execution_run in execution_runs:
            score_run = judgments.get(str(execution_run.get("run_id") or ""))
            if score_run is None or score_run.get("status") != "accepted":
                continue
            position = int(execution_run.get("icp_position") or 0)
            try:
                source_ids: set[str] = set()
                authorities = self._score_run_attribution(
                    score_run,
                    execution_run,
                    round_id=round_id,
                    runs_by_id=runs_by_id,
                    source_ids=source_ids,
                    evidence_reads=evidence_reads,
                    round_context=round_context,
                )
                if not authorities:
                    raise scoring.ScoringError(
                        "scoring attribution authority is missing"
                    )
            except (
                ArenaContractError,
                ArenaStoreError,
                KeyError,
                TypeError,
                ValueError,
                scoring.ScoringError,
            ):
                unattributed_positions.add(position)
                by_position.pop(position, None)
                continue
            if position in unattributed_positions:
                continue
            versions = versions_by_position.setdefault(position, set())
            for source_id in source_ids:
                source = runs_by_id.get(source_id) or {}
                document = source.get("result_doc")
                resource = document.get("resource_summary") if isinstance(document, Mapping) else None
                resource = resource if isinstance(resource, Mapping) else {}
                commit = str(resource.get("validator_source_commit") or "")
                if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                    dirty = str(resource.get("validator_source_dirty") or "unknown")
                    versions.add((str(source.get("runner_hotkey") or ""), commit,
                                  dirty if dirty in ("clean", "dirty") else "unknown"))
            combined = by_position.setdefault(position, {})
            for hotkey, reused in authorities.items():
                combined[hotkey] = combined.get(hotkey, False) or reused

        validator_counts: Dict[str, Dict[str, int]] = {}
        for authorities in by_position.values():
            for hotkey, reused in authorities.items():
                counts = validator_counts.setdefault(
                    hotkey, {"icp_count": 0, "reused_icp_count": 0}
                )
                counts["icp_count"] += 1
                counts["reused_icp_count"] += int(reused)

        code_versions: Dict[tuple, list] = {}
        for position, versions in versions_by_position.items():
            if position in public_positions and position not in unattributed_positions:
                for version in versions:
                    code_versions.setdefault(version, []).append(position)
        return {
            "code_versions": [
                {"validator_hotkey": key[0], "commit": key[1], "working_tree": key[2],
                 "icp_positions": sorted(set(positions))}
                for key, positions in sorted(code_versions.items())
            ],
            "validators": [
                {"hotkey": hotkey, **validator_counts[hotkey]}
                for hotkey in sorted(validator_counts)
            ],
            "icps": [
                {
                    "icp_position": position,
                    "validator_hotkeys": sorted(
                        by_position.get(position, {})
                    ),
                    "reused_judgment": any(
                        by_position.get(position, {}).values()
                    ),
                }
                for position in sorted(
                    set(by_position).union(unattributed_positions)
                    & public_positions
                )
            ],
            "unattributed_icp_count": len(unattributed_positions),
        }

    def public_results(self, round_id: str, submission_id: str) -> Dict[str, Any]:
        if not submission_id or not isinstance(submission_id, str):
            raise ServiceError("submission_missing", 404)  # an empty id must never mean "every submission"
        row = self._public_results_round(round_id)
        round_status = str(row["status"])
        public_reads = getattr(self, "_public_result_reads", None)
        read_objects = (
            public_reads
            if round_status == "published" and public_reads is not None
            and public_reads.source is self._objects
            else None
        )
        completed = self.completed_submission_scores(row).get(submission_id) if round_status != "published" else None
        if round_status != "published" and completed is None:
            raise ServiceError("results_not_public", 403)
        publication = row.get("publication_doc") or {}
        participants = (publication.get("participants") if round_status == "published" else row.get("participants")) or []
        participant = next(
            (
                item
                for item in participants
                if item.get("submission_id") == submission_id
            ),
            None,
        )
        if participant is None:
            raise ServiceError("submission_missing", 404)
        combined_all_reader = getattr(self._store, "list_public_result_runs", None)
        combined_reader = getattr(self._store, "list_public_result_execution_runs", None)
        disclosure_metadata = (
            icp_disclosure.disclosure_metadata(row)
            if round_status == "published"
            and (combined_all_reader is not None or combined_reader is not None)
            else None
        )
        baselines = (
            [p for p in row.get("participants") or [] if p.get("is_king") is True]
            if disclosure_metadata is not None else []
        )
        combined_score_runs = None
        if disclosure_metadata is not None and len(baselines) == 1:
            baseline_id = str(baselines[0]["submission_id"])
            combined_runs = (
                combined_all_reader(round_id, baseline_id, submission_id)
                if combined_all_reader is not None
                else combined_reader(round_id, baseline_id, submission_id)
            )
            disclosure = icp_disclosure.baseline_disclosure(
                row,
                [run for run in combined_runs
                 if run.get("submission_id") == baseline_id and run.get("kind") == "execute"],
                self.now(),
            )
            source_execution_runs = [
                run for run in combined_runs
                if run.get("submission_id") == submission_id and run.get("kind") == "execute"
            ]
            if combined_all_reader is not None:
                combined_score_runs = [
                    run for run in combined_runs
                    if run.get("submission_id") == submission_id and run.get("kind") == "score"
                ]
        else:
            disclosure = self._public_icp_disclosure(row)
            source_execution_runs = self._store.list_runs(
                round_id, kind="execute", submission_id=submission_id
            )
        public_positions = set(disclosure["public_positions"]) if disclosure else set()
        if completed is not None:
            # Projection only: preserve every stored output and write-once score.
            projected = {run["run_id"]: run for run in completed["execution_runs"]}
            source_execution_runs = [projected.get(run["run_id"], run) for run in source_execution_runs]
        execution_runs = [
            run for run in source_execution_runs
            if run.get("submission_id") == submission_id
        ]
        runs = [
            run for run in execution_runs
            if run.get("icp_position") in public_positions
        ]
        outputs = {}
        for run in runs:
            if run.get("output_ref"):
                try:
                    raw = (read_objects or self._objects).get_bounded(
                        str(run["output_ref"]), MAX_OUTPUT_BYTES,
                    )
                    outputs[run["run_id"]] = validate_output_document(json.loads(raw.decode("utf-8")))
                except Exception as exc:
                    raise ServiceError("public_output_unavailable", 503) from exc
        scores = {
            "stage_1": [
                {"run_id": run["run_id"], "icp_position": run["icp_position"], "per_icp_score": run["per_icp_score"]}
                for run in runs
                if int(run.get("stage") or 0) == 1 and run.get("per_icp_score") is not None
            ],
            "stage_2": [
                {"run_id": run["run_id"], "icp_position": run["icp_position"], "per_icp_score": run["per_icp_score"]}
                for run in runs
                if int(run.get("stage") or 0) == 2 and run.get("per_icp_score") is not None
            ],
        }
        stage1_entry = next((item for item in publication.get("stage1_ranking") or [] if item.get("submission_id") == submission_id), None)
        final_entry = next((item for item in publication.get("final_ranking") or [] if item.get("submission_id") == submission_id), None)
        if completed is not None:
            final_entry = completed
        run_results = [run["result_doc"] for run in runs if run.get("result_doc")]
        validated_results = []
        for document in run_results:
            try:
                validated_results.append(contracts.validate_run_result(document))
            except ArenaContractError as exc:
                raise ServiceError("public_result_unavailable", 503) from exc
        run_results = validated_results
        if not execution_runs:
            score_runs = []
        elif combined_score_runs is not None:
            score_runs = combined_score_runs
        else:
            score_runs = self._store.list_runs(
                round_id, kind="score", submission_id=submission_id
            )
        verification_required = (
            contact_policy.enabled(row.get("configuration_doc") or {})
            or integrity.enabled(row.get("configuration_doc") or {})
        )
        cached_rows, source_rows = self._public_result_authority_reads(
            round_id, execution_runs, score_runs,
            set(outputs) if verification_required else set(), round_context=row,
        )
        evidence_reads = _PublicResultEvidenceReads()
        try:
            self._prefetch_public_result_evidence(
                evidence_reads, round_id=round_id, execution_runs=execution_runs,
                score_runs=score_runs, public_run_ids=set(outputs),
                source_rows=source_rows, round_context=row,
            )
        except ArenaStoreUnavailable:
            raise
        except ArenaStoreError as exc:
            raise ServiceError("public_result_unavailable", 503) from exc
        result = {
            "round_id": round_id, "submission_id": submission_id, "submission": {
                "miner_hotkey": participant.get("miner_hotkey"),
                "is_baseline": bool(participant.get("is_baseline", participant.get("is_king", False))),
            },
            "outputs": outputs, "run_results": run_results,
            "scores": scores,
            "public_icp_status": "ready" if disclosure else "pending",
            "public_icp_count": len(public_positions),
            "benchmark_icp_count": contracts.benchmark_icp_count(row.get("configuration_doc")),
            "score_status": "published" if round_status == "published" else "complete",
            "submission_scores": {
                "stage_1": None if stage1_entry is None else stage1_entry.get("stage1_score"),
                "final": None if final_entry is None else final_entry.get("final_score"),
            },
            "scoring_attribution": self._public_scoring_attribution(
                round_id,
                submission_id,
                execution_runs,
                source_execution_runs,
                public_positions,
                score_runs=score_runs,
                source_rows=source_rows,
                evidence_reads=evidence_reads,
                round_context=row,
            ),
        }
        if verification_required:
            contacts_required = contact_policy.enabled(row.get("configuration_doc") or {})
            judgments = {}
            for stage in {
                int(run.get("stage") or 0)
                for run in runs
                if int(run.get("icp_position") or 0) in public_positions
            }:
                judgments.update(self._select_scoring_outputs(
                    [score_run for score_run in score_runs if score_run.get("stage") == stage],
                    round_row=row,
                ))
            icps = []
            if outputs:
                if read_objects is None:
                    icps = self._benchmark_icps_from_row(round_id, row)
                else:
                    icps = self._benchmark_icps_from_row(
                        round_id, row, get_object=read_objects.get,
                    )
            contacts = {}
            diagnostics = []
            for run in runs:
                run_id = str(run["run_id"])
                judge = judgments.get(run_id)
                if run_id not in outputs or not judge or judge.get("status") != "accepted":
                    continue
                try:
                    breakdowns = self._verified_breakdowns(
                        judge, icp=icps[int(run["icp_position"])], companies=outputs[run_id]["companies"],
                        policy=row["configuration_doc"]["scorer_policy"],
                        cached_rows=cached_rows, source_rows=source_rows,
                        evidence_reads=evidence_reads,
                        round_context=row,
                    )
                except scoring.ScoringError as exc:
                    raise ServiceError(
                        "public_contact_verification_unavailable" if contacts_required
                        else "public_company_verification_unavailable", 503
                    ) from exc
                if contacts_required:
                    contacts[run_id] = [
                        {key: value for key, value in verify.redact_breakdown(item).items()
                         if key in {"company_index", "company_qualified", "contact_qualified", "contact_identity_key", "email_status", "contact_verification"}}
                        for item in breakdowns
                    ]
                diagnostics.extend(
                    public_dashboard.company_diagnostic(
                        item,
                        outputs[run_id]["companies"][item["company_index"]],
                        icp_position=int(run["icp_position"]),
                        contacts_required=contacts_required,
                    )
                    for item in breakdowns
                )
            if contacts_required:
                result["contact_verifications"] = contacts
            result["company_diagnostics"] = diagnostics
        return result
