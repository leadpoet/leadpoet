"""Bounded semantic Sentry telemetry for restart and weight operations.

This module is deliberately stdlib-only.  It never imports the vendor SDK;
the single SDK boundary remains :mod:`sentry_bootstrap`.  Every helper is a
best-effort no-op and every field crosses an explicit allowlist before it can
reach the bootstrap scrubber.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from . import sentry_bootstrap


INCIDENT_FAILURE_CODES = frozenset(
    {
        "release.source_tree_mismatch",
        "release.schema_contract_mismatch",
        "release.channel_unavailable",
        "release.builder_resource_exhausted",
        "release.artifact_auth_failed",
        "runtime.enclave_relay_unavailable",
        "authority.dependency_unreadable",
        "restart.stage_deadline_exceeded",
        "weight.allocation_authority_missing",
        "weight.settlement_history_incomplete",
        "weight.ancestry_bounds_exceeded",
        "weight.frontier_source_invalid",
        "weight.gateway_endpoint_unavailable",
        "weight.block_drift_exhausted",
        "weight.authoritative_result_invalid",
        "weight.sdk_response_invalid",
        "weight.chain_transport_poisoned",
        "weight.finalization_missing",
        "release.pcr0_mismatch",
        "weight.bundle_divergence",
        "restart.terminal_failure",
    }
)

_SAFE_FIELDS = frozenset(
    {
        "allocation_hash",
        "aggregate_hash",
        "aggregate_score",
        "approved_sha",
        "attempt",
        "attempts",
        "assignment_hash",
        "authority_epoch_id",
        "authority_hash",
        "authority_mode",
        "authority_stage",
        "backoff_seconds",
        "backlog_count",
        "benchmark_attempt",
        "benchmark_bundle_id",
        "benchmark_date",
        "baseline_last_update",
        "block",
        "blocks_remaining",
        "blocked_stages",
        "boot_identity_hash",
        "broadcast_block",
        "build_manifest_hash",
        "bundle_hash",
        "bytes_processed",
        "cache_hit",
        "candidate_sha",
        "chain_endpoint_alias",
        "cleanup_result",
        "checkpoint_hash",
        "checkpoint_attempt_count",
        "checkpoint_sequence",
        "coalesced_count",
        "commit_sha",
        "completed_icp_count",
        "component",
        "concurrency",
        "confirmation_attempt",
        "confirmation_stage",
        "conditional_icp_count",
        "correlation_id",
        "current_block",
        "cutover_epoch_id",
        "dependency",
        "destination_count",
        "decision_code",
        "disk_available_bytes",
        "disk_required_bytes",
        "duration_ms",
        "duration_seconds",
        "elapsed_blocks",
        "event_sequence",
        "epoch_block",
        "epoch_id",
        "endpoint_kind",
        "error_code",
        "exception_class",
        "expected_block",
        "expected_commit_sha",
        "expected_pcr0_hash",
        "expected_vector_count",
        "expected_icp_count",
        "extrinsic_hash",
        "fail_closed",
        "failure_code",
        "finalization_block",
        "finalization_hash",
        "finalized_block",
        "finalized_block_hash",
        "first_missing_epoch",
        "first_unfinished_stage",
        "frame_limit_bytes",
        "frontier_epoch",
        "frontier_hash",
        "gateway_restart_invocation_id",
        "http_status",
        "icp_score",
        "inclusion_block",
        "icp_ordinal",
        "icp_ref_hash",
        "last_successful_stage",
        "last_update",
        "lineage_hash",
        "logical_bytes",
        "manifest_hash",
        "max_block_drift",
        "max_depth",
        "memory_high_water_bytes",
        "migration_version",
        "mechanism_id",
        "missing_milestones",
        "model_artifact_hash",
        "netuid",
        "observed_block",
        "observed_pcr0_hash",
        "observed_vector_count",
        "operation",
        "pages",
        "parent_count",
        "persistence_hash",
        "physical_role",
        "policy_hash",
        "private_icp_count",
        "provider_id",
        "publication_hash",
        "public_icp_count",
        "query_fingerprint",
        "reason_code",
        "receipt_count",
        "recovered_icp_count",
        "release_correlation_id",
        "release_attempts",
        "release_status",
        "request_hash",
        "request_bytes",
        "response_bytes",
        "restart_invocation_id",
        "retryable",
        "retry_round",
        "retry_rounds",
        "role",
        "root_receipt_hash",
        "row_count",
        "rpc_method",
        "runtime_sha",
        "schema_version",
        "score_summary_hash",
        "scoring_run_id",
        "selected_icp_count",
        "sequence",
        "settlement_attempt",
        "settlement_hash",
        "sdk_response_class",
        "sdk_version",
        "shutdown_started",
        "snapshot_hash",
        "source_epoch_id",
        "sqlstate",
        "stage",
        "stage_ledger",
        "status",
        "result_hash",
        "scored_company_count",
        "result_status",
        "sourced_company_count",
        "subnet_epoch_index",
        "submission_epoch_id",
        "submission_attempt",
        "submitted_block",
        "superseded_sha",
        "terminal",
        "timed_out",
        "transaction_name",
        "transport_session_id",
        "uid",
        "unresolved_icp_count",
        "validator_id_hash",
        "validator_restart_invocation_id",
        "validator_role",
        "vector_hash",
        "vector_matches",
        "weight_correlation_id",
        "weight_finalization_event_hash",
        "weight_submission_event_hash",
        "weights_hash",
        "window_hash",
        "worker_id_hash",
    }
)

_TAG_FIELDS = frozenset(
    {
        "component",
        "failure_code",
        "physical_role",
        "role",
        "stage",
        "validator_role",
        "retryable",
        "terminal",
        "fail_closed",
        "release_status",
    }
)

_EVENT_TAG_FIELDS = _TAG_FIELDS | frozenset(
    {
        "approved_sha",
        "bundle_hash",
        "candidate_sha",
        "correlation_id",
        "epoch_id",
        "netuid",
        "release_correlation_id",
        "restart_invocation_id",
        "runtime_sha",
        "validator_id_hash",
        "weight_correlation_id",
        "weights_hash",
    }
)

_HASH_FIELDS = frozenset(
    {
        name for name in _SAFE_FIELDS if name.endswith("_hash")
    }
    | {
        "allocation_hash",
        "bundle_hash",
        "extrinsic_hash",
        "finalization_hash",
        "frontier_hash",
        "publication_hash",
        "request_hash",
        "root_receipt_hash",
        "snapshot_hash",
        "vector_hash",
        "weight_submission_event_hash",
        "weights_hash",
    }
)

_SHA_RE = re.compile(r"^(?:(?:sha256:|0x)?[0-9a-f]{16,128})$", re.I)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,200}$")
_current_context: ContextVar[Dict[str, Any]] = ContextVar(
    "leadpoet_sentry_operation_context", default={}
)
_operation_logger = logging.getLogger("leadpoet.operations")
_event_sequence = 0
_event_sequence_lock = threading.Lock()


def _next_event_sequence() -> int:
    global _event_sequence
    with _event_sequence_lock:
        _event_sequence += 1
        return _event_sequence


def _emit_operation_log(event_type: str, fields: Mapping[str, Any]) -> None:
    """Emit one bounded single-line event into ordinary process logs."""

    try:
        payload = safe_fields(fields)
        payload["event_sequence"] = _next_event_sequence()
        payload = {
            "event_type": str(event_type),
            **payload,
        }
        level = (
            logging.WARNING
            if payload.get("status") in {"failed", "rejected", "blocked"}
            or payload.get("terminal") is True
            else logging.INFO
        )
        _operation_logger.log(
            level,
            "leadpoet_operation_event %s",
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    except BaseException:
        return


def hash_identifier(value: Any) -> str:
    """Return a non-reversible join key for a public or private identifier."""

    encoded = str(value or "").encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def release_correlation_id(commit_sha: Any) -> Optional[str]:
    commit = str(commit_sha or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return "release:" + hashlib.sha256(
        ("leadpoet-release-v1:" + commit).encode("ascii")
    ).hexdigest()


def weight_correlation_id(
    *, runtime_sha: Any, netuid: Any, epoch_id: Any, bundle_hash: Any = None
) -> Optional[str]:
    # Keep one operation identity before and after canonical bundle creation.
    # The bundle hash remains an independent comparison field; making it the
    # correlation key would split retrieval and finalization, and divergent
    # auditor bundles would become harder to compare in the same trace.
    del bundle_hash
    commit = str(runtime_sha or "").strip().lower()
    try:
        network = int(netuid)
        epoch = int(epoch_id)
    except (TypeError, ValueError):
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    material = f"leadpoet-weight-v1:{commit}:{network}:{epoch}"
    return "weight:" + hashlib.sha256(material.encode("ascii")).hexdigest()


def _trace_id(fields: Mapping[str, Any]) -> Optional[str]:
    try:
        identity = (
            fields.get("weight_correlation_id")
            or fields.get("restart_invocation_id")
            or fields.get("release_correlation_id")
            or fields.get("correlation_id")
        )
        if not identity:
            return None
        return hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:32]
    except BaseException:
        return None


def _bounded_scalar(name: str, value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if name in {"blocked_stages", "missing_milestones"}:
        if not isinstance(value, (list, tuple)):
            return []
        return [
            str(item)[:100]
            for item in value[:40]
            if _IDENTIFIER_RE.fullmatch(str(item)[:200])
        ]
    if name == "stage_ledger":
        if not isinstance(value, (list, tuple)):
            return []
        output = []
        for item in value[:80]:
            if not isinstance(item, Mapping):
                continue
            output.append(
                {
                    key: _bounded_scalar(key, child)
                    for key, child in item.items()
                    if key in {"stage", "status", "elapsed_seconds", "duration_seconds"}
                }
            )
        return output
    text = str(value).strip()
    if name in _HASH_FIELDS:
        return text.lower()[:140] if _SHA_RE.fullmatch(text) else None
    if name.endswith("_sha") or name == "commit_sha":
        return text.lower() if re.fullmatch(r"[0-9a-f]{40}", text.lower()) else None
    if name.endswith("_id") or name.endswith("_code") or name in {
        "component",
        "dependency",
        "operation",
        "physical_role",
        "role",
        "stage",
        "status",
        "validator_role",
    }:
        return text[:200] if _IDENTIFIER_RE.fullmatch(text[:200]) else None
    return sentry_bootstrap.scrub_sentry_value(text, 200)


def safe_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop unknown fields and bound every retained value."""

    output: Dict[str, Any] = {}
    try:
        items = fields.items()
    except BaseException:
        return output
    for name, value in items:
        try:
            key = str(name)
            if key not in _SAFE_FIELDS:
                continue
            bounded = _bounded_scalar(key, value)
            if bounded is not None:
                output[key] = bounded
        except BaseException:
            continue
    return output


def configure_sentry_context(**fields: Any) -> Dict[str, Any]:
    """Merge safe process/task context and expose low-cardinality tags."""

    try:
        merged = dict(_current_context.get())
        merged.update(safe_fields(fields))
        if not merged.get("release_correlation_id"):
            merged["release_correlation_id"] = release_correlation_id(
                merged.get("runtime_sha") or merged.get("candidate_sha")
            )
        merged = safe_fields(merged)
        _current_context.set(merged)
        for key in _TAG_FIELDS:
            if key in merged:
                sentry_bootstrap.set_sentry_tag(key, merged[key])
        for key in (
            "correlation_id",
            "release_correlation_id",
            "restart_invocation_id",
            "weight_correlation_id",
        ):
            if key in merged:
                sentry_bootstrap.set_sentry_tag(key, merged[key])
        return merged
    except BaseException:
        try:
            return dict(_current_context.get())
        except BaseException:
            return {}


@contextmanager
def sentry_context(**fields: Any) -> Iterator[Dict[str, Any]]:
    token = None
    try:
        previous = _current_context.get()
        merged = dict(previous)
        merged.update(safe_fields(fields))
        token = _current_context.set(safe_fields(merged))
        current = _current_context.get()
    except BaseException:
        current = {}
    try:
        yield current
    finally:
        if token is not None:
            try:
                _current_context.reset(token)
            except BaseException:
                pass


class _TerminalLimiter:
    def __init__(self, max_entries: int = 1024) -> None:
        self._max_entries = int(max_entries)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def admit(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen[key] = None
            while len(self._seen) > self._max_entries:
                self._seen.popitem(last=False)
            return True


_terminal_limiter = _TerminalLimiter()


def record_retry(
    failure_code: str,
    *,
    component: str,
    stage: str,
    attempt: int,
    attempts: Optional[int] = None,
    **fields: Any,
) -> None:
    """Record one retry as a bounded breadcrumb, never an event."""

    try:
        payload = dict(_current_context.get())
        payload.update(fields)
        payload.update(
            {
                "failure_code": failure_code,
                "component": component,
                "stage": stage,
                "attempt": attempt,
                "attempts": attempts,
                "terminal": False,
            }
        )
        safe = safe_fields(payload)
        _emit_operation_log("retry", safe)
        sentry_bootstrap.add_sentry_breadcrumb(
            category="leadpoet.retry",
            message=failure_code,
            level="warning",
            data=safe,
        )
    except BaseException:
        return


def record_stage(
    *,
    component: str,
    stage: str,
    status: str,
    duration_seconds: Optional[float] = None,
    _emit_trace: bool = True,
    **fields: Any,
) -> None:
    try:
        payload = dict(_current_context.get())
        payload.update(fields)
        payload.update({"component": component, "stage": stage, "status": status})
        if duration_seconds is not None:
            payload["duration_seconds"] = round(float(duration_seconds), 3)
        safe = safe_fields(payload)
        _emit_operation_log("stage", safe)
        sentry_bootstrap.add_sentry_breadcrumb(
            category="leadpoet.stage",
            message=f"{component}.{stage}.{status}",
            level=(
                "info"
                if status in {"passed", "completed", "ready"}
                else "warning"
            ),
            data=safe,
        )
        if duration_seconds is not None:
            sentry_bootstrap.record_sentry_distribution(
                "leadpoet.stage.duration",
                float(duration_seconds),
                unit="second",
                tags={
                    "component": component,
                    "stage": stage,
                    "status": status,
                },
            )
        if _emit_trace and status in {"passed", "completed", "ready"}:
            with sentry_bootstrap.start_sentry_span(
                operation="leadpoet.stage",
                description=f"{component}.{stage}",
                data=safe,
                trace_id=_trace_id(safe),
            ):
                pass
    except BaseException:
        return


@contextmanager
def sentry_stage(
    *,
    component: str,
    operation: str,
    stage: str,
    **fields: Any,
) -> Iterator[None]:
    """Create one sampled manual span plus deterministic stage telemetry."""

    started = time.monotonic()
    with sentry_context(component=component, operation=operation, stage=stage, **fields):
        record_stage(
            component=component,
            stage=stage,
            status="started",
            _emit_trace=False,
            **fields,
        )
        with sentry_bootstrap.start_sentry_span(
            operation=f"leadpoet.{operation}",
            description=f"{component}.{stage}",
            data=safe_fields({**_current_context.get(), **fields}),
            trace_id=_trace_id({**_current_context.get(), **fields}),
        ):
            try:
                yield
            except BaseException:
                record_stage(
                    component=component,
                    stage=stage,
                    status="failed",
                    duration_seconds=time.monotonic() - started,
                    _emit_trace=False,
                    **fields,
                )
                raise
            else:
                record_stage(
                    component=component,
                    stage=stage,
                    status="passed",
                    duration_seconds=time.monotonic() - started,
                    _emit_trace=False,
                    **fields,
                )


def capture_failure(
    failure_code: str,
    *,
    component: str,
    stage: str,
    exception: Optional[BaseException] = None,
    terminal: bool = True,
    retryable: bool = False,
    fail_closed: bool = True,
    **fields: Any,
) -> bool:
    """Capture one semantic failure.  Repeated terminal calls are coalesced."""

    try:
        code = str(failure_code)
        if code not in INCIDENT_FAILURE_CODES:
            code = "restart.terminal_failure"
        payload = dict(_current_context.get())
        payload.update(fields)
        payload.update(
            {
                "failure_code": code,
                "component": component,
                "stage": stage,
                "terminal": bool(terminal),
                "retryable": bool(retryable),
                "fail_closed": bool(fail_closed),
                "exception_class": type(exception).__name__ if exception else None,
            }
        )
        safe = safe_fields(payload)
        correlation = (
            safe.get("weight_correlation_id")
            or safe.get("restart_invocation_id")
            or safe.get("release_correlation_id")
            or safe.get("correlation_id")
        )
        if not correlation:
            epoch_id = safe.get("epoch_id") or safe.get("submission_epoch_id")
            if epoch_id is not None:
                correlation = ":".join(
                    (
                        "epoch",
                        str(safe.get("runtime_sha") or "unknown"),
                        str(safe.get("netuid") or "unknown"),
                        str(epoch_id),
                    )
                )
        correlation = str(correlation or "process")
        logical_actor = str(
            safe.get("validator_id_hash")
            or safe.get("physical_role")
            or safe.get("validator_role")
            or "process"
        )
        # The first causal stage wins. Outer wrappers often observe and
        # re-raise the same logical failure at a broader stage; including the
        # stage here would turn that propagation into duplicate events.
        limiter_key = "|".join((correlation, code, component, logical_actor))
        if terminal and not _terminal_limiter.admit(limiter_key):
            return False
        if not terminal:
            retry_fields = dict(safe)
            retry_fields.pop("attempt", None)
            retry_fields.pop("attempts", None)
            retry_fields.pop("component", None)
            retry_fields.pop("stage", None)
            retry_fields.pop("failure_code", None)
            record_retry(
                code,
                component=component,
                stage=stage,
                attempt=int(safe.get("attempt") or 1),
                attempts=safe.get("attempts"),
                **retry_fields,
            )
            return False
        _emit_operation_log("failure", safe)
        return sentry_bootstrap.capture_sentry_failure(
            failure_code=code,
            component=component,
            stage=stage,
            exception=exception,
            tags={key: safe[key] for key in _EVENT_TAG_FIELDS if key in safe},
            context=safe,
        )
    except BaseException:
        return False


_SIGNATURE_CODES = (
    (re.compile(r"cannot import name|out of sync", re.I), "release.source_tree_mismatch"),
    (re.compile(r"schema cache|does not exist|undefined (?:column|function)|PGRST", re.I), "release.schema_contract_mismatch"),
    (re.compile(r"approved .*release.*(?:unavailable|not published)|release channel unavailable", re.I), "release.channel_unavailable"),
    (re.compile(r"no space left|insufficient disk|stale .*mount|builder.*(?:disk|space)", re.I), "release.builder_resource_exhausted"),
    (
        re.compile(
            r"artifact.*(?:auth|decrypt|capacity)|credential envelope|"
            r"encrypted storage document authentication failed|artifact vault.*(?:full|capacity)",
            re.I,
        ),
        "release.artifact_auth_failed",
    ),
    (re.compile(r"vsock|enclave.*(?:listener|relay|not ready)|relay.*(?:stopped|failed)", re.I), "runtime.enclave_relay_unavailable"),
    (
        re.compile(
            r"(?:chain|supabase|transparency|authority).*(?:unavailable|unreadable|timeout|timed out|connection failed|5\d\d)|"
            r"(?:websocket|rpc).*(?:connection failed|unavailable|timed out)",
            re.I,
        ),
        "authority.dependency_unreadable",
    ),
    (re.compile(r"historical compute fallback lacks finalized allocation authority", re.I), "weight.allocation_authority_missing"),
    (re.compile(r"chain realized settlement history is incomplete", re.I), "weight.settlement_history_incomplete"),
    (re.compile(r"ancestry.*(?:size|bound|frame|too large)|frame.*(?:limit|exceed)|weight input response.*(?:size|frame)", re.I), "weight.ancestry_bounds_exceeded"),
    (re.compile(r"allocation_frontier_bootstrap_source_invalid", re.I), "weight.frontier_source_invalid"),
    (re.compile(r"pinned_gateway_request_failed|requested URL returned error: 502|connection refused.*8000", re.I), "weight.gateway_endpoint_unavailable"),
    (re.compile(r"block drift is too large", re.I), "weight.block_drift_exhausted"),
    (re.compile(r"authoritative weight result fields are invalid|another-boot|non-canonical", re.I), "weight.authoritative_result_invalid"),
    (re.compile(r"ExtrinsicResponse|SDK.*response.*(?:invalid|shape|unpack)", re.I), "weight.sdk_response_invalid"),
    (re.compile(r"open subscriptions|Unable to reconnect", re.I), "weight.chain_transport_poisoned"),
    (
        re.compile(
            r"LastUpdate|finalized-chain proof|finalization.*missing|"
            r"timelocked commitment.*missing|archive.*finalized block.*missing|"
            r"exact vector readback.*missing",
            re.I,
        ),
        "weight.finalization_missing",
    ),
    (re.compile(r"PCR0.*(?:mismatch|differs|disagree)|expected_pcr0.*observed_pcr0", re.I), "release.pcr0_mismatch"),
    (re.compile(r"bundle.*(?:mismatch|differs|diverg)|weights.*(?:mismatch|differs|diverg)", re.I), "weight.bundle_divergence"),
    (re.compile(r"timed out|deadline exceeded|did not .* after \d+ attempts", re.I), "restart.stage_deadline_exceeded"),
)

_GATEWAY_STAGE_CODES = {
    "git_prepare": "release.source_tree_mismatch",
    "git_prepared_tree_verification": "release.source_tree_mismatch",
    "worker_import_preflight": "release.source_tree_mismatch",
    "validator_weight_input_storage_preflight": "release.schema_contract_mismatch",
    "v2_release_acquisition": "release.channel_unavailable",
    "docker_disk_cleanup": "release.builder_resource_exhausted",
    "attested_runtime_and_enclave_build": "release.builder_resource_exhausted",
    "v2_offline_artifact_prepare": "release.artifact_auth_failed",
    "v2_credential_envelope_preparation": "release.artifact_auth_failed",
    "dependency_preflight": "authority.dependency_unreadable",
    "v2_pre_shutdown_preflight": "authority.dependency_unreadable",
    "v2_runtime_bootstrap": "runtime.enclave_relay_unavailable",
    "v2_runtime_readiness": "runtime.enclave_relay_unavailable",
    "ancestry_frontier_recovery": "weight.settlement_history_incomplete",
    "ancestry_postcheckpoint": "weight.settlement_history_incomplete",
    "validator_weight_input_repair": "weight.allocation_authority_missing",
    "validator_weight_input_http_check": "weight.gateway_endpoint_unavailable",
    "gateway_health_check": "weight.gateway_endpoint_unavailable",
}

_VALIDATOR_STAGE_CODES = {
    "release_acquisition": "release.channel_unavailable",
    "pre_shutdown_gateway_alignment": "weight.gateway_endpoint_unavailable",
    "runtime_rebuild": "runtime.enclave_relay_unavailable",
    "pre_shutdown_validation": "release.pcr0_mismatch",
    "validator_application_start": "runtime.enclave_relay_unavailable",
    "gateway_alignment": "weight.gateway_endpoint_unavailable",
}

_RELEASE_STAGE_CODES = {
    "stale_cleanup": "release.builder_resource_exhausted",
    "source_checkout": "release.source_tree_mismatch",
    "source_prepare": "release.source_tree_mismatch",
    "schema_preflight": "release.schema_contract_mismatch",
    "host_memory_guard": "release.builder_resource_exhausted",
    "storage_reclaim": "release.builder_resource_exhausted",
    "gateway_validator_build": "release.builder_resource_exhausted",
    "evidence_upload": "release.channel_unavailable",
    "evidence_download": "release.channel_unavailable",
    "manifest_assembly": "release.source_tree_mismatch",
    "immutable_publication": "release.channel_unavailable",
    "final_cleanup": "release.builder_resource_exhausted",
}

_RESTART_MILESTONES = {
    "gateway": (
        "release_ready",
        "pre_shutdown_checks_complete",
        "candidate_activated",
        "attested_runtime_staged",
        "v2_runtime_ready",
        "gateway_base_health_ready",
        "gateway_v2_health_ready",
        "validator_weight_http_handoff_ready",
        "completed",
    ),
    "validator": (
        "release_ready",
        "pre_shutdown_checks_complete",
        "destructive_phase_started",
        "attested_enclave_ready",
        "validator_application_ready",
        "completed",
    ),
}


def _read_bounded_tail(paths: Iterable[Path], limit: int = 64 * 1024) -> str:
    parts = []
    for path in paths:
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > limit:
                    handle.seek(size - limit)
                parts.append(handle.read(limit).decode("utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)[-limit:]


def classify_restart_failure(
    *, component: str, stage: str, evidence_paths: Iterable[Path] = ()
) -> str:
    evidence = _read_bounded_tail(evidence_paths)
    for pattern, code in _SIGNATURE_CODES:
        if pattern.search(evidence):
            return code
    mapping = _GATEWAY_STAGE_CODES if component == "gateway" else _VALIDATOR_STAGE_CODES
    return mapping.get(stage, "restart.terminal_failure")


def failure_code_for_exception(
    exception: BaseException,
    *,
    default: str,
) -> str:
    """Classify a known incident locally without exporting its message."""

    fallback = (
        default
        if default in INCIDENT_FAILURE_CODES
        else "restart.terminal_failure"
    )
    try:
        current: Optional[BaseException] = exception
        seen = set()
        fragments = []
        while current is not None and id(current) not in seen and len(fragments) < 6:
            seen.add(id(current))
            fragments.append(type(current).__name__)
            try:
                fragments.append(str(current)[:500])
            except BaseException:
                pass
            current = current.__cause__ or current.__context__
        evidence = " ".join(fragments)
        for pattern, code in _SIGNATURE_CODES:
            if pattern.search(evidence):
                return code
    except BaseException:
        return fallback
    return fallback


def load_restart_ledger(path: Path) -> list[Dict[str, Any]]:
    """Read only the bounded timing schema produced by the restart scripts."""

    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if len(raw) > 512 * 1024:
        raw = raw[-512 * 1024 :]
    records = []
    for line in raw.decode("utf-8", errors="replace").splitlines()[-160:]:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, Mapping):
            continue
        stage = str(value.get("stage") or "")
        status = str(value.get("status") or "")
        try:
            elapsed = float(value.get("elapsed_seconds"))
        except (TypeError, ValueError):
            continue
        if not _IDENTIFIER_RE.fullmatch(stage) or not _IDENTIFIER_RE.fullmatch(status):
            continue
        records.append({"stage": stage, "status": status, "elapsed_seconds": elapsed})
    previous = 0.0
    for record in records:
        elapsed = max(previous, float(record["elapsed_seconds"]))
        record["duration_seconds"] = round(max(0.0, elapsed - previous), 3)
        previous = elapsed
    return records


def emit_restart_summary(
    *,
    component: str,
    status: str,
    stage: str,
    ledger_path: Path,
    restart_invocation_id: str,
    candidate_sha: Optional[str] = None,
    shutdown_started: bool = False,
    release_attempts: Optional[int] = None,
    evidence_paths: Iterable[Path] = (),
) -> None:
    """Emit bounded restart metrics and one terminal event on failure."""

    records = load_restart_ledger(ledger_path)
    reached = {str(item["stage"]) for item in records}
    expected = _RESTART_MILESTONES.get(component, ())
    missing = [item for item in expected if item not in reached]
    passed = [
        str(item["stage"])
        for item in records
        if item.get("status") in {"passed", "reached", "ready"}
    ]
    last_success = passed[-1] if passed else None
    total = float(records[-1]["elapsed_seconds"]) if records else None
    try:
        stage_deadline = min(
            7200.0,
            max(
                1.0,
                float(
                    os.environ.get(
                        "LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS",
                        "900",
                    )
                ),
            ),
        )
    except (TypeError, ValueError):
        stage_deadline = 900.0
    slowest = max(
        records,
        key=lambda item: float(item.get("duration_seconds") or 0.0),
        default=None,
    )
    configure_sentry_context(
        component=component,
        candidate_sha=candidate_sha,
        runtime_sha=candidate_sha,
        restart_invocation_id=restart_invocation_id,
        correlation_id=restart_invocation_id,
        release_correlation_id=release_correlation_id(candidate_sha),
        stage=stage,
        status=status,
        release_attempts=release_attempts,
    )
    for record in records:
        record_stage(
            component=component,
            stage=str(record["stage"]),
            status=str(record["status"]),
            duration_seconds=float(record["duration_seconds"]),
            restart_invocation_id=restart_invocation_id,
            candidate_sha=candidate_sha,
        )
    if status == "passed":
        sentry_bootstrap.record_sentry_distribution(
            "leadpoet.restart.duration",
            total or 0.0,
            unit="second",
            tags={"component": component, "status": "passed"},
        )
        if slowest is not None and float(slowest["duration_seconds"]) > stage_deadline:
            capture_failure(
                "restart.stage_deadline_exceeded",
                component=component,
                stage=str(slowest["stage"]),
                terminal=True,
                retryable=False,
                fail_closed=False,
                restart_invocation_id=restart_invocation_id,
                candidate_sha=candidate_sha,
                duration_seconds=float(slowest["duration_seconds"]),
                timed_out=False,
                last_successful_stage="completed",
                stage_ledger=records,
                release_attempts=release_attempts,
            )
        return
    code = classify_restart_failure(
        component=component,
        stage=stage,
        evidence_paths=evidence_paths,
    )
    if (
        code == "restart.terminal_failure"
        and slowest is not None
        and float(slowest["duration_seconds"]) > stage_deadline
    ):
        code = "restart.stage_deadline_exceeded"
    capture_failure(
        code,
        component=component,
        stage=stage,
        terminal=True,
        retryable=code in {
            "release.channel_unavailable",
            "authority.dependency_unreadable",
            "weight.gateway_endpoint_unavailable",
        },
        fail_closed=True,
        restart_invocation_id=restart_invocation_id,
        candidate_sha=candidate_sha,
        shutdown_started=shutdown_started,
        duration_seconds=total,
        last_successful_stage=last_success,
        first_unfinished_stage=stage,
        missing_milestones=missing,
        blocked_stages=missing,
        stage_ledger=records,
        release_attempts=release_attempts,
    )


def emit_release_summary(
    *,
    component: str,
    physical_role: str,
    status: str,
    candidate_sha: str,
    stage_statuses: Iterable[tuple[str, str]],
    stage_durations: Optional[Mapping[str, float]] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """Emit one bounded release-job summary without reading build artifacts."""

    try:
        release_id = release_correlation_id(candidate_sha)
        durations = dict(stage_durations or {})
        records = []
        for stage, stage_status in list(stage_statuses)[:40]:
            normalized_stage = str(stage)[:120]
            normalized_status = str(stage_status or "unexercised")[:40]
            if not _IDENTIFIER_RE.fullmatch(normalized_stage):
                continue
            if not _IDENTIFIER_RE.fullmatch(normalized_status):
                normalized_status = "unexercised"
            record: Dict[str, Any] = {
                "stage": normalized_stage,
                "status": normalized_status,
            }
            try:
                stage_duration = float(durations.get(normalized_stage))
            except (TypeError, ValueError):
                stage_duration = None
            if stage_duration is not None and 0.0 <= stage_duration <= 43_200.0:
                record["duration_seconds"] = round(stage_duration, 3)
            records.append(record)
        configure_sentry_context(
            component=component,
            physical_role=physical_role,
            role="release-builder",
            candidate_sha=candidate_sha,
            runtime_sha=candidate_sha,
            correlation_id=release_id,
            release_correlation_id=release_id,
            status=status,
        )
        for record in records:
            record_stage(
                component=component,
                stage=record["stage"],
                status=record["status"],
                duration_seconds=record.get("duration_seconds"),
                candidate_sha=candidate_sha,
                release_correlation_id=release_id,
            )
        if duration_seconds is not None:
            sentry_bootstrap.record_sentry_distribution(
                "leadpoet.release.duration",
                float(duration_seconds),
                unit="second",
                tags={
                    "component": component,
                    "physical_role": physical_role,
                    "status": status,
                },
            )
        if status == "passed":
            return
        failed = next(
            (
                record
                for record in records
                if record["status"] not in {"success", "passed", "skipped"}
            ),
            None,
        )
        stage = str((failed or {}).get("stage") or "release_job")
        code = _RELEASE_STAGE_CODES.get(stage, "restart.terminal_failure")
        passed = [
            record["stage"]
            for record in records
            if record["status"] in {"success", "passed"}
        ]
        blocked = [
            record["stage"]
            for record in records
            if record["status"] in {"skipped", "unexercised", "cancelled"}
        ]
        capture_failure(
            code,
            component=component,
            stage=stage,
            terminal=True,
            retryable=code == "release.channel_unavailable",
            fail_closed=True,
            candidate_sha=candidate_sha,
            release_correlation_id=release_id,
            duration_seconds=duration_seconds,
            last_successful_stage=passed[-1] if passed else None,
            first_unfinished_stage=stage,
            blocked_stages=blocked,
            missing_milestones=blocked,
            stage_ledger=records,
        )
    except BaseException:
        return


def _reset_for_tests() -> None:
    global _event_sequence, _terminal_limiter
    _current_context.set({})
    _terminal_limiter = _TerminalLimiter()
    _event_sequence = 0
