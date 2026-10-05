"""Opt-in, infra-only OpenTelemetry for the gateway HOST process.

Scope is deliberately narrow and privacy-preserving:

- It emits ONE server span per HTTP request capturing only operational
  metadata: request method, the matched route *template* (never the concrete
  path, so ids/hotkeys/epochs do not leak), response status class, and
  duration. It never records request/response bodies, query strings, headers,
  database statements, or any LLM prompt/completion content.
- It does NOT auto-instrument the database, outbound HTTP clients, or the
  Research Lab / scoring / model / Langfuse code paths. Training data and model
  I/O are never touched.
- It runs only in the gateway host process (`python -m gateway.main`), never in
  the attested enclaves, and adds no new dependencies (it uses the OpenTelemetry
  packages already pinned in requirements.txt), so it cannot change any enclave
  measurement / PCR0.

The boundary is CODE-ENFORCED, not discipline-enforced:

1. The exporter destination is passed EXPLICITLY from private, namespaced
   environment variables (``GATEWAY_OTEL_ENDPOINT``, ``GATEWAY_OTEL_TOKEN``).
   A non-empty token is REQUIRED (the pinned exporter falls back to ambient
   header variables when the explicit headers dict is empty), and
   initialization is REFUSED outright if any ambient standard exporter
   variable is present in the runtime environment — the exporter also
   consults those for TLS certificates, client keys, timeout, and
   compression, and a CI grep cannot see variables injected by a restart
   script or the live process environment. The service name is a CONSTANT,
   not an environment value. The provider resource is a fixed attribute set
   (never the SDK's env-merging factory), so ambient resource variables
   cannot leak either.
2. Every span passes a FAIL-CLOSED schema validator that checks the COMPLETE
   span envelope before export: the ``gateway.http`` instrumentation scope
   with no version/schema metadata, ``SERVER`` kind, a root span (no parent,
   empty trace state), EXACTLY the four approved attributes with the
   approved types, a standard HTTP method, a span name equal to exactly
   ``<method> <route>``, no events, no links, no status description, the
   fixed resource, and a route label that is either a registered route
   template or the literal ``/_unmatched``. Anything else is DROPPED (never
   mutated, never exported) and counted in a warning that does not include
   the rejected values.
3. ``tests/test_otel_boundary_guard.py`` fails CI if the boundary is crossed
   anywhere in the repo (auto-instrumentation packages, the
   process-wrapping launcher, a global tracer provider, ambient exporter
   or resource env vars, the env-merging resource factory in this module,
   or OTLP exporter imports outside this module).

The same module also wires the **Lab Arena sidecar** (``leadpoet-arena``),
which runs the Arena competition pipeline in its own process. It reuses the
identical fail-closed machinery under two additional scopes:

- ``arena.http`` — one SERVER span per sidecar request, carrying the SAME four
  attributes. At the gateway every Arena call collapses into the catch-all
  ``/arena/{arena_path:path}`` template, so the operation that was invoked is
  not recoverable from gateway telemetry at all; the sidecar's own route
  templates restore it without exporting a single client-controlled string.
- ``arena.task`` — one INTERNAL span per driver / worker stage, so a stalled
  pipeline stops being indistinguishable from an idle one. Its attribute set
  is a separate, equally fixed allowlist: a stage name from a frozen
  vocabulary, an outcome from a frozen vocabulary, an exception CLASS name
  (code-controlled, shape-validated) and a bounded integer count. No round id,
  submission id, hotkey, source, prompt, score, or model output can pass it.

Both Arena scopes share the gateway's destination (the same private
``GATEWAY_OTEL_*`` values, read from the same protected env file) so enabling
them requires no new secret and no new host configuration.

It is a complete no-op unless ``GATEWAY_OTEL_ENABLED`` is truthy AND
``GATEWAY_OTEL_ENDPOINT`` AND ``GATEWAY_OTEL_TOKEN`` are all set. Any failure
while wiring it up is swallowed so it can never delay or break gateway
startup. No endpoint or token is ever hard-coded or committed.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

_TRUTHY = {"1", "true", "yes", "on"}

# Route templates whose spans are suppressed entirely — health/liveness noise.
_SUPPRESSED_ROUTES = {"/health", "/health/live", "/health/ready"}

# Route label used whenever the request did not resolve to a registered route
# template. A client-controlled path segment must NEVER be exported.
UNMATCHED_ROUTE_LABEL = "/_unmatched"

# The instrumentation scope every exported span must carry.
INSTRUMENTATION_SCOPE = "gateway.http"

# Fixed service identity — a constant, never an environment value.
SERVICE_NAME = "leadpoet-gateway"

# The Arena sidecar is a DIFFERENT process with a DIFFERENT fixed identity.
# Never let an Arena reading inherit the gateway's service name.
ARENA_SERVICE_NAME = "leadpoet-arena"

# The Arena sidecar's two scopes: request envelopes and pipeline stages.
ARENA_HTTP_SCOPE = "arena.http"
ARENA_TASK_SCOPE = "arena.task"

# Every pipeline stage that may be reported, as a FROZEN vocabulary. A stage
# name outside this set is dropped whole, so a future caller cannot smuggle a
# computed or client-derived string out through the stage label.
ARENA_TASK_STAGES = frozenset(
    {
        "driver_tick",
        "promote_baselines",
        "active_rounds",
        "advance_round",
        "ensure_daily_round",
        "activate_rewards",
        "reconcile_provider_costs",
        "review_submissions",
    }
)

# The only outcomes a stage may report.
ARENA_TASK_OUTCOMES = frozenset({"ok", "idle", "failed"})

# An exception CLASS name is code-controlled, not client-controlled, but it is
# still shape-checked so no message, path, or value can ever ride along.
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
ARENA_NO_ERROR = "-"

# Upper bound on a reported count: an operational magnitude, never a payload.
_ARENA_MAX_COUNT = 1_000_000

# ---------------------------------------------------------------------------
# Arena provider calls, run outcomes, and provider-gate contention
# ---------------------------------------------------------------------------
#
# Each of these is a SEPARATE envelope with its own exact attribute allowlist
# and its own frozen vocabularies. Every vocabulary below is a copy of a
# server-side constant that already exists in ``lab_arena`` — copied rather
# than imported so this module stays importable with nothing but the standard
# library, and so a change to the pipeline can never silently widen what is
# exported. ``tests/test_otel_boundary_guard.py`` asserts the copies match.

ARENA_PROVIDER_SCOPE = "arena.provider"
ARENA_RUN_SCOPE = "arena.run"
ARENA_GATE_SCOPE = "arena.gate"

# ``lab_arena.contracts.PROVIDERS`` plus the fixed label used when an
# operation is not in the table at all.
ARENA_PROVIDER_UNKNOWN = "unknown"
ARENA_PROVIDERS = frozenset(
    {"scrapingdog", "deepline", "openrouter", "unknown"}
)

# How a provider call ended, from the gateway's point of view:
#   ok        — the provider answered and the ledger settled
#   refused   — the call was declined before or at the provider (budget, cap,
#               policy, credentials) and the ledger settled
#   uncertain — the ledger could not be settled; cost is reconciled later
#   failed    — the broker raised instead of returning a result
ARENA_PROVIDER_OUTCOMES = frozenset({"ok", "refused", "uncertain", "failed"})

# Every error code the broker and the operation table can produce, as a frozen
# vocabulary. A code outside this set is dropped whole.
ARENA_PROVIDER_ERROR_CODES = frozenset(
    {
        # broker._error_result / BrokerError
        "broker_unavailable",
        "budget_refused",
        "call_refused",
        "call_uncertain",
        "invalid_request",
        "lease_stale",
        "miner_credentials_unavailable",
        "miner_provider_not_configured",
        "model_not_allowed",
        "provider_request_refused",
        "provider_unavailable",
        # operations.ERROR_CODES
        "no_matching_operation",
        "request_too_large",
        "invalid_body",
        "invalid_query",
        "unknown_header",
        "forbidden_header",
        "unknown_field",
        "forbidden_field",
        "missing_field",
        "invalid_field",
        "invalid_url",
        "invalid_response",
        "response_too_large",
    }
)

# ``lab_arena.contracts.TERMINAL_CAUSES`` — how one evaluation run ended.
ARENA_TERMINAL_CAUSES = frozenset(
    {
        "accepted",
        "model_timeout",
        "invalid_output",
        "budget_exhausted",
        "credential_error",
        "model_error",
        "lease_expired",
        "worker_lost",
        "result_rejected",
        "provider_error",
        "stage_closed",
        "judge_error",
        "judge_timeout",
    }
)

# A run is either a miner execution or an Arena judge scoring pass.
ARENA_RUN_KINDS = frozenset({"execute", "score"})

# How a wait on the shared OpenRouter concurrency gate ended. A call admitted
# with no wait at all emits nothing, so gate spans appear only under
# contention.
ARENA_GATE_OUTCOMES = frozenset(
    {"admitted_after_wait", "timed_out", "cancelled", "no_capacity"}
)

# Exact operation-to-provider pairs from ``lab_arena.operations.OPERATIONS``.
# Exa operations use Deepline, so the operation prefix is not the provider.
# The service must reduce unknown caller-supplied operation ids to "unknown".
ARENA_OPERATION_PROVIDERS = {
    "deepline.execute": "deepline",
    "exa.contents": "deepline",
    "exa.search": "deepline",
    "openrouter.chat": "openrouter",
    "openrouter.responses": "openrouter",
    "scrapingdog.google": "scrapingdog",
    "scrapingdog.google_jobs": "scrapingdog",
    "scrapingdog.google_news": "scrapingdog",
    "scrapingdog.indeed": "scrapingdog",
    "scrapingdog.instagram_profile": "scrapingdog",
    "scrapingdog.jobs": "scrapingdog",
    "scrapingdog.linkedinjobs": "scrapingdog",
    "scrapingdog.profile": "scrapingdog",
    "scrapingdog.profile_post": "scrapingdog",
    "scrapingdog.scrape": "scrapingdog",
    "scrapingdog.tiktok_profile": "scrapingdog",
    "scrapingdog.x_post": "scrapingdog",
    "scrapingdog.x_profile": "scrapingdog",
    "scrapingdog.youtube_channel": "scrapingdog",
    "scrapingdog.youtube_search": "scrapingdog",
    "scrapingdog.youtube_transcripts": "scrapingdog",
    "scrapingdog.youtube_video": "scrapingdog",
}
ARENA_OPERATION_UNKNOWN = "unknown"

# A refusal code is the literal prefix of a ``ServiceError`` code, i.e. the
# part before the first ":". The suffix can carry an exception message and is
# never exported; the prefix is always a literal written in the source.
_ARENA_DENIAL_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
ARENA_NO_DENIAL = "-"

# Only explicit source-defined refusal codes may leave the sidecar. A shape
# check alone would admit a caller-controlled identifier such as "private_key".
ARENA_DENIAL_CODES = frozenset(
    {
        "accepted_weight_epoch_not_current",
        "accepted_weight_epoch_scope_unavailable",
        "accepted_weight_state_conflict",
        "accepted_weight_state_invalid",
        "accepted_weight_state_reward_conflict",
        "arena_schema_version_invalid",
        "arena_store_unavailable",
        "baseline_cost_eligibility_schema_invalid",
        "baseline_cost_eligibility_schema_unavailable",
        "baseline_hotkey_missing",
        "baseline_hotkey_reserved",
        "baseline_promoter_unavailable",
        "baseline_promotion_pending",
        "baseline_registration_failed",
        "baseline_source_fetcher_missing",
        "baseline_source_invalid",
        "baseline_source_not_ready",
        "baseline_source_rejected",
        "baseline_submission_invalid",
        "benchmark_data_invalid",
        "benchmark_disclosure_activation_invalid",
        "benchmark_disclosure_policy_invalid",
        "benchmark_icp_count_invalid",
        "benchmark_not_committed",
        "benchmark_not_public",
        "cancel_reason_invalid",
        "chain_outcome_conflict",
        "chain_outcome_scope_mismatch",
        "chain_outcome_signature_invalid",
        "chain_outcome_state_unknown",
        "champion_funding_freeze_failed",
        "champion_funding_schema_unavailable",
        "champion_funding_state_invalid",
        "closed_provider_cost_candidate_invalid",
        "code_review_pending",
        "code_review_required",
        "code_review_schema_unavailable",
        "code_review_unavailable",
        "company_judgment_cache_invalid",
        "company_judgment_output_invalid",
        "company_quality_activation_invalid",
        "company_quality_requires_integrity",
        "company_quality_schema_invalid",
        "company_quality_schema_unavailable",
        "company_quality_scorer_policy_mismatch",
        "contact_activation_invalid",
        "contact_policy_requires_integrity",
        "contact_schema_invalid",
        "contact_schema_unavailable",
        "contract",
        "credential_validation_unavailable",
        "daily_cutoff_hour_invalid",
        "daily_icp_source_invalid",
        "daily_round_dates_exhausted",
        "daily_runner_capacity_insufficient",
        "declared_parallelism_invalid",
        "dynamic_benchmark_schema_unavailable",
        "execution_sequence_activation_invalid",
        "frame_invalid",
        "function_probe_invalid",
        "function_unavailable",
        "hotkey_banned",
        "hotkey_unregistered",
        "integrity_activation_invalid",
        "integrity_schema_invalid",
        "integrity_schema_unavailable",
        "integrity_scorer_policy_mismatch",
        "intent_details_activation_invalid",
        "intent_details_policy_requires_integrity",
        "intent_details_scorer_policy_mismatch",
        "judgment_cache_authority_invalid",
        "judgment_cache_invalid",
        "lease_expired",
        "lease_inactive",
        "lease_invalid",
        "lease_stale",
        "lease_token_invalid",
        "mode_invalid",
        "mode_off",
        "netuid_invalid",
        "network_name_invalid",
        "object_store_mismatch",
        "object_store_unavailable",
        "output_invalid",
        "parallel_execution_schema_unavailable",
        "parallel_twenty_icp_execution_invalid",
        "participant_missing",
        "participant_source_missing",
        "per_icp_cost_schema_unavailable",
        "pinned_round_id_invalid",
        "promotion_commit_mismatch",
        "promotion_margin_invalid",
        "promotion_plan_missing",
        "promotion_round_missing",
        "promotion_source_size_mismatch",
        "promotion_winner_invalid",
        "public_output_unavailable",
        "public_result_unavailable",
        "quota_unavailable",
        "request_invalid",
        "results_not_public",
        "reward_history_invalid",
        "reward_signer_unavailable",
        "round_cost_policy_missing",
        "round_create_failed",
        "round_ended",
        "round_evaluation_date_invalid",
        "round_missing",
        "round_mode_mismatch",
        "round_network_mismatch",
        "round_scope_mismatch",
        "round_unknown",
        "run_missing",
        "run_result_budget_unproved",
        "run_result_cause_kind_mismatch",
        "run_result_invalid",
        "run_round_mismatch",
        "run_runner_mismatch",
        "run_source_integrity_failed",
        "run_source_unavailable",
        "runner_banned",
        "runner_benchmark_eligibility_unavailable",
        "runner_hotkey_unregistered",
        "runner_slot_ceiling_invalid",
        "runner_validator_authority_unavailable",
        "runner_validator_required",
        "score_invalid",
        "scored_run_missing",
        "scorer_image_access_unavailable",
        "scorer_image_access_unsupported",
        "scorer_image_invalid",
        "scores_not_recorded",
        "scoring_plan_invalid",
        "scoring_plan_missing",
        "scoring_plan_run_mismatch",
        "signature_invalid",
        "source_upload_not_configured",
        "source_upload_unavailable",
        "stage_invalid",
        "submission_conflict",
        "submission_credentials_immutable",
        "submission_credentials_missing",
        "submission_finalize_failed",
        "submission_id_mismatch",
        "submission_missing",
        "submission_not_uploading",
        "submission_owner_changed",
        "submission_rate_limited",
        "submission_registration_failed",
        "submission_rejected",
        "submission_replacement_closed",
        "submission_replacement_ineligible",
        "submission_replacement_limit_reached",
        "submission_replacement_schema_unavailable",
        "submission_superseded",
        "submission_transport_mismatch",
        "submission_window_closed",
        "successful_call_cost_schema_unavailable",
        "table_unavailable",
        "trajectory_invalid",
        "trajectory_unavailable",
        "unsupported_integrity_policy",
        "validator_checkpoint_upgrade_required",
        "validator_output_schema_upgrade_required",
        "validator_proxy_execution_upgrade_required",
        "validator_scoring_authority_schema_unavailable",
        "validator_snapshot_unavailable",
        "weight_state_request_invalid",
        "weight_state_scope_mismatch",
    }
)


# Upper bound on a reported cost. Micro-USD, so this is $1,000 per call.
_ARENA_MAX_MICROUSD = 1_000_000_000

# Upper bound on reported provider attempts (champion credential retries).
_ARENA_MAX_ATTEMPTS = 4

# Fixed batching limits. Passing every value explicitly prevents ambient
# ``OTEL_BSP_*`` variables from changing gateway memory or shutdown behavior.
_BATCH_MAX_QUEUE_SIZE = 2048
_BATCH_SCHEDULE_DELAY_MILLIS = 5000
_BATCH_MAX_EXPORT_BATCH_SIZE = 512
_BATCH_EXPORT_TIMEOUT_MILLIS = 30000

# The standard exporter env-var prefix that must NOT exist at runtime. The
# pinned exporter consults these for headers, TLS certificates, client keys,
# timeout, and compression even when endpoint/headers are passed explicitly.
# (Split literal so the repo-wide CI grep for the ambient prefix stays
# meaningful everywhere else.)
_AMBIENT_EXPORTER_PREFIX = "OTEL_" + "EXPORTER_"

# Standard HTTP request methods; anything else is a client-controlled string.
_ALLOWED_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)

# The ONLY attributes a span may carry out of this process, with their
# required types (bool is explicitly rejected for the numeric fields).
SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "duration_ms": (int, float),
}

SPAN_ATTRIBUTE_ALLOWLIST = frozenset(SPAN_ATTRIBUTE_TYPES)

# The ONLY attributes an Arena pipeline-stage span may carry. A separate,
# equally exact allowlist — the HTTP four do not apply to a task span.
TASK_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.stage": (str,),
    "arena.outcome": (str,),
    "arena.error_type": (str,),
    "arena.count": (int,),
    "duration_ms": (int, float),
}

TASK_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(TASK_SPAN_ATTRIBUTE_TYPES)

# The Arena sidecar's HTTP envelope: the gateway's four, plus the refusal code
# the sidecar itself produced. The gateway's own envelope is UNCHANGED — this
# fifth attribute exists only on the sidecar, where a 4xx with no reason is
# the difference between "a runner was denied a claim" and "a runner is
# broken". Its vocabulary is the literal prefix of a ``ServiceError`` code.
ARENA_HTTP_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "arena.denial": (str,),
    "duration_ms": (int, float),
}

ARENA_HTTP_ATTRIBUTE_ALLOWLIST = frozenset(ARENA_HTTP_ATTRIBUTE_TYPES)

# The ONLY attributes a provider-call span may carry. Every text field is a
# member of a frozen vocabulary above; every numeric field is bounded. No
# prompt, completion, model output, miner hotkey, run id, submission id,
# credential, URL, or response body can pass this allowlist.
PROVIDER_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.provider": (str,),
    "arena.operation": (str,),
    "arena.outcome": (str,),
    "arena.error_code": (str,),
    "arena.error_type": (str,),
    "arena.http_status": (int,),
    "arena.provider_status": (int,),
    "arena.attempts": (int,),
    "arena.cost_microusd": (int,),
    "duration_ms": (int, float),
}

PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(PROVIDER_SPAN_ATTRIBUTE_TYPES)

# The ONLY attributes a run-outcome span may carry. There is deliberately no
# duration here: the sidecar records when a run ENDS, and the run row carries
# no start time, so any duration would be invented rather than measured.
RUN_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.run_kind": (str,),
    "arena.terminal_cause": (str,),
    "arena.outcome": (str,),
}

RUN_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(RUN_SPAN_ATTRIBUTE_TYPES)

# The ONLY attributes a provider-gate span may carry.
GATE_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.gate_outcome": (str,),
    "duration_ms": (int, float),
}

GATE_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(GATE_SPAN_ATTRIBUTE_TYPES)

# Log tokens, written out in full so the CI boundary guard can grep them.
_GATEWAY_LOG = {
    "refused": "gateway_otel_bootstrap_refused",
    "skipped": "gateway_otel_bootstrap_skipped",
    "dropped": "gateway_otel_span_dropped",
    "emit_failed": "gateway_otel_emit_failed",
}
_ARENA_LOG = {
    "refused": "arena_otel_bootstrap_refused",
    "skipped": "arena_otel_bootstrap_skipped",
    "dropped": "arena_otel_span_dropped",
    "emit_failed": "arena_otel_emit_failed",
}


def _enabled() -> bool:
    return (
        os.getenv("GATEWAY_OTEL_ENABLED", "").strip().lower() in _TRUTHY
        and bool(os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip())
    )


def _ambient_exporter_env_names() -> List[str]:
    return sorted(k for k in os.environ if k.startswith(_AMBIENT_EXPORTER_PREFIX))


def _safe_route_label(scope_or_request: Any) -> str:
    """Return the low-cardinality route template, never the concrete path.

    Using the matched route template (e.g. ``/research-lab/allocations/attested/
    {epoch}``) keeps ids, hotkeys, and epoch numbers out of telemetry. When the
    template cannot be resolved the label is the fixed ``/_unmatched`` literal —
    a client-controlled path segment is never exported.
    """
    try:
        scope = getattr(scope_or_request, "scope", scope_or_request)
        route = scope.get("route")
        template = getattr(route, "path", None)
        if isinstance(template, str) and template:
            return template
    except Exception:
        pass
    return UNMATCHED_ROUTE_LABEL


def _span_scope(span: Any) -> Any:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None:  # older SDK naming
        scope = getattr(span, "instrumentation_info", None)
    return scope


def _attributes_conform(
    attributes: Dict[str, Any],
    types: Optional[Dict[str, tuple]] = None,
) -> bool:
    required = SPAN_ATTRIBUTE_TYPES if types is None else types
    if set(attributes) != set(required):
        return False
    for key, allowed_types in required.items():
        value = attributes[key]
        if isinstance(value, bool) or not isinstance(value, allowed_types):
            return False
    return True


class _GatewayOtelMiddleware:
    """Raw-ASGI request telemetry that cannot affect request behavior."""

    def __init__(
        self,
        app: Any,
        *,
        emit: Callable[..., None],
        on_request_start: Optional[Callable[[], None]] = None,
    ) -> None:
        self.app = app
        self._emit = emit
        # Optional per-request reset, so a value recorded while serving one
        # request can never be read back while serving the next one.
        self._on_request_start = on_request_start

    def _emit_safely(
        self,
        scope: Dict[str, Any],
        status_code: int,
        start_ns: int,
        start_mono: float,
    ) -> None:
        try:
            self._emit(scope, status_code, start_ns, start_mono)
        except BaseException as exc:
            # Telemetry is observational only. Log the exception type without
            # values that could contain request or exporter data.
            try:
                print(
                    "gateway_otel_emit_failed error=%s"
                    % type(exc).__name__,
                    flush=True,
                )
            except BaseException:
                pass

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") in _SUPPRESSED_ROUTES:
            await self.app(scope, receive, send)
            return

        if self._on_request_start is not None:
            try:
                self._on_request_start()
            except BaseException:
                # Telemetry setup must never refuse to serve a request.
                pass

        start_ns = time.time_ns()
        start_mono = time.monotonic()
        status_code = 500
        response_started = False

        async def _send_with_status(message: Dict[str, Any]) -> None:
            nonlocal response_started, status_code
            if message.get("type") == "http.response.start":
                observed = message.get("status")
                if isinstance(observed, int) and not isinstance(observed, bool):
                    status_code = observed
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, _send_with_status)
        except BaseException:
            self._emit_safely(
                scope,
                status_code if response_started else 500,
                start_ns,
                start_mono,
            )
            raise
        self._emit_safely(scope, status_code, start_ns, start_mono)


def _validating_exporter(
    delegate: Any,
    *,
    allowed_routes: Callable[[], set],
    expected_resource: Dict[str, Any],
    scopes: Optional[Dict[str, str]] = None,
    log: Optional[Dict[str, str]] = None,
) -> Any:
    """Wrap a span exporter in a FAIL-CLOSED complete-envelope validator.

    A span is exported only when every check passes; a non-conforming span is
    dropped entirely — never mutated, never partially exported. Drops are
    logged as a per-reason count WITHOUT the rejected values.

    ``scopes`` maps each ACCEPTED instrumentation scope to its envelope kind.
    Each kind has its own exact attribute allowlist and its own frozen
    vocabularies:

    ``http``        SERVER span, the four approved gateway HTTP attributes
    ``arena_http``  SERVER span, those four plus the Arena refusal code
    ``task``        INTERNAL span, the five approved pipeline-stage attributes
    ``provider``    INTERNAL span, the ten approved provider-call attributes
    ``run``         INTERNAL span, the four approved run-outcome attributes
    ``gate``        INTERNAL span, the two approved provider-gate attributes

    A span whose scope is not listed is dropped, so one process can never emit
    through another's envelope and one envelope can never borrow another's
    allowlist.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import SpanKind

    accepted_scopes = (
        {INSTRUMENTATION_SCOPE: "http"} if scopes is None else dict(scopes)
    )
    tokens = _GATEWAY_LOG if log is None else log

    def _http_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes):
            return "attributes"
        method = attributes["http.request.method"]
        if method not in _ALLOWED_HTTP_METHODS:
            return "method"
        route = attributes["http.route"]
        if span.name != "%s %s" % (method, route):
            return "name"
        if route != UNMATCHED_ROUTE_LABEL and route not in allowed_routes():
            return "route"
        return None

    def _arena_http_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, ARENA_HTTP_ATTRIBUTE_TYPES):
            return "attributes"
        denial = attributes["arena.denial"]
        if denial != ARENA_NO_DENIAL and (
            not _ARENA_DENIAL_RE.fullmatch(denial)
            or denial not in ARENA_DENIAL_CODES
        ):
            return "denial"
        trimmed = {
            key: value
            for key, value in attributes.items()
            if key != "arena.denial"
        }
        return _http_violation(span, trimmed)

    def _provider_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, PROVIDER_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        provider = attributes["arena.provider"]
        if provider not in ARENA_PROVIDERS:
            return "provider"
        operation = attributes["arena.operation"]
        if operation != ARENA_OPERATION_UNKNOWN and ARENA_OPERATION_PROVIDERS.get(operation) != provider:
            return "operation"
        if attributes["arena.outcome"] not in ARENA_PROVIDER_OUTCOMES:
            return "outcome"
        error_code = attributes["arena.error_code"]
        if (
            error_code != ARENA_NO_ERROR
            and error_code not in ARENA_PROVIDER_ERROR_CODES
        ):
            return "error_code"
        error_type = attributes["arena.error_type"]
        if error_type != ARENA_NO_ERROR and not _ERROR_TYPE_RE.fullmatch(error_type):
            return "error_type"
        for key, ceiling in (
            ("arena.http_status", 999),
            ("arena.provider_status", 999),
            ("arena.cost_microusd", _ARENA_MAX_MICROUSD),
        ):
            value = attributes[key]
            if value < 0 or value > ceiling:
                return key.split(".", 1)[-1]
        attempts = attributes["arena.attempts"]
        if attempts < 1 or attempts > _ARENA_MAX_ATTEMPTS:
            return "attempts"
        if span.name != "arena.provider.%s" % provider:
            return "name"
        return None

    def _run_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, RUN_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        run_kind = attributes["arena.run_kind"]
        if run_kind not in ARENA_RUN_KINDS:
            return "run_kind"
        if attributes["arena.terminal_cause"] not in ARENA_TERMINAL_CAUSES:
            return "terminal_cause"
        if attributes["arena.outcome"] not in ARENA_TASK_OUTCOMES:
            return "outcome"
        if span.name != "arena.run.%s" % run_kind:
            return "name"
        return None

    def _gate_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, GATE_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        if attributes["arena.gate_outcome"] not in ARENA_GATE_OUTCOMES:
            return "gate_outcome"
        if span.name != "arena.gate":
            return "name"
        return None

    def _task_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, TASK_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        stage = attributes["arena.stage"]
        if stage not in ARENA_TASK_STAGES:
            return "stage"
        if attributes["arena.outcome"] not in ARENA_TASK_OUTCOMES:
            return "outcome"
        error_type = attributes["arena.error_type"]
        if error_type != ARENA_NO_ERROR and not _ERROR_TYPE_RE.fullmatch(error_type):
            return "error_type"
        count = attributes["arena.count"]
        if count < 0 or count > _ARENA_MAX_COUNT:
            return "count"
        if span.name != "arena.%s" % stage:
            return "name"
        return None

    _ENVELOPE_CHECKERS = {
        "http": _http_violation,
        "arena_http": _arena_http_violation,
        "task": _task_violation,
        "provider": _provider_violation,
        "run": _run_violation,
        "gate": _gate_violation,
    }

    def _violation(span: Any) -> Optional[str]:
        try:
            scope = _span_scope(span)
            envelope = accepted_scopes.get(str(getattr(scope, "name", "") or ""))
            if envelope is None:
                return "scope"
            if getattr(scope, "version", None) or getattr(scope, "schema_url", None):
                return "scope_metadata"
            expected_kind = (
                SpanKind.SERVER
                if envelope in ("http", "arena_http")
                else SpanKind.INTERNAL
            )
            if getattr(span, "kind", None) is not expected_kind:
                return "kind"
            if getattr(span, "parent", None) is not None:
                return "parent"
            context = getattr(span, "context", None) or span.get_span_context()
            trace_state = getattr(context, "trace_state", None)
            if trace_state is not None and len(trace_state) > 0:
                return "trace_state"
            attributes = dict(span.attributes or {})
            checker = _ENVELOPE_CHECKERS.get(envelope)
            if checker is None:
                return "envelope"
            shape = checker(span, attributes)
            if shape is not None:
                return shape
            if getattr(span, "events", None):
                return "events"
            if getattr(span, "links", None):
                return "links"
            status = getattr(span, "status", None)
            if status is not None and getattr(status, "description", None):
                return "status_description"
            resource = getattr(span, "resource", None)
            if dict(getattr(resource, "attributes", {}) or {}) != expected_resource:
                return "resource"
            return None
        except Exception:
            # Fail closed: a span we cannot fully validate is never exported.
            return "validation_error"

    class _FailClosedSpanExporter(SpanExporter):
        def export(self, spans):  # type: ignore[override]
            accepted = []
            dropped: Dict[str, int] = {}
            for span in spans:
                reason = _violation(span)
                if reason is None:
                    accepted.append(span)
                else:
                    dropped[reason] = dropped.get(reason, 0) + 1
            if dropped:
                try:
                    # Reasons and counts only — never the rejected values.
                    print(
                        "%s %s"
                        % (
                            tokens["dropped"],
                            " ".join("%s=%d" % kv for kv in sorted(dropped.items())),
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
            if not accepted:
                return SpanExportResult.SUCCESS
            return delegate.export(accepted)

        def shutdown(self) -> None:
            try:
                delegate.shutdown()
            except Exception:
                pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            try:
                return bool(delegate.force_flush(timeout_millis))
            except Exception:
                return True

    return _FailClosedSpanExporter()


def _build_otlp_exporter(log: Dict[str, str]) -> Optional[Any]:
    """Build the single OTLP exporter, or None when it must stay off.

    The destination is passed EXPLICITLY from the private ``GATEWAY_OTEL_*``
    variables; ambient exporter variables are never consulted. Both the
    gateway and the Arena sidecar share this one construction site, which is
    what keeps ``tests/test_otel_boundary_guard.py``'s "exporters live in one
    module" guarantee true.
    """
    token = os.getenv("GATEWAY_OTEL_TOKEN", "").strip()
    if not token:
        # An empty explicit headers dict would let the pinned exporter
        # fall back to ambient header variables — require the token.
        try:
            print(log["refused"] + " token_missing", flush=True)
        except Exception:
            pass
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    endpoint = os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip()
    headers = {"Authorization": "Bearer " + token}
    # Explicit arguments only: the destination exists solely inside
    # this exporter object, never in ambient environment variables.
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _build_span_processor(validating_exporter: Any, *, simple: bool) -> Any:
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    if simple:
        return SimpleSpanProcessor(validating_exporter)
    return BatchSpanProcessor(
        validating_exporter,
        max_queue_size=_BATCH_MAX_QUEUE_SIZE,
        schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MILLIS,
        max_export_batch_size=_BATCH_MAX_EXPORT_BATCH_SIZE,
        export_timeout_millis=_BATCH_EXPORT_TIMEOUT_MILLIS,
    )


def configure_gateway_otel(app: Any, *, span_exporter: Optional[Any] = None) -> bool:
    """Wire infra-only request spans onto the gateway app. No-op unless enabled.

    Returns True if instrumentation was installed, False otherwise.
    ``span_exporter`` is a test seam; production always builds the OTLP
    exporter with an EXPLICIT endpoint + headers from the private
    ``GATEWAY_OTEL_*`` variables (measure A) — ambient exporter variables
    are never read, and their presence at runtime refuses initialization.
    """
    if span_exporter is None and not _enabled():
        return False
    ambient = _ambient_exporter_env_names()
    if ambient:
        # A CI grep cannot see variables injected by a restart script or the
        # live process environment; refuse at runtime instead. Names only —
        # never the values.
        try:
            print(
                "gateway_otel_bootstrap_refused ambient_exporter_env=%s"
                % ",".join(ambient),
                flush=True,
            )
        except Exception:
            pass
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind, StatusCode, Status
        from opentelemetry.context import Context

        # FIXED resource: the plain constructor takes exactly these attributes.
        # The SDK's env-merging factory is deliberately avoided so ambient
        # resource variables can never leak into exported metadata.
        expected_resource = {"service.name": SERVICE_NAME}
        resource = Resource(expected_resource)

        def _registered_routes() -> set:
            try:
                return {
                    template
                    for template in (
                        getattr(route, "path", None)
                        for route in getattr(app, "routes", [])
                    )
                    if isinstance(template, str) and template
                }
            except Exception:
                return set()

        delegate = span_exporter
        if delegate is None:
            delegate = _build_otlp_exporter(_GATEWAY_LOG)
            if delegate is None:
                return False
        validating_exporter = _validating_exporter(
            delegate,
            allowed_routes=_registered_routes,
            expected_resource=expected_resource,
            scopes={INSTRUMENTATION_SCOPE: "http"},
            log=_GATEWAY_LOG,
        )
        processor = _build_span_processor(
            validating_exporter,
            simple=span_exporter is not None,
        )

        # A dedicated provider, NOT the global one, so nothing else (e.g. Langfuse)
        # is affected and no ambient instrumentation is picked up.
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(INSTRUMENTATION_SCOPE)

        def _emit(scope, status_code, start_ns, start_mono):
            # Resolve the route template only after routing has run, so ids stay
            # out of the label. Suppression already handled static health paths.
            route = _safe_route_label(scope)
            method = str(scope.get("method") or "")
            span = tracer.start_span(
                "%s %s" % (method, route),
                kind=SpanKind.SERVER,
                # A fresh root context: never adopt a caller-supplied parent
                # or trace state.
                context=Context(),
                start_time=start_ns,
            )
            # Only method / route template / status / duration — nothing else.
            span.set_attribute("http.request.method", method)
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", int(status_code))
            span.set_attribute("duration_ms", (time.monotonic() - start_mono) * 1000.0)
            if int(status_code) >= 500:
                span.set_status(Status(StatusCode.ERROR))
            span.end()

        # Register a raw ASGI middleware. FastAPI's ``@app.middleware("http")``
        # silently wraps the function in BaseHTTPMiddleware and changes
        # exception, cancellation, streaming, and context behavior.
        app.add_middleware(_GatewayOtelMiddleware, emit=_emit)

        return True
    except Exception as exc:  # never let telemetry break the gateway
        try:
            # The exception message may contain a configured endpoint or
            # exporter detail. Log only its class.
            print(
                "gateway_otel_bootstrap_skipped error=%s"
                % type(exc).__name__,
                flush=True,
            )
        except Exception:
            pass
        return False


class ArenaTelemetry:
    """Records Arena spans. Never raises, never blocks, never reorders work.

    Four INTERNAL envelopes, each a root span with its own exact allowlist:
    pipeline stages, provider calls, run outcomes, and provider-gate waits.
    The caller supplies values from the frozen vocabularies; anything else is
    dropped whole by the same fail-closed validator that guards the HTTP
    envelope. Every public method swallows its own failures — an exporter
    problem can never surface as an Arena failure.
    """

    def __init__(
        self,
        tracer: Any,
        *,
        provider_tracer: Any = None,
        run_tracer: Any = None,
        gate_tracer: Any = None,
    ) -> None:
        self._tracer = tracer
        self._provider_tracer = provider_tracer
        self._run_tracer = run_tracer
        self._gate_tracer = gate_tracer
        # Per-request refusal code, read back by the Arena HTTP middleware.
        # A raw ASGI chain shares one context per request, so this is set and
        # read within a single request and never crosses between them.
        import contextvars

        # A worker thread receives a copy of the request context. Sharing one
        # mutable value lets a sync route report its denial back to ASGI.
        self._denial = contextvars.ContextVar("arena_denial", default=None)

    # -- refusal code ----------------------------------------------------

    def note_denial(self, code: str) -> None:
        """Record the refusal code for the request currently being served."""
        try:
            state = self._denial.get()
            if state is None:
                state = [ARENA_NO_DENIAL]
                self._denial.set(state)
            state[0] = str(code)
        except BaseException:
            pass

    def reset_denial(self) -> None:
        try:
            self._denial.set([ARENA_NO_DENIAL])
        except BaseException:
            pass

    def take_denial(self) -> str:
        """Return the refusal code's exported form, or the fixed no-denial mark.

        Only the literal prefix before the first ":" is ever returned; a
        ``ServiceError`` code can interpolate an exception message after it.
        """
        try:
            state = self._denial.get()
            raw = str(state[0]) if state is not None else ""
        except BaseException:
            return ARENA_NO_DENIAL
        prefix = raw.split(":", 1)[0]
        if _ARENA_DENIAL_RE.fullmatch(prefix) and prefix in ARENA_DENIAL_CODES:
            return prefix
        return ARENA_NO_DENIAL

    # -- provider calls --------------------------------------------------

    def record_provider(
        self,
        provider: str,
        operation: str,
        outcome: str,
        *,
        error_code: str = ARENA_NO_ERROR,
        error_type: str = ARENA_NO_ERROR,
        http_status: int = 0,
        provider_status: int = 0,
        attempts: int = 1,
        cost_microusd: int = 0,
        duration_ms: float = 0.0,
        start_ns: Optional[int] = None,
    ) -> None:
        try:
            self._record_provider(
                provider,
                operation,
                outcome,
                error_code=error_code,
                error_type=error_type,
                http_status=http_status,
                provider_status=provider_status,
                attempts=attempts,
                cost_microusd=cost_microusd,
                duration_ms=duration_ms,
                start_ns=start_ns,
            )
        except BaseException as exc:
            self._log_emit_failure(exc)

    def _record_provider(
        self,
        provider: str,
        operation: str,
        outcome: str,
        *,
        error_code: str,
        error_type: str,
        http_status: int,
        provider_status: int,
        attempts: int,
        cost_microusd: int,
        duration_ms: float,
        start_ns: Optional[int],
    ) -> None:
        if self._provider_tracer is None:
            return
        from opentelemetry.trace import Status, StatusCode

        span = self._start(
            self._provider_tracer, "arena.provider.%s" % provider, start_ns
        )
        span.set_attribute("arena.provider", str(provider))
        span.set_attribute("arena.operation", str(operation))
        span.set_attribute("arena.outcome", str(outcome))
        span.set_attribute("arena.error_code", str(error_code))
        span.set_attribute("arena.error_type", str(error_type))
        span.set_attribute("arena.http_status", int(http_status))
        span.set_attribute("arena.provider_status", int(provider_status))
        span.set_attribute("arena.attempts", int(attempts))
        span.set_attribute("arena.cost_microusd", int(cost_microusd))
        span.set_attribute("duration_ms", float(duration_ms))
        if outcome in ("failed", "uncertain"):
            span.set_status(Status(StatusCode.ERROR))
        span.end()

    # -- run outcomes ----------------------------------------------------

    def record_run(self, run_kind: str, terminal_cause: str) -> None:
        try:
            if self._run_tracer is None:
                return
            from opentelemetry.trace import Status, StatusCode

            span = self._start(
                self._run_tracer, "arena.run.%s" % run_kind, None
            )
            outcome = "ok" if terminal_cause == "accepted" else "failed"
            span.set_attribute("arena.run_kind", str(run_kind))
            span.set_attribute("arena.terminal_cause", str(terminal_cause))
            span.set_attribute("arena.outcome", outcome)
            if outcome == "failed":
                span.set_status(Status(StatusCode.ERROR))
            span.end()
        except BaseException as exc:
            self._log_emit_failure(exc)

    # -- provider-gate contention ----------------------------------------

    def record_gate(
        self,
        gate_outcome: str,
        *,
        duration_ms: float = 0.0,
        start_ns: Optional[int] = None,
    ) -> None:
        try:
            if self._gate_tracer is None:
                return
            from opentelemetry.trace import Status, StatusCode

            span = self._start(self._gate_tracer, "arena.gate", start_ns)
            span.set_attribute("arena.gate_outcome", str(gate_outcome))
            span.set_attribute("duration_ms", float(duration_ms))
            if gate_outcome in ("timed_out", "no_capacity"):
                span.set_status(Status(StatusCode.ERROR))
            span.end()
        except BaseException as exc:
            self._log_emit_failure(exc)

    # -- shared ----------------------------------------------------------

    def _start(self, tracer: Any, name: str, start_ns: Optional[int]) -> Any:
        from opentelemetry.context import Context
        from opentelemetry.trace import SpanKind

        return tracer.start_span(
            name,
            kind=SpanKind.INTERNAL,
            # A fresh root context: these spans never adopt a caller's parent.
            context=Context(),
            start_time=start_ns if start_ns is not None else time.time_ns(),
        )

    def _log_emit_failure(self, exc: BaseException) -> None:
        try:
            print(
                "%s error=%s" % (_ARENA_LOG["emit_failed"], type(exc).__name__),
                flush=True,
            )
        except BaseException:
            pass

    def record(
        self,
        stage: str,
        outcome: str,
        *,
        error_type: str = ARENA_NO_ERROR,
        count: int = 0,
        duration_ms: float = 0.0,
        start_ns: Optional[int] = None,
    ) -> None:
        try:
            self._record(
                stage,
                outcome,
                error_type=error_type,
                count=count,
                duration_ms=duration_ms,
                start_ns=start_ns,
            )
        except BaseException as exc:
            # Telemetry is observational only — the pipeline must not notice.
            self._log_emit_failure(exc)

    def _record(
        self,
        stage: str,
        outcome: str,
        *,
        error_type: str,
        count: int,
        duration_ms: float,
        start_ns: Optional[int],
    ) -> None:
        from opentelemetry.trace import Status, StatusCode

        span = self._start(self._tracer, "arena.%s" % stage, start_ns)
        span.set_attribute("arena.stage", str(stage))
        span.set_attribute("arena.outcome", str(outcome))
        span.set_attribute("arena.error_type", str(error_type))
        span.set_attribute("arena.count", int(count))
        span.set_attribute("duration_ms", float(duration_ms))
        if outcome == "failed":
            span.set_status(Status(StatusCode.ERROR))
        span.end()


def configure_arena_otel(
    app: Any = None, *, span_exporter: Optional[Any] = None
) -> Optional[ArenaTelemetry]:
    """Wire Arena sidecar telemetry. No-op (returns None) unless enabled.

    Installs, on the SAME fail-closed pipeline the gateway uses:

    - request spans under ``arena.http`` for the sidecar's own route
      templates, which the gateway's ``/arena/{arena_path:path}`` catch-all
      cannot express, each carrying the refusal code the sidecar produced;
    - a recorder for pipeline-stage spans under ``arena.task``;
    - a recorder for provider-call spans under ``arena.provider``;
    - a recorder for run-outcome spans under ``arena.run``;
    - a recorder for provider-gate contention under ``arena.gate``.

    The returned recorder is what ``lab_arena.telemetry`` installs; when this
    returns None the pipeline keeps calling a no-op and nothing changes.
    """
    if span_exporter is None and not _enabled():
        return None
    ambient = _ambient_exporter_env_names()
    if ambient:
        try:
            print(
                "%s ambient_exporter_env=%s"
                % (_ARENA_LOG["refused"], ",".join(ambient)),
                flush=True,
            )
        except Exception:
            pass
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind, StatusCode, Status
        from opentelemetry.context import Context

        # FIXED resource, distinct service identity: an Arena reading must
        # never be attributed to the gateway (or inherit its sensors).
        expected_resource = {"service.name": ARENA_SERVICE_NAME}
        resource = Resource(expected_resource)

        def _registered_routes() -> set:
            try:
                return {
                    template
                    for template in (
                        getattr(route, "path", None)
                        for route in getattr(app, "routes", [])
                    )
                    if isinstance(template, str) and template
                }
            except Exception:
                return set()

        delegate = span_exporter
        if delegate is None:
            delegate = _build_otlp_exporter(_ARENA_LOG)
            if delegate is None:
                return None
        validating_exporter = _validating_exporter(
            delegate,
            allowed_routes=_registered_routes,
            expected_resource=expected_resource,
            scopes={
                ARENA_HTTP_SCOPE: "arena_http",
                ARENA_TASK_SCOPE: "task",
                ARENA_PROVIDER_SCOPE: "provider",
                ARENA_RUN_SCOPE: "run",
                ARENA_GATE_SCOPE: "gate",
            },
            log=_ARENA_LOG,
        )
        processor = _build_span_processor(
            validating_exporter,
            simple=span_exporter is not None,
        )

        # A dedicated provider, never the global one.
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)

        telemetry = ArenaTelemetry(
            provider.get_tracer(ARENA_TASK_SCOPE),
            provider_tracer=provider.get_tracer(ARENA_PROVIDER_SCOPE),
            run_tracer=provider.get_tracer(ARENA_RUN_SCOPE),
            gate_tracer=provider.get_tracer(ARENA_GATE_SCOPE),
        )

        if app is not None:

            http_tracer = provider.get_tracer(ARENA_HTTP_SCOPE)

            def _emit(scope, status_code, start_ns, start_mono):
                route = _safe_route_label(scope)
                method = str(scope.get("method") or "")
                span = http_tracer.start_span(
                    "%s %s" % (method, route),
                    kind=SpanKind.SERVER,
                    context=Context(),
                    start_time=start_ns,
                )
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", int(status_code))
                span.set_attribute("arena.denial", telemetry.take_denial())
                span.set_attribute(
                    "duration_ms", (time.monotonic() - start_mono) * 1000.0
                )
                if int(status_code) >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

            app.add_middleware(
                _GatewayOtelMiddleware,
                emit=_emit,
                on_request_start=telemetry.reset_denial,
            )

        return telemetry
    except Exception as exc:  # never let telemetry break the Arena service
        try:
            print(
                "%s error=%s" % (_ARENA_LOG["skipped"], type(exc).__name__),
                flush=True,
            )
        except Exception:
            pass
        return None
