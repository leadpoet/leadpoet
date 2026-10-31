"""Opt-in, infra-only OpenTelemetry for the gateway HOST process.

Scope is deliberately narrow and privacy-preserving:

- It emits ONE server span per HTTP request capturing only operational
  metadata: request method, the matched route *template* (never the concrete
  path, so ids/hotkeys/epochs do not leak), response status class, duration,
  and a fixed pair for capacity shedding. It never records bodies, queries,
  headers, database statements, or any LLM prompt/completion content.
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
   empty trace state), the four approved HTTP attributes and an optional
   fixed capacity-shed admission/class pair, with the
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
identical fail-closed machinery under separate Arena scopes:

- ``arena.http`` — one SERVER span per sidecar request, carrying the SAME four
  attributes plus a bounded refusal code. A matched public round route may
  also carry a canonical calendar-date round id from its path parameter. At
  the gateway every Arena call collapses into the catch-all
  ``/arena/{arena_path:path}`` template, so the sidecar's own route templates
  identify the operation without exporting the concrete URL.
- ``arena.task`` — one INTERNAL span per driver / worker stage, so a stalled
  pipeline stops being indistinguishable from an idle one. Its attribute set
  is a separate, equally fixed allowlist: a stage name from a frozen
  vocabulary, an outcome from a frozen vocabulary, an exception CLASS name
  (code-controlled, shape-validated) and a bounded integer count. No round id,
  submission id, hotkey, source, prompt, score, or model output can pass it.
- ``arena.provider`` and ``arena.run`` — fixed operational fields and, after
  exact lease match or terminal commit, seven bounded identifiers from the trusted
  run row. No request content, ICP text, provider body, or credential is read.
- ``arena.runtime`` — one span for an accepted runtime event upload that
  inserted at least one event. Batch flags describe event kinds present in the
  upload; mixed new/replayed batches do not prove a fresh lifecycle transition.
- ``arena.review`` — one span after a submission review attempt commits. It
  carries only the bounded decision, rejection categories, safe diagnostic,
  provider HTTP status, retry flag, proven funding route, and validated
  submission identity.

All Arena scopes share the gateway's destination (the same private
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
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional

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
        "reconcile_active_deepline_costs",
        "reconcile_deepline_cost",
        "reconcile_deepline_billing_lookup",
        "review_submissions",
        "review_guard",
        "service_startup",
        "service_shutdown",
        "service_preflight",
        # Steps inside one advance_round. The parent stage reports only that
        # the whole transition failed, which is not enough to tell a slow
        # billing read from a blocked service lock from a slow SQL function.
        "advance_billing_read",
        "advance_admission",
        "advance_lock_wait",
        "advance_expire_leases",
        "advance_list_scoring_runs",
        "advance_commit_benchmark",
        "advance_open_stage",
        "advance_close_stage",
        "advance_commit_scoring_plan",
        "advance_open_scoring",
        "advance_close_scoring",
        "advance_score_stage",
        "advance_publish",
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
ARENA_RUNTIME_SCOPE = "arena.runtime"
ARENA_GATE_SCOPE = "arena.gate"
ARENA_REVIEW_SCOPE = "arena.review"

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
        "capability_unavailable",
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

# A failed run already reports WHICH cause ended it, never WHERE it failed or
# WHAT the failure was. Both come from the completion contract's bounded
# ``failure_diagnostic`` choice sets, so neither can carry judge or miner text.
# An accepted run has no diagnostic and reports the no-error sentinel.
ARENA_FAILURE_STAGES = frozenset({
    "sandbox", "sandbox_output", "scoring_output", "scorer", "provider_call",
    ARENA_NO_ERROR,
})
ARENA_FAILURE_CLASSES = frozenset({
    "judge_timeout", "sandbox_output_error", "missing_output",
    "execution_output_invalid", "scoring_output_invalid", "judge_error",
    "provider_unavailable", ARENA_NO_ERROR,
})
ARENA_REVIEW_STATUSES = frozenset({"passed", "rejected", "error"})
ARENA_REVIEW_FUNDING_ROUTES = frozenset({"miner_key", "none"})
ARENA_REVIEW_CATEGORIES = frozenset({
    "hardcoded_prepared_answers", "fabricated_evidence", "malicious_behavior",
    "reviewer_manipulation", "duplicate_submission",
})
ARENA_REVIEW_ERROR_CODES = frozenset({
    "-",
    "code_review_source_size_mismatch", "code_review_source_digest_mismatch",
    "review_context_limit_invalid", "review_model_invalid",
    "review_output_cannot_report_coverage", "review_output_limit_invalid",
    "review_source_empty", "review_source_exceeds_context",
    "review_source_incomplete", "review_source_invalid",
    "review_source_not_text", "review_source_not_utf8",
    "review_source_unreadable", "source_archive_invalid",
    "source_archive_failed", "source_archive_too_large",
    "source_archive_unsafe", "source_contains_credentials",
    "source_directory_missing", "source_empty",
    "source_entry_type_invalid", "source_file_count_exceeded",
    "source_git_automation_forbidden", "source_license_invalid",
    "source_path_invalid", "source_unpacked_too_large",
    "source_unreadable", "source_credentials_invalid",
    "harness_file_missing", "harness_invalid", "harness_too_large",
    "code_review_preparation_unavailable", "code_review_provider_authentication",
    "code_review_provider_credit", "code_review_provider_rate_limited",
    "code_review_provider_request_rejected", "code_review_provider_timeout",
    "code_review_provider_unavailable", "code_review_transport_error",
    "code_review_credential_echo", "code_review_cost_unavailable",
    "code_review_deadline_exceeded", "code_review_round_closed",
    "code_review_submission_ineligible", "code_review_research_unverified",
    "review_response_invalid",
})

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

# ``lab_arena.operations.DEEPLINE_TOOLS``. Every Deepline call goes out as
# ``deepline.execute``, so the operation id alone cannot say WHICH upstream
# tool answered or failed. The tool is the missing half of that attribution,
# and it is a closed set of literals written here like every other exported
# string. ``ARENA_TOOL_NONE`` is the value for a call that has no tool — any
# non-Deepline provider, and a Deepline operation with no resolvable tool.
ARENA_TOOL_NONE = "-"
ARENA_DEEPLINE_TOOLS = frozenset(
    {
        "contextdev_get_web_scrape_markdown",
        "contextdev_post_news_search",
        "contextdev_post_web_search",
        "exa_answer",
        "exa_company_search",
        "exa_contents",
        "exa_people_search",
        "exa_search",
        "firecrawl_scrape",
        "free_simple_company_search",
        "generic_http_request",
        "harvestapi_get_company",
        "harvestapi_get_profile",
        "harvestapi_search_leads",
        "hunter_email_finder",
        "limadata_find_work_email",
        "datagma_find_email",
        "leadmagic_email_finder",
        "zerobounce_validate",
        "bounceban_verify_single",
        "bounceban_get_single_status",
        "harvestapi_get_job",
        "harvestapi_get_post",
        "hunter_discover",
        "predictleads_company_financing_events",
        "predictleads_company_job_openings",
        "predictleads_company_news_events",
        "twitterapi_tweets_by_ids",
    }
)

# A refusal code is the literal prefix of a ``ServiceError`` code, i.e. the
# part before the first ":". The suffix can carry an exception message and is
# never exported; the prefix is always a literal written in the source.
_ARENA_DENIAL_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
ARENA_NO_DENIAL = "-"

# Only explicit source-defined refusal codes may leave the sidecar. A shape
# check alone would admit a caller-controlled identifier such as "private_key".
ARENA_DENIAL_CODES = frozenset(
    {
        "accepted_judgment_binding_invalid",
        "accepted_judgment_score_missing",
        "accepted_judgment_source_invalid",
        "accepted_weight_epoch_not_current",
        "accepted_weight_epoch_scope_unavailable",
        "accepted_weight_state_conflict",
        "accepted_weight_state_invalid",
        "accepted_weight_state_reward_conflict",
        "accepted_weight_state_scope_conflict",
        "arena_schema_version_invalid",
        "arena_store_unavailable",
        "baseline_cost_eligibility_schema_invalid",
        "baseline_cost_eligibility_schema_unavailable",
        "baseline_hotkey_missing",
        "baseline_hotkey_reserved",
        "baseline_promoter_unavailable",
        "baseline_promotion_invalid",
        "baseline_promotion_pending",
        "baseline_recovery_scores_not_recorded",
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
        "claim_recovery_invalid",
        "claim_recovery_unavailable",
        "closed_provider_cost_candidate_invalid",
        "code_review_pending",
        "code_review_required",
        "code_review_schema_unavailable",
        "code_review_unavailable",
        "company_judgment_activation_invalid",
        "company_judgment_cache_invalid",
        "company_judgment_output_invalid",
        "company_judgment_schema_unavailable",
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
        "credit_retry_owner_required",
        "credit_retry_request_invalid",
        "credit_retry_source_unavailable",
        "credit_retry_unavailable",
        "daily_cutoff_hour_invalid",
        "daily_icp_source_invalid",
        "daily_round_dates_exhausted",
        "daily_runner_capacity_insufficient",
        "declared_parallelism_invalid",
        "deepline_catalog_schema_unavailable",
        "dynamic_benchmark_schema_unavailable",
        "execution_sequence_activation_invalid",
        "execution_terminal_evidence_invalid",
        "frame_invalid",
        "function_probe_invalid",
        "function_unavailable",
        "hotkey_banned",
        "hotkey_unregistered",
        "infrastructure_retry_cost_schema_unavailable",
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
        "legacy_checkpoint_invalid",
        "legacy_checkpoint_output_invalid",
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
        "recovered_judgment_binding_invalid",
        "recovered_judgment_scores_not_recorded",
        "reasoning_baseline_source_mismatch",
        "reasoning_capture_activation_invalid",
        "reasoning_capture_activation_not_ready",
        "request_invalid",
        "results_not_public",
        "reviewer_runtime_audit_invalid",
        "reviewer_runtime_invalid",
        "reviewer_runtime_lease_changed",
        "reward_history_invalid",
        "reward_history_signature_invalid",
        "reward_policy_projection_invalid",
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
        "round_opening",
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
        "scoring_generation_floor_invalid",
        "scoring_plan_invalid",
        "scoring_plan_missing",
        "scoring_plan_run_mismatch",
        "signature_invalid",
        "similarity_reference_limit",
        "similarity_reference_unavailable",
        "signal_judgment_activation_invalid",
        "signal_judgment_cache_invalid",
        "signal_judgment_claim_invalid",
        "signal_judgment_completion_invalid",
        "signal_judgment_input_invalid",
        "signal_judgment_lease_invalid",
        "signal_judgment_policy_unavailable",
        "signal_judgment_request_invalid",
        "signal_judgment_result_invalid",
        "source_not_public",
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
        "submission_similarity_schema_unavailable",
        "submission_source_immutable",
        "submission_superseded",
        "submission_transport_mismatch",
        "submission_window_closed",
        "successful_call_cost_schema_unavailable",
        "table_unavailable",
        "trajectory_invalid",
        "trajectory_unavailable",
        "training_input_conflict",
        "training_input_invalid",
        "training_input_unavailable",
        "training_observation_segment_conflict",
        "training_observation_segment_invalid",
        "training_observation_segment_unavailable",
        "training_observations_conflict",
        "training_observations_digest_mismatch",
        "training_observations_invalid",
        "training_observations_unavailable",
        "unsupported_integrity_policy",
        "validator_events_correlation_invalid",
        "validator_events_invalid",
        "validator_events_owner_required",
        "validator_events_rate_limited",
        "validator_events_scope_invalid",
        "validator_events_too_large",
        "validator_events_unavailable",
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

# Base HTTP fields. Only gateway capacity shedding may add the fixed pair below.
# Bool is explicitly rejected for the numeric fields.
SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "duration_ms": (int, float),
}

SPAN_ATTRIBUTE_ALLOWLIST = frozenset(SPAN_ATTRIBUTE_TYPES)

GATEWAY_ADMISSION_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "gateway.admission": (str,),
    "gateway.priority_class": (str,),
}
GATEWAY_PRIORITY_CLASSES = frozenset({"miner", "validator", "other"})

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
BILLING_LOOKUP_LANES = frozenset({"execute", "score", "general"})
BILLING_TASK_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    **TASK_SPAN_ATTRIBUTE_TYPES,
    "arena.round_id": (str,),
    "arena.billing_lane": (str,),
    "arena.cursor_entry_id": (int,),
    "arena.run_scoped": (bool,),
}
# Two rounds can be open at once. A round-advance stage says which round it
# served and when that round's final scoring close falls, so the last-resort
# cancellation two hours later is a deadline that can be read rather than
# inferred. Both values are Arena-generated and shape-checked.
ROUND_TASK_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    **TASK_SPAN_ATTRIBUTE_TYPES,
    "arena.round_id": (str,),
    "arena.scoring_deadline": (str,),
}
_ARENA_ISO_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?"
    r"(?:Z|\+00:00)$"
)

ARENA_LIFECYCLE_STAGES = frozenset(
    {"service_startup", "service_shutdown", "service_preflight"}
)
LIFECYCLE_TASK_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    **TASK_SPAN_ATTRIBUTE_TYPES,
    "arena.source_sha": (str,),
}

# The Arena sidecar's HTTP envelope: the gateway's four, plus the refusal code
# the sidecar itself produced. The gateway's own envelope is UNCHANGED — this
# fifth required attribute exists only on the sidecar, where a 4xx with no
# reason is the difference between "a runner was denied a claim" and "a runner
# is broken". Its vocabulary is the literal prefix of a ``ServiceError`` code.
# A matched public round route may add an optional canonical round id.
# Temporary store failures may also carry the existing bounded machine fields;
# the final exporter admits those only for the matching 503 denial.
ARENA_HTTP_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "arena.denial": (str,),
    "duration_ms": (int, float),
}

ARENA_HTTP_ATTRIBUTE_ALLOWLIST = (
    frozenset(ARENA_HTTP_ATTRIBUTE_TYPES) | frozenset({"arena.round_id"})
)

# A registered public round route can name the requested round. The concrete
# path, query, headers, and other path parameters remain private.
_ARENA_PUBLIC_ROUND_ROUTE_PREFIX = "/arena/v1/rounds/{round_id}"
_ARENA_HTTP_ROUND_ID_RE = re.compile(r"^arena-([0-9]{4}-[0-9]{2}-[0-9]{2})$")


def _is_public_round_route(route: str) -> bool:
    return route == _ARENA_PUBLIC_ROUND_ROUTE_PREFIX or route.startswith(
        _ARENA_PUBLIC_ROUND_ROUTE_PREFIX + "/"
    )


def _canonical_http_round_id(value: Any) -> bool:
    if type(value) is not str:
        return False
    match = _ARENA_HTTP_ROUND_ID_RE.fullmatch(value)
    if match is None:
        return False
    try:
        date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return True


def _matched_http_round_id(scope: Mapping[str, Any], route: str) -> Optional[str]:
    if not _is_public_round_route(route):
        return None
    params = scope.get("path_params")
    if not isinstance(params, Mapping):
        return None
    value = params.get("round_id")
    return value if _canonical_http_round_id(value) else None

# The only per-run fields admitted on provider, terminal, and runtime spans. The Arena
# service takes these from the authenticated database run row, not a request.
# No ICP content, miner output, provider body, or credential is admitted.
RUN_IDENTITY_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.run_id": (str,),
    "arena.round_id": (str,),
    "arena.submission_id": (str,),
    "arena.runner_hotkey": (str,),
    "arena.stage": (int,),
    "arena.icp_position": (int,),
    "arena.attempt": (int,),
}
RUN_IDENTITY_ATTRIBUTE_ALLOWLIST = frozenset(RUN_IDENTITY_ATTRIBUTE_TYPES)
_ARENA_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_ARENA_ROUND_ID_RE = re.compile(r"^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-[a-z0-9]{1,16})?$")
_ARENA_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_ARENA_RUNNER_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")

# The base provider envelope plus the complete trusted run identity when the
# service has one. Partial identities and any additional field drop the span.
PROVIDER_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.provider": (str,),
    "arena.operation": (str,),
    "arena.tool": (str,),
    "arena.outcome": (str,),
    "arena.error_code": (str,),
    "arena.error_type": (str,),
    "arena.http_status": (int,),
    "arena.provider_status": (int,),
    "arena.attempts": (int,),
    "arena.cost_microusd": (int,),
    "duration_ms": (int, float),
}

STORE_ERROR_SPAN_FIELDS = frozenset({
    "arena.store_operation", "arena.store_resource", "arena.store_http_status",
    "arena.store_code", "arena.store_transport_error",
})

PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(PROVIDER_SPAN_ATTRIBUTE_TYPES)

# The ONLY attributes a run-outcome span may carry. There is deliberately no
# duration here: the sidecar records when a run ENDS, and the run row carries
# no start time, so any duration would be invented rather than measured.
RUN_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.run_kind": (str,),
    "arena.terminal_cause": (str,),
    "arena.outcome": (str,),
    "arena.failure_stage": (str,),
    "arena.failure_class": (str,),
}

RUN_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(RUN_SPAN_ATTRIBUTE_TYPES)

# Runtime uploads report only batch counts and lifecycle flags from the
# validated batch. has_error covers runtime.error or a runtime.finished event
# with a known failed status. Mixed replay does not prove a fresh transition.
RUNTIME_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.inserted_count": (int,),
    "arena.replayed_count": (int,),
    "arena.has_started": (bool,),
    "arena.has_finished": (bool,),
    "arena.has_error": (bool,),
}
RUNTIME_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(
    RUNTIME_SPAN_ATTRIBUTE_TYPES
) | RUN_IDENTITY_ATTRIBUTE_ALLOWLIST

REVIEW_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.review_status": (str,),
    "arena.error_code": (str,),
    "arena.review_categories": (str,),
    "arena.provider_http_status": (int,),
    "arena.retryable": (bool,),
    "arena.funding_route": (str,),
    "arena.round_id": (str,),
    "arena.submission_id": (str,),
    "arena.miner_hotkey": (str,),
    "arena.attempt": (int,),
}
REVIEW_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(REVIEW_SPAN_ATTRIBUTE_TYPES)

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
    {epoch}``) keeps ids, hotkeys, and epoch numbers out of HTTP spans. When the
    template cannot be resolved the label is the fixed ``/_unmatched`` literal —
    a client-controlled path segment is never exported.
    """
    try:
        scope = getattr(scope_or_request, "scope", scope_or_request)
        route = scope.get("route")
        # FastAPI 0.137+ keeps included routes in an effective context. The
        # original route's path omits include-time prefixes, but the context's
        # path is the complete matched template. Only use it for that route.
        fastapi_scope = scope.get("fastapi")
        if isinstance(fastapi_scope, Mapping):
            context = fastapi_scope.get("effective_route_context")
            if getattr(context, "original_route", None) is route:
                template = getattr(context, "path", None)
                if isinstance(template, str) and template:
                    return template
        template = getattr(route, "path", None)
        if isinstance(template, str) and template:
            return template
    except Exception:
        pass
    return UNMATCHED_ROUTE_LABEL


def _registered_route_templates(app: Any) -> set:
    """Read registered templates, including FastAPI's lazy included routers.

    FastAPI 0.137+ exposes effective route contexts for included routers. Older
    FastAPI and Starlette keep routes directly in ``app.routes``. Neither path
    reads a request's concrete URL, and an unavailable route tree fails closed.
    """
    try:
        routes = getattr(app, "routes", [])
        try:
            from fastapi.routing import iter_route_contexts
        except ImportError:
            contexts = routes
        else:
            contexts = iter_route_contexts(routes)
        return {
            template
            for route in contexts
            if isinstance((template := getattr(route, "path", None)), str)
            and template
        }
    except Exception:
        return set()


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
        if isinstance(value, bool):
            if bool not in allowed_types:
                return False
        elif not isinstance(value, allowed_types):
            return False
    return True


def _attributed_run_violation(
    attributes: Dict[str, Any], base_types: Dict[str, tuple]
) -> Optional[str]:
    """Accept the base envelope or all seven bounded run fields, never a part."""

    if _attributes_conform(attributes, base_types):
        return None
    full_types = {**base_types, **RUN_IDENTITY_ATTRIBUTE_TYPES}
    if not _attributes_conform(attributes, full_types):
        return "attributes"
    if not _ARENA_RUN_ID_RE.fullmatch(attributes["arena.run_id"]):
        return "run_identity"
    if not _ARENA_ROUND_ID_RE.fullmatch(attributes["arena.round_id"]):
        return "run_identity"
    if not _ARENA_SUBMISSION_ID_RE.fullmatch(attributes["arena.submission_id"]):
        return "run_identity"
    if not _ARENA_RUNNER_HOTKEY_RE.fullmatch(attributes["arena.runner_hotkey"]):
        return "run_identity"
    if attributes["arena.stage"] not in (1, 2):
        return "run_identity"
    if not 0 <= attributes["arena.icp_position"] < 100:
        return "run_identity"
    if not 1 <= attributes["arena.attempt"] <= _ARENA_MAX_COUNT:
        return "run_identity"
    return None


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

    ``http``        SERVER span, four HTTP fields plus optional capacity pair
    ``arena_http``  SERVER span, those four plus the Arena refusal code and
                    optional canonical public round id
    ``task``        INTERNAL span, the five approved pipeline-stage attributes
    ``provider``    INTERNAL span, the ten approved provider-call attributes
    ``run``         INTERNAL span, the four approved run-outcome attributes
    ``runtime``     INTERNAL span, a bounded accepted runtime-upload batch
    ``gate``        INTERNAL span, the two approved provider-gate attributes
    ``review``      INTERNAL span, one committed submission review decision

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
            if not _attributes_conform(attributes, {
                **SPAN_ATTRIBUTE_TYPES, **GATEWAY_ADMISSION_ATTRIBUTE_TYPES,
            }):
                return "attributes"
            if (
                attributes["gateway.admission"] != "capacity_shed"
                or attributes["gateway.priority_class"] not in GATEWAY_PRIORITY_CLASSES
                or attributes["http.response.status_code"] != 503
            ):
                return "admission"
        method = attributes["http.request.method"]
        if method not in _ALLOWED_HTTP_METHODS:
            return "method"
        route = attributes["http.route"]
        if span.name != "%s %s" % (method, route):
            return "name"
        if route != UNMATCHED_ROUTE_LABEL and route not in allowed_routes():
            return "route"
        return None

    def _without_store_error_fields(attributes: Dict[str, Any], *, allowed: bool) -> Optional[Dict[str, Any]]:
        fields = {key: value for key, value in attributes.items() if key in STORE_ERROR_SPAN_FIELDS}
        if fields:
            from lab_arena.store import safe_store_error_metadata

            metadata = {key.removeprefix("arena.store_"): value for key, value in fields.items()}
            if not allowed or safe_store_error_metadata(metadata) != metadata:
                return None
        return {key: value for key, value in attributes.items() if key not in fields}

    def _arena_http_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        attributes = _without_store_error_fields(attributes, allowed=(
            attributes.get("http.response.status_code") == 503
            and attributes.get("arena.denial") == "arena_store_unavailable"
        ))
        if attributes is None:
            return "store_error"
        has_round_id = "arena.round_id" in attributes
        expected_types = ARENA_HTTP_ATTRIBUTE_TYPES
        if has_round_id:
            expected_types = {**expected_types, "arena.round_id": (str,)}
        if not _attributes_conform(attributes, expected_types):
            return "attributes"
        if has_round_id and (
            not _is_public_round_route(attributes["http.route"])
            or not _canonical_http_round_id(attributes["arena.round_id"])
        ):
            return "round_id"
        denial = attributes["arena.denial"]
        if denial != ARENA_NO_DENIAL and (
            not _ARENA_DENIAL_RE.fullmatch(denial)
            or denial not in ARENA_DENIAL_CODES
        ):
            return "denial"
        trimmed = {
            key: value
            for key, value in attributes.items()
            if key not in ("arena.denial", "arena.round_id")
        }
        return _http_violation(span, trimmed)

    def _provider_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        from lab_arena.store import safe_store_error_metadata

        store_fields = {key: value for key, value in attributes.items() if key in STORE_ERROR_SPAN_FIELDS}
        if store_fields:
            metadata = {key.removeprefix("arena.store_"): value for key, value in store_fields.items()}
            if safe_store_error_metadata(metadata) != metadata or attributes.get("arena.outcome") != "failed":
                return "store_error"
        provider_fields = {key: value for key, value in attributes.items() if key not in store_fields}
        run_identity_error = _attributed_run_violation(provider_fields, PROVIDER_SPAN_ATTRIBUTE_TYPES)
        if run_identity_error is not None:
            return run_identity_error
        provider = attributes["arena.provider"]
        if provider not in ARENA_PROVIDERS:
            return "provider"
        operation = attributes["arena.operation"]
        if operation != ARENA_OPERATION_UNKNOWN and ARENA_OPERATION_PROVIDERS.get(operation) != provider:
            return "operation"
        tool = attributes["arena.tool"]
        if tool != ARENA_TOOL_NONE and (
            tool not in ARENA_DEEPLINE_TOOLS or provider != "deepline"
        ):
            return "tool"
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
        run_identity_error = _attributed_run_violation(attributes, RUN_SPAN_ATTRIBUTE_TYPES)
        if run_identity_error is not None:
            return run_identity_error
        run_kind = attributes["arena.run_kind"]
        if run_kind not in ARENA_RUN_KINDS:
            return "run_kind"
        if attributes["arena.terminal_cause"] not in ARENA_TERMINAL_CAUSES:
            return "terminal_cause"
        if attributes["arena.outcome"] not in ARENA_TASK_OUTCOMES:
            return "outcome"
        if attributes["arena.failure_stage"] not in ARENA_FAILURE_STAGES:
            return "failure_stage"
        if attributes["arena.failure_class"] not in ARENA_FAILURE_CLASSES:
            return "failure_class"
        if span.name != "arena.run.%s" % run_kind:
            return "name"
        return None

    def _runtime_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if set(attributes) != RUNTIME_SPAN_ATTRIBUTE_ALLOWLIST:
            return "attributes"
        run_identity_error = _attributed_run_violation(
            attributes, RUNTIME_SPAN_ATTRIBUTE_TYPES
        )
        if run_identity_error is not None:
            return run_identity_error
        inserted = attributes["arena.inserted_count"]
        replayed = attributes["arena.replayed_count"]
        if inserted < 1 or replayed < 0 or inserted + replayed > 32:
            return "runtime_count"
        if not any(attributes[key] for key in (
            "arena.has_started", "arena.has_finished", "arena.has_error",
        )):
            return "runtime_kind"
        if span.name != "arena.runtime.upload":
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

    def _review_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, REVIEW_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        if attributes["arena.review_status"] not in ARENA_REVIEW_STATUSES:
            return "review_status"
        if attributes["arena.error_code"] not in ARENA_REVIEW_ERROR_CODES:
            return "error_code"
        if attributes["arena.review_status"] != "error" and attributes["arena.error_code"] != "-":
            return "error_code"
        if attributes["arena.review_status"] == "error" and attributes["arena.error_code"] == "-":
            return "error_code"
        categories = attributes["arena.review_categories"]
        if categories != "-":
            values = categories.split(";")
            if (
                attributes["arena.review_status"] != "rejected"
                or not 1 <= len(values) <= len(ARENA_REVIEW_CATEGORIES)
                or values != sorted(set(values))
                or any(value not in ARENA_REVIEW_CATEGORIES for value in values)
            ):
                return "review_categories"
        if attributes["arena.funding_route"] not in ARENA_REVIEW_FUNDING_ROUTES:
            return "funding_route"
        if not 0 <= attributes["arena.provider_http_status"] <= 599:
            return "provider_http_status"
        if not _ARENA_ROUND_ID_RE.fullmatch(attributes["arena.round_id"]):
            return "review_identity"
        if not _ARENA_SUBMISSION_ID_RE.fullmatch(attributes["arena.submission_id"]):
            return "review_identity"
        if not _ARENA_RUNNER_HOTKEY_RE.fullmatch(attributes["arena.miner_hotkey"]):
            return "review_identity"
        if not 1 <= attributes["arena.attempt"] <= _ARENA_MAX_COUNT:
            return "review_identity"
        if span.name != "arena.review.decision":
            return "name"
        return None

    def _task_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        attributes = _without_store_error_fields(attributes, allowed=(
            attributes.get("arena.outcome") == "failed"
            and attributes.get("arena.error_type") in (
                "ArenaStoreError", "ArenaStoreUnavailable"
            )
        ))
        if attributes is None:
            return "store_error"
        stage = attributes.get("arena.stage")
        if stage in ARENA_LIFECYCLE_STAGES:
            if not _attributes_conform(attributes, LIFECYCLE_TASK_SPAN_ATTRIBUTE_TYPES):
                return "attributes"
            source_sha = attributes["arena.source_sha"]
            if source_sha != "unknown" and not re.fullmatch(r"[0-9a-f]{40}", source_sha):
                return "source_sha"
        elif isinstance(stage, str) and stage.startswith("advance_") and _attributes_conform(
            attributes, ROUND_TASK_SPAN_ATTRIBUTE_TYPES
        ):
            if (not _ARENA_ROUND_ID_RE.fullmatch(attributes["arena.round_id"])
                    or not _ARENA_ISO_UTC_RE.fullmatch(
                        attributes["arena.scoring_deadline"])):
                return "round_scope"
        elif stage == "reconcile_deepline_billing_lookup" and _attributes_conform(
            attributes, BILLING_TASK_SPAN_ATTRIBUTE_TYPES
        ):
            if (not _ARENA_ROUND_ID_RE.fullmatch(attributes["arena.round_id"])
                    or attributes["arena.billing_lane"] not in BILLING_LOOKUP_LANES
                    or not 0 <= attributes["arena.cursor_entry_id"] <= 9223372036854775807):
                return "billing_lookup"
        elif not _attributes_conform(attributes, TASK_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
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
        "runtime": _runtime_violation,
        "gate": _gate_violation,
        "review": _review_violation,
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
    import math
    import random
    import requests

    class _ConnectionRetrySession(requests.Session):
        def post(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return super().post(*args, **kwargs)
            except requests.ConnectionError as exc:
                timeout = kwargs.get("timeout")
                if (isinstance(exc, requests.Timeout)
                    or type(timeout) not in (int, float)
                    or not math.isfinite(timeout) or timeout <= 0
                    or not isinstance(kwargs.get("data"), bytes)):
                    raise
                # The pinned exporter does not retry response-less disconnects.
                # Retry the identical immutable batch once within its original
                # timeout. HTTP status retries remain owned by the exporter.
                deadline = started + timeout
                delay = random.uniform(0.1, 0.2)
                if deadline - time.monotonic() <= delay:
                    raise
                time.sleep(delay)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                return super().post(*args, **dict(kwargs, timeout=remaining))

    endpoint = os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip()
    headers = {"Authorization": "Bearer " + token}
    # Explicit arguments only: the destination exists solely inside
    # this exporter object, never in ambient environment variables.
    return OTLPSpanExporter(endpoint=endpoint, headers=headers, session=_ConnectionRetrySession())


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
            return _registered_route_templates(app)

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
            # Keep request content private, including failures before routing.
            span.set_attribute("http.request.method", method)
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", int(status_code))
            span.set_attribute("duration_ms", (time.monotonic() - start_mono) * 1000.0)
            if "_gateway_capacity_shed_class" in scope:
                span.set_attribute("gateway.admission", "capacity_shed")
                # Set only by PriorityMiddleware on its own 503 response.
                # This class names a request pool, not a verified identity.
                span.set_attribute(
                    "gateway.priority_class", scope["_gateway_capacity_shed_class"]
                )
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
    """Records Arena spans without raising or blocking normal work.

    Six INTERNAL envelopes, each a root span with its own exact allowlist:
    pipeline stages, provider calls, run outcomes, runtime uploads,
    provider-gate waits, and submission reviews.
    The caller supplies values from the frozen vocabularies; anything else is
    dropped whole by the same fail-closed validator that guards the HTTP
    envelope. A separate bounded flush is available at graceful shutdown.
    Every public method swallows its own failures — an exporter problem can
    never surface as an Arena failure.
    """

    def __init__(
        self,
        tracer: Any,
        *,
        provider_tracer: Any = None,
        run_tracer: Any = None,
        runtime_tracer: Any = None,
        gate_tracer: Any = None,
        review_tracer: Any = None,
        flush_spans: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self._tracer = tracer
        self._provider_tracer = provider_tracer
        self._run_tracer = run_tracer
        self._runtime_tracer = runtime_tracer
        self._gate_tracer = gate_tracer
        self._review_tracer = review_tracer
        self._flush_spans = flush_spans
        # Per-request refusal code, read back by the Arena HTTP middleware.
        # A raw ASGI chain shares one context per request, so this is set and
        # read within a single request and never crosses between them.
        import contextvars

        # A worker thread receives a copy of the request context. Sharing one
        # mutable value lets a sync route report its denial back to ASGI.
        self._denial = contextvars.ContextVar("arena_denial", default=None)

    def flush(self, *, timeout_millis: int = 1000) -> None:
        """Flush queued spans within a bounded graceful-shutdown window."""
        try:
            if self._flush_spans is not None:
                self._flush_spans(timeout_millis)
        except BaseException:
            pass

    # -- refusal code ----------------------------------------------------

    def note_denial(self, code: str, *, store_error: Optional[Mapping[str, Any]] = None) -> None:
        """Record the refusal code for the request currently being served."""
        try:
            state = self._denial.get()
            if state is None:
                state = [ARENA_NO_DENIAL, {}]
                self._denial.set(state)
            from lab_arena.store import safe_store_error_metadata

            state[0] = str(code)
            state[1] = safe_store_error_metadata(store_error)
        except BaseException:
            pass

    def reset_denial(self) -> None:
        try:
            self._denial.set([ARENA_NO_DENIAL, {}])
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

    def take_store_error(self) -> Mapping[str, Any]:
        """Return only bounded store diagnostics from this request's context."""
        try:
            from lab_arena.store import safe_store_error_metadata

            state = self._denial.get()
            return safe_store_error_metadata(state[1]) if state is not None else {}
        except BaseException:
            return {}

    # -- provider calls --------------------------------------------------

    def record_provider(
        self,
        provider: str,
        operation: str,
        outcome: str,
        *,
        error_code: str = ARENA_NO_ERROR,
        error_type: str = ARENA_NO_ERROR,
        tool: str = ARENA_TOOL_NONE,
        http_status: int = 0,
        provider_status: int = 0,
        attempts: int = 1,
        cost_microusd: int = 0,
        duration_ms: float = 0.0,
        start_ns: Optional[int] = None,
        run_identity: Optional[Mapping[str, Any]] = None,
        store_error: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            self._record_provider(
                provider,
                operation,
                outcome,
                error_code=error_code,
                error_type=error_type,
                tool=tool,
                http_status=http_status,
                provider_status=provider_status,
                attempts=attempts,
                cost_microusd=cost_microusd,
                duration_ms=duration_ms,
                start_ns=start_ns,
                run_identity=run_identity,
                store_error=store_error,
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
        tool: str,
        http_status: int,
        provider_status: int,
        attempts: int,
        cost_microusd: int,
        duration_ms: float,
        start_ns: Optional[int],
        run_identity: Optional[Mapping[str, Any]],
        store_error: Optional[Mapping[str, Any]],
    ) -> None:
        if self._provider_tracer is None:
            return
        if run_identity is not None and set(run_identity) != RUN_IDENTITY_ATTRIBUTE_ALLOWLIST:
            return
        from opentelemetry.trace import Status, StatusCode

        span = self._start(
            self._provider_tracer, "arena.provider.%s" % provider, start_ns
        )
        span.set_attribute("arena.provider", str(provider))
        span.set_attribute("arena.operation", str(operation))
        span.set_attribute("arena.tool", str(tool))
        span.set_attribute("arena.outcome", str(outcome))
        span.set_attribute("arena.error_code", str(error_code))
        span.set_attribute("arena.error_type", str(error_type))
        span.set_attribute("arena.http_status", int(http_status))
        span.set_attribute("arena.provider_status", int(provider_status))
        span.set_attribute("arena.attempts", int(attempts))
        span.set_attribute("arena.cost_microusd", int(cost_microusd))
        span.set_attribute("duration_ms", float(duration_ms))
        if store_error:
            from lab_arena.store import safe_store_error_metadata
            for key, value in safe_store_error_metadata(store_error).items():
                span.set_attribute("arena.store_" + key, value)
        if run_identity is not None:
            for key, value in run_identity.items():
                span.set_attribute(key, value)
        if outcome in ("failed", "uncertain"):
            span.set_status(Status(StatusCode.ERROR))
        span.end()

    # -- run outcomes ----------------------------------------------------

    def record_run(
        self, run_kind: str, terminal_cause: str,
        *, run_identity: Optional[Mapping[str, Any]] = None,
        failure_stage: str = ARENA_NO_ERROR,
        failure_class: str = ARENA_NO_ERROR,
    ) -> None:
        try:
            if self._run_tracer is None:
                return
            if run_identity is not None and set(run_identity) != RUN_IDENTITY_ATTRIBUTE_ALLOWLIST:
                return
            from opentelemetry.trace import Status, StatusCode

            span = self._start(
                self._run_tracer, "arena.run.%s" % run_kind, None
            )
            outcome = "ok" if terminal_cause == "accepted" else "failed"
            span.set_attribute("arena.run_kind", str(run_kind))
            span.set_attribute("arena.terminal_cause", str(terminal_cause))
            span.set_attribute("arena.outcome", outcome)
            span.set_attribute("arena.failure_stage", str(failure_stage))
            span.set_attribute("arena.failure_class", str(failure_class))
            if run_identity is not None:
                for key, value in run_identity.items():
                    span.set_attribute(key, value)
            if outcome == "failed":
                span.set_status(Status(StatusCode.ERROR))
            span.end()
        except BaseException as exc:
            self._log_emit_failure(exc)

    # -- accepted runtime uploads --------------------------------------

    def record_runtime_upload(
        self,
        *,
        inserted_count: int,
        replayed_count: int,
        has_started: bool,
        has_finished: bool,
        has_error: bool,
        run_identity: Mapping[str, Any],
    ) -> None:
        try:
            if self._runtime_tracer is None:
                return
            if set(run_identity) != RUN_IDENTITY_ATTRIBUTE_ALLOWLIST:
                return
            span = self._start(self._runtime_tracer, "arena.runtime.upload", None)
            span.set_attribute("arena.inserted_count", inserted_count)
            span.set_attribute("arena.replayed_count", replayed_count)
            span.set_attribute("arena.has_started", has_started)
            span.set_attribute("arena.has_finished", has_finished)
            span.set_attribute("arena.has_error", has_error)
            for key, value in run_identity.items():
                span.set_attribute(key, value)
            span.end()
        except BaseException as exc:
            self._log_emit_failure(exc)

    # -- committed submission reviews -----------------------------------

    def record_review(
        self, status: str, *, error_code: str, categories: str,
        provider_http_status: int, retryable: bool, funding_route: str,
        round_id: str, submission_id: str, miner_hotkey: str, attempt: int,
    ) -> None:
        try:
            if self._review_tracer is None:
                return
            from opentelemetry.trace import Status, StatusCode

            span = self._start(self._review_tracer, "arena.review.decision", None)
            span.set_attribute("arena.review_status", status)
            span.set_attribute("arena.error_code", error_code)
            span.set_attribute("arena.review_categories", categories)
            span.set_attribute("arena.provider_http_status", provider_http_status)
            span.set_attribute("arena.retryable", retryable)
            span.set_attribute("arena.funding_route", funding_route)
            span.set_attribute("arena.round_id", round_id)
            span.set_attribute("arena.submission_id", submission_id)
            span.set_attribute("arena.miner_hotkey", miner_hotkey)
            span.set_attribute("arena.attempt", attempt)
            if status == "error":
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
        source_sha: Optional[str] = None,
        store_error: Optional[Mapping[str, Any]] = None,
        billing_lookup: Optional[Mapping[str, Any]] = None,
        round_scope: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            self._record(
                stage,
                outcome,
                error_type=error_type,
                count=count,
                duration_ms=duration_ms,
                start_ns=start_ns,
                source_sha=source_sha,
                store_error=store_error,
                billing_lookup=billing_lookup,
                round_scope=round_scope,
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
        source_sha: Optional[str],
        store_error: Optional[Mapping[str, Any]],
        billing_lookup: Optional[Mapping[str, Any]],
        round_scope: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from opentelemetry.trace import Status, StatusCode

        span = self._start(self._tracer, "arena.%s" % stage, start_ns)
        span.set_attribute("arena.stage", str(stage))
        span.set_attribute("arena.outcome", str(outcome))
        span.set_attribute("arena.error_type", str(error_type))
        span.set_attribute("arena.count", int(count))
        span.set_attribute("duration_ms", float(duration_ms))
        if source_sha is not None:
            span.set_attribute("arena.source_sha", str(source_sha))
        if stage.startswith("advance_") and isinstance(round_scope, Mapping):
            round_id = round_scope.get("round_id")
            deadline = round_scope.get("scoring_deadline")
            if (type(round_id) is str and _ARENA_ROUND_ID_RE.fullmatch(round_id)
                    and type(deadline) is str
                    and _ARENA_ISO_UTC_RE.fullmatch(deadline)):
                span.set_attribute("arena.round_id", round_id)
                span.set_attribute("arena.scoring_deadline", deadline)
        if stage == "reconcile_deepline_billing_lookup" and isinstance(billing_lookup, Mapping):
            round_id = billing_lookup.get("round_id")
            lane = billing_lookup.get("lane")
            cursor = billing_lookup.get("cursor_entry_id")
            run_scoped = billing_lookup.get("run_scoped")
            if (type(round_id) is str and _ARENA_ROUND_ID_RE.fullmatch(round_id)
                    and type(lane) is str and lane in BILLING_LOOKUP_LANES
                    and type(cursor) is int and 0 <= cursor <= 9223372036854775807
                    and type(run_scoped) is bool):
                span.set_attribute("arena.round_id", round_id)
                span.set_attribute("arena.billing_lane", lane)
                span.set_attribute("arena.cursor_entry_id", cursor)
                span.set_attribute("arena.run_scoped", run_scoped)
        if store_error:
            from lab_arena.store import safe_store_error_metadata

            for key, value in safe_store_error_metadata(store_error).items():
                span.set_attribute("arena.store_" + key, value)
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
            return _registered_route_templates(app)

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
                ARENA_RUNTIME_SCOPE: "runtime",
                ARENA_GATE_SCOPE: "gate",
                ARENA_REVIEW_SCOPE: "review",
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
            runtime_tracer=provider.get_tracer(ARENA_RUNTIME_SCOPE),
            gate_tracer=provider.get_tracer(ARENA_GATE_SCOPE),
            review_tracer=provider.get_tracer(ARENA_REVIEW_SCOPE),
            flush_spans=lambda timeout: provider.force_flush(timeout_millis=timeout),
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
                round_id = _matched_http_round_id(scope, route)
                if round_id is not None:
                    span.set_attribute("arena.round_id", round_id)
                denial = telemetry.take_denial()
                span.set_attribute("arena.denial", denial)
                if int(status_code) == 503 and denial == "arena_store_unavailable":
                    for key, value in telemetry.take_store_error().items():
                        span.set_attribute("arena.store_" + key, value)
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
