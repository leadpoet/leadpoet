"""Arena durable-state access through the restricted PostgREST service role.

Writes use the SECURITY DEFINER RPCs declared by the numbered SQL migrations;
reads use bounded table selects. The HTTP/1.1 client sends the scoped
``sb_secret_`` API key only as ``apikey``.

``PsycopgTransport`` supports tests and local tooling. It calls the same SQL
functions through a PostgreSQL driver so disposable-PostgreSQL coverage
exercises the production function contract.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
import json
import re
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import httpx

from lab_arena.contracts import (
    ArenaContractError,
    LEASE_TTL_SECONDS,
    RUNNER_SLOT_CEILING,
    canonical_json,
    validate_submission_costs,
)
from lab_arena.owner_admission import OwnerAdmission

WHOAMI_SCHEMA_VERSION = "leadpoet.lab_arena.whoami.v1"
CODE_REVIEW_SCHEMA_VERSION = "leadpoet.lab_arena.code_review.v1"
VALIDATOR_SCORING_AUTHORITY_SCHEMA_VERSION = (
    "leadpoet.lab_arena.validator_scoring_authority.v1"
)
COMPANY_QUALITY_SCHEMA_VERSION = "leadpoet.lab_arena.company_quality_schema.v1"
SUCCESSFUL_CALL_COST_SCHEMA_VERSION = (
    "leadpoet.lab_arena.successful_call_cost_schema.v1"
)
PER_ICP_COST_SCHEMA_VERSION = "leadpoet.lab_arena.per_icp_cost_schema.v1"
PARALLEL_EXECUTION_SCHEMA_VERSION = (
    "leadpoet.lab_arena.parallel_execution_schema.v1"
)
PARALLEL_EXECUTION_SCHEMA_V2_VERSION = (
    "leadpoet.lab_arena.parallel_execution_schema.v2"
)
DYNAMIC_BENCHMARK_SCHEMA_VERSION = "leadpoet.lab_arena.dynamic_benchmark_schema.v1"
RUN_QUOTA_SNAPSHOT_SCHEMA_VERSION = "leadpoet.lab_arena.quota_snapshot.v1"
SERVICE_ROLE_NAME = "lab_arena_service"

# Parameter order and PostgreSQL casts for every service-callable function.
# PostgREST matches JSON keys to parameter names; the psycopg transport uses
# named notation with explicit casts so both transports share this table.
SCORE_BATCH_SIZE = 500

FUNCTION_SIGNATURES: Dict[str, Sequence[tuple]] = {
    "lab_arena_append_validator_events_v1": (
        ("p_validator_hotkey", "text"), ("p_network", "text"),
        ("p_netuid", "integer"), ("p_events", "jsonb"),
        ("p_gateway_source_commit", "text"),
    ),
    "lab_arena_retry_credit_failures_v1": (
        ("p_round_id", "text"), ("p_submission_id", "text"),
        ("p_miner_hotkey", "text"), ("p_request_hash", "text"),
    ),
    "lab_arena_operator_hold_active_v1": (),
    "lab_arena_append_trajectory_events_v1": (
        ("p_run_id", "text"),
        ("p_lease_token_hash", "text"),
        ("p_events", "jsonb"),
    ),
    "lab_arena_per_icp_cost_schema_v1": (),
    "lab_arena_deepline_catalog_schema_v1": (),
    "lab_arena_next_closed_deepline_reconciliation_v1": (
        ("p_mode", "text"), ("p_network_name", "text"),
        ("p_netuid", "integer"), ("p_round_id", "text"),
        ("p_after_entry_id", "bigint"),
    ),
    "lab_arena_next_closed_provider_reconciliation_v1": (
        ("p_mode", "text"), ("p_network_name", "text"),
        ("p_netuid", "integer"), ("p_round_id", "text"),
        ("p_after_entry_id", "bigint"),
    ),
    "lab_arena_freeze_champion_funding": (("p_round_id", "text"),),
    "lab_arena_provider_funding": (("p_run_id", "text"), ("p_provider", "text")),
    "lab_arena_run_quota_snapshot_v1": (
        ("p_run_id", "text"),
        ("p_lease_token_hash", "text"),
    ),
    "lab_arena_run_quota_snapshot_v2": (
        ("p_run_id", "text"),
        ("p_lease_token_hash", "text"),
    ),
    "lab_arena_mark_champion_provider_fallback": (
        ("p_run_id", "text"), ("p_lease_token_hash", "text"),
        ("p_provider", "text"), ("p_evidence", "jsonb"),
    ),
    "lab_arena_champion_funding_schema_v1": (),
    "lab_arena_whoami": (),
    "lab_arena_has_recent_participation_v1": (
        ("p_network", "text"),
        ("p_netuid", "integer"),
        ("p_runner_hotkey", "text"),
    ),
    "lab_arena_schema_version_v1": (),
    "lab_arena_code_review_schema_v1": (),
    "lab_arena_participation_schema_v1": (),
    "lab_arena_validator_scoring_authority_schema_v1": (),
    "lab_arena_integrity_schema_v1": (),
    "lab_arena_twenty_icp_promotion_schema_v1": (),
    "lab_arena_baseline_cost_eligibility_schema_v1": (),
    "lab_arena_parallel_execution_schema_v1": (),
    "lab_arena_parallel_execution_schema_v2": (),
    "lab_arena_dynamic_benchmark_schema_v1": (),
    "lab_arena_submission_replacement_schema_v1": (),
    "lab_arena_contact_schema_v1": (),
    "lab_arena_company_quality_schema_v1": (),
    "lab_arena_successful_call_cost_schema_v1": (),
    "lab_arena_weight_state_schema_v1": (),
    "lab_arena_current_daily_icp_set": (("p_set_id", "bigint"),),
    "lab_arena_submission_costs": (("p_submission_id", "text"),),
    "lab_arena_icp_cost_eligibility": (
        ("p_round_id", "text"),
        ("p_submission_id", "text"),
        ("p_icp_position", "integer"),
        ("p_qualified_company_count", "integer"),
    ),
    "lab_arena_commit_round_v2": (
        ("p_round_id", "text"),
        ("p_participants", "jsonb"),
        ("p_benchmark_ref", "text"),
        ("p_evaluation_date", "text"),
        ("p_icp_set_date", "date"),
        ("p_scorer_image_digest", "text"),
        ("p_scorer_image_reference", "text"),
    ),
    "lab_arena_commit_round_v3": (
        ("p_round_id", "text"), ("p_participants", "jsonb"),
        ("p_benchmark_ref", "text"), ("p_evaluation_date", "text"),
        ("p_icp_set_date", "date"), ("p_scorer_image_digest", "text"),
        ("p_scorer_image_reference", "text"), ("p_deepline_catalog", "jsonb"),
    ),
    "lab_arena_create_round": (("p_round_id", "text"), ("p_configuration_doc", "jsonb")),
    "lab_arena_transition_round": (("p_round_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_activate_reward": (("p_round_id", "text"), ("p_reward_basis", "jsonb"), ("p_signing_key_doc", "jsonb")),
    "lab_arena_reward_slot_snapshot": (("p_round_id", "text"), ("p_slot_policy", "jsonb")),
    "lab_arena_prepare_promotion": (("p_round_id", "text"), ("p_plan", "jsonb")),
    "lab_arena_complete_promotion": (("p_round_id", "text"), ("p_plan", "jsonb")),
    "lab_arena_register_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_doc", "jsonb")),
    "lab_arena_register_submission_v2": (
        ("p_round_id", "text"),
        ("p_submission_id", "text"),
        ("p_miner_hotkey", "text"),
        ("p_doc", "jsonb"),
        ("p_owner_coldkey", "text"),
        ("p_owner_block_number", "bigint"),
        ("p_owner_block_hash", "text"),
    ),
    "lab_arena_update_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_accept_submission_with_credentials": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_credentials", "jsonb")),
    "lab_arena_accept_submission_source_with_credentials": (
        ("p_round_id", "text"), ("p_submission_id", "text"),
        ("p_miner_hotkey", "text"), ("p_credentials", "jsonb"),
        ("p_archive_sha256", "text"), ("p_normalized_sha256", "text"),
    ),
    "lab_arena_submission_duplicate_schema_v1": (),
    "lab_arena_submission_similarity_champion": (("p_round_id", "text"),),
    "lab_arena_get_submission_credential": (("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_provider", "text")),
    "lab_arena_begin_submission_review": (
        ("p_submission_id", "text"),
        ("p_miner_hotkey", "text"),
        ("p_claim_token_hash", "text"),
        ("p_reservation_microusd", "bigint"),
        ("p_review_model", "text"),
        ("p_file_count", "integer"),
        ("p_source_bytes", "bigint"),
    ),
    "lab_arena_finish_submission_review": (
        ("p_submission_id", "text"),
        ("p_miner_hotkey", "text"),
        ("p_claim_token_hash", "text"),
        ("p_status", "text"),
        ("p_review_doc", "jsonb"),
        ("p_actual_microusd", "bigint"),
    ),
    "lab_arena_open_stage": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_participants", "jsonb"), ("p_icp_positions", "integer[]")),
    "lab_arena_open_parallel_execution_v1": (("p_round_id", "text"), ("p_participants", "jsonb")),
    "lab_arena_close_parallel_execution_v1": (("p_round_id", "text"),),
    "lab_arena_activate_preexecuted_stage2_v1": (("p_round_id", "text"),),
    "lab_arena_claim_assignment": (("p_round_id", "text"), ("p_runner_hotkey", "text"), ("p_declared_parallelism", "integer"), ("p_slot_ceiling", "integer"), ("p_excluded_miner_hotkeys", "text[]"), ("p_request_id", "text"), ("p_request_hash", "text"), ("p_lease_token_hash", "text"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_reserve_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_operation_id", "text"), ("p_provider", "text"), ("p_funding_source", "text"), ("p_amount_microusd", "bigint"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_reserve_judgment_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_operation_id", "text"), ("p_provider", "text"), ("p_funding_source", "text"), ("p_amount_microusd", "bigint"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_dispatched": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text")),
    "lab_arena_settle_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_actual_microusd", "bigint"), ("p_terminal_response", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_uncertain": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_list_openrouter_cost_reconciliations_v1": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_after_entry_id", "bigint"),
        ("p_limit", "integer"),
    ),
    "lab_arena_reconcile_openrouter_cost_v1": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_call_identity", "text"),
        ("p_uncertain_entry_id", "bigint"),
        ("p_generation_id", "text"),
        ("p_credential_fingerprint", "text"),
        ("p_actual_microusd", "bigint"),
        ("p_cost_units", "text"),
    ),
    "lab_arena_list_deepline_cost_reconciliations_v1": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_after_entry_id", "bigint"),
        ("p_limit", "integer"),
    ),
    "lab_arena_list_deepline_cost_reconciliations_v2": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_after_entry_id", "bigint"),
        ("p_limit", "integer"),
        ("p_successful_execute_only", "boolean"),
    ),
    "lab_arena_list_deepline_cost_reconciliations_v3": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_after_entry_id", "bigint"),
        ("p_limit", "integer"),
    ),
    "lab_arena_reconcile_deepline_cost_v1": (
        ("p_round_id", "text"),
        ("p_run_id", "text"),
        ("p_call_identity", "text"),
        ("p_uncertain_entry_id", "bigint"),
        ("p_request_id", "text"),
        ("p_operation", "text"),
        ("p_credential_fingerprint", "text"),
        ("p_actual_microusd", "bigint"),
        ("p_cost_units", "text"),
    ),
    "lab_arena_reconcile_deepline_cost_v2": (
        ("p_round_id", "text"), ("p_run_id", "text"),
        ("p_call_identity", "text"), ("p_uncertain_entry_id", "bigint"),
        ("p_request_id", "text"), ("p_operation", "text"),
        ("p_credential_fingerprint", "text"), ("p_actual_microusd", "bigint"),
        ("p_cost_units", "text"), ("p_execution_key", "text"),
        ("p_recovered_request_id", "text"),
    ),
    "lab_arena_deepline_response_schema_v1": (),
    "lab_arena_recover_deepline_response_v1": (
        ("p_run_id", "text"), ("p_lease_token_hash", "text"),
        ("p_call_identity", "text"), ("p_request_hash", "text"),
        ("p_execution_key", "text"), ("p_credential_fingerprint", "text"),
        ("p_request_id", "text"), ("p_operation", "text"),
        ("p_actual_microusd", "bigint"), ("p_terminal_response", "jsonb"),
        ("p_lease_ttl_seconds", "integer"),
    ),
    "lab_arena_complete_attempt": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_result", "jsonb"), ("p_terminal_cause", "text"), ("p_output_ref", "text")),
    "lab_arena_complete_attempt_v2": (
        ("p_run_id", "text"),
        ("p_lease_token_hash", "text"),
        ("p_result", "jsonb"),
        ("p_terminal_cause", "text"),
        ("p_output_ref", "text"),
        ("p_judgment_evidence", "jsonb"),
        ("p_judgment_evidence_hash", "text"),
    ),
    "lab_arena_complete_attempt_v3": (
        ("p_run_id", "text"),
        ("p_lease_token_hash", "text"),
        ("p_result", "jsonb"),
        ("p_terminal_cause", "text"),
        ("p_output_ref", "text"),
        ("p_output_hash", "text"),
        ("p_company_judgment_evidence", "jsonb"),
        ("p_completion_request_hash", "text"),
    ),
    "lab_arena_expire_leases": (("p_round_id", "text"),),
    "lab_arena_close_stage": (("p_round_id", "text"), ("p_stage", "smallint")),
    "lab_arena_open_scoring": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_work_items", "jsonb")),
    "lab_arena_open_scoring_v2": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_work_items", "jsonb")),
    "lab_arena_open_scoring_v3": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_work_items", "jsonb")),
    "lab_arena_close_scoring": (("p_round_id", "text"), ("p_stage", "smallint")),
    "lab_arena_cancel_round": (("p_round_id", "text"), ("p_reason", "text")),
    "lab_arena_record_run_scores": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_scores", "jsonb")),
    "lab_arena_publish_weight_state_v1": (("p_network", "text"), ("p_netuid", "integer"), ("p_epoch", "bigint"), ("p_state_hash", "text"), ("p_state_doc", "jsonb")),
    "lab_arena_record_chain_outcome_v1": (("p_network", "text"), ("p_netuid", "integer"), ("p_epoch", "bigint"), ("p_validator_hotkey", "text"), ("p_request_id", "text"), ("p_extrinsic_hash", "text"), ("p_outcome_doc", "jsonb")),
}

TABLES = (
    "lab_arena_rounds",
    "lab_arena_published_results_v1",
    "lab_arena_submissions",
    "lab_arena_runs",
    "lab_arena_ledger",
    "lab_arena_trajectory_events",
    "lab_arena_validator_events",
    "lab_arena_accepted_weight_states",
    "lab_arena_chain_outcomes",
    "lab_arena_judgment_cache",
    "lab_arena_company_judgments",
)
ROUND_MODE_FILTER = "configuration_doc->>mode"
PROMOTION_OUTCOME_FILTER = "publication_doc->king_decision->>outcome"
COMPETITION_CONFIGURATION_FIELDS = (
    "mode", "network_name", "netuid", "schedule",
    "benchmark_disclosure_policy", "stage_1_icp_count", "stage_2_icp_count",
    "promotion_margin", "execution_sequence_policy", "sourcing_cost_eligibility_policy",
    "recovery_source_round_id",
)
CURRENT_CONFIGURATION_FIELDS = (
    "mode", "schedule", "stage_1_icp_count", "stage_2_icp_count", "promotion_margin",
    "integrity_policy", "contact_policy", "company_quality_policy", "intent_details_policy",
)
PUBLIC_RESULTS_CONFIGURATION_FIELDS = (
    "mode", "network_name", "netuid", "schedule",
    "benchmark_disclosure_policy", "stage_1_icp_count", "stage_2_icp_count",
    "integrity_policy", "contact_policy", "company_quality_policy",
    "intent_details_policy", "scorer_policy",
)
# Fixed public audit projections. Never fetch source or provider documents.
_RUNTIME_JSON_COLUMNS = {
    "publication_doc:lab_arena_competition_publication_v1":
        "public.lab_arena_competition_publication_v1(lab_arena_rounds) AS publication_doc",
    "source_commit:result_doc->resource_summary->>validator_source_commit": "result_doc #>> '{resource_summary,validator_source_commit}' AS source_commit",
    "source_dirty:result_doc->resource_summary->>validator_source_dirty": "result_doc #>> '{resource_summary,validator_source_dirty}' AS source_dirty",
    "source_commit:content->>validator_source_commit": "content ->> 'validator_source_commit' AS source_commit",
    "source_dirty:content->>validator_source_dirty": "content ->> 'validator_source_dirty' AS source_dirty",
    "start_lease_generation:content->>lease_generation": "content ->> 'lease_generation' AS start_lease_generation",
    "status:content->>status": "content ->> 'status' AS status",
    "error_class:content->>error_class": "content ->> 'error_class' AS error_class",
    "failure_stage:content->>failure_stage": "content ->> 'failure_stage' AS failure_stage",
    **{
        "cfg_%s:configuration_doc->%s::text" % (key, key):
        "(configuration_doc -> '%s')::text AS cfg_%s" % (key, key)
        for key in COMPETITION_CONFIGURATION_FIELDS + CURRENT_CONFIGURATION_FIELDS
        + PUBLIC_RESULTS_CONFIGURATION_FIELDS + ("baseline_hotkey",)
    },
}
ROUND_NETWORK_COLUMN = "arena_network_name"
ROUND_NETUID_COLUMN = "arena_netuid"


DEADLOCK_SQLSTATE = "40P01"
DEADLOCK_RETRIES = 3
# Stage creation, scoring queue creation, and score batches can
# take longer than an ordinary RPC. Let their bounded database work return a
# result before giving up on the response.
BULK_ROUND_RPC_READ_TIMEOUT_SECONDS = 65.0
# Publication validates the complete frozen ranking in one database statement.
PUBLICATION_RPC_READ_TIMEOUT_SECONDS = 605.0
# Closed-round billing scans historical ledger entries; one slow read must not
# use the short timeout shared with ordinary RPCs.
CLOSED_PROVIDER_RECONCILIATION_READ_TIMEOUT_SECONDS = 20.0


class ArenaStoreError(RuntimeError):
    """A durable-state operation failed. Messages never carry credentials."""


class ArenaStoreUnavailable(ArenaStoreError):
    """A transient database transport failure; RPC transport errors are not replayed."""


class ArenaRoleError(ArenaStoreError):
    """The database identity is not the least-privilege Arena service role."""


def new_lease_token() -> str:
    return secrets.token_hex(32)


def hash_lease_token(token: str) -> str:
    return "sha256:" + hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def _check_filter_value(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in ",.()\"'\n\r\t") or len(text) > 200:
        raise ArenaStoreError("filter value contains reserved characters")
    return text


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class StoreTransport:
    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def select(
        self,
        table: str,
        *,
        filters: Optional[Mapping[str, Any]] = None,
        order: Optional[str] = None,
        descending: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        after_run_id: Optional[str] = None,
        after_entry_id: Optional[int] = None,
        before_round: Optional[tuple[str, str]] = None,
        status_in: Optional[Sequence[str]] = None,
        submission_ids: Optional[Sequence[str]] = None,
        run_ids: Optional[Sequence[str]] = None,
        cache_keys: Optional[Sequence[str]] = None,
        columns: str = "*",
    ) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        return None


def create_http1_client(timeout_seconds: float) -> httpx.Client:
    """The HTTP/1.1-pinned construction copied from ``gateway/db/client.py``.

    postgrest-py enables HTTP/2 by default, but its shared HPACK encoder is
    not safe when multiple threads encode headers concurrently. HTTP/1.1 keeps
    connection pooling and parallel requests without that shared table.
    """

    return httpx.Client(
        http1=True,
        http2=False,
        timeout=httpx.Timeout(float(timeout_seconds)),
        follow_redirects=False,
        trust_env=False,
    )


class PostgrestTransport(StoreTransport):
    """PostgREST over HTTP/1.1 with one least-privilege credential."""

    def __init__(
        self,
        base_url: str,
        *,
        service_key: str,
        timeout_seconds: float = 8.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        parsed = urlsplit(base_url)
        is_https = parsed.scheme == "https" and bool(parsed.hostname)
        is_loopback_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not (is_https or is_loopback_http)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ArenaStoreError("PostgREST base URL must be https (or loopback for tests)")
        if not service_key.startswith("sb_secret_"):
            raise ArenaStoreError("scoped service key has an invalid shape")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "apikey": service_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client = http_client or create_http1_client(timeout_seconds)
        self.deadlock_retries = 0

    def __repr__(self) -> str:  # never expose the token
        return "PostgrestTransport(%r)" % self._base_url

    def _raise_for_status(self, response: httpx.Response, context: str) -> None:
        if 200 <= response.status_code < 300:
            return
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = " code=%s message=%s" % (body.get("code"), str(body.get("message") or "")[:200])
        except ValueError:
            detail = ""
        raise ArenaStoreError("%s failed: HTTP %d%s" % (context, response.status_code, detail))

    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:
        if function not in FUNCTION_SIGNATURES:
            raise ArenaStoreError("unknown Arena function")
        content = canonical_json(dict(params)).encode("utf-8")
        request_options = {}
        read_timeout = None
        if function in {
            "lab_arena_open_stage",
            "lab_arena_open_scoring",
            "lab_arena_open_scoring_v2",
            "lab_arena_open_scoring_v3",
            "lab_arena_record_run_scores",
        }:
            read_timeout = BULK_ROUND_RPC_READ_TIMEOUT_SECONDS
        elif (
            function == "lab_arena_transition_round"
            and params.get("p_expected_status") == "scored"
            and params.get("p_next_status") == "published"
        ):
            read_timeout = PUBLICATION_RPC_READ_TIMEOUT_SECONDS
        elif function == "lab_arena_next_closed_provider_reconciliation_v1":
            read_timeout = CLOSED_PROVIDER_RECONCILIATION_READ_TIMEOUT_SECONDS
        if read_timeout is not None:
            timeout = self._client.timeout
            request_options["timeout"] = httpx.Timeout(
                connect=timeout.connect,
                read=read_timeout,
                write=timeout.write,
                pool=timeout.pool,
            )
        for attempt in range(DEADLOCK_RETRIES + 1):
            try:
                response = self._client.post("%s/rest/v1/rpc/%s" % (self._base_url, function), headers=self._headers, content=content, **request_options)
            except httpx.TransportError as exc:
                # A mutating RPC may have committed before its response failed.
                # Surface availability to the API, but never replay the POST.
                raise ArenaStoreUnavailable("rpc %s transport failure: %s" % (function, type(exc).__name__)) from exc
            except httpx.HTTPError as exc:
                raise ArenaStoreError("rpc %s transport failure: %s" % (function, type(exc).__name__)) from exc
            if response.status_code >= 400 and attempt < DEADLOCK_RETRIES:
                try:
                    code = response.json().get("code")
                except ValueError:
                    code = None
                if code == DEADLOCK_SQLSTATE:
                    self.deadlock_retries += 1
                    time.sleep(0.01 * (attempt + 1))
                    continue
            if (
                function == "lab_arena_append_trajectory_events_v1"
                and response.status_code in (502, 503, 504)
            ):
                # This RPC is idempotent by (run_id, event_id), so its caller
                # can safely replay one ambiguous upstream availability result.
                # Other mutating RPCs retain the no-replay policy.
                raise ArenaStoreUnavailable(
                    "rpc %s temporarily unavailable: HTTP %d"
                    % (function, response.status_code)
                )
            self._raise_for_status(response, "rpc %s" % function)
            try:
                return response.json()
            except ValueError as exc:
                if function == "lab_arena_append_trajectory_events_v1":
                    # A successful non-JSON response can follow a committed
                    # insert. Replaying the same event UUID is idempotent.
                    raise ArenaStoreUnavailable(
                        "rpc %s returned non-JSON" % function
                    ) from exc
                raise ArenaStoreError("rpc %s returned non-JSON" % function) from exc
        raise ArenaStoreError("rpc %s failed after deadlock retries" % function)

    def select(
        self,
        table,
        *,
        filters=None,
        order=None,
        descending=False,
        limit=None,
        offset=None,
        after_run_id=None,
        after_entry_id=None,
        before_round=None,
        status_in=None,
        submission_ids=None,
        run_ids=None,
        cache_keys=None,
        columns="*",
    ):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        query: List[tuple] = [("select", columns)]
        for key, value in (filters or {}).items():
            if key not in (ROUND_MODE_FILTER, PROMOTION_OUTCOME_FILTER) and not key.replace("_", "").isalnum():
                raise ArenaStoreError("invalid filter column")
            if (key == "event_kind" and value in ("runtime.started", "runtime.error")
                    and table == "lab_arena_trajectory_events"):
                query.append((key, "eq." + value))
            elif key == "operation_id" and value is not None:
                operation = str(value)
                if not re.fullmatch(r"[a-z0-9_.]{1,64}", operation):
                    raise ArenaStoreError("invalid operation filter")
                query.append((key, "eq." + operation))
            else:
                query.append((key, "is.null" if value is None else "eq." + _check_filter_value(value)))
        if status_in is not None:
            if table != "lab_arena_rounds" or "status" in (filters or {}):
                raise ArenaStoreError("status inclusion filter is invalid")
            statuses = tuple(_check_filter_value(value) for value in status_in)
            if not statuses:
                raise ArenaStoreError("status inclusion filter is empty")
            query.append(("status", "in.(%s)" % ",".join(statuses)))
        if submission_ids is not None:
            if table not in ("lab_arena_runs", "lab_arena_trajectory_events") or "submission_id" in (filters or {}):
                raise ArenaStoreError("submission inclusion filter is invalid")
            ids = tuple(_check_filter_value(value) for value in submission_ids)
            if not ids:
                raise ArenaStoreError("submission inclusion filter is empty")
            query.append(("submission_id", "in.(%s)" % ",".join(ids)))
        if run_ids is not None:
            if table not in ("lab_arena_runs", "lab_arena_trajectory_events") or "run_id" in (filters or {}):
                raise ArenaStoreError("run inclusion filter is invalid")
            ids = tuple(str(value) for value in run_ids)
            if not ids or any(not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", value) for value in ids):
                raise ArenaStoreError("run inclusion ids are invalid")
            query.append(("run_id", "in.(%s)" % ",".join(ids)))
        if cache_keys is not None:
            if table != "lab_arena_judgment_cache" or "cache_key" in (filters or {}):
                raise ArenaStoreError("cache-key inclusion filter is invalid")
            keys = tuple(_check_filter_value(value) for value in cache_keys)
            if not keys:
                raise ArenaStoreError("cache-key inclusion filter is empty")
            query.append(("cache_key", "in.(%s)" % ",".join(keys)))
        if order == "history_round":
            if table != "lab_arena_rounds" or not descending:
                raise ArenaStoreError("history cursor requires descending rounds")
            query.append(("order", "created_at.desc,round_id.desc"))
        elif order:
            query.append(("order", "%s.%s" % (order, "desc" if descending else "asc")))
        if before_round is not None:
            if table != "lab_arena_rounds" or order != "history_round":
                raise ArenaStoreError("history cursor requires descending rounds")
            created, round_id = str(before_round[0]), _check_filter_value(before_round[1])
            try:
                parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError()
                created = parsed.isoformat()
            except ValueError:
                raise ArenaStoreError("history timestamp is invalid")
            query.append(("or", "(created_at.lt.%s,and(created_at.eq.%s,round_id.lt.%s))" % (created, created, round_id)))
        if limit is not None:
            query.append(("limit", str(int(limit))))
        if offset is not None:
            offset = int(offset)
            if offset < 0:
                raise ArenaStoreError("offset must be nonnegative")
            query.append(("offset", str(offset)))
        if after_run_id is not None:
            if table != "lab_arena_runs" or order != "run_id" or "run_id" in (filters or {}):
                raise ArenaStoreError("run cursor requires ordered Arena runs")
            query.append(("run_id", "gt." + _check_filter_value(after_run_id)))
        if after_entry_id is not None:
            if (table != "lab_arena_ledger" or order != "entry_id"
                    or descending or "entry_id" in (filters or {})
                    or type(after_entry_id) is not int or after_entry_id < 0):
                raise ArenaStoreError("ledger cursor requires ordered Arena ledger")
            query.append(("entry_id", "gt." + str(after_entry_id)))
        # Only SELECT is safe to replay after an ambiguous read failure.
        # RPCs can mutate state and retain their separate retry policy.
        for attempt in range(2):
            try:
                response = self._client.get(
                    "%s/rest/v1/%s" % (self._base_url, table),
                    headers=self._headers,
                    params=query,
                )
                break
            except (httpx.ReadError, httpx.ReadTimeout) as exc:
                if attempt:
                    raise ArenaStoreUnavailable(
                        "select %s transport failure: %s" % (table, type(exc).__name__)
                    ) from exc
            except httpx.HTTPError as exc:
                raise ArenaStoreError("select %s transport failure: %s" % (table, type(exc).__name__)) from exc
        self._raise_for_status(response, "select %s" % table)
        rows = response.json()
        if not isinstance(rows, list):
            raise ArenaStoreError("select %s returned a non-list" % table)
        return rows

    def close(self) -> None:
        self._client.close()


class PsycopgTransport(StoreTransport):
    """Test/local transport calling the same functions through psycopg2.

    Connections come from a bounded pool so concurrency tests that model
    separate service instances never exhaust the server. ``role`` is applied
    with ``SET ROLE`` on every connection so the least-privilege grants are
    exercised.
    """

    def __init__(self, connect: Callable[[], Any], *, role: Optional[str] = SERVICE_ROLE_NAME, pool_size: int = 6) -> None:
        import queue

        self._connect = connect
        self._role = role
        self._pool_size = max(1, int(pool_size))
        self._idle: "queue.LifoQueue[Any]" = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()
        self._connections: List[Any] = []
        self._closed = False
        self.deadlock_retries = 0
        self.last_deadlock_detail = ""

    def _acquire(self):
        import queue

        with self._lock:
            if self._closed:
                raise ArenaStoreError("transport is closed")
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self._pool_size:
                self._created += 1
                create = True
            else:
                create = False
        if create:
            try:
                connection = self._connect()
                connection.autocommit = True
                if self._role:
                    with connection.cursor() as cursor:
                        cursor.execute("SET ROLE %s" % self._role)
            except Exception:
                with self._lock:
                    self._created -= 1
                raise
            with self._lock:
                self._connections.append(connection)
            return connection
        return self._idle.get(timeout=120)

    def _release(self, connection: Any) -> None:
        if getattr(connection, "closed", 0):
            with self._lock:
                self._created -= 1
            return
        self._idle.put(connection)

    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:
        signature = FUNCTION_SIGNATURES.get(function)
        if signature is None:
            raise ArenaStoreError("unknown Arena function")
        names = [name for name, _ in signature]
        unknown = set(params) - set(names)
        if unknown:
            raise ArenaStoreError("unknown parameters for %s: %s" % (function, sorted(unknown)))
        placeholders = []
        values = []
        for name, cast in signature:
            if name not in params:
                continue
            value = params[name]
            if cast == "jsonb":
                value = json.dumps(value, sort_keys=True) if value is not None else None
            placeholders.append("%s => %%s::%s" % (name, cast))
            values.append(value)
        sql = "SELECT public.%s(%s)" % (function, ", ".join(placeholders))
        for attempt in range(DEADLOCK_RETRIES + 1):
            connection = self._acquire()
            try:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, values)
                        row = cursor.fetchone()
                    return row[0] if row else None
                except Exception as exc:  # psycopg2 errors carry no secrets here
                    if getattr(exc, "pgcode", None) == DEADLOCK_SQLSTATE and attempt < DEADLOCK_RETRIES:
                        # The aborted call changed nothing; every Arena function is
                        # idempotent by identity, so one bounded retry is safe.
                        self.deadlock_retries += 1
                        self.last_deadlock_detail = (getattr(getattr(exc, "diag", None), "message_detail", None) or "")[:2000]
                        time.sleep(0.01 * (attempt + 1))
                        continue
                    diag = getattr(exc, "diag", None)
                    detail = (getattr(diag, "message_detail", None) or "")[:2000]
                    context = (getattr(diag, "context", None) or "")[:600]
                    raise ArenaStoreError("rpc %s failed: %s%s%s" % (function, str(exc).splitlines()[0][:200], (" [" + detail + "]") if detail else "", (" {" + context + "}") if context else "")) from exc
            finally:
                self._release(connection)
        raise ArenaStoreError("rpc %s failed after deadlock retries" % function)

    def select(
        self,
        table,
        *,
        filters=None,
        order=None,
        descending=False,
        limit=None,
        offset=None,
        after_run_id=None,
        after_entry_id=None,
        before_round=None,
        status_in=None,
        submission_ids=None,
        run_ids=None,
        cache_keys=None,
        columns="*",
    ):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        clauses = []
        values: List[Any] = []
        for key, value in (filters or {}).items():
            if key == ROUND_MODE_FILTER:
                expression = "configuration_doc ->> 'mode'"
            elif key == PROMOTION_OUTCOME_FILTER:
                expression = "publication_doc #>> '{king_decision,outcome}'"
            elif key.replace("_", "").isalnum():
                expression = key
            else:
                raise ArenaStoreError("invalid filter column")
            if value is None:
                clauses.append("%s IS NULL" % expression)
            else:
                clauses.append("%s = %%s" % expression)
                values.append(value)
        if status_in is not None:
            if table != "lab_arena_rounds" or "status" in (filters or {}):
                raise ArenaStoreError("status inclusion filter is invalid")
            statuses = tuple(_check_filter_value(value) for value in status_in)
            if not statuses:
                raise ArenaStoreError("status inclusion filter is empty")
            clauses.append("status = ANY(%s)")
            values.append(list(statuses))
        if after_run_id is not None:
            if table != "lab_arena_runs" or order != "run_id" or "run_id" in (filters or {}):
                raise ArenaStoreError("run cursor requires ordered Arena runs")
            clauses.append("run_id > %s")
            values.append(_check_filter_value(after_run_id))
        if after_entry_id is not None:
            if (table != "lab_arena_ledger" or order != "entry_id"
                    or descending or "entry_id" in (filters or {})
                    or type(after_entry_id) is not int or after_entry_id < 0):
                raise ArenaStoreError("ledger cursor requires ordered Arena ledger")
            clauses.append("entry_id > %s")
            values.append(after_entry_id)
        if before_round is not None:
            if table != "lab_arena_rounds" or order != "history_round":
                raise ArenaStoreError("history cursor requires descending rounds")
            clauses.append("(created_at, round_id) < (%s, %s)")
            values.extend(before_round)
        if submission_ids is not None:
            if table not in ("lab_arena_runs", "lab_arena_trajectory_events") or "submission_id" in (filters or {}):
                raise ArenaStoreError("submission inclusion filter is invalid")
            ids = tuple(_check_filter_value(value) for value in submission_ids)
            if not ids:
                raise ArenaStoreError("submission inclusion filter is empty")
            clauses.append("submission_id = ANY(%s)")
            values.append(list(ids))
        if run_ids is not None:
            if table not in ("lab_arena_runs", "lab_arena_trajectory_events") or "run_id" in (filters or {}):
                raise ArenaStoreError("run inclusion filter is invalid")
            ids = tuple(str(value) for value in run_ids)
            if not ids or any(not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", value) for value in ids):
                raise ArenaStoreError("run inclusion ids are invalid")
            clauses.append("run_id = ANY(%s)")
            values.append(list(ids))
        if cache_keys is not None:
            if table != "lab_arena_judgment_cache" or "cache_key" in (filters or {}):
                raise ArenaStoreError("cache-key inclusion filter is invalid")
            keys = tuple(_check_filter_value(value) for value in cache_keys)
            if not keys:
                raise ArenaStoreError("cache-key inclusion filter is empty")
            clauses.append("cache_key = ANY(%s)")
            values.append(list(keys))
        projected_columns = ",".join(_RUNTIME_JSON_COLUMNS.get(column, column) for column in columns.split(","))
        sql = "SELECT row_to_json(t) FROM (SELECT %s FROM public.%s" % (projected_columns, table)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order == "history_round":
            if table != "lab_arena_rounds" or not descending:
                raise ArenaStoreError("history cursor requires descending rounds")
            sql += " ORDER BY created_at DESC, round_id DESC"
        elif order:
            if not order.replace("_", "").isalnum():
                raise ArenaStoreError("invalid order column")
            sql += " ORDER BY %s %s" % (order, "DESC" if descending else "ASC")
        if limit is not None:
            sql += " LIMIT %d" % int(limit)
        if offset is not None:
            offset = int(offset)
            if offset < 0:
                raise ArenaStoreError("offset must be nonnegative")
            sql += " OFFSET %d" % offset
        sql += ") t"
        connection = self._acquire()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, values)
                    return [row[0] for row in cursor.fetchall()]
            except Exception as exc:
                raise ArenaStoreError("select %s failed: %s" % (table, str(exc).splitlines()[0][:200])) from exc
        finally:
            self._release(connection)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connections = list(self._connections)
            self._connections = []
        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaStoreError("%s returned a non-object result" % context)
    return dict(value)


class ArenaStore:
    """Typed wrapper over the Arena functions and reads (section 15.1)."""

    def __init__(self, transport: StoreTransport, *, lease_ttl_seconds: int = LEASE_TTL_SECONDS) -> None:
        self._transport = transport
        self._lease_ttl_seconds = int(lease_ttl_seconds)

    # -- identity ---------------------------------------------------------

    def freeze_champion_funding(self, round_id: str) -> Dict[str, Any]:
        """Pin the last promoted miner independently of the baseline source."""
        return _require_mapping(self._transport.rpc(
            "lab_arena_freeze_champion_funding", {"p_round_id": round_id}
        ), "freeze_champion_funding")

    def provider_funding(self, run_id: str, provider: str) -> Dict[str, Any]:
        """Read the run's immutable payer, including historical billing calls."""
        return _require_mapping(self._transport.rpc(
            "lab_arena_provider_funding", {"p_run_id": run_id, "p_provider": provider}
        ), "provider_funding")

    def run_quota_snapshot(
        self,
        run_id: str,
        lease_token_hash: str,
        *,
        include_sourcing_cost: bool = False,
    ) -> Dict[str, Any]:
        """Read bounded counters for one active lease without renewing it."""

        result = _require_mapping(
            self._transport.rpc(
                (
                    "lab_arena_run_quota_snapshot_v2"
                    if include_sourcing_cost
                    else "lab_arena_run_quota_snapshot_v1"
                ),
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                },
            ),
            "run_quota_snapshot",
        )
        if include_sourcing_cost:
            from lab_arena import lab_arena_checkpoint

            try:
                return lab_arena_checkpoint.validate_quota_cost_snapshot(result)
            except lab_arena_checkpoint.QuotaUnavailable:
                raise ArenaStoreError(
                    "run quota snapshot schema mismatch"
                ) from None
        if set(result) != {"schema_version", "providers"} or result.get(
            "schema_version"
        ) != RUN_QUOTA_SNAPSHOT_SCHEMA_VERSION:
            raise ArenaStoreError("run quota snapshot schema mismatch")
        providers = result.get("providers")
        if not isinstance(providers, Mapping) or set(providers) != {
            "scrapingdog",
            "deepline",
            "openrouter",
        }:
            raise ArenaStoreError("run quota snapshot schema mismatch")
        for provider, counters in providers.items():
            if not isinstance(counters, Mapping) or set(counters) != {
                "limit",
                "used",
                "remaining",
                "inflight",
            }:
                raise ArenaStoreError("run quota snapshot schema mismatch")
            limit = counters.get("limit")
            used = counters.get("used")
            remaining = counters.get("remaining")
            inflight = counters.get("inflight")
            unlimited = provider == "deepline" and limit == 0
            if (
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in (limit, used, inflight))
                or (unlimited and remaining is not None)
                or (not unlimited and (
                    isinstance(remaining, bool) or not isinstance(remaining, int)
                    or limit < 1 or used > limit or remaining != limit - used
                ))
                or used < 0
                or inflight < 0
                or inflight > used
            ):
                raise ArenaStoreError("run quota snapshot schema mismatch")
        return result

    def append_validator_events(self, hotkey: str, network: str, netuid: int,
                                events: Sequence[Mapping[str, Any]],
                                gateway_source_commit: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc(
            "lab_arena_append_validator_events_v1", {
                "p_validator_hotkey": hotkey, "p_network": network,
                "p_netuid": netuid, "p_events": [dict(item) for item in events],
                "p_gateway_source_commit": gateway_source_commit,
            }), "append_validator_events")

    def append_trajectory_events(
        self,
        run_id: str,
        lease_token_hash: str,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Append one bounded batch under the run's active lease."""

        return _require_mapping(
            self._transport.rpc(
                "lab_arena_append_trajectory_events_v1",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_events": [dict(item) for item in events],
                },
            ),
            "append_trajectory_events",
        )

    def champion_provider_restart_required(self, run_id: str, provider: str) -> bool:
        return self.provider_funding(run_id, provider).get("restart_required") is True

    def mark_champion_provider_fallback(
        self, run_id: str, lease_token_hash: str, provider: str,
        evidence: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc(
            "lab_arena_mark_champion_provider_fallback", {
                "p_run_id": run_id, "p_lease_token_hash": lease_token_hash,
                "p_provider": provider, "p_evidence": dict(evidence),
            }
        ), "mark_champion_provider_fallback")

    def champion_funding_schema(self) -> Dict[str, Any]:
        result = _require_mapping(self._transport.rpc(
            "lab_arena_champion_funding_schema_v1", {}
        ), "champion_funding_schema")
        if result != {"version": 227, "provider_attempts": 4, "reward_factor_ppm": 500000}:
            raise ArenaStoreError("champion funding schema mismatch")
        return result

    def whoami(self) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_whoami", {}), "whoami")

    def require_service_role(self) -> Dict[str, Any]:
        """Refuse to run unless the role is ``lab_arena_service`` without superuser/BYPASSRLS."""

        identity = self.whoami()
        if identity.get("schema_version") != WHOAMI_SCHEMA_VERSION:
            raise ArenaRoleError("whoami schema mismatch")
        if identity.get("current_user") != SERVICE_ROLE_NAME:
            raise ArenaRoleError("database role is not %s" % SERVICE_ROLE_NAME)
        if identity.get("rolsuper") is not False or identity.get("rolbypassrls") is not False:
            raise ArenaRoleError("database role must not be superuser or BYPASSRLS")
        if identity.get("rolcanlogin") is not False:
            raise ArenaRoleError("database role must be NOLOGIN")
        return identity

    def current_daily_icp_set(self, set_id: int) -> Dict[str, Any]:
        """Read only the active UTC-day ICP set exposed to the Arena."""

        return _require_mapping(
            self._transport.rpc(
                "lab_arena_current_daily_icp_set", {"p_set_id": int(set_id)}
            ),
            "current_daily_icp_set",
        )

    def code_review_schema(self) -> Dict[str, Any]:
        """Require the independently deployable pre-scoring review capability."""

        result = _require_mapping(
            self._transport.rpc("lab_arena_code_review_schema_v1", {}),
            "code_review_schema",
        )
        if (
            result.get("schema_version") != CODE_REVIEW_SCHEMA_VERSION
            or result.get("version") != 207
            or result.get("claim_ttl_seconds") != 600
            or result.get("retry_backoff_seconds") != 60
            or result.get("max_attempts") != 3
        ):
            raise ArenaStoreError("code review schema mismatch")
        retry_policy = result.get("retry_policy")
        if (
            retry_policy != "bounded_transient_v1"
            or result.get("max_transient_attempts") != 6
            or result.get("transient_retry_backoff_seconds")
                != [60, 120, 240, 480, 900]
            or result.get("legacy_max_attempts") != 3
            or result.get("retry_window")
                != "replacement_freeze_to_benchmark_deadline"
        ):
            raise ArenaStoreError("code review schema mismatch")
        return result

    def validator_scoring_authority_schema(self) -> Dict[str, Any]:
        """Require gateway-authorized validator claims and completions."""

        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_validator_scoring_authority_schema_v1", {}
            ),
            "validator_scoring_authority_schema",
        )
        if (
            result.get("schema_version")
            != VALIDATOR_SCORING_AUTHORITY_SCHEMA_VERSION
            or result.get("version") != 208
            or result.get("authority") != "gateway_subnet_validator_role"
        ):
            raise ArenaStoreError("validator scoring authority schema mismatch")
        return result

    def company_quality_schema(self) -> Dict[str, Any]:
        """Require the per-company accepted-judgment capability."""

        result = _require_mapping(
            self._transport.rpc("lab_arena_company_quality_schema_v1", {}),
            "company_quality_schema",
        )
        if (
            result.get("schema_version") != COMPANY_QUALITY_SCHEMA_VERSION
            or result.get("version") != 1
        ):
            raise ArenaStoreError("company quality schema mismatch")
        return result

    def successful_call_cost_schema(self) -> Dict[str, Any]:
        """Require the successful-sourcing-call aggregate and SQL guard."""

        result = _require_mapping(
            self._transport.rpc("lab_arena_successful_call_cost_schema_v1", {}),
            "successful_call_cost_schema",
        )
        if (
            result.get("schema_version") != SUCCESSFUL_CALL_COST_SCHEMA_VERSION
            or result.get("version") != 230
            or result.get("policy") != "successful_calls_v1"
        ):
            raise ArenaStoreError("successful-call cost schema mismatch")
        return result

    def deepline_response_schema(self) -> Dict[str, Any]:
        value = _require_mapping(self._transport.rpc(
            "lab_arena_deepline_response_schema_v1", {}), "deepline_response_schema")
        if value != {"status": "ok", "schema_version":
                     "leadpoet.lab_arena.deepline_response_schema.v1", "version": 426}:
            raise ArenaStoreError("deepline response recovery schema is incomplete")
        return value

    def deepline_catalog_schema(self) -> Dict[str, Any]:
        """Require exact recovery and budget-only SQL before dynamic rollout."""
        result = _require_mapping(
            self._transport.rpc("lab_arena_deepline_catalog_schema_v1", {}),
            "deepline_catalog_schema",
        )
        if result != {
            "schema_version": "leadpoet.lab_arena.deepline_catalog_schema.v1",
            "version": 415,
        }:
            raise ArenaStoreError("Deepline catalog schema mismatch")
        return result

    def per_icp_cost_schema(self) -> Dict[str, Any]:
        result = _require_mapping(
            self._transport.rpc("lab_arena_per_icp_cost_schema_v1", {}),
            "per_icp_cost_schema",
        )
        if result != {
            "schema_version": PER_ICP_COST_SCHEMA_VERSION,
            "version": 289,
            "policy": "successful_calls_per_icp_v1",
        }:
            raise ArenaStoreError("per-ICP cost schema mismatch")
        return result

    def parallel_execution_schema(self, max_parallel_icps: int = 20) -> Dict[str, Any]:
        """Require the capability version for the configured runner ceiling."""

        expanded = max_parallel_icps > 20
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_parallel_execution_schema_v2" if expanded
                else "lab_arena_parallel_execution_schema_v1", {}
            ),
            "parallel_execution_schema",
        )
        if (
            set(result) != {"schema_version", "version", "max_parallel_icps"}
            or result["schema_version"] != (
                PARALLEL_EXECUTION_SCHEMA_V2_VERSION if expanded
                else PARALLEL_EXECUTION_SCHEMA_VERSION
            )
            or type(result["version"]) is not int
            or result["version"] != (401 if expanded else 255)
            or type(result["max_parallel_icps"]) is not int
            or result["max_parallel_icps"] != (
                RUNNER_SLOT_CEILING if expanded else 20
            )
        ):
            raise ArenaStoreError("parallel execution schema mismatch")
        return result

    def dynamic_benchmark_schema(self) -> Dict[str, Any]:
        """Require SQL guards for frozen benchmark counts and promotion margin."""

        result = _require_mapping(
            self._transport.rpc("lab_arena_dynamic_benchmark_schema_v1", {}),
            "dynamic_benchmark_schema",
        )
        if result != {
            "schema_version": DYNAMIC_BENCHMARK_SCHEMA_VERSION,
            "version": 353,
            "max_benchmark_icps": 100,
            "default_benchmark_icps": 10,
            "default_promotion_margin": 0.5,
        }:
            raise ArenaStoreError("dynamic benchmark schema mismatch")
        return result

    # -- accepted weight state ------------------------------------------

    def has_recent_participation(
        self, network: str, netuid: int, runner_hotkey: str
    ) -> bool:
        """Require an original accepted job in the database's last 24 hours."""

        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_has_recent_participation_v1",
                {
                    "p_network": str(network),
                    "p_netuid": int(netuid),
                    "p_runner_hotkey": str(runner_hotkey),
                },
            ),
            "has_recent_participation",
        )
        if set(result) != {"eligible"} or type(result["eligible"]) is not bool:
            raise ArenaStoreError("participation response is invalid")
        return result["eligible"]

    def get_weight_state(self, network: str, netuid: int, epoch: int) -> Optional[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_accepted_weight_states",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch)},
            limit=2,
            columns="network,netuid,epoch,state_hash,state_doc,created_at",
        )
        if len(rows) > 1:
            raise ArenaStoreError("multiple accepted weight states exist for one epoch")
        return rows[0] if rows else None

    def publish_weight_state(self, network: str, netuid: int, epoch: int, state_hash: str, state_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_publish_weight_state_v1",
                {"p_network": str(network), "p_netuid": int(netuid), "p_epoch": int(epoch), "p_state_hash": str(state_hash), "p_state_doc": dict(state_doc)},
            ),
            "publish_weight_state",
        )

    def record_chain_outcome(self, *, network: str, netuid: int, epoch: int, validator_hotkey: str, request_id: str, extrinsic_hash: str, outcome_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_record_chain_outcome_v1",
                {"p_network": str(network), "p_netuid": int(netuid), "p_epoch": int(epoch), "p_validator_hotkey": str(validator_hotkey), "p_request_id": str(request_id), "p_extrinsic_hash": str(extrinsic_hash), "p_outcome_doc": dict(outcome_doc)},
            ),
            "record_chain_outcome",
        )

    def list_chain_outcomes(self, network: str, netuid: int, epoch: int) -> List[Dict[str, Any]]:
        return self._transport.select(
            "lab_arena_chain_outcomes",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch)},
            order="created_at",
            descending=False,
            limit=200,
            columns="network,netuid,epoch,validator_hotkey,request_id,extrinsic_hash,outcome_doc,created_at",
        )

    def get_chain_outcome(self, network: str, netuid: int, epoch: int, validator_hotkey: str, request_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_chain_outcomes",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch), "validator_hotkey": str(validator_hotkey), "request_id": str(request_id)},
            limit=1,
            columns="network,netuid,epoch,validator_hotkey,request_id,extrinsic_hash,outcome_doc,created_at",
        )
        return rows[0] if rows else None

    # -- rounds -----------------------------------------------------------

    def create_round(self, round_id: str, configuration: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_create_round",
                {"p_round_id": round_id, "p_configuration_doc": dict(configuration)},
            ),
            "create_round",
        )

    def transition_round(self, round_id: str, expected_status: str, next_status: str, patch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_transition_round",
                {"p_round_id": round_id, "p_expected_status": expected_status, "p_next_status": next_status, "p_patch": dict(patch or {})},
            ),
            "transition_round",
        )

    def commit_round_v2(
        self,
        round_id: str,
        *,
        participants: Sequence[Mapping[str, Any]],
        benchmark_ref: str,
        evaluation_date: str,
        icp_set_date: str,
        scorer_image_digest: str,
        scorer_image_reference: str,
        deepline_catalog: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically commit an explicit Day 0 bank date for new-policy rounds."""

        return _require_mapping(
            self._transport.rpc(
                ("lab_arena_commit_round_v3" if deepline_catalog is not None
                 else "lab_arena_commit_round_v2"),
                {
                    "p_round_id": round_id,
                    "p_participants": [dict(item) for item in participants],
                    "p_benchmark_ref": benchmark_ref,
                    "p_evaluation_date": evaluation_date,
                    "p_icp_set_date": icp_set_date,
                    "p_scorer_image_digest": scorer_image_digest,
                    "p_scorer_image_reference": scorer_image_reference,
                    **({"p_deepline_catalog": dict(deepline_catalog)}
                       if deepline_catalog is not None else {}),
                },
            ),
            "commit_round_v2",
        )

    def reward_slot_snapshot(
        self, round_id: str, slot_policy: Mapping[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_reward_slot_snapshot",
                {"p_round_id": round_id, "p_slot_policy": dict(slot_policy)},
            ),
            "reward_slot_snapshot",
        )
        slots = result.get("reward_slots")
        if not isinstance(slots, list) or len(slots) != 3:
            raise ArenaStoreError("reward_slot_snapshot returned invalid reward_slots")
        return [
            None if slot is None else _require_mapping(slot, "reward_slot_snapshot")
            for slot in slots
        ]

    def activate_reward(self, round_id: str, reward_basis: Mapping[str, Any], signing_key_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_activate_reward",
                {"p_round_id": round_id, "p_reward_basis": dict(reward_basis), "p_signing_key_doc": dict(signing_key_doc)},
            ),
            "activate_reward",
        )

    def prepare_promotion(self, round_id: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_prepare_promotion",
                {"p_round_id": round_id, "p_plan": dict(plan)},
            ),
            "prepare_promotion",
        )

    def complete_promotion(self, round_id: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_complete_promotion",
                {"p_round_id": round_id, "p_plan": dict(plan)},
            ),
            "complete_promotion",
        )

    def pending_promotions(
        self, *, pinned_round_id: Optional[str] = None,
        network_name: Optional[str] = None, netuid: Optional[int] = None,
        limit: int = 100, offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        filters: Dict[str, Any] = {
            "status": "published",
            ROUND_MODE_FILTER: "live",
            "promotion_required": True,
            "baseline_promoted_at": None,
            PROMOTION_OUTCOME_FILTER: "crowned",
        }
        if pinned_round_id is not None:
            filters["round_id"] = pinned_round_id
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        rows = self._transport.select(
            "lab_arena_rounds",
            filters=filters,
            order="created_at",
            descending=False,
            limit=limit,
            offset=offset,
            columns="round_id,publication_doc,promotion_doc,published_at,created_at",
        )
        return rows

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_rounds", filters={"round_id": round_id}, limit=1)
        return rows[0] if rows else None

    def get_published_results_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Read only published result authorities and the configuration they use."""

        columns = (
            "round_id,status,participants,benchmark_ref,evaluation_date,icp_set_date,publication_doc,"
            + ",".join(
                "cfg_%s:configuration_doc->%s::text" % (key, key)
                for key in PUBLIC_RESULTS_CONFIGURATION_FIELDS
            )
        )
        rows = self._transport.select(
            "lab_arena_published_results_v1",
            filters={"round_id": round_id, "status": "published"},
            limit=1, columns=columns,
        )
        return rows[0] if rows else None

    def list_rounds(
        self,
        *,
        status: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
        network_name: Optional[str] = None,
        netuid: Optional[int] = None,
        limit: int = 100,
        offset: Optional[int] = None,
        columns: str = "*",
        evaluation_date: Optional[str] = None,
        before_round: Optional[tuple[str, str]] = None,
        history_order: bool = False,
    ) -> List[Dict[str, Any]]:
        if status is not None and statuses is not None:
            raise ArenaStoreError("round status filters are mutually exclusive")
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        filters: Dict[str, Any] = {}
        if evaluation_date is not None:
            filters["evaluation_date"] = evaluation_date
        if status is not None:
            filters["status"] = status
        if mode is not None:
            filters[ROUND_MODE_FILTER] = mode
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        return self._transport.select(
            "lab_arena_rounds",
            filters=filters or None,
            status_in=statuses,
            order="history_round" if history_order else "created_at",
            descending=True,
            limit=limit,
            offset=offset,
            columns=columns,
            **({"before_round": before_round} if before_round is not None else {}),
        )

    def latest_published_day(
        self, *, network_name: str, netuid: int
    ) -> Optional[Dict[str, Any]]:
        """Read the newest published live day in one chain scope."""

        rows = self._transport.select(
            "lab_arena_rounds",
            filters={
                "status": "published",
                ROUND_MODE_FILTER: "live",
                ROUND_NETWORK_COLUMN: str(network_name),
                ROUND_NETUID_COLUMN: int(netuid),
            },
            order="evaluation_date", descending=True, limit=1,
            columns="round_id,evaluation_date",
        )
        return rows[0] if rows else None

    def published_reward_bases(
        self, *, mode: Optional[str] = None,
        network_name: Optional[str] = None, netuid: Optional[int] = None,
        limit: int = 200, public_only: bool = False,
    ) -> List[Dict[str, Any]]:
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        # Activation is a separate signed commitment. A competition replay can
        # reopen its round while the activated basis still governs weights.
        filters: Dict[str, Any] = {}
        if mode is not None:
            filters[ROUND_MODE_FILTER] = mode
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        activated: List[Dict[str, Any]] = []
        offset = 0
        page_size = min(max(int(limit), 1), 200)
        while len(activated) < limit:
            rows = self._transport.select(
                "lab_arena_rounds",
                filters=filters or None,
                order="effective_reward_epoch",
                descending=True,
                limit=page_size,
                offset=offset,
                columns=(
                    "round_id,status,arena_network_name,arena_netuid,rewards_enabled,effective_reward_epoch,king_outcome,king_hotkey,king_start_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,published_at,"
                    + ("cfg_mode:configuration_doc->mode::text,cfg_baseline_hotkey:configuration_doc->baseline_hotkey::text"
                       if public_only else "configuration_doc")
                ),
            )
            activated.extend(
                row for row in rows
                if row.get("reward_activated_at")
                and row.get("reward_basis_doc")
                and row.get("signing_key_doc")
                and row.get("rewards_enabled") is True
            )
            if len(rows) < page_size:
                break
            offset += page_size
        return activated[:limit]

    # -- submissions ------------------------------------------------------

    def submission_replacement_schema(self) -> Dict[str, Any]:
        result = _require_mapping(
            self._transport.rpc("lab_arena_submission_replacement_schema_v1", {}),
            "submission_replacement_schema",
        )
        if (result.get("schema_version")
                != "leadpoet.lab_arena.submission_replacement_schema.v1"
                or result.get("version") != 258
                or result.get("replacement_freeze_seconds") != 3600
                or type(result.get("max_replacement_attempts")) is not int
                or result["max_replacement_attempts"] != 1):
            raise ArenaStoreError("submission replacement schema mismatch")
        return result

    def submission_similarity_schema(self) -> Dict[str, Any]:
        result = _require_mapping(
            self._transport.rpc("lab_arena_submission_duplicate_schema_v1", {}),
            "submission_similarity_schema",
        )
        if result != {
            "schema_version": "leadpoet.lab_arena.submission_duplicate.v1",
            "version": 405,
            "digest_format": "sha256:hex",
            "cross_hotkey_scope": "round",
        }:
            raise ArenaStoreError("submission similarity schema mismatch")
        return result

    def submission_similarity_champion(self, round_id: str) -> Dict[str, Any]:
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_submission_similarity_champion",
                {"p_round_id": round_id},
            ),
            "submission_similarity_champion",
        )
        if (result.get("status") not in ("ready", "promotion_pending")
                or result.get("submission_id") is not None
                and not isinstance(result.get("submission_id"), str)):
            raise ArenaStoreError("submission similarity champion invalid")
        return result

    def register_submission(
        self,
        round_id: str,
        submission_id: str,
        miner_hotkey: str,
        doc: Mapping[str, Any],
        *,
        owner_admission: Optional[OwnerAdmission] = None,
    ) -> Dict[str, Any]:
        function = "lab_arena_register_submission"
        params = {
            "p_round_id": round_id,
            "p_submission_id": submission_id,
            "p_miner_hotkey": miner_hotkey,
            "p_doc": dict(doc),
        }
        if owner_admission is not None:
            function = "lab_arena_register_submission_v2"
            params.update({
                "p_owner_coldkey": owner_admission.coldkey,
                "p_owner_block_number": owner_admission.block_number,
                "p_owner_block_hash": owner_admission.block_hash,
            })
        return _require_mapping(
            self._transport.rpc(function, params),
            "register_submission",
        )

    def update_submission(self, round_id: str, submission_id: str, expected_status: str, next_status: str, patch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_update_submission",
                {"p_round_id": round_id, "p_submission_id": submission_id, "p_expected_status": expected_status, "p_next_status": next_status, "p_patch": dict(patch or {})},
            ),
            "update_submission",
        )

    def retry_credit_failures(
        self, round_id: str, submission_id: str, miner_hotkey: str,
        request_hash: str,
    ) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc(
            "lab_arena_retry_credit_failures_v1",
            {"p_round_id": round_id, "p_submission_id": submission_id,
             "p_miner_hotkey": miner_hotkey, "p_request_hash": request_hash},
        ), "retry_credit_failures")

    def accept_submission_with_credentials(
        self,
        round_id: str,
        submission_id: str,
        miner_hotkey: str,
        encrypted_credentials: Mapping[str, str],
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_accept_submission_with_credentials",
                {
                    "p_round_id": round_id,
                    "p_submission_id": submission_id,
                    "p_miner_hotkey": miner_hotkey,
                    "p_credentials": dict(encrypted_credentials),
                },
            ),
            "accept_submission_with_credentials",
        )

    def accept_submission_source_with_credentials(
        self, round_id: str, submission_id: str, miner_hotkey: str,
        encrypted_credentials: Mapping[str, str], archive_sha256: str,
        normalized_sha256: str,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_accept_submission_source_with_credentials",
                {
                    "p_round_id": round_id,
                    "p_submission_id": submission_id,
                    "p_miner_hotkey": miner_hotkey,
                    "p_credentials": dict(encrypted_credentials),
                    "p_archive_sha256": archive_sha256,
                    "p_normalized_sha256": normalized_sha256,
                },
            ),
            "accept_submission_source_with_credentials",
        )

    def get_submission_credential(
        self, submission_id: str, miner_hotkey: str, provider: str
    ) -> Optional[Dict[str, Any]]:
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_get_submission_credential",
                {
                    "p_submission_id": submission_id,
                    "p_miner_hotkey": miner_hotkey,
                    "p_provider": provider,
                },
            ),
            "get_submission_credential",
        )
        return result if result.get("status") == "available" else None

    def begin_submission_review(
        self,
        submission_id: str,
        miner_hotkey: str,
        claim_token: str,
        reservation_microusd: int,
        review_model: str,
        file_count: int,
        source_bytes: int,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_begin_submission_review",
                {
                    "p_submission_id": str(submission_id),
                    "p_miner_hotkey": str(miner_hotkey),
                    "p_claim_token_hash": hash_lease_token(claim_token),
                    "p_reservation_microusd": int(reservation_microusd),
                    "p_review_model": str(review_model),
                    "p_file_count": int(file_count),
                    "p_source_bytes": int(source_bytes),
                },
            ),
            "begin_submission_review",
        )

    def finish_submission_review(
        self,
        submission_id: str,
        miner_hotkey: str,
        claim_token: str,
        status: str,
        review_doc: Mapping[str, Any],
        actual_microusd: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_finish_submission_review",
                {
                    "p_submission_id": str(submission_id),
                    "p_miner_hotkey": str(miner_hotkey),
                    "p_claim_token_hash": hash_lease_token(claim_token),
                    "p_status": str(status),
                    "p_review_doc": dict(review_doc),
                    "p_actual_microusd": (
                        None if actual_microusd is None else int(actual_microusd)
                    ),
                },
            ),
            "finish_submission_review",
        )

    def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_submissions", filters={"submission_id": submission_id}, limit=1)
        return rows[0] if rows else None

    def list_submissions(
        self,
        round_id: str,
        *,
        status: Optional[str] = None,
        columns: str = "*",
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if status:
            filters["status"] = status
        return self._transport.select(
            "lab_arena_submissions",
            filters=filters,
            order="created_at",
            columns=columns,
        )

    # -- stages and assignments -------------------------------------------

    def open_stage(self, round_id: str, stage: int, participants: Sequence[Mapping[str, Any]], icp_positions: Sequence[int]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_open_stage",
                {
                    "p_round_id": round_id,
                    "p_stage": int(stage),
                    "p_participants": [dict(item) for item in participants],
                    "p_icp_positions": [int(item) for item in icp_positions],
                },
            ),
            "open_stage",
        )

    def open_parallel_execution(
        self, round_id: str, participants: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_open_parallel_execution_v1",
                {
                    "p_round_id": round_id,
                    "p_participants": [dict(item) for item in participants],
                },
            ),
            "open_parallel_execution",
        )

    def close_parallel_execution(self, round_id: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_close_parallel_execution_v1",
                {"p_round_id": round_id},
            ),
            "close_parallel_execution",
        )

    def activate_preexecuted_stage2(self, round_id: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_activate_preexecuted_stage2_v1",
                {"p_round_id": round_id},
            ),
            "activate_preexecuted_stage2",
        )

    def claim_assignment(
        self,
        *,
        round_id: str,
        runner_hotkey: str,
        declared_parallelism: int,
        slot_ceiling: int,
        excluded_miner_hotkeys: Sequence[str],
        request_id: str,
        request_hash: str,
        lease_token_hash: str,
        lease_ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_claim_assignment",
                {
                    "p_round_id": round_id,
                    "p_runner_hotkey": runner_hotkey,
                    "p_declared_parallelism": int(declared_parallelism),
                    "p_slot_ceiling": int(slot_ceiling),
                    "p_excluded_miner_hotkeys": list(excluded_miner_hotkeys),
                    "p_request_id": request_id,
                    "p_request_hash": request_hash,
                    "p_lease_token_hash": lease_token_hash,
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "claim_assignment",
        )

    def recover_claim_response(
        self,
        *,
        round_id: str,
        runner_hotkey: str,
        request_id: str,
        request_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Read an exact prior lease response without allocating new work."""

        rows = self._transport.select(
            "lab_arena_runs",
            filters={
                "round_id": round_id,
                "runner_hotkey": runner_hotkey,
                "claim_request_id": request_id,
                "claim_request_hash": request_hash,
            },
            limit=1,
            columns="claim_response",
        )
        if not rows or not isinstance(rows[0], Mapping):
            return None
        response = rows[0].get("claim_response")
        if not isinstance(response, Mapping) or response.get("status") != "leased":
            return None
        return dict(response)

    def reserve_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, operation_id: str, provider: str, funding_source: str, amount_microusd: int, call_doc: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_reserve_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_operation_id": operation_id,
                    "p_provider": provider,
                    "p_funding_source": funding_source,
                    "p_amount_microusd": int(amount_microusd),
                    "p_call_doc": dict(call_doc),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "reserve_call",
        )

    def reserve_judgment_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, operation_id: str, provider: str, funding_source: str, amount_microusd: int, call_doc: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Atomically claim or replay one gateway-scoped verifier request."""

        return _require_mapping(
            self._transport.rpc(
                "lab_arena_reserve_judgment_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_operation_id": operation_id,
                    "p_provider": provider,
                    "p_funding_source": funding_source,
                    "p_amount_microusd": int(amount_microusd),
                    "p_call_doc": dict(call_doc),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "reserve_judgment_call",
        )

    def icp_cost_eligibility(
        self, *, round_id: str, submission_id: str, icp_position: int,
        qualified_company_count: int,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_icp_cost_eligibility",
                {
                    "p_round_id": round_id,
                    "p_submission_id": submission_id,
                    "p_icp_position": int(icp_position),
                    "p_qualified_company_count": int(qualified_company_count),
                },
            ),
            "icp_cost_eligibility",
        )

    def mark_dispatched(self, *, run_id: str, lease_token_hash: str, call_identity: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_mark_dispatched",
                {"p_run_id": run_id, "p_lease_token_hash": lease_token_hash, "p_call_identity": call_identity},
            ),
            "mark_dispatched",
        )

    def settle_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, actual_microusd: int, terminal_response: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_settle_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_actual_microusd": int(actual_microusd),
                    "p_terminal_response": dict(terminal_response),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "settle_call",
        )

    def recover_deepline_response(
        self, *, run_id: str, lease_token_hash: str, call_identity: str,
        request_hash: str, execution_key: str, credential_fingerprint: str,
        request_id: str, operation: str, actual_microusd: Optional[int],
        terminal_response: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc(
            "lab_arena_recover_deepline_response_v1", {
                "p_run_id": run_id, "p_lease_token_hash": lease_token_hash,
                "p_call_identity": call_identity, "p_request_hash": request_hash,
                "p_execution_key": execution_key,
                "p_credential_fingerprint": credential_fingerprint,
                "p_request_id": request_id, "p_operation": operation,
                "p_actual_microusd": actual_microusd,
                "p_terminal_response": dict(terminal_response),
                "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
            }), "recover_deepline_response")

    def mark_uncertain(self, *, run_id: str, lease_token_hash: str, call_identity: str, call_doc: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_mark_uncertain",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_call_doc": dict(call_doc),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "mark_uncertain",
        )

    def list_openrouter_cost_reconciliations(
        self,
        round_id: str,
        *,
        run_id: str = "",
        after_entry_id: int = 0,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_list_openrouter_cost_reconciliations_v1",
                {
                    "p_round_id": str(round_id),
                    "p_run_id": str(run_id),
                    "p_after_entry_id": int(after_entry_id),
                    "p_limit": int(limit),
                },
            ),
            "list_openrouter_cost_reconciliations",
        )
        if result.get("status") != "ok" or not isinstance(result.get("items"), list):
            raise ArenaStoreError("openrouter cost reconciliation list is malformed")
        return [
            _require_mapping(item, "openrouter cost reconciliation item")
            for item in result["items"]
        ]

    def reconcile_openrouter_cost(
        self,
        *,
        round_id: str,
        run_id: str,
        call_identity: str,
        uncertain_entry_id: int,
        generation_id: str,
        credential_fingerprint: str,
        actual_microusd: int,
        cost_units: str,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_reconcile_openrouter_cost_v1",
                {
                    "p_round_id": str(round_id),
                    "p_run_id": str(run_id),
                    "p_call_identity": str(call_identity),
                    "p_uncertain_entry_id": int(uncertain_entry_id),
                    "p_generation_id": str(generation_id),
                    "p_credential_fingerprint": str(credential_fingerprint),
                    "p_actual_microusd": int(actual_microusd),
                    "p_cost_units": str(cost_units),
                },
            ),
            "reconcile_openrouter_cost",
        )

    def next_closed_deepline_reconciliation(
        self, *, mode: str, network_name: str, netuid: int,
        round_id: str = "", after_entry_id: int = 0,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_next_closed_deepline_reconciliation_v1",
                {
                    "p_mode": mode, "p_network_name": network_name,
                    "p_netuid": int(netuid), "p_round_id": round_id,
                    "p_after_entry_id": int(after_entry_id),
                },
            ),
            "next_closed_deepline_reconciliation",
        )

    def next_closed_provider_reconciliation(
        self, *, mode: str, network_name: str, netuid: int,
        round_id: str = "", after_entry_id: int = 0,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_next_closed_provider_reconciliation_v1",
                {"p_mode": str(mode), "p_network_name": str(network_name),
                 "p_netuid": int(netuid), "p_round_id": str(round_id),
                 "p_after_entry_id": int(after_entry_id)},
            ),
            "next_closed_provider_reconciliation",
        )

    def list_deepline_cost_reconciliations(
        self,
        round_id: str,
        *,
        run_id: str = "",
        after_entry_id: int = 0,
        limit: int = 1,
        successful_execute_only: bool = False,
        score_only: bool = False,
    ) -> List[Dict[str, Any]]:
        if successful_execute_only and score_only:
            raise ValueError("Deepline candidate lanes are mutually exclusive")
        params = {
            "p_round_id": str(round_id),
            "p_run_id": str(run_id),
            "p_after_entry_id": int(after_entry_id),
            "p_limit": int(limit),
        }
        rpc = "lab_arena_list_deepline_cost_reconciliations_v1"
        if successful_execute_only:
            rpc = "lab_arena_list_deepline_cost_reconciliations_v2"
            params["p_successful_execute_only"] = True
        elif score_only:
            rpc = "lab_arena_list_deepline_cost_reconciliations_v3"
        result = _require_mapping(
            self._transport.rpc(rpc, params),
            "list_deepline_cost_reconciliations",
        )
        if result.get("status") != "ok" or not isinstance(result.get("items"), list):
            raise ArenaStoreError("deepline cost reconciliation list is malformed")
        return [
            _require_mapping(item, "deepline cost reconciliation item")
            for item in result["items"]
        ]

    def reconcile_deepline_cost(
        self,
        *,
        round_id: str,
        run_id: str,
        call_identity: str,
        uncertain_entry_id: int,
        request_id: str,
        operation: str,
        credential_fingerprint: str,
        actual_microusd: int,
        cost_units: str,
        execution_key: Optional[str] = None,
        recovered_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if execution_key is not None and recovered_request_id is None:
            raise ArenaStoreError("Deepline execution recovery requires key and request ID")
        return _require_mapping(
            self._transport.rpc(
                ("lab_arena_reconcile_deepline_cost_v2" if recovered_request_id is not None
                 else "lab_arena_reconcile_deepline_cost_v1"),
                {
                    "p_round_id": str(round_id),
                    "p_run_id": str(run_id),
                    "p_call_identity": str(call_identity),
                    "p_uncertain_entry_id": int(uncertain_entry_id),
                    "p_request_id": str(request_id),
                    "p_operation": str(operation),
                    "p_credential_fingerprint": str(credential_fingerprint),
                    "p_actual_microusd": int(actual_microusd),
                    "p_cost_units": str(cost_units),
                    **({"p_execution_key": execution_key,
                        "p_recovered_request_id": str(recovered_request_id)}
                       if recovered_request_id is not None else {}),
                },
            ),
            "reconcile_deepline_cost",
        )

    def complete_attempt(
        self,
        *,
        run_id: str,
        lease_token_hash: str,
        result: Mapping[str, Any],
        terminal_cause: str,
        output_ref: str,
        judgment_evidence: Optional[Mapping[str, Any]] = None,
        judgment_evidence_hash: str = "",
        company_judgment_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
        output_hash: str = "",
        completion_request_hash: str = "",
    ) -> Dict[str, Any]:
        function = "lab_arena_complete_attempt"
        params: Dict[str, Any] = {
            "p_run_id": run_id,
            "p_lease_token_hash": lease_token_hash,
            "p_result": dict(result),
            "p_terminal_cause": terminal_cause,
            "p_output_ref": output_ref,
        }
        if judgment_evidence is not None and company_judgment_evidence is not None:
            raise ArenaStoreError("whole-item and company judgment evidence conflict")
        if company_judgment_evidence is not None:
            function = "lab_arena_complete_attempt_v3"
            params.update({
                "p_output_hash": str(output_hash),
                "p_company_judgment_evidence": [
                    dict(item) for item in company_judgment_evidence
                ],
                "p_completion_request_hash": str(completion_request_hash),
            })
        elif judgment_evidence is not None:
            function = "lab_arena_complete_attempt_v2"
            params.update({
                "p_judgment_evidence": dict(judgment_evidence),
                "p_judgment_evidence_hash": str(judgment_evidence_hash),
            })
        return _require_mapping(
            self._transport.rpc(function, params),
            "complete_attempt",
        )

    def expire_leases(self, round_id: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_expire_leases", {"p_round_id": round_id}), "expire_leases")

    def operator_hold_active(self) -> bool:
        result = self._transport.rpc("lab_arena_operator_hold_active_v1", {})
        if not isinstance(result, bool):
            raise ArenaStoreError("operator hold state is malformed")
        return result

    def close_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_close_stage", {"p_round_id": round_id, "p_stage": int(stage)}), "close_stage")

    def open_scoring(
        self,
        round_id: str,
        stage: int,
        work_items: Sequence[Mapping[str, Any]],
        *,
        integrity_cache: bool = False,
        company_quality_cache: bool = False,
    ) -> Dict[str, Any]:
        """Turn the committed scoring plan into claimable scoring assignments (one per work item)."""

        if integrity_cache and company_quality_cache:
            raise ArenaStoreError("scoring cache modes conflict")
        function = (
            "lab_arena_open_scoring_v3"
            if company_quality_cache
            else "lab_arena_open_scoring_v2"
            if integrity_cache
            else "lab_arena_open_scoring"
        )

        return _require_mapping(
            self._transport.rpc(function, {"p_round_id": round_id, "p_stage": int(stage), "p_work_items": [dict(item) for item in work_items]}),
            "open_scoring",
        )

    def get_judgment_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_judgment_cache",
            filters={"cache_key": str(cache_key)},
            limit=2,
            columns="cache_key,scope_doc,scoring_input_hash,evidence_hash,evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey,created_at",
        )
        if len(rows) > 1:
            raise ArenaStoreError("multiple accepted judgments exist for one cache key")
        return rows[0] if rows else None

    def get_judgment_caches(self, cache_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch only requested immutable cache authorities in bounded batches."""

        found: Dict[str, Dict[str, Any]] = {}
        keys = sorted(set(str(key) for key in cache_keys))
        for first in range(0, len(keys), 25):
            batch = keys[first:first + 25]
            rows = self._transport.select(
                "lab_arena_judgment_cache", cache_keys=batch, limit=50,
                columns=("cache_key,scope_doc,scoring_input_hash,evidence_hash,"
                         "evidence_doc,source_score_run_id,source_scored_run_id,"
                         "source_runner_hotkey,created_at"),
            )
            for row in rows:
                key = str(row.get("cache_key") or "")
                if key not in batch or key in found:
                    raise ArenaStoreError("accepted judgment cache bulk read is invalid")
                found[key] = row
        return found

    def get_company_judgments(self, cache_key: str) -> List[Dict[str, Any]]:
        """Read immutable authority variants for one gateway-built company key."""

        return self._transport.select(
            "lab_arena_company_judgments",
            filters={"cache_key": str(cache_key)},
            order="authority_slot",
            limit=100,
            columns=(
                "cache_key,authority_slot,scope_doc,company_input_hash,"
                "evidence_hash,evidence_doc,source_score_run_id,"
                "source_scored_run_id,source_runner_hotkey,created_at"
            ),
        )

    def get_company_judgment(
        self, cache_key: str, authority_slot: int
    ) -> Optional[Dict[str, Any]]:
        """Read one exact immutable company judgment authority variant."""

        rows = self._transport.select(
            "lab_arena_company_judgments",
            filters={
                "cache_key": str(cache_key),
                "authority_slot": int(authority_slot),
            },
            limit=2,
            columns=(
                "cache_key,authority_slot,scope_doc,company_input_hash,"
                "evidence_hash,evidence_doc,source_score_run_id,"
                "source_scored_run_id,source_runner_hotkey,created_at"
            ),
        )
        if len(rows) > 1:
            raise ArenaStoreError(
                "multiple company judgments exist for one authority slot"
            )
        return rows[0] if rows else None

    def close_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_close_scoring", {"p_round_id": round_id, "p_stage": int(stage)}), "close_scoring")

    def cancel_round(self, round_id: str, reason: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_cancel_round", {"p_round_id": round_id, "p_reason": reason}), "cancel_round")

    def record_run_scores(self, round_id: str, stage: int, scores: Sequence[Mapping[str, Any]], *, batch_size: int = SCORE_BATCH_SIZE) -> Dict[str, Any]:
        """Record per-run scores in bounded batches.

        The SQL function is idempotent per run (an equal score counts as
        existing, a different one is refused), so a stage of thousands of
        runs is written in batches that stay within request-size limits and a
        retry after a partial write completes the remainder.
        """

        items = [dict(item) for item in scores]
        if batch_size < 1:
            raise ArenaStoreError("score batch size must be positive")
        totals = {"status": "ok", "recorded": 0, "existing": 0, "batches": 0}
        for start in range(0, len(items), batch_size) or (0,):
            result = _require_mapping(
                self._transport.rpc(
                    "lab_arena_record_run_scores",
                    {"p_round_id": round_id, "p_stage": int(stage), "p_scores": items[start:start + batch_size]},
                ),
                "record_run_scores",
            )
            if result.get("status") != "ok":
                return dict(result)
            totals["recorded"] += int(result.get("recorded") or 0)
            totals["existing"] += int(result.get("existing") or 0)
            totals["batches"] += 1
        return totals

    # -- reads ------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_runs", filters={"run_id": run_id}, limit=1)
        return rows[0] if rows else None

    def get_runs(self, run_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch only referenced runs, without a round-wide scan."""

        found: Dict[str, Dict[str, Any]] = {}
        ids = sorted(set(str(run_id) for run_id in run_ids))
        for first in range(0, len(ids), 25):
            batch = ids[first:first + 25]
            rows = self._transport.select("lab_arena_runs", run_ids=batch, limit=50)
            for row in rows:
                run_id = str(row.get("run_id") or "")
                if run_id not in batch or run_id in found:
                    raise ArenaStoreError("referenced run bulk read is invalid")
                found[run_id] = row
        return found

    def list_trajectory_events(self, run_id: str) -> List[Dict[str, Any]]:
        """Read one private run in bounded pages, including long trajectories."""

        if not run_id:
            raise ArenaStoreError("trajectory run id is required")
        rows: List[Dict[str, Any]] = []
        from lab_arena.trajectory import MAX_EVENTS_PER_RUN

        for offset in range(0, MAX_EVENTS_PER_RUN, 500):
            page = self._transport.select(
                "lab_arena_trajectory_events", filters={"run_id": run_id},
                order="trajectory_id", limit=500, offset=offset,
            )
            rows.extend(page)
            if len(page) < 500:
                break
        return rows

    def list_abandoned_billing_runs(self, round_id: str) -> List[str]:
        """Prioritize retained bills, without granting lease recovery authority.

        Only leased runs are read, then their indexed runtime errors. Billing
        and expiry keep their existing identity and all-settled checks.
        """
        leased = self.list_runs(round_id, status="leased", columns="run_id")
        ids = sorted({str(row["run_id"]) for row in leased})
        abandoned = set()
        for first in range(0, len(ids), 25):
            rows = self._transport.select(
                "lab_arena_trajectory_events",
                filters={"round_id": round_id, "event_kind": "runtime.error"},
                run_ids=ids[first:first + 25], order="trajectory_id",
                descending=True, limit=500,
                columns=("run_id,status:content->>status,"
                         "error_class:content->>error_class,"
                         "failure_stage:content->>failure_stage"),
            )
            abandoned.update(
                str(row["run_id"]) for row in rows
                if row.get("status") == "abandoned"
                and row.get("error_class") == "RuntimeHostError"
                and row.get("failure_stage") == "runtime"
            )
        return sorted(abandoned.intersection(ids))

    def list_runtime_starts(self, round_id: str, *, run_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Read indexed start receipts only for runs missing source metadata.

        Small batches keep PostgREST URLs bounded, including maximum-length IDs.
        The run index avoids scanning a round's many provider trajectory events.
        """
        rows: List[Dict[str, Any]] = []
        ids = sorted(set(run_ids))
        for first in range(0, len(ids), 25):
            offset = 0
            while True:
                page = self._transport.select(
                    "lab_arena_trajectory_events",
                    filters={"round_id": round_id, "event_kind": "runtime.started"},
                    run_ids=ids[first:first + 25], order="trajectory_id", limit=500, offset=offset,
                    columns=("trajectory_id,run_id,submission_id,runner_hotkey,assignment_id,attempt,"
                             "source_commit:content->>validator_source_commit,"
                             "source_dirty:content->>validator_source_dirty,"
                             "start_lease_generation:content->>lease_generation"),
                )
                rows.extend(page)
                if len(page) < 500:
                    break
                offset += 500
        return rows

    def list_runs(self, round_id: str, *, stage: Optional[int] = None, status: Optional[str] = None, submission_id: Optional[str] = None, kind: Optional[str] = None, columns: str = "*", submission_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if stage is not None:
            filters["stage"] = int(stage)
        if status:
            filters["status"] = status
        if submission_id:
            filters["submission_id"] = submission_id
        if kind:
            filters["kind"] = kind
        # A full miner set creates more rows per stage than PostgREST's usual
        # 1,000-row response cap. Keep every caller's run_id ordering intact.
        rows: List[Dict[str, Any]] = []
        page_size = 500
        after_run_id = None
        while True:
            page = self._transport.select(
                "lab_arena_runs", filters=filters, order="run_id",
                limit=page_size, after_run_id=after_run_id,
                **({"columns": columns} if columns != "*" else {}),
                **({"submission_ids": submission_ids} if submission_ids is not None else {}),
            )
            rows.extend(page)
            if len(page) < page_size:
                return rows
            after_run_id = str(page[-1]["run_id"])

    def list_public_result_execution_runs(
        self, round_id: str, baseline_id: str, submission_id: str
    ) -> List[Dict[str, Any]]:
        """Read baseline and requested executions in one fresh ordered query."""

        return self.list_runs(
            round_id, kind="execute",
            submission_ids=tuple(dict.fromkeys((baseline_id, submission_id))),
        )

    def list_ledger(
        self,
        *,
        run_id: Optional[str] = None,
        call_identity: Optional[str] = None,
        miner_hotkey: Optional[str] = None,
        submission_id: Optional[str] = None,
        provider: Optional[str] = None,
        entry_kind: Optional[str] = None,
        operation_id: Optional[str] = None,
        after_entry_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {}
        if run_id:
            filters["run_id"] = run_id
        if call_identity:
            filters["call_identity"] = call_identity
        if miner_hotkey:
            filters["miner_hotkey"] = miner_hotkey
        if submission_id:
            filters["submission_id"] = submission_id
        if provider:
            filters["provider"] = provider
        if operation_id:
            filters["operation_id"] = operation_id
        if entry_kind:
            filters["entry_kind"] = entry_kind
        if limit is not None and (
            type(limit) is not int or not 1 <= limit <= 1_000
        ):
            raise ArenaStoreError("ledger limit must be from 1 through 1000")
        arguments: Dict[str, Any] = {
            "filters": filters or None,
            "order": "entry_id",
        }
        if limit is not None:
            arguments["limit"] = limit
        if after_entry_id is not None:
            if type(after_entry_id) is not int or after_entry_id < 0:
                raise ArenaStoreError("ledger cursor must be nonnegative")
            arguments["after_entry_id"] = after_entry_id
        return self._transport.select("lab_arena_ledger", **arguments)

    def submission_costs(self, submission_id: str) -> Dict[str, Any]:
        """Return strictly validated aggregate costs across every retry run."""

        try:
            result = validate_submission_costs(
                self._transport.rpc(
                    "lab_arena_submission_costs",
                    {"p_submission_id": str(submission_id)},
                )
            )
        except ArenaContractError as exc:
            raise ArenaStoreError("submission_costs returned an invalid result") from exc
        if result["submission_id"] != str(submission_id):
            raise ArenaStoreError("submission_costs returned the wrong submission")
        return result

    def close(self) -> None:
        self._transport.close()
