"""Bounded, sanitized retry policy for submission code review failures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, Mapping, Optional


LEGACY_MAX_ATTEMPTS = 3
TRANSIENT_BACKOFF_SECONDS = (60, 120, 240, 480, 900)
TRANSIENT_ERROR_CODES = frozenset({
    "code_review_preparation_unavailable", "code_review_provider_rate_limited",
    "code_review_provider_timeout", "code_review_provider_unavailable",
    "code_review_transport_error", "review_response_invalid",
    "code_review_research_unverified",
})
OPENROUTER_LIMIT_SOURCES = frozenset({
    "openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget",
})

SOURCE_ERROR_CODES = frozenset({
    "code_review_source_size_mismatch",
    "code_review_source_digest_mismatch",
    "review_context_limit_invalid",
    "review_model_invalid",
    "review_output_cannot_report_coverage",
    "review_output_limit_invalid",
    "review_source_empty",
    "review_source_exceeds_context",
    "review_source_incomplete",
    "review_source_invalid",
    "review_source_not_text",
    "review_source_not_utf8",
    "review_source_unreadable",
    "source_archive_invalid",
    "source_archive_failed",
    "source_archive_too_large",
    "source_archive_unsafe",
    "source_contains_credentials",
    "source_directory_missing",
    "source_empty",
    "source_entry_type_invalid",
    "source_file_count_exceeded",
    "source_git_automation_forbidden",
    "source_license_invalid",
    "source_path_invalid",
    "source_unpacked_too_large",
    "source_unreadable",
    "source_credentials_invalid",
    "harness_file_missing",
    "harness_invalid",
    "harness_too_large",
})

RESPONSE_ERROR_REASONS = frozenset({
    "envelope", "model_mismatch", "choice", "not_finished", "message",
    "refusal_or_tools", "content", "content_json", "document_keys",
    "verdict", "summary", "coverage", "coverage_order", "findings",
    "verdict_findings", "finding_keys", "finding_classification",
    "finding_evidence", "finding_evidence_mismatch", "finding_explanation",
    "context_integrity",
    "research_keys", "research_binding", "research_classification",
    "research_evidence", "research_evidence_mismatch", "research_limitations",
})

ERROR_CODES = frozenset({
    *SOURCE_ERROR_CODES,
    "code_review_preparation_unavailable",
    "code_review_provider_authentication",
    "code_review_provider_credit",
    "code_review_provider_rate_limited",
    "code_review_provider_request_rejected",
    "code_review_provider_timeout",
    "code_review_provider_unavailable",
    "code_review_transport_error",
    "code_review_credential_echo",
    "code_review_cost_unavailable",
    "code_review_deadline_exceeded",
    "code_review_round_closed",
    "code_review_submission_ineligible",
    "review_response_invalid",
    "code_review_research_unverified",
})

# Private diagnostics are observations, never admission, billing or retry input.
REVIEW_FINISH_REASONS = frozenset({
    "stop", "length", "tool_calls", "function_call", "content_filter", "error",
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal",
    "cancelled", "completed", "missing", "invalid", "other",
})
REVIEW_CONTEXT_REASONS = frozenset({
    "not_required", "verified", "metadata_missing", "requested_model", "strategy",
    "attempt", "pipeline", "endpoints_missing", "endpoints_available",
    "endpoints_total", "endpoint_shape", "selected_count", "selected_model",
    "usage_missing", "prompt_tokens", "context_budget",
})
REVIEW_PROVIDER_NAMES = frozenset({"Anthropic", "Amazon Bedrock", "Google", "Google Vertex"})
_GENERATION_ID = re.compile(r"gen-[0-9]{1,20}-[A-Za-z0-9_-]{1,128}\Z")


def private_response_diagnostic(document: Any, prepared: Any, *, generation_id: Any = None) -> dict:
    """Copy only bounded metadata from a credential-checked provider reply.

    Unknown provider text and identifiers are omitted. No source, message,
    reasoning, error prose, headers or arbitrary routing metadata is retained.
    """
    from lab_arena import code_review

    response = document if isinstance(document, Mapping) else {}
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None

    def finish(name):
        if not isinstance(choice, Mapping):
            return "invalid"
        value = choice.get(name)
        if value is None:
            return "missing"
        if not isinstance(value, str):
            return "invalid"
        return value if len(value) <= 32 and value in REVIEW_FINISH_REASONS else "other"

    context = "not_required"
    if prepared.requires_context_integrity:
        context = code_review.context_integrity_failure(response, prepared) or "verified"
    result = {
        "finish_reason": finish("finish_reason"),
        "native_finish_reason": finish("native_finish_reason"),
        "max_output_tokens": prepared.parameters["max_tokens"],
        "context_integrity": context,
    }
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        for name in ("prompt_tokens", "completion_tokens"):
            value = usage.get(name)
            if type(value) is int and 0 <= value <= 1_000_000_000:
                result[name] = value
        details = usage.get("completion_tokens_details")
        value = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
        if type(value) is int and 0 <= value <= 1_000_000_000:
            result["reasoning_tokens"] = value
    models = {prepared.parameters["model"]}
    canonical = code_review._REVIEW_MODEL_CANONICAL_SLUGS.get(prepared.parameters["model"])
    if canonical:
        models.add(canonical)
    model = response.get("model")
    if isinstance(model, str) and model in models:
        result["response_model"] = model
    metadata = response.get("openrouter_metadata")
    endpoints = metadata.get("endpoints") if isinstance(metadata, Mapping) else None
    available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
    if isinstance(available, list) and len(available) <= 128:
        selected = [row for row in available if isinstance(row, Mapping) and row.get("selected") is True]
        if len(selected) == 1:
            model, provider = selected[0].get("model"), selected[0].get("provider")
            if isinstance(model, str) and model in models:
                result["selected_model"] = model
            if isinstance(provider, str) and provider in REVIEW_PROVIDER_NAMES:
                result["selected_provider"] = provider
    if isinstance(generation_id, str) and len(generation_id) <= 153 and _GENERATION_ID.fullmatch(generation_id):
        result["generation_id"] = generation_id
    return result


def diagnostic(
    error_code: str,
    *,
    provider_http_status: Optional[int] = None,
    error_reason: Optional[str] = None,
    retryable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the only diagnostic fields allowed to cross the persistence edge."""

    if error_code not in ERROR_CODES:
        error_code = "code_review_provider_unavailable"
        retryable = None
    result: Dict[str, Any] = {"error_code": error_code}
    if (
        isinstance(provider_http_status, int)
        and not isinstance(provider_http_status, bool)
        and 100 <= provider_http_status <= 599
    ):
        result["provider_http_status"] = provider_http_status
    if error_reason in RESPONSE_ERROR_REASONS:
        result["error_reason"] = error_reason
    if isinstance(retryable, bool):
        result["retryable"] = retryable
    return result


def source_diagnostic(error_code: str) -> Dict[str, Any]:
    """Source rejection is permanent and never consumes another paid attempt."""

    if error_code not in SOURCE_ERROR_CODES:
        error_code = "code_review_preparation_unavailable"
    return diagnostic(error_code, retryable=False)


def provider_response_observation(
    document: Any, *, raw_http_status: int, effective_http_status: Optional[int] = None,
) -> Dict[str, Any]:
    """Retain fixed provider observations without granting retry or cost authority."""

    result: Dict[str, Any] = {"provider_limit_source": "unknown"}
    for name, value in (
        ("provider_raw_http_status", raw_http_status),
        ("provider_effective_http_status", effective_http_status),
    ):
        if type(value) is int and 100 <= value <= 599:
            result[name] = value
    error = document.get("error") if isinstance(document, Mapping) else None
    if not isinstance(error, Mapping) or type(error.get("code")) is not int:
        return result
    metadata = error.get("metadata")
    source = metadata.get("limit_source") if isinstance(metadata, Mapping) else None
    if (error["code"] in (402, 429)
        and error["code"] in (raw_http_status, effective_http_status)
        and isinstance(source, str) and source in OPENROUTER_LIMIT_SOURCES):
        result["provider_limit_source"] = source
    return result


def provider_http_diagnostic(status: int) -> Dict[str, Any]:
    """Classify only explicit HTTP outcomes; provider prose is never retained."""

    if status in (401, 403):
        return diagnostic(
            "code_review_provider_authentication",
            provider_http_status=status,
            retryable=False,
        )
    if status == 402:
        return diagnostic(
            "code_review_provider_credit",
            provider_http_status=status,
            retryable=False,
        )
    if status == 408:
        return diagnostic(
            "code_review_provider_timeout",
            provider_http_status=status,
            retryable=True,
        )
    if status == 429:
        return diagnostic(
            "code_review_provider_rate_limited",
            provider_http_status=status,
            retryable=True,
        )
    if status == 404:
        # OpenRouter can use 404 when no provider satisfies the route policy.
        return diagnostic(
            "code_review_provider_unavailable",
            provider_http_status=status,
            retryable=True,
        )
    if 500 <= status <= 599:
        return diagnostic(
            "code_review_provider_unavailable",
            provider_http_status=status,
            retryable=True,
        )
    if 400 <= status <= 499:
        return diagnostic(
            "code_review_provider_request_rejected",
            provider_http_status=status,
            retryable=False,
        )
    return diagnostic(
        "code_review_provider_request_rejected",
        provider_http_status=status,
        retryable=False,
    )


def transient_failure(document: Any) -> bool:
    """Use classified failure codes, never provider prose or stale retry flags."""

    code = document.get("error_code") if isinstance(document, Mapping) else None
    return isinstance(code, str) and code in TRANSIENT_ERROR_CODES


def review_attempt_limit(row: Mapping[str, Any]) -> Optional[int]:
    """Known reviewer failures use the admission deadline, never an attempt cap.

    Reclassify old diagnostics at read time: a cached retry flag is not a code
    verdict. Unknown legacy errors retain their historical fail-closed limit.
    """

    attempts = max(0, int(row.get("code_review_attempts") or 0))
    document = row.get("code_review_doc")
    if transient_failure(document):
        return None
    retryable = document.get("retryable") if isinstance(document, Mapping) else None
    if retryable is False:
        return attempts
    return LEGACY_MAX_ATTEMPTS


def review_retry_available(row: Mapping[str, Any]) -> bool:
    """Return whether an error row can consume another attempt."""

    attempts = max(0, int(row.get("code_review_attempts") or 0))
    limit = review_attempt_limit(row)
    return row.get("code_review_status") == "error" and (limit is None or attempts < limit)


def retry_ready_at(row: Mapping[str, Any]) -> Optional[datetime]:
    """Return the deterministic local backoff time for an error row."""

    if not review_retry_available(row):
        return None
    raw = row.get("code_review_started_at")
    if raw is None:
        return None
    started = (
        raw
        if isinstance(raw, datetime)
        else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    )
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    attempt = max(1, int(row.get("code_review_attempts") or 0))
    retryable = review_attempt_limit(row) is None
    delay = (
        TRANSIENT_BACKOFF_SECONDS[min(attempt - 1, len(TRANSIENT_BACKOFF_SECONDS) - 1)]
        if retryable else TRANSIENT_BACKOFF_SECONDS[0]
    )
    return started.astimezone(timezone.utc) + timedelta(seconds=delay)
