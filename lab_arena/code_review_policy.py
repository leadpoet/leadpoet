"""Bounded, sanitized retry policy for submission code review failures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional


LEGACY_MAX_ATTEMPTS = 3
MAX_TRANSIENT_ATTEMPTS = 6
TRANSIENT_BACKOFF_SECONDS = (60, 120, 240, 480, 900)

SOURCE_ERROR_CODES = frozenset({
    "code_review_source_size_mismatch",
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
})


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


def review_attempt_limit(row: Mapping[str, Any]) -> int:
    """Return the row's effective cap, including a consumed permanent failure."""

    attempts = max(0, int(row.get("code_review_attempts") or 0))
    document = row.get("code_review_doc")
    retryable = document.get("retryable") if isinstance(document, Mapping) else None
    if retryable is True:
        return MAX_TRANSIENT_ATTEMPTS
    if retryable is False:
        return attempts
    return LEGACY_MAX_ATTEMPTS


def review_retry_available(row: Mapping[str, Any]) -> bool:
    """Return whether an error row can consume another attempt."""

    attempts = max(0, int(row.get("code_review_attempts") or 0))
    return row.get("code_review_status") == "error" and attempts < review_attempt_limit(row)


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
    retryable = isinstance(row.get("code_review_doc"), Mapping) and (
        row["code_review_doc"].get("retryable") is True
    )
    delay = (
        TRANSIENT_BACKOFF_SECONDS[min(attempt - 1, len(TRANSIENT_BACKOFF_SECONDS) - 1)]
        if retryable else TRANSIENT_BACKOFF_SECONDS[0]
    )
    return started.astimezone(timezone.utc) + timedelta(seconds=delay)
