"""Sanitized candidate-generation failure diagnostics.

The hosted worker and public projection both need to explain why a loop ended
with no buildable image candidate without leaking prompts, source payloads, or
provider bodies. This module keeps that taxonomy in one place.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence


NO_BUILDABLE_CANDIDATE_EVENT_TYPE = "no_buildable_candidate"
NO_BUILDABLE_CANDIDATE_LABEL = "No buildable candidate"

_TEXT_LIMIT = 500
_SECRET_DIAGNOSTIC_RE = re.compile(
    r"(sk-[a-z0-9_-]+|sk-or-[a-z0-9_-]+|api[_-]?key|secret|token|password|"
    r"authorization|bearer\s+[a-z0-9._-]+|hidden[_-]?prompt|judge[_-]?prompt|"
    r"raw[_-]?(?:prompt|response|body|secret)|private[_-]?key|service[_-]?role|"
    r"candidate_patch_manifest|private_model_manifest_doc|image_digest)",
    re.IGNORECASE,
)

_REASON_SUMMARIES = {
    "provider_route_unavailable": "Provider route unavailable during candidate generation.",
    "provider_privacy_verification_failed": "Provider privacy verification failed before generation.",
    "source_inspection_failed": "Source inspection failed before enough editable context was available.",
    "source_context_empty_or_unread": "Source inspection did not return editable files for a patch.",
    "loop_direction_plan_invalid": "The loop plan could not be parsed or validated.",
    "binding_plan_source_missing": "The selected plan did not match inspected editable source.",
    "no_viable_patch": "Planner or drafter could not find a safe patch.",
    "probe_blocked": "A provider probe was blocked by the query guard before upstream contact.",
    "probe_budget_exhausted": "The loop's provider-probe budget was exhausted.",
    "probe_upstream_error": "A provider probe failed at the upstream provider.",
    "provider_probe_refuted_hypothesis": "A provider probe refuted the loop's working hypothesis.",
    "candidate_patch_parse_failed": "Drafted patch response could not be parsed.",
    "candidate_patch_empty_or_noop": "Drafted patch applied but made no repository change.",
    "candidate_patch_apply_failed": "Drafted patch did not apply to the extracted source.",
    "candidate_repair_exhausted": "Patch repair attempts were exhausted.",
    "candidate_patch_test_failed": "Drafted patch failed private tests.",
    "candidate_artifact_missing": "Candidate build did not produce an artifact manifest.",
    "candidate_image_build_failed": "Candidate image build failed.",
    "candidate_build_failed": "Candidate build failed.",
    "no_valid_image_build_finalists": "The loop ended without a valid image-build finalist.",
}

_BUILDABLE_REASONS = frozenset(_REASON_SUMMARIES)


def sanitize_diagnostic_text(value: Any, *, max_length: int = _TEXT_LIMIT) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if _SECRET_DIAGNOSTIC_RE.search(text):
        return "[redacted]"
    return text[: max(0, int(max_length))]


def build_candidate_generation_failure_summary(
    loop_event_rows: Sequence[Mapping[str, Any]] | None,
    *,
    queue_reason: str = "",
    terminal_error: str = "",
    candidate_count: int = 0,
) -> dict[str, Any]:
    events = canonical_loop_event_order(loop_event_rows)
    event_types = [_event_type(row) for row in events if _event_type(row)]
    stage_counts = {
        key: int(value)
        for key, value in Counter(event_types).most_common(20)
        if key
    }
    primary_reason = _classify(events, queue_reason=queue_reason, terminal_error=terminal_error)
    latest_event_type = event_types[-1] if event_types else ""
    latest_stage = _latest_stage(events)
    sample_detail = _sample_detail(events, queue_reason=queue_reason, terminal_error=terminal_error)
    summary = {
        "schema_version": "1.0",
        "primary_reason": primary_reason,
        "public_label": NO_BUILDABLE_CANDIDATE_LABEL,
        "public_summary": _REASON_SUMMARIES.get(primary_reason, _REASON_SUMMARIES["no_valid_image_build_finalists"]),
        "candidate_count": max(0, int(candidate_count or 0)),
        "stage_counts": stage_counts,
        "latest_event_type": latest_event_type,
        "latest_stage": latest_stage,
    }
    if sample_detail:
        summary["sample_detail"] = sample_detail
    if queue_reason:
        summary["queue_reason"] = sanitize_diagnostic_text(queue_reason, max_length=160)
    return summary


def public_candidate_generation_failure_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    primary_reason = str(summary.get("primary_reason") or "no_valid_image_build_finalists")
    if primary_reason not in _BUILDABLE_REASONS:
        primary_reason = "no_valid_image_build_finalists"
    stage_counts = summary.get("stage_counts") if isinstance(summary.get("stage_counts"), Mapping) else {}
    public_doc = {
        "schema_version": "1.0",
        "primary_reason": primary_reason,
        "public_label": NO_BUILDABLE_CANDIDATE_LABEL,
        "public_summary": _REASON_SUMMARIES.get(primary_reason, _REASON_SUMMARIES["no_valid_image_build_finalists"]),
        "stage_counts": {
            sanitize_diagnostic_text(key, max_length=80): int(value)
            for key, value in list(stage_counts.items())[:12]
            if key
        },
    }
    latest_event_type = sanitize_diagnostic_text(summary.get("latest_event_type"), max_length=80)
    latest_stage = sanitize_diagnostic_text(summary.get("latest_stage"), max_length=120)
    if latest_event_type:
        public_doc["latest_event_type"] = latest_event_type
    if latest_stage:
        public_doc["latest_stage"] = latest_stage
    return public_doc


def canonical_loop_event_order(
    loop_event_rows: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Return deterministic chronological order for DB, report, and in-memory rows."""

    indexed = list(enumerate(loop_event_rows or ()))

    def _key(item: tuple[int, Mapping[str, Any]]) -> tuple[Any, ...]:
        original_index, row = item
        try:
            seq = int(row.get("seq"))
        except (TypeError, ValueError):
            seq = None
        created_at = str(row.get("created_at") or "")
        if seq is not None and seq >= 0:
            return (0, seq, created_at, original_index)
        return (1, created_at, original_index)

    return [row for _index, row in sorted(indexed, key=_key)]


def _classify(
    events: Sequence[Mapping[str, Any]],
    *,
    queue_reason: str,
    terminal_error: str,
) -> str:
    text = _combined_text(events, queue_reason=queue_reason, terminal_error=terminal_error)
    event_types = {_event_type(row) for row in events}
    cost_stop_reasons = {_cost_stop_reason(row) for row in events}
    latest_no_viable = _terminal_no_viable_doc(events)
    structured_failure_class = str(latest_no_viable.get("failure_class") or "").strip().lower()

    if "no endpoints found" in text or "no endpoint found" in text:
        return "provider_route_unavailable"
    if "workspace privacy verification failed" in text or "privacy verification failed" in text:
        return "provider_privacy_verification_failed"
    if structured_failure_class == "binding_plan_unimplementable":
        return "binding_plan_source_missing"
    if structured_failure_class == "provider_probe_refuted_hypothesis":
        return "provider_probe_refuted_hypothesis"
    if "loop_direction_plan_parse_failed" in text or "resume_loop_direction_plan_parse_failed" in text:
        return "loop_direction_plan_invalid"
    if _has_binding_plan_source_gap(latest_no_viable):
        return "binding_plan_source_missing"
    if _terminal_source_failure(events):
        return "source_inspection_failed"
    if _terminal_source_unread(events):
        return "source_context_empty_or_unread"
    if "candidate_patch_parse_failed" in event_types or "code_edit_parse_failed" in cost_stop_reasons:
        return "candidate_patch_parse_failed"
    if "candidate_patch_empty_or_noop" in event_types or "no repository changes" in text or "no-op" in text:
        return "candidate_patch_empty_or_noop"
    if "candidate_repair_exhausted" in event_types or "patch_apply_repair_exhausted" in text:
        return "candidate_repair_exhausted"
    if "candidate_patch_apply_failed" in event_types:
        return "candidate_patch_apply_failed"
    if "candidate_patch_test_failed" in event_types or "candidate_test_failed" in event_types:
        return "candidate_patch_test_failed"
    if "candidate_artifact_missing" in event_types or "artifact manifest output" in text:
        return "candidate_artifact_missing"
    if "candidate_image_build_failed" in event_types:
        return "candidate_image_build_failed"
    if "candidate_build_failed" in event_types:
        return "candidate_build_failed"
    if latest_no_viable:
        # W4: a refusal that cites probe evidence is a learning outcome, not a
        # planner failure — surface it as its own reason.
        if structured_failure_class == "provider_probe_refuted_hypothesis" or (
            "probe" in text and "refut" in text
        ):
            return "provider_probe_refuted_hypothesis"
        return "no_viable_patch"
    if "probe_budget_exhausted" in text:
        return "probe_budget_exhausted"
    if "probe_upstream_error" in text:
        return "probe_upstream_error"
    if "probe_blocked" in event_types:
        return "probe_blocked"
    if "no_valid_image_build_finalists" in text:
        return "no_valid_image_build_finalists"
    return "no_valid_image_build_finalists"


def _has_binding_plan_source_gap(latest: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(latest.get(key) or "")
        for key in ("stage", "error", "reason", "stop_reason", "failure_class")
    ).lower()
    if "binding_plan_unimplementable" in text:
        return True
    gap_markers = (
        "not present",
        "not listed",
        "no safe source path",
        "outside editable",
        "cannot be inspected",
        "selected path",
        "selected_path",
        "no existing test file",
        "no existing test is listed",
        "no test file appears",
        "no test file is listed",
    )
    return bool(latest) and any(marker in text for marker in gap_markers)


_CANDIDATE_GENERATION_SIGNAL_TYPES = frozenset(
    {
        "source_inspection_requested",
        "source_inspection_resolved",
        "source_inspection_failed",
        "code_edit_validation_failed",
        "candidate_generation_fallback_requested",
        "candidate_generation_fallback_drafted",
        "candidate_generation_fallback_failed",
        "candidate_patch_parse_failed",
        "candidate_patch_empty_or_noop",
        "candidate_repair_exhausted",
        "candidate_patch_apply_failed",
        "candidate_patch_test_failed",
        "candidate_test_failed",
        "candidate_artifact_missing",
        "candidate_image_build_failed",
        "candidate_build_failed",
        "candidate_built",
        "candidate_selected",
        "no_viable_patch",
    }
)


def _terminal_no_viable_doc(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    latest_signal_index = _last_matching_index(
        events,
        lambda row: _event_type(row) in _CANDIDATE_GENERATION_SIGNAL_TYPES,
    )
    if latest_signal_index < 0 or _event_type(events[latest_signal_index]) != "no_viable_patch":
        return {}
    return _event_doc(events[latest_signal_index])


def _successful_source_read(row: Mapping[str, Any]) -> bool:
    if _event_type(row) != "source_inspection_resolved":
        return False
    doc = _event_doc(row)
    try:
        if int(doc.get("read_file_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    read_files = doc.get("read_files")
    return isinstance(read_files, Sequence) and not isinstance(
        read_files, (str, bytes, bytearray)
    ) and bool(read_files)


def _last_matching_index(
    events: Sequence[Mapping[str, Any]], predicate: Any
) -> int:
    found = -1
    for index, row in enumerate(events):
        if predicate(row):
            found = index
    return found


def _event_iteration(row: Mapping[str, Any]) -> int | None:
    doc = _event_doc(row)
    for value in (doc.get("iteration"), doc.get("loop_iteration"), row.get("iteration")):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _terminal_iteration_events(
    events: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    iterations = [iteration for row in events if (iteration := _event_iteration(row)) is not None]
    if not iterations:
        return events
    terminal_iteration = max(iterations)
    return [row for row in events if _event_iteration(row) == terminal_iteration]


def _terminal_source_failure(events: Sequence[Mapping[str, Any]]) -> bool:
    events = _terminal_iteration_events(events)
    last_failure = _last_matching_index(
        events, lambda row: _event_type(row) == "source_inspection_failed"
    )
    last_success = _last_matching_index(events, _successful_source_read)
    return last_failure >= 0 and last_failure > last_success


def _terminal_source_unread(events: Sequence[Mapping[str, Any]]) -> bool:
    events = _terminal_iteration_events(events)

    def _is_unread(row: Mapping[str, Any]) -> bool:
        if _event_type(row) != "code_edit_validation_failed":
            return False
        doc = _event_doc(row)
        return (
            str(doc.get("error") or "") == "code_edit_no_source_files_read"
            or str(doc.get("stage") or "") == "source_inspection_exhausted_without_read"
        )

    last_unread = _last_matching_index(events, _is_unread)
    last_success = _last_matching_index(events, _successful_source_read)
    return last_unread >= 0 and last_unread > last_success


def _combined_text(
    events: Sequence[Mapping[str, Any]],
    *,
    queue_reason: str,
    terminal_error: str,
) -> str:
    chunks = [str(queue_reason or ""), str(terminal_error or "")]
    for row in events:
        chunks.append(_event_type(row))
        doc = _event_doc(row)
        for key in ("stage", "error", "reason", "stop_reason", "failure_reason", "failure_class"):
            if key in doc:
                chunks.append(str(doc.get(key) or ""))
        provider_usage = row.get("provider_usage")
        if isinstance(provider_usage, Mapping):
            chunks.append(str(provider_usage.get("call_stage") or ""))
            request_doc = provider_usage.get("failed_request")
            if isinstance(request_doc, Mapping):
                chunks.append(str(request_doc.get("error_class") or ""))
                chunks.append(str(request_doc.get("http_status") or ""))
        elif isinstance(provider_usage, list):
            for item in provider_usage[:8]:
                if not isinstance(item, Mapping):
                    continue
                chunks.append(str(item.get("call_stage") or ""))
                request_doc = item.get("failed_request")
                if isinstance(request_doc, Mapping):
                    chunks.append(str(request_doc.get("error_class") or ""))
                    chunks.append(str(request_doc.get("http_status") or ""))
    return " ".join(chunks).lower()


def _sample_detail(
    events: Sequence[Mapping[str, Any]],
    *,
    queue_reason: str,
    terminal_error: str,
) -> str:
    for raw in (terminal_error, queue_reason):
        detail = sanitize_diagnostic_text(raw, max_length=240)
        if detail:
            return detail
    for row in reversed(events):
        doc = _event_doc(row)
        for key in ("error", "reason", "stop_reason", "stage"):
            detail = sanitize_diagnostic_text(doc.get(key), max_length=240)
            if detail:
                return detail
    return ""


def _latest_stage(events: Sequence[Mapping[str, Any]]) -> str:
    for row in reversed(events):
        doc = _event_doc(row)
        stage = sanitize_diagnostic_text(doc.get("stage"), max_length=120)
        if stage:
            return stage
        event_type = sanitize_diagnostic_text(_event_type(row), max_length=120)
        if event_type:
            return event_type
    return ""


def _event_type(row: Mapping[str, Any]) -> str:
    return sanitize_diagnostic_text(row.get("event_type"), max_length=120)


def _cost_stop_reason(row: Mapping[str, Any]) -> str:
    ledger = row.get("cost_ledger")
    if not isinstance(ledger, Mapping):
        return ""
    return sanitize_diagnostic_text(ledger.get("stop_reason"), max_length=120)


def _event_doc(row: Mapping[str, Any]) -> Mapping[str, Any]:
    doc = row.get("event_doc")
    return doc if isinstance(doc, Mapping) else {}
