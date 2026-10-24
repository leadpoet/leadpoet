"""Deterministic future-champion evidence checks; never changes round execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lab_arena import code_review, contracts, reasoning_capture, training_export, training_steps, training_thinking


def _result(reasons, **counts):
    return {"eligible": not reasons, "reasons": sorted(set(reasons)), "counts": counts}


def assess_source(submission: Mapping, *, enabled: bool) -> dict:
    """Require the stored complete-source review to bind the exact archive."""
    if not enabled:
        return _result([], enabled=0)
    review = submission.get("code_review_doc") or {}
    review = review if isinstance(review, Mapping) else {}
    research = review.get("research_control")
    digest = submission.get("source_archive_sha256")
    reasons = []
    if submission.get("code_review_status") != "passed" or review.get("passed") is not True:
        reasons.append("source_review_not_passed")
    if not isinstance(digest, str) or contracts.SHA256_RE.fullmatch(digest) is None:
        reasons.append("source_archive_hash_missing")
    if (not isinstance(research, Mapping)
            or research.get("policy_id") != code_review.RESEARCH_POLICY_ID
            or research.get("policy_hash") != code_review.RESEARCH_POLICY_HASH
            or research.get("reviewer_version") != code_review.RESEARCH_REVIEWER_VERSION
            or research.get("source_archive_sha256") != digest
            or research.get("classification") != "model_directed"
            or set(research.get("evidence_claims") or []) != {"observation_to_action", "finish_continue"}):
        reasons.append("model_directed_source_unverified")
    return _result(reasons, enabled=1, reviewed=int(not reasons))


def _at(example, path):
    value = example
    for part in path:
        value = value[part]
    return value


def _readable_full_segments(example, report, call_index, delivered, segments):
    """Every admitted text segment must point to the identical retained raw value."""
    call = example["provider_calls"][call_index]
    digest = contracts.document_hash(delivered)
    roots = []
    if call.get("response_sha256") == digest and call.get("response") == delivered:
        roots.append(["provider_calls", call_index, "response"])
    for receipt_index, receipt in enumerate(call.get("response_receipts") or []):
        if receipt.get("response_sha256") == digest and receipt.get("response") == delivered:
            roots.append(["provider_calls", call_index, "response_receipts", receipt_index, "response"])
    if not roots:
        return False
    events = [event for event in report["calls"][call_index]["events"]
              if event.get("reasoning_field") and "source_path" in event]
    indexed = {tuple(event["source_path"]): event for event in events}
    for segment in segments:
        found = False
        for root in roots:
            for source_path in segment["source_paths"]:
                path = root + source_path
                event = indexed.get(tuple(path))
                if (event is not None and event.get("value") == segment["text"]
                        and event.get("value_sha256") == contracts.document_hash(segment["text"])):
                    try:
                        found = _at(example, path) == segment["text"]
                    except (KeyError, IndexError, TypeError):
                        found = False
                    if found:
                        break
            if found:
                break
        if not found:
            return False
    return True


def assess_capture(submission: Mapping, examples_by_run: Mapping[str, Mapping | None],
                   selected_run_ids: Sequence[str], *, enabled: bool) -> dict:
    """Check each selected final execute run against its persisted raw export.

    A missing export or malformed record is an ineligible result, never an
    execution error. The caller chooses final runs; this helper cannot promote
    an omitted run or use a score run as a substitute.
    """
    if not enabled:
        return _result([], enabled=0)
    selected = list(selected_run_ids)
    counts = {"enabled": 1, "selected_runs": len(selected), "checked_runs": 0,
              "model_calls": 0, "full_thinking_calls": 0, "tool_only_empty_calls": 0}
    reasons = []
    if not selected or len(set(selected)) != len(selected) or any(not isinstance(item, str) or not item for item in selected):
        return _result(["selected_runs_invalid"], **counts)
    baseline = submission.get("is_king") is True
    source_digest = submission.get("source_archive_sha256")
    if (not baseline and
            (not isinstance(source_digest, str) or contracts.SHA256_RE.fullmatch(source_digest) is None)):
        reasons.append("source_archive_hash_missing")
    for run_id in selected:
        example = examples_by_run.get(run_id)
        if not isinstance(example, Mapping):
            reasons.append("selected_run_export_missing")
            continue
        counts["checked_runs"] += 1
        try:
            source = example.get("source") or {}
            baseline_source_ok = (baseline and source_digest in (None, "")
                                  and source.get("hash_origin") == "computed_from_frozen_object"
                                  and isinstance(source.get("sha256"), str)
                                  and contracts.SHA256_RE.fullmatch(source["sha256"]) is not None)
            if (example.get("schema_version") != training_export.SCHEMA_VERSION
                    or example.get("example_hash") != contracts.document_hash({
                        key: value for key, value in example.items() if key != "example_hash"})
                    or example.get("run_id") != run_id
                    or example.get("submission_id") != submission.get("submission_id")
                    or example.get("role") != ("baseline" if baseline else "miner")
                    or not (isinstance(source.get("sha256"), str)
                            and contracts.SHA256_RE.fullmatch(source["sha256"]) is not None
                            and (source["sha256"] == source_digest or baseline_source_ok))
                    or (example.get("configuration") or {}).get("reasoning_capture_policy") != reasoning_capture.POLICY):
                reasons.append("selected_run_identity_unverified")
                continue
            completeness = example.get("completeness") or {}
            if completeness.get("complete") is not True or completeness.get("incomplete_reasons"):
                reasons.append("trajectory_evidence_incomplete")
                continue
            if (not isinstance(example.get("output"), Mapping)
                    or not isinstance(example.get("input_snapshots"), Mapping)
                    or run_id not in example["input_snapshots"]):
                reasons.append("input_or_output_unavailable")
                continue
            report = training_thinking.readable_trace(example)
            if report.get("projection_truncated") is not False or report.get("capture_gaps"):
                reasons.append("readable_trace_incomplete")
                continue
            calls = [(index, call) for index, call in enumerate(example.get("provider_calls") or [])
                     if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
            if not calls:
                reasons.append("execution_model_call_missing")
                continue
            registry = (example.get("configuration") or {}).get("reasoning_route_registry")
            has_full = False
            for index, call in calls:
                counts["model_calls"] += 1
                candidate = training_steps._candidate(example, call)
                digest = candidate["source_refs"][0]["delivered_response_sha256"]
                delivered = training_thinking._delivered_response(call, digest)
                full_segments, status = training_thinking.full_text_segments(
                    {**call, "response": delivered, "response_sha256": digest},
                    example["role"], registry)
                if not status["route_reviewed"] or not status["provider_route_verified"]:
                    reasons.append("full_thinking_route_unverified")
                    continue
                if full_segments:
                    if not _readable_full_segments(example, report, index, delivered, full_segments):
                        reasons.append("readable_full_thinking_unverified")
                    else:
                        counts["full_thinking_calls"] += 1
                        has_full = True
                elif training_thinking.verified_tool_only_empty(call, delivered, status):
                    counts["tool_only_empty_calls"] += 1
                else:
                    reasons.append("full_thinking_unavailable")
            if not has_full:
                reasons.append("full_thinking_call_missing")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            reasons.append("selected_run_evidence_invalid")
    return _result(reasons, **counts)


__all__ = ["assess_source", "assess_capture"]
