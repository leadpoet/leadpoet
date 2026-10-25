"""Deterministic future-champion evidence checks; never changes round execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lab_arena import code_review, contracts, reasoning_capture, training_coverage, training_export, training_steps, training_thinking


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


def _all_model_routes_verified(example, run_id):
    calls = [call for call in example.get("provider_calls") or []
             if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
    if not calls:
        return False
    registry = (example.get("configuration") or {}).get("reasoning_route_registry")
    try:
        return all((state := training_thinking.attempt_coverage(
            call, example.get("role", "miner"), registry))["route_reviewed"]
            and state["provider_route_verified"] for call in calls)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _only_benign_empty_thinking(example, run_id):
    calls = [call for call in example.get("provider_calls") or []
             if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
    registry = (example.get("configuration") or {}).get("reasoning_route_registry")
    try:
        states = [training_thinking.attempt_coverage(call, example.get("role", "miner"), registry)
                  for call in calls]
        return bool(states) and any(state["returned"] == "requested_empty" for state in states) and all(
            state["route_reviewed"] and state["provider_route_verified"]
            and state["returned"] in {"requested_empty", "full_thinking"} for state in states)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _documented_selected_privacy_gap(example, run_id):
    return any(
        call.get("run_id") == run_id and (
            (call.get("request_capture") or {}).get("redacted") is True
            or ("credential" in (call.get("response_privacy_redactions") or [])
                and any("credential" in (receipt.get("privacy_redactions") or [])
                        for receipt in call.get("response_receipts") or [])))
        for call in example.get("provider_calls") or [])


def _selected_cost_gap_verified(example, run_id):
    """Recognize an uncertain charge only after selected receipt proof survives."""
    proof = example.get("completeness") or {}
    reasons = set(proof.get("incomplete_reasons") or [])
    allowed = {"provider_cost_unresolved", "response_privacy_nonexact"}
    if "provider_cost_unresolved" not in reasons or not reasons <= allowed:
        return False
    dimensions = proof.get("dimensions") or {}
    by_dimension = proof.get("reasons_by_dimension") or {}
    if (any(dimensions.get(name) is not True or by_dimension.get(name) != []
            for name in ("context", "source_identity", "exact_inputs", "input_output_label"))
            or dimensions.get("costs") is not False
            or dimensions.get("research_process") is not False
            or set(by_dimension.get("costs") or []) != {"provider_cost_unresolved"}
            or set(by_dimension.get("research_process") or []) != reasons
            or ("response_privacy_nonexact" in reasons
                and not _documented_selected_privacy_gap(example, run_id))
            or training_thinking._selected_input_output_binding(example) is None
            or not _all_model_routes_verified(example, run_id)):
        return False
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if (payload.get("complete") is not True
            or artifact.get("payload_sha256") != contracts.document_hash(payload)):
        return False
    selected = [call for call in example.get("provider_calls") or []
                if call.get("run_id") == run_id]
    cost_calls = [call for call in selected
                  if "provider_cost_unresolved" in (call.get("evidence_gaps") or [])]
    if not cost_calls:
        return False
    try:
        for call in selected:
            gaps = set(call.get("evidence_gaps") or [])
            if not gaps <= allowed:
                return False
            if call in cost_calls and (
                call.get("accounting_state") != "uncertain"
                or call.get("actual_microusd") is not None
                or type(call.get("terminal_entry_id")) is not int
                or call["terminal_entry_id"] < 1):
                return False
            if call.get("operation_id") in reasoning_capture.OPERATIONS:
                try:
                    training_steps._candidate(example, call)
                except ValueError as exc:
                    privacy_gap = (str(exc) == "response_privacy_nonexact"
                                   and "response_privacy_nonexact" in reasons)
                    truncated = False
                    if str(exc) == "response_not_complete":
                        state = training_thinking.attempt_coverage(
                            call, example.get("role", "miner"),
                            (example.get("configuration") or {}).get("reasoning_route_registry"))
                        truncated = (state["returned"] == "truncated"
                                     and state["route_reviewed"] and state["provider_route_verified"])
                    if not privacy_gap and not truncated:
                        return False
            request = call.get("request") or {}
            provenance = call.get("response_provenance") or {}
            response = call.get("response")
            if (not isinstance(request, Mapping)
                    or request.get("operation_id") != call.get("operation_id")
                    or request.get("request_hash") != call.get("request_hash")
                    or not isinstance(request.get("hashed_parameters"), Mapping)
                    or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")
                    or not isinstance(response, Mapping)
                    or contracts.document_hash(response) != call.get("response_sha256")
                    or ("credential" in (call.get("response_privacy_redactions") or [])
                        and "response_privacy_nonexact" not in reasons)
                    or (any("credential" in (receipt.get("privacy_redactions") or [])
                            for receipt in call.get("response_receipts") or [])
                        and "response_privacy_nonexact" not in reasons)
                    or provenance.get("agent_receipt") != "reported_exact_body_received"
                    or provenance.get("delivery_to_worker") != "socket_written"):
                return False
            worker_ref = provenance.get("worker_receipt_ref") or {}
            if (worker_ref.get("run_id") != run_id
                    or worker_ref.get("payload_sha256") != artifact["payload_sha256"]):
                return False
            writes = [row for row in payload.get("response_receipts") or []
                      if row.get("ordinal") == worker_ref.get("ordinal")
                      and row.get("call_identity") == call.get("call_identity")
                      and row.get("operation_id") == call.get("operation_id")
                      and row.get("action_sequence") == call.get("action_sequence")
                      and row.get("write_state") == "socket_written"]
            refs = provenance.get("agent_receipt_refs") or []
            reads = [row for row in payload.get("agent_observations") or []
                     if row.get("kind") == "response_received"
                     and any(ref.get("id") == row.get("id") and ref.get("run_id") == run_id
                             and ref.get("payload_sha256") == artifact["payload_sha256"]
                             for ref in refs)]
            receipts = [row for row in call.get("response_receipts") or []
                        if row.get("entry_id") in (provenance.get("worker_received_receipt_entry_ids") or [])
                        and row.get("response_sha256") == call.get("response_sha256")]
            if (len(writes) != 1 or not reads or not receipts
                    or any((row.get("privacy") or {}).get("redacted") is True for row in reads)
                    or not any((read.get("data") or {}).get("call_identity") == call.get("call_identity")
                               and (read.get("data") or {}).get("body_sha256") == receipt.get("wire_body_sha256")
                               and (read.get("data") or {}).get("body_bytes") == receipt.get("wire_body_bytes")
                               and writes[0].get("body_sha256") == receipt.get("wire_body_sha256")
                               and writes[0].get("body_bytes") == receipt.get("wire_body_bytes")
                               for read in reads for receipt in receipts)):
                return False
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return True


def assess_capture(submission: Mapping, examples_by_run: Mapping[str, Mapping | None],
                   selected_run_ids: Sequence[str], *, enabled: bool,
                   minimum_coverage_percent: int = 100) -> dict:
    """Check each selected final execute run against its persisted raw export.

    A missing export or malformed record is an ineligible result, never an
    execution error. The caller chooses final runs; this helper cannot promote
    an omitted run or use a score run as a substitute.
    """
    if not enabled:
        return _result([], enabled=0)
    selected = list(selected_run_ids)
    required = (training_coverage.required_count(len(selected), minimum_coverage_percent)
                if selected else 0)
    counts = {"enabled": 1, "selected_runs": len(selected), "checked_runs": 0,
              "model_calls": 0, "full_thinking_calls": 0, "tool_only_empty_calls": 0,
              "required_complete_runs": required, "complete_runs": 0,
              "incomplete_runs": 0, "minimum_coverage_percent": minimum_coverage_percent}
    reasons = []
    if not selected or len(set(selected)) != len(selected) or any(not isinstance(item, str) or not item for item in selected):
        return _result(["selected_runs_invalid"], **counts)
    baseline = submission.get("is_king") is True
    source_digest = submission.get("source_archive_sha256")
    if (not baseline and
            (not isinstance(source_digest, str) or contracts.SHA256_RE.fullmatch(source_digest) is None)):
        reasons.append("source_archive_hash_missing")
    for run_id in selected:
        run_reasons = []
        example = examples_by_run.get(run_id)
        if not isinstance(example, Mapping):
            run_reasons.append("selected_run_export_missing")
            counts["incomplete_runs"] += 1
            if minimum_coverage_percent == 100:
                reasons.extend(run_reasons)
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
                run_reasons.append("selected_run_identity_unverified")
                continue
            # A self-contradictory completeness flag cannot hide an unsettled
            # selected call. The narrow 80% verifier below must prove it.
            if any(call.get("run_id") == run_id and (
                    call.get("accounting_state") != "settlement"
                    or type(call.get("actual_microusd")) is not int)
                    for call in example.get("provider_calls") or []):
                run_reasons.append("trajectory_evidence_incomplete")
                continue
            ancillary_gaps = training_thinking.selected_research_evidence(example)
            if ancillary_gaps is None:
                run_reasons.append("trajectory_evidence_incomplete")
                continue
            if (not isinstance(example.get("output"), Mapping)
                    or not isinstance(example.get("input_snapshots"), Mapping)
                    or run_id not in example["input_snapshots"]):
                run_reasons.append("input_or_output_unavailable")
                continue
            report = training_thinking.readable_trace(example)
            if (report.get("projection_truncated") is not False
                    or not set(report.get("capture_gaps") or []) <= ancillary_gaps):
                run_reasons.append("readable_trace_incomplete")
                continue
            calls = [(index, call) for index, call in enumerate(example.get("provider_calls") or [])
                     if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
            if not calls:
                run_reasons.append("execution_model_call_missing")
                continue
            registry = (example.get("configuration") or {}).get("reasoning_route_registry")
            has_full = False
            for index, call in calls:
                counts["model_calls"] += 1
                try:
                    candidate = training_steps._candidate(example, call)
                except ValueError as exc:
                    # _candidate checks the request, acknowledged receipt,
                    # exact response hash, and privacy before parsing the
                    # target. A provider's explicit length stop can therefore
                    # consume one incomplete ICP without excusing bad proof.
                    if minimum_coverage_percent < 100 and str(exc) == "response_not_complete":
                        state = training_thinking.attempt_coverage(
                            call, example.get("role", "miner"), registry)
                        if (state["returned"] == "truncated" and state["route_reviewed"]
                                and state["provider_route_verified"]):
                            run_reasons.append("provider_response_truncated")
                            continue
                    raise
                digest = candidate["source_refs"][0]["delivered_response_sha256"]
                delivered = training_thinking._delivered_response(call, digest)
                full_segments, status = training_thinking.full_text_segments(
                    {**call, "response": delivered, "response_sha256": digest},
                    example["role"], registry)
                if not status["route_reviewed"] or not status["provider_route_verified"]:
                    run_reasons.append("full_thinking_route_unverified")
                    continue
                if full_segments:
                    if not _readable_full_segments(example, report, index, delivered, full_segments):
                        run_reasons.append("readable_full_thinking_unverified")
                    else:
                        counts["full_thinking_calls"] += 1
                        has_full = True
                elif training_thinking.verified_tool_only_empty(call, delivered, status):
                    counts["tool_only_empty_calls"] += 1
                else:
                    run_reasons.append("full_thinking_unavailable")
            if not has_full:
                run_reasons.append("full_thinking_call_missing")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            run_reasons.append("selected_run_evidence_invalid")
        finally:
            if run_reasons:
                counts["incomplete_runs"] += 1
                if minimum_coverage_percent == 100:
                    reasons.extend(run_reasons)
                else:
                    soft = {"provider_response_truncated"}
                    soft |= ({"full_thinking_unavailable", "full_thinking_call_missing"}
                            if _only_benign_empty_thinking(example, run_id) else set())
                    if run_reasons == ["trajectory_evidence_incomplete"]:
                        proof_reasons = set((example.get("completeness") or {}).get("incomplete_reasons") or [])
                        if (proof_reasons and proof_reasons <= {"response_privacy_nonexact"}
                                and _documented_selected_privacy_gap(example, run_id)
                                and _all_model_routes_verified(example, run_id)):
                            soft.add("trajectory_evidence_incomplete")
                        if _selected_cost_gap_verified(example, run_id):
                            soft.add("trajectory_evidence_incomplete")
                    reasons.extend(reason for reason in run_reasons if reason not in soft)
            else:
                counts["complete_runs"] += 1
    if minimum_coverage_percent < 100 and counts["complete_runs"] < required:
        reasons.append("training_coverage_incomplete")
    if minimum_coverage_percent == 100:
        for key in ("required_complete_runs", "complete_runs", "incomplete_runs", "minimum_coverage_percent"):
            counts.pop(key)
    return _result(reasons, **counts)


__all__ = ["assess_source", "assess_capture"]
