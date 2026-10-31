"""Deterministic future-champion evidence checks; never changes round execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lab_arena import baseline_reasoning, code_review, contracts, reasoning_capture, training_coverage, training_export, training_steps, training_thinking, training_worker_proof


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


def _empty_stopped_chat_output(response):
    """Recognize only the observed null Chat result, with explicit zero usage."""
    if (not isinstance(response, Mapping)
            or not set(response) <= {"id", "object", "created", "model", "provider", "choices", "usage",
                                     "system_fingerprint", "service_tier", "openrouter_metadata"}):
        return False
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return False
    choice = choices[0]
    target = choice.get("message")
    if (not set(choice) <= {"index", "finish_reason", "native_finish_reason", "message", "logprobs"}
            or choice.get("finish_reason") != "stop"
            or choice.get("native_finish_reason") not in (None, "stop")
            or choice.get("logprobs") is not None
            or not isinstance(target, Mapping) or target.get("role") != "assistant"
            or "content" not in target or target["content"] is not None
            or not set(target) <= {"role", "content", "reasoning", "reasoning_content", "refusal", "tool_calls"}
            or any(target.get(field) is not None for field in ("reasoning", "reasoning_content", "refusal"))
            or target.get("tool_calls") not in (None, [])):
        return False
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return False
    sources = (usage, usage.get("completion_tokens_details", {}), usage.get("output_tokens_details", {}))
    if not all(isinstance(source, Mapping) for source in sources):
        return False
    tokens = [source["reasoning_tokens"] for source in sources if "reasoning_tokens" in source]
    return bool(tokens) and all(type(value) is int and value == 0 for value in tokens)


def _verified_provider_output_fault(example, call, *, worker_proofs=None):
    """Prove an unusable generated action without admitting a training target.

    Only the observed output faults qualify. Reaching these target errors
    must still prove the exact request, input history and host-written response.
    Actual provider routing and settled cost are independent requirements.
    """
    configuration = example.get("configuration") or {}
    if (training_coverage.MISSING_REASONING_SETTING not in configuration
            or not training_coverage.uses_host_result_evidence(configuration)
            or call.get("accounting_state") != "settlement"
            or type(call.get("actual_microusd")) is not int
            or call["actual_microusd"] < 0 or call.get("evidence_gaps")
            or not training_thinking._terminal_accounting_matches(call, "settlement", call["actual_microusd"])):
        return False
    try:
        # Reject ordinary outputs before hashing a potentially large artifact.
        # This filter grants nothing; the complete candidate proof follows.
        response = call.get("response") or {}
        op = call.get("operation_id")
        target = (response.get("output") if op == "openrouter.responses"
                  else response["choices"][0]["message"])
        reasoning_only = (op == "openrouter.responses" and isinstance(target, list) and bool(target)
                          and all(isinstance(item, Mapping) and item.get("type") == "reasoning"
                                  for item in target))
        empty_chat = (op == "openrouter.chat" and _empty_stopped_chat_output(response)
                      and not call.get("response_privacy_redactions")
                      and not any(receipt.get("privacy_redactions") for receipt in call.get("response_receipts") or []))
        if not reasoning_only and not empty_chat:
            try:
                training_steps._validate_target_tools(
                    call["request"]["outbound"]["body"], target, op)
            except ValueError as exc:
                if str(exc) != "undeclared_tool_call":
                    return False
            else:
                return False
        evidence = training_steps._host_result(example, call, worker_proofs=worker_proofs)
        if (evidence["write"]["status"] != 200
                or not isinstance(call.get("response"), Mapping)
                or contracts.document_hash(call["response"]) != call.get("response_sha256")
                or call["response"] != evidence["response"]
                or not set(call.get("response_privacy_redactions") or []) <= {"hidden_reasoning"}):
            return False
        state = training_thinking.attempt_coverage(
            call, example.get("role", "miner"), baseline_reasoning.for_example(example).get("reasoning_route_registry"))
        if not state["route_reviewed"] or not state["provider_route_verified"]:
            return False
        try:
            training_steps._candidate(example, call, worker_proofs=worker_proofs)
        except ValueError as exc:
            if str(exc) == "assistant_shape_unsupported" and empty_chat:
                # The closed null shape and explicit zero counters above must
                # still agree with the independent route/accounting projection.
                return state["tokens_reported"] == 0
            if str(exc) == "undeclared_tool_call":
                return True
            if str(exc) == "response_choice_ambiguous" and call.get("operation_id") == "openrouter.responses":
                output = evidence["response"].get("output")
                return (isinstance(output, list) and bool(output)
                        and all(isinstance(item, Mapping) and item.get("type") == "reasoning"
                                for item in output))
        return False
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def assess_capture(submission: Mapping, examples_by_run: Mapping[str, Mapping | None],
                   selected_run_ids: Sequence[str], *, enabled: bool,
                   minimum_coverage_percent: int = 100,
                   expected_run_count: int | None = None,
                   maximum_missing_reasoning_percent: int = 0,
                   unavailable_exports: Mapping[str, str] | None = None,
                   unavailable_outputs: Mapping[str, str] | None = None,
                   unavailable_artifacts: Mapping[str, Mapping[str, str]] | None = None) -> dict:
    """Check each selected final execute run against its persisted raw export.

    A missing export or malformed record is an incomplete ICP, never an
    execution error. The caller chooses final runs; this helper cannot promote
    an omitted run or use a score run as a substitute.
    """
    if not enabled:
        return _result([], enabled=0)
    if type(maximum_missing_reasoning_percent) is not int or not 0 <= maximum_missing_reasoning_percent <= 100:
        raise ValueError("invalid_maximum_missing_reasoning_percent")
    selected = list(selected_run_ids)
    expected = len(selected) if expected_run_count is None else expected_run_count
    if type(expected) is not int or expected < len(selected) or expected < 1:
        return _result(["selected_runs_invalid"], enabled=1, selected_runs=len(selected))
    required = training_coverage.required_count(expected, minimum_coverage_percent)
    missing_selected = expected - len(selected)
    counts = {"enabled": 1, "selected_runs": len(selected), "checked_runs": 0,
              "model_calls": 0, "full_thinking_calls": 0, "tool_only_empty_calls": 0,
              "required_complete_runs": required, "complete_runs": 0,
              "incomplete_runs": missing_selected, "minimum_coverage_percent": minimum_coverage_percent}
    if missing_selected:
        counts["missing_selected_runs"] = missing_selected
    reasoning_totals = {"completed_untruncated_verified_reasoning_tokens": 0,
                        "missing_reasoning_tokens": 0, "missing_reasoning_calls": 0,
                        "unknown_reasoning_usage_calls": 0, "truncated_model_calls": 0,
                        "verified_failed_model_attempts": 0}
    reasoning_by_run = []
    policy_stats_enabled = False
    reasons = []
    rejected_runs = []
    if any(not isinstance(item, str) or not item for item in selected) or len(set(selected)) != len(selected):
        return _result(["selected_runs_invalid"], **counts)
    if missing_selected and minimum_coverage_percent == 100:
        reasons.append("selected_execution_coverage_incomplete")
    baseline = submission.get("is_king") is True
    source_digest = submission.get("source_archive_sha256")
    if (not baseline and
            (not isinstance(source_digest, str) or contracts.SHA256_RE.fullmatch(source_digest) is None)):
        reasons.append("source_archive_hash_missing")
    for run_id in selected:
        run_reasons = []
        worker_proofs = report = None
        reasoning_counts = {key: 0 for key in reasoning_totals}
        example = examples_by_run.get(run_id)
        if not isinstance(example, Mapping):
            error_class = (unavailable_exports or {}).get(run_id)
            run_reasons.append("selected_run_export_invalid" if error_class else "selected_run_export_missing")
            counts["incomplete_runs"] += 1
            rejected_runs.append({"run_id": run_id, "reasons": run_reasons,
                                  **({"error_class": error_class} if error_class else {})})
            if minimum_coverage_percent == 100:
                reasons.extend(run_reasons)
            continue
        counts["checked_runs"] += 1
        new_policy = (isinstance(example.get("configuration"), Mapping)
                      and training_coverage.MISSING_REASONING_SETTING in example["configuration"])
        policy_stats_enabled |= new_policy
        if (unavailable_artifacts or {}).get(run_id):
            run_reasons.append("selected_artifact_temporarily_unavailable")
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
                    or baseline_reasoning.for_example(example).get("reasoning_capture_policy") != reasoning_capture.POLICY
                    or training_coverage.missing_reasoning_percent_from_configuration(
                        example.get("configuration") or {}) != maximum_missing_reasoning_percent):
                run_reasons.append("selected_run_identity_unverified")
                continue
            worker_proofs = training_worker_proof.WorkerProofs()
            example = worker_proofs.prepare(example)
            ancillary_gaps = training_thinking.selected_research_evidence(
                example, worker_proofs=worker_proofs)
            if ancillary_gaps is None:
                run_reasons.append("trajectory_evidence_incomplete")
                continue
            if (not isinstance(example.get("output"), Mapping)
                    or not isinstance(example.get("input_snapshots"), Mapping)
                    or run_id not in example["input_snapshots"]):
                run_reasons.append("input_or_output_unavailable")
                continue
            report = training_thinking.readable_trace(example, worker_proofs=worker_proofs)
            if (report.get("projection_truncated") is not False
                    or not set(report.get("capture_gaps") or []) <= ancillary_gaps):
                run_reasons.append("readable_trace_incomplete")
                continue
            calls = [(index, call) for index, call in enumerate(example.get("provider_calls") or [])
                     if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
            if not calls:
                run_reasons.append("execution_model_call_missing")
                continue
            registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
            has_full = False
            for index, call in calls:
                counts["model_calls"] += 1
                if new_policy and training_thinking.verified_failed_attempt(example, call, worker_proofs=worker_proofs):
                    reasoning_counts["verified_failed_model_attempts"] += 1
                    continue
                try:
                    candidate = training_steps._candidate(example, call, worker_proofs=worker_proofs)
                except ValueError as exc:
                    if new_policy and _verified_provider_output_fault(example, call, worker_proofs=worker_proofs):
                        run_reasons.append("provider_output_untrainable")
                        continue
                    if new_policy and training_thinking.verified_cap_truncated_target(example, call, worker_proofs=worker_proofs):
                        reasoning_counts["truncated_model_calls"] += 1
                        continue
                    # _candidate checks the request, acknowledged receipt,
                    # exact response hash, and privacy before parsing the
                    # target. A provider's explicit length stop can therefore
                    # consume one incomplete ICP without excusing bad proof.
                    if str(exc) == "response_not_complete":
                        state = training_thinking.attempt_coverage(
                            call, example.get("role", "miner"), registry)
                        if (state["returned"] == "truncated" and state["route_reviewed"]
                                and state["provider_route_verified"]):
                            reasoning_counts["truncated_model_calls"] += 1
                            if not new_policy:
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
                tokens = status["tokens_reported"]
                if tokens is None:
                    reasoning_counts["unknown_reasoning_usage_calls"] += 1
                else:
                    reasoning_counts["completed_untruncated_verified_reasoning_tokens"] += tokens
                if full_segments:
                    if not _readable_full_segments(example, report, index, delivered, full_segments):
                        run_reasons.append("readable_full_thinking_unverified")
                    else:
                        counts["full_thinking_calls"] += 1
                        has_full = True
                elif (training_thinking.verified_no_thinking(call, delivered, status)
                      if new_policy else
                      training_thinking.verified_tool_only_empty(call, delivered, status)):
                    counts["tool_only_empty_calls"] += 1
                elif new_policy and training_thinking.verified_missing_thinking(status):
                    reasoning_counts["missing_reasoning_calls"] += 1
                    reasoning_counts["missing_reasoning_tokens"] += tokens
                else:
                    run_reasons.append("full_thinking_unavailable")
            if not has_full:
                run_reasons.append("full_thinking_call_missing")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            run_reasons.append("selected_run_evidence_invalid")
        finally:
            if new_policy:
                reasoning_by_run.append({"run_id": run_id, **reasoning_counts})
                for key, value in reasoning_counts.items():
                    reasoning_totals[key] += value
                if (reasoning_counts["missing_reasoning_tokens"] * 100
                        > reasoning_counts["completed_untruncated_verified_reasoning_tokens"]
                        * maximum_missing_reasoning_percent):
                    run_reasons.append("missing_reasoning_share_exceeded")
            unavailable_ref = (unavailable_outputs or {}).get(run_id)
            if unavailable_ref is not None:
                identity = example.get("output_identity") or {}
                output = example.get("output")
                if (unavailable_ref == "arena/%s/outputs/%s.json" % (example.get("round_id"), run_id)
                        and identity.get("ref") == unavailable_ref
                        and isinstance(output, Mapping)
                        and isinstance(identity.get("document_hash"), str)
                        and contracts.SHA256_RE.fullmatch(identity["document_hash"]) is not None):
                    # The complete export was checked against the exact bytes
                    # retained by this publication attempt. Availability is
                    # a separate one-ICP coverage gap, never a raw export edit.
                    run_reasons.append("accepted_output_unavailable")
                else:
                    run_reasons.append("selected_run_evidence_invalid")
            if run_reasons:
                counts["incomplete_runs"] += 1
                if minimum_coverage_percent == 100:
                    reasons.extend(run_reasons)
                rejected_runs.append({"run_id": run_id, "reasons": sorted(set(run_reasons))})
            else:
                counts["complete_runs"] += 1
            # The service loads the next export on demand. Drop both the owned
            # worker view and references into this export before that read.
            worker_proofs = example = report = None
            calls = call = candidate = delivered = full_segments = None
    if rejected_runs:
        counts["rejected_runs"] = rejected_runs
    if policy_stats_enabled:
        counts.update(reasoning_totals)
        counts["maximum_missing_reasoning_percent"] = maximum_missing_reasoning_percent
        counts["reasoning_by_run"] = reasoning_by_run
    if minimum_coverage_percent < 100 and counts["complete_runs"] < required:
        reasons.append("training_coverage_incomplete")
    if minimum_coverage_percent == 100:
        for key in ("required_complete_runs", "complete_runs", "incomplete_runs", "minimum_coverage_percent"):
            counts.pop(key)
    return _result(reasons, **counts)


__all__ = ["assess_source", "assess_capture"]
