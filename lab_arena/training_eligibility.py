"""Deterministic future-champion evidence checks; never changes round execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lab_arena import code_review, contracts, operations, reasoning_capture, training_coverage, training_export, training_observations, training_steps, training_thinking


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
        new_policy = training_coverage.MISSING_REASONING_SETTING in (example.get("configuration") or {})
        return all(new_policy and training_thinking.verified_failed_attempt(example, call) or (
            (state := training_thinking.attempt_coverage(
                call, example.get("role", "miner"), registry))["route_reviewed"]
            and state["provider_route_verified"]) for call in calls)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


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


def _verified_provider_output_fault(example, call):
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
        evidence = training_steps._host_result(example, call)
        if (evidence["write"]["status"] != 200
                or not isinstance(call.get("response"), Mapping)
                or contracts.document_hash(call["response"]) != call.get("response_sha256")
                or call["response"] != evidence["response"]
                or not set(call.get("response_privacy_redactions") or []) <= {"hidden_reasoning"}):
            return False
        state = training_thinking.attempt_coverage(
            call, example.get("role", "miner"), configuration.get("reasoning_route_registry"))
        if not state["route_reviewed"] or not state["provider_route_verified"]:
            return False
        try:
            training_steps._candidate(example, call)
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


def _only_benign_thinking_gaps(example, run_id, *, new_policy=False):
    """Verified omissions and length stops share one incomplete ICP budget."""
    calls = [call for call in example.get("provider_calls") or []
             if call.get("run_id") == run_id and call.get("operation_id") in reasoning_capture.OPERATIONS]
    registry = (example.get("configuration") or {}).get("reasoning_route_registry")
    try:
        output_fault = new_policy and any(_verified_provider_output_fault(example, call) for call in calls)
        checked = [training_thinking.full_text_segments(call, example.get("role", "miner"), registry)
                   for call in calls if not (new_policy and (
                       training_thinking.verified_failed_attempt(example, call)
                       or training_thinking.verified_cap_truncated_target(example, call)
                       or _verified_provider_output_fault(example, call)))]
        # An execution with only verified failed calls has no training target.
        # It consumes an incomplete ICP, while every failed receipt stays proven.
        if not checked:
            return bool(calls)
        # Coverage labels preserve ancillary summaries and removed opaque state.
        # Admission uses the proven segments instead of those display labels.
        return (output_fault or any(state["returned"] in {"requested_empty", "truncated", "missing_thinking", "summary", "opaque"}
                                     and not full for full, state in checked)) and all(
            state["route_reviewed"] and state["provider_route_verified"]
            and (full or state["returned"] in {"requested_empty", "truncated", "missing_thinking"}
                 or new_policy and training_thinking.verified_missing_thinking(state))
            for full, state in checked)
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


def _selected_cost_gap_verified(example, run_id, *, unavailable_source=False):
    """Recognize an uncertain charge only after selected receipt proof survives.

    Only the separately proved source-timeout path can supply the missing
    source dimension. All input/output and provider evidence remains required.
    """
    example, _ = training_thinking.provider_visible_view(example)
    proof = example.get("completeness") or {}
    source_reasons = {"source_unverified"} if unavailable_source else set()
    new_policy = training_coverage.MISSING_REASONING_SETTING in (example.get("configuration") or {})
    stored_reasons = set(proof.get("incomplete_reasons") or [])
    if not source_reasons <= stored_reasons:
        return False
    reasons = stored_reasons - source_reasons
    cost_snapshot_unavailable = ((example.get("competition_cost_eligibility") or {}).get("status")
                                 == "unavailable")
    call_allowed = {"provider_cost_unresolved", "response_privacy_nonexact"}
    allowed = set(call_allowed)
    if new_policy and "full_response_unavailable" in reasons:
        unavailable = [call for call in example.get("provider_calls") or []
                       if call.get("run_id") == run_id
                       and "full_response_unavailable" in (call.get("evidence_gaps") or [])]
        if not unavailable or not all(
                training_thinking._verified_unavailable_upstream_failure(example, call)
                and training_thinking.verified_failed_attempt(example, call) for call in unavailable):
            return False
        allowed.add("full_response_unavailable")
    if cost_snapshot_unavailable:
        allowed.add("competition_cost_eligibility_unavailable")
    if "provider_cost_unresolved" not in reasons or not reasons <= allowed:
        return False
    dimensions = proof.get("dimensions") or {}
    by_dimension = proof.get("reasons_by_dimension") or {}
    if (any(dimensions.get(name) is not True or by_dimension.get(name) != []
            for name in ("context", "exact_inputs", "input_output_label"))
            or dimensions.get("source_identity") is not (not source_reasons)
            or set(by_dimension.get("source_identity") or []) != source_reasons
            or dimensions.get("costs") is not False
            or dimensions.get("research_process") is not False
            or set(by_dimension.get("costs") or []) != (
                {"provider_cost_unresolved", "competition_cost_eligibility_unavailable"}
                if cost_snapshot_unavailable else {"provider_cost_unresolved"})
            or set(by_dimension.get("research_process") or []) != (
                reasons - {"competition_cost_eligibility_unavailable"})
            or (cost_snapshot_unavailable !=
                ("competition_cost_eligibility_unavailable" in reasons))
            or ("response_privacy_nonexact" in reasons
                and not _documented_selected_privacy_gap(example, run_id))
            or training_thinking._selected_input_output_binding(
                example, allow_selected_cost_unavailable=cost_snapshot_unavailable) is None
            or not _all_model_routes_verified(example, run_id)):
        return False
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if (not training_thinking._provider_capture_complete(example, run_id)
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
            permitted = call_allowed | ({"full_response_unavailable"} if (
                new_policy and training_thinking._verified_unavailable_upstream_failure(example, call)
                and training_thinking.verified_failed_attempt(example, call)) else set())
            if not gaps <= permitted:
                return False
            if "response_privacy_nonexact" not in reasons and (
                "credential" in (call.get("response_privacy_redactions") or [])
                or any("credential" in (receipt.get("privacy_redactions") or [])
                       for receipt in call.get("response_receipts") or [])):
                return False
            if call in cost_calls and (
                call.get("accounting_state") != "uncertain"
                or call.get("actual_microusd") is not None
                or type(call.get("terminal_entry_id")) is not int
                or call["terminal_entry_id"] < 1):
                return False
            if training_thinking.verified_recovered_responses_retry(example, call):
                continue
            if (call in cost_calls and new_policy
                    and training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                    and training_thinking.verified_failed_attempt(example, call)):
                # A host-written bridge failure can have no upstream response.
                # Reuse its exact existing proof; an unknown bill makes this
                # whole ICP incomplete, never free or a strict training target.
                evidence = training_steps._host_result(example, call, allow_worker_error=True)
                if ("response" in call or "response_sha256" in call) and (
                    not isinstance(call.get("response"), Mapping)
                    or contracts.document_hash(call["response"]) != call.get("response_sha256")
                    or call["response"] != evidence["response"]):
                    return False
                continue
            if (call.get("operation_id") in reasoning_capture.OPERATIONS
                    and not (new_policy and training_thinking.verified_failed_attempt(example, call))):
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
                    if not privacy_gap and not truncated and not _verified_provider_output_fault(example, call):
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
                    or (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                        and provenance.get("agent_receipt") != "reported_exact_body_received")
                    or provenance.get("delivery_to_worker") != "socket_written"):
                return False
            if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
                evidence = training_steps._host_result(example, call, allow_privacy=True)
                if evidence["response"] != response:
                    return False
            else:
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


def _selected_privacy_cascade_verified(example, run_id):
    """Count a documented redaction as an incomplete ICP, never exact training."""
    example, _ = training_thinking.provider_visible_view(example)
    proof = example.get("completeness") or {}
    reasons = set(proof.get("incomplete_reasons") or [])
    allowed = {"context_privacy_nonexact", "request_privacy_nonexact", "response_privacy_nonexact"}
    by_dimension = proof.get("reasons_by_dimension") or {}
    dimensions = proof.get("dimensions") or {}
    if (not reasons or not reasons <= allowed
            or not {"request_privacy_nonexact", "response_privacy_nonexact"} & reasons
            or any(dimensions.get(name) is not True or by_dimension.get(name) != []
                   for name in ("context", "source_identity", "exact_inputs", "input_output_label", "costs"))
            or dimensions.get("research_process") is not False
            or set(by_dimension.get("research_process") or []) != reasons
            or ("context_privacy_nonexact" in reasons) !=
               ("credential" in ((example.get("privacy") or {}).get("context_redactions") or []))
            or training_thinking._selected_input_output_binding(example) is None
            or not _all_model_routes_verified(example, run_id)):
        return False
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if (not training_thinking._provider_capture_complete(example, run_id)
            or artifact.get("payload_sha256") != contracts.document_hash(payload)
            or any(training_thinking._required_observation_redacted(example, row, agent=field == "agent_observations")
                   for field in ("observations", "agent_observations")
                   for row in payload.get(field) or [])):
        return False
    selected = [call for call in example.get("provider_calls") or [] if call.get("run_id") == run_id]
    selected_privacy = set()
    attributed = set()
    try:
        for call in example.get("provider_calls") or []:
            gaps = set(call.get("evidence_gaps") or [])
            if call.get("run_id") == run_id:
                if not gaps <= allowed - {"context_privacy_nonexact"}:
                    return False
                selected_privacy.update(gaps)
            elif gaps:
                if (not any(row.get("run_id") == call.get("run_id") and row.get("kind") == "score"
                            for row in example.get("attempts") or [])
                        or not gaps <= allowed - {"context_privacy_nonexact"}):
                    return False
            if "request_privacy_nonexact" in gaps:
                capture = call.get("request_capture") or {}
                if (capture.get("redacted") is not True
                        or "credential" not in (capture.get("redactions") or [])):
                    return False
            if "response_privacy_nonexact" in gaps and not (
                    "credential" in (call.get("response_privacy_redactions") or [])
                    and (call.get("run_id") != run_id or any(
                        "credential" in (row.get("privacy_redactions") or [])
                        for row in call.get("response_receipts") or []))):
                return False
            attributed.update(gaps)
        if not selected_privacy or attributed | (
                {"context_privacy_nonexact"} if "context_privacy_nonexact" in reasons else set()) != reasons:
            return False
        for call in selected:
            gaps = set(call.get("evidence_gaps") or [])
            if training_thinking.verified_recovered_responses_retry(example, call):
                continue
            if call.get("accounting_state") == "refusal" and training_thinking.verified_failed_attempt(example, call):
                continue
            if call.get("accounting_state") != "settlement" or type(call.get("actual_microusd")) is not int:
                return False
            if call.get("operation_id") in reasoning_capture.OPERATIONS:
                try:
                    training_steps._candidate(example, call)
                except ValueError as exc:
                    if str(exc) not in gaps and not _verified_provider_output_fault(example, call):
                        return False
            request = call.get("request") or {}
            response = call.get("response")
            provenance = call.get("response_provenance") or {}
            capture = call.get("request_capture") or {}
            redacted_request = "request_privacy_nonexact" in gaps
            if (not isinstance(request, Mapping)
                    or request.get("operation_id") != call.get("operation_id")
                    or (request.get("effective_operation_id") is not None
                        and request["effective_operation_id"] != call.get("operation_id"))
                    or request.get("request_hash") != call.get("request_hash")
                    or (not redacted_request and (not isinstance(request.get("hashed_parameters"), Mapping)
                        or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")))
                    or (redacted_request and ((request.get("privacy") or {}).get("redacted") is not True
                        or capture.get("status") not in {"stored", "referenced"}
                        or not isinstance(capture.get("sha256"), str) or not capture["sha256"]
                        or not isinstance(capture.get("ref"), str) or not capture["ref"]))
                    or (not redacted_request and capture.get("redacted") is True)
                    or not isinstance(response, Mapping)
                    or contracts.document_hash(response) != call.get("response_sha256")
                    or (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                        and (provenance.get("agent_receipt") != "reported_exact_body_received"
                             or provenance.get("selected_response_acknowledged") is not True))
                    or provenance.get("delivery_to_worker") != "socket_written"):
                return False
            if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
                evidence = training_steps._host_result(example, call, allow_privacy=True)
                if evidence["response"] != response:
                    return False
            else:
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
                                 and ref.get("payload_sha256") == artifact["payload_sha256"] for ref in refs)]
                receipts = [row for row in call.get("response_receipts") or []
                            if row.get("entry_id") in (provenance.get("worker_received_receipt_entry_ids") or [])
                            and row.get("response_sha256") == call.get("response_sha256")]
                if (len(writes) != 1 or not reads or not receipts
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


def _transient_artifact_gap_verified(example, submission, artifacts):
    """Bind retryable object gaps without excusing any retained evidence error.

    The service supplies only saved references which exhausted retryable
    storage reads. Missing objects, access denial and malformed bytes never
    produce this marker. The completed export still checks all other paths.
    """
    example, _ = training_thinking.provider_visible_view(example)
    if not isinstance(artifacts, Mapping) or not artifacts or not set(artifacts) <= {
            "source", "benchmark", "input", "judge_input", "judge_output"}:
        return False
    try:
        run_id, round_id = example["run_id"], example["round_id"]
        if any(not isinstance(ref, str) or not ref for ref in artifacts.values()):
            return False
        attempts = {row["run_id"]: row for row in example.get("attempts") or []}
        expected = {name: set() for name in (
            "context", "source_identity", "exact_inputs", "input_output_label", "costs", "research_process")}
        missing_inputs = set()
        if "source" in artifacts:
            if (artifacts["source"] != "arena/%s/sources/%s.tar.gz" % (round_id, submission["submission_id"])
                    or example.get("source") is not None):
                return False
            expected["source_identity"].add("source_unverified")
        if "benchmark" in artifacts:
            if example.get("icp") is not None:
                return False
            expected["context"].add("frozen_icp_unavailable")
            expected["input_output_label"].add("frozen_context_unverified")
        for kind in ("input", "judge_input"):
            if kind not in artifacts:
                continue
            matching = [key for key, row in attempts.items()
                        if artifacts[kind] == "private/arena-training/runs/%s/leases/%s/input.json.gz" % (
                            key, row.get("lease_generation"))
                        and ((kind == "input" and key == run_id and row.get("kind") == "execute")
                             or (kind == "judge_input" and row.get("kind") == "score"))]
            if len(matching) != 1 or matching[0] in (example.get("input_snapshots") or {}):
                return False
            missing_inputs.add(matching[0])
            expected["exact_inputs"].add("runtime_input_snapshot_missing")
            expected["research_process"].add("runtime_input_snapshot_missing")
        if "judge_output" in artifacts:
            if (example.get("judgment") is not None or example.get("score_run_id") is not None
                    or not any(row.get("kind") == "score" and row.get("status") == "accepted"
                               and artifacts["judge_output"] == "arena/%s/outputs/%s.json" % (round_id, key)
                               for key, row in attempts.items())):
                return False
            expected["input_output_label"].add("judge_evidence_unverified")
            expected["research_process"].add("judge_evidence_unverified")
        proof = example.get("completeness") or {}
        by_dimension = proof.get("reasons_by_dimension") or {}
        dimensions = proof.get("dimensions") or {}
        reasons = set().union(*expected.values())
        # Source bytes can be unavailable at the same time that a selected
        # provider bill is unsettled. Prove the source marker above, then
        # reuse the exact cost/receipt verifier without rewriting raw proof.
        # Missing input or judgment evidence cannot use this exception.
        if "credential" in ((example.get("privacy") or {}).get("context_redactions") or []):
            return False
        if (set(artifacts) == {"source"} and proof.get("complete") is False
                and all(not call.get("evidence_gaps") for call in example.get("provider_calls") or []
                        if call.get("run_id") != run_id)
                and _selected_cost_gap_verified(example, run_id, unavailable_source=True)):
            return True
        if (proof.get("complete") is not False or set(proof.get("incomplete_reasons") or []) != reasons
                or any(set(by_dimension.get(key) or []) != value
                       or dimensions.get(key) is not (not value) for key, value in expected.items())
                or "credential" in ((example.get("privacy") or {}).get("context_redactions") or [])):
            return False
        output = example.get("output")
        if not training_thinking.verified_output_projection(example):
            return False
        snapshots = example.get("input_snapshots") or {}
        if run_id not in snapshots and run_id not in missing_inputs:
            return False
        for snapshot in snapshots.values():
            document = snapshot.get("input")
            if (not isinstance(document, Mapping)
                    or snapshot.get("input_sha256") != contracts.document_hash(document)
                    or (snapshot.get("privacy") or {}).get("redacted") is not False):
                return False
        # Every retained provider call still has to prove its own exact data.
        registry = (example.get("configuration") or {}).get("reasoning_route_registry")
        for call in example.get("provider_calls") or []:
            if call.get("evidence_gaps"):
                return False
            if call.get("run_id") != run_id:
                continue
            if call.get("accounting_state") == "refusal" and training_thinking.verified_failed_attempt(example, call):
                continue
            if (call.get("accounting_state") != "settlement"
                    or type(call.get("actual_microusd")) is not int):
                return False
            if call.get("operation_id") in reasoning_capture.OPERATIONS:
                if (training_coverage.MISSING_REASONING_SETTING in (example.get("configuration") or {})
                        and training_thinking.verified_failed_attempt(example, call)):
                    continue
                try:
                    training_steps._candidate(example, call)
                except ValueError as exc:
                    state = training_thinking.attempt_coverage(call, example.get("role", "miner"), registry)
                    if (not _verified_provider_output_fault(example, call)
                            and (str(exc) != "response_not_complete" or state["returned"] != "truncated"
                                 or not state["route_reviewed"] or not state["provider_route_verified"])):
                        return False
            else:
                # The exporter validated the graph. Recheck retained request and
                # response hashes so an availability marker cannot hide tamper.
                request = call.get("request") or {}
                response = call.get("response")
                if (request.get("operation_id") != call.get("operation_id")
                        or request.get("request_hash") != call.get("request_hash")
                        or not isinstance(request.get("hashed_parameters"), Mapping)
                        or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")
                        or not isinstance(response, Mapping)
                        or contracts.document_hash(response) != call.get("response_sha256")
                        or (call.get("request_capture") or {}).get("redacted")
                        or call.get("response_privacy_redactions")):
                    return False
                if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
                    evidence = training_steps._host_result(example, call)
                    if evidence["response"] != response:
                        return False
                else:
                    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
                    payload = artifact.get("payload") or {}
                    payload_hash = contracts.document_hash(payload)
                    provenance = call.get("response_provenance") or {}
                    worker_ref = provenance.get("worker_receipt_ref") or {}
                    if (artifact.get("payload_sha256") != payload_hash
                            or provenance.get("agent_receipt") != "reported_exact_body_received"
                            or provenance.get("delivery_to_worker") != "socket_written"
                            or worker_ref.get("run_id") != run_id
                            or worker_ref.get("payload_sha256") != payload_hash):
                        return False
                    writes = [row for row in payload.get("response_receipts") or []
                              if row.get("ordinal") == worker_ref.get("ordinal")
                              and row.get("call_identity") == call.get("call_identity")
                              and row.get("operation_id") == call.get("operation_id")
                              and row.get("action_sequence") == call.get("action_sequence")
                              and row.get("write_state") == "socket_written"]
                    refs = provenance.get("agent_receipt_refs") or []
                    reads = [row for row in payload.get("agent_observations") or []
                             if row.get("kind") == "response_received" and any(
                                 ref.get("id") == row.get("id") and ref.get("run_id") == run_id
                                 and ref.get("payload_sha256") == payload_hash for ref in refs)]
                    receipts = [row for row in call.get("response_receipts") or []
                                if row.get("entry_id") in (provenance.get("worker_received_receipt_entry_ids") or [])
                                and row.get("response_sha256") == call.get("response_sha256")
                                and not row.get("privacy_redactions")]
                    if (len(writes) != 1 or not reads or not receipts
                            or any((row.get("privacy") or {}).get("redacted") is not False for row in reads)
                            or not all(any(
                                (read.get("data") or {}).get("call_identity") == call.get("call_identity")
                                and (read.get("data") or {}).get("status") == (receipt.get("metadata") or {}).get("status")
                                and all((read.get("data") or {}).get(key) == writes[0].get(key)
                                        for key in ("status", "body_sha256", "body_bytes"))
                                and writes[0].get("body_sha256") == receipt.get("wire_body_sha256")
                                and writes[0].get("body_bytes") == receipt.get("wire_body_bytes")
                                for receipt in receipts) for read in reads)):
                        return False
        return True
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def _retained_delivery_evidence_exact(call):
    """Check every retained result and API route, including unselected receipts."""
    if not training_steps._outbound_request_exact(call):
        return False
    lifecycle_ids = {row.get("entry_id") for row in call.get("lifecycle") or []}
    permitted = {"hidden_reasoning"} if call.get("operation_id") in reasoning_capture.OPERATIONS else set()
    receipts = call.get("response_receipts") or []
    if (not receipts or len({row.get("entry_id") for row in receipts}) != len(receipts)):
        return False
    selected_receipts = [row for row in receipts if row.get("entry_id") == training_export.response_entry_id(call)]
    if (len(selected_receipts) != 1
            or (selected_receipts[0].get("metadata") or {}).get("status") != call.get("response_status")):
        return False
    for receipt in receipts:
        if (type(receipt.get("entry_id")) is not int or receipt["entry_id"] not in lifecycle_ids
                or receipt.get("response_unavailable")
                or not set(receipt.get("privacy_redactions") or []) <= permitted
                or not training_steps._exact(training_steps._receipt_response(call, receipt), response=True)
                or type((receipt.get("metadata") or {}).get("status")) is not int
                or not 100 <= receipt["metadata"]["status"] <= 599
                or type(receipt.get("wire_body_bytes")) is not int or receipt["wire_body_bytes"] < 0
                or not isinstance(receipt.get("wire_body_sha256"), str)
                or contracts.SHA256_RE.fullmatch(receipt["wire_body_sha256"]) is None):
            return False
    upstream = call.get("upstream_response_receipts") or []
    if len({row.get("entry_id") for row in upstream}) != len(upstream):
        return False
    for receipt in upstream:
        value = receipt.get("response")
        if (receipt.get("privacy_redactions") or not training_steps._exact(value, response=True)
                or receipt.get("provenance") != "private_upstream_response_receipt"
                or receipt.get("delivery_to_worker") != "not_proved"
                or type(receipt.get("entry_id")) is not int
                or receipt["entry_id"] not in lifecycle_ids
                or type(receipt.get("status")) is not int or not 100 <= receipt["status"] <= 599
                or receipt.get("body_sha256") != receipt.get("wire_body_sha256")
                or type(receipt.get("body_bytes")) is not int or receipt["body_bytes"] < 0
                or receipt["body_bytes"] != receipt.get("wire_body_bytes")
                or not isinstance(receipt.get("body_sha256"), str)
                or contracts.SHA256_RE.fullmatch(receipt["body_sha256"]) is None
                or receipt.get("response_sha256") != contracts.document_hash(value)):
            return False
    return True


def _failed_delivery_call_verified(example, call):
    """Prove an exact settled result without claiming delivery or a target."""
    if ((call.get("response_provenance") or {}).get("delivery_to_worker") != "socket_write_failed"
            or call.get("run_id") != example.get("run_id") or call.get("evidence_gaps")
            or call.get("accounting_state") != "settlement"
            or type(call.get("actual_microusd")) is not int or call["actual_microusd"] < 0
            or not training_thinking._terminal_accounting_matches(call, "settlement", call["actual_microusd"])
            or not _retained_delivery_evidence_exact(call)):
        return False
    evidence = training_steps._failed_host_result(example, call)
    provenance = call.get("response_provenance") or {}
    entry_ids = provenance.get("worker_received_receipt_entry_ids") or []
    if (provenance.get("worker_body_source") != "ledger_terminal"
            or provenance.get("worker_body_exact_in_artifact") is not True
            or len(entry_ids) != len(set(entry_ids))
            or set(entry_ids) != {row["entry_id"] for row in evidence["receipts"]}):
        return False
    response = call.get("response")
    if (not isinstance(response, Mapping) or response != evidence["response"]
            or contracts.document_hash(response) != call.get("response_sha256")
            or call.get("response_status") != evidence["write"]["status"]
            or evidence["write"]["status"] >= 400 and not call.get("upstream_response_receipts")):
        return False
    if call.get("operation_id") in reasoning_capture.OPERATIONS:
        op, body, tools = training_steps._model_request(example, call)
        registry = (example.get("configuration") or {}).get("reasoning_route_registry")
        if (not training_thinking._policy(call, registry)
                or not (training_thinking._legacy_tested_controls_match(call)
                        or training_thinking._captured_reasoning_controls_match(call, registry))):
            return False
        if evidence["write"]["status"] < 400:
            state = training_thinking.attempt_coverage(call, example.get("role", "miner"), registry)
            if not state["route_reviewed"] or not state["provider_route_verified"]:
                return False
        if op == "openrouter.chat":
            training_steps._chat_messages(body.get("messages"), allow_reasoning_history=True,
                                           allow_tool_error_history=True, tools=tools)
        else:
            training_steps._responses_input(body.get("input"), tools=tools,
                max_tool_calls=body.get("max_tool_calls"), allow_reasoning_history=True,
                allow_tool_error_history=True)
    return True


def _failed_delivery_gap_verified(example):
    """Proven failed writes or undispatched requests consume one coverage slot."""
    example, _ = training_thinking.provider_visible_view(example)
    if not training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
        return None
    run_id = example.get("run_id")
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    try:
        if (set(artifact) != {"schema_version", "run_id", "kind", "lease_generation",
                              "claim_request_hash", "payload", "payload_sha256"}
                or artifact.get("schema_version") != training_observations.SCHEMA_VERSION
                or artifact.get("run_id") != run_id or artifact.get("kind") != "execute"
                or type(artifact.get("lease_generation")) is not int or artifact["lease_generation"] < 1
                or not isinstance(artifact.get("claim_request_hash"), str)
                or contracts.SHA256_RE.fullmatch(artifact["claim_request_hash"]) is None
                or artifact.get("payload_sha256") != contracts.document_hash(payload)
                or payload.get("complete") is not False
                or payload.get("failures") not in (["capture_incomplete"],
                    ["capture_incomplete", training_observations.OPTIONAL_CAPTURE_INCOMPLETE])
                or training_observations._validate_payload(payload) != payload
                or any(payload[count] != len(payload[rows]) for count, rows in (
                    ("observation_count", "observations"), ("response_receipt_count", "response_receipts"),
                    ("agent_observation_count", "agent_observations")))
                or payload["local_refusal_count"] != sum(row["kind"] == "local_refusal" for row in payload["observations"])
                or example.get("partial_worker_observation_artifacts")):
            return None
        calls = [call for call in example.get("provider_calls") or [] if call.get("run_id") == run_id]
        undispatched = [call for call in calls if training_thinking.verified_undispatched_transport(example, call)]
        delivered = [call for call in calls if call not in undispatched]
        receipts = payload["response_receipts"]
        identities = [call["call_identity"] for call in calls]
        if (not any(call.get("operation_id") in reasoning_capture.OPERATIONS for call in calls)
                or len(identities) != len(set(identities))
                or len(receipts) != len(delivered)
                or {row["call_identity"] for row in receipts} != {call["call_identity"] for call in delivered}
                or any(row["write_state"] not in {"socket_written", "socket_write_failed"} for row in receipts)
                or not undispatched and not any(row["write_state"] == "socket_write_failed" for row in receipts)):
            return None
        # The failed-write proof explains only the generic flag. Every
        # optional failure needs the same independent proof as optional-only
        # capture; unknown or untagged runtime failures remain hard.
        optional = training_observations.OPTIONAL_CAPTURE_INCOMPLETE in payload["failures"]
        failures = training_observations._capture_failure_rows(payload)
        if (bool(failures) is not optional
                or not all(training_observations.optional_capture_failure(row) for row in failures)):
            return None
        for call in delivered:
            if (call.get("accounting_state") != "settlement"
                    or type(call.get("actual_microusd")) is not int or call["actual_microusd"] < 0
                    or not _retained_delivery_evidence_exact(call)
                    or not training_thinking._terminal_accounting_matches(call, "settlement", call["actual_microusd"])):
                return None
        reasons = set((example.get("completeness") or {}).get("incomplete_reasons") or [])
        return training_thinking._attributed_score_only_gaps(
            example, reasons, failed_delivery_proof=_failed_delivery_call_verified,
            undispatched_proof=training_thinking.verified_undispatched_transport if undispatched else None)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


def assess_capture(submission: Mapping, examples_by_run: Mapping[str, Mapping | None],
                   selected_run_ids: Sequence[str], *, enabled: bool,
                   minimum_coverage_percent: int = 100,
                   maximum_missing_reasoning_percent: int = 0,
                   unavailable_outputs: Mapping[str, str] | None = None,
                   unavailable_artifacts: Mapping[str, Mapping[str, str]] | None = None) -> dict:
    """Check each selected final execute run against its persisted raw export.

    A missing export or malformed record is an ineligible result, never an
    execution error. The caller chooses final runs; this helper cannot promote
    an omitted run or use a score run as a substitute.
    """
    if not enabled:
        return _result([], enabled=0)
    if type(maximum_missing_reasoning_percent) is not int or not 0 <= maximum_missing_reasoning_percent <= 100:
        raise ValueError("invalid_maximum_missing_reasoning_percent")
    selected = list(selected_run_ids)
    required = (training_coverage.required_count(len(selected), minimum_coverage_percent)
                if selected else 0)
    counts = {"enabled": 1, "selected_runs": len(selected), "checked_runs": 0,
              "model_calls": 0, "full_thinking_calls": 0, "tool_only_empty_calls": 0,
              "required_complete_runs": required, "complete_runs": 0,
              "incomplete_runs": 0, "minimum_coverage_percent": minimum_coverage_percent}
    reasoning_totals = {"completed_untruncated_verified_reasoning_tokens": 0,
                        "missing_reasoning_tokens": 0, "missing_reasoning_calls": 0,
                        "unknown_reasoning_usage_calls": 0, "truncated_model_calls": 0,
                        "verified_failed_model_attempts": 0}
    reasoning_by_run = []
    policy_stats_enabled = False
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
        failed_delivery_verified = False
        reasoning_counts = {key: 0 for key in reasoning_totals}
        example = examples_by_run.get(run_id)
        if not isinstance(example, Mapping):
            run_reasons.append("selected_run_export_missing")
            counts["incomplete_runs"] += 1
            if minimum_coverage_percent == 100:
                reasons.extend(run_reasons)
            continue
        counts["checked_runs"] += 1
        new_policy = (isinstance(example.get("configuration"), Mapping)
                      and training_coverage.MISSING_REASONING_SETTING in example["configuration"])
        policy_stats_enabled |= new_policy
        transient_verified = _transient_artifact_gap_verified(
            example, submission, (unavailable_artifacts or {}).get(run_id))
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
                    or not ((isinstance(source.get("sha256"), str)
                             and contracts.SHA256_RE.fullmatch(source["sha256"]) is not None
                             and (source["sha256"] == source_digest or baseline_source_ok))
                            or transient_verified and "source" in (unavailable_artifacts or {}).get(run_id, {}))
                    or (example.get("configuration") or {}).get("reasoning_capture_policy") != reasoning_capture.POLICY
                    or training_coverage.missing_reasoning_percent_from_configuration(
                        example.get("configuration") or {}) != maximum_missing_reasoning_percent):
                run_reasons.append("selected_run_identity_unverified")
                continue
            # A self-contradictory completeness flag cannot hide an unsettled
            # selected call. The narrow 80% verifier below must prove it.
            if any(call.get("run_id") == run_id
                    and not (new_policy and call.get("accounting_state") == "refusal"
                             and training_thinking.verified_failed_attempt(example, call))
                    and not (new_policy and training_thinking.verified_undispatched_transport(example, call)) and (
                    call.get("accounting_state") != "settlement"
                    or type(call.get("actual_microusd")) is not int)
                    for call in example.get("provider_calls") or []):
                run_reasons.append("trajectory_evidence_incomplete")
                continue
            ancillary_gaps = training_thinking.selected_research_evidence(
                example, target_omission_proof=_verified_provider_output_fault if new_policy else None)
            if transient_verified:
                ancillary_gaps = frozenset((example.get("completeness") or {}).get("incomplete_reasons") or [])
                run_reasons.append("selected_artifact_temporarily_unavailable")
            if ancillary_gaps is None:
                if new_policy and minimum_coverage_percent < 100:
                    delivery_gaps = _failed_delivery_gap_verified(example)
                    if delivery_gaps is not None:
                        failed_delivery_verified = True
                        run_reasons.append("worker_delivery_failed")
                        continue
                run_reasons.append("trajectory_evidence_incomplete")
                continue
            if (not isinstance(example.get("output"), Mapping)
                    or not isinstance(example.get("input_snapshots"), Mapping)
                    or run_id not in example["input_snapshots"] and not transient_verified):
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
                if new_policy and training_thinking.verified_failed_attempt(example, call):
                    reasoning_counts["verified_failed_model_attempts"] += 1
                    continue
                try:
                    candidate = training_steps._candidate(example, call)
                except ValueError as exc:
                    if new_policy and _verified_provider_output_fault(example, call):
                        run_reasons.append("provider_output_untrainable")
                        continue
                    if new_policy and training_thinking.verified_cap_truncated_target(example, call):
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
                else:
                    # The loop above still proves each candidate or exact
                    # length stop. Mixed verified gaps consume the same ICP;
                    # route, receipt and other evidence failures remain hard.
                    soft = ({"provider_response_truncated", "provider_output_untrainable", "full_thinking_unavailable",
                             "full_thinking_call_missing", "missing_reasoning_share_exceeded"}
                            if _only_benign_thinking_gaps(
                                example, run_id, new_policy=new_policy)
                            else set())
                    if unavailable_ref is not None:
                        soft.add("accepted_output_unavailable")
                    if failed_delivery_verified:
                        soft.add("worker_delivery_failed")
                    if transient_verified:
                        soft.add("selected_artifact_temporarily_unavailable")
                        # A composed source/cost proof already checked every
                        # retained input, route and receipt before this gate.
                        soft.add("trajectory_evidence_incomplete")
                    if set(run_reasons) - {"accepted_output_unavailable"} == {
                            "trajectory_evidence_incomplete"}:
                        proof_reasons = set((example.get("completeness") or {}).get("incomplete_reasons") or [])
                        if (proof_reasons and proof_reasons <= {"response_privacy_nonexact"}
                                and _documented_selected_privacy_gap(example, run_id)
                                and _all_model_routes_verified(example, run_id)):
                            soft.add("trajectory_evidence_incomplete")
                        if _selected_cost_gap_verified(example, run_id):
                            soft.add("trajectory_evidence_incomplete")
                        if _selected_privacy_cascade_verified(example, run_id):
                            soft.add("trajectory_evidence_incomplete")
                    reasons.extend(reason for reason in run_reasons if reason not in soft)
            else:
                counts["complete_runs"] += 1
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
