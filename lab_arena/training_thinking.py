"""Read-only coverage and strict generated-thinking candidates from Arena exports.

The saved provider result and host-delivery proof remain in training_steps. A request
annotation establishes route semantics; returned text alone never does.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from lab_arena import contracts, operations, reasoning_capture, training_capture, training_coverage, training_dataset, training_export, training_steps

SCHEMA_VERSION = "leadpoet.lab_arena.full_thinking_dataset.v1"
RECORD_SCHEMA_VERSION = "leadpoet.lab_arena.full_thinking_step.v1"
POLICY = "full_generated_thinking_v1"
PARSER_VERSION = "arena_openrouter_reasoning_v1"
_ROUTE_FIELDS = ("model_id", "model_version", "provider_route", "api",
                 "parser_version", "probe_id", "probe_manifest_sha256", "registry_sha256")
def _policy(call: Mapping[str, Any], registry: Any = None) -> bool:
    annotation = call.get("reasoning_capture") or {}
    model = (((call.get("request") or {}).get("outbound") or {}).get("body") or {}).get("model")
    operation = call.get("operation_id")
    allowed_fields = ({"output.content.reasoning_text"} if operation == "openrouter.responses" else
                      {"reasoning", "reasoning_content", "reasoning_details.text"} if operation == "openrouter.chat" else set())
    mode = annotation.get("mode") if isinstance(annotation, Mapping) else None
    mode_valid = (isinstance(mode, Mapping) and (
        set(mode) == {"effort"} and isinstance(mode["effort"], str)
        and mode["effort"] in {"low", "medium", "high", "xhigh", "max"}
        or operation == "openrouter.chat" and set(mode) == {"budget_tokens"}
        and type(mode["budget_tokens"]) is int and mode["budget_tokens"] >= 1024
        or operation == "openrouter.chat" and mode == {"enabled": True}))
    if not (isinstance(annotation, Mapping) and annotation.get("policy") == POLICY
            and annotation.get("requested") is True and annotation.get("reviewed_route") is True
            and annotation.get("semantic_class") == "full_generated_thinking"
            and all(isinstance(annotation.get(key), str) and annotation[key] for key in _ROUTE_FIELDS)
            and all(re.fullmatch(r"[0-9a-f]{64}", annotation[key]) for key in
                    ("probe_manifest_sha256", "registry_sha256"))
            and mode_valid and annotation.get("parser_version") == PARSER_VERSION
            and isinstance(annotation.get("full_text_fields"), list)
            and bool(annotation["full_text_fields"])
            and all(isinstance(field, str) for field in annotation["full_text_fields"])
            and len(annotation["full_text_fields"]) == len(set(annotation["full_text_fields"]))
            and all(field in allowed_fields for field in annotation["full_text_fields"])
            and call.get("provider") == "openrouter"
            and annotation.get("model_id") == model
            and annotation.get("api") == call.get("operation_id")):
        return False
    try:
        frozen = reasoning_capture.validate_registry(registry)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    if contracts.document_hash(frozen).split(":", 1)[1] != annotation["registry_sha256"]:
        return False
    matches = [entry for entry in frozen["approved_routes"]
               if entry["model_id"] == model and entry["api"] == operation
               and entry["provider"] == annotation.get("provider_route")]
    if len(matches) != 1:
        return False
    entry = matches[0]
    expected = {"model_version": entry["model_version"], "provider_route": entry["provider"],
                "provider_name": entry["provider_name"], "api": entry["api"],
                "mode": entry["mode"], "full_text_fields": entry["full_text_fields"],
                "parser_version": entry["parser_version"], "context_tokens": entry["context_tokens"],
                "max_output_tokens": entry["max_output_tokens"],
                "price_ceiling": entry["price_ceiling"],
                "probe_id": entry["evidence"]["probe_id"],
                "probe_manifest_sha256": entry["evidence"]["probe_manifest_sha256"]}
    return all(annotation.get(key) == value for key, value in expected.items())


def _observation_matches(observation: Any, annotation: Any) -> bool:
    if not isinstance(observation, Mapping) or not isinstance(annotation, Mapping):
        return False
    expected = {"requested_model": annotation.get("model_id"),
                "requested_provider_route": annotation.get("provider_route"),
                "expected_provider_name": annotation.get("provider_name"),
                "expected_model_version": annotation.get("model_version"),
                "response_model": annotation.get("model_id"),
                "response_provider_name": annotation.get("provider_name"),
                "selected_endpoint_model": annotation.get("model_version"),
                "selected_endpoint_provider_name": annotation.get("provider_name")}
    # Responses can omit the redundant top-level provider. The selected
    # endpoint still has to prove both the exact model and provider identity.
    if (annotation.get("api") == "openrouter.responses"
            and observation.get("response_provider_name") is None):
        expected.pop("response_provider_name")
    return (observation.get("status") == "matched" and
            all(isinstance(value, str) and value and observation.get(key) == value
                for key, value in expected.items()))


def _approved_segments(segments: list[dict], allowed: set[str]) -> bool:
    return bool(segments) and all(
        segment["field"] in allowed and
        all(alias in allowed for alias in segment.get("alias_fields", []))
        for segment in segments)


def full_text_segments(call: Mapping[str, Any], role: str, registry: Any = None) -> tuple[list[dict], dict]:
    """Admit only reviewed full text; keep ancillary summaries and opaque state separate."""
    status = attempt_coverage(call, role, registry)
    segments, _, _ = _segments(call.get("response"), str(call.get("operation_id")))
    if (not status["route_reviewed"] or not status["provider_route_verified"]
            or status["storage"] != "saved"
            or status["returned"] in {"redacted", "truncated", "interrupted",
                                   "storage_failed", "size_limited", "unavailable"}):
        return [], status
    allowed = set(call["reasoning_capture"]["full_text_fields"])
    full = [segment for segment in segments if _approved_segments([segment], allowed)]
    # A summary is ancillary. Other unapproved readable text has no verified
    # full-text route and must not enter the generated-thinking target.
    if any(segment not in full and "summary" not in segment["field"] for segment in segments):
        return [], status
    return full, status


def verified_no_thinking(call: Mapping[str, Any], delivered: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    """Admit an explicit zero-token completed turn without inventing thinking.

    A tool call, a note plus a tool call, and a direct answer use the same
    rule. Missing usage is unknown; positive billed reasoning without text
    is a capture gap. Summaries and opaque evidence do not prove zero work.
    The caller still verifies the exact delivered request and response.
    """
    if (not status["route_reviewed"] or not status["provider_route_verified"]
            or status["storage"] != "saved" or status["returned"] != "requested_empty"
            or status["tokens_reported"] != 0
            or status["segment_count"] or status["contains_summary"] or status["contains_opaque"]):
        return False
    if call.get("operation_id") == "openrouter.chat":
        choices = delivered.get("choices") or []
        return (len(choices) == 1 and isinstance(choices[0], Mapping)
                and choices[0].get("finish_reason") in {"stop", "tool_calls"})
    return call.get("operation_id") == "openrouter.responses" and delivered.get("status") == "completed"


def verified_missing_thinking(status: Mapping[str, Any]) -> bool:
    """Positive reported reasoning with no proved full text, never a target."""
    return (status["route_reviewed"] and status["provider_route_verified"]
            and status["storage"] == "saved"
            and status["returned"] in {"missing_thinking", "summary", "opaque"}
            and type(status["tokens_reported"]) is int and status["tokens_reported"] > 0)


def verified_tool_only_empty(call: Mapping[str, Any], delivered: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    """Keep the historical unknown-usage tool-only exception explicit.

    Unknown usage remains unknown. Only the prior tool-only shape retains its
    compatibility admission; a note or direct answer needs reported zero.
    Positive billed reasoning without text never uses the exception.
    """
    if verified_no_thinking(call, delivered, status):
        return True
    if (not status["route_reviewed"] or not status["provider_route_verified"]
            or status["storage"] != "saved" or status["returned"] != "requested_empty"
            or status["tokens_reported"] is not None):
        return False
    if call.get("operation_id") == "openrouter.chat":
        choices = delivered.get("choices") or []
        return (len(choices) == 1 and choices[0].get("finish_reason") == "tool_calls"
                and bool((choices[0].get("message") or {}).get("tool_calls"))
                and not (choices[0].get("message") or {}).get("content"))
    output = delivered.get("output") or []
    return (call.get("operation_id") == "openrouter.responses" and bool(output)
            and any(item.get("type") == "function_call" for item in output)
            and not any(item.get("type") == "message" for item in output))


def _segments(response: Any, operation: str) -> tuple[list[dict], bool, bool]:
    """Return exact visible segments, summary present, and opaque marker present."""
    segments, summary, opaque = [], False, False
    if not isinstance(response, Mapping):
        return segments, summary, opaque
    if operation == "openrouter.chat":
        for ci, choice in enumerate(response.get("choices") or []):
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if not isinstance(message, Mapping):
                continue
            for field in ("reasoning", "reasoning_content"):
                value = message.get(field)
                if isinstance(value, str) and value.strip():
                    segments.append({"source_path": ["choices", ci, "message", field], "field": field, "text": value})
                elif value not in (None, ""):
                    opaque = True
            if any(message.get(field) not in (None, "", []) for field in ("hidden_reasoning", "encrypted_content")):
                opaque = True
            for ri, item in enumerate(message.get("reasoning_details") or []):
                if not isinstance(item, Mapping):
                    opaque = True
                    continue
                kind = item.get("type")
                if kind in {"reasoning.summary", "summary_text"} or item.get("summary"):
                    summary = True
                    value = item.get("summary")
                    if isinstance(value, str) and value.strip():
                        segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri, "summary"],
                                         "field": "reasoning_details.summary", "text": value})
                    elif isinstance(value, list):
                        for pi, part in enumerate(value):
                            if (isinstance(part, Mapping) and part.get("type") == "summary_text"
                                    and isinstance(part.get("text"), str) and part["text"].strip()):
                                segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri,
                                                                 "summary", pi, "text"],
                                                 "field": "reasoning_details.summary", "text": part["text"]})
                if training_capture.reasoning_kind(item) == "opaque":
                    opaque = True
                    continue
                for field in ("text", "thinking", "content"):
                    value = item.get(field)
                    field_id = "reasoning_details.text" if kind == "reasoning.text" and field == "text" else "reasoning_details.unproved"
                    if isinstance(value, str) and value.strip():
                        segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri, field],
                                         "field": field_id, "text": value})
                    elif isinstance(value, list):
                        for pi, part in enumerate(value):
                            if isinstance(part, Mapping) and isinstance(part.get("text"), str) and part["text"].strip():
                                segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri, field, pi, "text"],
                                                 "field": field_id, "text": part["text"]})
    elif operation == "openrouter.responses":
        for oi, item in enumerate(response.get("output") or []):
            if not isinstance(item, Mapping) or item.get("type") != "reasoning":
                continue
            if item.get("summary"):
                summary = True
                for pi, part in enumerate(item["summary"] if isinstance(item["summary"], list) else []):
                    if (isinstance(part, Mapping) and part.get("type") == "summary_text"
                            and isinstance(part.get("text"), str) and part["text"].strip()):
                        segments.append({"source_path": ["output", oi, "summary", pi, "text"],
                                         "field": "output.summary.summary_text", "text": part["text"]})
            if item.get("encrypted_content") or item.get("redacted_thinking"):
                opaque = True
            for field in ("text", "thinking", "content"):
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    segments.append({"source_path": ["output", oi, field], "field": "output." + field, "text": value})
                elif isinstance(value, list):
                    for pi, part in enumerate(value):
                        if isinstance(part, Mapping) and part.get("type") == "summary_text":
                            summary = True
                        elif isinstance(part, Mapping) and isinstance(part.get("text"), str) and part["text"].strip():
                            segments.append({"source_path": ["output", oi, field, pi, "text"],
                                             "field": "output.content.reasoning_text" if field == "content" and part.get("type") == "reasoning_text" else "output.unproved",
                                             "text": part["text"]})
    # Providers can repeat one generated string in an alias and a typed block.
    # Keep both source locations but train and count the text once.
    unique = []
    by_text = {}
    for segment in segments:
        existing = by_text.get(segment["text"])
        if existing is None:
            segment["source_paths"] = [segment.pop("source_path")]
            by_text[segment["text"]] = segment
            unique.append(segment)
        else:
            existing["source_paths"].append(segment["source_path"])
            if existing["field"] != segment["field"]:
                existing.setdefault("alias_fields", []).append(segment["field"])
    return unique, summary, opaque


def _route_observation(call: Mapping[str, Any]) -> Any:
    matching = [receipt for receipt in call.get("response_receipts") or []
                if receipt.get("response_sha256") == call.get("response_sha256")]
    observations = [(receipt.get("metadata") or {}).get("reasoning_route_observation")
                    for receipt in matching]
    if call.get("route_observation") is not None:
        observations.append(call["route_observation"])
    # Append-only recovery can retain the same delivered body again. Repeated
    # identical proof is one observation; missing or conflicting proof is not.
    if (observations and all(reasoning_capture.valid_route_observation(value) for value in observations)
            and len({contracts.canonical_json(value) for value in observations}) == 1):
        return observations[0]
    return None


def _execution_model_call(example: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    return (call.get("operation_id") in {"openrouter.chat", "openrouter.responses"}
            and call.get("run_id") == example.get("run_id")
            and _role(example, call) in {"miner", "baseline"})


def _reasoning_request_state(call: Mapping[str, Any], annotation: Mapping[str, Any]) -> str:
    """Report what the saved outbound request asked for, without route approval."""
    body = (((call.get("request") or {}).get("outbound") or {}).get("body"))
    if isinstance(body, Mapping):
        control = body.get("reasoning")
        if isinstance(control, Mapping):
            if control.get("exclude") is True or control.get("enabled") is False:
                return "not_requested"
            if (control.get("effort") not in (None, "", "none")
                    or control.get("summary") not in (None, "", "none")
                    or control.get("enabled") is True
                    or any(type(control.get(key)) is int and control[key] > 0
                           for key in ("max_tokens", "budget_tokens"))):
                return "requested"
        if (body.get("include_reasoning") is True
                or body.get("reasoning_effort") not in (None, "", "none")):
            return "requested"
        return "requested" if annotation.get("requested") is True else "not_requested"
    return "requested" if annotation.get("requested") is True else "unknown"


def attempt_coverage(call: Mapping[str, Any], role: str, registry: Any = None) -> dict:
    """Describe one stored provider attempt without asserting model cognition."""
    annotation = call.get("reasoning_capture") or {}
    annotation = annotation if isinstance(annotation, Mapping) else {}
    response = call.get("response")
    segments, summary, opaque = _segments(response, str(call.get("operation_id")))
    policy_verified = _policy(call, registry)
    observation_verified = _observation_matches(_route_observation(call), annotation)
    route_verified = policy_verified and observation_verified
    allowed = set(annotation["full_text_fields"]) if policy_verified else set()
    unproved = bool(segments) and not _approved_segments(segments, allowed)
    redactions = set(call.get("response_privacy_redactions") or [])
    for receipt in call.get("response_receipts") or []:
        redactions.update(receipt.get("privacy_redactions") or [])
    finish = None
    if isinstance(response, Mapping):
        if call.get("operation_id") == "openrouter.chat":
            choices = response.get("choices") if isinstance(response.get("choices"), list) else []
            finish = choices[0].get("finish_reason") if len(choices) == 1 and isinstance(choices[0], Mapping) else None
        else:
            finish = response.get("status")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    output_details = usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, Mapping) else {}
    reported = [value for value in (usage.get("reasoning_tokens"), details.get("reasoning_tokens"),
                                   output_details.get("reasoning_tokens"))
                if type(value) is int and value >= 0]
    # A conflicting zero field cannot conceal positively billed reasoning.
    tokens = max(reported) if reported else None
    capture_reason = (call.get("upstream_response_capture") or {}).get("reason")
    storage = ("saved" if isinstance(response, Mapping) else
               "storage_failed" if capture_reason == "response_capture_failed" else
               "size_limited" if capture_reason == "response_size_limit" else "unavailable")
    request_state = _reasoning_request_state(call, annotation)
    contains_opaque = opaque or "hidden_reasoning" in redactions
    if storage == "storage_failed":
        content_class = "storage_failed"
    elif storage == "size_limited":
        content_class = "size_limited"
    elif not isinstance(response, Mapping):
        content_class = "unavailable"
    elif "credential" in redactions:
        content_class = "redacted"
    elif finish in {"length", "max_tokens", "incomplete"} or response.get("incomplete_details"):
        content_class = "truncated"
    elif (call.get("operation_id") == "openrouter.chat" and finish not in {"stop", "tool_calls"}
          or call.get("operation_id") == "openrouter.responses" and finish != "completed"):
        content_class = "interrupted"
    elif summary:
        content_class = "summary"
    elif unproved:
        content_class = "unproved_text"
    elif segments:
        content_class = "full_thinking" if route_verified and not contains_opaque else "unproved_text"
    elif contains_opaque:
        content_class = "opaque"
    elif tokens is not None and tokens > 0:
        content_class = "missing_thinking"
    else:
        content_class = "requested_empty" if request_state == "requested" else request_state
    return {"run_id": call.get("run_id"), "call_identity": call.get("call_identity"),
            "provider_attempt": call.get("provider_attempt"), "role": role,
            "model_id": annotation.get("model_id") or (((call.get("request") or {}).get("outbound") or {}).get("body") or {}).get("model"),
            "requested_provider_route": annotation.get("provider_route"), "api": call.get("operation_id"),
            "requested": request_state == "requested", "request_state": request_state,
            "route_reviewed": policy_verified,
            "provider_route_verified": observation_verified,
            "returned": content_class, "storage": storage, "finish_reason": finish,
            "characters": sum(len(part["text"]) for part in segments),
            "tokens_reported": tokens, "segment_count": len(segments),
            "contains_summary": summary, "contains_opaque": contains_opaque,
            "redactions": sorted(redactions)}


def _ignored_provider_failure(response, request_body, annotation) -> bool:
    """Prove the observed account-policy routing failure, not arbitrary 404s."""
    error = response.get("error") or {}
    metadata = error.get("metadata") or {}
    routing = response.get("openrouter_metadata") or {}
    provider = request_body.get("provider") or {}
    funnel = metadata.get("routing_funnel")
    endpoints = (routing.get("endpoints") or {}).get("available")
    if (type(error.get("code")) is not int or error["code"] != 404
            or not isinstance(error.get("message"), str)
            or not error["message"].startswith("All providers have been ignored.")
            or metadata.get("failed_routing_step") != "Filter by Ignored Providers"
            or provider.get("only") != [annotation.get("provider_route")]
            or provider.get("allow_fallbacks") is not False
            or routing.get("requested") != annotation.get("model_id")
            or routing.get("strategy") != "direct"
            or not isinstance(funnel, list) or not funnel
            or not isinstance(endpoints, list) or not endpoints
            or any(not isinstance(row, Mapping) or row.get("selected") is not False for row in endpoints)):
        return False
    allowed = [row for row in funnel if isinstance(row, Mapping)
               and row.get("step") == "Filter by Allowed Providers"]
    target = [row for row in endpoints if row.get("model") == annotation.get("model_version")
              and row.get("provider") == annotation.get("provider_name")]
    return (len(allowed) == 1 and funnel[-1] == allowed[0]
            and type(allowed[0].get("endpoint_count")) is int and allowed[0]["endpoint_count"] == 1
            and allowed[0].get("reason") == "not in your request's provider.only preference"
            and len(target) == 1)


def _bound_admission_refusal(example, call) -> bool:
    """Prove one zero-charge SQL admission denial; it has no provider output."""
    configuration = example.get("configuration") or {}
    row = call.get("refusal_receipt") or {}
    doc = row.get("entry_doc") or {}
    saved = doc.get("call") or {}
    lifecycle = call.get("lifecycle") or []
    return (training_coverage.MISSING_REASONING_SETTING in configuration
            and configuration.get("reasoning_capture_policy") == reasoning_capture.POLICY
            and call.get("accounting_state") == "refusal"
            and type(call.get("actual_microusd")) is int and call["actual_microusd"] == 0
            and call.get("reservation_entry_id") is None
            and call.get("terminal_entry_id") is None
            and not call.get("response_receipts") and not call.get("upstream_response_receipts")
            and not any(key in call for key in ("response", "response_sha256", "provider_cost"))
            and row.get("entry_kind") == "refusal"
            and type(row.get("entry_id")) is int and row["entry_id"] == call.get("refusal_entry_id")
            and type(row.get("amount_microusd")) is int and row["amount_microusd"] == 0
            and doc.get("reason") in {"per_icp_quota", "stage_quota", "money_cap"}
            and type(doc.get("requested_microusd")) is int and doc["requested_microusd"] >= 0
            and not row.get("terminal_response")
            and all(row.get(key) == call.get(key) for key in
                    ("run_id", "call_identity", "operation_id", "provider", "funding_source"))
            and row.get("round_id") == example.get("round_id")
            and row.get("submission_id") == example.get("submission_id")
            and all(saved.get(key) == call.get(key) for key in
                    ("request_hash", "base_call_identity", "provider_attempt", "action_sequence", "reasoning_capture"))
            and saved.get("request_capture") == call.get("request_capture")
            and isinstance(saved.get("base_call_identity"), str)
            and type(saved.get("action_sequence")) is int and saved["action_sequence"] >= 0
            and type(saved.get("provider_attempt")) is int and saved["provider_attempt"] >= 1
            and len(lifecycle) == 1 and lifecycle[0].get("entry_kind") == "refusal"
            and lifecycle[0].get("entry_id") == row["entry_id"]
            and lifecycle[0].get("amount_microusd") == 0)


def verified_failed_attempt(example: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    """Prove a delivered retryable error without making it a model target.

    Provider errors need no observed generation route: no generation is
    claimed. The reviewed request, saved failure bytes, host write and agent
    read must agree. Unsettled billing remains a separate eligibility gate.
    """
    try:
        run_id = example["run_id"]
        admission_refusal = call.get("accounting_state") == "refusal"
        if admission_refusal:
            if not _bound_admission_refusal(example, call):
                return False
            try:
                admitted = operations.validate_operation_request(
                    call.get("operation_id"), (call.get("request") or {}).get("admitted_parameters"),
                    allow_full_thinking=True)
            except operations.OperationError:
                return False
            if admitted.get("model") != (((call.get("request") or {}).get("outbound") or {}).get("body") or {}).get("model"):
                return False
        operation = call.get("operation_id")
        registry = (example.get("configuration") or {}).get("reasoning_route_registry")
        request = call.get("request") or {}
        outbound = request.get("outbound") or {}
        body = outbound.get("body")
        if (call.get("run_id") != run_id or call.get("provider") != "openrouter"
                or operation not in reasoning_capture.OPERATIONS
                or not _policy(call, registry)
                or (call.get("request_capture") or {}).get("redacted")
                or not training_steps._exact(request)
                or request.get("operation_id") != operation
                or request.get("request_hash") != call.get("request_hash")
                or not isinstance(request.get("hashed_parameters"), Mapping)
                or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")
                or not isinstance(body, Mapping)
                or contracts.document_hash(body) != outbound.get("body_sha256")
                or outbound.get("method") != "POST" or outbound.get("host") != "openrouter.ai"
                or outbound.get("path") != ("/api/v1/responses" if operation.endswith("responses")
                                              else "/api/v1/chat/completions")
                or (request.get("effective_parameters") or {}).get("model") != body.get("model")
                or not set(call.get("evidence_gaps") or []) <= {"provider_cost_unresolved"}
                or set((example.get("completeness") or {}).get("incomplete_reasons") or []) & {
                    "worker_response_receipt_link_invalid", "agent_response_receipt_link_invalid",
                    "agent_dependency_link_invalid", "runtime_input_context_mismatch"}):
            return False
        if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
            evidence = training_steps._host_result(example, call, allow_worker_error=True)
            status = evidence["write"]["status"]
            if status not in ({402} if admission_refusal else {404, 429, 502, 503, 504}):
                return False
            failure_bodies = [evidence["response"]]
            if admission_refusal:
                return (not evidence["receipts"]
                        and evidence["response"] == {"error": {"code": "budget_refused"}})
        else:
            provenance = call.get("response_provenance") or {}
            artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
            payload = artifact.get("payload") or {}
            payload_hash = contracts.document_hash(payload)
            worker_ref = provenance.get("worker_receipt_ref") or {}
            if (artifact.get("payload_sha256") != payload_hash
                    or provenance.get("agent_receipt") != "reported_exact_body_received"
                    or provenance.get("delivery_to_worker") != "socket_written"
                    or worker_ref.get("run_id") != run_id
                    or worker_ref.get("payload_sha256") != payload_hash):
                return False
            writes = [row for row in payload.get("response_receipts") or []
                      if row.get("ordinal") == worker_ref.get("ordinal")]
            if len(writes) != 1:
                return False
            write = writes[0]
            status = write.get("status")
            if (type(status) is not int or status not in ({402} if admission_refusal else {404, 429, 502, 503, 504})
                    or write.get("write_state") != "socket_written"
                    or write.get("operation_id") != operation
                    or write.get("action_sequence") != call.get("action_sequence")
                    or write.get("call_identity") != call.get("call_identity")
                    or type(write.get("body_bytes")) is not int or write["body_bytes"] < 0
                    or not isinstance(write.get("body_sha256"), str)
                    or contracts.SHA256_RE.fullmatch(write["body_sha256"]) is None):
                return False
            refs = provenance.get("agent_receipt_refs") or []
            reads = [row for row in payload.get("agent_observations") or []
                     if row.get("kind") == "response_received" and any(
                         ref.get("id") == row.get("id") and ref.get("run_id") == run_id
                         and ref.get("payload_sha256") == payload_hash for ref in refs)]
            if (not reads or any((row.get("privacy") or {}).get("redacted") is not False
                                or any((row.get("data") or {}).get(key) != write.get(key)
                                       for key in ("call_identity", "status", "body_sha256", "body_bytes"))
                                for row in reads)):
                return False
            failure_bodies = []
            for receipt in call.get("response_receipts") or []:
                if (receipt.get("entry_id") not in (provenance.get("worker_received_receipt_entry_ids") or [])
                        or (receipt.get("metadata") or {}).get("status") != status
                        or receipt.get("wire_body_sha256") != write["body_sha256"]
                        or receipt.get("wire_body_bytes") != write["body_bytes"]):
                    continue
                if receipt.get("privacy_redactions") or receipt.get("response_unavailable"):
                    return False
                failure_bodies.append(_delivered_response(call, receipt["response_sha256"]))
            if not failure_bodies:
                ref = provenance.get("worker_error_observation_ref") or {}
                if (provenance.get("worker_error_body_acknowledged") is not True
                        or ref.get("run_id") != run_id or ref.get("payload_sha256") != payload_hash
                        or any(receipt.get("wire_body_sha256") for receipt in call.get("response_receipts") or [])):
                    return False
                observations = [row for row in payload.get("observations") or []
                                if row.get("ordinal") == ref.get("ordinal")]
                if len(observations) != 1:
                    return False
                observed = observations[0]
                response = (observed.get("outcome") or {}).get("response") or {}
                raw = base64.b64decode(response["body_b64"], validate=True)
                if ((observed.get("privacy") or {}).get("redacted") is not False
                        or any(observed.get(key) != call.get(key)
                               for key in ("call_identity", "operation_id", "action_sequence"))
                        or response.get("status") != status or len(raw) != write["body_bytes"]
                        or contracts.hash_bytes(raw) != write["body_sha256"]):
                    return False
                if admission_refusal and (observed.get("request") or {}).get("admitted_parameters") != request.get("admitted_parameters"):
                    return False
                failure_bodies.append(json.loads(raw))
            if admission_refusal:
                return (provenance.get("worker_body_source") == "worker_error_observation"
                        and bool(failure_bodies)
                        and all(value == {"error": {"code": "budget_refused"}} for value in failure_bodies))
        return all(isinstance(value, Mapping) and bool(value.get("error"))
                   and not value.get("choices") and not value.get("output")
                   and (status != 404 or _ignored_provider_failure(
                       value, body, call.get("reasoning_capture") or {}))
                   for value in failure_bodies)
    except (KeyError, TypeError, ValueError, AttributeError, binascii.Error):
        return False


def verified_cap_truncated_target(example, call) -> bool:
    """Recognize exact cap-hit tool JSON ending at EOF, never repair a target.

    Some providers report completed despite exhausting the requested output
    limit. All normal request, host, route, output-shape and declaration checks
    must pass; only the unfinished JSON argument may differ from a valid target.
    """
    if not training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
        return False
    try:
        body = call["request"]["outbound"]["body"]
        operation = call.get("operation_id")
        cap_key, usage_key = (("max_tokens", "completion_tokens") if operation == "openrouter.chat"
                              else ("max_output_tokens", "output_tokens"))
        cap = body.get(cap_key)
        # Cheap rejection only: a possible cap hit still needs the full exact
        # selected-host proof below, including older delivered receipt bodies.
        bodies = [call.get("response")] + [row.get("response") for row in call.get("response_receipts") or []]
        if not any(isinstance(value, Mapping) and type((value.get("usage") or {}).get(usage_key)) is int
                   and (value.get("usage") or {})[usage_key] == cap for value in bodies):
            return False
        evidence = training_steps._host_result(example, call)
        response = evidence["response"]
        usage = response.get("usage") or {}
        checked = dict(call, response=response, response_sha256=contracts.document_hash(response))
        state = attempt_coverage(checked, _role(example, call),
                                 (example.get("configuration") or {}).get("reasoning_route_registry"))
        if (operation not in reasoning_capture.OPERATIONS or evidence["write"]["status"] != 200
                or type(cap) is not int or cap < 1
                or cap != (call.get("reasoning_capture") or {}).get("max_output_tokens")
                or type(usage.get(usage_key)) is not int or usage[usage_key] != cap
                or not state["requested"] or not state["route_reviewed"]
                or not state["provider_route_verified"] or state["redactions"]):
            return False
        try:
            training_steps._candidate(example, call)
        except ValueError as exc:
            if str(exc) != "tool_call_shape_unsupported":
                return False
        else:
            return False  # A valid generation at its cap remains a valid target.
        unfinished = []

        def argument_shape(value):
            if training_steps._arguments(value):
                return True
            if not isinstance(value, str) or not value.lstrip().startswith("{"):
                return False
            try:
                json.loads(value, parse_constant=training_steps._reject_constant)
            except json.JSONDecodeError as exc:
                if exc.pos == len(value):
                    unfinished.append(value)
                    return True
            except (ValueError, RecursionError):
                pass
            return False

        # This validates the unchanged raw target's remaining shape only. It
        # never returns a training candidate or inserts missing JSON syntax.
        target, _ = training_steps._target(body, response, operation,
            allow_reasoning_history=True, argument_validator=argument_shape,
            allow_tool_error_history=True)
        training_steps._validate_target_tools(body, target, operation)
        return bool(unfinished)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False


def _role(example: Mapping[str, Any], call: Mapping[str, Any]) -> str:
    if call.get("run_id") == example.get("run_id"):
        return example.get("role") if example.get("role") in {"miner", "baseline"} else "miner"
    for attempt in example.get("attempts") or []:
        if attempt.get("run_id") == call.get("run_id"):
            return "judge" if attempt.get("kind") == "score" else example.get("role", "miner")
    return "unknown"


def coverage(examples) -> dict:
    counts, grouped, trajectories = Counter(), defaultdict(Counter), Counter()
    attempts = []
    full_text_calls = 0
    seen = set()
    for example in examples:
        if not isinstance(example, Mapping) or example.get("schema_version") != training_dataset.EXAMPLE_SCHEMA_VERSION:
            continue
        source = example.get("example_hash")
        if source != contracts.document_hash({key: value for key, value in example.items() if key != "example_hash"}):
            continue
        if source in seen:
            continue
        seen.add(source)
        bad = False
        has_text = False
        configuration = example.get("configuration") or {}
        registry = configuration.get("reasoning_route_registry")
        new_policy = training_coverage.MISSING_REASONING_SETTING in configuration
        try:
            maximum_missing = training_coverage.missing_reasoning_percent_from_configuration(configuration)
        except (ValueError, TypeError, AttributeError):
            maximum_missing = 0
            bad = True
        total_reasoning_tokens = missing_reasoning_tokens = 0
        for call in example.get("provider_calls") or []:
            row = attempt_coverage(call, _role(example, call), registry)
            full, _ = full_text_segments(call, _role(example, call), registry)
            attempts.append(row)
            counts[row["returned"]] += 1
            key = (row["role"], row["model_id"], row["requested_provider_route"], row["api"])
            grouped[key][row["returned"]] += 1
            grouped[key]["requested"] += row["requested"]
            grouped[key]["saved"] += row["storage"] == "saved"
            execution_model = _execution_model_call(example, call)
            full_text_calls += execution_model and bool(full)  # Raw exposed text inventory.
            full_round = (example.get("configuration") or {}).get("reasoning_capture_policy") == POLICY
            failed = new_policy and verified_failed_attempt(example, call)
            omitted_truncation = False
            target_usable = not new_policy
            if new_policy and execution_model and not failed:
                try:
                    training_steps._candidate(example, call)
                    target_usable = True
                except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
                    omitted_truncation = verified_cap_truncated_target(example, call) or (
                        isinstance(exc, ValueError) and str(exc) == "response_not_complete"
                        and row["returned"] == "truncated" and row["route_reviewed"]
                        and row["provider_route_verified"])
            if (execution_model and new_policy and row["route_reviewed"] and row["provider_route_verified"]
                    and row["returned"] not in {"truncated", "interrupted", "redacted"}
                    and row["tokens_reported"] is not None and not failed and target_usable):
                total_reasoning_tokens += row["tokens_reported"]
                if not full and verified_missing_thinking(row):
                    missing_reasoning_tokens += row["tokens_reported"]
            if omitted_truncation and row["returned"] != "truncated":
                row["target_omission_reason"] = "inferred_output_cap_json_eof"
            has_text |= execution_model and bool(full) and target_usable
            bad |= execution_model and new_policy and not (target_usable or failed or omitted_truncation)
            empty = (verified_no_thinking(call, call.get("response") or {}, row) if new_policy
                     else verified_tool_only_empty(call, call.get("response") or {}, row))
            bad |= (execution_model and (row["requested"] or full_round) and
                    not (full or empty or failed or omitted_truncation
                         or new_policy and verified_missing_thinking(row)))
        if new_policy:
            bad |= missing_reasoning_tokens * 100 > total_reasoning_tokens * maximum_missing
        trajectories["ineligible"] += bad
        trajectories["with_full_thinking"] += has_text
        trajectories["eligible_without_disqualifying_capture"] += not bad
        trajectories["complete_full_thinking"] += bool(
            not bad and has_text and (example.get("completeness") or {}).get("complete") is True)
        trajectories["selected_research_full_thinking"] += bool(
            not bad and has_text and selected_research_evidence(example) is not None)
    return {"attempts": attempts, "returned_classes": dict(sorted(counts.items())),
            "distinct_verified_exports": len(seen),
            "saved_calls": sum(row["storage"] == "saved" for row in attempts),
            "generated_thinking_texts": full_text_calls,
            "summary_calls": sum(row["returned"] == "summary" for row in attempts),
            "by_role_model_route": [{"role": key[0], "model_id": key[1], "requested_provider_route": key[2],
                                     "api": key[3], "counts": dict(sorted(value.items()))}
                                    for key, value in sorted(grouped.items(), key=lambda item: str(item[0]))],
            "trajectories": dict(sorted(trajectories.items()))}


def readable_trace(example: Mapping[str, Any]) -> dict:
    """Index a private export for review without replacing its complete raw JSON.

    Paths point into the supplied, hash-verified export. No text is shortened.
    Provider receipts are distinct from the response confirmed as delivered.
    """
    if (not isinstance(example, Mapping)
            or example.get("example_hash") != contracts.document_hash(
                {key: value for key, value in example.items() if key != "example_hash"})):
        raise ValueError("example_hash_mismatch")
    registry = (example.get("configuration") or {}).get("reasoning_route_registry")
    calls = []
    for ci, call in enumerate(example.get("provider_calls") or []):
        prefix = ["provider_calls", ci]
        body = ((call.get("request") or {}).get("outbound") or {}).get("body")
        events = []

        def add(kind, path, value, *, delivery=None, reasoning_field=None, alias_paths=()):
            events.append({"kind": kind, "source_path": path, "value": copy.deepcopy(value),
                           "value_sha256": contracts.document_hash(value),
                           **({"delivery": delivery} if delivery else {}),
                           **({"reasoning_field": reasoning_field} if reasoning_field else {}),
                           **({"alias_source_paths": [path_prefix for path_prefix in alias_paths]}
                              if alias_paths else {})})

        def add_derived(kind, evidence_path, *, delivery):
            events.append({"kind": kind, "derived": True,
                           "evidence_source_path": evidence_path, "value": True,
                           "delivery": delivery})

        if isinstance(body, Mapping):
            for key in sorted(body):
                if key not in {"messages", "input", "tools"}:
                    add("request_parameter", prefix + ["request", "outbound", "body", key], body[key])
            if isinstance(body.get("messages"), list):
                for mi, message in enumerate(body["messages"]):
                    if not isinstance(message, Mapping):
                        continue
                    path = prefix + ["request", "outbound", "body", "messages", mi]
                    add("input_message", path, message)
            if isinstance(body.get("input"), list):
                for ii, item in enumerate(body["input"]):
                    add("input_item", prefix + ["request", "outbound", "body", "input", ii], item)
            elif isinstance(body.get("input"), str):
                add("input_text", prefix + ["request", "outbound", "body", "input"], body["input"])
            if isinstance(body.get("tools"), list):
                for ti, tool in enumerate(body["tools"]):
                    add("tool_definition", prefix + ["request", "outbound", "body", "tools", ti], tool)

        def response_events(response, path, delivery):
            if not isinstance(response, Mapping):
                return
            operation = call.get("operation_id")
            for key in sorted(response):
                if key not in {"choices", "output", "usage"}:
                    add("response_metadata", path + [key], response[key], delivery=delivery)
            segments, summary, opaque = _segments(response, operation)
            full, _ = full_text_segments({**call, "response": response}, _role(example, call), registry)
            for segment in segments:
                classification = ("summary" if "summary" in segment["field"] else
                                  "full_generated_thinking" if (delivery == "selected_terminal"
                                                                 and segment in full) else
                                  "unverified_reasoning_text")
                sources = segment["source_paths"]
                add(classification, path + sources[0], segment["text"], delivery=delivery,
                    reasoning_field=segment["field"],
                    alias_paths=[path + source for source in sources[1:]])
            if summary and not any("summary" in segment["field"] for segment in segments):
                add_derived("summary_present_without_readable_text", path, delivery=delivery)
            if opaque or "hidden_reasoning" in (call.get("response_privacy_redactions") or []):
                add_derived("opaque_reasoning_removed", path, delivery=delivery)
            if isinstance(response.get("usage"), Mapping):
                add("usage", path + ["usage"], response["usage"], delivery=delivery)
            if operation == "openrouter.chat":
                for choice_index, choice in enumerate(response.get("choices") or []):
                    if not isinstance(choice, Mapping):
                        continue
                    message = choice.get("message")
                    if isinstance(message, Mapping):
                        base = path + ["choices", choice_index, "message"]
                        if "content" in message:
                            add("assistant_output", base + ["content"], message["content"], delivery=delivery)
                        for tool_index, tool in enumerate(message.get("tool_calls") or []):
                            add("tool_call", base + ["tool_calls", tool_index], tool, delivery=delivery)
            elif operation == "openrouter.responses":
                for output_index, item in enumerate(response.get("output") or []):
                    if not isinstance(item, Mapping):
                        continue
                    if item.get("type") in {"message", "function_call", "function_call_output"}:
                        add("assistant_output" if item.get("type") == "message" else "tool_call",
                            path + ["output", output_index], item, delivery=delivery)

        if isinstance(call.get("response"), Mapping):
            response_events(call["response"], prefix + ["response"], "selected_terminal")
        receipts = []
        for ri, receipt in enumerate(call.get("response_receipts") or []):
            record = {"entry_id": receipt.get("entry_id"), "entry_kind": receipt.get("entry_kind"),
                      "created_at": receipt.get("created_at"), "metadata": copy.deepcopy(receipt.get("metadata")),
                      "response_sha256": receipt.get("response_sha256"),
                      "wire_body_sha256": receipt.get("wire_body_sha256"),
                      "wire_body_bytes": receipt.get("wire_body_bytes"),
                      "privacy_redactions": copy.deepcopy(receipt.get("privacy_redactions")),
                      "source_path": prefix + ["response_receipts", ri]}
            if isinstance(receipt.get("response"), Mapping):
                response_events(receipt["response"], record["source_path"] + ["response"], "terminal_receipt")
            else:
                record["response_ref"] = copy.deepcopy(receipt.get("response_ref"))
            receipts.append(record)
        calls.append({"call_identity": call.get("call_identity"), "run_id": call.get("run_id"),
                      "operation_id": call.get("operation_id"), "provider": call.get("provider"),
                      "request_path": prefix + ["request"],
                      "request_sha256": (call.get("request_capture") or {}).get("sha256"),
                      "response_path": prefix + ["response"], "response_sha256": call.get("response_sha256"),
                      "response_status": call.get("response_status"),
                      "response_provenance": copy.deepcopy(call.get("response_provenance")),
                      "reasoning_coverage": attempt_coverage(call, _role(example, call), registry),
                      **({"target_omission_reason": "inferred_output_cap_json_eof"}
                         if verified_cap_truncated_target(example, call) else {}),
                      "actual_microusd": call.get("actual_microusd"),
                      "provider_cost": copy.deepcopy(call.get("provider_cost")),
                      "lifecycle": copy.deepcopy(call.get("lifecycle")),
                      "response_receipts": receipts,
                      "upstream_response_receipts": copy.deepcopy(call.get("upstream_response_receipts")),
                      "evidence_gaps": copy.deepcopy(call.get("evidence_gaps") or []),
                      "events": events})
    observations = []
    for field in ("worker_observation_artifacts", "partial_worker_observation_artifacts"):
        for run_id, artifact in sorted((example.get(field) or {}).items()):
            payload = artifact.get("payload") if isinstance(artifact, Mapping) else None
            if not isinstance(payload, Mapping):
                continue
            for collection in ("observations", "agent_observations", "response_receipts"):
                for index, row in enumerate(payload.get(collection) or []):
                    observations.append({"run_id": run_id, "kind": row.get("kind", collection)
                                         if isinstance(row, Mapping) else collection,
                                         "source_path": [field, run_id, "payload", collection, index],
                                         "value": copy.deepcopy(row),
                                         "value_sha256": contracts.document_hash(row),
                                         "payload_sha256": artifact.get("payload_sha256")})
    core_context = ("icp", "configuration", "input_snapshots", "output", "judgment",
                    "runtime", "costs", "competition_cost_eligibility", "completeness")
    separate = {"provider_calls", "worker_observation_artifacts",
                "partial_worker_observation_artifacts", "example_hash"}
    context_keys = list(core_context) + sorted(set(example) - set(core_context) - separate)
    document_context = [
        {"field": key, "status": "present", "source_path": [key],
         "value_sha256": contracts.document_hash(example[key]), "value": copy.deepcopy(example[key])}
        if key in example else {"field": key, "status": "absent", "source_path": [key]}
        for key in context_keys]
    completeness = example.get("completeness") or {}
    incomplete_reasons = completeness.get("incomplete_reasons") if isinstance(completeness, Mapping) else []
    return {"schema_version": "leadpoet.lab_arena.readable_trace.v1",
            "run_id": example.get("run_id"), "example_hash": example["example_hash"],
            "raw_source": "private_training_export", "projection_truncated": False,
            "runtime": copy.deepcopy(example.get("runtime")),
            "capture_limits": {"request_max_bytes": training_capture.MAX_OBJECT_BYTES,
                               "receipt_max_bytes": training_export.MAX_RECEIPT_BYTES},
            "document_context": document_context,
            "capture_gaps": sorted(set(incomplete_reasons or []) |
                                   {gap for call in calls for gap in call["evidence_gaps"]}),
            "calls": calls, "observations": observations, "worker_observation_paths": [
                ["worker_observation_artifacts", key] for key in sorted(example.get("worker_observation_artifacts") or {})],
            "partial_worker_observation_paths": [
                ["partial_worker_observation_artifacts", key] for key in sorted(example.get("partial_worker_observation_artifacts") or {})]}


def readable_trace_markdown(report: Mapping[str, Any], *, raw_export_name: str | None = None) -> str:
    """Render every indexed visible value, with paths back to the private export."""
    if report.get("schema_version") != "leadpoet.lab_arena.readable_trace.v1":
        raise ValueError("readable_trace_schema_invalid")

    def block(value):
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        longest = max((len(match[0]) for match in re.finditer(r"`+", rendered)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{fence}\n{rendered}\n{fence}"

    def event_lines(event):
        if event.get("derived") is True:
            return [f"#### {event['kind']}", "Derived status: `true`  ",
                    f"Evidence path: `{contracts.canonical_json(event['evidence_source_path'])}`  ",
                    *([f"Receipt source: `{event['delivery']}`  "] if event.get("delivery") else []),
                    "This marker is inferred from the saved response or redaction metadata; it is not a raw value.", ""]
        return [f"#### {event['kind']}",
                *([f"Reasoning field: `{event['reasoning_field']}`  "] if event.get("reasoning_field") else []),
                *([f"Receipt source: `{event['delivery']}`  "] if event.get("delivery") else []),
                f"Raw path: `{contracts.canonical_json(event['source_path'])}`  ",
                *([f"Alias paths: `{contracts.canonical_json(event['alias_source_paths'])}`  "]
                  if event.get("alias_source_paths") else []),
                f"Value hash: `{event['value_sha256']}`",
                block(event["value"]), ""]

    lines = ["# Arena readable trace", "",
             f"Run: `{report.get('run_id')}`  ",
             f"Raw export hash: `{report.get('example_hash')}`  ",
             "Projection truncation: none. Capture gaps and size limits appear below.", ""]
    if raw_export_name is not None:
        if (not raw_export_name or any(character in raw_export_name for character in "<>\n\r")
                or raw_export_name.startswith("/") or ".." in raw_export_name.split("/")):
            raise ValueError("raw_export_name_invalid")
        lines.extend([f"[Open the raw private export](<{raw_export_name}>)", ""])
    for key in ("runtime", "capture_limits", "capture_gaps"):
        lines.extend([f"## {key.replace('_', ' ').title()}", "", block(report.get(key)), ""])
    lines.extend(["## Document context", "",
                  "Each present value is copied from the hash-verified private export. Absent means the field was not saved.", ""])
    for entry in report.get("document_context") or []:
        lines.extend([f"### {entry['field']}", "", f"Status: `{entry['status']}`  ",
                      f"Raw path: `{contracts.canonical_json(entry['source_path'])}`  "])
        if entry["status"] == "present":
            lines.extend([f"Value hash: `{entry['value_sha256']}`", block(entry["value"]), ""])
        else:
            lines.append("")
    for index, call in enumerate(report.get("calls") or []):
        lines.extend([f"## Call {index + 1}: {call.get('operation_id')}", "",
                      f"Identity: `{call.get('call_identity')}`  ",
                      f"Request path: `{contracts.canonical_json(call.get('request_path'))}`  ",
                      f"Request hash: `{call.get('request_sha256')}`  ",
                      f"Response path: `{contracts.canonical_json(call.get('response_path'))}`  ",
                      f"Response hash: `{call.get('response_sha256')}`  ",
                      f"HTTP status: `{call.get('response_status')}`  ",
                      f"Actual cost (micro USD): `{call.get('actual_microusd')}`", ""])
        for label, key in (("Reasoning classification", "reasoning_coverage"),
                           ("Provider cost", "provider_cost"), ("Lifecycle timing", "lifecycle"),
                           ("Terminal receipts", "response_receipts"),
                           ("Upstream receipts", "upstream_response_receipts"),
                           ("Delivery proof", "response_provenance"), ("Evidence gaps", "evidence_gaps")):
            lines.extend([f"### {label}", "", block(call.get(key)), ""])
        lines.extend(["### Visible events", ""])
        for event in call.get("events") or []:
            lines.extend(event_lines(event))
    lines.extend(["## Worker observations and receipt events", ""])
    for event in report.get("observations") or []:
        lines.extend(event_lines(event))
    return "\n".join(lines) + "\n"


def _prior_observations(example: Mapping[str, Any], call: Mapping[str, Any]) -> list[dict]:
    artifact = (example.get("worker_observation_artifacts") or {}).get(call.get("run_id")) or {}
    payload = artifact.get("payload") or {}
    if contracts.document_hash(payload) != artifact.get("payload_sha256"):
        raise ValueError("worker_payload_hash_mismatch")
    sequence = call.get("action_sequence")
    if type(sequence) is not int:
        raise ValueError("action_sequence_unavailable")
    if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
        return []  # New targets include only the exact provider-visible input.
    result = []
    for row in payload.get("agent_observations") or []:
        previous = row.get("after_action_sequence")
        if type(previous) is not int or previous >= sequence:
            continue
        if row.get("kind") not in {"web_fetch", "web_fetch_binary", "http_response_metadata", "state_snapshot"}:
            continue
        if (row.get("privacy") or {}).get("redacted"):
            raise ValueError("prior_observation_redacted")
        data = copy.deepcopy(row.get("data") or {})
        if row.get("kind") == "web_fetch_binary":
            data.pop("body_b64", None)
        result.append({"id": row.get("id"), "kind": row["kind"], "observed_at": row.get("observed_at"),
                       "after_action_sequence": previous, "depends_on": copy.deepcopy(row.get("depends_on")),
                       "data": data})
    return result


def _delivered_response(call: Mapping[str, Any], digest: str) -> Mapping[str, Any]:
    """Use the acknowledged body, never a later billing recovery as teacher."""
    values = []
    if call.get("response_sha256") == digest and isinstance(call.get("response"), Mapping):
        values.append(call["response"])
    for receipt in call.get("response_receipts") or []:
        if receipt.get("response_sha256") == digest and isinstance(receipt.get("response"), Mapping):
            values.append(receipt["response"])
    if not values or any(contracts.document_hash(value) != digest for value in values):
        raise ValueError("delivered_response_unavailable")
    if len({contracts.canonical_json(value) for value in values}) != 1:
        raise ValueError("delivered_response_ambiguous")
    return values[0]


def verified_output_projection(example: Mapping[str, Any]) -> bool:
    """Bind the normalized export while retaining the stored document hash.

    New exports record both identities. Old exports can prove only an output
    whose normalized projection has the same document hash as the stored JSON.
    """
    output = example.get("output")
    identity = example.get("output_identity") or {}
    stored_hash = identity.get("document_hash")
    projected_hash = identity.get("normalized_document_hash", stored_hash)
    return (isinstance(output, Mapping)
            and isinstance(stored_hash, str) and contracts.SHA256_RE.fullmatch(stored_hash) is not None
            and isinstance(projected_hash, str) and contracts.SHA256_RE.fullmatch(projected_hash) is not None
            and projected_hash == contracts.document_hash(output))


def _selected_input_output_binding(
    example: Mapping[str, Any], *, allow_selected_cost_unavailable: bool = False,
) -> set[str] | None:
    """Verify the accepted score, output hash, and both exact input snapshots."""
    run_id = example.get("run_id")
    score_id = example.get("score_run_id")
    output = example.get("output")
    judgment = example.get("judgment")
    output_identity = example.get("output_identity") or {}
    snapshots = example.get("input_snapshots") or {}
    if (not isinstance(run_id, str) or not isinstance(score_id, str)
            or not isinstance(output, Mapping) or not isinstance(judgment, Mapping)
            or type(judgment.get("per_icp_score")) not in (int, float)
            or not verified_output_projection(example)):
        return None
    attempts = example.get("attempts") or []
    score_runs = [row for row in attempts if row.get("run_id") == score_id
                  and row.get("kind") == "score" and row.get("status") == "accepted"]
    if len(score_runs) != 1:
        return None
    score_ids = {row.get("run_id") for row in attempts if row.get("kind") == "score"
                 and isinstance(row.get("run_id"), str)}
    for identity in (run_id, score_id):
        snapshot = snapshots.get(identity) or {}
        document = snapshot.get("input")
        if (not isinstance(document, Mapping)
                or (snapshot.get("privacy") or {}).get("redacted") is not False
                or snapshot.get("input_sha256") != contracts.document_hash(document)):
            return None
    score_input = snapshots[score_id]["input"]
    if (score_input.get("scored_run_id") != run_id
            or score_input.get("companies") != output.get("companies")):
        return None
    cost = example.get("competition_cost_eligibility") or {}
    if cost.get("status") != "available":
        if (not allow_selected_cost_unavailable
                or cost.get("status") != "unavailable"
                or cost.get("reason") != "provider_cost_uncertain"
                or cost.get("source_rpc") != "lab_arena_icp_cost_eligibility"
                or cost.get("policy") != (example.get("configuration") or {}).get(
                    "sourcing_cost_eligibility_policy")
                or not isinstance(cost.get("observed_at"), str)
                or not cost["observed_at"]):
            return None
    return score_ids


def _attributed_score_only_gaps(example: Mapping[str, Any], reasons: set[str]) -> frozenset[str] | None:
    """Use per-run export attribution only when the selected execution is exact."""
    proof = example.get("completeness") or {}
    dimensions = proof.get("dimensions") or {}
    by_dimension = proof.get("reasons_by_dimension") or {}
    if any(dimensions.get(name) is not True or by_dimension.get(name) != []
           for name in ("context", "source_identity", "exact_inputs", "input_output_label")):
        return None
    if (_selected_input_output_binding(example) is None
            or not isinstance(example.get("related_run_gap_attribution"), Mapping)):
        return None
    run_id = example["run_id"]
    rows = example["related_run_gap_attribution"]
    attempts = {row.get("run_id"): row for row in example.get("attempts") or []
                if isinstance(row, Mapping) and isinstance(row.get("run_id"), str)}
    if set(rows) != set(attempts) or run_id not in rows or len(attempts) != len(example.get("attempts") or []):
        return None
    research, costs = set(), set()
    for source_id, row in rows.items():
        attempt = attempts[source_id]
        if (not isinstance(row, Mapping) or row.get("kind") != attempt.get("kind")
                or row.get("status") != attempt.get("status")
                or not isinstance(row.get("research_process_gaps"), list)
                or not isinstance(row.get("cost_gaps"), list)):
            return None
        source_research = set(row["research_process_gaps"])
        source_costs = set(row["cost_gaps"])
        if (any(not isinstance(gap, str) for gap in source_research | source_costs)
                or not source_costs <= source_research):
            return None
        if source_id == run_id:
            if (row["kind"] != "execute" or row["status"] != "accepted"
                    or source_research or source_costs):
                return None
        elif row["kind"] != "score":
            return None
        research.update(source_research)
        costs.update(source_costs)
    if (research != set(by_dimension.get("research_process") or [])
            or costs != set(by_dimension.get("costs") or [])
            or proof.get("complete") is not False
            or dimensions.get("research_process") is not (not research)
            or dimensions.get("costs") is not (not costs)
            or reasons != research | costs
            or not reasons):
        return None
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if (payload.get("complete") is not True
            or artifact.get("payload_sha256") != contracts.document_hash(payload)
            or any(_required_observation_redacted(example, row, agent=field == "agent_observations")
                   for field in ("observations", "agent_observations")
                   for row in payload.get(field) or [])):
        return None
    selected = [call for call in example.get("provider_calls") or [] if call.get("run_id") == run_id]
    if not selected:
        return None
    registry = (example.get("configuration") or {}).get("reasoning_route_registry")
    for call in example.get("provider_calls") or []:
        call_gaps = set(call.get("evidence_gaps") or [])
        if call.get("run_id") not in rows or not call_gaps <= set(rows[call["run_id"]]["research_process_gaps"]):
            return None
        if call.get("run_id") != run_id:
            continue
        if call.get("accounting_state") == "refusal" and verified_failed_attempt(example, call):
            continue
        provenance = call.get("response_provenance") or {}
        request = call.get("request") or {}
        response = call.get("response")
        model = call.get("operation_id") in reasoning_capture.OPERATIONS
        permitted_redactions = {"hidden_reasoning"} if model else set()
        if (call_gaps or call.get("accounting_state") != "settlement"
                or type(call.get("actual_microusd")) is not int
                or (call.get("request_capture") or {}).get("redacted") is True
                or not set(call.get("response_privacy_redactions") or []) <= permitted_redactions
                or any(not set(row.get("privacy_redactions") or []) <= permitted_redactions
                       for row in call.get("response_receipts") or [])
                or not isinstance(request, Mapping)
                or request.get("operation_id") != call.get("operation_id")
                or request.get("request_hash") != call.get("request_hash")
                or not isinstance(request.get("hashed_parameters"), Mapping)
                or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")
                or not isinstance(response, Mapping)
                or contracts.document_hash(response) != call.get("response_sha256")
                or (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                    and (provenance.get("agent_receipt") != "reported_exact_body_received"
                         or provenance.get("selected_response_acknowledged") is not True))
                or provenance.get("delivery_to_worker") != "socket_written"):
            return None
        if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
            try:
                evidence = training_steps._host_result(example, call)
                if evidence["response"] != response:
                    return None
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
        else:
            worker_ref = provenance.get("worker_receipt_ref") or {}
            if (worker_ref.get("run_id") != run_id
                    or worker_ref.get("payload_sha256") != artifact["payload_sha256"]):
                return None
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
                return None
        if model:
            try:
                training_steps._candidate(example, call)
                state = attempt_coverage(call, _role(example, call), registry)
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            if not state["route_reviewed"] or not state["provider_route_verified"]:
                return None
    return frozenset(reasons)


def _selected_research_evidence(example: Mapping[str, Any]) -> frozenset[str] | None:
    """Return attributable judge-only gaps, or reject an inexact selected replay.

    Raw export completeness remains an all-related-runs audit. A score worker's
    private provider response is not part of the accepted agent's prior context
    or target, provided the score input and published label are still bound.
    """
    selected_id = example.get("run_id")
    if any(isinstance(call, Mapping) and call.get("run_id") == selected_id
            and not (call.get("accounting_state") == "refusal" and verified_failed_attempt(example, call)) and (
            "provider_cost_unresolved" in (call.get("evidence_gaps") or [])
            or ("accounting_state" in call and call.get("accounting_state") != "settlement")
            or ("actual_microusd" in call and type(call.get("actual_microusd")) is not int))
            for call in example.get("provider_calls") or []):
        return None
    proof = example.get("completeness") or {}
    reasons = set(proof.get("incomplete_reasons") or [])
    if proof.get("complete") is True and not reasons:
        if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
            try:
                for call in example.get("provider_calls") or []:
                    if call.get("run_id") != selected_id:
                        continue
                    if call.get("accounting_state") == "refusal":
                        if not verified_failed_attempt(example, call):
                            return None
                    else:
                        training_steps._host_result(example, call, allow_worker_error=True)
            except (KeyError, ValueError, TypeError, AttributeError):
                return None
        return frozenset()
    allowed = {"request_privacy_nonexact", "response_privacy_nonexact", "provider_cost_unresolved"}
    if reasons and not reasons <= allowed:
        return _attributed_score_only_gaps(example, reasons)
    if not reasons:
        return None
    dimensions = proof.get("dimensions") or {}
    by_dimension = proof.get("reasons_by_dimension") or {}
    for name in ("context", "source_identity", "exact_inputs", "input_output_label"):
        if dimensions.get(name) is not True or by_dimension.get(name) != []:
            return None
    if (set(by_dimension.get("research_process") or []) != reasons
            or set(by_dimension.get("costs") or []) != (reasons & {"provider_cost_unresolved"})
            or dimensions.get("research_process") is not False
            or dimensions.get("costs") is not ("provider_cost_unresolved" not in reasons)):
        return None
    score_ids = _selected_input_output_binding(example)
    if score_ids is None:
        return None
    run_id = example["run_id"]
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if (payload.get("complete") is not True
            or artifact.get("payload_sha256") != contracts.document_hash(payload)
            or any(_required_observation_redacted(example, row, agent=field == "agent_observations")
                   for field in ("observations", "agent_observations")
                   for row in payload.get(field) or [])):
        return None
    attributed = set()
    for call in example.get("provider_calls") or []:
        call_reasons = set(call.get("evidence_gaps") or [])
        if call.get("run_id") in score_ids:
            if not call_reasons <= allowed:
                return None
            if ("request_privacy_nonexact" in call_reasons
                    and ((call.get("request_capture") or {}).get("redacted") is not True
                         or "credential" not in ((call.get("request_capture") or {}).get("redactions") or []))):
                return None
            attributed.update(call_reasons)
        elif call_reasons:
            return None
        if call.get("run_id") == run_id:
            if call.get("accounting_state") == "refusal" and verified_failed_attempt(example, call):
                continue
            redactions = set(call.get("response_privacy_redactions") or [])
            model = call.get("operation_id") in reasoning_capture.OPERATIONS
            permitted = {"hidden_reasoning"} if model else set()
            if ((call.get("request_capture") or {}).get("redacted") is True
                    or not redactions <= permitted
                    or any(not set(row.get("privacy_redactions") or []) <= permitted
                           for row in call.get("response_receipts") or [])
                    or call.get("accounting_state") != "settlement"
                    or type(call.get("actual_microusd")) is not int):
                return None
    if attributed != reasons:
        return None
    return frozenset(reasons)


_LOCAL_WEB_DIAGNOSTICS = frozenset({"direct_web_usage_unverified", "direct_web_content_not_captured",
                                   "direct_web_coverage_unverified"})
_OPTIONAL_LOCAL_KINDS = frozenset({"web_fetch", "web_fetch_binary", "http_response_metadata", "state_snapshot"})


def _required_observation_redacted(example, row, *, agent=False):
    return ((row.get("privacy") or {}).get("redacted") is not False
            and not (agent and row.get("kind") in _OPTIONAL_LOCAL_KINDS
                     and training_coverage.uses_host_result_evidence(example.get("configuration") or {})))


def _provider_visible_view(example):
    """Ignore only attributed local-runtime diagnostics for provider proof.

    This is a validation view. The saved export, completeness, hashes and
    optional observations remain unchanged. All callers still prove the
    retained gateway inputs/results, host writes, source/output and costs.
    """
    if not training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
        return example, frozenset()
    proof = example.get("completeness") or {}
    reasons = set(proof.get("incomplete_reasons") or [])
    ignored = reasons & _LOCAL_WEB_DIAGNOSTICS
    artifacts = example.get("worker_observation_artifacts") or {}
    optional_redacted = False
    optional_binary = False
    for artifact in artifacts.values():
        payload = artifact.get("payload") or {}
        if artifact.get("payload_sha256") != contracts.document_hash(payload):
            return example, frozenset()
        if any(_required_observation_redacted(example, row) for row in payload.get("observations") or []):
            return example, frozenset()
        for row in payload.get("agent_observations") or []:
            if _required_observation_redacted(example, row, agent=True):
                return example, frozenset()
            optional_redacted |= (row.get("kind") in _OPTIONAL_LOCAL_KINDS
                                  and (row.get("privacy") or {}).get("redacted") is True)
            optional_binary |= row.get("kind") == "web_fetch_binary"
    if optional_redacted and "worker_observation_privacy_nonexact" in reasons:
        ignored.add("worker_observation_privacy_nonexact")
    if optional_binary and "binary_privacy_review_required" in reasons:
        ignored.add("binary_privacy_review_required")
    if not ignored:
        return example, frozenset()
    by_dimension = proof.get("reasons_by_dimension") or {}
    dimensions = proof.get("dimensions") or {}
    if (proof.get("complete") is not False or not by_dimension
            or set().union(*(set(value) for value in by_dimension.values())) != reasons
            or any(dimensions.get(name) is not (not value) for name, value in by_dimension.items())
            or not ignored <= set(by_dimension.get("research_process") or [])
            or any(ignored & set(value) for name, value in by_dimension.items() if name != "research_process")):
        return example, frozenset()
    view = copy.deepcopy(example)
    result = view["completeness"]
    result["incomplete_reasons"] = sorted(reasons - ignored)
    result["reasons_by_dimension"]["research_process"] = sorted(set(by_dimension["research_process"]) - ignored)
    result["dimensions"]["research_process"] = not result["reasons_by_dimension"]["research_process"]
    result["complete"] = not result["incomplete_reasons"]
    for row in (view.get("related_run_gap_attribution") or {}).values():
        row["research_process_gaps"] = sorted(set(row.get("research_process_gaps") or []) - ignored)
    return view, frozenset(ignored)


def provider_visible_view(example):
    try:
        return _provider_visible_view(example)
    except (TypeError, ValueError, KeyError, AttributeError):
        return example, frozenset()  # Invalid diagnostics never grant an exception.


def selected_research_evidence(example: Mapping[str, Any]) -> frozenset[str] | None:
    view, diagnostic_gaps = provider_visible_view(example)
    if diagnostic_gaps and _selected_input_output_binding(view) is None:
        return None
    result = _selected_research_evidence(view)
    return None if result is None else result | diagnostic_gaps


def build(examples, rules: training_steps.Rules | None = None):
    """Return private candidates only for proven full generated thinking."""
    rules = rules or training_steps.Rules()
    examples = list(examples)
    splits, historical = training_steps.build(examples, rules)
    # Reuse existing exact request, agent read, receipt and split checks. Its
    # summary-only candidates are merely a search space, never admitted here.
    by_hash = {example.get("example_hash"): example for example in examples}
    scoped = {key: selected_research_evidence(example) for key, example in by_hash.items()}
    output = {"train": [], "eval": [], "analysis": []}
    records_by_split: dict[str, dict[str, dict]] = {"train": {}, "eval": {}}
    reasons = Counter()
    for split in ("train", "eval"):
        for candidate in splits[split]:
            for ref in candidate["source_refs"]:
                example = by_hash.get(ref["example_hash"])
                if not example:
                    continue
                call = next((row for row in example.get("provider_calls") or []
                             if row.get("call_identity") == ref["call_identity"] and row.get("run_id") == ref["run_id"]), None)
                if call is None:
                    continue
                ancillary_gaps = scoped[ref["example_hash"]]
                if ancillary_gaps is None:
                    reasons["trajectory_evidence_incomplete"] += 1
                    continue
                try:
                    delivered = _delivered_response(call, ref["delivered_response_sha256"])
                except ValueError as exc:
                    reasons[str(exc)] += 1
                    continue
                delivered_call = dict(call, response=delivered,
                                      response_sha256=ref["delivered_response_sha256"])
                registry = (example.get("configuration") or {}).get("reasoning_route_registry")
                trajectory_valid = True
                new_policy = training_coverage.MISSING_REASONING_SETTING in (example.get("configuration") or {})
                try:
                    maximum_missing = training_coverage.missing_reasoning_percent_from_configuration(
                        example.get("configuration") or {})
                except (TypeError, ValueError, AttributeError):
                    reasons["frozen_reasoning_policy_invalid"] += 1
                    continue
                total_reasoning_tokens = missing_reasoning_tokens = 0
                for row in example.get("provider_calls") or []:
                    if not _execution_model_call(example, row):
                        continue
                    if new_policy and verified_failed_attempt(example, row):
                        continue
                    try:
                        linked = training_steps._candidate(example, row)["source_refs"][0]
                        response = _delivered_response(row, linked["delivered_response_sha256"])
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        if new_policy and verified_cap_truncated_target(example, row):
                            continue
                        # _candidate reaches this finish check only after exact
                        # request, agent read, response hash and privacy proof.
                        # Keep other independently reconstructable targets.
                        if (new_policy and isinstance(exc, ValueError)
                                and str(exc) == "response_not_complete"):
                            try:
                                state = attempt_coverage(row, _role(example, row), registry)
                            except (ValueError, TypeError, KeyError, AttributeError):
                                trajectory_valid = False
                                break
                            if (state["returned"] == "truncated" and state["route_reviewed"]
                                    and state["provider_route_verified"]):
                                continue
                        trajectory_valid = False
                        break
                    checked = dict(row, response=response,
                                   response_sha256=linked["delivered_response_sha256"])
                    full, state = full_text_segments(checked, _role(example, row), registry)
                    if (state["route_reviewed"] and state["provider_route_verified"]
                            and state["tokens_reported"] is not None):
                        total_reasoning_tokens += state["tokens_reported"]
                        if not full and verified_missing_thinking(state):
                            missing_reasoning_tokens += state["tokens_reported"]
                    if (not full and not (verified_no_thinking(row, response, state) if new_policy
                                          else verified_tool_only_empty(row, response, state))
                            and not (new_policy and verified_missing_thinking(state))):
                        trajectory_valid = False
                        break
                if (new_policy
                        and missing_reasoning_tokens * 100 > total_reasoning_tokens * maximum_missing):
                    reasons["missing_reasoning_share_exceeded"] += 1
                    continue
                if not trajectory_valid:
                    reasons["trajectory_reasoning_ineligible"] += 1
                    continue
                segments, status = full_text_segments(delivered_call, _role(example, call), registry)
                if not segments:
                    reasons[status["returned"]] += 1
                    continue
                try:
                    prior = _prior_observations(example, call)
                except ValueError as exc:
                    reasons[str(exc)] += 1
                    continue
                diagnostic_gaps = provider_visible_view(example)[1]
                record = {"schema_version": RECORD_SCHEMA_VERSION, "format": candidate["format"],
                          "input": copy.deepcopy(candidate["input"]), "prior_observations": prior,
                          "generated_thinking": segments, "target": copy.deepcopy(candidate["target"]),
                          "quality": copy.deepcopy(candidate["quality"]),
                          "source_refs": [{**{key: copy.deepcopy(value) for key, value in ref.items()
                                               if key != "teacher_reasoning_metadata"},
                                           "response_sha256": ref["delivered_response_sha256"],
                                           "route": copy.deepcopy(call["reasoning_capture"]),
                                           "selected_research_evidence": {
                                               "complete": True,
                                               **({"scope": "provider_visible_request_results"} if training_coverage.uses_host_result_evidence(example.get("configuration") or {}) else {}),
                                               "ancillary_judge_gaps": sorted(ancillary_gaps - diagnostic_gaps),
                                               **({"diagnostic_capture_gaps": sorted(diagnostic_gaps)}
                                                  if training_coverage.uses_host_result_evidence(example.get("configuration") or {}) else {})}}]}
                record["candidate_id"] = contracts.document_hash({key: record[key] for key in
                    ("format", "input", "prior_observations", "generated_thinking", "target")})
                existing = records_by_split[split].get(record["candidate_id"])
                if existing is None:
                    records_by_split[split][record["candidate_id"]] = record
                else:
                    existing["source_refs"].extend(record["source_refs"])
    for split, records in records_by_split.items():
        for record in records.values():
            record["source_refs"] = sorted({contracts.canonical_json(ref): ref for ref in record["source_refs"]}.values(),
                                           key=contracts.canonical_json)
        output[split] = [records[key] for key in sorted(records)]
    output["analysis"] = copy.deepcopy(splits["analysis"])
    manifest = {"schema_version": SCHEMA_VERSION, "rules": historical["rules"],
                "training_permission": copy.deepcopy(training_dataset.TRAINING_PERMISSION),
                "source_dataset_version": historical["dataset_version"],
                "counts": {"records_by_split": {key: len(rows) for key, rows in output.items()},
                           "exclusion_reasons": dict(sorted(reasons.items()))},
                "coverage": coverage(examples)}
    manifest["dataset_version"] = contracts.document_hash(manifest)
    return output, manifest
