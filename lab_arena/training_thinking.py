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

from lab_arena import baseline_reasoning, contracts, operations, reasoning_capture, training_capture, training_coverage, training_dataset, training_export, training_steps, training_worker_proof

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
    if (contracts.document_hash(frozen).split(":", 1)[1] != annotation["registry_sha256"]
            or not reasoning_capture.explicit_provider_matches(call.get("request") or {}, annotation.get("provider_route"))):
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
    # The route annotation describes a probe. Report the saved outbound controls
    # separately: an absent control is different from an unavailable request.
    body = (((call.get("request") or {}).get("outbound") or {}).get("body"))
    outbound_controls = ({key: copy.deepcopy(body[key])
                          for key in ("reasoning", "reasoning_effort", "include_reasoning")
                          if key in body} if isinstance(body, Mapping) else None)
    controls_status = ("unavailable" if outbound_controls is None else
                       "present" if outbound_controls else "absent")
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
            "outbound_reasoning_controls": outbound_controls,
            "outbound_reasoning_controls_status": controls_status,
            "probe_reasoning_mode": copy.deepcopy(annotation.get("mode")),
            "route_reviewed": policy_verified,
            "provider_route_verified": observation_verified,
            "returned": content_class, "storage": storage, "finish_reason": finish,
            "characters": sum(len(part["text"]) for part in segments),
            "tokens_reported": tokens, "segment_count": len(segments),
            "contains_summary": summary, "contains_opaque": contains_opaque,
            "redactions": sorted(redactions)}


def _ignored_provider_failure(response, request_body, annotation) -> bool:
    """Prove a bound routing failure before selection, not arbitrary 404s.

    The surrounding failure check proves the saved request and response.
    Routing metadata proves no endpoint was selected; it does not prove a
    zero bill. Error wording and endpoint multiplicity are not authority.
    """
    error = response.get("error") or {}
    metadata = error.get("metadata") or {}
    routing = response.get("openrouter_metadata") or {}
    provider = request_body.get("provider") or {}
    funnel = metadata.get("routing_funnel")
    endpoints = (routing.get("endpoints") or {}).get("available")
    if (type(error.get("code")) is not int or error["code"] != 404
            or not isinstance(metadata.get("failed_routing_step"), str)
            or not metadata["failed_routing_step"].strip()
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
    # The available catalog can include endpoints removed by earlier filters.
    # All listed endpoints must still explicitly report no selection above.
    return (len(allowed) == 1 and funnel[-1] == allowed[0]
            and type(allowed[0].get("endpoint_count")) is int
            and 0 < allowed[0]["endpoint_count"] <= len(target)
            and metadata["failed_routing_step"] not in {row.get("step") for row in funnel
                                                       if isinstance(row, Mapping)})


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


def _legacy_tested_controls_match(call) -> bool:
    """Retain the old forced shape, without accepting contradictory aliases."""
    try:
        request = call["request"]
        body = request["outbound"]["body"]
        annotation = call["reasoning_capture"]
        operation = call["operation_id"]
        if operation not in reasoning_capture.OPERATIONS:
            return False
        field = "max_tokens" if operation == "openrouter.chat" else "max_output_tokens"
        mode = annotation["mode"]
        expected = {"max_tokens": mode["budget_tokens"]} if "budget_tokens" in mode else dict(mode)
        allowed = (expected, {**expected, "exclude": False}) if operation == "openrouter.chat" else (expected,)
        if (type(body.get(field)) is not int or body[field] != annotation["max_output_tokens"]
                or body.get("reasoning") not in allowed
                or "max_tokens" in expected and expected["max_tokens"] >= body[field]):
            return False
        for parameters in (body, request.get("hashed_parameters") or {}, request.get("effective_parameters") or {}):
            if (any(key in parameters for key in ("reasoning_effort", "include_reasoning"))
                    or any(key in parameters and parameters[key] != body[key] for key in (field, "reasoning"))):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _captured_reasoning_controls_match(call, registry) -> bool:
    """Bind preserved settings to the saved hash, not the route probe's mode."""
    try:
        request = call["request"]
        hashed = request["hashed_parameters"]
        body = request["outbound"]["body"]
        operation = call["operation_id"]
        # Hash agreement alone cannot make Chat-only controls valid Responses
        # parameters. Validate the saved request without changing its bytes.
        operations.validate_operation_request(operation, hashed, allow_full_thinking=True)
        cap_field = "max_tokens" if operation == "openrouter.chat" else "max_output_tokens"
        fields = (cap_field, "reasoning", "reasoning_effort", "include_reasoning")
        controls = {key: hashed[key] for key in fields if key in hashed}
        if (contracts.document_hash(hashed) != call.get("request_hash")
                or request.get("request_hash") != call.get("request_hash")
                or type(controls.get(cap_field)) is not int
                or not 1 <= controls[cap_field] <= operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS
                or controls != {key: body[key] for key in fields if key in body}):
            return False
        checked, annotation = reasoning_capture.apply(operation, hashed, policy=POLICY, registry=registry)
        # Defaults must already be explicit in the hashed provider request.
        # Reject disabled/ambiguous controls; never repair saved evidence.
        return annotation is not None and controls == {key: checked[key] for key in fields if key in checked}
    except (KeyError, TypeError, ValueError, AttributeError, operations.OperationError):
        return False


def verified_recovered_provider_retry(example: Mapping[str, Any], call: Mapping[str, Any], *, worker_proofs=None) -> bool:
    """Prove a hidden throttle recovered by the same request's host write.

    A managed retry is provider evidence, never a delivered response or target.
    Retain the settled-zero later-action proof. Broker-internal attempts instead
    need exact same-action lineage; their saved billing remains authoritative,
    including uncertainty. Both require saved 429 bytes and a successful write.
    """
    try:
        registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
        request = call.get("request") or {}
        outbound = request.get("outbound") or {}
        body = outbound.get("body")
        provenance = call.get("response_provenance") or {}
        terminal_id = call.get("terminal_entry_id")
        sequence = call.get("action_sequence")
        operation = call.get("operation_id")
        internal = call.get("response_status") == 402
        expected_status = 402 if internal else 502
        expected_error = {"error": {"code": "miner_credentials_unavailable" if internal else "provider_unavailable"}}
        digest = contracts.document_hash(expected_error)
        amount = call.get("actual_microusd")
        accounting = call.get("accounting_state")
        response_id = training_export.response_entry_id(call) if internal else terminal_id
        accounting_view = call
        if internal and response_id != terminal_id:
            # The existing retained-response primitive proves the delayed bill
            # points back to this error. Verify both amounts; order recovery
            # against the saved error, never the later accounting-only head.
            prior = [row for row in call.get("lifecycle") or [] if row.get("entry_id") == response_id]
            if (len(prior) != 1 or prior[0].get("entry_kind") != "uncertain"
                    or type(prior[0].get("amount_microusd")) is not int or prior[0]["amount_microusd"] != 0):
                return False
            accounting_view = {**call, "lifecycle": [row for row in call.get("lifecycle") or []
                                                    if row.get("entry_id") != response_id]}
        if internal:
            base, attempt = call.get("base_call_identity"), call.get("provider_attempt")
            if (call.get("provider") != "openrouter" or call.get("funding_source") != "miner_key"
                    or not isinstance(base, str) or contracts.SHA256_RE.fullmatch(base) is None
                    or type(attempt) is not int or attempt < 1
                    or call.get("call_identity") != (base if attempt == 1 else contracts.document_hash(
                        {"base_call_identity": base, "provider_attempt": attempt}))
                    or not (accounting == "uncertain" and "actual_microusd" in call and amount is None
                            or accounting == "settlement" and type(amount) is int and amount >= 0)
                    or not _terminal_accounting_matches(accounting_view, accounting, 0 if amount is None else amount)
                    or type(call.get("reservation_entry_id")) is not int
                    or not 0 < call["reservation_entry_id"] < response_id
                    or set(call.get("evidence_gaps") or []) != (
                        {"provider_cost_unresolved"} if accounting == "uncertain" else set())):
                return False
        elif (accounting != "settlement" or type(amount) is not int or amount != 0
              or call.get("evidence_gaps")):
            return False
        if (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                or call.get("run_id") != example.get("run_id")
                or operation not in reasoning_capture.OPERATIONS
                or not internal and operation != "openrouter.responses"
                or call.get("funding_source") not in {"host", "miner_key"}
                or type(sequence) is not int or sequence < 0 or type(terminal_id) is not int or terminal_id < 1
                or not _policy(call, registry)
                or (call.get("request_capture") or {}).get("redacted")
                or not training_steps._exact(request)
                or request.get("operation_id") != operation
                or request.get("request_hash") != call.get("request_hash")
                or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")
                or not isinstance(body, Mapping)
                or contracts.document_hash(body) != outbound.get("body_sha256")
                or outbound.get("method") != "POST" or outbound.get("host") != "openrouter.ai"
                or outbound.get("path") != operations.OPERATIONS[operation].path
                or call.get("response_status") != expected_status or call.get("response") != expected_error
                or call.get("response_sha256") != digest or call.get("response_privacy_redactions")
                or provenance.get("delivery_to_worker") != "not_proved"
                or provenance.get("agent_receipt") not in (None, "not_reported")
                or any(provenance.get(key) for key in ("worker_receipt_ref", "worker_error_observation_ref",
                    "agent_receipt_refs", "selected_response_acknowledged", "worker_error_body_acknowledged",
                    "worker_received_receipt_entry_ids", "worker_body_source", "worker_body_exact_in_artifact"))):
            return False
        provider = body.get("provider") or {}
        if (provider.get("only") != [call["reasoning_capture"]["provider_route"]]
                or provider.get("allow_fallbacks") is not False or provider.get("zdr") is not True
                or provider.get("data_collection") != "deny" or provider.get("require_parameters") is not True
                or not (_legacy_tested_controls_match(call)
                        or _captured_reasoning_controls_match(call, registry))):
            return False
        if not internal:
            settlements = [row for row in call.get("lifecycle") or []
                           if row.get("entry_id") == terminal_id and row.get("entry_kind") == "settlement"]
            if (len(settlements) != 1 or type(settlements[0].get("amount_microusd")) is not int
                    or settlements[0]["amount_microusd"] != 0):
                return False
        receipts = [row for row in call.get("response_receipts") or [] if row.get("entry_id") == response_id]
        if (len(receipts) != 1 or receipts[0].get("response_sha256") != digest
                or receipts[0].get("wire_body_sha256") != digest
                or receipts[0].get("wire_body_bytes") != len(contracts.canonical_json(expected_error).encode())
                or (receipts[0].get("metadata") or {}).get("status") != expected_status
                or (receipts[0].get("metadata") or {}).get("call_succeeded") is not False
                or receipts[0].get("response_ref") != {"entry_id": response_id, "field": "response"}
                or receipts[0].get("response_unavailable") or receipts[0].get("privacy_redactions")):
            return False
        if internal and (receipts[0].get("entry_kind") != ("uncertain" if response_id != terminal_id else accounting)
                or (receipts[0].get("metadata") or {}).get("account_failure_evidence") != {
                    "error_class": "account_credential_failure", "provider_status": 429,
                    "action_sequence": sequence, "provider_attempt": attempt, "base_call_identity": base}):
            return False
        upstream = call.get("upstream_response_receipts") or []
        if not upstream:
            return False
        for receipt in upstream:
            response = receipt.get("response")
            if (receipt.get("status") != 429 or not isinstance(response, Mapping)
                    or type(receipt.get("entry_id")) is not int or not 0 < receipt["entry_id"] <= response_id
                    or not isinstance(response.get("error"), Mapping) or response["error"].get("code") != 429
                    or response.get("choices") or response.get("output")
                    or receipt.get("provenance") != "private_upstream_response_receipt"
                    or receipt.get("delivery_to_worker") != "not_proved"
                    or receipt.get("privacy_redactions") or not training_steps._exact(response, response=True)
                    or receipt.get("response_sha256") != contracts.document_hash(response)
                    or not isinstance(receipt.get("body_sha256"), str)
                    or contracts.SHA256_RE.fullmatch(receipt["body_sha256"]) is None
                    or type(receipt.get("body_bytes")) is not int or receipt["body_bytes"] < 1
                    or receipt.get("body_sha256") != receipt.get("wire_body_sha256")
                    or receipt.get("body_bytes") != receipt.get("wire_body_bytes")):
                return False
        payload = ((example.get("worker_observation_artifacts") or {}).get(call["run_id"]) or {}).get("payload") or {}
        if not _provider_capture_complete(example, call["run_id"], worker_proofs=worker_proofs) or any(row.get("call_identity") == call.get("call_identity")
               for row in payload.get("response_receipts") or []):
            return False
        for recovered in example.get("provider_calls") or []:
            later_request = recovered.get("request") or {}
            recovered_amount = recovered.get("actual_microusd")
            if internal:
                later_attempt = recovered.get("provider_attempt")
                recovered_accounting = recovered.get("accounting_state")
                heads = [row for row in recovered.get("lifecycle") or []
                         if row.get("entry_id") == recovered.get("terminal_entry_id")]
                if (recovered.get("base_call_identity") != base
                        or recovered.get("provider") != "openrouter"
                        or recovered.get("funding_source") != "miner_key"
                        or type(later_attempt) is not int or later_attempt <= attempt
                        or recovered.get("call_identity") != contracts.document_hash(
                            {"base_call_identity": base, "provider_attempt": later_attempt})
                        or recovered.get("action_sequence") != sequence
                        or type(recovered.get("reservation_entry_id")) is not int
                        or recovered["reservation_entry_id"] <= response_id
                        or type(recovered.get("terminal_entry_id")) is not int
                        or recovered["terminal_entry_id"] <= recovered["reservation_entry_id"]
                        or not (recovered_accounting == "uncertain" and "actual_microusd" in recovered
                                and recovered_amount is None
                                or recovered_accounting == "settlement" and type(recovered_amount) is int
                                and recovered_amount >= 0)
                        or len(heads) != 1 or heads[0].get("entry_kind") != recovered_accounting
                        or type(heads[0].get("amount_microusd")) is not int
                        or heads[0]["amount_microusd"] != (0 if recovered_amount is None else recovered_amount)
                        or ("provider_cost_unresolved" in (recovered.get("evidence_gaps") or []))
                            != (recovered_accounting == "uncertain")):
                    continue
            elif (type(recovered.get("action_sequence")) is not int
                  or recovered["action_sequence"] <= sequence
                  or recovered.get("accounting_state") != "settlement"
                  or type(recovered_amount) is not int or recovered_amount < 0):
                continue
            if (recovered.get("run_id") != call["run_id"] or recovered.get("call_identity") == call.get("call_identity")
                    or recovered.get("operation_id") != operation
                    or type(recovered.get("terminal_entry_id")) is not int or recovered["terminal_entry_id"] <= response_id
                    or recovered.get("request_hash") != call.get("request_hash")
                    or later_request.get("hashed_parameters") != request["hashed_parameters"]
                    or (later_request.get("outbound") or {}).get("body") != body
                    or (later_request.get("outbound") or {}).get("body_sha256") != outbound["body_sha256"]
                    or recovered.get("response_status") != 200
                    or not _policy(recovered, registry)):
                continue
            # Recovery proves a delivered provider response, not target quality.
            # Its own eligibility checks still validate or omit a cap-hit target.
            evidence = training_steps._host_result(example, recovered, worker_proofs=worker_proofs)
            if (evidence["write"]["status"] != 200 or evidence["response"] != recovered.get("response")
                    or contracts.document_hash(evidence["response"]) != recovered.get("response_sha256")):
                continue
            state = attempt_coverage(recovered, _role(example, recovered), registry)
            if state["route_reviewed"] and state["provider_route_verified"]:
                return True
        return False
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _terminal_accounting_matches(call, kind, amount) -> bool:
    """Check the terminal ledger projection for the two coverage exceptions."""
    lifecycle = call.get("lifecycle")
    if not isinstance(lifecycle, list) or not all(isinstance(row, Mapping) for row in lifecycle):
        return False
    terminal_id = call.get("terminal_entry_id")
    terminals = [row for row in lifecycle if row.get("entry_id") == terminal_id
                 or row.get("entry_kind") in {"settlement", "uncertain", "refusal"}]
    return (type(terminal_id) is int and terminal_id > 0 and len(terminals) == 1
            and type(terminals[0].get("entry_id")) is int and terminals[0]["entry_id"] == terminal_id
            and terminals[0].get("entry_kind") == kind
            and type(terminals[0].get("amount_microusd")) is int
            and terminals[0]["amount_microusd"] == amount)


def _verified_unavailable_upstream_failure(example, call, *, worker_proofs=None) -> bool:
    """Prove the observed no-upstream-response bridge error, not a free call."""
    configuration = example.get("configuration") or {}
    if (training_coverage.MISSING_REASONING_SETTING not in configuration
            or not training_coverage.uses_host_result_evidence(configuration)
            or call.get("accounting_state") != "uncertain"
            or "actual_microusd" not in call
            or call.get("actual_microusd") is not None
            or not _terminal_accounting_matches(call, "uncertain", 0)
            or type(call.get("terminal_entry_id")) is not int or call["terminal_entry_id"] < 1
            or any(key in call for key in ("response", "response_sha256", "response_status"))
            or any(key in call and (not isinstance(call[key], list) or call[key])
                   for key in ("upstream_response_receipts", "response_privacy_redactions"))):
        return False
    receipts = call.get("response_receipts")
    if not isinstance(receipts, list) or len(receipts) != 1:
        return False
    receipt = receipts[0]
    if (not isinstance(receipt, Mapping)
            or not set(receipt) <= {"created_at", "entry_id", "entry_kind", "metadata", "response_unavailable"}
            or type(receipt.get("entry_id")) is not int or receipt["entry_id"] != call["terminal_entry_id"]
            or receipt.get("entry_kind") != "uncertain" or receipt.get("metadata") != {}
            or receipt.get("response_unavailable") is not True):
        return False
    try:
        evidence = training_steps._host_result(example, call, allow_worker_error=True, worker_proofs=worker_proofs)
        return (evidence["write"]["status"] == 502 and not evidence["receipts"]
                and evidence["response"] == {"error": {"code": "provider_unavailable"}})
    except (KeyError, TypeError, ValueError, AttributeError, binascii.Error):
        return False


def verified_undispatched_transport(example, call, *, require_observation_ref=True, worker_proofs=None) -> bool:
    """Prove the retained provider request never dispatched; invent no reply."""
    from lab_arena import training_observations
    operation = operations.get_operation(call.get("operation_id"))
    if (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
            or call.get("run_id") != example.get("run_id")
            or operation is None or call.get("operation_id") not in {"deepline.execute", *reasoning_capture.OPERATIONS}
            or call.get("provider") != operation.provider
            or call.get("funding_source") not in {"host", "miner_key"}
            or call.get("accounting_state") != "recovery"
            or type(call.get("actual_microusd")) is not int or call["actual_microusd"] != 0
            or not set(call.get("evidence_gaps") or []) <= {"full_response_unavailable"}
            or any(key in call for key in ("response", "response_sha256", "response_status", "provider_cost"))
            or call.get("upstream_response_receipts") or call.get("response_privacy_redactions")
            or call.get("response_provenance") != {
                "source": "stored_provider_ledger_receipt", "delivery_to_worker": "not_proved"}):
        return False
    try:
        identity, sequence = call["call_identity"], call["action_sequence"]
        request = call["request"]
        capture = call["request_capture"]
        outbound = request["outbound"]
        if (not isinstance(capture, Mapping) or capture.get("schema_version") != training_capture.SCHEMA_VERSION
                or capture.get("status") not in {"stored", "referenced"} or capture.get("encoding") != "gzip"
                or not isinstance(capture.get("sha256"), str)
                or contracts.SHA256_RE.fullmatch(capture["sha256"]) is None
                or capture.get("ref") != training_capture.OBJECT_PREFIX + capture["sha256"][7:] + ".json.gz"
                or capture.get("redacted") is not False or capture.get("redactions") != []
                or request.get("schema_version") != training_capture.SCHEMA_VERSION
                or request.get("privacy") != {"redacted": False, "redactions": []}
                or contracts.document_hash(request) != capture["sha256"]
                or type(capture.get("size_bytes")) is not int
                or capture["size_bytes"] != len(contracts.canonical_json(request).encode())
                or not 0 < capture["size_bytes"] <= training_capture.MAX_OBJECT_BYTES
                or type(capture.get("compressed_bytes")) is not int
                or not 0 < capture["compressed_bytes"] <= training_capture.MAX_COMPRESSED_BYTES
                or not isinstance(outbound, Mapping) or not isinstance(outbound.get("body"), Mapping)
                or not isinstance(outbound.get("query"), Mapping) or outbound.get("content_type") != "application/json"
                or not training_steps._outbound_request_exact(call)):
            return False
        lifecycle = call["lifecycle"]
        if (not isinstance(identity, str) or contracts.SHA256_RE.fullmatch(identity) is None
                or type(sequence) is not int or sequence < 0
                or len(lifecycle) != 2 or [row["entry_kind"] for row in lifecycle] != ["reservation", "recovery"]
                or any(type(row["entry_id"]) is not int or row["entry_id"] < 1 for row in lifecycle)
                or lifecycle[0]["entry_id"] >= lifecycle[1]["entry_id"]
                or type(call.get("reservation_entry_id")) is not int or type(call.get("terminal_entry_id")) is not int
                or call["reservation_entry_id"] != lifecycle[0]["entry_id"]
                or call["terminal_entry_id"] != lifecycle[1]["entry_id"]
                or any(row.get("funding_source") != call.get("funding_source") for row in lifecycle)
                or type(lifecycle[0]["amount_microusd"]) is not int or lifecycle[0]["amount_microusd"] < 0
                or type(lifecycle[1]["amount_microusd"]) is not int or lifecycle[1]["amount_microusd"] != 0
                or lifecycle[1]["accounting"] != {"reason": "attempt_completed_before_dispatch",
                    "reserved_microusd": lifecycle[0]["amount_microusd"]}
                or type(lifecycle[1]["accounting"].get("reserved_microusd")) is not int
                or any(lifecycle[0]["accounting"].get(key) != call.get(key) for key in (
                    "request_hash", "action_sequence", "base_call_identity", "provider_attempt"))
                or type(call.get("provider_attempt")) is not int or call["provider_attempt"] < 1
                or not isinstance(call.get("base_call_identity"), str)
                or contracts.SHA256_RE.fullmatch(call["base_call_identity"]) is None
                or (call.get("request_capture") or {}).get("redacted")
                or not training_steps._exact(request)
                or request.get("operation_id") != call["operation_id"]
                or request.get("request_hash") != call.get("request_hash")
                or not isinstance(request.get("hashed_parameters"), Mapping)
                or contracts.document_hash(request["hashed_parameters"]) != call.get("request_hash")):
            return False
        receipt = call["response_receipts"]
        if (not isinstance(receipt, list) or len(receipt) != 1
                or receipt[0] != {"created_at": lifecycle[1].get("created_at"),
                    "entry_id": call["terminal_entry_id"], "entry_kind": "recovery",
                    "metadata": {}, "response_unavailable": True}):
            return False
        artifact = (example.get("worker_observation_artifacts") or {})[call["run_id"]]
        payload = artifact["payload"]
        if (set(artifact) != {"schema_version", "run_id", "kind", "lease_generation",
                              "claim_request_hash", "payload", "payload_sha256"}
                or artifact["schema_version"] != training_observations.SCHEMA_VERSION
                or artifact["run_id"] != call["run_id"] or artifact["kind"] != "execute"
                or type(artifact["lease_generation"]) is not int or artifact["lease_generation"] < 1
                or not isinstance(artifact["claim_request_hash"], str)
                or contracts.SHA256_RE.fullmatch(artifact["claim_request_hash"]) is None
                or artifact["payload_sha256"] != training_worker_proof.payload_hash(payload, worker_proofs)
                or training_observations._validate_payload(payload) != payload
                or any(row["call_identity"] == identity for row in payload["response_receipts"])
                or any(row["kind"] == "response_received" and row["data"]["call_identity"] == identity
                       for row in payload["agent_observations"])):
            return False
        observations = [row for row in payload["observations"]
                        if row["action_sequence"] == sequence or row["call_identity"] == identity]
        if not observations:
            # SQL closes only an unused reservation under the same run lock
            # as dispatch. That exact ledger chain proves no provider call;
            # a worker exception is optional, and no reply is manufactured.
            return "worker_observation_ref" not in call
        if len(observations) != 1:
            return False
        row = observations[0]
        ref = {"run_id": call["run_id"], "ordinal": row["ordinal"], "payload_sha256": artifact["payload_sha256"]}
        outcome = row["outcome"]
        exception_only = set(outcome) == {"protocol", "exception_class"}
        local_error = (set(outcome) == {"protocol", "exception_class", "error_code", "response"}
                       and outcome["error_code"] == "worker_unavailable"
                       and (outcome["protocol"] == "frame" and outcome["response"] == {"error": "worker_unavailable"}
                            or outcome["protocol"] == "http" and outcome["response"] == {
                                "status": 503, "headers": {}, "body_b64": base64.b64encode(json.dumps(
                                    {"error": {"code": "worker_unavailable"}}, sort_keys=True).encode()).decode()}))
        return (row["action_sequence"] == sequence
                and row["kind"] == "transport_failure" and row["operation_id"] == call["operation_id"]
                and row["call_identity"] in (None, identity)
                and row["contributes_to_provider_call_count"] is True and row["privacy"]["redacted"] is False
                and row["request"]["admitted_parameters"] == request.get("admitted_parameters")
                and (exception_only or local_error) and bool(outcome["exception_class"].strip())
                and (not require_observation_ref and "worker_observation_ref" not in call
                     or call.get("worker_observation_ref") == ref))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def verified_failed_attempt(example: Mapping[str, Any], call: Mapping[str, Any], *, worker_proofs=None) -> bool:
    """Prove no generation or a delivered error without making a model target.

    Provider errors need no observed generation route: no generation is
    claimed. The reviewed request, saved failure bytes, host write and agent
    read must agree. Unsettled billing does not establish whether the delivered bytes are usable.
    """
    if (verified_recovered_provider_retry(example, call, worker_proofs=worker_proofs)
            or verified_undispatched_transport(example, call, worker_proofs=worker_proofs)):
        return True
    try:
        run_id = example["run_id"]
        # Both delivery policies reject the referenced write's status 200.
        # This is only a negative filter; no response or target is proved here.
        try:
            provenance = call.get("response_provenance") or {}
            ref = provenance.get("worker_receipt_ref") or {}
            artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
            writes = [row for row in (artifact.get("payload") or {}).get("response_receipts") or []
                      if row.get("ordinal") == ref.get("ordinal")]
            if (call.get("run_id") == ref.get("run_id") == run_id
                    and call.get("provider") == "openrouter" and call.get("operation_id") in reasoning_capture.OPERATIONS
                    and type(call.get("action_sequence")) is int and call["action_sequence"] >= 0
                    and type(ref.get("ordinal")) is int and ref["ordinal"] >= 0
                    and isinstance(ref.get("payload_sha256"), str)
                    and contracts.SHA256_RE.fullmatch(ref["payload_sha256"]) is not None
                    and ref["payload_sha256"] == artifact.get("payload_sha256")
                    and provenance.get("delivery_to_worker") == "socket_written" and len(writes) == 1
                    and writes[0].get("write_state") == "socket_written"
                    and type(writes[0].get("status")) is int and writes[0]["status"] == 200
                    and all(writes[0].get(key) == call.get(key)
                            for key in ("call_identity", "operation_id", "action_sequence"))
                    and type(writes[0].get("body_bytes")) is int and writes[0]["body_bytes"] >= 0
                    and isinstance(writes[0].get("body_sha256"), str)
                    and contracts.SHA256_RE.fullmatch(writes[0]["body_sha256"]) is not None):
                return False
        except (KeyError, TypeError, ValueError, AttributeError):
            pass  # Malformed or ambiguous references keep the full proof path.
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
        registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
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
                or not set(call.get("evidence_gaps") or []) <= (
                    {"provider_cost_unresolved", "full_response_unavailable"}
                    if "full_response_unavailable" in (call.get("evidence_gaps") or [])
                    and _verified_unavailable_upstream_failure(example, call, worker_proofs=worker_proofs)
                    else {"provider_cost_unresolved"})
                or (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
                    and set((example.get("completeness") or {}).get("incomplete_reasons") or []) & {
                        "worker_response_receipt_link_invalid", "agent_response_receipt_link_invalid",
                        "agent_dependency_link_invalid", "runtime_input_context_mismatch"})):
            return False
        if training_coverage.uses_host_result_evidence(example.get("configuration") or {}):
            evidence = training_steps._host_result(example, call, allow_worker_error=True, worker_proofs=worker_proofs)
            status = evidence["write"]["status"]
            if status not in ({402} if admission_refusal else {404, 429, 500, 502, 503, 504}):
                return False
            failure_bodies = [evidence["response"]]
            if admission_refusal:
                return (not evidence["receipts"]
                        and evidence["response"] == {"error": {"code": "budget_refused"}})
        else:
            provenance = call.get("response_provenance") or {}
            artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
            payload = artifact.get("payload") or {}
            payload_hash = training_worker_proof.payload_hash(payload, worker_proofs)
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
            if (type(status) is not int or status not in ({402} if admission_refusal else {404, 429, 500, 502, 503, 504})
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


def _incomplete_json_object_prefix(value: str) -> bool:
    """Recognize strict JSON cut at EOF; never add bytes or produce a target.

    The standard decoder reports an unfinished string or scalar at its start,
    rather than at EOF. Its first syntax error still proves the preceding
    document valid. Accept only an unfinished token that occupies the entire
    remaining suffix. A complete object, invalid token, or trailing garbage
    cannot use this omission rule.
    """
    if not isinstance(value, str) or not value.lstrip(" \t\r\n").startswith("{"):
        return False
    try:
        json.loads(value, parse_constant=training_steps._reject_constant)
    except json.JSONDecodeError as exc:
        suffix = value[exc.pos:]
        if exc.pos == len(value):
            return True
        if exc.msg == "Unterminated string starting at":
            # JSON's string scanner already rejected bad escapes and controls.
            return True
        if exc.msg == "Invalid \\uXXXX escape":
            return re.fullmatch(r"u[0-9a-fA-F]{0,4}", suffix) is not None
        if exc.msg == "Expecting value":
            return suffix == "-" or any(
                token.startswith(suffix) and len(suffix) < len(token)
                for token in ("true", "false", "null"))
        if exc.msg == "Expecting ',' delimiter":
            # A number decoder stops before an unfinished decimal/exponent.
            # Verify its left token too, so a closed string or container with
            # the same suffix cannot masquerade as a partial number.
            number_token = r"-?(?:0|[1-9][0-9]*)"
            if suffix != ".":
                number_token += r"(?:\.[0-9]+)?"
            number = re.search(r"(?:^|[\[,:\s])(" + number_token + r")\Z", value[:exc.pos])
            return number is not None and re.fullmatch(r"\.|[eE][+-]?", suffix) is not None
        return False
    except (ValueError, RecursionError):
        return False
    return False


def verified_private_explicit_truncation(example, call, *, worker_proofs=None) -> bool:
    """Discard a proved explicit cap stop; never admit its private response.

    A redacted source excerpt can coexist with a provider's explicit truncation
    status. The entire turn stays outside targets and reasoning-token totals.
    Its exact request, input history, selected route and host-written receipt
    must still be proved. Later requests keep their ordinary privacy checks.
    """
    if (not training_coverage.uses_host_result_evidence(example.get("configuration") or {})
            or call.get("operation_id") != "openrouter.responses"
            or not isinstance(call.get("response"), Mapping)
            or call["response"].get("status") != "incomplete"
            or call["response"].get("incomplete_details") != {"reason": "max_output_tokens"}
            or call["response"].get("error")
            or "credential" not in (call.get("response_privacy_redactions") or [])
            or not set(call.get("evidence_gaps") or []) <= {
                "response_privacy_nonexact", "provider_cost_unresolved"}):
        return False
    try:
        operation, body, _ = training_steps._model_request(example, call, worker_proofs=worker_proofs)
        evidence = training_steps._host_result(example, call, allow_privacy=True, worker_proofs=worker_proofs)
        response = evidence["response"]
        if (evidence["write"]["status"] != 200 or response != call["response"]
                or contracts.document_hash(response) != call.get("response_sha256")):
            return False
        registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
        state = attempt_coverage(call, _role(example, call), registry)
        if (not state["requested"] or not state["route_reviewed"]
                or not state["provider_route_verified"]
                or not (_legacy_tested_controls_match(call)
                        or _captured_reasoning_controls_match(call, registry))):
            return False
        try:
            # This preserves input-history and retrieval checks before the
            # ordinary explicit-incomplete branch. It never returns a target.
            training_steps._target(body, response, operation, allow_reasoning_history=True,
                                   allow_tool_error_history=True)
        except ValueError as exc:
            return str(exc) == "response_not_complete"
        return False
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def verified_cap_truncated_target(example, call, *, worker_proofs=None) -> bool:
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
        evidence = training_steps._host_result(example, call, worker_proofs=worker_proofs)
        response = evidence["response"]
        usage = response.get("usage") or {}
        checked = dict(call, response=response, response_sha256=contracts.document_hash(response))
        state = attempt_coverage(checked, _role(example, call),
                                 baseline_reasoning.for_example(example).get("reasoning_route_registry"))
        if (operation not in reasoning_capture.OPERATIONS or evidence["write"]["status"] != 200
                or type(cap) is not int or cap < 1
                or not (_legacy_tested_controls_match(call)
                        or _captured_reasoning_controls_match(call,
                            baseline_reasoning.for_example(example).get("reasoning_route_registry")))
                or type(usage.get(usage_key)) is not int or usage[usage_key] != cap
                or not state["requested"] or not state["route_reviewed"]
                or not state["provider_route_verified"] or state["redactions"]):
            return False
        try:
            training_steps._candidate(example, call, worker_proofs=worker_proofs)
        except ValueError as exc:
            if str(exc) != "tool_call_shape_unsupported":
                return False
        else:
            return False  # A valid generation at its cap remains a valid target.
        unfinished = []

        def argument_shape(value):
            if training_steps._arguments(value):
                return True
            if _incomplete_json_object_prefix(value):
                unfinished.append(value)
                return True
            return False

        # This validates the unchanged raw target's remaining shape only. It
        # never returns a training candidate or inserts missing JSON syntax.
        target, _ = training_steps._target(body, response, operation,
            allow_reasoning_history=True, argument_validator=argument_shape,
            allow_tool_error_history=True, allow_empty_messages=True)
        training_steps._validate_target_tools(body, target, operation)
        # A cut must be the final generated action. Reasoning and empty
        # response frames may follow in provider output without proving that
        # another action was completed. All frames were validated above.
        if len(unfinished) != 1:
            return False
        if operation == "openrouter.chat":
            calls = target['tool_calls']
            if any('index' in item for item in calls):
                indexes = [item.get('index') for item in calls]
                if (any(type(index) is not int for index in indexes)
                        or len(set(indexes)) != len(indexes)):
                    return False
                final = max(calls, key=lambda item: item['index'])
            else:
                final = calls[-1]
            return final['function']['arguments'] == unfinished[0]
        actions = [item for item in response['output'] if item['type'] != 'reasoning'
                   and not training_steps._empty_target_message(item)]
        final = actions[-1]
        return final.get("type") == "function_call" and final.get("arguments") == unfinished[0]
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
        registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
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
            full_round = baseline_reasoning.for_example(example).get("reasoning_capture_policy") == POLICY
            failed = new_policy and verified_failed_attempt(example, call)
            omitted_truncation = False
            target_usable = not new_policy
            if new_policy and execution_model and not failed:
                try:
                    training_steps._candidate(example, call)
                    target_usable = True
                except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
                    private_truncation = verified_private_explicit_truncation(example, call)
                    omitted_truncation = private_truncation or verified_cap_truncated_target(example, call) or (
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
                row["target_omission_reason"] = ("explicit_output_cap_private_response" if private_truncation
                                                 else "inferred_output_cap_json_eof")
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


def readable_trace(example: Mapping[str, Any], *, worker_proofs=None) -> dict:
    """Index a private export for review without replacing its complete raw JSON.

    Paths point into the supplied, hash-verified export. No text is shortened.
    Provider receipts are distinct from the response confirmed as delivered.
    """
    if (not isinstance(example, Mapping)
            or example.get("example_hash") != contracts.document_hash(
                {key: value for key, value in example.items() if key != "example_hash"})):
        raise ValueError("example_hash_mismatch")
    registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
    calls = []
    for ci, call in enumerate(example.get("provider_calls") or []):
        prefix = ["provider_calls", ci]
        body = ((call.get("request") or {}).get("outbound") or {}).get("body")
        events = []

        def add(kind, path, value, *, delivery=None, reasoning_field=None, alias_paths=(), execution=None):
            events.append({"kind": kind, "source_path": path, "value": copy.deepcopy(value),
                           "value_sha256": contracts.document_hash(value),
                           **({"delivery": delivery} if delivery else {}),
                           **({"reasoning_field": reasoning_field} if reasoning_field else {}),
                           **({"execution": execution} if execution else {}),
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
            first_event = len(events)
            operation = call.get("operation_id")
            for key in sorted(response):
                if key not in {"choices", "output", "usage"}:
                    add("response_metadata", path + [key], response[key], delivery=delivery)
            segments, summary, opaque = _segments(response, operation)
            full, _ = full_text_segments({**call, "response": response}, _role(example, call), registry)
            def add_segments(values):
                for segment in values:
                    classification = ("summary" if "summary" in segment["field"] else
                                      "full_generated_thinking" if (delivery == "selected_terminal"
                                                                     and segment in full) else
                                      "unverified_reasoning_text")
                    sources = segment["source_paths"]
                    add(classification, path + sources[0], segment["text"], delivery=delivery,
                        reasoning_field=segment["field"],
                        alias_paths=[path + source for source in sources[1:]])

            if operation != "openrouter.responses":
                add_segments(segments)
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
                        if isinstance(message.get("annotations"), list) and message["annotations"]:
                            add("citation_annotations", base + ["annotations"], message["annotations"], delivery=delivery)
                        for tool_index, tool in enumerate(message.get("tool_calls") or []):
                            add("tool_call", base + ["tool_calls", tool_index], tool, delivery=delivery)
            elif operation == "openrouter.responses":
                for output_index, item in enumerate(response.get("output") or []):
                    kind = item.get("type") if isinstance(item, Mapping) else None
                    hosted = kind in training_steps._HOSTED_SEARCH_TYPES
                    event_kind = ("assistant_output" if kind == "message" else
                                  "provider_hosted_tool" if hosted else
                                  "tool_call" if kind in {"function_call", "custom_tool_call"} else
                                  "tool_result" if kind in {"function_call_output", "custom_tool_call_output"} else
                                  "reasoning_item" if kind == "reasoning" else "provider_output_item")
                    # Whole native items preserve IDs, status, citations and
                    # exposed search sources. The list order is the provider's
                    # wire order; no internal tool response is synthesized.
                    add(event_kind, path + ["output", output_index], item, delivery=delivery,
                        execution="provider" if hosted else None)
                    add_segments([segment for segment in segments
                                  if segment["source_paths"][0][:2] == ["output", output_index]])

            # Retain fields outside the typed events (for example native Chat
            # finish reasons and reasoning-detail format/index). Skip whole
            # indexed values and text aliases so large reasoning is not copied
            # again. Walk the saved key/list order without moving typed events.
            indexed = {tuple(source)
                       for event in events[first_event:]
                       for source in ([event["source_path"]] if "source_path" in event else [])
                       + event.get("alias_source_paths", [])}

            def add_unindexed(value, source):
                if tuple(source) in indexed:
                    return
                if isinstance(value, Mapping) and value:
                    for key, item in value.items():
                        add_unindexed(item, source + [key])
                elif isinstance(value, list) and value:
                    for index, item in enumerate(value):
                        add_unindexed(item, source + [index])
                else:
                    add("response_metadata", source, value, delivery=delivery)

            add_unindexed(response, path)

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
                      **({"target_omission_reason": "explicit_output_cap_private_response"}
                         if verified_private_explicit_truncation(example, call, worker_proofs=worker_proofs) else {}),
                      **({"target_omission_reason": "inferred_output_cap_json_eof"}
                         if verified_cap_truncated_target(example, call, worker_proofs=worker_proofs) else {}),
                      "actual_microusd": call.get("actual_microusd"),
                      "provider_cost": copy.deepcopy(call.get("provider_cost")),
                      "lifecycle": copy.deepcopy(call.get("lifecycle")),
                      "response_receipts": receipts,
                      "upstream_response_receipts": copy.deepcopy(call.get("upstream_response_receipts")),
                      "evidence_gaps": copy.deepcopy(call.get("evidence_gaps") or []),
                      "events": events})
    observations = []
    for field in ("worker_observation_artifacts", "partial_worker_observation_artifacts",
                  "unverified_worker_observation_artifacts"):
        for run_id, artifact in sorted((example.get(field) or {}).items()):
            payload = artifact.get("payload") if isinstance(artifact, Mapping) else None
            if not isinstance(payload, Mapping):
                continue
            if field == "unverified_worker_observation_artifacts" and artifact.get("unverified_reasons"):
                reasons = artifact["unverified_reasons"]
                observations.append({"run_id": run_id, "kind": "unverified_snapshot",
                                     "source_path": [field, run_id, "unverified_reasons"],
                                     "value": copy.deepcopy(reasons),
                                     "value_sha256": contracts.document_hash(reasons),
                                     "payload_sha256": artifact.get("payload_sha256")})
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
                "partial_worker_observation_artifacts", "unverified_worker_observation_artifacts", "example_hash"}
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
                ["partial_worker_observation_artifacts", key] for key in sorted(example.get("partial_worker_observation_artifacts") or {})],
            **({"unverified_worker_observation_paths": [
                ["unverified_worker_observation_artifacts", key]
                for key in sorted(example["unverified_worker_observation_artifacts"])]}
               if example.get("unverified_worker_observation_artifacts") else {})}


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
                *([f"Execution: `{event['execution']}`  "] if event.get("execution") else []),
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
    artifact = training_steps._worker_artifact(example, call, allow_journal_prefix=True)
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


def _selected_input_output_binding(example: Mapping[str, Any]) -> set[str] | None:
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
    selected_runs = [row for row in attempts if row.get("run_id") == run_id]
    score_runs = [row for row in attempts if row.get("run_id") == score_id
                  and row.get("kind") == "score" and row.get("status") == "accepted"]
    if (len(score_runs) != 1 or len(selected_runs) != 1
            or selected_runs[0].get("kind") != "execute" or selected_runs[0].get("status") != "accepted"):
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
    return score_ids


def _selected_research_evidence(example: Mapping[str, Any], *, worker_proofs=None) -> frozenset[str] | None:
    """Bind selected source and labels without an all-attempt or billing veto.

    Each candidate separately proves its exact provider input, required tool
    history and delivered result. The eligibility gate aggregates those checks
    into ICP coverage; one unusable target cannot erase another valid example.
    """
    try:
        configuration = example.get("configuration") or {}
        training_coverage.missing_reasoning_percent_from_configuration(configuration)
        training_coverage.percent_from_configuration(configuration)
        if not reasoning_capture.enabled(baseline_reasoning.for_example(example)):
            return None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if _selected_input_output_binding(example) is None:
        return None
    proof = example.get("completeness") or {}
    dimensions = proof.get("dimensions") or {}
    by_dimension = proof.get("reasons_by_dimension") or {}
    source_verified = (dimensions.get("source_identity") is True
                       and by_dimension.get("source_identity") == [])
    # The validator checkout marker is best-effort operational metadata, not
    # proof of the miner archive or captured bytes. Retain its warning while
    # requiring every other source, scorer, input/output and privacy check.
    checkout_diagnostic_only = (
        dimensions.get("source_identity") is False
        and by_dimension.get("source_identity") == ["runtime_source_unverified"]
        and proof.get("complete") is False
        and isinstance(proof.get("incomplete_reasons"), list)
        and all(isinstance(reason, str) for reason in proof["incomplete_reasons"])
        and all(isinstance(value, list) and all(isinstance(reason, str) for reason in value)
                for value in by_dimension.values())
        and set(proof["incomplete_reasons"]) == set().union(*(set(value) for value in by_dimension.values()))
        and all(dimensions.get(name) is (not value) for name, value in by_dimension.items()))
    if (dimensions.get("context") is not True or by_dimension.get("context") != []
            or not (source_verified or checkout_diagnostic_only)):
        return None
    return frozenset(proof.get("incomplete_reasons") or [])


_LOCAL_WEB_DIAGNOSTICS = frozenset({"direct_web_usage_unverified", "direct_web_content_not_captured",
                                   "direct_web_coverage_unverified"})
_OPTIONAL_LOCAL_KINDS = frozenset({"web_fetch", "web_fetch_binary", "http_response_metadata", "state_snapshot"})


def _provider_capture_complete(example, run_id, *, worker_proofs=None):
    """Keep old failures hard; only new explicit optional tags narrow capture scope."""
    from lab_arena import training_observations
    artifact = (example.get("worker_observation_artifacts") or {}).get(run_id) or {}
    payload = artifact.get("payload") or {}
    if payload.get("complete") is True:
        return not payload.get("failures")
    return (training_coverage.uses_host_result_evidence(example.get("configuration") or {})
            and set(artifact) == {"schema_version", "run_id", "kind", "lease_generation",
                                  "claim_request_hash", "payload", "payload_sha256"}
            and artifact.get("schema_version") == training_observations.SCHEMA_VERSION
            and artifact.get("run_id") == run_id
            and artifact.get("kind") in {"execute", "score"}
            and type(artifact.get("lease_generation")) is int and artifact["lease_generation"] > 0
            and isinstance(artifact.get("claim_request_hash"), str)
            and contracts.SHA256_RE.fullmatch(artifact["claim_request_hash"]) is not None
            and artifact.get("payload_sha256") == training_worker_proof.payload_hash(payload, worker_proofs)
            and training_observations.optional_capture_only(payload))


def _optional_capture_runs_proven(example, *, worker_proofs=None):
    """Every run that contributed the shared gap needs its own full host proof."""
    rows = example.get("related_run_gap_attribution") or {}
    attempts = {row.get("run_id"): row for row in example.get("attempts") or []}
    artifacts = example.get("worker_observation_artifacts") or {}
    if (not rows or set(rows) != set(attempts) or len(attempts) != len(example.get("attempts") or [])
            or not set(artifacts) <= set(rows) or example.get("partial_worker_observation_artifacts")):
        return False
    affected = []
    for run_id, row in rows.items():
        if (row.get("kind") != attempts[run_id].get("kind")
                or row.get("status") != attempts[run_id].get("status")):
            return False
        artifact = artifacts.get(run_id) or {}
        if ((artifact.get("payload") or {}).get("complete") is False
                and "worker_observations_incomplete" not in (row.get("research_process_gaps") or [])):
            return False
        if "worker_observations_incomplete" in (row.get("research_process_gaps") or []):
            if not _provider_capture_complete(example, run_id, worker_proofs=worker_proofs) or artifact.get("kind") != row.get("kind"):
                return False
            # A complete unrelated snapshot cannot explain an incomplete journal.
            payload = example["worker_observation_artifacts"][run_id]["payload"]
            if payload.get("complete") is not False:
                return False
            affected.append(run_id)
    for run_id in affected:
        calls = [call for call in example.get("provider_calls") or [] if call.get("run_id") == run_id]
        delivered = []
        for call in calls:
            if verified_recovered_provider_retry(example, call, worker_proofs=worker_proofs):
                continue
            training_steps._host_result(example, call, allow_worker_error=True, worker_proofs=worker_proofs)
            delivered.append(call["call_identity"])
        receipts = example["worker_observation_artifacts"][run_id]["payload"]["response_receipts"]
        identities = [row["call_identity"] for row in receipts]
        if len(delivered) != len(set(delivered)) or len(identities) != len(set(identities)) or set(identities) != set(delivered):
            return False
    return bool(affected)


def _required_observation_redacted(example, row, *, agent=False):
    return ((row.get("privacy") or {}).get("redacted") is not False
            and not (agent and row.get("kind") in _OPTIONAL_LOCAL_KINDS
                     and training_coverage.uses_host_result_evidence(example.get("configuration") or {})))


def _provider_visible_view(example, *, not_dispatched_host_observations=None, worker_proofs=None):
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
    if "worker_observations_incomplete" in reasons and _optional_capture_runs_proven(example, worker_proofs=worker_proofs):
        ignored.add("worker_observations_incomplete")
    artifacts = example.get("worker_observation_artifacts") or {}
    optional_redacted = False
    optional_binary = False
    private_local_host = False
    not_dispatched_host_observations = not_dispatched_host_observations or {}
    for artifact in artifacts.values():
        payload = artifact.get("payload") or {}
        if artifact.get("payload_sha256") != training_worker_proof.payload_hash(payload, worker_proofs):
            return example, frozenset()
        for row in payload.get("observations") or []:
            if _required_observation_redacted(example, row):
                # Eligibility alone supplies rows already proved by the exact
                # no-ledger host classifier. Keep their privacy flags intact.
                if (artifact.get("run_id") != example.get("run_id")
                        or row != not_dispatched_host_observations.get(row.get("call_identity"))):
                    return example, frozenset()
                private_local_host = True
        for row in payload.get("agent_observations") or []:
            if _required_observation_redacted(example, row, agent=True):
                return example, frozenset()
            optional_redacted |= (row.get("kind") in _OPTIONAL_LOCAL_KINDS
                                  and (row.get("privacy") or {}).get("redacted") is True)
            optional_binary |= row.get("kind") == "web_fetch_binary"
    if optional_redacted and not private_local_host and "worker_observation_privacy_nonexact" in reasons:
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


def provider_visible_view(example, *, not_dispatched_host_observations=None, worker_proofs=None):
    try:
        return _provider_visible_view(example, not_dispatched_host_observations=not_dispatched_host_observations, worker_proofs=worker_proofs)
    except (TypeError, ValueError, KeyError, AttributeError):
        return example, frozenset()  # Invalid diagnostics never grant an exception.


def selected_research_evidence(example: Mapping[str, Any], *, worker_proofs=None) -> frozenset[str] | None:
    # Raw diagnostic capture remains visible without becoming a second
    # admission gate for independently verified provider-visible pairs.
    view, diagnostic_gaps = provider_visible_view(example, worker_proofs=worker_proofs)
    if diagnostic_gaps and _selected_input_output_binding(view) is None:
        return None
    result = _selected_research_evidence(view, worker_proofs=worker_proofs)
    return None if result is None else result | diagnostic_gaps


def _combined_coverage(reports):
    result = coverage([])
    returned, trajectories, grouped = Counter(), Counter(), defaultdict(Counter)
    for report in reports:
        result["attempts"].extend(report["attempts"])
        returned.update(report["returned_classes"])
        trajectories.update(report["trajectories"])
        for key in ("distinct_verified_exports", "saved_calls", "generated_thinking_texts", "summary_calls"):
            result[key] += report[key]
        for row in report["by_role_model_route"]:
            key = tuple(row[name] for name in ("role", "model_id", "requested_provider_route", "api"))
            grouped[key].update(row["counts"])
    result["returned_classes"] = dict(sorted(returned.items()))
    result["trajectories"] = dict(sorted(trajectories.items()))
    result["by_role_model_route"] = [dict(zip(("role", "model_id", "requested_provider_route", "api"), key),
                                         counts=dict(sorted(value.items())))
                                      for key, value in sorted(grouped.items(), key=lambda item: str(item[0]))]
    return result


def build(examples, rules: training_steps.Rules | None = None):
    """Return private candidates only for proven full generated thinking."""
    rules = rules or training_steps.Rules()
    output = {"train": [], "eval": [], "analysis": []}
    records_by_split: dict[str, dict[str, dict]] = {"train": {}, "eval": {}}
    reasons = Counter()
    diagnostic_gaps_by_source: dict[str, frozenset[str]] = {}
    projected = {}
    projected_refs = set()
    scoped = {}
    coverage_reports = []
    coverage_sources = set()

    def project_candidate(example, candidate):
        # The shared builder calls this while this one source is in memory.
        # Global candidate deduplication and cluster assignment still run there.
        ref = candidate["source_refs"][0]
        ref_key = contracts.canonical_json(ref)
        if ref_key in projected_refs:
            return
        projected_refs.add(ref_key)
        call = next((row for row in example.get("provider_calls") or []
                     if row.get("call_identity") == ref["call_identity"] and row.get("run_id") == ref["run_id"]), None)
        if call is None:
            return
        source = ref["example_hash"]
        if source not in scoped:
            scoped[source] = selected_research_evidence(example)
        ancillary_gaps = scoped[source]
        if ancillary_gaps is None:
            reasons["trajectory_evidence_incomplete"] += 1
            return
        try:
            delivered = _delivered_response(call, ref["delivered_response_sha256"])
        except ValueError as exc:
            reasons[str(exc)] += 1
            return
        delivered_call = dict(call, response=delivered,
                              response_sha256=ref["delivered_response_sha256"])
        registry = baseline_reasoning.for_example(example).get("reasoning_route_registry")
        source = ref["example_hash"]
        segments, status = full_text_segments(delivered_call, _role(example, call), registry)
        if not segments:
            reasons[status["returned"]] += 1
            return
        try:
            prior = _prior_observations(example, call)
        except ValueError as exc:
            reasons[str(exc)] += 1
            return
        if source not in diagnostic_gaps_by_source:
            diagnostic_gaps_by_source[source] = provider_visible_view(example)[1]
        diagnostic_gaps = diagnostic_gaps_by_source[source]
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
        projected[contracts.canonical_json(ref)] = record

    def sources():
        for example in examples:
            source = example.get("example_hash")
            if source not in coverage_sources:
                report = coverage([example])
                if report["distinct_verified_exports"]:
                    coverage_sources.add(source)
                    coverage_reports.append(report)
            yield example
            del example

    splits, historical = training_steps._build(sources(), rules, full_thinking_candidate=project_candidate)
    for split in ("train", "eval"):
        for candidate in splits[split]:
            for ref in candidate["source_refs"]:
                record = projected.pop(contracts.canonical_json(ref), None)
                if record is None:
                    continue
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
                "split_source_hashes": historical["split_source_hashes"],
                "counts": {"records_by_split": {key: len(rows) for key, rows in output.items()},
                           "exclusion_reasons": dict(sorted(reasons.items()))},
                "coverage": _combined_coverage(coverage_reports)}
    manifest["dataset_version"] = contracts.document_hash(manifest)
    return output, manifest
