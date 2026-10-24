"""Read-only coverage and strict generated-thinking candidates from Arena exports.

The provider receipt and agent read proof remain in training_steps. A request
annotation establishes route semantics; returned text alone never does.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from lab_arena import contracts, reasoning_capture, training_capture, training_dataset, training_export, training_steps

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
               if entry["model_id"] == model and entry["api"] == operation]
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


def verified_tool_only_empty(call: Mapping[str, Any], delivered: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    if (not status["route_reviewed"] or not status["provider_route_verified"]
            or status["storage"] != "saved" or status["returned"] != "requested_empty"):
        return False
    if call.get("operation_id") == "openrouter.chat":
        choices = delivered.get("choices") or []
        return (len(choices) == 1 and choices[0].get("finish_reason") == "tool_calls"
                and bool((choices[0].get("message") or {}).get("tool_calls")))
    if call.get("operation_id") == "openrouter.responses":
        output = delivered.get("output") or []
        return bool(output) and any(item.get("type") == "function_call" for item in output) and not any(
            item.get("type") == "message" for item in output)
    return False


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
                if isinstance(value, str) and value.strip() and "[REDACTED]" not in value:
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
                    if isinstance(value, str) and value.strip() and "[REDACTED]" not in value:
                        segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri, "summary"],
                                         "field": "reasoning_details.summary", "text": value})
                    elif isinstance(value, list):
                        for pi, part in enumerate(value):
                            if (isinstance(part, Mapping) and part.get("type") == "summary_text"
                                    and isinstance(part.get("text"), str) and part["text"].strip()
                                    and "[REDACTED]" not in part["text"]):
                                segments.append({"source_path": ["choices", ci, "message", "reasoning_details", ri,
                                                                 "summary", pi, "text"],
                                                 "field": "reasoning_details.summary", "text": part["text"]})
                if training_capture.reasoning_kind(item) == "opaque":
                    opaque = True
                    continue
                for field in ("text", "thinking", "content"):
                    value = item.get(field)
                    field_id = "reasoning_details.text" if kind == "reasoning.text" and field == "text" else "reasoning_details.unproved"
                    if isinstance(value, str) and value.strip() and "[REDACTED]" not in value:
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
                            and isinstance(part.get("text"), str) and part["text"].strip()
                            and "[REDACTED]" not in part["text"]):
                        segments.append({"source_path": ["output", oi, "summary", pi, "text"],
                                         "field": "output.summary.summary_text", "text": part["text"]})
            if item.get("encrypted_content") or item.get("redacted_thinking"):
                opaque = True
            for field in ("text", "thinking", "content"):
                value = item.get(field)
                if isinstance(value, str) and value.strip() and "[REDACTED]" not in value:
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
    if isinstance(call.get("route_observation"), Mapping):
        return call["route_observation"]
    matching = [receipt for receipt in call.get("response_receipts") or []
                if receipt.get("response_sha256") == call.get("response_sha256")]
    if len(matching) == 1:
        return (matching[0].get("metadata") or {}).get("reasoning_route_observation")
    return None


def _execution_model_call(example: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    return (call.get("operation_id") in {"openrouter.chat", "openrouter.responses"}
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
    tokens = next((value for value in (usage.get("reasoning_tokens"), details.get("reasoning_tokens"),
                                      output_details.get("reasoning_tokens"))
                   if type(value) is int and value >= 0), None)
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
        registry = (example.get("configuration") or {}).get("reasoning_route_registry")
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
            has_text |= execution_model and bool(full)
            full_text_calls += execution_model and bool(full)
            full_round = (example.get("configuration") or {}).get("reasoning_capture_policy") == POLICY
            bad |= (execution_model and (row["requested"] or full_round) and
                    not (full or verified_tool_only_empty(call, call.get("response") or {}, row)))
        trajectories["ineligible"] += bad
        trajectories["with_full_thinking"] += has_text
        trajectories["eligible_without_disqualifying_capture"] += not bad
        trajectories["complete_full_thinking"] += bool(
            not bad and has_text and (example.get("completeness") or {}).get("complete") is True)
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


def build(examples, rules: training_steps.Rules | None = None):
    """Return private candidates only for proven full generated thinking."""
    rules = rules or training_steps.Rules()
    examples = list(examples)
    splits, historical = training_steps.build(examples, rules)
    # Reuse existing exact request, agent read, receipt and split checks. Its
    # summary-only candidates are merely a search space, never admitted here.
    by_hash = {example.get("example_hash"): example for example in examples}
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
                if (example.get("completeness") or {}).get("complete") is not True:
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
                for row in example.get("provider_calls") or []:
                    if not _execution_model_call(example, row):
                        continue
                    try:
                        linked = training_steps._candidate(example, row)["source_refs"][0]
                        response = _delivered_response(row, linked["delivered_response_sha256"])
                    except (ValueError, TypeError, KeyError, AttributeError):
                        trajectory_valid = False
                        break
                    checked = dict(row, response=response,
                                   response_sha256=linked["delivered_response_sha256"])
                    full, state = full_text_segments(checked, _role(example, row), registry)
                    if not full and not verified_tool_only_empty(row, response, state):
                        trajectory_valid = False
                        break
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
                record = {"schema_version": RECORD_SCHEMA_VERSION, "format": candidate["format"],
                          "input": copy.deepcopy(candidate["input"]), "prior_observations": prior,
                          "generated_thinking": segments, "target": copy.deepcopy(candidate["target"]),
                          "quality": copy.deepcopy(candidate["quality"]),
                          "source_refs": [{**{key: copy.deepcopy(value) for key, value in ref.items()
                                               if key != "teacher_reasoning_metadata"},
                                           "response_sha256": ref["delivered_response_sha256"],
                                           "route": copy.deepcopy(call["reasoning_capture"])}]}
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
