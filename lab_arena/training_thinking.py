"""Read-only coverage and strict generated-thinking candidates from Arena exports.

The provider receipt and agent read proof remain in training_steps. A request
annotation establishes route semantics; returned text alone never does.
"""
from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from lab_arena import contracts, reasoning_capture, training_capture, training_dataset, training_steps

SCHEMA_VERSION = "leadpoet.lab_arena.full_thinking_dataset.v1"
RECORD_SCHEMA_VERSION = "leadpoet.lab_arena.full_thinking_step.v1"
POLICY = "full_generated_thinking_v1"
PARSER_VERSION = "arena_openrouter_reasoning_v1"
_ROUTE_FIELDS = ("model_id", "model_version", "provider_route", "api",
                 "parser_version", "probe_id", "probe_manifest_sha256", "registry_sha256")
_BAD_CLASSES = frozenset({"summary", "opaque", "redacted", "truncated", "interrupted",
                          "storage_failed", "size_limited", "unavailable", "unproved_text"})


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
    tokens = usage.get("reasoning_tokens", details.get("reasoning_tokens"))
    if type(tokens) is not int or tokens < 0:
        tokens = None
    capture_reason = (call.get("upstream_response_capture") or {}).get("reason")
    storage = ("saved" if isinstance(response, Mapping) else
               "storage_failed" if capture_reason == "response_capture_failed" else
               "size_limited" if capture_reason == "response_size_limit" else "unavailable")
    if storage == "storage_failed":
        content_class = "storage_failed"
    elif storage == "size_limited":
        content_class = "size_limited"
    elif not isinstance(response, Mapping):
        content_class = "unavailable"
    elif "credential" in redactions:
        content_class = "redacted"
    elif "hidden_reasoning" in redactions:
        content_class = "opaque"
    elif finish in {"length", "max_tokens", "incomplete"} or response.get("incomplete_details"):
        content_class = "truncated"
    elif (call.get("operation_id") == "openrouter.chat" and finish not in {"stop", "tool_calls"}
          or call.get("operation_id") == "openrouter.responses" and finish != "completed"):
        content_class = "interrupted"
    elif summary:
        content_class = "summary"
    elif opaque:
        content_class = "opaque"
    elif unproved:
        content_class = "unproved_text"
    elif segments:
        content_class = "full_thinking" if route_verified else "unproved_text"
    else:
        content_class = "empty" if route_verified else "unsupported"
    return {"run_id": call.get("run_id"), "call_identity": call.get("call_identity"),
            "provider_attempt": call.get("provider_attempt"), "role": role,
            "model_id": annotation.get("model_id") or (((call.get("request") or {}).get("outbound") or {}).get("body") or {}).get("model"),
            "requested_provider_route": annotation.get("provider_route"), "api": call.get("operation_id"),
            "requested": annotation.get("requested") is True, "route_reviewed": policy_verified,
            "provider_route_verified": observation_verified,
            "returned": content_class, "storage": storage, "finish_reason": finish,
            "characters": sum(len(part["text"]) for part in segments),
            "tokens_reported": tokens, "segment_count": len(segments),
            "contains_summary": summary, "contains_opaque": opaque,
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
            attempts.append(row)
            counts[row["returned"]] += 1
            key = (row["role"], row["model_id"], row["requested_provider_route"], row["api"])
            grouped[key][row["returned"]] += 1
            grouped[key]["requested"] += row["requested"]
            grouped[key]["saved"] += row["storage"] == "saved"
            execution_model = _execution_model_call(example, call)
            has_text |= execution_model and row["returned"] == "full_thinking"
            full_round = (example.get("configuration") or {}).get("reasoning_capture_policy") == POLICY
            bad |= (execution_model and (row["requested"] or full_round) and
                    (row["returned"] in _BAD_CLASSES or not row["route_reviewed"]
                     or not row["provider_route_verified"]))
        trajectories["ineligible"] += bad
        trajectories["with_full_thinking"] += has_text
        trajectories["eligible_without_disqualifying_capture"] += not bad
        trajectories["complete_full_thinking"] += bool(
            not bad and has_text and (example.get("completeness") or {}).get("complete") is True)
    return {"attempts": attempts, "returned_classes": dict(sorted(counts.items())),
            "distinct_verified_exports": len(seen),
            "saved_calls": sum(row["storage"] == "saved" for row in attempts),
            "generated_thinking_texts": sum(row["returned"] == "full_thinking" for row in attempts),
            "summary_calls": sum(row["returned"] == "summary" for row in attempts),
            "by_role_model_route": [{"role": key[0], "model_id": key[1], "requested_provider_route": key[2],
                                     "api": key[3], "counts": dict(sorted(value.items()))}
                                    for key, value in sorted(grouped.items(), key=lambda item: str(item[0]))],
            "trajectories": dict(sorted(trajectories.items()))}


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
                                      response_sha256=ref["delivered_response_sha256"],
                                      response_privacy_redactions=[],
                                      response_receipts=[receipt for receipt in call.get("response_receipts") or []
                                                         if receipt.get("response_sha256") == ref["delivered_response_sha256"]])
                registry = (example.get("configuration") or {}).get("reasoning_route_registry")
                status = attempt_coverage(delivered_call, _role(example, call), registry)
                all_status = [attempt_coverage(row, _role(example, row), registry)
                              for row in example.get("provider_calls") or []
                              if _execution_model_call(example, row)]
                full_round = (example.get("configuration") or {}).get("reasoning_capture_policy") == POLICY
                if any((row["requested"] or full_round) and
                       (row["returned"] in _BAD_CLASSES or not row["route_reviewed"]
                        or not row["provider_route_verified"])
                       for row in all_status):
                    reasons["trajectory_reasoning_ineligible"] += 1
                    continue
                if status["returned"] != "full_thinking" or status["storage"] != "saved":
                    reasons[status["returned"]] += 1
                    continue
                try:
                    prior = _prior_observations(example, call)
                except ValueError as exc:
                    reasons[str(exc)] += 1
                    continue
                segments, summary, opaque = _segments(delivered, call.get("operation_id"))
                allowed = set(call["reasoning_capture"]["full_text_fields"])
                if summary or opaque or not _approved_segments(segments, allowed):
                    reasons["delivered_reasoning_not_full"] += 1
                    continue
                if not segments:
                    reasons["full_thinking_text_unavailable"] += 1
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
