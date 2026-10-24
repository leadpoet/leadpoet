"""Exact reviewed routes for full provider-generated thinking.

Only routes in the committed registry pass admission. Each approval records
provider/API identity and a probe of raw text, tool continuity, privacy, and
structured output; round activation separately requires a verified baseline.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from lab_arena import operations

POLICY = "full_generated_thinking_v1"
REGISTRY_SCHEMA_VERSION = 1
PARSER_VERSION = "arena_openrouter_reasoning_v1"
OPERATIONS = frozenset({"openrouter.chat", "openrouter.responses"})
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MAX_ROUTES = 4096
PRICE_COMPONENTS = frozenset({"prompt", "completion", "request", "internal_reasoning"})


def enabled(configuration: Mapping[str, Any]) -> bool:
    if "reasoning_capture_policy" not in configuration:
        return False
    if configuration["reasoning_capture_policy"] != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    return True


def round_policy(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    on = enabled(configuration)
    return {"mode": "requested" if on else "off", "policy": POLICY if on else None}


def validate_registry(document: Any) -> Dict[str, Any]:
    """Validate exact, reviewed routes; catalog capability flags grant nothing."""
    if (not isinstance(document, Mapping) or set(document) != {"schema_version", "approved_routes", "baseline_route"}
            or document["schema_version"] != REGISTRY_SCHEMA_VERSION
            or not isinstance(document["approved_routes"], list)
            or len(document["approved_routes"]) > MAX_ROUTES):
        raise ValueError("invalid reasoning route registry")
    seen = set()
    for entry in document["approved_routes"]:
        if (not isinstance(entry, Mapping)
                or set(entry) != {"model_id", "model_version", "provider", "provider_name", "api", "mode", "parser_version", "semantic_class", "full_text_fields", "max_output_tokens", "answer_headroom_tokens", "context_tokens", "price_ceiling", "evidence", "allowed_fallback_routes", "status"}
                or not isinstance(entry["model_id"], str)
                or operations._MODEL_ID_RE.fullmatch(entry["model_id"]) is None
                or not isinstance(entry["model_version"], str)
                or operations._MODEL_ID_RE.fullmatch(entry["model_version"]) is None
                or not isinstance(entry["provider"], str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", entry["provider"]) is None
                or not isinstance(entry["provider_name"], str)
                or not 1 <= len(entry["provider_name"]) <= 128
                or entry["api"] not in OPERATIONS
                or entry["parser_version"] != PARSER_VERSION
                or entry["semantic_class"] != "full_generated_thinking"
                or not isinstance(entry["full_text_fields"], list)
                or not entry["full_text_fields"]
                or any(not isinstance(field, str) for field in entry["full_text_fields"])
                or len(entry["full_text_fields"]) != len(set(entry["full_text_fields"]))
                or any(field not in ({"reasoning", "reasoning_content", "reasoning_details.text"}
                                     if entry["api"] == "openrouter.chat"
                                     else {"output.content.reasoning_text"})
                       for field in entry["full_text_fields"])
                or entry["status"] != "approved"
                or type(entry["max_output_tokens"]) is not int
                or not 2048 <= entry["max_output_tokens"] <= operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS
                or type(entry["answer_headroom_tokens"]) is not int
                or not 1024 <= entry["answer_headroom_tokens"] < entry["max_output_tokens"]
                or type(entry["context_tokens"]) is not int
                or entry["context_tokens"] < entry["max_output_tokens"]
                or not isinstance(entry["mode"], Mapping)
                or not isinstance(entry["allowed_fallback_routes"], list)
                or entry["allowed_fallback_routes"]):
            raise ValueError("invalid approved reasoning route")
        pair = (entry["model_id"], entry["api"])
        if pair in seen:
            raise ValueError("duplicate approved reasoning route")
        seen.add(pair)
        mode = entry["mode"]
        if (set(mode) == {"effort"} and mode["effort"] in EFFORTS[2:]):
            if entry["api"] == "openrouter.responses" and mode["effort"] == "max":
                raise ValueError("unsupported Responses reasoning effort")
        elif set(mode) == {"budget_tokens"} and type(mode["budget_tokens"]) is int and 1024 <= mode["budget_tokens"] <= 2048 and entry["api"] == "openrouter.chat":
            pass
        elif set(mode) == {"enabled"} and mode["enabled"] is True and entry["api"] == "openrouter.chat":
            pass
        else:
            raise ValueError("invalid generated thinking mode")
        ceiling = entry["price_ceiling"]
        if not isinstance(ceiling, Mapping) or set(ceiling) != PRICE_COMPONENTS:
            raise ValueError("invalid exact route price ceiling")
        for component, value in ceiling.items():
            if (not isinstance(value, str)
                    or re.fullmatch(r"[0-9]+(?:\.[0-9]{1,12})?", value) is None
                    or Decimal(value) > (Decimal("1000") if component == "request" else Decimal("1"))):
                raise ValueError("invalid exact route price ceiling")
        evidence = entry["evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {"documentation_url", "probe_id", "verified_at", "probe_manifest_sha256"}:
            raise ValueError("missing exact route evidence")
        url = evidence["documentation_url"]
        if (not isinstance(url, str) or len(url) > 2048 or urlsplit(url).scheme != "https"
                or not urlsplit(url).hostname or not isinstance(evidence["probe_id"], str)
                or not 8 <= len(evidence["probe_id"]) <= 128
                or not isinstance(evidence["verified_at"], str)
                or not evidence["verified_at"].endswith("Z")
                or not isinstance(evidence["probe_manifest_sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", evidence["probe_manifest_sha256"]) is None):
            raise ValueError("invalid exact route evidence")
        try:
            timestamp = datetime.fromisoformat(evidence["verified_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid exact route evidence timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("invalid exact route evidence timestamp")
    baseline = document["baseline_route"]
    if baseline is not None and (not isinstance(baseline, Mapping)
            or set(baseline) != {"model_id", "api"}
            or (baseline["model_id"], baseline["api"]) not in seen):
        raise ValueError("baseline reasoning route is unapproved")
    return copy.deepcopy(dict(document))


def load_registry() -> Dict[str, Any]:
    return validate_registry(json.loads(Path(__file__).with_name("reasoning_models.json").read_text()))


def _apply_full(operation_id: str, parameters: Mapping[str, Any],
                registry: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    frozen = validate_registry(registry)
    if operation_id not in OPERATIONS:
        raise ValueError("reasoning_route_unsupported")
    result = copy.deepcopy(dict(parameters))
    model = result.get("model")
    entry = next((item for item in frozen["approved_routes"]
                  if item["model_id"] == model and item["api"] == operation_id), None)
    if entry is None:
        raise ValueError("reasoning_route_unsupported")
    if operation_id == "openrouter.responses":
        tools = result.get("tools")
        items = result.get("input")
        if (isinstance(tools, list) and any(
                isinstance(tool, Mapping) and tool.get("type") == "openrouter:web_search"
                for tool in tools)
                or isinstance(items, list) and any(
                    isinstance(item, Mapping) and item.get("type") == "web_search_call"
                    for item in items)):
            raise ValueError("reasoning_route_unsupported")
    field = "max_output_tokens" if operation_id == "openrouter.responses" else "max_tokens"
    cap = min(entry["max_output_tokens"], operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS)
    budget = entry["mode"].get("budget_tokens", 1024)
    required = budget + entry["answer_headroom_tokens"]
    if required > cap:
        raise ValueError("reasoning_headroom_unavailable")
    # Use the exact output envelope exercised by the route probe. Shrinking
    # it for a short agent request can consume the answer in thinking tokens.
    result[field] = cap
    result.pop("reasoning_effort", None)
    result.pop("include_reasoning", None)
    mode = entry["mode"]
    if "effort" in mode:
        reasoning = {"effort": mode["effort"]}
    elif "budget_tokens" in mode:
        reasoning = {"max_tokens": mode["budget_tokens"]}
    else:
        reasoning = {"enabled": True}
    if operation_id == "openrouter.chat":
        reasoning["exclude"] = False
    result["reasoning"] = reasoning
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    evidence = entry["evidence"]
    annotation = {
        "policy": POLICY, "requested": True, "reviewed_route": True,
        "semantic_class": "full_generated_thinking", "model_id": model,
        "model_version": entry["model_version"], "provider_route": entry["provider"],
        "provider_name": entry["provider_name"],
        "api": operation_id, "mode": copy.deepcopy(mode),
        "full_text_fields": list(entry["full_text_fields"]),
        "context_tokens": entry["context_tokens"],
        "max_output_tokens": entry["max_output_tokens"],
        "price_ceiling": dict(entry["price_ceiling"]),
        "parser_version": entry["parser_version"],
        "probe_id": evidence["probe_id"],
        "probe_manifest_sha256": evidence["probe_manifest_sha256"],
        "registry_sha256": hashlib.sha256(canonical).hexdigest(),
        "changes": ["reasoning", "tested_output_allowance"],
    }
    return result, annotation


def apply(operation_id: str, parameters: Mapping[str, Any], *, policy: Optional[str],
          registry: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    result = copy.deepcopy(dict(parameters))
    if policy is None or operation_id not in OPERATIONS:
        return result, None
    if policy != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    return _apply_full(operation_id, parameters, registry)


def route_observation(response: Any, annotation: Mapping[str, Any]) -> Dict[str, Any]:
    """Record actual route fields without mistaking a pin for provider proof."""
    document = response if isinstance(response, Mapping) else {}
    metadata = document.get("openrouter_metadata")
    endpoints = metadata.get("endpoints") if isinstance(metadata, Mapping) else None
    available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
    selected = [item for item in available if isinstance(item, Mapping) and item.get("selected") is True] if isinstance(available, list) else []
    endpoint = selected[0] if len(selected) == 1 else {}

    def bounded(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and 1 <= len(value) <= 128 else None

    observed = {
        "requested_model": annotation["model_id"],
        "requested_provider_route": annotation["provider_route"],
        "expected_provider_name": annotation["provider_name"],
        "expected_model_version": annotation["model_version"],
        "response_model": bounded(document.get("model")),
        "response_provider_name": bounded(document.get("provider")),
        "selected_endpoint_model": bounded(endpoint.get("model")),
        "selected_endpoint_provider_name": bounded(endpoint.get("provider")),
    }
    checks = (("response_model", "requested_model"),
              ("response_provider_name", "expected_provider_name"),
              ("selected_endpoint_model", "expected_model_version"),
              ("selected_endpoint_provider_name", "expected_provider_name"))
    values = [observed[key] for key, _ in checks]
    observed["status"] = ("unverified" if any(value is None for value in values)
                          else "matched" if all(observed[key] == observed[expected] for key, expected in checks)
                          else "mismatch")
    return observed


def valid_route_observation(value: Any) -> bool:
    """Check stored diagnostic bounds without granting it replay authority."""
    required = {
        "requested_model": 129,
        "requested_provider_route": 128,
        "expected_provider_name": 128,
        "expected_model_version": 129,
    }
    optional = {
        "response_model", "response_provider_name",
        "selected_endpoint_model", "selected_endpoint_provider_name",
    }
    if not isinstance(value, Mapping) or set(value) != set(required) | optional | {"status"}:
        return False
    if value["status"] not in ("matched", "mismatch", "unverified"):
        return False
    if any(not isinstance(value[key], str) or not 1 <= len(value[key]) <= limit
           for key, limit in required.items()):
        return False
    return all(field is None or isinstance(field, str) and 1 <= len(field) <= 128
               for field in (value[key] for key in optional))


__all__ = ["POLICY", "PARSER_VERSION", "apply", "enabled",
           "load_registry", "round_policy", "route_observation",
           "valid_route_observation", "validate_registry"]
