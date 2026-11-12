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
        pair = (entry["model_id"], entry["api"], entry["provider"])
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
            or not any(key[:2] == (baseline["model_id"], baseline["api"]) for key in seen)):
        raise ValueError("baseline reasoning route is unapproved")
    return copy.deepcopy(dict(document))


def load_registry() -> Dict[str, Any]:
    return validate_registry(json.loads(Path(__file__).with_name("reasoning_models.json").read_text()))


def validate_judge_registry(registry: Any, scorer_policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Freeze judge routes independently of sourcing and its baseline gates."""
    from lab_arena import contracts
    scorer_policy = contracts.validate_scorer_policy(scorer_policy)
    exceptions = contracts.judge_reasoning_exceptions(scorer_policy)
    frozen = validate_registry(registry)
    bindings = scorer_policy.get("env_bindings") or {}
    if bindings.get("ARENA_JUDGE_REASONING_CAPTURE_POLICY") != POLICY:
        raise ValueError("judge reasoning policy unavailable")
    routes = json.loads(bindings.get("ARENA_JUDGE_PROVIDER_ROUTES", ""))
    models = scorer_policy.get("judge_models") or {}
    approved = {(row["model_id"], row["provider"], row["api"]): row
                for row in frozen["approved_routes"]}
    options_raw = bindings.get("ARENA_JUDGE_REQUEST_OPTIONS")
    options = json.loads(options_raw) if options_raw is not None else {}
    for role, model in models.items():
        if model in exceptions:
            continue
        # Company lookup also runs a Chat tool controller. Conditional social
        # and podcast Stage 3 paths need hosted search as well as ordinary Chat.
        apis = ({"openrouter.chat", "openrouter.responses"} if role in {
            "company_fit_reverification", "intent_three_stage_stage3"} else {"openrouter.chat"})
        for api in apis:
            route = approved.get((model, routes.get(model), api))
            if route is None:
                raise ValueError("judge reasoning route unavailable")
            if options:
                option = options[model]
                mode = option["reasoning"]
                route_mode = route["mode"]
                budget = "max_tokens" in mode and "budget_tokens" in route_mode
                maximum = contracts.judge_request_max_output_tokens(option, model=model,
                    industry_model=models.get("industry_evidence_review"), operation_id=api, role=role)
                if (maximum > route["max_output_tokens"]
                        or not (budget or set(mode) == set(route_mode))
                        or api == "openrouter.responses" and mode.get("effort") == "max"):
                    raise ValueError("judge request options differ from approved route")
    return frozen


def validate_judge_provider_routing(document: Any, registry: Any,
                                    scorer_policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a host transport plan without changing the grading policy."""
    frozen = validate_judge_registry(registry, scorer_policy)
    bindings = scorer_policy["env_bindings"]
    primary = json.loads(bindings["ARENA_JUDGE_PROVIDER_ROUTES"])
    if (not isinstance(document, Mapping) or not document or not set(document) <= set(primary)
            or any(not isinstance(routes, list) or not 1 <= len(routes) <= 2
                   or any(not isinstance(route, str) or re.fullmatch(
                       r"[a-z0-9][a-z0-9._/-]{0,127}", route) is None for route in routes)
                   or len(routes) != len(set(routes)) for routes in document.values())):
        raise ValueError("invalid judge provider routing")
    for model, routes in document.items():
        # Judge registry validation already determines every API a role needs.
        apis = {row["api"] for row in frozen["approved_routes"]
                if row["model_id"] == model and row["provider"] == primary[model]}
        for api in apis:
            original = next(row for row in frozen["approved_routes"]
                            if (row["model_id"], row["api"], row["provider"]) == (model, api, primary[model]))
            candidates = [next((row for row in frozen["approved_routes"]
                               if (row["model_id"], row["api"], row["provider"]) == (model, api, provider)), None)
                          for provider in routes]
            _validate_transport_candidates(original, candidates)
    return copy.deepcopy(dict(document))


def _validate_transport_candidates(original: Mapping[str, Any], candidates: Any) -> None:
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 2
            or any(not isinstance(row, Mapping) for row in candidates)):
        raise ValueError("unapproved judge transport route")
    if len({row["provider"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate judge transport route")
    if len({row["provider_name"] for row in candidates}) != len(candidates):
        raise ValueError("ambiguous judge transport provider proof")
    for row in candidates:
        if (any(row[field] != original[field] for field in (
                "model_id", "model_version", "api", "parser_version", "semantic_class", "full_text_fields"))
                or set(row["mode"]) != set(original["mode"])
                or row["context_tokens"] < original["context_tokens"]
                or row["max_output_tokens"] < original["max_output_tokens"]):
            raise ValueError("incompatible judge transport route")


def judge_transport_candidates(annotation: Mapping[str, Any], registry: Any,
                               providers: Any) -> list[Dict[str, Any]]:
    """Bind each transport candidate to its existing exact reviewed registry row."""
    frozen = validate_registry(registry)
    original = next((row for row in frozen["approved_routes"]
                     if (row["model_id"], row["api"], row["provider"]) ==
                     (annotation["model_id"], annotation["api"], annotation["provider_route"])), None)
    if original is None or not isinstance(providers, list):
        raise ValueError("unapproved judge transport route")
    candidates = [next((row for row in frozen["approved_routes"]
                       if (row["model_id"], row["api"], row["provider"]) ==
                       (annotation["model_id"], annotation["api"], provider)), None)
                  for provider in providers]
    _validate_transport_candidates(original, candidates)
    return copy.deepcopy(candidates)


def recorded_judge_transport_routes(annotation: Mapping[str, Any], registry: Any,
                                    document: Any, scorer_policy: Mapping[str, Any]) -> list[str]:
    """Keep recorded priority while requiring the exact authorized candidate set."""
    plan = validate_judge_provider_routing(document, registry, scorer_policy)
    primary = json.loads(scorer_policy["env_bindings"]["ARENA_JUDGE_PROVIDER_ROUTES"])
    model = annotation.get("model_id")
    if not primary.get(model) or annotation.get("provider_route") != primary[model]:
        raise ValueError("unapproved judge transport route")
    expected = judge_transport_candidates(annotation, registry, plan.get(model))
    recorded = annotation.get("judge_provider_routing")
    if (not isinstance(recorded, list) or len(recorded) != len(expected)
            or any(not isinstance(row, Mapping) for row in recorded)):
        raise ValueError("unapproved judge transport route")
    providers = [row.get("provider") for row in recorded]
    authorized = {row["provider"]: row for row in expected}
    if (any(not isinstance(provider, str) for provider in providers)
            or len(set(providers)) != len(providers) or set(providers) != set(authorized)):
        raise ValueError("unapproved judge transport route")
    for row, provider in zip(recorded, providers):
        # Canonical equality keeps every reviewed field and JSON type exact;
        # registry approval alone cannot add a provider or change its descriptor.
        if json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(
                authorized[provider], sort_keys=True, separators=(",", ":"), allow_nan=False):
            raise ValueError("unapproved judge transport route")
    return providers


def _apply_full(operation_id: str, parameters: Mapping[str, Any],
                registry: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    frozen = validate_registry(registry)
    if operation_id not in OPERATIONS:
        raise ValueError("reasoning_route_unsupported")
    result = copy.deepcopy(dict(parameters))
    model = result.get("model")
    preference = result.get("provider") or {}
    selected = preference.get("only") if isinstance(preference, Mapping) else None
    if selected is not None and (not isinstance(selected, list) or len(selected) != 1
                                 or not isinstance(selected[0], str)):
        raise ValueError("reasoning_route_unsupported")
    # Existing row order is the default. Appending tested alternatives cannot
    # silently redirect a caller which did not select a provider.
    entry = next((item for item in frozen["approved_routes"]
                  if item["model_id"] == model and item["api"] == operation_id
                  and (selected is None or item["provider"] == selected[0])), None)
    if entry is None:
        raise ValueError("reasoning_route_unsupported")
    mode = entry["mode"]
    changes = []
    reasoning = result.get("reasoning", {})
    if not isinstance(reasoning, Mapping):
        raise ValueError("reasoning_controls_invalid")
    # Admission may refuse disabled or ambiguous controls, but capture must
    # not turn them on or replace an explicit effort/budget with the probe's.
    if (reasoning.get("enabled") is False or reasoning.get("exclude") is True
            or result.get("include_reasoning") is False):
        raise ValueError("reasoning_controls_invalid")
    if "include_reasoning" in result and type(result["include_reasoning"]) is not bool:
        raise ValueError("reasoning_controls_invalid")
    for key in ("enabled", "exclude"):
        if key in reasoning and type(reasoning[key]) is not bool:
            raise ValueError("reasoning_controls_invalid")
    efforts = [value for present, value in (
        ("effort" in reasoning, reasoning.get("effort")),
        ("reasoning_effort" in result, result.get("reasoning_effort")),
    ) if present]
    if any(value not in EFFORTS[1:] for value in efforts) or len(set(efforts)) > 1:
        raise ValueError("reasoning_controls_invalid")
    budgets = [reasoning[key] for key in ("max_tokens", "budget_tokens") if key in reasoning]
    if (any(type(value) is not int or value <= 0 for value in budgets)
            or len(set(budgets)) > 1 or (efforts and budgets)):
        raise ValueError("reasoning_controls_invalid")
    if not (efforts or budgets or reasoning.get("enabled") is True
            or result.get("include_reasoning") is True):
        if "reasoning" in result:
            # A summary/context request is not permission to select an effort.
            # Refuse insufficient full-thinking controls without rewriting them.
            raise ValueError("reasoning_controls_invalid")
        # Retain the established route default only when the caller did not
        # select a reasoning mode. It is a default, not a newly verified mode.
        reasoning = dict(reasoning)
        reasoning.update({"effort": mode["effort"]} if "effort" in mode else
                         {"max_tokens": mode["budget_tokens"]} if "budget_tokens" in mode else
                         {"enabled": True})
        if operation_id == "openrouter.chat":
            reasoning.setdefault("exclude", False)
        result["reasoning"] = reasoning
        changes.append("reasoning_default")
    field = "max_output_tokens" if operation_id == "openrouter.responses" else "max_tokens"
    if field not in result:
        result[field] = operations.OPENROUTER_MAX_OUTPUT_TOKENS
        changes.append("output_limit_default")
    actual_budget = (result.get("reasoning") or {}).get("max_tokens",
                     (result.get("reasoning") or {}).get("budget_tokens"))
    if actual_budget is not None and (type(result[field]) is not int or actual_budget >= result[field]):
        raise ValueError("reasoning_headroom_unavailable")
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    evidence = entry["evidence"]
    annotation = {
        "policy": POLICY, "requested": True, "reviewed_route": True,
        "semantic_class": "full_generated_thinking", "model_id": model,
        "model_version": entry["model_version"], "provider_route": entry["provider"],
        "provider_name": entry["provider_name"],
        # Mode and allowance describe the frozen probe, not the actual call.
        # The hashed parameters and captured outbound body retain that call.
        "api": operation_id, "mode": copy.deepcopy(mode),
        "full_text_fields": list(entry["full_text_fields"]),
        "context_tokens": entry["context_tokens"],
        "max_output_tokens": entry["max_output_tokens"],
        "price_ceiling": dict(entry["price_ceiling"]),
        "parser_version": entry["parser_version"],
        "probe_id": evidence["probe_id"],
        "probe_manifest_sha256": evidence["probe_manifest_sha256"],
        "registry_sha256": hashlib.sha256(canonical).hexdigest(),
        "changes": changes,
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


def explicit_provider_matches(request: Mapping[str, Any], provider_route: Any,
                              *, transport_routes: Any = None) -> bool:
    """Bind explicit miner selection without changing selector-absent records."""
    if not isinstance(request, Mapping):
        return False
    selections = []
    for field in ("admitted_parameters", "hashed_parameters"):
        parameters = request.get(field) or {}
        preference = parameters.get("provider") if isinstance(parameters, Mapping) else None
        selections.append((isinstance(preference, Mapping) and "only" in preference,
                           preference.get("only") if isinstance(preference, Mapping) else None))
    if transport_routes is None and not any(present for present, _ in selections):
        return True
    expected = [provider_route]
    outbound = request.get("outbound") or {}
    body = outbound.get("body") if isinstance(outbound, Mapping) else None
    provider = body.get("provider") if isinstance(body, Mapping) else None
    if transport_routes is not None:
        return (isinstance(provider_route, str) and bool(provider_route)
                and all(present and isinstance(value, list) and value == expected for present, value in selections)
                and isinstance(provider, Mapping) and provider.get("only") == transport_routes
                and provider.get("order") == transport_routes
                and provider.get("allow_fallbacks") is (len(transport_routes) > 1)
                and provider.get("zdr") is True and provider.get("data_collection") == "deny"
                and provider.get("require_parameters") is True)
    return (isinstance(provider_route, str) and bool(provider_route)
            and all(present and isinstance(value, list) and value == expected for present, value in selections)
            and isinstance(provider, Mapping) and provider.get("only") == expected)


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
    candidates = annotation.get("judge_provider_routing")
    if candidates is not None:
        matching = [row for row in candidates if isinstance(row, Mapping)
                    and row.get("provider_name") == observed["selected_endpoint_provider_name"]
                    and row.get("model_version") == observed["selected_endpoint_model"]]
        candidate = matching[0] if len(matching) == 1 else None
        observed["selected_provider_route"] = candidate["provider"] if candidate else None
        provider_name = candidate["provider_name"] if candidate else None
        values = [observed["response_model"], observed["selected_endpoint_model"],
                  observed["selected_endpoint_provider_name"]]
        if annotation.get("api") != "openrouter.responses" or document.get("provider") is not None:
            values.append(observed["response_provider_name"])
        observed["status"] = ("unverified" if any(value is None for value in values)
                              else "matched" if candidate is not None
                              and observed["response_model"] == annotation["model_id"]
                              and observed["response_provider_name"] in (provider_name, None)
                              else "mismatch")
        return observed
    checks = (("response_model", "requested_model"),
              ("response_provider_name", "expected_provider_name"),
              ("selected_endpoint_model", "expected_model_version"),
              ("selected_endpoint_provider_name", "expected_provider_name"))
    # OpenRouter Responses identifies its provider in selected endpoint
    # metadata; its Chat-only top-level provider field can be absent.
    if annotation.get("api") == "openrouter.responses" and document.get("provider") is None:
        checks = tuple(pair for pair in checks if pair[0] != "response_provider_name")
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
    if not isinstance(value, Mapping) or set(value) not in (
            set(required) | optional | {"status"},
            set(required) | optional | {"status", "selected_provider_route"}):
        return False
    if "selected_provider_route" in value and value["selected_provider_route"] is not None and (
            not isinstance(value["selected_provider_route"], str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", value["selected_provider_route"]) is None):
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
