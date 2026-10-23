"""Round-frozen request policy for provider-returned readable reasoning.

Only execute calls in opted-in rounds change. Returned text is a provider
summary or thinking output; opaque continuity state is not training text.
The policy runs before hashing, accounting, capture and dispatch.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple

from lab_arena import operations

POLICY = "agent_readable_reasoning_v1"
OPERATIONS = frozenset({"openrouter.chat", "openrouter.responses"})
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_FLOOR = "low"
# Live sourcing probes returned no text at low, and readable summaries at
# medium, for effort-controlled Gemini. Budget-style models keep their budget.
GOOGLE_EFFORT_FLOOR = "medium"
SUMMARY = "auto"
HEADROOM_TOKENS = 1024
BUDGET_TOKENS = 1024
MAX_PROFILES = 4096
_PROFILE_KEYS = frozenset({"mandatory", "default_enabled", "default_effort",
                           "supported_efforts", "supports_max_tokens"})
_STATEFUL_FAMILIES = ("anthropic/", "google/")


def enabled(configuration: Mapping[str, Any]) -> bool:
    if "reasoning_capture_policy" not in configuration:
        return False
    if configuration["reasoning_capture_policy"] != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    return True


def round_policy(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    on = enabled(configuration)
    return {"mode": "requested" if on else "off", "policy": POLICY if on else None}


def profiles_from_models_response(response: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Keep capability metadata from the existing startup catalog, not a new GET."""
    profiles = {}
    for item in response.get("data") or ():
        if not isinstance(item, Mapping):
            continue
        model, meta = item.get("id"), item.get("reasoning")
        parameters = item.get("supported_parameters")
        if (not isinstance(model, str) or operations._MODEL_ID_RE.fullmatch(model) is None
            or not (isinstance(meta, Mapping)
                    or isinstance(parameters, list) and "reasoning" in parameters)):
            continue
        meta = meta if isinstance(meta, Mapping) else {}
        profile = {"mandatory": meta.get("mandatory") is True}
        for field in ("default_enabled", "supports_max_tokens"):
            if type(meta.get(field)) is bool:
                profile[field] = meta[field]
        if meta.get("default_effort") in EFFORTS:
            profile["default_effort"] = meta["default_effort"]
        if "supported_efforts" in meta:
            supported = meta["supported_efforts"]
            if supported is None:
                profile["supported_efforts"] = None  # All effort values accepted.
            elif isinstance(supported, list):
                efforts = [effort for effort in EFFORTS if effort in supported]
                if efforts:
                    profile["supported_efforts"] = efforts
        profiles[model] = profile
    return dict(sorted(profiles.items())[:MAX_PROFILES])


def validate_profiles(document: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(document, Mapping) or len(document) > MAX_PROFILES:
        raise ValueError("reasoning profiles must be a bounded object")
    for model, profile in document.items():
        if not isinstance(model, str) or operations._MODEL_ID_RE.fullmatch(model) is None:
            raise ValueError("reasoning profile model is invalid")
        if not isinstance(profile, Mapping) or not set(profile) <= _PROFILE_KEYS:
            raise ValueError("reasoning profile has unknown fields")
        if type(profile.get("mandatory")) is not bool:
            raise ValueError("reasoning profile requires mandatory")
        for field in ("default_enabled", "supports_max_tokens"):
            if field in profile and type(profile[field]) is not bool:
                raise ValueError("reasoning profile %s is invalid" % field)
        if "default_effort" in profile and profile["default_effort"] not in EFFORTS:
            raise ValueError("reasoning profile default_effort is invalid")
        if "supported_efforts" in profile and profile["supported_efforts"] is not None:
            supported = profile["supported_efforts"]
            if (not isinstance(supported, list) or not supported
                or supported != [effort for effort in EFFORTS if effort in supported]):
                raise ValueError("reasoning profile supported_efforts is invalid")
    # A round must not share mutable lists with the runtime catalog.
    return copy.deepcopy(dict(document))


def _annotation(treatment: str, requested: bool, changes: List[str],
                effort: Optional[str] = None, skipped: Optional[str] = None) -> Dict[str, Any]:
    result = {"policy": POLICY, "treatment": treatment, "requested": requested,
              "changes": sorted(set(changes))}
    if isinstance(effort, str) and effort in EFFORTS:
        result["effort"] = effort
    if skipped is not None:
        result["skipped"] = skipped
    return result


def _output_limit(parameters: Mapping[str, Any], field: str) -> int:
    value = parameters.get(field)
    return value if type(value) is int else operations.OPENROUTER_MAX_OUTPUT_TOKENS


def _allowance(parameters: Dict[str, Any], field: str, cap: int) -> str:
    """Restore up to 1,024 tokens of visible allowance, within the existing cap.

    This is output room, not a hard bound on effort-controlled thinking.
    At the cap, added thinking necessarily shares the original output limit.
    """
    previous = _output_limit(parameters, field)
    parameters[field] = min(previous + HEADROOM_TOKENS, cap)
    return "headroom" if previous + HEADROOM_TOKENS <= cap else "headroom_capped"


def _tool_state(parameters: Mapping[str, Any], responses: bool) -> bool:
    if parameters.get("tools") or parameters.get("tool_choice") not in (None, "auto", "none"):
        return True
    items = parameters.get("input" if responses else "messages")
    if not isinstance(items, list):
        return False
    return any(isinstance(item, Mapping) and (
        item.get("role") == "tool" or item.get("tool_calls")
        or item.get("type") in {"function_call", "function_call_output", "web_search_call"}
    ) for item in items)


def _anthropic_sampling_conflict(parameters: Mapping[str, Any]) -> bool:
    messages = parameters.get("messages") or []
    prefill = bool(messages) and isinstance(messages[-1], Mapping) and messages[-1].get("role") == "assistant"
    return ((parameters.get("temperature") is not None and parameters["temperature"] != 1)
            or parameters.get("top_k") is not None
            or parameters.get("top_p") is not None and parameters["top_p"] < 0.95
            or prefill)


def _control(model: str, profile: Mapping[str, Any]) -> str:
    if profile.get("supports_max_tokens") is True:
        return "budget"
    if "supported_efforts" in profile:
        return "effort"
    # Legacy catalog entries omit supports_max_tokens. Only these documented
    # budget-style families get a budget fallback; an explicit false wins.
    if "supports_max_tokens" not in profile and (
        model.startswith("anthropic/") and profile.get("mandatory") is not True
        or model.startswith("google/gemini-2.5-")
    ):
        return "budget"
    return "toggle"


def apply(operation_id: str, parameters: Mapping[str, Any], *, policy: Optional[str],
          profiles: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    result = copy.deepcopy(dict(parameters))
    if policy is None or operation_id not in OPERATIONS:
        return result, None
    if policy != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    model = str(result.get("model") or "")
    profile = (profiles or {}).get(model)
    if profile is None:
        return result, _annotation("no_reasoning_support", False, [])
    responses = operation_id == "openrouter.responses"
    treatment = "responses" if responses else "chat"
    original = copy.deepcopy(result)
    supplied = result.get("reasoning")
    reasoning = dict(supplied) if isinstance(supplied, Mapping) else {}
    changes: List[str] = []
    legacy_effort = result.pop("reasoning_effort", None)
    legacy_include = result.pop("include_reasoning", None)
    if legacy_effort is not None:
        reasoning.setdefault("effort", legacy_effort)
        changes.append("legacy_effort")
    if legacy_include is not None:
        reasoning.setdefault("exclude", legacy_include is False)
        if legacy_include is True:
            reasoning.setdefault("enabled", True)
        changes.append("legacy_include")
    effort, budget = reasoning.get("effort"), reasoning.get("max_tokens")
    disabled = reasoning.get("enabled") is False or effort == "none" or budget == 0
    positive_budget = type(budget) is int and budget > 0
    explicit_on = not disabled and (reasoning.get("enabled") is True or positive_budget
        or effort is not None or legacy_include is True)
    default_on = profile.get("mandatory") is True or profile.get("default_enabled") is True
    was_on = not disabled and (explicit_on or default_on)
    current_effort = effort if effort is not None else (
        profile.get("default_effort") if not positive_budget and was_on else None
    )
    control = _control(model, profile)
    floor = GOOGLE_EFFORT_FLOOR if model.startswith("google/") and control == "effort" else EFFORT_FLOOR
    raise_reasoning = not was_on or (
        isinstance(current_effort, str) and current_effort in EFFORTS
        and EFFORTS.index(current_effort) < EFFORTS.index(floor)
    )
    skipped = None
    if raise_reasoning:
        if model.startswith(_STATEFUL_FAMILIES) and _tool_state(original, responses):
            skipped = "thinking_tool_state_unavailable"
        elif (model.startswith("anthropic/") and profile.get("mandatory") is not True
              and (responses or _anthropic_sampling_conflict(original))):
            skipped = "anthropic_thinking_conflict"
        supported = profile.get("supported_efforts")
        if skipped is None and control == "effort" and supported is not None and floor not in supported:
            skipped = "no_%s_effort" % floor
        if skipped is None and responses and control != "effort":
            skipped = "operation_reasoning_control_unavailable"
        field = "max_output_tokens" if responses else "max_tokens"
        cap = (operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS if responses
               else operations.OPENROUTER_MAX_OUTPUT_TOKENS)
        if skipped is None and control == "toggle" and _output_limit(result, field) + HEADROOM_TOKENS > cap:
            skipped = "headroom_unavailable"
        if skipped is not None and not was_on:
            return original, _annotation(treatment, False, [], skipped=skipped)
        if skipped is None:
            reasoning.pop("enabled", None)
            reasoning.pop("effort", None)
            reasoning.pop("max_tokens", None)
            if control == "budget":
                reasoning["max_tokens"] = BUDGET_TOKENS
                changes.append("budget")
            elif control == "effort":
                reasoning["effort"] = floor
                changes.append("effort")
            else:
                reasoning["enabled"] = True
                changes.append("enabled")
            changes.append(_allowance(result, field, cap))
            if disabled:
                changes.append("agent_disabled_overridden")
    elif effort is None and not explicit_on and current_effort is not None:
        # Freeze the catalog's default effort rather than trusting future defaults.
        reasoning["effort"] = current_effort
        changes.append("effort")
    if not responses and reasoning.get("exclude") is not False:
        changes.append("exclude_cleared" if reasoning.get("exclude") is True else "exclude")
        reasoning["exclude"] = False
    if (responses or model.startswith("openai/")) and reasoning.get("summary") not in ("auto", "concise", "detailed"):
        reasoning["summary"] = SUMMARY
        changes.append("summary")
    # OpenRouter returns visible Claude summaries by default (unlike native
    # Claude 5.5), and maps Gemini visibility through this exclusion setting.
    if reasoning:
        result["reasoning"] = reasoning
    return result, _annotation(treatment, True, changes, reasoning.get("effort"), skipped)


def unapplied(annotation: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return dict(_annotation(str(annotation.get("treatment")), False, []), unapplied=reason)


__all__ = ["POLICY", "apply", "enabled", "profiles_from_models_response",
           "round_policy", "unapplied", "validate_profiles"]
