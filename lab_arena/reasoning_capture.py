"""Readable model reasoning for Arena agent calls.

A round may freeze ``reasoning_capture_policy``. Under it the gateway broker
makes every OpenRouter call from an agent (an ``execute`` run) reason at
least at low effort and return readable reasoning: a provider summary or
thinking text. Judge (``score``) runs are never changed.

Each round also freezes the per-model reasoning profiles from OpenRouter's
model catalog at creation. The request rules are a pure function of the
policy, those profiles and the request, so a call's request hash, identity
and captured outbound body bind the parameters actually sent, and a retried
call reproduces them exactly.

Readable reasoning is text a provider deliberately returns. Encrypted
reasoning, signatures and redacted thinking stay opaque. They are never
counted as captured reasoning and never become training text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from lab_arena import operations

POLICY = "agent_reasoning_v1"
OPERATIONS = frozenset({"openrouter.chat", "openrouter.responses"})
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_FLOOR = "low"
# OpenAI's "auto" selects the most detailed summarizer a model offers.
SUMMARY = "auto"
# Added output allowance when the policy raises reasoning, so new reasoning
# tokens do not consume the agent's visible answer.
HEADROOM_TOKENS = 1024
# Anthropic's minimum extended-thinking budget.
ANTHROPIC_BUDGET_TOKENS = 1024
MAX_PROFILES = 4096
_PROFILE_KEYS = frozenset({"mandatory", "default_enabled", "default_effort", "supported_efforts"})


def enabled(configuration: Mapping[str, Any]) -> bool:
    if "reasoning_capture_policy" not in configuration:
        return False
    if configuration["reasoning_capture_policy"] != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    return True


def round_policy(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    on = enabled(configuration)
    return {"mode": "forced" if on else "off", "policy": POLICY if on else None}


def profiles_from_models_response(response: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Reasoning profiles for every catalog model that accepts ``reasoning``."""

    profiles: Dict[str, Dict[str, Any]] = {}
    for item in response.get("data") or ():
        if not isinstance(item, Mapping):
            continue
        model = item.get("id")
        meta = item.get("reasoning")
        parameters = item.get("supported_parameters")
        accepts = isinstance(parameters, list) and "reasoning" in parameters
        if (not isinstance(model, str) or operations._MODEL_ID_RE.fullmatch(model) is None
                or not (accepts or isinstance(meta, Mapping))):
            continue
        meta = meta if isinstance(meta, Mapping) else {}
        profile: Dict[str, Any] = {"mandatory": meta.get("mandatory") is True}
        if isinstance(meta.get("default_enabled"), bool):
            profile["default_enabled"] = meta["default_enabled"]
        if meta.get("default_effort") in EFFORTS:
            profile["default_effort"] = meta["default_effort"]
        supported = meta.get("supported_efforts")
        if isinstance(supported, list):
            efforts = [effort for effort in EFFORTS if effort in supported]
            if efforts:
                profile["supported_efforts"] = efforts
        profiles[model] = profile
    # A bounded, deterministic subset; the catalog must never stop a startup.
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
        if "default_enabled" in profile and type(profile["default_enabled"]) is not bool:
            raise ValueError("reasoning profile default_enabled is invalid")
        if "default_effort" in profile and profile["default_effort"] not in EFFORTS:
            raise ValueError("reasoning profile default_effort is invalid")
        supported = profile.get("supported_efforts", [EFFORT_FLOOR])
        if (not isinstance(supported, list) or not supported
                or list(supported) != [effort for effort in EFFORTS if effort in supported]):
            raise ValueError("reasoning profile supported_efforts is invalid")
    return {model: dict(profile) for model, profile in document.items()}


def _rank(effort: Any) -> int:
    return EFFORTS.index(effort) if effort in EFFORTS else -1


def _at_least_floor(effort: Optional[str], profile: Mapping[str, Any]) -> str:
    """The agent's or model's effort, raised to the floor the model accepts."""
    target = max(_rank(effort), _rank(EFFORT_FLOOR))
    supported = profile.get("supported_efforts") or EFFORTS
    allowed = [value for value in supported if _rank(value) >= target]
    return min(allowed, key=_rank) if allowed else max(supported, key=_rank)


def _annotation(treatment: str, requested: bool, changes: List[str],
                effort: Optional[str] = None) -> Dict[str, Any]:
    annotation = {"policy": POLICY, "treatment": treatment, "requested": requested,
                  "changes": sorted(set(changes))}
    if effort is not None:
        annotation["effort"] = effort
    return annotation


def _headroom(parameters: Dict[str, Any], field: str, cap: int, tokens: int) -> bool:
    current = parameters.get(field)
    if type(current) is not int or current >= cap:
        return False
    parameters[field] = min(cap, current + tokens)
    return True


def _forced_tool_choice(value: Any) -> bool:
    return value not in (None, "auto", "none")


def _anthropic_sampling_conflict(parameters: Mapping[str, Any]) -> bool:
    """Extended thinking refuses these agent settings; never rewrite them."""
    temperature = parameters.get("temperature")
    top_p = parameters.get("top_p")
    return ((temperature is not None and temperature != 1) or parameters.get("top_k") is not None
            or (top_p is not None and top_p < 0.95) or _forced_tool_choice(parameters.get("tool_choice")))


def _default_on(profile: Mapping[str, Any]) -> bool:
    return profile.get("mandatory") is True or profile.get("default_enabled") is True


def _effort_plan(requested: Any, profile: Mapping[str, Any], *, disabled: bool) -> Tuple[Optional[str], bool]:
    """Return (effort to send or None to leave unset, whether reasoning rises).

    An agent's explicit effort is kept unless it is below the floor. Without
    one, a model that reasons by default keeps its catalog default, and one
    that does not is turned on at the floor.
    """

    if requested is not None and not disabled:
        if requested not in EFFORTS or _rank(requested) >= _rank(EFFORT_FLOOR):
            return None, False  # At or above the floor: the agent's choice stands.
        effort = _at_least_floor(requested, profile)
        return effort, True
    if _default_on(profile) and not disabled:
        default = profile.get("default_effort")
        if default is None:
            return None, False  # Already reasoning; only ask for readable output.
        effort = _at_least_floor(default, profile)
        return effort, _rank(effort) > _rank(default)
    return _at_least_floor(None, profile), True


def _responses(parameters: Dict[str, Any], profile: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if profile is None:
        # The round's catalog lists no reasoning support for this model.
        return parameters, _annotation("no_reasoning_support", False, [])
    reasoning = dict(parameters.get("reasoning") or {})
    changes: List[str] = []
    effort, raised = _effort_plan(reasoning.get("effort"), profile,
                                  disabled=reasoning.get("effort") == "none")
    if effort is not None and reasoning.get("effort") != effort:
        reasoning["effort"] = effort
        changes.append("effort")
    if "summary" not in reasoning:
        reasoning["summary"] = SUMMARY
        changes.append("summary")
    parameters["reasoning"] = reasoning
    if raised and _headroom(parameters, "max_output_tokens",
                            operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS, HEADROOM_TOKENS):
        changes.append("headroom")
    return parameters, _annotation("responses", True, changes, reasoning.get("effort"))


def _chat(parameters: Dict[str, Any], profile: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model = str(parameters.get("model") or "")
    if profile is None:
        # The round's catalog lists no reasoning support for this model.
        return parameters, _annotation("no_reasoning_support", False, [])
    original = dict(parameters)
    reasoning = parameters.get("reasoning")
    reasoning = dict(reasoning) if isinstance(reasoning, Mapping) else {}
    changes: List[str] = []
    # Fold the legacy fields into one reasoning object so the provider never
    # sees two conflicting reasoning settings.
    legacy_effort = parameters.pop("reasoning_effort", None)
    if legacy_effort is not None:
        changes.append("legacy_effort")
        reasoning.setdefault("effort", legacy_effort)
    if parameters.pop("include_reasoning", None) is not None:
        changes.append("legacy_include")
    disabled = (reasoning.get("enabled") is False or reasoning.get("effort") == "none"
                or reasoning.get("max_tokens") == 0)
    budget = reasoning.get("max_tokens")
    agent_budget = type(budget) is int and budget > 0 and not disabled
    anthropic_optional = model.startswith("anthropic/") and profile.get("mandatory") is not True
    raised = False
    effort: Optional[str] = None
    agent_effort = reasoning.get("effort") is not None and not disabled
    if agent_budget:
        pass  # The agent chose its own thinking budget.
    elif anthropic_optional and not agent_effort and not _default_on(profile):
        # Anthropic thinking is turned on by budget. It refuses temperature,
        # top_k, low top_p and forced tools, so a conflicting call is skipped.
        if _anthropic_sampling_conflict(parameters):
            return original, _annotation("anthropic_sampling_conflict", False, [])
        reasoning.pop("effort", None)
        reasoning["max_tokens"] = ANTHROPIC_BUDGET_TOKENS
        changes.append("budget")
        raised = True
    else:
        effort, raised = _effort_plan(reasoning.get("effort"), profile, disabled=disabled)
    if effort is not None and reasoning.get("effort") != effort:
        reasoning["effort"] = effort
        changes.append("effort")
    if effort is not None and reasoning.get("max_tokens") == 0:
        del reasoning["max_tokens"]
    if reasoning.get("enabled") is False:
        changes.append("enabled")
        del reasoning["enabled"]
    # Returning reasoning text is OpenRouter's default; only an agent's
    # explicit exclusion needs reversing.
    if reasoning.get("exclude") is True:
        reasoning["exclude"] = False
        changes.append("exclude_cleared")
    if model.startswith("openai/") and "summary" not in reasoning:
        reasoning["summary"] = SUMMARY
        changes.append("summary")
    if reasoning:
        parameters["reasoning"] = reasoning
    allowance = ANTHROPIC_BUDGET_TOKENS if "budget" in changes else HEADROOM_TOKENS
    if raised and _headroom(parameters, "max_tokens", operations.OPENROUTER_MAX_OUTPUT_TOKENS, allowance):
        changes.append("headroom")
    return parameters, _annotation("chat", True, changes, reasoning.get("effort"))


def apply(
    operation_id: str,
    parameters: Mapping[str, Any],
    *,
    policy: Optional[str],
    profiles: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return an agent call's parameters under a frozen round policy.

    The second value annotates the reservation. It is ``None`` when the round
    has no policy or the operation is not a model call.
    """

    result = dict(parameters)
    if policy is None or operation_id not in OPERATIONS:
        return result, None
    if policy != POLICY:
        raise ValueError("unsupported reasoning capture policy")
    profile = (profiles or {}).get(str(result.get("model") or ""))
    if operation_id == "openrouter.responses":
        return _responses(result, profile)
    return _chat(result, profile)


def unapplied(annotation: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """The agent's own request was sent because the policy form was invalid."""
    return dict(_annotation(str(annotation.get("treatment")), False, []), unapplied=reason)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _chat_texts(message: Any) -> List[str]:
    if not isinstance(message, dict):
        return []
    texts = [_text(message.get("reasoning"))]
    for item in message.get("reasoning_details") or ():
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning.summary":
            texts.append(_text(item.get("summary")))
        elif item.get("type") == "reasoning.text":
            # Gemini can return this type with only a signature: no text.
            texts.append(_text(item.get("text")))
    return texts


def _responses_texts(item: Any) -> List[str]:
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        return []
    texts = []
    for name, part_type in (("summary", "summary_text"), ("content", "reasoning_text")):
        for part in item.get(name) or ():
            if isinstance(part, dict) and part.get("type") == part_type:
                texts.append(_text(part.get("text")))
    return texts


def readable_reasoning(document: Any) -> List[str]:
    """Return the readable reasoning texts in one OpenRouter response body.

    Chat completions carry them in ``message.reasoning`` and readable
    ``reasoning_details`` items; the Responses API carries them in reasoning
    output items. Duplicates are kept once, in their original order.
    """

    if not isinstance(document, dict):
        return []
    texts: List[str] = []
    choices = document.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                texts.extend(_chat_texts(choice.get("message")))
    output = document.get("output")
    if isinstance(output, list):
        for item in output:
            texts.extend(_responses_texts(item))
    unique: List[str] = []
    for text in texts:
        if text and text not in unique:
            unique.append(text)
    return unique


__all__ = [
    "POLICY", "apply", "enabled", "profiles_from_models_response", "readable_reasoning",
    "round_policy", "unapplied", "validate_profiles",
]
