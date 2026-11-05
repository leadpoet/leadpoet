"""Small OpenRouter request options used by company scoring."""

from __future__ import annotations

import os
import copy
import json
import re
from collections.abc import Mapping


JUDGE_REASONING_POLICY = "full_generated_thinking_v1"
JUDGE_REASONING_TIMEOUT_SECONDS = 120


def judge_reasoning_enabled() -> bool:
    """The scorer environment is bound from the round's frozen policy."""
    return os.getenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY") == JUDGE_REASONING_POLICY


def prepare_judge_request(payload: Mapping, *, hosted_search: bool = False,
                          output_token_floor: int | None = None) -> tuple[str, dict]:
    """Prepare existing judge HTTP calls; legacy rounds keep their exact body.

    Native Responses is used for hosted search so the gateway can retain its
    exposed search events and citations. This helper performs no I/O.
    """
    body = copy.deepcopy(dict(payload))
    if not judge_reasoning_enabled():
        return "/chat/completions", body
    try:
        routes = json.loads(os.environ["ARENA_JUDGE_PROVIDER_ROUTES"])
        provider = routes[body["model"]]
        if (not isinstance(routes, dict) or not isinstance(provider, str)
                or re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", provider) is None):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("judge_provider_route_unavailable") from None
    option = None
    options_raw = os.environ.get("ARENA_JUDGE_REQUEST_OPTIONS")
    if options_raw:
        try:
            option = json.loads(options_raw)[body["model"]]
            mode = option["reasoning"]
            maximum = option["max_output_tokens"]
            if (not isinstance(option, dict) or set(option) != {"reasoning", "max_output_tokens"}
                    or type(maximum) is not int or not 1 <= maximum <= 32768
                    or not isinstance(mode, dict)
                    or not ((set(mode) == {"effort"} and mode["effort"] in {"minimal", "low", "medium", "high", "xhigh", "max"})
                            or (set(mode) == {"enabled"} and mode["enabled"] is True)
                            or (set(mode) == {"max_tokens"} and type(mode["max_tokens"]) is int and 1 <= mode["max_tokens"] < maximum))):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("judge_request_options_unavailable") from None
    # This is a selector, not a caller-owned provider policy. The broker
    # installs ZDR, data_collection=deny, no fallback and exact price bounds.
    body["provider"] = {"only": [provider]}
    if "parallel_tool_calls" in body:
        if body.pop("parallel_tool_calls") is not False:
            raise ValueError("judge_parallel_tools_unsupported")
        # These routes support tools, but reject this optional Chat parameter.
        # The existing investigator still validates exactly one tool per turn.
        instruction = {"role": "system", "content": "Return exactly one tool call per assistant turn."}
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("judge_messages_unsupported")
        if instruction not in messages:
            messages.insert(0, instruction)
    controls = body.get("reasoning") or {}
    if not isinstance(controls, Mapping):
        raise ValueError("judge_reasoning_controls_unsupported")
    # Keep routine evidence checks bounded. Some reasoning routes map medium
    # to high compute; an explicit caller setting still takes precedence.
    effort = controls.get("effort", body.get("reasoning_effort", "low"))
    if (effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}
            or "reasoning_effort" in body and body["reasoning_effort"] != effort
            or hosted_search and effort == "max"):
        raise ValueError("judge_reasoning_controls_unsupported")
    mode = {"effort": effort}
    if option is not None:
        mode = copy.deepcopy(option["reasoning"])
        explicit_effort = controls.get("effort", body.get("reasoning_effort"))
        explicit_budget = controls.get("max_tokens", controls.get("budget_tokens"))
        if controls.get("enabled") is False or controls.get("exclude") is True:
            raise ValueError("judge_reasoning_controls_unsupported")
        if explicit_effort is not None:
            if "effort" not in mode:
                raise ValueError("judge_reasoning_controls_incompatible")
            mode = {"effort": effort}
        if explicit_budget is not None:
            if ("max_tokens" not in mode or type(explicit_budget) is not int
                    or not 1 <= explicit_budget < option["max_output_tokens"]):
                raise ValueError("judge_reasoning_controls_incompatible")
            mode = {"max_tokens": explicit_budget}
        if hosted_search and mode.get("effort") == "max":
            raise ValueError("judge_reasoning_controls_unsupported")
        body.pop("reasoning_effort", None)
    body["reasoning"] = {**mode, "exclude": False}
    if body.get("include_reasoning") is False:
        body["include_reasoning"] = True
    limit = body.get("max_tokens", 8192)
    if type(limit) is not int or not 1 <= limit <= 32768:
        raise ValueError("judge_output_limit_invalid")
    if output_token_floor is not None and (type(output_token_floor) is not int or not 1 <= output_token_floor <= 8192):
        raise ValueError("judge_output_floor_invalid")
    # Optional claim comparison is allowed to keep its small original budget.
    # Incomplete comparisons are cache misses, never factual verdicts.
    if option is not None:
        body["max_tokens"] = (option["max_output_tokens"] if output_token_floor is None
                              else min(option["max_output_tokens"], max(output_token_floor, limit)))
    else:
        body["max_tokens"] = max(8192 if output_token_floor is None else output_token_floor, limit)
    if not hosted_search:
        return "/chat/completions", body
    messages = body.pop("messages")
    body["input"] = _responses_messages(messages)
    body["max_output_tokens"] = body.pop("max_tokens")
    # Responses exposes thinking text natively; these are Chat-only controls.
    body.pop("include_reasoning", None)
    body.pop("reasoning_effort", None)
    body.pop("temperature", None)
    body["reasoning"].pop("exclude")
    response_format = body.pop("response_format", None)
    if response_format is not None:
        if not isinstance(response_format, Mapping):
            raise ValueError("judge_response_format_unsupported")
        if response_format.get("type") == "json_schema":
            body["text"] = {"format": {"type": "json_schema", **response_format["json_schema"]}}
        elif response_format.get("type") in {"json_object", "text"}:
            body["text"] = {"format": {"type": response_format["type"]}}
        else:
            raise ValueError("judge_response_format_unsupported")
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise ValueError("judge_tool_definition_unsupported")
        tools.append({"type": "function", **tool["function"]})
    tools.append({"type": "openrouter:web_search", "parameters": {
        "engine": "exa", "max_uses": 1, "max_total_results": 5}})
    body["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, Mapping):
        if choice.get("type") != "function":
            raise ValueError("judge_tool_choice_unsupported")
        body["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    body["max_tool_calls"] = 1
    return "/responses", body


def _responses_messages(messages) -> list:
    if not isinstance(messages, list):
        raise ValueError("judge_messages_unsupported")
    result = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("judge_messages_unsupported")
        role = message.get("role")
        content = message.get("content")
        if any(message.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
            # Search judges are one-shot requests. Do not silently convert a
            # Chat-specific thinking history into a different native protocol.
            raise ValueError("judge_messages_unsupported")
        if role == "tool":
            result.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                           "output": copy.deepcopy(content)})
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("judge_messages_unsupported")
        if content is not None:
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "output_text" if role == "assistant" else "input_text",
                                      "text": part["text"]})
                    else:
                        raise ValueError("judge_messages_unsupported")
                content = parts
            elif not isinstance(content, str):
                raise ValueError("judge_messages_unsupported")
            result.append({"role": role, "content": copy.deepcopy(content)})
        for call in message.get("tool_calls") or []:
            if role != "assistant" or call.get("type") != "function":
                raise ValueError("judge_messages_unsupported")
            result.append({"type": "function_call", "call_id": call["id"],
                           "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    return result


def _chat_annotations(annotations: list) -> list:
    result = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("judge_citation_malformed")
        if annotation.get("type") == "url_citation":
            citation = annotation.get("url_citation", annotation)
            if not isinstance(citation, Mapping) or not isinstance(citation.get("url"), str) or not citation["url"]:
                raise ValueError("judge_citation_malformed")
            if "url_citation" not in annotation:
                annotation = {"type": "url_citation", "url_citation": {
                    key: copy.deepcopy(value) for key, value in citation.items() if key != "type"}}
        result.append(copy.deepcopy(dict(annotation)))
    return result


def project_judge_response(payload: Mapping, *, hosted_search: bool = False) -> dict:
    """Project only the completed native final answer for existing judges.

    The gateway stores the original native response, including all reasoning,
    search events and citations. This projection is not a training transcript.
    """
    body = copy.deepcopy(dict(payload))
    if not judge_reasoning_enabled():
        return body
    if "output" not in body:
        if hosted_search:
            raise ValueError("judge_response_malformed")
        return body
    if (body.get("status") != "completed" or body.get("error") or body.get("errors")
            or body.get("refusal") or body.get("incomplete_details") or not isinstance(body.get("output"), list)):
        raise ValueError("judge_response_incomplete")
    messages, final_messages = [], []
    searched = False
    for item in body["output"]:
        if not isinstance(item, Mapping):
            raise ValueError("judge_response_malformed")
        kind = item.get("type")
        if kind in {"web_search_call", "openrouter:web_search"}:
            if item.get("status") != "completed":
                raise ValueError("judge_search_incomplete")
            searched = True
        elif kind == "reasoning":
            if item.get("status") not in (None, "completed"):
                raise ValueError("judge_response_incomplete")
        elif kind == "message":
            if (item.get("role") != "assistant" or item.get("status") not in (None, "completed")
                    or item.get("refusal") or not isinstance(item.get("content"), list)):
                raise ValueError("judge_response_malformed")
            text, annotations = [], []
            for part in item["content"]:
                if not isinstance(part, Mapping) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise ValueError("judge_response_malformed")
                text.append(part["text"])
                if not isinstance(part.get("annotations", []), list):
                    raise ValueError("judge_response_malformed")
                annotations.extend(_chat_annotations(part.get("annotations", [])))
            phase = item.get("phase")
            if phase not in (None, "commentary", "final_answer"):
                raise ValueError("judge_response_phase_unsupported")
            message = {"role": "assistant", "content": "".join(text),
                       **({"annotations": annotations} if annotations else {})}
            if phase == "final_answer":
                final_messages.append(message)
            elif phase is None:
                messages.append(message)
        else:
            # A function call or refusal is unfinished judging, not a verdict.
            raise ValueError("judge_response_unfinished_tool")
    if hosted_search and not searched:
        raise ValueError("judge_search_missing")
    if len(final_messages) > 1:
        raise ValueError("judge_response_final_ambiguous")
    selected = final_messages or messages
    if not selected or not selected[-1]["content"].strip():
        raise ValueError("judge_response_missing_answer")
    return {**{key: copy.deepcopy(body[key]) for key in ("id", "model", "provider", "usage") if key in body},
            "choices": [{"index": 0, "finish_reason": "stop", "message": selected[-1]}]}


def preserve_judge_reasoning(provider_message: Mapping, replay_message: Mapping) -> dict:
    """Keep native Chat thinking on tool continuations under the frozen opt-in."""
    result = copy.deepcopy(dict(replay_message))
    if judge_reasoning_enabled():
        for key in ("reasoning", "reasoning_content"):
            value = provider_message.get(key)
            if isinstance(value, str) and value:
                result[key] = value
        details = provider_message.get("reasoning_details")
        if isinstance(details, list) and details and all(isinstance(item, Mapping) for item in details):
            result["reasoning_details"] = copy.deepcopy(details)
    return result


def include_reasoning_default() -> bool:
    return str(os.getenv("QUALIFICATION_LLM_INCLUDE_REASONING", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reasoning_request_unsupported(status: int, message: str) -> bool:
    if int(status) not in {400, 404, 422}:
        return False
    text = str(message or "").lower()
    return "reasoning" in text or "include_reasoning" in text
