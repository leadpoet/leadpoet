"""Small OpenRouter request options used by company scoring."""

from __future__ import annotations

import os
import copy
import json
import re
from collections.abc import Mapping


JUDGE_REASONING_POLICY = "full_generated_thinking_v1"


def judge_reasoning_enabled() -> bool:
    """The scorer environment is bound from the round's frozen policy."""
    return os.getenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY") == JUDGE_REASONING_POLICY


def prepare_judge_request(payload: Mapping, *, hosted_search: bool = False) -> tuple[str, dict]:
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
    # This is a selector, not a caller-owned provider policy. The broker
    # installs ZDR, data_collection=deny, no fallback and exact price bounds.
    body["provider"] = {"only": [provider]}
    controls = body.get("reasoning") or {}
    if not isinstance(controls, Mapping):
        raise ValueError("judge_reasoning_controls_unsupported")
    effort = controls.get("effort", body.get("reasoning_effort", "medium"))
    if (effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}
            or "reasoning_effort" in body and body["reasoning_effort"] != effort
            or hosted_search and effort == "max"):
        raise ValueError("judge_reasoning_controls_unsupported")
    body["reasoning"] = {"effort": effort, "exclude": False}
    if body.get("include_reasoning") is False:
        body["include_reasoning"] = True
    limit = body.get("max_tokens", 8192)
    if type(limit) is not int or not 1 <= limit <= 32768:
        raise ValueError("judge_output_limit_invalid")
    body["max_tokens"] = max(8192, limit)
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
    messages = []
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
            messages.append({"role": "assistant", "content": "".join(text),
                             **({"annotations": annotations} if annotations else {})})
        else:
            # A function call or refusal is unfinished judging, not a verdict.
            raise ValueError("judge_response_unfinished_tool")
    if hosted_search and not searched:
        raise ValueError("judge_search_missing")
    if not messages or not messages[-1]["content"].strip():
        raise ValueError("judge_response_missing_answer")
    return {**{key: copy.deepcopy(body[key]) for key in ("id", "model", "provider", "usage") if key in body},
            "choices": [{"index": 0, "finish_reason": "stop", "message": messages[-1]}]}


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
