"""Compose only the bounded single-tool protocol repair onto frozen research loops."""
from __future__ import annotations

import ast

CORRECTION_ID = "single_tool_protocol_repair_v1"


class ProtocolRepairSeamError(ValueError):
    """Frozen protocol source does not have the reviewed before/after shape."""


_HELPER = '''def _valid_multi_research_reply(
    tool_calls: list[Any], *, required_tool: str,
    required_search_locators: Optional[tuple[str, str]] = None,
) -> bool:
    """Check a rejected research batch without executing or selecting any call."""

    call_ids: set[str] = set()
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return False
        call_id = call.get("id")
        function = call.get("function")
        if (
            not isinstance(call_id, str) or not call_id or call_id in call_ids
            or call.get("type") != "function"
            or not isinstance(function, Mapping)
        ):
            return False
        call_ids.add(call_id)
        name = function.get("name")
        if not isinstance(name, str) or name not in {"search_web", "fetch_page"} or (
            required_tool and name != required_tool
        ):
            return False
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            return False
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            return False
        key = "query" if name == "search_web" else "url"
        if not isinstance(arguments, Mapping) or set(arguments) != {key}:
            return False
        value = arguments[key]
        if not isinstance(value, str) or not value.strip():
            return False
        if name == "search_web" and len(value) > 500:
            return False
        if name == "search_web" and required_search_locators is not None and not any(
            locator in _normalized_span(value) for locator in required_search_locators
        ):
            return False
        if name == "fetch_page" and (len(value) < 10 or not _safe_https_url(value)):
            return False
    return True


'''

_LOOP_OLD = '''                if (
                    not isinstance(message, Mapping)
                    or not isinstance(tool_calls, list)
                    or len(tool_calls) != 1
                    or not isinstance(tool_calls[0], Mapping)
                ):
                    raise ValueError("reasoning_tool_count_invalid")
'''

_REPAIR = '''                if (
                    isinstance(choices, list)
                    and len(choices) == 1
                    and isinstance(message, Mapping)
                    and isinstance(tool_calls, list)
                    and len(tool_calls) > 1
                    and _valid_multi_research_reply(
                        tool_calls,
                        required_tool=(
                            "submit_findings" if submit_only_response else required_tool
                        ),
                        required_search_locators=(
                            (_normalized_span(search_name), search_domain)
                            if forced_stage_search_call else None
                        ),
                    )
                    and not correction_turn
                    and _turn + 1 < MAX_REASONING_TURNS
                    and incomplete_submit_retries < MAX_INCOMPLETE_SUBMIT_RETRIES
                    and time.monotonic() - started < ADMISSION_DEADLINE_SECONDS
                ):
                    # No call from an ambiguous batch is executed. Reuse the
                    # one serialization repair, keeping research/turn limits.
                    incomplete_submit_retries += 1
                    forced_next_tool = required_tool
                    messages.append({
                        "role": "user",
                        "content": (
                            "The prior response contained multiple tool calls. "
                            "None was executed. Return exactly one complete tool "
                            "call using the currently allowed tools and required "
                            "tool choice. Do not emit multiple calls or prose."
                        ),
                    })
                    continue
'''

_NATIVE_REPAIR = _REPAIR.replace("_turn + 1 < MAX_REASONING_TURNS",
                                  "_turn + 1 < reasoning_turn_limit")


def convert_protocol_repair_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProtocolRepairSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise ProtocolRepairSeamError("grading source must use LF newlines")
    functions = {name: [node for node in module.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == name]
                 for name in ("_safe_https_url", "investigate_company_evidence",
                              "_valid_multi_research_reply")}
    if any(len(functions[name]) != 1 for name in (
        "_safe_https_url", "investigate_company_evidence",
    )) or len(functions["_valid_multi_research_reply"]) > 1:
        raise ProtocolRepairSeamError("protocol helper is missing or ambiguous")
    lines = text.splitlines(keepends=True)
    loop = functions["investigate_company_evidence"][0]
    loop_text = "".join(lines[loop.lineno - 1:loop.end_lineno])
    # Keep the frozen research budget, including turns reserved for review.
    bounds = [bound for bound in ("MAX_REASONING_TURNS", "reasoning_turn_limit")
              if f"for _turn in range({bound} + 1):" in loop_text]
    if len(bounds) != 1:
        raise ProtocolRepairSeamError("unrecognized protocol turn bound")
    repair = _REPAIR if bounds[0] == "MAX_REASONING_TURNS" else _NATIVE_REPAIR
    uses = [node for node in ast.walk(module) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_valid_multi_research_reply"]
    if text.count(_LOOP_OLD) != 1 or loop_text.count(_LOOP_OLD) != 1:
        raise ProtocolRepairSeamError("unrecognized protocol rejection block")
    corrected = bool(functions["_valid_multi_research_reply"])
    if corrected:
        helper = functions["_valid_multi_research_reply"][0]
        helper_text = "".join(lines[helper.lineno - 1:helper.end_lineno])
        if (helper_text != _HELPER.rstrip("\n") + "\n"
                or text.count(_HELPER) != 1 or len(uses) != 1
                or loop_text.count(repair + _LOOP_OLD) != 1):
            raise ProtocolRepairSeamError("partially corrected protocol source")
        return source
    if uses or _REPAIR in text or _NATIVE_REPAIR in text:
        raise ProtocolRepairSeamError("partially corrected protocol source")
    anchor = "def _safe_https_url(value: Any) -> str:\n"
    if text.count(anchor) != 1:
        raise ProtocolRepairSeamError("unrecognized URL helper anchor")
    text = text.replace(anchor, _HELPER + anchor, 1)
    text = text.replace(_LOOP_OLD, repair + _LOOP_OLD, 1)
    compile(text, "frozen_protocol_repair.py", "exec")
    return text.encode("utf-8")
