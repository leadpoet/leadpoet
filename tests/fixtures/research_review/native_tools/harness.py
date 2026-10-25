"""Native assistant/tool transcript; execute only model-selected research calls."""

import json

_model = _tools = None
_trace = None
MAX_STEPS = 4


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    messages = [{"role": "user", "content": json.dumps(icp)}]
    for _ in range(MAX_STEPS):
        reply = _model({"messages": messages, "tools": ["search", "fetch"], "tool_choice": "auto"})
        if not isinstance(reply, dict) or reply.get("role") != "assistant":
            raise ValueError("invalid assistant message")
        calls = reply.get("tool_calls")
        if not calls:
            try:
                final = json.loads(reply["content"])
                companies = final["companies"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid final companies") from exc
            if not isinstance(companies, list) or any(not isinstance(x, dict) for x in companies):
                raise ValueError("invalid final companies")
            return companies
        if not isinstance(calls, list) or len(calls) > 3:
            raise ValueError("invalid tool calls")
        decoded, ids = [], set()
        for call in calls:
            try:
                call_id, function = call["id"], call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid tool calls") from exc
            if not isinstance(call_id, str) or not call_id or call_id in ids or not isinstance(name, str) or name not in {"search", "fetch"} or not isinstance(arguments, dict):
                raise ValueError("invalid tool calls")
            ids.add(call_id)
            decoded.append((call_id, name, arguments))
        messages.append(reply)
        for call_id, name, arguments in decoded:
            result = _tools(name, arguments)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
            if _trace is not None:
                _trace.append({"action": name, "tool_call_id": call_id})
    raise RuntimeError("research step limit reached")
