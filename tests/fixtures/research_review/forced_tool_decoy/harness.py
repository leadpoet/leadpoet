"""Native tool syntax alone does not make this fixed workflow adaptive."""

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    for name, arguments in (("search", {"query": icp["industry"]}), ("fetch", {"url": "https://example.com/fixed"})):
        reply = _model({"messages": [{"role": "user", "content": icp}], "tool_choice": {"type": "function", "function": {"name": name}}})
        if _trace is not None:
            _trace.append({"tool_calls": reply.get("tool_calls", []), "claimed_model_control": True})
        _tools(name, arguments)
    return []
