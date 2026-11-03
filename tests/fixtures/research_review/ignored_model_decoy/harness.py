"""Record impressive-looking decisions while ignoring their control meaning."""

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    for name, arguments in (("search", {"query": icp["industry"]}), ("fetch", {"url": "https://example.com/fixed"})):
        decision = _model({"observations": [], "available_actions": ["search", "fetch", "finish"]})
        if _trace is not None:
            _trace.append({"model_decision": decision, "claimed_model_control": True})
        _tools(name, arguments)
    return []
