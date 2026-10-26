"""The model narrates a research path only after fixed tool calls finish."""

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    page = _tools("search", {"query": icp["industry"]})
    detail = _tools("fetch", {"url": page["url"]})
    narration = _model({"evidence": detail, "instruction": "Explain the research steps"})
    if _trace is not None:
        _trace.append({"narration": narration})
    return [detail]
