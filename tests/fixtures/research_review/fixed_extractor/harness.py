"""Research stages are fixed; the model only formats already collected facts."""

import json

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    results = _tools("search", {"query": icp["industry"]})
    page = _tools("fetch", {"url": results["url"]})
    final = _model({"icp": icp, "evidence": page, "schema": "companies"})
    final = json.loads(final) if isinstance(final, str) else final
    return final["companies"]
