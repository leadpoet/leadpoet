"""A model plans initial queries, then Python runs every fixed research stage."""

import json

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    plan = _model({"icp": icp, "stage": "plan"})
    plan = json.loads(plan) if isinstance(plan, str) else plan
    pages = []
    for query in plan["queries"][:2]:
        results = _tools("search", {"query": query})
        pages.append(_tools("fetch", {"url": results["url"]}))
    final = _model({"stage": "extract", "evidence": pages})
    final = json.loads(final) if isinstance(final, str) else final
    return final["companies"]
