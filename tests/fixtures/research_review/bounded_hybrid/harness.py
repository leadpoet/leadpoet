"""One deterministic discovery step, then model-owned research or early finish."""

import json

_model = _tools = None
_trace = None


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def run_icp(icp):
    observations = [{"tool": "search", "result": _tools("search", {"query": icp["industry"]})}]
    for _ in range(3):
        action = _model({"icp": icp, "observations": observations})
        action = json.loads(action) if isinstance(action, str) else action
        if not isinstance(action, dict):
            raise ValueError("invalid model action")
        if action.get("action") == "finish":
            companies = action.get("companies")
            if not isinstance(companies, list) or any(not isinstance(x, dict) for x in companies):
                raise ValueError("invalid final companies")
            return companies
        name, arguments = action.get("action"), action.get("arguments")
        if not isinstance(name, str) or name not in {"search", "fetch"} or not isinstance(arguments, dict):
            raise ValueError("invalid model action")
        observations.append({"tool": name, "result": _tools(name, arguments)})
        if _trace is not None:
            _trace.append({"action": name})
    raise RuntimeError("research step limit reached")
