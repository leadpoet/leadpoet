"""Bounded JSON action protocol with injectable local model and research tools."""

import json

_model = _tools = None
_trace = None
MAX_STEPS = 4


def configure(model, tools, trace=None):
    global _model, _tools, _trace
    _model, _tools, _trace = model, tools, trace


def _decode(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("invalid model action")
    return value


def run_icp(icp):
    observations = []
    for _ in range(MAX_STEPS):
        action = _decode(_model({"icp": icp, "observations": observations}))
        if action.get("action") == "finish":
            companies = action.get("companies")
            if not isinstance(companies, list) or any(not isinstance(x, dict) for x in companies):
                raise ValueError("invalid final companies")
            return companies
        if not isinstance(action.get("action"), str) or action["action"] not in {"search", "fetch"} or not isinstance(action.get("arguments"), dict):
            raise ValueError("invalid model action")
        name, arguments = action["action"], action["arguments"]
        observation = _tools(name, arguments)
        observations.append({"tool": name, "arguments": arguments, "result": observation})
        if _trace is not None:
            _trace.append({"action": name})
    raise RuntimeError("research step limit reached")
