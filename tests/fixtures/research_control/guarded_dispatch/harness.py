"""A runtime guard can dispatch to a controller absent from this source bundle."""

import importlib

_model = _tools = None
_trace = None
_controllers = {}


def configure(model, tools, trace=None, controllers=None):
    global _model, _tools, _trace, _controllers
    _model, _tools, _trace = model, tools, trace
    _controllers = {} if controllers is None else controllers


def run_icp(icp):
    if icp.get("use_external_controller"):
        name = icp["controller"]
        controller = _controllers.get(name)
        if controller is None:
            controller = importlib.import_module(name).run
        return controller(icp, _model, _tools)
    _tools("search", {"query": icp["industry"]})
    return []
