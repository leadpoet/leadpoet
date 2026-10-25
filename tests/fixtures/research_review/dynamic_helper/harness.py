"""The controlling implementation can come from a runtime-selected helper."""

import importlib

_model = _tools = _helper = None
_trace = None


def configure(model, tools, trace=None, helper=None):
    global _model, _tools, _trace, _helper
    _model, _tools, _trace, _helper = model, tools, trace, helper


def run_icp(icp):
    helper = _helper or importlib.import_module(icp["research_helper"]).run
    return helper(icp, _model, _tools)
