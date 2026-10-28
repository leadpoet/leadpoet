"""Resolve the immutable capture contract of a promoted baseline source."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from lab_arena import reasoning_capture

FIELD = "baseline_reasoning_configuration"
EXPORT_FIELD = "execution_reasoning_configuration"
KEYS = frozenset({"reasoning_capture_policy", "reasoning_route_registry"})


def validate(value):
    if not isinstance(value, Mapping) or set(value) - KEYS:
        raise ValueError("baseline reasoning configuration is invalid")
    if value:
        if (set(value) != KEYS
                or value.get("reasoning_capture_policy") != reasoning_capture.POLICY):
            raise ValueError("baseline reasoning configuration is incomplete")
        reasoning_capture.validate_registry(value["reasoning_route_registry"])
    return copy.deepcopy(dict(value))


def from_configuration(configuration):
    return {key: copy.deepcopy(configuration[key]) for key in KEYS if key in configuration}


def for_run(round_row, run):
    """Use only the service's frozen baseline participant, never caller flags."""
    configuration = round_row.get("configuration_doc") or {}
    if FIELD not in configuration or run.get("kind", "execute") != "execute":
        return from_configuration(configuration)
    baseline_id = "baseline-" + str(round_row["round_id"]).removeprefix("arena-")
    if run.get("submission_id") != baseline_id:
        return from_configuration(configuration)
    matches = [participant for participant in round_row.get("participants") or []
               if participant.get("submission_id") == baseline_id
               and participant.get("is_king") is True
               and participant.get("miner_hotkey") == configuration.get("baseline_hotkey")
               and participant.get("miner_hotkey") == run.get("miner_hotkey")]
    if len(matches) != 1:
        raise ValueError("baseline reasoning participant is not frozen")
    return validate(configuration[FIELD])


def for_example(example):
    configuration = example.get("configuration") or {}
    if (example.get("role") == "baseline" and FIELD in configuration
            and EXPORT_FIELD not in example):
        raise ValueError("baseline reasoning export is missing")
    if EXPORT_FIELD in example:
        if example.get("role") != "baseline":
            raise ValueError("only the baseline has a separate reasoning contract")
        if configuration.get(FIELD) != example[EXPORT_FIELD]:
            raise ValueError("baseline reasoning export differs from frozen configuration")
        return validate(example[EXPORT_FIELD])
    return from_configuration(example.get("configuration") or {})
