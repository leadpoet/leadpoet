"""Frozen minimum ICP coverage for future champion training evidence."""

from __future__ import annotations

from collections.abc import Mapping


SETTING = "training_minimum_icp_coverage_percent"
NEW_ROUND_PERCENT = 80
LEGACY_PERCENT = 100
MISSING_REASONING_SETTING = "training_maximum_missing_reasoning_percent"
NEW_MISSING_REASONING_PERCENT = 1


def missing_reasoning_percent_from_configuration(configuration: Mapping) -> int:
    """Keep the zero-tolerance rule for rounds frozen before this setting."""
    value = configuration.get(MISSING_REASONING_SETTING, 0)
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("training maximum missing reasoning percent is invalid")
    return value


def percent_from_configuration(configuration: Mapping) -> int:
    """An absent setting retains the full-coverage rule of frozen old rounds."""

    if SETTING not in configuration:
        return LEGACY_PERCENT
    value = configuration[SETTING]
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("training minimum ICP coverage percent is invalid")
    return value


def required_count(bank_size: int, percent: int) -> int:
    """Round the required number of complete ICPs up without floating point."""

    if type(bank_size) is not int or bank_size < 1:
        raise ValueError("ICP bank size is invalid")
    if type(percent) is not int or not 1 <= percent <= 100:
        raise ValueError("training minimum ICP coverage percent is invalid")
    return (bank_size * percent + 99) // 100


def uses_host_result_evidence(configuration: Mapping) -> bool:
    """Only validated new frozen full-thinking rounds omit client callbacks."""
    from lab_arena import reasoning_capture
    try:
        if (not isinstance(configuration, Mapping)
                or configuration.get("reasoning_capture_policy") != reasoning_capture.POLICY
                or MISSING_REASONING_SETTING not in configuration):
            return False
        missing_reasoning_percent_from_configuration(configuration)
        percent_from_configuration(configuration)
        reasoning_capture.validate_registry(configuration.get("reasoning_route_registry"))
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
