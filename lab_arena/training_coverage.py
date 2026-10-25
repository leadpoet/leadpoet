"""Frozen minimum ICP coverage for future champion training evidence."""

from __future__ import annotations

from collections.abc import Mapping


SETTING = "training_minimum_icp_coverage_percent"
NEW_ROUND_PERCENT = 80
LEGACY_PERCENT = 100


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
