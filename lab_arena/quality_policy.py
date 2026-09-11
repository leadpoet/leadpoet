"""Frozen policy for company identity, required fields and ICP coverage.

This is additive to the existing integrity/contact policies: an absent marker
always selects the historical behavior, including its arithmetic mean.
"""

from __future__ import annotations

from typing import Any, Mapping

POLICY = "company_quality_v1"
OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v3"
CONTACT_OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v4"
AGGREGATION = "sqrt_mean_v1"


def enabled(configuration: Mapping[str, Any]) -> bool:
    if "company_quality_policy" not in configuration:
        return False
    if configuration["company_quality_policy"] != POLICY:
        raise ValueError("unsupported company quality policy")
    return True


def scorer_enabled(policy: Mapping[str, Any]) -> bool:
    return enabled(policy)


def output_schema(*, contacts_required: bool = False) -> str:
    return CONTACT_OUTPUT_SCHEMA if contacts_required else OUTPUT_SCHEMA
