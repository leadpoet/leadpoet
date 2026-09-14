"""Frozen output and scoring boundary for company-level intent details."""

from __future__ import annotations

from typing import Any, Mapping


POLICY = "intent_details_v1"
OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v5"


def enabled(document: Mapping[str, Any]) -> bool:
    if "intent_details_policy" not in document:
        return False
    if document["intent_details_policy"] != POLICY:
        raise ValueError("unsupported intent details policy")
    return True


def scorer_enabled(policy: Mapping[str, Any]) -> bool:
    return enabled(policy)
