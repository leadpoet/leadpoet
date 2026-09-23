"""Frozen output and scoring boundary for company-level intent details."""

from __future__ import annotations

from typing import Any, Mapping


POLICY = "intent_details_v1"
# Historical intent-details rounds required contacts and wrote V5. Keep the
# compatibility name pinned to that frozen schema.
OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v5"
COMPANY_ONLY_OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v6"


def output_schema(*, contacts_required: bool) -> str:
    """Select the intent-details shape without changing its scoring policy."""

    return OUTPUT_SCHEMA if contacts_required else COMPANY_ONLY_OUTPUT_SCHEMA


def enabled(document: Mapping[str, Any]) -> bool:
    if "intent_details_policy" not in document:
        return False
    if document["intent_details_policy"] != POLICY:
        raise ValueError("unsupported intent details policy")
    return True


def scorer_enabled(policy: Mapping[str, Any]) -> bool:
    return enabled(policy)
