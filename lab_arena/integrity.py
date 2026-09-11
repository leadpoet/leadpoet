"""Versioned Arena integrity policy and safe agent-facing ICP projection."""

from __future__ import annotations

import json
from typing import Any, Mapping

POLICY = "arena_integrity_v1"
SCORING_ADAPTER = "qualification_integrity_v2"

# Buyer requirements only. Generation receipts and examples never cross this
# boundary, including in nested signal objects added by future generators.
ICP_FIELDS = frozenset({
    "icp_id", "prompt", "industry", "sub_industry", "employee_count",
    "company_stage", "geography", "country", "state", "product_service",
    "required_attribute", "excluded_companies", "intent_signals",
    "intent_signal", "intent_category", "bonus_intents", "intent_max_age_days",
    "max_companies", "intent_signal_evidence_types",
})
SIGNAL_FIELDS = frozenset({
    "text", "intent_signal", "signal", "intent_category", "category",
    "evidence_type", "max_age_days", "description",
    "intent_max_age_days",
})
ATTRIBUTE_FIELDS = frozenset({"text", "description", "required"})
CONTACT_ICP_FIELDS = frozenset({"target_roles", "target_seniority", "contact_geography"})
CONTACT_GEOGRAPHY_FIELDS = frozenset({"countries", "regions", "cities"})
CONTACT_POLICY = "contacts_v1"


def enabled(configuration: Mapping[str, Any]) -> bool:
    if "integrity_policy" not in configuration:
        return False
    if configuration["integrity_policy"] != POLICY:
        raise ValueError("unsupported Arena integrity policy")
    return True


def _project(value: Any, allowed: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _project(item, allowed)
            for key, item in value.items() if key in allowed
        }
    if isinstance(value, list):
        return [_project(item, allowed) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("ICP values must be JSON values")


def _bounded_string_list(
    value: Any, field: str, *, maximum_items: int, maximum_length: int
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{field} must be a bounded list")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain strings")
        text = " ".join(item.strip().split())
        if not text or len(text) > maximum_length:
            raise ValueError(f"{field} contains an invalid value")
        if text not in result:
            result.append(text)
    return result


def agent_visible_icp(
    icp: Mapping[str, Any], *, contacts_required: bool = False
) -> dict[str, Any]:
    if not isinstance(icp, Mapping):
        raise ValueError("ICP must be an object")
    result = {}
    for key in ICP_FIELDS:
        if key not in icp:
            continue
        allowed = ATTRIBUTE_FIELDS if key == "required_attribute" else SIGNAL_FIELDS
        result[key] = _project(icp[key], allowed)
    if contacts_required:
        target_roles = icp.get("target_roles", [])
        target_seniority = icp.get("target_seniority", "")
        contact_geography = icp.get(
            "contact_geography",
            {"countries": [], "regions": [], "cities": []},
        )
        if not isinstance(target_seniority, str):
            raise ValueError("target_seniority must be a string")
        target_seniority = " ".join(target_seniority.strip().split())
        if len(target_seniority) > 80:
            raise ValueError("target_seniority is too long")
        if not isinstance(contact_geography, Mapping):
            raise ValueError("contact_geography must be an object")
        unexpected_geography = set(contact_geography) - CONTACT_GEOGRAPHY_FIELDS
        if unexpected_geography:
            raise ValueError("contact_geography contains unsupported fields")
        result.update(
            {
                "contact_policy": CONTACT_POLICY,
                "target_roles": _bounded_string_list(
                    target_roles,
                    "target_roles",
                    maximum_items=5,
                    maximum_length=120,
                ),
                "target_seniority": target_seniority,
                "contact_geography": {
                    field: _bounded_string_list(
                        contact_geography.get(field, []),
                        f"contact_geography.{field}",
                        maximum_items=25,
                        maximum_length=120,
                    )
                    for field in ("countries", "regions", "cities")
                },
            }
        )
    # Validate the actual serialized boundary; never return a view into private
    # benchmark objects which a downstream caller could mutate.
    return json.loads(json.dumps(result, allow_nan=False))


def requirement_fingerprint(
    icp: Mapping[str, Any], *, contacts_required: bool = False
) -> str:
    """Stable semantic input fingerprint excluding only its assigned identifier."""
    from lab_arena.contracts import document_hash

    visible = agent_visible_icp(icp, contacts_required=contacts_required)
    visible.pop("icp_id", None)
    if visible.get("industry") and (visible.get("intent_signal") or visible.get("intent_signals")):
        visible.pop("prompt", None)
    return document_hash(visible)
