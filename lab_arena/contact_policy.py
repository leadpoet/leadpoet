"""Version boundary for rounds that require a verified contact per company."""

from __future__ import annotations

from typing import Any, Mapping

POLICY = "contacts_v1"
SCORING_ADAPTER = "qualification_contacts_v3"
OUTPUT_SCHEMA = "leadpoet.lab_arena.output.v2"


def enabled(config: Mapping[str, Any]) -> bool:
    if "contact_policy" not in config:
        return False
    if config["contact_policy"] != POLICY:
        raise ValueError("unsupported contact policy")
    return True


def scorer_enabled(policy: Mapping[str, Any]) -> bool:
    return policy.get("scoring_adapter_version") == SCORING_ADAPTER


def integrity_adapter(version: str) -> bool:
    return version in ("qualification_integrity_v2", SCORING_ADAPTER)


def output_schema(config: Mapping[str, Any]) -> str:
    from lab_arena import quality_policy
    if quality_policy.enabled(config):
        return quality_policy.output_schema(contacts_required=enabled(config))
    return OUTPUT_SCHEMA if enabled(config) else "leadpoet.lab_arena.output.v1"



def validate_icp(icp: Mapping[str, Any]) -> dict[str, Any]:
    """A contact benchmark needs explicit, bounded buyer roles before intake closes."""
    from lab_arena import integrity
    from qualification.contact_models import normalize_country_code

    if icp.get("contact_policy") != POLICY:
        raise ValueError("contact benchmark requires contact policy")
    visible = integrity.agent_visible_icp(icp, contacts_required=True)
    if not visible["target_roles"]:
        raise ValueError("contact benchmark requires target roles")
    for country in visible["contact_geography"]["countries"]:
        normalize_country_code(country)
    return visible


def redact_contact_verification(value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish verdicts and evidence digests, never profile/provider bodies."""
    out = {key: value[key][:200] for key in ("decision", "reason", "verified_at")
           if isinstance(value.get(key), str)}
    checks = value.get("subchecks")
    if isinstance(checks, Mapping):
        out["subchecks"] = {
            key: {field: checks[key][field][:200] for field in ("status", "reason")
                  if isinstance(checks[key].get(field), str)}
            for key in ("claim", "identity", "source", "company", "role", "location", "email_attribution", "email_verification")
            if isinstance(checks.get(key), Mapping)
        }
    for field in ("evidence_hashes", "evidence_timestamps"):
        items = value.get(field)
        if isinstance(items, Mapping):
            out[field] = {key: items[key][:100] for key in ("source", "zerobounce", "bounceban")
                          if isinstance(items.get(key), str)}
    return out


def validate_contact_breakdown(row: Mapping[str, Any]) -> None:
    """Reject positive credit without a complete, successful contact verdict."""
    from lab_arena.contracts import ArenaContractError

    qualified = row.get("contact_qualified")
    status = row.get("email_status")
    verification = row.get("contact_verification")
    identity = row.get("contact_identity_key")
    if (type(qualified) is not bool or status not in {"valid", "catch_all", "invalid", "unknown"}
        or not isinstance(identity, str) or not identity or len(identity) > 1024
        or not isinstance(verification, Mapping)):
        raise ArenaContractError("invalid contact qualification receipt")
    decision = verification.get("decision")
    if decision not in {"verified", "mismatch", "unverified", "unavailable", "not_evaluated"}:
        raise ArenaContractError("invalid contact decision")
    if qualified != (decision == "verified"):
        raise ArenaContractError("contact qualification contradicts decision")
    if qualified != row.get("company_qualified"):
        raise ArenaContractError("company and contact qualification must agree")
    if qualified:
        required = {"claim", "identity", "source", "company", "role", "location", "email_attribution", "email_verification"}
        checks = verification.get("subchecks")
        if status not in {"valid", "catch_all"} or not isinstance(checks, Mapping) or any(
            not isinstance(checks.get(key), Mapping) or checks[key].get("status") != "pass"
            for key in required
        ):
            raise ArenaContractError("contact qualification requires every check")
    elif row.get("company_qualified") or float(row.get("final_score") or 0) != 0:
        raise ArenaContractError("contact failure cannot receive credit")
