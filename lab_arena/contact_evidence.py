"""Resolve submitted source references using the scored execution's ledger."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping, Sequence

from lab_arena import contracts, operations
from qualification.contact_models import validate_contact_claim
from qualification.scoring.contact_verification import contact_source_semantics

INVALID_REFERENCE = {"invalid": True, "reason": "email_source_reference_invalid"}
_MAX_SOURCE_RESPONSE_BYTES = 1024 * 1024
_SETTLEMENT_SCAN_LIMIT = (
    max(profile["deepline"] for profile in contracts.EXECUTION_CALL_QUOTA_PROFILES)
    + 1
)


def source_key(company: Mapping[str, Any]) -> str:
    contact = company.get("contact")
    source = contact.get("email_source") if isinstance(contact, Mapping) else None
    if not isinstance(source, Mapping):
        return ""
    return str(source.get("broker_call_id") or source.get("record_id") or "")


def _decode_response(row: Mapping[str, Any]) -> Any:
    terminal = row.get("terminal_response") or {}
    try:
        raw = base64.b64decode(terminal["body_b64"], validate=True)
        if (
            len(raw) > _MAX_SOURCE_RESPONSE_BYTES
            or not 200 <= int(terminal["status"]) < 300
        ):
            raise ValueError("source response invalid")
        return json.loads(raw)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None


def _trusted_call(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    call_id: str,
    tool: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]] | None:
    reserves = [row for row in rows if row.get("entry_kind") == "reservation"]
    settled = [row for row in rows if row.get("entry_kind") == "settlement"]
    uncertain = [row for row in rows if row.get("entry_kind") == "uncertain"]
    finder = tool in operations.DEEPLINE_CONTACT_FINDER_IDENTITY_FIELDS
    if len(reserves) != 1 or len(settled) > 1:
        return None
    if len(settled) == 1:
        terminal_row = settled[0]
    elif (
        finder
        and len(uncertain) == 1
        and (uncertain[0].get("terminal_response") or {}).get("call_succeeded")
        is True
    ):
        # Confirmed-cost accounting can retain a successful provider response
        # while its final bill is still unknown. The exact call remains
        # pending in the ledger; its response may still prove the contact.
        terminal_row = uncertain[0]
    else:
        return None
    reserve = reserves[0]
    if any(
        row.get("run_id") != run_id
        or row.get("call_identity") != call_id
        or row.get("provider") != "deepline"
        or row.get("operation_id") != "deepline.execute"
        for row in (reserve, terminal_row)
    ) or (reserve.get("entry_doc") or {}).get("tool") != tool:
        return None
    response = _decode_response(terminal_row)
    return (response, terminal_row, reserve) if response is not None else None


def _finder_input(reserve: Mapping[str, Any], tool: str) -> Mapping[str, Any] | None:
    """Read only the host-sealed identity projection for a reviewed finder."""

    if tool not in operations.DEEPLINE_CONTACT_FINDER_IDENTITY_FIELDS:
        return None
    sealed = (reserve.get("entry_doc") or {}).get("contact_finder_input")
    if (
        not isinstance(sealed, Mapping)
        or set(sealed) != {"schema_version", "tool", "payload"}
        or sealed.get("schema_version")
        != operations.CONTACT_FINDER_INPUT_SCHEMA_VERSION
        or sealed.get("tool") != tool
        or not isinstance(sealed.get("payload"), Mapping)
    ):
        return None
    payload = dict(sealed["payload"])
    allowed = set(operations.DEEPLINE_CONTACT_FINDER_IDENTITY_FIELDS[tool])
    if not payload or set(payload) - allowed:
        return None
    if any(
        not isinstance(value, str) or not value.strip() or len(value) > 512
        for value in payload.values()
    ):
        return None
    return payload


def _record_matches(
    response: Any,
    evidence: Mapping[str, Any],
    contact: Mapping[str, Any],
    record_id: str,
) -> bool:
    try:
        target = contact_source_semantics(
            {
                **evidence,
                "response": {
                    "data": {
                        "element": {
                            "id": record_id,
                            "linkedinUrl": contact["linkedin_url"],
                            "email": contact["email"],
                        }
                    }
                },
            }
        )
        observed = contact_source_semantics({**evidence, "response": response})
    except (TypeError, ValueError):
        return False
    target_profiles = ((target or {}).get("response") or {}).get("profiles") or []
    observed_profiles = (
        ((observed or {}).get("response") or {}).get("profiles") or []
    )
    if len(target_profiles) != 1:
        return False
    expected = target_profiles[0]
    matches = [
        profile
        for profile in observed_profiles
        if profile.get("id") == expected.get("id")
        and profile.get("linkedin") == expected.get("linkedin")
        and contact["email"] in profile.get("emails", [])
    ]
    return len(matches) == 1


def resolve_sources(store: Any, run: Mapping[str, Any], companies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No miner-supplied response body or cross-execution reference is trusted."""
    resolved: dict[str, Any] = {}
    record_settlements: list[Mapping[str, Any]] | None = None
    record_settlements_complete = True
    run_id = str(run["run_id"])
    for company in companies[:5]:
        try:
            contact = validate_contact_claim(company.get("contact"))
        except (ValueError, TypeError):
            continue
        source = contact["email_source"]
        key = source_key(company)
        if key in resolved:
            continue
        evidence = {"provider": source["provider"], "tool": source["tool"]}
        call_id = source.get("broker_call_id")
        if not call_id:
            if source["tool"] != "harvestapi_get_profile":
                resolved[key] = dict(INVALID_REFERENCE)
                continue
            evidence["input"] = {
                "url": contact["linkedin_url"],
                "findEmail": "true",
            }
            if record_settlements is None:
                record_settlements = store.list_ledger(
                    run_id=run_id,
                    provider="deepline",
                    entry_kind="settlement",
                    limit=_SETTLEMENT_SCAN_LIMIT,
                )
                record_settlements_complete = (
                    len(record_settlements) < _SETTLEMENT_SCAN_LIMIT
                )
            candidates: list[str] = []
            if record_settlements_complete:
                for row in record_settlements:
                    if (
                        row.get("run_id") != run_id
                        or row.get("provider") != "deepline"
                        or row.get("operation_id") != "deepline.execute"
                        or row.get("entry_kind") != "settlement"
                    ):
                        continue
                    response = _decode_response(row)
                    candidate_id = str(row.get("call_identity") or "")
                    if candidate_id and response is not None and _record_matches(
                        response, evidence, contact, str(source["record_id"])
                    ):
                        candidates.append(candidate_id)
            if len(candidates) == 1:
                candidate_id = candidates[0]
                rows = store.list_ledger(run_id=run_id, call_identity=candidate_id)
                trusted = _trusted_call(
                    rows,
                    run_id=run_id,
                    call_id=candidate_id,
                    tool=source["tool"],
                )
                if trusted is not None and _record_matches(
                    trusted[0], evidence, contact, str(source["record_id"])
                ):
                    response, terminal_row, _reserve = trusted
                    resolved[key] = {
                        **evidence,
                        "response": response,
                        "call_identity": {
                            "call_id": candidate_id,
                            "record_id": str(source["record_id"]),
                        },
                        "observed_at": str(terminal_row.get("created_at") or ""),
                    }
                    continue
            # The trusted verifier re-fetches when no unique same-run source exists.
            resolved[key] = evidence
            continue
        rows = store.list_ledger(run_id=run_id, call_identity=call_id)
        trusted = _trusted_call(
            rows, run_id=run_id, call_id=call_id, tool=source["tool"]
        )
        if trusted is None:
            resolved[key] = dict(INVALID_REFERENCE)
            continue
        response, terminal_row, reserve = trusted
        if source["tool"] == "harvestapi_get_profile":
            evidence["input"] = {
                "url": contact["linkedin_url"],
                "findEmail": "true",
            }
        else:
            finder_input = _finder_input(reserve, source["tool"])
            if finder_input is None:
                resolved[key] = dict(INVALID_REFERENCE)
                continue
            evidence["input"] = finder_input
        resolved[key] = {**evidence, "response": response, "call_identity": call_id,
                         "observed_at": str(terminal_row.get("created_at") or "")}
    return resolved
