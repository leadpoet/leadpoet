"""Resolve submitted source references using the scored execution's ledger."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping, Sequence

from qualification.contact_models import validate_contact_claim

INVALID_REFERENCE = {"invalid": True, "reason": "email_source_reference_invalid"}


def source_key(company: Mapping[str, Any]) -> str:
    contact = company.get("contact")
    source = contact.get("email_source") if isinstance(contact, Mapping) else None
    if not isinstance(source, Mapping):
        return ""
    return str(source.get("broker_call_id") or source.get("record_id") or "")


def resolve_sources(store: Any, run: Mapping[str, Any], companies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No miner-supplied response body or cross-execution reference is trusted."""
    resolved: dict[str, Any] = {}
    for company in companies[:5]:
        try:
            contact = validate_contact_claim(company.get("contact"))
        except (ValueError, TypeError):
            continue
        source = contact["email_source"]
        key = source_key(company)
        if key in resolved:
            continue
        evidence = {"provider": source["provider"], "tool": source["tool"],
                    "input": {"url": contact["linkedin_url"], "findEmail": "true"}}
        call_id = source.get("broker_call_id")
        if not call_id:
            # The trusted verifier re-fetches the profile and checks record_id.
            resolved[key] = evidence
            continue
        rows = store.list_ledger(run_id=str(run["run_id"]), call_identity=call_id)
        reserves = [row for row in rows if row.get("entry_kind") == "reservation"]
        settled = [row for row in rows if row.get("entry_kind") == "settlement"]
        if not reserves or not settled:
            resolved[key] = dict(INVALID_REFERENCE)
            continue
        reserve, terminal_row = reserves[-1], settled[-1]
        if any(row.get("run_id") != run["run_id"] or row.get("call_identity") != call_id
               or row.get("provider") != "deepline" or row.get("operation_id") != "deepline.execute"
               for row in (reserve, terminal_row)) or (reserve.get("entry_doc") or {}).get("tool") != source["tool"]:
            resolved[key] = dict(INVALID_REFERENCE)
            continue
        terminal = terminal_row.get("terminal_response") or {}
        try:
            raw = base64.b64decode(terminal["body_b64"], validate=True)
            if len(raw) > 1024 * 1024 or not 200 <= int(terminal["status"]) < 300:
                raise ValueError("source response invalid")
            response = json.loads(raw)
        except (KeyError, TypeError, ValueError, binascii.Error):
            resolved[key] = dict(INVALID_REFERENCE)
            continue
        resolved[key] = {**evidence, "response": response, "call_identity": call_id,
                         "observed_at": str(terminal_row.get("created_at") or "")}
    return resolved
