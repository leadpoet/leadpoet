"""Deterministic attestation response and PCR0 allowlist helpers."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def load_pcr0_allowlist(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load the current gateway/validator PCR0 allowlist JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "gateway": list(data.get("gateway_pcr0") or []),
        "validator": list(data.get("validator_pcr0") or []),
        "verifier": list(data.get("verifier_pcr0") or []),
    }


def is_pcr0_allowed(
    pcr0: str,
    allowlist: Mapping[str, List[Mapping[str, Any]]],
    *,
    role: str,
) -> bool:
    """Return True when pcr0 is present in the role-specific allowlist."""
    normalized_role = role.strip().lower()
    if normalized_role.endswith("_pcr0"):
        normalized_role = normalized_role[:-5]
    wanted = (pcr0 or "").lower()
    return any(
        str(entry.get("pcr0", "")).lower() == wanted
        for entry in allowlist.get(normalized_role, [])
    )


def validate_attestation_response_shape(response: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate public attestation endpoint shape without sealed crypto deps.

    This is necessary but not sufficient for enclave proof. A shape-valid
    response and allowlisted PCR0 still need the cryptographic Nitro/COSE
    verification path before callers treat the response as an attested
    execution result.
    """
    errors: List[str] = []
    trust_level = str(response.get("trust_level") or "")
    if trust_level not in {"full_nitro", "signature_only"}:
        errors.append("trust_level must be full_nitro or signature_only")

    attestation_document = response.get("attestation_document")
    if trust_level == "full_nitro":
        if not isinstance(attestation_document, str) or not attestation_document:
            errors.append("full_nitro responses must include attestation_document")
        elif not _is_base64(attestation_document):
            errors.append("attestation_document must be base64")

    pubkey = response.get("enclave_pubkey")
    if not _is_hex_length(pubkey, 64):
        errors.append("enclave_pubkey must be 32-byte hex")

    code_hash = response.get("code_hash")
    if not _is_hex_length(code_hash, 64):
        errors.append("code_hash must be sha256 hex")

    pcr0 = response.get("pcr0")
    if pcr0 is not None and not _is_hex_length(pcr0, 96):
        errors.append("pcr0 must be 48-byte hex when present")

    return {"passed": not errors, "errors": errors}


def _is_hex_length(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and bool(_HEX_RE.match(value))


def _is_base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
        return True
    except Exception:
        return False
