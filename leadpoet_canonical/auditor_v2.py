"""Independent PCR0 and Nitro verification for auditor v2 weight bundles."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_receipts import SCORING_ROLE, WEIGHT_ROLE
from leadpoet_canonical.weight_bundle_v2 import validate_weight_bundle_v2


IDENTITY_CACHE_SCHEMA_VERSION = "leadpoet.independent_pcr0_identities.v1"
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class AuditorV2Error(ValueError):
    """A v2 bundle lacks independently rebuilt code identity evidence."""


def load_identity_cache(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditorV2Error("independent PCR0 identity cache is unavailable") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != IDENTITY_CACHE_SCHEMA_VERSION
        or not isinstance(document.get("entries"), list)
    ):
        raise AuditorV2Error("independent PCR0 identity cache schema is invalid")
    return {"schema_version": IDENTITY_CACHE_SCHEMA_VERSION, "entries": list(document["entries"])}


def _identity_for_receipt(
    cache: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    role = str(receipt.get("role") or "")
    commit_sha = str(receipt.get("commit_sha") or "").lower()
    if role not in {SCORING_ROLE, WEIGHT_ROLE} or not _COMMIT_RE.fullmatch(commit_sha):
        raise AuditorV2Error("receipt code identity is invalid")
    matches = [
        entry
        for entry in cache.get("entries", [])
        if isinstance(entry, Mapping)
        and entry.get("role") == role
        and str(entry.get("commit_sha") or "").lower() == commit_sha
    ]
    if len(matches) != 1:
        raise AuditorV2Error("receipt has no unique independently rebuilt PCR0 identity")
    entry = dict(matches[0])
    pcr0 = str(entry.get("pcr0") or "").lower()
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise AuditorV2Error("independent PCR0 identity is invalid")
    if int(entry.get("verified_build_count") or 0) < 3:
        raise AuditorV2Error("independent PCR0 identity needs three matching builds")
    return entry


def verify_attested_weight_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    identity_cache: Mapping[str, Any],
    nitro_verifier: Optional[Callable[..., Any]] = None,
    require_allocation_ancestry: bool = True,
) -> Dict[str, Any]:
    """Verify canonical weights and every receipt against independent PCR0s."""

    verified = validate_weight_bundle_v2(
        bundle,
        require_allocation_ancestry=require_allocation_ancestry,
    )
    if nitro_verifier is None:
        from leadpoet_canonical.nitro import verify_nitro_attestation_full

        nitro_verifier = verify_nitro_attestation_full

    receipts = [bundle["weight_receipt"], *list(bundle.get("parent_receipts") or [])]
    observed = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise AuditorV2Error("v2 receipt is not an object")
        identity = _identity_for_receipt(identity_cache, receipt)
        role = str(receipt["role"])
        valid, data = nitro_verifier(
            attestation_b64=str(receipt.get("attestation_document_b64") or ""),
            expected_pubkey=str(receipt.get("enclave_pubkey") or ""),
            expected_purpose=str(receipt.get("purpose") or ""),
            expected_epoch_id=int(receipt.get("epoch_id", -1)),
            role="validator" if role == WEIGHT_ROLE else "gateway",
            skip_pcr0_verification=True,
        )
        if not valid:
            raise AuditorV2Error("AWS Nitro attestation verification failed")
        if data.get("purpose") != receipt.get("purpose"):
            raise AuditorV2Error("attested receipt purpose mismatch")
        if int(data.get("epoch_id", -1)) != int(receipt.get("epoch_id", -2)):
            raise AuditorV2Error("attested receipt epoch mismatch")
        if data.get("enclave_pubkey") != receipt.get("enclave_pubkey"):
            raise AuditorV2Error("attested receipt public key mismatch")
        if str(data.get("pcr0") or "").lower() != str(identity["pcr0"]).lower():
            raise AuditorV2Error("attested receipt PCR0 differs from independent build")
        observed.append(
            {
                "receipt_hash": str(receipt["receipt_hash"]),
                "role": role,
                "commit_sha": str(receipt["commit_sha"]),
                "pcr0": str(identity["pcr0"]),
                "verified_build_count": int(identity["verified_build_count"]),
            }
        )
    return {**verified, "independent_receipt_identities": observed}
