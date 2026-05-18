"""
Hashing utilities for the lead fulfillment commit-reveal scheme.

Inlined from Leadpoet/utils/hashing.py so the gateway doesn't depend
on the Leadpoet package (which isn't installed on the gateway server).
"""

import json
import hashlib

HASH_SCHEMA_VERSION = 1

_JSON_NATIVE_TYPES = (str, int, float, bool, type(None), list, dict)


def canonical_json(obj: dict) -> str:
    """Deterministic JSON serialization with sorted keys and no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _check_json_native(obj, path: str = "root") -> None:
    """Raise TypeError if any value in the nested structure is not JSON-native."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_json_native(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_json_native(v, path=f"{path}[{i}]")
    elif not isinstance(obj, _JSON_NATIVE_TYPES):
        raise TypeError(
            f"Non-JSON-native type at {path}: {type(obj).__name__} = {obj!r}. "
            "Call model_dump(mode='json') before hashing."
        )


def hash_data(data: str) -> str:
    """SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# Fields that are present on the gateway-side FulfillmentLead model but
# must NOT participate in the commit-reveal hash. They're either:
#   - reveal-time-only metadata (miner-cited evidence about a lead)
#   - server-added fields with defaults that older miner code doesn't emit
# Including them in the hash breaks backward compatibility for any miner
# running pre-deploy code, because pydantic auto-fills the default at
# parse time and the canonical_json then disagrees with the miner's
# original (no-such-key) JSON. Once a field is in this set it can never
# be removed without bumping HASH_SCHEMA_VERSION.
_HASH_EXCLUDED_FIELDS = frozenset({
    # Added 2026-05-18 (commit 188bc8f0). Optional per-attribute miner-cited
    # evidence; supplied at reveal time, not part of lead identity. Including
    # it in the hash made every pre-deploy miner fail commit-reveal because
    # pydantic auto-fills `attribute_evidence: []` on parse.
    "attribute_evidence",
})


def hash_lead(lead_data: dict, schema_version: int = HASH_SCHEMA_VERSION) -> str:
    """Hash a complete lead dict (including PII) for commit-reveal binding.

    Strips fields in _HASH_EXCLUDED_FIELDS before hashing so the hash
    captures lead IDENTITY, not server-side metadata or reveal-only
    fields that older miner code doesn't emit.
    """
    _check_json_native(lead_data)
    hashable = {k: v for k, v in lead_data.items() if k not in _HASH_EXCLUDED_FIELDS}
    return hash_data(f"v{schema_version}:" + canonical_json(hashable))


def hash_request(icp_details: dict) -> str:
    """Hash ICP request details for transparency logging."""
    return hash_data(canonical_json(icp_details))


def verify_commit(
    committed_hash: str,
    lead_data: dict,
    schema_version: int = HASH_SCHEMA_VERSION,
) -> bool:
    """Check whether ``lead_data`` matches the previously committed hash."""
    return hash_lead(lead_data, schema_version) == committed_hash
