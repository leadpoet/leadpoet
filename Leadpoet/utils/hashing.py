"""
Shared hashing utilities for the lead fulfillment commit-reveal scheme.

Lives in Leadpoet/utils/ so both miners and the gateway import the same
``hash_lead()`` function — single source of truth for hash agreement.
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


# Fields that must NOT participate in the commit-reveal hash. Keep in sync
# with gateway/fulfillment/hashing.py — the two functions must produce
# byte-identical hashes for the same lead.
_HASH_EXCLUDED_FIELDS = frozenset({
    "attribute_evidence",  # added 188bc8f0; optional reveal-time miner evidence
})


def hash_lead(lead_data: dict, schema_version: int = HASH_SCHEMA_VERSION) -> str:
    """
    Hash a complete lead dict (including PII) for commit-reveal binding.

    ``lead_data`` **must** be the output of ``FulfillmentLead.model_dump(mode='json')``.

    Fields in ``_HASH_EXCLUDED_FIELDS`` are stripped before hashing so that
    server-side metadata or reveal-time-only fields (like ``attribute_evidence``)
    can be added to ``FulfillmentLead`` without breaking commit-reveal for
    miners running pre-deploy code.
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
