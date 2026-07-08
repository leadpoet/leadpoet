"""Miner-facing SOURCE_ADD submission helpers.

The gateway intake endpoint accepts a full adapter manifest, while the miner UX
only asks for API/source details. These helpers convert those details into the
minimal private manifest shape accepted by the operator-run SOURCE_ADD funnel.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlparse


SOURCE_ADD_SOURCE_KINDS: tuple[str, ...] = (
    "web",
    "filing",
    "news",
    "registry",
    "procurement",
    "social",
)

SOURCE_ADD_DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = (
    "evidence_refs",
    "snapshot_refs",
    "content_hashes",
    "normalized_text_hashes",
    "metadata_refs",
)


def normalize_source_add_domain(value: str) -> str:
    """Normalize user-provided host or URL into a bare domain."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    domain = (parsed.hostname or raw).strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.split(":", 1)[0]


def parse_source_add_domains(value: str) -> tuple[str, ...]:
    domains: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[\s,]+", str(value or "")):
        domain = normalize_source_add_domain(item)
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return tuple(domains)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return (slug or "source")[:48]


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_source_add_submission_docs(
    *,
    miner_hotkey: str,
    source_name: str,
    source_kind: str,
    declared_base_domains: tuple[str, ...],
    endpoint_summary: str,
    claimed_output_type: str,
    credential_supplied: bool,
    max_trial_cost_cents: int = 500,
    max_request_cost_cents: int = 50,
    max_latency_ms: int = 30_000,
) -> tuple[dict[str, object], str, str]:
    """Return ``(manifest, source_brief, idempotency_key)`` for intake.

    No raw credential is accepted here. The CLI passes the credential separately
    so the gateway can KMS-encrypt it and exclude it from the signed payload.
    """

    normalized_kind = str(source_kind or "").strip().lower()
    if normalized_kind not in SOURCE_ADD_SOURCE_KINDS:
        raise ValueError("source_kind must be one of: " + ", ".join(SOURCE_ADD_SOURCE_KINDS))
    domains = tuple(domain for domain in declared_base_domains if domain)
    if not domains:
        raise ValueError("at least one base domain is required")
    clean_name = str(source_name or "").strip()
    if not clean_name:
        raise ValueError("source_name is required")
    if max_request_cost_cents <= 0 or max_trial_cost_cents <= 0:
        raise ValueError("cost caps must be positive")
    if max_request_cost_cents > max_trial_cost_cents:
        raise ValueError("max_request_cost_cents cannot exceed max_trial_cost_cents")

    seed = {
        "miner_hotkey": str(miner_hotkey),
        "source_name": clean_name,
        "source_kind": normalized_kind,
        "declared_base_domains": list(domains),
        "endpoint_summary": str(endpoint_summary or "").strip()[:2000],
        "claimed_output_type": str(claimed_output_type or "").strip()[:200],
        "credential_supplied": bool(credential_supplied),
    }
    digest = _sha256_json(seed).split(":", 1)[1]
    adapter_id = f"adapter:{_slug(clean_name)}-{digest[:12]}"
    manifest = {
        "adapter_id": adapter_id,
        "miner_ref": f"miner:{miner_hotkey}",
        "source_name": clean_name,
        "source_kind": normalized_kind,
        "declared_base_domains": list(domains),
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": list(SOURCE_ADD_DEFAULT_OUTPUT_FIELDS),
        "submitted_artifact_ref": f"miner_source_suggestion:{digest[:16]}",
        "code_bundle_hash": f"sha256:{digest}",
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": int(max_trial_cost_cents),
        "max_request_cost_cents": int(max_request_cost_cents),
        "max_latency_ms": int(max_latency_ms),
        "credential_policy": "credential_ref_only" if credential_supplied else "no_credentials",
        "fixture_refs": [f"fixture:operator-trial:{digest[:16]}"],
    }
    source_brief = "\n".join(
        line
        for line in (
            f"Source name: {clean_name}",
            f"Source kind: {normalized_kind}",
            f"Base domains: {', '.join(domains)}",
            f"Claimed output type: {seed['claimed_output_type'] or 'unspecified'}",
            f"Endpoint details: {seed['endpoint_summary'] or 'not supplied'}",
            f"Auth material submitted separately: {'yes' if credential_supplied else 'no'}",
        )
        if line
    )
    idempotency_key = f"research-source-add:{miner_hotkey}:{digest[:24]}"
    return manifest, source_brief, idempotency_key
