"""Miner-facing SOURCE_ADD submission helpers.

The gateway intake endpoint accepts a full adapter manifest, while the miner UX
only asks for API/source details. These helpers convert those details into the
minimal private manifest shape accepted by the operator-run SOURCE_ADD funnel.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from research_lab.source_add import SourceAddSourceKind


SOURCE_ADD_SOURCE_KINDS: tuple[str, ...] = tuple(kind.value for kind in SourceAddSourceKind)

SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS: dict[str, str] = {
    "web": "general web, search, or crawl data",
    "filing": "regulatory, legal, or company filings",
    "news": "news articles and media coverage",
    "registry": "official business or public registries",
    "procurement": "contracts, tenders, and RFPs",
    "social": "social networks and public community activity",
    "hiring": "job postings, ATS data, and workforce changes",
    "tech_stack": "technologies a company uses, adds, or removes",
    "funding": "funding rounds, investors, acquisitions, and exits",
    "firmographic": "company attributes such as size, revenue, and location",
    "people": "professional profiles, roles, and career history",
    "intent": "research, demand, and buyer-intent signals",
    "reviews": "product, vendor, employer, and customer reviews",
    "events": "conferences, webinars, speakers, and attendance",
}

SOURCE_ADD_DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = (
    "evidence_refs",
    "snapshot_refs",
    "content_hashes",
    "normalized_text_hashes",
    "metadata_refs",
)

SOURCE_ADD_AUTH_TYPES: tuple[str, ...] = (
    "none",
    "api_key_header",
    "api_key_query",
    "bearer",
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


def normalize_source_add_url(value: str, *, field_name: str) -> str:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an http(s) URL")
    return raw.rstrip("/")


def _clean_short_text(value: str, *, field_name: str, max_length: int) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return cleaned


def _clean_optional_text(value: str, *, max_length: int) -> str:
    return " ".join(str(value or "").strip().split())[:max_length]


def _normalize_endpoint_example(item: Mapping[str, Any], index: int) -> dict[str, str]:
    method = str(item.get("method") or "").strip().upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise ValueError(f"endpoint_examples[{index}].method must be GET, POST, PUT, PATCH, or DELETE")
    path = str(item.get("path") or "").strip()
    if not path.startswith("/") or "://" in path or any(ch.isspace() for ch in path):
        raise ValueError(f"endpoint_examples[{index}].path must be a relative API path like /v1/search")
    purpose = _clean_short_text(
        str(item.get("purpose") or ""),
        field_name=f"endpoint_examples[{index}].purpose",
        max_length=300,
    )
    example_query = _clean_short_text(
        str(item.get("example_query") or item.get("example") or ""),
        field_name=f"endpoint_examples[{index}].example_query",
        max_length=500,
    )
    return {
        "method": method,
        "path": path[:160],
        "purpose": purpose,
        "example_query": example_query,
    }


def normalize_source_add_endpoint_examples(examples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    if not examples:
        raise ValueError("at least one endpoint example is required")
    normalized = tuple(_normalize_endpoint_example(item, idx) for idx, item in enumerate(examples, start=1))
    if len(normalized) > 12:
        raise ValueError("at most 12 endpoint examples are allowed")
    return normalized


def build_source_add_metadata(
    *,
    api_base_url: str,
    documentation_url: str,
    auth_type: str,
    endpoint_examples: Sequence[Mapping[str, Any]],
    rate_limit_notes: str,
    data_provenance_notes: str = "",
    third_party_refs: Sequence[str] = (),
) -> dict[str, object]:
    normalized_auth = str(auth_type or "").strip().lower()
    if normalized_auth not in SOURCE_ADD_AUTH_TYPES:
        raise ValueError("auth_type must be one of: " + ", ".join(SOURCE_ADD_AUTH_TYPES))
    api_url = normalize_source_add_url(api_base_url, field_name="api_base_url")
    docs_url = normalize_source_add_url(documentation_url, field_name="documentation_url")
    examples = normalize_source_add_endpoint_examples(endpoint_examples)
    notes = _clean_short_text(rate_limit_notes, field_name="rate_limit_notes", max_length=1000)
    refs: list[str] = []
    for ref in third_party_refs:
        cleaned = str(ref or "").strip()
        if not cleaned:
            continue
        refs.append(normalize_source_add_url(cleaned, field_name="third_party_refs"))
    return {
        "api_base_url": api_url,
        "documentation_url": docs_url,
        "auth_type": normalized_auth,
        "endpoint_examples": list(examples),
        "rate_limit_notes": notes,
        "data_provenance_notes": _clean_optional_text(data_provenance_notes, max_length=1000),
        "third_party_refs": refs[:8],
    }


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
    declared_base_domains: tuple[str, ...] = (),
    endpoint_summary: str = "",
    claimed_output_type: str = "",
    credential_supplied: bool = False,
    api_base_url: str = "",
    documentation_url: str = "",
    auth_type: str = "",
    endpoint_examples: Sequence[Mapping[str, Any]] = (),
    rate_limit_notes: str = "",
    data_provenance_notes: str = "",
    third_party_refs: Sequence[str] = (),
    max_trial_cost_cents: int = 500,
    max_request_cost_cents: int = 50,
    max_latency_ms: int = 30_000,
) -> tuple[dict[str, object], str, str, dict[str, object]]:
    """Return ``(manifest, source_brief, idempotency_key, metadata)`` for intake.

    No raw credential is accepted here. The CLI passes the credential separately
    so the gateway can KMS-encrypt it and exclude it from the signed payload.
    """

    normalized_kind = str(source_kind or "").strip().lower()
    if normalized_kind not in SOURCE_ADD_SOURCE_KINDS:
        raise ValueError("source_kind must be one of: " + ", ".join(SOURCE_ADD_SOURCE_KINDS))
    metadata: dict[str, object] = {}
    if api_base_url or documentation_url or auth_type or endpoint_examples or rate_limit_notes:
        metadata = build_source_add_metadata(
            api_base_url=api_base_url,
            documentation_url=documentation_url,
            auth_type=auth_type,
            endpoint_examples=endpoint_examples,
            rate_limit_notes=rate_limit_notes,
            data_provenance_notes=data_provenance_notes,
            third_party_refs=third_party_refs,
        )
        derived_domains = parse_source_add_domains(
            " ".join(
                [
                    str(metadata["api_base_url"]),
                    str(metadata["documentation_url"]),
                    " ".join(str(item) for item in metadata.get("third_party_refs", [])),
                ]
            )
        )
        domains = tuple(domain for domain in (*declared_base_domains, *derived_domains) if domain)
    else:
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
        "source_metadata": metadata,
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
            f"API base URL: {metadata.get('api_base_url', '')}" if metadata else "",
            f"Documentation URL: {metadata.get('documentation_url', '')}" if metadata else "",
            f"Auth type: {metadata.get('auth_type', '')}" if metadata else "",
            f"Rate limit notes: {metadata.get('rate_limit_notes', '')}" if metadata else "",
            "Endpoint examples: "
            + json.dumps(metadata.get("endpoint_examples", []), sort_keys=True, separators=(",", ":"))[:900]
            if metadata
            else "",
            f"Claimed output type: {seed['claimed_output_type'] or 'unspecified'}",
            f"Endpoint details: {seed['endpoint_summary'] or 'not supplied'}",
            f"Auth material submitted separately: {'yes' if credential_supplied else 'no'}",
        )
        if line
    )
    idempotency_key = f"research-source-add:{miner_hotkey}:{digest[:24]}"
    return manifest, source_brief[:2000], idempotency_key, metadata
