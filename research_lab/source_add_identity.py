"""Canonical identity helpers for SOURCE_ADD submissions."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Mapping, Sequence

from .canonical import sha256_json


def normalize_source_add_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urllib.parse.urlsplit(text)
        host = parsed.netloc or parsed.path.split("/", 1)[0]
    except Exception:
        host = text
    host = host.split("@")[-1].split(":", 1)[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_source_add_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urllib.parse.urlsplit(text)
    except Exception:
        return ""
    scheme = (parsed.scheme or "https").lower()
    host = normalize_source_add_domain(parsed.netloc or parsed.path)
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    return urllib.parse.urlunsplit((scheme, host, path, "", ""))


def source_identity_payload(
    *,
    api_base_url: str = "",
    documentation_url: str = "",
    declared_base_domains: Sequence[str] = (),
) -> dict[str, Any]:
    api_url = normalize_source_add_url(api_base_url)
    docs_url = normalize_source_add_url(documentation_url)
    domains = {
        normalize_source_add_domain(item)
        for item in declared_base_domains
        if normalize_source_add_domain(item)
    }
    for url in (api_url, docs_url):
        domain = normalize_source_add_domain(url)
        if domain:
            domains.add(domain)
    return {
        "api_base_url": api_url,
        "documentation_url": docs_url,
        "domains": sorted(domains),
    }


def source_identity_hash(
    *,
    api_base_url: str = "",
    documentation_url: str = "",
    declared_base_domains: Sequence[str] = (),
) -> str:
    payload = source_identity_payload(
        api_base_url=api_base_url,
        documentation_url=documentation_url,
        declared_base_domains=declared_base_domains,
    )
    if not payload["api_base_url"] and not payload["documentation_url"] and not payload["domains"]:
        return ""
    return sha256_json({"source_identity": payload})


def source_identity_hash_from_metadata(
    metadata: Mapping[str, Any],
    *,
    declared_base_domains: Sequence[str] = (),
) -> str:
    return source_identity_hash(
        api_base_url=str(metadata.get("api_base_url") or ""),
        documentation_url=str(metadata.get("documentation_url") or ""),
        declared_base_domains=declared_base_domains,
    )
