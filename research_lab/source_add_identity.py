"""Canonical private duplicate identities for SOURCE_ADD submissions."""

from __future__ import annotations

import re
import urllib.parse
import ipaddress
from typing import Any, Mapping, Sequence

from .canonical import sha256_json


SOURCE_ADD_IDENTITY_VERSION = "v2"
SOURCE_ADD_LEGACY_IDENTITY_VERSION = "v1"
SOURCE_ADD_PROVIDER_ORIGIN_VERSION = "v1"


def _source_add_url_candidate(value: str) -> str:
    text = str(value or "").strip()
    if "://" in text:
        return text
    if text.count(":") >= 2 and not text.startswith("["):
        return "https://[%s]" % text
    return "https://" + text


def _canonical_source_add_host(value: str) -> str:
    candidate = _source_add_url_candidate(value)
    try:
        parsed = urllib.parse.urlsplit(candidate)
        host = str(parsed.hostname or "").strip().lower().strip(".")
    except (TypeError, ValueError):
        return ""
    if host.startswith("www."):
        host = host[4:]
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        try:
            return host.encode("idna").decode("ascii")
        except UnicodeError:
            return ""


def normalize_source_add_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _canonical_source_add_host(text)


def normalize_source_add_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urllib.parse.urlsplit(_source_add_url_candidate(text))
    except (TypeError, ValueError):
        return ""
    scheme = (parsed.scheme or "https").lower()
    host = _canonical_source_add_host(text)
    if not host:
        return ""
    url_host = "[%s]" % host if ":" in host else host
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    return urllib.parse.urlunsplit((scheme, url_host, path, "", ""))


def normalize_source_add_api_base(value: str) -> str:
    """Normalize an API origin/base path without query or fragment aliases."""

    normalized = normalize_source_add_url(value)
    if not normalized:
        return ""
    parsed = urllib.parse.urlsplit(normalized)
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    return urllib.parse.urlunsplit(("https", parsed.netloc.lower(), path, "", ""))


def normalize_source_add_provider_origin(value: str) -> str:
    """Return the exact normalized provider host, independent of API paths."""

    text = str(value or "").strip()
    if not text or any(character.isspace() for character in text):
        return ""
    try:
        parsed = urllib.parse.urlsplit(text)
    except (TypeError, ValueError):
        return ""
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return ""
    authority = str(parsed.netloc or "")
    if authority.startswith("["):
        close = authority.find("]")
        if close <= 1 or authority[close + 1 :] not in ("", ":443"):
            return ""
        if "%" in authority[1:close]:
            return ""
    elif ":" in authority:
        if authority.count(":") != 1 or not authority.endswith(":443"):
            return ""
    host = str(parsed.hostname).lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host or len(host) > 253:
        return ""
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        return ""
    try:
        address = ipaddress.ip_address(host)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            return ""
        return address.compressed
    except ValueError:
        pass
    if re.fullmatch(r"[0-9.]+", host):
        return ""
    labels = host.split(".")
    if len(labels) < 2 or any(
        not label
        or len(label) > 63
        or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
        is None
        for label in labels
    ):
        return ""
    return host


def source_provider_origin_payload(api_base_url: str = "") -> dict[str, str]:
    """Build the independent provider-origin identity without changing v1/v2."""

    return {
        "identity_version": SOURCE_ADD_PROVIDER_ORIGIN_VERSION,
        "identity_kind": "provider_origin",
        "provider_host": normalize_source_add_provider_origin(api_base_url),
    }


def source_provider_origin_hash(api_base_url: str = "") -> str:
    """Hash one exact provider host so path aliases share one reservation."""

    payload = source_provider_origin_payload(api_base_url)
    if not payload["provider_host"]:
        return ""
    return sha256_json({"source_identity": payload})


def source_provider_origin_hash_from_metadata(metadata: Mapping[str, Any]) -> str:
    return source_provider_origin_hash(str(metadata.get("api_base_url") or ""))


def normalize_source_add_documentation_alias(value: str) -> str:
    """Return a controlled docs alias that cannot be widened by third parties."""

    normalized = normalize_source_add_url(value)
    if not normalized:
        return ""
    parsed = urllib.parse.urlsplit(normalized)
    path = parsed.path.rstrip("/") or "/"
    # Documentation products commonly move within one of these stable roots.
    # Keeping only the first stable root segment prevents a harmless quickstart
    # path change from opening a duplicate identity while preserving host scope.
    segments = [item for item in path.split("/") if item]
    alias_path = "/"
    if segments and segments[0].lower() in {
        "api",
        "apis",
        "developer",
        "developers",
        "doc",
        "docs",
        "reference",
    }:
        alias_path = "/" + segments[0].lower()
    return urllib.parse.urlunsplit(("https", parsed.netloc.lower(), alias_path, "", ""))


def source_identity_payload(
    *,
    api_base_url: str = "",
    documentation_url: str = "",
    declared_base_domains: Sequence[str] = (),
) -> dict[str, Any]:
    api_url = normalize_source_add_api_base(api_base_url)
    # The primary identity must not change when a miner edits only the docs URL.
    # Documentation has its own controlled alias reservation below. Third-party
    # references and declared domains are deliberately excluded from v2.
    return {
        "identity_version": SOURCE_ADD_IDENTITY_VERSION,
        "identity_kind": "api_base",
        "api_base_url": api_url,
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
    if not payload["api_base_url"]:
        return ""
    return sha256_json({"source_identity": payload})


def source_documentation_identity_hash(documentation_url: str = "") -> str:
    """Hash one normalized first-party documentation alias independently."""

    alias = normalize_source_add_documentation_alias(documentation_url)
    if not alias:
        return ""
    return sha256_json(
        {
            "source_identity": {
                "identity_version": SOURCE_ADD_IDENTITY_VERSION,
                "identity_kind": "documentation_alias",
                "documentation_alias": alias,
            }
        }
    )


def source_identity_alias_hashes_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return controlled secondary v2 reservations for submitted metadata."""

    docs_hash = source_documentation_identity_hash(
        str(metadata.get("documentation_url") or "")
    )
    return (docs_hash,) if docs_hash else ()


def legacy_source_identity_hash(
    *,
    api_base_url: str = "",
    documentation_url: str = "",
    declared_base_domains: Sequence[str] = (),
) -> str:
    """Reproduce the deployed v1 identity during the migration window."""

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
    payload = {
        "api_base_url": api_url,
        "documentation_url": docs_url,
        "domains": sorted(domains),
    }
    if not api_url and not docs_url and not domains:
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


__all__ = [
    "SOURCE_ADD_LEGACY_IDENTITY_VERSION",
    "SOURCE_ADD_IDENTITY_VERSION",
    "SOURCE_ADD_PROVIDER_ORIGIN_VERSION",
    "legacy_source_identity_hash",
    "normalize_source_add_api_base",
    "normalize_source_add_documentation_alias",
    "normalize_source_add_domain",
    "normalize_source_add_provider_origin",
    "normalize_source_add_url",
    "source_identity_hash",
    "source_identity_alias_hashes_from_metadata",
    "source_identity_hash_from_metadata",
    "source_identity_payload",
    "source_documentation_identity_hash",
    "source_provider_origin_hash",
    "source_provider_origin_hash_from_metadata",
    "source_provider_origin_payload",
]
