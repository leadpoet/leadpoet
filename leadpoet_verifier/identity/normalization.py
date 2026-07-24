from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

import idna


NORMALIZATION_VERSION = "company-identity-normalization-v1"
PSL_SNAPSHOT_VERSION = "2026-07-15_18-13-59_UTC"
PSL_SNAPSHOT_SHA256 = "545d7c4b561104b293fe182b1b3f40e636209c5089d4d9f4d5481e4defc1f35a"

_ASCII_EDGE_WHITESPACE = " \t\r\n\f\v"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_LEGAL_SUFFIXES = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "limited", "llc", "llp", "ltd", "nv", "oy", "plc",
    "pte", "pty", "sa", "sas", "sarl", "spa", "srl",
}
_LINKEDIN_HOST = re.compile(r"^(?:[a-z]{2}\.)?(?:www\.)?linkedin\.com$")


class NormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class DomainParts:
    ascii_host: str
    unicode_host: str
    public_suffix: str
    registrable_domain: str
    is_private_suffix: bool


@dataclass(frozen=True)
class NormalizedUrl:
    url: str
    scheme: str
    ascii_host: str
    unicode_host: str
    port: Optional[int]
    origin: str
    path: str
    query: str
    domain: DomainParts


@dataclass(frozen=True)
class LinkedInUrl:
    canonical_url: str
    normalized_slug: str
    company_id: Optional[str]


class PublicSuffixSnapshot:
    def __init__(self, text: str) -> None:
        self.exact: dict[str, bool] = {}
        self.wildcards: dict[str, bool] = {}
        self.exceptions: dict[str, bool] = {}
        private = False
        for raw in text.splitlines():
            line = raw.strip()
            if line == "// ===BEGIN PRIVATE DOMAINS===":
                private = True
                continue
            if not line or line.startswith("//"):
                continue
            rule = line.lstrip("!*. ")
            if rule.isascii():
                normalized = rule.lower()
            else:
                try:
                    normalized = ".".join(
                        idna.encode(label, uts46=True, transitional=False, std3_rules=True).decode("ascii")
                        for label in rule.split(".")
                    ).lower()
                except idna.IDNAError as exc:
                    raise RuntimeError(f"invalid rule in vendored PSL: {line}") from exc
            if line.startswith("!"):
                self.exceptions[normalized] = private
            elif line.startswith("*."):
                self.wildcards[normalized] = private
            else:
                self.exact[normalized] = private

    def split(self, ascii_host: str) -> tuple[str, str, bool]:
        labels = ascii_host.split(".")
        exception: Optional[tuple[int, bool]] = None
        matches: list[tuple[int, bool]] = []
        for index in range(len(labels)):
            candidate = ".".join(labels[index:])
            if candidate in self.exceptions:
                exception = (len(labels) - index - 1, self.exceptions[candidate])
                break
            if candidate in self.exact:
                matches.append((len(labels) - index, self.exact[candidate]))
            if index + 1 < len(labels):
                wildcard_base = ".".join(labels[index + 1:])
                if wildcard_base in self.wildcards:
                    matches.append((len(labels) - index, self.wildcards[wildcard_base]))
        suffix_labels, private = exception or max(matches, default=(1, False), key=lambda item: item[0])
        suffix = ".".join(labels[-suffix_labels:])
        if len(labels) <= suffix_labels:
            raise NormalizationError("host is a public or private suffix, not a company domain")
        registrable = ".".join(labels[-(suffix_labels + 1):])
        return suffix, registrable, private


@lru_cache(maxsize=1)
def public_suffix_snapshot() -> PublicSuffixSnapshot:
    path = Path(__file__).with_name("public_suffix_list.dat")
    payload = path.read_bytes()
    import hashlib

    if hashlib.sha256(payload).hexdigest() != PSL_SNAPSHOT_SHA256:
        raise RuntimeError("vendored Public Suffix List digest mismatch")
    return PublicSuffixSnapshot(payload.decode("utf-8"))


def normalize_host(value: str) -> DomainParts:
    host = value.strip(_ASCII_EDGE_WHITESPACE).rstrip(".")
    if not host or _CONTROL.search(host) or "\\" in host:
        raise NormalizationError("invalid hostname")
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        raise NormalizationError("IP literals cannot be company domains")
    try:
        ascii_host = idna.encode(
            host, uts46=True, transitional=False, std3_rules=True
        ).decode("ascii").lower()
        unicode_host = idna.decode(ascii_host.encode("ascii"), uts46=True, std3_rules=True)
    except idna.IDNAError as exc:
        raise NormalizationError("hostname is not valid UTS #46/IDNA") from exc
    if len(ascii_host) > 253 or "." not in ascii_host:
        raise NormalizationError("hostname must be a bounded multi-label domain")
    suffix, registrable, private = public_suffix_snapshot().split(ascii_host)
    return DomainParts(ascii_host, unicode_host, suffix, registrable, private)


def _normalized_split(raw: str, *, allow_bare_domain: bool) -> SplitResult:
    value = raw.strip(_ASCII_EDGE_WHITESPACE)
    if not value or _CONTROL.search(value) or "\\" in value or _BAD_PERCENT.search(value):
        raise NormalizationError("URL contains unsafe or malformed characters")
    if allow_bare_domain and "://" not in value:
        value = "https://" + value
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise NormalizationError("URL must be absolute HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise NormalizationError("URL userinfo is forbidden")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise NormalizationError("URL port is invalid") from exc
    return parsed


def normalize_url(raw: str, *, allow_bare_domain: bool = False) -> NormalizedUrl:
    parsed = _normalized_split(raw, allow_bare_domain=allow_bare_domain)
    domain = normalize_host(parsed.hostname or "")
    scheme = parsed.scheme.lower()
    port = parsed.port
    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    netloc = domain.ascii_host if port is None else f"{domain.ascii_host}:{port}"
    path = parsed.path or "/"
    # Decoding is validation only. Keep the escaped path exactly as supplied.
    try:
        unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise NormalizationError("URL path is not valid UTF-8") from exc
    normalized = urlunsplit((scheme, netloc, path, parsed.query, ""))
    return NormalizedUrl(
        normalized, scheme, domain.ascii_host, domain.unicode_host, port,
        f"{scheme}://{netloc}", path, parsed.query, domain,
    )


def normalize_linkedin_company_url(raw: str) -> LinkedInUrl:
    parsed = _normalized_split(raw, allow_bare_domain=False)
    host = (parsed.hostname or "").lower().rstrip(".")
    if not _LINKEDIN_HOST.fullmatch(host):
        raise NormalizationError("not a supported LinkedIn host")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) != 2 or segments[0].lower() != "company":
        raise NormalizationError("LinkedIn URL must be an exact company route")
    slug = segments[1].casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", slug):
        raise NormalizationError("LinkedIn company slug or ID is invalid")
    return LinkedInUrl(
        canonical_url=f"https://linkedin.com/company/{slug}",
        normalized_slug=slug,
        company_id=slug if slug.isdigit() else None,
    )


def normalize_name(value: str, *, strip_legal_suffix: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("&", " and ")
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    if strip_legal_suffix:
        while tokens and tokens[-1] in _LEGAL_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


def exact_or_legal_name_match(left: str, right: str) -> bool:
    exact_left, exact_right = normalize_name(left), normalize_name(right)
    if exact_left and exact_left == exact_right:
        return True
    legal_left = normalize_name(left, strip_legal_suffix=True)
    legal_right = normalize_name(right, strip_legal_suffix=True)
    return bool(legal_left and legal_left == legal_right)


def is_label_subdomain(child: str, parent: str) -> bool:
    child_host = normalize_host(child).ascii_host
    parent_host = normalize_host(parent).ascii_host
    return child_host != parent_host and child_host.endswith("." + parent_host)
