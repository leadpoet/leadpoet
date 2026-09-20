"""
Qualification System: Company-existence verification (company-mode only)

This is the company-mode equivalent of ``db_verification.py``.  In
lead-mode the model is required to point at a real row in the published
leads table; we verify by comparing field-for-field against Supabase.

Company-mode has no leads table to compare against — the model is asked
to surface companies from the open web.  Instead, we verify that the
``CompanyOutput.company_website`` resolves to a live, real-looking
company page that actually mentions the claimed company name.  This
plays the same anti-gaming role as DB verification: it makes it
expensive to invent a fictitious company.

Design constraints:

* Cheap (single HTTP GET, ~3-5s budget).  Validators score many leads
  back-to-back; a slow verification step bottlenecks the pipeline.
* No paid APIs.  Apify / LinkedIn / etc. are deliberately excluded —
  baking those into base-miner scoring would force every miner who
  builds on the model to take on licensing risk.  See
  ``gateway/qualification/models.py::CompanyOutput`` for the rationale.
* Three-state outcome.  A verified homepage-name binding is ``match``;
  a parked or contradicted identity is ``mismatch``; transport failure or
  insufficient evidence is ``unavailable``.  A domain that resembles the
  company name is never sufficient proof by itself.

Public API:
    verify_company_exists(company_name, company_website, timeout_secs=5,
                          company_linkedin=...) -> CompanyFitDecisionResult
"""

from __future__ import annotations

import asyncio
import codecs
from html.parser import HTMLParser
import json
import logging
import re
from typing import Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp

from leadpoet_verifier.identity.normalization import NormalizationError, normalize_url
from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult,
    company_fit_match,
    company_fit_mismatch,
    company_fit_unavailable,
    evaluate_company_identity,
)


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

_HTTP_TIMEOUT_SECS = 5
# Large first-party sites can place identity metadata after bundled style data.
# Keep the single fetch bounded while covering the observed 1.5 MB homepage.
_MAX_BYTES = 2_000_000
_TRANSIENT_FETCH_ATTEMPTS = 2
_TRANSIENT_FETCH_RETRY_DELAY_SECS = 0.25
_MAX_ORGANIZATION_LEGAL_NAME_ALIASES = 3
_MAX_ORGANIZATION_NAME_LENGTH = 200
_HTML_ENCODING_SNIFF_BYTES = 1024
_IDENTITY_SOURCE_HOST_LABELS = frozenset({"media", "news", "newsroom", "press"})

# Python also exposes binary transforms as codecs. This explicit text-codec
# subset prevents an untrusted HTTP or HTML label from selecting one of them.
_SAFE_HTML_CODECS = frozenset({
    "ascii",
    "big5",
    "cp866",
    "cp874",
    *(f"cp125{index}" for index in range(9)),
    "euc_jp",
    "euc_kr",
    "gb18030",
    "gbk",
    *(f"iso8859-{index}" for index in (*range(1, 12), *range(13, 17))),
    "iso2022_jp",
    "koi8-r",
    "koi8-u",
    "mac-cyrillic",
    "mac-roman",
    "shift_jis",
    "utf-8",
    "utf-16",
    "utf-16-be",
    "utf-16-le",
})
_CHARSET_PARAMETER_RE = re.compile(
    r"(?:^|;)\s*charset\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^;\s]*))",
    re.IGNORECASE,
)

# Headers that look like a real browser.  Some company sites refuse
# default ``python-aiohttp/...`` user agents with 403, which would
# falsely fail otherwise-legitimate companies.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Patterns indicating a parked / for-sale / GoDaddy-style landing page.
# Hit any of these and the company verification fails, since the URL
# is not a real company website even if it returns HTTP 200.
_PARKED_DOMAIN_PATTERNS = [
    r"\bthis domain (?:is|may be) for sale\b",
    r"\bbuy this domain\b",
    r"\bdomain.*available for purchase\b",
    r"\bregister(?:ed)? this domain\b",
    r"\bsedo(?:parking)?\b",
    r"\bgodaddy\b.*\bparked\b",
    r"\bnamecheap\b.*\bparked\b",
    r"\bhostgator\b.*\bdefault\b",
    r"\bunder construction\b",
    # Keep these terms local. Minified product pages can mention a coming-soon
    # feature and an unrelated email domain hundreds of kilobytes later.
    r"\bcoming soon\b.{0,80}\bdomain\b",
    r"\bdefault web site page\b",
]
_PARKED_DOMAIN_RE = re.compile("|".join(_PARKED_DOMAIN_PATTERNS), re.IGNORECASE)
_COPYRIGHT_LEGAL_NAME_RE = re.compile(
    r"(?:©|\bcopyright\b)\s*"
    r"(?:\d{4}(?:\s*[-\N{EN DASH}]\s*\d{4})?\s*)?"
    r"(?P<name>[a-z][a-z0-9&.,'’ -]{1,180}?\b(?:limited|ltd\.?|"
    r"incorporated|inc\.?|corporation|corp\.?|llc|plc|pty limited|"
    r"pty ltd\.?))"
    r"(?=\s+(?:abn|acn|all rights reserved)\b|[\s.]*$)",
    re.IGNORECASE,
)
# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _upgrade_plain_http_company_url(company_website: str) -> str:
    """Upgrade a conventional HTTP company URL to the V2 HTTPS transport.

    Only the default HTTP port is eligible.  Explicit nonstandard ports,
    credentials, malformed URLs, and every non-HTTP scheme are left unchanged
    so the measured provider transport can reject them fail-closed.
    """

    value = str(company_website or "").strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return value
    if (
        parsed.scheme.lower() != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 80)
    ):
        return value
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return urlunsplit(("https", host, parsed.path, parsed.query, parsed.fragment))


def _charset_parameter(value: object) -> str:
    """Return one declared charset label from a Content-Type value."""

    match = _CHARSET_PARAMETER_RE.search(str(value or ""))
    if match is None:
        return ""
    return next((item.strip() for item in match.groups() if item), "")


class _EarlyMetaCharsetParser(HTMLParser):
    """Read the first real HTML charset declaration from the bounded prefix."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.charset = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if self.charset or tag.casefold() != "meta":
            return
        attributes = {}
        for key, value in attrs:
            # HTML tokenization keeps the first duplicate attribute.
            attributes.setdefault(
                str(key or "").casefold(), str(value or "").strip()
            )
        direct = attributes.get("charset", "")
        candidate = direct
        if (
            not candidate
            and attributes.get("http-equiv", "").casefold() == "content-type"
        ):
            candidate = _charset_parameter(attributes.get("content", ""))
        # An invalid declaration does not hide a later valid declaration.
        self.charset = _safe_html_codec(candidate)
        if self.charset.startswith("utf-16"):
            # HTML meta declarations cannot select UTF-16; only BOM/HTTP can.
            self.charset = "utf-8"


def _early_meta_charset(raw: bytes) -> str:
    parser = _EarlyMetaCharsetParser()
    try:
        # HTML encoding labels are ASCII. Latin-1 preserves every source byte
        # while HTMLParser excludes declarations in comments and script data.
        parser.feed(raw[:_HTML_ENCODING_SNIFF_BYTES].decode("latin-1"))
        parser.close()
    except (AssertionError, ValueError):
        return ""
    return parser.charset


def _safe_html_codec(label: str) -> str:
    try:
        canonical = codecs.lookup(str(label or "").strip()).name
    except (LookupError, TypeError, ValueError):
        return ""
    return canonical if canonical in _SAFE_HTML_CODECS else ""


def _decode_homepage_html(raw: bytes, content_type: object = "") -> str:
    """Decode bounded homepage bytes without probabilistic encoding guesses."""

    value = bytes(raw)
    for marker, encoding in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF16_LE, "utf-16"),
    ):
        if value.startswith(marker):
            return value.decode(encoding, errors="replace")

    encoding = _safe_html_codec(_charset_parameter(content_type))
    if not encoding:
        encoding = _safe_html_codec(_early_meta_charset(value))
    try:
        return value.decode(encoding or "utf-8", errors="replace")
    except UnicodeError:
        return value.decode("utf-8", errors="replace")


def _content_type_header(headers: object) -> object:
    if not hasattr(headers, "items"):
        return ""
    for name, value in headers.items():
        if str(name).casefold() == "content-type":
            return value
    return ""


def _registrable_domain(url: str) -> str:
    """Extract the PSL-aware registrable domain from an HTTP(S) URL.

    ``https://www.ExampleCo.com/about`` -> ``exampleco.com``.
    """
    return normalize_url(url.strip()).domain.registrable_domain


def _normalize_for_match(s: str) -> str:
    """Lowercase + strip non-alphanumerics so 'Example, Co.' matches 'exampleco'."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _name_appears(haystack: str, company_name: str) -> bool:
    """Does ``company_name`` plausibly appear in ``haystack``?

    Tolerant of legal suffixes ('Inc', 'Ltd', 'LLC'), case, punctuation,
    and the company name being split across HTML tags.  We collapse both
    sides to alphanumerics-only before checking — same idea as company
    name fuzzy matching in pre_checks.
    """
    if not company_name or not haystack:
        return False

    # Strip only terminal legal suffixes from the claimed name. Tokens such
    # as "Group" and "AG" can be meaningful leading name terms.
    legal_suffixes = {
        "inc", "incorporated", "llc", "ltd", "limited", "corp",
        "corporation", "co", "company", "gmbh", "sa", "bv", "plc",
        "holdings", "group", "ag", "nv", "oy", "ab", "as", "pty",
        "pte", "kk", "srl", "spa",
    }
    words = re.findall(r"[a-z0-9]+", str(company_name or "").casefold())
    while words and words[-1] in legal_suffixes:
        words.pop()
    needle = "".join(words)
    if len(needle) < 3:
        # Avoid spurious matches on extremely short normalized names.
        # In practice, requiring 3+ characters costs us nothing; companies
        # that short are vanishingly rare and would need stronger signals
        # anyway.
        needle = _normalize_for_match(company_name)
    hay = _normalize_for_match(haystack)
    return bool(needle) and needle in hay


def _domain_matches_name(company_website: str, company_name: str) -> bool:
    """Legacy diagnostic heuristic; never sufficient for a fit decision."""
    try:
        domain = _registrable_domain(company_website)
    except Exception:
        return False
    if not domain:
        return False
    # Take only the part before the public suffix (e.g. 'exampleco' from
    # 'exampleco.com' or 'exampleco' from 'exampleco.co.uk').  This is a
    # diagnostic heuristic only. The verifier never uses it as proof.
    parts = domain.split(".")
    if len(parts) >= 2:
        base = parts[-2]
    else:
        base = domain
    return _name_appears(base, company_name)


def _canonical_linkedin_company_url(value: str) -> str:
    """Return a direct, canonical LinkedIn company URL or an empty string."""

    raw = str(value or "").strip()
    if raw.startswith("//"):
        raw = f"https:{raw}"
    try:
        parsed = urlsplit(raw)
    except (TypeError, ValueError):
        return ""
    host = str(parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._%+-]{0,99}", slug):
        return ""
    return f"https://www.linkedin.com/company/{slug}"


def _organization_identity_records(value) -> list[dict[str, object]]:
    """Extract bounded root Organization identity records from JSON-LD."""

    try:
        document = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    roots = document if isinstance(document, list) else [document]
    candidates: list[object] = []
    for root in roots[:20]:
        if not isinstance(root, dict):
            continue
        candidates.append(root)
        graph = root.get("@graph")
        if isinstance(graph, list):
            candidates.extend(graph[:20])

    records: list[dict[str, object]] = []
    for node in candidates:
        if not isinstance(node, dict):
            continue
        raw_type = node.get("@type", "")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if not any(str(item or "").casefold() == "organization" for item in types):
            continue
        raw_same_as = node.get("sameAs", [])
        same_as_candidates = (
            raw_same_as if isinstance(raw_same_as, list) else [raw_same_as]
        )
        linkedins = list(
            dict.fromkeys(
                canonical
                for candidate in same_as_candidates[:20]
                if (canonical := _canonical_linkedin_company_url(candidate))
            )
        )[:10]
        records.append(
            {
                "name": node.get("name"),
                "legal_name": node.get("legalName"),
                "url": node.get("url"),
                "linkedin_urls": linkedins,
            }
        )
        if len(records) == 10:
            break
    return records


def _organization_same_as_urls(value) -> list[str]:
    """Extract LinkedIn URLs from nested JSON-LD Organization ``sameAs`` data."""

    urls: list[str] = []

    try:
        document = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    def visit(node) -> None:
        if isinstance(node, dict):
            raw_type = node.get("@type", "")
            types = raw_type if isinstance(raw_type, list) else [raw_type]
            is_organization = any(
                str(item or "").casefold() == "organization" for item in types
            )
            if is_organization:
                raw_same_as = node.get("sameAs", [])
                candidates = (
                    raw_same_as if isinstance(raw_same_as, list) else [raw_same_as]
                )
                for candidate in candidates:
                    canonical = _canonical_linkedin_company_url(candidate)
                    if canonical:
                        urls.append(canonical)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(document)
    return list(dict.fromkeys(urls))[:10]


def _is_https_url(value: str) -> bool:
    """Whether a final response URL is an absolute HTTPS URL without creds."""

    try:
        parsed = urlsplit(str(value or "").strip())
    except (TypeError, ValueError):
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
    )


def _homepage_company_linkedin_urls(page_text: str) -> list[str]:
    """Return bounded LinkedIn identities from parsed hrefs or Organization JSON-LD."""

    parser = _HomepageIdentityParser()
    try:
        parser.feed(str(page_text or "")[:_MAX_BYTES])
        parser.close()
    except Exception:
        return []
    return list(dict.fromkeys(parser.linkedin_urls))[:10]


def _linked_identity_source_url(
    page_text: str,
    *,
    base_url: str,
    registrable_domain: str,
) -> str:
    """Return one linked first-party newsroom root, without guessing a path."""

    parser = _HomepageIdentityParser()
    try:
        parser.feed(str(page_text or "")[:_MAX_BYTES])
        parser.close()
    except Exception:
        return ""

    for href in parser.source_hrefs:
        try:
            absolute = urljoin(base_url, href)
            parsed = urlsplit(absolute)
            candidate_port = parsed.port
            candidate_domain = _registrable_domain(absolute)
        except (NormalizationError, TypeError, ValueError):
            continue
        if (
            _is_https_url(absolute)
            and candidate_port in (None, 443)
            and candidate_domain == registrable_domain
            and parsed.path.rstrip("/") == ""
            and str(parsed.hostname or "").casefold().split(".")[0]
            in _IDENTITY_SOURCE_HOST_LABELS
        ):
            return urlunsplit(
                ("https", parsed.netloc.casefold(), "/", parsed.query, "")
            )
    return ""


class _HomepageIdentityParser(HTMLParser):
    """Extract bounded first-party name metadata without trusting input text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._title_parts: list[str] = []
        self.metadata_names: list[str] = []
        self.copyright_legal_names: list[str] = []
        self.linkedin_urls: list[str] = []
        self.source_hrefs: list[str] = []
        self.organization_records: list[dict[str, object]] = []
        self._json_ld_parts: list[str] | None = None
        self._nonvisible_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag_name = tag.casefold()
        attributes = {
            str(key or "").casefold(): str(value or "").strip()
            for key, value in attrs
        }
        if tag_name in {"script", "style", "template"}:
            self._nonvisible_depth += 1
        if tag_name == "title":
            self._in_title = True
            return
        if tag_name in {"a", "link"}:
            href = attributes.get("href", "")
            if (
                tag_name == "a"
                and href
                and len(href) <= 2048
            ):
                self.source_hrefs.append(href)
            linkedin = _canonical_linkedin_company_url(href)
            if linkedin:
                self.linkedin_urls.append(linkedin)
        if tag_name == "script":
            script_type = attributes.get("type", "").split(";", 1)[0].casefold()
            if script_type == "application/ld+json":
                self._json_ld_parts = []
            return
        if tag_name != "meta":
            return
        key = (
            attributes.get("property")
            or attributes.get("name")
            or attributes.get("itemprop")
            or ""
        ).casefold()
        if key in {"og:site_name", "application-name"}:
            content = attributes.get("content", "")
            if content:
                self.metadata_names.append(content[:200])

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.casefold()
        if tag_name == "title":
            self._in_title = False
        elif tag_name == "script" and self._json_ld_parts is not None:
            json_ld = "".join(self._json_ld_parts)
            records = _organization_identity_records(json_ld)
            self.organization_records.extend(records)
            self.linkedin_urls.extend(_organization_same_as_urls(json_ld))
            self._json_ld_parts = None
        if tag_name in {"script", "style", "template"}:
            self._nonvisible_depth = max(0, self._nonvisible_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self._title_parts.append(data.strip())
        if self._json_ld_parts is not None:
            self._json_ld_parts.append(data)
        if self._nonvisible_depth == 0:
            self.copyright_legal_names.extend(
                match.group("name").strip()[:200]
                for match in _COPYRIGHT_LEGAL_NAME_RE.finditer(data[:500])
            )

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()[:300]


def _homepage_company_names(page_text: str) -> list[str]:
    """Return names actually observed in title or first-party metadata."""

    parser = _HomepageIdentityParser()
    try:
        parser.feed(str(page_text or "")[:_MAX_BYTES])
    except Exception:
        return []
    candidates = [*parser.metadata_names, *parser.copyright_legal_names]
    if parser.title:
        candidates.append(parser.title)
        candidates.extend(
            part.strip()
            # Product homepages commonly use ``Brand: tagline`` without a
            # space before the colon. Keep the conservative whitespace rule
            # for dash-like separators so hyphenated brand names are intact.
            for part in re.split(
                r"\s+(?:\||-|\N{EN DASH}|\N{EM DASH})\s+|\s*:\s*",
                parser.title,
            )
            if part.strip()
        )
    generic = {"home", "homepage", "welcome", "official site", "website"}
    return list(dict.fromkeys(
        candidate
        for candidate in candidates
        if candidate.casefold() not in generic and 2 < len(candidate) <= 200
    ))[:10]


def _verified_organization_legal_name_aliases(
    page_text: str,
    *,
    observed_domain: str,
    observed_name: str,
    observed_linkedin: str,
    company_quality: bool = False,
) -> list[str]:
    """Return legal names explicitly bound to the verified homepage entity."""

    parser = _HomepageIdentityParser()
    try:
        parser.feed(str(page_text or "")[:_MAX_BYTES])
        parser.close()
    except Exception:
        return []
    aliases: list[str] = []
    for record in parser.organization_records[:10]:
        brand = record.get("name")
        legal_name = record.get("legal_name")
        organization_url = record.get("url")
        linkedins = record.get("linkedin_urls")
        if not all(
            isinstance(value, str) and value.strip() and len(value.strip()) <= limit
            for value, limit in (
                (brand, _MAX_ORGANIZATION_NAME_LENGTH),
                (legal_name, _MAX_ORGANIZATION_NAME_LENGTH),
                (organization_url, 2048),
            )
        ) or not isinstance(linkedins, list):
            continue
        try:
            organization_domain = _registrable_domain(organization_url)
        except Exception:
            continue
        if organization_domain != observed_domain or observed_linkedin not in linkedins:
            continue
        brand_receipt = evaluate_company_identity(
            submitted_name=observed_name,
            submitted_website=f"https://{observed_domain}",
            submitted_linkedin=observed_linkedin,
            observed_name=brand,
            observed_website=organization_url,
            observed_linkedin=observed_linkedin,
            evidence_source="company_homepage",
            company_quality=company_quality,
        )
        if brand_receipt["decision"] != "match":
            continue
        alias = legal_name.strip()
        if alias not in aliases:
            aliases.append(alias)
        if len(aliases) == _MAX_ORGANIZATION_LEGAL_NAME_ALIASES:
            break
    return aliases


def _identity_result(
    receipt: Mapping[str, object],
    reason: str,
    *,
    actual_final_url: str = "",
    source_fetch_failed: bool = False,
    verified_homepage_transport_domain: str = "",
) -> CompanyFitDecisionResult:
    details = {
        "identity": dict(receipt),
        "actual_final_url": str(actual_final_url or ""),
        **(
            {
                "verified_homepage_transport_domain": (
                    verified_homepage_transport_domain
                )
            }
            if verified_homepage_transport_domain
            else {}
        ),
        **({"failure_reason_code": "source_blocked"} if source_fetch_failed else {}),
    }
    if receipt["decision"] == "match":
        return company_fit_match(reason, details=details)
    if receipt["decision"] == "mismatch":
        return company_fit_mismatch(reason, details=details)
    return company_fit_unavailable(reason, details=details)


async def _fetch_bounded_html(
    session: aiohttp.ClientSession,
    url: str,
) -> tuple[int, str, str]:
    async with session.get(url, allow_redirects=True) as resp:
        raw = bytearray()
        while len(raw) < _MAX_BYTES:
            chunk = await resp.content.read(_MAX_BYTES - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        return (
            resp.status,
            str(getattr(resp, "url", url) or url),
            _decode_homepage_html(
                raw,
                _content_type_header(getattr(resp, "headers", {})),
            ),
        )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

async def verify_company_exists(
    company_name: str,
    company_website: str,
    timeout_secs: float = _HTTP_TIMEOUT_SECS,
    *,
    company_linkedin: str = "",
    require_https_transport: bool = False,
    company_quality: bool = False,
) -> CompanyFitDecisionResult:
    """Verify that ``company_website`` is a real page for ``company_name``.

    Returns a ``company-fit-decision:v1`` result with named decision state.

      * ``"verified: name 'ExampleCo' found in homepage"`` (best case)
      * ``"homepage identity evidence unavailable: ..."``
      * ``"website is a parked / for-sale page"``
      * ``"website unreachable: ..."``
    """
    submitted_identity = evaluate_company_identity(
        submitted_name=company_name,
        submitted_website=company_website,
        submitted_linkedin=company_linkedin,
        observed_name="",
        observed_website="",
        observed_linkedin="",
        evidence_source="company_homepage",
        company_quality=company_quality,
    )
    if submitted_identity["decision"] == "mismatch":
        return _identity_result(
            submitted_identity,
            "submitted company identity is missing or invalid",
        )

    request_url = (
        _upgrade_plain_http_company_url(company_website)
        if require_https_transport
        else company_website.strip()
    )

    try:
        domain = _registrable_domain(request_url)
    except NormalizationError:
        submitted_identity.update(
            decision="mismatch", reason_code="identity_unresolved"
        )
        return _identity_result(
            submitted_identity,
            f"company_website has no valid registrable domain: {company_website!r}",
        )
    except Exception as exc:  # Pinned identity primitive is unavailable.
        return _identity_result(
            submitted_identity,
            f"company identity normalization unavailable: {type(exc).__name__}",
        )

    timeout = aiohttp.ClientTimeout(total=timeout_secs, connect=min(3.0, timeout_secs))

    try:
        for attempt in range(_TRANSIENT_FETCH_ATTEMPTS):
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    headers=_HEADERS,
                ) as session:
                    status, observed_url, text = await _fetch_bounded_html(
                        session,
                        request_url,
                    )
                break
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt + 1 >= _TRANSIENT_FETCH_ATTEMPTS:
                    raise
                await asyncio.sleep(_TRANSIENT_FETCH_RETRY_DELAY_SECS)
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
        return _identity_result(
            submitted_identity,
            f"website unreachable: {type(e).__name__}: {str(e)[:120]}",
            source_fetch_failed=True,
        )
    except aiohttp.ClientError as e:
        return _identity_result(
            submitted_identity,
            f"website unreachable: {type(e).__name__}: {str(e)[:120]}",
        )
    except Exception as e:  # noqa: BLE001 - log everything else and fall through
        return _identity_result(
            submitted_identity,
            f"website fetch error: {type(e).__name__}: {str(e)[:120]}",
        )

    # ----- Status checks ----------------------------------------------------
    # 200 = ideal.  3xx are already followed by aiohttp.  4xx/5xx mean
    # the page itself is not a usable company page.
    if status >= 400:
        return _identity_result(
            submitted_identity, f"website returned HTTP {status}",
            actual_final_url=observed_url,
            # These are failures of this bounded page fetch. Account and
            # quota refusals remain distinct from unavailable source evidence.
            source_fetch_failed=status in {408, 425, 500, 502, 503, 504},
        )

    if require_https_transport and not _is_https_url(observed_url):
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: final URL is not HTTPS",
            actual_final_url=observed_url,
        )

    # ----- Parked-domain detection ------------------------------------------
    if _PARKED_DOMAIN_RE.search(text):
        submitted_identity.update(decision="mismatch", reason_code="identity_mismatch")
        return _identity_result(
            submitted_identity,
            "website is a parked / for-sale page",
            actual_final_url=observed_url,
        )

    try:
        observed_domain = _registrable_domain(observed_url)
    except Exception as exc:
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: observed domain "
            f"normalization failed ({type(exc).__name__})",
            actual_final_url=observed_url,
        )
    if observed_domain != domain:
        redirect_receipt = dict(submitted_identity)
        redirect_receipt.update(
            decision="mismatch",
            reason_code="identity_mismatch",
            observed_domain=observed_domain,
        )
        return _identity_result(
            redirect_receipt,
            f"company identity conflict: redirect changed registrable domain "
            f"from {domain!r} to {observed_domain!r}",
            actual_final_url=observed_url,
        )

    observed_names = _homepage_company_names(text)
    observed_linkedins = _homepage_company_linkedin_urls(text)
    if observed_names and not observed_linkedins:
        source_url = _linked_identity_source_url(
            text,
            base_url=observed_url,
            registrable_domain=domain,
        )
        if source_url:
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    headers=_HEADERS,
                ) as session:
                    source_status, source_final_url, source_text = (
                        await _fetch_bounded_html(session, source_url)
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            else:
                try:
                    source_domain = _registrable_domain(source_final_url)
                except Exception:
                    source_domain = ""
                source_linkedins = _homepage_company_linkedin_urls(source_text)
                if (
                    source_status < 400
                    and _is_https_url(source_final_url)
                    and source_domain == domain
                    and len(source_linkedins) == 1
                    and any(
                        evaluate_company_identity(
                            submitted_name=homepage_name,
                            submitted_website=f"https://{domain}",
                            submitted_linkedin=source_linkedins[0],
                            observed_name=source_name,
                            observed_website=f"https://{domain}",
                            observed_linkedin=source_linkedins[0],
                            evidence_source="company_homepage",
                            company_quality=False,
                        )["decision"] == "match"
                        for homepage_name in observed_names
                        for source_name in _homepage_company_names(source_text)
                    )
                ):
                    observed_linkedins = source_linkedins
    if not observed_names:
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: company name metadata not found",
            actual_final_url=observed_url,
            verified_homepage_transport_domain=domain,
        )
    if not observed_linkedins:
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: LinkedIn company binding not found",
            actual_final_url=observed_url,
            verified_homepage_transport_domain=domain,
        )
    identity_receipts = [
        evaluate_company_identity(
            submitted_name=company_name,
            submitted_website=company_website,
            submitted_linkedin=company_linkedin,
            observed_name=observed_name,
            observed_website=observed_url,
            observed_linkedin=observed_linkedin,
            evidence_source="company_homepage",
            company_quality=company_quality,
        )
        for observed_name in observed_names
        for observed_linkedin in observed_linkedins
    ]
    matched = next(
        (receipt for receipt in identity_receipts if receipt["decision"] == "match"),
        None,
    )
    if matched is not None:
        matched_receipt: dict[str, object] = dict(matched)
        matched_linkedin = (
            "https://www.linkedin.com/company/"
            f"{matched['observed_linkedin_slug']}"
        )
        legal_name_aliases = _verified_organization_legal_name_aliases(
            text,
            observed_domain=matched["observed_domain"],
            observed_name=matched["observed_name"],
            observed_linkedin=matched_linkedin,
            company_quality=company_quality,
        )
        if legal_name_aliases:
            matched_receipt["verified_legal_name_aliases"] = legal_name_aliases
        return _identity_result(
            matched_receipt,
            "verified: independently observed first-party name, final domain, "
            "and exact LinkedIn company identity",
            actual_final_url=observed_url,
        )
    conflict = next(
        (receipt for receipt in identity_receipts if receipt["decision"] == "mismatch"),
        None,
    )
    if conflict is not None:
        # Page titles can be marketing copy, and outgoing company links can
        # name a parent, partner, or an old LinkedIn slug. Their disagreement
        # is not a proven entity conflict on the same final domain. Defer to
        # the existing independent web verifier; do not mark this as a match
        # or supply it as a verified identity anchor. Invalid domains, parked
        # pages, and cross-domain redirects remain terminal checks above.
        unresolved = dict(conflict)
        unresolved.update(decision="unavailable", reason_code="identity_not_proven")
        return _identity_result(
            unresolved,
            "homepage identity evidence unavailable: page metadata and links "
            "do not prove the submitted identity",
            actual_final_url=observed_url,
            verified_homepage_transport_domain=domain,
        )
    unavailable = next(
        (receipt for receipt in identity_receipts if receipt["decision"] == "unavailable"),
        None,
    )
    if unavailable is not None:
        return _identity_result(
            unavailable,
            "homepage identity evidence unavailable: complete identity not proven",
            actual_final_url=observed_url,
            verified_homepage_transport_domain=domain,
        )
    return _identity_result(
        submitted_identity,
        "homepage identity evidence unavailable: complete identity not proven",
        actual_final_url=observed_url,
        verified_homepage_transport_domain=domain,
    )
