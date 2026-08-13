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
                          company_linkedin=...) -> (passed, reason)
"""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
import json
import logging
import re
from urllib.parse import urlsplit, urlunsplit

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
_MAX_BYTES = 200_000  # cap body read to 200KB — plenty for title + first paragraph
_TRANSIENT_FETCH_ATTEMPTS = 2
_TRANSIENT_FETCH_RETRY_DELAY_SECS = 0.25

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
    r"\bcoming soon\b.*\bdomain\b",
    r"\bdefault web site page\b",
]
_PARKED_DOMAIN_RE = re.compile("|".join(_PARKED_DOMAIN_PATTERNS), re.IGNORECASE)
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


def _organization_same_as_urls(value) -> list[str]:
    """Extract LinkedIn URLs only from JSON-LD Organization ``sameAs`` data."""

    try:
        document = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    urls: list[str] = []

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
                    raw_same_as
                    if isinstance(raw_same_as, list)
                    else [raw_same_as]
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


class _HomepageIdentityParser(HTMLParser):
    """Extract bounded first-party name metadata without trusting input text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._title_parts: list[str] = []
        self.metadata_names: list[str] = []
        self.linkedin_urls: list[str] = []
        self._json_ld_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {
            str(key or "").casefold(): str(value or "").strip()
            for key, value in attrs
        }
        if tag.casefold() == "title":
            self._in_title = True
            return
        if tag.casefold() in {"a", "link"}:
            linkedin = _canonical_linkedin_company_url(attributes.get("href", ""))
            if linkedin:
                self.linkedin_urls.append(linkedin)
        if tag.casefold() == "script":
            script_type = attributes.get("type", "").split(";", 1)[0].casefold()
            if script_type == "application/ld+json":
                self._json_ld_parts = []
            return
        if tag.casefold() != "meta":
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
        if tag.casefold() == "title":
            self._in_title = False
        elif tag.casefold() == "script" and self._json_ld_parts is not None:
            self.linkedin_urls.extend(
                _organization_same_as_urls("".join(self._json_ld_parts))
            )
            self._json_ld_parts = None

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self._title_parts.append(data.strip())
        if self._json_ld_parts is not None:
            self._json_ld_parts.append(data)

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
    candidates = list(parser.metadata_names)
    if parser.title:
        candidates.append(parser.title)
        candidates.extend(
            part.strip()
            for part in re.split(r"\s+(?:\||-|\N{EN DASH}|\N{EM DASH}|:)\s+", parser.title)
            if part.strip()
        )
    generic = {"home", "homepage", "welcome", "official site", "website"}
    return list(dict.fromkeys(
        candidate
        for candidate in candidates
        if candidate.casefold() not in generic and 2 < len(candidate) <= 200
    ))[:10]


def _identity_result(
    receipt: dict[str, str],
    reason: str,
    *,
    actual_final_url: str = "",
) -> CompanyFitDecisionResult:
    details = {
        "identity": dict(receipt),
        "actual_final_url": str(actual_final_url or ""),
    }
    if receipt["decision"] == "match":
        return company_fit_match(reason, details=details)
    if receipt["decision"] == "mismatch":
        return company_fit_mismatch(reason, details=details)
    return company_fit_unavailable(reason, details=details)


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
) -> CompanyFitDecisionResult:
    """Verify that ``company_website`` is a real page for ``company_name``.

    Returns a ``company-fit-decision:v1`` result.  Historical callers may
    still unpack it as ``(passed, reason)``.

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
                    async with session.get(request_url, allow_redirects=True) as resp:
                        status = resp.status
                        observed_url = str(
                            getattr(resp, "url", request_url) or request_url
                        )
                        # Read at most _MAX_BYTES so a giant single-page-app
                        # download can't stall the scorer.
                        raw = await resp.content.read(_MAX_BYTES)
                        try:
                            text = raw.decode("utf-8", errors="replace")
                        except Exception:
                            text = ""
                break
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt + 1 >= _TRANSIENT_FETCH_ATTEMPTS:
                    raise
                await asyncio.sleep(_TRANSIENT_FETCH_RETRY_DELAY_SECS)
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
    if not observed_names:
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: company name metadata not found",
            actual_final_url=observed_url,
        )
    observed_linkedins = _homepage_company_linkedin_urls(text)
    if not observed_linkedins:
        return _identity_result(
            submitted_identity,
            "homepage identity evidence unavailable: LinkedIn company binding not found",
            actual_final_url=observed_url,
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
        )
        for observed_name in observed_names
        for observed_linkedin in observed_linkedins
    ]
    matched = next(
        (receipt for receipt in identity_receipts if receipt["decision"] == "match"),
        None,
    )
    if matched is not None:
        return _identity_result(
            matched,
            "verified: independently observed homepage name, final domain, "
            "and exact LinkedIn company identity",
            actual_final_url=observed_url,
        )
    conflict = next(
        (receipt for receipt in identity_receipts if receipt["decision"] == "mismatch"),
        None,
    )
    if conflict is not None:
        return _identity_result(
            conflict,
            f"company identity conflict: {conflict['reason_code']}",
            actual_final_url=observed_url,
        )
    return _identity_result(
        submitted_identity,
        "homepage identity evidence unavailable: complete identity not proven",
        actual_final_url=observed_url,
    )
