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
* Soft on legitimate edge cases (anti-bot pages, JS-rendered sites,
  302 redirects, IDN domains).  We err on the side of accepting a
  weakly-verified company rather than rejecting a real one, because
  intent verification (which IS strict) still has to pass on every
  signal.  Net result: a fake company can pass company verification
  but is extraordinarily unlikely to also produce verifiable intent
  signals at the same domain.

Public API:
    verify_company_exists(company_name, company_website,
                          timeout_secs=5) -> (passed, reason)
"""

from __future__ import annotations

import logging
import re
from typing import Tuple
from urllib.parse import urlparse

import aiohttp


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

_HTTP_TIMEOUT_SECS = 5
_MAX_BYTES = 200_000  # cap body read to 200KB — plenty for title + first paragraph

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

def _registrable_domain(url: str) -> str:
    """Extract the registrable hostname from a URL (lowercased, no www).

    ``https://www.ExampleCo.com/about`` -> ``exampleco.com``.
    """
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


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

    # Strip common legal suffixes from the claimed name before normalizing,
    # so 'ExampleCo Inc.' still matches a homepage that only says 'ExampleCo'.
    cleaned = re.sub(
        r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|bv|plc)\.?\b",
        "",
        company_name,
        flags=re.IGNORECASE,
    )
    needle = _normalize_for_match(cleaned)
    if len(needle) < 3:
        # Avoid spurious matches on extremely short normalized names.
        # In practice, requiring 3+ characters costs us nothing; companies
        # that short are vanishingly rare and would need stronger signals
        # anyway.
        needle = _normalize_for_match(company_name)
    hay = _normalize_for_match(haystack)
    return bool(needle) and needle in hay


def _domain_matches_name(company_website: str, company_name: str) -> bool:
    """Is the second-level domain a plausible derivation of the company name?

    Fallback path when the homepage HTML cannot be retrieved (anti-bot,
    JS-only render, transient network error).  If the URL host already
    encodes the company name, we treat that as weak-but-positive evidence
    of company existence and pass with a soft reason.
    """
    domain = _registrable_domain(company_website)
    if not domain:
        return False
    # Take only the part before the public suffix (e.g. 'exampleco' from
    # 'exampleco.com' or 'exampleco' from 'exampleco.co.uk').  This is a
    # heuristic — not a full PSL parse — but is sufficient since false
    # positives just turn into "weak verification, pass anyway".
    parts = domain.split(".")
    if len(parts) >= 2:
        base = parts[-2]
    else:
        base = domain
    return _name_appears(base, company_name)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

async def verify_company_exists(
    company_name: str,
    company_website: str,
    timeout_secs: float = _HTTP_TIMEOUT_SECS,
) -> Tuple[bool, str]:
    """Verify that ``company_website`` is a real page for ``company_name``.

    Returns ``(passed, reason)``.  Examples of reasons:

      * ``"verified: name 'ExampleCo' found in homepage"`` (best case)
      * ``"verified (soft): homepage unreachable but domain 'exampleco.com' matches name"``
      * ``"company name 'ExampleCo' not found in homepage and not in domain"``
      * ``"website is a parked / for-sale page"``
      * ``"website unreachable: ..."`` (when the domain itself is invalid)

    Soft-pass behavior: if the homepage GET fails for any transient
    reason BUT the registrable domain encodes the company name, we pass
    with reason annotated as ``(soft)``.  This is deliberate — see the
    module docstring.
    """
    if not company_website or not company_website.strip():
        return False, "company_website is empty"

    domain = _registrable_domain(company_website)
    if not domain:
        return False, f"company_website has no valid hostname: {company_website!r}"

    timeout = aiohttp.ClientTimeout(total=timeout_secs, connect=min(3.0, timeout_secs))

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=_HEADERS) as session:
            async with session.get(company_website, allow_redirects=True) as resp:
                status = resp.status
                # Read at most _MAX_BYTES so a giant single-page-app
                # download can't stall the scorer.
                raw = await resp.content.read(_MAX_BYTES)
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    text = ""
    except aiohttp.ClientError as e:
        # Network-level failure.  Try the domain-name fallback.
        if _domain_matches_name(company_website, company_name):
            return True, (
                f"verified (soft): homepage unreachable ({type(e).__name__}) "
                f"but domain {domain!r} matches name {company_name!r}"
            )
        return False, f"website unreachable: {type(e).__name__}: {str(e)[:120]}"
    except Exception as e:  # noqa: BLE001 - log everything else and fall through
        if _domain_matches_name(company_website, company_name):
            return True, (
                f"verified (soft): homepage fetch raised ({type(e).__name__}) "
                f"but domain {domain!r} matches name {company_name!r}"
            )
        return False, f"website fetch error: {type(e).__name__}: {str(e)[:120]}"

    # ----- Status checks ----------------------------------------------------
    # 200 = ideal.  3xx are already followed by aiohttp.  4xx/5xx mean
    # the page itself is not a usable company page.
    if status >= 400:
        if _domain_matches_name(company_website, company_name):
            return True, (
                f"verified (soft): homepage returned HTTP {status} but "
                f"domain {domain!r} matches name {company_name!r}"
            )
        return False, f"website returned HTTP {status}"

    # ----- Parked-domain detection ------------------------------------------
    if _PARKED_DOMAIN_RE.search(text):
        return False, "website is a parked / for-sale page"

    # ----- Name-presence check ---------------------------------------------
    if _name_appears(text, company_name):
        return True, f"verified: name {company_name!r} found in homepage"

    if _domain_matches_name(company_website, company_name):
        return True, (
            f"verified (soft): name not found in homepage text but "
            f"domain {domain!r} matches name {company_name!r}"
        )

    return False, (
        f"company name {company_name!r} not found in homepage "
        f"and not in domain {domain!r}"
    )
