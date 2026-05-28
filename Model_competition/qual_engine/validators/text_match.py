"""Deterministic name / size / domain matchers — lifted from fulfillment patterns."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

_TRADEMARK_RE = re.compile(r"[®™©℠]")  # Universal Unicode chars
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_company_name(name: str) -> str:
    """Lowercase + diacritic-fold + alnum-only. No legal-suffix stripping
    (would require a hardcoded list); name matching uses containment instead."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = _TRADEMARK_RE.sub("", s)
    s = s.lower()
    s = _NON_ALNUM.sub("", s)
    return s.strip()


def validate_name_match(claimed: str, source: str) -> Tuple[bool, str]:
    """Token-boundary name match (stricter than substring).

    A single-token query is matched if it appears as a full token in the
    multi-token source, NOT as a substring of a longer token. Single-token
    matches on common/short names remain ambiguous on their own; callers
    pair this check with a domain check to disambiguate.
    """
    a_norm = normalize_company_name(claimed)
    b_norm = normalize_company_name(source)
    if not a_norm or not b_norm:
        return False, f"empty name: claimed={claimed!r} source={source!r}"
    if a_norm == b_norm:
        return True, "exact"

    def _tokens(s: str) -> list[str]:
        return [normalize_company_name(t) for t in (s or "").split()
                if normalize_company_name(t)]
    a_toks = _tokens(claimed)
    b_toks = _tokens(source)
    if not a_toks or not b_toks:
        return False, f"no tokens: claimed={claimed!r} source={source!r}"

    shorter, longer = (a_toks, b_toks) if len(a_toks) <= len(b_toks) else (b_toks, a_toks)
    if all(t in longer for t in shorter):
        return True, "token-boundary"
    return False, f"token mismatch: {a_toks!r} vs {b_toks!r}"


# -----------------------------------------------------------------------------
# Employee count
# -----------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+")
_RANGE_SEP = re.compile(r"\s*(?:-|–|—|to)\s*", re.IGNORECASE)


def parse_size_range(s: str) -> Optional[Tuple[int, int]]:
    """Parse '201-500', '1,001-5,000 employees', '5000+', '50 to 200' → (lo, hi)."""
    if not s:
        return None
    cleaned = re.sub(r"employees?", "", s, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace(",", "")
    # 5000+ → (5000, 50000)
    if "+" in cleaned:
        m = _NUM_RE.search(cleaned)
        if m:
            lo = int(m.group(0))
            return (lo, max(lo * 5, lo + 1))
    # Find all numbers in the cleaned string
    nums = [int(m.group(0)) for m in _NUM_RE.finditer(cleaned)]
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (min(nums), max(nums))


def validate_size_match(claimed: str, source: str) -> Tuple[bool, str]:
    """Two size bands match if their numeric ranges overlap."""
    a = parse_size_range(claimed)
    b = parse_size_range(source)
    if not a or not b:
        return False, f"unparseable: claimed={claimed!r} source={source!r}"
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo <= hi:
        return True, f"overlap [{lo}, {hi}]"
    return False, f"no overlap: {a} vs {b}"


# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

def normalize_domain(url_or_host: str) -> str:
    if not url_or_host:
        return ""
    s = url_or_host.strip().lower()
    if s.startswith("http://") or s.startswith("https://"):
        s = (urlparse(s).hostname or "").lower()
    if s.startswith("www."):
        s = s[4:]
    # Registrable domain only (last 2 segments)
    parts = s.split(".")
    if len(parts) >= 2:
        # Handle .co.uk style
        tld = parts[-1]
        country_tlds = {"uk", "au", "nz", "br", "in", "jp", "mx", "ru", "za", "tr", "fr"}
        if tld in country_tlds and len(parts) >= 3 and len(parts[-2]) <= 3:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])
    return s


def domain_root_stem(domain: str) -> str:
    """Return the registrable-domain stem (everything before the final TLD).

    Used for fuzzy-matching two domain variants of the same company:
      stipplebio.com → stipplebio
      stipple.bio     → stipple
      stipple.co      → stipple
      terremoto.la    → terremoto
      terremoto.com   → terremoto

    Note: when company brand re-renders as `<brand>.<short-tld>` vs
    `<brand><tld-noise>.com`, the stem will still differ slightly (stipplebio
    vs stipple). Callers can compare with a substring rule on top.
    """
    if not domain:
        return ""
    norm = normalize_domain(domain)
    parts = norm.split(".")
    if not parts:
        return ""
    return parts[0]


def domains_likely_same_company(a: str, b: str) -> bool:
    """True when two domains plausibly refer to the same company.

    Match conditions:
      1. Normalized registrable domains are equal.
      2. One root-stem is a substring of the other (with both ≥3 chars).

    Handles cases like stipplebio.com vs stipple.bio, terremoto.com vs
    terremoto.la, biossil.com vs biossil.co.
    """
    if not a or not b:
        return False
    na, nb = normalize_domain(a), normalize_domain(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    sa, sb = domain_root_stem(a), domain_root_stem(b)
    if not sa or not sb or len(sa) < 3 or len(sb) < 3:
        return False
    if sa == sb:
        return True
    if sa in sb or sb in sa:
        return True
    return False


def extract_linkedin_slug(linkedin_url: str) -> Optional[str]:
    """Extract slug from URL like https://linkedin.com/company/ramp/about → 'ramp'."""
    if not linkedin_url:
        return None
    m = re.search(r"linkedin\.com/company/([^/?#]+)", linkedin_url.lower())
    if m:
        return m.group(1)
    return None


# -----------------------------------------------------------------------------
# Country matching
# -----------------------------------------------------------------------------
# Country extraction and matching use LLM (qual_engine.utils.ai_classifiers).
# We compare the LLM-extracted ISO-2 codes directly — both inputs normalized
# upstream, so equality is the only check here.


def country_match(icp_country: str, candidate_country: str) -> bool:
    """Both inputs are expected to be ISO-2 (already normalized via LLM).
    Empty on either side = compatible."""
    if not icp_country or not candidate_country:
        return True
    return icp_country.upper().strip()[:2] == candidate_country.upper().strip()[:2]


# Negation detection and generic-marketing detection have moved to
# qual_engine.utils.ai_classifiers (LLM-driven, no hardcoded patterns).
