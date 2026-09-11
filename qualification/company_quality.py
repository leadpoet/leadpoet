"""Pure claim normalization for the versioned Arena company requirements.

Parsing preserves invalid new fields so one bad claim occupies a zero-valued
company slot instead of discarding the other companies in a valid document.
These checks do not verify facts; the independent company-fit judge does that.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from leadpoet_verifier.identity.normalization import normalize_linkedin_company_url
from qualification.scoring.country_data import US_STATES

_US_NAMES = frozenset({"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america", "america"})
_STATES = {str(key).casefold(): str(value) for key, value in US_STATES.items()}
_STATES.update({"district of columbia": "District of Columbia", "dc": "District of Columbia", "d.c.": "District of Columbia", "washington dc": "District of Columbia", "washington, dc": "District of Columbia", "washington d.c.": "District of Columbia"})
_COMPANY_TABS = frozenset({"about", "jobs", "posts", "people", "life"})


def is_united_states(value: Any) -> bool:
    return isinstance(value, str) and " ".join(value.split()).casefold() in _US_NAMES


def canonical_us_state(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _STATES.get(" ".join(value.split()).casefold(), "")


def canonical_company_linkedin(value: Any) -> str:
    from qualification.competition_models import public_http_url

    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError("company_linkedin must identify a LinkedIn company page")
    public = public_http_url(value)
    parsed = urlsplit(public)
    if parsed.port not in (None, 443 if parsed.scheme == "https" else 80):
        raise ValueError("company_linkedin must use a standard HTTP port")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 3 and parts[2].casefold() in _COMPANY_TABS:
        public = urlunsplit((parsed.scheme, parsed.netloc, "/".join(["", *parts[:2]]), "", ""))
    return normalize_linkedin_company_url(public).canonical_url


def normalize_company_claim(company: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize valid claims and return deterministic company-local issues."""
    row = dict(company)
    errors: list[str] = []
    try:
        row["company_linkedin"] = canonical_company_linkedin(row.get("company_linkedin"))
    except (TypeError, ValueError):
        errors.append("company_linkedin_required_or_invalid")
    if is_united_states(row.get("country")):
        row["country"] = "United States"
        state = canonical_us_state(row.get("state"))
        if not state:
            errors.append("us_headquarters_state_required_or_invalid")
        else:
            row["state"] = state
    elif row.get("state") is None:
        row["state"] = ""
    elif not isinstance(row.get("state", ""), str):
        errors.append("state_must_be_text")
    return row, tuple(errors)
