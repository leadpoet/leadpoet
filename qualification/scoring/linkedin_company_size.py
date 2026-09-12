"""Bounded current LinkedIn company-size evidence from Exa Contents."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, Mapping, Optional, TypedDict, Union
from urllib.parse import urlsplit

import aiohttp

from gateway.qualification.models import candidate_linkedin_prompt_slug
from qualification.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS


logger = logging.getLogger(__name__)

PROFILE_MAX_CHARACTERS = 4_000
PROFILE_TIMEOUT_SECONDS = 30.0
# Exa defaults live crawls to 10 seconds.  Keep this below the outer request
# timeout while giving the exact fresh-profile fetch more time to complete.
PROFILE_LIVECRAWL_TIMEOUT_MILLISECONDS = 20_000
CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE: Literal[
    "insufficient_evidence"
] = "insufficient_evidence"

_ABOUT_SECTION_HEADING = re.compile(
    r"(?:#{1,6}\s*)?(?:About(?: us)?|Over ons)",
    re.IGNORECASE,
)
_ABOUT_SECTION_END_HEADING = re.compile(
    r"(?:#{1,6}\s*)?(?:Employees(?: at\b.*)?|Medewerkers van(?:\b.*)?|Updates)",
    re.IGNORECASE,
)
_COMPANY_SIZE_FIELD = re.compile(
    r"(?:\*\*)?(?:Company size|Bedrijfsgrootte)(?:\*\*)?\s*:?(?:\s+(?P<value>.+))?",
    re.IGNORECASE,
)
_EMPLOYEE_SUFFIX = re.compile(r"\s+(?:employees?|medewerkers)\s*$", re.IGNORECASE)

# These are exact LinkedIn authentication-page titles, paired with credential
# fields. A public company page can contain sign-in links, so links or isolated
# authentication words are not enough to classify the fetch as blocked.
_ACCESS_WALL_TITLES = {
    "aanmelden | linkedin",
    "cadastre-se | linkedin",
    "entrar | linkedin",
    "inloggen | linkedin",
    "join linkedin",
    "linkedin login, sign in",
    "sign in | linkedin",
    "sign up | linkedin",
}
_ACCESS_WALL_CREDENTIAL_FIELDS = (
    ("email or phone", "password"),
    ("e-mail", "senha"),
    ("e-mail", "wachtwoord"),
)
_BLOCKED_PAGE_TITLES = {
    "access denied",
    "access denied | linkedin",
    "robot check",
    "robot check | linkedin",
    "security check | linkedin",
    "security verification | linkedin",
}
_BLOCKED_PAGE_MARKERS = (
    "access to this page has been denied",
    "additional verification required",
    "checking your browser",
    "request blocked",
    "security check",
    "verify you are human",
    "verifying you are human",
    "you don't have permission to access",
)


class CurrentLinkedInCompanySizeEvidence(TypedDict):
    employee_count: str
    quote: str
    url: str


class CurrentLinkedInCompanySizeInsufficientEvidence(TypedDict):
    outcome: Literal["insufficient_evidence"]
    url: str


CurrentLinkedInCompanySizeResult = Union[
    CurrentLinkedInCompanySizeEvidence,
    CurrentLinkedInCompanySizeInsufficientEvidence,
]


def linkedin_company_page_slug(value: Any) -> str:
    """Return one strict LinkedIn company slug from an HTTP(S) profile URL."""

    try:
        slug = candidate_linkedin_prompt_slug(
            value,
            "employee_size_evidence_url",
        )
        parsed = urlsplit(str(value))
    except (TypeError, ValueError):
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "company":
        return ""
    return slug


def is_linkedin_evidence_url(value: Any) -> bool:
    """Identify an absolute LinkedIn URL without accepting its claim."""

    if not isinstance(value, str) or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return False
    host = str(parsed.hostname or "").casefold()
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and parsed.username is None
        and parsed.password is None
        and (port is None or 1 <= port <= 65535)
        and (host == "linkedin.com" or host.endswith(".linkedin.com"))
    )


def _is_linkedin_access_wall(text: str) -> bool:
    """Return whether exact-profile text is an authentication or block wall."""

    visible_lines = [
        line.strip().lstrip("#").strip().casefold()
        for line in text[:PROFILE_MAX_CHARACTERS].splitlines()
        if line.strip()
    ]
    top_lines = visible_lines[:4]
    folded = "\n".join(visible_lines)
    blocked_body = "\n".join(
        line for line in visible_lines if line not in _BLOCKED_PAGE_TITLES
    )
    authentication_wall = (
        any(line in _ACCESS_WALL_TITLES for line in top_lines)
        and any(
            all(field in folded for field in fields)
            for fields in _ACCESS_WALL_CREDENTIAL_FIELDS
        )
    )
    blocked_page = (
        any(line in _BLOCKED_PAGE_TITLES for line in top_lines)
        and any(marker in blocked_body for marker in _BLOCKED_PAGE_MARKERS)
    )
    return authentication_wall or blocked_page


def _canonical_linkedin_company_size(value: str) -> Optional[str]:
    """Map an exact English or Dutch LinkedIn band to its canonical form."""

    raw = _EMPLOYEE_SUFFIX.sub("", value.strip().strip("*").strip())
    for band in LINKEDIN_EMPLOYEE_BUCKETS:
        localized = re.escape(band)
        localized = localized.replace(r"\-", r"\s*[-\u2013]\s*")
        localized = localized.replace(",", r"[,.]")
        if re.fullmatch(localized, raw, re.IGNORECASE):
            return band
    return None


def extract_linkedin_company_size(text: Any) -> Optional[dict[str, str]]:
    """Extract only LinkedIn's literal Company size field from About."""

    if not isinstance(text, str):
        return None
    lines = text[:PROFILE_MAX_CHARACTERS].splitlines()
    about_start: Optional[int] = None
    about_end = len(lines)
    for index, line in enumerate(lines):
        visible = line.strip().strip("*").strip()
        if about_start is None:
            if _ABOUT_SECTION_HEADING.fullmatch(visible):
                about_start = index + 1
            continue
        if _ABOUT_SECTION_END_HEADING.fullmatch(visible):
            about_end = index
            break
    if about_start is None:
        return None

    for index in range(about_start, about_end):
        field_match = _COMPANY_SIZE_FIELD.fullmatch(lines[index].strip())
        if field_match is None:
            continue
        raw_value = str(field_match.group("value") or "").strip().strip("*").strip()
        quote_end = index
        if not raw_value:
            for next_index in range(index + 1, min(about_end, index + 4)):
                candidate = lines[next_index].strip().strip("*").strip()
                if candidate:
                    raw_value = candidate
                    quote_end = next_index
                    break
        employee_count = _canonical_linkedin_company_size(raw_value)
        if employee_count is None:
            return None
        quote = "\n".join(lines[index : quote_end + 1]).strip()
        if not quote:
            return None
        return {
            "employee_count": employee_count,
            "quote": quote[:500],
        }
    return None


async def fetch_current_linkedin_company_size(
    profile_url: str,
) -> Optional[CurrentLinkedInCompanySizeResult]:
    """Return exact size proof, explicit no-size content, or retryable failure.

    A successful exact-profile About result with no canonical Company size, or
    the provider's exact-profile ``CRAWL_NOT_FOUND`` result, returns the explicit
    insufficient-evidence outcome. Authentication or block walls, transport
    failures, other status failures, and malformed results return ``None``.
    """

    requested_slug = linkedin_company_page_slug(profile_url)
    key = str(os.environ.get("EXA_API_KEY") or "").strip()
    if not requested_slug or not key:
        return None
    payload = {
        "ids": [profile_url],
        "text": {"maxCharacters": PROFILE_MAX_CHARACTERS},
        "maxAgeHours": 0,
        "livecrawlTimeout": PROFILE_LIVECRAWL_TIMEOUT_MILLISECONDS,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=PROFILE_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://api.exa.ai/contents",
                json=payload,
                headers={"x-api-key": key, "Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "current_linkedin_size_unavailable status=%s",
                        response.status,
                    )
                    return None
                body = await response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "current_linkedin_size_unavailable error=%s",
            type(exc).__name__,
        )
        return None
    if (
        not isinstance(body, Mapping)
        or body.get("error")
        or body.get("errors")
        or str(body.get("status") or "").casefold()
        in {"error", "failed", "failure"}
    ):
        return None
    statuses = body.get("statuses")
    results = body.get("results")
    if isinstance(statuses, list) and len(statuses) == 1:
        status_item = statuses[0]
        status_error = (
            status_item.get("error")
            if isinstance(status_item, Mapping)
            and isinstance(status_item.get("error"), Mapping)
            else {}
        )
        status_slug = (
            linkedin_company_page_slug(status_item.get("id"))
            if isinstance(status_item, Mapping)
            else ""
        )
        if (
            isinstance(status_item, Mapping)
            and str(status_item.get("status") or "").casefold() == "error"
            and status_error.get("httpStatusCode") == 404
            and status_error.get("tag") == "CRAWL_NOT_FOUND"
            and status_slug == requested_slug
            and results == []
        ):
            return {
                "outcome": CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE,
                "url": str(status_item["id"]),
            }
    if statuses is not None and (
        not isinstance(statuses, list)
        or not statuses
        or any(
            not isinstance(item, Mapping)
            or str(item.get("status") or "").casefold() != "success"
            for item in statuses
        )
    ):
        return None
    result = results[0] if isinstance(results, list) and len(results) == 1 else None
    if (
        not isinstance(result, Mapping)
        or result.get("error")
        or result.get("errors")
        or str(result.get("status") or "").casefold()
        in {"error", "failed", "failure"}
    ):
        return None
    returned_url = result.get("url")
    returned_slug = linkedin_company_page_slug(returned_url)
    if not isinstance(returned_url, str) or not returned_slug:
        return None
    if returned_slug != requested_slug:
        return None
    text = result.get("text")
    if not isinstance(text, str):
        return None
    if _is_linkedin_access_wall(text):
        return None
    extracted = extract_linkedin_company_size(text)
    if extracted is None:
        return {
            "outcome": CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE,
            "url": returned_url,
        }
    return {**extracted, "url": returned_url}
