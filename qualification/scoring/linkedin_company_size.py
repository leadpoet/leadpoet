"""Bounded current LinkedIn company-size evidence from Exa Contents."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

import aiohttp

from gateway.qualification.models import candidate_linkedin_prompt_slug
from qualification.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS


logger = logging.getLogger(__name__)

PROFILE_MAX_CHARACTERS = 4_000
PROFILE_TIMEOUT_SECONDS = 30.0


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
            if re.fullmatch(
                r"(?:#{1,6}\s*)?About(?: us)?",
                visible,
                re.IGNORECASE,
            ):
                about_start = index + 1
            continue
        if re.fullmatch(
            r"(?:#{1,6}\s*)?(?:Employees(?: at\b.*)?|Updates)",
            visible,
            re.IGNORECASE,
        ):
            about_end = index
            break
    if about_start is None:
        return None

    bands = "|".join(re.escape(value) for value in LINKEDIN_EMPLOYEE_BUCKETS)
    value_pattern = re.compile(
        rf"^(?P<band>{bands})(?:\s+employees?)?$",
        re.IGNORECASE,
    )
    field_pattern = re.compile(
        r"^(?:\*\*)?Company size(?:\*\*)?\s*:?(?:\s+(?P<value>.+))?$",
        re.IGNORECASE,
    )
    for index in range(about_start, about_end):
        field_match = field_pattern.fullmatch(lines[index].strip())
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
        value_match = value_pattern.fullmatch(raw_value)
        if value_match is None:
            return None
        quote = "\n".join(lines[index : quote_end + 1]).strip()
        if not quote:
            return None
        return {
            "employee_count": value_match.group("band"),
            "quote": quote[:500],
        }
    return None


async def fetch_current_linkedin_company_size(
    profile_url: str,
) -> Optional[dict[str, str]]:
    """Make one uncached Exa Contents request for an exact profile URL."""

    requested_slug = linkedin_company_page_slug(profile_url)
    key = str(os.environ.get("EXA_API_KEY") or "").strip()
    if not requested_slug or not key:
        return None
    payload = {
        "ids": [profile_url],
        "text": {"maxCharacters": PROFILE_MAX_CHARACTERS},
        "maxAgeHours": 0,
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
    results = body.get("results")
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
    extracted = extract_linkedin_company_size(result.get("text"))
    if extracted is None:
        return None
    return {**extracted, "url": returned_url}
