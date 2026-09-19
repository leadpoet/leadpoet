"""Bounded current LinkedIn company-size evidence from Exa Contents."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Literal, Mapping, Optional, TypedDict, Union
from urllib.parse import urlsplit

import aiohttp

from gateway.qualification.models import candidate_linkedin_prompt_slug
from leadpoet_verifier.identity.normalization import (
    NormalizationError,
    normalize_host,
)
from qualification.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS


logger = logging.getLogger(__name__)

PROFILE_MAX_CHARACTERS = 4_000
PROFILE_TIMEOUT_SECONDS = 30.0
# Exa defaults live crawls to 10 seconds.  Keep this below the outer request
# timeout while giving the exact fresh-profile fetch more time to complete.
PROFILE_LIVECRAWL_TIMEOUT_MILLISECONDS = 20_000
STRUCTURED_PROFILE_TIMEOUT_SECONDS = 30.0
STRUCTURED_PROFILE_PROVIDER = "harvestapi_get_company"
STRUCTURED_PROFILE_SOURCE_FIELD = "employeeCountRange"
STRUCTURED_PROFILE_COMPANY_TYPE_SOURCE_FIELD = "companyType"
STRUCTURED_PROFILE_PUBLIC_COMPANY_TYPE = "Public Company"
CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE: Literal[
    "insufficient_evidence"
] = "insufficient_evidence"

_ABOUT_SECTION_HEADING = re.compile(
    r"(?:#{1,6}\s*)?(?:About(?: us)?|Over ons|Sobre nós)",
    re.IGNORECASE,
)
_ABOUT_SECTION_END_HEADING = re.compile(
    r"(?:#{1,6}\s*)?(?:Employees(?: at\b.*)?|Medewerkers van(?:\b.*)?|"
    r"Funcionários da(?:\b.*)?|Updates)",
    re.IGNORECASE,
)
_COMPANY_SIZE_FIELD = re.compile(
    r"(?:\*\*)?(?:Company size|Bedrijfsgrootte|Tamanho da empresa)"
    r"(?:\*\*)?\s*:?(?:\s+(?P<value>.+))?",
    re.IGNORECASE,
)
_EMPLOYEE_SUFFIX = re.compile(
    r"\s+(?:employees?|medewerkers|funcionários)\s*$",
    re.IGNORECASE,
)

# These are exact LinkedIn authentication-page titles, paired with credential
# fields. A public company page can contain sign-in links, so links or isolated
# authentication words are not enough to classify the fetch as blocked.
_ACCESS_WALL_TITLES = {
    "aanmelden | linkedin",
    "anmelden | linkedin",
    "cadastre-se | linkedin",
    "daftar | linkedin",
    "entrar | linkedin",
    "inloggen | linkedin",
    "inschrijven | linkedin",
    "iscriviti | linkedin",
    "join linkedin",
    "linkedin login, sign in",
    "registrarse | linkedin",
    "zarejestruj się | linkedin",
    "s’inscrire | linkedin",
    "sign in | linkedin",
    "sign up | linkedin",
    "đăng ký | linkedin",
    "εγγραφή | linkedin",
    "الاشتراك | linkedin",
    "ลงทะเบียน | linkedin",
}
_ACCESS_WALL_CREDENTIAL_FIELDS = (
    ("adres e-mail lub numer telefonu", "hasło"),
    ("e-mail-adresse/telefon", "passwort"),
    ("e-mail of telefoonnummer", "wachtwoord"),
    ("e-mel atau nombor telefon", "kata laluan"),
    ("email atau telepon", "kata sandi"),
    ("email or phone", "password"),
    ("email o telefono", "password"),
    ("email ή τηλέφωνο", "κωδικός πρόσβασης"),
    ("email", "contraseña"),
    ("email", "mật khẩu"),
    ("e-mail", "senha"),
    ("e-mail", "mot de passe"),
    ("e-mail", "wachtwoord"),
    ("البريد الإلكتروني أو رقم الهاتف", "كلمة المرور"),
    ("อีเมลหรือโทรศัพท์", "รหัสผ่าน"),
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


class StructuredLinkedInCompanySizeEvidence(TypedDict):
    employee_count: str
    provider: str
    source_field: str
    url: str
    website: str


class StructuredLinkedInPublicCompanyEvidence(TypedDict):
    company_type: str
    provider: str
    source_field: str
    url: str
    website: str

VERIFIER_FAILURE_REASON_KEY = "failure_reason"
SOURCE_BLOCKED_FAILURE_REASON = "source_blocked"
MALFORMED_RESPONSE_FAILURE_REASON = "malformed_response"
PROVIDER_ERROR_FAILURE_REASON = "provider_error"
UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON = "unexpected_verifier_error"


def _set_failure_reason(diagnostic: Optional[dict[str, str]], reason: str) -> None:
    """Set one fixed diagnostic code without retaining provider data."""

    if diagnostic is not None:
        diagnostic[VERIFIER_FAILURE_REASON_KEY] = reason


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


def _canonical_company_domain(value: Any) -> str:
    """Return one normalized exact website host for identity comparison."""

    if not isinstance(value, str) or not value.strip():
        return ""
    raw = value.strip()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return ""
    host = str(parsed.hostname or "").casefold().rstrip(".").removeprefix("www.")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80, 443}
        or "." not in host
    ):
        return ""
    return host


def _structured_company_domain_matches(
    requested_domain: Any,
    observed_website: Any,
) -> bool:
    """Match an exact host or a child of a requested registrable root."""

    requested = _canonical_company_domain(requested_domain)
    observed = _canonical_company_domain(observed_website)
    if not requested or not observed:
        return False
    if observed == requested:
        return True
    try:
        requested_parts = normalize_host(requested)
        observed_parts = normalize_host(observed)
    except NormalizationError:
        return False
    return (
        requested_parts.ascii_host == requested_parts.registrable_domain
        and observed_parts.registrable_domain
        == requested_parts.registrable_domain
        and observed_parts.ascii_host.endswith(
            "." + requested_parts.registrable_domain
        )
    )


def _strict_linkedin_company_profile_url(value: Any) -> str:
    """Return one canonical HTTPS LinkedIn company URL, or an empty string."""

    slug = linkedin_company_page_slug(value)
    if not slug or not isinstance(value, str) or value != value.strip():
        return ""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return ""
    host = str(parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme.casefold() != "https"
        or host not in {"linkedin.com", "www.linkedin.com"}
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return f"https://www.linkedin.com/company/{slug}"


def _structured_company_elements(value: Any) -> list[Mapping[str, Any]]:
    current = value
    for _ in range(8):
        if not isinstance(current, Mapping):
            return []
        element = current.get("element")
        if isinstance(element, Mapping):
            if type(current.get("status")) is not int or current["status"] != 200:
                return []
            return [element]
        elements = current.get("elements")
        if isinstance(elements, list):
            if type(current.get("status")) is not int or current["status"] != 200:
                return []
            return [item for item in elements[:10] if isinstance(item, Mapping)]
        for key in ("toolResponse", "rawV2", "raw", "result", "data", "output"):
            child = current.get(key)
            if isinstance(child, Mapping) and child is not current:
                current = child
                break
        else:
            return []
    return []


def _structured_provider_reported_error(value: Any, *, depth: int = 0) -> bool:
    if not isinstance(value, Mapping) or depth > 5:
        return False
    if value.get("error") not in (None, "", False, [], {}):
        return True
    if value.get("errors") not in (None, "", False, [], {}):
        return True
    status = value.get("status")
    if type(status) is int and status >= 400:
        return True
    if str(status or "").casefold() in {
        "error",
        "failed",
        "failure",
    }:
        return True
    return any(
        _structured_provider_reported_error(value.get(key), depth=depth + 1)
        for key in ("toolResponse", "rawV2", "raw", "result", "data", "output")
    )


def _canonical_employee_range(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    start = value.get("start")
    end = value.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or (end is not None and not isinstance(end, int))
    ):
        return ""
    return {
        (0, 1): "0-1",
        (2, 10): "2-10",
        (11, 50): "11-50",
        (51, 200): "51-200",
        (201, 500): "201-500",
        (501, 1_000): "501-1,000",
        (1_001, 5_000): "1,001-5,000",
        (5_001, 10_000): "5,001-10,000",
        (10_001, None): "10,001+",
    }.get((start, end), "")


def project_structured_linkedin_company_size(
    requested_domain: str,
    requested_profile_url: str,
    payload: Any,
) -> Optional[StructuredLinkedInCompanySizeEvidence]:
    """Project only an exact-identity canonical structured employee range."""

    domain = _canonical_company_domain(requested_domain)
    profile_url = _strict_linkedin_company_profile_url(requested_profile_url)
    requested_slug = linkedin_company_page_slug(profile_url)
    if not domain or not requested_slug or _structured_provider_reported_error(payload):
        return None
    for element in _structured_company_elements(payload):
        if not _structured_company_domain_matches(
            domain,
            element.get("website"),
        ):
            continue
        returned_profile = _strict_linkedin_company_profile_url(
            element.get("linkedinUrl") or element.get("linkedin_url")
        )
        if linkedin_company_page_slug(returned_profile) != requested_slug:
            continue
        employee_count = _canonical_employee_range(element.get("employeeCountRange"))
        if not employee_count:
            continue
        return {
            "employee_count": employee_count,
            "provider": STRUCTURED_PROFILE_PROVIDER,
            "source_field": STRUCTURED_PROFILE_SOURCE_FIELD,
            "url": profile_url,
            "website": f"https://{domain}/",
        }
    return None


def project_structured_linkedin_public_company(
    requested_domain: str,
    requested_profile_url: str,
    payload: Any,
) -> Optional[StructuredLinkedInPublicCompanyEvidence]:
    """Project only an exact-identity structured Public Company value."""

    domain = _canonical_company_domain(requested_domain)
    profile_url = _strict_linkedin_company_profile_url(requested_profile_url)
    requested_slug = linkedin_company_page_slug(profile_url)
    if not domain or not requested_slug or _structured_provider_reported_error(payload):
        return None
    for element in _structured_company_elements(payload):
        if not _structured_company_domain_matches(
            domain,
            element.get("website"),
        ):
            continue
        returned_profile = _strict_linkedin_company_profile_url(
            element.get("linkedinUrl") or element.get("linkedin_url")
        )
        if linkedin_company_page_slug(returned_profile) != requested_slug:
            continue
        if element.get("companyType") != STRUCTURED_PROFILE_PUBLIC_COMPANY_TYPE:
            continue
        return {
            "company_type": STRUCTURED_PROFILE_PUBLIC_COMPANY_TYPE,
            "provider": STRUCTURED_PROFILE_PROVIDER,
            "source_field": STRUCTURED_PROFILE_COMPANY_TYPE_SOURCE_FIELD,
            "url": profile_url,
            "website": f"https://{domain}/",
        }
    return None


async def fetch_structured_linkedin_company_size(
    requested_domain: str,
    profile_url: str,
    *,
    diagnostic: Optional[dict[str, str]] = None,
    public_company_evidence: Optional[dict[str, str]] = None,
) -> Optional[StructuredLinkedInCompanySizeEvidence]:
    """Fetch one profile and project size plus optional Public evidence."""

    key = str(os.environ.get("DEEPLINE_API_KEY") or "").strip()
    canonical_profile = _strict_linkedin_company_profile_url(profile_url)
    domain = _canonical_company_domain(requested_domain)
    if not key:
        return None
    if not canonical_profile or not domain:
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    try:
        timeout = aiohttp.ClientTimeout(total=STRUCTURED_PROFILE_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://code.deepline.com/api/v2/integrations/"
                "harvestapi_get_company/execute",
                json={"payload": {"url": canonical_profile}},
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
                    return None
                try:
                    body = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
                    return None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
        return None
    except Exception:  # noqa: BLE001
        _set_failure_reason(diagnostic, UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON)
        return None
    if _structured_provider_reported_error(body):
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
        return None
    if not _structured_company_elements(body):
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    public_evidence = project_structured_linkedin_public_company(
        domain,
        canonical_profile,
        body,
    )
    if public_company_evidence is not None and public_evidence is not None:
        public_company_evidence.update(public_evidence)
    evidence = project_structured_linkedin_company_size(
        domain,
        canonical_profile,
        body,
    )
    if evidence is None:
        # A valid one-company reply that lacks the requested exact identity or
        # canonical range is local to this profile. Arena may retry it, then
        # isolate only this company instead of cancelling the scoring item.
        _set_failure_reason(diagnostic, SOURCE_BLOCKED_FAILURE_REASON)
    return evidence


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
            all(field.casefold() in folded for field in fields)
            for fields in _ACCESS_WALL_CREDENTIAL_FIELDS
        )
    )
    blocked_page = (
        any(line in _BLOCKED_PAGE_TITLES for line in top_lines)
        and any(marker in blocked_body for marker in _BLOCKED_PAGE_MARKERS)
    )
    return authentication_wall or blocked_page


def _canonical_linkedin_company_size(value: str) -> Optional[str]:
    """Map an exact English, Dutch, or Portuguese band to canonical form."""

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
    *,
    diagnostic: Optional[dict[str, str]] = None,
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
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
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
                    _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
                    return None
                try:
                    body = await response.json()
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    logger.warning(
                        "current_linkedin_size_unavailable error=%s",
                        type(exc).__name__,
                    )
                    _set_failure_reason(
                        diagnostic, MALFORMED_RESPONSE_FAILURE_REASON
                    )
                    return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning(
            "current_linkedin_size_unavailable error=%s",
            type(exc).__name__,
        )
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "current_linkedin_size_unavailable error=%s",
            type(exc).__name__,
        )
        _set_failure_reason(diagnostic, UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON)
        return None
    if not isinstance(body, Mapping):
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    if (
        body.get("error")
        or body.get("errors")
        or str(body.get("status") or "").casefold()
        in {"error", "failed", "failure"}
    ):
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
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
        if (
            isinstance(status_item, Mapping)
            and str(status_item.get("status") or "").casefold() == "error"
            and status_error.get("httpStatusCode") == 409
            and status_error.get("tag") == "CRAWL_NON_CANONICAL"
            and status_slug == requested_slug
            and results == []
        ):
            # Exa classified this exact profile as noncanonical. This is
            # source-local, like an authentication wall, rather than evidence
            # that the Exa account is unavailable.
            _set_failure_reason(diagnostic, SOURCE_BLOCKED_FAILURE_REASON)
            return None
        if (
            isinstance(status_item, Mapping)
            and str(status_item.get("status") or "").casefold() == "error"
            and status_error.get("httpStatusCode") == 504
            and status_error.get("tag") == "CRAWL_LIVECRAWL_TIMEOUT"
            and status_slug == requested_slug
            and results == []
        ):
            # Exa could not live-crawl this exact profile within its bounded
            # source timeout. Other Exa or outer HTTP failures stay systemic.
            _set_failure_reason(diagnostic, SOURCE_BLOCKED_FAILURE_REASON)
            return None
    if statuses is not None and (
        not isinstance(statuses, list)
        or not statuses
        or any(
            not isinstance(item, Mapping)
            or str(item.get("status") or "").casefold() != "success"
            for item in statuses
        )
    ):
        if isinstance(statuses, list) and any(
            isinstance(item, Mapping)
            and str(item.get("status") or "").casefold() == "error"
            for item in statuses
        ):
            _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
        else:
            _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    result = results[0] if isinstance(results, list) and len(results) == 1 else None
    if (
        not isinstance(result, Mapping)
    ):
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    if (
        result.get("error")
        or result.get("errors")
        or str(result.get("status") or "").casefold()
        in {"error", "failed", "failure"}
    ):
        _set_failure_reason(diagnostic, PROVIDER_ERROR_FAILURE_REASON)
        return None
    returned_url = result.get("url")
    returned_slug = linkedin_company_page_slug(returned_url)
    if not isinstance(returned_url, str) or not returned_slug:
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    if returned_slug != requested_slug:
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    text = result.get("text")
    if not isinstance(text, str):
        _set_failure_reason(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)
        return None
    if _is_linkedin_access_wall(text):
        _set_failure_reason(diagnostic, SOURCE_BLOCKED_FAILURE_REASON)
        return None
    extracted = extract_linkedin_company_size(text)
    if extracted is None:
        return {
            "outcome": CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE,
            "url": returned_url,
        }
    return {**extracted, "url": returned_url}
