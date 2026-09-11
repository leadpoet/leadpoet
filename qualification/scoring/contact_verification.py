"""Bounded verification for the required Arena company contact.

The submitted contact is a claim.  Provider evidence is accepted only through
the separate ``source_evidence`` argument, which the trusted scorer populates
from broker records.  Raw evidence embedded in a miner's company/contact value
is deliberately ignored.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence
from urllib.parse import urlparse

import httpx
from leadpoet_verifier.identity.normalization import (
    NormalizationError,
    normalize_url,
)


DEEPLINE_EXECUTE_URL = "https://code.deepline.com/api/v2/integrations/{tool}/execute"
DEEPLINE_TIMEOUT_SECONDS = 30.0
PROVIDER_ATTEMPTS = 2
BOUNCEBAN_POLL_ATTEMPTS = 3
BOUNCEBAN_POLL_DELAYS_SECONDS = (1.0, 2.0, 4.0)

_SUPPORTED_SOURCE_PROVIDER = "harvestapi"
_SUPPORTED_SOURCE_TOOL = "harvestapi_get_profile"
_GENERIC_MAILBOXES = frozenset(
    {
        "admin",
        "billing",
        "careers",
        "contact",
        "customerservice",
        "hello",
        "help",
        "hr",
        "info",
        "jobs",
        "legal",
        "marketing",
        "office",
        "privacy",
        "recruiting",
        "sales",
        "security",
        "support",
        "team",
    }
)
_COUNTRY_ALIASES = {
    "us": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "great britain": "united kingdom",
    "uae": "united arab emirates",
}
_TITLE_EXPANSIONS = {
    "ceo": "chief executive officer",
    "coo": "chief operating officer",
    "cto": "chief technology officer",
    "cio": "chief information officer",
    "cfo": "chief financial officer",
    "cmo": "chief marketing officer",
    "cro": "chief revenue officer",
    "ciso": "chief information security officer",
    "chro": "chief human resources officer",
    "svp": "senior vice president",
    "evp": "executive vice president",
    "vp": "vice president",
    "dir": "director",
    "revops": "revenue operations",
    "salesops": "sales operations",
    "gtm": "go to market",
    "bd": "business development",
}
_KNOWN_INVALID_EMAIL_STATUSES = frozenset(
    {"invalid", "abuse", "do_not_mail", "disposable", "spamtrap", "toxic"}
)
_CATCH_ALL_EMAIL_STATUSES = frozenset(
    {"catch_all", "catch-all", "accept_all", "valid_accept_all", "ok_for_all"}
)

Execute = Callable[[str, Mapping[str, Any]], Awaitable[Any]]
RoleClassifier = Callable[..., Any]


class _ProviderUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    raw = unicodedata.normalize("NFKD", _text(value))
    letters = "".join(char for char in raw if not unicodedata.combining(char))
    return re.sub(r"[\W_]+", " ", letters.casefold()).strip()


def _norm_country(value: Any) -> str:
    try:
        from qualification.contact_models import normalize_country_code

        return normalize_country_code(value)
    except (TypeError, ValueError):
        normalized = _norm(value)
        return _COUNTRY_ALIASES.get(normalized, normalized)


def _canonical_linkedin(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if host not in {"linkedin.com", "m.linkedin.com"} or not path.casefold().startswith("/in/"):
        return ""
    return f"linkedin.com{path.casefold()}"


def _hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity_key(contact: Mapping[str, Any]) -> str:
    linkedin = _canonical_linkedin(contact.get("linkedin_url"))
    basis = {
        "linkedin": linkedin,
        "name": _norm(contact.get("full_name")),
        "email": _text(contact.get("email")).casefold(),
    }
    return f"contact:{_hash(basis)}"


def contact_identity_key(contact: Any) -> str:
    """Return the stable contact key used by evaluated and skipped receipts."""
    return _identity_key(contact if isinstance(contact, Mapping) else {})


def _result(
    contact: Mapping[str, Any],
    decision: str,
    reason: str,
    *,
    email_status: str = "unknown",
    subchecks: Optional[Mapping[str, Any]] = None,
    evidence_hashes: Optional[Mapping[str, str]] = None,
    evidence_timestamps: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    verification = {
        "decision": decision,
        "reason": reason,
        "subchecks": dict(subchecks or {}),
        "evidence_hashes": dict(evidence_hashes or {}),
        "evidence_timestamps": dict(evidence_timestamps or {}),
        "verified_at": _utc_now(),
    }
    receipt: dict[str, Any] = {
        "gate": "contact",
        "decision": decision,
        "reason": reason,
    }
    if decision == "unavailable":
        receipt["failure_class"] = "contact_provider_error"
    return {
        "contact_qualified": decision == "verified",
        "contact_identity_key": _identity_key(contact),
        "email_status": email_status,
        "contact_verification": verification,
        "verifier_gate_receipts": [receipt],
    }


async def _default_execute(tool: str, payload: Mapping[str, Any]) -> Any:
    provider, separator, _ = tool.partition("_")
    if not separator or not provider:
        raise _ProviderUnavailable("contact_provider_malformed_tool")
    # The Arena sandbox matches ``tool`` from the fixed URL and accepts only
    # the caller payload. Its broker adds provider/operation to the real
    # outbound Deepline request after policy validation.
    body = {"payload": dict(payload)}
    headers = {"Authorization": "Bearer arena-trusted-scorer"}
    try:
        async with httpx.AsyncClient(timeout=DEEPLINE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                DEEPLINE_EXECUTE_URL.format(tool=tool), json=body, headers=headers
            )
    except httpx.TimeoutException as exc:
        raise _ProviderUnavailable("contact_provider_timeout") from exc
    except httpx.HTTPError as exc:
        raise _ProviderUnavailable("contact_provider_error") from exc
    if response.status_code == 429:
        raise _ProviderUnavailable("contact_provider_rate_limited")
    if response.status_code >= 500:
        raise _ProviderUnavailable("contact_provider_error")
    if response.status_code >= 400:
        raise _ProviderUnavailable("contact_provider_error")
    try:
        return response.json()
    except ValueError as exc:
        raise _ProviderUnavailable("contact_provider_malformed_response") from exc


def _response_error(value: Any) -> Optional[str]:
    records = [value]
    unwrapped = _unwrap_data(value)
    if unwrapped is not value:
        records.append(unwrapped)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        status_code = record.get("status_code", record.get("statusCode"))
        if status_code is None and isinstance(record.get("status"), int):
            status_code = record.get("status")
        try:
            code = int(status_code) if status_code is not None else 0
        except (TypeError, ValueError):
            code = 0
        if code == 429:
            return "contact_provider_rate_limited"
        if code >= 500:
            return "contact_provider_error"
        status = _norm(record.get("status"))
        error = _norm(record.get("error"))
        not_found = code == 404 and (
            "not found" in error or "no profile" in error
        )
        if status in {"failed", "error"} or (error and not not_found):
            return "contact_provider_error"
    return None


async def _call(execute: Execute, tool: str, payload: Mapping[str, Any]) -> Any:
    last: Optional[_ProviderUnavailable] = None
    for attempt in range(PROVIDER_ATTEMPTS):
        try:
            value = execute(tool, payload)
            if inspect.isawaitable(value):
                value = await value
            error = _response_error(value)
            if error:
                raise _ProviderUnavailable(error)
            return value
        except (asyncio.TimeoutError, TimeoutError) as exc:
            last = _ProviderUnavailable("contact_provider_timeout")
            last.__cause__ = exc
        except _ProviderUnavailable as exc:
            last = exc
        except (httpx.TimeoutException,) as exc:
            last = _ProviderUnavailable("contact_provider_timeout")
            last.__cause__ = exc
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            last = _ProviderUnavailable("contact_provider_error")
            last.__cause__ = exc
        if attempt + 1 < PROVIDER_ATTEMPTS:
            await asyncio.sleep(0)
    raise last or _ProviderUnavailable("contact_provider_error")


def _unwrap_data(value: Any) -> Any:
    """Unwrap only documented/common provider envelopes, with a hard depth cap."""
    current = value
    for _ in range(8):
        if not isinstance(current, Mapping):
            return current
        status = _norm(current.get("status"))
        if status in {"pending", "queued", "processing", "running"}:
            return current
        moved = False
        for key in (
            "toolResponse",
            "tool_response",
            "rawV2",
            "raw_v2",
            "raw",
            "result",
            "data",
            "output",
        ):
            child = current.get(key)
            if (
                isinstance(child, (Mapping, list, tuple))
                and child is not current
            ):
                current = child
                moved = True
                break
        if not moved:
            return current
    return current


def _profile_candidates(value: Any, depth: int = 0) -> list[Mapping[str, Any]]:
    if depth > 5:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        found: list[Mapping[str, Any]] = []
        for item in list(value)[:25]:
            found.extend(_profile_candidates(item, depth + 1))
        return found
    if not isinstance(value, Mapping):
        return []
    profile_keys = {
        "publicIdentifier",
        "linkedinUrl",
        "linkedin_url",
        "firstName",
        "lastName",
        "currentPosition",
        "currentPositions",
        "experience",
    }
    found = [value] if profile_keys.intersection(value) else []
    for key in ("elements", "items", "profiles", "profile", "element", "results"):
        if key in value:
            found.extend(_profile_candidates(value[key], depth + 1))
    return found


def _is_completed_not_found(response: Any) -> bool:
    """Recognize a successful HarvestAPI response containing zero profiles."""
    data = _unwrap_data(response)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return len(data) == 0
    if isinstance(data, Mapping):
        status = _norm(data.get("status")).replace(" ", "_")
        if status in {"404", "not_found", "no_profile", "profile_not_found"}:
            return True
        error = _norm(data.get("error"))
        if "not found" in error or "no profile" in error:
            return True
        if data.get("element", object()) is None and not error:
            return True
        for key in ("elements", "items", "profiles", "results"):
            value = data.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return len(value) == 0
    return False


def _profile_linkedin(profile: Mapping[str, Any]) -> str:
    direct = _canonical_linkedin(
        profile.get("linkedinUrl")
        or profile.get("linkedin_url")
        or profile.get("profileUrl")
        or profile.get("url")
    )
    if direct:
        return direct
    identifier = _text(profile.get("publicIdentifier") or profile.get("public_identifier"))
    return _canonical_linkedin(f"linkedin.com/in/{identifier}") if identifier else ""


def _profile_name(profile: Mapping[str, Any]) -> str:
    first = _text(profile.get("firstName") or profile.get("first_name"))
    last = _text(profile.get("lastName") or profile.get("last_name"))
    return f"{first} {last}".strip() or _text(profile.get("fullName") or profile.get("full_name"))


def _is_current_experience(item: Mapping[str, Any]) -> bool:
    if item.get("current") is False or item.get("isCurrent") is False:
        return False
    if item.get("current") is True or item.get("isCurrent") is True:
        return True
    end = item.get("endDate", item.get("end_date"))
    if end is None or end == "":
        return True
    if isinstance(end, Mapping):
        if not any(end.values()):
            return True
        if _norm(end.get("text")) in {"present", "current", "now"}:
            return True
    return _norm(end) in {"present", "current", "now"}


def _current_positions(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    positions: list[Mapping[str, Any]] = []
    for key in ("currentPosition", "currentPositions", "current_position"):
        current = profile.get(key)
        if isinstance(current, Mapping) and _is_current_experience(current):
            positions.append(current)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            positions.extend(
                item
                for item in current[:10]
                if isinstance(item, Mapping) and _is_current_experience(item)
            )
    experience = profile.get("experience") or profile.get("experiences") or []
    if isinstance(experience, Sequence) and not isinstance(experience, (str, bytes, bytearray)):
        positions.extend(
            item for item in experience[:25] if isinstance(item, Mapping) and _is_current_experience(item)
        )
    unique: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for position in positions:
        digest = _hash(position)
        if digest not in seen:
            seen.add(digest)
            unique.append(position)
    return unique


def _position_title(position: Mapping[str, Any]) -> str:
    return _text(
        position.get("title")
        or position.get("position")
        or position.get("role")
        or position.get("jobTitle")
    )


def _company_identifiers(company: Any) -> dict[str, str]:
    linkedin = _text(
        _get(company, "linkedin_url")
        or _get(company, "linkedinUrl")
        or _get(company, "linkedin_company_url")
        or _get(company, "company_linkedin")
    )
    linkedin_id = _text(_get(company, "linkedin_id") or _get(company, "linkedinId"))
    if not linkedin_id and linkedin:
        linkedin_id = urlparse(linkedin if "://" in linkedin else f"https://{linkedin}").path.rstrip("/").split("/")[-1]
    domain = _registrable_domain(
        _get(company, "domain")
        or _get(company, "website")
        or _get(company, "company_website")
    )
    return {
        "name": _norm_company_name(
            _get(company, "company_name") or _get(company, "name")
        ),
        "domain": domain,
        "linkedin_slug": _norm(linkedin_id),
    }


def _position_company(position: Mapping[str, Any]) -> dict[str, str]:
    company = position.get("company") if isinstance(position.get("company"), Mapping) else {}
    name = (
        position.get("companyName")
        or position.get("company_name")
        or position.get("employerName")
        or company.get("name")
    )
    domain = position.get("companyDomain") or position.get("company_domain") or company.get("domain") or company.get("website")
    linkedin = position.get("companyLinkedinUrl") or position.get("companyLinkedInUrl") or company.get("linkedinUrl") or company.get("linkedin_url")
    # LinkedIn's profile companyId can be numeric while the canonical company
    # URL uses a slug. Compare the URL slug and never equate those two forms.
    linkedin_slug = ""
    if linkedin:
        linkedin_slug = urlparse(
            _text(linkedin) if "://" in _text(linkedin) else f"https://{linkedin}"
        ).path.rstrip("/").split("/")[-1]
    domain_text = _registrable_domain(domain)
    return {
        "name": _norm_company_name(name),
        "domain": domain_text,
        "linkedin_slug": _norm(linkedin_slug),
    }


def _registrable_domain(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    try:
        return normalize_url(
            raw, allow_bare_domain=True
        ).domain.registrable_domain.casefold()
    except (NormalizationError, TypeError, ValueError):
        return ""


def _norm_company_name(value: Any) -> str:
    words = _norm(value).split()
    legal_suffixes = {
        "co",
        "company",
        "corp",
        "corporation",
        "gmbh",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "ltd",
        "plc",
    }
    while words and words[-1] in legal_suffixes:
        words.pop()
    return " ".join(words)


def _company_matches(expected: Mapping[str, str], observed: Mapping[str, str]) -> bool:
    strong = [
        expected[key] == observed[key]
        for key in ("domain", "linkedin_slug")
        if expected.get(key) and observed.get(key)
    ]
    if strong:
        # One explicit contradiction defeats a weaker name match (and even a
        # second strong match); provider evidence must describe one company.
        return all(strong)
    return bool(
        expected.get("name")
        and observed.get("name")
        and expected["name"] == observed["name"]
    )


def _normalize_title(value: Any) -> str:
    words = _norm(value).split()
    expanded: list[str] = []
    for word in words:
        expanded.extend(_TITLE_EXPANSIONS.get(word, word).split())
    return " ".join(word for word in expanded if word not in {"of", "the", "and"})


def _seniority(value: Any) -> str:
    title = _normalize_title(value)
    if re.search(r"\bchief\b.*\bofficer\b", title):
        return "c_level"
    if (
        "managing partner" in title
        or "managing director" in title
        or re.search(r"\b(owner|founder)\b", title)
        or (re.search(r"\bpresident\b", title) and "vice president" not in title)
    ):
        return "c_level"
    if "vice president" in title:
        return "vp"
    if "head " in f"{title} ":
        return "head"
    if "director" in title:
        return "director"
    if "manager" in title:
        return "manager"
    return "other"


def _deterministic_role_match(actual: str, targets: Sequence[str], target_seniority: str) -> Optional[bool]:
    actual_norm = _normalize_title(actual)
    if not actual_norm:
        return False
    if any(
        re.search(rf"\b{marker}\b", actual_norm)
        for marker in ("assistant", "former", "formerly", "retired", "previous", "previously", "ex")
    ):
        return False
    if not _target_seniority_matches(actual_norm, target_seniority):
        return False
    for target in targets:
        target_norm = _normalize_title(target)
        if actual_norm == target_norm:
            return True
    if not targets and _norm(target_seniority):
        return True
    return None


def _target_seniority_matches(actual: str, requested: str) -> bool:
    requested_raw = _text(requested).casefold()
    target = _norm(requested)
    if not target:
        return True
    actual_level = _seniority(actual)
    canonical = {
        "c level": "c_level",
        "c suite": "c_level",
        "executive": "c_level",
        "vp": "vp",
        "vice president": "vp",
        "head": "head",
        "head of": "head",
        "director": "director",
        "manager": "manager",
    }
    if (
        ("+" in requested_raw and target in {"vp", "vice president"})
        or target in {"vp above", "vp and above", "vice president above", "vice president and above"}
    ):
        return actual_level in {"vp", "c_level"}
    if (
        ("+" in requested_raw and target == "director")
        or target in {"director above", "director and above"}
    ):
        return actual_level in {"director", "head", "vp", "c_level"}
    expected = canonical.get(target)
    return actual_level == expected if expected else False


async def _role_matches(
    actual: str,
    targets: Sequence[str],
    target_seniority: str,
    classify_role: Optional[RoleClassifier],
) -> bool:
    deterministic = _deterministic_role_match(actual, targets, target_seniority)
    if deterministic is not None:
        return deterministic
    if classify_role is None:
        return False
    try:
        result = classify_role(actual, list(targets), target_seniority)
        if inspect.isawaitable(result):
            result = await result
    except _ProviderUnavailable:
        raise
    except Exception as exc:
        raise _ProviderUnavailable("contact_role_provider_error") from exc
    if isinstance(result, Mapping):
        return result.get("match") is True or result.get("qualified") is True
    return result is True


def _profile_location(profile: Mapping[str, Any]) -> dict[str, str]:
    location = profile.get("location") if isinstance(profile.get("location"), Mapping) else {}
    parsed = location.get("parsed") if isinstance(location.get("parsed"), Mapping) else {}
    return {
        "country": _norm_country(
            profile.get("country")
            or profile.get("countryName")
            or location.get("country")
            or location.get("countryName")
            or location.get("countryCode")
            or parsed.get("country")
            or parsed.get("countryFull")
            or parsed.get("countryCode")
        ),
        "region": _norm(
            profile.get("region")
            or profile.get("state")
            or location.get("region")
            or location.get("state")
            or parsed.get("state")
            or parsed.get("regionCode")
        ),
        "city": _norm(profile.get("city") or location.get("city") or parsed.get("city")),
    }


def _location_check(claim: Mapping[str, Any], profile: Mapping[str, Any], icp: Any) -> tuple[str, str]:
    claimed = claim.get("location") if isinstance(claim.get("location"), Mapping) else {}
    observed = _profile_location(profile)
    country = _norm_country(claimed.get("country"))
    if not observed["country"]:
        return "unknown", "contact_location_unverified"
    if observed["country"] != country:
        return "fail", "contact_location_mismatch"
    for part in ("region", "city"):
        expected = _norm(claimed.get(part))
        if expected:
            if not observed[part]:
                return "unknown", "contact_location_unverified"
            if expected != observed[part]:
                return "fail", "contact_location_mismatch"

    geography = _get(icp, "contact_geography") or {}
    if not isinstance(geography, Mapping):
        geography = {}
    constraints = {
        "country": geography.get("countries") or [],
        "region": geography.get("regions") or [],
        "city": geography.get("cities") or [],
    }
    for part, allowed in constraints.items():
        if not allowed:
            continue
        normalize = _norm_country if part == "country" else _norm
        allowed_values = {normalize(item) for item in allowed if _text(item)}
        value = observed[part]
        if value and value not in allowed_values:
            return "fail", "contact_geography_mismatch"
        if not value:
            return "unknown", "contact_geography_unverified"
    return "pass", "contact_location_verified"


def _extract_emails(profile: Mapping[str, Any]) -> set[str]:
    emails: set[str] = set()
    direct_keys = ("email", "workEmail", "work_email", "professionalEmail", "professional_email")
    for key in direct_keys:
        value = profile.get(key)
        if isinstance(value, str) and "@" in value:
            emails.add(value.strip().casefold())
    collection = profile.get("emails") or profile.get("emailAddresses") or []
    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
        for item in collection[:20]:
            value = item.get("email") or item.get("value") if isinstance(item, Mapping) else item
            if isinstance(value, str) and "@" in value:
                emails.add(value.strip().casefold())
    return emails


def contact_source_semantics(source: Any) -> Any:
    """Project only source facts that can change a contact verdict.

    Deepline job IDs, billing fields, profile photos, follower counts, and
    timestamps do not affect verification and therefore do not split cache
    entries. Candidate order is retained because the verifier selects the
    first profile matching the claimed LinkedIn URL.
    """
    if not isinstance(source, Mapping):
        return None
    if source.get("invalid") is True:
        return {
            "invalid": True,
            "reason": _text(source.get("reason")),
        }
    response = source.get("response")
    profiles = _profile_candidates(_unwrap_data(response))
    projected_profiles = []
    for profile in profiles[:25]:
        positions = [
            {
                "title": _normalize_title(_position_title(position)),
                "company": _position_company(position),
            }
            for position in _current_positions(profile)
        ]
        projected_profiles.append(
            {
                "id": _text(
                    profile.get("recordId")
                    or profile.get("record_id")
                    or profile.get("id")
                ),
                "linkedin": _profile_linkedin(profile),
                "name": _norm(_profile_name(profile)),
                "positions": positions,
                "location": _profile_location(profile),
                "emails": sorted(_extract_emails(profile)),
            }
        )
    if projected_profiles:
        response_semantics: Any = {"profiles": projected_profiles}
    elif response is None:
        response_semantics = {"state": "refetch_required"}
    elif _is_completed_not_found(response):
        response_semantics = {"state": "not_found"}
    else:
        response_semantics = {"state": "malformed"}
    source_input = source.get("input") if isinstance(source.get("input"), Mapping) else {}
    return {
        "provider": source.get("provider"),
        "tool": source.get("tool"),
        "input": {
            key: source_input.get(key)
            for key in ("url", "publicIdentifier", "profileId", "findEmail")
            if source_input.get(key) is not None
        },
        "response": response_semantics,
    }


def _normalized_email_status(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _truthy_flag(value: Any) -> bool:
    return value is True or _norm(value) in {"1", "true", "yes"}


def _email_evidence_unsafe(*records: Any) -> bool:
    flag_names = (
        "is_disposable",
        "isDisposable",
        "disposable",
        "is_abuse",
        "isAbuse",
        "abuse",
        "do_not_mail",
        "doNotMail",
        "is_do_not_mail",
        "isDoNotMail",
        "is_spamtrap",
        "isSpamtrap",
        "spamtrap",
        "is_toxic",
        "isToxic",
        "toxic",
    )
    status_names = ("status", "sub_status", "result", "verdict", "classification")
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if any(_truthy_flag(record.get(name)) for name in flag_names):
            return True
        if any(
            _normalized_email_status(record.get(name))
            in _KNOWN_INVALID_EMAIL_STATUSES
            for name in status_names
        ):
            return True
    return False


def _email_evidence_is_catch_all(*records: Any) -> bool:
    flag_names = (
        "is_accept_all",
        "isAcceptAll",
        "accept_all",
        "acceptAll",
        "catchall_domain",
        "catchAllDomain",
        "catch_all_domain",
    )
    status_names = ("status", "sub_status", "result", "verdict", "classification")
    catch_all = set(_CATCH_ALL_EMAIL_STATUSES) | {"catchall_domain"}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if any(_truthy_flag(record.get(name)) for name in flag_names):
            return True
        if any(
            _normalized_email_status(record.get(name)) in catch_all
            for name in status_names
        ):
            return True
    return False


def _require_requested_email(records: Sequence[Any], requested_email: str) -> None:
    returned: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for key in ("address", "email"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                returned.append(value.strip().casefold())
    if returned and any(value != requested_email.casefold() for value in returned):
        raise _ProviderUnavailable("contact_provider_malformed_response")


def _email_status(value: Any, requested_email: str) -> str:
    data = _unwrap_data(value)
    if not isinstance(data, Mapping):
        raise _ProviderUnavailable("contact_provider_malformed_response")
    nested = data.get("result") if isinstance(data.get("result"), Mapping) else None
    records = (data, nested)
    _require_requested_email(records, requested_email)
    if _email_evidence_unsafe(*records):
        return "invalid"
    if _email_evidence_is_catch_all(*records):
        return "catch_all"
    status = _normalized_email_status(
        data.get("status") or data.get("result") or data.get("verdict")
    )
    if status == "valid":
        return "valid"
    if status in _KNOWN_INVALID_EMAIL_STATUSES:
        return "invalid"
    if status in {"unknown", "pending", "queued", "processing", "running", ""}:
        return "unknown"
    return "unknown"


def _bounceban_job_id(value: Any) -> str:
    data = _unwrap_data(value)
    if not isinstance(data, Mapping):
        return ""
    return _text(data.get("id") or data.get("job_id") or data.get("request_id"))


def _bounceban_verdict(value: Any, requested_email: str) -> tuple[str, bool]:
    data = _unwrap_data(value)
    if not isinstance(data, Mapping):
        raise _ProviderUnavailable("contact_provider_malformed_response")
    api_state = _norm(data.get("status") or data.get("state"))
    raw_result = data.get("result")
    result = raw_result if isinstance(raw_result, Mapping) else data
    records = (data, result if result is not data else None)
    _require_requested_email(records, requested_email)
    verdict = _norm(
        (
            result.get("status")
            or result.get("result")
            or result.get("verdict")
            or result.get("classification")
        )
        if isinstance(raw_result, Mapping)
        else raw_result
        or result.get("verdict")
        or result.get("classification")
        or result.get("status")
    ).replace(" ", "_")
    if api_state in {"pending", "queued", "queue", "processing", "running", "verifying"} and verdict in {
        "pending",
        "queued",
        "queue",
        "processing",
        "running",
        "verifying",
        "",
    }:
        return "pending", False
    if _email_evidence_unsafe(*records):
        return "invalid", False
    if verdict == "risky":
        explanation = _norm(
            result.get("reason") or result.get("sub_status") or result.get("details")
        )
        invalid_markers = {"invalid", "disposable", "spamtrap", "abuse", "do not mail"}
        catchall = (
            result.get("is_accept_all") is True
            or "catch all" in explanation
            or "accept all" in explanation
        )
        contradicted = (
            result.get("is_disposable") is True
            or any(marker in explanation for marker in invalid_markers)
        )
        return ("catch_all", True) if catchall and not contradicted else ("unknown", False)
    if _email_evidence_is_catch_all(*records):
        return "catch_all", True
    if verdict in {"deliverable", "valid", "safe"}:
        return "valid", True
    if verdict in {"catch_all", "catch-all", "accept_all"}:
        return "catch_all", True
    if verdict in _KNOWN_INVALID_EMAIL_STATUSES or verdict in {"undeliverable", "disposable"}:
        return "invalid", False
    return "unknown", False


async def _verify_email(execute: Execute, email: str) -> tuple[str, bool, dict[str, str]]:
    hashes: dict[str, str] = {}
    zero = await _call(execute, "zerobounce_validate", {"email": email})
    hashes["zerobounce"] = _hash(zero)
    status = _email_status(zero, email)
    if status in {"valid", "catch_all"}:
        return status, True, hashes
    if status == "invalid":
        return "invalid", False, hashes

    bounce = await _call(execute, "bounceban_verify_single", {"email": email})
    hashes["bounceban"] = _hash(bounce)
    bounce_status, accepted = _bounceban_verdict(bounce, email)
    if bounce_status != "pending":
        return bounce_status, accepted, hashes
    job_id = _bounceban_job_id(bounce)
    if not job_id:
        raise _ProviderUnavailable("contact_provider_malformed_response")
    for delay in BOUNCEBAN_POLL_DELAYS_SECONDS[:BOUNCEBAN_POLL_ATTEMPTS]:
        await asyncio.sleep(delay)
        polled = await _call(execute, "bounceban_get_single_status", {"id": job_id})
        hashes["bounceban"] = _hash(polled)
        bounce_status, accepted = _bounceban_verdict(polled, email)
        if bounce_status != "pending":
            return bounce_status, accepted, hashes
    raise _ProviderUnavailable("contact_provider_timeout")


def _source_timestamp(source: Mapping[str, Any]) -> str:
    identity = source.get("call_identity") if isinstance(source.get("call_identity"), Mapping) else {}
    return _text(
        source.get("observed_at")
        or source.get("timestamp")
        or identity.get("observed_at")
        or identity.get("timestamp")
        or identity.get("completed_at")
    )


def _source_reference_matches(contact: Mapping[str, Any], source: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    attribution = contact.get("email_source") if isinstance(contact.get("email_source"), Mapping) else {}
    if _norm(attribution.get("provider")) != _SUPPORTED_SOURCE_PROVIDER:
        return False
    if _norm(attribution.get("tool")).replace(" ", "_") != _SUPPORTED_SOURCE_TOOL:
        return False
    raw_identity = source.get("call_identity")
    identity = raw_identity if isinstance(raw_identity, Mapping) else {}
    broker_claim = _text(attribution.get("broker_call_id"))
    broker_actual = _text(
        identity.get("broker_call_id")
        or identity.get("call_id")
        or identity.get("id")
        or (raw_identity if isinstance(raw_identity, str) else "")
    )
    if broker_claim and broker_claim != broker_actual:
        return False
    record_claim = _text(attribution.get("record_id"))
    identity_record = _text(identity.get("record_id"))
    profile_record = _text(
        profile.get("recordId") or profile.get("record_id") or profile.get("id")
    )
    if record_claim:
        available_records = [value for value in (identity_record, profile_record) if value]
        if not available_records or any(record_claim != value for value in available_records):
            return False
    return bool(broker_claim or record_claim)


async def verify_contact(
    company: Any,
    icp: Any,
    *,
    source_evidence: Optional[Mapping[str, Any]] = None,
    execute: Optional[Execute] = None,
    classify_role: Optional[RoleClassifier] = None,
) -> dict[str, Any]:
    """Verify one claimed contact against supported broker and email evidence."""
    raw_contact = _get(company, "contact")
    try:
        from qualification.contact_models import validate_contact_claim

        contact = validate_contact_claim(raw_contact)
    except (ValueError, TypeError, ImportError, AttributeError):
        contact = raw_contact if isinstance(raw_contact, Mapping) else {}
        return _result(contact, "mismatch", "contact_claim_invalid")

    if not isinstance(contact, Mapping):
        return _result({}, "mismatch", "contact_claim_invalid")
    subchecks: dict[str, Any] = {"claim": {"status": "pass", "reason": "contact_claim_valid"}}
    evidence_hashes: dict[str, str] = {}
    evidence_timestamps: dict[str, str] = {}

    email = _text(contact.get("email")).casefold()
    local_part = email.partition("@")[0].replace(".", "").replace("_", "").replace("-", "")
    if local_part in _GENERIC_MAILBOXES:
        subchecks["email_attribution"] = {"status": "fail", "reason": "contact_role_mailbox"}
        return _result(contact, "mismatch", "contact_role_mailbox", subchecks=subchecks)

    if not isinstance(source_evidence, Mapping):
        subchecks["source"] = {"status": "unknown", "reason": "contact_source_missing"}
        return _result(contact, "unverified", "contact_source_missing", subchecks=subchecks)
    if source_evidence.get("invalid") is True:
        reason = _text(source_evidence.get("reason")) or "email_source_reference_invalid"
        subchecks["source"] = {"status": "fail", "reason": reason}
        return _result(contact, "mismatch", reason, subchecks=subchecks)

    provider = _norm(source_evidence.get("provider"))
    tool = _norm(source_evidence.get("tool")).replace(" ", "_")
    if provider != _SUPPORTED_SOURCE_PROVIDER or tool != _SUPPORTED_SOURCE_TOOL:
        subchecks["source"] = {"status": "fail", "reason": "contact_source_unsupported"}
        return _result(contact, "mismatch", "contact_source_unsupported", subchecks=subchecks)

    runner = execute or _default_execute
    response = source_evidence.get("response")
    try:
        if response is None:
            source_input = source_evidence.get("input") if isinstance(source_evidence.get("input"), Mapping) else {}
            linked = _text(
                source_input.get("url")
                or source_input.get("publicIdentifier")
                or source_input.get("profileId")
                or contact.get("linkedin_url")
            )
            payload = {
                key: source_input[key]
                for key in ("url", "publicIdentifier", "profileId")
                if _text(source_input.get(key))
            }
            if not payload:
                payload["url"] = linked
            payload["findEmail"] = "true"
            response = await _call(runner, _SUPPORTED_SOURCE_TOOL, payload)
        evidence_hashes["source"] = _hash(response)
        timestamp = _source_timestamp(source_evidence)
        if timestamp:
            evidence_timestamps["source"] = timestamp
        profiles = _profile_candidates(_unwrap_data(response))
        claimed_linkedin = _canonical_linkedin(contact.get("linkedin_url"))
        profile = next((item for item in profiles if _profile_linkedin(item) == claimed_linkedin), None)
        if profile is None:
            if not profiles:
                if _is_completed_not_found(response):
                    subchecks["source"] = {
                        "status": "unknown",
                        "reason": "contact_source_not_found",
                    }
                    return _result(
                        contact,
                        "unverified",
                        "contact_source_not_found",
                        subchecks=subchecks,
                        evidence_hashes=evidence_hashes,
                        evidence_timestamps=evidence_timestamps,
                    )
                raise _ProviderUnavailable("contact_provider_malformed_response")
            subchecks["identity"] = {"status": "fail", "reason": "contact_person_mismatch"}
            return _result(
                contact,
                "mismatch",
                "contact_person_mismatch",
                subchecks=subchecks,
                evidence_hashes=evidence_hashes,
                evidence_timestamps=evidence_timestamps,
            )
    except _ProviderUnavailable as exc:
        subchecks["source"] = {"status": "unknown", "reason": exc.reason}
        return _result(
            contact,
            "unavailable",
            exc.reason,
            subchecks=subchecks,
            evidence_hashes=evidence_hashes,
            evidence_timestamps=evidence_timestamps,
        )

    observed_name = _norm(_profile_name(profile))
    if not observed_name or observed_name != _norm(contact.get("full_name")):
        subchecks["identity"] = {"status": "fail", "reason": "contact_person_mismatch"}
        return _result(contact, "mismatch", "contact_person_mismatch", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["identity"] = {"status": "pass", "reason": "contact_identity_verified"}

    if not _source_reference_matches(contact, source_evidence, profile):
        subchecks["source"] = {"status": "fail", "reason": "email_source_reference_invalid"}
        return _result(contact, "mismatch", "email_source_reference_invalid", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["source"] = {"status": "pass", "reason": "contact_source_verified"}

    expected_company = _company_identifiers(company)
    matching_positions = [
        position
        for position in _current_positions(profile)
        if _company_matches(expected_company, _position_company(position))
    ]
    if not matching_positions:
        subchecks["company"] = {"status": "fail", "reason": "contact_company_mismatch"}
        return _result(contact, "mismatch", "contact_company_mismatch", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["company"] = {"status": "pass", "reason": "contact_company_verified"}

    claimed_role = _text(contact.get("role"))
    claimed_role_normalized = _normalize_title(claimed_role)
    actual_roles = [
        _position_title(position)
        for position in matching_positions
        if _normalize_title(_position_title(position)) == claimed_role_normalized
    ]
    if not actual_roles:
        subchecks["role"] = {"status": "fail", "reason": "contact_role_mismatch"}
        return _result(contact, "mismatch", "contact_role_mismatch", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    targets = _get(icp, "target_roles") or []
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes, bytearray)):
        targets = []
    target_seniority = _text(_get(icp, "target_seniority"))
    role_ok = False
    try:
        for role in actual_roles:
            if await _role_matches(
                role,
                [str(item) for item in targets],
                target_seniority,
                classify_role,
            ):
                role_ok = True
                break
    except _ProviderUnavailable as exc:
        subchecks["role"] = {"status": "unknown", "reason": exc.reason}
        return _result(
            contact,
            "unavailable",
            exc.reason,
            subchecks=subchecks,
            evidence_hashes=evidence_hashes,
            evidence_timestamps=evidence_timestamps,
        )
    if not role_ok:
        subchecks["role"] = {"status": "fail", "reason": "contact_role_not_targeted"}
        return _result(contact, "mismatch", "contact_role_not_targeted", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["role"] = {"status": "pass", "reason": "contact_role_verified"}

    location_status, location_reason = _location_check(contact, profile, icp)
    subchecks["location"] = {"status": location_status, "reason": location_reason}
    if location_status == "fail":
        return _result(contact, "mismatch", location_reason, subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    if location_status == "unknown":
        return _result(contact, "unverified", location_reason, subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)

    if email not in _extract_emails(profile):
        subchecks["email_attribution"] = {"status": "fail", "reason": "contact_email_mismatch"}
        return _result(contact, "mismatch", "contact_email_mismatch", subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["email_attribution"] = {"status": "pass", "reason": "contact_email_attributed"}

    try:
        status, accepted, email_hashes = await _verify_email(runner, email)
        evidence_hashes.update(email_hashes)
    except _ProviderUnavailable as exc:
        subchecks["email_verification"] = {"status": "unknown", "reason": exc.reason}
        return _result(contact, "unavailable", exc.reason, subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    if not accepted:
        reason = "contact_email_invalid" if status == "invalid" else "contact_email_unverified"
        decision = "mismatch" if status == "invalid" else "unverified"
        subchecks["email_verification"] = {"status": "fail" if status == "invalid" else "unknown", "reason": reason}
        return _result(contact, decision, reason, email_status=status, subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
    subchecks["email_verification"] = {"status": "pass", "reason": "contact_email_verified"}
    return _result(contact, "verified", "contact_verified", email_status=status, subchecks=subchecks, evidence_hashes=evidence_hashes, evidence_timestamps=evidence_timestamps)
