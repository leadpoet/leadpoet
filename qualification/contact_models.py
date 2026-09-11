"""Strict, normalized contact claims for Arena company results."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Mapping, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# This is deliberately a code-owned allowlist. Adding a broker operation is a
# reviewed contract change; miner output cannot name an arbitrary tool.
CONTACT_SOURCE_REGISTRY = frozenset(
    {
        ("harvestapi", "harvestapi_get_profile"),
    }
)

_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_LINKEDIN_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,119}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/~-]{0,199}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CREDENTIAL_RE = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~-]{12,}|sk-[a-z0-9_-]{12,}|"
    r"(?:api[_ -]?key|password|secret|token)\s*[:=])"
)


def _bounded_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = " ".join(value.strip().split())
    if not text or len(text) > max_length or _CONTROL_RE.search(value):
        raise ValueError(f"{field} is missing or too long")
    if _CREDENTIAL_RE.search(text):
        raise ValueError(f"{field} must not contain credentials")
    return text


@lru_cache(maxsize=1)
def _country_codes() -> dict[str, str]:
    aliases = {
        "uk": "GB",
        "u.k.": "GB",
        "great britain": "GB",
        "united states of america": "US",
        "usa": "US",
        "u.s.a.": "US",
        "u.s.": "US",
        "south korea": "KR",
        "north korea": "KP",
        "russia": "RU",
        "vietnam": "VN",
        "laos": "LA",
        "bolivia": "BO",
        "tanzania": "TZ",
        "venezuela": "VE",
    }
    try:
        import geonamescache

        for country in geonamescache.GeonamesCache().get_countries().values():
            iso2 = str(country.get("iso") or "").upper()
            if len(iso2) != 2:
                continue
            for candidate in (
                country.get("name"),
                country.get("iso"),
                country.get("iso3"),
            ):
                if candidate:
                    aliases[str(candidate).strip().casefold()] = iso2
    except (ImportError, AttributeError, TypeError):
        # The production requirements include geonamescache. These core
        # aliases keep validation deterministic in minimal tooling images.
        aliases.update(
            {
                "united states": "US",
                "canada": "CA",
                "united kingdom": "GB",
                "australia": "AU",
                "new zealand": "NZ",
                "ireland": "IE",
                "singapore": "SG",
                "india": "IN",
                "germany": "DE",
                "france": "FR",
            }
        )
    return aliases


def normalize_country_code(value: Any) -> str:
    """Normalize a supported ISO-2, ISO-3, or common country name to ISO-2."""

    text = _bounded_text(value, "location.country", max_length=80)
    normalized = _country_codes().get(text.casefold())
    if normalized is None:
        raise ValueError("location.country must be a recognized country")
    return normalized


def normalize_linkedin_profile_url(value: Any) -> str:
    """Return one canonical LinkedIn person-profile URL without tracking data."""

    text = _bounded_text(value, "linkedin_url", max_length=512)
    parsed = urlsplit(text)
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not hostname
        or not (hostname == "linkedin.com" or hostname.endswith(".linkedin.com"))
        or parsed.username
        or parsed.password
    ):
        raise ValueError("linkedin_url must be an absolute LinkedIn profile URL")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "in":
        raise ValueError("linkedin_url must identify one LinkedIn person profile")
    slug = parts[1]
    if _LINKEDIN_SLUG_RE.fullmatch(slug) is None:
        raise ValueError("linkedin_url profile slug is invalid")
    return urlunsplit(
        ("https", "www.linkedin.com", f"/in/{quote(slug, safe='._~-')}/", "", "")
    )


class ContactLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    country: str
    region: Optional[str] = Field(default=None, max_length=120)
    city: Optional[str] = Field(default=None, max_length=120)

    @field_validator("country", mode="before")
    @classmethod
    def validate_country(cls, value: Any) -> str:
        return normalize_country_code(value)

    @field_validator("region", "city", mode="before")
    @classmethod
    def validate_optional_location(cls, value: Any, info: Any) -> Optional[str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return _bounded_text(value, f"location.{info.field_name}", max_length=120)


class ContactEmailSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: str = Field(min_length=1, max_length=80)
    tool: str = Field(min_length=1, max_length=120)
    broker_call_id: Optional[str] = Field(default=None, max_length=200)
    record_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("provider", "tool", mode="before")
    @classmethod
    def normalize_source_name(cls, value: Any, info: Any) -> str:
        return _bounded_text(value, f"email_source.{info.field_name}", max_length=120).casefold()

    @field_validator("broker_call_id", "record_id", mode="before")
    @classmethod
    def validate_source_id(cls, value: Any, info: Any) -> Optional[str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        text = _bounded_text(value, f"email_source.{info.field_name}", max_length=200)
        if _SOURCE_ID_RE.fullmatch(text) is None:
            raise ValueError(f"email_source.{info.field_name} is invalid")
        return text

    @model_validator(mode="after")
    def validate_registered_source(self) -> "ContactEmailSource":
        if (self.provider, self.tool) not in CONTACT_SOURCE_REGISTRY:
            raise ValueError("email_source provider and tool are unsupported")
        if not self.broker_call_id and not self.record_id:
            raise ValueError("email_source requires broker_call_id or record_id")
        return self


class ContactClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    full_name: str = Field(min_length=1, max_length=160)
    role: str = Field(min_length=1, max_length=200)
    linkedin_url: str = Field(min_length=1, max_length=512)
    location: ContactLocation
    email: str = Field(min_length=3, max_length=254)
    email_source: ContactEmailSource

    @field_validator("full_name", "role", mode="before")
    @classmethod
    def validate_identity_text(cls, value: Any, info: Any) -> str:
        maximum = 160 if info.field_name == "full_name" else 200
        return _bounded_text(value, info.field_name, max_length=maximum)

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def validate_linkedin_url(cls, value: Any) -> str:
        return normalize_linkedin_profile_url(value)

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, value: Any) -> str:
        text = _bounded_text(value, "email", max_length=254).casefold()
        if _EMAIL_RE.fullmatch(text) is None:
            raise ValueError("email must be a valid work email address")
        local, domain = text.rsplit("@", 1)
        try:
            domain = domain.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("email domain is invalid") from exc
        normalized = f"{local}@{domain}"
        if len(normalized) > 254:
            raise ValueError("email is too long")
        return normalized


def validate_contact_claim(value: Any) -> dict[str, Any]:
    """Validate a contact claim and return its normalized JSON representation."""

    if not isinstance(value, Mapping):
        raise ValueError("contact must be an object")
    try:
        return ContactClaim.model_validate(value).model_dump(mode="json", exclude_none=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("contact fails the contact claim contract") from exc
