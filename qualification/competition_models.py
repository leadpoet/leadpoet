"""Shared input and output models for the public sourcing competition."""

from __future__ import annotations

import ipaddress
from datetime import date
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def public_http_url(value: Any, *, allow_empty: bool = False) -> str:
    """Return one normalized public HTTP URL."""

    text = str(value or "").strip()
    if not text and allow_empty:
        return ""
    parsed = urlsplit(text)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("must be an absolute public HTTP URL")
    hostname = parsed.hostname.rstrip(".").lower()
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
        parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("must be a public URL") from exc
    if hostname == "localhost" or hostname.endswith(
        (".internal", ".invalid", ".local", ".localhost", ".onion", ".test")
    ):
        raise ValueError("must be a public URL")
    try:
        address = ipaddress.ip_address(ascii_hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("must be a public URL")
    if address is None:
        labels = ascii_hostname.split(".")
        if len(labels) < 2 or not any(character.isalpha() for character in labels[-1]):
            raise ValueError("must be a public URL")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


class CompetitionIntentSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    matched_icp_signal: int = Field(ge=0)
    description: str = Field(min_length=1)
    date: date
    why_now: str = Field(min_length=1)
    url: str
    snippet: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return public_http_url(value)


class CompetitionRequiredAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    passed: bool
    evidence_url: str
    evidence_quote: str = Field(min_length=1)
    explanation: str = Field(min_length=1)

    @field_validator("evidence_url")
    @classmethod
    def validate_evidence_url(cls, value: str) -> str:
        return public_http_url(value)


class CompetitionCompany(BaseModel):
    """The ordinary company result returned by every public agent bundle."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    company_name: str = Field(min_length=1)
    company_website: str
    company_linkedin: str = ""
    industry: str
    employee_count: str
    company_stage: str = ""
    country: str
    state: str = ""
    fit_summary: str = Field(min_length=1)
    fit_evidence_urls: list[str]
    intent_signals: list[CompetitionIntentSignal] = Field(min_length=1)
    required_attribute: Optional[CompetitionRequiredAttribute] = None

    @field_validator("company_website")
    @classmethod
    def validate_website(cls, value: str) -> str:
        return public_http_url(value)

    @field_validator("company_linkedin")
    @classmethod
    def validate_linkedin(cls, value: str) -> str:
        return public_http_url(value, allow_empty=True)

    @field_validator("fit_evidence_urls")
    @classmethod
    def validate_fit_urls(cls, values: list[str]) -> list[str]:
        return [public_http_url(value) for value in values]


def validate_companies(values: Any, *, max_companies: int) -> list[dict[str, Any]]:
    """Validate one result list and return plain JSON-ready dictionaries."""

    if not isinstance(values, list):
        raise ValueError("companies must be a list")
    if len(values) > int(max_companies):
        raise ValueError("too many companies")
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("each company must be an object")
        rows.append(CompetitionCompany.model_validate(value).model_dump(mode="json"))
    return rows
