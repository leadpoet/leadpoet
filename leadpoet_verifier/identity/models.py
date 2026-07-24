from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from .normalization import (
    NORMALIZATION_VERSION,
    PSL_SNAPSHOT_VERSION,
    normalize_host,
    normalize_linkedin_company_url,
    normalize_name,
    normalize_url,
)


RESOLVER_POLICY_VERSION = "company-identity-v1"
IDENTITY_SCHEMA_VERSION = "company-identity-schema-v1"
ANCHOR_SCHEMA_VERSION = "company-identity-anchor-v1"
IDENTITY_MATCH_KEY_VERSION = "identity-match-v1"


class StrictIdentityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after", check_fields=False)
    @classmethod
    def require_timezone_aware_datetimes(cls, value):
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("identity timestamps must include an explicit timezone")
        return value

    @field_serializer("*", when_used="json", check_fields=False)
    def serialize_hash_critical_values(self, value):
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
        return value


class EvidenceRef(StrictIdentityModel):
    ref: str = Field(min_length=1, max_length=160)
    evidence_type: Literal[
        "submitted", "linkedin", "first_party", "redirect", "prior_receipt", "provider"
    ]
    source_group: str = Field(min_length=1, max_length=80)
    source_url: Optional[str] = Field(default=None, max_length=2048)
    content_sha256: Optional[str] = None
    observed_at: datetime

    @field_validator("content_sha256")
    @classmethod
    def validate_digest(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("content_sha256 must be lowercase SHA-256")
        return value


class AnchorSet(StrictIdentityModel):
    anchor_schema_version: Literal["company-identity-anchor-v1"] = ANCHOR_SCHEMA_VERSION
    submitted_name: str = Field(min_length=1, max_length=500)
    submitted_domain: Optional[str] = Field(default=None, max_length=253)
    submitted_website: Optional[str] = Field(default=None, max_length=2048)
    submitted_linkedin_url: Optional[str] = Field(default=None, max_length=2048)
    model_resolved_linkedin_url: Optional[str] = Field(default=None, max_length=2048)
    model_resolved_domain: Optional[str] = Field(default=None, max_length=253)
    source_observations: list[EvidenceRef] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_anchors(self) -> "AnchorSet":
        if self.submitted_domain:
            normalize_host(self.submitted_domain)
        if self.model_resolved_domain:
            normalize_host(self.model_resolved_domain)
        if self.submitted_website:
            normalize_url(self.submitted_website, allow_bare_domain=True)
        for value in (self.submitted_linkedin_url, self.model_resolved_linkedin_url):
            if value:
                normalize_linkedin_company_url(value)
        if len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode()) > 32768:
            raise ValueError("anchor set exceeds 32 KiB")
        return self


class CompanyName(StrictIdentityModel):
    display_value: str = Field(min_length=1, max_length=500)
    normalized_value: str = Field(min_length=1, max_length=500)
    type: Literal["legal", "common", "brand", "former", "acronym", "translated"]
    locale: Optional[str] = Field(default=None, max_length=35)
    country_code: Optional[str] = Field(default=None, pattern=r"^[A-Z]{2}$")
    source_evidence_ref: str = Field(min_length=1, max_length=160)
    observed_at: datetime


class LinkedInCompanyIdentity(StrictIdentityModel):
    company_id: Optional[str] = Field(default=None, pattern=r"^[0-9]{1,30}$")
    canonical_url: str = Field(max_length=2048)
    normalized_slug: str = Field(min_length=1, max_length=100)
    former_slugs: list[str] = Field(default_factory=list, max_length=16)
    listed_website_host: Optional[str] = Field(default=None, max_length=253)
    source_evidence_ref: str = Field(min_length=1, max_length=160)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_linkedin(self) -> "LinkedInCompanyIdentity":
        parsed = normalize_linkedin_company_url(self.canonical_url)
        if parsed.normalized_slug != self.normalized_slug:
            raise ValueError("LinkedIn slug does not match canonical URL")
        if self.company_id and parsed.company_id and self.company_id != parsed.company_id:
            raise ValueError("LinkedIn numeric IDs conflict")
        if self.listed_website_host:
            normalize_host(self.listed_website_host)
        return self


class RedirectHop(StrictIdentityModel):
    source_url: str = Field(max_length=2048)
    target_url: str = Field(max_length=2048)
    status: int = Field(ge=300, le=399)
    elapsed_ms: int = Field(ge=0, le=120000)


class WebsiteObservation(StrictIdentityModel):
    requested_url: str = Field(max_length=2048)
    final_url: str = Field(max_length=2048)
    status: int = Field(ge=100, le=599)
    fetched_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    names: list[str] = Field(default_factory=list, max_length=24)
    linkedin_company_urls: list[str] = Field(default_factory=list, max_length=16)
    outbound_hosts: list[str] = Field(default_factory=list, max_length=64)
    canonical_url: Optional[str] = Field(default=None, max_length=2048)
    redirects: list[RedirectHop] = Field(default_factory=list, max_length=5)
    parked: bool = False
    aggregator: bool = False
    shared_infrastructure: bool = False
    contradictory_names: list[str] = Field(default_factory=list, max_length=16)
    transient_failure: bool = False
    unsafe_target: bool = False
    source_evidence_ref: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_urls(self) -> "WebsiteObservation":
        normalize_url(self.requested_url)
        normalize_url(self.final_url)
        for value in self.linkedin_company_urls:
            normalize_linkedin_company_url(value)
        for value in self.outbound_hosts:
            normalize_host(value)
        return self


class PriorReceiptSummary(StrictIdentityModel):
    receipt_id: UUID
    same_tenant: bool
    decision: Literal["resolved", "ambiguous", "unavailable", "rejected"]
    positive_rule_ids: list[str] = Field(max_length=8)
    stable_linkedin_key: Optional[str] = Field(default=None, max_length=80)
    verified_domain: str = Field(max_length=253)
    identity_match_key: str = Field(pattern=r"^idmk1:[0-9a-f]{64}$")
    valid_until: datetime
    superseded: bool = False


class ResolverInput(StrictIdentityModel):
    request_id: UUID
    run_id: UUID
    candidate_id: UUID
    anchor_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchor_set: AnchorSet
    website_observations: list[WebsiteObservation] = Field(default_factory=list, max_length=8)
    linkedin_identities: list[LinkedInCompanyIdentity] = Field(default_factory=list, max_length=4)
    prior_receipt: Optional[PriorReceiptSummary] = None
    verified_primary_domains: list[str] = Field(default_factory=list, max_length=8)
    relationship_conflict: bool = False
    country_conflict: bool = False
    observed_at: datetime


class CompanyDomain(StrictIdentityModel):
    ascii_host: str
    unicode_host: str
    registrable_domain: str
    public_suffix: str
    role: Literal[
        "primary", "alias", "regional", "careers", "newsroom", "support",
        "product", "former", "parent", "subsidiary", "shared_platform", "unverified",
    ]
    relation_to_identity: Literal[
        "same_entity", "related_entity", "shared_infrastructure", "unknown"
    ]
    ownership_status: Literal["verified", "provisional", "contradicted", "expired"]
    http_origin: str
    redirect_target_host: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list, max_length=16)


class CanonicalCompanyIdentity(StrictIdentityModel):
    schema_version: Literal["company-identity-schema-v1"] = IDENTITY_SCHEMA_VERSION
    resolver_policy_version: Literal["company-identity-v1"] = RESOLVER_POLICY_VERSION
    normalization_version: Literal["company-identity-normalization-v1"] = NORMALIZATION_VERSION
    public_suffix_snapshot_version: Literal["2026-07-15_18-13-59_UTC"] = PSL_SNAPSHOT_VERSION
    company_identity_id: UUID
    identity_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_match_key: Optional[str] = Field(
        default=None,
        pattern=r"^idmk1:[0-9a-f]{64}$",
    )
    identity_match_key_version: Literal["identity-match-v1"] = IDENTITY_MATCH_KEY_VERSION
    resolution_status: Literal["resolved", "ambiguous", "unavailable", "rejected"]
    identity_tier: Literal["exact", "strong", "provisional", "ambiguous", "rejected"]
    positive_rule_ids: list[str] = Field(default_factory=list, max_length=8)
    negative_rule_ids: list[str] = Field(default_factory=list, max_length=16)
    conflicts: list[str] = Field(default_factory=list, max_length=16)
    canonical_name: CompanyName
    legal_name: Optional[CompanyName] = None
    aliases: list[CompanyName] = Field(default_factory=list, max_length=24)
    linkedin_company: Optional[LinkedInCompanyIdentity] = None
    domains: list[CompanyDomain] = Field(default_factory=list, max_length=24)
    evidence_refs: list[str] = Field(default_factory=list, max_length=64)
    resolved_at: datetime
    valid_until: datetime
    supersedes_identity_receipt_id: Optional[UUID] = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "CanonicalCompanyIdentity":
        eligible = {
            "ID-A1-LINKEDIN-ID-WEBSITE", "ID-A2-RECIPROCAL-LINKEDIN-SLUG",
            "ID-A3-FRESH-PRIOR-RECEIPT", "ID-B1-CONTROLLED-SUBDOMAIN",
            "ID-B2-RECIPROCAL-REGIONAL-ALIAS", "ID-B3-VERIFIED-DOMAIN-MIGRATION",
        }
        if self.resolution_status == "resolved":
            if self.identity_tier not in {"exact", "strong"}:
                raise ValueError("resolved identity requires exact or strong tier")
            if not set(self.positive_rule_ids) & eligible:
                raise ValueError("resolved identity requires an enumerated v1 rule")
            if not any(
                domain.relation_to_identity == "same_entity"
                and domain.ownership_status == "verified"
                for domain in self.domains
            ):
                raise ValueError("resolved identity requires a verified same-entity domain")
        elif self.identity_match_key is not None:
            raise ValueError("unresolved identities cannot expose a match key")
        return self


class IdentityDecision(StrictIdentityModel):
    outcome: Literal["resolved", "ambiguous", "unavailable", "rejected"]
    identity_tier: Literal["exact", "strong", "provisional", "ambiguous", "rejected"]
    positive_rule_ids: list[str] = Field(default_factory=list, max_length=8)
    negative_rule_ids: list[str] = Field(default_factory=list, max_length=16)
    conflicts: list[str] = Field(default_factory=list, max_length=16)
    reason_codes: list[str] = Field(min_length=1, max_length=16)
    canonical_identity: CanonicalCompanyIdentity


def submitted_company_name(anchor: AnchorSet, observed_at: datetime) -> CompanyName:
    return CompanyName(
        display_value=anchor.submitted_name,
        normalized_value=normalize_name(anchor.submitted_name),
        type="common",
        source_evidence_ref="submitted-company-name",
        observed_at=observed_at,
    )
