from __future__ import annotations

import hashlib
from datetime import timedelta
from uuid import uuid4

from .canonical import canonical_sha256
from .models import (
    CanonicalCompanyIdentity,
    CompanyDomain,
    IdentityDecision,
    LinkedInCompanyIdentity,
    ResolverInput,
    WebsiteObservation,
    submitted_company_name,
)
from .normalization import (
    exact_or_legal_name_match,
    is_label_subdomain,
    normalize_host,
    normalize_linkedin_company_url,
    normalize_url,
)


ELIGIBLE_RULES = frozenset({
    "ID-A1-LINKEDIN-ID-WEBSITE",
    "ID-A2-RECIPROCAL-LINKEDIN-SLUG",
    "ID-A3-FRESH-PRIOR-RECEIPT",
    "ID-B1-CONTROLLED-SUBDOMAIN",
    "ID-B2-RECIPROCAL-REGIONAL-ALIAS",
    "ID-B3-VERIFIED-DOMAIN-MIGRATION",
})


def _candidate_host(value: ResolverInput) -> str | None:
    for candidate in (
        value.anchor_set.model_resolved_domain,
        value.anchor_set.submitted_domain,
        value.anchor_set.submitted_website,
    ):
        if not candidate:
            continue
        try:
            if "://" in candidate:
                return normalize_url(candidate, allow_bare_domain=True).ascii_host
            return normalize_host(candidate).ascii_host
        except ValueError:
            continue
    return None


def _observation_hosts(observation: WebsiteObservation) -> set[str]:
    hosts = {
        normalize_url(observation.requested_url).ascii_host,
        normalize_url(observation.final_url).ascii_host,
    }
    for hop in observation.redirects:
        hosts.add(normalize_url(hop.source_url).ascii_host)
        hosts.add(normalize_url(hop.target_url).ascii_host)
    return hosts


def _name_agrees(value: ResolverInput, observation: WebsiteObservation) -> bool:
    return any(
        exact_or_legal_name_match(value.anchor_set.submitted_name, name)
        for name in observation.names
    )


def _strong_conflicts(value: ResolverInput) -> list[str]:
    conflicts: set[str] = set()
    ids = {item.company_id for item in value.linkedin_identities if item.company_id}
    slugs = {item.normalized_slug for item in value.linkedin_identities}
    if len(ids) > 1:
        conflicts.add("conflicting_linkedin_company_ids")
    if not ids and len(slugs) > 1:
        conflicts.add("conflicting_linkedin_company_slugs")
    if value.relationship_conflict:
        conflicts.add("related_entity_not_same_identity")
    if value.country_conflict:
        conflicts.add("conflicting_company_countries")
    if any(item.contradictory_names for item in value.website_observations):
        conflicts.add("conflicting_first_party_names")
    if value.prior_receipt and value.prior_receipt.stable_linkedin_key and value.linkedin_identities:
        current_keys = {
            key
            for item in value.linkedin_identities
            for key in (item.company_id, item.normalized_slug)
            if key
        }
        if value.prior_receipt.stable_linkedin_key not in current_keys:
            conflicts.add("prior_receipt_linkedin_identity_changed")
    return sorted(conflicts)


def _website_for_host(value: ResolverInput, host: str) -> WebsiteObservation | None:
    for observation in value.website_observations:
        if host in _observation_hosts(observation):
            return observation
    return None


def _is_usable_site(observation: WebsiteObservation | None) -> bool:
    return bool(
        observation
        and 200 <= observation.status < 400
        and not observation.parked
        and not observation.aggregator
        and not observation.shared_infrastructure
        and not observation.unsafe_target
        and not observation.transient_failure
    )


def _site_links_linkedin(
    observation: WebsiteObservation | None,
    linkedin: LinkedInCompanyIdentity,
) -> bool:
    return bool(
        observation
        and linkedin.canonical_url in observation.linkedin_company_urls
    )


def _is_direct_host_fetch(observation: WebsiteObservation | None, host: str) -> bool:
    return bool(
        observation
        and not observation.redirects
        and normalize_url(observation.requested_url).ascii_host == host
        and normalize_url(observation.final_url).ascii_host == host
    )


def _prior_anchor_matches(
    value: ResolverInput,
    linkedin: LinkedInCompanyIdentity | None,
) -> bool:
    prior = value.prior_receipt
    if not prior or not prior.stable_linkedin_key or not linkedin:
        return False
    return prior.stable_linkedin_key in {
        linkedin.company_id,
        linkedin.normalized_slug,
    }


def _reciprocal_region(
    value: ResolverInput,
    candidate_host: str,
    linkedin: LinkedInCompanyIdentity,
) -> bool:
    candidate_observation = _website_for_host(value, candidate_host)
    if not _is_usable_site(candidate_observation) or not _site_links_linkedin(
        candidate_observation, linkedin
    ):
        return False
    for primary in value.verified_primary_domains:
        primary_host = normalize_host(primary).ascii_host
        if normalize_host(primary_host).registrable_domain == normalize_host(candidate_host).registrable_domain:
            continue
        primary_observation = _website_for_host(value, primary_host)
        if not _is_usable_site(primary_observation):
            continue
        if (
            primary_host in candidate_observation.outbound_hosts
            and candidate_host in primary_observation.outbound_hosts
            and _site_links_linkedin(primary_observation, linkedin)
        ):
            return True
    return False


def _domain_migration(
    value: ResolverInput,
    candidate_host: str,
    linkedin: LinkedInCompanyIdentity,
) -> bool:
    prior = value.prior_receipt
    if not prior or not prior.same_tenant or prior.superseded or prior.valid_until < value.observed_at:
        return False
    if not _prior_anchor_matches(value, linkedin):
        return False
    old_host = normalize_host(prior.verified_domain).ascii_host
    for observation in value.website_observations:
        if normalize_url(observation.requested_url).ascii_host != old_host:
            continue
        if normalize_url(observation.final_url).ascii_host != candidate_host:
            continue
        if any(
            hop.source_url.startswith("https://") and hop.target_url.startswith("http://")
            for hop in observation.redirects
        ):
            continue
        if (
            old_host in observation.outbound_hosts
            and _is_usable_site(observation)
            and _site_links_linkedin(observation, linkedin)
        ):
            return True
    return False


def _make_match_key(rule: str, value: ResolverInput, candidate_host: str) -> str:
    if rule == "ID-A1-LINKEDIN-ID-WEBSITE":
        linkedin_id = next(item.company_id for item in value.linkedin_identities if item.company_id)
        anchor = f"linkedin-company-id:{linkedin_id}"
    elif rule in {"ID-A3-FRESH-PRIOR-RECEIPT", "ID-B3-VERIFIED-DOMAIN-MIGRATION"}:
        assert value.prior_receipt is not None
        return value.prior_receipt.identity_match_key
    elif rule in {"ID-B1-CONTROLLED-SUBDOMAIN", "ID-B2-RECIPROCAL-REGIONAL-ALIAS"}:
        primary = normalize_host(value.verified_primary_domains[0]).registrable_domain
        anchor = f"verified-domain:{primary}"
    else:
        anchor = f"verified-domain:{normalize_host(candidate_host).registrable_domain}"
    return "idmk1:" + hashlib.sha256(anchor.encode("utf-8")).hexdigest()


def _decision(
    value: ResolverInput,
    *,
    outcome: str,
    tier: str,
    positive: list[str],
    negative: list[str],
    conflicts: list[str],
    reasons: list[str],
    candidate_host: str | None,
) -> IdentityDecision:
    observed_at = value.observed_at
    canonical_name = submitted_company_name(value.anchor_set, observed_at)
    resolved = outcome == "resolved" and candidate_host is not None
    rule = positive[0] if positive else None
    match_key = _make_match_key(rule, value, candidate_host) if resolved and rule else None
    domains: list[CompanyDomain] = []
    if candidate_host:
        parts = normalize_host(candidate_host)
        observation = _website_for_host(value, candidate_host)
        domains.append(CompanyDomain(
            ascii_host=parts.ascii_host,
            unicode_host=parts.unicode_host,
            registrable_domain=parts.registrable_domain,
            public_suffix=parts.public_suffix,
            role="primary" if resolved else ("shared_platform" if parts.is_private_suffix else "unverified"),
            relation_to_identity="same_entity" if resolved else (
                "shared_infrastructure" if parts.is_private_suffix else "unknown"
            ),
            ownership_status="verified" if resolved else (
                "contradicted" if negative else "provisional"
            ),
            http_origin=(
                normalize_url(observation.final_url).origin
                if observation else f"https://{parts.ascii_host}"
            ),
            redirect_target_host=(
                normalize_url(observation.final_url).ascii_host
                if observation and normalize_url(observation.final_url).ascii_host != parts.ascii_host
                else None
            ),
            evidence_refs=[observation.source_evidence_ref] if observation else [],
        ))

    identity_id = uuid4()
    identity_material = {
        "schema_version": "company-identity-schema-v1",
        "resolver_policy_version": "company-identity-v1",
        "normalization_version": "company-identity-normalization-v1",
        "public_suffix_snapshot_version": "2026-07-15_18-13-59_UTC",
        "company_identity_id": str(identity_id),
        "identity_match_key": match_key,
        "identity_match_key_version": "identity-match-v1",
        "resolution_status": outcome,
        "identity_tier": tier,
        "positive_rule_ids": sorted(positive),
        "negative_rule_ids": sorted(negative),
        "conflicts": sorted(conflicts),
        "canonical_name": canonical_name,
        "legal_name": None,
        "aliases": [],
        "linkedin_company": value.linkedin_identities[0] if value.linkedin_identities else None,
        "domains": domains,
        "evidence_refs": sorted({
            item.source_evidence_ref for item in value.website_observations
        } | {
            item.source_evidence_ref for item in value.linkedin_identities
        }),
        "resolved_at": observed_at,
        "valid_until": observed_at + timedelta(hours=24),
        "supersedes_identity_receipt_id": None,
    }
    snapshot_hash = canonical_sha256(identity_material)
    identity = CanonicalCompanyIdentity(
        **identity_material,
        identity_snapshot_hash=snapshot_hash,
    )
    return IdentityDecision(
        outcome=outcome,
        identity_tier=tier,
        positive_rule_ids=sorted(positive),
        negative_rule_ids=sorted(negative),
        conflicts=sorted(conflicts),
        reason_codes=reasons,
        canonical_identity=identity,
    )


def resolve_identity(value: ResolverInput) -> IdentityDecision:
    """Evaluate the closed v1 allowlist. No weak-signal accumulation is permitted."""

    if canonical_sha256(value.anchor_set) != value.anchor_set_hash:
        raise ValueError("anchor_set_hash does not match canonical anchor set")
    candidate_host = _candidate_host(value)
    if not candidate_host:
        return _decision(
            value, outcome="rejected", tier="rejected", positive=[],
            negative=["ID-N-INVALID-DOMAIN"], conflicts=[],
            reasons=["invalid_or_missing_company_domain"], candidate_host=None,
        )

    conflicts = _strong_conflicts(value)
    if conflicts:
        return _decision(
            value, outcome="ambiguous", tier="ambiguous", positive=[], negative=[],
            conflicts=conflicts, reasons=["strong_identity_conflict"],
            candidate_host=candidate_host,
        )

    parts = normalize_host(candidate_host)
    site = _website_for_host(value, candidate_host)
    negative: list[str] = []
    if parts.is_private_suffix or (site and site.shared_infrastructure):
        negative.append("ID-N-SHARED-INFRASTRUCTURE")
    if site and site.parked:
        negative.append("ID-N-PARKED-DOMAIN")
    if site and site.aggregator:
        negative.append("ID-N-AGGREGATOR-HOST")
    if site and site.unsafe_target:
        negative.append("ID-N-UNSAFE-TARGET")
    if negative:
        return _decision(
            value, outcome="rejected", tier="rejected", positive=[], negative=negative,
            conflicts=[], reasons=["deterministic_identity_rejection"],
            candidate_host=candidate_host,
        )
    if site and site.transient_failure:
        return _decision(
            value, outcome="unavailable", tier="provisional", positive=[], negative=[],
            conflicts=[], reasons=["required_first_party_source_unavailable"],
            candidate_host=candidate_host,
        )

    usable = _is_usable_site(site)
    name_agrees = bool(site and _name_agrees(value, site))
    linkedin = value.linkedin_identities[0] if value.linkedin_identities else None
    listed_host = normalize_host(linkedin.listed_website_host).ascii_host if linkedin and linkedin.listed_website_host else None
    site_linkedin = {
        normalize_linkedin_company_url(url).canonical_url
        for url in (site.linkedin_company_urls if site else [])
    }

    rule: str | None = None
    tier = "provisional"
    if (
        linkedin and linkedin.company_id and listed_host == candidate_host
        and usable and name_agrees
    ):
        rule, tier = "ID-A1-LINKEDIN-ID-WEBSITE", "exact"
    elif (
        linkedin and listed_host == candidate_host and usable and name_agrees
        and linkedin.canonical_url in site_linkedin
    ):
        rule, tier = "ID-A2-RECIPROCAL-LINKEDIN-SLUG", "exact"
    elif (
        value.prior_receipt and value.prior_receipt.same_tenant
        and not value.prior_receipt.superseded
        and value.prior_receipt.valid_until >= value.observed_at
        and value.prior_receipt.decision == "resolved"
        and set(value.prior_receipt.positive_rule_ids) & {
            "ID-A1-LINKEDIN-ID-WEBSITE", "ID-A2-RECIPROCAL-LINKEDIN-SLUG"
        }
        and normalize_host(value.prior_receipt.verified_domain).ascii_host == candidate_host
        and usable and _is_direct_host_fetch(site, candidate_host)
        and _prior_anchor_matches(value, linkedin)
    ):
        rule, tier = "ID-A3-FRESH-PRIOR-RECEIPT", "exact"
    elif (
        value.verified_primary_domains
        and any(is_label_subdomain(candidate_host, primary) for primary in value.verified_primary_domains)
        and usable and name_agrees and not parts.is_private_suffix
    ):
        rule, tier = "ID-B1-CONTROLLED-SUBDOMAIN", "strong"
    elif linkedin and usable and name_agrees and _reciprocal_region(
        value, candidate_host, linkedin
    ):
        rule, tier = "ID-B2-RECIPROCAL-REGIONAL-ALIAS", "strong"
    elif linkedin and usable and name_agrees and _domain_migration(
        value, candidate_host, linkedin
    ):
        rule, tier = "ID-B3-VERIFIED-DOMAIN-MIGRATION", "strong"

    if rule:
        return _decision(
            value, outcome="resolved", tier=tier, positive=[rule], negative=[],
            conflicts=[], reasons=["verified_company_identity"], candidate_host=candidate_host,
        )
    return _decision(
        value, outcome="rejected", tier="provisional", positive=[], negative=[],
        conflicts=[], reasons=["insufficient_verified_identity_anchors"],
        candidate_host=candidate_host,
    )
