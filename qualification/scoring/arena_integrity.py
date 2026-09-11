"""Pure helpers for the versioned Arena score-integrity policy.

The helpers in this module deliberately do not depend on the scorer models.  The
Arena parent can therefore use the same identity grouping before a retry that
the scorer uses after validation, without importing the networked judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Mapping, Optional, Sequence

from leadpoet_verifier.identity.normalization import (
    NormalizationError,
    normalize_name,
    normalize_url,
)


ARENA_INTEGRITY_POLICY = "arena_integrity_v1"
ARENA_INTEGRITY_SCORER_ADAPTER = "qualification_integrity_v2"

_VERIFIED_IDENTITY_SOURCES = frozenset({
    "company_homepage",
    "company_web_reverification",
})
_DATE_NOTE_RE = re.compile(
    r"^(?P<kind>source_event_date|source_publication_date):"
    r"(?P<value>\d{4}-\d{2}-\d{2})$"
)


@dataclass(frozen=True)
class ArenaCompanyIdentity:
    """One conservative, comparison-safe company identity candidate."""

    key: str
    registrable_domain: str
    website_host: str
    normalized_name: str
    verified_linkedin_slug: str = ""
    verified: bool = False


@dataclass(frozen=True)
class ArenaCompanyIdentityDecision:
    """Stable duplicate annotation for one company in the original slice."""

    company_index: int
    identity: ArenaCompanyIdentity
    duplicate_company: bool
    duplicate_of_index: Optional[int] = None


@dataclass(frozen=True)
class SourceDateVerdict:
    """Freshness result based only on source-grounded dates."""

    verdict: str
    authoritative_date: Optional[str]
    basis: str
    age_days: Optional[int]


def _identity_values(
    company: Mapping[str, Any],
    verified_identity_receipt: Optional[Mapping[str, Any]],
) -> tuple[str, str, str, bool]:
    receipt = verified_identity_receipt or {}
    receipt_verified = bool(
        receipt.get("decision") == "match"
        and receipt.get("evidence_source") in _VERIFIED_IDENTITY_SOURCES
    )
    website = (
        f"https://{receipt.get('observed_domain')}"
        if receipt_verified and receipt.get("observed_domain")
        else str(company.get("company_website") or "")
    )
    name = (
        str(receipt.get("observed_name") or "")
        if receipt_verified
        else str(company.get("company_name") or "")
    )
    linkedin_slug = (
        str(receipt.get("observed_linkedin_slug") or "").strip().casefold()
        if receipt_verified
        else ""
    )
    return website, name, linkedin_slug, receipt_verified


def canonical_company_identity(
    company: Mapping[str, Any],
    *,
    verified_identity_receipt: Optional[Mapping[str, Any]] = None,
) -> ArenaCompanyIdentity:
    """Build a stable company key without trusting a submitted LinkedIn alias.

    A registrable domain alone is not an identity: holding-company and hosted
    marketplace domains can represent multiple businesses.  The conservative
    submitted key therefore combines the vendored-PSL domain with an exact or
    legal-suffix-normalized company name.  A verifier-confirmed receipt may
    replace those values and may key on its independently observed LinkedIn
    slug. Submitted LinkedIn URLs never merge or split identities because they
    have not yet been corroborated.
    """

    website, name, verified_linkedin_slug, verified = _identity_values(
        company, verified_identity_receipt
    )
    try:
        normalized_url = normalize_url(website, allow_bare_domain=True)
    except (NormalizationError, TypeError, ValueError):
        host = ""
        registrable = ""
    else:
        host = normalized_url.ascii_host.removeprefix("www.")
        registrable = normalized_url.domain.registrable_domain
    normalized = normalize_name(name, strip_legal_suffix=True)
    key = (
        f"domain:{registrable}|linkedin:{verified_linkedin_slug}"
        if verified and registrable and verified_linkedin_slug
        else f"domain:{registrable}|name:{normalized}"
    )
    return ArenaCompanyIdentity(
        key=key,
        registrable_domain=registrable,
        website_host=host,
        normalized_name=normalized,
        verified_linkedin_slug=verified_linkedin_slug,
        verified=verified,
    )


def same_company_identity(
    left: ArenaCompanyIdentity, right: ArenaCompanyIdentity
) -> bool:
    """Compare exact names or independently verified LinkedIn identities."""

    return bool(
        left.registrable_domain
        and left.registrable_domain == right.registrable_domain
        and (
            (
                left.verified
                and right.verified
                and left.verified_linkedin_slug
                and left.verified_linkedin_slug == right.verified_linkedin_slug
            )
            or (
                left.normalized_name
                and left.normalized_name == right.normalized_name
            )
        )
    )


def company_identity_alias_keys(identity: ArenaCompanyIdentity) -> tuple[str, ...]:
    """Return bounded corroborated handles that may identify one company."""

    aliases: list[str] = []
    if identity.registrable_domain and identity.normalized_name:
        aliases.append(
            f"domain:{identity.registrable_domain}|name:{identity.normalized_name}"
        )
    if (
        identity.verified
        and identity.registrable_domain
        and identity.verified_linkedin_slug
    ):
        aliases.append(
            "domain:"
            f"{identity.registrable_domain}|linkedin:{identity.verified_linkedin_slug}"
        )
    return tuple(dict.fromkeys(aliases))


def mark_duplicate_companies(
    companies: Sequence[Mapping[str, Any]], *, limit: int
) -> list[ArenaCompanyIdentityDecision]:
    """Annotate the original first-N order using conservative identity groups."""

    decisions: list[ArenaCompanyIdentityDecision] = []
    first_by_key: dict[str, int] = {}
    for company_index, company in enumerate(list(companies)[: max(0, int(limit))]):
        identity = canonical_company_identity(company)
        duplicate_of = first_by_key.get(identity.key)
        duplicate = bool(
            identity.registrable_domain
            and identity.normalized_name
            and duplicate_of is not None
        )
        if not duplicate and identity.registrable_domain and identity.normalized_name:
            first_by_key[identity.key] = company_index
        decisions.append(
            ArenaCompanyIdentityDecision(
                company_index=company_index,
                identity=identity,
                duplicate_company=duplicate,
                duplicate_of_index=duplicate_of if duplicate else None,
            )
        )
    return decisions


def verified_identity_receipt(
    verifier_gate_receipts: Any,
) -> Optional[Mapping[str, Any]]:
    """Extract the independently observed identity receipt from a fit result."""

    if not isinstance(verifier_gate_receipts, Sequence) or isinstance(
        verifier_gate_receipts, (str, bytes, bytearray)
    ):
        return None
    for gate in verifier_gate_receipts:
        if not isinstance(gate, Mapping) or gate.get("gate") != "company_fit":
            continue
        dimension_evidence = gate.get("dimension_evidence")
        identity = (
            dimension_evidence.get("identity")
            if isinstance(dimension_evidence, Mapping)
            else None
        )
        if not isinstance(identity, Mapping):
            continue
        receipt = identity.get("web_identity_receipt")
        if (
            isinstance(receipt, Mapping)
            and receipt.get("decision") == "match"
            and receipt.get("evidence_source") in _VERIFIED_IDENTITY_SOURCES
        ):
            return receipt
    return None


def company_fit_verified(receipts_or_breakdown: Any) -> bool:
    """Return whether the independent company-fit gate reached ``match``.

    This is intentionally narrower than ``company_qualified``: a company can
    have verified identity and fit while its primary intent evidence fails.
    Retry-wide duplicate state must still reserve that verified identity.
    """

    receipts = (
        receipts_or_breakdown.get("verifier_gate_receipts")
        if isinstance(receipts_or_breakdown, Mapping)
        else receipts_or_breakdown
    )
    return bool(
        isinstance(receipts, Sequence)
        and not isinstance(receipts, (str, bytes, bytearray))
        and any(
            isinstance(receipt, Mapping)
            and receipt.get("gate") == "company_fit"
            and receipt.get("decision") == "match"
            for receipt in receipts
        )
    )


def source_dates_from_verdict(
    verdict: Mapping[str, Any], publication_dates: Sequence[Any] = ()
) -> tuple[Optional[str], list[str]]:
    """Read bounded date tags emitted by the Arena-only Stage-3 prompt."""

    event_dates: list[str] = []
    grounded_publications: list[str] = []
    publications: list[str] = []
    for value in publication_dates:
        text = str(value or "").strip()
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            continue
        canonical = parsed.isoformat()
        if canonical not in grounded_publications:
            grounded_publications.append(canonical)
    for note in verdict.get("risk_notes") or []:
        match = _DATE_NOTE_RE.fullmatch(str(note or "").strip())
        if match is None:
            continue
        try:
            canonical = date.fromisoformat(match.group("value")).isoformat()
        except ValueError:
            continue
        if match.group("kind") == "source_event_date":
            if canonical not in event_dates:
                event_dates.append(canonical)
        elif (
            canonical in grounded_publications
            and canonical not in publications
        ):
            publications.append(canonical)
    if len(event_dates) > 1:
        # Conflicting event dates cannot be repaired by choosing a newer page
        # publication date. Preserve uncertainty at the independent date gate.
        return None, []
    return (event_dates[0] if event_dates else None), publications


def source_grounded_date_verdict(
    *,
    event_date: Optional[str],
    publication_dates: Sequence[str],
    buyer_cap_days: int,
    evaluated_on: date,
) -> SourceDateVerdict:
    """Reject only a single, source-grounded date clearly outside the window.

    An explicit event date is authoritative over page publication metadata, so
    republishing an old event cannot make it fresh.  Conflicting or absent dates
    remain uncertain and therefore do not create a false negative.
    """

    candidates: list[tuple[str, str]] = []
    if event_date:
        candidates.append(("event_date", event_date))
    else:
        unique_publications = list(dict.fromkeys(publication_dates))
        if len(unique_publications) == 1:
            candidates.append(("publication_date", unique_publications[0]))
    if len(candidates) != 1:
        return SourceDateVerdict("uncertain", None, "missing_or_conflicting", None)
    basis, raw = candidates[0]
    try:
        grounded = date.fromisoformat(raw)
    except (TypeError, ValueError):
        return SourceDateVerdict("uncertain", None, "invalid_source_date", None)
    age_days = (evaluated_on - grounded).days
    if age_days < 0:
        return SourceDateVerdict("uncertain", grounded.isoformat(), basis, age_days)
    if age_days > max(1, int(buyer_cap_days)):
        return SourceDateVerdict(
            "out_of_window", grounded.isoformat(), basis, age_days
        )
    return SourceDateVerdict("in_window", grounded.isoformat(), basis, age_days)
