"""Private, versioned company-identity resolution contracts.

DEFERRED PRIMITIVES (explicit, per the PR-28 audit): this package ports the
site verifier's identity-resolution building blocks — PSL-pinned host
normalization, float-forbidding canonical hashing, the SSRF-hardened fetcher,
the closed-allowlist resolution policy, observation parsing, and evidence
binding — with their full test suites, but NO lab runtime calls them yet.
The site runs identity resolution as its own queue worker with a Postgres
run-store; the lab equivalent (wiring resolve_identity into
qualification/scoring/company_verification.py behind a mode flag, with its
own durable receipts) is a follow-up integration, not part of this port.
Until then these are library primitives: importable, tested, unwired.
"""

from .models import (
    AnchorSet,
    CanonicalCompanyIdentity,
    CompanyDomain,
    CompanyName,
    EvidenceRef,
    IdentityDecision,
    LinkedInCompanyIdentity,
    ResolverInput,
    WebsiteObservation,
)
from .policy import resolve_identity

__all__ = [
    "AnchorSet",
    "CanonicalCompanyIdentity",
    "CompanyDomain",
    "CompanyName",
    "EvidenceRef",
    "IdentityDecision",
    "LinkedInCompanyIdentity",
    "ResolverInput",
    "WebsiteObservation",
    "resolve_identity",
]
