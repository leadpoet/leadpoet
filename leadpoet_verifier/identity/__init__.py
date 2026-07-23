"""Private, versioned company-identity resolution contracts."""

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
