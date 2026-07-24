from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from leadpoet_verifier.contracts import VerificationDecision

from .models import CanonicalCompanyIdentity, StrictIdentityModel
from .normalization import normalize_url


class EvidenceIdentityBinding(StrictIdentityModel):
    normalized_source_url: str = Field(max_length=2048)
    source_host: str = Field(max_length=253)
    binding: Literal[
        "same_entity", "related_entity", "third_party", "ambiguous", "unavailable"
    ]
    matched_domain_role: Optional[str] = Field(default=None, max_length=40)
    identity_rule_ids: list[str] = Field(default_factory=list, max_length=8)
    identity_receipt_id: str = Field(min_length=1, max_length=80)


class CounterfactualIdentityComparison(StrictIdentityModel):
    baseline_outcome: Literal["accepted", "rejected", "unavailable"]
    identity_bound_outcome: Literal["accepted", "rejected", "unavailable"]
    reason_code: str = Field(min_length=1, max_length=120)
    claim_score_unchanged: bool
    evidence_bindings: list[EvidenceIdentityBinding] = Field(max_length=5)


def bind_evidence_url(
    raw_url: str,
    identity: CanonicalCompanyIdentity,
    *,
    identity_receipt_id: str,
) -> EvidenceIdentityBinding:
    try:
        source = normalize_url(raw_url)
    except ValueError:
        return EvidenceIdentityBinding(
            normalized_source_url=raw_url[:2048],
            source_host="invalid",
            binding="ambiguous",
            identity_receipt_id=identity_receipt_id,
        )
    for domain in identity.domains:
        if domain.ascii_host != source.ascii_host:
            continue
        if domain.relation_to_identity == "same_entity" and domain.ownership_status == "verified":
            binding = "same_entity"
        elif domain.relation_to_identity == "related_entity":
            binding = "related_entity"
        elif domain.relation_to_identity == "shared_infrastructure":
            binding = "ambiguous"
        else:
            binding = "third_party"
        return EvidenceIdentityBinding(
            normalized_source_url=source.url,
            source_host=source.ascii_host,
            binding=binding,
            matched_domain_role=domain.role,
            identity_rule_ids=identity.positive_rule_ids if binding == "same_entity" else [],
            identity_receipt_id=identity_receipt_id,
        )
    return EvidenceIdentityBinding(
        normalized_source_url=source.url,
        source_host=source.ascii_host,
        binding="third_party",
        identity_receipt_id=identity_receipt_id,
    )


def compare_identity_bound_claim(
    claim: VerificationDecision,
    identity: CanonicalCompanyIdentity,
    evidence_urls: list[str],
    *,
    identity_receipt_id: str,
) -> CounterfactualIdentityComparison:
    bindings = [
        bind_evidence_url(url, identity, identity_receipt_id=identity_receipt_id)
        for url in evidence_urls[:5]
    ]
    identity_eligible = identity.resolution_status == "resolved" and identity.identity_tier in {
        "exact", "strong"
    }
    if not identity_eligible:
        counterfactual = "unavailable" if identity.resolution_status == "unavailable" else "rejected"
        reason = f"identity_{identity.resolution_status}"
    else:
        counterfactual = claim.outcome
        reason = claim.reason_code
    return CounterfactualIdentityComparison(
        baseline_outcome=claim.outcome,
        identity_bound_outcome=counterfactual,
        reason_code=reason,
        # Identity is a conjunctive gate only. It never changes verifier_score.
        claim_score_unchanged=True,
        evidence_bindings=bindings,
    )
