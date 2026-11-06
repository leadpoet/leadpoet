"""Apply only reviewed identity-repair finality to exact frozen 2dc.

The source hash binds all existing scoring code. Only the reviewed 196c1bdbb
changes are applied; the later identity-investigation ordering is not added.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "completed_missing_linkedin_identity_repair_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:2dc2a4e7ac758527d20629a05f4046a3705d92f712c5ae7f28663cf4a55b161f"
ORIGINAL_SHA256 = "f2417700fb3ff72311e0911498573ed27e735a0371cdca970b1cbcf1b926ae93"
REPAIRED_SHA256 = "787d41f8fd35ad90858bab0b4d5c4bbcb456735f468c69f73eb5a68e8c988a74"
# Exact reviewed context blocks, in original file order.
_DELTAS = (
    ('''    )


def _is_verified_homepage_web_identity_conflict(
    receipt: Optional[Mapping[str, Any]],
) -> bool:
''', '''    )


def _is_explicitly_missing_web_linkedin_identity(
    verdict: Mapping[str, Any],
    receipt: Optional[Mapping[str, Any]],
) -> bool:
    """Recognize a typed, same-entity observation without LinkedIn proof."""
    value = receipt or {}
    return bool(
        all(isinstance(verdict.get(field), str) for field in (
            "observed_company_name", "observed_company_website",
            "observed_company_linkedin",
        ))
        and verdict.get("observed_company_linkedin") == ""
        and value.get("decision") == COMPANY_FIT_UNAVAILABLE
        and value.get("reason_code") == "identity_not_proven"
        and value.get("evidence_source") == "company_web_reverification"
        and all(isinstance(value.get(field), str) and bool(value[field].strip())
                for field in ("submitted_name", "submitted_domain", "observed_name",
                              "observed_domain", "submitted_linkedin_slug"))
        and value.get("submitted_name") == value.get("observed_name")
        and value.get("submitted_domain") == value.get("observed_domain")
        and value.get("observed_linkedin_slug") == ""
    )


def _is_verified_homepage_web_identity_conflict(
    receipt: Optional[Mapping[str, Any]],
) -> bool:
'''),
    ('''    linkedin_refresh_outcome: str = "",
    identity_receipt: Optional[Mapping[str, Any]] = None,
    investigation_receipt: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Recognize unavailable proof after one complete repair response."""

''', '''    linkedin_refresh_outcome: str = "",
    identity_receipt: Optional[Mapping[str, Any]] = None,
    investigation_receipt: Optional[Mapping[str, Any]] = None,
    completed_identity_repair: bool = False,
) -> bool:
    """Recognize unavailable proof after one complete repair response."""

'''),
    ('''            continue
        if dimension == "identity":
            if not (
                _is_same_domain_unproven_web_identity(identity_receipt)
                or _is_verified_homepage_web_identity_conflict(identity_receipt)
                or (
                    completed_investigation
''', '''            continue
        if dimension == "identity":
            if not (
                (
                    completed_identity_repair
                    and _is_explicitly_missing_web_linkedin_identity(verdict, identity_receipt)
                    and not review_has_transport_fault
                    and linkedin_refresh_outcome != "retryable_failure"
                    and (investigation_receipt is None or completed_investigation)
                )
                or _is_same_domain_unproven_web_identity(identity_receipt)
                or _is_verified_homepage_web_identity_conflict(identity_receipt)
                or (
                    completed_investigation
'''),
    ('''    icp_attribute: str,
    icp_stage: str,
    current_profile_cache: Mapping[str, Any],
) -> Optional[CompanyFitDecisionResult]:
    """Reuse bounded-investigation finality before semantic early returns.

''', '''    icp_attribute: str,
    icp_stage: str,
    current_profile_cache: Mapping[str, Any],
    completed_identity_repair: bool = False,
) -> Optional[CompanyFitDecisionResult]:
    """Reuse bounded-investigation finality before semantic early returns.

'''),
    ('''         "geography": "geography"}[target]
        for target in investigation_targets
    }
    if not incomplete or not investigation_targets or not set(incomplete).issubset(
        investigated_dimensions
    ):
''', '''         "geography": "geography"}[target]
        for target in investigation_targets
    }
    if completed_identity_repair:
        investigated_dimensions.add("identity")
    if not incomplete or not investigation_targets or not set(incomplete).issubset(
        investigated_dimensions
    ):
'''),
    ('''            verdict,
            incomplete,
            icp=icp,
            linkedin_refresh_outcome=(
                CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE
                if malformed_profile_exhausted or resolved_profile_obligations else refresh_outcome
''', '''            verdict,
            incomplete,
            icp=icp,
            completed_identity_repair=completed_identity_repair,
            linkedin_refresh_outcome=(
                CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE
                if malformed_profile_exhausted or resolved_profile_obligations else refresh_outcome
'''),
    ('''        # may describe the newly observed entity, so reject the whole response
        # instead of mixing them with the prior entity's validated findings.
        return result
    repaired_verdict = _preserve_unrelated_validated_stage_evidence(
        verdict,
        result,
''', '''        # may describe the newly observed entity, so reject the whole response
        # instead of mixing them with the prior entity's validated findings.
        return result
    # The existing repair explicitly requested fresh identity research. A typed
    # same-entity answer that still has no LinkedIn is completed missing proof,
    # never a match. Do not grant this finality to an initial answer or failure.
    completed_identity_repair = bool(
        "identity" in incomplete
        and not repair_error
        and not repair_diagnostic.get(VERIFIER_FAILURE_REASON_KEY)
        and not (
            verified_homepage_identity is not None
            and _homepage_identity_has_retryable_failure(verified_homepage_identity)
        )
        and _is_explicitly_missing_web_linkedin_identity(
            repaired_verdict, repaired_identity_receipt,
        )
    )
    repaired_verdict = _preserve_unrelated_validated_stage_evidence(
        verdict,
        result,
'''),
    ('''                    investigation_targets=post_repair_investigation_targets,
                    icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                    current_profile_cache=current_profile_cache,
                )
                if settled is not None:
                    return settled
''', '''                    investigation_targets=post_repair_investigation_targets,
                    icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                    current_profile_cache=current_profile_cache,
                    completed_identity_repair=completed_identity_repair,
                )
                if settled is not None:
                    return settled
'''),
    ('''                        icp_attribute=icp_attribute, icp_stage=icp_stage,
                    ),
                    icp=icp,
                    linkedin_refresh_outcome=str(
                        current_profile_cache.get("refresh_outcome") or ""
                    ),
''', '''                        icp_attribute=icp_attribute, icp_stage=icp_stage,
                    ),
                    icp=icp,
                    completed_identity_repair=completed_identity_repair,
                    linkedin_refresh_outcome=str(
                        current_profile_cache.get("refresh_outcome") or ""
                    ),
'''),
    ('''                investigation_targets=investigation_targets,
                icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                current_profile_cache=current_profile_cache,
            )
            if settled is not None and settled.details.get("_completed_investigation_settled") is True:
                return settled
''', '''                investigation_targets=investigation_targets,
                icp=icp, icp_attribute=icp_attribute, icp_stage=icp_stage,
                current_profile_cache=current_profile_cache,
                completed_identity_repair=completed_identity_repair,
            )
            if settled is not None and settled.details.get("_completed_investigation_settled") is True:
                return settled
'''),
    ('''                repaired_verdict,
                other_incomplete,
                icp=icp,
                linkedin_refresh_outcome=linkedin_refresh_outcome,
                identity_receipt=(
                    repaired_result.details.get("identity_receipt")
''', '''                repaired_verdict,
                other_incomplete,
                icp=icp,
                completed_identity_repair=completed_identity_repair,
                linkedin_refresh_outcome=linkedin_refresh_outcome,
                identity_receipt=(
                    repaired_result.details.get("identity_receipt")
'''),
    ('''                repaired_verdict,
                other_than_entity_conflicts,
                icp=icp,
                linkedin_refresh_outcome=linkedin_refresh_outcome,
                identity_receipt=(
                    repaired_result.details.get("identity_receipt")
''', '''                repaired_verdict,
                other_than_entity_conflicts,
                icp=icp,
                completed_identity_repair=completed_identity_repair,
                linkedin_refresh_outcome=linkedin_refresh_outcome,
                identity_receipt=(
                    repaired_result.details.get("identity_receipt")
'''),
    ('''            repaired_verdict,
            repaired_incomplete,
            icp=icp,
            linkedin_refresh_outcome=linkedin_refresh_outcome,
            identity_receipt=(
                repaired_result.details.get("identity_receipt")
''', '''            repaired_verdict,
            repaired_incomplete,
            icp=icp,
            completed_identity_repair=completed_identity_repair,
            linkedin_refresh_outcome=linkedin_refresh_outcome,
            identity_receipt=(
                repaired_result.details.get("identity_receipt")
'''),
)


class IdentityRepairFinalitySeamError(ValueError):
    """The input is not the exact reviewed frozen generation."""


def convert_identity_repair_finality_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise IdentityRepairFinalitySeamError("unrecognized or partial frozen identity-repair-finality source")
    text = source.decode("utf-8")
    for before, after in _DELTAS:
        if text.count(before) != 1:
            raise IdentityRepairFinalitySeamError("frozen identity finality context differs")
        text = text.replace(before, after, 1)
    result = text.encode("utf-8")
    compile(result, "frozen_identity_repair_finality.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise IdentityRepairFinalitySeamError("identity repair finality differs from reviewed source")
    return result
