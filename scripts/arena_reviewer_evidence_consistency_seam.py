"""Exact frozen2ec correction; no current prompts, rules, models or host changes."""
import hashlib

from scripts.arena_reviewer_public_buyer_seam import FROZEN_IMAGE_DIGEST

CORRECTIONS = [
    "public_issuer_acquisition_target_binding_v1",
    "bound_linkedin_headquarters_feedback_v1",
    "grounded_word_punctuation_citation_v1",
    "supported_claim_uncertain_date_policy_v1",
    "bound_headquarters_subdivision_country_v1",
]
ORIGINAL_SHA256 = {'qualification/scoring/intent_verification_three_stage.py': 'ab95f6d71e8914f94415c8c00615b6a3c9a76c8de22f73c66727623b485d31df', 'qualification/scoring/company_evidence_investigator.py': 'ec74bdfb2cc4b657afc301cb183c799985915b25a8151ea056786c933c6113c5', 'qualification/scoring/lead_scorer.py': 'f1874c0f8554a92d955067c7833569eb285ca63b425359df3d0c679c97e6b49c'}
REPAIRED_SHA256 = {'qualification/scoring/intent_verification_three_stage.py': '802baac6853e03389d472353d609535d6c339cc1fd5f6a0066ffe1a77cb9ff8b', 'qualification/scoring/company_evidence_investigator.py': '823dffe1fde7ef61d66f2847dca249660ecc3446ebd23246d68f5390f608cc1a', 'qualification/scoring/lead_scorer.py': '24f82207f0ff87e014be56024351e4e2ae87f1612c660591d76bfcafadd60feb'}
FILES = tuple(ORIGINAL_SHA256)

class EvidenceConsistencySeamError(ValueError):
    """Only the independently tested before/after files are admitted."""


def verified_consistency_files(original: dict[str, bytes], reviewed: dict[str, bytes]) -> dict[str, bytes]:
    if set(original) != set(FILES) or set(reviewed) != set(FILES):
        raise EvidenceConsistencySeamError("evidence consistency files differ")
    for name in FILES:
        if hashlib.sha256(original[name]).hexdigest() != ORIGINAL_SHA256[name]:
            raise EvidenceConsistencySeamError("unrecognized frozen evidence source")
        if hashlib.sha256(reviewed[name]).hexdigest() != REPAIRED_SHA256[name]:
            raise EvidenceConsistencySeamError("reviewed evidence correction differs")
        compile(reviewed[name], name, "exec")
    return dict(reviewed)
