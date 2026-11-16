"""Apply only PR1249's public-buyer evidence correction to verified frozen source."""
import hashlib

CORRECTION_ID = "public_issuer_acquisition_target_binding_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:2ec6357b4517b0af50cbece24a2d130afd5f8352e548b0a77f0fa02b33ef0c63"
ORIGINAL_SHA256 = "f1874c0f8554a92d955067c7833569eb285ca63b425359df3d0c679c97e6b49c"
REPAIRED_SHA256 = "15275b9a300a24ee6026f89df39597228a6573dfec97ccda2d56c412bd1a4d82"
HEADQUARTERS_FILE = "qualification/scoring/company_evidence_investigator.py"
HEADQUARTERS_CORRECTION_ID = "bound_linkedin_headquarters_feedback_v1"
HEADQUARTERS_ORIGINAL_SHA256 = "ec74bdfb2cc4b657afc301cb183c799985915b25a8151ea056786c933c6113c5"
HEADQUARTERS_REPAIRED_SHA256 = "36ba690c61c12fa55eb4a679c7fe27b14141eaee1474c9792a7114b896d7f8d5"

_OLD_0 = r'''def _stage_quote_supports_observation(observed: str, quote: str) -> bool:
'''
_NEW_0 = r'''def _stage_quote_supports_observation(
    observed: str, quote: str, *, identity_names: Sequence[str] = (),
) -> bool:
'''

_OLD_1 = r'''        reject_historical=True,
    )
'''
_NEW_1 = r'''        reject_historical=True,
    )
    if public and acquired and identity_names:
        # Ignore only a completed acquisition of an explicitly different named
        # target. Keep passive/parent/ambiguous claims and all listing guards.
        acquisition_text = text
        for match in reversed(list(_BOUND_ACQUISITION_SUBJECT_PATTERNS[3].finditer(text))):
            target = match.group("subject")
            acquisition_clause = _compact_company_name(
                re.split(r"[.!?;\n]", text[match.start():], maxsplit=1)[0]
            )
            if (
                not re.match(r"(?:the|its|our|their|this|that|it|us|them)\b", target, re.I)
                and _compact_company_name(target) not in {
                    "company", "issuer", "registrant", "business", "target",
                }
                and _acquired_stage_quote_supports_names((target,), match.group(0))
                and not any(
                    _compact_company_name(name) in acquisition_clause
                    for name in identity_names if _compact_company_name(name)
                )
            ):
                acquisition_text = (
                    acquisition_text[:match.start()] + " " * (match.end() - match.start())
                    + acquisition_text[match.end():]
                )
        acquired = _has_affirmed_stage_proof(
            acquisition_text, _ACQUIRED_STAGE_PROOF_PATTERNS, reject_historical=True,
        )
'''

_OLD_2 = r'''    lexical_support = _stage_quote_supports_observation(normalized_stage, quote)
'''
_NEW_2 = r'''    lexical_support = _stage_quote_supports_observation(
        normalized_stage, quote, identity_names=identity_names,
    )
'''

DELTAS = ((_OLD_0, _NEW_0), (_OLD_1, _NEW_1), (_OLD_2, _NEW_2),)

class PublicBuyerSeamError(ValueError):
    """The source differs from the independently verified grading file."""


_HEADQUARTERS_DELTAS = (
    ('MISSING_INDUSTRY_OBSERVATION_REASON = (\n    "industry finding requires a nonempty source-grounded observed_industry; "\n    "the exact quote and company binding were not rejected"\n)\nREQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON = (\n    "supporting quote did not identify the investigated company"\n',
     'MISSING_INDUSTRY_OBSERVATION_REASON = (\n    "industry finding requires a nonempty source-grounded observed_industry; "\n    "the exact quote and company binding were not rejected"\n)\nBOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON = (\n    "Bound LinkedIn headquarters evidence must quote its exact About "\n    "Headquarters field and prove the observed country and required state"\n)\nREQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON = (\n    "supporting quote did not identify the investigated company"\n'),
    ('                    evidence_url="",\n                    evidence_quote="",\n                    reason=(\n                        "headquarters evidence must use independently bound "\n                        "first-party requested and fetched final URLs"\n                        if target == "geography" else\n                        "activity and headquarters evidence must use an "\n',
     '                    evidence_url="",\n                    evidence_quote="",\n                    reason=(\n                        BOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON\n                        if (\n                            target == "geography"\n                            and _bound_linkedin_headquarters_source(\n                                evidence_url,\n                                str((fetched_final_urls or {}).get(evidence_url) or ""),\n                                fetched_text, identity_anchor or {},\n                            )\n                        ) else "headquarters evidence must use independently bound "\n                        "first-party requested and fetched final URLs"\n                        if target == "geography" else\n                        "activity and headquarters evidence must use an "\n'),
    ('                                source_url=source_url,\n                                source_context=source_context,\n                            )\n                        if target in {"industry", "required_attribute", "headcount"} and reason in {\n                            "source quote did not identify the investigated company",\n                            REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,\n',
     '                                source_url=source_url,\n                                source_context=source_context,\n                            )\n                        if reason == BOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON:\n                            # The quote excerpt can omit the actual About field.\n                            # Show the existing validator\'s exact proof, not a\n                            # generic page prefix or an inferred location.\n                            headquarters_field = _linkedin_about_headquarters_field(\n                                fetched_pages.get(source_url, ""),\n                            )\n                            rejected_finding["headquarters_proof"] = {\n                                "about_headquarters_field": headquarters_field,\n                                "submitted_country": submitted.get("observed_country", ""),\n                                "submitted_state": submitted.get("observed_state", ""),\n                                "field_supports_submitted_location": bool(\n                                    headquarters_field and _quote_supports_headquarters(\n                                        headquarters_field,\n                                        observed_country=submitted.get("observed_country", ""),\n                                        observed_state=submitted.get("observed_state", ""),\n                                    )\n                                ),\n                            }\n                        if target in {"industry", "required_attribute", "headcount"} and reason in {\n                            "source quote did not identify the investigated company",\n                            REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,\n'),
    ('                                    "missing fields do not establish that the fact is absent. "\n                                    if any(\n                                        item.get("reason") == MISSING_CITATION_REASON\n                                        for item in rejected\n                                    )\n                                    else ""\n',
     '                                    "missing fields do not establish that the fact is absent. "\n                                    if any(\n                                        item.get("reason") == MISSING_CITATION_REASON\n                                        for item in rejected\n                                    )\n                                    else ""\n                                )\n                                + (\n                                    "The LinkedIn profile identity passed, but its submitted "\n                                    "headquarters proof did not. Quote its exact About "\n                                    "Headquarters field only if that field proves the observed "\n                                    "country and required state. Do not infer the country from a city "\n                                    "or substitute generic office locations or mailing addresses. "\n                                    "headquarters_proof contains the exact loaded field and whether "\n                                    "it supports your submitted location. A false result means that "\n                                    "field cannot establish that location; repeating it or combining "\n                                    "it with generic locations does not repair the proof. Otherwise use qualifying "\n                                    "already fetched first-party headquarters evidence, or submit "\n                                    "explicit UNPROVEN with empty citations when the bounded research "\n                                    "does not establish the location. Keep unrelated validated findings. "\n                                    if any(\n                                        item.get("reason") == BOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON\n                                        for item in rejected\n                                    )\n                                    else ""\n'),
)


def convert_headquarters_feedback(source: bytes) -> bytes:
    """Retain the exact PR1252 composition even after prospective code changes."""
    if hashlib.sha256(source).hexdigest() == HEADQUARTERS_REPAIRED_SHA256:
        return source
    if hashlib.sha256(source).hexdigest() != HEADQUARTERS_ORIGINAL_SHA256:
        raise PublicBuyerSeamError("unrecognized headquarters feedback source")
    text = source.decode("utf-8")
    for old, new in _HEADQUARTERS_DELTAS:
        if text.count(old) != 1:
            raise PublicBuyerSeamError("headquarters feedback anchor differs")
        text = text.replace(old, new, 1)
    corrected = text.encode("utf-8")
    return verified_headquarters_feedback(source, corrected)


def verified_headquarters_feedback(source: bytes, reviewed_source: bytes) -> bytes:
    """Accept only the reviewed before/after pair; never copy other source drift."""
    if hashlib.sha256(source).hexdigest() != HEADQUARTERS_ORIGINAL_SHA256:
        raise PublicBuyerSeamError("unrecognized headquarters feedback source")
    if hashlib.sha256(reviewed_source).hexdigest() != HEADQUARTERS_REPAIRED_SHA256:
        raise PublicBuyerSeamError("headquarters feedback differs from reviewed source")
    compile(reviewed_source, HEADQUARTERS_FILE, "exec")
    return reviewed_source


def convert_public_buyer_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise PublicBuyerSeamError("unrecognized or partial public buyer source")
    text = source.decode("utf-8")
    for old, new in DELTAS:
        if text.count(old) != 1:
            raise PublicBuyerSeamError("public buyer correction anchor differs")
        text = text.replace(old, new, 1)
    result = text.encode("utf-8")
    compile(result, "frozen_public_buyer.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise PublicBuyerSeamError("public buyer correction differs from reviewed source")
    return result
