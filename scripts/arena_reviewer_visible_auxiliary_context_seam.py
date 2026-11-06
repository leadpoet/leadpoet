"""Use the consumer's visible surface for optional contexts on exact2dc.

This seam follows the reviewed identity-finality correction. No current grader,
consumer, mandatory context or provider policy is imported or replaced.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "visible_optional_company_context_v1"
ORIGINAL_SHA256 = "787d41f8fd35ad90858bab0b4d5c4bbcb456735f468c69f73eb5a68e8c988a74"
REPAIRED_SHA256 = "64696a903629676cf2328768f4bb97525b35dde3e699aab1882fe8b15c894317"
_DELTAS = (
    ('''                prior = source_sink.get(url)
                if isinstance(prior, Mapping):
                    page, final_url = prior.get("text"), prior.get("final_url")
            if _validated_first_party_source_context(
                url, {"final_url": final_url, "text": page}, identity,
                company_name=company.company_name,
            ) is None:
                continue
            phrase_overlap = _source_overlap_score(company.intent_details, page)
            term_overlap = _source_term_overlap(company.intent_details, page)
            if phrase_overlap or term_overlap >= 3:
''', '''                prior = source_sink.get(url)
                if isinstance(prior, Mapping):
                    page, final_url = prior.get("text"), prior.get("final_url")
            context = _validated_first_party_source_context(
                url, {"final_url": final_url, "text": page}, identity,
                company_name=company.company_name,
            )
            if context is None:
                continue
            page = context["text"]
            phrase_overlap = _source_overlap_score(company.intent_details, page)
            term_overlap = _source_term_overlap(company.intent_details, page)
            if phrase_overlap or term_overlap >= 3:
'''),
    ('''        {source_url: entry}, submitted_source_urls=(source_url,),
    )
    text = pages.get(source_url)
    if (
        not name
        or not isinstance(text, str) or name not in _compact_company_name(text)
''', '''        {source_url: entry}, submitted_source_urls=(source_url,),
    )
    text = pages.get(source_url)
    if isinstance(text, str):
        from qualification.scoring.intent_details import _source_evidence_text

        text = _source_evidence_text(text)
    if (
        not name
        or not isinstance(text, str) or name not in _compact_company_name(text)
'''),
)


class VisibleAuxiliaryContextSeamError(ValueError):
    """The source is not the exact reviewed identity-corrected2dc generation."""


def convert_visible_auxiliary_context_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise VisibleAuxiliaryContextSeamError("unrecognized or partial visible auxiliary source")
    text = source.decode("utf-8")
    for before, after in _DELTAS:
        if text.count(before) != 1:
            raise VisibleAuxiliaryContextSeamError("frozen visible auxiliary context differs")
        text = text.replace(before, after, 1)
    result = text.encode("utf-8")
    compile(result, "frozen_visible_auxiliary_context.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise VisibleAuxiliaryContextSeamError("visible auxiliary context postimage differs")
    return result
