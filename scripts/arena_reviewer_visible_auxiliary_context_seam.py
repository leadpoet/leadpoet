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


# The submitted-attribute auxiliary uses the same unchanged visible-body
# validator. This independent one-file correction preserves the d657 runtime.
SUBMITTED_CORRECTION_ID = "visible_submitted_company_context_v1"
SUBMITTED_FROZEN_IMAGE_DIGEST = "sha256:d6574b368e228b08f2136322e4eaf52f5d8bc9ab7b02f979d0195e60e51a8bc0"
SUBMITTED_ORIGINAL_SHA256 = "f783730a48ea0196fce6983dfa75f42b068662219bfb965cd216ec5c0efc8c3a"
SUBMITTED_REPAIRED_SHA256 = "d1dbdd48d89ec52218c0e82270771f88924d40b4a4216fab98ba444a174cd08c"
_SUBMITTED_BEFORE = '''    submitted_name = (
        _compact_company_name(company.company_name)
        if company is not None else ""
    )
    identity = verified_identity_receipt([company_fit.receipt("company_fit")])
    source_entry = (
        matched_company_source_cache.get(submitted_url)
        if isinstance(matched_company_source_cache, Mapping)
        and len(matched_company_source_cache)
        <= _max_retained_investigator_contexts()
        else None
    )
    if (
        company is not None
        and company_fit.decision == COMPANY_FIT_MATCH
        and submitted_name
        and isinstance(identity, Mapping)
        and submitted_url.startswith("https://")
        and _registrable_domain(submitted_url)
        == str(identity.get("observed_domain") or "").casefold()
        and isinstance(submitted_quote, str)
        and isinstance(source_entry, Mapping)
        and set(source_entry) == {"final_url", "text"}
        and isinstance(source_entry["final_url"], str)
        and source_entry["final_url"].startswith("https://")
        and _valid_web_evidence_url(source_entry["final_url"])
        == source_entry["final_url"]
        and _registrable_domain(source_entry.get("final_url"))
        == _registrable_domain(submitted_url)
        and isinstance(source_entry.get("text"), str)
        and 0 < len(source_entry["text"]) <= MAX_PAGE_CHARACTERS
        and _quote_occurs(submitted_quote, source_entry["text"])
        and submitted_name in _compact_company_name(source_entry["text"])
    ):
        contexts.append({
            "dimension": "first_party_company",
            "url": submitted_url,
            "text": source_entry["text"],
        })
'''
_SUBMITTED_AFTER = '''    identity = verified_identity_receipt([company_fit.receipt("company_fit")])
    source_entry = (
        matched_company_source_cache.get(submitted_url)
        if isinstance(matched_company_source_cache, Mapping)
        and len(matched_company_source_cache)
        <= _max_retained_investigator_contexts()
        else None
    )
    # Use the paragraph consumer's visible body and identity binding. Locator
    # metadata can contain the company name but cannot admit an optional page.
    submitted_context = (
        _validated_first_party_source_context(
            submitted_url, source_entry, identity,
            company_name=company.company_name,
        )
        if company is not None else None
    )
    if (
        company is not None
        and company_fit.decision == COMPANY_FIT_MATCH
        and submitted_context is not None
        and isinstance(submitted_quote, str)
        and _quote_occurs(submitted_quote, submitted_context["text"])
    ):
        contexts.append({
            "dimension": "first_party_company",
            "url": submitted_url,
            "text": submitted_context["text"],
        })
'''


def convert_submitted_company_context_seam(source: bytes) -> bytes:
    """Correct only the exact frozen submitted-context producer."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == SUBMITTED_REPAIRED_SHA256:
        return source
    if digest != SUBMITTED_ORIGINAL_SHA256:
        raise VisibleAuxiliaryContextSeamError("unrecognized submitted company context source")
    text = source.decode("utf-8")
    if text.count(_SUBMITTED_BEFORE) != 1:
        raise VisibleAuxiliaryContextSeamError("frozen submitted company context differs")
    result = text.replace(_SUBMITTED_BEFORE, _SUBMITTED_AFTER, 1).encode("utf-8")
    compile(result, "frozen_submitted_company_context.py", "exec")
    if hashlib.sha256(result).hexdigest() != SUBMITTED_REPAIRED_SHA256:
        raise VisibleAuxiliaryContextSeamError("submitted company context postimage differs")
    return result
