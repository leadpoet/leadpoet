"""Optional company context must pass the consumer's existing visible surface."""
import copy

import pytest

from qualification.scoring import intent_details, lead_scorer
from qualification.scoring.company_evidence_investigator import _IDENTITY_LINK_CONTEXT_MARKER
from tests.test_intent_details_fetched_first_party_context import (
    ABOUT, HOME, HOME_QUOTE, INVESTOR, PARAGRAPH, PRODUCT, REGULATION,
    _case, _retained_contexts,
)


def test_locator_only_company_name_cannot_break_the_real_paragraph_review():
    visible = "Accept payments online, automate split payments and manage subscriptions."
    raw = visible + _IDENTITY_LINK_CONTEXT_MARKER + " https://ryftpay.com/about-us Ryft"
    pages = {
        HOME: {"final_url": HOME, "text": HOME_QUOTE},
        ABOUT: {"final_url": ABOUT, "text": raw},
    }
    original = copy.deepcopy(pages)
    company, icp, fit, signals, _retained, contexts = _retained_contexts(pages=pages)
    assert not any(context['dimension']=='first_party_context' for context in contexts)
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt('company_fit'), company_source_contexts=contexts,
    )
    admitted = {source.get('source_url') for source in document['admitted_evidence']}
    assert ABOUT not in admitted
    assert HOME in admitted and INVESTOR in admitted
    assert pages == original


def test_locator_terms_cannot_win_the_single_optional_retention_slot():
    company, icp, fit, signals = _case()
    metadata_only = "https://ryftpay.com/a-contact"
    good_visible = "About Ryft. " + PRODUCT + " " + REGULATION
    bad_raw = "About Ryft. Contact us." + _IDENTITY_LINK_CONTEXT_MARKER + PARAGRAPH
    pages = {
        metadata_only: {"final_url": metadata_only, "text": bad_raw},
        ABOUT: {"final_url": ABOUT, "text": good_visible},
        HOME: {"final_url": HOME, "text": HOME_QUOTE},
    }
    company, icp, fit, signals, retained, contexts = _retained_contexts(pages=pages)
    assert ABOUT in retained
    assert metadata_only not in retained
    assert any(context['url']==ABOUT and context['dimension']=='first_party_context'
               for context in contexts)
    intent_details.review_evidence(company, icp, signals, fit.receipt('company_fit'),
                                   company_source_contexts=contexts)


@pytest.mark.parametrize('with_metadata', [False, True])
def test_visible_optional_context_survives_without_modifying_mandatory_text(with_metadata):
    suffix = _IDENTITY_LINK_CONTEXT_MARKER + " https://ryftpay.com/discovery" if with_metadata else ""
    visible = "About Ryft. " + PRODUCT + " " + REGULATION
    mandatory = HOME_QUOTE + suffix
    pages = {
        HOME: {"final_url": HOME, "text": mandatory},
        ABOUT: {"final_url": ABOUT, "text": visible+suffix},
    }
    original = copy.deepcopy(pages)
    company, icp, fit, signals, retained, contexts = _retained_contexts(pages=pages)
    optional = next(context for context in contexts if context['dimension']=='first_party_context')
    assert optional['text'] == visible
    assert next(context for context in contexts if context['dimension']=='industry')['text'] == mandatory
    assert retained[ABOUT]['text'] == pages[ABOUT]['text']
    intent_details.review_evidence(company, icp, signals, fit.receipt('company_fit'),
                                   company_source_contexts=contexts)
    assert pages == original
