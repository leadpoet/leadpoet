"""Final cited industry support survives the bounded paragraph handoff."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import intent_details, lead_scorer
from tests.test_intent_details_fetched_first_party_context import (
    _case, HOME, HOME_QUOTE, PRODUCT, INVESTOR, FUNDING, PARAGRAPH,
)


JOB = 'https://job-boards.greenhouse.io/ryft/jobs/12345'


def case_with_support(monkeypatch, *, enabled=True, support_quote=PRODUCT):
    if enabled:
        monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    else:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    company, icp, fit, signals = _case()
    # This fixture starts after the native industry reviewer has bound this
    # exact employer posting and cited its exact product statement as support.
    fit.details['dimension_evidence']['industry']['web_evidence']['supporting_evidence'] = [
        {'url': JOB, 'quote': support_quote},
    ]
    pages = {HOME: {'final_url': HOME, 'text': HOME_QUOTE},
             JOB: {'final_url': JOB, 'text': PRODUCT},
             INVESTOR: {'final_url': INVESTOR, 'text': FUNDING}}
    cache = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        cache, fit, {investigator.PRIVATE_FETCHED_PAGES_KEY: pages}, company=company,
    )
    return company, icp, fit, signals, cache


def test_exact_industry_support_reaches_actual_paragraph_document(monkeypatch):
    company, icp, fit, signals, cache = case_with_support(monkeypatch)
    before = deepcopy(fit.details)
    assert JOB in cache
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, cache, paragraph=PARAGRAPH, company=company,
        verified_signal_source_urls=lead_scorer._verified_signal_context_urls(signals),
    )
    assert JOB in {context['url'] for context in contexts}
    assert len(contexts) <= 2
    assert len({context['dimension'] for context in contexts}) == len(contexts)
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt('company_fit'), company_source_contexts=contexts,
    )
    sources = [s for s in document['admitted_evidence']
               if s['evidence_kind'] == 'verified_company_source_context' and s['source_url'] == JOB]
    assert len(sources) == 1 and PRODUCT in sources[0]['admitted_text'][0]
    assert any(source['source_url'] == HOME and HOME_QUOTE in source['admitted_text']
               for source in document['admitted_evidence'])
    assert any(source['source_url'] == JOB and PRODUCT in source['admitted_text']
               for source in document['admitted_evidence'])
    assert fit.details == before


@pytest.mark.parametrize('enabled,quote', [(False, PRODUCT), (True, 'An invented product statement.')])
def test_frozen_policy_or_unbound_support_cannot_supply_paragraph_facts(monkeypatch, enabled, quote):
    company, _, fit, signals, cache = case_with_support(monkeypatch, enabled=enabled, support_quote=quote)
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, cache, paragraph=PARAGRAPH, company=company,
        verified_signal_source_urls=lead_scorer._verified_signal_context_urls(signals),
    )
    assert JOB not in {context['url'] for context in contexts}


@pytest.mark.parametrize('fault', ['uncited_url', 'changed_quote', 'unmatched_industry',
                                  'incomplete_identity', 'frozen_policy'])
def test_paragraph_consumer_rechecks_exact_final_support(monkeypatch, fault):
    company, icp, fit, signals, _ = case_with_support(monkeypatch)
    context = {'dimension': 'industry', 'url': JOB, 'text': PRODUCT}
    if fault == 'uncited_url':
        context['url'] = JOB + '9'
    elif fault == 'changed_quote':
        context['text'] = 'No cited product statement is present.'
    elif fault == 'unmatched_industry':
        fit.details['dimension_evidence']['industry']['decision'] = 'unavailable'
    elif fault == 'incomplete_identity':
        fit.details['dimension_evidence']['identity']['web_identity_receipt']['observed_linkedin_slug'] = ''
    else:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    with pytest.raises(ValueError):
        intent_details.review_evidence(
            company, icp, signals, fit.receipt('company_fit'), company_source_contexts=[context],
        )
