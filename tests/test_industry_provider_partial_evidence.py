"""Industry transport faults must not erase checked facts from other targets."""
import asyncio
from copy import deepcopy

import aiohttp
import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import _attribute_finding, _finding
from tests.test_company_evidence_refusal_fallback import URL, SUBSCRIPTION, IDENTITY
from tests.test_native_industry_review import install


def native_result(monkeypatch, fault, **kwargs):
    run, calls, _, reviews = install(monkeypatch, finding=[
        _finding('industry'), _attribute_finding('VERIFIED', url=URL, quote=SUBSCRIPTION),
    ])
    original_entry = c.investigate_company_evidence

    async def entry(**kwargs):
        return await original_entry(**dict(kwargs, targets=('industry', 'required_attribute')))

    monkeypatch.setattr(c, 'investigate_company_evidence', entry)
    original_fetch = c._fetch_page

    async def fetch(*args, **kwargs):
        result = await original_fetch(*args, **kwargs)
        if result.get('ok'):
            result = {**result, 'text': result['text'] + ' ' + SUBSCRIPTION}
        return result

    monkeypatch.setattr(c, '_fetch_page', fetch)
    original_post = c._post_json
    attempts = []

    async def post(session, url, **kwargs):
        if 'response_format' in kwargs['payload']:
            attempts.append(kwargs['payload'])
            if fault == 'http':
                return 400, {'error': {'code': 'offline_fixture'}}
            if fault == 'provider_body':
                return 200, {'error': {'code': 'offline_fixture'}}
            if fault == 'timeout':
                raise asyncio.TimeoutError()
            if fault == 'connection':
                raise aiohttp.ClientConnectionError('offline fixture')
            if fault == 'cancel':
                raise asyncio.CancelledError()
            if fault == 'contract':
                raise TypeError('offline contract fixture')
        return await original_post(session, url, **kwargs)

    monkeypatch.setattr(c, '_post_json', post)
    return asyncio.run(run(**kwargs)), calls, reviews, attempts


@pytest.mark.parametrize('fault', ['http', 'provider_body', 'timeout', 'connection', 'healthy'])
def test_final_industry_fault_preserves_other_validated_claims_and_exact_sources(monkeypatch, fault):
    result, calls, reviews, attempts = native_result(monkeypatch, fault)
    assert result['claims']['required_attribute']['status'] == 'VERIFIED'
    assert result['claims']['required_attribute']['evidence_quote'] == SUBSCRIPTION
    pages = result[c.PRIVATE_FETCHED_PAGES_KEY]
    assert len(pages) == len(calls['primary']) == 2
    assert all(page['text'].endswith(SUBSCRIPTION) for page in pages.values())
    checked, finals = c._validated_returned_fetched_pages(pages, submitted_source_urls=tuple(pages))
    assert checked == {url: page['text'] for url, page in pages.items()}
    assert finals == {url: page['final_url'] for url, page in pages.items()}
    assert len(attempts) == result['usage']['industry_review_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    industry = result['claims']['industry']
    assert industry['_industry_review']['packet_sha256']
    assert set(industry['_industry_review']['source_urls']) == set(pages)
    if fault == 'healthy':
        assert result['_completed_submit'] is True
        assert result['failure_reason'] == ''
        assert industry['status'] == 'VERIFIED'
        assert len(reviews) == 1
    else:
        assert result['_completed_submit'] is False
        assert result['failure_reason'] == c.PROVIDER_ERROR_FAILURE_REASON
        assert result['_checked_source_provenance'] == {}
        assert industry['status'] == 'UNPROVEN'
        assert industry['evidence_url'] == industry['evidence_quote'] == ''
        assert industry['supporting_evidence'] == []
        assert industry['_industry_review']['decision'] == 'unavailable'
        assert industry['_industry_review']['evidence'] == []
        projected = lead_scorer._project_investigator_industry({
            'industry_matches': True, 'industry_evidence_url': URL,
            'industry_evidence_quote': SUBSCRIPTION, 'attribute_satisfied': True,
        }, industry)
        assert projected['industry_matches'] is None
        assert projected['industry_evidence_url'] == projected['industry_evidence_quote'] == ''
        assert projected['attribute_satisfied'] is True


def test_industry_transport_cancellation_propagates(monkeypatch):
    with pytest.raises(asyncio.CancelledError):
        native_result(monkeypatch, 'cancel')


def test_industry_contract_bug_keeps_existing_malformed_response_handling(monkeypatch):
    result, _, _, _ = native_result(monkeypatch, 'contract')
    assert result['claims'] == {}
    assert result['failure_reason'] == c.MALFORMED_RESPONSE_FAILURE_REASON


@pytest.mark.parametrize('fault', ['http', 'timeout'])
def test_initial_advisory_provider_failure_still_stops_incomplete(monkeypatch, fault):
    result, calls, _, attempts = native_result(monkeypatch, fault,
        prefetched_pages={URL: {'final_url': URL, 'text': SUBSCRIPTION}},
        prior_observations={'submitted_source_urls': [URL]})
    assert result['claims'] == {}
    assert result['failure_reason'] == c.PROVIDER_ERROR_FAILURE_REASON
    assert result.get('_completed_submit') is not True
    assert len(attempts) == 1
    assert not calls['model'] and not calls['primary'] and not calls['search']


@pytest.mark.parametrize('fault', ['http', 'timeout'])
def test_failed_final_review_remains_incomplete_through_caller_and_source_retry(monkeypatch, fault):
    from tests.test_matched_company_retry_source_continuity import _company, _icp, _verdict

    investigation, _, _, _ = native_result(monkeypatch, fault)
    company = _company().model_copy(update={
        'company_name': 'Coreworks', 'company_website': 'https://www.coreworks.ai/',
        'company_linkedin': 'https://www.linkedin.com/company/coreworks-ai',
    })
    verdict = {**_verdict(), 'observed_company_name': 'Coreworks',
               'observed_company_website': company.company_website,
               'observed_company_linkedin': company.company_linkedin}
    cache, seen = {}, []

    async def investigated(**kwargs):
        seen.append(kwargs)
        return deepcopy(investigation)

    monkeypatch.setattr(lead_scorer, 'investigate_company_evidence', investigated)

    async def invoke():
        return await lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=_icp(), verdict=verdict, investigation_targets=('industry',),
            icp_attribute=_icp().required_attribute, icp_stage='', verified_identity=IDENTITY,
            verified_transport_domain='coreworks.ai', structured_employee_size_evidence=None,
            structured_public_company_evidence=None, employee_size_conflict=False,
            company_quality=True, matched_company_source_sink={},
            matched_company_retry_source_cache=cache,
        )

    projected, result, claims, _, _ = asyncio.run(invoke())
    assert projected['industry_matches'] is None
    assert result.decision == 'unavailable'
    assert result.details[lead_scorer.VERIFIER_FAILURE_DETAIL_KEY] == 'provider_error'
    receipt = result.details['investigation_receipt']
    assert receipt['completed_submitted_findings'] is False
    assert receipt['failure_reason'] == 'provider_error'
    assert claims['required_attribute']['status'] == 'VERIFIED'
    assert lead_scorer._settle_completed_company_investigation(
        result, projected, investigation_targets=('industry',), icp=_icp(),
        icp_attribute=_icp().required_attribute, icp_stage='', current_profile_cache={},
    ) is None
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert retained == investigation[c.PRIVATE_FETCHED_PAGES_KEY]
    for key in IDENTITY:
        assert lead_scorer._matched_company_retry_prefetched_pages(
            cache, {**IDENTITY, key: 'another-company'},
        ) == {}
    assert set(cache) == {'verified_identity', 'pages'}  # Source facts, no verdict or success flag.
    asyncio.run(invoke())
    assert seen[-1]['prefetched_pages'] == retained
