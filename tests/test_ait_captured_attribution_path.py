"""Replay captured AIT inputs; fresh recovery evidence is labelled separately."""
import asyncio
from copy import deepcopy
from datetime import date
import json

import pytest

from qualification.scoring import company_evidence_investigator as c, industry_classification as industry
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_industry_attribution_recovery import CAPTURE, DOMAIN


ORIGINAL = CAPTURE['original_capture']
INPUT = ORIGINAL['investigator_input']
SOURCE = ORIGINAL['reviewer_source']
ALTERNATIVE = CAPTURE['fresh_alternative_source']


def _captured_anchor():
    locator, observed = INPUT['company_locator'], INPUT['prior_observations']
    assert INPUT['verified_homepage_identity'] == {}
    return {'submitted_name': locator['name'], 'submitted_domain': DOMAIN,
            'submitted_linkedin_slug': 'ait-worldwide-logistics',
            'observed_name': observed['observed_company_name'], 'observed_domain': DOMAIN,
            'observed_linkedin_slug': 'ait-worldwide-logistics',
            'verified_name': None, 'verified_domain': '', 'verified_linkedin_slug': ''}


def test_original_packet_and_empty_verified_anchor_are_exactly_reconstructed():
    metadata = ORIGINAL['reviewer_metadata']
    packet = industry.prepare(metadata['criterion'], metadata['company_locator'], [SOURCE],
                              evaluation_date=metadata['evaluation_date'])
    assert packet.digest == ORIGINAL['packet_sha256'] == '601a5e0a9535e643f04a7e6b3478182dafc20f8350d737a3cdafc1b16ccbe8f6'
    assert SOURCE['body'] == c._visible_quote_surface(INPUT['prefetched_sources'][0]['text']).strip()
    assert 'AIT Worldwide Logistics' not in SOURCE['body']
    assert 'https://www.linkedin.com/company/24870' in INPUT['prefetched_sources'][0]['text']
    assert not c._complete_verified_first_party_identity(SOURCE['source_url'], {DOMAIN}, _captured_anchor())
    raw = {**ORIGINAL['final_claims']['industry'], 'status': 'VERIFIED',
           'evidence_url': SOURCE['source_url'], 'evidence_quote': CAPTURE['evidence'][0]['quote']}
    checked = c._validated_findings({'findings': [raw]}, targets=('industry',),
        fetched_pages={SOURCE['source_url']: SOURCE['body']},
        # This actual rejection does not depend on an assumed fetched-final URL.
        fetched_final_urls={}, first_party_domains={DOMAIN},
        identity_names={'aitworldwidelogistics'}, identity_anchor=_captured_anchor())['industry']
    assert checked['status'] == 'UNPROVEN'
    assert checked['reason'] == ORIGINAL['final_claims']['industry']['reason']


def install_captured_run(monkeypatch, *, alternative=False):
    searches, reviews, controllers, investigations = [], [], [], []
    profile = ORIGINAL['linkedin_tool_result']
    original_investigate = c.investigate_company_evidence

    async def search(_session, query, **options):
        searches.append((query, options))
        # Original-corpus control returns no new source; the alternative is a
        # separately captured current official page, never an original response.
        sources = ({ALTERNATIVE['requested_url']: ALTERNATIVE['text']}
                   if alternative and options.get('read_source_text') else {})
        return {'results': [], '_source_pages': sources, 'notice': 'discovery_only_not_evidence'}

    async def fetch(_session, url, **kwargs):
        assert url == profile['url']
        return {key: value for key, value in profile.items() if key != 'remaining_research_budget'}

    async def post(_session, _url, *, payload, **kwargs):
        if 'response_format' in payload:
            reviews.append(payload)
            spans = [s for message in payload['messages'][2:]
                     for page in json.loads(message['content'])['untrusted_source_pages']
                     for s in page['spans']]
            captured = CAPTURE['reason']
            response = {'observed_activity': 'North American truckload brokerage and logistics',
                        'reason': captured, 'missing_requirements': [], 'role': 'supplier_operator',
                        'evidence_span_ids': [s['span_id'] for s in spans], 'decision': 'match'}
            if 'research_query' in payload['response_format']['json_schema']['schema']['properties']:
                response['research_query'] = ''
            return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response)}}]}
        controllers.append(payload)
        if len(controllers) == 1:
            return 200, _response('fetch_page', {'url': profile['url']})
        if payload.get('tool_choice') == {'type': 'function', 'function': {'name': 'search_web'}}:
            return 200, _response('search_web', {'query': 'AIT Worldwide Logistics own freight brokerage identity'})
        findings = []
        for finding in ORIGINAL['final_claims'].values():
            raw = deepcopy(finding)
            for index, support in enumerate(raw.pop('supporting_evidence', []), 1):
                raw[f'supporting_evidence_url_{index}'] = support['url']
                raw[f'supporting_evidence_quote_{index}'] = support['quote']
            findings.append(raw)
        return 200, _response('submit_findings', {'findings': findings})

    async def investigate(**kwargs):
        investigations.append(deepcopy(kwargs))
        return await original_investigate(**kwargs)

    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-key')
    monkeypatch.setenv('EXA_API_KEY', 'offline-key')
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'offline-captured-review')
    monkeypatch.setattr(c, 'evaluation_date', lambda: date(2026, 11, 9))
    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_fetch_page', fetch)

    def arguments():
        return dict(company_locator=INPUT['company_locator'], targets=tuple(INPUT['requested_targets']),
            requested_industry=INPUT['requested_industry'], requested_subindustry=INPUT['requested_subindustry'],
            requested_product_service=INPUT['requested_product_service'], requested_attribute=INPUT['requested_attribute'],
            requested_employee_buckets=INPUT['requested_employee_buckets'],
            prior_observations=INPUT['prior_observations'], verified_homepage_identity={},
            prefetched_pages={p['url']: {'text': p['text'], 'final_url': p['url']} for p in INPUT['prefetched_sources']},
            positive_semantic_review=INPUT['positive_semantic_review'])

    return investigate, arguments, searches, reviews, controllers, investigations


@pytest.mark.parametrize('alternative', [False, True])
def test_captured_three_target_path_preserves_original_uncertainty_or_recovers_named_evidence(monkeypatch, alternative):
    investigate, arguments, searches, reviews, _, investigations = install_captured_run(monkeypatch, alternative=alternative)
    result = asyncio.run(investigate(**arguments()))
    assert investigations[0]['verified_homepage_identity'] == {}
    assert result['_completed_submit'] is True
    assert result['failure_reason'] == ''
    assert result['claims']['headcount']['status'] == 'VERIFIED'
    assert result['claims']['required_attribute']['status'] == 'VERIFIED'
    assert result['claims']['industry']['status'] == ('VERIFIED' if alternative else 'UNPROVEN')
    assert len(searches) == result['usage']['search_calls'] == 1
    assert searches[0][1]['include_domains'] == (DOMAIN,)
    assert searches[0][1]['read_source_text'] is True
    assert result['usage']['fetch_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert len(reviews) == 2
    if alternative:
        assert result['claims']['industry']['evidence_url'] == ALTERNATIVE['requested_url']
    else:
        assert result['claims']['industry']['reason'] == ORIGINAL['final_claims']['industry']['reason']


@pytest.mark.parametrize('company_quality', [False, True], ids=['standard-fit-policy', 'company-quality-policy'])
@pytest.mark.parametrize('alternative', [False, True])
def test_shared_company_scorer_runs_captured_attribution_through_final_fit_receipt(monkeypatch, company_quality, alternative):
    from gateway.qualification.models import CompanyOutput
    from qualification.scoring import lead_scorer
    from qualification.scoring.company_fit_decision import company_fit_unavailable
    from tests.test_company_evidence_investigator import _icp, _complete_verdict
    from lab_arena.contracts import check_strict_document

    investigate, arguments, searches, _, _, investigations = install_captured_run(monkeypatch, alternative=alternative)
    verdict = _complete_verdict(**INPUT['prior_observations'], attribute_satisfied=True,
                                employee_size_matches=None, reason='Captured identity and source observations')

    async def provider(**kwargs):
        return deepcopy(verdict), ''

    async def keep_size(current, *args, **kwargs):
        return current

    # All source-review, legacy validation, projection and fit logic is real.
    # The already captured official body is hydrated at the existing source
    # transport boundary. Restore captured source hints that the saved row
    # omits; the original fetched-final URL is not claimed known.
    async def captured_investigation(**kwargs):
        assert kwargs['verified_homepage_identity'] == {}
        kwargs['prefetched_pages'] = arguments()['prefetched_pages']
        kwargs['prior_observations'] = {**kwargs['prior_observations'],
            'submitted_source_urls': INPUT['prior_observations']['submitted_source_urls']}
        return await investigate(**kwargs)

    monkeypatch.setattr(lead_scorer, '_request_company_reverify_json', provider)
    monkeypatch.setattr(lead_scorer, '_refresh_linkedin_employee_size_observation', keep_size)
    monkeypatch.setattr(lead_scorer, 'investigate_company_evidence', captured_investigation)
    from qualification.scoring.competition import _normalized_company
    company = CompanyOutput(**_normalized_company(ORIGINAL['submitted_company'], integrity_policy=True))
    icp = _icp(industry=INPUT['requested_industry'], sub_industry=INPUT['requested_subindustry'],
               product_service=INPUT['requested_product_service'], required_attribute=INPUT['requested_attribute'],
               employee_count='1,001-5,000', geography='United States', company_stage='')
    homepage = company_fit_unavailable('Captured incomplete homepage identity', details={
        'identity': CAPTURE['identity'], 'verified_homepage_transport_domain': DOMAIN,
        'actual_final_url': 'https://www.aitworldwide.com/'})
    result = asyncio.run(lead_scorer._llm_reverify_company(company, icp,
        require_company_fit_dimensions=True, evidence_investigator=True, company_quality=company_quality,
        verified_homepage_identity=homepage))
    stored = json.loads(json.dumps(result.receipt('company_fit')))
    check_strict_document(stored)
    assert investigations and all(x['verified_homepage_identity'] == {} for x in investigations)
    assert 'dimension_decisions' in stored, json.dumps(stored, indent=2)
    assert stored['dimension_decisions']['industry'] == ('match' if alternative else 'unavailable')
    assert stored['decision'] == ('match' if alternative else 'unavailable')
    assert stored['dimension_decisions']['employee_size'] == 'match'
    assert stored['dimension_decisions']['geography'] == 'match'
    assert stored['required_attribute_decision'] == 'match'
    assert len(searches) == 1
    if alternative:
        assert stored['dimension_evidence']['industry']['url'] == ALTERNATIVE['requested_url']
    else:
        assert stored['investigation_receipt']['claims']['industry']['reason'] == ORIGINAL['final_claims']['industry']['reason']
