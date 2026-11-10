"""Replay retained public evidence; mocked corrections do not prove model accuracy."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _response


DATA = json.loads((Path(__file__).parent / 'fixtures/company_evidence/tvscientific_headcount_quote_repair.json').read_text())
URL, TEXT = DATA['source_url'], DATA['source_text']
NAMES = {'tvscientific'}
LOCATOR = {'name': 'tvScientific', 'website': 'http://tvscientific.com/',
           'linkedin': URL}
IDENTITY = {'submitted_name': 'tvScientific', 'submitted_domain': 'tvscientific.com',
            'submitted_linkedin_slug': 'tvscientificbypinterest',
            'observed_name': 'tvScientific', 'observed_domain': 'tvscientific.com',
            'observed_linkedin_slug': 'tvscientific'}


def validate(finding, text=TEXT):
    return inv._validated_findings(
        {'findings': [finding]}, targets=('headcount',), fetched_pages={URL: text},
        fetched_final_urls={URL: URL}, first_party_domains={'tvscientific.com'},
        identity_names=NAMES, identity_anchor=IDENTITY,
    )['headcount']


def test_actual_rejected_quote_needs_both_named_span_and_non_null_count():
    raw = deepcopy(DATA['rejected_finding'])
    assert validate(raw)['reason'] == 'source quote did not identify the investigated company'
    candidate = inv._company_bound_quote_candidate(raw['evidence_quote'], TEXT, NAMES)
    assert candidate and len(candidate) <= 2000 and inv._quote_occurs(candidate, TEXT)
    raw['evidence_quote'] = candidate
    assert validate(raw)['reason'] == 'source quote did not prove the submitted company-wide headcount'
    raw['observed_value'] = '51-200'
    assert validate(raw)['status'] == 'VERIFIED'


@pytest.mark.parametrize('fault', ['null', 'wrong_count', 'ellipsis', 'invented', 'unnamed', 'missing_source'])
def test_headcount_feedback_keeps_existing_acceptance_checks(fault):
    raw = deepcopy(DATA['rejected_finding'])
    raw.update(observed_value='51-200', evidence_quote=inv._company_bound_quote_candidate(raw['evidence_quote'], TEXT, NAMES))
    text = TEXT
    if fault == 'null': raw['observed_value'] = None
    elif fault == 'wrong_count': raw['observed_value'] = '201-500'
    elif fault == 'ellipsis': raw['evidence_quote'] = 'tvScientific ... Company size 51-200 employees'
    elif fault == 'invented': raw['evidence_quote'] += ' invented'
    elif fault == 'unnamed': raw['evidence_quote'] = DATA['rejected_finding']['evidence_quote']
    else: text = 'Another company. Company size 51-200 employees'
    assert validate(raw, text)['status'] == 'UNPROVEN'


@pytest.mark.parametrize('corrected_status', ['VERIFIED', 'UNPROVEN'])
def test_actual_tool_args_receive_exact_repair_context_without_new_authority(monkeypatch, corrected_status):
    requests = []
    reads = []
    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 3
        if len(requests) == 1:
            return 200, _response('fetch_page', {'url': URL})
        raw = deepcopy(DATA['rejected_finding'])
        if len(requests) == 3:
            feedback = json.loads(payload['messages'][-1]['content'])
            rejection = feedback['rejected_findings'][0]
            assert rejection['reason'] == 'source quote did not identify the investigated company'
            if 'company_bound_quote_candidate' not in rejection:
                pytest.fail('headcount repair feedback lacks its exact named source span')
            raw['evidence_quote'] = rejection['company_bound_quote_candidate']
            assert inv._quote_occurs(raw['evidence_quote'], TEXT)
            assert 'non-null' in feedback['instruction'] and 'observed_value' in feedback['instruction']
            assert 'does not prove the claim' in feedback['instruction']
            raw['observed_value'] = '51-200'
            if corrected_status == 'UNPROVEN':
                raw.update(status='UNPROVEN', evidence_url='', evidence_quote='', observed_value=None)
        return 200, _response('submit_findings', {'findings': [raw]}, call=len(requests))
    async def retained_fetch(_session, url, **_kwargs):
        reads.append(url)
        assert reads == [URL]
        return {'ok': True, 'url': URL, 'text': TEXT}
    async def forbidden(*_args, **_kwargs):
        pytest.fail('retained source repair requires no new provider read')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    monkeypatch.setenv('EXA_API_KEY', 'test')
    monkeypatch.setattr(inv, '_post_json', post)
    monkeypatch.setattr(inv, '_search_web', forbidden)
    monkeypatch.setattr(inv, '_fetch_page', retained_fetch)
    monkeypatch.setattr(inv, 'time', _Clock())
    monkeypatch.setattr(inv, 'MAX_REASONING_TURNS', 2)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator=LOCATOR, targets=('headcount',),
        requested_employee_buckets=('51-200', '201-500', '501-1,000'),
        prior_observations={'observed_company_name': 'tvScientific',
                            'observed_company_website': 'https://www.tvscientific.com',
                            'observed_company_linkedin': 'https://www.linkedin.com/company/tvscientific'},
        verified_homepage_identity={},
    ))
    assert result.get('_completed_submit') is True, (result, len(requests))
    assert result['claims']['headcount']['status'] == corrected_status
    assert len(requests) == 3 and reads == [URL]
    assert result['usage']['search_calls'] == 0
    assert result['usage']['fetch_calls'] == 1
