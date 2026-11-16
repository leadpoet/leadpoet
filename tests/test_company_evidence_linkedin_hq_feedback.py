"""Keep LinkedIn source identity separate from missing headquarters proof."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response

CASE = json.loads((Path(__file__).parent / 'fixtures/company_evidence/linkedin_hq_country_missing.json').read_text())
URL = CASE['source']['url']
TEXT = CASE['source']['page']['text']
FINDING = CASE['finding']
REASON = ('Bound LinkedIn headquarters evidence must quote its exact About '
          'Headquarters field and prove the observed country and required state')
ANCHOR = {'verified_name': 'Starling', 'verified_domain': 'starlingbank.com',
          'verified_linkedin_slug': 'starlingbank'}


def validate(finding=None, text=TEXT, anchor=None, final_url=URL):
    return inv._validated_findings(
        {'findings': [finding or FINDING]}, targets=('geography',),
        fetched_pages={URL: text}, fetched_final_urls={URL: final_url},
        first_party_domains={'starlingbank.com'}, identity_names={'starling'},
        identity_anchor=ANCHOR if anchor is None else anchor,
    )['geography']


def test_actual_bound_profile_does_not_report_an_identity_failure():
    assert inv._bound_linkedin_headquarters_source(URL, URL, TEXT, ANCHOR)
    field = inv._linkedin_about_headquarters_field(TEXT)
    assert field == FINDING['evidence_quote']
    assert not inv._quote_supports_headquarters(
        field, observed_country='United Kingdom', observed_state='London')
    result = validate()
    assert result['status'] == 'UNPROVEN'
    assert result['evidence_quote'] == result['evidence_url'] == ''
    assert result['reason'] == REASON


def test_explicit_country_in_bound_headquarters_field_still_passes():
    quote = FINDING['evidence_quote'].replace('London, London', 'London, United Kingdom')
    text = TEXT.replace(FINDING['evidence_quote'], quote)[:inv.PROFILE_MAX_CHARACTERS]
    result = validate({**FINDING, 'evidence_quote': quote}, text)
    assert result['status'] == 'VERIFIED'
    assert result['evidence_quote'] == quote


@pytest.mark.parametrize('fault', ['identity', 'redirect', 'invented_quote', 'other_office'])
def test_identity_quote_and_office_controls_remain_rejected(fault):
    finding, text, anchor, final = deepcopy(FINDING), TEXT, dict(ANCHOR), URL
    if fault == 'identity': anchor['verified_linkedin_slug'] = 'other-company'
    if fault == 'redirect': final = 'https://www.linkedin.com/company/other-company/'
    if fault == 'invented_quote': finding['evidence_quote'] += ', United Kingdom'
    if fault == 'other_office':
        text = ('Regional office: United Kingdom\n' + text)[:inv.PROFILE_MAX_CHARACTERS]
    result = validate(finding, text, anchor, final)
    assert result['status'] == 'UNPROVEN'
    assert result['evidence_url'] == result['evidence_quote'] == ''
    if fault in {'identity', 'redirect', 'invented_quote'}:
        assert result['reason'] != REASON


@pytest.mark.parametrize('explicit_unproven', [True, False])
def test_existing_final_repair_preserves_completed_finding(monkeypatch, explicit_unproven):
    clock = _Clock()
    requests = []
    industry_url = 'https://www.starlingbank.com/about/'
    industry_quote = 'Starling supplies digital banking accounts through its mobile app.'
    industry = {**_finding(url=industry_url, quote=industry_quote),
                'observed_industry': 'Banking', 'observed_subindustry': 'Digital banking'}

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        if len(requests) == 1:
            clock.now = inv.ADMISSION_DEADLINE_SECONDS - inv.JUDGMENT_ADMISSION_RESERVE_SECONDS
            findings = [FINDING, industry]
        else:
            feedback = next(json.loads(m['content']) for m in reversed(payload['messages'])
                            if m.get('role') == 'tool' and 'rejected_findings' in json.loads(m['content']))
            assert feedback['rejected_findings'][0]['reason'] == REASON
            assert 'profile identity passed' in feedback['instruction']
            assert 'Do not infer the country from a city' in feedback['instruction']
            assert 'UNPROVEN' in feedback['instruction']
            findings = [{**FINDING, 'status': 'UNPROVEN', 'evidence_url': '',
                         'evidence_quote': '', 'reason': 'The headquarters field does not establish the country.'}
                        if explicit_unproven else FINDING]
        return 200, _response('submit_findings', {'findings': findings}, call=len(requests))

    async def no_research(*_args, **_kwargs):
        raise AssertionError('Final repair must not add research')

    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    monkeypatch.setenv('EXA_API_KEY', 'test-exa-key')
    monkeypatch.setattr(inv, 'time', clock)
    monkeypatch.setattr(inv, '_post_json', post)
    monkeypatch.setattr(inv, '_search_web', no_research)
    monkeypatch.setattr(inv, '_fetch_page', no_research)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={'name': 'Starling', 'website': 'https://www.starlingbank.com/', 'linkedin': URL},
        targets=('industry', 'geography'), requested_industry='Banking',
        requested_geography='United Kingdom, London', verified_homepage_identity=CASE['identity'],
        prefetched_pages={URL: CASE['source']['page'], industry_url: {'final_url': industry_url, 'text': industry_quote}},
        prior_observations={'submitted_source_urls': [URL, industry_url]},
    ))
    assert len(requests) == (2 if explicit_unproven else 3)
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['industry']['evidence_quote'] == industry_quote
    assert result['claims']['geography']['status'] == 'UNPROVEN'
    assert result['_completed_submit'] is explicit_unproven
    assert bool(result['failure_reason']) is not explicit_unproven
    assert result['usage']['fetch_calls'] == result['usage']['search_calls'] == 0
