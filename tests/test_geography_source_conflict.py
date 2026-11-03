"""Captured dual-HQ conflicts use the existing single bounded investigation."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_mismatch

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/airwallex_oct23_headquarters_conflict.json').read_text())
# This synthetic private receipt uses the current literal representation.
# Keep the captured first-party article, submitted company and truth unchanged.
from qualification.scoring.linkedin_company_size import project_structured_linkedin_company_headquarters
FIXTURE['headquarters'] = project_structured_linkedin_company_headquarters(
    'airwallex.com', FIXTURE['headquarters']['url'], 'Airwallex', {
        'status': 200, 'element': {'name': 'Airwallex', 'website': 'https://airwallex.com/',
            'linkedinUrl': FIXTURE['headquarters']['url'],
            'locations': [{'headquarter': True, 'country': FIXTURE['headquarters']['country']}]},
    },
)
IDENTITY = {'normalized_name': 'Airwallex', 'registrable_dns_domain': 'airwallex.com', 'linkedin_company_slug': 'airwallex'}
ANCHOR = {'submitted_name': 'airwallex', 'observed_name': 'airwallex', 'submitted_domain': 'airwallex.com', 'observed_domain': 'airwallex.com', 'verified_name': 'Airwallex', 'verified_domain': 'airwallex.com', 'verified_linkedin_slug': 'airwallex'}


def company():
    value = deepcopy(FIXTURE['company'])
    value['state'] = value.get('state') or ''
    for signal in value['intent_signals']:
        signal.setdefault('source', 'news')
        signal.setdefault('snippet', signal['description'])
    return CompanyOutput.model_validate(value)


def icp():
    return ICPPrompt(icp_id='airwallex-captured', industry='financial services', sub_industry='', product_service='', employee_count='1,001-5,000', country='Singapore', geography='Singapore', company_stage='Series C+')


def verdict():
    return {'observed_company_name': 'Airwallex', 'observed_company_website': 'https://airwallex.com/',
        'observed_company_linkedin': FIXTURE['headquarters']['url'], 'observed_employee_count': '1,001-5,000',
        'employee_size_matches': True, 'employee_size_evidence_url': 'https://airwallex.com/about',
        'employee_size_evidence_quote': 'Airwallex has 2,300 employees.',
        'observed_industry': 'financial services', 'observed_subindustry': '', 'industry_matches': True,
        'industry_activity_role': 'supplier_operator', 'industry_evidence_url': 'https://airwallex.com/about',
        'industry_evidence_quote': 'Airwallex operates payments infrastructure.',
        'observed_company_stage': 'Series C+', 'stage_matches': True,
        'stage_evidence_url': company().company_stage_evidence[2].url,
        'stage_evidence_quote': company().company_stage_evidence[2].quote,
        'observed_hq_country': 'Hong Kong', 'observed_hq_state': '', 'geography_matches': False,
        'geography_evidence_url': FIXTURE['headquarters']['url'], 'geography_evidence_quote': FIXTURE['headquarters']['text']}


def result():
    return scorer._reverify_decision(verdict(), '', 'Series C+', icp=icp(), company=company(),
        verified_homepage_identity=IDENTITY, verified_homepage_transport_domain='airwallex.com', company_quality=True,
        structured_profile_headquarters_evidence=FIXTURE['headquarters'])


def finding(quote=None, *, url=None, body=None, status='VERIFIED'):
    url = url or FIXTURE['source_url']
    quote = quote or FIXTURE['quote']
    return investigator._validated_findings({'findings': [{
        'target': 'geography', 'status': status, 'observed_country': 'Singapore',
        'observed_state': '', 'evidence_url': url, 'evidence_quote': quote,
    }]}, targets=('geography',), fetched_pages={url: body or quote},
        fetched_final_urls={url: url}, first_party_domains={'airwallex.com'},
        identity_names={'airwallex'}, identity_anchor=ANCHOR)['geography']


def project(proof, *, observation=None):
    return scorer._project_verified_linkedin_headquarters(observation or verdict(), verified_identity=IDENTITY,
        invocation_cache={'structured_profile_headquarters_evidence': FIXTURE['headquarters']},
        linkedin_profile_source={}, icp=icp(), company=company(), company_quality=True,
        validated_geography_finding=proof)


def test_captured_airwallex_input_opens_geography_in_existing_stage_investigation():
    prior = result()
    assert prior.details['dimension_decisions']['geography'] == 'mismatch'
    assert scorer._targeted_company_investigation_dimensions(prior, icp_stage='Series C+', icp=icp(),
        employee_size_conflict=False, company=company(), verified_homepage_identity=IDENTITY,
        verified_transport_domain='airwallex.com', verified_linkedin_employee_size=True,
        verified_linkedin_headquarters=True) == ('stage', 'geography')


def test_validated_current_captured_coheadquarters_survives_profile_projection():
    proof = finding(body=FIXTURE['source_text'])
    assert proof['status'] == 'VERIFIED'
    projected = project(proof)
    assert projected['observed_hq_country'] == 'Singapore'
    assert projected['geography_evidence_url'] == FIXTURE['source_url']
    assert scorer._decision_from_observed_geography(projected, icp(), company=company(), company_quality=True) == 'match'


@pytest.mark.parametrize('quote', [
    'Airwallex maintains dual global headquarters in Singapore and San Francisco.',
    'Airwallex has headquarters in Singapore and San Francisco.',
    'About Airwallex\nWe maintain our global headquarters in Singapore and San Francisco.',
    'Airwallex operates from global headquarters in Singapore and San Francisco.',
])
def test_validated_current_headquarters_does_not_require_one_sentence_template(quote):
    proof = finding(quote)
    assert proof['status'] == 'VERIFIED'
    assert project(proof)['observed_hq_country'] == 'Singapore'


@pytest.mark.parametrize('quote', [
    'Airwallex has an office in Singapore, but its headquarters is in Hong Kong.',
    'Airwallex is headquartered in Hong Kong. Its customer is co-headquartered in Singapore.',
    'Airwallex was co-headquartered in Singapore and San Francisco.',
    'Airwallex will be co-headquartered in Singapore and San Francisco.',
    'Airwallex is not co-headquartered in Singapore and San Francisco.',
    'Airwallex may be co-headquartered in Singapore and San Francisco.',
    'Airwallex has a regional headquarters in Singapore.',
    'Airwallex has offices in Singapore. Airwallex is co-headquartered in Hong Kong and London.',
    'Other Company is co-headquartered in Singapore and San Francisco.',
])
def test_completed_semantic_negative_does_not_override_profile(quote):
    # The existing investigator decides these meanings. This offline check
    # verifies integration of its negative decision; actual-model controls
    # must separately establish that it interprets each source correctly.
    projected = project(finding(quote, status='UNPROVEN'))
    assert projected['observed_hq_country'] == 'Hong Kong'
    assert projected['geography_evidence_url'] == FIXTURE['headquarters']['url']


def test_exact_quote_and_firstparty_binding_stay_required():
    for proof in (finding(url='https://other.example/news'), finding(body='Airwallex news with no quote.')):
        assert proof['status'] == 'UNPROVEN'
        assert project(proof)['observed_hq_country'] == 'Hong Kong'


@pytest.mark.parametrize('variant', ['no-identity', 'no-firstparty-source', 'no-verified-profile', 'outside-icp'])
def test_no_blanket_mismatch_retry(variant):
    prior = result()
    candidate = company()
    anchor = IDENTITY
    verified = True
    stage = 'Series C+'
    request = icp()
    if variant == 'outside-icp': request = request.model_copy(update={'country': 'Germany', 'geography': 'Germany'})
    elif variant == 'no-identity':
        prior.details['identity_decision'] = 'unavailable'
    elif variant == 'no-firstparty-source':
        candidate = candidate.model_copy(update={'company_stage_evidence': [], 'fit_evidence_urls': [], 'intent_signals': [candidate.intent_signals[0]], 'required_attribute': None})
    elif variant == 'no-verified-profile': verified = False
    targets = scorer._targeted_company_investigation_dimensions(prior, icp_stage=stage, icp=request,
        employee_size_conflict=False, company=candidate, verified_homepage_identity=anchor,
        verified_transport_domain='airwallex.com', verified_linkedin_employee_size=True,
        verified_linkedin_headquarters=verified)
    assert 'geography' not in targets


def test_existing_targeted_investigator_projects_fetched_coheadquarters(monkeypatch):
    captured = {}
    async def investigate(**kwargs):
        captured.update(kwargs)
        return {'claims': {'geography': finding(body=FIXTURE['source_text'])}, '_completed_submit': True, 'failure_reason': ''}
    monkeypatch.setattr(scorer, 'investigate_company_evidence', investigate)
    projected, updated, *_ = asyncio.run(scorer._run_targeted_company_evidence_investigation(
        company=company(), icp=icp(), verdict=verdict(), investigation_targets=('geography',),
        icp_attribute='', icp_stage='Series C+', verified_identity=IDENTITY,
        verified_transport_domain='airwallex.com', structured_employee_size_evidence=None,
        structured_public_company_evidence=None, employee_size_conflict=False, company_quality=True,
        structured_profile_headquarters_evidence=FIXTURE['headquarters'],
        structured_profile_invocation_cache={'structured_profile_headquarters_evidence': FIXTURE['headquarters']},
        prior_result=result()))
    assert captured['targets'] == ('geography',)
    assert FIXTURE['source_url'] in captured['prior_observations']['submitted_source_urls']
    assert projected['observed_hq_country'] == 'Singapore'
    assert updated.details['dimension_decisions']['geography'] == 'match'


@pytest.mark.parametrize('stage_required,proof_kind', [(True, 'current'), (False, 'current'), (False, 'office'), (False, 'regional'), (False, 'customer'), (False, 'old'), (False, 'future'), (False, 'wrong-entity'), (False, 'missing')])
def test_captured_company_runs_through_full_company_reverification(monkeypatch, stage_required, proof_kind):
    from unittest.mock import AsyncMock
    from qualification.scoring.company_fit_decision import company_fit_match
    from tests.test_linkedin_first_company_fit import install_profile, CAPTURE

    body = deepcopy(CAPTURE)
    body['element'].update(name='Airwallex', website='https://airwallex.com/',
        linkedinUrl=FIXTURE['headquarters']['url'],
        employeeCountRange={'start': 1001, 'end': 5000},
        locations=[{'headquarter': True, 'country': 'HK', 'city': 'Global'}])
    install_profile(monkeypatch, body)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-placeholder')
    broad = verdict()
    broad.update(observed_hq_country='Singapore', geography_matches=True,
        geography_evidence_url=FIXTURE['source_url'], geography_evidence_quote=FIXTURE['quote'])
    monkeypatch.setattr(scorer, '_request_company_reverify_json', AsyncMock(return_value=(broad, '')))
    monkeypatch.setattr(scorer, 'fetch_current_linkedin_company_size', AsyncMock(return_value={'outcome': 'insufficient_evidence'}))
    captured = []
    async def investigate(**kwargs):
        captured.append(kwargs)
        stage = company().company_stage_evidence[2]
        quote = FIXTURE['quote']
        if proof_kind == 'office': quote = 'Airwallex has an office in Singapore.'
        elif proof_kind == 'regional': quote = 'Airwallex has a regional headquarters in Singapore.'
        elif proof_kind == 'customer': quote = 'Airwallex customer Nium has headquarters in Singapore.'
        elif proof_kind == 'old': quote = 'Airwallex formerly had headquarters in Singapore.'
        elif proof_kind == 'future': quote = 'Airwallex will establish headquarters in Singapore.'
        elif proof_kind == 'wrong-entity': quote = 'Other Company is co-headquartered in Singapore and San Francisco.'
        # Mock completed semantic negatives, not a claim that the literal
        # citation validator determines chronology or branch relationships.
        semantic_negative = proof_kind in {'regional', 'customer', 'old', 'future'}
        findings = [{'target': 'geography', 'status': 'UNPROVEN' if semantic_negative else 'VERIFIED', 'observed_country': 'Singapore',
            'observed_state': '', 'evidence_url': FIXTURE['source_url'], 'evidence_quote': quote}]
        if stage_required:
            findings.insert(0, {'target': 'stage', 'status': 'VERIFIED', 'observed_value': 'Series C+',
                'evidence_url': stage.url, 'evidence_quote': stage.quote})
        claims = investigator._validated_findings({'findings': findings}, targets=kwargs['targets'],
            fetched_pages={stage.url: stage.quote, FIXTURE['source_url']: 'Unavailable' if proof_kind == 'missing' else quote},
            fetched_final_urls={stage.url: stage.url, FIXTURE['source_url']: FIXTURE['source_url']},
            first_party_domains={'airwallex.com'}, identity_names={'airwallex'}, identity_anchor=ANCHOR,
            requested_stage='Series C+' if stage_required else '')
        if proof_kind == 'current': assert all(v['status'] == 'VERIFIED' for v in claims.values())
        return {'claims': claims, '_validated_stage_finding': claims.get('stage'), '_completed_submit': True, 'failure_reason': ''}
    monkeypatch.setattr(scorer, 'investigate_company_evidence', investigate)
    homepage = company_fit_match('verified homepage', details={'identity': {
        'decision': 'match', 'evidence_source': 'company_homepage', 'observed_name': 'Airwallex',
        'observed_domain': 'airwallex.com', 'observed_linkedin_slug': 'airwallex'},
        'verified_homepage_transport_domain': 'airwallex.com'})
    request = icp() if stage_required else icp().model_copy(update={'company_stage': ''})
    final = asyncio.run(scorer._llm_reverify_company(company(), request, require_company_fit_dimensions=True,
        verified_homepage_identity=homepage, company_quality=True, evidence_investigator=True))
    assert len(captured) == 1
    assert captured[0]['targets'] == (('stage', 'geography') if stage_required else ('geography',))
    expected = 'match' if proof_kind == 'current' else 'mismatch'
    assert final.details['dimension_decisions']['geography'] == expected
    assert final.decision == expected


def test_broad_web_observation_cannot_resolve_profile_conflict():
    broad = verdict()
    broad.update(observed_hq_country='Singapore', geography_matches=True,
        geography_evidence_url=FIXTURE['source_url'], geography_evidence_quote=FIXTURE['quote'])
    # A plausible raw observation is not an investigator-validated finding.
    assert project({}, observation=broad)['observed_hq_country'] == 'Hong Kong'


@pytest.mark.parametrize('status', ['UNPROVEN', 'CONTRADICTED'])
def test_nonverified_investigator_finding_retains_profile(status):
    assert project(finding(status=status))['observed_hq_country'] == 'Hong Kong'
