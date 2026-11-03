"""An old vanity needs its own authoritative canonical response, never relabeling."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer as scorer, linkedin_company_size as linkedin
from lab_arena import company_judgments, contracts, scoring
from qualification.scoring.company_fit_decision import company_fit_match, company_fit_unavailable
from tests.test_fit_evidence_url_compat import _company, _complete_verdict, _icp


DATA = json.loads((Path(__file__).parent / 'lab_arena/fixtures/nov3_mychn_homepage_identity.json').read_text())
OLD_URL = 'https://www.linkedin.com/company/community-health-network-fqhc'
CURRENT_URL = 'https://www.linkedin.com/company/mychn'


def homepage():
    return company_fit_unavailable('Saved bound homepage identity conflict', details=deepcopy(DATA['homepage']['details']))


def web_receipt():
    return scorer._web_identity_receipt(SimpleNamespace(**DATA['company']), DATA['web_observation'],
                                        verified_homepage_transport_domain='mychn.org')


def canonical_response_proof(payload=None, requested=OLD_URL):
    # This models a future response to the actual OLD URL. The saved payload
    # was fetched for CURRENT_URL; no historical receipt is changed or reused.
    return linkedin.project_structured_linkedin_company_identity(
        'mychn.org', requested, deepcopy(payload or DATA['current_profile_payload']),
        observed_profile_url=CURRENT_URL, expected_company_name='MyCHN',
    )


def test_saved_dual_vanity_conflict_selects_old_url_and_needs_its_own_reply():
    home, web = homepage(), web_receipt()
    assert scorer._unresolved_homepage_web_identity_conflict(home, web)
    lookup = scorer._conflicting_homepage_numeric_profile_lookup(home, web)
    assert lookup['requested_profile_url'] == OLD_URL
    assert lookup['observed_profile_url'] == CURRENT_URL
    current_only = canonical_response_proof(requested=CURRENT_URL)
    assert 'requested_url' not in current_only
    assert scorer._unresolved_homepage_web_identity_conflict(home, {**web, 'structured_profile_identity': current_only})
    proof = canonical_response_proof()
    assert proof['requested_url'] == OLD_URL and proof['url'] == CURRENT_URL
    assert not scorer._unresolved_homepage_web_identity_conflict(home, {**web, 'structured_profile_identity': proof})


@pytest.mark.parametrize('field,value', [
    ('id', True), ('id', 0), ('id', '0'), ('id', '١٢٣'), ('id', ' 123'),
    ('universalName', 'mychn-other'), ('universalName', ''),
    ('linkedinUrl', 'https://www.linkedin.com/company/other'),
    ('name', 'MyCHN Parent'), ('name', 'Different Entity'),
    ('website', 'https://unrelated.example/'),
])
def test_wrong_canonical_main_profile_cannot_resolve_old_vanity(field, value):
    payload = deepcopy(DATA['current_profile_payload'])
    payload['data']['element'][field] = value
    assert canonical_response_proof(payload) is None


@pytest.mark.parametrize('siblings', [[None], [{}], ['malformed']])
def test_malformed_or_ambiguous_main_slots_cannot_prove_alias(siblings):
    element = deepcopy(DATA['current_profile_payload']['data']['element'])
    payload = {'data': {'status': 200, 'elements': [element, *siblings]}}
    assert canonical_response_proof(payload) is None


@pytest.mark.parametrize('fault', ['provider_error', 'both_slots', 'ambiguous_wrapper', 'description_name'])
def test_new_alias_has_no_error_wrapper_or_description_fallback(fault):
    payload = deepcopy(DATA['current_profile_payload'])
    if fault == 'provider_error':
        payload['errors'] = [{'code': 'provider_failure'}]
    elif fault == 'both_slots':
        payload['data']['elements'] = [deepcopy(payload['data']['element'])]
    elif fault == 'ambiguous_wrapper':
        payload['result'] = deepcopy(payload['data'])
    else:
        payload['data']['element'].update(name='MyCHN Parent', description='MyCHN is a healthcare provider.')
    assert canonical_response_proof(payload) is None


@pytest.mark.parametrize('field,value', [
    ('requested_url', CURRENT_URL), ('requested_url', 'https://www.linkedin.com/company/other'),
    ('url', OLD_URL), ('company_id', '0'), ('company_id', '١٢٣'),
    ('name', 'MyCHN Parent'), ('website', 'https://unrelated.example/'),
    ('provider', 'model'), ('source_field', 'description.leading_legal_name'),
])
def test_outer_assembly_revalidates_every_alias_binding(field, value):
    proof = canonical_response_proof()
    assert proof is not None
    proof[field] = value
    assert scorer._unresolved_homepage_web_identity_conflict(homepage(), {**web_receipt(), 'structured_profile_identity': proof})


def test_current_profile_or_prior_attempt_cannot_be_relabelled_as_old_lookup(monkeypatch):
    current = canonical_response_proof(requested=CURRENT_URL)
    cache = {'structured_attempted': True, 'structured_company_identity_evidence': current}
    async def forbidden(*args, **kwargs): pytest.fail('second structured lookup')
    monkeypatch.setattr(scorer, 'fetch_structured_linkedin_company_size', forbidden)
    lookup = scorer._conflicting_homepage_numeric_profile_lookup(homepage(), web_receipt())
    asyncio.run(scorer._fetch_structured_linkedin_profile_once(lookup, cache, collect_identity=True))
    assert cache['structured_company_identity_evidence'] == current
    assert 'requested_url' not in current
    assert scorer._unresolved_homepage_web_identity_conflict(homepage(), {**web_receipt(), 'structured_profile_identity': current})


@pytest.mark.parametrize('fault', ['wrong_web_name', 'wrong_web_domain', 'wrong_current_slug', 'no_bound_organization'])
def test_old_lookup_cannot_be_selected_from_unbound_identity(fault):
    home, web = homepage(), web_receipt()
    if fault == 'wrong_web_name': web['observed_name'] = 'mychnparent'
    elif fault == 'wrong_web_domain': web['observed_domain'] = 'unrelated.example'
    elif fault == 'wrong_current_slug': web['submitted_linkedin_slug'] = 'unrelated'
    else: home.details.pop('observed_homepage_organization_identity')
    assert not scorer._conflicting_homepage_numeric_profile_lookup(home, web)


@pytest.mark.parametrize('outcome', ['canonical', 'wrong_parent', 'transport', 'current_only'])
def test_real_web_path_replaces_current_lookup_with_one_old_vanity_request(monkeypatch, outcome):
    requests = []
    verdict = _complete_verdict(name='MyCHN', website='https://mychn.org/')
    verdict.update(DATA['web_observation'])
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-placeholder')
    async def request(**kwargs): return deepcopy(verdict), ''
    async def attribute(value, **kwargs): return dict(value), None
    async def employee(value, **kwargs): return dict(value)
    async def refresh(value, *args, **kwargs): return dict(value)
    async def structured(domain, url, **kwargs):
        requests.append((domain, url))
        if outcome == 'transport':
            kwargs['diagnostic'][linkedin.VERIFIER_FAILURE_REASON_KEY] = 'provider_error'
            return None
        payload = deepcopy(DATA['current_profile_payload'])
        if outcome == 'wrong_parent': payload['data']['element']['name'] = 'MyCHN Parent'
        proof = linkedin.project_structured_linkedin_company_identity(
            domain, CURRENT_URL if outcome == 'current_only' else url, payload,
            observed_profile_url=kwargs['company_identity_observed_profile_url'],
            expected_company_name=kwargs['expected_company_name'],
        )
        kwargs['company_identity_evidence'].update(proof or {})
        return None
    monkeypatch.setattr(scorer, '_request_company_reverify_json', request)
    monkeypatch.setattr(scorer, '_ground_required_attribute_evidence', attribute)
    monkeypatch.setattr(scorer, '_ground_employee_size_observation', employee)
    monkeypatch.setattr(scorer, '_refresh_linkedin_employee_size_observation', refresh)
    monkeypatch.setattr(scorer, 'fetch_structured_linkedin_company_size', structured)
    company = _company(**DATA['company'], company_stage='')
    result = asyncio.run(scorer._llm_reverify_company(company, _icp().model_copy(update={'company_stage': ''}),
        require_company_fit_dimensions=True, verified_homepage_identity=homepage()))
    assert requests == [('mychn.org', OLD_URL)]
    receipt = result.details['identity_receipt']
    if outcome == 'canonical':
        assert receipt['structured_profile_identity']['requested_url'] == OLD_URL
    else:
        assert not receipt.get('structured_profile_identity')
    assert scorer._unresolved_homepage_web_identity_conflict(homepage(), receipt) is (outcome != 'canonical')


def test_normal_matching_profile_keeps_existing_shape_and_needs_no_alias_lookup():
    web = web_receipt()
    matched_home = company_fit_match(details={'identity': {**web, 'evidence_source': 'company_homepage'}})
    assert not scorer._conflicting_homepage_numeric_profile_lookup(matched_home, web)
    ordinary = canonical_response_proof(requested=CURRENT_URL)
    assert set(ordinary) == {'name', 'provider', 'source_field', 'url', 'website'}
    assert ordinary['url'] == CURRENT_URL


@pytest.mark.parametrize('proof_fault,industry,expected', [
    (None, 'match', 'match'), ('absent', 'match', 'unavailable'),
    ('parent_name', 'match', 'unavailable'), ('current_lookup', 'match', 'unavailable'),
    (None, 'mismatch', 'mismatch'), (None, 'unavailable', 'unavailable'),
])
def test_final_company_fit_and_legacy_cache_keep_exact_proof_and_other_gates(monkeypatch, proof_fault, industry, expected):
    receipt = web_receipt()
    proof = canonical_response_proof(requested=CURRENT_URL if proof_fault == 'current_lookup' else OLD_URL)
    if proof_fault == 'parent_name': proof['name'] = 'MyCHN Parent'
    if proof_fault != 'absent': receipt['structured_profile_identity'] = proof
    async def identity(*args, **kwargs): return homepage()
    async def precheck(*args, **kwargs): return company_fit_match()
    async def completed(*args, **kwargs):
        return company_fit_match(details={
            'identity_decision': 'match', 'identity_receipt': receipt,
            'dimension_decisions': {'employee_size': 'match', 'industry': industry, 'geography': 'match', 'stage': 'match'},
            'dimension_evidence': {name: {'url': 'https://mychn.org/', 'quote': 'Independent dimension control.'}
                                   for name in ('employee_size', 'industry', 'geography')},
            'required_attribute_decision': 'match',
        })
    monkeypatch.setattr(scorer, 'verify_company_exists', identity)
    monkeypatch.setattr(scorer, 'run_company_zero_checks', precheck)
    monkeypatch.setattr(scorer, '_llm_reverify_company', completed)
    fit = asyncio.run(scorer._verify_company_fit(
        _company(**DATA['company'], company_stage=''), _icp().model_copy(update={'company_stage': ''}),
        0, 1, set(), require_https_transport=True,
    ))
    assert fit.decision == expected
    assert fit.details['company_fit_dimensions']['industry'] == industry
    if expected != 'match': return
    # The same seven-field host proof survives ordinary raw-output validation,
    # cache evidence serialization and reconstruction without a new authority.
    raw = {'company_judgment_policy': 'company_judgment_v1', 'final_score': 0.0,
           'company_identity_key': 'domain:mychn.org', 'company_identity_alias_keys': ['domain:mychn.org'],
           'verifier_gate_receipts': [fit.receipt('company_fit')], 'intent_signals_detail': []}
    assert company_judgments.raw_judgment_is_cacheable(raw)
    miss = {'company_index': 0, 'cache_key': 'sha256:' + 'a' * 64,
            'company_input_hash': 'sha256:' + 'b' * 64, 'authority_slot': 0}
    row = {**miss, 'raw_judgment': raw}
    output = scoring.build_scoring_output('old-vanity-control', [], company_judgments=[row])
    copied = scoring.validate_scoring_output_document(json.loads(contracts.canonical_json(output)))
    lease = {'schema_version': company_judgments.LEASE_SCHEMA_VERSION, 'hits': [], 'misses': [miss]}
    assert company_judgments.validate_new_company_judgments(copied['company_judgments'], lease_context=lease) == [row]
    retained = copied['company_judgments'][0]['raw_judgment']['verifier_gate_receipts'][0][
        'dimension_evidence']['identity']['web_identity_receipt']['structured_profile_identity']
    assert retained == proof
