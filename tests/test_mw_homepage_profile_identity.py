"""Saved MW observations must use the exact old homepage profile as a locator."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from qualification.scoring import lead_scorer as scorer, linkedin_company_size as linkedin
from qualification.scoring.company_fit_decision import company_fit_unavailable

ROWS = json.loads((Path(__file__).parent / 'lab_arena/fixtures/nov10_mw_homepage_identity.json').read_text())
OLD = 'https://www.linkedin.com/company/m-w-components'
CURRENT = 'https://www.linkedin.com/company/mwcomponents'

@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['run'])
def test_saved_mw_slug_conflict_never_passes_without_exact_profile_reply(row):
    details = deepcopy(row['identity'])
    web = details.pop('web_identity_receipt')
    home = company_fit_unavailable('Saved homepage conflict', details=details)
    if web['observed_name'] != web['submitted_name']:
        assert scorer._unresolved_homepage_web_identity_conflict(home, web)
        assert not scorer._conflicting_homepage_numeric_profile_lookup(home, web)
        return
    lookup = scorer._conflicting_homepage_numeric_profile_lookup(home, web)
    assert lookup['requested_profile_url'] == OLD
    assert lookup['observed_profile_url'] == CURRENT
    if web['decision'] == 'match':
        assert scorer._unresolved_homepage_web_identity_conflict(home, web)

@pytest.mark.parametrize('observed_slug', ['', 'mwcomponents'])
@pytest.mark.parametrize('outcome', ['canonical', 'absent', 'provider_failure', 'different_company', 'different_domain', 'current_only'])
def test_fresh_request_uses_one_old_locator_and_requires_authoritative_reply(monkeypatch, observed_slug, outcome):
    import asyncio
    from tests.test_fit_evidence_url_compat import _company, _complete_verdict, _icp
    requests = []
    details = deepcopy(ROWS[0]['identity'])
    details.pop('web_identity_receipt')
    home = company_fit_unavailable('Saved homepage conflict', details=details)
    verdict = _complete_verdict(name='MW Components', website='https://mwcomponents.com/')
    verdict['observed_company_linkedin'] = CURRENT if observed_slug else ''
    # Model omission is retained as an observation, never silently normalized.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-placeholder')
    async def request(**kwargs): return deepcopy(verdict), ''
    async def attribute(value, **kwargs): return dict(value), None
    async def employee(value, **kwargs): return dict(value)
    async def refresh(value, *args, **kwargs): return dict(value)
    async def structured(domain, url, **kwargs):
        requests.append((domain, url))
        if outcome in {'absent', 'provider_failure'}:
            if outcome == 'provider_failure':
                kwargs['diagnostic'][linkedin.VERIFIER_FAILURE_REASON_KEY] = 'provider_error'
            return None
        payload = {'data': {'status': 200, 'element': {
            'id': '12345', 'name': 'MW Components', 'universalName': 'mwcomponents',
            'linkedinUrl': CURRENT, 'website': 'https://mwcomponents.com/',
        }}}
        if outcome == 'different_company': payload['data']['element']['name'] = 'MW Industries'
        if outcome == 'different_domain': payload['data']['element']['website'] = 'https://unrelated.example/'
        proof = linkedin.project_structured_linkedin_company_identity(
            domain, CURRENT if outcome == 'current_only' else url, payload,
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
    company = _company(company_name='MW Components', company_website='https://mwcomponents.com/',
                       company_linkedin=CURRENT, company_stage='')
    result = asyncio.run(scorer._llm_reverify_company(
        company, _icp().model_copy(update={'company_stage': ''}),
        require_company_fit_dimensions=True, verified_homepage_identity=home,
    ))
    assert requests == [('mwcomponents.com', OLD)]
    receipt = result.details['identity_receipt']
    if outcome == 'canonical':
        assert receipt['decision'] == 'match'
        assert receipt['structured_profile_identity']['requested_url'] == OLD
        assert receipt['raw_observed_linkedin_slug'] == observed_slug
        assert not scorer._unresolved_homepage_web_identity_conflict(home, receipt)
    else:
        assert receipt.get('decision') != 'match' or scorer._unresolved_homepage_web_identity_conflict(home, receipt)

    from qualification.scoring.company_fit_decision import company_fit_match
    async def identity(*args, **kwargs): return home
    async def precheck(*args, **kwargs): return company_fit_match()
    async def completed(*args, **kwargs): return result
    monkeypatch.setattr(scorer, 'verify_company_exists', identity)
    monkeypatch.setattr(scorer, 'run_company_zero_checks', precheck)
    monkeypatch.setattr(scorer, '_llm_reverify_company', completed)
    fit = asyncio.run(scorer._verify_company_fit(
        company, _icp().model_copy(update={'company_stage': ''}), 0, 1, set(),
        require_https_transport=True,
    ))
    assert fit.decision == ('match' if outcome == 'canonical' else 'unavailable')
