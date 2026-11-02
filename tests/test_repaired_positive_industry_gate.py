"""A schema repair must not turn a skipped industry audit into acceptance."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena import scoring
from qualification.scoring import lead_scorer as s
from qualification.scoring.company_fit_decision import company_fit_match, company_fit_mismatch
from tests.test_positive_semantic_fit_review import _company, _icp, _verdict, _finding

MODEL = scoring.DEFAULT_JUDGE_MODELS['industry_evidence_review']


def run_case(monkeypatch, *, enabled=True, remains_mismatch=False,
             initially_reviewed=False, industry_status='VERIFIED',
             identity='linkedin_conflict'):
    company = _company()
    icp = _icp().model_copy(update={'required_attribute': ''})
    initial = _verdict(
        observed_company_linkedin='https://www.linkedin.com/company/foreign-peer',
        observed_hq_country='United States' if initially_reviewed else 'Canada',
        observed_hq_state='California' if initially_reviewed else 'Ontario',
        geography_matches=initially_reviewed,
        geography_evidence_quote=('PeerConnect is headquartered in California, United States.'
            if initially_reviewed else 'PeerConnect is headquartered in Ontario, Canada.'),
    )
    if identity == 'domain_conflict':
        initial['observed_company_website'] = 'https://foreign.example/'
        initial['observed_company_linkedin'] = company.company_linkedin
    elif identity == 'missing':
        initial.update(observed_company_name='', observed_company_website='', observed_company_linkedin='')
    elif identity == 'match':
        initial['observed_company_linkedin'] = company.company_linkedin
    repaired = _verdict(observed_company_stage='Series A', stage_matches=True,
        stage_evidence_url='https://peer.example/funding',
        stage_evidence_quote='PeerConnect announced its Series A funding round.')
    if remains_mismatch:
        repaired.update(observed_hq_country='Canada', observed_hq_state='Ontario',
            geography_matches=False, geography_evidence_quote=initial['geography_evidence_quote'])
    requests, investigations = [], []
    async def request(**kw):
        requests.append(kw['telemetry_purpose'])
        assert len(requests) <= 2
        return deepcopy(initial if len(requests) == 1 else repaired), ''
    async def unchanged(verdict, *a, **kw): return dict(verdict)
    async def refresh(verdict, *a, invocation_cache, **kw):
        invocation_cache['refresh_outcome'] = 'verified'
        return dict(verdict)
    async def prechecks(*a, **kw): return company_fit_match()
    async def homepage(*a, **kw):
        return company_fit_match(details={'identity': {'decision': 'match',
            'evidence_source': 'company_homepage', 'observed_name': 'peerconnect',
            'observed_domain': 'peer.example', 'observed_linkedin_slug': 'peerconnect'},
            'verified_homepage_transport_domain': 'peer.example'})
    async def investigate(**kw):
        investigations.append(kw)
        claims = {}
        if 'stage' in kw['targets']:
            claims['stage'] = _finding('stage')
        if 'industry' in kw['targets']:
            claims['industry'] = _finding('industry', status=industry_status,
                observed_value='', observed_industry='Education',
                observed_subindustry='Enrollment software', activity_role='supplier_operator',
                evidence_url='https://peer.example/platform',
                evidence_quote='PeerConnect supplies a student enrollment platform to universities.',
                _industry_review={'schema': 'industry_evidence_review_v1', 'model': MODEL,
                    'decision': {'VERIFIED': 'match', 'CONTRADICTED': 'mismatch', 'UNPROVEN': 'unavailable'}[industry_status],
                    'packet_sha256': 'a'*64})
            if industry_status == 'UNPROVEN':
                claims['industry'].update(activity_role='unresolved', evidence_url='', evidence_quote='')
            elif industry_status == 'CONTRADICTED':
                claims['industry'].update(activity_role='customer_user',
                    evidence_quote='PeerConnect uses an enrollment platform for its own students.')
        return {'claims': claims, '_validated_stage_finding': claims.get('stage', {}),
            '_completed_submit': True, 'failure_reason': '',
            'usage': {'reasoning_turns': 1, 'search_calls': 1, 'fetch_calls': 1,
                      'fetch_outcomes': []}}
    models = dict(scoring.DEFAULT_JUDGE_MODELS)
    if not enabled: models.pop('industry_evidence_review')
    policy = scoring.build_scorer_policy(judge_models=models)
    for name in scoring.CREDENTIAL_ENV_NAMES: monkeypatch.delenv(name, raising=False)
    for name in ('ARENA_INDUSTRY_EVIDENCE_MODEL', 'ARENA_INTENT_EVIDENCE_MODEL',
                 'ARENA_INTENT_DETAILS_REVIEW_MODEL', 'ARENA_ROLE_BATCH_MODEL'):
        monkeypatch.delenv(name, raising=False)
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment,
        credentials={name: 'offline-synthetic-key' for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items(): monkeypatch.setenv(name, value)
    for name, fn in [('run_company_zero_checks', prechecks), ('verify_company_exists', homepage),
        ('_request_company_reverify_json', request), ('_ground_employee_size_observation', unchanged),
        ('_refresh_linkedin_employee_size_observation', refresh), ('investigate_company_evidence', investigate)]:
        monkeypatch.setattr(s, name, fn)
    monkeypatch.setattr(s, '_resolve_observed_linkedin_redirect_alias', lambda company, verdict: unchanged(verdict))
    result = asyncio.run(s._verify_company_fit(company, icp, 0., 0., set(),
        require_https_transport=True, evidence_investigator=True, company_quality=True))
    return result, requests, investigations


@pytest.mark.parametrize('status,expected', [('VERIFIED', 'match'),
    ('CONTRADICTED', 'mismatch'), ('UNPROVEN', 'mismatch')])
def test_new_role_shares_first_review_before_repair_can_create_acceptance(monkeypatch, status, expected):
    result, requests, inv = run_case(monkeypatch, industry_status=status)
    assert requests == (['lead_scorer_reverify'] if status == 'UNPROVEN'
        else ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair'])
    assert [call['targets'] for call in inv] == [('stage', 'industry')]
    assert inv[0]['positive_semantic_review'] is True
    assert result.decision == expected
    receipt = result.receipt('company_fit')
    assert receipt['company_fit_dimensions']['stage'] == 'match'
    assert receipt['company_fit_dimensions']['identity'] == ('unavailable' if status == 'UNPROVEN' else 'match')
    assert receipt['company_fit_dimensions']['industry'] == {
        'VERIFIED': 'match', 'CONTRADICTED': 'mismatch', 'UNPROVEN': 'unavailable'}[status]
    investigation = next(r for r in receipt['supporting_receipts'] if r.get('gate') == 'company_evidence_investigation')
    assert investigation['claims']['stage'] == _finding('stage')
    assert investigation['claims']['industry']['_industry_review']['model'] == MODEL


def test_repaired_independent_terminal_mismatch_keeps_short_circuit(monkeypatch):
    result, requests, inv = run_case(monkeypatch, remains_mismatch=True, identity='match')
    assert result.decision == 'mismatch'
    assert [call['targets'] for call in inv] == [('stage',)]
    assert requests == ['lead_scorer_reverify']


def test_frozen_legacy_role_keeps_prior_path(monkeypatch):
    result, _, inv = run_case(monkeypatch, enabled=False)
    assert result.decision == 'match'
    assert [call['targets'] for call in inv] == [('stage',)]


def test_current_role_already_reviewed_industry_is_not_reviewed_twice(monkeypatch):
    result, _, inv = run_case(monkeypatch, initially_reviewed=True)
    assert result.decision == 'match'
    assert [call['targets'] for call in inv] == [('stage', 'industry')]


def test_domain_conflict_keeps_identity_first_then_existing_postrepair_review(monkeypatch):
    result, requests, inv = run_case(monkeypatch, identity='domain_conflict')
    assert result.decision == 'match'
    assert requests == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert [call['targets'] for call in inv] == [('stage', 'industry')]


def test_missing_identity_does_not_reopen_an_independent_terminal_mismatch(monkeypatch):
    result, requests, inv = run_case(monkeypatch, identity='missing')
    assert result.decision == 'mismatch'
    assert requests == ['lead_scorer_reverify']
    assert [call['targets'] for call in inv] == [('stage',)]


@pytest.mark.parametrize('reason,targets,identity,industry,attribute,enabled,expected', [
    ('web_linkedin_conflicts_with_verified_homepage', ('stage',), 'unavailable', 'match', 'match', True, True),
    ('web_domain_conflicts_with_verified_homepage', (), 'unavailable', 'match', 'match', True, True),
    ('identity_not_proven', ('stage',), 'unavailable', 'match', 'match', True, False),
    ('', ('stage',), 'unavailable', 'match', 'match', True, False),
    ('web_linkedin_conflicts_with_verified_homepage', ('stage', 'rebrand'), 'unavailable', 'match', 'match', True, False),
    ('web_linkedin_conflicts_with_verified_homepage', ('stage',), 'match', 'match', 'match', True, False),
    ('web_linkedin_conflicts_with_verified_homepage', ('stage',), 'unavailable', 'match', 'match', False, False),
    ('web_linkedin_conflicts_with_verified_homepage', ('stage',), 'unavailable', 'unavailable', 'match', True, False),
    ('web_linkedin_conflicts_with_verified_homepage', ('stage',), 'unavailable', 'match', 'mismatch', True, True),
    ('', ('stage',), 'match', 'match', 'mismatch', True, False),
])
def test_exact_current_role_full_identity_repair_predicate(monkeypatch, reason, targets,
        identity, industry, attribute, enabled, expected):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL if enabled else '')
    result = company_fit_mismatch('independent mismatch', details={
        'identity_decision': identity, 'identity_receipt': {'reason_code': reason},
        'dimension_decisions': {'industry': industry, 'geography': 'mismatch'},
        'required_attribute_decision': attribute})
    icp = SimpleNamespace(required_attribute='Product capability' if attribute == 'mismatch' else '',
        company_stage='Series A')
    assert s._positive_semantic_review_needed(result, icp, targets) is expected


def test_unavailable_industry_keeps_normal_target_selection():
    result = company_fit_mismatch('geography mismatch', details={
        'dimension_decisions': {'industry': 'unavailable', 'geography': 'mismatch'},
        'identity_decision': 'match'})
    assert 'industry' in s._targeted_company_investigation_dimensions(
        result, icp_stage='', employee_size_conflict=False)
