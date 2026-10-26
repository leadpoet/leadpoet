import asyncio
import json

import pytest

from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_refusal_fallback import _install_loop, FEATURES, URL


MODEL = 'openai/gpt-6.1-sol'


def install(monkeypatch, *, decision='match', missing=None, mutation=None, **kwargs):
    run, calls, clock = _install_loop(
        monkeypatch, target='industry', status=200,
        body='Coreworks connects account workflows to its analysis and reporting product.',
        urls=('https://www.coreworks.ai/product', URL), finding=_finding('industry'),
        **kwargs)
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    original = c._post_json
    review_calls = []

    async def post(session, url, *, headers, payload):
        if payload.get('model') != MODEL:
            return await original(session, url, headers=headers, payload=payload)
        review_calls.append(payload)
        pages = [page for message in payload['messages'][2:]
                 for page in json.loads(message['content'])['untrusted_source_pages']]
        spans = [span for page in pages for span in page['spans']]
        result = {'observed_activity': 'Analysis, reporting and account workflow software',
                  'reason': 'The connected product is supported by the two pages.',
                  'missing_requirements': missing or [], 'role': 'supplier_operator',
                  'evidence_span_ids': [span['span_id'] for span in spans], 'decision': decision}
        if mutation:
            mutation(result)
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(c, '_post_json', post)
    return run, calls, clock, review_calls


@pytest.mark.parametrize('positive_review', [False, True])
def test_native_loop_retains_both_sources_through_fit_evidence(monkeypatch, positive_review):
    run, calls, clock, reviews = install(monkeypatch)
    # Exercise both normal and prior-positive modes through the same entrypoint.
    original = c.investigate_company_evidence

    async def invoke(**kwargs):
        kwargs['positive_semantic_review'] = positive_review
        return await original(**kwargs)

    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    result = asyncio.run(run(requested_product_service='Supplies reporting and account workflows'))
    assert result['_completed_submit'] is True
    finding = result['claims']['industry']
    assert finding['status'] == 'VERIFIED'
    assert len(finding['_industry_review']['evidence']) == 2
    assert len(finding['supporting_evidence']) == 1
    projected = lead_scorer._project_investigator_industry({'attribute_satisfied': False}, finding)
    stored = json.loads(json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry')))
    assert stored['quote'] == FEATURES
    assert stored['supporting_evidence'] == finding['supporting_evidence']
    assert projected['attribute_satisfied'] is False
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS


def test_native_discovery_requests_pass_the_real_arena_request_validator(monkeypatch):
    from lab_arena.operations import validate_operation_request
    run, calls, _, reviews = install(monkeypatch)
    result = asyncio.run(run())
    assert result['_completed_submit'] is True
    for request in [*calls['model'], *reviews]:
        validate_operation_request('openrouter.chat', request)


def test_unproven_industry_review_cannot_leave_an_old_pass(monkeypatch):
    run, _, _, _ = install(monkeypatch, decision='unavailable', missing=['Voice support'])
    result = asyncio.run(run())
    finding = result['claims']['industry']
    assert finding['status'] == 'UNPROVEN'
    old = {'industry_matches': True, 'industry_activity_role': 'supplier_operator',
           'industry_evidence_url': URL, 'industry_evidence_quote': FEATURES,
           'dimension_evidence': {'industry': {'url': URL, 'quote': FEATURES}},
           'stage_matches': True}
    projected = lead_scorer._project_investigator_industry(old, finding)
    assert projected['industry_matches'] is None
    assert projected['stage_matches'] is True
    assert lead_scorer._dimension_web_evidence(projected, 'industry') == {'url': '', 'quote': ''}
    assert old['industry_matches'] is True


@pytest.mark.parametrize('mutation', [
    lambda result: result['evidence_span_ids'].append('foreign-span'),
    lambda result: result.update(missing_requirements=['Unsupported capability']),
    lambda result: result.update(role='customer_user'),
])
def test_invalid_strong_positive_is_not_an_admission(monkeypatch, mutation):
    run, _, _, _ = install(monkeypatch, mutation=mutation)
    result = asyncio.run(run())
    assert result['failure_reason'] == 'malformed_response'
    assert not result.get('_completed_submit')


def test_no_extra_model_call_after_admission_deadline(monkeypatch):
    run, _, _, reviews = install(monkeypatch, case='deadline')
    result = asyncio.run(run())
    assert not reviews
    assert result['failure_reason'] == 'admission_budget_interrupted'


def test_an_unbound_source_does_not_reach_strong_judge(monkeypatch):
    run, _, _, reviews = install(monkeypatch, identity={
        'normalized_name': 'Other', 'registrable_dns_domain': 'other.example'})
    result = asyncio.run(run())
    assert not reviews
    assert result['claims']['industry']['status'] == 'UNPROVEN'


@pytest.mark.parametrize('enabled', [False, True])
def test_frozen_policy_controls_industry_model_and_blocks_ambient_override(enabled):
    models = dict(scoring.DEFAULT_JUDGE_MODELS)
    if not enabled:
        models.pop('industry_evidence_review')
    policy = scoring.build_scorer_policy(judge_models=models)
    credentials = {name: 'offline-secret' for name in scoring.CREDENTIAL_ENV_NAMES}
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment, credentials=credentials)
    assert environment['ARENA_INDUSTRY_EVIDENCE_MODEL'] == (MODEL if enabled else '')
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy,
            environ={'ARENA_INDUSTRY_EVIDENCE_MODEL': 'another/model'}, credentials=credentials)


@pytest.mark.parametrize('model,expected', [('', False), (MODEL, True)])
def test_positive_industry_review_does_not_require_a_separate_attribute(monkeypatch, model, expected):
    from types import SimpleNamespace
    from qualification.scoring.company_fit_decision import company_fit_match
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', model)
    result = company_fit_match('All active company requirements passed', details={
        'dimension_decisions': {'industry': 'match'},
        'required_attribute_decision': 'not_required',
    })
    icp = SimpleNamespace(required_attribute='', industry='Software', product_service='Reporting')
    assert lead_scorer._positive_semantic_review_needed(result, icp, ()) is expected


def test_new_industry_review_preserves_an_independent_final_rejection(monkeypatch):
    from types import SimpleNamespace
    from qualification.scoring.company_fit_decision import company_fit_mismatch
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    result = company_fit_mismatch('Country mismatch', details={
        'dimension_decisions': {'industry': 'match', 'geography': 'mismatch'},
        'required_attribute_decision': 'not_required',
    })
    assert lead_scorer._positive_semantic_review_needed(
        result, SimpleNamespace(required_attribute=''), ()) is False
