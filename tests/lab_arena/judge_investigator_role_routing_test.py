"""Native serial controllers retain their frozen semantic and generic roles."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import openrouter_options as options


@pytest.mark.parametrize('variant', ['industry', 'generic', 'positive_semantic'])
@pytest.mark.parametrize('enabled', [False, True])
def test_native_controller_wire_uses_only_its_assigned_role(monkeypatch, variant, enabled):
    models = {'company': 'test/company-fit', 'generic': 'test/generic-controller',
              'positive_semantic': 'test/semantic-controller', 'industry': 'openai/gpt-6.1-sol'}
    routes = {model: 'example/' + role for role, model in models.items() if role != 'industry'}
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-not-secret')
    monkeypatch.setenv('EXA_API_KEY', 'test-not-secret')
    monkeypatch.setenv('ARENA_COMPANY_REVERIFY_MODEL', models['company'])
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', models['industry'])
    monkeypatch.setattr(investigator, 'INVESTIGATOR_MODEL', models['generic'])
    monkeypatch.setattr(investigator, 'POSITIVE_SEMANTIC_REVIEW_MODEL', models['positive_semantic'])
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps(routes))
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps({model: {
        'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096} for model in routes}))
    monkeypatch.setenv('ARENA_JUDGE_REASONING_EXCEPTIONS', json.dumps({models['industry']: {
        'roles': ['industry_evidence_review'], 'reason': 'quality_gate_not_met'}}) if enabled else '')
    calls = []
    async def capture(_session, _url, **kwargs):
        if not str(_url).startswith('https://openrouter.ai/'):
            return 200, {'results': []}
        calls.append(kwargs['payload'])
        if kwargs['payload'].get('tools'):
            return 400, {}
        # A malformed advisory permits the bounded native controller to run.
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
    monkeypatch.setattr(investigator, '_post_json', capture)
    url = 'https://acme.example/'
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={'name': 'Acme', 'website': url},
        targets=('industry',) if variant == 'industry' else ('required_attribute',),
        requested_industry='Software', requested_attribute='Supplies reporting software',
        positive_semantic_review=variant == 'positive_semantic',
        prior_observations={'submitted_source_urls': [url]},
        verified_homepage_identity={'normalized_name': 'acme', 'registrable_dns_domain': 'acme.example'},
        prefetched_pages={url: {'final_url': url, 'text': 'Acme supplies reporting software to businesses.'}}))
    controllers = [call for call in calls if call.get('tools')]
    assert len(controllers) == 1, (result, calls)
    expected = models['company' if enabled else 'industry'] if variant == 'industry' else models[variant]
    assert controllers[0]['model'] == expected
    if enabled:
        assert controllers[0]['provider'] == {'only': [routes[expected]]}
        assert controllers[0]['reasoning'] == {'effort': 'low', 'exclude': False}
        assert controllers[0]['max_tokens'] == 4096
    else:
        assert 'provider' not in controllers[0]
    if variant == 'industry':
        reviewers = [call for call in calls if not call.get('tools')]
        assert reviewers and {call['model'] for call in reviewers} == {models['industry']}
