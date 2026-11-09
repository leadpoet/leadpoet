"""Native controller repairs use its actual wire cap, without changing requests."""
import asyncio
from copy import deepcopy
from datetime import date
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring
from qualification.scoring import company_evidence_investigator as investigator
from tests.lab_arena.industry_controller_capture_test import tool_reply
from tests.lab_arena.judge_default_activation_test import EXCEPTIONS, PREVIOUS_OPTIONS, ROUTES

SOL = 'openai/gpt-6.1-sol'
GLM = 'z-ai/glm-5.3-flash'

# Complete explicit-length requests with the reviewed industry evidence rules.
# The November 7 audit changes only the criterion-rule system text;
# stage-field descriptions are covered by the stage contract tests. Repair
# instructions, tools, budgets and provider routing here are unchanged.
# Budget-only repairs must still match explicit-length requests byte for byte.
REQUEST_GOLDENS = {
    ('glm', 'empty'): 'sha256:217d4c152636764ac11f28aca5171c25444966abb901df7c2d596fb8316635b9',
    ('glm', 'malformed_submit'): 'sha256:8061648a09d166dc2a8f9e1ee924b8c3364fcd1e51f4ea64cb29b4b9e05342d2',
    ('sol_legacy', 'empty'): 'sha256:5bfb0148b11b872427fd5e2312d208600fcf8eab8fe59264c57d082d70061d68',
    ('sol_legacy', 'malformed_submit'): 'sha256:77d6d345e41ce0c258c029bbd775dc8dc4d3d2cbd95ed4772297b19bfb119505',
    ('historical_open', 'empty'): 'sha256:eda699e774b58be8fb368c574f507343bad88c3109a07ff860a92d43c9f03ed4',
    ('historical_open', 'malformed_submit'): 'sha256:f9839ee2663f72540bfd1759410266083a27f5a0d2500ccd9d739db55c0a03e1',
}


def run_controller(monkeypatch, variant, shape, finish, *, repeat=False, completion_delta=0, expired=False,
                   raw_calls=None):
    url = 'https://acme.example/'
    calls, events = [], []
    clock = SimpleNamespace(now=0.0)
    async def review(*_, **__):
        return {'target': 'industry', 'status': 'UNPROVEN', 'evidence_url': '', 'evidence_quote': '',
                '_industry_review': {'decision': 'match', 'missing_requirements': [],
                                     'reason': 'Fixture only', 'source_urls': [url], 'research_query': ''}}, '', 1
    async def post(_session, endpoint, *, headers, payload):
        calls.append((endpoint, deepcopy(payload)))
        if len(calls) > 1 and not repeat:
            return 400, {}
        reply = ({'choices': [{'finish_reason': finish, 'message': {'content': '', 'tool_calls': []}}]}
                 if shape == 'empty' else tool_reply('submit_findings', '{"findings": [', finish=finish))
        reply['usage'] = {'completion_tokens': payload['max_tokens'] + completion_delta}
        if expired:
            clock.now = investigator.ADMISSION_DEADLINE_SECONDS + investigator.SERIALIZATION_REPAIR_GRACE_SECONDS
        return 200, reply
    native_post = investigator._post_openrouter_json
    async def observe_native_post(*args, **kwargs):
        if raw_calls is not None:
            raw_calls.append(deepcopy(kwargs['payload']))
        return await native_post(*args, **kwargs)
    monkeypatch.setattr(investigator, '_review_industry_sources', review)
    monkeypatch.setattr(investigator, '_post_json', post)
    monkeypatch.setattr(investigator, '_post_openrouter_json', observe_native_post)
    monkeypatch.setattr(investigator, 'record_judge_tool_event', lambda event, **_: events.append(deepcopy(event)))
    monkeypatch.setattr(investigator, 'evaluation_date', lambda: date(2026, 11, 6))
    monkeypatch.setattr(investigator, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-placeholder')
    monkeypatch.setenv('EXA_API_KEY', 'offline-placeholder')
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', SOL)
    monkeypatch.setenv('ARENA_INDUSTRY_RESEARCH_CONTROLLER_MODEL', '' if variant != 'glm' else GLM)
    monkeypatch.setenv('ARENA_COMPANY_REVERIFY_MODEL', 'deepseek/deepseek-v4.1-flash-1017')
    enabled = variant != 'sol_legacy'
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', 'full_generated_thinking_v1' if enabled else '')
    monkeypatch.setenv('ARENA_JUDGE_REASONING_EXCEPTIONS', json.dumps(EXCEPTIONS))
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps(ROUTES))
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps(PREVIOUS_OPTIONS) if variant == 'glm' else '')
    monkeypatch.setattr(investigator, 'REASONING_MAX_TOKENS', 16384 if enabled else 3000)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={'name': 'Acme', 'website': url}, targets=('industry',), requested_industry='Software',
        prior_observations={'submitted_source_urls': [url]},
        verified_homepage_identity={'normalized_name': 'acme', 'registrable_dns_domain': 'acme.example'},
        prefetched_pages={url: {'final_url': url, 'text': 'Acme supplies software to companies.'}}))
    return result, calls, events


@pytest.mark.parametrize('variant,cap', [('glm', 8192), ('sol_legacy', 3000), ('historical_open', 16384)])
@pytest.mark.parametrize('shape', ['empty', 'malformed_submit'])
@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_native_budget_and_explicit_length_keep_one_existing_repair(monkeypatch, variant, cap, shape, finish):
    result, calls, events = run_controller(monkeypatch, variant, shape, finish)
    assert len(calls) == 2, (result, len(calls))
    assert all(payload['max_tokens'] == cap for _, payload in calls)
    assert contracts.document_hash(calls) == REQUEST_GOLDENS[variant, shape]
    if variant == 'glm':
        assert all(payload['model'] == GLM and payload['provider'] == {'only': ['near-ai/fp8']}
                   and payload['reasoning'] == {'effort': 'low', 'exclude': False} for _, payload in calls)
        assert events[0]['disposition'] == 'repair'
    assert investigator.MAX_INCOMPLETE_SUBMIT_RETRIES == 1
    assert result.get('_completed_submit') is not True


@pytest.mark.parametrize('shape', ['empty', 'malformed_submit'])
def test_repeated_glm_exhaustion_cannot_expand_the_repair_allowance(monkeypatch, shape):
    result, calls, _ = run_controller(monkeypatch, 'glm', shape, 'stop', repeat=True)
    assert len(calls) == 2
    assert result.get('_completed_submit') is not True


@pytest.mark.parametrize('shape', ['empty', 'malformed_submit'])
def test_effective_cap_preserves_original_native_payload_on_every_turn(monkeypatch, shape):
    raw_calls = []
    _, calls, _ = run_controller(monkeypatch, 'glm', shape, 'stop', raw_calls=raw_calls)
    assert [payload['max_tokens'] for payload in raw_calls] == [16384, 16384]
    assert [payload['max_tokens'] for _, payload in calls] == [8192, 8192]
    assert contracts.document_hash(calls) == REQUEST_GOLDENS['glm', shape]


@pytest.mark.parametrize('shape', ['empty', 'malformed_submit'])
@pytest.mark.parametrize('guard', ['below_cap', 'deadline'])
def test_effective_cap_does_not_relax_completion_or_deadline_guards(monkeypatch, shape, guard):
    result, calls, _ = run_controller(monkeypatch, 'glm', shape, 'stop',
        completion_delta=-1 if guard == 'below_cap' else 0, expired=guard == 'deadline')
    assert len(calls) == 1
    assert result.get('_completed_submit') is not True
