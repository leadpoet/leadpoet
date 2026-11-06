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

SOL = 'openai/gpt-6.1-sol'
GLM = 'z-ai/glm-5.3-flash'

# Complete explicit-length request sequences from parent c4bde26, including
# the unchanged existing repair instruction. Budget-only repairs must match.
REQUEST_GOLDENS = {
    ('glm', 'empty'): 'sha256:ac6f28237560888c2a2992ee0be3ea2d2a2f2a7267483c02a13efd60461d0876',
    ('glm', 'malformed_submit'): 'sha256:794dedbd135130775b0e13bd9ceec03e900bb52965aefaf1e420cd1845b98b18',
    ('sol_legacy', 'empty'): 'sha256:e227de00da28ac0b5b2d1a4eb46b7f492c92488f2356e7c6d9cc0419b67f84e7',
    ('sol_legacy', 'malformed_submit'): 'sha256:6492aff7542d6563a0e429478714d5c13fdab372e493fdc9f369beba76d3c6b9',
    ('historical_open', 'empty'): 'sha256:1cdec36d703b8571278650112c84371614ec51a0b08e3ec928751942feefd275',
    ('historical_open', 'malformed_submit'): 'sha256:6fe708a6cf9babdacdf86408250993919e66e6beb2fd7adfceaf147c150cedc8',
}


def run_controller(monkeypatch, variant, shape, finish, *, repeat=False, completion_delta=0, expired=False):
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
    monkeypatch.setattr(investigator, '_review_industry_sources', review)
    monkeypatch.setattr(investigator, '_post_json', post)
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
    monkeypatch.setenv('ARENA_JUDGE_REASONING_EXCEPTIONS', json.dumps(scoring.DEFAULT_JUDGE_REASONING_EXCEPTIONS))
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps(scoring.DEFAULT_JUDGE_PROVIDER_ROUTES))
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps(scoring.DEFAULT_JUDGE_REQUEST_OPTIONS) if variant == 'glm' else '')
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
@pytest.mark.parametrize('guard', ['below_cap', 'deadline'])
def test_effective_cap_does_not_relax_completion_or_deadline_guards(monkeypatch, shape, guard):
    result, calls, _ = run_controller(monkeypatch, 'glm', shape, 'stop',
        completion_delta=-1 if guard == 'below_cap' else 0, expired=guard == 'deadline')
    assert len(calls) == 1
    assert result.get('_completed_submit') is not True
