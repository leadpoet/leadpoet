"""Incomplete reasoning output can use only the existing identical-request budget."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import openrouter_options as options
from tests.test_intent_three_stage_json_retry import _Client, _Response

MODEL = 'openai/gpt-oss-120b'
FIXTURE = json.loads((Path(__file__).parent/'fixtures/intent_length_retry_recovered.json').read_text())


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-not-a-secret')
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY)
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: 'near-ai/fp4'}))
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps({MODEL: {
        'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096}}))
    monkeypatch.delenv('ARENA_JUDGE_REASONING_EXCEPTIONS', raising=False)
    sleeps = []
    async def sleep(seconds):sleeps.append(seconds)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    return sleeps


def call(responses, **kwargs):
    client = _Client([_Response(deepcopy(r)) for r in responses])
    result = asyncio.run(verifier._call_openrouter(client, MODEL, 'Judge this exact source.', **kwargs))
    return result, client


def test_captured_length_shape_gets_fresh_complete_output_in_same_request_budget(enabled):
    result, client = call([FIXTURE['incomplete'], FIXTURE['complete']])
    assert client.calls == 2 and enabled == [1]
    assert result['answer'] == json.loads(FIXTURE['complete']['choices'][0]['message']['content'])
    first, second = [r['kwargs']['json'] for r in client.requests]
    assert first == second
    assert first['max_tokens'] == 4096 and first['provider'] == {'only': ['near-ai/fp4']}
    assert first['response_format']['json_schema']['strict'] is True


@pytest.mark.parametrize('maximum, expected', [(0, 1), (1, 1), (2, 2), (3, 3), (99, 3)])
def test_length_exhaustion_keeps_terminal_error_and_attempt_cap(enabled, maximum, expected):
    result, client = call([FIXTURE['incomplete']]*3, max_attempts=maximum)
    assert client.calls == expected and enabled == [1]*(expected-1)
    assert result['_error'] == 'incomplete_response' and 'answer' not in result


@pytest.mark.parametrize('finish', [None, 'content_filter', 'tool_calls', 'refusal', 'unknown'])
def test_other_incomplete_finish_reasons_remain_terminal(enabled, finish):
    response = deepcopy(FIXTURE['incomplete'])
    response['choices'][0]['finish_reason'] = finish
    result, client = call([response])
    assert client.calls == 1 and not enabled
    assert result['_error'] == 'incomplete_response' and 'answer' not in result


def test_length_and_malformed_content_share_one_budget(enabled):
    malformed = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{'}}]}
    result, client = call([FIXTURE['incomplete'], malformed, FIXTURE['complete']])
    assert client.calls == 3 and enabled == [1, 1] and 'answer' in result
    assert all(r['kwargs']['json'] == client.requests[0]['kwargs']['json'] for r in client.requests)


def test_legacy_finished_json_behavior_is_unchanged(enabled, monkeypatch):
    monkeypatch.delenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY')
    response = deepcopy(FIXTURE['complete']);response['choices'][0]['finish_reason'] = 'length'
    result, client = call([response])
    assert client.calls == 1 and 'answer' in result and not enabled


def test_finished_negative_verdict_does_not_trigger_a_fresh_judgment(enabled):
    response = deepcopy(FIXTURE['complete'])
    answer = json.loads(response['choices'][0]['message']['content'])
    answer['overall_verdict'] = 'disqualified'
    answer['signal_evaluations'][0]['signal_status'] = 'contradicted'
    response['choices'][0]['message']['content'] = json.dumps(answer)
    result, client = call([response])
    assert client.calls == 1 and not enabled and result['answer'] == answer


def test_production_cache_keeps_length_failure_unshared(enabled):
    from lab_arena import broker
    result, client = call([FIXTURE['incomplete'], FIXTURE['complete']])
    request = client.requests[0]['kwargs']['json']
    for response, expected in [(FIXTURE['incomplete'], False), (FIXTURE['complete'], True)]:
        assert broker._complete_judgment_response(
            'openrouter.chat', request, status=200,
            body=json.dumps(response).encode(), call_succeeded=True,
        ) is expected
