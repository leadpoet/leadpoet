"""The company-fit request uses the existing opt-in reasoning deadline.

Synthetic durations exercise caller deadline selection; they do not predict
whether a real provider generation will complete within the selected window.
"""
import asyncio
import copy
import json

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring import openrouter_options as options

MODEL = 'deepseek/deepseek-v4.1-flash-1017'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('elapsed', [20, 60, 121])
def test_company_fit_uses_shared_reasoning_deadline_without_request_changes(monkeypatch, enabled, elapsed):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: 'nebius/fp8'}))
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps({MODEL: {
        'reasoning': {'effort': 'low'}, 'max_output_tokens': 8192}}))
    monkeypatch.setattr(scorer, '_SCORER_REVERIFY_MODEL', MODEL)
    expected_path, expected_body = options.prepare_judge_request({
        'model': MODEL,
        'messages': [{'role': 'system', 'content': scorer._SCORER_REVERIFY_SYSTEM_PROMPT},
                     {'role': 'user', 'content': 'Verify existing source evidence.'}],
        'temperature': 0.0,
    }, hosted_search=True)
    calls, timeouts = [], []

    class Response:
        status = 503  # No synthetic company verdict or qualification is supplied.
        async def __aenter__(self):
            if elapsed > timeouts[-1]:
                raise TimeoutError
            return self
        async def __aexit__(self, *args):
            pass

    class Session:
        def __init__(self, *, timeout):
            timeouts.append(timeout.total)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, url, **kwargs):
            calls.append((url, copy.deepcopy(kwargs)))
            return Response()

    monkeypatch.setattr(scorer.aiohttp, 'ClientSession', Session)
    diagnostic = {}
    verdict, error = asyncio.run(scorer._request_company_reverify_json(
        key='offline-test-only', prompt='Verify existing source evidence.',
        telemetry_purpose='test', diagnostic=diagnostic))
    expected_timeout = options.JUDGE_REASONING_TIMEOUT_SECONDS if enabled else scorer._SCORER_REVERIFY_TIMEOUT_S
    assert timeouts == [expected_timeout]
    assert len(calls) == 1  # The change adds no retry.
    assert calls[0] == ('https://openrouter.ai/api/v1' + expected_path, {
        'json': expected_body,
        'headers': {'Authorization': 'Bearer offline-test-only', 'Content-Type': 'application/json'},
    })
    assert verdict is None
    assert error == ('provider HTTP 503' if elapsed <= expected_timeout else 'provider transport error: TimeoutError')
    assert scorer.PROVIDER_ERROR_FAILURE_REASON in diagnostic.values()


def test_company_fit_preserves_reasoning_model_exception_refusal(monkeypatch):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY)
    monkeypatch.setenv('ARENA_JUDGE_REASONING_EXCEPTIONS', json.dumps({
        'openai/gpt-6.1-sol': {'roles': ['industry_evidence_review'], 'reason': 'quality_gate_not_met'}}))
    monkeypatch.setattr(scorer, '_SCORER_REVERIFY_MODEL', 'openai/gpt-6.1-sol')
    def forbidden(**kwargs):
        raise AssertionError('Unsupported hosted-search model must not create a client')
    monkeypatch.setattr(scorer.aiohttp, 'ClientSession', forbidden)
    with pytest.raises(ValueError, match='judge_proprietary_exception_api_unsupported'):
        asyncio.run(scorer._request_company_reverify_json(
            key='offline-test-only', prompt='Verify', telemetry_purpose='test'))
