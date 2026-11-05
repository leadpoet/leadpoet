"""Reasoning reviews share one bounded window; legacy limits are unchanged."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details as review, verification_helpers as helpers
from qualification.scoring import openrouter_options as options
from tests.test_intent_details_complete_protocol_correction import captured


@pytest.mark.parametrize('enabled', [False, True])
def test_reasoning_http_timeout_is_bounded_and_opt_in(monkeypatch, enabled):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({'test/model': 'example/verified'}))
    timeouts = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            timeouts.append(kwargs['timeout'])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})

    monkeypatch.setattr(helpers.httpx, 'AsyncClient', Client)
    assert asyncio.run(helpers.openrouter_chat('Check', 'test/model', api_key='test')) == '{}'
    assert timeouts == [120 if enabled else helpers.LLM_TIMEOUT]


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('elapsed', [40, 130])
def test_existing_repair_uses_remaining_window_not_a_new_deadline(monkeypatch, enabled, elapsed):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    document, initial = captured('AHF')
    fresh = copy.deepcopy(initial)
    fresh['facts_supported'] = True
    monkeypatch.setattr(review, 'review_evidence', lambda *a, **k: copy.deepcopy(document))
    clock, calls, timeouts = [100.0], [], []

    async def judge(*args, **kwargs):
        calls.append(kwargs)
        clock[0] += elapsed
        return json.dumps(initial if len(calls) == 1 else fresh)

    async def wait_for(awaitable, *, timeout):
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(helpers, 'openrouter_chat', judge)
    monkeypatch.setattr(review, 'asyncio', SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: clock[0]),
        wait_for=wait_for, TimeoutError=asyncio.TimeoutError))
    result = asyncio.run(review.review_intent_details(None, None, [], {}))
    total = 120 if enabled else review.REVIEW_TIMEOUT_SECONDS
    assert timeouts == ([total, total - elapsed] if elapsed < total else [total])
    assert len(calls) == (2 if elapsed < total else 1)
    assert all(call['max_retries'] == 0 for call in calls)
    assert result['decision'] == ('mismatch' if elapsed < total else 'unavailable')


@pytest.mark.parametrize('enabled', [False, True])
def test_industry_response_contract_is_explicit_only_for_reasoning_policy(monkeypatch, enabled):
    import time
    from qualification.scoring import company_evidence_investigator as investigator
    from qualification.scoring import industry_classification as industry

    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    calls = []

    async def capture(session, url, **kwargs):
        calls.append(kwargs['payload'])
        return 503, {}

    monkeypatch.setattr(investigator, '_post_openrouter_json', capture)
    url = 'https://acme.example/'
    asyncio.run(investigator._review_industry_sources(
        None, model='test/model', openrouter_key='test', started=time.monotonic(),
        company_locator={'name': 'Acme', 'website': url},
        criteria={'industry': 'Software', 'subindustry': '', 'product_service': 'Reporting'},
        fetched_pages={url: 'Acme supplies reporting software.'}, fetched_final_urls={url: url},
        first_party_domains={'acme.example'}, identity_names={'acme'},
        identity_anchor={'submitted_domain': 'acme.example', 'observed_domain': 'acme.example',
                         'verified_domain': 'acme.example'}, evaluation_date_value='2026-11-05'))
    assert len(calls) == 1
    assert (industry.REASONING_RESPONSE_CONTRACT in calls[0]['messages'][0]['content']) is enabled
    assert calls[0]['max_tokens'] == (16384 if enabled else 2000)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_claim_comparison_keeps_500_token_15_second_bound(monkeypatch, enabled, finish):
    from qualification.scoring import claim_equivalence
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY if enabled else '')
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({'test/model': 'example/verified'}))
    monkeypatch.setattr(helpers, 'OPENROUTER_API_KEY', 'test-not-secret')
    calls = []

    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'choices': [{'finish_reason': finish, 'message': {'content':
                    '{"decision":"equivalent","material_differences":[]}'}}]})

    monkeypatch.setattr(helpers.httpx, 'AsyncClient', Client)
    source = {'description': 'Acme partnered with Wiz.'}
    current = {'description': 'Wiz and Acme partnered.'}
    result = asyncio.run(claim_equivalence.verify_claim_equivalence(source, current, model='test/model'))
    assert len(calls) == 1
    assert calls[0]['json']['max_tokens'] == 500
    assert calls[0]['timeout'] == claim_equivalence.TIMEOUT_SECONDS == 15
    if enabled:
        assert calls[0]['json']['provider'] == {'only': ['example/verified']}
        assert calls[0]['json']['reasoning']['exclude'] is False
        assert (result is None) == (finish == 'length')
    else:
        assert calls[0]['json']['provider'] == {'data_collection': 'deny', 'zdr': True}
        assert 'reasoning' not in calls[0]['json']
