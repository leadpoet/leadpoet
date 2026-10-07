"""Confirmed tool refusals cannot consume a healthy operation's local guard."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from lab_arena import contracts, runner as rn


class OperationApi:
    def __init__(self, refusal='miner_credentials_unavailable'):
        self.refusal = refusal
        self.frames = []
        self.lock = threading.Lock()
        self.budget_spent = False

    def provider(self, run_id, token, frame):
        with self.lock:
            self.frames.append(frame)
        bad_tool = frame['operation_id'] == 'exa.search' or (
            frame['operation_id'] == 'deepline.execute'
            and frame['parameters'].get('tool') == 'exa_search')
        error = 'budget_refused' if self.budget_spent else self.refusal if bad_tool else None
        call = {'operation_id': frame['operation_id'], 'action_sequence': frame['action_sequence'],
                'call_identity': contracts.document_hash(frame), 'funding_source': 'miner_key',
                'outcome': 'settled', 'actual_microusd': 0, 'provider_status': 200}
        if error:
            call.update(error_code=error, provider_status=402)
            body = {'error': {'code': error}}
        else:
            body = {'results': []}
        if self.budget_spent:
            call.update(outcome='refused', reason='execution_cap_exceeded')
        return {'status': 402 if error else 200, 'headers': {'content-type': 'application/json'},
                'body_b64': base64.b64encode(json.dumps(body).encode()).decode(), 'call': call}


def worker(tmp_path, api):
    return rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
                                rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'))


def execute(w, tool='exa_search', operation='deepline.execute'):
    return w._dispatch_once(operation, {'tool': tool, 'payload': {'query': 'original query'}}, 1000)


def exhaust_exa(w):
    for _ in range(rn.MAX_REFUSED_FRAMES):
        error, doc = execute(w)
        assert error is None and doc['call']['error_code'] == 'miner_credentials_unavailable'


def test_exa_credit_refusals_preserve_healthy_providers_and_free_tools(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    exhaust_exa(w)
    assert execute(w) == ('miner_credentials_unavailable', None)
    for tool, operation in [('contextdev_post_web_search', 'deepline.execute'),
                            ('exa_contents', 'deepline.execute'),
                            ('', 'openrouter.chat'), ('', 'openrouter.responses')]:
        error, doc = execute(w, tool, operation)
        assert error is None and doc['status'] == 200
    assert len(api.frames) == rn.MAX_REFUSED_FRAMES + 4
    assert len(w._state.calls) == len(api.frames)
    assert w._state.refusals == rn.MAX_REFUSED_FRAMES
    assert w._state.operation_refusals == {('deepline.execute', 'exa_search'): (25, 'miner_credentials_unavailable')}


def test_exa_compatibility_alias_cannot_bypass_the_tool_guard(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    exhaust_exa(w)
    assert execute(w, operation='exa.search') == ('miner_credentials_unavailable', None)
    error, doc = execute(w, tool='exa_contents', operation='exa.contents')
    assert error is None and doc['status'] == 200
    assert len(api.frames) == rn.MAX_REFUSED_FRAMES + 1


@pytest.mark.parametrize('code', ['budget_refused', 'budget_exhausted', 'miner_provider_not_configured'])
def test_same_refused_operation_is_bounded_with_its_actual_error(tmp_path, code):
    api = OperationApi(refusal=code)
    w = worker(tmp_path, api)
    for _ in range(rn.MAX_REFUSED_FRAMES):
        assert execute(w)[0] is None
    assert execute(w) == (code, None)
    assert len(api.frames) == rn.MAX_REFUSED_FRAMES
    assert len(w._state.calls) == rn.MAX_REFUSED_FRAMES


def test_http_guard_preserves_credential_failure_status_and_body(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    exhaust_exa(w)
    status, headers, body = w.handle_http('POST', 'https://api.exa.ai/search',
                                         json.dumps({'query': 'original query'}).encode(), {})
    assert status == 402 and headers == {}
    assert json.loads(body) == {'error': {'code': 'miner_credentials_unavailable'}}
    assert len(api.frames) == rn.MAX_REFUSED_FRAMES


def test_real_global_budget_is_still_enforced_by_the_gateway(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    exhaust_exa(w)
    api.budget_spent = True
    for tool, operation in [('contextdev_post_web_search', 'deepline.execute'), ('', 'openrouter.chat')]:
        error, doc = execute(w, tool, operation)
        assert error is None
        assert doc['status'] == 402 and doc['call']['error_code'] == 'budget_refused'
        assert doc['call']['reason'] == 'execution_cap_exceeded'
    assert len(api.frames) == rn.MAX_REFUSED_FRAMES + 2


def test_unknown_nonrefusal_errors_do_not_build_a_refusal_guard(tmp_path):
    api = OperationApi(refusal='provider_unavailable')
    w = worker(tmp_path, api)
    for _ in range(rn.MAX_REFUSED_FRAMES + 2):
        assert execute(w)[0] is None
    assert w._state.operation_refusals == {} and w._state.refusals == 0
    assert len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 2


def test_concurrent_operation_counters_and_action_ids_are_independent(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    # Stop below the refusal boundary: every admitted response must settle and
    # retain its own identity, even while other operation responses complete.
    requests = [('exa_search', 'deepline.execute')] * 20 + [('contextdev_post_web_search', 'deepline.execute')] * 20 + [('', 'openrouter.chat')] * 20
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda args: execute(w, *args), requests))
    assert all(error is None for error, _doc in results)
    assert len(api.frames) == len(w._state.calls) == 60
    assert sorted(frame['action_sequence'] for frame in api.frames) == list(range(60))
    assert len({call['call_identity'] for call in w._state.calls}) == 60
    assert w._state.operation_refusals == {('deepline.execute', 'exa_search'): (20, 'miner_credentials_unavailable')}
    for _ in range(5):
        assert execute(w)[0] is None
    assert execute(w) == ('miner_credentials_unavailable', None)
    assert execute(w, 'contextdev_post_web_search')[1]['status'] == 200
    assert len(api.frames) == len(w._state.calls) == 66


def test_inflight_calls_keep_their_original_accounting_at_the_guard_boundary(tmp_path):
    api = OperationApi()
    w = worker(tmp_path, api)
    for _ in range(rn.MAX_REFUSED_FRAMES - 1):
        assert execute(w)[0] is None
    barrier = threading.Barrier(8)
    provider = api.provider

    def concurrent_provider(*args):
        # Every call passed the local guard before any of these replies arrive.
        barrier.wait(timeout=5)
        return provider(*args)

    api.provider = concurrent_provider
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: execute(w), range(8)))
    assert all(error is None for error, _doc in results)
    assert len(api.frames) == len(w._state.calls) == 32
    assert sorted(frame['action_sequence'] for frame in api.frames) == list(range(32))
    assert len({call['call_identity'] for call in w._state.calls}) == 32
    assert w._state.operation_refusals[('deepline.execute', 'exa_search')][0] == 32
    assert execute(w) == ('miner_credentials_unavailable', None)
    assert len(api.frames) == len(w._state.calls) == 32
