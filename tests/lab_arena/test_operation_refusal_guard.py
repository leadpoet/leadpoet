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


def test_expired_cooldown_allows_one_requested_probe_and_success_reopens_tool(tmp_path):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    assert execute(w) == ('miner_credentials_unavailable', None)
    assert len(api.frames) == 25
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    api.refusal = None
    assert execute(w)[1]['status'] == 200
    assert execute(w)[1]['status'] == 200
    assert len(api.frames) == len(w._state.calls) == 27
    assert w._state.operation_refusals == {}
    assert w._state.local_refusals == 1


@pytest.mark.parametrize('status,error_code', [
    (400, None), (404, None), (422, None),
    (403, 'provider_request_refused'),
])
def test_definitive_request_error_clears_stale_credential_guard(
    tmp_path, status, error_code,
):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    api.refusal = None
    original_provider = api.provider

    def terminal_probe(*args):
        document = original_provider(*args)
        if len(api.frames) == rn.MAX_REFUSED_FRAMES + 1:
            document['status'] = status
            document['call'].update(
                provider_status=status, outcome='uncertain',
                actual_microusd=None,
            )
            if error_code:
                document['call']['error_code'] = error_code
            document['body_b64'] = base64.b64encode(json.dumps({
                'error': {'code': error_code or 'request_failed'},
            }).encode()).decode()
        return document

    api.provider = terminal_probe
    assert execute(w)[1]['status'] == status
    assert w._state.operation_refusals == {}
    # This is a new model-requested call, not an automatic retry of the probe.
    assert execute(w)[1]['status'] == 200
    assert len(api.frames) == len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 2
    assert w._state.refusals == rn.MAX_REFUSED_FRAMES


@pytest.mark.parametrize('status,error_code,outcome', [
    (401, 'miner_credentials_unavailable', 'uncertain'),
    (402, 'budget_refused', 'refused'),
    (403, None, 'uncertain'),
    (408, None, 'uncertain'),
    (429, 'provider_unavailable', 'uncertain'),
    (502, 'provider_unavailable', 'uncertain'),
    (404, 'broker_unavailable', 'unknown'),
])
def test_nondefinitive_probe_keeps_guard_and_does_not_retry(
    tmp_path, status, error_code, outcome,
):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    api.refusal = None
    original_provider = api.provider

    def failed_probe(*args):
        document = original_provider(*args)
        document['status'] = status
        document['call'].update(
            provider_status=status, outcome=outcome, actual_microusd=None,
        )
        if error_code:
            document['call']['error_code'] = error_code
        document['body_b64'] = base64.b64encode(json.dumps({
            'error': {'code': error_code or 'request_failed'},
        }).encode()).decode()
        return document

    api.provider = failed_probe
    assert execute(w)[1]['status'] == status
    assert execute(w) == ('miner_credentials_unavailable' if outcome != 'refused' else 'budget_refused', None)
    assert len(api.frames) == len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 1
    assert w._state.operation_refusals


def test_failed_probe_rearms_from_completion_and_does_not_replay(tmp_path):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    original_provider = api.provider

    def delayed_provider(*args):
        clock[0] += 5.0
        return original_provider(*args)

    api.provider = delayed_provider
    assert execute(w)[1]['status'] == 402
    assert w._state.operation_refusal_until[('deepline.execute', 'exa_search')] == 165.0
    assert execute(w) == ('miner_credentials_unavailable', None)
    assert len(api.frames) == 26
    clock[0] = 165.0
    assert execute(w)[1]['status'] == 402
    assert len(api.frames) == len(w._state.calls) == 27


def test_only_one_concurrent_probe_is_dispatched_after_cooldown(tmp_path):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    entered, release = threading.Event(), threading.Event()
    original_provider = api.provider

    def blocked_provider(*args):
        entered.set()
        assert release.wait(timeout=5)
        return original_provider(*args)

    api.provider = blocked_provider
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(execute, w)
        assert entered.wait(timeout=5)
        assert execute(w) == ('miner_credentials_unavailable', None)
        assert len(api.frames) == 25
        release.set()
        assert first.result(timeout=5)[1]['status'] == 402
    assert len(api.frames) == len(w._state.calls) == 26


@pytest.mark.parametrize('recovery_status', [200, 404])
def test_older_refusal_cannot_overwrite_later_same_tool_recovery(
    tmp_path, recovery_status,
):
    api = OperationApi()
    w = worker(tmp_path, api)
    for _ in range(rn.MAX_REFUSED_FRAMES - 1):
        assert execute(w)[1]['status'] == 402
    entered, release = threading.Event(), threading.Event()
    original_provider = api.provider

    def out_of_order_provider(*args):
        if args[2]['action_sequence'] == rn.MAX_REFUSED_FRAMES - 1:
            entered.set()
            assert release.wait(timeout=5)
            previous_refusal = api.refusal
            try:
                api.refusal = 'miner_credentials_unavailable'
                return original_provider(*args)
            finally:
                api.refusal = previous_refusal
        document = original_provider(*args)
        if recovery_status == 404:
            document['status'] = 404
            document['call'].update(
                provider_status=404, outcome='uncertain', actual_microusd=None,
            )
        return document

    api.provider = out_of_order_provider
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(execute, w)
        assert entered.wait(timeout=5)
        api.refusal = None
        assert execute(w)[1]['status'] == recovery_status
        release.set()
        assert old.result(timeout=5)[1]['status'] == 402
    key = ('deepline.execute', 'exa_search')
    assert w._state.operation_latest_calls[key]['action_sequence'] == rn.MAX_REFUSED_FRAMES
    assert w._state.operation_refusals == {}
    assert len(api.frames) == len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 1


def test_malformed_probe_releases_slot_and_records_unknown_broker_failure(tmp_path):
    clock = [100.0]
    api = OperationApi()
    w = rn.WorkerSocketServer(tmp_path / 'worker.sock', api,
        rn.RunState(lease={'run_id': 'r1', 'kind': 'execute'}, lease_token='token'),
        monotonic=lambda: clock[0])
    exhaust_exa(w)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    api.provider = lambda *_args: {}
    assert execute(w) == ('worker_unavailable', None)
    key = ('deepline.execute', 'exa_search')
    assert not w._state.operation_probe_inflight
    assert w._state.operation_latest_calls[key]['error_code'] == 'broker_unavailable'
    assert w._state.calls[-1]['outcome'] == 'unknown'
    assert len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 1
    assert execute(w) == ('miner_credentials_unavailable', None)
    clock[0] += rn.REFUSAL_PROBE_COOLDOWN_SECONDS
    assert execute(w) == ('worker_unavailable', None)
    assert len(w._state.calls) == rn.MAX_REFUSED_FRAMES + 2
