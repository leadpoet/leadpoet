"""Frozen judge opt-in reuses exact reasoning routing and existing persistence."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from lab_arena import broker as br, contracts, reasoning_capture as rc, service as arena_service
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.champion_baseline_reasoning_test import baseline_round
from tests.lab_arena.reasoning_capture_test import _execute, _reviewed_registry


def test_judge_frozen_opt_in_pins_route_and_saves_same_annotation_as_source():
    registry = _reviewed_registry()
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry)
    result, store, transport = _execute(context)
    assert result.status == 200
    sent = json.loads(transport.sent[0]['body'])
    assert sent['reasoning'] == {'effort': 'high', 'exclude': False}
    assert sent['messages'] == bt.CHAT['messages']
    assert sent['provider']['only'] == ['example/verified']
    assert sent['provider']['zdr'] is True
    assert sent['provider']['allow_fallbacks'] is False
    assert sent['provider']['require_parameters'] is True
    normalized = br.normalized_request('openrouter.chat', bt.CHAT,
        reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
    assert result.call['request_hash'] == contracts.document_hash(normalized)
    saved = store.calls[result.call['call_identity']]['call_doc']['reasoning_capture']
    assert saved['requested'] is True and saved['reviewed_route'] is True
    assert saved['registry_sha256'] == contracts.document_hash(registry).split(':', 1)[1]
    assert result.call['funding_source'] == 'host'


def test_source_thinking_policy_does_not_change_legacy_score_requests():
    legacy, _, legacy_transport = _execute(replace(bt.CONTEXT, kind='score'))
    round_with_thinking = replace(bt.CONTEXT, kind='score', reasoning_capture_policy=rc.POLICY,
                                  reasoning_route_registry=_reviewed_registry())
    unchanged, store, transport = _execute(round_with_thinking)
    assert legacy.status == unchanged.status == 200
    assert transport.sent[0]['body'] == legacy_transport.sent[0]['body']
    assert legacy.call['request_hash'] == unchanged.call['request_hash']
    assert 'reasoning_capture' not in store.calls[unchanged.call['call_identity']]['call_doc']


def test_unreviewed_judge_route_is_refused_before_provider_or_reservation():
    registry = _reviewed_registry(model='other/model')
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry)
    result, store, transport = _execute(context)
    assert result.status == 400
    assert not transport.sent and not store.calls


def test_service_binds_judge_to_round_not_promoted_baseline_capture():
    row, baseline = baseline_round({})
    row['configuration_doc']['scorer_policy']['env_bindings']['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] = rc.POLICY
    run = dict(baseline, run_id='run', round_id=row['round_id'], assignment_id='assignment',
               attempt=1, icp_position=0, stage=1, kind='score')
    service = object.__new__(arena_service.ArenaService)
    service._store = SimpleNamespace(get_run=lambda _id: run)
    service._require_round_ownership = lambda _id: None
    service._hot_round = lambda _id: row
    _, context = service._run_context('run', 'token')
    assert context.judge_reasoning_capture_policy == rc.POLICY
    assert context.judge_reasoning_route_registry == row['configuration_doc']['reasoning_route_registry']
    run['kind'] = 'execute'
    _, source = service._run_context('run', 'token')
    assert source.reasoning_capture_policy is None
    assert br._reasoning_contract(source) == (None, None)


@pytest.mark.parametrize('model_id,filename', [
    ('deepseek/deepseek-v4.1-flash-1017', 'deepseek--deepseek-v4.1-flash-1017-chat-probe-price-bound.json'),
    ('mistralai/mistral-large-4-0', 'mistralai--mistral-large-4-0-chat-probe-price-bound.json'),
])
def test_saved_real_provider_tool_continuations_pass_as_judge(model_id, filename):
    from tests.lab_arena.full_thinking_saved_probe_test import historical_registry
    from tests.lab_arena.reasoning_capture_test import EVIDENCE_DIR
    proof = json.loads((EVIDENCE_DIR / filename).read_text())
    registry = historical_registry()
    route = next(row for row in registry['approved_routes'] if row['model_id'] == model_id)
    table = bt.price_table()
    table['models'][model_id] = {component: route['price_ceiling'].get(component, '0')
                               for component in br.PRICED_COMPONENTS}
    store = bt.FakeLedgerStore()
    transport = bt.FakeTransport([(row['status'], row['response']) for row in proof['attempts']])
    broker = br.Broker(store=store, key_for=lambda provider: bt.HOST_KEYS[provider],
                       price_table=table, judge_models=(model_id,), transport=transport)
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry)
    for index, attempt in enumerate(proof['attempts']):
        request = attempt['request']
        result = broker.execute(context, operation_id='openrouter.chat',
            parameters={k: v for k, v in request.items() if k not in {'provider', 'stream'}},
            action_sequence=index, timeout_ms=30000)
        assert result.status == 200
        assert json.loads(transport.sent[-1]['body']) == request
        assert json.loads(result.body) == attempt['response']
        annotation = store.calls[result.call['call_identity']]['call_doc']['reasoning_capture']
        assert annotation['provider_route'] == route['provider']
        observation = br.broker_observations(store.calls[result.call['call_identity']]['terminal'])
        assert observation['reasoning_route_observation']['status'] == 'matched'


@pytest.mark.parametrize('interface', ['frame', 'http'])
@pytest.mark.parametrize('enabled', [False, True])
def test_worker_accepts_long_judge_history_only_with_frozen_opt_in(tmp_path, interface, enabled):
    from lab_arena import runner, shim, training_capture
    from scripts.verify_arena_live_capture import BrokerSocketApi
    from tests.lab_arena.full_thinking_continuation_limit_test import broker_fixture, continuation
    from tests.lab_arena.test_lab_arena_runner import lease
    broker, store, transport, context, objects = broker_fixture()
    broker._judge_models = (bt.CHAT['model'],)
    context = replace(context, kind='score', judge_reasoning_capture_policy=rc.POLICY if enabled else None,
                      judge_reasoning_route_registry=context.reasoning_route_registry)
    api = BrokerSocketApi(broker, context)
    state = runner.RunState(lease=dict(lease(), kind='score', judge_reasoning_capture_policy=rc.POLICY if enabled else None),
                            lease_token='host-token')
    worker = runner.WorkerSocketServer(tmp_path / 'worker.sock', api, state)
    parameters = continuation('reasoning', 160_000)
    if interface == 'frame':
        reply = json.loads(worker.handle_frame(shim.build_operation_frame('openrouter.responses', parameters, 30000)))
        status = reply.get('status', 400)
    else:
        status, _, _ = worker.handle_http('POST', 'https://openrouter.ai/api/v1/responses',
                                         json.dumps(parameters).encode(), {'content-type': 'application/json'})
    if not enabled:
        assert status == 400
        assert not transport.sent
        return
    assert status == 200
    assert len(transport.sent) == 1
    saved = training_capture.read_request(objects, next(iter(store.calls.values()))['call_doc']['request_capture'])
    assert saved['admitted_parameters']['input'] == parameters['input']
    assert saved['outbound']['body']['input'] == parameters['input']
    assert saved['request_hash'] == contracts.document_hash(saved['hashed_parameters'])


def test_worker_source_policy_cannot_enable_a_legacy_judge():
    from lab_arena import runner
    assert not runner._full_thinking_lease({'kind': 'score', 'reasoning_capture_policy': rc.POLICY})
    assert not runner._full_thinking_lease({'reasoning_capture_policy': rc.POLICY})
    assert runner._full_thinking_lease({'kind': 'execute', 'reasoning_capture_policy': rc.POLICY})


@pytest.mark.parametrize('enabled', [False, True])
def test_trusted_judge_consumption_reaches_existing_worker_receipt_path(monkeypatch, enabled):
    import tempfile
    from pathlib import Path
    from lab_arena import runner, shim
    from scripts.verify_arena_live_capture import BrokerSocketApi
    from tests.lab_arena.full_thinking_continuation_limit_test import broker_fixture, continuation
    broker, store, transport, context, objects = broker_fixture()
    broker._judge_models = (bt.CHAT['model'],)
    context = replace(context, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=context.reasoning_route_registry)
    state = runner.RunState(lease={'run_id': context.run_id, 'kind': 'score',
        'judge_reasoning_capture_policy': rc.POLICY if enabled else None}, lease_token='test-token')
    with tempfile.TemporaryDirectory(prefix='judge-receipt-', dir='/tmp') as directory:
        worker = runner.WorkerSocketServer(Path(directory) / 'w.sock', BrokerSocketApi(broker, context), state)
        worker.start()
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(Path(directory) / 'w.sock'))
        monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, '1')
        monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', rc.POLICY if enabled else '')
        try:
            status, headers, body = shim.dispatch('openrouter.responses', continuation('reasoning', 10), 1000)
            assert status == 200
        finally:
            worker.stop()
    reports = state.worker_observations.agent_observations
    assert len(reports) == int(enabled)
    if enabled:
        assert reports[0]['kind'] == 'response_received'
        assert reports[0]['data']['call_identity'] == headers[shim.CALL_IDENTITY_HEADER]
        assert reports[0]['data']['body_sha256'] == contracts.hash_bytes(body)
        assert reports[0]['data']['body_bytes'] == len(body)


@pytest.mark.parametrize('enabled', [False, True])
def test_gateway_frame_size_gate_uses_frozen_judge_opt_in(enabled):
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    from tests.lab_arena.full_thinking_continuation_limit_test import api_service, continuation
    service, transport, reads = api_service(full=True, kind='score')
    broker = service._broker_for('round-test')
    broker._judge_models = (bt.CHAT['model'],)
    original = service._run_context

    def context(run_id, token):
        run, value = original(run_id, token)
        return run, replace(value, judge_reasoning_capture_policy=rc.POLICY if enabled else None,
                            judge_reasoning_route_registry=value.reasoning_route_registry)

    service._run_context = context
    frame = {'operation_id': 'openrouter.responses', 'parameters': continuation('arguments', 160_000, glyph='é'),
             'timeout_ms': 1000, 'action_sequence': 0}
    with TestClient(create_app(service)) as client:
        answer = client.post('/arena/v1/runs/r1/provider', content=contracts.canonical_json(frame),
                             headers={'x-lab-arena-lease': 'a' * 64})
    assert len(reads) == 1
    assert answer.status_code == (200 if enabled else 400)
    assert len(transport.sent) == int(enabled)
    if enabled:
        assert answer.json()['status'] == 200
        assert json.loads(transport.sent[0]['body'])['input'] == frame['parameters']['input']
