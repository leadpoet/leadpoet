"""A missing mandatory privacy route is infrastructure, not a caller/model 404."""
from dataclasses import replace
import base64
import copy
import json
import os

import pytest

from lab_arena import broker as br, contracts, operations, runner, runtime, shim
from tests.lab_arena import test_lab_arena_broker as broker_tests
from tests.lab_arena import test_lab_arena_runner as runner_tests

MODEL = 'google/gemini-2.5-flash'
FALLBACK = 'anthropic/claude-3.5-haiku'
MESSAGE = ('No endpoints found matching your data policy (Zero data retention). '
           'Configure: https://openrouter.ai/settings/privacy')
ERROR = {'error': {'code': 404, 'message': MESSAGE}}
SUCCESS = {'id': 'gen-local-route-control',
           'choices': [{'message': {'role': 'assistant', 'content': 'synthetic completion'},
                        'finish_reason': 'stop'}], 'usage': {'cost': '0.000001'}}
CHAT = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'synthetic request'}],
        'max_tokens': 32}


def make_broker(responses, *, ledger=None):
    prices = broker_tests.price_table()
    # Synthetic admission only; this is not a production price-table snapshot.
    prices['models'][MODEL] = dict(prices['models']['openai/gpt-4o-mini'])
    ledger = ledger or broker_tests.FakeLedgerStore()
    transport = broker_tests.FakeTransport(responses)
    broker = br.Broker(store=ledger, transport=transport, price_table=prices,
        key_for=lambda provider: broker_tests.HOST_KEYS[provider],
        credential_for=lambda _context, provider: broker_tests.HOST_KEYS[provider],
        funding_source_for=lambda _context: 'miner_key')
    return broker, ledger, transport


@pytest.mark.parametrize('message', [
    MESSAGE,
    'No endpoints found matching your data policy (Zero data retention).',
    'No endpoints found matching your data policy.',
    'No endpoints found matching your data policy. Configure: https://openrouter.ai/settings/privacy',
])
def test_known_data_policy_route_messages_are_recognized(message):
    body = json.dumps({'error': {'code': 404, 'message': message}}).encode()
    assert br._openrouter_data_policy_route_unavailable(404, body)


@pytest.mark.parametrize('status,document', [
    (404, None), (404, []), (404, {'error': None}), (404, {'error': []}),
    (404, {'error': {'code': '404', 'message': MESSAGE}}),
    (404, {'error': {'code': 400, 'message': MESSAGE}}),
    (404, {'error': {'code': True, 'message': MESSAGE}}),
    (404, {'error': {'code': 404, 'message': ['not a string']}}),
    (404, {'error': {'code': 404, 'message': 'Requested model does not exist'}}),
    (404, {'error': {'code': 404, 'message': 'Requested page not found'}}),
    (404, {'error': {'code': 404, 'message': 'Quoted user input: ' + MESSAGE}}),
    (404, {'choices': [{'message': {'content': MESSAGE}}]}),
    (200, ERROR), (400, ERROR), (429, ERROR), (True, ERROR),
])
def test_caller_errors_spoofed_content_and_mismatched_status_do_not_match(status, document):
    assert not br._openrouter_data_policy_route_unavailable(status, json.dumps(document).encode())


@pytest.mark.parametrize('body', [b'not-json', b'\xff', b'{"error":'])
def test_malformed_data_policy_bodies_do_not_match(body):
    assert not br._openrouter_data_policy_route_unavailable(404, body)


@pytest.mark.parametrize('http_status', [404, 200])
@pytest.mark.parametrize('actual', ['0', '0.0000091'])
def test_live_and_settled_replay_share_classification_without_rewrite_or_redispatch(http_status, actual):
    payload = copy.deepcopy(ERROR)
    if actual != '0':
        payload['usage'] = {'cost': actual}
    broker, ledger, transport = make_broker([(http_status, payload)])
    arguments = dict(operation_id='openrouter.chat', parameters=CHAT,
                     action_sequence=0, timeout_ms=5000)
    first = broker.execute(broker_tests.CONTEXT, **arguments)
    preserved = copy.deepcopy(ledger.calls)
    # A fresh broker must derive the classification from the pre-existing
    # native terminal alone, without memory of the first response or new fields.
    replay_broker, _, replay_transport = make_broker([], ledger=ledger)
    second = replay_broker.execute(broker_tests.CONTEXT, **arguments)
    assert len(transport.sent) == 1 and not replay_transport.sent and ledger.calls == preserved
    assert first.status == second.status == 502
    assert first.body == second.body == operations.GENERIC_UNAVAILABLE_BODY
    assert first.call['error_code'] == second.call['error_code'] == 'provider_unavailable'
    assert first.call['provider_status'] == second.call['provider_status'] == 404
    assert first.call['outcome'] == second.call['outcome'] == 'settled'
    assert first.call['actual_microusd'] == second.call['actual_microusd'] == (0 if actual == '0' else 10)
    assert first.call['response_hash'] == second.call['response_hash'] == contracts.hash_bytes(first.body)
    assert second.call['idempotent'] is True
    terminal = ledger.calls[first.call['call_identity']]['terminal']
    assert terminal['status'] == 404 and terminal['call_succeeded'] is False
    assert json.loads(base64.b64decode(terminal['body_b64'])) == payload
    assert ledger.log.count('dispatch') == ledger.log.count('settle') == 1
    outbound = json.loads(transport.sent[0]['body'])
    assert outbound['provider'] == dict(operations.OPENROUTER_STRICT_PROVIDER_POLICY)


def test_same_message_from_a_source_provider_remains_source_404():
    broker, ledger, transport = make_broker([(404, ERROR)])
    result = broker.execute(broker_tests.CONTEXT, operation_id='scrapingdog.scrape',
        parameters={'url': 'https://example.com/missing'}, action_sequence=0, timeout_ms=5000)
    assert result.status == 404 and result.call.get('error_code') is None
    assert result.call['provider_status'] == 404 and result.call['actual_microusd'] == 0
    assert json.loads(result.body) == ERROR and len(transport.sent) == 1


def test_mapped_worker_error_export_keeps_native_receipt_and_cost_separate(tmp_path):
    from tests.lab_arena import training_observations_test as capture
    broker, ledger, _ = make_broker([(404, {**ERROR, 'usage': {'cost': '0.0000091'}})])
    result = broker.execute(broker_tests.CONTEXT, operation_id='openrouter.chat',
        parameters=CHAT, action_sequence=0, timeout_ms=5000)
    state = runner.RunState(lease=runner_tests.lease(), lease_token=capture.TOKEN)
    server = runner.WorkerSocketServer(tmp_path / 'worker.sock',
        runner_tests.FakeApi([], broker_documents=[result.to_document()]), state)
    returned = json.loads(server.handle_frame(shim.build_operation_frame('openrouter.chat', CHAT, 5000)))
    observation = state.worker_observations.observations[0]
    assert observation['outcome']['response'] == returned
    assert returned['status'] == 502
    assert base64.b64decode(returned['body_b64']) == operations.GENERIC_UNAVAILABLE_BODY
    receipt, = state.worker_observations.response_receipts
    assert receipt['call_identity'] == result.call['call_identity']
    assert receipt['status'] == 502
    assert receipt['body_sha256'] == contracts.hash_bytes(operations.GENERIC_UNAVAILABLE_BODY)
    assert receipt['body_bytes'] == len(operations.GENERIC_UNAVAILABLE_BODY)
    # handle_frame prepares the reply; this fixture does not write to a socket.
    assert receipt['write_state'] == 'prepared'
    assert receipt['socket_written_at'] is None

    values = capture.fixture()
    capture.add_call(values, pointer=False)
    identity = result.call['call_identity']
    for row in values['ledger_rows']:
        row['call_identity'] = identity
    values['ledger_rows'][-1].update(amount_microusd=10,
        terminal_response=copy.deepcopy(ledger.calls[identity]['terminal']))
    capture.save(values, state.worker_observations.document())
    exported = capture.assemble_example(**values)
    call = exported['provider_calls'][0]
    assert call['worker_observation_ref']['ordinal'] == observation['ordinal']
    assert call['response'] == {**ERROR, 'usage': {'cost': '0.0000091'}}
    assert call['response_receipts'][0]['metadata']['status'] == 404
    assert call['response_provenance']['delivery_to_worker'] == 'prepared'
    assert call['response_provenance']['worker_body_source'] == 'worker_error_observation'
    assert 'worker_provider_response_mismatch' in exported['completeness']['incomplete_reasons']
    assert call['actual_microusd'] == exported['costs']['confirmed_actual_microusd'] == 10
    assert not exported['privacy']['exact_replay']


@pytest.mark.parametrize('scenario,steps,output,exit_code,terminal,unresolved', [
    ('empty', [('zdr', MODEL)], 'empty', 0, 'provider_error', 1),
    ('no_output', [('zdr', MODEL)], 'none', 1, 'provider_error', 1),
    ('fallback_nonempty', [('zdr', MODEL), ('success', FALLBACK)], 'nonempty', 0, 'accepted', 0),
    ('fallback_empty', [('zdr', MODEL), ('success', FALLBACK)], 'empty', 0, 'accepted', 0),
    ('renewed_failure', [('zdr', MODEL), ('success', FALLBACK), ('zdr', MODEL)], 'empty', 0, 'provider_error', 1),
    ('invalid_model', [('invalid', MODEL)], 'empty', 0, 'accepted', 0),
    ('missing_model', [('invalid', 'unknown/not-admitted-model')], 'empty', 0, 'accepted', 0),
    ('source404', [('source404', None)], 'empty', 0, 'accepted', 0),
    ('rate_limit', [('rate_limit', MODEL)], 'empty', 0, 'provider_error', 1),
    ('auth', [('auth', MODEL)], 'empty', 0, 'credential_error', 1),
    ('spoof_success', [('spoof_success', MODEL)], 'empty', 0, 'accepted', 0),
])
def test_real_broker_worker_runner_transition(tmp_path, scenario, steps, output, exit_code, terminal, unresolved):
    replies = {
        'zdr': (404, ERROR), 'success': (200, SUCCESS),
        'invalid': (404, {'error': {'code': 404, 'message': 'Requested model does not exist'}}),
        'source404': (404, {'error': {'code': 404, 'message': 'Requested page not found'}}),
        'rate_limit': (429, {'error': {'code': 429, 'message': 'Rate limit exceeded'}}),
        'auth': (401, {'error': {'code': 401, 'message': 'Invalid API key'}}),
        'spoof_success': (200, {**SUCCESS, 'choices': [{'message': {'role': 'assistant', 'content': MESSAGE}, 'finish_reason': 'stop'}]}),
    }
    broker, ledger, transport = make_broker([replies[kind] for kind, _model in steps])
    claimed = runner_tests.lease()
    context = replace(broker_tests.CONTEXT, assignment_id=claimed['assignment_id'],
        round_id=claimed['round_id'], lease_token_hash=contracts.document_hash(claimed['lease_token']))
    calls = []
    class Api(runner_tests.FakeApi):
        def provider(self, run_id, token, frame):
            self.provider_frames.append(dict(frame))
            result = broker.execute(context, operation_id=frame['operation_id'],
                parameters=frame['parameters'], action_sequence=frame['action_sequence'],
                timeout_ms=frame['timeout_ms'])
            calls.append(result.call)
            return result.to_document()
    class Sandbox(runner_tests.BridgingRuntime):
        def run_icp(self, spec, **_kwargs):
            self.specs.append(spec)
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for kind, model in steps:
                    operation = 'scrapingdog.scrape' if kind == 'source404' else 'openrouter.chat'
                    parameters = {'url': 'https://example.com/missing'} if kind == 'source404' else {**CHAT, 'model': model}
                    shim.dispatch(operation, parameters, 5000)
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            if output != 'none':
                rows = [runner_tests.valid_company(1)] if output == 'nonempty' else []
                spec.output_path.write_bytes(json.dumps({'companies': rows}).encode())
            return runtime.fake_result(exit_code=exit_code, output_bytes=runtime.read_output(spec))
    api = Api([claimed])
    (tmp_path / 'work').mkdir()
    instance = runner.Runner(runner_tests.make_config(tmp_path, api, Sandbox(calls=0)))
    assert instance.run_once() == 1 and instance.abandoned == 0
    complete = api.completions[0]['body']
    result = contracts.validate_run_result(complete['result'])
    assert result['terminal_status'] == terminal
    assert result['resource_summary']['provider_unresolved_failure_count'] == unresolved
    assert result['resource_summary']['provider_call_count'] == len(steps)
    assert (complete['output'] is None) == (terminal != 'accepted')
    if output == 'nonempty':
        assert len(complete['output']['companies']) == 1
    if scenario == 'missing_model':
        assert not transport.sent and calls[0]['error_code'] == 'model_not_allowed'
    for sent in transport.sent:
        if sent['method'] == 'POST' and 'openrouter.ai' in sent['url']:
            assert json.loads(sent['body'])['provider'] == dict(operations.OPENROUTER_STRICT_PROVIDER_POLICY)
    if scenario == 'renewed_failure':
        assert [call.get('error_code') for call in calls] == ['provider_unavailable', None, 'provider_unavailable']
