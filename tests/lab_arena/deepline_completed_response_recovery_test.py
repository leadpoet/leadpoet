"""Recover authenticated saved responses through the ordinary broker path, without replay."""
import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote

import pytest

from lab_arena import broker as br, deepline_catalog
from tests.lab_arena.deepline_pending_async_receipt_test import ZeroHoldStore
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, make_broker

FIXTURE = Path(__file__).parent / 'fixtures/deepline/completed_key_lookup.json'
KEY = 'arena:' + 'a' * 64
NATIVE = 'public-execute:11111111-2222-4333-8444-555555555555'


def record():
    # Exact field structure observed on the real keyed Parallel lookup; values are synthetic.
    return json.loads(FIXTURE.read_text())


def catalog():
    return deepline_catalog.freeze_catalog({'tools': [row('parallel_search', provider='parallel',
        categories=['research'], inputSchema={'type': 'object', 'properties': {'objective': {'type': 'string'}}, 'required': ['objective']})]})


class RecoveryTransport:
    def __init__(self, document=None, *, headers=None, lookup_status=200, raw=None, lookup_error=False, bill_final=True, lost_error=None):
        self.document = document if document is not None else record()
        self.headers = headers if headers is not None else {'X-Deepline-Idempotency-Supported': 'true'}
        self.lookup_status, self.raw, self.lookup_error = lookup_status, raw, lookup_error
        self.bill_final, self.lost_error = bill_final, lost_error
        self.requests = []
        self.operation, self.provider = 'parallel_search', 'parallel'

    def send(self, **request):
        self.requests.append(request)
        if request['method'] == 'POST':
            wire = json.loads(request['body'])
            self.operation, self.provider = wire['operation'], wire['provider']
            key = request['headers'].get('idempotency-key')
            if key:
                self.document['executionRecovery']['idempotencyKey'] = key
            raise self.lost_error or br.ProviderTransportError('ReadTimeout')
        assert request['method'] == 'GET'
        if '/executions/by-key/' in request['url']:
            if self.lookup_error:
                raise br.ProviderTransportError('ReadTimeout')
            return br.ProviderResponse(self.lookup_status, self.headers,
                self.raw if self.raw is not None else json.dumps(self.document).encode())
        assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE, safe='')
        return br.ProviderResponse(200, {}, json.dumps({'recent': {'request_id': NATIVE, 'entries': [{
            'id': 'fixture-exact-charge', 'request_id': NATIVE, 'provider': self.provider, 'operation': self.operation,
            'credits': .02, 'delta': -.02, 'charge_state': 'posted' if self.bill_final else 'temporary_hold',
            'charge_finality': 'final' if self.bill_final else 'pending', 'billing_mode': 'deduct_on_settle', 'metadata': {},
        }]}}).encode())


def readback(transport, **patch):
    arguments = dict(transport=transport, secret=DL_KEY, execution_key=KEY, operation='parallel_search',
                     operation_aliases=(), max_response_bytes=4096)
    arguments.update(patch)
    return br._deepline_execution_response_readback(**arguments)


def execute(broker, context):
    return broker.execute(context, operation_id='deepline.execute',
        parameters={'tool': 'parallel_search', 'payload': {'objective': 'Find the official example.com website'}},
        action_sequence=0, timeout_ms=1000)


def test_live_shaped_saved_response_has_exact_identity_numeric_types_and_bounded_get_only():
    transport = RecoveryTransport()
    recovered = readback(transport)
    assert recovered.status == 200
    assert json.loads(recovered.body) == record()['response']
    assert isinstance(json.loads(recovered.body)['billing']['credits_charged'], float)
    assert len(transport.requests) == 1 and transport.requests[0]['method'] == 'GET'
    assert 'executionRecovery' not in json.loads(recovered.body)
    assert KEY not in recovered.body.decode()


@pytest.mark.parametrize('change', [
    'key', 'tool', 'native', 'local_native', 'saved_id', 'alias_conflict', 'absent_state', 'pending',
    'absent_response', 'response_list', 'status_bool', 'status_string', 'status_low', 'status_high',
    'secret_echo', 'encoded_secret_echo', 'nan',
])
def test_unproven_or_malformed_lookup_cannot_return_saved_response(change):
    document = record()
    if change == 'key': document['executionRecovery']['idempotencyKey'] = 'arena:' + 'b' * 64
    elif change == 'tool': document['toolId'] = 'another_search'
    elif change == 'native': document['requestId'] = 'bad/id'
    elif change == 'local_native': document['requestId'] = 'ctx-tool-' + 'a' * 32
    elif change == 'saved_id': document['response']['job_id'] = 'different-request'
    elif change == 'alias_conflict': document['response']['requestId'] = 'different-request'
    elif change == 'absent_state': del document['executionRecovery']['state']
    elif change == 'pending': document['executionRecovery']['state'] = 'running'
    elif change == 'absent_response': del document['response']
    elif change == 'response_list': document['response'] = []
    elif change == 'status_bool': document['responseStatus'] = True
    elif change == 'status_string': document['responseStatus'] = '200'
    elif change == 'status_low': document['responseStatus'] = 199
    elif change == 'status_high': document['responseStatus'] = 600
    elif change == 'secret_echo': document['response']['result'] = {'text': DL_KEY}
    elif change == 'encoded_secret_echo': document['response']['result'] = {'text': ''.join('%%%02X' % ord(c) for c in DL_KEY)}
    else: document['response']['billing']['credits_charged'] = float('nan')
    assert readback(RecoveryTransport(document)) is None


@pytest.mark.parametrize('options', [
    {'headers': {}}, {'headers': {'x-deepline-idempotency-supported': 'false'}},
    {'headers': {'x-deepline-idempotency-supported': 'true', 'X-Deepline-Idempotency-Supported': 'true'}},
    {'lookup_status': 500}, {'lookup_status': 404}, {'lookup_error': True},
    {'raw': b'not JSON'}, {'raw': b'[]'}, {'raw': b'\xff'},
])
def test_unavailable_or_unsupported_lookup_does_not_recover(options):
    assert readback(RecoveryTransport(**options)) is None


@pytest.mark.parametrize('patch', [{'observed_status': 201}])
def test_prior_transport_status_cannot_be_overwritten(patch):
    assert readback(RecoveryTransport(), **patch) is None


def test_alias_and_matching_prior_transport_status_are_accepted():
    document = record(); document['toolId'] = 'search'
    assert readback(RecoveryTransport(document), operation_aliases=['search'],
                    observed_status=200).status == 200


def test_oversized_lookup_and_saved_body_are_rejected():
    document = record(); document['response']['result']['text'] = 'x' * 5000
    assert readback(RecoveryTransport(document)) is None
    document['response']['result']['text'] = 'x' * 25000
    assert readback(RecoveryTransport(document)) is None


def test_unkeyed_failure_never_reads_key_or_reexecutes():
    transport = RecoveryTransport()
    assert readback(transport, execution_key=None) is None
    assert transport.requests == []


@pytest.mark.parametrize('billing', ['inline', 'exact', 'pending'])
def test_timeout_recovery_uses_normal_success_billing_and_no_paid_replay(monkeypatch, billing):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    document = record()
    if billing != 'inline': del document['response']['billing']
    transport = RecoveryTransport(document, bill_final=billing != 'pending')
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    context = replace(CONTEXT, deepline_catalog=catalog())
    result = execute(broker, context)
    assert result.status == 200 and json.loads(result.body)['result'] == record()['response']['result']
    assert result.call['transport_recovery'] == 'deepline_execution_lookup'
    saved = store.calls[result.call['call_identity']]
    assert saved['amount'] == 0
    if billing == 'pending':
        assert result.call['outcome'] == 'uncertain' and 'actual_microusd' not in result.call
        assert saved['uncertain_doc']['call_succeeded'] is True
    else:
        assert result.call['actual_microusd'] == 2000 and saved['terminal']['call_succeeded'] is True
    before = len(transport.requests)
    execute(broker, context)
    assert len(transport.requests) == before + (2 if billing == 'pending' else 0)
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1


def test_recovered_provider_error_is_not_fabricated_as_success(monkeypatch):
    document = record(); document['responseStatus'] = 500
    document['response'].update(status='failed', error={'code': 'UPSTREAM_FAILURE'})
    transport = RecoveryTransport(document)
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    result = execute(broker, replace(CONTEXT, deepline_catalog=catalog()))
    assert result.status == 502 and result.call['actual_microusd'] == 2000
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is False
    assert 'deepline_response_missing' not in result.call
    assert 'deepline_response_missing' not in store.calls[result.call['call_identity']]['terminal']
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1


def test_vercel_trace_hint_does_not_override_exact_key_native_identity():
    trace = 'iad1::fwh5j-1791330233666-aeb0e7021ee6'
    transport = RecoveryTransport(lost_error=br.ProviderTransportError('ReadTimeout',
        observed_status=200, deepline_job_id=trace))
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    result = execute(broker, replace(CONTEXT, deepline_catalog=catalog()))
    assert result.status == 200 and result.call['actual_microusd'] == 2000
    saved = store.calls[result.call['call_identity']]['terminal']
    assert saved['provider_cost']['request_id'] == NATIVE and saved['call_succeeded'] is True
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1


def test_recovered_async_start_keeps_owned_handle_and_unconfirmed_parent_cost(monkeypatch):
    from tests.lab_arena.deepline_pending_async_receipt_test import receipt, JOB
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    start = row('vendor_batch_scrape', categories=['automation'],
        inputSchema={'type': 'object', 'properties': {'urls': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['urls']},
        asyncFlow={'startAction': 'vendor_batch_scrape', 'pollActions': ['vendor_get_status'], 'finishAction': None},
        asyncOperation={'job': {'idPaths': ['id', 'data.id']}})
    poll = row('vendor_get_status', categories=['admin'],
        inputSchema={'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']})
    frozen = deepline_catalog.freeze_catalog({'tools': [start, poll]})
    document = record(); document['toolId'] = 'vendor_batch_scrape'; document['response'] = receipt()
    document['response']['job_id'] = NATIVE
    transport = RecoveryTransport(document, bill_final=False)
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    context = replace(CONTEXT, deepline_catalog=frozen)
    result = broker.execute(context, operation_id='deepline.execute',
        parameters={'tool': 'vendor_batch_scrape', 'payload': {'urls': ['https://example.com']}},
        action_sequence=0, timeout_ms=1000)
    assert result.status == 200 and result.call['outcome'] == 'uncertain'
    assert json.loads(result.body)['status'] == 'running' and 'actual_microusd' not in result.call
    saved = store.calls[result.call['call_identity']]
    assert saved['amount'] == 0 and saved['uncertain_doc']['call_succeeded'] is True
    assert saved['uncertain_doc']['deepline_async_job_ids'] == [JOB]
    assert broker._owns_deepline_async_job(context, deepline_catalog.tool_entry(frozen, 'vendor_get_status'), {'id': JOB})
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
