"""Admission denials bind real ledger and worker proof without a generation."""
import base64
from copy import deepcopy

import pytest

from lab_arena import contracts, training_capture, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_call_semantics_test import _failed_example
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_export_test import Objects
from tests.lab_arena.training_steps_test import seal


def refusal_example(reason='per_icp_quota', *, missing_request=False):
    example = _failed_example(402, worker_error=True)
    original = example['provider_calls'][0]
    original['request']['admitted_parameters'].pop('provider', None)
    original['request']['admitted_parameters'].pop('stream', None)
    original['request']['admitted_parameters']['reasoning'].pop('exclude', None)
    original['response'] = {'error': {'code': 'budget_refused'}}
    wire = contracts.canonical_json(original['response']).encode()
    payload = example['worker_observation_artifacts']['run']['payload']
    for observation_row in payload['agent_observations']:
        observation_row.setdefault('depends_on', [])
    observation = payload['observations'][0]
    observation['request'] = {'admitted_parameters': deepcopy(original['request']['admitted_parameters'])}
    observation['outcome']['response']['body_b64'] = base64.b64encode(wire).decode()
    for row in payload['response_receipts'] + payload['agent_observations']:
        data = row.get('data', row)
        if data.get('call_identity') == original['call_identity']:
            data.update(body_sha256=contracts.hash_bytes(wire), body_bytes=len(wire))
    objects = Objects()
    pointer = training_capture.RequestCapture(objects)(original['request'])
    row = {key: original[key] for key in ('run_id', 'provider', 'operation_id', 'call_identity', 'funding_source')}
    row.update(round_id=example['round_id'], submission_id=example['submission_id'],
               entry_id=999, entry_kind='refusal', amount_microusd=0,
               entry_doc={'reason': reason, 'requested_microusd': 100,
                          'call': {**{key: original[key] for key in ('request_hash', 'action_sequence', 'reasoning_capture')},
                                   'base_call_identity': original['call_identity'], 'provider_attempt': 1,
                                   'request_capture': pointer}})
    if missing_request:
        objects.values.clear()
    calls, gaps = training_export._calls([row], {'run'}, objects, {'run': 0})
    assert len(calls) == 1
    snapshot = example['worker_observation_artifacts']['run']
    snapshot['payload_sha256'] = contracts.document_hash(payload)
    # Link only this call against its actual write/read rows. Other successful
    # calls keep their existing proof and receive the updated artifact digest.
    selected_payload = deepcopy(payload)
    selected_payload['response_receipts'] = [r for r in payload['response_receipts'] if r['call_identity'] == original['call_identity']]
    selected_payload['agent_observations'] = [r for r in payload['agent_observations'] if r.get('data', {}).get('call_identity') == original['call_identity']]
    selected_snapshot = dict(snapshot, payload=selected_payload, payload_sha256=contracts.document_hash(selected_payload))
    errors = training_export._link_worker_receipts({'run_id': 'run', 'kind': 'execute'}, calls, selected_snapshot, require_completed_phases=False)
    # The whole artifact remains the authority; restore the full artifact refs.
    for call in calls + example['provider_calls'][1:]:
        provenance = call['response_provenance']
        for ref in provenance.get('agent_receipt_refs', []) + [provenance['worker_receipt_ref']] + ([provenance['worker_error_observation_ref']] if 'worker_error_observation_ref' in provenance else []):
            ref['payload_sha256'] = snapshot['payload_sha256']
    example['provider_calls'][0] = calls[0]
    return seal(example), row, objects, gaps, errors


@pytest.mark.parametrize('reason', ['per_icp_quota', 'stage_quota', 'money_cap'])
def test_bound_zero_charge_refusal_keeps_previous_targets_and_eligibility(reason):
    example, row, _, gaps, errors = refusal_example(reason)
    call = example['provider_calls'][0]
    assert not gaps and not errors
    assert call['refusal_receipt'] == row
    assert call['response_receipts'] == []
    assert 'response' not in call and 'reservation_entry_id' not in call
    assert training_thinking.verified_failed_attempt(example, call)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(record['source_refs'][0]['call_identity'] != call['call_identity'] for record in records)


@pytest.mark.parametrize('damage', ['ledger_hash', 'ledger_identity', 'charge', 'dispatch', 'reason', 'request', 'worker_hash', 'ack', 'admitted', 'generation', 'unbound'])
def test_refusal_cannot_hide_invalid_request_or_forged_evidence(damage):
    example, _, _, _, _ = refusal_example()
    call = example['provider_calls'][0]
    if damage == 'ledger_hash':
        call['refusal_receipt']['entry_doc']['call']['request_hash'] = contracts.hash_bytes(b'bad')
    elif damage == 'ledger_identity':
        call['refusal_receipt']['call_identity'] = 'unbound'
    elif damage == 'charge':
        call['actual_microusd'] = 1
    elif damage == 'dispatch':
        call['lifecycle'].append({'entry_kind': 'dispatch'})
    elif damage == 'reason':
        call['refusal_receipt']['entry_doc']['reason'] = 'invalid_request'
    elif damage == 'request':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'worker_hash':
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'ack':
        call['response_provenance']['agent_receipt'] = 'not_reported'
    elif damage == 'admitted':
        call['request']['admitted_parameters'] = {'model': 'forged'}
    elif damage == 'generation':
        call['response'] = {'output': [{'type': 'message'}]}
    else:
        call.pop('refusal_receipt')
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


def test_missing_refusal_request_is_a_real_gap_not_an_invalid_receipt():
    example, _, _, gaps, errors = refusal_example(missing_request=True)
    assert gaps == {'full_request_unavailable'}
    assert not errors
    assert not training_thinking.verified_failed_attempt(example, example['provider_calls'][0])


@pytest.mark.parametrize('policy', ['frozen', 'off'])
def test_legacy_policy_does_not_gain_refusal_admission(policy):
    example, _, _, _, _ = refusal_example()
    if policy == 'frozen':
        example['configuration'].pop('training_maximum_missing_reasoning_percent')
    else:
        example['configuration']['reasoning_capture_policy'] = None
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, example['provider_calls'][0])


@pytest.mark.parametrize('policy', ['full', 'off'])
@pytest.mark.parametrize('damage', [None, 'pointer', 'hash', 'reason'])
def test_broker_persists_only_bound_full_policy_refusal_without_dispatch(policy, damage):
    from dataclasses import replace
    from lab_arena import reasoning_capture as rc
    from tests.lab_arena import test_lab_arena_broker as bt
    from tests.lab_arena.reasoning_capture_test import _reviewed_registry

    class RefusalStore(bt.FakeLedgerStore):
        def reserve_call(self, **kwargs):
            self.log.append('reserve')
            self.row = {key: kwargs[key] for key in ('run_id', 'call_identity', 'operation_id', 'provider', 'funding_source')}
            saved = deepcopy(kwargs['call_doc'])
            if damage == 'pointer':
                saved['request_capture']['sha256'] = contracts.hash_bytes(b'forged')
            if damage == 'hash':
                saved['request_hash'] = contracts.hash_bytes(b'forged')
            self.row.update(entry_kind='refusal', entry_id=1, amount_microusd=0,
                            entry_doc={'reason': 'invalid_request' if damage == 'reason' else 'per_icp_quota',
                                       'requested_microusd': kwargs['amount_microusd'], 'call': saved})
            return {'status': 'refused', 'idempotent': len(self.log) > 1,
                    'call_identity': kwargs['call_identity'], 'reason': 'per_icp_quota'}

        def list_ledger(self, **kwargs):
            return [deepcopy(self.row)]

    objects = Objects()
    store = RefusalStore()
    broker, _, transport = bt.make_broker(store=store, capture_request=training_capture.RequestCapture(objects).prepare)
    context = bt.CONTEXT
    if policy == 'full':
        context = replace(context, reasoning_capture_policy=rc.POLICY,
                          reasoning_route_registry=_reviewed_registry())
    parameters = {'model': 'openai/gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Find evidence.'}], 'max_tokens': 1500}
    for _ in range(2):
        result = broker.execute(context, operation_id='openrouter.chat', parameters=parameters,
                                action_sequence=0, timeout_ms=1000)
        assert result.status == 402
        assert result.body == b'{"error":{"code":"budget_refused"}}'
    assert transport.sent == [] and store.log == ['reserve', 'reserve']
    assert len(objects.values) == int(policy == 'full' and damage is None)
    if policy == 'full' and damage is None:
        request = training_capture.read_request(objects, store.row['entry_doc']['call']['request_capture'])
        assert request['request_hash'] == store.row['entry_doc']['call']['request_hash']
