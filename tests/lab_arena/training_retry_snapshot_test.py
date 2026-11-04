"""A fresh accepted retry keeps exact predecessor capture and all costs."""
import base64
import copy
import gzip

import pytest

from lab_arena import broker, contracts, training_capture, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import judge_only_gap_example
from tests.lab_arena.training_export_test import CLAIM_HASH, per_icp_cost_fixture, per_icp_cost_reader
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_observations_test import observation


def save_snapshot(values, run, document):
    snapshot, raw = training_observations.build_snapshot(run['run_id'], run['kind'], document,
        lease_generation=run['lease_generation'], claim_request_hash=run['claim_request_hash'])
    values['objects'].put(training_observations.object_ref(
        run['run_id'], run['lease_generation'], run['claim_request_hash']), raw)
    return snapshot


def retry_values(*, paid=False):
    values = per_icp_cost_fixture()
    values['cost_eligibility_reader'] = per_icp_cost_reader(spend=3, reason='eligible')
    source = without_client_callbacks(judge_only_gap_example())
    values['round_row']['configuration_doc'].update(source['configuration'])
    run = values['runs'][0]
    run['attempt'] = 2
    predecessor = copy.deepcopy(run)
    predecessor.update(run_id='execute:prior', attempt=1, status='failed',
        terminal_cause='lease_expired', result_doc=None, output_ref=None)
    values['runs'].append(predecessor)
    # Materialize exact selected provider requests, results and host receipts.
    ledger = []
    selected_buffer = training_observations.Buffer()
    for call in source['provider_calls']:
        if call['run_id'] != source['run_id']:
            continue
        pointer = training_capture.RequestCapture(values['objects'])(call['request'])
        terminal = broker._terminal_response_document(call['response_status'], {},
            contracts.canonical_json(call['response']).encode(), call_succeeded=True)
        terminal['reasoning_route_observation'] = copy.deepcopy(
            call['response_receipts'][0]['metadata']['reasoning_route_observation'])
        context = {key: run[key] for key in ('run_id', 'round_id', 'submission_id', 'miner_hotkey')}
        context.update(provider=call['provider'], operation_id=call['operation_id'],
            call_identity=contracts.hash_bytes(call['call_identity'].encode()), funding_source='miner')
        entry_id = call['terminal_entry_id']
        ledger.extend([
            dict(context, entry_id=entry_id-2, entry_kind='reservation', amount_microusd=0,
                entry_doc={'request_hash': call['request_hash'], 'action_sequence': call['action_sequence'],
                    'base_call_identity': call['call_identity'], 'provider_attempt': 1,
                    'request_capture': pointer, 'reasoning_capture': call['reasoning_capture']}),
            dict(context, entry_id=entry_id-1, entry_kind='dispatch', amount_microusd=0, entry_doc={}),
            dict(context, entry_id=entry_id, entry_kind='settlement', amount_microusd=call['actual_microusd'],
                entry_doc={}, terminal_response=terminal)])
        raw = base64.b64decode(terminal['body_b64'])
        selected_buffer.append_response_receipt({'ordinal': call['action_sequence'],
            'action_sequence': call['action_sequence'], 'operation_id': call['operation_id'],
            'call_identity': context['call_identity'], 'protocol': 'http', 'status': call['response_status'],
            'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw),
            'write_state': 'socket_written', 'socket_written_at': '2026-10-28T00:00:00Z'})
    values['ledger_rows'] = ledger
    run['result_doc']['resource_summary']['provider_call_count'] = len(ledger) // 3
    save_snapshot(values, run, selected_buffer.document())
    buffer = training_observations.Buffer()
    if paid:
        # The prior model call remains paid but is not a retry target.
        rows = copy.deepcopy(ledger[:3])
        for row in rows:
            row.update(run_id=predecessor['run_id'], entry_id=row['entry_id'] + 10000,
                call_identity=contracts.hash_bytes(b'prior-call'))
        reservation, _, terminal = rows
        reservation['entry_doc'].pop('reasoning_capture')
        reservation['entry_doc']['base_call_identity'] = reservation['call_identity']
        terminal['terminal_response'].pop('reasoning_route_observation')
        raw = base64.b64decode(terminal['terminal_response']['body_b64'])
        buffer.append_response_receipt({'ordinal': 0,
            'action_sequence': reservation['entry_doc']['action_sequence'],
            'operation_id': reservation['operation_id'], 'call_identity': reservation['call_identity'],
            'protocol': 'http', 'status': terminal['terminal_response']['status'],
            'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw),
            'write_state': 'socket_written', 'socket_written_at': '2026-10-28T00:00:00Z'})
        values['ledger_rows'].extend(rows)
    document = buffer.document()
    save_snapshot(values, predecessor, document)
    return values, predecessor, document


@pytest.mark.parametrize('paid', [False, True])
def test_failed_predecessor_uses_exact_complete_snapshot_without_completion_summary(paid):
    values, predecessor, _ = retry_values(paid=paid)
    before = copy.deepcopy(values['runs'])
    example = training_export.assemble_example(**values)
    assert example['related_run_gap_attribution'][predecessor['run_id']] == {
        'kind': 'execute', 'status': 'failed', 'research_process_gaps': ['direct_web_usage_unverified'],
        'cost_gaps': []}
    assert example['completeness']['incomplete_reasons'] == ['direct_web_usage_unverified']
    assert training_thinking.selected_research_evidence(example) == {'direct_web_usage_unverified'}
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    rows = strict['train'] + strict['eval']
    assert len(rows) == 2
    assert {ref['run_id'] for row in rows for ref in row['source_refs']} == {example['run_id']}
    assert example['costs']['all_related_attempts_included'] is True
    assert example['costs']['confirmed_actual_microusd'] == sum(
        row['amount_microusd'] for row in values['ledger_rows'] if row['entry_kind'] == 'settlement')
    assert values['runs'] == before


@pytest.mark.parametrize('kind', ['local_catalog', 'local_refusal'])
def test_predecessor_local_actions_use_snapshot_counts_without_phantom_bills(kind):
    values, predecessor, _ = retry_values()
    buffer = training_observations.Buffer()
    row = observation(kind=kind, operation='deepline.tools.list' if kind == 'local_catalog' else 'openrouter.chat',
        contributes=kind == 'local_catalog')
    row['request']['admitted_parameters'] = {}
    buffer.append(row)
    document = buffer.document()
    assert document['complete'] and document['observation_count'] == 1
    assert document['local_refusal_count'] == int(kind == 'local_refusal')
    save_snapshot(values, predecessor, document)
    example = training_export.assemble_example(**values)
    assert example['completeness']['incomplete_reasons'] == ['direct_web_usage_unverified']
    assert not [call for call in example['provider_calls'] if call['run_id'] == predecessor['run_id']]
    assert training_thinking.selected_research_evidence(example) == {'direct_web_usage_unverified'}


@pytest.mark.parametrize('damage', ['absent', 'incomplete', 'hash', 'claim', 'legacy',
    'counter_conflict', 'refusal_conflict', 'counter_invalid', 'accepted', 'missing_receipt',
    'receipt_identity', 'receipt_hash', 'unknown_cost', 'untracked_ledger', 'request_privacy', 'summary_conflict'])
def test_predecessor_summary_inference_requires_exact_complete_accounted_capture(damage):
    values, predecessor, document = retry_values(paid=damage in {
        'missing_receipt', 'receipt_identity', 'receipt_hash', 'unknown_cost', 'request_privacy'})
    ref = training_observations.object_ref(predecessor['run_id'], 1, CLAIM_HASH)
    if damage == 'absent':
        del values['objects'].values[ref]
    elif damage in {'incomplete', 'legacy', 'missing_receipt', 'receipt_identity', 'receipt_hash'}:
        if damage == 'incomplete': document.update(complete=False, failures=['capture_incomplete'])
        elif damage == 'legacy':
            document = {key: document[key] for key in (
                'observations', 'observation_count', 'local_refusal_count', 'complete', 'failures')}
        elif damage == 'missing_receipt': document.update(response_receipts=[], response_receipt_count=0)
        elif damage == 'receipt_identity': document['response_receipts'][0]['call_identity'] = contracts.hash_bytes(b'wrong')
        else: document['response_receipts'][0]['body_sha256'] = contracts.hash_bytes(b'wrong')
        save_snapshot(values, predecessor, document)
    elif damage in {'hash', 'claim'}:
        snapshot = training_observations.read_snapshot(values['objects'], predecessor['run_id'], 'execute', 1, CLAIM_HASH)
        snapshot['payload_sha256' if damage == 'hash' else 'claim_request_hash'] = contracts.hash_bytes(b'wrong')
        values['objects'].put(ref, gzip.compress(contracts.canonical_json(snapshot).encode(), mtime=0))
    elif damage in {'counter_conflict', 'refusal_conflict', 'counter_invalid'}:
        predecessor['result_doc'] = {'resource_summary': {
            'provider_local_refusal_count' if damage == 'refusal_conflict' else 'provider_call_count':
                False if damage == 'counter_invalid' else 1}}
    elif damage == 'summary_conflict':
        predecessor['result_doc'] = {'resource_summary': {'worker_observations': {'status': 'failed'}}}
    elif damage == 'accepted': predecessor['status'] = 'accepted'
    elif damage == 'unknown_cost': values['ledger_rows'][-1]['entry_kind'] = 'uncertain'
    elif damage == 'untracked_ledger':
        values['ledger_rows'].append(dict(values['ledger_rows'][0], run_id=predecessor['run_id'],
            entry_id=20000, call_identity=None))
    elif damage == 'request_privacy':
        values['ledger_rows'][-3]['entry_doc']['request_capture']['redacted'] = True
    example = training_export.assemble_example(**values)
    gaps = example['related_run_gap_attribution'][predecessor['run_id']]
    assert 'worker_local_refusals_unverified' in gaps['research_process_gaps']
    assert not example['completeness']['complete']


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_selected_execution_and_score_attempts_still_require_completion_counters(kind):
    values, _, _ = retry_values()
    selected = next(run for run in values['runs'] if run['kind'] == kind and run['status'] == 'accepted')
    summary = selected['result_doc']['resource_summary']
    summary.pop('provider_call_count')
    summary.pop('provider_local_refusal_count')
    save_snapshot(values, selected, training_observations.Buffer().document())
    example = training_export.assemble_example(**values)
    gaps = example['related_run_gap_attribution'][selected['run_id']]
    assert {'provider_capture_count_unverified', 'worker_local_refusals_unverified'} <= set(gaps['research_process_gaps'])


@pytest.mark.parametrize('damage', ['policy_off', 'legacy_configuration', 'invalid_registry',
    'same_attempt', 'future_attempt', 'invalid_prior_attempt', 'zero_prior_attempt',
    'invalid_selected_attempt'])
def test_snapshot_counter_fallback_is_only_for_prior_attempts_in_host_result_rounds(damage):
    values, predecessor, _ = retry_values()
    configuration = values['round_row']['configuration_doc']
    if damage == 'policy_off': configuration['reasoning_capture_policy'] = None
    elif damage == 'legacy_configuration': configuration.pop('training_maximum_missing_reasoning_percent')
    elif damage == 'invalid_registry': configuration['reasoning_route_registry'] = {}
    elif damage == 'same_attempt': predecessor['attempt'] = values['runs'][0]['attempt']
    elif damage == 'future_attempt': predecessor['attempt'] = values['runs'][0]['attempt'] + 1
    elif damage == 'invalid_prior_attempt': predecessor['attempt'] = True
    elif damage == 'zero_prior_attempt': predecessor['attempt'] = 0
    elif damage == 'invalid_selected_attempt': values['runs'][0]['attempt'] = '2'
    example = training_export.assemble_example(**values)
    gaps = example['related_run_gap_attribution'][predecessor['run_id']]
    assert {'provider_capture_count_unverified', 'worker_local_refusals_unverified'} <= set(gaps['research_process_gaps'])
    assert gaps['cost_gaps'] == ['provider_capture_count_unverified']

