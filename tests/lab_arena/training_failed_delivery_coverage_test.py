"""Exact failed host delivery consumes an ICP, without claiming a read or target."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    _add_settled_exa_call, assess_ten, judge_only_gap_example, refresh_response_links)
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_steps_test import seal


def refresh(example):
    artifact = example['worker_observation_artifacts']['run']
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in example['provider_calls']:
        if call.get('run_id') == 'run':
            for name in ('worker_receipt_ref', 'worker_error_observation_ref'):
                if name in call['response_provenance']:
                    call['response_provenance'][name]['payload_sha256'] = artifact['payload_sha256']
    return seal(example)


def failed_delivery(*, model=False, status=502, api='openrouter.chat'):
    example = judge_only_gap_example()
    example['provider_calls'] = [call for call in example['provider_calls'] if call['run_id'] == 'run']
    if api == 'openrouter.responses':
        from tests.lab_arena.training_cap_eof_test import cap_example
        responses = cap_example(api)
        responses['provider_calls'][0]['response']['output'][1]['arguments'] = '{}'
        responses['provider_calls'][0]['response']['usage']['output_tokens'] = 50
        refresh_response_links(responses, 0)
        for key in ('provider_calls', 'configuration', 'worker_observation_artifacts'):
            example[key] = responses[key]
    if not model:
        _add_settled_exa_call(example)
    call = example['provider_calls'][0 if model else -1]
    call['response_status'] = status
    if status != 200:
        call.update(response={'error': {'code': 'provider_unavailable'}}, actual_microusd=0)
    call['lifecycle'][-1].update(entry_id=call['terminal_entry_id'], amount_microusd=call['actual_microusd'])
    call['response_receipts'][0]['metadata'].update(status=status, call_succeeded=status == 200)
    refresh_response_links(example, example['provider_calls'].index(call))
    example = without_client_callbacks(example)
    call = example['provider_calls'][0 if model else -1]
    if status != 200:
        raw = b'{"error":{"message":"Saved upstream failure"}}'
        call['upstream_response_receipts'] = [{
            'entry_id': call['terminal_entry_id'], 'status': status,
            'response': {'error': {'message': 'Saved upstream failure'}}, 'privacy_redactions': [],
            'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw),
            'wire_body_sha256': contracts.hash_bytes(raw), 'wire_body_bytes': len(raw),
            'response_sha256': contracts.document_hash({'error': {'message': 'Saved upstream failure'}}),
            'provenance': 'private_upstream_response_receipt', 'delivery_to_worker': 'not_proved'}]
    buffer = training_observations.Buffer()
    old = example['worker_observation_artifacts']['run']['payload']['response_receipts']
    for item, write in zip(example['provider_calls'], old):
        item['call_identity'] = contracts.hash_bytes(item['call_identity'].encode())
        write = {**write, 'call_identity': item['call_identity'], 'protocol': 'frame',
                 'status': item['response_status'], 'socket_written_at': '2026-10-27T00:00:00Z'}
        if item is call:
            write.update(write_state='prepared', socket_written_at=None)
        assert buffer.append_response_receipt(write)
        if item is call:
            buffer.mark_socket_write(write['ordinal'], written_at=None)
    snapshot, _ = training_observations.build_snapshot('run', 'execute', buffer.document(),
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b'claim'))
    example['worker_observation_artifacts']['run'] = snapshot
    # Derive the truthful links from the production exporter.
    assert not training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, example['provider_calls'], snapshot,
        require_client_ack=False, configuration=example['configuration'])
    example['attempts'] = [{'run_id': 'run', 'kind': 'execute', 'status': 'accepted'},
                           {'run_id': 'score', 'kind': 'score', 'status': 'accepted'}]
    example['related_run_gap_attribution'] = {
        'run': {'kind': 'execute', 'status': 'accepted', 'research_process_gaps': ['worker_observations_incomplete'], 'cost_gaps': []},
        'score': {'kind': 'score', 'status': 'accepted', 'research_process_gaps': [], 'cost_gaps': []}}
    dimensions = {name: [] for name in ('context', 'source_identity', 'exact_inputs', 'input_output_label', 'costs', 'research_process')}
    dimensions['research_process'] = ['worker_observations_incomplete']
    example['completeness'] = {'complete': False, 'incomplete_reasons': ['worker_observations_incomplete'],
        'dimensions': {name: not gaps for name, gaps in dimensions.items()}, 'reasons_by_dimension': dimensions}
    return refresh(example)
