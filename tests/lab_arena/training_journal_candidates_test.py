"""Validated journal pairs survive a missing final upload, without a full ICP."""
import base64
from copy import deepcopy
import gzip
import json

import pytest

from lab_arena import (contracts, reasoning_capture, training_capture, training_eligibility,
                       training_export, training_observations as obs, training_steps, training_thinking)
from tests.lab_arena.training_export_receipts_test import values_with_receipts
from tests.lab_arena.training_export_test import CLAIM_HASH
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import seal


RUN = 'execute:1'


def journal_values(*, legacy=False, write_state='socket_written'):
    values, payload = values_with_receipts(acknowledged=False)
    native = without_client_callbacks()
    original = native['provider_calls'][1]
    configuration = values['round_row']['configuration_doc']
    configuration.update({key: deepcopy(native['configuration'][key]) for key in
                          ('reasoning_capture_policy', 'reasoning_route_registry')})
    configuration['training_minimum_icp_coverage_percent'] = 80
    if not legacy:
        configuration['training_maximum_missing_reasoning_percent'] = 1
    capture = {key: deepcopy(original['request'][key]) for key in
               ('operation_id', 'request_hash', 'hashed_parameters', 'admitted_parameters',
                'effective_parameters', 'outbound')}
    capture['reasoning_capture'] = deepcopy(original['reasoning_capture'])
    pointer = training_capture.RequestCapture(values['objects'])(capture)
    wire = contracts.canonical_json(original['response']).encode()
    annotation = deepcopy(original['reasoning_capture'])
    for row in values['ledger_rows']:
        if row['entry_kind'] == 'reservation':
            row['entry_doc'].update(request_hash=capture['request_hash'], request_capture=pointer,
                                    reasoning_capture=annotation)
        elif row['entry_kind'] == 'settlement':
            row['terminal_response'].update(body_b64=base64.b64encode(wire).decode(),
                reasoning_route_observation=reasoning_capture.route_observation(original['response'], annotation))
    receipt = payload['response_receipts'][0]
    receipt.update(body_sha256=contracts.hash_bytes(wire), body_bytes=len(wire), write_state=write_state,
                   socket_written_at=None if write_state != 'socket_written' else receipt['socket_written_at'])
    payload['agent_observations'] = []
    payload['agent_observation_count'] = 0
    segment_payload = {'observation_from': 0, 'receipt_from': 0, 'agent_from': 0,
        'observations': [], 'response_receipts': payload['response_receipts'], 'receipt_updates': [],
        'agent_observations': [], 'observation_count': 0, 'response_receipt_count': 1,
        'agent_observation_count': 0, 'local_refusal_count': 0, 'failures': []}
    segment, raw = obs.build_segment(RUN, 'execute', segment_payload, lease_generation=1,
        claim_request_hash=CLAIM_HASH, number=0, previous_sha256=None,
        cumulative_bytes=len(contracts.canonical_json(segment_payload).encode()))
    values['objects'].put(obs.segment_ref(RUN, 1, CLAIM_HASH, 0), raw)
    objects = values['objects']
    objects.get_optional_bounded = lambda ref, maximum: objects.get_bounded(ref, maximum) if ref in objects.values else None
    return values, segment


def records(example, builder=training_thinking.build):
    splits, _ = builder([example], training_steps.Rules(production_rounds_only=False))
    return splits['train'] + splits['eval']


def test_actual_export_keeps_individual_journal_model_pair_without_qualifying_icp():
    values, segment = journal_values()
    example = training_export.assemble_example(**values)
    assert RUN not in example['worker_observation_artifacts']
    assert example['partial_worker_observation_artifacts'][RUN]['payload']['complete'] is False
    assert 'worker_observations_incomplete' in example['completeness']['incomplete_reasons']
    assert records(example), 'Missing final upload erased an independently persisted exact model pair'
    assert records(example, training_steps.build)
    assessment = training_eligibility.assess_capture(values['submission_row'], {RUN: example}, [RUN],
        enabled=True, expected_run_count=1, minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=1)
    assert not assessment['eligible']
    assert assessment['counts']['complete_runs'] == 0
    assert example['configuration']['training_minimum_icp_coverage_percent'] == 80
    assert example['configuration']['training_maximum_missing_reasoning_percent'] == 1
    ref = records(example)[0]['source_refs'][0]['worker_receipt_ref']
    assert ref['artifact_field'] == 'partial_worker_observation_artifacts'
    assert example[ref['artifact_field']][RUN]['segment_sha256'] == [segment['segment_sha256']]
    assert example[ref['artifact_field']][RUN]['segment_refs'] == [obs.segment_ref(RUN, 1, CLAIM_HASH, 0)]
    assert example[ref['artifact_field']][RUN]['payload_sha256'] == ref['payload_sha256']
    timeline = training_thinking.readable_trace(example)
    assert timeline['partial_worker_observation_paths'] == [['partial_worker_observation_artifacts', RUN]]
    assert timeline['worker_observation_paths'] == []
    assert 'worker_observations_incomplete' in timeline['capture_gaps']


@pytest.mark.parametrize('write_state', ['prepared', 'socket_write_failed'])
def test_journal_cannot_export_an_unwritten_model_result(write_state):
    values, _ = journal_values(write_state=write_state)
    assert not records(training_export.assemble_example(**values))


def test_historical_policy_cannot_use_journal_as_final_capture():
    values, _ = journal_values(legacy=True)
    assert not records(training_export.assemble_example(**values))


@pytest.mark.parametrize('damage', ['first_segment', 'segment_identity', 'segment_hash', 'segment_privacy',
                                   'lease_generation', 'claim_request_hash'])
def test_original_segment_identity_hash_privacy_and_lease_are_checked_before_linking(damage):
    values, segment = journal_values()
    ref = obs.segment_ref(RUN, 1, CLAIM_HASH, 0)
    if damage == 'first_segment':
        values['objects'].values[ref] = b'corrupt'
    elif damage in {'lease_generation', 'claim_request_hash'}:
        values['runs'][0][damage] = 2 if damage == 'lease_generation' else contracts.hash_bytes(b'other lease')
    else:
        changed = deepcopy(segment)
        if damage == 'segment_identity':
            changed['run_id'] = 'different-run'
        elif damage == 'segment_hash':
            changed['payload']['response_receipts'][0]['body_bytes'] += 1
        else:
            receipt = changed['payload']['response_receipts'][0]
            changed['payload']['agent_observations'] = [{
                'id': 0, 'kind': 'response_received', 'after_action_sequence': 0,
                'observed_at': '2026-10-16T01:00:01Z', 'depends_on': [],
                'privacy': {'redacted': True, 'redactions': ['credential']},
                'data': {key: receipt[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}}]
            changed['payload']['agent_observation_count'] = 1
            changed['payload_sha256'] = contracts.document_hash(changed['payload'])
            changed['cumulative_bytes'] = len(contracts.canonical_json(changed['payload']).encode())
            changed['segment_sha256'] = contracts.document_hash({key: value for key, value in changed.items() if key != 'segment_sha256'})
        values['objects'].values[ref] = gzip.compress(contracts.canonical_json(changed).encode())
    example = training_export.assemble_example(**values)
    assert not records(example)
    if damage != 'segment_privacy':
        assert example['provider_calls'][0]['response_provenance']['delivery_to_worker'] == 'not_proved'


@pytest.mark.parametrize('damage', ['payload_hash', 'lease', 'claim', 'segment_ref', 'missing_refs',
                                   'schema', 'complete', 'request_privacy', 'response_privacy', 'body',
                                   'other_lease_prefix', 'missing_input', 'input_privacy'])
def test_exported_prefix_cannot_bypass_individual_identity_or_privacy_checks(damage):
    values, _ = journal_values()
    example = training_export.assemble_example(**values)
    artifact = example['partial_worker_observation_artifacts'][RUN]
    call = example['provider_calls'][0]
    if damage == 'payload_hash': artifact['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'lease': artifact['lease_generation'] += 1
    elif damage == 'claim': artifact['claim_request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'segment_ref': artifact['segment_refs'][0] = 'other-ref'
    elif damage == 'missing_refs': artifact['segment_refs'] = []
    elif damage == 'schema': artifact['schema_version'] = obs.SCHEMA_VERSION
    elif damage == 'complete': artifact['complete'] = True
    elif damage == 'request_privacy': call['request_capture']['redacted'] = True
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'other_lease_prefix':
        artifact['claim_request_hash'] = contracts.hash_bytes(b'another valid prefix lease')
        artifact['segment_refs'] = [obs.segment_ref(RUN, artifact['lease_generation'], artifact['claim_request_hash'], 0)]
    elif damage == 'missing_input': example['input_snapshots'].pop(RUN)
    elif damage == 'input_privacy': example['input_snapshots'][RUN]['privacy']['redacted'] = True
    else: call['response']['choices'][0]['message']['content'] = 'changed response'
    seal(example)
    assert not records(example)


@pytest.mark.parametrize('corrupt_tail', [False, True])
def test_missing_or_corrupt_later_segment_keeps_only_exact_persisted_pairs(corrupt_tail):
    values, _ = journal_values()
    first = deepcopy(values['ledger_rows'])
    for row in first:
        row['call_identity'] = contracts.hash_bytes(b'uncaptured later model result')
        row['entry_id'] += 3
        row['entry_doc']['action_sequence'] = 1
        row['entry_doc']['base_call_identity'] = row['call_identity']
        if row['entry_kind'] == 'settlement':
            response = json.loads(base64.b64decode(row['terminal_response']['body_b64']))
            response['choices'][0]['message']['content'] = 'Uncaptured later result'
            row['terminal_response']['body_b64'] = base64.b64encode(contracts.canonical_json(response).encode()).decode()
    values['ledger_rows'].extend(first)
    values['runs'][0]['result_doc']['resource_summary']['provider_call_count'] = 2
    if corrupt_tail:
        values['objects'].put(obs.segment_ref(RUN, 1, CLAIM_HASH, 1), b'corrupt tail')
    example = training_export.assemble_example(**values)
    result = records(example)
    assert len(result) == 1
    assert {r['call_identity'] for item in result for r in item['source_refs']} == {values['ledger_rows'][0]['call_identity']}
    assert example['partial_worker_observation_artifacts'][RUN]['payload']['complete'] is False
    assert example['provider_calls'][1]['response_provenance']['delivery_to_worker'] == 'not_proved'


def test_final_snapshot_path_remains_authoritative_and_unchanged():
    from tests.lab_arena.training_export_receipts_test import save
    values, _ = journal_values()
    _, prefix, _ = obs.read_journal_prefix(values['objects'], RUN, 'execute', 1, CLAIM_HASH)
    prefix.update(complete=True, failures=[])
    snapshot = save(values, prefix)
    example = training_export.assemble_example(**values)
    assert example['worker_observation_artifacts'][RUN] == snapshot
    assert not example.get('partial_worker_observation_artifacts')
    assert 'artifact_field' not in example['provider_calls'][0]['response_provenance']['worker_receipt_ref']
    assert records(example)
    example['worker_observation_artifacts'][RUN]['schema_version'] = obs.JOURNAL_SCHEMA_VERSION
    seal(example)
    assert not records(example), 'A final-snapshot entry cannot masquerade as a journal prefix'
