"""Unverified final labels cannot erase exact unlabeled observed pairs."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_partial_outcome_test import built, partial_example
from tests.lab_arena.training_steps_test import seal


def unverified_label_example(*, signal_gap=False):
    example = partial_example()
    example['attempts'][-1].update(status='accepted', terminal_cause='accepted')
    proof = example['completeness']
    for name, values in proof['reasons_by_dimension'].items():
        proof['reasons_by_dimension'][name] = [
            'judge_evidence_unverified' if value == 'final_judge_unavailable' else value
            for value in values]
    if signal_gap:
        for name in ('input_output_label', 'research_process'):
            proof['reasons_by_dimension'][name].append('signal_authority_unverified')
    proof['incomplete_reasons'] = sorted({
        value for values in proof['reasons_by_dimension'].values() for value in values})
    return seal(example)


@pytest.mark.parametrize('signal_gap', [False, True])
@pytest.mark.parametrize('population', ['miner', 'baseline'])
def test_accepted_unverified_judge_keeps_exact_source_pair_without_labels(signal_gap, population):
    example = unverified_label_example(signal_gap=signal_gap)
    example['role'] = population
    seal(example)
    original = deepcopy(example)
    records, _ = built(example)
    assert len(records) == 1
    assert example == original
    record = records[0]
    assert record['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    ref = record['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['training_provenance']['label_authority']['origin'] == 'unavailable'
    assert ref['training_provenance']['company_outcomes'] == []
    assert ref['selected_research_evidence']['complete'] is False
    assert ref['selected_research_evidence']['outcome_availability'] == 'unavailable'
    assert 'judgment' not in example and 'score_run_id' not in example and example['companies'] == []


@pytest.mark.parametrize('damage', ['request_privacy', 'request_hash', 'response_privacy',
    'response_body', 'delivery', 'worker_hash', 'input_hash', 'input_privacy', 'input_icp',
    'output_hash', 'unknown_label_gap', 'source_identity'])
def test_unverified_labels_do_not_waive_exact_source_pair_checks(damage):
    example = unverified_label_example(signal_gap=True)
    call = example['provider_calls'][0]
    snapshot = example['input_snapshots'][example['run_id']]
    proof = example['completeness']
    if damage == 'request_privacy': call['request_capture'] = {'redacted': True}
    elif damage == 'request_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'response_body': call['response']['changed'] = True
    elif damage == 'delivery': call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'worker_hash':
        example['worker_observation_artifacts'][example['run_id']]['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'input_hash': snapshot['input_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'input_privacy': snapshot['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'input_icp':
        snapshot['input']['icp'] = {'different': True}
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    elif damage == 'output_hash': example['output_identity']['document_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'source_identity':
        proof['dimensions']['source_identity'] = False
        proof['reasons_by_dimension']['source_identity'] = ['source_unverified']
    else:
        proof['reasons_by_dimension']['input_output_label'].append('unknown_label_failure')
    proof['incomplete_reasons'] = sorted({
        value for values in proof['reasons_by_dimension'].values() for value in values})
    assert built(seal(example))[0] == []


def test_unverified_label_pairs_never_prove_a_full_trajectory():
    example = unverified_label_example(signal_gap=True)
    assert len(built(example, role='source')[0]) == 1
    assert len(built(example, role='judge')[0]) == 1
    assert training_thinking.selected_research_evidence(example) is None
    assert training_thinking.selected_research_evidence(example, role='judge') is None
    assert training_thinking._partial_research_evidence(example, role='judge') is not None


@pytest.mark.parametrize('signal_gap', [False, True])
@pytest.mark.parametrize('population', ['miner', 'baseline'])
def test_accepted_judge_with_unverified_final_labels_keeps_only_exact_unlabeled_pair(signal_gap, population):
    example = unverified_label_example(signal_gap=signal_gap)
    example['role'] = population
    seal(example)
    original = deepcopy(example)
    records, _ = built(example, role='judge')
    assert len(records) == 1
    plain, _ = training_steps._build([example], host_observed=True, role='judge')
    assert len(plain['train'] + plain['eval']) == 1
    ref = records[0]['source_refs'][0]
    assert ref['judge_context']['attempt_status'] == 'accepted'
    assert ref['judge_context']['outcome_availability'] == 'unavailable'
    assert ref['judge_context']['judgment_sha256'] is None
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    provenance = ref['training_provenance']
    assert provenance['label_authority']['origin'] == 'unavailable'
    assert provenance['label_authority']['input_output_label_verified'] is False
    assert provenance['score_run_id'] is provenance['judgment_sha256'] is None
    assert provenance['company_outcomes'] == []
    evidence = ref['selected_judge_evidence']
    assert evidence['scope'] == 'individual_model_request_result'
    assert evidence['complete'] is evidence['judge_full_thinking_complete'] is False
    assert example == original


@pytest.mark.parametrize('damage', ['source_input_missing', 'source_input_privacy', 'source_input_claim',
    'source_input_icp', 'source_status', 'judge_input_missing', 'judge_input_hash', 'judge_input_claim',
    'judge_input_companies', 'judge_input_policy', 'judge_input_icp', 'judge_input_icp_missing',
    'judge_input_scored_run', 'judge_scored_run', 'judge_runtime', 'judge_runtime_grading',
    'output_hash', 'output_body', 'request_privacy', 'request_hash', 'response_privacy', 'response_body',
    'delivery', 'worker_hash', 'route', 'history', 'unknown_label_gap'])
def test_unverified_final_labels_do_not_waive_accepted_judge_pair_evidence(damage):
    example = unverified_label_example()
    source_id, judge_id = example['run_id'], example['attempts'][-1]['run_id']
    source_input = example['input_snapshots'][source_id]
    judge_input = example['input_snapshots'][judge_id]
    call = example['provider_calls'][-1]
    if damage == 'source_input_missing': del example['input_snapshots'][source_id]
    elif damage == 'source_input_privacy': source_input['privacy']['redacted'] = True
    elif damage == 'source_input_claim': source_input['claim_request_hash'] = contracts.hash_bytes(b'foreign')
    elif damage == 'source_input_icp': source_input['input']['icp'] = {'foreign': True}
    elif damage == 'source_status': example['attempts'][0]['status'] = 'failed'
    elif damage == 'judge_input_missing': del example['input_snapshots'][judge_id]
    elif damage == 'judge_input_hash': judge_input['input_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'judge_input_claim': judge_input['claim_request_hash'] = contracts.hash_bytes(b'foreign')
    elif damage == 'judge_input_companies': judge_input['input']['companies'] = [{'foreign': True}]
    elif damage == 'judge_input_policy': judge_input['input']['scorer_policy'] = {}
    elif damage == 'judge_input_icp': judge_input['input']['icp'] = {'foreign': True}
    elif damage == 'judge_input_icp_missing': judge_input['input'].pop('icp')
    elif damage == 'judge_input_scored_run': judge_input['input']['scored_run_id'] = 'foreign'
    elif damage == 'judge_scored_run': example['attempts'][-1]['scored_run_id'] = 'foreign'
    elif damage == 'judge_runtime': example['attempts'][-1].pop('runtime_identity')
    elif damage == 'judge_runtime_grading':
        example['attempts'][-1]['runtime_identity']['grading_image_digest'] = 'sha256:' + 'd' * 64
    elif damage == 'output_hash': example['output_identity']['document_hash'] = contracts.hash_bytes(b'foreign')
    elif damage == 'output_body': example['output']['companies'] = [{'foreign': True}]
    elif damage == 'request_privacy': call['request_capture'] = {'redacted': True}
    elif damage == 'request_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'response_body': call['response']['changed'] = True
    elif damage == 'delivery': call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'worker_hash':
        example['worker_observation_artifacts'][judge_id]['payload_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'route': call['reasoning_capture']['provider_route'] = 'foreign'
    elif damage == 'history':
        call['request']['outbound']['body']['messages'].append({'role': 'tool', 'content': 'unbound', 'tool_call_id': 'foreign'})
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    else:
        proof = example['completeness']
        proof['reasons_by_dimension']['input_output_label'].append('unknown_label_failure')
        proof['incomplete_reasons'].append('unknown_label_failure')
    if damage in {'source_input_icp', 'judge_input_companies', 'judge_input_policy',
                  'judge_input_icp', 'judge_input_icp_missing', 'judge_input_scored_run'}:
        for snapshot in (source_input, judge_input):
            snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    assert built(seal(example), role='judge')[0] == []


def test_real_export_with_complete_worker_capture_and_missing_judge_output_keeps_only_pair():
    from lab_arena import training_observations as obs
    from tests.lab_arena.training_export_test import CLAIM_HASH
    from tests.lab_arena.training_journal_candidates_test import RUN, journal_values, records

    values, _ = journal_values()
    objects = values['objects']
    _, payload, _ = obs.read_journal_prefix(objects, RUN, 'execute', 1, CLAIM_HASH)
    payload.update(complete=True, failures=[])
    snapshot, raw = obs.build_snapshot(RUN, 'execute', payload,
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    objects.put(obs.object_ref(RUN, 1, CLAIM_HASH), raw)
    values['runs'][0]['result_doc']['resource_summary']['worker_observations'] = {
        'status': 'stored', 'complete': True, 'payload_sha256': snapshot['payload_sha256'],
        'observation_count': payload['observation_count'], 'local_refusal_count': payload['local_refusal_count']}
    judge = values['runs'][1]
    assert judge['status'] == judge['terminal_cause'] == 'accepted'
    del objects.values[judge['output_ref']]

    example = training_export.assemble_example(**values)
    assert example['completeness']['incomplete_reasons'] == ['judge_evidence_unverified']
    candidates = records(example)
    assert len(candidates) == 1
    ref = candidates[0]['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['selected_research_evidence']['complete'] is False
    assert training_thinking.selected_research_evidence(example) is None
    assessment = training_eligibility.assess_capture(values['submission_row'], {RUN: example}, [RUN],
        enabled=True, expected_run_count=1, minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=1)
    assert not assessment['eligible'] and assessment['counts']['complete_runs'] == 0
    assert assessment['counts']['rejected_runs'] == [{'run_id': RUN, 'reasons': ['trajectory_evidence_incomplete']}]
    assert example['configuration']['training_minimum_icp_coverage_percent'] == 80
    assert example['configuration']['training_maximum_missing_reasoning_percent'] == 1
