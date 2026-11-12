"""Unverified ancillary labels cannot erase exact unlabeled source pairs."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_thinking
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


def test_unverified_label_exception_is_source_only_and_never_full_trajectory():
    example = unverified_label_example(signal_gap=True)
    assert len(built(example, role='source')[0]) == 1
    assert built(example, role='judge')[0] == []
    assert training_thinking.selected_research_evidence(example) is None
    assert training_thinking.selected_research_evidence(example, role='judge') is None
    assert training_thinking._partial_research_evidence(example, role='judge') is None


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
