"""A retained score reference is not a verified final training label."""
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, scoring, signal_judgments, training_export, training_steps, training_thinking
from tests.lab_arena.training_partial_outcome_test import built
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_unverified_label_source_pairs_test import unverified_label_example


def retained_signal_label_example():
    example = unverified_label_example(signal_gap=True)
    # Native assembly sets this before signal-authority reconstruction. On
    # failure it removes judgment/companies, but retains this audit reference.
    example['score_run_id'] = example['attempts'][-1]['run_id']
    proof = example['completeness']
    for dimension, reasons in proof['reasons_by_dimension'].items():
        remaining = [reason for reason in reasons if reason != 'judge_evidence_unverified']
        proof['reasons_by_dimension'][dimension] = remaining
        proof['dimensions'][dimension] = not remaining
    proof['incomplete_reasons'] = ['signal_authority_unverified']
    return seal(example)


@pytest.mark.parametrize('role', ['source', 'judge'])
@pytest.mark.parametrize('population', ['miner', 'baseline'])
def test_retained_score_without_signal_authority_keeps_only_verified_unlabeled_pairs(role, population):
    example = retained_signal_label_example()
    example['role'] = population
    seal(example)
    before = deepcopy(example)
    call = example['provider_calls'][0 if role == 'source' else -1]
    # Byte, privacy, input, output-shape and delivery proofs already pass.
    assert training_steps._candidate(example, call, role=role)
    records, _ = built(example, role=role)
    assert len(records) == 1
    plain, _ = training_steps._build([example], host_observed=True, role=role)
    for record in records + plain['train'] + plain['eval']:
        ref = record['source_refs'][0]
        assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
        provenance = ref['training_provenance']
        assert provenance['label_authority']['origin'] == 'unavailable'
        assert provenance['label_authority']['input_output_label_verified'] is False
        assert provenance['score_run_id'] is provenance['judgment_sha256'] is None
        assert provenance['company_outcomes'] == []
        if role == 'judge':
            assert ref['judge_context']['attempt_status'] == 'accepted'
            assert ref['judge_context']['judgment_sha256'] is None
            assert ref['judge_context']['outcome_availability'] == 'unavailable'
    evidence = records[0]['source_refs'][0][
        'selected_judge_evidence' if role == 'judge' else 'selected_research_evidence']
    assert evidence['complete'] is False
    assert evidence['scope'] == 'individual_model_request_result'
    assert evidence['outcome_availability'] == 'unavailable'
    if role == 'judge':
        assert evidence['judge_full_thinking_complete'] is False
    assert training_thinking.selected_research_evidence(example, role=role) is None
    assert example == before


@pytest.mark.parametrize('role', ['source', 'judge'])
@pytest.mark.parametrize('damage', [
    'request_hash', 'response_hash', 'request_privacy', 'response_privacy', 'delivery',
    'worker_hash', 'source_input', 'source_privacy', 'source_identity', 'output',
    'selected_score', 'score_kind', 'score_status', 'score_source', 'score_input',
    'score_input_missing', 'score_input_privacy', 'score_runtime', 'score_terminal',
    'score_claim', 'score_lease', 'unknown_label',
])
def test_unavailable_signal_label_keeps_pair_and_role_specific_context_checks(role, damage):
    example = retained_signal_label_example()
    call = example['provider_calls'][0 if role == 'source' else -1]
    source_input = example['input_snapshots'][example['run_id']]
    score_input = example['input_snapshots'][example['score_run_id']]
    score = example['attempts'][-1]
    if damage == 'request_hash':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'response_hash':
        call['response']['changed'] = True
    elif damage == 'request_privacy':
        call['request_capture'] = {'redacted': True}
    elif damage == 'response_privacy':
        call['response_privacy_redactions'] = ['credential']
    elif damage == 'delivery':
        call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'worker_hash':
        example['worker_observation_artifacts'][call['run_id']]['payload_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'source_input':
        source_input['input_sha256'] = contracts.hash_bytes(b'foreign')
    elif damage == 'source_privacy':
        source_input['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'source_identity':
        proof = example['completeness']
        proof['dimensions']['source_identity'] = False
        proof['reasons_by_dimension']['source_identity'] = ['source_unverified']
        proof['incomplete_reasons'].append('source_unverified')
    elif damage == 'output':
        example['output']['companies'] = [{'foreign': True}]
    elif damage == 'selected_score':
        example['score_run_id'] = 'foreign-score'
    elif damage == 'score_kind':
        score['kind'] = 'execute'
    elif damage == 'score_status':
        score['status'] = 'failed'
        score['terminal_cause'] = 'judge_timeout'
    elif damage == 'score_source':
        score['scored_run_id'] = 'foreign-source'
    elif damage == 'score_input':
        score_input['input']['companies'] = [{'foreign': True}]
        score_input['input_sha256'] = contracts.document_hash(score_input['input'])
    elif damage == 'score_input_missing':
        del example['input_snapshots'][example['score_run_id']]
    elif damage == 'score_input_privacy':
        score_input['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'score_runtime':
        score.pop('runtime_identity')
    elif damage == 'score_terminal':
        score['terminal_cause'] = 'judge_timeout'
    elif damage == 'score_claim':
        score_input['claim_request_hash'] = contracts.hash_bytes(b'foreign')
    elif damage == 'score_lease':
        score_input['lease_generation'] += 1
    else:
        proof = example['completeness']
        for dimension in ('input_output_label', 'research_process'):
            proof['reasons_by_dimension'][dimension].append('unknown_label_failure')
        proof['incomplete_reasons'].append('unknown_label_failure')
    # The score reference supplies no pair authority. A damaged judge context
    # must not erase the independent exact source exchange.
    diagnostic_only = damage in {'selected_score', 'score_status'} or (
        role == 'source' and damage.startswith('score_'))
    assert len(built(seal(example), role=role)[0]) == int(diagnostic_only)


@pytest.mark.parametrize('damaged_role', ['source', 'judge'])
@pytest.mark.parametrize('damage', ['request_privacy', 'response_privacy', 'request_hash', 'delivery'])
def test_private_or_unverified_turn_cannot_erase_independent_opposite_role(damaged_role, damage):
    example = retained_signal_label_example()
    call = example['provider_calls'][0 if damaged_role == 'source' else -1]
    if damage == 'request_privacy':
        call['request_capture'] = {'redacted': True}
    elif damage == 'response_privacy':
        call['response_privacy_redactions'] = ['credential']
    elif damage == 'request_hash':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'foreign')
    else:
        call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    seal(example)
    assert not built(example, role=damaged_role)[0]
    other_role = 'judge' if damaged_role == 'source' else 'source'
    assert len(built(example, role=other_role)[0]) == 1


def clone_judge(example, identity, *, status, scored_run_id=None):
    original = example['score_run_id']
    attempt = deepcopy(example['attempts'][-1])
    attempt.update(run_id=identity, status=status, terminal_cause='judge_timeout' if status == 'failed' else 'accepted',
                   scored_run_id=scored_run_id or example['run_id'])
    example['attempts'].append(attempt)
    snapshot = deepcopy(example['input_snapshots'][original])
    snapshot['run_id'] = identity
    example['input_snapshots'][identity] = snapshot
    call = deepcopy(example['provider_calls'][-1])
    call.update(run_id=identity, call_identity=identity + '-call')
    artifact = deepcopy(example['worker_observation_artifacts'][original])
    artifact['run_id'] = identity
    for row in artifact['payload']['response_receipts']:
        row['call_identity'] = call['call_identity']
    for row in artifact['payload']['agent_observations']:
        if row['kind'] == 'response_received':
            row['data']['call_identity'] = call['call_identity']
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    example['worker_observation_artifacts'][identity] = artifact
    provenance = call['response_provenance']
    for ref in provenance['agent_receipt_refs'] + [provenance['worker_receipt_ref']]:
        ref.update(run_id=identity, payload_sha256=artifact['payload_sha256'])
    example['provider_calls'].append(call)


def test_retained_unlabeled_judge_and_bound_failed_attempt_keep_no_episode_label():
    example = retained_signal_label_example()
    clone_judge(example, 'failed-judge', status='failed')
    clone_judge(example, 'foreign-judge', status='failed', scored_run_id='foreign-source')
    records, _ = built(seal(example), role='judge')
    refs = [ref for record in records for ref in record['source_refs']]
    assert {ref['run_id'] for ref in refs} == {example['score_run_id'], 'failed-judge'}
    assert all(ref['outcome'] == {'per_icp_score': None, 'score_run_id': None} for ref in refs)
    assert all(ref['judge_context']['judgment_sha256'] is None for ref in refs)


def native_signal_values():
    from lab_arena import training_observations as observations
    from tests.lab_arena.training_export_signal_test import _signal_fixture
    from tests.lab_arena.training_export_test import save_input
    from tests.lab_arena.training_journal_candidates_test import journal_values

    values = _signal_fixture()
    native, segment = journal_values()
    source, judge = values['runs'][:2]
    configuration = values['round_row']['configuration_doc']
    configuration.update({key: deepcopy(native['round_row']['configuration_doc'][key]) for key in (
        'reasoning_capture_policy', 'reasoning_route_registry',
        'training_minimum_icp_coverage_percent', 'training_maximum_missing_reasoning_percent')})
    annotation = native['ledger_rows'][0]['entry_doc']['reasoning_capture']
    configuration['scorer_policy'] = contracts.validate_scorer_policy({
        **scoring.build_scorer_policy(
            judge_models={role: annotation['model_id'] for role in scoring.DEFAULT_JUDGE_MODELS},
            judge_provider_routes={annotation['model_id']: annotation['provider_route']},
            scoring_adapter_version='qualification_integrity_v2', company_judgment_cache=True),
        'signal_judgment_policy': signal_judgments.POLICY})
    for ref, raw in native['objects'].values.items():
        if ref not in values['objects'].values:
            values['objects'].put(ref, raw)
    icp = json.loads(values['objects'].values['benchmark'])['icps'][0]
    output = json.loads(values['objects'].values[source['output_ref']])
    save_input(values['objects'], judge['run_id'], 'score', scoring.build_scoring_input(
        scored_run_id=source['run_id'], icp=icp, companies=output['companies'],
        policy=configuration['scorer_policy'], evaluation_date=values['round_row']['evaluation_date']))
    values['ledger_rows'] = []
    for index, run in enumerate((source, judge)):
        identity = contracts.hash_bytes(run['run_id'].encode())
        for original in native['ledger_rows']:
            row = deepcopy(original)
            row.update(run_id=run['run_id'], call_identity=identity, entry_id=row['entry_id'] + index * 100)
            row.update({key: source[key] for key in ('round_id', 'submission_id', 'miner_hotkey')})
            values['ledger_rows'].append(row)
        buffer = observations.Buffer()
        receipt = deepcopy(segment['payload']['response_receipts'][0])
        receipt['call_identity'] = identity
        assert buffer.append_response_receipt(receipt)
        snapshot, raw = observations.build_snapshot(
            run['run_id'], run['kind'], buffer.document(), lease_generation=run['lease_generation'],
            claim_request_hash=run['claim_request_hash'])
        values['objects'].put(observations.object_ref(
            run['run_id'], run['lease_generation'], run['claim_request_hash']), raw)
        summary = run['result_doc']['resource_summary']
        summary['provider_call_count'] = 1
        summary['worker_observations'] = {
            'status': 'stored', 'payload_sha256': snapshot['payload_sha256'],
            'observation_count': 0, 'local_refusal_count': 0, 'complete': True}
    # Only the final signal authority is unavailable. Both model exchanges and
    # host snapshots are independently retained and validated by native assembly.
    values['signal_judgment_rows'] = []
    return values


@pytest.mark.parametrize('role', ['source', 'judge'])
def test_native_signal_authority_absence_keeps_clean_pairs_without_final_eligibility(role):
    from lab_arena import training_eligibility

    values = native_signal_values()
    example = training_export.assemble_example(**values)
    assert example['score_run_id'] == values['runs'][1]['run_id']
    assert 'judgment' not in example and example['companies'] == []
    assert example['completeness']['reasons_by_dimension']['input_output_label'] == ['signal_authority_unverified']
    assert len(example['worker_observation_artifacts']) == 2
    records, _ = built(example, role=role)
    assert len(records) == 1
    ref = records[0]['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['training_provenance']['company_outcomes'] == []
    assert ref['selected_judge_evidence' if role == 'judge' else 'selected_research_evidence']['complete'] is False
    assessment = training_eligibility.assess_capture(
        values['submission_row'], {example['run_id']: example}, [example['run_id']],
        enabled=True, expected_run_count=1, minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=1)
    assert not assessment['eligible'] and assessment['counts']['complete_runs'] == 0
    assert example['configuration']['training_minimum_icp_coverage_percent'] == 80
    assert example['configuration']['training_maximum_missing_reasoning_percent'] == 1
