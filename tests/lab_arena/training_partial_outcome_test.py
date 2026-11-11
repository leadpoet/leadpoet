"""Exact delivered turns survive absent outcomes, without creating labels."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_export, training_steps, training_thinking
from tests.lab_arena.judge_thinking_export_test import judge_example
from tests.lab_arena.training_steps_test import seal


def partial_example(*, failed_execution=False, final_judge=False):
    example = judge_example()
    configuration = example['configuration']
    configuration['scorer_image_reference'] = 'registry.example/scorer@' + configuration['scorer_image_digest']
    dimensions = ('context', 'source_identity', 'input_output_label', 'exact_inputs', 'costs', 'research_process')
    by_dimension = {name: [] for name in dimensions}
    if not final_judge:
        example.pop('judgment')
        example.pop('score_run_id')
        example['companies'] = []
        by_dimension['input_output_label'] = ['final_judge_unavailable']
        by_dimension['research_process'] = ['final_judge_unavailable']
    for attempt in example['attempts']:
        identity, kind = attempt['run_id'], attempt['kind']
        attempt.update(lease_generation=1, claim_request_hash=contracts.hash_bytes(identity.encode()))
        if kind == 'score':
            attempt.update(status='failed', scored_run_id=example['run_id'], runtime_identity={
                'selection_basis': 'frozen_round',
                **{prefix + '_image_' + kind: configuration['scorer_image_' + kind]
                   for prefix in ('grading', 'runtime') for kind in ('digest', 'reference')}})
        snapshot = example['input_snapshots'][identity]
        snapshot.update(schema_version='leadpoet.lab_arena.training_input.v1', run_id=identity,
                        kind=kind, lease_generation=1, claim_request_hash=attempt['claim_request_hash'])
        snapshot['input'].update(evaluation_date=example.get('evaluation_date'), icp=example['icp'])
        if kind == 'score':
            snapshot['input']['scorer_policy'] = example['configuration']['scorer_policy']
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    if failed_execution:
        example['attempts'][0]['status'] = 'failed'
        example.pop('output')
        example.pop('output_identity')
        by_dimension['input_output_label'].append('failed_execution_output_unavailable')
        by_dimension['research_process'].append('failed_execution_output_unavailable')
    example['completeness'] = {
        'complete': not any(by_dimension.values()),
        'dimensions': {key: not value for key, value in by_dimension.items()},
        'reasons_by_dimension': by_dimension,
        'incomplete_reasons': sorted({reason for values in by_dimension.values() for reason in values}),
    }
    return seal(example)


def built(example, role='source'):
    splits, manifest = training_thinking.build([seal(example)], role=role)
    return splits['train'] + splits['eval'], manifest


@pytest.mark.parametrize('failed_execution', [False, True])
@pytest.mark.parametrize('role', ['source', 'judge'])
def test_partial_outcomes_keep_only_proven_turns_and_unknown_labels(role, failed_execution):
    example = partial_example(failed_execution=failed_execution)
    original = deepcopy(example)
    records, _ = built(example, role)
    assert len(records) == 1
    assert example == original
    record = records[0]
    assert record['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    ref = record['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['training_provenance']['label_authority']['origin'] == 'unavailable'
    proof = ref['selected_judge_evidence' if role == 'judge' else 'selected_research_evidence']
    assert proof['complete'] is False
    assert proof['scope'] == 'individual_model_request_result'
    assert proof['outcome_availability'] == 'unavailable'
    assert training_thinking.selected_research_evidence(example, role=role) is None


@pytest.mark.parametrize('role', ['source', 'judge'])
@pytest.mark.parametrize('damage', ['request', 'response', 'privacy', 'failed_write', 'summary', 'source', 'context', 'diagnostics'])
def test_partial_outcome_does_not_waive_byte_privacy_delivery_or_context_checks(role, damage):
    example = partial_example()
    call = example['provider_calls'][0 if role == 'source' else -1]
    if damage == 'request': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response': call['response']['changed'] = True
    elif damage == 'privacy': call['request_capture'] = {'redacted': True}
    elif damage == 'failed_write': call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'summary': call['reasoning_capture']['semantic_class'] = 'summary'
    elif damage in {'source', 'context'}:
        example['completeness']['dimensions']['source_identity' if damage == 'source' else 'context'] = False
    else: example['completeness']['incomplete_reasons'].append('invented')
    assert built(example, role)[0] == []


def test_failed_judge_requires_its_saved_execution_and_policy_binding():
    for damage in ('run', 'snapshot', 'policy', 'lease', 'privacy'):
        example = partial_example()
        attempt = example['attempts'][-1]
        snapshot = example['input_snapshots'][attempt['run_id']]
        if damage == 'run': attempt['scored_run_id'] = 'foreign'
        elif damage == 'snapshot': del example['input_snapshots'][attempt['run_id']]
        elif damage == 'policy': snapshot['input']['scorer_policy'] = {}
        elif damage == 'lease': snapshot['lease_generation'] = 2
        else: snapshot['privacy'] = {'redacted': True, 'redactions': ['credential']}
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
        assert built(example, 'judge')[0] == []


def test_absent_judge_does_not_remove_exact_source_pairs():
    example = partial_example()
    score_id = example['attempts'][-1]['run_id']
    example['attempts'].pop()
    example['provider_calls'].pop()
    example['input_snapshots'].pop(score_id)
    example['worker_observation_artifacts'].pop(score_id)
    assert len(built(example)[0]) == 1
    plain, _ = training_steps.build([seal(example)])
    assert len(plain['train'] + plain['eval']) == 1


@pytest.mark.parametrize('damage', ['hash', 'privacy', 'claim', 'icp', 'lease', 'date'])
def test_partial_outcome_does_not_waive_present_execution_input_integrity(damage):
    example = partial_example()
    snapshot = example['input_snapshots'][example['run_id']]
    if damage == 'hash': snapshot['input_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'privacy': snapshot['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'claim': snapshot['claim_request_hash'] = contracts.hash_bytes(b'other')
    elif damage == 'lease': snapshot['lease_generation'] = 2
    else:
        snapshot['input']['icp' if damage == 'icp' else 'evaluation_date'] = {} if damage == 'icp' else '1900-01-01'
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    assert built(example)[0] == []


def test_failed_judge_turn_does_not_borrow_a_later_accepted_attempt_outcome():
    example = partial_example(final_judge=True)
    accepted = deepcopy(example['attempts'][-1])
    accepted.update(run_id='later-accepted-judge', status='accepted')
    snapshot = deepcopy(example['input_snapshots'][example['score_run_id']])
    snapshot.update(run_id=accepted['run_id'])
    example['score_run_id'] = accepted['run_id']
    example['input_snapshots'][accepted['run_id']] = snapshot
    example['attempts'].append(accepted)
    seal(example)
    before = deepcopy(example)
    source, _ = built(example)
    judges, _ = built(example, 'judge')
    assert len(source) == len(judges) == 1
    assert source[0]['source_refs'][0]['outcome']['per_icp_score'] == 1
    ref = judges[0]['source_refs'][0]
    assert ref['run_id'] != example['score_run_id']
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['judge_context']['judgment_sha256'] is None
    assert ref['judge_context']['attempt_status'] == 'failed'
    assert ref['training_provenance']['label_authority']['origin'] == 'unavailable'
    assert ref['training_provenance']['score_run_id'] is None
    assert ref['training_provenance']['judgment_sha256'] is None
    assert ref['training_provenance']['company_outcomes'] == []
    assert ref['selected_judge_evidence']['judge_full_thinking_complete'] is False
    assert example == before


def test_export_marks_failed_execution_absence_and_binds_score_attempts():
    from tests.lab_arena.training_export_test import fixture
    values = fixture()
    values['runs'][0].update(status='failed', terminal_cause='worker_error', output_ref=None)
    values['runs'][1].update(status='failed', terminal_cause='judge_timeout')
    example = training_export.assemble_example(**values)
    assert 'failed_execution_output_unavailable' in example['completeness']['incomplete_reasons']
    assert 'output' not in example and 'judgment' not in example and not example['companies']
    assert example['attempts'][-1]['scored_run_id'] == example['run_id']
    assert example['attempts'][0]['claim_request_hash'] == values['runs'][0]['claim_request_hash']


@pytest.mark.parametrize('damage', [None, 'claim', 'privacy', 'hash'])
def test_failed_judge_journal_prefix_keeps_exact_turn_and_unknown_outcome(damage):
    from lab_arena import training_observations as obs
    example = partial_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    score_id = example['attempts'][-1]['run_id']
    call = example['provider_calls'][-1]
    artifact = example['worker_observation_artifacts'].pop(score_id)
    claim = example['input_snapshots'][score_id]['claim_request_hash']
    artifact.update(schema_version=obs.JOURNAL_SCHEMA_VERSION, complete=False, lease_generation=1,
                    claim_request_hash=claim, segment_sha256=['sha256:' + 'd' * 64],
                    segment_refs=[obs.segment_ref(score_id, 1, claim, 0)])
    artifact['payload']['complete'] = False
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    call['response_provenance']['worker_receipt_ref']['artifact_field'] = 'partial_worker_observation_artifacts'
    example['partial_worker_observation_artifacts'] = {score_id: artifact}
    proof = example['completeness']
    proof['incomplete_reasons'].append('worker_observations_incomplete')
    proof['reasons_by_dimension']['research_process'].append('worker_observations_incomplete')
    if damage == 'claim': artifact['claim_request_hash'] = contracts.hash_bytes(b'other')
    elif damage == 'privacy': example['input_snapshots'][score_id]['privacy']['redacted'] = True
    elif damage == 'hash': artifact['payload_sha256'] = contracts.hash_bytes(b'wrong')
    records, _ = built(example, 'judge')
    assert len(records) == (1 if damage is None else 0)
    if records:
        assert records[0]['source_refs'][0]['selected_judge_evidence']['complete'] is False


@pytest.mark.parametrize("missing_result", [False, True])
def test_real_failed_run_export_keeps_proven_journal_pair_without_final_output_or_judge(missing_result):
    from lab_arena.training_reader import export_run
    from tests.lab_arena.training_journal_candidates_test import journal_values
    values, _ = journal_values()
    values['runs'][0].update(status='failed', terminal_cause='lease_expired', output_ref=None)
    values['runs'] = values['runs'][:1]
    if missing_result:
        values['runs'][0]['result_doc'] = {}
    class Store:
        def get_run(self, run_id): return values['runs'][0]
        def get_round(self, round_id): return values['round_row']
        def get_submission(self, submission_id): return values['submission_row']
        def list_runs(self, round_id): return values['runs']
        def list_ledger(self, **identity): return values['ledger_rows']
    example = export_run(Store(), values['objects'], values['run_id'])
    records, _ = built(example)
    assert len(records) == 1
    assert records[0]['source_refs'][0]['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert 'output' not in example and 'judgment' not in example and not example['companies']
    assert training_thinking.selected_research_evidence(example) is None


@pytest.mark.parametrize('failed_execution', [False, True])
def test_real_partial_judge_export_keeps_exact_journal_reply_without_final_label(failed_execution):
    from lab_arena import scoring, training_observations as obs
    from tests.lab_arena.training_export_test import CLAIM_HASH, save_input
    from tests.lab_arena.training_journal_candidates_test import journal_values
    values, original_segment = journal_values()
    configuration = values['round_row']['configuration_doc']
    source, judge = values['runs']
    annotation = values['ledger_rows'][0]['entry_doc']['reasoning_capture']
    configuration['scorer_policy'] = scoring.build_scorer_policy(
        judge_models={role: annotation['model_id'] for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={annotation['model_id']: annotation['provider_route']})
    old_input = training_export._json_object(values['objects'], values['round_row']['benchmark_ref'], 1024 * 1024)
    output = training_export._json_object(values['objects'], source['output_ref'], 1024 * 1024)
    save_input(values['objects'], judge['run_id'], 'score', scoring.build_scoring_input(
        scored_run_id=source['run_id'], icp=old_input['icps'][0], companies=output['companies'],
        policy=configuration['scorer_policy'], evaluation_date=values['round_row']['evaluation_date']))
    judge.update(status='failed', terminal_cause='judge_timeout')
    judge['result_doc']['resource_summary']['provider_call_count'] = 1
    for row in values['ledger_rows']:
        row['run_id'] = judge['run_id']
    payload = original_segment['payload']
    _, raw = obs.build_segment(judge['run_id'], 'score', payload, lease_generation=1,
        claim_request_hash=CLAIM_HASH, number=0, previous_sha256=None,
        cumulative_bytes=len(contracts.canonical_json(payload).encode()))
    values['objects'].put(obs.segment_ref(judge['run_id'], 1, CLAIM_HASH, 0), raw)
    if failed_execution:
        source.update(status='failed', terminal_cause='worker_error', output_ref=None)
    example = training_export.assemble_example(**values)
    records, manifest = built(example, 'judge')
    assert len(records) == 1, manifest
    assert example['companies'] == [] and 'judgment' not in example
    assert records[0]['source_refs'][0]['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    if failed_execution:
        assert 'judge_input_context_mismatch' in example['completeness']['incomplete_reasons']
        assert records[0]['source_refs'][0]['judge_context']['scored_output_sha256'] is None


@pytest.mark.parametrize('role', ['source', 'judge'])
def test_partial_turns_still_deduplicate_without_mutating_raw_attempts(role):
    example = partial_example()
    before = deepcopy(example)
    once, manifest = training_thinking.build([example], role=role)
    twice, duplicate_manifest = training_thinking.build([example, deepcopy(example)], role=role)
    assert once == twice
    assert manifest['split_source_hashes'] == duplicate_manifest['split_source_hashes']
    assert example == before


@pytest.mark.parametrize('reason', ['runtime_input_snapshot_invalid', 'runtime_input_privacy_nonexact',
                                    'runtime_input_context_mismatch', 'judge_input_context_mismatch'])
def test_partial_source_missing_input_does_not_erase_invalid_removed_input(reason):
    example = partial_example()
    del example['input_snapshots'][example['run_id']]
    proof = example['completeness']
    proof['incomplete_reasons'].extend(['runtime_input_snapshot_missing', reason])
    proof['reasons_by_dimension']['exact_inputs'] = ['runtime_input_snapshot_missing', reason]
    proof['dimensions']['exact_inputs'] = False
    assert built(example)[0] == []


@pytest.mark.parametrize('damage', ['missing_runtime', 'runtime_grading', 'runtime_lease', 'runtime_reference', 'runtime_gap'])
def test_failed_judge_cannot_borrow_unverified_runtime_identity(damage):
    example = partial_example()
    attempt = example['attempts'][-1]
    runtime = attempt['runtime_identity']
    if damage == 'missing_runtime': attempt.pop('runtime_identity')
    elif damage == 'runtime_grading': runtime['grading_image_digest'] = 'sha256:' + 'd' * 64
    elif damage == 'runtime_lease':
        runtime.update(selection_basis='persisted_lease_binding', schema_version='leadpoet.lab_arena.reviewer_runtime.v1',
                       run_id=attempt['run_id'], lease_generation=2)
    elif damage == 'runtime_reference': runtime['runtime_image_reference'] = 'unbound'
    else:
        example['related_run_gap_attribution'] = {attempt['run_id']: {
            'research_process_gaps': ['prior_attempt_reviewer_runtime_identity_unverified']}}
    assert built(example, 'judge')[0] == []


@pytest.mark.parametrize('role', ['source', 'judge'])
def test_completed_tool_choice_survives_later_run_failure_without_claiming_tool_execution(role):
    from tests.lab_arena.judge_chat_annotations_test import bind_response
    example = partial_example(failed_execution=True)
    if role == 'source':
        example['provider_calls'].pop()  # Bind the source call with the same byte-proof helper.
    call = example['provider_calls'][-1]
    body = call['request']['outbound']['body']
    body['tools'] = [{'type': 'function', 'function': {
        'name': 'lookup', 'parameters': {'type': 'object'}}}]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    choice = call['response']['choices'][0]
    choice['finish_reason'] = 'tool_calls'
    choice['message'].update(content=None, tool_calls=[{
        'id': 'lookup-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}])
    bind_response(example)
    records, _ = built(example, role)
    assert len(records) == 1
    assert records[0]['target']['tool_calls'] == choice['message']['tool_calls']
    assert records[0]['source_refs'][0]['outcome']['per_icp_score'] is None
    assert records[0]['quality']['correct_action'] == 'not_established'


def test_partial_source_missing_runtime_input_retains_exact_pair_and_explicit_absence():
    example = partial_example()
    del example['input_snapshots'][example['run_id']]
    proof = example['completeness']
    proof['incomplete_reasons'].append('runtime_input_snapshot_missing')
    proof['reasons_by_dimension']['exact_inputs'] = ['runtime_input_snapshot_missing']
    proof['dimensions']['exact_inputs'] = False
    records, _ = built(example)
    assert len(records) == 1
    evidence = records[0]['source_refs'][0]['selected_research_evidence']
    assert evidence['complete'] is False
    assert evidence['runtime_input_availability'] == 'unavailable'
    assert evidence['missing_runtime_input_run_ids'] == [example['run_id']]
    assert 'output' not in records[0] and 'judgment' not in records[0]
