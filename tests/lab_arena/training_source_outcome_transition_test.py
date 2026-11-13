"""A final judge label cannot erase an independently verified source turn."""
from copy import deepcopy
import gzip
import json

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_input, training_steps, training_thinking
from tests.lab_arena.training_partial_outcome_test import partial_example
from tests.lab_arena.training_retained_unlabeled_pairs_test import retained_signal_label_example
from tests.lab_arena.training_retry_snapshot_test import retry_values
from tests.lab_arena.training_steps_test import seal


RULES = training_steps.Rules(production_rounds_only=False)


def transition(damage):
    values, _, _ = retry_values()
    values['round_row']['configuration_doc']['training_maximum_missing_reasoning_percent'] = 1
    judge = next(row for row in values['runs'] if row['kind'] == 'score')
    ref = training_input.object_ref(judge['run_id'], judge['lease_generation'])
    if damage == 'missing':
        del values['objects'].values[ref]
    elif damage == 'gzip':
        values['objects'].values[ref] = b'invalid gzip'
    elif damage in {'hash', 'schema'}:
        snapshot = json.loads(gzip.decompress(values['objects'].values[ref]))
        snapshot['input_sha256' if damage == 'hash' else 'schema_version'] = 'invalid'
        values['objects'].values[ref] = gzip.compress(contracts.canonical_json(snapshot).encode())
    pending = deepcopy(values)
    next(row for row in pending['runs'] if row['kind'] == 'score').update(status='pending', output_ref=None)
    return (training_export.assemble_example(**pending),
            training_export.assemble_example(**values), values['submission_row'])


def records(example, *, thinking=True, role='source'):
    splits, manifest = (training_thinking.build([example], RULES, role=role) if thinking
                        else training_steps._build([example], RULES, role=role))
    return splits['train'] + splits['eval'], manifest


def assert_unlabeled(record):
    ref = record['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    provenance = ref['training_provenance']
    assert provenance['label_authority']['origin'] == 'unavailable'
    assert provenance['label_authority']['input_output_label_verified'] is False
    assert provenance['score_run_id'] is provenance['judgment_sha256'] is None
    assert provenance['company_outcomes'] == []


@pytest.mark.parametrize('damage', ['missing', 'gzip', 'hash', 'schema'])
def test_native_final_label_keeps_source_pairs_without_unverified_outcomes(damage):
    before, after, submission = transition(damage)
    original = deepcopy(after)
    source_calls = lambda e: [c for c in e['provider_calls'] if c['run_id'] == e['run_id']]
    assert source_calls(before) == source_calls(after)
    assert before['input_snapshots'][before['run_id']] == after['input_snapshots'][after['run_id']]
    assert after['judgment']['per_icp_score'] == 12.75
    assert after['completeness']['dimensions']['input_output_label'] is True
    assert after['completeness']['dimensions']['exact_inputs'] is False
    for call in source_calls(after):
        assert training_steps._candidate(after, call)
    baseline, _ = records(before)
    retained, _ = records(after)
    assert len(baseline) == len(retained) == 2
    assert {r['candidate_id'] for r in baseline} == {r['candidate_id'] for r in retained}
    for record in retained:
        assert_unlabeled(record)
        evidence = record['source_refs'][0]['selected_research_evidence']
        assert evidence['scope'] == 'individual_model_request_result'
        assert evidence['complete'] is False
        assert evidence['outcome_availability'] == 'unavailable'
    plain, _ = records(after, thinking=False)
    assert len(plain) == 3
    for record in plain:
        assert_unlabeled(record)
    for example in (before, after):
        assessment = training_eligibility.assess_capture(
            submission, {example['run_id']: example}, [example['run_id']], enabled=True,
            expected_run_count=1, minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
        assert assessment['eligible'] is False
        assert assessment['counts']['complete_runs'] == 0
    assert after == original


def test_verified_final_input_keeps_existing_outcome_and_pairs():
    before, after, _ = transition('valid')
    original = deepcopy(after)
    baseline, _ = records(before)
    for thinking, count in ((True, 2), (False, 3)):
        retained, _ = records(after, thinking=thinking)
        assert len(retained) == count
        for record in retained:
            ref = record['source_refs'][0]
            assert ref['outcome'] == {'per_icp_score': 12.75, 'score_run_id': after['score_run_id']}
            assert ref['training_provenance']['label_authority']['origin'] == 'automated_native_scorer'
        if thinking:
            assert {r['candidate_id'] for r in baseline} == {r['candidate_id'] for r in retained}
    assert after == original


@pytest.mark.parametrize('damage', ['valid', 'missing', 'gzip', 'hash', 'schema'])
def test_historical_provenance_projection_preserves_outcome_scope(damage):
    from tests.lab_arena.training_thinking_reuse_test import historical_provenance_projection

    _, example, _ = transition(damage)
    current, _ = records(example)
    with historical_provenance_projection():
        historical, _ = records(example)
    projected = deepcopy(current)
    for record in projected:
        for ref in record['source_refs']:
            ref.pop('training_provenance')
    assert len(historical) == 2
    assert historical == projected
    assert all(record['source_refs'][0]['selected_research_evidence']['complete']
               is (damage == 'valid') for record in historical)


@pytest.mark.parametrize('damage', ['request_hash', 'response_body', 'request_privacy',
    'response_privacy', 'delivery', 'source_input_hash', 'source_input_privacy',
    'source_identity', 'output', 'foreign_judge', 'unknown_label_gap'])
def test_unverified_judge_input_does_not_waive_source_integrity(damage):
    _, example, _ = transition('gzip')
    before, _ = records(transition('valid')[1])
    identity = before[0]['source_refs'][0]['call_identity']
    call = next(c for c in example['provider_calls'] if c['call_identity'] == identity)
    if damage == 'request_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'response_body': call['response']['changed'] = True
    elif damage == 'request_privacy': call['request_capture'] = {'redacted': True}
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'delivery': call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'source_input_hash': example['input_snapshots'][example['run_id']]['input_sha256'] = 'invalid'
    elif damage == 'source_input_privacy': example['input_snapshots'][example['run_id']]['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'source_identity':
        proof = example['completeness']
        proof['reasons_by_dimension']['source_identity'] = ['source_unverified']
        proof['dimensions']['source_identity'] = False
        proof['incomplete_reasons'].append('source_unverified')
    elif damage == 'output': example['output']['changed'] = True
    elif damage == 'foreign_judge':
        next(r for r in example['attempts'] if r['kind'] == 'score')['scored_run_id'] = 'foreign'
    else:
        proof = example['completeness']
        proof['reasons_by_dimension']['input_output_label'].append('unknown_label_failure')
        proof['dimensions']['input_output_label'] = False
        proof['incomplete_reasons'].append('unknown_label_failure')
    retained, _ = records(seal(example))
    local_damage = damage in {'request_hash', 'response_body', 'request_privacy', 'response_privacy', 'delivery'}
    assert len(retained) == int(local_damage)
    assert not any(ref['call_identity'] == identity for r in retained for ref in r['source_refs'])


def test_final_label_does_not_rescue_invalid_judge_turns():
    example = retained_signal_label_example()
    final = partial_example(final_judge=True)
    example['judgment'], example['companies'] = final['judgment'], final.get('companies', [])
    del example['input_snapshots'][example['score_run_id']]
    proof = example['completeness']
    proof['reasons_by_dimension']['input_output_label'] = []
    proof['dimensions']['input_output_label'] = True
    proof['reasons_by_dimension']['exact_inputs'] = ['runtime_input_snapshot_invalid', 'runtime_input_snapshot_missing']
    proof['dimensions']['exact_inputs'] = False
    proof['reasons_by_dimension']['research_process'] = proof['reasons_by_dimension']['exact_inputs'][:]
    proof['dimensions']['research_process'] = False
    proof['incomplete_reasons'] = sorted(set(proof['reasons_by_dimension']['exact_inputs']))
    seal(example)
    source, _ = records(example)
    judges, _ = records(example, role='judge')
    assert len(source) == 1
    assert_unlabeled(source[0])
    assert judges == []


def test_prior_judge_input_warning_does_not_strip_verified_selected_outcome():
    _, example, _ = transition('valid')
    proof = example['completeness']
    gaps = ['runtime_input_snapshot_invalid', 'runtime_input_snapshot_missing']
    proof['reasons_by_dimension']['exact_inputs'] = gaps
    proof['dimensions']['exact_inputs'] = False
    proof['reasons_by_dimension']['research_process'] = sorted(set(proof['reasons_by_dimension']['research_process'] + gaps))
    proof['dimensions']['research_process'] = False
    proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons'] + gaps))
    proof['complete'] = False
    retained, _ = records(seal(example))
    assert len(retained) == 2
    assert all(r['source_refs'][0]['outcome']['per_icp_score'] == 12.75 for r in retained)


@pytest.mark.parametrize('damage', ['hash', 'schema'])
def test_present_invalid_judge_input_and_missing_prior_input_keep_only_source_pairs(damage):
    _, example, _ = transition('gzip')
    _, complete, _ = transition('valid')
    identity = example['score_run_id']
    snapshot = deepcopy(complete['input_snapshots'][identity])
    snapshot['input_sha256' if damage == 'hash' else 'schema_version'] = 'invalid'
    example['input_snapshots'][identity] = snapshot
    prior = deepcopy(next(row for row in example['attempts'] if row['run_id'] == identity))
    prior.update(run_id='prior-judge', status='failed', terminal_cause='judge_timeout')
    example['attempts'].append(prior)
    seal(example)
    original = deepcopy(example)
    assert not training_steps._individual_run_input_valid(example, identity, 'score')
    # Native assembly removes invalid envelopes (covered above). This direct
    # malformed-export control checks label masking without tightening the
    # historical selected-input binder or full-ICP assessment in this change.
    if damage == 'hash':
        assert training_thinking.selected_research_evidence(example) is None
    for thinking, count in ((True, 2), (False, 3)):
        retained, _ = records(example, thinking=thinking)
        assert len(retained) == count
        for record in retained:
            assert_unlabeled(record)
    assert example == original


@pytest.mark.parametrize('thinking,expected', [(False, 1), (True, 2)])
def test_source_outcome_validation_runs_once_per_example_pass(monkeypatch, thinking, expected):
    _, example, _ = transition('gzip')
    calls = []
    original = training_steps._source_outcome_unavailable

    def checked(value):
        calls.append(value['example_hash'])
        return original(value)

    monkeypatch.setattr(training_steps, '_source_outcome_unavailable', checked)
    retained, _ = records(example, thinking=thinking)
    assert len(retained) == (2 if thinking else 3)
    # Thinking also has one independent coverage pass. Neither pass hashes
    # source/judge inputs again for each of the three source calls.
    assert calls == [example['example_hash']] * expected
