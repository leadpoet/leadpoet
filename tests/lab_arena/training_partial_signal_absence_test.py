"""Missing final signal labels cannot erase independently proved source turns."""
from copy import deepcopy

import pytest

from lab_arena import (contracts, scoring, signal_judgments, training_dataset,
                       training_eligibility, training_export, training_thinking)
from tests.lab_arena.training_journal_candidates_test import journal_values, records
from tests.lab_arena.training_steps_test import seal


def failed_signal_values():
    values, _ = journal_values()
    configuration = values['round_row']['configuration_doc']
    annotation = values['ledger_rows'][0]['entry_doc']['reasoning_capture']
    policy = scoring.build_scorer_policy(
        judge_models={role: annotation['model_id'] for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={annotation['model_id']: annotation['provider_route']},
        scoring_adapter_version='qualification_integrity_v2', company_judgment_cache=True)
    configuration['scorer_policy'] = contracts.validate_scorer_policy({
        **policy, 'signal_judgment_policy': signal_judgments.POLICY})
    values['runs'][0].update(status='failed', terminal_cause='lease_expired', output_ref=None)
    values['runs'] = values['runs'][:1]
    return values


def test_real_shaped_failed_source_keeps_exact_pair_without_final_signal_or_label():
    values = failed_signal_values()
    example = training_export.assemble_example(**values)
    expected_label_gaps = ['failed_execution_output_unavailable', 'final_judge_unavailable',
                           'signal_authority_unverified']
    assert example['completeness']['reasons_by_dimension']['input_output_label'] == expected_label_gaps
    assert 'output' not in example and 'judgment' not in example and example['companies'] == []
    original = deepcopy(example)
    eligibility = training_eligibility.assess_capture(values['submission_row'], {values['run_id']: example},
        [values['run_id']], enabled=True, expected_run_count=1, minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=1)
    targets = records(example)
    assert len(targets) == 1, 'Absent final signal authority erased an independently proved pair'
    ref = targets[0]['source_refs'][0]
    assert ref['outcome'] == {'per_icp_score': None, 'score_run_id': None}
    assert ref['training_provenance']['label_authority']['origin'] == 'unavailable'
    assert ref['training_provenance']['company_outcomes'] == []
    assert targets[0]['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    assert ref['selected_research_evidence']['scope'] == 'individual_model_request_result'
    assert ref['selected_research_evidence']['complete'] is False
    assert ref['selected_research_evidence']['outcome_availability'] == 'unavailable'
    assert training_thinking.selected_research_evidence(example) is None
    full, _ = training_dataset.build([example], training_dataset.Rules(production_rounds_only=False))
    assert not full['train'] and not full['eval']
    assert not eligibility['eligible'] and eligibility['counts']['complete_runs'] == 0
    assert eligibility == training_eligibility.assess_capture(values['submission_row'], {values['run_id']: example},
        [values['run_id']], enabled=True, expected_run_count=1, minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=1)
    assert example == original


@pytest.mark.parametrize('damage', ['request_hash', 'response_hash', 'privacy', 'delivery',
                                    'call_run', 'claim', 'lease', 'context', 'source', 'output_gap'])
def test_absent_signal_label_does_not_waive_individual_proof_checks(damage):
    example = training_export.assemble_example(**failed_signal_values())
    call = example['provider_calls'][0]
    if damage == 'request_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response_hash': call['response']['choices'][0]['message']['content'] = 'changed bytes'
    elif damage == 'privacy': call['request_capture']['redacted'] = True
    elif damage == 'delivery': call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    elif damage == 'call_run': call['run_id'] = 'foreign-run'
    elif damage in {'claim', 'lease'}:
        snapshot = example['input_snapshots'][example['run_id']]
        snapshot['claim_request_hash' if damage == 'claim' else 'lease_generation'] = (
            contracts.hash_bytes(b'wrong') if damage == 'claim' else 2)
    elif damage in {'context', 'source'}:
        dimension = 'context' if damage == 'context' else 'source_identity'
        proof = example['completeness']
        proof['dimensions'][dimension] = False
        proof['reasons_by_dimension'][dimension] = ['context_invalid']
        proof['incomplete_reasons'].append('context_invalid')
    else:
        # A contradictory output reference is not the absence exception.
        proof = example['completeness']
        for key in ('input_output_label', 'research_process'):
            proof['reasons_by_dimension'][key] = [
                'execution_output_reference_mismatch' if reason == 'failed_execution_output_unavailable' else reason
                for reason in proof['reasons_by_dimension'][key]]
        proof['incomplete_reasons'] = sorted(set().union(*(
            set(value) for value in proof['reasons_by_dimension'].values())))
    assert records(seal(example)) == []


@pytest.mark.parametrize('damage', ['source_membership', 'signal_identity', 'signal_authority'])
def test_actual_membership_and_signal_authority_validation_remain_strict(damage):
    if damage == 'source_membership':
        values = failed_signal_values()
        values['submission_row']['miner_hotkey'] = 'foreign-miner'
        with pytest.raises(training_export.ExportError, match='context_membership_mismatch'):
            training_export.assemble_example(**values)
        return
    from tests.lab_arena.training_export_signal_test import _signal_fixture
    values = _signal_fixture()
    if damage == 'signal_identity':
        values['signal_judgment_rows'][0]['scope_doc']['verified_company_identity']['observed_name'] = 'Foreign'
    else:
        values['signal_judgment_rows'] = []
    example = training_export.assemble_example(**values)
    assert 'judgment' not in example
    assert 'signal_authority_unverified' in example['completeness']['incomplete_reasons']
    assert example.get('score_run_id') is not None
    assert training_thinking._partial_research_evidence(example) is None
