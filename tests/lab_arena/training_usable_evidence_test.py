"""Usable accepted data is independent of unrelated failures and billing."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    assess_ten, failed_score_gap_example, judge_only_gap_example, refresh_response_links,
    submission, ten_selected_examples,
)
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import seal


def usable_example():
    example = judge_only_gap_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    return seal(example)


def check(example):
    return training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)


def exported(example):
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    return splits['train'] + splits['eval']


def test_eight_usable_icps_survive_two_corrupt_icps_with_rejection_diagnostics():
    examples = ten_selected_examples()
    for run_id in ('run-8', 'run-9'):
        examples[run_id]['provider_calls'][2]['reasoning_capture']['provider_route'] = 'wrong-route'
        seal(examples[run_id])
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 8
    assert {row['run_id'] for row in result['counts']['rejected_runs']} == {'run-8', 'run-9'}
    assert all(row['reasons'] for row in result['counts']['rejected_runs'])
    examples['run-7'] = None
    assert not assess_ten(examples)['eligible']


def test_unpriced_exact_selected_pairs_are_usable_without_changing_billing():
    example = usable_example()
    call = example['provider_calls'][1]
    call.update(accounting_state='uncertain', actual_microusd=None,
                evidence_gaps=['provider_cost_unresolved'])
    example['competition_cost_eligibility'].update(status='unavailable', reason='provider_cost_uncertain')
    proof = example['completeness']
    proof['incomplete_reasons'].append('provider_cost_unresolved')
    proof['dimensions']['costs'] = False
    proof['reasons_by_dimension']['costs'] = ['provider_cost_unresolved']
    proof['reasons_by_dimension']['research_process'].append('provider_cost_unresolved')
    seal(example)
    original = deepcopy(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2
    assert example == original


def test_failed_execute_attempt_cannot_poison_exact_accepted_context():
    example = without_client_callbacks(failed_score_gap_example())
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    example['attempts'][-1]['kind'] = 'execute'
    example['related_run_gap_attribution']['failed-score']['kind'] = 'execute'
    for key in ('incomplete_reasons',):
        example['completeness'][key].append('agent_dependency_link_invalid')
    example['completeness']['reasons_by_dimension']['research_process'].append('agent_dependency_link_invalid')
    example['related_run_gap_attribution']['failed-score']['research_process_gaps'].append('agent_dependency_link_invalid')
    seal(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2


@pytest.mark.parametrize('damage', ['request_hash', 'response_hash', 'privacy', 'required_tool_context', 'source', 'score'])
def test_selected_corrupt_private_or_unbound_data_is_not_usable(damage):
    example = usable_example()
    call = example['provider_calls'][1]
    if damage == 'request_hash':
        call['request']['request_hash'] = 'sha256:' + '0' * 64
    elif damage == 'response_hash':
        call['response']['changed'] = True
    elif damage == 'privacy':
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'required_tool_context':
        body = call['request']['outbound']['body']
        body['messages'].append({'role': 'tool', 'tool_call_id': 'missing', 'content': 'result'})
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    elif damage == 'source':
        example['completeness']['dimensions']['source_identity'] = False
        example['completeness']['reasons_by_dimension']['source_identity'] = ['source_unverified']
        example['completeness']['incomplete_reasons'].append('source_unverified')
    else:
        example['input_snapshots']['score']['input']['scored_run_id'] = 'wrong-run'
    seal(example)
    assert not check(example)['eligible']
    records = exported(example)
    if damage in {'source', 'score'}:
        assert not records
    else:
        assert len(records) == 1
        assert all(ref['call_identity'] != call['call_identity']
                   for record in records for ref in record['source_refs'])


def test_exporter_retains_valid_thinking_from_a_reasoning_ineligible_icp():
    example = usable_example()
    call = example['provider_calls'][2]
    message = call['response']['choices'][0]['message']
    for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
        message.pop(field, None)
    call['response']['usage']['completion_tokens_details']['reasoning_tokens'] = 900
    refresh_response_links(example, 2)
    assert not check(example)['eligible']
    records = exported(example)
    assert len(records) == 1
    assert records[0]['source_refs'][0]['call_identity'] == example['provider_calls'][1]['call_identity']
    assert records[0]['generated_thinking']


@pytest.mark.parametrize('damage', ['score_hash', 'score_binding', 'output', 'context'])
def test_complete_flag_cannot_hide_a_broken_selected_label(damage):
    from tests.lab_arena.training_eligibility_test import captured_example
    example = captured_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    if damage == 'score_hash':
        example['input_snapshots'][example['score_run_id']]['input_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'score_binding':
        snapshot = example['input_snapshots'][example['score_run_id']]
        snapshot['input']['scored_run_id'] = 'other'
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    elif damage == 'context':
        example['completeness']['dimensions'].pop('context')
    else:
        example['output']['companies'] = [{'company': 'changed'}]
    seal(example)
    assert not check(example)['eligible']
    assert not exported(example)


def test_unrelated_worker_dependency_does_not_poison_self_contained_provider_input():
    example = usable_example()
    example['related_run_gap_attribution'] = {'run': {
        'kind': 'execute', 'status': 'accepted', 'research_process_gaps': [], 'cost_gaps': []}}
    artifact = example['worker_observation_artifacts']['run']
    artifact['payload']['agent_observations'][0].update(
        depends_on=['sha256:' + '0' * 64], after_action_sequence=0)
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in example['provider_calls']:
        if call['run_id'] != 'run':
            continue
        provenance = call['response_provenance']
        provenance['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
        for ref in provenance.get('agent_receipt_refs') or []:
            ref['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2


@pytest.mark.parametrize('damage', ['request', 'receipt', 'privacy', 'route'])
def test_bad_later_target_does_not_erase_an_independent_earlier_pair(damage):
    example = usable_example()
    bad = example['provider_calls'][2]
    if damage == 'request':
        bad['request']['outbound']['body_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'receipt':
        bad['response_receipts'][0]['wire_body_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'privacy':
        bad['response_privacy_redactions'] = ['credential']
    else:
        bad['reasoning_capture']['provider_route'] = 'wrong-route'
    seal(example)
    assert not check(example)['eligible']
    records = exported(example)
    assert len(records) == 1
    assert records[0]['source_refs'][0]['call_identity'] == example['provider_calls'][1]['call_identity']


def test_unrelated_external_tool_receipt_does_not_erase_exact_provider_visible_input():
    from tests.lab_arena.training_eligibility_test import _add_settled_exa_call
    example = _add_settled_exa_call(usable_example())
    assert check(example)['eligible']
    example['provider_calls'][-1]['response_receipts'][0]['wire_body_sha256'] = 'sha256:' + '0' * 64
    seal(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2


def test_unrelated_attempt_link_diagnostics_do_not_poison_a_proven_failed_model_attempt():
    from tests.lab_arena.training_call_semantics_test import _failed_example
    example = _failed_example(502, worker_error=True)
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    example['completeness']['incomplete_reasons'] = ['agent_dependency_link_invalid']
    example['completeness']['complete'] = False
    example['attempts'].append({'run_id': 'unrelated', 'kind': 'execute', 'status': 'failed'})
    seal(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2


@pytest.mark.parametrize('setting,value', [
    ('training_maximum_missing_reasoning_percent', True),
    ('training_maximum_missing_reasoning_percent', 101),
    ('training_minimum_icp_coverage_percent', 0),
    ('reasoning_capture_policy', 'unsupported'),
])
def test_invalid_frozen_policy_is_not_salvaged_as_training_data(setting, value):
    example = usable_example()
    example['configuration'][setting] = value
    seal(example)
    assert not check(example)['eligible']
    assert not exported(example)


@pytest.mark.parametrize('field,value', [
    ('run_id', 'another-run'), ('kind', 'score'), ('schema_version', 'unknown'),
    ('lease_generation', True), ('lease_generation', 0), ('claim_request_hash', 'invalid'),
])
def test_present_worker_snapshot_identity_metadata_must_agree(field, value):
    example = usable_example()
    example['worker_observation_artifacts']['run'][field] = value
    seal(example)
    assert not check(example)['eligible']
    assert not exported(example)


def test_present_valid_worker_snapshot_identity_metadata_preserves_exact_pairs():
    from lab_arena import training_observations
    example = usable_example()
    example['worker_observation_artifacts']['run'].update(
        run_id='run', kind='execute', schema_version=training_observations.SCHEMA_VERSION,
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b'fixture-claim'))
    seal(example)
    assert check(example)['eligible']
    assert len(exported(example)) == 2


@pytest.mark.parametrize('field,value', [
    ('lease_generation', 2), ('claim_request_hash', contracts.hash_bytes(b'another-claim')),
])
def test_present_snapshot_identity_must_match_known_attempt_metadata(field, value):
    example = usable_example()
    expected = {'lease_generation': 1, 'claim_request_hash': contracts.hash_bytes(b'fixture-claim')}
    example['attempts'][0].update(expected)
    artifact = example['worker_observation_artifacts']['run']
    artifact.update(expected)
    artifact[field] = value
    seal(example)
    assert not check(example)['eligible']
    assert not exported(example)
