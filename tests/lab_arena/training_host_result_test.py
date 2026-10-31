"""New frozen policy uses existing organizer capture, without client callbacks."""
from copy import deepcopy

import pytest

from lab_arena import contracts, reasoning_capture, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_steps_test import seal


def without_client_callbacks(example=None):
    example = deepcopy(example or captured_example())
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    for run_id, artifact in example['worker_observation_artifacts'].items():
        payload = artifact['payload']
        payload['agent_observations'] = []
        artifact['payload_sha256'] = contracts.document_hash(payload)
        for call in example['provider_calls']:
            if call['run_id'] != run_id:
                continue
            provenance = call['response_provenance']
            for key in ('agent_receipt_refs', 'selected_response_acknowledged', 'worker_error_body_acknowledged'):
                provenance.pop(key, None)
            provenance['agent_receipt'] = 'not_reported'
            for name in ('worker_receipt_ref', 'worker_error_observation_ref'):
                if name in provenance:
                    provenance[name]['payload_sha256'] = artifact['payload_sha256']
    return seal(example)


def test_full_icp_and_strict_training_need_no_client_ack_or_named_phase():
    example = without_client_callbacks()
    for call in example['provider_calls']:
        training_steps._candidate(example, call)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits['train']) + len(splits['eval']) == 2
    assert all(c['response_provenance']['agent_receipt'] == 'not_reported' for c in example['provider_calls'])


def test_exporter_new_policy_does_not_demand_named_sdk_phases_or_read_reports():
    example = without_client_callbacks()
    snapshot = example['worker_observation_artifacts']['run']
    errors = training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, example['provider_calls'], snapshot,
        require_client_ack=False)
    assert not errors


@pytest.mark.parametrize('legacy', ['absent', 'off', 'invalid_percent', 'invalid_registry'])
def test_old_or_invalid_policy_cannot_disable_client_checks(legacy):
    example = without_client_callbacks()
    if legacy == 'absent':
        example['configuration'].pop('training_maximum_missing_reasoning_percent')
    elif legacy == 'off':
        example['configuration']['reasoning_capture_policy'] = None
    elif legacy == 'invalid_percent':
        example['configuration']['training_maximum_missing_reasoning_percent'] = False
    else:
        example['configuration']['reasoning_route_registry'] = {}
    seal(example)
    with pytest.raises(ValueError):
        training_steps._candidate(example, example['provider_calls'][0])


@pytest.mark.parametrize('damage', ['host_hash', 'host_status', 'host_length', 'host_run', 'host_call',
                                   'request_hash', 'response_hash', 'missing_snapshot', 'failed_write',
                                   'forged_ack_claim', 'forged_ack_ref'])
def test_no_client_requirement_preserves_host_and_gateway_integrity(damage):
    example = without_client_callbacks()
    call = example['provider_calls'][0]
    provenance = call['response_provenance']
    if damage == 'host_hash': provenance['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_status': example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['status'] = 201
    elif damage == 'host_length': example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['body_bytes'] += 1
    elif damage == 'host_run': provenance['worker_receipt_ref']['run_id'] = 'wrong'
    elif damage == 'host_call': call['call_identity'] = 'wrong'
    elif damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response_hash': call['response']['tamper'] = True
    elif damage == 'missing_snapshot': example['worker_observation_artifacts'] = {}
    elif damage == 'failed_write': provenance['delivery_to_worker'] = 'write_failed'
    elif damage == 'forged_ack_claim': provenance['agent_receipt'] = 'reported_exact_body_received'
    else: provenance['agent_receipt_refs'] = [{'run_id': 'run', 'id': 'made-up', 'payload_sha256': provenance['worker_receipt_ref']['payload_sha256']}]
    seal(example)
    with pytest.raises(ValueError):
        training_steps._candidate(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


@pytest.mark.parametrize('new_policy', [False, True])
def test_actual_export_assembly_derives_callback_policy_from_frozen_configuration(new_policy):
    import base64
    from lab_arena import training_observations
    from tests.lab_arena.training_export_test import fixture, add_call
    from tests.lab_arena.training_observations_test import save
    values = fixture()
    add_call(values)
    config = values['round_row']['configuration_doc']
    config.update(reasoning_capture_policy=reasoning_capture.POLICY,
                  reasoning_route_registry=captured_example()['configuration']['reasoning_route_registry'])
    if new_policy:
        config['training_maximum_missing_reasoning_percent'] = 0
    terminal = values['ledger_rows'][-1]
    raw = base64.b64decode(terminal['terminal_response']['body_b64'])
    snapshot = {'observations': [], 'observation_count': 0, 'local_refusal_count': 0,
                'complete': True, 'failures': [], 'response_receipt_count': 1,
                'response_receipts': [{'ordinal': 0, 'action_sequence': 0,
                    'call_identity': terminal['call_identity'], 'operation_id': terminal['operation_id'],
                    'protocol': 'frame', 'status': 200, 'body_sha256': contracts.hash_bytes(raw),
                    'body_bytes': len(raw), 'write_state': 'socket_written',
                    'socket_written_at': '2026-10-25T00:00:00Z'}],
                'agent_observations': [], 'agent_observation_count': 0}
    save(values, snapshot)
    result = training_export.assemble_example(**values)
    reasons = set(result['completeness']['incomplete_reasons'])
    if new_policy:
        assert result['completeness']['complete'], reasons
    else:
        assert reasons == {'agent_response_receipts_unverified', 'agent_task_state_unverified'}


def test_missing_host_snapshot_remains_a_gap_in_new_policy():
    from tests.lab_arena.training_export_test import fixture, add_call
    values = fixture()
    add_call(values)
    values['round_row']['configuration_doc'].update(
        reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=captured_example()['configuration']['reasoning_route_registry'],
        training_maximum_missing_reasoning_percent=0)
    result = training_export.assemble_example(**values)
    assert not result['completeness']['complete']
    assert 'worker_observations_unavailable' in result['completeness']['incomplete_reasons']


@pytest.mark.parametrize('error_kind', ['provider_error', 'budget_refusal'])
def test_exact_failed_attempt_needs_no_client_ack(error_kind):
    if error_kind == 'provider_error':
        from tests.lab_arena.training_call_semantics_test import _failed_example
        source = _failed_example(502, worker_error=True)
    else:
        from tests.lab_arena.training_admission_refusal_test import refusal_example
        source = refusal_example()[0]
    example = without_client_callbacks(source)
    assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    assert training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


def test_later_recovery_never_replaces_the_actual_host_written_target():
    example = without_client_callbacks()
    call = example['provider_calls'][1]
    original = deepcopy(call['response'])
    receipt = call['response_receipts'][0]
    receipt['response'] = original
    receipt.pop('response_ref', None)
    later = deepcopy(original)
    later['choices'][0]['message']['content'] = 'Later billing recovery was never delivered.'
    call.update(terminal_entry_id=999, response=later, response_sha256=contracts.document_hash(later))
    call['response_receipts'].append({'entry_id': 999, 'metadata': {'status': 200},
        'response_sha256': call['response_sha256'], 'response_ref': {'field': 'response', 'entry_id': 999},
        'wire_body_sha256': contracts.document_hash(later), 'wire_body_bytes': 10})
    seal(example)
    resolved = training_steps._host_result(example, call)
    assert resolved['response'] == original
    candidate = training_steps._candidate(example, call)
    assert candidate['source_refs'][0]['delivered_response_sha256'] == contracts.document_hash(original)
    assert 'Later billing recovery' not in str(candidate['target'])


def local_runtime_diagnostics(*, redacted=True):
    from tests.lab_arena.training_eligibility_test import judge_only_gap_example
    source = judge_only_gap_example()
    source['provider_calls'] = [call for call in source['provider_calls'] if call['run_id'] == source['run_id']]
    example = without_client_callbacks(source)
    reasons = {'direct_web_content_not_captured'}
    artifact = example['worker_observation_artifacts']['run']
    if redacted:
        artifact['payload']['agent_observations'] = [{'id': 'optional-state', 'kind': 'state_snapshot',
            'after_action_sequence': 0, 'depends_on': [],
            'privacy': {'redacted': True, 'redactions': ['credential']},
            'data': {'name': 'optional-debug', 'value': '[REDACTED]'}}]
        reasons.add('worker_observation_privacy_nonexact')
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in example['provider_calls']:
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    dimensions = {key: [] for key in ('context', 'source_identity', 'exact_inputs', 'input_output_label', 'costs', 'research_process')}
    dimensions['research_process'] = sorted(reasons)
    example['completeness'] = {'complete': False, 'incomplete_reasons': sorted(reasons),
        'dimensions': {key: not value for key, value in dimensions.items()}, 'reasons_by_dimension': dimensions}
    return seal(example)


@pytest.mark.parametrize('web_gap', ['direct_web_usage_unverified', 'direct_web_content_not_captured', 'direct_web_coverage_unverified'])
def test_combined_no_callbacks_local_web_and_redacted_sdk_state_preserves_provider_pair(web_gap):
    example = local_runtime_diagnostics()
    proof = example['completeness']
    proof['incomplete_reasons'].remove('direct_web_content_not_captured')
    proof['incomplete_reasons'].append(web_gap)
    proof['reasons_by_dimension']['research_process'] = sorted(proof['incomplete_reasons'])
    seal(example)
    before = deepcopy(example)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(record['prior_observations'] == [] for record in records)
    assert all(record['source_refs'][0]['capture_scope'] == 'provider_visible_request_result' for record in records)
    assert example == before and example['completeness']['complete'] is False


@pytest.mark.parametrize('damage', ['host_row_redacted', 'provider_request_redacted', 'host_hash',
                                   'response_hash', 'output_hash', 'judge_input', 'optional_ack', 'wrong_dimension'])
def test_local_diagnostics_do_not_waive_provider_or_scored_evidence(damage):
    example = local_runtime_diagnostics()
    call = example['provider_calls'][0]
    if damage == 'host_row_redacted':
        example['worker_observation_artifacts']['run']['payload']['observations'] = [{'privacy': {'redacted': True}}]
    elif damage == 'provider_request_redacted': call['request_capture']['redacted'] = True
    elif damage == 'host_hash': call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'response_hash': call['response']['changed'] = True
    elif damage == 'output_hash': example['output']['changed'] = True
    elif damage == 'judge_input': example['input_snapshots']['score']['input']['companies'] = [{'forged': True}]
    elif damage == 'optional_ack':
        example['worker_observation_artifacts']['run']['payload']['agent_observations'][0]['kind'] = 'response_received'
    else:
        example['completeness']['reasons_by_dimension']['source_identity'] = ['direct_web_content_not_captured']
        example['completeness']['dimensions']['source_identity'] = False
    seal(example)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


def test_readiness_and_training_keep_local_diagnostics_distinct_from_judge_gaps():
    from lab_arena import training_readiness
    example = local_runtime_diagnostics()
    report = training_readiness.build([example], production_rounds_only=False)
    capture = report['sources'][0]['capture']
    assert capture['scope'] == 'provider_visible_request_results'
    assert capture['ancillary_judge_gaps'] == []
    assert set(capture['diagnostic_capture_gaps']) == set(example['completeness']['incomplete_reasons'])
    assert capture['status'] == 'incomplete'
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    for record in splits['train'] + splits['eval']:
        evidence = record['source_refs'][0]['selected_research_evidence']
        assert evidence['ancillary_judge_gaps'] == []
        assert set(evidence['diagnostic_capture_gaps']) == set(capture['diagnostic_capture_gaps'])


@pytest.mark.parametrize('gap', ['cost', 'source', 'source_cost'])
@pytest.mark.parametrize('tamper', [False, True])
def test_local_diagnostics_compose_with_bounded_source_and_cost_availability(gap, tamper):
    from tests.lab_arena.training_eligibility_test import _unsettled_selected_call, ten_selected_examples
    example = local_runtime_diagnostics()
    local = set(example['completeness']['incomplete_reasons'])
    if gap in {'cost', 'source_cost'}:
        _unsettled_selected_call(example)
        proof = example['completeness']
        proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | local)
        proof['reasons_by_dimension']['research_process'] = sorted(set(proof['reasons_by_dimension']['research_process']) | local)
    else:
        proof = example['completeness']
    artifacts = {}
    if gap in {'source', 'source_cost'}:
        example['source'] = None
        source_ref = 'arena/%s/sources/%s.tar.gz' % (example['round_id'], example['submission_id'])
        artifacts['source'] = source_ref
        proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | {'source_unverified'})
        proof['dimensions']['source_identity'] = False
        proof['reasons_by_dimension']['source_identity'] = ['source_unverified']
    if tamper:
        example['provider_calls'][1]['response']['changed'] = True
    seal(example)
    examples = ten_selected_examples()
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 0
        seal(other)
    examples.pop('run-9')
    examples['run'] = example
    result = training_eligibility.assess_capture(submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, unavailable_artifacts={'run': artifacts} if artifacts else None)
    assert result['eligible'], result
    if not tamper:
        assert result['counts']['complete_runs'] == (10 if gap == 'cost' and not tamper else 9)
        assert result['counts']['incomplete_runs'] == (0 if gap == 'cost' and not tamper else 1)
