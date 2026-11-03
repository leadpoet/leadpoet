"""Call admission follows exact reasoning evidence, not a coverage display label."""

import pytest

from lab_arena import training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    capture, captured_example, refresh_response_links, submission, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal


@pytest.mark.parametrize('extra', ['summary', 'hidden_removed'])
def test_mixed_truncation_and_proven_full_text_remains_a_coverage_gap(extra):
    examples = ten_selected_examples()
    example = examples['run-9']
    example['provider_calls'][1]['response']['choices'][0]['finish_reason'] = 'length'
    refresh_response_links(example, 1)
    call = example['provider_calls'][2]
    if extra == 'summary':
        call['response']['choices'][0]['message']['reasoning_details'].append(
            {'type': 'reasoning.summary', 'summary': 'Ancillary summary'})
    else:
        call['response_privacy_redactions'] = ['hidden_reasoning']
        call['response_receipts'][0]['privacy_redactions'] = ['hidden_reasoning']
    refresh_response_links(example, 2)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
    )
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 9


@pytest.mark.parametrize('tokens,accepted', [(0, True), (12, False), (None, False)])
@pytest.mark.parametrize('target', ['tool', 'note_tool', 'answer'])
def test_only_explicit_zero_usage_proves_a_completed_turn_did_not_think(tokens, accepted, target):
    example = captured_example()
    call = example['provider_calls'][0]
    message = call['response']['choices'][0]['message']
    message['content'] = 'I will use the tool.' if target == 'note_tool' else None
    if target == 'answer':
        message.pop('tool_calls')
        message['content'] = 'A complete direct answer.'
        call['response']['choices'][0]['finish_reason'] = 'stop'
    usage = call['response']['usage']
    usage.pop('completion_tokens_details', None)
    if tokens is not None:
        usage['completion_tokens_details'] = {'reasoning_tokens': tokens}
    refresh_response_links(example, 0)
    registry = example['configuration']['reasoning_route_registry']
    status = training_thinking.attempt_coverage(call, 'miner', registry)
    assert status['tokens_reported'] == tokens
    assert training_thinking.verified_no_thinking(call, call['response'], status) is accepted
    # Historical tool-only admission is preserved without labeling usage zero.
    compatible = tokens is None and target == 'tool'
    assert capture(example)['eligible'] is (accepted or compatible)


@pytest.mark.parametrize('kind', ['summary', 'unproved', 'opaque'])
def test_zero_usage_does_not_turn_other_reasoning_evidence_into_no_thinking(kind):
    example = captured_example()
    call = example['provider_calls'][0]
    message = call['response']['choices'][0]['message']
    if kind == 'summary':
        message['reasoning_details'] = [{'type': 'reasoning.summary', 'summary': 'Only summary'}]
    elif kind == 'unproved':
        message['reasoning_content'] = 'Not a route-approved field'
    else:
        call['response_privacy_redactions'] = ['hidden_reasoning']
        call['response_receipts'][0]['privacy_redactions'] = ['hidden_reasoning']
    refresh_response_links(example, 0)
    assert capture(example)['eligible'] is False


def _failed_example(status, *, worker_error=False):
    import base64
    from lab_arena import contracts
    example = captured_example()
    # Failed-call tolerance belongs to the newly frozen policy, including 0.
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][0]
    call['response'] = {'error': {'message': 'Provider unavailable'}}
    call['response_status'] = status
    call['response_receipts'][0]['metadata'] = {'status': status}
    refresh_response_links(example, 0)
    payload = example['worker_observation_artifacts']['run']['payload']
    for row in payload['response_receipts'] + payload['agent_observations']:
        data = row.get('data', row)
        if data.get('call_identity') == call['call_identity']:
            data['status'] = status
    if worker_error:
        wire = contracts.canonical_json(call['response']).encode()
        payload['observations'] = [{
            'ordinal': 1, 'call_identity': call['call_identity'],
            'operation_id': call['operation_id'], 'action_sequence': call['action_sequence'],
            'privacy': {'redacted': False},
            'request': {'admitted_parameters': call['request']['admitted_parameters']},
            'outcome': {'response': {'status': status, 'body_b64': base64.b64encode(wire).decode()}},
        }]
        provenance = call['response_provenance']
        provenance['worker_error_body_acknowledged'] = True
        provenance['worker_error_observation_ref'] = {
            'run_id': 'run', 'ordinal': 1,
            'payload_sha256': '',
        }
        call['response_receipts'] = []
    # Refresh all references after adding the status or worker error bytes.
    payload_hash = contracts.document_hash(payload)
    example['worker_observation_artifacts']['run']['payload_sha256'] = payload_hash
    for item in example['provider_calls']:
        provenance = item['response_provenance']
        for ref in provenance['agent_receipt_refs'] + [provenance['worker_receipt_ref']]:
            ref['payload_sha256'] = payload_hash
        if 'worker_error_observation_ref' in provenance:
            provenance['worker_error_observation_ref']['payload_sha256'] = payload_hash
    return seal(example)


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
@pytest.mark.parametrize('worker_error', [False, True])
def test_exact_failed_attempt_preserves_successful_training_and_entrant(status, worker_error):
    from lab_arena import training_steps
    example = _failed_example(status, worker_error=worker_error)
    assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    assert capture(example)['eligible'] is True
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(record['source_refs'][0]['call_identity'] != example['provider_calls'][0]['call_identity']
               for record in records)
    assert example['provider_calls'][0]['response']['error']


@pytest.mark.parametrize('damage', ['request', 'ack', 'wire', 'route', 'generation'])
def test_failed_attempt_cannot_mask_invalid_proof_or_generated_output(damage):
    from lab_arena import contracts
    example = _failed_example(502)
    call = example['provider_calls'][0]
    if damage == 'request':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'ack':
        call['response_provenance']['agent_receipt'] = 'not_reported'
    elif damage == 'wire':
        call['response_receipts'][0]['wire_body_bytes'] += 1
    elif damage == 'route':
        call['reasoning_capture']['provider_route'] = 'wrong-route'
    else:
        call['response']['choices'] = [{'message': {'role': 'assistant', 'content': 'A generation'}}]
        refresh_response_links(example, 0)
    seal(example)
    assert training_thinking.verified_failed_attempt(example, call) is False
    assert capture(example)['eligible'] is False


@pytest.mark.parametrize('missing,total,accepted', [(1, 100, True), (2, 100, False), (1, 101, True)])
@pytest.mark.parametrize('kind', ['missing', 'summary', 'opaque'])
def test_missing_reasoning_budget_uses_reported_tokens_on_verified_completed_calls(missing, total, accepted, kind):
    example = captured_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    for index, tokens in enumerate((missing, 40, total - missing - 40)):
        call = example['provider_calls'][index]
        call['response']['usage']['completion_tokens_details']['reasoning_tokens'] = tokens
        if index == 0 and kind == 'summary':
            call['response']['choices'][0]['message']['reasoning_details'] = [
                {'type': 'reasoning.summary', 'summary': 'Summary with no full text'}]
        if index == 0 and kind == 'opaque':
            call['response_privacy_redactions'] = ['hidden_reasoning']
            call['response_receipts'][0]['privacy_redactions'] = ['hidden_reasoning']
        refresh_response_links(example, index)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1,
    )
    assert result['eligible'] is accepted, result
    assert result['counts']['missing_reasoning_tokens'] == missing
    assert result['counts']['completed_untruncated_verified_reasoning_tokens'] == total
    assert result['counts']['complete_runs'] == int(accepted)
    assert result['reasons'] == ([] if accepted else ['training_coverage_incomplete'])


def test_unknown_usage_and_truncation_do_not_enter_token_denominator():
    example = captured_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    example['provider_calls'][0]['response']['usage'].pop('completion_tokens_details')
    example['provider_calls'][1]['response']['choices'][0]['finish_reason'] = 'length'
    example['provider_calls'][1]['response']['usage']['completion_tokens_details']['reasoning_tokens'] = 99999
    for index in (0, 1):
        refresh_response_links(example, index)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1,
    )
    assert result['counts']['completed_untruncated_verified_reasoning_tokens'] == 13
    assert result['counts']['unknown_reasoning_usage_calls'] == 1
    assert result['counts']['truncated_model_calls'] == 1
    assert result['counts']['incomplete_runs'] == 1


def test_conflicting_zero_usage_cannot_conceal_positive_billed_reasoning():
    example = captured_example()
    call = example['provider_calls'][0]
    call['response']['usage']['reasoning_tokens'] = 0
    call['response']['usage']['completion_tokens_details']['reasoning_tokens'] = 10
    refresh_response_links(example, 0)
    status = training_thinking.attempt_coverage(call, 'miner', example['configuration']['reasoning_route_registry'])
    assert status['tokens_reported'] == 10
    assert status['returned'] == 'missing_thinking'
    assert capture(example)['eligible'] is False


def test_per_icp_reasoning_share_and_truncation_preserve_eight_of_ten_and_strict_boundary():
    from lab_arena import training_steps
    examples = ten_selected_examples()
    for run_id, example in examples.items():
        example['configuration']['training_maximum_missing_reasoning_percent'] = 1
        for index, tokens in enumerate((0, 40, 59)):
            example['provider_calls'][index]['response']['usage']['completion_tokens_details']['reasoning_tokens'] = tokens
            refresh_response_links(example, index)
    for run_id, missing in [('run-7', 1), ('run-8', 2)]:
        example = examples[run_id]
        example['provider_calls'][0]['response']['usage']['completion_tokens_details']['reasoning_tokens'] = missing
        refresh_response_links(example, 0)
    for index, call in enumerate(examples['run-9']['provider_calls']):
        call['response']['choices'][0]['finish_reason'] = 'length'
        refresh_response_links(examples['run-9'], index)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1,
    )
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 8
    assert result['counts']['incomplete_runs'] == 2
    per_run = {row['run_id']: row for row in result['counts']['reasoning_by_run']}
    assert per_run['run-7']['missing_reasoning_tokens'] == 1
    assert per_run['run-8']['missing_reasoning_tokens'] == 2
    assert per_run['run-9']['truncated_model_calls'] == 3
    assert per_run['run-9']['completed_untruncated_verified_reasoning_tokens'] == 0
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    strict_runs = {ref['run_id'] for split in ('train', 'eval') for record in strict[split]
                   for ref in record['source_refs']}
    assert strict_runs == set(examples) - {'run-9'}


def test_one_truncated_call_does_not_make_new_policy_icp_incomplete():
    from lab_arena import training_steps
    example = captured_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    example['provider_calls'][1]['response']['choices'][0]['finish_reason'] = 'length'
    refresh_response_links(example, 1)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1,
    )
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 1
    assert result['counts']['truncated_model_calls'] == 1
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == 1


@pytest.mark.parametrize('operation,provider,accepted', [
    ('openrouter.responses', None, True),
    ('openrouter.responses', 'wrong-provider', False),
    ('openrouter.chat', None, False),
])
def test_missing_top_provider_is_only_optional_for_responses_with_exact_selected_endpoint(operation, provider, accepted):
    example = captured_example()
    call = example['provider_calls'][0]
    annotation = {**call['reasoning_capture'], 'api': operation}
    observed = dict(call['response_receipts'][0]['metadata']['reasoning_route_observation'])
    observed['response_provider_name'] = provider
    assert training_thinking._observation_matches(observed, annotation) is accepted
    observed['selected_endpoint_provider_name'] = 'wrong-endpoint'
    assert training_thinking._observation_matches(observed, annotation) is False


@pytest.mark.parametrize('kind', ['source', 'benchmark', 'input', 'judge_input', 'judge_output'])
def test_proven_transient_artifact_gap_consumes_one_icp_and_keeps_retained_integrity_hard(kind):
    from copy import deepcopy
    from lab_arena import contracts
    from tests.lab_arena.training_eligibility_test import judge_only_gap_example
    examples = ten_selected_examples()
    examples.pop('run-9')
    example = judge_only_gap_example()
    example['provider_calls'] = example['provider_calls'][:3]
    example['attempts'] = [
        {'run_id': 'run', 'kind': 'execute', 'status': 'accepted', 'lease_generation': 1},
        {'run_id': 'score', 'kind': 'score', 'status': 'accepted', 'lease_generation': 1},
    ]
    dimensions = {key: set() for key in (
        'context', 'source_identity', 'exact_inputs', 'input_output_label', 'costs', 'research_process')}
    if kind == 'source':
        example.pop('source')
        ref = f"arena/{example['round_id']}/sources/{example['submission_id']}.tar.gz"
        dimensions['source_identity'].add('source_unverified')
    elif kind == 'benchmark':
        example.pop('icp', None)
        ref = 'saved/benchmark.json'
        dimensions['context'].add('frozen_icp_unavailable')
        dimensions['input_output_label'].add('frozen_context_unverified')
    elif kind in {'input', 'judge_input'}:
        source_id = 'run' if kind == 'input' else 'score'
        example['input_snapshots'].pop(source_id)
        ref = f'private/arena-training/runs/{source_id}/leases/1/input.json.gz'
        dimensions['exact_inputs'].add('runtime_input_snapshot_missing')
        dimensions['research_process'].add('runtime_input_snapshot_missing')
    else:
        example.pop('judgment')
        example.pop('score_run_id')
        ref = f"arena/{example['round_id']}/outputs/score.json"
        dimensions['input_output_label'].add('judge_evidence_unverified')
        dimensions['research_process'].add('judge_evidence_unverified')
    example['completeness'] = {
        'complete': False, 'incomplete_reasons': sorted(set().union(*dimensions.values())),
        'dimensions': {key: not value for key, value in dimensions.items()},
        'reasons_by_dimension': {key: sorted(value) for key, value in dimensions.items()},
    }
    examples['run'] = seal(example)
    def assess():
        return training_eligibility.assess_capture(
            submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
            unavailable_artifacts={'run': {kind: ref}},
        )
    result = assess()
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    original = deepcopy(example)
    for damage in ('request', 'receipt', 'route'):
        bad = deepcopy(original)
        call = bad['provider_calls'][2]
        if damage == 'request':
            call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'bad')
        elif damage == 'receipt':
            call['response_receipts'][0]['response_sha256'] = contracts.hash_bytes(b'bad')
        else:
            call['reasoning_capture']['provider_route'] = 'wrong-route'
        examples['run'] = seal(bad)
        assert assess()['eligible'] is True
        assert assess()['counts']['complete_runs'] == 9


def test_honest_failed_attempt_with_unsettled_cost_consumes_only_one_icp():
    from tests.lab_arena.training_eligibility_test import _unsettled_selected_call
    examples = ten_selected_examples()
    example = _failed_example(502)
    # The cost-gap fixture supplies exact score/output/input binding and
    # marks the existing selected call's still-unknown bill explicitly.
    example = _unsettled_selected_call(example)
    examples.pop('run-9')
    examples['run'] = example
    assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
    )
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 10


def _unsettled_worker_failure(status=502, *, retained_response=False):
    from tests.lab_arena.training_eligibility_test import _unsettled_selected_call
    example = _unsettled_selected_call(_failed_example(status, worker_error=True))
    call = example['provider_calls'][0]
    call['response_provenance']['worker_error_observation_ref']['payload_sha256'] = (
        example['worker_observation_artifacts']['run']['payload_sha256'])
    if not retained_response:
        call.pop('response')
        call.pop('response_sha256')
    return seal(example)


@pytest.mark.parametrize('status', [500, 502])
@pytest.mark.parametrize('retained_response', [False, True])
def test_exact_host_failure_with_unknown_bill_consumes_one_icp(status, retained_response):
    from copy import deepcopy
    from lab_arena import training_steps
    example = _unsettled_worker_failure(status, retained_response=retained_response)
    original = deepcopy(example)
    call = example['provider_calls'][0]
    assert training_thinking.verified_failed_attempt(example, call)
    examples = ten_selected_examples()
    examples.pop('run-9')
    examples['run'] = example
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 10
    assert result['counts']['incomplete_runs'] == 0
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == 2
    assert call['accounting_state'] == 'uncertain' and call['actual_microusd'] is None
    assert example == original
    # Combine independent availability failures at the unchanged floor.
    examples['run-8'] = None
    assert training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80)['eligible']
    examples['run-7'] = None
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80)
    assert result['eligible'] and result['counts']['complete_runs'] == 8
    examples['run-6'] = None
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80)
    assert not result['eligible'] and result['counts']['complete_runs'] == 7
    assert 'training_coverage_incomplete' in result['reasons']


@pytest.mark.parametrize('damage', ['request_hash', 'outbound_hash', 'host_hash', 'host_write',
    'host_body_missing', 'wrong_route', 'response_hash', 'response_changed', 'response_redaction_unreported'])
def test_unknown_bill_cannot_excuse_invalid_or_missing_host_failure(damage):
    from lab_arena import contracts
    example = _unsettled_worker_failure(retained_response=True)
    call = example['provider_calls'][0]
    payload = example['worker_observation_artifacts']['run']['payload']
    if damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_hash': call['response_provenance']['worker_error_observation_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_write': payload['response_receipts'][0]['write_state'] = 'write_failed'
    elif damage == 'host_body_missing': payload['observations'] = []
    elif damage == 'wrong_route': call['reasoning_capture']['provider_route'] = 'unapproved'
    elif damage == 'response_hash': call['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response_redaction_unreported': call['response_privacy_redactions'] = ['credential']
    else:
        call['response'] = {'choices': [{'message': {'content': 'Contradictory generated output'}}]}
        call['response_sha256'] = contracts.document_hash(call['response'])
    if damage in {'host_write', 'host_body_missing'}:
        # Keep the object/reference hashes valid so these controls exercise
        # absent delivery/body proof independently of hash-tamper rejection.
        digest = contracts.document_hash(payload)
        example['worker_observation_artifacts']['run']['payload_sha256'] = digest
        for item in example['provider_calls']:
            provenance = item['response_provenance']
            refs = provenance.get('agent_receipt_refs', []) + [
                provenance[key] for key in ('worker_receipt_ref', 'worker_error_observation_ref')
                if key in provenance]
            for reference in refs:
                reference['payload_sha256'] = digest
    seal(example)
    examples = ten_selected_examples()
    examples.pop('run-9')
    examples['run'] = example
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=100)
    if damage in {'response_hash', 'response_changed', 'response_redaction_unreported'}:
        # The authoritative host failure bytes are exact; no failed response
        # projection is admitted as a generated training target.
        assert result['eligible']
        strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
        assert all(ref['call_identity'] != call['call_identity']
                   for split in ('train', 'eval') for record in strict[split] for ref in record['source_refs'])
    else:
        assert not result['eligible']


def test_live_parameter_cannot_override_frozen_zero_tolerance():
    example = captured_example()
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1,
    )
    assert result['eligible'] is False
    assert 'selected_run_identity_unverified' in result['counts']['rejected_runs'][0]['reasons']


def test_output_projection_keeps_raw_hash_and_proves_normalized_hash_without_legacy_guessing():
    from lab_arena import contracts
    raw = {'companies': [{'company_name': '  Acme  '}]}
    normalized = {'companies': [{'company_name': 'Acme'}]}
    example = {'output': normalized, 'output_identity': {
        'document_hash': contracts.document_hash(raw),
        'normalized_document_hash': contracts.document_hash(normalized),
    }}
    assert training_thinking.verified_output_projection(example)
    for field in ('document_hash', 'normalized_document_hash'):
        prior = example['output_identity'][field]
        example['output_identity'][field] = 'invalid'
        assert not training_thinking.verified_output_projection(example)
        example['output_identity'][field] = prior
    example['output_identity'].pop('normalized_document_hash')
    assert not training_thinking.verified_output_projection(example)
    example['output'] = raw
    assert training_thinking.verified_output_projection(example)


@pytest.mark.parametrize('frozen_setting', ['absent', 0])
@pytest.mark.parametrize('case', ['truncated', 'unknown_tool', 'missing_positive'])
def test_explicit_zero_limit_keeps_new_semantics_distinct_from_absent_legacy(frozen_setting, case):
    from lab_arena import training_steps
    example = captured_example()
    if frozen_setting != 'absent':
        example['configuration']['training_maximum_missing_reasoning_percent'] = frozen_setting
    if case == 'truncated':
        example['provider_calls'][1]['response']['choices'][0]['finish_reason'] = 'length'
        index = 1
    else:
        index = 0
        usage = example['provider_calls'][0]['response']['usage']
        if case == 'unknown_tool':
            usage.pop('completion_tokens_details')
        else:
            usage['completion_tokens_details']['reasoning_tokens'] = 1
    refresh_response_links(example, index)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=0,
    )
    expected = (case == 'truncated' if frozen_setting == 0 else case == 'unknown_tool')
    assert result['eligible'] is expected, result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert strict['train'] or strict['eval']  # Other exact full targets remain exportable.


def _source_timeout_and_cost_example(*, privacy=False, truncated=False):
    from tests.lab_arena.training_eligibility_test import _unsettled_selected_call
    example = _unsettled_selected_call(captured_example(), privacy=privacy)
    example.pop('source')
    proof = example['completeness']
    proof['dimensions']['source_identity'] = False
    proof['reasons_by_dimension']['source_identity'] = ['source_unverified']
    proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | {'source_unverified'})
    if truncated:
        example['provider_calls'][2]['response']['choices'][0]['finish_reason'] = 'length'
        refresh_response_links(example, 2)
    return seal(example)


@pytest.mark.parametrize('privacy', [False, True])
@pytest.mark.parametrize('truncated', [False, True])
def test_source_timeout_and_verified_cost_gap_consume_one_icp(privacy, truncated):
    from lab_arena import training_steps
    example = _source_timeout_and_cost_example(privacy=privacy, truncated=truncated)
    examples = ten_selected_examples()
    examples.pop('run-9')
    examples['run'] = example
    source_ref = f"arena/{example['round_id']}/sources/{example['submission_id']}.tar.gz"
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
        unavailable_artifacts={'run': {'source': source_ref}},
    )
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    assert all(ref['run_id'] != 'run' for split in ('train', 'eval') for record in strict[split]
               for ref in record['source_refs'])


@pytest.mark.parametrize('damage', ['source_ref', 'request', 'receipt', 'route', 'cost', 'input'])
def test_source_timeout_plus_cost_does_not_mask_retained_evidence_damage(damage):
    from lab_arena import contracts
    example = _source_timeout_and_cost_example()
    examples = ten_selected_examples()
    examples.pop('run-9')
    source_ref = f"arena/{example['round_id']}/sources/{example['submission_id']}.tar.gz"
    call = example['provider_calls'][0]
    if damage == 'source_ref':
        source_ref += '.wrong'
    elif damage == 'request':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'receipt':
        call['response_receipts'][0]['response_sha256'] = contracts.hash_bytes(b'bad')
    elif damage == 'route':
        call['reasoning_capture']['provider_route'] = 'wrong-route'
    elif damage == 'cost':
        call['actual_microusd'] = 7
    else:
        example['input_snapshots']['run']['input_sha256'] = contracts.hash_bytes(b'bad')
    examples['run'] = seal(example)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=100,
        unavailable_artifacts={'run': {'source': source_ref}},
    )
    assert result['eligible'] is False


@pytest.mark.parametrize('frozen_setting', ['absent', 0, 1])
@pytest.mark.parametrize('status', [429, 500])
def test_frozen_error_policy_presence_controls_capture_build_and_coverage(frozen_setting, status):
    from lab_arena import training_steps
    example = _failed_example(status)
    example['configuration'].pop('training_maximum_missing_reasoning_percent', None)
    if frozen_setting != 'absent':
        example['configuration']['training_maximum_missing_reasoning_percent'] = frozen_setting
    seal(example)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80,
        maximum_missing_reasoning_percent=0 if frozen_setting == 'absent' else frozen_setting,
    )
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    report = training_thinking.coverage([example])
    permitted = frozen_setting != 'absent'
    assert result['eligible'] is permitted, result
    assert result['counts']['complete_runs'] == int(permitted)
    assert len(strict['train']) + len(strict['eval']) == 2
    assert report['trajectories']['complete_full_thinking'] == int(permitted)
    assert report['trajectories']['ineligible'] == int(not permitted)
