"""A proved pre-generation credit failure does not erase earlier thinking."""
import base64
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_call_semantics_test import _failed_example
from tests.lab_arena.training_eligibility_test import _unsettled_selected_call, refresh_response_links, submission
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import seal

BROKER_ERROR = {'error': {'code': 'miner_credentials_unavailable'}}
CREDIT_ERROR = {'error': {'code': 402,
    'message': 'This request requires more credits, or fewer max_tokens.',
    'metadata': {'provider_name': None}}}


def upstream(value, entry_id=3):
    digest = contracts.document_hash(value)
    size = len(contracts.canonical_json(value).encode())
    return {'entry_id': entry_id, 'status': 402, 'response': deepcopy(value),
        'response_sha256': digest, 'body_sha256': digest, 'wire_body_sha256': digest,
        'body_bytes': size, 'wire_body_bytes': size, 'privacy_redactions': [],
        'provenance': 'private_upstream_response_receipt', 'delivery_to_worker': 'not_proved'}


def credit_example(*, worker_error=True, client_ack=False, settled=False):
    example = _failed_example(402)
    call = example['provider_calls'][0]
    call.update(response=deepcopy(BROKER_ERROR), funding_source='miner_key',
                accounting_state='settlement' if settled else 'uncertain',
                actual_microusd=7 if settled else None,
                evidence_gaps=[] if settled else ['provider_cost_unresolved'])
    refresh_response_links(example, 0)
    if not settled:
        _unsettled_selected_call(example)
    call['upstream_response_receipts'] = [upstream(CREDIT_ERROR, call['terminal_entry_id'])]
    if worker_error:
        payload = example['worker_observation_artifacts']['run']['payload']
        raw = contracts.canonical_json(BROKER_ERROR).encode()
        payload['observations'] = [{'ordinal': 1, 'call_identity': call['call_identity'],
            'operation_id': call['operation_id'], 'action_sequence': call['action_sequence'],
            'privacy': {'redacted': False},
            'request': {'admitted_parameters': call['request']['admitted_parameters']},
            'outcome': {'response': {'status': 402, 'body_b64': base64.b64encode(raw).decode()}}}]
        call['response_receipts'] = []
        call['response_provenance'].update(worker_error_body_acknowledged=True,
            worker_error_observation_ref={'run_id': 'run', 'ordinal': 1, 'payload_sha256': ''})
        digest = contracts.document_hash(payload)
        example['worker_observation_artifacts']['run']['payload_sha256'] = digest
        for item in example['provider_calls']:
            prov = item['response_provenance']
            for ref in prov['agent_receipt_refs'] + [prov['worker_receipt_ref']]:
                ref['payload_sha256'] = digest
        call['response_provenance']['worker_error_observation_ref']['payload_sha256'] = digest
    if not client_ack:
        example = without_client_callbacks(example)
    return seal(example)


@pytest.mark.parametrize('worker_error', [False, True])
@pytest.mark.parametrize('client_ack', [False, True])
@pytest.mark.parametrize('settled', [False, True])
@pytest.mark.parametrize('funding', ['miner_key', 'host'])
def test_exact_credit_refusal_preserves_prior_targets_and_accounting(worker_error, client_ack, settled, funding):
    example = credit_example(worker_error=worker_error, client_ack=client_ack, settled=settled)
    call = example['provider_calls'][0]
    call['funding_source'] = funding
    seal(example)
    before = deepcopy(example)
    assert training_steps._host_result(example, call, allow_worker_error=True)['write']['status'] == 402
    assert training_thinking.verified_failed_attempt(example, call)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    assert result['counts']['reasoning_by_run'][0]['verified_failed_model_attempts'] == 1
    with pytest.raises(ValueError):
        training_steps._candidate(example, call)
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    expected = deepcopy(example)
    expected['provider_calls'].pop(0)
    seal(expected)
    prior, _ = training_thinking.build([expected], training_steps.Rules(production_rounds_only=False))
    keys = ('candidate_id', 'input', 'target', 'generated_thinking')
    assert [{k: r[k] for k in keys} for r in records] == [
        {k: r[k] for k in keys} for r in prior['train'] + prior['eval']]
    assert len(records) == 2
    assert example == before


@pytest.mark.parametrize('damage', ['missing_upstream', 'wrong_status', 'empty_message', 'empty',
    'malformed_error', 'wrong_code', 'boolean_code', 'generated_output', 'generated_choices',
    'generated_usage', 'malformed_usage', 'mixed_success', 'private_upstream', 'upstream_hash',
    'wire_hash', 'wire_size', 'text_target', 'nested_target', 'success_status', 'encrypted_target', 'wrong_entry', 'claimed_delivery', 'wrong_provenance',
    'broker_error', 'request_hash', 'request_privacy', 'host_hash', 'host_privacy', 'host_status', 'unauthorized_403'])
def test_unproved_credit_refusals_and_generated_work_remain_hard(damage):
    example = credit_example()
    call = example['provider_calls'][0]
    receipt = call['upstream_response_receipts'][0]
    body = deepcopy(receipt['response'])
    if damage == 'missing_upstream': call.pop('upstream_response_receipts')
    elif damage == 'wrong_status': receipt['status'] = 429
    elif damage == 'empty_message': body['error']['message'] = ' '
    elif damage == 'empty': body = {}
    elif damage == 'malformed_error': body['error'] = 'This request requires more credits'
    elif damage == 'wrong_code': body['error']['code'] = 429
    elif damage == 'boolean_code': body['error']['code'] = True
    elif damage == 'generated_output': body['output'] = [{'type': 'message', 'content': 'generated'}]
    elif damage == 'generated_choices': body['choices'] = [{'message': {'content': 'generated'}}]
    elif damage == 'generated_usage': body['usage'] = {'output_tokens': 1}
    elif damage == 'malformed_usage': body['usage'] = {'output_tokens': '0'}
    elif damage == 'mixed_success': call['upstream_response_receipts'].append(upstream({'output': ['generated']}))
    elif damage == 'private_upstream': receipt['privacy_redactions'] = ['credential']
    elif damage == 'upstream_hash': receipt['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'wire_hash': receipt['wire_body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'wire_size': receipt['wire_body_bytes'] += 1
    elif damage == 'text_target': body['text'] = 'Generated answer'
    elif damage == 'nested_target': body['response'] = {'output': ['generated']}
    elif damage == 'success_status': body['status'] = 'completed'
    elif damage == 'encrypted_target': body['encrypted_content'] = 'opaque'
    elif damage == 'wrong_entry': receipt['entry_id'] += 1
    elif damage == 'claimed_delivery': receipt['delivery_to_worker'] = 'socket_written'
    elif damage == 'wrong_provenance': receipt['provenance'] = 'agent_report'
    elif damage == 'broker_error':
        observation = example['worker_observation_artifacts']['run']['payload']['observations'][0]
        observation['outcome']['response']['body_b64'] = base64.b64encode(b'{}').decode()
    elif damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'request_privacy': call['request_capture']['redacted'] = True
    elif damage == 'host_hash': call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_privacy': example['worker_observation_artifacts']['run']['payload']['observations'][0]['privacy']['redacted'] = True
    elif damage == 'host_status': example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['status'] = 200
    elif damage == 'unauthorized_403': receipt['status'] = 403; body['error']['code'] = 403
    if damage in {'empty_message', 'empty', 'malformed_error', 'wrong_code', 'boolean_code',
                  'generated_output', 'generated_choices', 'generated_usage', 'malformed_usage',
                  'text_target', 'nested_target', 'success_status', 'encrypted_target'}:
        call['upstream_response_receipts'][0] = upstream(body, receipt['entry_id'])
    seal(example)
    before = deepcopy(example)
    assert not training_thinking.verified_failed_attempt(example, call), damage
    assert example == before


def credit_lifecycle_values(*, reconciled=False):
    from lab_arena import broker, training_capture, training_observations
    from tests.lab_arena.training_retry_snapshot_test import retry_values, save_snapshot
    values, predecessor, _ = retry_values()
    values['runs'].remove(predecessor)
    run = values['runs'][0]
    reserve = deepcopy(next(row for row in values['ledger_rows'] if row['entry_kind'] == 'reservation'))
    identity = contracts.hash_bytes(b'synthetic credit refusal')
    sequence = max(row['entry_doc'].get('action_sequence', -1) for row in values['ledger_rows']) + 1
    reserve.update(entry_id=10001, call_identity=identity, funding_source='miner_key')
    reserve['entry_doc'].update(action_sequence=sequence, base_call_identity=identity)
    request = training_capture.read_request(values['objects'], reserve['entry_doc']['request_capture'])
    raw = contracts.canonical_json(BROKER_ERROR).encode()
    terminal = broker._terminal_response_document(402, {}, raw, call_succeeded=False)
    provider = broker._capture_upstream_response(broker.ProviderResponse(
        402, {}, contracts.canonical_json(CREDIT_ERROR).encode()),
        operation_id=reserve['operation_id'], request_hash=reserve['entry_doc']['request_hash'], call_identity=identity)
    context = {key: reserve[key] for key in ('run_id', 'round_id', 'submission_id', 'miner_hotkey',
        'provider', 'operation_id', 'call_identity', 'funding_source')}
    values['ledger_rows'].extend([reserve,
        dict(context, entry_id=10002, entry_kind='dispatch', amount_microusd=0, entry_doc={}),
        dict(context, entry_id=10003, entry_kind='uncertain', amount_microusd=0,
            entry_doc={'reason': 'worker_reported', 'call': {'reason': 'missing_provider_cost',
                'provider_status': 402, 'call_succeeded': False, 'openrouter_generation_present': True,
                'upstream_response_receipt': provider, 'openrouter_terminal_response': terminal}})])
    if reconciled:
        values['ledger_rows'].append(dict(context, entry_id=10004, entry_kind='settlement', amount_microusd=7,
            entry_doc={'openrouter_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 10003},
            terminal_response=broker._terminal_response_document(502, {}, b'{"error":{"code":"provider_unavailable"}}', call_succeeded=False)))
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    ordinal = len(payload['response_receipts'])
    payload['response_receipts'].append({'ordinal': ordinal, 'action_sequence': sequence,
        'operation_id': reserve['operation_id'], 'call_identity': identity, 'protocol': 'http', 'status': 402,
        'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw), 'write_state': 'socket_written',
        'socket_written_at': '2026-10-28T00:00:00Z'})
    payload['observations'].append({'ordinal': ordinal, 'action_sequence': sequence,
        'operation_id': reserve['operation_id'], 'call_identity': identity, 'kind': 'nonstandard_response',
        'request': {'admitted_parameters': request['admitted_parameters'], 'timeout_ms': 1000},
        'outcome': {'protocol': 'http', 'response': {'status': 402, 'headers': {}, 'body_b64': base64.b64encode(raw).decode()}},
        'privacy': {'redacted': False, 'redactions': []}, 'contributes_to_provider_call_count': True})
    payload.update(observation_count=len(payload['observations']), response_receipt_count=len(payload['response_receipts']))
    save_snapshot(values, run, payload)
    run['result_doc']['resource_summary']['provider_call_count'] += 1
    return values, identity


@pytest.mark.parametrize('reconciled', [False, True])
def test_native_lifecycle_and_selective_reader_preserve_error_bills_and_prior_pairs(reconciled):
    from types import SimpleNamespace
    from lab_arena import training_export, training_reader
    from tests.lab_arena.training_undispatched_coverage_test import assess_bank
    values, identity = credit_lifecycle_values(reconciled=reconciled)
    before = deepcopy(values['ledger_rows'])
    direct = training_export.assemble_example(**values)
    def ledger(*, run_id, after_entry_id, limit, training_responses):
        assert training_responses is True
        return [row for row in values['ledger_rows'] if row['run_id'] == run_id
                and (after_entry_id is None or row['entry_id'] > after_entry_id)][:limit]
    store = SimpleNamespace(get_run=lambda identity: next(row for row in values['runs'] if row['run_id'] == identity),
        get_round=lambda _: values['round_row'], get_submission=lambda _: values['submission_row'],
        list_runs=lambda _: values['runs'], list_ledger=ledger,
        icp_cost_eligibility=lambda **identity: values['cost_eligibility_reader'](**identity)['eligibility'])
    selective = training_reader.export_run(store, values['objects'], values['run_id'])
    for example in (direct, selective):
        call = next(c for c in example['provider_calls'] if c['call_identity'] == identity)
        assert training_thinking.verified_failed_attempt(example, call)
        assert call['response_status'] == 402 and call['response'] == BROKER_ERROR
        assert call['accounting_state'] == ('settlement' if reconciled else 'uncertain')
        assert call['actual_microusd'] == (7 if reconciled else None)
        assert ('provider_cost_unresolved' in call['evidence_gaps']) is not reconciled
        assert training_export.response_entry_id(call) == 10003
        assert call['terminal_entry_id'] == (10004 if reconciled else 10003)
        assert training_thinking.selected_research_evidence(example) is not None
        result = assess_bank(values, example)
        assert result['eligible'], result
        splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
        records = splits['train'] + splits['eval']
        assert len(records) == 2
        assert all(r['source_refs'][0]['call_identity'] != identity for r in records)
    keys = ('provider_calls', 'costs', 'worker_observation_artifacts')
    assert {k: direct[k] for k in keys} == {k: selective[k] for k in keys}
    assert values['ledger_rows'] == before


@pytest.mark.parametrize('incomplete,eligible', [(2, True), (3, False)])
def test_credit_failure_does_not_change_eight_of_ten_or_one_percent_policy(incomplete, eligible):
    from tests.lab_arena.training_eligibility_test import ten_selected_examples
    examples = ten_selected_examples()
    examples.pop('run-9')
    examples['run'] = credit_example()
    for example in examples.values():
        example['configuration']['training_maximum_missing_reasoning_percent'] = 1
        seal(example)
    for index in range(incomplete):
        examples['run-' + str(index)] = None
    result = training_eligibility.assess_capture(submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    assert result['eligible'] is eligible, result
    assert result['counts']['complete_runs'] == 10 - incomplete
    assert result['counts']['incomplete_runs'] == incomplete
    failed = examples['run']['provider_calls'][0]
    assert failed['accounting_state'] == 'uncertain' and failed['actual_microusd'] is None
    assert failed['evidence_gaps'] == ['provider_cost_unresolved']


@pytest.mark.parametrize('missing,complete', [(1, True), (2, False)])
def test_credit_error_cannot_raise_the_one_percent_missing_reasoning_budget(missing, complete):
    example = credit_example(worker_error=False, client_ack=True)
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    for index, tokens in ((1, missing), (2, 100 - missing)):
        call = example['provider_calls'][index]
        call['response']['usage']['completion_tokens_details']['reasoning_tokens'] = tokens
        if index == 1:
            for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
                call['response']['choices'][0]['message'].pop(field, None)
        refresh_response_links(example, index)
    seal(example)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    assert result['eligible'] is complete, result
    row = result['counts']['reasoning_by_run'][0]
    assert row['missing_reasoning_tokens'] == missing
    assert row['completed_untruncated_verified_reasoning_tokens'] == 100
    assert row['verified_failed_model_attempts'] == 1


@pytest.mark.parametrize('message', ['Insufficient credits', 'Payment required', 'Credit balance exhausted', '请求需要更多额度'])
@pytest.mark.parametrize('funding', ['miner_key', 'host'])
def test_bound_no_generation_payment_error_does_not_depend_on_wording_or_funding(message, funding):
    example = credit_example()
    call = example['provider_calls'][0]
    call['funding_source'] = funding
    value = deepcopy(CREDIT_ERROR)
    value['error']['message'] = message
    call['upstream_response_receipts'] = [upstream(value, call['terminal_entry_id'])]
    assert training_thinking.verified_failed_attempt(example, call)


@pytest.mark.parametrize('worker_error', [False, True])
def test_historical_delivery_proof_still_requires_exact_acknowledged_credit_bytes(worker_error):
    from lab_arena import training_coverage
    example = credit_example(worker_error=worker_error, client_ack=True)
    example['configuration'].pop('training_maximum_missing_reasoning_percent')
    seal(example)
    assert not training_coverage.uses_host_result_evidence(example['configuration'])
    call = example['provider_calls'][0]
    assert training_thinking.verified_failed_attempt(example, call)
    call['response_provenance']['agent_receipt'] = 'not_reported'
    assert not training_thinking.verified_failed_attempt(example, call)
