"""Exact failed host delivery consumes an ICP, without claiming a read or target."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    _add_settled_exa_call, assess_ten, judge_only_gap_example, refresh_response_links)
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_steps_test import seal


def refresh(example):
    artifact = example['worker_observation_artifacts']['run']
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in example['provider_calls']:
        if call.get('run_id') == 'run':
            for name in ('worker_receipt_ref', 'worker_error_observation_ref'):
                if name in call['response_provenance']:
                    call['response_provenance'][name]['payload_sha256'] = artifact['payload_sha256']
    return seal(example)


def failed_delivery(*, model=False, status=502, api='openrouter.chat'):
    example = judge_only_gap_example()
    example['provider_calls'] = [call for call in example['provider_calls'] if call['run_id'] == 'run']
    if api == 'openrouter.responses':
        from tests.lab_arena.training_cap_eof_test import cap_example
        responses = cap_example(api)
        responses['provider_calls'][0]['response']['output'][1]['arguments'] = '{}'
        responses['provider_calls'][0]['response']['usage']['output_tokens'] = 50
        refresh_response_links(responses, 0)
        for key in ('provider_calls', 'configuration', 'worker_observation_artifacts'):
            example[key] = responses[key]
    if not model:
        _add_settled_exa_call(example)
    call = example['provider_calls'][0 if model else -1]
    call['response_status'] = status
    if status != 200:
        call.update(response={'error': {'code': 'provider_unavailable'}}, actual_microusd=0)
    call['lifecycle'][-1].update(entry_id=call['terminal_entry_id'], amount_microusd=call['actual_microusd'])
    call['response_receipts'][0]['metadata'].update(status=status, call_succeeded=status == 200)
    refresh_response_links(example, example['provider_calls'].index(call))
    example = without_client_callbacks(example)
    call = example['provider_calls'][0 if model else -1]
    if status != 200:
        raw = b'{"error":{"message":"Saved upstream failure"}}'
        call['upstream_response_receipts'] = [{
            'entry_id': call['terminal_entry_id'], 'status': status,
            'response': {'error': {'message': 'Saved upstream failure'}}, 'privacy_redactions': [],
            'body_sha256': contracts.hash_bytes(raw), 'body_bytes': len(raw),
            'wire_body_sha256': contracts.hash_bytes(raw), 'wire_body_bytes': len(raw),
            'response_sha256': contracts.document_hash({'error': {'message': 'Saved upstream failure'}}),
            'provenance': 'private_upstream_response_receipt', 'delivery_to_worker': 'not_proved'}]
    buffer = training_observations.Buffer()
    old = example['worker_observation_artifacts']['run']['payload']['response_receipts']
    for item, write in zip(example['provider_calls'], old):
        item['call_identity'] = contracts.hash_bytes(item['call_identity'].encode())
        write = {**write, 'call_identity': item['call_identity'], 'protocol': 'frame',
                 'status': item['response_status'], 'socket_written_at': '2026-10-27T00:00:00Z'}
        if item is call:
            write.update(write_state='prepared', socket_written_at=None)
        assert buffer.append_response_receipt(write)
        if item is call:
            buffer.mark_socket_write(write['ordinal'], written_at=None)
    snapshot, _ = training_observations.build_snapshot('run', 'execute', buffer.document(),
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b'claim'))
    example['worker_observation_artifacts']['run'] = snapshot
    # Derive the truthful links from the production exporter.
    assert not training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, example['provider_calls'], snapshot,
        require_client_ack=False, configuration=example['configuration'])
    example['attempts'] = [{'run_id': 'run', 'kind': 'execute', 'status': 'accepted'},
                           {'run_id': 'score', 'kind': 'score', 'status': 'accepted'}]
    example['related_run_gap_attribution'] = {
        'run': {'kind': 'execute', 'status': 'accepted', 'research_process_gaps': ['worker_observations_incomplete'], 'cost_gaps': []},
        'score': {'kind': 'score', 'status': 'accepted', 'research_process_gaps': [], 'cost_gaps': []}}
    dimensions = {name: [] for name in ('context', 'source_identity', 'exact_inputs', 'input_output_label', 'costs', 'research_process')}
    dimensions['research_process'] = ['worker_observations_incomplete']
    example['completeness'] = {'complete': False, 'incomplete_reasons': ['worker_observations_incomplete'],
        'dimensions': {name: not gaps for name, gaps in dimensions.items()}, 'reasons_by_dimension': dimensions}
    return refresh(example)


@pytest.mark.parametrize('model,status,api', [(False, 502, 'openrouter.chat'), (False, 200, 'openrouter.chat'),
    (True, 502, 'openrouter.chat'), (True, 200, 'openrouter.chat'),
    (True, 502, 'openrouter.responses'), (True, 200, 'openrouter.responses')])
def test_exact_failed_delivery_is_one_incomplete_icp(model, status, api):
    example = failed_delivery(model=model, status=status, api=api)
    before = deepcopy(example)
    examples = bank(example)
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    assert not assess_ten(examples, percent=100)['eligible']
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    runs = {ref['run_id'] for split in ('train', 'eval') for row in strict[split] for ref in row['source_refs']}
    assert runs == set(examples) - {'run'}
    assert example == before


def test_failed_delivery_shares_the_existing_coverage_floor():
    examples = bank(failed_delivery())
    examples['run-8'] = None
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 8
    examples['run-7'] = None
    result = assess_ten(examples)
    assert result['reasons'] == ['training_coverage_incomplete']


@pytest.mark.parametrize('damage', [
    'provenance', 'write_state', 'written_at', 'missing_host', 'extra_host', 'duplicate_call',
    'body_hash', 'body_length', 'body_status', 'response_hash', 'changed_response',
    'request_hash', 'outbound_hash', 'host_ref', 'read_claim', 'read_ref', 'selected_ack',
    'error_ack', 'actual_read', 'body_source', 'body_exact', 'extra_receipt_id', 'duplicate_receipt_id',
    'uncertain', 'negative_cost', 'terminal_amount',
    'terminal_missing', 'request_privacy', 'response_privacy', 'receipt_privacy',
    'upstream_privacy', 'upstream_hash', 'upstream_entry', 'upstream_missing', 'upstream_delivered',
    'upstream_response', 'upstream_missing_hash', 'upstream_none', 'upstream_duplicate',
    'terminal_duplicate', 'provider', 'host', 'snapshot_count', 'unknown_capture', 'prepared_other', 'partial',
    'other_response', 'other_target', 'other_route', 'other_cost', 'other_privacy',
    'other_upstream_privacy', 'other_upstream_hash', 'other_upstream_duplicate',
    'input_hash', 'output_hash', 'source', 'attribution', 'unknown_gap', 'legacy'])
def test_failed_delivery_cannot_hide_inconsistent_or_unrelated_evidence(damage):
    example = failed_delivery()
    call = example['provider_calls'][-1]
    other = example['provider_calls'][1]
    provenance = call['response_provenance']
    payload = example['worker_observation_artifacts']['run']['payload']
    write = payload['response_receipts'][-1]
    upstream = call['upstream_response_receipts'][0]
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'provenance': provenance['delivery_to_worker'] = 'socket_written'
    elif damage == 'write_state': write['write_state'] = 'prepared'
    elif damage == 'written_at': write['socket_written_at'] = '2026-10-27T00:00:00Z'
    elif damage == 'missing_host': payload['response_receipts'].pop(); payload['response_receipt_count'] -= 1
    elif damage == 'extra_host':
        row = deepcopy(write); row.update(call_identity=wrong, ordinal=100)
        payload['response_receipts'].append(row); payload['response_receipt_count'] += 1
    elif damage == 'duplicate_call': example['provider_calls'].append(deepcopy(call))
    elif damage == 'body_hash': write['body_sha256'] = wrong
    elif damage == 'body_length': write['body_bytes'] += 1
    elif damage == 'body_status': write['status'] = 503
    elif damage == 'response_hash': call['response_sha256'] = wrong
    elif damage == 'changed_response':
        call['response'] = {'error': {'code': 'changed'}}
        call['response_sha256'] = contracts.document_hash(call['response'])
    elif damage == 'request_hash': call['request']['request_hash'] = wrong
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = wrong
    elif damage == 'host_ref': provenance['worker_receipt_ref']['ordinal'] = 999
    elif damage == 'read_claim': provenance['agent_receipt'] = 'reported_exact_body_received'
    elif damage == 'read_ref': provenance['agent_receipt_refs'] = [{'id': 0, 'run_id': 'run', 'payload_sha256': wrong}]
    elif damage == 'selected_ack': provenance['selected_response_acknowledged'] = True
    elif damage == 'error_ack': provenance['worker_error_body_acknowledged'] = True
    elif damage == 'body_source': provenance['worker_body_source'] = 'worker_error_observation'
    elif damage == 'body_exact': provenance['worker_body_exact_in_artifact'] = False
    elif damage == 'extra_receipt_id': provenance['worker_received_receipt_entry_ids'].append(123456)
    elif damage == 'duplicate_receipt_id': provenance['worker_received_receipt_entry_ids'] *= 2
    elif damage == 'actual_read':
        payload['agent_observations'] = [{'id': 0, 'kind': 'response_received', 'observed_at': '2026-10-27T00:00:00Z',
            'after_action_sequence': call['action_sequence'], 'depends_on': [],
            'privacy': {'redacted': False, 'redactions': []},
            'data': {key: write[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}}]
        payload['agent_observation_count'] = 1
    elif damage == 'uncertain': call.update(accounting_state='uncertain', actual_microusd=None)
    elif damage == 'negative_cost': call['actual_microusd'] = -1
    elif damage == 'terminal_amount': call['lifecycle'][-1]['amount_microusd'] += 1
    elif damage == 'terminal_missing': call['lifecycle'].pop()
    elif damage == 'request_privacy': call['request_capture']['redacted'] = True
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'receipt_privacy': call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'upstream_privacy': upstream['privacy_redactions'] = ['credential']
    elif damage == 'upstream_hash': upstream['wire_body_sha256'] = wrong
    elif damage == 'upstream_entry': upstream['entry_id'] = 9999
    elif damage == 'upstream_missing': call.pop('upstream_response_receipts')
    elif damage == 'upstream_delivered': upstream['delivery_to_worker'] = 'socket_written'
    elif damage == 'upstream_response': upstream['response']['error']['message'] = 'Changed upstream body'
    elif damage == 'upstream_missing_hash':
        upstream.pop('response_sha256'); upstream['response'] = {'invented': 'value'}
    elif damage == 'upstream_none':
        upstream.pop('response_sha256'); upstream['response'] = None
    elif damage == 'upstream_duplicate': call['upstream_response_receipts'].append(deepcopy(upstream))
    elif damage == 'terminal_duplicate': call['response_receipts'].append(deepcopy(call['response_receipts'][0]))
    elif damage == 'provider': call['provider'] = 'openrouter'
    elif damage == 'host': call['request']['outbound']['host'] = 'wrong.invalid'
    elif damage == 'snapshot_count': payload['observation_count'] += 1
    elif damage == 'unknown_capture': payload['failures'].append('unknown_capture_loss')
    elif damage == 'prepared_other': payload['response_receipts'][0].update(write_state='prepared', socket_written_at=None)
    elif damage == 'partial': example['partial_worker_observation_artifacts'] = {'score': {'complete': False}}
    elif damage == 'other_response': other['response']['changed'] = True
    elif damage == 'other_target':
        other['response']['choices'] = []
        for item in example['provider_calls']:
            item['response_provenance'].setdefault('agent_receipt_refs', [])
        refresh_response_links(example, 1)
    elif damage == 'other_route': other['reasoning_capture']['provider_route'] = 'unapproved'
    elif damage == 'other_cost': other['actual_microusd'] = None
    elif damage == 'other_privacy': other['response_privacy_redactions'] = ['credential']
    elif damage.startswith('other_upstream'):
        copied = deepcopy(upstream)
        copied['entry_id'] = other['terminal_entry_id']
        other['upstream_response_receipts'] = [copied]
        if damage == 'other_upstream_privacy': copied['privacy_redactions'] = ['credential']
        elif damage == 'other_upstream_hash': copied['response_sha256'] = wrong
        else: other['upstream_response_receipts'].append(deepcopy(copied))
    elif damage == 'input_hash': example['input_snapshots']['score']['input_sha256'] = wrong
    elif damage == 'output_hash': example['output_identity']['document_hash'] = wrong
    elif damage == 'source': example['source']['sha256'] = wrong
    elif damage == 'attribution': example['related_run_gap_attribution']['run']['research_process_gaps'] = []
    elif damage == 'unknown_gap':
        example['completeness']['incomplete_reasons'].append('unknown_integrity_failure')
        example['completeness']['reasons_by_dimension']['research_process'].append('unknown_integrity_failure')
    else: example['configuration'].pop('training_maximum_missing_reasoning_percent')
    refresh(example)
    result = assess_ten(bank(example))
    assert not result['eligible'], (damage, result)


@pytest.mark.parametrize('damage', ['route', 'observation', 'controls', 'history', 'request_api', 'outbound', 'unavailable'])
def test_undelivered_generation_still_requires_exact_model_request_and_route(damage):
    example = failed_delivery(model=True, status=200)
    call = example['provider_calls'][0]
    if damage == 'route': call['reasoning_capture']['provider_route'] = 'unapproved'
    elif damage == 'observation': call['response_receipts'][0]['metadata']['reasoning_route_observation']['status'] = 'mismatched'
    elif damage == 'controls':
        call['request']['outbound']['body']['reasoning'] = {'enabled': False}
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    elif damage == 'history':
        body = call['request']['outbound']['body']
        body['messages'].append({'role': 'tool', 'tool_call_id': 'missing', 'content': 'unpaired'})
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    elif damage == 'request_api': call['request']['operation_id'] = 'openrouter.responses'
    elif damage == 'outbound': call['request']['outbound']['host'] = 'wrong.invalid'
    else: call['response_receipts'][0]['response_unavailable'] = True
    refresh(example)
    assert not assess_ten(bank(example))['eligible']


@pytest.mark.parametrize('cost_gap', [False, True])
@pytest.mark.parametrize('tamper', [False, True])
def test_failed_delivery_composes_only_with_independently_attributed_judge_gaps(cost_gap, tamper):
    example = failed_delivery()
    judge = judge_only_gap_example(cost_gap=cost_gap)
    example['provider_calls'].append(judge['provider_calls'][-1])
    gaps = judge['completeness']['incomplete_reasons']
    proof = example['completeness']
    proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | set(gaps))
    proof['reasons_by_dimension']['research_process'] = proof['incomplete_reasons'][:]
    proof['reasons_by_dimension']['costs'] = ['provider_cost_unresolved'] if cost_gap else []
    proof['dimensions']['costs'] = not cost_gap
    example['related_run_gap_attribution']['score'].update(
        research_process_gaps=gaps[:], cost_gaps=proof['reasons_by_dimension']['costs'][:])
    if tamper:
        example['provider_calls'][1]['response_privacy_redactions'] = ['credential']
    seal(example)
    result = assess_ten(bank(example))
    assert result['eligible'] is (not tamper), result
    assert result['counts']['incomplete_runs'] == 1


def test_multiple_exact_failed_writes_still_consume_only_one_icp():
    example = failed_delivery()
    for call, write in zip(example['provider_calls'][:2],
                           example['worker_observation_artifacts']['run']['payload']['response_receipts'][:2]):
        write.update(write_state='socket_write_failed', socket_written_at=None)
        call['response_provenance']['delivery_to_worker'] = 'socket_write_failed'
    refresh(example)
    result = assess_ten(bank(example))
    assert result['eligible'] and result['counts']['incomplete_runs'] == 1, result


def test_exact_failed_delivery_does_not_hide_a_missing_execution_model_call():
    example = failed_delivery()
    example['provider_calls'] = example['provider_calls'][-1:]
    payload = example['worker_observation_artifacts']['run']['payload']
    payload['response_receipts'] = payload['response_receipts'][-1:]
    payload['response_receipt_count'] = 1
    refresh(example)
    assert not assess_ten(bank(example))['eligible']


@pytest.mark.parametrize('model,status', [(False, 502), (True, 200)])
def test_worker_failed_write_through_actual_export_assembly_and_eligibility(model, status):
    from lab_arena import broker, training_capture
    from tests.lab_arena.training_export_test import per_icp_cost_fixture, per_icp_cost_reader
    from tests.lab_arena.training_observations_test import save
    source = failed_delivery(model=model, status=status)
    values = per_icp_cost_fixture()
    config = values['round_row']['configuration_doc']
    config.update(source['configuration'])
    run = values['runs'][0]
    run_id = run['run_id']
    ledger = []
    for call in source['provider_calls']:
        request = deepcopy(call['request'])
        if call.get('reasoning_capture'):
            request['reasoning_capture'] = deepcopy(call['reasoning_capture'])
        pointer = training_capture.RequestCapture(values['objects'])(request)
        context = {'run_id': run_id, 'round_id': run['round_id'], 'submission_id': run['submission_id'],
            'miner_hotkey': run['miner_hotkey'],
            'provider': call['provider'], 'operation_id': call['operation_id'],
            'call_identity': call['call_identity'], 'funding_source': 'miner'}
        entry_id = call['terminal_entry_id']
        terminal = broker._terminal_response_document(call['response_status'], {},
            contracts.canonical_json(call['response']).encode(), call_succeeded=call['response_status'] == 200)
        if call.get('reasoning_capture'):
            terminal['reasoning_route_observation'] = deepcopy(call['response_receipts'][0]['metadata']['reasoning_route_observation'])
        if call.get('upstream_response_receipts'):
            upstream = call['upstream_response_receipts'][0]
            terminal['upstream_response_receipt'] = broker._capture_upstream_response(
                broker.ProviderResponse(upstream['status'], {}, contracts.canonical_json(upstream['response']).encode()),
                operation_id=call['operation_id'], request_hash=call['request_hash'], call_identity=call['call_identity'])
        ledger.extend([
            dict(context, entry_id=entry_id-2, entry_kind='reservation', amount_microusd=0,
                 entry_doc={'request_hash': call['request_hash'], 'action_sequence': call['action_sequence'],
                    'base_call_identity': call['call_identity'], 'provider_attempt': 1,
                    'request_capture': pointer, **({'reasoning_capture': call['reasoning_capture']} if call.get('reasoning_capture') else {})}),
            dict(context, entry_id=entry_id-1, entry_kind='dispatch', amount_microusd=0, entry_doc={}),
            dict(context, entry_id=entry_id, entry_kind='settlement', amount_microusd=call['actual_microusd'],
                 entry_doc={}, terminal_response=terminal)])
    values['ledger_rows'] = ledger
    run['result_doc']['resource_summary']['provider_call_count'] = len(source['provider_calls'])
    save(values, deepcopy(source['worker_observation_artifacts']['run']['payload']))
    example = training_export.assemble_example(**values,
        cost_eligibility_reader=per_icp_cost_reader(spend=3, reason='eligible'))
    before = deepcopy(example)
    assert example['completeness']['incomplete_reasons'] == ['worker_observations_incomplete']
    assert training_eligibility._failed_delivery_gap_verified(example) == {'worker_observations_incomplete'}
    examples = bank(source)
    examples.pop('run')
    examples[run_id] = example
    digest = example['source']['sha256']
    for identity, control in examples.items():
        if identity != run_id:
            control['source']['sha256'] = digest
            control['source']['hash_origin'] = 'computed_from_frozen_object'
            control['submission_id'] = example['submission_id']
            control['role'] = example['role']
            seal(control)
    result = training_eligibility.assess_capture(values['submission_row'], examples, list(examples),
        enabled=True, minimum_coverage_percent=80)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    failed = next(call for call in example['provider_calls'] if call['response_provenance']['delivery_to_worker'] == 'socket_write_failed')
    assert failed['response_provenance']['agent_receipt'] == 'not_reported'
    with pytest.raises(ValueError, match='worker_write_unverified'):
        training_steps._host_result(example, failed)
    assert example == before
