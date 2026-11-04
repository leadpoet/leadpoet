import base64
import copy
import json
import os

import pytest

from lab_arena import contracts as c, training_steps as steps
from scripts.prepare_arena_agent_dataset import main


@pytest.mark.parametrize("value", [
    "Bearer [REDACTED]suffix",
    "Basic [REDACTED]suffix",
    "Bearer [REDACTED].synthetic-tail",
    "Bearer [REDACTED]:synthetic-tail",
    "Bearer [REDACTED]%2Fsynthetic-tail",
    "Bearer %5BREDACTED%5Dsynthetic-tail",
    "Basic %5BREDACTED%5Dsynthetic-tail",
    'Bearer \\"[REDACTED]synthetic-tail\\"',
    '{\\"Authorization\\":\\"[REDACTED]suffix\\"}',
    '{\\\\"Authorization\\\\":\\\\"[REDACTED]suffix\\\\"}',
    '"Proxy-Authorization": "[REDACTED]suffix"',
])
def test_attached_auth_marker_is_not_exact_training_text(value):
    assert steps._exact(value) is False
    assert steps._exact({"messages": [{"role": "tool", "content": value}]}) is False


def test_standalone_auth_marker_remains_exact_training_text():
    assert steps._exact("Bearer [REDACTED]") is True
    assert steps._exact("Bearer %5BREDACTED%5D") is True


def seal(example):
    example['example_hash'] = c.document_hash({k: v for k, v in example.items() if k != 'example_hash'})
    return example


def bind_selected_output(example):
    """Attach the exact accepted output and score inputs used by strict exports."""
    proof = example.setdefault('completeness', {})
    for name in ('context', 'source_identity'):
        proof.setdefault('dimensions', {}).setdefault(name, True)
        proof.setdefault('reasons_by_dimension', {}).setdefault(name, [])
    run_id = example['run_id']
    score_id = example.get('score_run_id', 'score-' + run_id)
    example['score_run_id'] = score_id
    example.setdefault('output', {'companies': []})
    example.setdefault('output_identity', {'document_hash': c.document_hash(example['output'])})
    attempts = example.setdefault('attempts', [])
    for identity, kind in ((run_id, 'execute'), (score_id, 'score')):
        if not any(row.get('run_id') == identity for row in attempts):
            attempts.append({'run_id': identity, 'kind': kind, 'status': 'accepted'})
    snapshots = example.setdefault('input_snapshots', {})
    for identity, document in ((run_id, {'icp': example.get('icp', {})}),
                               (score_id, {'scored_run_id': run_id, 'companies': example['output']['companies']})):
        snapshots.setdefault(identity, {'input': document, 'input_sha256': c.document_hash(document),
                                        'privacy': {'redacted': False, 'redactions': []}})
    for snapshot in snapshots.values():
        snapshot.setdefault('privacy', {'redacted': False, 'redactions': []})
        snapshot['input_sha256'] = c.document_hash(snapshot['input'])
    return seal(example)


def fixture():
    body = {'model': 'test/model', 'messages': [{'role': 'user', 'content': 'Find a company'}], 'temperature': 0.2}
    response = {'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Observed answer', 'reasoning_content': 'Readable summary'}}]}
    rh = c.document_hash({'model': 'test/model'})
    wire = c.document_hash(response)
    data = {'call_identity': 'call-1', 'status': 200, 'body_sha256': wire, 'body_bytes': 100}
    payload = {'response_receipts': [dict(data, ordinal=1, write_state='socket_written', operation_id='openrouter.chat', action_sequence=0)], 'agent_observations': [{'kind': 'response_received', 'id': 'ack', 'privacy': {'redacted': False}, 'data': data}]}
    ph = c.document_hash(payload)
    call = {'run_id': 'run', 'action_sequence': 0, 'call_identity': 'call-1', 'provider': 'openrouter', 'operation_id': 'openrouter.chat', 'request_hash': rh,
            'request': {'operation_id': 'openrouter.chat', 'request_hash': rh, 'hashed_parameters': {'model': 'test/model'},
                        'effective_parameters': {'model': 'test/model'}, 'outbound': {'method': 'POST', 'host': 'openrouter.ai', 'path': '/api/v1/chat/completions', 'body': body, 'body_sha256': c.document_hash(body)}},
            'response': response, 'terminal_entry_id': 5,
            'response_receipts': [{'entry_id': 5, 'response_ref': {'field': 'response', 'entry_id': 5}, 'response_sha256': c.document_hash(response), 'wire_body_sha256': wire, 'wire_body_bytes': 100, 'metadata': {'status': 200}}],
            'response_provenance': {'agent_receipt': 'reported_exact_body_received', 'delivery_to_worker': 'socket_written', 'worker_received_receipt_entry_ids': [5], 'worker_receipt_ref': {'run_id': 'run', 'ordinal': 1, 'payload_sha256': ph}, 'agent_receipt_refs': [{'run_id': 'run', 'id': 'ack', 'payload_sha256': ph}]}}
    return bind_selected_output({'schema_version': 'leadpoet.lab_arena.training_example.v1', 'run_id': 'run', 'round_id': 'arena-2026-10-20', 'round_status': 'published', 'submission_id': 'sub', 'stage': 1, 'icp_position': 0, 'assignment_id': 'a', 'attempt': 1, 'icp': {'industry': 'test'}, 'provider_calls': [call], 'worker_observation_artifacts': {'run': {'payload': payload, 'payload_sha256': ph}}, 'judgment': {'per_icp_score': 1, 'future_secret': 'FUTURE JUDGE'}, 'completeness': {'complete': False, 'incomplete_reasons': ['direct_web_content_not_captured']}})


def records(example):
    splits, manifest = steps.build([example])
    return splits['train'] + splits['eval'], manifest


def test_observed_not_correct_and_no_future_labels():
    rows, manifest = records(fixture())
    assert len(rows) == 1
    row = rows[0]
    assert row['input']['temperature'] == 0.2
    assert row['target'] == {'role': 'assistant', 'content': 'Observed answer'}
    assert row['source_refs'][0]['teacher_reasoning_metadata'][0]['text'] == 'Readable summary'
    assert 'FUTURE JUDGE' not in json.dumps(row)
    assert row['quality']['status'] == 'unreviewed'
    assert manifest['training_permission']['status'] == 'unconfirmed'


def test_private_binary_bytes_never_enter_agent_dataset_splits_or_manifest():
    source = fixture()
    marker = b'UNREVIEWED_BINARY_PAYLOAD\x00\xff'
    encoded = base64.b64encode(marker).decode()
    artifact = source['worker_observation_artifacts']['run']
    artifact['payload']['agent_observations'].append({
        'id': 'binary', 'kind': 'web_fetch_binary',
        'privacy': {'redacted': False},
        'data': {'body_b64': encoded, 'privacy_review': 'unreviewed_binary'}})
    artifact['payload_sha256'] = c.document_hash(artifact['payload'])
    provenance = source['provider_calls'][0]['response_provenance']
    for ref in provenance['agent_receipt_refs'] + [provenance['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    splits, manifest = steps.build([seal(source)])
    assert len(splits['train']) + len(splits['eval']) == 1
    output = json.dumps([splits, manifest])
    assert encoded not in output and 'UNREVIEWED_BINARY_PAYLOAD' not in output
    assert 'body_b64' not in output


@pytest.mark.parametrize('mutate', [
    lambda e: e.update(example_hash='bad'),
    lambda e: e['provider_calls'][0]['request'].update(request_hash='bad'),
    lambda e: e['provider_calls'][0]['request']['outbound'].update(body_sha256='bad'),
    lambda e: e['provider_calls'][0]['response_receipts'][0].update(response_sha256='bad'),
    lambda e: e['provider_calls'][0]['response']['choices'][0].update(finish_reason='length'),
    lambda e: e['provider_calls'][0]['response']['choices'][0]['message'].update(refusal='No'),
    lambda e: e['provider_calls'][0]['response_receipts'][0].update(privacy_redactions=['credential']),
    lambda e: e['provider_calls'][0]['response']['choices'][0]['message'].update(reasoning_details=[{'type': 'reasoning.encrypted', 'data': 'secret'}]),
    lambda e: e['provider_calls'][0]['response_provenance'].update(agent_receipt='not_reported'),
    lambda e: e['provider_calls'][0]['request']['outbound']['body'].update(previous_response_id='external'),
    lambda e: e['provider_calls'][0]['request']['outbound']['body'].update(stream=True),
    lambda e: e['provider_calls'][0].update(run_id='judge'),
])
def test_fail_closed(mutate):
    e = fixture()
    mutate(e)
    # Preserve deliberate example mismatch, otherwise bind the mutation.
    if e['example_hash'] != 'bad':
        call = e['provider_calls'][0]
        call['response_receipts'][0]['response_sha256'] = call['response_receipts'][0]['response_sha256'] if call['response_receipts'][0]['response_sha256'] == 'bad' else c.document_hash(call['response'])
        seal(e)
    assert not records(e)[0]


@pytest.mark.parametrize('marker', ['[REDACTED]', '%5BREDACTED%5D'])
def test_literal_marker_in_acknowledged_tool_input_is_exact(marker):
    e = fixture()
    call = e['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['messages'] = [
        {'role': 'user', 'content': 'Research the source'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'search-1', 'type': 'function',
            'function': {'name': 'search', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'search-1', 'content': json.dumps({'snippet': marker})},
    ]
    body['tools'] = [{'type': 'function', 'function': {
        'name': 'search', 'parameters': {'type': 'object'}}}]
    call['request']['outbound']['body_sha256'] = c.document_hash(body)
    call['request_capture'] = {'redacted': False, 'redactions': []}
    rows, _ = records(seal(e))
    assert len(rows) == 1
    assert rows[0]['input']['messages'][2]['content'] == json.dumps({'snippet': marker})
    assert rows[0]['source_refs'][0]['delivered_response_sha256'] == c.document_hash(call['response'])

    call['request_capture'] = {'redacted': True, 'redactions': ['credential']}
    assert not records(seal(e))[0]
    call['request_capture'] = {'redacted': False, 'redactions': []}
    call['request']['effective_parameters']['api_key'] = 'synthetic-secret'
    assert not records(seal(e))[0]


def test_literal_model_reply_is_not_a_privacy_redaction_without_receipt_flag():
    e = fixture()
    call = e['provider_calls'][0]
    call['response']['choices'][0]['message']['content'] = '[REDACTED]'
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    assert records(seal(e))[0][0]['target']['content'] == '[REDACTED]'
    call['response_receipts'][0]['privacy_redactions'] = ['credential']
    assert not records(seal(e))[0]


def test_select_acknowledged_earlier_reply_not_billing_recovery():
    e = fixture()
    call = e['provider_calls'][0]
    old = copy.deepcopy(call['response'])
    call['response_receipts'][0].pop('response_ref')
    call['response_receipts'][0]['response'] = old
    call['terminal_entry_id'] = 9
    call['response'] = {'billing_recovery': True}
    assert records(seal(e))[0][0]['target']['content'] == 'Observed answer'


def test_ambiguous_ack_rejected():
    e = fixture()
    artifact = e['worker_observation_artifacts']['run']
    ack = copy.deepcopy(artifact['payload']['agent_observations'][0])
    ack['data']['body_sha256'] = 'different'
    artifact['payload']['agent_observations'].append(ack)
    artifact['payload_sha256'] = c.document_hash(artifact['payload'])
    e['provider_calls'][0]['response_provenance']['agent_receipt_refs'][0]['payload_sha256'] = artifact['payload_sha256']
    assert not records(seal(e))[0]


def test_dedup_and_split_repeatable():
    one = fixture()
    two = copy.deepcopy(one)
    two['attempt'] = 2
    seal(two)
    a, m = steps.build([one, two])
    b, n = steps.build([two, one])
    assert (a, m) == (b, n)
    rows = a['train'] + a['eval']
    assert len(rows) == 1 and len(rows[0]['source_refs']) == 2


def test_nonproduction_and_private_cli(tmp_path):
    e = fixture()
    e['round_status'] = 'open'
    seal(e)
    assert not records(e)[0]
    assert steps.build([e], steps.Rules(production_rounds_only=False))[1]['counts']['candidate_executions'] == 1
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(fixture()))
    output = tmp_path / 'output'
    assert main(['--examples', str(source), '--output', str(output)]) == 0
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in output.iterdir())
    assert main(['--examples', str(source), '--output', str(output)]) == 1


def response_fixture():
    e = fixture()
    call = e['provider_calls'][0]
    op = 'openrouter.responses'
    call['operation_id'] = call['request']['operation_id'] = op
    body = {'model': 'test/model', 'input': 'Research this company', 'reasoning': {'effort': 'low'}}
    call['request']['outbound'].update(path='/api/v1/responses', body=body, body_sha256=c.document_hash(body))
    response = {'status': 'completed', 'output': [
        {'type': 'reasoning', 'id': 'r', 'summary': [{'type': 'summary_text', 'text': 'Readable teacher summary'}]},
        {'type': 'message', 'id': 'm', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': 'Answer', 'annotations': []}]}]}
    call['response'] = response
    call['response_receipts'][0]['response_sha256'] = c.document_hash(response)
    artifact = e['worker_observation_artifacts']['run']
    artifact['payload']['response_receipts'][0]['operation_id'] = op
    artifact['payload_sha256'] = c.document_hash(artifact['payload'])
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    return seal(e)


def test_responses_readable_summary_is_metadata_only():
    rows, _ = records(response_fixture())
    assert len(rows) == 1
    assert rows[0]['target'][0]['content'][0]['text'] == 'Answer'
    assert 'Readable teacher summary' not in json.dumps(rows[0]['target'])
    assert rows[0]['source_refs'][0]['teacher_reasoning_metadata'][0]['text'] == 'Readable teacher summary'


@pytest.mark.parametrize('change', [
    lambda b, r: b.update(tools=[{'type': 'web_search'}]),
    lambda b, r: b.update(input=[{'type': 'item_reference', 'id': 'external'}]),
    lambda b, r: r.update(status='incomplete'),
    lambda b, r: r['output'][0].update(encrypted_content='DO NOT WRITE'),
    lambda b, r: r['output'].append({'type': 'web_search_call', 'status': 'completed'}),
    lambda b, r: r['output'].append(copy.deepcopy(r['output'][1])),
])
def test_responses_unsupported_and_opaque(change):
    e = response_fixture()
    call = e['provider_calls'][0]
    change(call['request']['outbound']['body'], call['response'])
    call['request']['outbound']['body_sha256'] = c.document_hash(call['request']['outbound']['body'])
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    assert not records(seal(e))[0]


def test_tool_mismatch_and_multiple_choices():
    for choice in [
        {'finish_reason': 'stop', 'message': {'role': 'assistant', 'tool_calls': [{'id': 'x', 'type': 'function', 'function': {'name': 'unknown', 'arguments': '{}'}}]}},
        {'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'tool_calls': [{'id': 'x', 'type': 'function', 'function': {'name': 'unknown', 'arguments': '{}'}}]}},
    ]:
        e = fixture()
        call = e['provider_calls'][0]
        call['response']['choices'] = [choice]
        call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
        assert not records(seal(e))[0]
    e = fixture()
    call = e['provider_calls'][0]
    call['response']['choices'] *= 2
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    assert not records(seal(e))[0]


@pytest.mark.parametrize('content,identifier,name,arguments', [
    ({'bad': 'object'}, 'id', 'search', '{}'),
    (None, '', 'search', '{}'),
    (None, 'id', '', '{}'),
    (None, 'id', 'search', 'not json'),
    (None, 'id', 'search', '[]'),
    (None, 'id', 'search', '{"x": NaN}'),
])
def test_invalid_tool_fields_rejected(content, identifier, name, arguments):
    e = fixture()
    call = e['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'] = [{'type': 'function', 'function': {'name': 'search', 'parameters': {'type': 'object'}}}]
    call['request']['outbound']['body_sha256'] = c.document_hash(body)
    call['response']['choices'] = [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': content, 'tool_calls': [{'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]}}]
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    assert not records(seal(e))[0]


def test_only_opaque_response_omission_is_allowed():
    e = fixture()
    e['provider_calls'][0]['response_receipts'][0]['privacy_redactions'] = ['hidden_reasoning']
    row = records(seal(e))[0][0]
    assert row['source_refs'][0]['response_projection'] == {'opaque_reasoning_omitted': True, 'exact_provider_replay': False}
    e['provider_calls'][0]['response_receipts'][0]['privacy_redactions'] = ['credential']
    assert not records(seal(e))[0]


def test_responses_server_conversation_rejected():
    e = response_fixture()
    call = e['provider_calls'][0]
    call['request']['outbound']['body']['conversation'] = 'external-conversation'
    call['request']['outbound']['body_sha256'] = c.document_hash(call['request']['outbound']['body'])
    assert not records(seal(e))[0]


def test_valid_declared_chat_tool_call_candidate():
    e = fixture()
    call = e['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'] = [{'type': 'function', 'function': {'name': 'search', 'parameters': {'type': 'object'}}}]
    call['request']['outbound']['body_sha256'] = c.document_hash(body)
    call['response']['choices'] = [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'next', 'type': 'function', 'function': {'name': 'search', 'arguments': '{"query":"company"}'}}]}}]
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    assert records(seal(e))[0][0]['target']['tool_calls'][0]['function']['name'] == 'search'


def test_related_nonidentical_candidates_stay_in_one_split():
    one = fixture()
    two = copy.deepcopy(one)
    two['attempt'] = 2
    call = two['provider_calls'][0]
    call['request']['outbound']['body']['messages'][0]['content'] = 'A different context'
    call['request']['outbound']['body_sha256'] = c.document_hash(call['request']['outbound']['body'])
    seal(two)
    splits, manifest = steps.build([one, two])
    assert sorted([len(splits['train']), len(splits['eval'])]) == [0, 2]
    assert manifest['counts']['candidate_executions'] == 1


def test_teacher_summary_variation_dedup_is_order_independent():
    one = fixture()
    two = copy.deepcopy(one)
    call = two['provider_calls'][0]
    call['response']['choices'][0]['message']['reasoning_content'] = 'Other readable summary'
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    seal(two)
    assert steps.build([one, two]) == steps.build([two, one])


@pytest.mark.parametrize('field,value', [('prompt', {'id': 'server-prompt'}), ('plugins', [{'id': 'web'}]), ('transforms', ['middle-out'])])
def test_external_prompt_rewrites_rejected(field, value):
    e = response_fixture()
    call = e['provider_calls'][0]
    call['request']['outbound']['body'][field] = value
    call['request']['outbound']['body_sha256'] = c.document_hash(call['request']['outbound']['body'])
    assert not records(seal(e))[0]


def test_malformed_call_does_not_abort_valid_sibling():
    e = fixture()
    e['provider_calls'].append(None)
    rows, manifest = records(seal(e))
    assert len(rows) == 1
    assert manifest['counts']['exclusion_reasons']['malformed_call'] == 1


def test_unrelated_page_privacy_gap_does_not_reject_exact_step():
    e = fixture()
    e['completeness']['incomplete_reasons'].append('worker_observation_privacy_nonexact')
    assert len(records(seal(e))[0]) == 1


@pytest.mark.parametrize('field,value', [('reasoning', None), ('reasoning', ''), ('reasoning_content', None), ('reasoning_content', ''), ('reasoning_details', None), ('reasoning_details', [])])
def test_empty_optional_reasoning_is_not_a_target(field, value):
    e = fixture()
    call = e['provider_calls'][0]
    call['response']['choices'][0]['message'][field] = value
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    row = records(seal(e))[0][0]
    assert row['target'] == {'role': 'assistant', 'content': 'Observed answer'}


def test_typed_readable_chat_reasoning_is_metadata_only():
    e = fixture()
    call = e['provider_calls'][0]
    details = [{'type': 'reasoning.summary', 'summary': 'A readable summary', 'index': 0, 'format': 'unknown'}]
    call['response']['choices'][0]['message']['reasoning_details'] = details
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    row = records(seal(e))[0][0]
    assert details[0] in row['source_refs'][0]['teacher_reasoning_metadata']
    assert 'reasoning' not in json.dumps(row['target'])


def test_own_ack_redaction_rejected_despite_unrelated_gap_allowance():
    e = fixture()
    e['completeness']['incomplete_reasons'].append('worker_observation_privacy_nonexact')
    artifact = e['worker_observation_artifacts']['run']
    artifact['payload']['agent_observations'][0]['privacy']['redacted'] = True
    artifact['payload_sha256'] = c.document_hash(artifact['payload'])
    call = e['provider_calls'][0]
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    assert not records(seal(e))[0]


@pytest.mark.parametrize('make_example', [fixture, response_fixture])
@pytest.mark.parametrize('field', ['citations', 'search_results', 'annotations'])
@pytest.mark.parametrize('value', [[{'url': 'https://example.org/source'}], [], None])
def test_top_level_retrieval_markers_excluded(make_example, field, value):
    e = make_example()
    call = e['provider_calls'][0]
    call['response'][field] = value
    call['response_receipts'][0]['response_sha256'] = c.document_hash(call['response'])
    rows, manifest = records(seal(e))
    assert len(rows) == (0 if value else 1)
    if value:
        assert manifest['counts']['exclusion_reasons'] == {'retrieval_context_unsupported': 1}


@pytest.mark.parametrize('value', [{'search_context_size': 'low'}, {}, None])
def test_request_web_search_options_excluded(value):
    e = fixture()
    call = e['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['web_search_options'] = value
    call['request']['outbound']['body_sha256'] = c.document_hash(body)
    # An empty options object can enable search with provider defaults.
    assert len(records(seal(e))[0]) == (1 if value is None else 0)
