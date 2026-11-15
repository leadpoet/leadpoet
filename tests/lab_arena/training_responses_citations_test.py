"""Provider citation metadata is an exact output, not a new inbound contract."""
from copy import deepcopy

import pytest

from lab_arena import contracts, operations, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


def cited_example(*, uncited=True, truncated=False):
    value = cap_example('openrouter.responses')
    call = value['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'].append({'type': 'openrouter:web_search', 'parameters': {
        'engine': 'exa', 'max_uses': 1, 'max_total_results': 5}})
    body['max_tool_calls'] = 1
    if not truncated:
        call['response']['output'][-1]['arguments'] = '{}'
    call['response']['output'][1:1] = [
        {'type': 'openrouter:web_search', 'id': 'search', 'status': 'completed',
         'action': {'type': 'search', 'query': 'public company source', 'sources': [
             {'type': 'url', 'url': 'https://example.com/source'}]}},
        {'type': 'message', 'id': 'source-message', 'role': 'assistant', 'status': 'completed',
         'phase': 'commentary', 'content': [{'type': 'output_text', 'text': 'Retained source.',
             'annotations': [{'type': 'url_citation', 'url': 'https://example.com/source',
                 'title': 'Original source', 'content': 'Substantive retained provider excerpt.',
                 'start_index': 0, 'end_index': 8, 'uncited': uncited}]}]},
    ]
    return refresh(value)


def refresh(value):
    call = value['provider_calls'][0]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    refresh_response_links(value, 0)
    return seal(value)


def annotation(value):
    return value['provider_calls'][0]['response']['output'][2]['content'][0]['annotations'][0]


def records(builder, value):
    splits, _ = builder([value], training_steps.Rules(production_rounds_only=False))
    return splits['train'] + splits['eval']


@pytest.mark.parametrize('uncited', [True, False])
def test_exact_provider_citations_survive_shared_target_strict_export_and_eligibility(uncited):
    value = cited_example(uncited=uncited)
    before = deepcopy(value)
    call = value['provider_calls'][0]
    target, _ = training_steps._target(call['request']['outbound']['body'], call['response'],
        'openrouter.responses', allow_reasoning_history=True, allow_tool_error_history=True)
    assert target == call['response']['output']
    for builder in (training_steps.build, training_thinking.build):
        got = records(builder, value)
        assert len(got) == 3
        cited = next(row for row in got if row['format'] == 'openrouter.responses')
        assert cited['target'] == target
        assert cited['input'] == call['request']['outbound']['body']
    report = training_thinking.readable_trace(value)
    assert any(event['value'] == call['response']['output'][2]
               for event in report['calls'][0]['events'] if event['kind'] == 'assistant_output')
    result = training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True)
    assert result['eligible'], result
    assert not training_thinking.verified_cap_truncated_target(value, call)
    assert value == before


@pytest.mark.parametrize('damage', ['string', 'integer', 'null', 'unknown_field', 'missing_url',
    'bad_content', 'unsafe_url', 'annotation_count', 'annotation_not_mapping'])
def test_malformed_provider_citations_remain_excluded(damage):
    value = cited_example()
    citation = annotation(value)
    if damage in {'string', 'integer', 'null'}:
        citation['uncited'] = {'string': 'true', 'integer': 1, 'null': None}[damage]
    elif damage == 'unknown_field':
        citation['unknown'] = 'unsupported'
    elif damage == 'missing_url':
        citation.pop('url')
    elif damage == 'bad_content':
        citation['content'] = ['not text']
    elif damage == 'unsafe_url':
        citation['url'] = 'file:///private/source'
    elif damage == 'annotation_count':
        value['provider_calls'][0]['response']['output'][2]['content'][0]['annotations'] *= 129
    else:
        value['provider_calls'][0]['response']['output'][2]['content'][0]['annotations'] = [1]
    refresh(value)
    call = value['provider_calls'][0]
    with pytest.raises(ValueError, match='assistant_shape_unsupported'):
        training_steps._target(call['request']['outbound']['body'], call['response'],
            'openrouter.responses', allow_reasoning_history=True, allow_tool_error_history=True)
    for builder in (training_steps.build, training_thinking.build):
        assert len(records(builder, value)) == 2  # The independent calls survive.
    assert not training_thinking.verified_cap_truncated_target(value, call)
    assert not training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True)['eligible']


def test_inbound_history_search_declaration_and_limits_remain_strict():
    value = cited_example()
    call = value['provider_calls'][0]
    body, response = call['request']['outbound']['body'], call['response']
    message = deepcopy(response['output'][2])
    with pytest.raises(operations.OperationRequestError) as error:
        operations.validate_operation_request('openrouter.responses',
            {'model': body['model'], 'input': [message]}, allow_full_thinking=True)
    assert error.value.code == 'unknown_field'
    history = dict(body, input=[message])
    with pytest.raises(ValueError, match='input_shape_unsupported'):
        training_steps._target(history, response, 'openrouter.responses', allow_reasoning_history=True)
    undeclared = dict(body, tools=body['tools'][:1])
    undeclared.pop('max_tool_calls')
    with pytest.raises(ValueError, match='hosted_tool_undeclared'):
        training_steps._target(undeclared, response, 'openrouter.responses', allow_reasoning_history=True)
    with pytest.raises(ValueError, match='input_shape_unsupported'):
        training_steps._target(dict(body, max_tool_calls=10), response,
            'openrouter.responses', allow_reasoning_history=True)


@pytest.mark.parametrize('damage', ['response_body', 'receipt_hash', 'ack', 'privacy', 'incomplete'])
def test_citation_metadata_cannot_bypass_delivery_privacy_or_completion(damage):
    value = cited_example()
    call = value['provider_calls'][0]
    if damage == 'response_body':
        call['response']['output'][2]['content'][0]['text'] = 'Unverified changed answer.'
    elif damage == 'receipt_hash':
        call['response_receipts'][0]['response_sha256'] = 'forged'
    elif damage == 'ack':
        call['response_provenance']['agent_receipt'] = 'not_reported'
    elif damage == 'privacy':
        call['response_privacy_redactions'] = ['credential']
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
    else:
        call['response']['status'] = 'incomplete'
        refresh(value)
    seal(value)
    for builder in (training_steps.build, training_thinking.build):
        assert not any(row['format'] == 'openrouter.responses' for row in records(builder, value))


def test_cap_eof_shape_check_uses_same_citation_rules_without_repairing_output():
    value = cited_example(truncated=True)
    before = deepcopy(value)
    assert training_thinking.verified_cap_truncated_target(value, value['provider_calls'][0])
    for builder in (training_steps.build, training_thinking.build):
        assert len(records(builder, value)) == 2
    assert value == before
    annotation(value)['uncited'] = 'true'
    refresh(value)
    assert not training_thinking.verified_cap_truncated_target(value, value['provider_calls'][0])


def test_metadata_free_output_stays_exact_and_is_not_mutated():
    value = cited_example()
    annotation(value).pop('uncited')
    call = value['provider_calls'][0]
    before = deepcopy(call['response'])
    target, _ = training_steps._target(call['request']['outbound']['body'], call['response'],
        'openrouter.responses', allow_reasoning_history=True, allow_tool_error_history=True)
    assert target == call['response']['output'] == before['output']
