"""Pure judge request adapters preserve legacy requests and native evidence."""
from copy import deepcopy
import json

import pytest

from lab_arena import operations
from qualification.scoring import openrouter_options as options


MODEL = 'deepseek/deepseek-v4.1-flash-1017'
PROVIDER = 'nebius/fp8'


@pytest.fixture(autouse=True)
def frozen_environment(monkeypatch):
    monkeypatch.delenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', raising=False)
    monkeypatch.delenv('ARENA_JUDGE_PROVIDER_ROUTES', raising=False)


def enable(monkeypatch):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY)
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: PROVIDER}))


def request():
    return {'model': MODEL, 'max_tokens': 700, 'temperature': 0,
            'messages': [{'role': 'system', 'content': 'Check the source before deciding.'},
                         {'role': 'user', 'content': 'Does this firm make valves?'}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'verdict',
                'strict': True, 'schema': {'type': 'object', 'properties': {'match': {'type': 'boolean'}},
                                         'required': ['match'], 'additionalProperties': False}}}}


def native():
    return {'id': 'resp-1', 'model': MODEL, 'provider': 'Nebius', 'status': 'completed',
            'output': [
                {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'Inspect the evidence.'}]},
                {'type': 'message', 'role': 'assistant', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': 'I will check.'}]},
                {'type': 'openrouter:web_search', 'status': 'completed', 'id': 'search-1',
                 'action': {'type': 'search', 'query': 'valve manufacturer'}},
                {'type': 'message', 'role': 'assistant', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': '{"match":true}',
                              'annotations': [{'type': 'url_citation', 'url': 'https://example.com',
                                               'title': 'Factory', 'content': 'We make valves.'}]}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 50, 'cost': 0.001,
                      'output_tokens_details': {'reasoning_tokens': 30}}}


@pytest.mark.parametrize('search', [False, True])
def test_legacy_body_response_and_tool_replay_are_byte_equivalent(search):
    payload = request()
    payload['reasoning'] = {'effort': 'low', 'exclude': True}
    original = deepcopy(payload)
    path, sent = options.prepare_judge_request(payload, hosted_search=search)
    assert path == '/chat/completions' and sent == original == payload
    response = {'choices': [{'message': {'content': 'old'}, 'finish_reason': 'stop'}]}
    assert options.project_judge_response(response, hosted_search=search) == response
    replay = {'role': 'assistant', 'tool_calls': []}
    assert options.preserve_judge_reasoning({'reasoning_content': 'Do not add this'}, replay) == replay


@pytest.mark.parametrize('search', [False, True])
def test_future_request_uses_only_frozen_route_and_real_arena_validator(monkeypatch, search):
    enable(monkeypatch)
    payload = request()
    original = deepcopy(payload)
    path, body = options.prepare_judge_request(payload, hosted_search=search)
    assert payload == original
    assert body['provider'] == {'only': [PROVIDER]}
    assert body['reasoning']['effort'] == 'medium'
    assert body['max_output_tokens' if search else 'max_tokens'] == 8192
    operation, normalized = operations.match_request('POST', 'https://openrouter.ai/api/v1' + path,
        json.dumps(body).encode(), {'content-type': 'application/json'}, allow_full_thinking=True)
    assert operation == ('openrouter.responses' if search else 'openrouter.chat')
    if search:
        assert body['tools'] == [{'type': 'openrouter:web_search', 'parameters': {
            'engine': 'exa', 'max_uses': 1, 'max_total_results': 5}}]
        assert body['text']['format'] == {'type': 'json_schema', **payload['response_format']['json_schema']}
        assert body['input'] == payload['messages']
        assert body['max_tool_calls'] == 1
        assert 'exclude' not in body['reasoning']
    else:
        assert body['reasoning']['exclude'] is False
        assert body['messages'] == payload['messages']
        assert body['response_format'] == payload['response_format']
    assert normalized['provider'] == {'only': [PROVIDER]}


@pytest.mark.parametrize('routes', [None, '{}', '[]', 'null', '{"bad":',
    json.dumps({MODEL: ''}), json.dumps({MODEL: ['nebius/fp8']}), json.dumps({MODEL: 'bad route'})])
def test_opt_in_cannot_fall_back_to_ambient_or_unpinned_provider(monkeypatch, routes):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', options.JUDGE_REASONING_POLICY)
    if routes is not None:
        monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', routes)
    with pytest.raises(ValueError, match='judge_provider_route_unavailable'):
        options.prepare_judge_request(request())


def test_native_projection_keeps_final_json_citations_usage_and_original(monkeypatch):
    enable(monkeypatch)
    response = native()
    before = deepcopy(response)
    projected = options.project_judge_response(response, hosted_search=True)
    assert projected['choices'][0]['message']['content'] == '{"match":true}'
    assert projected['choices'][0]['message']['annotations'] == [{'type': 'url_citation', 'url_citation': {
        'url': 'https://example.com', 'title': 'Factory', 'content': 'We make valves.'}}]
    assert projected['usage'] == response['usage']
    assert response == before
    # The projection is for the existing scoring parser. The raw native
    # response remains the recorder's source, including all thinking/search.
    assert response['output'][0]['content'][0]['text'] == 'Inspect the evidence.'


@pytest.mark.parametrize('damage', ['incomplete', 'error', 'refusal', 'function', 'custom', 'search_missing',
    'search_failed', 'refusal_part', 'bad_text', 'empty', 'no_output'])
def test_native_response_requires_completed_search_and_final_answer(monkeypatch, damage):
    enable(monkeypatch)
    response = native()
    if damage == 'incomplete': response['status'] = 'incomplete'
    elif damage == 'error': response['error'] = {'code': 429}
    elif damage == 'refusal': response['refusal'] = 'Cannot do it'
    elif damage == 'function': response['output'].append({'type': 'function_call', 'name': 'read'})
    elif damage == 'custom': response['output'].append({'type': 'custom_tool_call', 'name': 'read'})
    elif damage == 'search_missing': response['output'].pop(2)
    elif damage == 'search_failed': response['output'][2]['status'] = 'failed'
    elif damage == 'refusal_part': response['output'][-1]['content'] = [{'type': 'refusal', 'refusal': 'No'}]
    elif damage == 'bad_text': response['output'][-1]['content'][0]['text'] = None
    elif damage == 'empty': response['output'][-1]['content'][0]['text'] = ''
    else: response = {'choices': [{'message': {'content': 'No native search proof'}}]}
    with pytest.raises(ValueError, match='judge_'):
        options.project_judge_response(response, hosted_search=True)


def test_tool_history_preserves_native_thinking_only_under_frozen_flag(monkeypatch):
    message = {'reasoning_content': 'Select a reliable source.', 'reasoning': None,
               'reasoning_details': [{'type': 'reasoning.text', 'text': 'Check the page.', 'index': 0}]}
    replay = {'role': 'assistant', 'tool_calls': [{'id': 'tool-1', 'type': 'function',
              'function': {'name': 'read', 'arguments': '{"url":"https://example.com"}'}}]}
    enable(monkeypatch)
    result = options.preserve_judge_reasoning(message, replay)
    assert result['tool_calls'] == replay['tool_calls']
    assert result['reasoning_content'] == message['reasoning_content']
    assert result['reasoning_details'] == message['reasoning_details']
    assert 'reasoning' not in result
    result['reasoning_details'][0]['text'] = 'Changed in the caller'
    assert message['reasoning_details'][0]['text'] == 'Check the page.'
    assert 'reasoning_content' not in replay


def test_native_projection_preserves_nested_and_unknown_annotations(monkeypatch):
    enable(monkeypatch)
    response = native()
    annotations = [{'type': 'url_citation', 'url_citation': {'url': 'https://example.com',
                    'title': 'Factory', 'start_index': 0, 'end_index': 14}},
                   {'type': 'provider_metadata', 'value': 'Keep this'}]
    response['output'][-1]['content'][0]['annotations'] = deepcopy(annotations)
    projected = options.project_judge_response(response, hosted_search=True)
    assert projected['choices'][0]['message']['annotations'] == annotations
    response['output'][-1]['content'][0]['annotations'] = [{'type': 'url_citation', 'title': 'No URL'}]
    with pytest.raises(ValueError, match='judge_citation_malformed'):
        options.project_judge_response(response, hosted_search=True)


def test_search_conversion_keeps_tools_results_and_schema(monkeypatch):
    enable(monkeypatch)
    payload = request()
    payload['messages'] += [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'id': 'call-1',
            'function': {'name': 'read', 'arguments': '{"url":"https://example.com"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Saved evidence'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Use this source evidence.'}]}]
    payload['tools'] = [{'type': 'function', 'function': {'name': 'read', 'description': 'Read one source',
        'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}}}}]
    _, body = options.prepare_judge_request(payload, hosted_search=True)
    assert body['input'][2]['type'] == 'function_call'
    assert body['input'][3] == {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'Saved evidence'}
    assert body['input'][4]['content'] == [{'type': 'input_text', 'text': 'Use this source evidence.'}]
    operations.validate_operation_request('openrouter.responses', body, allow_full_thinking=True)


def test_search_conversion_does_not_silently_drop_chat_thinking_history(monkeypatch):
    enable(monkeypatch)
    payload = request()
    payload['messages'].append({'role': 'assistant', 'content': 'Earlier', 'reasoning_content': 'Earlier reasoning'})
    with pytest.raises(ValueError, match='judge_messages_unsupported'):
        options.prepare_judge_request(payload, hosted_search=True)


@pytest.mark.parametrize('effort,limit', [('low', 8192), ('high', 8192), ('high', 16384), ('medium', 32768)])
@pytest.mark.parametrize('search', [False, True])
def test_stage_controls_remain_explicit_and_frozen_policy_exposes_reasoning(monkeypatch, effort, limit, search):
    enable(monkeypatch)
    payload = request()
    payload.update(reasoning={'effort': effort, 'exclude': True}, max_tokens=limit, include_reasoning=False)
    _, body = options.prepare_judge_request(payload, hosted_search=search)
    assert body['reasoning']['effort'] == effort
    assert body['max_output_tokens' if search else 'max_tokens'] == limit
    if not search:
        assert body['reasoning']['exclude'] is False and body['include_reasoning'] is True


@pytest.mark.parametrize('limit', [0, -1, True, '8192', 32769])
def test_invalid_output_limits_fail_before_dispatch(monkeypatch, limit):
    enable(monkeypatch)
    payload = request()
    payload['max_tokens'] = limit
    with pytest.raises(ValueError, match='judge_output_limit_invalid'):
        options.prepare_judge_request(payload)
