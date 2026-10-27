"""Normalize the observed native search replay before hashing, never its evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker, contracts, operations, reasoning_capture
from tests.lab_arena.codex_runtime_test import response
from tests.lab_arena.hosted_search_continuation_test import hosted_request
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


def native_item():
    return {'type': 'web_search_call', 'id': 'search-native', 'status': 'completed',
            'action': {'type': 'search', 'query': 'official fixture product'}}


def normalize(request):
    return broker.normalized_request('openrouter.responses', request,
        reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=_reviewed_registry(api='openrouter.responses'))


def test_observed_eleven_item_shape_uses_provider_type_at_item_nine():
    # Sanitized shape of actual TYCHE03 wire10: only type needed changing in
    # the independent 400/200 A/B. No provider-result fields are restored.
    request = {'model': 'openai/gpt-4o-mini', 'input': [
        *[{'role': 'user', 'content': f'Saved context {i}.'} for i in range(9)],
        native_item(), {'role': 'user', 'content': 'Continue the research.'}]}
    before = deepcopy(request)
    expected = deepcopy(request['input'])
    expected[9]['type'] = 'openrouter:web_search'
    actual = normalize(request)
    assert actual['input'] == expected
    assert request == before
    assert 'sources' not in actual['input'][9]['action']


def test_native_alias_conversion_preserves_order_sources_citations_and_local_results():
    request = hosted_request()
    for item in request['input']:
        if item.get('type') == 'openrouter:web_search':
            item['type'] = 'web_search_call'
    before = deepcopy(request)
    expected = deepcopy(request['input'])
    for item in expected:
        if item.get('type') == 'web_search_call':
            item['type'] = 'openrouter:web_search'
    assert normalize(request)['input'] == expected
    assert request == before


def test_broker_capture_hash_and_outbound_bind_canonical_input_but_keep_admitted_native():
    request = {'model': 'openai/gpt-4o-mini', 'input': [{'role': 'user', 'content': 'Find evidence.'}, native_item()]}
    original = deepcopy(request)
    captured = []
    def capture(document, **_):
        captured.append(deepcopy(document))
        return {'status': 'unavailable'}
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
                      reasoning_route_registry=_reviewed_registry(api='openrouter.responses'))
    instance, store, transport = make_broker(transport=FakeTransport([(200, response())]), capture_request=capture)
    result = instance.execute(context, operation_id='openrouter.responses', parameters=request,
                              action_sequence=0, timeout_ms=300_000)
    assert result.status == 200
    assert len(captured) == len(transport.sent) == 1
    saved = captured[0]
    expected = normalize(original)
    assert saved['admitted_parameters'] == original == request
    assert saved['normalized_parameters']['input'] == original['input']
    assert saved['hashed_parameters'] == saved['effective_parameters'] == expected
    assert saved['request_hash'] == result.call['request_hash'] == contracts.document_hash(expected)
    sent = json.loads(transport.sent[0]['body'])
    assert saved['outbound']['body'] == sent
    assert sent['input'] == expected['input']
    assert sent['input'][-1]['type'] == 'openrouter:web_search'
    assert saved['outbound']['body_sha256'] == contracts.hash_bytes(transport.sent[0]['body'])
    assert store.log == ['reserve', 'dispatch', 'settle']


def test_policy_off_native_search_keeps_historical_hash():
    request = {'model': 'openai/gpt-4o-mini', 'input': [{'role': 'user', 'content': 'Find evidence.'}, native_item()],
               'max_output_tokens': 400}
    actual = broker.normalized_request('openrouter.responses', request)
    assert actual == request
    assert contracts.document_hash(actual) == 'sha256:9053a9e60183d635cdf7f58c531ec2328c562f1ca7f6cd014c7f945c4098383f'


@pytest.mark.parametrize('kind', ['idless', 'open_page', 'find_in_page', 'in_progress', 'empty_query'])
def test_other_previously_supported_native_shapes_stay_unchanged(kind):
    item = native_item()
    if kind == 'idless': item.pop('id')
    elif kind == 'open_page': item['action'] = {'type': 'open_page', 'url': 'https://example.com/source'}
    elif kind == 'find_in_page': item['action'] = {'type': 'find_in_page', 'url': 'https://example.com/source', 'pattern': 'fact'}
    elif kind == 'in_progress': item['status'] = 'in_progress'
    else: item['action'] = {'type': 'search'}
    request = {'model': 'openai/gpt-4o-mini', 'input': [item]}
    assert normalize(request)['input'] == request['input']
    operations.validate_operation_request('openrouter.responses', normalize(request), allow_full_thinking=True)


def test_direct_provider_alias_is_preserved_without_rewriting_metadata():
    request = hosted_request()
    before = deepcopy(request)
    assert normalize(request)['input'] == before['input']
    assert request == before


def test_changed_native_item_has_no_shared_mutable_provider_fields():
    request = {'model': 'openai/gpt-4o-mini', 'input': [native_item()]}
    original = deepcopy(request)
    actual = broker.openrouter_normalized(request, allow_full_thinking=True)
    actual['input'][0]['action']['query'] = 'changed only in the returned copy'
    assert request == original


@pytest.mark.parametrize('explicit_limit', [False, True])
@pytest.mark.parametrize('overflow', [False, True])
def test_full_context_limit_measures_the_canonical_provider_body(monkeypatch, explicit_limit, overflow):
    route = _reviewed_registry(api='openrouter.responses')['approved_routes'][0]
    output_tokens = route['max_output_tokens'] if explicit_limit else operations.OPENROUTER_MAX_OUTPUT_TOKENS
    def tokens(parameters):
        if parameters['input'][0]['type'] == 'openrouter:web_search':
            return route['context_tokens'] - output_tokens + int(overflow)
        return 0
    monkeypatch.setattr(broker, 'bounded_input_tokens', tokens)
    request = {'model': 'openai/gpt-4o-mini', 'input': [native_item()]}
    if explicit_limit:
        request['max_output_tokens'] = output_tokens
    if overflow:
        with pytest.raises(broker.BrokerError, match='invalid_request'):
            normalize(request)
    else:
        assert normalize(request)['max_output_tokens'] == output_tokens


def test_strict_training_binds_canonical_hash_while_retaining_native_admitted_history():
    from lab_arena import training_steps, training_thinking
    from tests.lab_arena.training_cap_eof_test import cap_example
    from tests.lab_arena.training_eligibility_test import refresh_response_links
    from tests.lab_arena.training_steps_test import seal
    value = cap_example('openrouter.responses')
    call = value['provider_calls'][0]
    call['response']['output'][1]['arguments'] = '{}'
    call['response']['usage']['output_tokens'] = 50
    original_body = call['request']['outbound']['body']
    host_provider = deepcopy(original_body['provider'])
    admitted = {key: deepcopy(original_body[key]) for key in ('model', 'input', 'tools', 'max_output_tokens')}
    admitted['provider'] = {'only': host_provider['only']}
    admitted['input'] += [native_item(),
        {'type': 'function_call', 'status': 'completed', 'call_id': 'history-local',
         'name': 'lookup_evidence', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'history-local', 'output': 'Exact local source evidence.'}]
    before = deepcopy(admitted)
    canonical = broker.normalized_request('openrouter.responses', admitted,
        reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=value['configuration']['reasoning_route_registry'])
    request_hash = contracts.document_hash(canonical)
    call['request_hash'] = request_hash
    call['request'].update(request_hash=request_hash, admitted_parameters=admitted,
        normalized_parameters=operations.validate_operation_request('openrouter.responses', admitted, allow_full_thinking=True),
        hashed_parameters=deepcopy(canonical), effective_parameters=deepcopy(canonical))
    outbound = operations.build_outbound_request('openrouter.responses', canonical,
        openrouter_provider_policy=host_provider, allow_full_thinking=True)
    call['request']['outbound']['body'] = json.loads(outbound.body)
    call['request']['outbound']['body_sha256'] = contracts.hash_bytes(outbound.body)
    call['lifecycle'][0]['accounting']['request_hash'] = request_hash
    refresh_response_links(value, 0)
    seal(value)
    candidate = training_steps._candidate(value, call)
    assert candidate['input']['input'] == canonical['input']
    assert candidate['input']['input'][-3]['type'] == 'openrouter:web_search'
    assert candidate['input']['input'][-1]['output'] == 'Exact local source evidence.'
    assert contracts.document_hash(call['request']['hashed_parameters']) == request_hash
    assert call['request']['admitted_parameters'] == before
    strict, _ = training_thinking.build([value], training_steps.Rules(production_rounds_only=False))
    assert any(row['source_refs'][0]['call_identity'] == call['call_identity']
               for split in ('train', 'eval') for row in strict[split])
    call['request']['hashed_parameters']['input'][-3]['type'] = 'web_search_call'
    with pytest.raises(ValueError, match='request_identity_mismatch'):
        training_steps._candidate(value, call)


def test_existing_encoded_request_limit_applies_after_alias_expansion(monkeypatch):
    request = {'model': 'openai/gpt-4o-mini', 'input': [native_item()]}
    reviewed = _reviewed_registry(api='openrouter.responses')
    expanded, _ = reasoning_capture.apply('openrouter.responses', request,
        policy=reasoning_capture.POLICY, registry=reviewed)
    native_bytes = len(contracts.canonical_json(expanded).encode())
    operation = operations.OPERATIONS['openrouter.responses']
    monkeypatch.setattr(operations, 'OPERATIONS', dict(operations.OPERATIONS, **{
        'openrouter.responses': replace(operation, max_request_bytes=native_bytes)}))
    # Initial and reasoning-expanded native input fit. The longer canonical
    # provider type must still obey the same existing encoded request limit.
    operations.validate_operation_request('openrouter.responses', expanded, allow_full_thinking=True)
    with pytest.raises(broker.BrokerError, match='invalid_request'):
        normalize(request)


def test_native_and_provider_alias_retries_share_the_same_canonical_call_identity():
    request = {'model': 'openai/gpt-4o-mini', 'input': [native_item()]}
    direct = deepcopy(request)
    direct['input'][0]['type'] = 'openrouter:web_search'
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
                      reasoning_route_registry=_reviewed_registry(api='openrouter.responses'))
    instance, store, transport = make_broker(transport=FakeTransport([(200, response())]))
    arguments = dict(operation_id='openrouter.responses', action_sequence=0, timeout_ms=300_000)
    first = instance.execute(context, parameters=request, **arguments)
    retry = instance.execute(context, parameters=direct, **arguments)
    assert first.status == retry.status == 200
    assert first.body == retry.body
    assert first.call['request_hash'] == retry.call['request_hash']
    assert first.call['call_identity'] == retry.call['call_identity']
    assert retry.call['idempotent'] is True
    assert len(transport.sent) == store.log.count('dispatch') == store.log.count('settle') == 1
