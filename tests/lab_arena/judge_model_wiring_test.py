"""Frozen judge model wiring, native search, and thinking replay regressions."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring, operations
from qualification.scoring import openrouter_options as opts

MODEL = 'deepseek/deepseek-v4.1-flash-1017'
ROUTE = 'example/verified'
ROLES = tuple(scoring.DEFAULT_JUDGE_MODELS)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('This audit must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', opts.JUDGE_REASONING_POLICY)
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: ROUTE}))
    monkeypatch.setenv('ARENA_COMPANY_REVERIFY_MODEL', MODEL)


def chat(answer=None, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {
        'role': 'assistant', 'content': json.dumps(answer if answer is not None else {'ok': True}),
        'reasoning': 'Saved native thinking.'}}]}


def native(answer=None):
    return {'id': 'response-test', 'status': 'completed', 'model': MODEL,
        'output': [{'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'Saved thinking.'}]},
                   {'type': 'web_search_call', 'status': 'completed', 'action': {'type': 'search', 'query': 'company'}},
                   {'type': 'message', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'output_text', 'text': json.dumps(answer or {'ok': True}),
                     'annotations': [{'type': 'url_citation', 'url': 'https://example.com/product',
                                      'title': 'Product', 'start_index': 0, 'end_index': 1}]}]}]}


class Http:
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []
    async def post(self, url, **kwargs):
        self.calls.append((url, copy.deepcopy(kwargs['json'])))
        document = self.documents.pop(0)
        return SimpleNamespace(status_code=200, text=json.dumps(document), json=lambda: document,
                               raise_for_status=lambda: None)
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass


def check_call(call, search=False):
    url, payload = call
    assert url.endswith('/responses' if search else '/chat/completions')
    assert payload['model'] == MODEL
    assert payload['provider'] == {'only': [ROUTE]}
    assert payload['reasoning']['effort'] in {'low', 'medium', 'high'}
    assert payload.get('max_output_tokens', payload.get('max_tokens')) >= 8192
    if search:
        assert 'messages' not in payload and payload['input']
        assert payload['max_tool_calls'] == 1
        assert payload['tools'][-1] == {'type': 'openrouter:web_search', 'parameters': {
            'engine': 'exa', 'max_uses': 1, 'max_total_results': 5}}
    else:
        assert payload['reasoning']['exclude'] is False


def test_complete_frozen_policy_binds_all_seven_roles_and_helpers():
    models = {role: MODEL + '-' + str(i) for i, role in enumerate(ROLES)}
    policy = scoring.build_scorer_policy(judge_models=models,
        judge_provider_routes={model: ROUTE for model in models.values()})
    env = {}
    scoring.apply_policy_to_environment(policy, environ=env,
        credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    expected = {'ARENA_COMPANY_REVERIFY_MODEL': 'company_fit_reverification',
        'INTENT_THREE_STAGE_S1_MODEL': 'company_fit_reverification',
        'INTENT_THREE_STAGE_S3_MODEL': 'intent_three_stage_stage3',
        'ARENA_INTENT_SIGNAL_MODEL': 'intent_signal_judge',
        'ARENA_POSITIVE_SEMANTIC_REVIEW_MODEL': 'intent_fetched_evidence',
        'ARENA_INTENT_EVIDENCE_MODEL': 'intent_fetched_evidence',
        'ARENA_INTENT_DETAILS_REVIEW_MODEL': 'intent_details_review',
        'ARENA_ROLE_BATCH_MODEL': 'role_batch_check',
        'ARENA_INDUSTRY_EVIDENCE_MODEL': 'industry_evidence_review'}
    assert len(models) == 7
    assert {key: env[key] for key in expected} == {key: models[role] for key, role in expected.items()}
    assert env['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] == opts.JUDGE_REASONING_POLICY


@pytest.mark.parametrize('missing', ROLES)
def test_new_full_thinking_policy_cannot_omit_role(missing):
    models = {role: MODEL for role in ROLES if role != missing}
    with pytest.raises(contracts.ArenaContractError):
        scoring.build_scorer_policy(judge_models=models, judge_provider_routes={MODEL: ROUTE})


def test_legacy_policy_cannot_inherit_ambient_capture():
    policy = scoring.build_scorer_policy(judge_provider_routes={})
    env = {'ARENA_JUDGE_REASONING_CAPTURE_POLICY': opts.JUDGE_REASONING_POLICY,
           'ARENA_JUDGE_PROVIDER_ROUTES': json.dumps({MODEL: ROUTE})}
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ=env,
            credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    env = {}
    scoring.apply_policy_to_environment(policy, environ=env,
        credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    assert env['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] == env['ARENA_JUDGE_PROVIDER_ROUTES'] == ''


def test_legacy_request_body_and_response_are_unchanged(monkeypatch):
    monkeypatch.delenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', raising=False)
    payload = {'model': 'perplexity/sonar', 'messages': [{'role': 'user', 'content': 'Find it'}],
               'max_tokens': 700, 'temperature': 0}
    assert opts.prepare_judge_request(payload, hosted_search=True) == ('/chat/completions', payload)
    response = chat()
    assert opts.project_judge_response(response, hosted_search=True) == response


def test_search_conversion_preserves_schema_and_original_body(enabled):
    payload = {'model': MODEL, 'messages': [{'role': 'system', 'content': 'Rules'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Source'}]}], 'max_tokens': 900,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'decision', 'strict': True,
         'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'],
                    'additionalProperties': False}}}}
    saved = copy.deepcopy(payload)
    path, body = opts.prepare_judge_request(payload, hosted_search=True)
    check_call(('https://openrouter.ai/api/v1' + path, body), search=True)
    assert payload == saved
    assert body['text']['format'] == {'type': 'json_schema', **payload['response_format']['json_schema']}
    operations.validate_operation_request('openrouter.responses', body, allow_full_thinking=True)


@pytest.mark.parametrize('failure', ['no_search', 'incomplete_search', 'incomplete_answer', 'refusal', 'unfinished_tool'])
def test_search_requires_completed_search_and_answer(enabled, failure):
    response = native()
    if failure == 'no_search':
        response['output'].pop(1)
    elif failure == 'incomplete_search':
        response['output'][1]['status'] = 'in_progress'
    elif failure == 'incomplete_answer':
        response['status'] = 'incomplete'
    elif failure == 'refusal':
        response['output'][-1]['content'] = [{'type': 'refusal', 'refusal': 'No'}]
    else:
        response['output'].append({'type': 'function_call', 'name': 'foo', 'arguments': '{}'})
    with pytest.raises(ValueError):
        opts.project_judge_response(response, hosted_search=True)


def test_search_projection_preserves_raw_events_and_converts_citations(enabled):
    response = native()
    original = copy.deepcopy(response)
    result = opts.project_judge_response(response, hosted_search=True)
    assert response == original
    assert result['choices'][0]['message']['annotations'][0]['url_citation']['url'] == 'https://example.com/product'
    assert result['choices'][0]['finish_reason'] == 'stop'


@pytest.mark.parametrize('search', [False, True])
def test_three_stage_helper_uses_frozen_route_and_native_search(enabled, monkeypatch, search):
    from qualification.scoring import intent_verification_three_stage as mod
    monkeypatch.setattr(mod, '_get_openrouter_key', lambda: 'test-not-secret')
    client = Http([native() if search else chat()])
    result = asyncio.run(mod._call_openrouter(client, MODEL, 'Find source', hosted_search=search, max_attempts=1))
    assert result['answer'] == {'ok': True}
    check_call(client.calls[0], search)


def test_three_stage_native_correction_updates_input_without_chat_fields(enabled, monkeypatch):
    from qualification.scoring import intent_verification_three_stage as mod
    monkeypatch.setattr(mod, '_get_openrouter_key', lambda: 'test-not-secret')
    async def no_sleep(*args): pass
    monkeypatch.setattr(mod.asyncio, 'sleep', no_sleep)
    client = Http([native({'signal_evaluations': [{'signal_status': 'wrong_entity', 'same_entity_check': 'pass'}]}), native()])
    result = asyncio.run(mod._call_openrouter(client, MODEL, 'Find source', hosted_search=True, max_attempts=2))
    assert 'answer' in result
    second = client.calls[1][1]
    assert 'messages' not in second
    assert mod._STRUCTURED_VERDICT_CORRECTION in ''.join(item.get('content', '') for item in second['input'])
    operations.validate_operation_request('openrouter.responses', second, allow_full_thinking=True)


def test_three_stage_truncated_json_is_not_a_verdict(enabled, monkeypatch):
    from qualification.scoring import intent_verification_three_stage as mod
    monkeypatch.setattr(mod, '_get_openrouter_key', lambda: 'test-not-secret')
    client = Http([chat(finish='length')])
    result = asyncio.run(mod._call_openrouter(client, MODEL, 'Find source', max_attempts=1))
    assert 'answer' not in result and result.get('_error')


def test_intent_gate_truncated_json_is_not_a_verdict(enabled):
    from qualification.scoring import intent_signal_gate as mod
    client = Http([chat(finish='length')])
    answer, error, retry = asyncio.run(mod._call_openrouter_once(
        {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Audit'}]}, 'test-not-secret', client=client))
    check_call(client.calls[0])
    assert answer is None and error


def test_research_and_industry_wrapper_use_route(enabled, monkeypatch):
    from qualification.scoring import company_evidence_investigator as mod
    calls = []
    async def post(session, url, **kwargs):
        calls.append((url, kwargs['payload']))
        return 200, chat()
    monkeypatch.setattr(mod, '_post_json', post)
    result = asyncio.run(mod._post_openrouter_json(None, 'unused', headers={}, started=time.monotonic(),
        payload={'model': MODEL, 'messages': [{'role': 'user', 'content': 'Research'}]}))
    assert result[0] == 200
    check_call(calls[0])


def test_locator_search_preserves_domain_boundary(enabled, monkeypatch):
    from qualification.scoring import company_evidence_investigator as mod
    calls = []
    response = native()
    response['output'][-1]['content'][0]['text'] = 'https://evil.example.net/fake fabricated claim'
    async def post(session, url, **kwargs):
        calls.append((url, kwargs['payload']))
        return 200, response
    monkeypatch.setattr(mod, '_post_json', post)
    result = asyncio.run(mod._search_industry_locators(None, 'product', key='test-not-secret',
        company_locator={'name': 'Example'}, include_domains=['example.com']))
    check_call(calls[0], True)
    assert [row['url'] for row in result['results']] == ['https://example.com/product']
    assert result['notice'] == 'discovery_only_not_evidence'


def test_company_reverify_uses_native_search(enabled, monkeypatch):
    from qualification.scoring import lead_scorer as mod
    monkeypatch.setattr(mod, '_SCORER_REVERIFY_MODEL', MODEL)
    calls = []
    class Response:
        status = 200
        async def json(self): return native()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def post(self, url, **kwargs):
            calls.append((url, kwargs['json']))
            return Response()
    monkeypatch.setattr(mod.aiohttp, 'ClientSession', lambda **kwargs: Session())
    answer, error = asyncio.run(mod._request_company_reverify_json(key='test-not-secret', prompt='Verify', telemetry_purpose='test'))
    assert answer == {'ok': True} and not error
    check_call(calls[0], True)


def test_role_batch_uses_frozen_route(enabled, monkeypatch):
    from qualification.scoring import role_batch_check as mod
    monkeypatch.setattr(mod, 'MODEL', MODEL)
    client = Http([chat([{'id': 1, 'match': True, 'reason': 'Explicit role'}])])
    result = asyncio.run(mod._judge_chunk(client, 'test-not-secret', ['Engineer'], [{'id': 1, 'role': 'Engineer'}]))
    assert result[0]['match'] is True
    check_call(client.calls[0])


@pytest.mark.parametrize('quote,finish,expected', [
    ('Owns product roadmap', 'stop', 'VERIFIED'),
    ('Invented duties', 'stop', 'UNPROVEN'),
    ('Owns product roadmap', 'length', None),
])
def test_role_followup_keeps_evidence_and_completion_gates(enabled, monkeypatch, quote, finish, expected):
    from qualification.scoring import role_batch_check as mod
    monkeypatch.setattr(mod, 'MODEL', MODEL)
    client = Http([chat({'status': 'VERIFIED', 'target_role': 'Product Manager',
        'evidence_quote': quote, 'reason': 'Product ownership'}, finish=finish)])
    result = asyncio.run(mod.review_role_evidence(client, 'test-not-secret', actual_role='Product Manager',
        target_roles=['Product Manager'], target_seniority='', duties='Owns product roadmap and requirements.'))
    assert (result['status'] if result else None) == expected
    check_call(client.calls[0])


def test_shared_paragraph_details_helper_uses_frozen_route(enabled, monkeypatch):
    from qualification.scoring import verification_helpers as mod
    client = Http([chat()])
    monkeypatch.setattr(mod.httpx, 'AsyncClient', lambda **kwargs: client)
    result = asyncio.run(mod.openrouter_chat('Review', MODEL, api_key='test-not-secret'))
    assert json.loads(result) == {'ok': True}
    check_call(client.calls[0])


def test_schema_wire_restores_facts_before_decision_for_open_weight_model():
    required = ['observed_activity', 'reason', 'missing_requirements', 'role', 'evidence_span_ids', 'decision']
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Review'}], 'max_tokens': 8192,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'industry_proposal', 'strict': True,
          'schema': {'type': 'object', 'properties': {key: {'type': 'string'} for key in sorted(required)},
                     'required': required, 'additionalProperties': False}}}}
    route = {'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False,
             'max_price': {'prompt': 1, 'completion': 1, 'request': 0},
             'only': [ROUTE], 'require_parameters': True}
    request = operations.build_outbound_request('openrouter.chat', payload,
        openrouter_provider_policy=route, allow_full_thinking=True, industry_schema_order=True)
    body = json.loads(request.body)
    assert list(body['response_format']['json_schema']['schema']['properties']) == required
    assert payload['response_format']['json_schema']['schema']['properties'].keys() == set(required)


@pytest.mark.parametrize('enabled_capture', [False, True])
def test_research_tool_history_keeps_native_thinking_without_inventing_it(monkeypatch, enabled_capture):
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY',
                       opts.JUDGE_REASONING_POLICY if enabled_capture else '')
    provider = {'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {
            'name': 'fetch_page', 'arguments': '{"url":"https://example.com/product"}'}}],
        'reasoning': 'Read the exact company source before making a finding.',
        'reasoning_content': 'Additional native thinking field.',
        'reasoning_details': [{'type': 'reasoning.text', 'text': 'Keep this exact text.',
                               'id': 'reasoning-1', 'format': 'unknown', 'index': 0}]}
    saved = copy.deepcopy(provider)
    replay = {'role': 'assistant', 'tool_calls': copy.deepcopy(provider['tool_calls'])}
    result = opts.preserve_judge_reasoning(provider, replay)
    assert provider == saved
    assert result['tool_calls'] == provider['tool_calls']
    for key in ('reasoning', 'reasoning_content', 'reasoning_details'):
        if enabled_capture:
            assert result[key] == provider[key]
        else:
            assert key not in result
    if enabled_capture:
        operations.validate_operation_request('openrouter.chat', {'model': MODEL,
            'messages': [{'role': 'user', 'content': 'Find source'}, result], 'max_tokens': 8192},
            allow_full_thinking=True)
        result['reasoning_details'][0]['text'] = 'changed only in local replay'
        assert provider == saved
    assert opts.preserve_judge_reasoning({'role': 'assistant', 'content': 'No visible thinking'},
        {'role': 'assistant', 'content': 'No visible thinking'}) == {
            'role': 'assistant', 'content': 'No visible thinking'}


def test_default_new_round_policy_freezes_exact_current_model_and_route_defaults():
    policy = scoring.build_scorer_policy()
    assert policy['judge_models'] == scoring.DEFAULT_JUDGE_MODELS
    assert set(policy['judge_models']) == {
        'company_fit_reverification', 'intent_signal_judge', 'intent_details_review',
        'intent_three_stage_stage3', 'intent_fetched_evidence', 'role_batch_check',
        'industry_evidence_review'}
    routes = scoring.DEFAULT_JUDGE_PROVIDER_ROUTES
    if routes:
        assert set(routes) == set(policy['judge_models'].values())
        assert json.loads(policy['env_bindings']['ARENA_JUDGE_PROVIDER_ROUTES']) == routes
        assert policy['env_bindings']['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] == opts.JUDGE_REASONING_POLICY
    else:
        assert 'ARENA_JUDGE_REASONING_CAPTURE_POLICY' not in policy['env_bindings']
        assert 'ARENA_JUDGE_PROVIDER_ROUTES' not in policy['env_bindings']


def test_explicit_final_answer_precedes_trailing_commentary(enabled):
    response = native({'decision': 'approve'})
    response['output'][-1]['phase'] = 'final_answer'
    response['output'].append({'type': 'message', 'role': 'assistant', 'status': 'completed',
        'phase': 'commentary', 'content': [{'type': 'output_text', 'text': 'I checked the page.'}]})
    original = copy.deepcopy(response)
    projected = opts.project_judge_response(response, hosted_search=True)
    assert json.loads(projected['choices'][0]['message']['content']) == {'decision': 'approve'}
    assert response == original


@pytest.mark.parametrize('phases', [
    ['final_answer', 'final_answer'], ['unexpected'], ['commentary'],
])
def test_native_projection_rejects_ambiguous_or_missing_final_answer(enabled, phases):
    response = native()
    answer = response['output'].pop()
    for phase in phases:
        response['output'].append({**copy.deepcopy(answer), 'phase': phase})
    with pytest.raises(ValueError):
        opts.project_judge_response(response, hosted_search=True)
