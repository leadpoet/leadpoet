"""Candidate industry final route; historical Sol workflows stay frozen."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import time

import pytest

from lab_arena import broker, contracts, reasoning_capture, scoring
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import industry_classification, openrouter_options
from tests.lab_arena import test_lab_arena_broker as bt

MISTRAL = 'mistralai/mistral-large-4-0'
SOL = 'openai/gpt-6.1-sol'
CONTROLLER = 'industry_research_controller'
FINAL = 'industry_evidence_review'
OLD_MODELS = {
    'company_fit_reverification': 'deepseek/deepseek-v4.1-flash-1017',
    'intent_signal_judge': 'openai/gpt-oss-120b',
    'intent_details_review': 'deepseek/deepseek-v4.1-flash-1017',
    'intent_three_stage_stage3': 'deepseek/deepseek-v4.1-flash-1017',
    'intent_fetched_evidence': 'openai/gpt-oss-120b',
    'role_batch_check': 'openai/gpt-oss-120b', FINAL: SOL, CONTROLLER: SOL,
}
OLD_ROUTES = {'deepseek/deepseek-v4.1-flash-1017': 'nebius/fp8', 'openai/gpt-oss-120b': 'near-ai/fp4'}
OLD_OPTIONS = {
    'deepseek/deepseek-v4.1-flash-1017': {'reasoning': {'effort': 'low'}, 'max_output_tokens': 8192},
    'openai/gpt-oss-120b': {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096},
}


def historical_policy(*, controller=True, capture=True):
    models = dict(OLD_MODELS)
    if not controller:
        models.pop(CONTROLLER)
    roles = [FINAL, CONTROLLER] if controller else [FINAL]
    return scoring.build_scorer_policy(judge_models=models,
        judge_provider_routes=OLD_ROUTES if capture else {},
        judge_request_options=OLD_OPTIONS if capture else None,
        judge_reasoning_exceptions={SOL: {'roles': roles, 'reason': 'quality_gate_not_met'}} if capture else None)


def test_candidate_changes_only_final_role_and_its_frozen_controls():
    old = historical_policy()
    candidate = scoring.build_scorer_policy()
    models = {**OLD_MODELS, FINAL: MISTRAL}
    assert candidate['judge_models'] == models
    assert contracts.judge_reasoning_exceptions(candidate) == {
        SOL: {'roles': [CONTROLLER], 'reason': 'quality_gate_not_met'}}
    bindings = candidate['env_bindings']
    assert json.loads(bindings[contracts.JUDGE_PROVIDER_ROUTES_BINDING]) == {**OLD_ROUTES, MISTRAL: 'mistral/zdr'}
    assert json.loads(bindings[contracts.JUDGE_REQUEST_OPTIONS_BINDING]) == {
        **OLD_OPTIONS, MISTRAL: {'reasoning': {'effort': 'high'}, 'max_output_tokens': 16384}}
    assert {k: v for k, v in candidate.items() if k not in {'judge_models', 'env_bindings'}} == {
        k: v for k, v in old.items() if k not in {'judge_models', 'env_bindings'}}
    assert contracts.validate_scorer_policy(json.loads(json.dumps(candidate))) == candidate


@pytest.mark.parametrize('controller,capture,digest', [
    (True, True, '0148cfb34001c5162e20f81a0ecfc72170a1eee2560dc3592ecd16f109692595'),
    (False, True, '4edd7bc144d25877bfbe9f8f4bbaad7fae88e8d2c3434736b83d70d1f01c9ed0'),
    (True, False, '84d2f2fd42e670e2b49de015e0d254ab289cf3aab8b6ee882be5974710dc0f16'),
    (False, False, '87ff87ef78b91a0d358bdcf4fe7fd592261908166ea41cda00e696b34b3f99a7'),
])
def test_historical_policy_bytes_and_model_bindings_remain_exact(controller, capture, digest):
    policy = historical_policy(controller=controller, capture=capture)
    assert contracts.document_hash(policy) == 'sha256:' + digest
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment,
        credentials={name: 'offline-placeholder' for name in scoring.CREDENTIAL_ENV_NAMES})
    assert environment['ARENA_INDUSTRY_EVIDENCE_MODEL'] == SOL
    assert environment['ARENA_INDUSTRY_RESEARCH_CONTROLLER_MODEL'] == (SOL if controller else '')


def test_controller_only_exception_is_exact_and_does_not_need_mistral_responses():
    models = {**OLD_MODELS, FINAL: MISTRAL}
    raw = json.dumps({SOL: {'roles': [CONTROLLER], 'reason': 'quality_gate_not_met'}})
    assert contracts.validate_judge_reasoning_exceptions(raw, models)[SOL]['roles'] == [CONTROLLER]
    bad = {**models, 'intent_details_review': SOL}
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_judge_reasoning_exceptions(raw, bad)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_judge_reasoning_exceptions(raw, OLD_MODELS)
    registry = reasoning_capture.load_registry()
    registry['approved_routes'] = [row for row in registry['approved_routes']
        if (row['model_id'], row['api']) != (MISTRAL, 'openrouter.responses')]
    assert reasoning_capture.validate_judge_registry(registry, scoring.build_scorer_policy()) == registry


def test_saved_eight_role_sol_round_is_not_rebound_when_candidate_defaults_change():
    from tests.lab_arena.judge_policy_integration_test import NOW, service
    app = service()
    app._config.reasoning_registry_source = reasoning_capture.load_registry
    app._scorer_policy = historical_policy()
    old = app.create_round(NOW, round_id='arena-2026-11-06-oldsol')
    before = contracts.canonical_json(old)
    app._scorer_policy = scoring.build_scorer_policy()
    assert contracts.canonical_json(app.create_round(NOW, round_id=old['round_id'])) == before
    future = app.create_round(NOW + timedelta(days=1), round_id='arena-2026-11-07-candidate')
    assert future['scorer_policy']['judge_models'][FINAL] == MISTRAL
    assert future['scorer_policy']['judge_models'][CONTROLLER] == SOL
    assert contracts.canonical_json(app.create_round(NOW, round_id=old['round_id'])) == before


def candidate_broker(response):
    policy = scoring.build_scorer_policy()
    registry = reasoning_capture.load_registry()
    prices = bt.price_table()
    price = {component: '0' for component in broker.PRICED_COMPONENTS}
    for model in set(policy['judge_models'].values()):
        prices['models'][model] = dict(price)
    prices['models'][MISTRAL] = {**price, 'prompt': '0.00000136', 'completion': '0.00000418'}
    store, transport = bt.FakeLedgerStore(), bt.FakeTransport([(200, response)])
    instance = broker.Broker(store=store, transport=transport, price_table=prices,
        key_for=lambda provider: bt.HOST_KEYS[provider], judge_models=policy['judge_models'])
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=reasoning_capture.POLICY,
        judge_reasoning_route_registry=registry, judgment_cache_scope={'scorer_policy': policy})
    return instance, context, store, transport


def bind_environment(monkeypatch):
    environment = {}
    scoring.apply_policy_to_environment(scoring.build_scorer_policy(), environ=environment,
        credentials={name: 'offline-placeholder' for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def test_actual_native_final_review_and_broker_keep_native_packet_and_exact_route(monkeypatch):
    bind_environment(monkeypatch)
    url = 'https://coreworks.example/services'
    text = 'Coreworks supplies managed infrastructure services to enterprise customers.'
    packet = industry_classification.prepare({'industry': 'IT services', 'subindustry': '',
        'product_service': 'Managed infrastructure services'}, {'name': 'Coreworks', 'website': url},
        [{'source_id': 's0', 'source_url': url, 'body': text}], evaluation_date='2026-11-05')
    span = next(iter(packet.spans()))
    response = {'model': MISTRAL, 'provider': 'Mistral', 'usage': {'cost': '0.001'}, 'choices': [{
        'finish_reason': 'stop', 'message': {'role': 'assistant', 'reasoning': 'Inspect the supplied source.',
        'content': json.dumps({'observed_activity': 'Managed infrastructure services', 'reason': 'The supplied source states the offering.',
            'missing_requirements': [], 'role': 'supplier_operator', 'evidence_span_ids': [span], 'decision': 'match'})}}]}
    instance, context, _, transport = candidate_broker(response)
    sent = []
    async def post(_session, endpoint, *, headers, payload):
        sent.append(deepcopy(payload))
        result = instance.execute(context, operation_id='openrouter.chat', parameters=payload,
                                  action_sequence=0, timeout_ms=30000)
        return result.status, json.loads(result.body)
    monkeypatch.setattr(investigator, '_post_json', post)
    finding, failure, count = asyncio.run(investigator._review_industry_sources(None,
        model=MISTRAL, openrouter_key='offline-placeholder', started=time.monotonic(),
        company_locator={'name': 'Coreworks', 'website': url},
        criteria={'industry': 'IT services', 'subindustry': '', 'product_service': 'Managed infrastructure services'},
        evaluation_date_value='2026-11-05', fetched_pages={url: text}, fetched_final_urls={url: url},
        first_party_domains={'coreworks.example'}, identity_anchor={'submitted_domain': 'coreworks.example', 'observed_domain': 'coreworks.example'},
        identity_names={'coreworks'}))
    assert count == len(sent) == len(transport.sent) == 1 and not failure
    assert finding['status'] == 'VERIFIED'  # Fixture protocol result, not quality gold.
    wire = json.loads(transport.sent[0]['body'])
    original = industry_classification.request_payload(packet, model=MISTRAL)
    original['messages'][0]['content'] += industry_classification.REASONING_RESPONSE_CONTRACT
    assert wire['messages'] == original['messages']
    assert wire['response_format'] == original['response_format']
    assert wire['max_tokens'] == 16384 and wire['reasoning'] == {'effort': 'high', 'exclude': False}
    assert wire['provider'] == {'only': ['mistral/zdr'], 'allow_fallbacks': False,
        'require_parameters': True, 'data_collection': 'deny', 'zdr': True,
        'max_price': {'prompt': 1.36, 'completion': 4.18, 'request': 0}}
    assert openrouter_options.judge_reasoning_enabled(SOL) is False
    controller = {'model': SOL, 'messages': [{'role': 'user', 'content': 'Research source.'}],
                  'max_tokens': 3000, 'reasoning': {'effort': 'low'}, 'parallel_tool_calls': False}
    assert openrouter_options.prepare_judge_request(controller) == ('/chat/completions', controller)


def test_sol_controller_keeps_actual_broker_wire_and_price_ceiling():
    def dispatch(policy):
        prices = bt.price_table()
        for model in set(policy['judge_models'].values()):
            prices['models'][model] = {component: '0' for component in broker.PRICED_COMPONENTS}
        store, transport = bt.FakeLedgerStore(), bt.FakeTransport([(200, {
            'model': SOL, 'choices': [], 'usage': {'cost': '0.001'}})])
        instance = broker.Broker(store=store, transport=transport, price_table=prices,
            key_for=lambda provider: bt.HOST_KEYS[provider], judge_models=policy['judge_models'])
        capture = policy['env_bindings'].get(contracts.JUDGE_REASONING_CAPTURE_BINDING)
        context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=capture,
            judge_reasoning_route_registry=reasoning_capture.load_registry() if capture else None,
            judgment_cache_scope={'scorer_policy': policy})
        body = {'model': SOL, 'messages': [{'role': 'user', 'content': 'Research source.'}],
                'max_tokens': 3000, 'reasoning': {'effort': 'low'}, 'parallel_tool_calls': False}
        result = instance.execute(context, operation_id='openrouter.chat', parameters=body,
                                  action_sequence=0, timeout_ms=30000)
        assert result.status == 200
        return json.loads(transport.sent[0]['body']), result.call['reserved_microusd']
    before = dispatch(historical_policy())
    after = dispatch(scoring.build_scorer_policy())
    assert after == before
    assert after[0]['provider']['only'] == ['azure/us', 'azure/eu']
    assert after[0]['max_tokens'] == 3000 and after[0]['reasoning'] == {'effort': 'low'}
    historical_custom = scoring.build_scorer_policy(judge_provider_routes={})
    no_capture, _ = dispatch(historical_custom)
    assert no_capture['provider'] == {'allow_fallbacks': False, 'data_collection': 'deny', 'zdr': True}


@pytest.mark.parametrize('research,digest', [
    (False, '2d875d22b6ab625d1d26181de8ee7ccd005e60d3312a02b221b9f55a2479a34c'),
    (True, 'ec2bc1370dd82ba563793e1322b63cc720cb47c617fdb930bc4708a96b4d1d5d'),
])
def test_mistral_guidance_is_model_only_and_keeps_historical_sol_bytes(research, digest):
    packet = industry_classification.prepare({'industry': 'IT services', 'subindustry': '',
        'product_service': 'Managed infrastructure services'}, {'name': 'Coreworks', 'website': 'https://coreworks.example/services'},
        [{'source_id': 's0', 'source_url': 'https://coreworks.example/services',
          'body': 'Coreworks supplies managed infrastructure services to enterprise customers.'}], evaluation_date='2026-11-05')
    legacy = industry_classification.request_payload(packet, model=SOL, research=research)
    assert contracts.document_hash(legacy) == 'sha256:' + digest
    candidate = industry_classification.request_payload(packet, model=MISTRAL, research=research)
    assert candidate['messages'][0]['content'] == legacy['messages'][0]['content'] + industry_classification.MISTRAL_CRITERION_GUIDANCE
    assert candidate['messages'][1:] == legacy['messages'][1:]
    assert candidate['response_format'] == legacy['response_format']
    assert packet.digest == industry_classification.prepare(dict(zip(('industry', 'subindustry', 'product_service'), packet.criteria)),
        dict(zip(('name', 'website', 'linkedin'), packet.company)),
        [{'source_id': source_id, 'body': body, 'source_url': url} for source_id, body, url in packet.pages],
        evaluation_date=packet.evaluation_date).digest


@pytest.mark.parametrize('damage', ['other_route', 'over_cap'])
def test_candidate_broker_refuses_unfrozen_route_or_cap_before_billing(monkeypatch, damage):
    bind_environment(monkeypatch)
    instance, context, store, transport = candidate_broker({'choices': [], 'usage': {'cost': '0.001'}})
    _, body = openrouter_options.prepare_judge_request({'model': MISTRAL,
        'messages': [{'role': 'user', 'content': 'Review the supplied source.'}], 'max_tokens': 16384})
    if damage == 'other_route':
        body['provider']['only'] = ['mistral']
    else:
        body['max_tokens'] = 16385
    result = instance.execute(context, operation_id='openrouter.chat', parameters=body,
                              action_sequence=0, timeout_ms=30000)
    assert result.status == 400 and not store.calls and not transport.sent
