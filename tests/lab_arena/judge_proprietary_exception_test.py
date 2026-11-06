"""An explicit industry-only exception never grants open routes an escape hatch."""
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
import time

import pytest

from lab_arena import broker as br, contracts, reasoning_capture as rc, scoring, training_thinking
from qualification.scoring import openrouter_options as opts
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.judge_policy_integration_test import NOW, service
from tests.lab_arena.judge_thinking_export_test import judge_example, records
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.training_steps_test import seal

SOL = 'openai/gpt-6.1-sol'
MODEL = 'test/model'
BINDING = 'ARENA_JUDGE_REASONING_EXCEPTIONS'
EXCEPTION = {SOL: {'roles': ['industry_evidence_review'], 'reason': 'quality_gate_not_met'}}


def policy():
    value = scoring.build_scorer_policy(
        judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={MODEL: 'example/verified'})
    value['judge_models']['industry_evidence_review'] = SOL
    value['env_bindings'][BINDING] = json.dumps(EXCEPTION)
    return value


def registry():
    value = _reviewed_registry(model=MODEL)
    value['approved_routes'] += _reviewed_registry(model=MODEL, api='openrouter.responses')['approved_routes']
    return value


def test_exception_round_activation_freezes_reason_and_keeps_existing_round():
    app = service()
    old = app.create_round(NOW, round_id='arena-2026-11-05-old')
    app._scorer_policy = policy()
    app._config.reasoning_registry_source = registry
    assert app.create_round(NOW, round_id='arena-2026-11-05-old') == old
    future = app.create_round(NOW, round_id='arena-2026-11-05-new')
    assert contracts.validate_round_configuration(deepcopy(future)) == future
    assert future['scorer_policy']['env_bindings'][BINDING] == json.dumps(EXCEPTION)
    assert BINDING not in old['scorer_policy']['env_bindings']


@pytest.mark.parametrize('damage', ['open_model', 'other_role', 'shared_sol', 'blank_reason',
                                  'extra_field', 'route_bypass', 'no_marker'])
def test_exception_cannot_broaden_model_role_or_routes(damage):
    value = policy()
    exceptions = deepcopy(EXCEPTION)
    if damage == 'open_model':
        exceptions[MODEL] = exceptions.pop(SOL)
    elif damage == 'other_role':
        exceptions[SOL]['roles'] = ['company_fit_reverification']
    elif damage == 'shared_sol':
        value['judge_models']['role_batch_check'] = SOL
    elif damage == 'blank_reason':
        exceptions[SOL]['reason'] = ' '
    elif damage == 'extra_field':
        exceptions[SOL]['allow_unreviewed_route'] = True
    elif damage == 'route_bypass':
        value['env_bindings']['ARENA_JUDGE_PROVIDER_ROUTES'] = '{}'
    else:
        value['env_bindings'].pop('ARENA_JUDGE_REASONING_CAPTURE_POLICY')
    value['env_bindings'][BINDING] = json.dumps(exceptions)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_scorer_policy(value)


def test_exception_is_model_specific_and_ambient_binding_cannot_change_old_round(monkeypatch):
    value = contracts.validate_scorer_policy(policy())
    environment = {}
    scoring.apply_policy_to_environment(value, environ=environment,
        credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, content in environment.items():
        monkeypatch.setenv(name, content)
    legacy = {'model': SOL, 'messages': [{'role': 'user', 'content': 'Review source'}],
              'max_tokens': 2000, 'reasoning': {'effort': 'medium', 'exclude': True}}
    assert opts.judge_reasoning_enabled() is True
    assert opts.judge_reasoning_enabled(SOL) is False
    assert opts.judge_reasoning_enabled(MODEL) is True
    assert opts.prepare_judge_request(legacy) == ('/chat/completions', legacy)
    with pytest.raises(ValueError, match='proprietary_exception_api_unsupported'):
        opts.prepare_judge_request(legacy, hosted_search=True)
    prepared = opts.prepare_judge_request({**legacy, 'model': MODEL, 'reasoning': {'effort': 'low'}})[1]
    assert prepared['reasoning']['exclude'] is False
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(scoring.build_scorer_policy(),
            environ={BINDING: json.dumps(EXCEPTION)},
            credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})


def dispatch(context, model=SOL, *, changes=None, operation='openrouter.chat'):
    response = {'id': 'gen-industry', 'model': model, 'choices': [], 'usage': {'cost': '0.0001'}}
    table = bt.price_table()
    table['models'][SOL] = deepcopy(table['models'][bt.CHAT['model']])
    table['models'][MODEL] = deepcopy(table['models'][bt.CHAT['model']])
    store, transport = bt.FakeLedgerStore(), bt.FakeTransport([(200, response)])
    broker = br.Broker(store=store, transport=transport, price_table=table,
        key_for=lambda provider: bt.HOST_KEYS[provider],
        judge_models={'industry_evidence_review': SOL, 'company_fit_reverification': MODEL})
    body = {'model': model, 'messages': [{'role': 'user', 'content': 'Review source'}],
            'max_tokens': 2000, 'reasoning': {'effort': 'medium', 'exclude': True}}
    body.update(changes or {})
    result = broker.execute(context, operation_id=operation, parameters=body,
                            action_sequence=0, timeout_ms=30000)
    return result, store, transport


def test_sol_exception_dispatch_preserves_legacy_body_route_and_price_bound():
    legacy, _, old_transport = dispatch(replace(bt.CONTEXT, kind='score'))
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry(),
                      judgment_cache_scope={'scorer_policy': policy()})
    result, store, transport = dispatch(context)
    assert result.status == legacy.status == 200
    assert transport.sent[0]['body'] == old_transport.sent[0]['body']
    assert result.call['request_hash'] == legacy.call['request_hash']
    assert result.call['reserved_microusd'] == legacy.call['reserved_microusd']
    body = json.loads(transport.sent[0]['body'])
    assert body['provider']['only'] == ['azure/us', 'azure/eu']
    assert body['provider']['allow_fallbacks'] is True
    assert body['reasoning'] == {'effort': 'medium', 'exclude': True}
    annotation = store.calls[result.call['call_identity']]['call_doc']['reasoning_capture']
    assert annotation['policy'] == 'provider_exposed_only'
    assert annotation['full_thinking_available'] is False
    assert annotation['exception_reason'] == 'quality_gate_not_met'


def test_exception_does_not_allow_open_model_legacy_controls_or_unfrozen_sol():
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry(),
                      judgment_cache_scope={'scorer_policy': policy()})
    for changed in (replace(context, judgment_cache_scope=None),
                    replace(context, kind='execute', reasoning_capture_policy=rc.POLICY,
                            reasoning_route_registry=registry())):
        result, store, transport = dispatch(changed)
        assert result.status == (503 if changed.kind == 'score' else 400)
        assert not transport.sent and not store.calls
    result, store, transport = dispatch(context, MODEL)
    assert result.status == 400 and not transport.sent and not store.calls
    assert br._reasoning_contract(context, SOL, 'openrouter.responses')[0] == rc.POLICY
    result, store, transport = dispatch(context, operation='openrouter.responses', changes={
        'input': [{'role': 'user', 'content': 'Review source'}], 'messages': None})
    assert result.status == 400 and not transport.sent and not store.calls


def test_mixed_policy_keeps_exact_full_open_route_and_rejects_route_bypass():
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry(),
                      judgment_cache_scope={'scorer_policy': policy()})
    controls = {'reasoning': {'effort': 'low', 'exclude': False}}
    result, store, transport = dispatch(context, MODEL, changes=controls)
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    assert body['provider']['only'] == ['example/verified']
    assert body['provider']['allow_fallbacks'] is False
    assert body['provider']['zdr'] is True and body['provider']['data_collection'] == 'deny'
    annotation = store.calls[result.call['call_identity']]['call_doc']['reasoning_capture']
    assert annotation['policy'] == rc.POLICY and annotation['reviewed_route'] is True
    for provider in ({'only': ['azure/us']}, {'only': ['example/verified'], 'allow_fallbacks': True}):
        result, store, transport = dispatch(context, MODEL, changes={**controls, 'provider': provider})
        assert result.status == 400 and not transport.sent and not store.calls


def test_industry_packet_preserves_exact_legacy_body_while_open_controller_stays_enabled(monkeypatch):
    from qualification.scoring import company_evidence_investigator as investigator
    calls = []
    async def capture(session, url, **kwargs):
        calls.append((url, kwargs['payload']))
        return 400, {}
    monkeypatch.setattr(investigator, '_post_json', capture)
    url = 'https://acme.example/'
    kwargs = dict(model=SOL, openrouter_key='test-not-secret', started=time.monotonic(),
        company_locator={'name': 'Acme', 'website': url},
        criteria={'industry': 'Software', 'subindustry': '', 'product_service': 'Reporting'},
        fetched_pages={url: 'Acme supplies reporting software.'}, fetched_final_urls={url: url},
        first_party_domains={'acme.example'}, identity_names={'acme'},
        identity_anchor={'submitted_domain': 'acme.example', 'observed_domain': 'acme.example',
                         'verified_domain': 'acme.example'}, evaluation_date_value='2026-11-05')
    monkeypatch.delenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', raising=False)
    asyncio.run(investigator._review_industry_sources(None, **kwargs))
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', rc.POLICY)
    monkeypatch.setenv(BINDING, json.dumps(EXCEPTION))
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: 'example/verified'}))
    monkeypatch.setenv('ARENA_COMPANY_REVERIFY_MODEL', MODEL)
    asyncio.run(investigator._review_industry_sources(None, **kwargs))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[1][1]['max_tokens'] == 2000
    assert calls[1][1]['reasoning'] == {'effort': 'medium', 'exclude': True}
    assert opts.judge_reasoning_enabled() and opts.judge_reasoning_enabled(MODEL)


def test_builder_options_cover_only_full_models_and_explicit_capture_off_has_no_exception():
    models = {role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS}
    models['industry_evidence_review'] = SOL
    value = scoring.build_scorer_policy(judge_models=models,
        judge_provider_routes={MODEL: 'example/verified'}, judge_reasoning_exceptions=EXCEPTION,
        judge_request_options={MODEL: {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096}})
    assert rc.validate_judge_registry(registry(), value) == registry()
    assert BINDING not in scoring.build_scorer_policy(judge_provider_routes={})['env_bindings']
    value['env_bindings']['ARENA_JUDGE_REQUEST_OPTIONS'] = json.dumps({
        MODEL: {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096},
        SOL: {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096}})
    with pytest.raises(contracts.ArenaContractError, match='request options'):
        contracts.validate_scorer_policy(value)


def mixed_example():
    example = judge_example()
    models = {role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS}
    models['industry_evidence_review'] = SOL
    example['configuration']['scorer_policy'] = scoring.build_scorer_policy(
        judge_models=models, judge_provider_routes={MODEL: 'route-a'},
        judge_reasoning_exceptions=EXCEPTION)
    call = deepcopy(example['provider_calls'][-1])
    call.update(call_identity='industry-call', action_sequence=1)
    request = call['request']
    request['hashed_parameters']['model'] = request['effective_parameters']['model'] = SOL
    request['outbound']['body']['model'] = SOL
    request['outbound']['body_sha256'] = contracts.document_hash(request['outbound']['body'])
    request['request_hash'] = call['request_hash'] = contracts.document_hash(request['hashed_parameters'])
    message = call['response']['choices'][0]['message']
    message.pop('reasoning_content')
    message['reasoning_details'] = [{'type': 'reasoning.summary', 'summary': 'Exposed summary only'}]
    call['response_sha256'] = contracts.document_hash(call['response'])
    call['response_receipts'][0]['response_sha256'] = call['response_sha256']
    call['response_receipts'][0]['metadata'].pop('reasoning_route_observation')
    call['reasoning_capture'] = {'policy': 'provider_exposed_only', 'model_id': SOL,
        'requested': False, 'reviewed_route': False, 'full_thinking_available': False,
        'exception_reason': 'quality_gate_not_met'}
    artifact = example['worker_observation_artifacts'][example['score_run_id']]
    receipt = deepcopy(artifact['payload']['response_receipts'][0])
    receipt.update(call_identity=call['call_identity'], ordinal=2, action_sequence=1)
    artifact['payload']['response_receipts'].append(receipt)
    ack = deepcopy(artifact['payload']['agent_observations'][0])
    ack['id'] = 'industry-ack'
    ack['data']['call_identity'] = call['call_identity']
    artifact['payload']['agent_observations'].append(ack)
    call['response_provenance']['worker_receipt_ref']['ordinal'] = 2
    call['response_provenance']['agent_receipt_refs'][0]['id'] = 'industry-ack'
    example['provider_calls'].append(call)
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for row in example['provider_calls'][1:]:
        for ref in row['response_provenance']['agent_receipt_refs'] + [row['response_provenance']['worker_receipt_ref']]:
            ref['payload_sha256'] = artifact['payload_sha256']
    return seal(example)


@pytest.mark.parametrize('missing_budget', [None, 0])
def test_mixed_export_keeps_approved_turn_targets_without_claiming_full_trajectory(missing_budget):
    example = mixed_example()
    if missing_budget is not None:
        example['configuration']['training_maximum_missing_reasoning_percent'] = missing_budget
        seal(example)
    before = deepcopy(example)
    exported, manifest = records(example, role='judge')
    assert len(exported) == 1
    assert exported[0]['input']['model'] == MODEL
    assert exported[0]['quality']['status'] == 'unreviewed'
    assert exported[0]['source_refs'][0]['judge_context']['scorer_policy_sha256'] == contracts.document_hash(
        example['configuration']['scorer_policy'])
    assert exported[0]['source_refs'][0]['selected_judge_evidence']['judge_full_thinking_complete'] is False
    assert manifest['counts']['exclusion_reasons']['proprietary_thinking_unavailable'] == 1
    coverage = manifest['coverage']
    assert coverage['generated_thinking_texts'] == 1
    assert coverage['trajectories']['complete_full_thinking'] == 0
    assert coverage['trajectories']['partial_judge_thinking'] == 1
    unavailable = next(row for row in coverage['attempts'] if row['model_id'] == SOL)
    assert unavailable['returned'] == 'summary'
    assert unavailable['full_thinking_available'] is False
    trace = training_thinking.readable_trace(example)
    industry = next(row for row in trace['calls'] if row['call_identity'] == 'industry-call')
    assert industry['reasoning_coverage']['full_thinking_available'] is False
    assert industry['reasoning_coverage']['exception_reason'] == 'quality_gate_not_met'
    assert any(event['kind'] == 'summary' and event['value'] == 'Exposed summary only'
               for event in industry['events'])
    assert not any(event['kind'] == 'full_generated_thinking' for event in industry['events'])
    assert example == before


def test_mixed_cli_export_retains_open_target_and_explicit_partial_status(tmp_path):
    from scripts.prepare_arena_full_thinking_dataset import main
    example_path = tmp_path / 'mixed.json'
    example_path.write_text(json.dumps(mixed_example()))
    output = tmp_path / 'judge-dataset'
    assert main(['--examples', str(example_path), '--output', str(output), '--role', 'judge']) == 0
    exported = [json.loads(line) for name in ('train.jsonl', 'eval.jsonl')
                for line in (output / name).read_text().splitlines()]
    assert len(exported) == 1 and exported[0]['input']['model'] == MODEL
    assert exported[0]['source_refs'][0]['selected_judge_evidence']['judge_full_thinking_complete'] is False
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['coverage']['trajectories']['partial_judge_thinking'] == 1
    assert manifest['coverage']['trajectories']['complete_full_thinking'] == 0
