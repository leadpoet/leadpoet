"""One host-owned transport plan preserves grading, action identity and proof."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc, scoring, training_capture, training_thinking
from lab_arena.store import ArenaStoreUnavailable
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.reasoning_capture_test import _reviewed_registry

MODEL = 'test/transport-judge'
PRIMARY = 'route-a'
SECONDARY = 'route-b'


class Objects:
    def __init__(self): self.data = {}
    def put(self, ref, value): self.data[ref] = bytes(value)
    def get_bounded(self, ref, limit):
        value = self.data[ref]
        assert len(value) <= limit
        return value


def setup(*, plan=(PRIMARY, SECONDARY), store=None, api='openrouter.chat', selected=SECONDARY):
    registry = _reviewed_registry(model=MODEL)
    registry['approved_routes'] = []
    for operation in ('openrouter.chat', 'openrouter.responses'):
        primary = _reviewed_registry(model=MODEL, api=operation)['approved_routes'][0]
        primary.update(provider=PRIMARY, provider_name='Route A')
        secondary = deepcopy(primary)
        secondary.update(provider=SECONDARY, provider_name='Route B')
        secondary['price_ceiling']['prompt'] = '0.000002'
        registry['approved_routes'] += [primary, secondary]
    policy = scoring.build_scorer_policy(
        judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={MODEL: PRIMARY},
        judge_request_options={MODEL: {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096}})
    context = replace(bt.CONTEXT, kind='score', judge_reasoning_capture_policy=rc.POLICY,
                      judge_reasoning_route_registry=registry, judgment_cache_scope={'scorer_policy': policy},
                      judge_provider_routing={MODEL: list(plan)} if plan is not None else None)
    name = 'Route A' if selected == PRIMARY else 'Route B'
    response = {'model': MODEL, 'provider': name, 'usage': {'cost': '0.0001'},
                'openrouter_metadata': {'endpoints': {'available': [
                    {'selected': True, 'model': MODEL, 'provider': name}]}},
                **({'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Approved',
                    'reasoning_details': [{'type': 'reasoning.text', 'text': 'Review actual retained evidence.'}]}}]}
                   if api == 'openrouter.chat' else {'status': 'completed', 'output': [
                       {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'Review actual retained evidence.'}], 'summary': []},
                       {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Approved'}]}]})}
    parameters = {'model': MODEL, 'provider': {'only': [PRIMARY]}, 'reasoning': {'effort': 'low'},
                  'max_tokens' if api == 'openrouter.chat' else 'max_output_tokens': 4096,
                  'messages' if api == 'openrouter.chat' else 'input': [{'role': 'user', 'content': 'Verify actual evidence.'}]}
    table = bt.price_table()
    table['models'][MODEL] = {key: '0.00000001' if key in ('prompt', 'completion') else '0'
                              for key in br.PRICED_COMPONENTS}
    store, transport, objects = store or bt.FakeLedgerStore(), bt.FakeTransport([(200, response)]), Objects()
    broker = br.Broker(store=store, transport=transport, key_for=lambda p: bt.HOST_KEYS[p],
                       price_table=table, judge_models=(MODEL,), capture_request=training_capture.RequestCapture(objects).prepare)
    return broker, context, parameters, store, transport, objects


def execute(parts, api='openrouter.chat'):
    broker, context, parameters, *_ = parts
    return broker.execute(context, operation_id=api, parameters=parameters, action_sequence=0, timeout_ms=30000)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('selected', [PRIMARY, SECONDARY])
def test_one_ordered_request_proves_actual_selected_route_and_preserves_settings(api, selected):
    parts = setup(api=api, selected=selected)
    _, context, original, store, transport, objects = parts
    policy = deepcopy(context.judgment_cache_scope['scorer_policy'])
    result = execute(parts, api)
    assert result.status == 200 and len(transport.sent) == 1 and len(store.calls) == 1
    saved = store.calls[result.call['call_identity']]
    request = training_capture.read_request(objects, saved['call_doc']['request_capture'])
    wire = json.loads(transport.sent[0]['body'])
    assert request['admitted_parameters'] == original
    assert request['hashed_parameters']['provider'] == {'only': [PRIMARY]}
    assert request['outbound']['body'] == wire
    assert result.call['request_hash'] == contracts.document_hash(request['hashed_parameters'])
    assert wire['provider'] == {'only': [PRIMARY, SECONDARY], 'order': [PRIMARY, SECONDARY],
        'allow_fallbacks': True, 'data_collection': 'deny', 'zdr': True,
        'require_parameters': True, 'max_price': {'prompt': 2, 'completion': 1, 'request': 0}}
    for field in ('model', 'reasoning', 'max_tokens', 'max_output_tokens', 'messages', 'input'):
        assert wire.get(field) == request['hashed_parameters'].get(field)
    assert context.judgment_cache_scope['scorer_policy'] == policy
    assert saved['amount'] == br._max_openrouter_cost_for_pricing(
        {**parts[0]._price_table['models'][MODEL], 'prompt': '0.000002', 'completion': '0.000001'},
        request['hashed_parameters'], max_output_tokens=4096, model=MODEL)
    annotation = saved['call_doc']['reasoning_capture']
    observation = br.broker_observations(saved['terminal'])['reasoning_route_observation']
    assert rc.valid_route_observation(observation)
    assert observation['status'] == 'matched' and observation['selected_provider_route'] == selected
    call = {'provider': 'openrouter', 'operation_id': api, 'request': request,
            'reasoning_capture': annotation, 'response': json.loads(result.body),
            'response_receipts': [{'metadata': {'reasoning_route_observation': observation}}]}
    text, status = training_thinking.full_text_segments(call, 'judge', context.judge_reasoning_route_registry)
    assert text and status['provider_route_verified'] and status['selected_provider_route'] == selected
    assert not training_thinking.full_text_segments(call, 'miner', context.judge_reasoning_route_registry)[0]


def test_single_route_correction_keeps_original_action_hash_and_selector():
    old, corrected = setup(plan=None), setup(plan=(SECONDARY,))
    first, second = execute(old), execute(corrected)
    assert first.call['call_identity'] == second.call['call_identity']
    assert first.call['request_hash'] == second.call['request_hash']
    assert json.loads(corrected[4].sent[0]['body'])['provider']['only'] == [SECONDARY]
    assert corrected[1].judgment_cache_scope == old[1].judgment_cache_scope


class CacheStore(bt.FakeLedgerStore):
    def reserve_judgment_call(self, **kwargs):
        self.cache_requests = getattr(self, 'cache_requests', []) + [deepcopy(kwargs)]
        return self.reserve_call(**kwargs)


def complete_cache_scope(parts):
    broker, context, parameters, store, transport, objects = parts
    scope = {'round_id': 'round-cache', 'evaluation_date': '2026-11-10',
             'scorer_image_digest': 'sha256:' + 'e' * 64,
             'scorer_image_reference': 'example@sha256:' + 'e' * 64,
             'scorer_policy': context.judgment_cache_scope['scorer_policy']}
    return broker, replace(context, round_id=scope['round_id'], judgment_cache_scope=scope), parameters, store, transport, objects


def test_complete_judgment_cache_scope_stays_identical_across_transport_plans():
    keys, scopes, identities, wires = [], [], [], []
    for plan in (None, (SECONDARY,), (PRIMARY, SECONDARY)):
        parts = complete_cache_scope(setup(plan=plan, store=CacheStore()))
        result = execute(parts)
        assert result.status == 200
        saved = parts[3].cache_requests[0]['call_doc']
        keys.append(saved['judgment_cache_key'])
        scopes.append(saved['judgment_cache_canonical'])
        identities.append(result.call['call_identity'])
        wires.append(parts[4].sent[0]['body'])
    assert len(set(keys)) == len(set(scopes)) == len(set(identities)) == 1
    assert len(set(wires)) == 3


@pytest.mark.parametrize('state', ['settled', 'pending'])
def test_complete_cache_scope_original_action_replay_does_not_dispatch_again(state):
    class StopStore(CacheStore):
        stop = state == 'pending'
        def mark_dispatched(self, **kwargs):
            if self.stop: raise ArenaStoreUnavailable('synthetic stop')
            return super().mark_dispatched(**kwargs)
    store = StopStore()
    old = complete_cache_scope(setup(plan=None, store=store, selected=PRIMARY))
    if state == 'pending':
        with pytest.raises(ArenaStoreUnavailable): execute(old)
    else: execute(old)
    prior = deepcopy(store.calls)
    store.stop = False
    corrected = complete_cache_scope(setup(plan=(SECONDARY,), store=store))
    result = execute(corrected)
    assert result.status == (200 if state == 'settled' else 503)
    assert store.calls == prior and not corrected[4].sent
    assert len({request['call_doc']['judgment_cache_key'] for request in store.cache_requests}) == 1


@pytest.mark.parametrize('state', ['settled', 'pending', 'dispatched', 'uncertain'])
def test_original_same_action_state_wins_after_route_activation(state):
    class StopBeforeDispatch(bt.FakeLedgerStore):
        stop = state == 'pending'
        def mark_dispatched(self, **kwargs):
            if self.stop: raise ArenaStoreUnavailable('synthetic stop before dispatch')
            return super().mark_dispatched(**kwargs)
    store = StopBeforeDispatch()
    old = setup(plan=None, store=store, selected=PRIMARY)
    if state == 'pending':
        with pytest.raises(ArenaStoreUnavailable): execute(old)
    else:
        execute(old)
        call = next(iter(store.calls.values()))
        if state in ('dispatched', 'uncertain'):
            call['kind'] = 'dispatch' if state == 'dispatched' else 'uncertain'
    prior = deepcopy(store.calls)
    store.stop = False
    corrected = setup(plan=(SECONDARY,), store=store)
    replay = execute(corrected)
    assert len(store.calls) == 1 and not corrected[4].sent
    assert store.calls == prior
    assert replay.status == {'settled': 200, 'pending': 503, 'dispatched': 409, 'uncertain': 409}[state]


@pytest.mark.parametrize('fault', ['model_version', 'api', 'context', 'output', 'mode', 'text_fields', 'provider_name', 'missing'])
def test_incompatible_or_missing_reviewed_secondary_is_refused_before_reservation(fault):
    parts = setup()
    registry = parts[1].judge_reasoning_route_registry
    row = registry['approved_routes'][1]
    if fault == 'model_version': row['model_version'] = 'test/different-version'
    elif fault == 'api': row['api'] = 'openrouter.responses'
    elif fault == 'context': row['context_tokens'] = 16384
    elif fault == 'output': row['max_output_tokens'] = 4096
    elif fault == 'mode': row['mode'] = {'enabled': True}
    elif fault == 'text_fields': row['full_text_fields'] = ['reasoning_content']
    elif fault == 'provider_name': row['provider_name'] = 'Route A'
    else: registry['approved_routes'].remove(row)
    result = execute(parts)
    assert result.status == 503 and not parts[3].calls and not parts[4].sent


@pytest.mark.parametrize('plan', [[], [PRIMARY, PRIMARY], [PRIMARY, SECONDARY, 'route-c'], ['unknown'], 'route-b', None])
def test_invalid_transport_plan_fails_closed(plan):
    parts = setup()
    object.__setattr__(parts[1], 'judge_provider_routing', {MODEL: plan})
    result = execute(parts)
    assert result.status == 503 and not parts[3].calls and not parts[4].sent


@pytest.mark.parametrize('fault', ['provider', 'version', 'missing', 'ambiguous'])
def test_unknown_or_ambiguous_actual_selection_never_proves_full_thinking(fault):
    parts = setup()
    result = execute(parts)
    saved = parts[3].calls[result.call['call_identity']]
    annotation = saved['call_doc']['reasoning_capture']
    response = json.loads(result.body)
    endpoint = response['openrouter_metadata']['endpoints']['available'][0]
    if fault == 'provider': endpoint['provider'] = response['provider'] = 'Unknown'
    elif fault == 'version': endpoint['model'] = 'test/other'
    elif fault == 'missing': response.pop('openrouter_metadata')
    else: response['openrouter_metadata']['endpoints']['available'].append(deepcopy(endpoint))
    observation = rc.route_observation(response, annotation)
    assert observation['status'] != 'matched'
    assert not training_thinking._observation_matches(observation, annotation)


def test_execute_with_transport_plan_keeps_existing_single_provider_behavior():
    parts = setup()
    context = replace(parts[1], kind='execute', reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=parts[1].judge_reasoning_route_registry)
    result = parts[0].execute(context, operation_id='openrouter.chat', parameters=parts[2],
                             action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    policy = json.loads(parts[4].sent[0]['body'])['provider']
    assert policy['only'] == [PRIMARY] and policy['allow_fallbacks'] is False and 'order' not in policy
    assert 'judge_provider_routing' not in next(iter(parts[3].calls.values()))['call_doc']['reasoning_capture']


def test_valid_negative_judgment_never_starts_another_gateway_call():
    parts = setup()
    response = parts[4].responses[0][1]
    response['choices'][0]['message']['content'] = '{"decision":"reject","reason":"evidence contradicts the claim"}'
    result = execute(parts)
    assert result.status == 200 and len(parts[4].sent) == 1 and len(parts[3].calls) == 1
    assert json.loads(result.body)['choices'][0]['message']['content'] == response['choices'][0]['message']['content']


def test_service_binds_transport_only_to_judge_context():
    from types import SimpleNamespace
    from lab_arena import service
    from tests.lab_arena.champion_baseline_reasoning_test import baseline_round
    row, baseline = baseline_round({})
    row['configuration_doc']['judge_provider_routing'] = {MODEL: [SECONDARY]}
    run = dict(baseline, run_id='run', round_id=row['round_id'], assignment_id='assignment',
               attempt=1, icp_position=0, stage=1, kind='score')
    app = object.__new__(service.ArenaService)
    app._store = SimpleNamespace(get_run=lambda _: run)
    app._require_round_ownership = lambda _: None
    app._hot_round = lambda _: row
    _, context = app._run_context('run', 'token')
    assert context.judge_provider_routing == {MODEL: [SECONDARY]}
    run['kind'] = 'execute'
    assert app._run_context('run', 'token')[1].judge_provider_routing is None


def test_provider_policy_cannot_enable_plan_without_explicit_host_authority():
    parts = setup()
    execute(parts)
    provider = json.loads(parts[4].sent[0]['body'])['provider']
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request('openrouter.chat', parts[2], allow_full_thinking=True,
                                          openrouter_provider_policy=provider)


@pytest.mark.parametrize('fault', ['zdr', 'data_collection', 'max_price'])
def test_transport_plan_cannot_bypass_existing_privacy_or_price_policy(fault):
    parts = setup()
    execute(parts)
    provider = json.loads(parts[4].sent[0]['body'])['provider']
    provider[fault] = False if fault == 'zdr' else 'allow' if fault == 'data_collection' else {'prompt': -1}
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request('openrouter.chat', parts[2], allow_full_thinking=True,
            openrouter_provider_policy=provider, judge_provider_routing=[PRIMARY, SECONDARY])


def export_example():
    from tests.lab_arena.judge_thinking_export_test import judge_example
    from tests.lab_arena.judge_chat_annotations_test import bind_response
    example = judge_example()
    registry = deepcopy(example['configuration']['reasoning_route_registry'])
    first = registry['approved_routes'][0]
    second = deepcopy(first)
    second.update(provider=SECONDARY, provider_name='Route B')
    registry['approved_routes'].append(second)
    for row in list(registry['approved_routes']):
        response_route = deepcopy(row)
        response_route.update(api='openrouter.responses', full_text_fields=['output.content.reasoning_text'])
        registry['approved_routes'].append(response_route)
    example['configuration']['judge_reasoning_route_registry'] = registry
    example['configuration']['judge_provider_routing'] = {'test/model': [PRIMARY, SECONDARY]}
    call = example['provider_calls'][-1]
    annotation = call['reasoning_capture']
    annotation['registry_sha256'] = contracts.document_hash(registry).split(':')[1]
    annotation['judge_provider_routing'] = rc.judge_transport_candidates(annotation, registry, [PRIMARY, SECONDARY])
    request = call['request']
    request['hashed_parameters']['provider'] = {'only': [PRIMARY]}
    request['admitted_parameters'] = deepcopy(request['hashed_parameters'])
    call['request_hash'] = request['request_hash'] = contracts.document_hash(request['hashed_parameters'])
    provider = {'only': [PRIMARY, SECONDARY], 'order': [PRIMARY, SECONDARY], 'allow_fallbacks': True,
                'zdr': True, 'data_collection': 'deny', 'require_parameters': True,
                'max_price': {'prompt': 1, 'completion': 2, 'request': 0}}
    request['outbound']['body']['provider'] = provider
    request['outbound']['body_sha256'] = contracts.document_hash(request['outbound']['body'])
    response = call['response']
    response.update(model='test/model', provider='Route B', openrouter_metadata={'endpoints': {'available': [
        {'selected': True, 'model': 'test/model', 'provider': 'Route B'}]}})
    call['response_receipts'][0]['metadata']['reasoning_route_observation'] = rc.route_observation(response, annotation)
    return bind_response(example)


def test_training_export_requires_transport_authority_and_keeps_actual_provider():
    from tests.lab_arena.judge_thinking_export_test import records
    example = export_example()
    call = example['provider_calls'][-1]
    before = deepcopy(example)
    assert not training_thinking._judge_frozen_route_error(example, call)
    got, manifest = records(example, role='judge')
    assert len(got) == 1
    assert got[0]['generated_thinking'][0]['text'] == 'Inspect the retained page, then answer.'
    assert manifest['coverage']['attempts'][-1]['selected_provider_route'] == SECONDARY
    assert example == before
    del example['configuration']['judge_provider_routing']
    assert training_thinking._judge_frozen_route_error(example, call) == 'judge_frozen_provider_route_mismatch'
    assert not records(example, role='judge')[0]


def test_historical_judge_export_preserves_recorded_order_after_priority_reorder():
    from tests.lab_arena.judge_thinking_export_test import records
    example = export_example()
    call = example['provider_calls'][-1]
    request = deepcopy(call['request'])
    source_before, _ = records(example)
    judges_before, _ = records(example, role='judge')
    assert len(source_before) == len(judges_before) == 1

    example['configuration']['judge_provider_routing']['test/model'].reverse()
    judges_after, manifest = records(example, role='judge')
    source_after, _ = records(example)
    assert len(judges_after) == 1
    assert not training_thinking._judge_frozen_route_error(example, call)
    for field in ('input', 'target', 'generated_thinking'):
        assert judges_after[0][field] == judges_before[0][field]
        assert source_after[0][field] == source_before[0][field]
    assert call['request'] == request
    assert call['request']['outbound']['body']['provider']['order'] == [PRIMARY, SECONDARY]
    assert manifest['coverage']['attempts'][-1]['selected_provider_route'] == SECONDARY


def test_historical_lineage_route_keeps_its_recorded_order_and_failed_status():
    example = export_example()
    call = example['provider_calls'][-1]
    attempt = next(row for row in example['attempts'] if row['run_id'] == call['run_id'])
    attempt['status'] = 'failed'
    example['score_run_id'] = 'later-accepted-judge'
    example['configuration']['judge_provider_routing']['test/model'].reverse()
    assert not training_thinking._judge_frozen_route_error(example, call, individual=True)
    assert attempt['status'] == 'failed'


@pytest.mark.parametrize('damage', [
    'missing_provider', 'extra_approved_provider', 'duplicate_provider', 'descriptor',
    'model', 'registry', 'privacy', 'wire_order', 'wire_only',
])
def test_historical_priority_reorder_preserves_exact_authority_and_request_checks(damage):
    from tests.lab_arena.judge_thinking_export_test import records
    example = export_example()
    call = example['provider_calls'][-1]
    config = example['configuration']
    config['judge_provider_routing']['test/model'].reverse()
    if damage == 'missing_provider':
        config['judge_provider_routing']['test/model'] = [SECONDARY]
    elif damage == 'extra_approved_provider':
        registry = config['judge_reasoning_route_registry']
        for route in list(registry['approved_routes']):
            if route['provider'] == PRIMARY:
                third = deepcopy(route)
                third.update(provider='route-c', provider_name='Route C')
                registry['approved_routes'].append(third)
        config['judge_provider_routing']['test/model'] = [SECONDARY, 'route-c']
        call['reasoning_capture']['registry_sha256'] = contracts.document_hash(registry).split(':')[1]
    elif damage == 'duplicate_provider':
        candidates = call['reasoning_capture']['judge_provider_routing']
        candidates[1] = deepcopy(candidates[0])
    elif damage == 'descriptor':
        call['reasoning_capture']['judge_provider_routing'][0]['price_ceiling']['prompt'] = '0.000002'
    elif damage == 'model':
        call['reasoning_capture']['model_id'] = 'different/model'
    elif damage == 'registry':
        config['judge_reasoning_route_registry']['approved_routes'][0]['evidence']['probe_id'] = 'changed-probe'
    elif damage == 'privacy':
        call['request_capture'] = {'redacted': True}
    else:
        field = 'order' if damage == 'wire_order' else 'only'
        body = call['request']['outbound']['body']
        body['provider'][field].reverse()
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    assert not records(example, role='judge')[0]


@pytest.mark.parametrize('fault', ['summary', 'opaque', 'response_privacy', 'request_privacy',
                                   'wrong_provider', 'wrong_version', 'forged_candidate', 'wrong_authority',
                                   'missing_selectors', 'wire_privacy'])
def test_fallback_does_not_weaken_full_thinking_or_private_pair_proof(fault):
    from tests.lab_arena.judge_thinking_export_test import records
    from tests.lab_arena.judge_chat_annotations_test import bind_response
    example = export_example()
    call = example['provider_calls'][-1]
    if fault == 'summary':
        message = call['response']['choices'][0]['message']
        message.pop('reasoning_content')
        message['reasoning_details'] = [{'type': 'reasoning.summary', 'summary': 'Only a summary.'}]
    elif fault == 'opaque':
        message = call['response']['choices'][0]['message']
        message.pop('reasoning_content')
        message['reasoning_details'] = [{'type': 'reasoning.encrypted', 'data': 'opaque'}]
    elif fault == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif fault == 'request_privacy': call['request_capture'] = {'redacted': True}
    elif fault in ('wrong_provider', 'wrong_version'):
        endpoint = call['response']['openrouter_metadata']['endpoints']['available'][0]
        if fault == 'wrong_provider': endpoint['provider'] = call['response']['provider'] = 'Unapproved'
        else: endpoint['model'] = 'test/wrong-version'
        call['response_receipts'][0]['metadata']['reasoning_route_observation'] = rc.route_observation(
            call['response'], call['reasoning_capture'])
    elif fault == 'forged_candidate': call['reasoning_capture']['judge_provider_routing'][1]['model_version'] = 'test/wrong-version'
    elif fault == 'missing_selectors':
        request = call['request']
        request['hashed_parameters'].pop('provider')
        request['admitted_parameters'].pop('provider')
        call['request_hash'] = request['request_hash'] = contracts.document_hash(request['hashed_parameters'])
    elif fault == 'wire_privacy':
        request = call['request']
        request['outbound']['body']['provider']['zdr'] = False
        request['outbound']['body_sha256'] = contracts.document_hash(request['outbound']['body'])
    else: example['configuration']['judge_provider_routing'] = {'test/model': [PRIMARY]}
    bind_response(example)
    assert not records(example, role='judge')[0]
