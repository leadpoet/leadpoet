"""Only frozen server authority selects a score provider and output ceiling."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, reasoning_capture as rc, scoring
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.reasoning_capture_test import _reviewed_registry

MODEL = 'test/frozen-judge'
FROZEN = 'preferred/verified'
OTHER = 'other/verified'


def dispatch(api, *, selector=None, mode=None, cap=4096, kind='score', damage=None,
             frozen_mode=None, options=True):
    registry = _reviewed_registry(model=MODEL, api=api)
    preferred = registry['approved_routes'][0]
    preferred.update(provider=FROZEN, provider_name='Preferred')
    alternate = deepcopy(preferred)
    alternate.update(provider=OTHER, provider_name='Other')
    registry['approved_routes'] = [alternate, preferred]
    policy = scoring.build_scorer_policy(
        judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={MODEL: FROZEN},
        judge_request_options={MODEL: {'reasoning': frozen_mode or {'effort': 'low'},
                                      'max_output_tokens': 4096}} if options else None)
    scope = {'scorer_policy': policy}
    if damage == 'missing_scope': scope = None
    elif damage == 'missing_policy': scope = {}
    elif damage == 'invalid_policy': policy['judge_models'].clear()
    elif damage == 'missing_marker': policy['env_bindings'].pop('ARENA_JUDGE_REASONING_CAPTURE_POLICY')
    body = {'model': MODEL}
    field = 'max_tokens' if api == 'openrouter.chat' else 'max_output_tokens'
    body[field] = cap
    body['messages' if api == 'openrouter.chat' else 'input'] = [{'role': 'user', 'content': 'Review supplied facts.'}]
    if mode is not None: body['reasoning'] = deepcopy(mode)
    if selector is not None: body['provider'] = {'only': [selector]}
    response = ({'choices': [], 'usage': {'cost': '0.0001'}} if api == 'openrouter.chat' else
                {'status': 'completed', 'output': [], 'usage': {'cost': '0.0001'}})
    table = bt.price_table()
    table['models'][MODEL] = deepcopy(next(iter(table['models'].values())))
    store, transport = bt.FakeLedgerStore(), bt.FakeTransport([(200, response)])
    broker = br.Broker(store=store, transport=transport, key_for=lambda p: bt.HOST_KEYS[p],
                       price_table=table, judge_models=(MODEL,))
    context = replace(bt.CONTEXT, kind='score' if kind == 'legacy' else kind,
        judge_reasoning_capture_policy=rc.POLICY if kind == 'score' else None,
        judge_reasoning_route_registry=registry if kind == 'score' else None,
        judgment_cache_scope=scope if kind == 'score' else None,
        reasoning_capture_policy=rc.POLICY if kind == 'execute' else None,
        reasoning_route_registry=registry if kind == 'execute' else None)
    result = broker.execute(context, operation_id=api, parameters=body,
                            action_sequence=0, timeout_ms=30000)
    return result, store, transport


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_score_installs_frozen_provider_and_reasoning_default_instead_of_first_probe(api):
    result, _, transport = dispatch(api)
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    assert body['provider']['only'] == [FROZEN]
    assert body['reasoning']['effort'] == 'low'


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_score_refuses_another_approved_provider_before_reservation(api):
    result, store, transport = dispatch(api, selector=OTHER, mode={'effort': 'low'})
    assert result.status == 400 and not store.calls and not transport.sent


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['missing_scope', 'missing_policy', 'invalid_policy', 'missing_marker'])
def test_score_cannot_replace_missing_or_invalid_frozen_authority_with_request_or_registry(api, damage):
    result, store, transport = dispatch(api, selector=FROZEN, mode={'effort': 'low'}, damage=damage)
    assert result.status == 503 and not store.calls and not transport.sent


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_frozen_output_ceiling_refuses_larger_request_before_reservation(api):
    result, store, transport = dispatch(api, selector=FROZEN, mode={'effort': 'low'}, cap=4097)
    assert result.status == 400 and not store.calls and not transport.sent


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_tighter_comparison_bound_and_explicit_compatible_effort_are_preserved(api):
    result, _, transport = dispatch(api, mode={'effort': 'high'}, cap=500)
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    assert body['reasoning']['effort'] == 'high'
    assert body['max_tokens' if api == 'openrouter.chat' else 'max_output_tokens'] == 500


@pytest.mark.parametrize('mode', [{'enabled': True}, {'max_tokens': 1024}, {'effort': 'high'}])
def test_chat_keeps_each_explicit_enabled_budget_or_effort_family(mode):
    result, _, transport = dispatch('openrouter.chat', mode=mode)
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    for key, value in mode.items(): assert body['reasoning'][key] == value


@pytest.mark.parametrize('mode', [{'enabled': True}, {'max_tokens': 1024}, {'effort': 'low'}])
def test_missing_chat_mode_uses_the_frozen_mode_family(mode):
    result, _, transport = dispatch('openrouter.chat', frozen_mode=mode)
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    for key, value in mode.items(): assert body['reasoning'][key] == value


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_execute_retains_its_explicit_approved_provider_choice(api):
    result, _, transport = dispatch(api, selector=OTHER, kind='execute')
    assert result.status == 200
    assert json.loads(transport.sent[0]['body'])['provider']['only'] == [OTHER]


def test_legacy_score_does_not_gain_a_provider_selector_or_reasoning_controls():
    result, _, transport = dispatch('openrouter.chat', kind='legacy')
    assert result.status == 200
    body = json.loads(transport.sent[0]['body'])
    assert 'only' not in body['provider'] and 'reasoning' not in body


def test_historical_full_policy_without_options_keeps_its_original_output_allowance():
    result, _, _ = dispatch('openrouter.chat', cap=4097, options=False)
    assert result.status == 200


def paired_export_example():
    from lab_arena import contracts, training_steps, training_thinking
    from tests.lab_arena.judge_thinking_export_test import judge_example
    from tests.lab_arena.training_steps_test import seal
    example = judge_example()
    registry = deepcopy(example['configuration']['reasoning_route_registry'])
    alternate = deepcopy(registry['approved_routes'][0])
    alternate.update(provider=OTHER, provider_name='Other')
    registry['approved_routes'].append(alternate)
    example['configuration']['judge_reasoning_route_registry'] = registry
    original = example['provider_calls'][-1]
    original['reasoning_capture']['registry_sha256'] = contracts.document_hash(registry).split(':')[1]
    other = deepcopy(original)
    other.update(call_identity='other-judge-call', action_sequence=1)
    other['reasoning_capture'].update(provider_route=OTHER, provider_name='Other')
    observation = other['response_receipts'][0]['metadata']['reasoning_route_observation']
    observation.update(requested_provider_route=OTHER, expected_provider_name='Other',
                       response_provider_name='Other', selected_endpoint_provider_name='Other')
    artifact = example['worker_observation_artifacts'][example['score_run_id']]
    receipt = deepcopy(artifact['payload']['response_receipts'][0])
    receipt.update(call_identity=other['call_identity'], ordinal=2, action_sequence=1)
    artifact['payload']['response_receipts'].append(receipt)
    ack = deepcopy(artifact['payload']['agent_observations'][0])
    ack['id'] = 'other-judge-ack'
    ack['data']['call_identity'] = other['call_identity']
    artifact['payload']['agent_observations'].append(ack)
    other['response_provenance']['worker_receipt_ref']['ordinal'] = 2
    other['response_provenance']['agent_receipt_refs'][0]['id'] = ack['id']
    example['provider_calls'].append(other)
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in (original, other):
        for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
            ref['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    # Both exact host/worker pairs and approved-provider observations are valid;
    # only the second call differs from the frozen provider selection.
    assert training_thinking.full_text_segments(other, 'judge', registry)[0]
    training_steps._candidate(example, other, role='judge')
    return example


def test_export_excludes_only_the_different_approved_provider_and_keeps_raw_trace():
    from lab_arena import training_thinking
    from tests.lab_arena.judge_thinking_export_test import records
    example = paired_export_example()
    before = deepcopy(example)
    exported, manifest = records(example, role='judge')
    assert len(exported) == 1
    assert {ref['call_identity'] for row in exported for ref in row['source_refs']} == {'judge-call'}
    assert manifest['counts']['exclusion_reasons']['judge_frozen_provider_route_mismatch'] == 1
    assert manifest['coverage']['trajectories']['complete_full_thinking'] == 0
    trace = training_thinking.readable_trace(example)
    invalid = next(row for row in trace['calls'] if row['call_identity'] == 'other-judge-call')
    assert invalid['reasoning_coverage']['full_thinking_status'] == 'judge_frozen_provider_route_mismatch'
    assert not any(event['kind'] == 'full_generated_thinking' for event in invalid['events'])
    assert example == before


@pytest.mark.parametrize('damage', ['missing_routes', 'invalid_policy'])
def test_export_cannot_replace_missing_frozen_authority_with_approved_route(damage):
    from tests.lab_arena.judge_thinking_export_test import judge_example, records
    example = judge_example()
    policy = example['configuration']['scorer_policy']
    if damage == 'missing_routes': policy['env_bindings'].pop('ARENA_JUDGE_PROVIDER_ROUTES')
    else: policy['judge_models'].clear()
    exported, _ = records(example, role='judge')
    assert exported == []
