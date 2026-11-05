"""Judge capture freezes its own registry without changing miner authority."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, reasoning_capture, scoring
from lab_arena.service import ArenaService, RoundDefaults, ServiceError
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.test_benchmark_disclosure_policy import _RoundStore

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)
MODEL = 'test/model'


def service():
    value = object.__new__(ArenaService)
    value._config = SimpleNamespace(defaults=RoundDefaults(
        baseline_hotkey='5' + 'B' * 47,
        baseline_source_url='https://example.com/baseline.tar.gz',
        scorer_image_digest='sha256:' + '1' * 64,
        scorer_image_reference='registry.example/scorer@sha256:' + '1' * 64),
        mode='shadow', network_name='test', netuid=401, pinned_round_id=None)
    value._store = _RoundStore()
    value._scorer_policy = scoring.build_scorer_policy(judge_provider_routes={})
    value.runner_settings = lambda: (['5' + 'C' * 47], [])
    return value


def enable(value, registry):
    value._scorer_policy = scoring.build_scorer_policy(
        judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={MODEL: 'example/verified'})
    value._config.reasoning_registry_source = lambda: registry


def test_new_judge_policy_freezes_registry_and_existing_round_keeps_legacy_policy():
    value = service()
    old = value.create_round(NOW, round_id='arena-2026-11-05-test')
    registry = _reviewed_registry(model=MODEL)
    registry['approved_routes'] += _reviewed_registry(model=MODEL, api='openrouter.responses')['approved_routes']
    enable(value, registry)
    assert value.create_round(NOW, round_id='arena-2026-11-05-test') == old
    future = value.create_round(NOW + timedelta(days=1), round_id='arena-2026-11-06-test')
    assert future['judge_reasoning_route_registry'] == registry
    assert 'judge_reasoning_route_registry' not in old
    assert 'reasoning_capture_policy' not in future
    assert 'reasoning_route_registry' not in future
    assert contracts.validate_round_configuration(deepcopy(future)) == future
    registry['approved_routes'].clear()
    assert future['judge_reasoning_route_registry']['approved_routes']


@pytest.mark.parametrize('registry', [None, {}, _reviewed_registry(model='other/model')])
def test_invalid_or_missing_judge_route_fails_before_new_round_persistence(registry):
    value = service()
    enable(value, registry)
    with pytest.raises(ServiceError, match='judge_reasoning_capture_activation_not_ready'):
        value.create_round(NOW, round_id='arena-2026-11-05-test')
    assert value._store.rows == {}


def test_judge_registry_cannot_enable_miner_capture_or_stand_alone():
    value = service()
    old = value.create_round(NOW, round_id='arena-2026-11-05-test')
    old['judge_reasoning_route_registry'] = _reviewed_registry(model=MODEL)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(old)
    assert not reasoning_capture.enabled(old)


@pytest.mark.parametrize('mode', [{'enabled': True}, {'effort': 'low'}, {'max_tokens': 1024}])
def test_frozen_options_roundtrip_and_remain_bound_to_exact_route(monkeypatch, mode):
    from qualification.scoring import openrouter_options as opts
    registry = _reviewed_registry(model=MODEL)
    registry['approved_routes'][0]['mode'] = (
        {'budget_tokens': 1024} if 'max_tokens' in mode else deepcopy(mode))
    search_model = 'test/search'
    registry['approved_routes'] += _reviewed_registry(
        model=search_model, api='openrouter.responses')['approved_routes']
    options = {MODEL: {'reasoning': mode, 'max_output_tokens': 4096},
               search_model: {'reasoning': {'effort': 'low'}, 'max_output_tokens': 4096}}
    models = {role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS}
    models['company_fit_reverification'] = search_model
    policy = scoring.build_scorer_policy(judge_models=models,
        judge_provider_routes={MODEL: 'example/verified', search_model: 'example/verified'},
        judge_request_options=options)
    assert contracts.validate_scorer_policy(deepcopy(policy)) == policy
    assert reasoning_capture.validate_judge_registry(registry, policy) == registry
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment,
        credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    original = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Check source'}], 'max_tokens': 200}
    path, prepared = opts.prepare_judge_request(original)
    assert path == '/chat/completions'
    assert prepared['reasoning'] == {**mode, 'exclude': False}
    assert prepared['max_tokens'] == 4096
    assert prepared['provider'] == {'only': ['example/verified']}
    assert original['max_tokens'] == 200 and 'reasoning' not in original
    _, comparison = opts.prepare_judge_request({**original, 'max_tokens': 500}, output_token_floor=500)
    assert comparison['max_tokens'] == 500
    if 'max_tokens' in mode:
        # Broker rejects a reasoning budget larger than the whole small reply.
        with pytest.raises(ValueError, match='reasoning_headroom_unavailable'):
            reasoning_capture.apply('openrouter.chat', comparison,
                policy=reasoning_capture.POLICY, registry=registry)
    else:
        checked, _ = reasoning_capture.apply('openrouter.chat', comparison,
            policy=reasoning_capture.POLICY, registry=registry)
        assert checked['max_tokens'] == 500
    broken = deepcopy(registry)
    broken['approved_routes'][0]['provider'] = 'different/provider'
    with pytest.raises(ValueError, match='route unavailable'):
        reasoning_capture.validate_judge_registry(broken, policy)
    broken = deepcopy(registry)
    broken['approved_routes'][0]['max_output_tokens'] = 2048
    with pytest.raises(ValueError, match='options differ'):
        reasoning_capture.validate_judge_registry(broken, policy)


@pytest.mark.parametrize('bad', [None, {}, {'test/model': {'reasoning': {'enabled': False}, 'max_output_tokens': 4096}},
    {'test/model': {'reasoning': {'effort': 'low'}, 'max_output_tokens': True}}])
def test_request_options_cannot_enable_missing_model_or_invalid_mode(bad):
    if bad is None:
        # An ambient options map cannot upgrade a legacy frozen round.
        with pytest.raises(scoring.ScorerPolicyConflict):
            scoring.apply_policy_to_environment(scoring.build_scorer_policy(),
                environ={'ARENA_JUDGE_REQUEST_OPTIONS': '{}'},
                credentials={name: 'test-not-secret' for name in scoring.CREDENTIAL_ENV_NAMES})
    else:
        with pytest.raises(contracts.ArenaContractError, match='request options'):
            scoring.build_scorer_policy(judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
                judge_provider_routes={MODEL: 'example/verified'}, judge_request_options=bad)


def test_compatible_explicit_effort_wins_but_enabled_route_rejects_effort(monkeypatch):
    import json
    from qualification.scoring import openrouter_options as opts
    monkeypatch.setenv('ARENA_JUDGE_REASONING_CAPTURE_POLICY', opts.JUDGE_REASONING_POLICY)
    monkeypatch.setenv('ARENA_JUDGE_PROVIDER_ROUTES', json.dumps({MODEL: 'example/verified'}))
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Check'}],
            'reasoning': {'effort': 'high'}, 'max_tokens': 200}
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps({MODEL: {
        'reasoning': {'effort': 'low'}, 'max_output_tokens': 2048}}))
    assert opts.prepare_judge_request(body)[1]['reasoning']['effort'] == 'high'
    monkeypatch.setenv('ARENA_JUDGE_REQUEST_OPTIONS', json.dumps({MODEL: {
        'reasoning': {'enabled': True}, 'max_output_tokens': 2048}}))
    with pytest.raises(ValueError, match='incompatible'):
        opts.prepare_judge_request(body)
