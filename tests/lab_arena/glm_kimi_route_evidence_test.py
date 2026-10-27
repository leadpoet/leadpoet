"""Exact evidence and append-only regression gates for GLM/Kimi routes.

Broker tests below use synthetic provider replies and an in-memory ledger.
They prove host request policy, not live model capability or production rollout.
Live capability comes from the separately hash-bound committed manifest.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, reasoning_capture as rc, training_capture, training_thinking
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.full_thinking_round_test import OPEN, opted_in
from tests.lab_arena.training_export_test import Objects


MANIFEST = Path(__file__).parents[2] / 'docs/evidence/full-thinking/glm-kimi-routes-20261027.json'
ORIGINAL_COUNT = 25
# origin/main 93dc29f4e9a96114b9abf8ea9234ac455e2458f8, before this extension.
ORIGINAL_HASH = 'sha256:e1ed90eb2aa84bd09b9b1d9b8afa371bf4480d83923744545ac0dfc9a3ed44d3'
ORIGINAL_BASELINE = {'model_id': 'deepseek/deepseek-v4.1-flash-1017', 'api': 'openrouter.responses'}


def key(route):
    return route['model_id'], route['api'], route['provider']


def original_registry(current):
    original = {'schema_version': 1, 'baseline_route': deepcopy(ORIGINAL_BASELINE),
                'approved_routes': deepcopy(current['approved_routes'][:ORIGINAL_COUNT])}
    assert current['schema_version'] == 1 and current['baseline_route'] == ORIGINAL_BASELINE
    assert contracts.document_hash(original) == ORIGINAL_HASH
    return rc.validate_registry(original)


@pytest.fixture(scope='module')
def evidence():
    assert MANIFEST.is_file(), 'release must include the exact GLM/Kimi evidence manifest'
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    assert manifest['schema_version'] == 1
    current = rc.load_registry()
    original_registry(current)
    captures = manifest['captures']
    capture_keys = {key(row['route_metadata']) for row in captures}
    additions = current['approved_routes'][ORIGINAL_COUNT:ORIGINAL_COUNT + len(captures)]
    assert additions, 'a route extension must contain evidence-backed additions'
    assert {key(row) for row in additions} == capture_keys
    assert len({key(row['route_metadata']) for row in captures}) == len(captures)
    assert {key(row) for row in additions} == {key(row['route_metadata']) for row in captures}
    return current, additions, manifest, hashlib.sha256(raw).hexdigest()


def request(route, *, explicit_provider=True):
    value = {'model': route['model_id'], **({'input': 'Inspect the supplied evidence.'}
        if route['api'] == 'openrouter.responses' else
        {'messages': [{'role': 'user', 'content': 'Inspect the supplied evidence.'}]})}
    cap = 'max_output_tokens' if route['api'] == 'openrouter.responses' else 'max_tokens'
    value[cap] = route['max_output_tokens']
    value['reasoning'] = deepcopy(route['mode'])
    if route['api'] == 'openrouter.chat':
        value['reasoning']['exclude'] = False
    if explicit_provider:
        value['provider'] = {'only': [route['provider']]}
    return value


def test_original_twenty_five_rows_order_and_baseline_are_byte_for_value_unchanged():
    original_registry(rc.load_registry())


def test_every_appended_route_matches_exact_committed_evidence(evidence):
    _current, additions, manifest, digest = evidence
    captures = {key(row['route_metadata']): row for row in manifest['captures']}
    for route in additions:
        capture = captures[key(route)]
        assert route['model_id'].startswith(('z-ai/glm-', 'moonshotai/kimi-'))
        assert {name: value for name, value in route.items() if name != 'evidence'} == capture['route_metadata']
        assert route['evidence'] == {'documentation_url': capture['documentation_url'],
            'probe_id': manifest['probe_id'], 'verified_at': manifest['verified_at'],
            'probe_manifest_sha256': digest}
        checks = capture['checks']
        assert checks['raw_wire_and_SQL_storage_exact'] is True
        assert checks['two_tool_continuations_and_strict_json'] is True
        assert checks['strict_candidate_targets'] >= 6
        assert checks['full_thinking_calls'] > 0
        assert checks['full_thinking_calls'] + checks['explicit_zero_calls'] == 6
        assert checks['unknown_usage_calls'] == 0
        cases = capture['cases']
        assert len(cases) == 2
        assert {case['scenario'] for case in cases} == {'acme', 'conflicting_evidence'}
        assert all(case['native_final_format'] in {'json_schema', 'json_object', 'text'} for case in cases)
        if capture.get('native_schema_contradiction'):
            assert all(case['native_final_format'] != 'json_schema' for case in cases)


def test_original_normalized_requests_hashes_and_provider_defaults_are_unchanged():
    current = rc.load_registry()
    original = original_registry(current)
    for route in original['approved_routes']:
        for explicit in (False, True):
            parameters = request(route, explicit_provider=explicit)
            before = br.normalized_request(route['api'], parameters,
                reasoning_capture_policy=rc.POLICY, reasoning_route_registry=original)
            after = br.normalized_request(route['api'], parameters,
                reasoning_capture_policy=rc.POLICY, reasoning_route_registry=current)
            assert before == after
            assert contracts.document_hash(before) == contracts.document_hash(after)
            old_selection = rc.apply(route['api'], parameters, policy=rc.POLICY, registry=original)[1]
            selection = rc.apply(route['api'], parameters, policy=rc.POLICY, registry=current)[1]
            assert selection['provider_route'] == old_selection['provider_route']


def test_new_routes_use_normal_broker_privacy_provider_caps_prices_capture_and_replay(evidence):
    current, additions, manifest, _digest = evidence
    # Production loads model-catalog prices. A pinned provider can cost less
    # than that catalog value; preserve the existing larger-price bound.
    catalog_prices = manifest['production_catalog_prices']
    assert set(catalog_prices) == {route['model_id'] for route in additions}
    prices = br.validate_price_table({'schema_version': br.PRICE_TABLE_SCHEMA_VERSION,
        'fetched_at': manifest['verified_at'], 'source': br.OPENROUTER_MODELS_URL,
        'models': catalog_prices})
    for route in additions:
        response = {'model': route['model_id'], 'provider': route['provider_name'],
            'openrouter_metadata': {'endpoints': {'available': [{'selected': True,
                'model': route['model_version'], 'provider': route['provider_name']}]}},
            'usage': {'cost': '0.00001'}, **({'status': 'completed', 'output': [{
                'type': 'message', 'role': 'assistant', 'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'Done.'}]}]}
                if route['api'] == 'openrouter.responses' else {'choices': [{
                    'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Done.'}}]})}
        objects, store, transport = Objects(), bt.FakeLedgerStore(), bt.FakeTransport([(200, response)])
        broker = br.Broker(store=store, transport=transport, price_table=prices,
            key_for=lambda provider: bt.HOST_KEYS[provider],
            capture_request=training_capture.RequestCapture(objects).prepare)
        context = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=current)
        arguments = {'operation_id': route['api'], 'parameters': request(route),
                     'action_sequence': 0, 'timeout_ms': 30000}
        result = broker.execute(context, **arguments)
        assert result.status == 200, key(route)
        sent = json.loads(transport.sent[0]['body'])
        assert sent['provider']['only'] == [route['provider']]
        assert sent['provider']['zdr'] is True and sent['provider']['data_collection'] == 'deny'
        assert sent['provider']['allow_fallbacks'] is False and sent['provider']['require_parameters'] is True
        effective_prices = {component: max(
            Decimal(catalog_prices[route['model_id']][component]),
            Decimal(route['price_ceiling'][component])) for component in route['price_ceiling']}
        assert sent['provider']['max_price'] == {
            'prompt': float(effective_prices['prompt'] * 1_000_000),
            'completion': float((effective_prices['completion']
                                 + effective_prices['internal_reasoning']) * 1_000_000),
            'request': float(effective_prices['request'])}
        cap = 'max_output_tokens' if route['api'] == 'openrouter.responses' else 'max_tokens'
        assert sent[cap] == route['max_output_tokens']
        saved = store.calls[result.call['call_identity']]
        captured = training_capture.read_request(objects, saved['call_doc']['request_capture'])
        assert captured['outbound']['body'] == sent
        annotation = captured['reasoning_capture']
        assert annotation['price_ceiling'] == route['price_ceiling']
        assert annotation['mode'] == route['mode'] and annotation['full_text_fields'] == route['full_text_fields']
        assert annotation['probe_manifest_sha256'] == route['evidence']['probe_manifest_sha256']
        assert annotation['registry_sha256'] == contracts.document_hash(current).split(':', 1)[1]
        observation = br.broker_observations(saved['terminal'])['reasoning_route_observation']
        assert observation['status'] == 'matched'
        call = {'provider': 'openrouter', 'operation_id': route['api'], 'request': captured,
                'reasoning_capture': annotation, 'route_observation': observation, 'response': response}
        assert training_thinking._policy(call, current)
        replay = broker.execute(context, **arguments)
        assert replay.body == result.body and len(transport.sent) == 1


def test_old_frozen_round_rejects_new_routes_and_new_round_freezes_expanded_registry(evidence):
    current, additions, _manifest, _digest = evidence
    original = original_registry(current)
    old_configuration = opted_in(original).create_round(OPEN)
    before = contracts.document_hash(old_configuration)
    new_configuration = opted_in(current).create_round(OPEN)
    assert new_configuration['reasoning_route_registry'] == current
    assert old_configuration['reasoning_route_registry'] == original
    assert old_configuration['baseline_source_url'] == new_configuration['baseline_source_url']
    assert old_configuration['reasoning_baseline_source_sha256'] == new_configuration['reasoning_baseline_source_sha256']
    assert contracts.validate_round_configuration(json.loads(json.dumps(new_configuration))) == new_configuration
    for route in additions:
        with pytest.raises(ValueError, match='reasoning_route_unsupported'):
            rc.apply(route['api'], request(route), policy=rc.POLICY,
                     registry=old_configuration['reasoning_route_registry'])
    assert contracts.document_hash(old_configuration) == before
