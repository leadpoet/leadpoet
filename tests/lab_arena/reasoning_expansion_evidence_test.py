"""Preserve prior route defaults and bind expansion routes to reviewed evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from lab_arena import contracts, reasoning_capture as rc

MANIFEST = Path(__file__).parents[2] / 'docs/evidence/full-thinking/reasoning-expansion-20261027.json'


def route_key(route):
    return route['model_id'], route['api'], route['provider']


def evidence():
    raw = MANIFEST.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_all_prior_routes_order_baseline_and_defaults_remain_unchanged():
    manifest, _digest = evidence()
    current = rc.load_registry()
    count = manifest['original_route_count']
    original = {**deepcopy(current), 'approved_routes': deepcopy(current['approved_routes'][:count])}
    assert contracts.document_hash(original) == manifest['original_registry_document_hash']
    assert current['baseline_route'] == manifest['original_baseline_route']
    for route in original['approved_routes']:
        parameters = {'model': route['model_id'], **({'input': 'Inspect evidence.'}
            if route['api'] == 'openrouter.responses' else
            {'messages': [{'role': 'user', 'content': 'Inspect evidence.'}]})}
        before = rc.apply(route['api'], parameters, policy=rc.POLICY, registry=original)
        after = rc.apply(route['api'], parameters, policy=rc.POLICY, registry=current)
        assert before[0] == after[0]
        assert before[1]['provider_route'] == after[1]['provider_route']


def test_added_routes_match_two_scenario_exact_capture_and_strict_target_evidence():
    manifest, digest = evidence()
    current = rc.load_registry()
    additions = current['approved_routes'][manifest['original_route_count']:]
    captures = {route_key(row['route_metadata']): row for row in manifest['captures']}
    assert {route_key(row) for row in additions} == set(captures)
    for route in additions:
        row = captures[route_key(route)]
        assert {k: v for k, v in route.items() if k != 'evidence'} == row['route_metadata']
        assert route['evidence'] == {'documentation_url': row['documentation_url'],
            'probe_id': manifest['probe_id'], 'verified_at': manifest['verified_at'],
            'probe_manifest_sha256': digest}
        checks = row['checks']
        for name in ['raw_wire_and_SQL_storage_exact', 'request_hash_and_SQL_reference_exact',
                     'worker_delivery_and_strict_targets_verified', 'readable_text_and_order_preserved',
                     'two_tool_continuations_and_expected_json']:
            assert checks[name] is True
        assert checks['strict_unscored_targets'] == 6
        assert checks['full_text_turns'] > 0
        assert checks['full_text_turns'] + checks['explicit_zero_no_text_turns'] == 6
        assert checks['unknown_or_missing_positive_reasoning_turns'] == 0
        assert checks['summary_or_opaque_turns'] == 0
        assert checks['cap_or_truncation_signals'] == 0
        assert {case['scenario'] for case in row['cases']} == {'acme', 'conflicting_evidence'}
        for case in row['cases']:
            assert case['native_final_format'] == 'json_schema'
            assert len(case['turns']) == 3
            identity = case['returned_identity']
            assert identity['response_models'] == [route['model_id']]
            assert identity['selected_endpoint_models'] == [route['model_version']]
            assert identity['selected_endpoint_provider_names'] == [route['provider_name']]
            assert identity['response_provider_names'] in ([[None], [route['provider_name']]]
                if route['api'] == 'openrouter.responses' else [[route['provider_name']]])
            for turn in case['turns']:
                assert turn['provider_route_observation']['status'] == 'matched'
                if turn['thinking_state'] == 'full_text':
                    assert turn['thinking_characters'] > 0
                else:
                    assert turn['thinking_state'] == 'explicit_zero_no_text'
                    assert turn['reported_reasoning_tokens'] == 0


def test_expansion_capture_preserves_explicit_miner_reasoning_and_output_allowance():
    manifest, _digest = evidence()
    current = rc.load_registry()
    for route in current['approved_routes'][manifest['original_route_count']:]:
        cap = 'max_output_tokens' if route['api'] == 'openrouter.responses' else 'max_tokens'
        parameters = {'model': route['model_id'], 'provider': {'only': [route['provider']]},
            'reasoning': {'effort': 'low'}, cap: 4096,
            **({'input': 'Inspect evidence.'} if route['api'] == 'openrouter.responses' else
               {'messages': [{'role': 'user', 'content': 'Inspect evidence.'}]})}
        selected, annotation = rc.apply(route['api'], parameters, policy=rc.POLICY, registry=current)
        assert selected['reasoning'] == parameters['reasoning']
        assert selected[cap] == 4096
        assert annotation['mode'] == route['mode']
        assert annotation['max_output_tokens'] == route['max_output_tokens']
