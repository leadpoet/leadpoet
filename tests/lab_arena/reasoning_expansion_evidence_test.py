"""Preserve prior route defaults and bind expansion routes to reviewed evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import contracts, reasoning_capture as rc

MANIFEST = Path(__file__).parents[2] / 'docs/evidence/full-thinking/reasoning-expansion-20261027.json'
WORKER_MANIFEST = Path(__file__).parents[2] / 'docs/evidence/full-thinking/glm-nearai-worker-capture-20261106.json'


def route_key(route):
    return route['model_id'], route['api'], route['provider']


def evidence(path=MANIFEST):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def capture_evidence(additions, manifests):
    captures = {}
    for manifest, digest in manifests:
        for row in manifest['captures']:
            key = route_key(row['route_metadata'])
            assert key not in captures, 'duplicate route evidence: %r' % (key,)
            captures[key] = row, manifest, digest
    assert {route_key(row) for row in additions} == set(captures), 'added routes differ from exact capture evidence'
    return captures


def test_capture_evidence_rejects_duplicate_routes_across_manifests():
    route = {'model_id': 'test/model', 'api': 'openrouter.chat', 'provider': 'test/provider'}
    manifest = {'captures': [{'route_metadata': route}]}
    with pytest.raises(AssertionError, match='duplicate route evidence'):
        capture_evidence([route], [(manifest, 'first-hash'), (deepcopy(manifest), 'second-hash')])


def test_capture_evidence_rejects_an_added_route_without_its_own_proof():
    route = {'model_id': 'test/model', 'api': 'openrouter.chat', 'provider': 'test/provider'}
    unknown = {**route, 'provider': 'test/unknown'}
    manifest = {'captures': [{'route_metadata': route}]}
    with pytest.raises(AssertionError, match='added routes differ from exact capture evidence'):
        capture_evidence([route, unknown], [(manifest, 'first-hash')])


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


def assert_retained_research_capture(row, owner_manifest):
    """Composite research proof is distinct from fresh synthetic conformance."""
    assert owner_manifest['schema_version'] == 'retained_native_worker_replay_v1'
    assert owner_manifest['fresh_provider_execution'] is False
    assert owner_manifest['production_activation'] is False
    assert owner_manifest['action_quality'] == 'unreviewed'
    checks = row['checks']
    for name in ('original_request_hashes_exact', 'original_response_hashes_exact',
                 'SQL_request_response_readback_exact', 'worker_delivery_and_ack_exact',
                 'strict_unscored_targets_verified', 'readable_thinking_preserved'):
        assert checks[name] is True
    assert checks['full_text_turns'] == checks['strict_unscored_targets'] == 8
    assert checks['unknown_or_missing_positive_reasoning_turns'] == 0
    assert checks['summary_or_opaque_turns'] == 0
    assert checks['cap_or_truncation_signals'] == 0
    assert len(row['cases']) == 2
    for case in row['cases']:
        assert case['proof_kind'] == 'retained_real_provider_bytes_replayed_locally'
        assert case['fresh_provider_calls'] == 0
        assert case['action_quality'] == 'unreviewed'
        assert len(case['turns']) == 4
        for turn in case['turns']:
            assert turn['original_request_sha256'] == turn['replayed_request_sha256']
            assert turn['original_response_sha256'] == turn['replayed_response_sha256']
            assert len(turn['original_request_sha256'].removeprefix('sha256:')) == 64
            assert len(turn['original_response_sha256'].removeprefix('sha256:')) == 64
            assert turn['thinking_characters'] > 0
            assert turn['reported_reasoning_tokens'] > 0
            assert turn['returned'] == 'full_thinking'
            assert turn['provider_route_verified'] is True
            assert turn['worker_socket_written'] is True
            assert turn['agent_exact_response_acknowledged'] is True
            assert turn['strict_unscored_target'] is True
            assert turn['outbound_reasoning_controls'] == {'effort': 'low', 'exclude': False}
            assert turn['tool_choice'] == 'required'
            assert turn['finish_reason'] == 'tool_calls'
    # Failed fresh controls remain evidence, not successful replay claims.
    counter = owner_manifest['fresh_synthetic_counterevidence']
    assert {case['tool_choice'] for case in counter} == {'named', 'auto', 'required'}
    for case in counter:
        assert case['calls'] == 6
        assert case['missing_positive_thinking_turns'] > 0
        assert case['missing_thinking_excluded_from_generated_targets'] is True
        assert case['raw_SQL_and_worker_fidelity_passed'] is True


def test_added_routes_match_their_exact_capture_and_strict_target_evidence():
    manifest, _digest = evidence()
    current = rc.load_registry()
    additions = current['approved_routes'][manifest['original_route_count']:]
    captures = capture_evidence(additions, [evidence(path) for path in (MANIFEST, WORKER_MANIFEST)])
    for route in additions:
        row, owner_manifest, digest = captures[route_key(route)]
        assert {k: v for k, v in route.items() if k != 'evidence'} == row['route_metadata']
        assert route['evidence'] == {'documentation_url': row['documentation_url'],
            'probe_id': owner_manifest['probe_id'], 'verified_at': owner_manifest['verified_at'],
            'probe_manifest_sha256': digest}
        if owner_manifest.get('schema_version') == 'retained_native_worker_replay_v1':
            assert_retained_research_capture(row, owner_manifest)
            continue
        assert owner_manifest == manifest
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


@pytest.mark.parametrize('damage', ['missing_text', 'request_hash', 'worker_ack', 'production_claim'])
def test_retained_research_proof_rejects_capture_gaps_or_overclaims(damage):
    manifest, _ = evidence(WORKER_MANIFEST)
    row = manifest['captures'][0]
    turn = row['cases'][0]['turns'][0]
    if damage == 'missing_text':
        turn['thinking_characters'] = 0
    elif damage == 'request_hash':
        turn['replayed_request_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'worker_ack':
        turn['agent_exact_response_acknowledged'] = False
    else:
        manifest['production_activation'] = True
    with pytest.raises(AssertionError):
        assert_retained_research_capture(row, manifest)


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
