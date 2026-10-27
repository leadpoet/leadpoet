"""New provider routes require exact paid evidence and preserve prior defaults."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from lab_arena import contracts, reasoning_capture


EVIDENCE = Path(__file__).parents[2] / 'docs/evidence/full-thinking'
PROOFS = [EVIDENCE / f'alternate-provider-{api}-proof-20261025.json'
          for api in ('chat', 'responses')]


def test_original_seven_routes_and_provider_defaults_are_unchanged():
    registry = reasoning_capture.load_registry()
    original = deepcopy(registry)
    original['approved_routes'] = original['approved_routes'][:7]
    original['baseline_route'] = None  # Preserve the historical route-evidence hash.
    assert contracts.document_hash(original) == (
        'sha256:bf758852eb6d47d37260ce7c42df397337ef78779e6d29524481027b507d1d46')
    for route in original['approved_routes']:
        params = {'model': route['model_id']}
        _, selected = reasoning_capture.apply(route['api'], params,
            policy=reasoning_capture.POLICY, registry=registry)
        assert selected['provider_route'] == route['provider']
        assert selected['max_output_tokens'] == route['max_output_tokens']


def test_every_added_route_matches_successful_exact_probe_and_can_be_selected():
    registry = reasoning_capture.load_registry()
    historical_added_count = sum(len(json.loads(path.read_bytes())['captures']) for path in PROOFS)
    # Keep this exact Oct25 evidence gate over its original appended rows.
    # New model additions have a separate current evidence manifest.
    pending = {(r['model_id'], r['api'], r['provider']): r
               for r in registry['approved_routes'][7:7 + historical_added_count]}
    for path in PROOFS:
        raw = path.read_bytes()
        for capture in json.loads(raw)['captures']:
            key = (capture['model_id'], capture['api'], capture['provider'])
            route = pending.pop(key)
            assert {k: v for k, v in route.items() if k != 'evidence'} == capture['route_metadata']
            assert route['evidence']['probe_manifest_sha256'] == hashlib.sha256(raw).hexdigest()
            checks = capture['checks']
            assert checks['raw_wire_and_SQL_storage_exact'] is True
            assert checks['two_tool_continuations_and_strict_json'] is True
            assert checks['client_callback_count'] == 0
            assert checks['strict_candidate_targets'] == 3
            assert checks['full_thinking_calls'] > 0
            assert checks['full_thinking_calls'] + checks['explicit_zero_calls'] == 3
            params = {'model': route['model_id'], 'provider': {'only': [route['provider']]}}
            _, selected = reasoning_capture.apply(route['api'], params,
                policy=reasoning_capture.POLICY, registry=registry)
            assert selected['provider_route'] == route['provider']
            assert selected['price_ceiling'] == route['price_ceiling']
    assert not pending
