"""Full-thinking inventory and projection reuse only episode-owned proofs."""
from contextlib import contextmanager
from copy import deepcopy
import gc
import weakref
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_steps, training_thinking, training_worker_proof
from tests.lab_arena.judge_thinking_export_test import judge_example
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_steps_worker_proof_reuse_test import published_example


@contextmanager
def stateless_proofs():
    payload_hash = training_worker_proof.payload_hash

    def original_hash(payload, worker_proofs=None):
        return payload_hash(payload)

    with patch.object(training_worker_proof.WorkerProofs, 'prepare', lambda self, example: example), \
            patch.object(training_worker_proof, 'payload_hash', side_effect=original_hash):
        yield


def source_example(count=8):
    example = published_example(count)
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    return seal(example)


def test_coverage_and_projection_worker_hashes_are_bounded_by_snapshots(monkeypatch):
    example = source_example()
    original = contracts.document_hash
    hashes = []

    def counted(value):
        if isinstance(value, dict) and 'response_receipts' in value and 'agent_observations' in value:
            hashes.append(len(value['response_receipts']))
        return original(value)

    monkeypatch.setattr(contracts, 'document_hash', counted)
    coverage = training_thinking.coverage([example])
    assert coverage['generated_thinking_texts'] == 8
    assert len(hashes) <= 2, hashes
    hashes.clear()
    splits, manifest = training_thinking.build([example])
    assert splits['train'] or splits['eval']
    assert len(hashes) <= 4, hashes
    assert manifest['counts']['records_by_split']['train'] + manifest['counts']['records_by_split']['eval'] > 0


@pytest.mark.parametrize('role', ['source', 'judge'])
@pytest.mark.parametrize('damage', ['none', 'privacy', 'worker_hash', 'target', 'route'])
def test_source_judge_coverage_and_full_outputs_equal_stateless_checks(role, damage):
    example = source_example(3) if role == 'source' else judge_example()
    example.update(round_id='arena-2026-10-28', round_status='published')
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    call = example['provider_calls'][0 if role == 'source' else -1]
    if damage == 'privacy':
        call['request_capture'] = {'redacted': True}
    elif damage == 'worker_hash':
        example['worker_observation_artifacts'][call['run_id']]['payload_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'target':
        call['response']['choices'][0]['finish_reason'] = 'length'
    elif damage == 'route':
        call['reasoning_capture']['reviewed_route'] = False
    seal(example)
    before = deepcopy(example)
    coverage_role = 'judge' if role == 'judge' else None
    with stateless_proofs():
        expected_coverage = training_thinking.coverage([example], role=coverage_role)
        expected_build = training_thinking.build([example], role=role)
    assert contracts.canonical_json(training_thinking.coverage([example], role=coverage_role)) == contracts.canonical_json(expected_coverage)
    assert contracts.canonical_json(training_thinking.build([example], role=role)) == contracts.canonical_json(expected_build)
    assert example == before
    if damage == 'none':
        assert expected_build[0]['train'] or expected_build[0]['eval']


def test_coverage_releases_owned_proof_before_next_episode(monkeypatch):
    native_owner = training_worker_proof.WorkerProofs
    owners = []

    class TrackedProofs(native_owner):
        def __init__(self):
            super().__init__()
            owners.append(weakref.ref(self))

    monkeypatch.setattr(training_worker_proof, 'WorkerProofs', TrackedProofs)
    first, second = source_example(2), source_example(2)

    def examples():
        yield first
        gc.collect()
        assert owners and all(owner() is None for owner in owners)
        second['worker_observation_artifacts']['run']['payload']['observation_count'] = 1
        seal(second)
        yield second

    report = training_thinking.coverage(examples())
    assert len(owners) == 2
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert report['trajectories']['ineligible'] >= 1
