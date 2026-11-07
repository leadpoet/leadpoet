"""Step exports own one worker proof per episode without changing candidates."""
from contextlib import contextmanager
from copy import deepcopy
import gc
import weakref
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_steps, training_worker_proof
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_worker_proof_reuse_test import large_example


def published_example(count):
    example = large_example(count)
    example.update(round_id='arena-2026-10-28', round_status='published')
    return seal(example)


@contextmanager
def original_worker_checks():
    candidate = training_steps._candidate

    def stateless(*args, **kwargs):
        kwargs.pop('worker_proofs', None)
        return candidate(*args, **kwargs)

    with patch.object(training_worker_proof.WorkerProofs, 'prepare', lambda self, example: example), \
            patch.object(training_steps, '_candidate', side_effect=stateless):
        yield


@pytest.mark.parametrize('host_observed', [False, True])
def test_many_step_calls_hash_the_owned_worker_payload_once(monkeypatch, host_observed):
    example = published_example(12)
    before = deepcopy(example)
    hashes = []
    native_hash = contracts.document_hash

    def counted(value):
        if isinstance(value, dict) and 'response_receipts' in value and 'agent_observations' in value:
            hashes.append(len(value['response_receipts']))
        return native_hash(value)

    monkeypatch.setattr(contracts, 'document_hash', counted)
    splits, manifest = training_steps._build([example], host_observed=host_observed)
    rows = splits['train'] + splits['eval']
    assert rows and len(rows[0]['source_refs']) == 12
    assert hashes == [12], hashes
    assert example == before
    assert manifest['counts']['records_by_split']['train'] + manifest['counts']['records_by_split']['eval'] == len(rows)


@pytest.mark.parametrize('host_observed', [False, True])
@pytest.mark.parametrize('damage', ['none', 'privacy', 'worker_hash', 'receipt_hash'])
def test_actual_multicall_records_exclusions_and_provenance_equal_stateless_checks(host_observed, damage):
    example = published_example(4)
    if damage == 'privacy':
        example['provider_calls'][1]['request_capture'] = {'redacted': True}
    elif damage == 'worker_hash':
        example['worker_observation_artifacts']['run']['payload_sha256'] = 'sha256:' + '0' * 64
    elif damage == 'receipt_hash':
        example['provider_calls'][1]['response_provenance']['worker_receipt_ref']['payload_sha256'] = 'sha256:' + '0' * 64
    seal(example)
    before = deepcopy(example)
    with original_worker_checks():
        expected = training_steps._build([example], host_observed=host_observed)
    actual = training_steps._build([example], host_observed=host_observed)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert example == before
    rows = actual[0]['train'] + actual[0]['eval']
    assert bool(rows) is (damage != 'worker_hash')
    if damage != 'none':
        assert actual[0]['analysis']


def test_step_proofs_are_released_before_reading_the_next_episode(monkeypatch):
    owners = []
    native_owner = training_worker_proof.WorkerProofs

    class TrackedProofs(native_owner):
        def __init__(self):
            super().__init__()
            owners.append(weakref.ref(self))

    monkeypatch.setattr(training_worker_proof, 'WorkerProofs', TrackedProofs)
    first, second = published_example(2), published_example(2)

    def examples():
        yield first
        gc.collect()
        assert owners and owners[-1]() is None
        # The next episode must validate its current bytes, not the prior proof.
        second['worker_observation_artifacts']['run']['payload']['observation_count'] = 1
        seal(second)
        yield second

    splits, _ = training_steps.build(examples())
    assert len(owners) == 2
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert splits['train'] or splits['eval']
    assert any(row['reason'] == 'worker_payload_hash_mismatch' for row in splits['analysis'])
