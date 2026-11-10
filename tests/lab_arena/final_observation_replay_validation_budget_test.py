"""Local validation must not consume all admission time for a final replay POST."""
from datetime import timedelta

import pytest

from lab_arena import capture_retention, training_observations as obs
from tests.lab_arena.final_observation_retention_test import accepted, observation_local
from tests.lab_arena.training_input_test import TOKEN
from tests.lab_arena.training_observations_test import observation, payload


def test_slow_local_validation_still_admits_one_bounded_final_post(tmp_path, monkeypatch):
    config, lease, _service, _objects, api, _clock = observation_local(tmp_path)
    retention = config.input_retention
    path = retention.observations.retain(lease, TOKEN, payload(observation()))
    monotonic = [100.0]
    monkeypatch.setattr(capture_retention.time, 'monotonic', lambda: monotonic[0])
    read = retention.observations._read

    def expensive_validation(*args):
        record = read(*args)
        monotonic[0] += capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS + 1
        return record

    monkeypatch.setattr(retention.observations, '_read', expensive_validation)
    retention._replay_batch()
    assert len(api.observation_calls) == 1
    assert 0 < api.observation_calls[0][3] <= capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS
    assert not path.exists()


def test_slow_final_records_use_one_post_each_and_rotate_between_batches(tmp_path, monkeypatch):
    config, lease, _service, _objects, api, _clock = observation_local(tmp_path)
    retention = config.input_retention
    paths = [retention.observations.retain(dict(lease, run_id='run-%d' % index),
             TOKEN, payload(observation())) for index in range(32)]
    monotonic = [100.0]
    monkeypatch.setattr(capture_retention.time, 'monotonic', lambda: monotonic[0])
    read = retention.observations._read
    validated = []

    def expensive_validation(*args):
        record = read(*args)
        validated.append(record['run_id'])
        monotonic[0] += capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS + 1
        return record

    def send(run, token, document, *, timeout_seconds=None):
        api.observation_calls.append((run, token, document, timeout_seconds))
        snapshot, _ = obs.build_snapshot(run, 'execute', document,
            lease_generation=lease['lease_generation'], claim_request_hash=lease['claim_request_hash'])
        return {'status': 'stored', 'schema_version': snapshot['schema_version'],
            'ref': obs.object_ref(run, lease['lease_generation'], lease['claim_request_hash']),
            'lease_generation': lease['lease_generation'], 'claim_request_hash': lease['claim_request_hash'],
            'payload_sha256': snapshot['payload_sha256']}

    monkeypatch.setattr(retention.observations, '_read', expensive_validation)
    api.training_observations = send
    for count in (30, 32):
        retention._replay_batch()
        assert len(validated) == len(api.observation_calls) == count
        assert sum(path.exists() for path in paths) == 32 - count
    assert len(set(validated)) == 32
    assert all(0 < call[3] <= capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS
               for call in api.observation_calls)
    retention._replay_batch()
    assert len(api.observation_calls) == 32


@pytest.mark.parametrize('condition', ['expired', 'stopped'])
def test_expiry_and_stop_after_slow_validation_still_prevent_post(tmp_path, monkeypatch, condition):
    config, lease, _service, _objects, api, clock = observation_local(tmp_path)
    retention = config.input_retention
    path = retention.observations.retain(lease, TOKEN, payload(observation()))
    original = path.read_bytes()
    monotonic = [100.0]
    monkeypatch.setattr(capture_retention.time, 'monotonic', lambda: monotonic[0])
    read = retention.observations._read

    def expensive_validation(*args):
        record = read(*args)
        monotonic[0] += capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS + 1
        if condition == 'expired':
            clock[0] += timedelta(seconds=capture_retention.FINAL_OBSERVATION_REPLAY_SECONDS)
        else:
            retention.stop()
        return record

    monkeypatch.setattr(retention.observations, '_read', expensive_validation)
    retention._replay_batch()
    retention._replay_batch()
    assert not api.observation_calls
    if condition == 'expired':
        assert not path.exists()
        audit = next((path.parent / 'audit').iterdir())
        assert audit.suffix == '.expired' and audit.read_bytes() == original
    else:
        assert path.read_bytes() == original


@pytest.mark.parametrize('denial', ['wrong_token', 'wrong_generation', 'wrong_claim', 'server_expired', 'immutable_conflict'])
def test_slow_validation_keeps_permanent_denials_terminal(tmp_path, monkeypatch, denial):
    config, lease, service, _objects, api, clock = observation_local(tmp_path)
    retention = config.input_retention
    path = retention.observations.retain(lease, TOKEN, payload(observation()))
    original = path.read_bytes()
    accepted(service, tmp_path / 'accepted')
    row = service._store.get_run('r1')
    if denial == 'wrong_token':
        row['lease_token_hash'] = 'sha256:' + '0' * 64
    elif denial == 'wrong_generation':
        row['lease_generation'] = 2
    elif denial == 'wrong_claim':
        row['claim_request_hash'] = 'invalid'
    elif denial == 'server_expired':
        clock[0] += timedelta(seconds=181)
    else:
        service.handle_training_observations('r1', TOKEN, payload())
    monotonic = [100.0]
    monkeypatch.setattr(capture_retention.time, 'monotonic', lambda: monotonic[0])
    read = retention.observations._read

    def expensive_validation(*args):
        record = read(*args)
        monotonic[0] += capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS + 1
        return record

    monkeypatch.setattr(retention.observations, '_read', expensive_validation)
    retention._replay_batch()
    retention._replay_batch()
    assert len(api.observation_calls) == 1 and not path.exists()
    audit = next((path.parent / 'audit').iterdir())
    assert audit.suffix == '.rejected' and audit.read_bytes() == original


def test_other_local_work_does_not_extend_final_network_budget(tmp_path, monkeypatch):
    config, lease, _service, _objects, api, clock = observation_local(tmp_path)
    retention = config.input_retention
    path = retention.observations.retain(lease, TOKEN, payload(observation()))
    monotonic = [100.0]
    monkeypatch.setattr(capture_retention.time, 'monotonic', lambda: monotonic[0])

    def expensive_clock():
        # _read calls this once. The expiry check calls it again after _read.
        monotonic[0] += capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS + 1
        return clock[0]

    config.clock = expensive_clock
    retention._replay_batch()
    assert not api.observation_calls and path.exists()
