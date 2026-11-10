"""A bounded replay worker must make progress at the real 30-second poll cadence."""
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import capture_retention, training_observations as obs
from tests.lab_arena.final_observation_retention_test import observation_local
from tests.lab_arena.training_input_test import TOKEN
from tests.lab_arena.training_observations_test import observation, payload


@pytest.mark.parametrize('post_seconds', [0.0, 5.0])
@pytest.mark.parametrize('failure', ['none', 'one_503', 'all_timeout'])
def test_poll_thirty_seconds_attempts_ten_slow_finals_before_terminal_fence(
        tmp_path, monkeypatch, post_seconds, failure):
    config, lease, _service, _objects, api, clock = observation_local(tmp_path)
    retention = config.input_retention
    assert config.claim_poll_seconds == 30
    paths = [retention.observations.retain(dict(lease, run_id='run-%d' % index),
             TOKEN, payload(observation())) for index in range(10)]
    original = {path: path.read_bytes() for path in paths}
    initial = clock[0]
    ticks = [180.0]
    next_poll = [210.0]
    clock[0] = initial + timedelta(seconds=ticks[0])
    monkeypatch.setattr(capture_retention, 'time', SimpleNamespace(monotonic=lambda: ticks[0]))
    attempted = []
    batch = [0]
    spans = []
    reader = retention.observations._read

    def set_time(value):
        ticks[0] = value
        clock[0] = initial + timedelta(seconds=value)

    def advance(seconds):
        target = ticks[0] + seconds
        while next_poll[0] <= target:
            point = next_poll[0]
            next_poll[0] += config.claim_poll_seconds
            set_time(point)
            # Use the actual poll function while its actual worker is alive.
            # It must retain that worker rather than start a second one.
            worker = retention._thread
            retention.poll()
            assert retention._thread is worker
        set_time(target)

    def expensive_validation(*args):
        record = reader(*args)
        advance(8.3)  # Measured local read time for the saved November10 45MB final.
        return record

    def send(run, token, document, *, timeout_seconds=None):
        require_before = ticks[0] < capture_retention.FINAL_OBSERVATION_REPLAY_SECONDS
        assert require_before, 'server terminal grace is closed'
        assert 0 < timeout_seconds <= capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS
        attempted.append((run, ticks[0], batch[0]))
        advance(post_seconds)
        if failure == 'all_timeout':
            raise httpx.ReadTimeout('synthetic lost final acknowledgement')
        if failure == 'one_503' and len(attempted) == 1:
            class Transient503(Exception):
                http_status = 503
            raise Transient503('synthetic transient final refusal')
        snapshot, _ = obs.build_snapshot(run, 'execute', document,
            lease_generation=lease['lease_generation'], claim_request_hash=lease['claim_request_hash'])
        return {'status': 'stored', 'schema_version': snapshot['schema_version'],
            'ref': obs.object_ref(run, lease['lease_generation'], lease['claim_request_hash']),
            'lease_generation': lease['lease_generation'], 'claim_request_hash': lease['claim_request_hash'],
            'payload_sha256': snapshot['payload_sha256']}

    monkeypatch.setattr(retention.observations, '_read', expensive_validation)
    api.training_observations = send
    while any(path.exists() for path in paths) and ticks[0] < 600:
        batch[0] += 1
        started = ticks[0]
        retention.poll()
        retention._thread.join(timeout=2)
        assert not retention._thread.is_alive()
        spans.append(ticks[0] - started)
        if any(path.exists() for path in paths):
            point = next_poll[0]
            next_poll[0] += config.claim_poll_seconds
            set_time(point)
    assert len({run for run, _, _ in attempted}) == 10
    if failure != 'all_timeout':
        assert len(attempted) == (11 if failure == 'one_503' else 10)
    assert all(at < 360 for _, at, _ in attempted)
    assert max(spans) <= capture_retention.INPUT_GRACE_SECONDS + 8.3 + post_seconds + 1e-9
    for group in {number for _, _, number in attempted}:
        runs = [run for run, _, number in attempted if number == group]
        assert len(runs) == len(set(runs))  # One pass, never retry a file in its batch.
    assert not any(path.exists() for path in paths)
    if failure == 'all_timeout':
        audits = list((paths[0].parent / 'audit').iterdir())
        assert len(audits) == 10 and all(path.suffix == '.expired' for path in audits)
        assert {path.read_bytes() for path in audits} == set(original.values())
    else:
        assert ticks[0] < 360
        if failure == 'none':
            assert ticks[0] == pytest.approx(180 + 10 * (8.3 + post_seconds))
        else:
            assert ticks[0] == pytest.approx((330 if post_seconds else 270) + 8.3 + post_seconds)
