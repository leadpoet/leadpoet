"""A bounded replay worker must make progress at the real 30-second poll cadence."""
import copy
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import capture_retention, training_observations as obs
from lab_arena.service import ServiceError
from tests.lab_arena.final_observation_retention_test import accepted, observation_local
from tests.lab_arena.training_input_test import TOKEN
from tests.lab_arena.training_observations_test import observation, payload


@pytest.mark.parametrize('replay_start,finished_seconds', [(180.0, 180.0), (210.0, 196.6), (226.6, 196.6)])
@pytest.mark.parametrize('post_seconds', [0.0, 5.0])
@pytest.mark.parametrize('failure', ['none', 'one_503', 'all_timeout'])
def test_poll_thirty_seconds_attempts_ten_slow_finals_before_terminal_fence(
        tmp_path, monkeypatch, post_seconds, failure, replay_start, finished_seconds):
    config, lease, service, objects, api, clock = observation_local(tmp_path)
    retention = config.input_retention
    assert config.claim_poll_seconds == 30
    paths = [retention.observations.retain(dict(lease, run_id='run-%d' % index),
             TOKEN, payload(observation())) for index in range(10)]
    original = {path: path.read_bytes() for path in paths}
    initial = clock[0]
    # Retained original claim expiry stays at120; the server owns the renewed
    # current lease and the real completion-based terminal window.
    accepted(service, tmp_path / 'accepted')
    row = service._store.get_run('r1')
    row['lease_expires_at'] = (initial + timedelta(seconds=3600)).isoformat()
    row['result_doc']['finished_at'] = (initial + timedelta(seconds=finished_seconds)).isoformat().replace('+00:00', 'Z')
    row['participation_accepted_at'] = (initial + timedelta(seconds=finished_seconds)).isoformat()
    rows = {'run-%d' % index: copy.deepcopy(row) for index in range(10)}
    for run, current in rows.items():
        current['run_id'] = current['claim_response']['run_id'] = run
    service._store.get_run = rows.get
    before = copy.deepcopy(rows)
    server_deadline = finished_seconds + capture_retention.INPUT_GRACE_SECONDS
    ticks = [replay_start]
    next_poll = [replay_start + 30]
    clock[0] = initial + timedelta(seconds=ticks[0])
    monkeypatch.setattr(capture_retention, 'time', SimpleNamespace(monotonic=lambda: ticks[0]))
    attempted = []
    denials = []
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
        try:
            service._training_final_observation_capture_run(run, token)
        except ServiceError as error:
            denials.append((run, ticks[0], error.code, error.status))
            return {'status': 'rejected', 'http_status': error.status}
        assert ticks[0] < server_deadline
        assert 0 < timeout_seconds <= capture_retention.INPUT_REPLAY_TIMEOUT_SECONDS
        attempted.append((run, ticks[0], batch[0]))
        if failure == 'one_503' and len(attempted) == 1:
            advance(post_seconds)
            class Transient503(Exception):
                http_status = 503
            raise Transient503('synthetic transient final refusal before object commit')
        # Actual native storage precedes the ACK delay. A lost ACK cannot be
        # treated as lost data when the admitted immutable write succeeded.
        response = service.handle_training_observations(run, token, document)
        advance(post_seconds)
        if failure == 'all_timeout':
            raise httpx.ReadTimeout('synthetic lost final acknowledgement')
        return response

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
    capacity_limit = failure == 'one_503' and post_seconds == 5 and replay_start == 226.6
    if failure != 'all_timeout':
        assert len(attempted) == (11 if failure == 'one_503' and not capacity_limit else 10)
    assert all(at < server_deadline for _, at, _ in attempted)
    assert max(spans) <= capture_retention.INPUT_GRACE_SECONDS + 8.3 + post_seconds + 1e-9
    for group in {number for _, _, number in attempted}:
        runs = [run for run, _, number in attempted if number == group]
        assert len(runs) == len(set(runs))
    assert not any(path.exists() for path in paths)
    assert rows == before
    audits = list((paths[0].parent / 'audit').iterdir()) if (paths[0].parent / 'audit').exists() else []
    if failure == 'all_timeout' or capacity_limit:
        assert len(audits) == (10 if failure == 'all_timeout' else 1)
        assert all(path.suffix == '.rejected' for path in audits)
        assert {path.read_bytes() for path in audits} <= set(original.values())
        assert denials and all(code == 'lease_expired' and status == 409 for _, _, code, status in denials)
    else:
        assert not audits and not denials
    captured = list(objects._root.rglob('*.gz'))
    assert len(captured) == (9 if capacity_limit else 10)
    if failure == 'one_503' and post_seconds == 5 and replay_start == 210:
        assert ticks[0] == pytest.approx(373.3)
        assert server_deadline == pytest.approx(376.6)
