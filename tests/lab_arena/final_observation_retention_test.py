"""Exact final evidence survives completion, process loss and bounded denials."""
import asyncio
import copy
from contextlib import contextmanager
from datetime import timedelta
import json
from pathlib import Path
import stat
import threading
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import capture_retention, runner, training_observations as obs
from lab_arena.api import create_app
from lab_arena.service import ServiceError
from tests.lab_arena.input_retention_test import local, finish_poll
from tests.lab_arena.training_input_grace_test import terminal_service
from tests.lab_arena.training_input_test import NOW, TOKEN, CLAIM_HASH
from tests.lab_arena.training_observations_test import observation, payload


def observation_local(tmp_path):
    config, lease, service, objects, api, clock = local(tmp_path)
    api.observation_calls = []
    api.fail_observations = 0
    def send(run_id, token, document, *, timeout_seconds=None):
        api.observation_calls.append((run_id, token, copy.deepcopy(document), timeout_seconds))
        if api.fail_observations:
            api.fail_observations -= 1
            raise httpx.ReadTimeout('synthetic final upload failure')
        try:
            return service.handle_training_observations(run_id, token, document)
        except ServiceError as error:
            return runner._HttpResponseDocument({'status': 'rejected'}, error.status)
    api.training_observations = send
    api.verify_training_observations = lambda *_: False
    return config, lease, service, objects, api, clock


def accepted(service, tmp_path):
    _other, _objects, terminal, _state = terminal_service(tmp_path)
    row = service._store.get_run('r1')
    expiry = row['lease_expires_at']
    row.update(copy.deepcopy(terminal))
    row['lease_expires_at'] = expiry


def test_original_final_is_private_before_http_and_survives_restart(tmp_path):
    config, lease, service, objects, api, _clock = observation_local(tmp_path)
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    state.worker_observations.append(observation())
    send = api.training_observations
    observed = []
    def checked(*args, **kwargs):
        paths = list((config.work_dir / 'pending-observations').glob('*.json'))
        assert len(paths) == 1
        assert stat.S_IMODE(paths[0].stat().st_mode) == 0o600
        assert stat.S_IMODE(paths[0].parent.stat().st_mode) == 0o700
        observed.append(paths[0].read_bytes())
        return send(*args, **kwargs)
    api.training_observations = checked
    api.fail_observations = 2
    summary = runner._record_training_observations(config, lease, TOKEN, state)
    assert summary['status'] == 'unavailable'
    assert len(observed) == 2 and observed[0] == observed[1]
    path = next((config.work_dir / 'pending-observations').glob('*.json'))
    original = json.loads(path.read_bytes())['observations']
    state.worker_observations.observations.clear()  # Old RAM cannot supply replay.
    accepted(service, tmp_path / 'accepted')
    before = copy.deepcopy(service._store.get_run('r1'))
    config.input_retention = capture_retention.InputRetention(config)
    finish_poll(config.input_retention)
    assert not path.exists()
    assert api.observation_calls[-1][2] == original
    assert 0 < api.observation_calls[-1][3] <= 5
    assert obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload'] == original
    assert service._store.get_run('r1') == before


@pytest.mark.parametrize('kind,cause', [('execute','accepted'), ('execute','model_error'), ('score','accepted'), ('score','judge_error')])
@pytest.mark.parametrize('seconds,allowed', [(179.999, True), (180, False), (181, False)])
def test_final_finish_grace_does_not_grant_other_authority(tmp_path, kind, cause, seconds, allowed):
    service, objects, row, _ = terminal_service(tmp_path, kind=kind, cause=cause, seconds=seconds)
    before = copy.deepcopy(row)
    try:
        result = service.handle_training_observations('r1', TOKEN, payload(observation()))
    except ServiceError as error:
        assert not allowed and error.status == 409
    else:
        assert allowed and result['status'] == 'stored'
        assert service.verify_training_observations('r1', TOKEN, result['payload_sha256']) == result['payload_sha256']
    assert row == before
    with pytest.raises(ServiceError):
        service._training_capture_run('r1', TOKEN)
    with pytest.raises(ServiceError):
        service.handle_training_observation_segment('r1', TOKEN, {})
    assert bool(list((tmp_path / 'objects').rglob('*.gz'))) is allowed


@pytest.mark.parametrize('delay,expected', [(179, 200), (180, 409), (181, 409)])
def test_queued_final_after_completion_has_exact_180_second_boundary(tmp_path, delay, expected):
    async def scenario():
        service, objects, row, _ = terminal_service(tmp_path, seconds=0)
        terminal = copy.deepcopy(row)
        row.update(status='leased', terminal_cause=None, result_doc=None, participation_accepted_at=None)
        current = [NOW]
        service._clock = lambda: current[0]
        entered, release = [], threading.Event()
        preauthorized = []
        authenticate = service._training_capture_run
        def auth(*args, **kwargs):
            preauthorized.append(row['status'])
            return authenticate(*args, **kwargs)
        service._training_capture_run = auth
        handle = service.handle_training_observations
        def blocked(*args):
            entered.append(True)
            if len(entered) <= 2:
                assert release.wait(5)
            return handle(*args)
        service.handle_training_observations = blocked
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            requests = []
            try:
                requests = [asyncio.create_task(client.post('/arena/v1/runs/r1/training-observations',
                    headers={'x-lab-arena-lease': TOKEN}, json=payload(observation()))) for _ in range(2)]
                for _ in range(200):
                    if len(entered) == 2:
                        break
                    await asyncio.sleep(.005)
                assert len(entered) == 2
                requests.append(asyncio.create_task(client.post('/arena/v1/runs/r1/training-observations',
                    headers={'x-lab-arena-lease': TOKEN}, json=payload(observation()))))
                for _ in range(200):
                    if len(preauthorized) >= 3:
                        break
                    await asyncio.sleep(.005)
                assert preauthorized[:3] == ['leased'] * 3 and len(entered) == 2
                row.update(terminal)
                current[0] = NOW + timedelta(seconds=delay)
            finally:
                release.set()
                responses = await asyncio.wait_for(asyncio.gather(*requests), 5)
            assert responses[-1].status_code == expected
            assert len(entered) == 3
            assert bool(list((tmp_path / 'objects').rglob('*.gz'))) is (expected == 200)
    asyncio.run(scenario())


@pytest.mark.parametrize('denial', ['expired_finish', 'wrong_token', 'immutable_conflict'])
def test_unusable_replay_parks_exact_final_for_audit(tmp_path, denial):
    config, lease, service, objects, api, clock = observation_local(tmp_path)
    original = payload(observation())
    path = config.input_retention.observations.retain(lease, TOKEN, original)
    original_bytes = path.read_bytes()
    accepted(service, tmp_path / 'accepted')
    row = service._store.get_run('r1')
    if denial == 'expired_finish':
        clock[0] += timedelta(seconds=181)
    elif denial == 'wrong_token':
        row['lease_token_hash'] = 'sha256:' + '0' * 64
    else:
        service.handle_training_observations('r1', TOKEN, payload())
    finish_poll(config.input_retention)
    attempts = 0 if denial == 'expired_finish' else 1
    assert not path.exists() and len(api.observation_calls) == attempts
    audit = next((path.parent / 'audit').iterdir())
    assert audit.read_bytes() == original_bytes
    assert audit.suffix == ('.expired' if denial == 'expired_finish' else '.rejected')
    finish_poll(config.input_retention)
    assert len(api.observation_calls) == attempts


@pytest.mark.parametrize('field,value', [('status','unavailable'), ('schema_version','wrong'), ('ref','wrong'),
    ('lease_generation', True), ('claim_request_hash','sha256:'+'f'*64), ('payload_sha256','sha256:'+'f'*64)])
def test_wrong_ack_keeps_exact_final(tmp_path, field, value):
    config, lease, service, objects, api, _ = observation_local(tmp_path)
    path = config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    before = path.read_bytes()
    send = api.training_observations
    def wrong(*args, **kwargs):
        return {**send(*args, **kwargs), field: value}
    api.training_observations = wrong
    assert not config.input_retention.observations.deliver(path)
    assert path.read_bytes() == before
    api.training_observations = send
    assert config.input_retention.observations.deliver(path)
    assert not path.exists()


def test_final_retention_has_no_second_replay_thread_or_unbounded_http(tmp_path):
    config, lease, _service, _objects, api, _ = observation_local(tmp_path)
    capture = config.input_retention
    input_path = capture.retain(lease, TOKEN, {'original': 'input'})
    final_path = capture.observations.retain(lease, TOKEN, payload(observation()))
    finish_poll(capture)
    assert not input_path.exists() and not final_path.exists()
    assert len(api.calls) == len(api.observation_calls) == 1
    assert 0 < api.calls[0][3] <= 5 and 0 < api.observation_calls[0][3] <= 5
    assert not hasattr(capture.observations, '_thread')

@pytest.mark.parametrize('damage', ['token','generation','claim','round','owner'])
def test_retained_identity_drift_never_uploads(tmp_path, damage):
    config, lease, _service, _objects, api, _ = observation_local(tmp_path)
    path = config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    data = json.loads(path.read_bytes())
    if damage == 'token': data['lease_token'] = 'b' * 64
    elif damage == 'generation': data['lease_generation'] = 2
    elif damage == 'claim': data['claim_request_hash'] = 'sha256:' + 'f' * 64
    elif damage == 'round': data['round_id'] = 'other-round'
    else: data['validator_hotkey'] = 'other-validator'
    path.write_text(json.dumps(data))
    finish_poll(config.input_retention)
    assert not api.observation_calls and not path.exists()
    assert next((path.parent/'audit').iterdir()).suffix == '.invalid'


def test_final_replay_keeps_privacy_and_truthful_incomplete_flags(tmp_path):
    config, lease, service, objects, _api, _ = observation_local(tmp_path)
    document = payload(observation())
    document['complete'] = False
    document['failures'] = ['capture_incomplete']
    document['observations'][0]['request']['admitted_parameters']['api_key'] = 'private-api-credential'
    document['observations'][0]['request']['admitted_parameters']['reasoning_content'] = 'private-hidden-reasoning'
    path = config.input_retention.observations.retain(lease, TOKEN, document)
    assert json.loads(path.read_bytes())['observations'] == document
    finish_poll(config.input_retention)
    saved = obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload']
    assert saved['complete'] is False and saved['failures'] == ['capture_incomplete']
    assert 'private-api-credential' not in json.dumps(saved)
    assert 'private-hidden-reasoning' not in json.dumps(saved)
    assert saved['observations'][0]['privacy']['redacted'] is True


def test_concurrent_controllers_share_the_retained_final_lock(tmp_path):
    config, lease, _service, _objects, api, _ = observation_local(tmp_path)
    path = config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    entered, release = threading.Event(), threading.Event()
    send = api.training_observations
    calls = []
    def paused(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return send(*args, **kwargs)
    api.training_observations = paused
    first, second = config.input_retention, capture_retention.InputRetention(config)
    try:
        first.poll()
        assert entered.wait(1)
        finish_poll(second)
        assert len(calls) == 1 and path.exists()
    finally:
        release.set()
        first._thread.join(timeout=3)
    assert not path.exists() and len(calls) == 1


@pytest.mark.parametrize('renewed', [False, True])
def test_foreground_keeps_publish_lock_until_exact_ack(tmp_path, monkeypatch, renewed):
    config, lease, service, _objects, api, clock = observation_local(tmp_path)
    if renewed:
        clock[0] += timedelta(minutes=10)
        service._store.get_run('r1')['lease_expires_at'] = (clock[0] + timedelta(minutes=2)).isoformat()
    capture = config.input_retention.observations
    retain = capture.retained_delivery
    paths = []
    @contextmanager
    def replay_after_publish(*args, **kwargs):
        with retain(*args, **kwargs) as retained:
            path, _descriptor = retained
            paths.append(path)
            original = path.read_bytes()
            # Use another controller: its stop flag is not the foreground's,
            # and flock must skip this inode instead of waiting or parking it.
            finish_poll(capture_retention.InputRetention(config))
            assert path.read_bytes() == original and not api.observation_calls
            yield retained
    monkeypatch.setattr(capture, 'retained_delivery', replay_after_publish)
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    state.worker_observations.append(observation())
    summary = runner._record_training_observations(config, lease, TOKEN, state)
    assert summary['status'] == 'stored'
    assert len(paths) == len(api.observation_calls) == 1
    assert not paths[0].exists()


def test_already_admitted_final_can_finish_after_completion_and_grace(tmp_path, monkeypatch):
    service, objects, row, _ = terminal_service(tmp_path, seconds=0)
    terminal = copy.deepcopy(row)
    row.update(status='leased', terminal_cause=None, result_doc=None, participation_accepted_at=None)
    clock = [NOW]
    service._clock = lambda: clock[0]
    build = obs.build_snapshot
    def delayed(*args, **kwargs):
        row.update(terminal)
        clock[0] = NOW + timedelta(seconds=181)
        return build(*args, **kwargs)
    monkeypatch.setattr(obs, 'build_snapshot', delayed)
    assert service.handle_training_observations('r1', TOKEN, payload(observation()))['status'] == 'stored'
    assert obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload']['complete'] is True
    with pytest.raises(ServiceError):
        service.handle_training_observations('r1', TOKEN, payload(observation()))

@pytest.mark.parametrize('damage', [None,'missing','status','resource_summary','runner'])
def test_cache_only_accepted_score_requires_exact_server_finish_event(tmp_path, damage):
    service, _objects, row, _ = terminal_service(tmp_path, kind='score')
    row['participation_accepted_at'] = None  # Native SQL221 all-hit semantics.
    event = service._store.get_input_finish_event('r1')
    if damage == 'missing':
        service._store.get_input_finish_event = lambda _: None
    elif damage == 'status':
        event['status'] = 'judge_error'
    elif damage == 'resource_summary':
        event['resource_summary']['provider_call_count'] = 1
    elif damage == 'runner':
        event['runner_hotkey'] = 'other-runner'
    if damage is None:
        assert service.handle_training_observations('r1', TOKEN, payload(observation()))['status'] == 'stored'
    else:
        with pytest.raises(ServiceError):
            service.handle_training_observations('r1', TOKEN, payload(observation()))
    with pytest.raises(ServiceError):
        service.handle_training_input('r1', TOKEN, {'original':'input'})
    assert row['participation_accepted_at'] is None

@pytest.mark.parametrize('damage', ['run','round','kind','event_kind','start','finish','future','naive',
    'before_start','boundary180','cancelled','token','claim_generation','claim_hash','claim_id','execute'])
def test_cache_only_finish_proof_rejects_every_unreliable_anchor(tmp_path, damage):
    service, _objects, row, round_state = terminal_service(tmp_path, kind='score')
    row['participation_accepted_at'] = None
    event = service._store.get_input_finish_event('r1')
    token = TOKEN
    if damage == 'run': event['run_id'] = 'other'
    elif damage == 'round': event['round_id'] = 'other'
    elif damage == 'kind': event['run_kind'] = 'execute'
    elif damage == 'event_kind': event['event_kind'] = 'provider.response'
    elif damage == 'start': event['started_at'] = NOW.isoformat()
    elif damage == 'finish': event['finished_at'] = NOW.isoformat()
    elif damage == 'future': event['created_at'] = (NOW+timedelta(seconds=1)).isoformat()
    elif damage == 'naive': event['created_at'] = NOW.replace(tzinfo=None).isoformat()
    elif damage == 'before_start': event['created_at'] = (NOW-timedelta(seconds=301)).isoformat()
    elif damage == 'boundary180': event['created_at'] = (NOW-timedelta(seconds=180)).isoformat()
    elif damage == 'cancelled': round_state['status'] = 'cancelled'
    elif damage == 'token': token = 'b'*64
    elif damage == 'claim_generation': row['lease_generation'] = 2
    elif damage == 'claim_hash': row['claim_request_hash'] = 'invalid'
    elif damage == 'claim_id': row['claim_request_id'] = '2'*32
    elif damage == 'execute':
        row['kind'] = row['claim_response']['kind'] = event['run_kind'] = 'execute'
    before = copy.deepcopy(row)
    with pytest.raises(ServiceError):
        service.handle_training_observations('r1', token, payload(observation()))
    assert row == before


@pytest.mark.parametrize('head_seconds', [0, 60])
def test_foreground_timeouts_leave_only_remaining_snapshot_replay_window(tmp_path, head_seconds):
    config, lease, _service, objects, api, clock = observation_local(tmp_path)
    timeouts = []
    def unavailable(_run, _token, _document, *, timeout_seconds=None):
        # Foreground keeps its existing 60-second phase bound. Replay alone is at most five seconds.
        delay = 60 if timeout_seconds is None else timeout_seconds
        timeouts.append(delay)
        clock[0] += timedelta(seconds=delay)
        raise httpx.ReadTimeout('synthetic queue longer than client allowance')
    def missing_head(*_args):
        clock[0] += timedelta(seconds=head_seconds)
        return False
    api.training_observations = unavailable
    api.verify_training_observations = missing_head
    api.provider = lambda *_args, **_kw: pytest.fail('capture cannot repeat paid work')
    api.complete = lambda *_args, **_kw: pytest.fail('capture cannot complete a run')
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    state.worker_observations.append(observation())
    result = runner._record_training_observations(config, lease, TOKEN, state)
    assert result['status'] == 'unavailable' and 'payload_sha256' not in result
    assert clock[0] == NOW + timedelta(seconds=120 + head_seconds) and timeouts == [60, 60]
    path = next((config.work_dir/'pending-observations').glob('*.json'))
    original = path.read_bytes()
    finish_poll(config.input_retention)
    if head_seconds == 0:
        assert len(timeouts) == 3 and timeouts[:2] == [60, 60] and 0 < timeouts[2] <= 5
        assert path.read_bytes() == original
    else:
        # Two full POST timeouts and a full HEAD consume the entire window.
        assert timeouts == [60, 60] and not path.exists()
    attempts = tuple(timeouts)
    assert not list((tmp_path/'objects').rglob('*.gz'))
    clock[0] = NOW + timedelta(seconds=180)
    finish_poll(config.input_retention)
    assert not path.exists() and tuple(timeouts) == attempts
    audit = next((path.parent/'audit').iterdir())
    assert audit.suffix == '.expired' and audit.read_bytes() == original
    finish_poll(config.input_retention)
    assert tuple(timeouts) == attempts


@pytest.mark.parametrize('seconds,allowed', [(179.999, True), (180, False), (181, False)])
def test_delayed_final_replay_keeps_snapshot_retention_bound_after_server_renewal(tmp_path, seconds, allowed):
    config, lease, service, _objects, api, clock = observation_local(tmp_path)
    path = config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    original = path.read_bytes()
    clock[0] = capture_retention._time(json.loads(original)['retained_at']) + timedelta(seconds=seconds)
    service._store.get_run('r1')['lease_expires_at'] = (clock[0] + timedelta(minutes=2)).isoformat()
    assert config.input_retention.observations.deliver(path, replay=True) is allowed
    assert not path.exists()
    assert len(api.observation_calls) == (1 if allowed else 0)
    if allowed:
        assert 0 < api.observation_calls[0][3] <= .001
    else:
        audit = next((path.parent / 'audit').iterdir())
        assert audit.suffix == '.expired' and audit.read_bytes() == original


def test_stopping_replay_keeps_active_foreground_final_delivery(tmp_path):
    config, lease, _service, objects, api, _clock = observation_local(tmp_path)
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    state.worker_observations.append(observation())
    document = state.worker_observations.document()
    path = config.input_retention.observations.retain(lease, TOKEN, document)
    original = path.read_bytes()
    config.input_retention.stop()  # Runner.close stops replay before workers drain.
    assert not config.input_retention.observations.deliver(path, replay=True)
    assert not api.observation_calls and path.read_bytes() == original
    summary = runner._record_training_observations(config, lease, TOKEN, state)
    assert summary['status'] == 'stored' and len(api.observation_calls) == 1
    assert not path.exists()
    assert obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload'] == document


def test_failed_foreground_context_releases_lock_and_keeps_final_for_restart(tmp_path):
    import os

    config, lease, _service, objects, api, _clock = observation_local(tmp_path)
    document = payload(observation())
    with pytest.raises(RuntimeError):
        with config.input_retention.observations.retained_delivery(lease, TOKEN, document) as (path, descriptor):
            original = path.read_bytes()
            raise RuntimeError('synthetic foreground process loss')
    assert path.read_bytes() == original and not api.observation_calls
    with pytest.raises(OSError):
        os.fstat(descriptor)
    finish_poll(capture_retention.InputRetention(config))
    assert not path.exists() and len(api.observation_calls) == 1
    assert obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload'] == document


@pytest.mark.parametrize('damage', ['naive', 'future', 'invalid', 'missing'])
def test_invalid_retention_clock_never_uploads_and_preserves_private_audit(tmp_path, damage):
    config, lease, _service, _objects, api, _clock = observation_local(tmp_path)
    path = config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    record = json.loads(path.read_bytes())
    if damage == 'naive':
        record['retained_at'] = NOW.replace(tzinfo=None).isoformat()
    elif damage == 'future':
        record['retained_at'] = (NOW + timedelta(seconds=1)).isoformat()
    elif damage == 'invalid':
        record['retained_at'] = 'invalid'
    else:
        del record['retained_at']
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    finish_poll(config.input_retention)
    assert not path.exists() and not api.observation_calls
    audit = next((path.parent / 'audit').iterdir())
    assert audit.suffix == '.invalid' and audit.read_bytes() == original


@pytest.mark.parametrize('field', ['clock', 'lease_expires_at'])
def test_invalid_initial_clock_or_claim_expiry_cannot_create_retained_final(tmp_path, field):
    config, lease, _service, _objects, api, clock = observation_local(tmp_path)
    if field == 'clock':
        clock[0] = NOW.replace(tzinfo=None)
    else:
        lease['lease_expires_at'] = 'invalid'
    with pytest.raises(ValueError):
        config.input_retention.observations.retain(lease, TOKEN, payload(observation()))
    assert not (config.work_dir / 'pending-observations').exists() and not api.observation_calls


def test_repeated_retention_never_refreshes_snapshot_replay_deadline(tmp_path):
    config, lease, _service, _objects, api, clock = observation_local(tmp_path)
    capture = config.input_retention.observations
    document = payload(observation())
    path = capture.retain(lease, TOKEN, document)
    original = path.read_bytes()
    clock[0] += timedelta(seconds=181)
    assert capture.retain(lease, TOKEN, document) == path
    assert path.read_bytes() == original
    assert not capture.deliver(path, replay=True) and not api.observation_calls
    assert next((path.parent / 'audit').iterdir()).read_bytes() == original


@pytest.mark.parametrize('seconds,allowed', [(179.999, True), (180, False)])
def test_legacy_pending_final_keeps_old_expiry_fence_without_new_retention_time(tmp_path, seconds, allowed):
    config, lease, service, _objects, api, clock = observation_local(tmp_path)
    capture = config.input_retention.observations
    document = payload(observation())
    path = capture.retain(lease, TOKEN, document)
    record = json.loads(path.read_bytes())
    record['schema_version'] = capture_retention._LEGACY_OBSERVATION_RECORD_SCHEMA
    del record['retained_at']
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    clock[0] = capture_retention._time(lease['lease_expires_at']) + timedelta(seconds=seconds)
    service._store.get_run('r1')['lease_expires_at'] = (clock[0] + timedelta(minutes=2)).isoformat()
    assert capture.retain(lease, TOKEN, document) == path and path.read_bytes() == original
    assert capture.deliver(path, replay=True) is allowed
    assert not path.exists() and len(api.observation_calls) == (1 if allowed else 0)
    if not allowed:
        audit = next((path.parent / 'audit').iterdir())
        assert audit.suffix == '.expired' and audit.read_bytes() == original
