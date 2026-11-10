"""Original input survives capture outages without owning paid work or completion."""
import copy
from datetime import timedelta
import errno
import json
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import capture_retention, contracts, runner, runtime, training_input
from lab_arena.service import ServiceError
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config, valid_company


def local(tmp_path):
    current = [NOW]
    service, objects = service_for(tmp_path)
    row = service._store.get_run('r1')
    row.update(claim_request_id='1' * 32,
        claim_response={'status': 'leased', 'request_id': '1' * 32, 'run_id': 'r1', 'lease_generation': 1, 'kind': 'execute'})
    service._store.get_round_capture_state = lambda _: {'round_id': row['round_id'], 'status': 'stage1'}
    service._clock = lambda: current[0]
    class Api:
        def __init__(self):
            self.calls = []
            self.fail = self.lost_ack = 0
            self.damage = {}
        def training_input(self, run_id, token, document, *, timeout_seconds=None):
            self.calls.append((run_id, token, copy.deepcopy(document), timeout_seconds))
            if self.fail:
                self.fail -= 1
                raise httpx.ReadTimeout('synthetic capture outage')
            try:
                response = service.handle_training_input(run_id, token, document)
            except ServiceError as exc:
                return runner._HttpResponseDocument({'status': 'rejected', 'code': exc.code}, exc.status)
            if self.lost_ack:
                self.lost_ack -= 1
                raise httpx.ReadTimeout('synthetic lost acknowledgment')
            return {**response, **self.damage}
    api = Api()
    config = SimpleNamespace(work_dir=tmp_path / 'work', identity=SimpleNamespace(hotkey='validator'),
        round_id=row['round_id'], clock=lambda: current[0], api=api, claim_poll_seconds=30)
    config.input_retention = capture_retention.InputRetention(config)
    active = {key: row[key] for key in ('run_id', 'round_id', 'kind', 'lease_generation', 'claim_request_hash', 'lease_expires_at')}
    return config, active, service, objects, api, current


def finish_poll(retention):
    retention._next_poll = 0
    retention.poll()
    retention._thread.join(timeout=2)
    assert not retention._thread.is_alive()


def test_lost_ack_and_process_restart_replay_identical_original_input(tmp_path):
    config, active, _service, objects, api, _clock = local(tmp_path)
    original = {'icp': {'prompt': 'Original é\ninput'}, 'catalog': [1, 2, 3]}
    path = config.input_retention.retain(active, TOKEN, original)
    original_bytes = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    api.lost_ack = 1
    with pytest.raises(httpx.ReadTimeout):
        config.input_retention.deliver(path, document=original)
    remote = objects.get(training_input.object_ref('r1', 1))
    assert path.read_bytes() == original_bytes
    # A fresh controller reads the file, not the old process's Python objects.
    config.input_retention = capture_retention.InputRetention(config)
    original['icp']['prompt'] = 'Changed after the saved runtime input'
    finish_poll(config.input_retention)
    assert not path.exists()
    assert objects.get(training_input.object_ref('r1', 1)) == remote
    assert api.calls[0][:3] == api.calls[1][:3]
    assert 0 < api.calls[1][3] <= 5
    assert training_input.read_input(objects, 'r1', 'execute', 1, CLAIM_HASH)['input']['icp']['prompt'] == 'Original é\ninput'


@pytest.mark.parametrize('field,value', [
    ('status', 'unavailable'), ('schema_version', 'wrong'), ('ref', 'wrong'),
    ('lease_generation', 2), ('lease_generation', True), ('claim_request_hash', 'sha256:' + 'd' * 64),
    ('input_sha256', 'sha256:' + 'e' * 64), ('privacy', {'redacted': True, 'redactions': ['credential']}),
    ('privacy', {'redacted': 0, 'redactions': []}),
])
def test_only_canonical_bound_ack_can_delete_the_retained_input(tmp_path, field, value):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': [1, 'exact']})
    before = path.read_bytes()
    api.damage = {field: value}
    assert not config.input_retention.deliver(path)
    assert path.read_bytes() == before
    api.damage = {}
    assert config.input_retention.deliver(path)
    assert not path.exists()


def test_expected_sanitized_ack_does_not_claim_raw_input_is_exactly_usable(tmp_path):
    config, active, _service, objects, _api, _clock = local(tmp_path)
    original = {'api_key': 'private original value', 'ordinary': 'retained'}
    path = config.input_retention.retain(active, TOKEN, original)
    assert json.loads(path.read_bytes())['input'] == original
    assert config.input_retention.deliver(path)
    saved = training_input.read_input(objects, 'r1', 'execute', 1, CLAIM_HASH)
    assert saved['privacy'] == {'redacted': True, 'redactions': ['credential']}
    assert saved['input']['api_key'] == '[REDACTED]'
    assert not path.exists()


def test_conflicting_original_input_never_replaces_the_first_record(tmp_path):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='conflict'):
        config.input_retention.retain(active, TOKEN, {'original': 2})
    with pytest.raises(ValueError, match='differs'):
        config.input_retention.deliver(path, document={'original': 2})
    assert path.read_bytes() == before and not api.calls


def test_concurrent_controllers_have_only_one_bounded_replay_operation(tmp_path):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    entered, release = threading.Event(), threading.Event()
    original = api.training_input
    calls = []
    def blocked(*args, **kwargs):
        calls.append(kwargs['timeout_seconds'])
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    api.training_input = blocked
    first, second = config.input_retention, capture_retention.InputRetention(config)
    try:
        first.poll()
        assert entered.wait(1)
        start = time.monotonic()
        for _ in range(5):
            first._next_poll = second._next_poll = 0
            first.poll()
            second.poll()
        assert time.monotonic() - start < .1
        second._thread.join(timeout=1)
        assert len(calls) == 1 and 0 < calls[0] <= 5
        assert path.exists()
    finally:
        release.set()
        first._thread.join(timeout=2)
    assert not path.exists() and len(api.calls) == 1
    assert first._thread.daemon


def test_expired_input_is_preserved_for_audit_without_more_automatic_http(tmp_path, capsys):
    config, active, _service, _objects, api, clock = local(tmp_path)
    original = {'original': 'no reconstruction'}
    path = config.input_retention.retain(active, TOKEN, original)
    clock[0] += timedelta(minutes=6)
    finish_poll(config.input_retention)
    assert not path.exists() and not api.calls
    audit = list((path.parent / 'audit').iterdir())
    assert len(audit) == 1 and json.loads(audit[0].read_bytes())['input'] == original
    assert stat.S_IMODE(audit[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(audit[0].parent.stat().st_mode) == 0o700
    finish_poll(config.input_retention)
    assert not api.calls and audit[0].exists()
    assert 'capture unavailable: expired' in capsys.readouterr().err


def test_stop_does_not_start_a_replay_or_remove_private_evidence(tmp_path):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': True})
    config.input_retention.stop()
    config.input_retention.poll()
    assert config.input_retention._thread is None
    assert not api.calls and path.exists()


@pytest.mark.parametrize('damage', ['mode', 'symlink', 'corrupt', 'foreign_validator'])
def test_local_file_security_and_binding_fail_closed(tmp_path, damage):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': True})
    external = tmp_path / 'unrelated'
    external.write_bytes(b'not input evidence')
    if damage == 'mode': path.chmod(0o644)
    elif damage == 'symlink':
        path.unlink()
        path.symlink_to(external)
    elif damage == 'corrupt': path.write_bytes(b'{broken')
    else: config.identity.hotkey = 'different-validator'
    finish_poll(config.input_retention)
    assert not api.calls
    assert external.read_bytes() == b'not input evidence'
    assert path.exists() or (path.parent / 'audit').exists()


def runtime_case(tmp_path, *, controller, failures, monkeypatch, storage_fault=None):
    service, objects = service_for(tmp_path)
    active = lease()
    active.update(lease_token=TOKEN)
    current = service._store.get_run('r1')
    current.update(round_id=active['round_id'], lease_expires_at=active['lease_expires_at'],
        claim_request_id='1' * 32,
        claim_response={'status': 'leased', 'request_id': '1' * 32, 'run_id': 'r1', 'lease_generation': 1, 'kind': 'execute'})
    service._store.get_round_capture_state = lambda _: {'round_id': active['round_id'], 'status': 'stage1'}
    class Api(FakeApi):
        def __init__(self):
            super().__init__([active])
            self.remaining_failures = failures
            self.input_calls = []
        def training_input(self, run_id, token, document, *, timeout_seconds=None):
            self.input_calls.append(copy.deepcopy(document))
            if self.remaining_failures:
                self.remaining_failures -= 1
                raise httpx.ReadTimeout('synthetic input outage')
            try:
                return service.handle_training_input(run_id, token, document)
            except ServiceError as exc:
                return runner._HttpResponseDocument({'status': 'rejected', 'code': exc.code}, exc.status)
        def complete(self, envelope):
            result = super().complete(envelope)
            current.update(status='accepted', terminal_cause='accepted', result_doc=copy.deepcopy(envelope['body']['result']),
                participation_accepted_at=config.clock().isoformat())
            return result
    api = Api()
    sandbox = BridgingRuntime(output={'companies': [valid_company(1)]}, calls=1)
    (tmp_path / 'work').mkdir()
    config = make_config(tmp_path, api, sandbox)
    service._clock = config.clock
    if storage_fault == 'permission':
        (config.work_dir / 'pending-inputs').mkdir(mode=0o755)
    elif storage_fault == 'disk_full':
        original = capture_retention.tempfile.mkstemp
        def fail(*args, **kwargs):
            if Path(kwargs['dir']).name == 'pending-inputs':
                raise OSError(errno.ENOSPC, 'synthetic disk full')
            return original(*args, **kwargs)
        monkeypatch.setattr(capture_retention.tempfile, 'mkstemp', fail)
    owner = runner.Runner(config) if controller else None
    if owner is not None:
        assert owner.run_once() == 1
        envelope = api.completions[0]
    else:
        envelope = runner.AssignmentExecutor(config).execute(active, TOKEN, active['icp'])
        assert api.complete(envelope)['status'] == 'accepted'
    return config, api, sandbox, service, objects, envelope, owner


@pytest.mark.parametrize('controller', [False, True])
def test_initial_two_and_final_one_outages_survive_normal_controller_restart(tmp_path, controller, monkeypatch):
    config, api, sandbox, service, objects, envelope, owner = runtime_case(
        tmp_path, controller=controller, failures=3, monkeypatch=monkeypatch)
    assert len(api.input_calls) == 3
    assert len(api.provider_frames) == len(api.completions) == 1
    assert envelope['body']['result']['terminal_status'] == 'accepted'
    pending = list((config.work_dir / 'pending-inputs').glob('*.json'))
    assert len(pending) == 1
    assert json.loads(pending[0].read_bytes())['input'] == sandbox.input_documents[0]
    assert not list(config.work_dir.glob('run-*'))
    config.input_retention = capture_retention.InputRetention(config)
    finish_poll(config.input_retention)
    saved = training_input.read_input(objects, 'r1', 'execute', 1, CLAIM_HASH)
    assert saved['input'] == sandbox.input_documents[0]
    assert saved['input_sha256'] == contracts.document_hash(sandbox.input_documents[0])
    assert not pending[0].exists()
    assert len(api.provider_frames) == len(api.completions) == 1
    if owner is not None:
        owner.close()


@pytest.mark.parametrize('fault', ['disk_full', 'permission'])
def test_local_retention_failure_keeps_only_original_input_without_changing_execution(tmp_path, fault, monkeypatch):
    config, api, sandbox, _service, _objects, envelope, _owner = runtime_case(
        tmp_path, controller=False, failures=3, storage_fault=fault, monkeypatch=monkeypatch)
    assert envelope['body']['result']['terminal_status'] == 'accepted'
    assert len(api.provider_frames) == len(api.completions) == 1
    paths = list(config.work_dir.glob('run-*'))
    assert len(paths) == 1
    files = list(paths[0].rglob('*'))
    assert {p.relative_to(paths[0]).as_posix() for p in files} == {'input', 'input/' + runtime.INPUT_FILE_NAME}
    original = paths[0] / 'input' / runtime.INPUT_FILE_NAME
    assert json.loads(original.read_bytes()) == sandbox.input_documents[0]
    assert stat.S_IMODE(paths[0].stat().st_mode) == 0o700
    assert stat.S_IMODE(original.stat().st_mode) == 0o600


def test_active_replay_cannot_remove_only_runtime_input_during_cleanup(tmp_path):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    original = {'original': 'the input actually written for runtime'}
    config.input_retention.retain(active, TOKEN, original)
    run_dir = config.work_dir / 'run-original'
    input_dir = run_dir / 'input'
    input_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    input_path = input_dir / runtime.INPUT_FILE_NAME
    raw = json.dumps(original, sort_keys=True).encode()
    input_path.write_bytes(raw)
    (run_dir / 'source').mkdir()
    (run_dir / 'source' / 'large-scratch').write_bytes(b'source scratch')
    (run_dir / 'output').mkdir()
    (run_dir / 'output' / 'result.json').write_text('{}')
    entered, release = threading.Event(), threading.Event()
    send = api.training_input
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return send(*args, **kwargs)
    api.training_input = blocked
    try:
        config.input_retention.poll()
        assert entered.wait(1)
        assert not config.input_retention.contains(active, TOKEN, original)
        runner._cleanup_runtime_input(run_dir, input_path, preserve=True)
        assert input_path.read_bytes() == raw
        assert {p.relative_to(run_dir).as_posix() for p in run_dir.rglob('*')} == {
            'input', 'input/' + runtime.INPUT_FILE_NAME}
    finally:
        release.set()
        config.input_retention._thread.join(timeout=2)
    assert input_path.read_bytes() == raw


def test_audit_link_is_fsynced_before_pending_name_is_removed(tmp_path, monkeypatch):
    config, active, _service, _objects, _api, clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    clock[0] += timedelta(minutes=6)
    events = []
    sync = capture_retention.sync_directory
    unlink = Path.unlink
    def record_sync(directory):
        events.append(('sync', directory.name))
        sync(directory)
    def record_unlink(candidate, *args, **kwargs):
        if candidate == path:
            events.append(('unlink', candidate.parent.name))
        return unlink(candidate, *args, **kwargs)
    monkeypatch.setattr(capture_retention, 'sync_directory', record_sync)
    monkeypatch.setattr(Path, 'unlink', record_unlink)
    finish_poll(config.input_retention)
    assert events == [('sync', 'audit'), ('unlink', 'pending-inputs'), ('sync', 'pending-inputs')]


def test_background_start_failure_cannot_escape_into_claim_or_drain(tmp_path, monkeypatch, capsys):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    with monkeypatch.context() as patch:
        def fail_start(_thread):
            raise RuntimeError('synthetic thread resource failure')
        patch.setattr(threading.Thread, 'start', fail_start)
        config.input_retention.poll()
    assert config.input_retention._thread is None
    assert path.exists() and not api.calls
    assert 'RuntimeError' in capsys.readouterr().err
    finish_poll(config.input_retention)
    assert not path.exists() and len(api.calls) == 1


def test_archive_link_survives_restart_before_pending_unlink(tmp_path):
    import os
    config, active, _service, _objects, api, clock = local(tmp_path)
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    audit = capture_retention.private_directory(path.parent, 'audit', create=True)
    target = audit / (path.stem + '.expired')
    os.link(path, target, follow_symlinks=False)
    assert path.stat().st_ino == target.stat().st_ino
    before = target.read_bytes()
    clock[0] += timedelta(minutes=6)
    config.input_retention = capture_retention.InputRetention(config)
    finish_poll(config.input_retention)
    assert not path.exists()
    assert target.read_bytes() == before and not api.calls
    finish_poll(config.input_retention)
    assert target.exists() and not api.calls


@pytest.mark.parametrize('denial', ['expired_terminal', 'rotated_token', 'immutable_conflict'])
def test_permanent_capture_denials_park_original_and_stop_replay(tmp_path, denial):
    from lab_arena.store import hash_lease_token
    from tests.lab_arena.training_input_grace_test import terminal_service
    config, active, service, objects, api, _clock = local(tmp_path)
    original = {'original': 'must remain available for audit'}
    path = config.input_retention.retain(active, TOKEN, original)
    row = service._store.get_run('r1')
    remote_before = None
    if denial == 'expired_terminal':
        _other, _objects, terminal, _state = terminal_service(tmp_path / 'terminal', seconds=181)
        row.update(status=terminal['status'], terminal_cause=terminal['terminal_cause'], result_doc=terminal['result_doc'])
    elif denial == 'rotated_token':
        row.update(lease_generation=2, claim_request_hash='sha256:' + 'd' * 64, lease_token_hash=hash_lease_token('b' * 64))
    else:
        service.handle_training_input('r1', TOKEN, {'existing': 'immutable other input'})
        remote_before = objects.get(training_input.object_ref('r1', 1))
    finish_poll(config.input_retention)
    assert len(api.calls) == 1 and not path.exists()
    audit = list((path.parent / 'audit').iterdir())
    assert len(audit) == 1 and audit[0].suffix == '.rejected'
    assert json.loads(audit[0].read_bytes())['input'] == original
    finish_poll(config.input_retention)
    assert len(api.calls) == 1 and audit[0].exists()
    if remote_before is not None:
        assert objects.get(training_input.object_ref('r1', 1)) == remote_before
    else:
        assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_active_capture_never_blocks_claim_poll_drain_or_exit(tmp_path):
    from tests.lab_arena.training_input_test import capture_ack
    api = FakeApi([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0))
    active = lease()
    active['lease_token'] = TOKEN
    path = config.input_retention.retain(active, TOKEN, {'original': 1})
    entered, release = threading.Event(), threading.Event()
    calls = []
    def blocked(run_id, token, document, *, timeout_seconds=None):
        calls.append(timeout_seconds)
        entered.set()
        assert release.wait(2)
        return capture_ack(run_id, document)
    api.training_input = blocked
    owner = runner.Runner(config)
    try:
        config.input_retention.poll()
        assert entered.wait(1)
        started = time.monotonic()
        assert owner.run_once(max_claims=0) == 0
        owner.close()
        assert time.monotonic() - started < .2
        assert config.input_retention._thread.is_alive() and config.input_retention._thread.daemon
        assert path.exists() and len(calls) == 1 and 0 < calls[0] <= 5
        config.input_retention._next_poll = 0
        config.input_retention.poll()
        assert len(calls) == 1 and 0 < calls[0] <= 5
    finally:
        release.set()
        config.input_retention._thread.join(timeout=2)
    assert not path.exists() and not api.provider_frames and not api.completions


def pending_burst(tmp_path, count, *, catalog_bytes=0):
    from tests.lab_arena.training_input_grace_test import terminal_service
    config, active, service, objects, api, clock = local(tmp_path)
    _template_service, _objects, template, _state = terminal_service(tmp_path / 'terminal-template', seconds=0, expiry_seconds=1)
    template['participation_accepted_at'] = NOW.isoformat()
    rows, paths = {}, []
    for index in range(count):
        run_id = 'burst-' + str(index)
        row = copy.deepcopy(template)
        row.update(run_id=run_id, claim_request_id='%032x' % (index + 1))
        row['claim_response'].update(run_id=run_id, request_id=row['claim_request_id'])
        rows[run_id] = row
        claimed = {key: row[key] for key in active}
        paths.append(config.input_retention.retain(claimed, TOKEN,
            {'position': index, 'catalog': 'public tool schema ' * (catalog_bytes // 19)}))
    service._store.get_run = lambda run_id: rows.get(run_id)
    return config, rows, paths, service, objects, api, clock


def test_recovered_gateway_drains_more_than_six_distinct_pending_inputs_per_poll(tmp_path):
    config, _rows, paths, _service, _objects, api, _clock = pending_burst(tmp_path, 12)
    finish_poll(config.input_retention)
    assert len(api.calls) == 12
    assert len({row[0] for row in api.calls}) == 12
    assert not any(path.exists() for path in paths)


def test_replay_batch_uses_bounded_attempts_and_never_retries_a_record_inside_it(tmp_path, monkeypatch):
    config, _rows, paths, _service, _objects, api, _clock = pending_burst(tmp_path, 8)
    monotonic = [0.0]
    monkeypatch.setattr(capture_retention, 'time', SimpleNamespace(monotonic=lambda: monotonic[0]))
    send = api.training_input
    def timed(*args, **kwargs):
        try:
            return send(*args, **kwargs)
        finally:
            monotonic[0] += 2
    api.training_input = timed
    api.fail = 1
    finish_poll(config.input_retention)
    assert len(api.calls) == 8
    assert len({row[0] for row in api.calls}) == 8
    assert [row[3] for row in api.calls] == [5] * 8
    assert sum(path.exists() for path in paths) == 1
    assert not config.input_retention._thread.is_alive()
    finish_poll(config.input_retention)
    finish_poll(config.input_retention)
    finish_poll(config.input_retention)
    assert len(api.calls) == 9 and not any(path.exists() for path in paths)


def test_real_process_restart_and_interprocess_lock_preserve_original_input(tmp_path):
    import subprocess
    import sys
    config, active, _service, objects, _api, _clock = local(tmp_path)
    original = {'runtime_document': ['Original process input', {'exact': 17}]}
    path = config.input_retention.retain(active, TOKEN, original)
    program = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from lab_arena import capture_retention
from tests.lab_arena.training_input_grace_test import terminal_service
from tests.lab_arena.training_input_test import NOW
root = Path(sys.argv[1])
service, objects, row, state = terminal_service(root)
calls = []
class Api:
    def training_input(self, run_id, token, document, *, timeout_seconds=None):
        calls.append(1)
        return service.handle_training_input(run_id, token, document)
config = SimpleNamespace(work_dir=root/'work', identity=SimpleNamespace(hotkey='validator'),
    round_id='arena-test', clock=lambda:NOW, api=Api(), claim_poll_seconds=30)
retention = capture_retention.InputRetention(config)
retention.poll()
retention._thread.join(timeout=5)
assert not retention._thread.is_alive()
print(len(calls))
'''
    def restart():
        result = subprocess.run([sys.executable, '-c', program, str(tmp_path)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return int(result.stdout.strip().splitlines()[-1])
    with capture_retention.locked_record(path, training_input.MAX_OBJECT_BYTES) as descriptor:
        assert descriptor is not None
        assert restart() == 0
        assert path.exists()
    assert restart() == 1
    assert not path.exists()
    assert training_input.read_input(objects, 'r1', 'execute', 1, CLAIM_HASH)['input'] == original


@pytest.mark.parametrize('raises', [False, True])
def test_http_413_is_parked_privately_and_never_retried(tmp_path, raises):
    config, active, _service, _objects, api, _clock = local(tmp_path)
    original = {'original': 'valid input refused by a proxy size limit'}
    path = config.input_retention.retain(active, TOKEN, original)
    def rejected(run_id, token, document, *, timeout_seconds=None):
        api.calls.append((run_id, token, document, timeout_seconds))
        if raises:
            raise runner.RunnerError('synthetic size refusal', http_status=413)
        return runner._HttpResponseDocument({'status': 'rejected'}, 413)
    api.training_input = rejected
    finish_poll(config.input_retention)
    assert not path.exists() and len(api.calls) == 1
    audit = list((path.parent / 'audit').iterdir())
    assert len(audit) == 1 and audit[0].suffix == '.rejected'
    assert json.loads(audit[0].read_bytes())['input'] == original
    assert stat.S_IMODE(audit[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(audit[0].parent.stat().st_mode) == 0o700
    finish_poll(config.input_retention)
    assert len(api.calls) == 1 and audit[0].exists()
