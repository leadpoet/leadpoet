"""One claim pause covers both canonical services; failures never release it."""
from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from scripts import restart_arena_pair as pair
from scripts.lab_arena_restart_claim_guard import _commitments, _identity
from scripts import lab_arena_restart_guard_handoff as handoff

SHA = "a" * 40
INV = "gateway-1234-5678"


def state(phase="gateway_ready", generation=7):
    guard, owner = _commitments(*_identity(SHA, INV))
    return dict(schema_version="leadpoet.lab_arena.restart_guard_state.v1",
                candidate_commit=SHA, guard_present=True, guard_active=True, paused=True, operator_paused=False,
                guard_generation=generation, restart_scope="all", restart_phase=phase,
                guard_commitment=guard, owner_commitment=owner,
                guard_expires_at=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat(),
                drain=dict(schema_version="leadpoet.lab_arena.restart_drain_state.v1",
                           captured_count=2, accepted_receipt_count=1, reported_terminal_receipt_count=1,
                           still_leased_count=0, lost_or_mutated_count=0, current_leased_count=0,
                           pending_retry_count=1, preserved=True))


@pytest.fixture(autouse=True)
def candidate_controller_blob(monkeypatch):
    original = subprocess.check_output
    def committed_blob(argv, **kwargs):
        if argv == ['git', 'show', SHA+':gw_restart.sh']:
            return (Path(__file__).parents[1]/'gw_restart.sh').read_bytes()
        return original(argv, **kwargs)
    monkeypatch.setattr(subprocess, 'check_output', committed_blob)


class FakePair(pair.Pair):
    def __init__(self, tmp_path):
        super().__init__(Namespace(commit=SHA, logs=tmp_path, gateway="gateway", validator="validator",
                                   identity=tmp_path/"identity"))
        self.events = []
        self.phase = "gateway_ready"
        self.fail_at = None
        self.generation_value = 7
        self.generation = 7
        self.receipt_state = state()

    def remote(self, host, command, **kwargs):
        if "sha256sum" in command:
            self.events.append("installed-preflight")
            if self.fail_at == "unsupported":
                raise PairErrorForTest()
        elif "gw_restart.sh --commit" in command:
            self.events.append("gateway")
            assert "GATEWAY_PAIRED_RESTART=1" in command
            if self.fail_at == "gateway":
                raise PairErrorForTest()
            kwargs["stdout"].write(pair.PREFIX + json.dumps(dict(invocation=INV, state=self.receipt_state))+"\n")
        else:
            self.events.append("validator-command")
            assert host == "validator" and kwargs["input"]
            assert b"SERVICE_KEY" not in kwargs["input"]
            assert "--check-only" not in command  # Only canonical script owns preflight.
            assert "git show " + SHA + ":validator_restart.sh" in command
            assert 'VALIDATOR_PAIR_PERMIT="$permit"' in command
        return subprocess.CompletedProcess([], 0)

    def guard(self, action, *extra):
        self.events.append(action)
        if self.fail_at == action:
            raise PairErrorForTest()
        if action == "drain":
            return dict(guard_generation=self.generation_value, restart_scope="all", preserved=True)
        if action == "authorize":
            assert self.phase == "gateway_ready"
            self.phase = "validator_destructive"
        elif action == "ready":
            assert self.phase == "validator_destructive"
            self.phase = "validator_ready"
        elif action == "release":
            assert self.phase == "validator_ready"
        return state(self.phase)

    def wait_validator_ready(self):
        self.events.append('operational-ready')
        if self.fail_at == 'operational-ready':
            raise PairErrorForTest()

    def restart_validator(self, permit, nonce):
        self.events.append("validator")
        assert self.phase == "validator_destructive"
        assert permit["guard_generation"] == 7
        assert permit["handoff_nonce"] == nonce
        assert permit["candidate_commit"] == SHA
        if self.fail_at == "validator":
            raise PairErrorForTest()
        super().restart_validator(permit, nonce)


class PairErrorForTest(RuntimeError):
    pass


def test_one_guard_across_pair_no_release_before_both_services_ready(tmp_path):
    p = FakePair(tmp_path)
    receipt = p.run()
    assert p.events == ["installed-preflight", "gateway", "drain", "authorize", "validator",
                        "validator-command", "operational-ready", "drain", "ready", "release"]
    assert receipt["guard_generation"] == 7
    logs = "".join(f.read_text() for f in tmp_path.glob("*.log"))
    assert "handoff_nonce" not in logs
    assert p.stop.is_set()


@pytest.mark.parametrize("failure", ["unsupported", "gateway", "drain", "authorize", "validator", "operational-ready", "ready"])
def test_failure_never_releases_guard_or_starts_unapproved_next_step(tmp_path, failure):
    p = FakePair(tmp_path)
    p.fail_at = failure
    with pytest.raises((PairErrorForTest, pair.PairError)):
        p.run()
    assert "release" not in p.events
    if failure in {"unsupported", "gateway", "drain", "authorize"}:
        assert "validator" not in p.events
    if failure == "unsupported":
        assert not (tmp_path/"gateway.log").exists()


def test_renewal_failure_after_primary_ready_keeps_guard(tmp_path):
    p = FakePair(tmp_path)
    original = p.restart_validator
    def ready_but_renewal_failed(permit, nonce):
        original(permit, nonce)
        p.renew_error = pair.PairError("renew failed")
    p.restart_validator = ready_but_renewal_failed
    with pytest.raises(pair.PairError):
        p.run()
    assert "ready" not in p.events and "release" not in p.events


@pytest.mark.parametrize("field,value", [("candidate_commit", "b"*40), ("restart_scope", "gateway"),
                                          ("restart_phase", "draining"), ("guard_generation", True),
                                          ("guard_active", False), ("paused", False)])
def test_changed_gateway_authority_cannot_restart_primary(tmp_path, field, value):
    p = FakePair(tmp_path)
    p.receipt_state[field] = value
    with pytest.raises(pair.PairError):
        p.run()
    assert "validator" not in p.events and "release" not in p.events


def test_changed_generation_during_renewal_cannot_restart_primary(tmp_path):
    p = FakePair(tmp_path)
    p.generation_value = 8
    with pytest.raises(pair.PairError):
        p.run()
    assert "validator" not in p.events


def test_renew_loop_uses_same_existing_guard_and_stops_on_error(tmp_path):
    p = FakePair(tmp_path)
    class Tick:
        calls = 0
        def wait(self, seconds):
            assert seconds == 30
            self.calls += 1
            return self.calls > 2
    p.stop = Tick()
    p.renew_loop()
    assert p.events == ["drain", "drain"]
    p.stop = Tick()
    p.fail_at = "drain"
    p.renew_loop()
    assert isinstance(p.renew_error, pair.PairError)


def test_private_permit_consumption_rejects_replay_wrong_generation_and_nonce(monkeypatch):
    # Existing helper owns the private-file checks; exercise its actual CLI action.
    import tempfile
    import shutil
    directory = Path(tempfile.mkdtemp(prefix="leadpoet-pair-test.", dir="/tmp"))
    path = directory/"leadpoet-permit.json"
    args = Namespace(candidate=SHA, invocation=INV, scope="all", nonce="c"*64,
                     authority_commit=SHA, expected_generation=7, path=path)
    try:
        value = handoff._permit(args, state("validator_destructive"))
        handoff._write_document(path, value)
        args.expected_generation = 8
        with pytest.raises(handoff.GuardError):
            handoff._validate_permit(args)
        assert path.exists()
        args.expected_generation = 7
        args.nonce = "d"*64
        with pytest.raises(handoff.GuardError):
            handoff._validate_permit(args)
        args.nonce = "c"*64
        argv = ["handoff", "consume-permit", "--path", str(path), "--candidate", SHA,
                "--invocation", INV, "--scope", "all", "--nonce", args.nonce,
                "--authority-commit", SHA, "--expected-generation", "7"]
        monkeypatch.setattr("sys.argv", argv)
        assert handoff.main() == 0
        assert not path.exists()
        assert handoff.main() == 1
    finally:
        shutil.rmtree(directory)


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('changed_generation', [False, True])
def test_real_gateway_tail_releases_only_standalone(tmp_path, paired, changed_generation):
    import shlex
    import sys
    text = (Path(__file__).parents[1]/'gw_restart.sh').read_text()
    start = text.index('if [ "$GATEWAY_PAIRED_RESTART" = 1 ]; then', text.index('GATEWAY_DEPLOY_STAGE="lab_arena_claim_guard_release"'))
    end = text.index('\nGATEWAY_DEPLOY_STAGE="completed"', start)
    log = tmp_path/'calls'
    program = f'''set -euo pipefail
GATEWAY_PAIRED_RESTART={int(paired)}
LEADPOET_REPO_ROOT=/exact
PREPARED_GATEWAY_SHA={SHA}
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID={INV}
LAB_ARENA_RESTART_GUARD_GENERATION=7
GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}
run_lab_arena_restart_guard() {{
  echo "$2" >> {shlex.quote(str(log))}
  if [ "$2" = owned ]; then echo {shlex.quote(json.dumps(state(generation=8 if changed_generation else 7)))}; fi
}}
{text[start:end]}
'''
    result = subprocess.run(['bash', '-c', program], capture_output=True, text=True)
    rejected = paired and changed_generation
    assert (result.returncode != 0) is rejected, result.stderr
    assert log.read_text().splitlines() == (['owned'] if paired else ['release'])
    assert (pair.PREFIX in result.stdout) is (paired and not rejected)


@pytest.mark.parametrize('bad', [None, 'candidate', 'nonce', 'generation', 'authority', 'expired'])
def test_actual_canonical_validator_permit_gate_before_stop(tmp_path, bad):
    import shlex
    import sys
    import tempfile
    import shutil
    text = (Path(__file__).parents[1]/'validator_restart.sh').read_text()
    start = text.index('if [ -n "${VALIDATOR_PAIR_PERMIT:-}" ]; then')
    end = text.index('\n# Only now drain', start)
    root = Path(tempfile.mkdtemp(prefix='leadpoet-gate-test.', dir='/tmp'))
    path = root/'leadpoet-permit.json'
    args = Namespace(candidate=SHA, invocation=INV, scope='all', nonce='c'*64,
                     authority_commit=SHA, expected_generation=7, path=path)
    permit = handoff._permit(args, state('validator_destructive'))
    if bad == 'candidate': permit['candidate_commit'] = 'b'*40
    if bad == 'expired': permit['guard_expires_at'] = '2000-01-01T00:00:00Z'
    handoff._write_document(path, permit)
    values = dict(RELEASE=str(Path(__file__).parents[1]), PYTHON=sys.executable, TARGET_SHA=SHA,
                  VALIDATOR_PAIR_PERMIT=str(path), VALIDATOR_PAIR_INVOCATION=INV,
                  VALIDATOR_PAIR_NONCE=('d'*64 if bad=='nonce' else 'c'*64),
                  VALIDATOR_PAIR_GENERATION=('8' if bad=='generation' else '7'),
                  VALIDATOR_PAIR_AUTHORITY_COMMIT=('b'*40 if bad=='authority' else SHA))
    assignments = '\n'.join(k+'='+shlex.quote(v) for k,v in values.items())
    try:
        result = subprocess.run(['bash','-c',f'''set -euo pipefail
fail() {{ echo "$*" >&2; exit 1; }}
{assignments}
{text[start:end]}
echo STOP_REACHED
'''], capture_output=True, text=True)
        assert ('STOP_REACHED' in result.stdout) is (bad is None), result.stderr
        assert (result.returncode == 0) is (bad is None)
        assert path.exists() is (bad is not None)
    finally:
        shutil.rmtree(root)


STARTED = '2026-10-21T04:00:00+00:00'
SESSION = '41ac7191-8aba-4232-9b20-1729b770abc8'


def readiness_rows(*, session=SESSION, candidate=SHA, timestamp='2026-10-21T04:00:10+00:00'):
    return [dict(validator_event_id=index, event_kind=kind, created_at=timestamp, occurred_at=timestamp,
                 content=dict(validator_source_commit=candidate, session_id=session, phase=phase,
                              ready_slots=7)) for index,(kind,phase) in enumerate(
            [('validator.startup','startup'), ('validator.ready','weights'), ('validator.ready','scoring_setup')],1)]


@pytest.mark.parametrize('bad', ['old-occurrence', 'old-created', 'candidate', 'split-session', 'zero-slots', 'missing-startup', 'error'])
def test_readiness_never_uses_old_or_incomplete_other_session(bad):
    rows = readiness_rows()
    if bad == 'old-occurrence':
        for row in rows: row['occurred_at'] = '2026-10-21T03:59:59+00:00'
    elif bad == 'old-created':
        for row in rows: row['created_at'] = '2026-10-21T03:59:59+00:00'
    elif bad == 'candidate': rows = readiness_rows(candidate='b'*40)
    elif bad == 'split-session': rows[-1]['content']['session_id'] = 'other-session'
    elif bad == 'zero-slots': rows[-1]['content']['ready_slots'] = 0
    elif bad == 'missing-startup': rows.pop(0)
    else:
        rows.append(dict(validator_event_id=4, event_kind='validator.error', created_at='2026-10-21T04:00:12+00:00',
                         occurred_at='2026-10-21T04:00:12+00:00',
                         content=dict(validator_source_commit=SHA, session_id=SESSION, phase='weights')))
    assert pair.ready_session(rows, candidate=SHA, started_at=STARTED) is None


def test_large_old_history_and_later_recovery_do_not_replace_current_session():
    rows = readiness_rows(timestamp='2026-10-21T03:59:59+00:00') * 500 + readiness_rows()
    error = dict(validator_event_id=4, event_kind='validator.error', created_at='2026-10-21T04:00:12+00:00',
                 occurred_at='2026-10-21T04:00:12+00:00',
                 content=dict(validator_source_commit=SHA, session_id=SESSION, phase='weights'))
    rows.append(error)
    rows.append({**error, 'validator_event_id':5, 'event_kind':'validator.recovered', 'occurred_at':'2026-10-21T04:00:13+00:00'})
    assert pair.ready_session(rows, candidate=SHA, started_at=STARTED) == SESSION


def test_waits_for_both_loops_then_rereads_same_primary_process(tmp_path, monkeypatch):
    p = FakePair(tmp_path)
    hotkey = '5' * 48
    (tmp_path/'validator.log').write_text('Arena local-wallet validator readiness is valid: hotkey='+hotkey+'\n')
    identity = dict(pid=123, created_at=STARTED, cwd='/exact/release', commit=SHA)
    checks = []
    p.primary_identity = lambda *args: (checks.append('identity') or identity)
    polls = [readiness_rows()[:-1], readiness_rows()]
    def events(key, lower, remaining):
        assert key == hotkey and lower == STARTED and 0 < remaining <= 300
        checks.append('events')
        return polls.pop(0)
    p.readiness_events = events
    monkeypatch.setattr(pair.time, 'sleep', lambda _: None)
    pair.Pair.wait_validator_ready(p)
    assert checks == ['identity','events','events','identity']
    assert json.loads((tmp_path/'validator-ready.json').read_text())['session_id'] == SESSION


def test_changed_process_or_missing_loop_keeps_guard(tmp_path, monkeypatch):
    p = FakePair(tmp_path)
    (tmp_path/'validator.log').write_text('Arena local-wallet validator readiness is valid: hotkey='+'5'*48+'\n')
    ids = iter([dict(pid=123,created_at=STARTED), dict(pid=124,created_at=STARTED)])
    p.primary_identity = lambda *args: next(ids)
    p.readiness_events = lambda *args: readiness_rows()
    with pytest.raises(pair.PairError, match='process changed'):
        pair.Pair.wait_validator_ready(p)
    assert not (tmp_path/'validator-ready.json').exists()
    p.primary_identity = lambda *args: dict(pid=123, created_at=STARTED)
    p.readiness_events = lambda *args: readiness_rows()[:-1]
    clock = iter([0,0,301])
    monkeypatch.setattr(pair.time, 'monotonic', lambda: next(clock))
    with pytest.raises(pair.PairError, match='did not become ready'):
        pair.Pair.wait_validator_ready(p)


def test_readiness_get_is_server_time_bounded_and_cap_fails_closed(tmp_path):
    import shlex
    p = FakePair(tmp_path)
    def remote(host, command, **kwargs):
        assert host == 'gateway' and kwargs['timeout'] == 25
        words = shlex.split(command)
        program = words[-1]
        # Execute only the generated private-read program with a fake transport.
        # The stale history fixture is filtered by the actual requested range.
        calls = []
        class Response:
            def raise_for_status(self): pass
            def json(self): return readiness_rows()
        class Client:
            def get(self, url, *, headers, params):
                calls.extend(params)
                assert ('created_at','gte.'+STARTED) in params
                assert any(k=='created_at' and v.startswith('lte.') for k,v in params)
                assert ('validator_hotkey','eq.'+'5'*48) in params
                assert ('limit','128') in params
                return Response()
        class Transport:
            def __init__(self, *args, **kwargs):
                assert kwargs['timeout_seconds'] == 8
                self._client=Client(); self._base_url='https://readonly.invalid'; self._headers={}
        import sys, types, io
        fake_store = types.ModuleType('lab_arena.store'); fake_store.PostgrestTransport=Transport
        from contextlib import redirect_stdout
        from unittest.mock import patch
        fake_guard = types.ModuleType('scripts.lab_arena_restart_claim_guard')
        fake_guard._read_scoped_environment=lambda _: dict(LAB_ARENA_SUPABASE_URL='https://readonly.invalid', LAB_ARENA_SERVICE_KEY='private-test-only')
        out = io.StringIO()
        with patch.dict(sys.modules, {'lab_arena.store':fake_store, 'scripts.lab_arena_restart_claim_guard':fake_guard}), redirect_stdout(out):
            exec(program, {})
        assert len(calls) == 6
        return subprocess.CompletedProcess([], 0, stdout=out.getvalue())
    p.remote = remote
    assert pair.Pair.readiness_events(p, '5'*48, STARTED, 25) == readiness_rows()
    p.remote = lambda *args,**kwargs: subprocess.CompletedProcess([],0,stdout=json.dumps(readiness_rows()*43))
    with pytest.raises(pair.PairError, match='incomplete'):
        pair.Pair.readiness_events(p, '5'*48, STARTED, 25)


def test_equal_timestamp_new_error_is_not_cleared_by_older_recovery():
    rows = readiness_rows()
    timestamp = rows[0]['occurred_at']
    fields = dict(validator_source_commit=SHA, session_id=SESSION, phase='scoring_setup')
    rows.extend([dict(validator_event_id=4, event_kind='validator.recovered', created_at=timestamp,
                      occurred_at=timestamp, content=fields),
                 dict(validator_event_id=5, event_kind='validator.error', created_at=timestamp,
                      occurred_at=timestamp, content=fields)])
    assert pair.ready_session(list(reversed(rows)), candidate=SHA, started_at=STARTED) is None
    rows.append(dict(validator_event_id=6, event_kind='validator.recovered', created_at=timestamp,
                     occurred_at=timestamp, content=fields))
    assert pair.ready_session(list(reversed(rows)), candidate=SHA, started_at=STARTED) == SESSION


@pytest.mark.parametrize('exact', [False, True])
def test_marker_only_installed_controller_is_refused_before_deployment(tmp_path, exact):
    import os
    p = FakePair(tmp_path)
    installed = tmp_path/'gw_restart.sh'
    expected = (Path(__file__).parents[1]/'gw_restart.sh').read_bytes()
    installed.write_bytes(expected if exact else b'GATEWAY_PAIRED_RESTART=1\necho old-controller\n')
    installed.chmod(0o700)
    tools = tmp_path/'bin'; tools.mkdir()
    (tools/'stat').write_text('#!/bin/sh\nprintf "%s:700\\n" "$(id -u)"\n')
    (tools/'sha256sum').write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    for tool in tools.iterdir(): tool.chmod(0o700)
    commands=[]
    def remote(host, command, **kwargs):
        commands.append(command)
        command=command.replace('/home/ec2-user/gw_restart.sh',str(installed))
        if len(commands)==1:
            return subprocess.run(['bash','-c',command],check=True,env={**os.environ,'PATH':str(tools)+':'+os.environ['PATH']},**kwargs)
        raise PairErrorForTest('preflight passed; stop before controller')
    p.remote=remote
    with pytest.raises((PairErrorForTest,subprocess.CalledProcessError)):
        p.run()
    assert len(commands)==(2 if exact else 1)
    assert (tmp_path/'gateway.log').exists() is exact


@pytest.mark.parametrize('fault', [None,'changed-file','wrong-commit','inactive','check-only'])
def test_actual_primary_identity_program_rejects_wrong_process_or_source(tmp_path, fault):
    import shlex,sys,types,io
    from contextlib import redirect_stdout
    from unittest.mock import patch
    root=tmp_path/'release';root.mkdir()
    files=['lab_arena/validator.py','lab_arena/runner.py','lab_arena/wiring.py','lab_arena/validator_logging.py']
    (root/'.release-commit').write_text('b'*40 if fault=='wrong-commit' else SHA)
    for name in files:
        target=root/name;target.parent.mkdir(exist_ok=True);target.write_bytes(b'exact-source')
    if fault=='changed-file':(root/files[0]).write_bytes(b'changed')
    p=FakePair(tmp_path)
    class Process:
        pid=123
        def __init__(self,pid):assert pid==123
        def cwd(self):return str(root)
        def cmdline(self):return ['python3','scripts/run_arena_validator.py']+(['--check-only'] if fault=='check-only' else [])
        def create_time(self):return datetime.fromisoformat(STARTED).timestamp()
    def output(argv,**kwargs):
        if argv[0]=='systemctl':return 'MainPID=123\nActiveState='+('inactive' if fault=='inactive' else 'active')+'\nSubState=running\n'
        assert argv[:3]==['git','-C',pair.VALIDATOR_ROOT]
        return b'exact-source'
    def remote(host,command,**kwargs):
        assert host=='validator' and kwargs['timeout']==17
        program=shlex.split(command)[-1]
        fake=types.ModuleType('psutil');fake.Process=Process
        out=io.StringIO()
        with patch.dict(sys.modules,{'psutil':fake}),patch.object(subprocess,'check_output',output),redirect_stdout(out):
            exec(program,{})
        return subprocess.CompletedProcess([],0,stdout=out.getvalue())
    p.remote=remote
    if fault:
        with pytest.raises(AssertionError):pair.Pair.primary_identity(p,17)
    else:
        assert pair.Pair.primary_identity(p,17)==dict(pid=123,created_at=STARTED,cwd=str(root),commit=SHA)
