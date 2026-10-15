"""Bind recovery stops to one exact private receipt and release owner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from scripts import manage_owned_process_group as owned

OWNER = 'gateway-1000-2000'
SHA = 'a' * 40


def fixture_state(tmp_path, monkeypatch):
    command = ['/fixture/python', '-u', 'owned-service.py']
    state = {'version': 1, 'pid': 123, 'pgid': 123, 'start_time_ticks': 500,
             'uid': os.getuid(), 'cwd': str(tmp_path), 'argv': command}
    path = tmp_path / 'process.json'
    path.write_text(json.dumps(state))
    path.chmod(0o600)
    args = argparse.Namespace(state_file=path, cwd=str(tmp_path), uid=os.getuid(),
        process_argv=command, verified_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        verified_invocation=OWNER, release_sha=SHA, term_seconds=1, kill_seconds=1,
        no_kill=True, signal_leader_only=True)
    live = {**state, 'state': 'S', 'session': 123}
    monkeypatch.setattr(owned, '_read_process', lambda _: live)
    monkeypatch.setattr(owned, '_verified_invocation', lambda *_: OWNER)
    return args, state, live


@pytest.mark.parametrize('mutation', ['receipt', 'missing', 'owner', 'pid'])
def test_strict_stop_rejects_replaced_identity_without_adoption(tmp_path, monkeypatch, mutation):
    args, state, live = fixture_state(tmp_path, monkeypatch)
    if mutation == 'receipt':
        args.state_file.write_text(json.dumps({**state, 'start_time_ticks': 501}))
    elif mutation == 'missing':
        args.state_file.unlink()
    elif mutation == 'owner':
        monkeypatch.setattr(owned, '_verified_invocation', lambda *_: 'gateway-9000-9000')
    else:
        live['start_time_ticks'] = 501
    monkeypatch.setattr(owned.os, 'kill', lambda *_: pytest.fail('changed process was signaled'))
    monkeypatch.setattr(owned.os, 'killpg', lambda *_: pytest.fail('changed process group was signaled'))
    monkeypatch.setattr(owned, '_discover_one', lambda *_: pytest.fail('strict stop must not adopt'))
    with pytest.raises(owned.OwnershipError):
        owned._stop(args)


def test_strict_stop_preserves_receipt_replaced_during_wait(tmp_path, monkeypatch):
    args, state, _ = fixture_state(tmp_path, monkeypatch)
    signals = []
    monkeypatch.setattr(owned.os, 'kill', lambda *items: signals.append(items))
    replacement = json.dumps({**state, 'start_time_ticks': 501})
    def completed(*_):
        args.state_file.write_text(replacement)
        return True
    monkeypatch.setattr(owned, '_wait_for_group_exit', completed)
    with pytest.raises(owned.OwnershipError, match='changed while'):
        owned._stop(args)
    assert signals == [(state['pid'], signal.SIGTERM)]
    assert args.state_file.read_text() == replacement


@pytest.mark.skipif(sys.platform != 'linux', reason='real Linux process ownership and SIGTERM proof')
@pytest.mark.parametrize('control', ['valid', 'wrong-owner', 'wrong-release', 'receipt-before', 'receipt-after'])
def test_linux_strict_stop_owns_and_reaps_its_child(tmp_path, monkeypatch, control):
    tmp_path.chmod(0o700)
    command = [sys.executable, '-u', '-c', 'import time\nwhile True: time.sleep(1)\n']
    environment = os.environ | {'GATEWAY_RESTART_INVOCATION_ID': OWNER,
        'LEADPOET_RESTART_INVOCATION_ID': OWNER,
        'GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID': OWNER,
        'LAB_ARENA_RELEASE_ROOT': str(tmp_path), 'LAB_ARENA_RELEASE_COMMIT': SHA}
    child = subprocess.Popen(command, cwd=tmp_path, env=environment, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        state = owned._read_process(child.pid)
        assert state is not None and state['argv'] == command
        path = tmp_path / 'process.json'
        owned._write_state(path, state)
        args = argparse.Namespace(state_file=path, cwd=str(tmp_path), uid=os.getuid(),
            process_argv=command, verified_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            verified_invocation=OWNER, release_sha=SHA, term_seconds=2, kill_seconds=1,
            no_kill=True, signal_leader_only=True)
        replacement = json.dumps({**json.loads(path.read_text()), 'start_time_ticks': state['start_time_ticks'] + 1})
        if control == 'wrong-owner':
            args.verified_invocation = 'gateway-9000-9000'
        elif control == 'wrong-release':
            args.release_sha = 'b' * 40
        elif control == 'receipt-before':
            path.write_text(replacement)
        elif control == 'receipt-after':
            wait = owned._wait_for_group_exit
            def replaced_during_wait(*items):
                path.write_text(replacement)
                return wait(*items)
            monkeypatch.setattr(owned, '_wait_for_group_exit', replaced_during_wait)
        if control == 'valid':
            owned._stop(args)
            child.wait(timeout=3)
            assert child.returncode == -signal.SIGTERM
            assert not path.exists()
        else:
            with pytest.raises(owned.OwnershipError):
                owned._stop(args)
            assert path.exists()
            if control == 'receipt-after':
                child.wait(timeout=3)
                assert child.returncode == -signal.SIGTERM
                assert path.read_text() == replacement
            else:
                assert child.poll() is None  # No signal reached the wrong identity.
    finally:
        # The fixture owns and reaps this exact Popen child. No discovery or group kills.
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)


@pytest.mark.parametrize('strict', [False, True])
def test_controller_stop_uses_strict_current_helper_only_for_recovery(tmp_path, strict):
    import shlex
    source = (Path(__file__).resolve().parents[1] / 'gw_restart.sh').read_text()
    function = source[source.index('stop_arena_component() {'):].split('\n}\n', 1)[0] + '\n}\n'
    output = tmp_path / 'operations.json'
    for name in ('old', 'current'):
        directory = tmp_path / name / 'scripts'
        directory.mkdir(parents=True)
        (directory / 'manage_owned_process_group.py').write_text(
            'import json,sys\nfrom pathlib import Path\n'
            f'path=Path({str(output)!r})\n'
            'rows=json.loads(path.read_text()) if path.exists() else []\n'
            f'rows.append([{name!r},sys.argv[1:]])\n'
            'path.write_text(json.dumps(rows))\n'
            'if sys.argv[1] == "verify": print(json.dumps({"ok":True,"state_sha256":"1"*64}))\n')
    setup = f'''set -euo pipefail
GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}
ARENA_CANDIDATE_TREE={shlex.quote(str(tmp_path / 'old'))}
GATEWAY_PREFLIGHT_TREE={shlex.quote(str(tmp_path / 'current'))}
verify_controller_process_helper() {{ return 0; }}
arena_component_command() {{ ARENA_COMPONENT_STATE_FILE=/fixture/state.json; ARENA_COMPONENT_COMMAND=(fixture-python service.py); }}
'''
    extra = f' {OWNER} {SHA}' if strict else ''
    result = subprocess.run(['bash', '-c', setup + function + f'stop_arena_component api 1 /fixture/tree{extra}\n'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    operations = json.loads(output.read_text())
    if strict:
        assert [row[0] for row in operations] == ['current', 'current']
        assert operations[0][1][0] == 'verify'
        arguments = operations[-1][1]
        for option, expected in (('--verified-state-sha256', '1' * 64), ('--verified-invocation', OWNER), ('--release-sha', SHA)):
            assert arguments[arguments.index(option) + 1] == expected
    else:
        assert operations[0][0] == 'old'
        assert operations[0][1][0] == 'stop'
        assert '--verified-invocation' not in operations[0][1]
