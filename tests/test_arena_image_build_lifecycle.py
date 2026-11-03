"""Builder lifecycle checks use fake Docker; Linux checks use real host locks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import build_arena_reviewer_runtime as builder

ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / 'scripts/build_lab_arena_judge_image.sh'
LOCK = ROOT / 'validator_tee/scripts/docker_operation_lock_v2.sh'
OLD = 'sha256:' + '1' * 64
NEW = 'sha256:' + '2' * 64


@pytest.mark.parametrize('platform,explicit,directory,expected', [
    ('darwin', True, True, False), ('linux', False, False, False),
    ('linux', True, False, True), ('linux', False, True, True),
])
def test_managed_host_boundary(monkeypatch, platform, explicit, directory, expected):
    monkeypatch.setattr(builder.sys, 'platform', platform)
    monkeypatch.setattr(builder.Path, 'is_dir', lambda self: directory)
    monkeypatch.delenv('LEADPOET_DOCKER_OPERATION_LOCK_FILE', raising=False)
    if explicit:
        monkeypatch.setenv('LEADPOET_DOCKER_OPERATION_LOCK_FILE', '/tmp/explicit.lock')
    assert builder._managed_docker_host() is expected


@pytest.mark.parametrize('publish,verify,fail,expected', [
    (True, False, False, ['guard', 'snapshot', 'cleanup', 'prune']),
    (False, False, False, ['guard', 'snapshot', 'prune']),
    (False, False, True, ['guard', 'snapshot', 'cleanup', 'prune']),
    (False, True, False, ['guard']),
])
def test_reviewer_lifecycle_and_inherited_descriptors(monkeypatch, publish, verify, fail, expected):
    monkeypatch.setattr(builder, '_managed_docker_host', lambda: True)
    monkeypatch.setattr(builder, '_require_managed_docker_locks', lambda: None)
    calls = []
    def run(args, **kwargs):
        assert kwargs['pass_fds'] == (7, 8)
        assert kwargs['stdout'] is sys.stderr
        calls.append('guard' if args[0] == sys.executable else
                     args[2] if args[0] == 'bash' else 'prune')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(builder.subprocess, 'run', run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError) as error:
            with builder._managed_build_images(publish=publish, verify_only=verify):
                raise subprocess.CalledProcessError(23, ['docker', 'build'])
        assert error.value.returncode == 23
    else:
        with builder._managed_build_images(publish=publish, verify_only=verify):
            pass
    assert calls == expected


@pytest.mark.parametrize('body_failed', [False, True])
def test_cleanup_failure_visible_without_masking_original_failure(monkeypatch, capsys, body_failed):
    monkeypatch.setattr(builder, '_managed_docker_host', lambda: True)
    monkeypatch.setattr(builder, '_require_managed_docker_locks', lambda: None)
    def run(args, **kwargs):
        if args[0] == 'bash' and args[2] == 'cleanup':
            raise subprocess.CalledProcessError(17, args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(builder.subprocess, 'run', run)
    expected = subprocess.CalledProcessError if body_failed else builder.CompositionError
    with pytest.raises(expected) as error:
        with builder._managed_build_images(publish=True, verify_only=False):
            if body_failed:
                raise subprocess.CalledProcessError(23, ['docker', 'build'])
    if body_failed:
        assert error.value.returncode == 23
    assert 'cleanup failed' in capsys.readouterr().err


def test_reexec_preserves_open_lock_descriptors(monkeypatch):
    monkeypatch.delenv('LEADPOET_DOCKER_OPERATION_LOCK_HELD', raising=False)
    seen = {}
    original = os.fstat
    def fstat(fd):
        return None if fd in (7, 8) else original(fd)
    def run(args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return SimpleNamespace(returncode=29)
    with monkeypatch.context() as patch:
        patch.setattr(builder.os, 'fstat', fstat)
        patch.setattr(builder.subprocess, 'run', run)
        with pytest.raises(SystemExit) as error:
            builder._require_managed_docker_locks()
    assert error.value.code == 29
    assert seen['kwargs']['pass_fds'] == (7, 8)
    assert 'leadpoet_acquire_docker_operation_lock_v2' in seen['args'][2]


def test_admission_marker_alone_is_not_accepted(monkeypatch):
    monkeypatch.setenv('LEADPOET_DOCKER_OPERATION_LOCK_HELD', '1')
    monkeypatch.delenv('LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD', raising=False)
    with pytest.raises(builder.CompositionError, match='both Docker locks'):
        builder._require_managed_docker_locks()


def fake_docker(tmp_path, *, managed):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    state = tmp_path / 'images.json'
    state.write_text(json.dumps([OLD]))
    log = tmp_path / 'docker.jsonl'
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import fcntl,json,os,sys
from pathlib import Path
args=sys.argv[1:]
if os.environ.get('TEST_MANAGED')=='1':
 for key in ('LEADPOET_DOCKER_OPERATION_LOCK_FILE','LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE'):
  descriptor=os.open(os.environ[key],os.O_RDWR)
  try:
   try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError: pass
   else: raise AssertionError('Docker command ran without both real locks')
  finally: os.close(descriptor)
with open(os.environ['TEST_DOCKER_LOG'],'a') as output: output.write(json.dumps(args)+'\\n')
state=Path(os.environ['TEST_DOCKER_STATE']);images=json.loads(state.read_text())
if args[:2]==['buildx','version']: raise SystemExit(1)
if args[:2]==['image','ls']: print('\\n'.join(images))
elif args[:2]==['image','rm']:
 image=args[-1]
 assert image!='sha256:'+'1'*64
 state.write_text(json.dumps([x for x in images if x!=image]))
elif args[0]=='build':
 state.write_text(json.dumps(sorted(set(images+['sha256:'+'2'*64]))))
 raise SystemExit(int(os.environ.get('TEST_BUILD_STATUS','0')))
elif args[0]=='push': raise SystemExit(int(os.environ.get('TEST_PUSH_STATUS','0')))
elif args[0]=='inspect': print('example.invalid/arena@sha256:'+'3'*64)
elif args[:2]==['builder','prune']: raise SystemExit(int(os.environ.get('TEST_PRUNE_STATUS','0')))
else: raise AssertionError('unexpected fake Docker command '+repr(args))
''')
    docker.chmod(0o700)
    git = bin_dir / 'git'
    git.write_text('''#!/usr/bin/env python3
import sys
if sys.argv[-2:]==['rev-parse','HEAD']: print('a'*40)
else: raise SystemExit('unexpected fake git command')
''')
    git.chmod(0o700)
    if not managed:
        uname = bin_dir / 'uname'
        uname.write_text('#!/bin/sh\necho Darwin\n')
        uname.chmod(0o700)
    env = {**os.environ, 'PATH': str(bin_dir)+os.pathsep+os.environ['PATH'],
           'TEST_MANAGED': '1' if managed else '0', 'TEST_DOCKER_LOG': str(log),
           'TEST_DOCKER_STATE': str(state), 'TMPDIR': str(tmp_path)}
    for key in list(env):
        if key.startswith('LEADPOET_DOCKER_OPERATION_'):
            env.pop(key)
    env['LEADPOET_DOCKER_OPERATION_LOCK_FILE'] = str(tmp_path/'operation.lock')
    return env, state, log


@pytest.mark.parametrize('no_push,build_status', [(False, 0), (True, 0), (False, 23)])
def test_unmanaged_local_build_preserves_existing_behavior(tmp_path, no_push, build_status):
    env,state,log = fake_docker(tmp_path, managed=False)
    env['TEST_BUILD_STATUS'] = str(build_status)
    command = ['bash',str(JUDGE),'example.invalid/arena'] + (['--no-push'] if no_push else [])
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == build_status, result.stderr
    assert json.loads(state.read_text()) == [OLD, NEW]
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[:2] in [['image','rm'],['builder','prune']] for call in calls)


linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='real /proc/FLOCK contract requires Linux')


@linux_only
@pytest.mark.parametrize('no_push,build_status,push_status,prune_status,expected', [
    (False,0,0,0,0), (True,0,0,0,0), (False,23,0,0,23),
    (False,0,31,0,31), (False,23,0,17,23), (False,0,0,17,17),
])
def test_managed_judge_holds_both_locks_and_cleans_only_own_images(
    tmp_path,no_push,build_status,push_status,prune_status,expected,
):
    env,state,log=fake_docker(tmp_path,managed=True)
    env.update(TEST_BUILD_STATUS=str(build_status),TEST_PUSH_STATUS=str(push_status),
               TEST_PRUNE_STATUS=str(prune_status))
    command=['bash',str(JUDGE),'example.invalid/arena']+(['--no-push'] if no_push else [])
    result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=15)
    assert result.returncode==expected,result.stderr
    assert json.loads(state.read_text())==([OLD,NEW] if no_push and build_status==0 else [OLD])
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert ['builder','prune','--all','--force'] in calls
    if prune_status:
        assert 'cleanup failed' in result.stderr
    if not no_push and build_status==push_status==0:
        assert result.stdout.strip().endswith('LAB_ARENA_SCORER_IMAGE=example.invalid/arena@sha256:'+'3'*64)


@linux_only
def test_managed_reviewer_cleanup_receives_real_inherited_lock_descriptors(tmp_path):
    env,state,log=fake_docker(tmp_path,managed=True)
    code='''
import subprocess
from scripts import build_arena_reviewer_runtime as builder
with builder._managed_build_images(publish=True, verify_only=False):
    subprocess.run(['docker','build'],check=True)
'''
    result=subprocess.run(['bash','-c',
        'set -euo pipefail; . "$1"; leadpoet_acquire_docker_operation_lock_v2; exec "$2" -c "$3"',
        'reviewer-test',str(LOCK),sys.executable,code],cwd=ROOT,env=env,
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert json.loads(state.read_text())==[OLD]
    assert ['builder','prune','--all','--force'] in [json.loads(x) for x in log.read_text().splitlines()]


@linux_only
def test_open_but_unlocked_descriptors_do_not_authorize_managed_build(tmp_path):
    env,state,log=fake_docker(tmp_path,managed=True)
    admission=env['LEADPOET_DOCKER_OPERATION_LOCK_FILE']+'.admission'
    env.update(LEADPOET_DOCKER_OPERATION_LOCK_HELD='1',
        LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD='1',
        LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE=admission)
    code='from scripts.build_arena_reviewer_runtime import _require_managed_docker_locks; _require_managed_docker_locks()'
    result=subprocess.run(['bash','-c','exec 7>>"$1"; exec 8>>"$2"; exec "$3" -c "$4"',
        'unlocked-test',env['LEADPOET_DOCKER_OPERATION_LOCK_FILE'],admission,sys.executable,code],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert not log.exists()
    assert json.loads(state.read_text())==[OLD]


@linux_only
def test_managed_reexec_acquires_real_locks_before_child_continues(tmp_path):
    env,state,log=fake_docker(tmp_path,managed=True)
    script=tmp_path/'reexec.py'
    script.write_text('''
import sys
sys.path.insert(0, %r)
from scripts import build_arena_reviewer_runtime as builder
builder.__file__=__file__
builder._require_managed_docker_locks()
import subprocess
subprocess.run(['docker','build'],check=True)
''' % str(ROOT))
    result=subprocess.run([sys.executable,str(script)],cwd=ROOT,env=env,
                          capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert json.loads(state.read_text())==[OLD,NEW]
    assert [json.loads(x) for x in log.read_text().splitlines()]==[['build']]
