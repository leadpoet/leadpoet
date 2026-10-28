from pathlib import Path
import os
import subprocess

import pytest


GATEWAY = Path('gw_restart.sh').read_text()
RECLAIM = Path('validator_tee/scripts/reclaim_docker_storage_v2.sh').read_text()


def function(source, name):
    start = source.index(name + '() {')
    return source[start:source.index('\n}\n', start) + 3]


def bash(script, **environment):
    return subprocess.run(['bash', '-c', 'set -euo pipefail\n' + script],
                          env={**os.environ, **environment}, text=True, capture_output=True)


@pytest.mark.parametrize('source', [GATEWAY, RECLAIM])
@pytest.mark.parametrize('state,status', [('0', 0), ('1', 0), ('', 1), ('unexpected', 0)])
def test_pending_reader_distinguishes_absence_from_failure(source, state, status):
    script = function(source, 'pending_docker_reconcile') + '''
sudo() { printf '%s' "$TEST_STATE"; return "$TEST_STATUS"; }
if pending_docker_reconcile; then echo present; else echo absent; fi
echo completed
'''
    result = bash(script, TEST_STATE=state, TEST_STATUS=str(status))
    if status or state not in ('0', '1'):
        assert result.returncode != 0 and 'completed' not in result.stdout
    else:
        assert result.returncode == 0 and ('present' if state == '1' else 'absent') in result.stdout


@pytest.mark.parametrize('pending,free,expected', [('0', 19_000_000, False), ('1', 19_000_000, True), ('0', 17_000_000, True)])
def test_canonical_preflight_uses_verified_authority_for_pending_or_low_space(tmp_path, pending, free, expected):
    authority = tmp_path / 'authority'
    scripts = authority / 'validator_tee/scripts'
    scripts.mkdir(parents=True)
    (scripts / 'docker_operation_lock_v2.sh').write_text('leadpoet_acquire_docker_operation_lock_v2() { :; }\nleadpoet_release_docker_operation_lock_v2() { :; }\n')
    (scripts / 'reclaim_docker_storage_v2.sh').write_text('echo candidate-helper\n')
    script = '\n'.join(function(GATEWAY, name) for name in (
        'pending_docker_reconcile', 'docker_build_required_kb', 'emergency_disk_preflight')) + '''
root_free_kb() { echo "$TEST_FREE"; }
sudo() { if [ "$1" = python3 ]; then echo "$TEST_PENDING"; fi; }
rm() { :; }
mkdir() { :; }
find() { :; }
run_bounded_restart_artifact_cleanup() { :; }
emergency_disk_preflight
'''
    result = bash(script, TEST_FREE=str(free), TEST_PENDING=pending,
        GATEWAY_RESTART_AUTHORITY_ROOT=str(authority), LEADPOET_REPO_ROOT=str(tmp_path / 'old-missing'),
        ENV_BACKUP_DIR=str(tmp_path / 'backups'), ENV_CLONE=str(tmp_path / 'clone'),
        ENV_SECRET=str(tmp_path / 'secret'), MIN_FREE_KB='10485760')
    assert result.returncode == 0, result.stderr
    assert ('candidate-helper' in result.stdout) == expected


def test_canonical_high_space_pending_read_error_is_not_ignored():
    script = '\n'.join(function(GATEWAY, name) for name in (
        'pending_docker_reconcile', 'docker_build_required_kb', 'emergency_disk_preflight')) + '''
root_free_kb() { echo 20000000; }
sudo() { return 1; }
emergency_disk_preflight
echo unsafe-success
'''
    result = bash(script, LEADPOET_REPO_ROOT='/old')
    assert result.returncode != 0 and 'unsafe-success' not in result.stdout


@pytest.mark.parametrize('value', ['0', '-1', 'bad', '9223372036854775808'])
def test_canonical_build_capacity_setting_is_strict(value):
    result = bash(function(GATEWAY, 'docker_build_required_kb') + '\ndocker_build_required_kb',
                  GATEWAY_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES=value)
    assert result.returncode != 0


@pytest.mark.parametrize('initial,current,ok', [(0, 0, True), (1, 1, True), (0, 1, False), (1, 0, False)])
def test_preserving_empty_lane_checks_gateway_presence_and_identity(initial, current, ok):
    script = function(RECLAIM, 'require_same_online_gateway') + '''
inspect_exact_host_gateway_runtime() { :; }
require_same_online_gateway test
'''
    result = bash(script, INITIAL_HOST_GATEWAY_LIVE=str(initial), HOST_GATEWAY_LIVE=str(current),
                  ONLINE_GATEWAY_PID='50', HOST_GATEWAY_PID='50',
                  ONLINE_GATEWAY_START_TIME_TICKS='90', HOST_GATEWAY_START_TIME_TICKS='90')
    assert (result.returncode == 0) is ok


@pytest.mark.parametrize('names', ['', 'retained-volume'])
def test_empty_lane_volume_manifest_is_stable_and_binds_metadata(names):
    script = function(RECLAIM, 'online_volume_manifest') + '''
run_bounded_daemon_inventory() {
  if [ "$2" = volume ] && [ "$3" = ls ]; then printf '%s' "$TEST_NAMES";
  else printf '[{"Name":"retained-volume","Driver":"local","Mountpoint":"/data","Options":null,"Labels":null,"Scope":"local","CreatedAt":"old"}]'; fi
}
online_volume_manifest
'''
    result = bash(script, TEST_NAMES=names)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64


def test_canonical_cleanup_has_no_broad_prune_and_preserving_paths_are_terminal():
    assert 'docker container prune' not in GATEWAY
    assert 'docker system prune' not in GATEWAY
    hook = function(GATEWAY, 'reset_orphaned_docker_storage_if_needed')
    assert 'docker_build_required_kb' in hook and 'pending_docker_reconcile' in hook
    assert 'VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1' in hook
    assert 'GATEWAY_RESTART_AUTHORITY_ROOT' in hook
    preserving = RECLAIM[RECLAIM.index('LIVE_CONTAINER_IDS='):RECLAIM.index('run_prune_with_retry image docker image prune')]
    assert 'if [ "$HOST_GATEWAY_LIVE" -eq 1 ] || [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" = "1" ]' in preserving
    assert preserving.count('exit 0') == 2
    assert 'ONLINE_VOLUME_MANIFEST=' in preserving
    assert 'zero-runtime reconciliation refuses live containers' in preserving
