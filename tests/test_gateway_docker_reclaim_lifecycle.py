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


@pytest.mark.parametrize('source', [GATEWAY, RECLAIM], ids=['gateway', 'reclaim'])
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
    assert 'docker_build_required_kb' in hook
    assert 'if [' not in hook
    assert 'VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1' in hook
    assert 'GATEWAY_RESTART_AUTHORITY_ROOT' in hook
    preserving = RECLAIM[RECLAIM.index('LIVE_CONTAINER_IDS='):RECLAIM.index('run_prune_with_retry image docker image prune')]
    assert 'if [ "$HOST_GATEWAY_LIVE" -eq 1 ] || [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" = "1" ]' in preserving
    assert preserving.count('exit 0') == 2
    assert 'ONLINE_VOLUME_MANIFEST=' in preserving
    assert 'zero-runtime reconciliation refuses live containers' in preserving


@pytest.mark.parametrize('allow,enters', [('0', False), ('1', True)])
def test_explicit_preserving_optin_audits_even_at_high_free_space(allow, enters):
    script = function(RECLAIM, 'protect_exact_host_gateway_runtime') + '''
inspect_exact_host_gateway_runtime() { HOST_GATEWAY_LIVE=1; }
available_bytes() { echo 50000000000; }
pending_docker_reconcile() { return 1; }
protect_exact_host_gateway_runtime entry
echo admitted
'''
    result = bash(script, ALLOW_LIVE_HOST_GATEWAY_PRUNE=allow,
                  LIVE_RUNTIME_MIN_FREE_BYTES='18000000000', REQUIRE_ZERO_RUNTIME_RECONCILE='0')
    assert result.returncode == 0
    assert ('admitted' in result.stdout) is enters


def test_explicit_optin_also_triggers_empty_runtime_audit_at_high_free_space():
    start = RECLAIM.index('  if [ "$AVAILABLE" -lt "$LIVE_RUNTIME_MIN_FREE_BYTES" ]', RECLAIM.index('ONLINE_RAW_RECLAIM_PERFORMED=0'))
    condition = RECLAIM[start:RECLAIM.index('; then', start)] + '; then echo audit; fi\n'
    assert 'ALLOW_LIVE_HOST_GATEWAY_PRUNE' in condition
    result = bash(condition, AVAILABLE='50000000000', LIVE_RUNTIME_MIN_FREE_BYTES='18000000000',
                  REQUIRE_ZERO_RUNTIME_RECONCILE='0', ALLOW_LIVE_HOST_GATEWAY_PRUNE='1')
    assert result.returncode == 0 and result.stdout.strip() == 'audit'


def test_normal_maintenance_always_calls_preserving_helper_with_retained_state(tmp_path):
    helper_dir = tmp_path / 'validator_tee/scripts'
    helper_dir.mkdir(parents=True)
    retained = tmp_path / 'retained-object'
    retained.write_bytes(b'rollback-image-and-volume')
    (helper_dir / 'reclaim_docker_storage_v2.sh').write_text('''
set -eu
[ "$VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE" = 1 ]
[ "$(cat "$TEST_RETAINED")" = rollback-image-and-volume ]
echo audited
''')
    script = '\n'.join(function(GATEWAY, name) for name in (
        'docker_build_required_kb', 'reset_orphaned_docker_storage_if_needed'))
    result = bash(script + '\nreset_orphaned_docker_storage_if_needed\n',
        LEADPOET_REPO_ROOT=str(tmp_path), TEST_RETAINED=str(retained))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'audited'
    assert retained.read_bytes() == b'rollback-image-and-volume'


@pytest.mark.parametrize('entry,cold', [('local', False), ('role', False), ('role', True)])
@pytest.mark.parametrize('managed,system', [(True, 'Linux'), (False, 'Linux'), (True, 'Darwin')])
@pytest.mark.parametrize('build_status,cleanup_status', [(0, 0), (17, 0), (0, 19), (17, 19)])
def test_build_batch_teardown_preserves_status_scope_and_retained_objects(tmp_path, entry, cold, managed, system, build_status, cleanup_status):
    source = Path('gateway/tee/build_' + ('local_release_v2' if entry == 'local' else 'role_enclaves') + '.sh').read_text()
    helper_dir = tmp_path / 'validator_tee/scripts'
    helper_dir.mkdir(parents=True)
    log = tmp_path / 'cleanup.log'
    retained = tmp_path / 'retained-image-volume-pg'
    retained.write_bytes(b'preserved')
    cold_root = tmp_path / '.gateway-eif-cold-build.test'
    if cold:
        cold_root.mkdir()
        (cold_root / 'owned-temporary').write_text('owned')
    managed_marker = tmp_path / 'restart-controller'
    if managed:
        managed_marker.mkdir()
    (helper_dir / 'docker_operation_lock_v2.sh').write_text('''
leadpoet_acquire_docker_operation_lock_v2() {
  export LEADPOET_DOCKER_OPERATION_LOCK_HELD=1 LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD=1
}
leadpoet_release_docker_operation_lock_v2() { echo unlock >> "$TEST_LOG"; }
''')
    (helper_dir / 'reclaim_docker_storage_v2.sh').write_text('''
set -eu
[ "$LEADPOET_DOCKER_OPERATION_LOCK_HELD" = 1 ]
[ "$LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD" = 1 ]
[ "$VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE" = 1 ]
[ "$(cat "$TEST_RETAINED")" = preserved ]
echo reclaim >> "$TEST_LOG"
exit "$TEST_CLEANUP_STATUS"
''')
    start = source.index('MANAGED_DOCKER_HOST=0')
    boundary = source[start:source.index('leadpoet_acquire_docker_operation_lock_v2', start) + len('leadpoet_acquire_docker_operation_lock_v2')]
    # The filesystem fixture replaces only the canonical managed-host directory.
    boundary = boundary.replace('/home/ec2-user/.config/leadpoet/restart-controller', str(managed_marker))
    setup = 'uname() { echo "$TEST_SYSTEM"; }\n' + boundary + '\n'
    if entry == 'local':
        setup += function(source, 'cleanup') + '\ntrap cleanup EXIT\n'
    else:
        setup += function(source, 'cleanup_cold_build_root') + '\n' + function(source, 'cleanup_build') + '\ntrap cleanup_build EXIT\n'
    environment = dict(TEST_SYSTEM=system, CANDIDATE_ROOT=str(tmp_path), REPO_ROOT=str(tmp_path),
        TEMPORARY_ROOT=str(tmp_path / 'owned-temporary'), COLD_BUILD_ROOT=str(cold_root) if cold else '',
        COLD_DOCKER_BUILD_STARTED='1' if cold else '0', EIF_ROOT=str(tmp_path),
        TEST_LOG=str(log), TEST_RETAINED=str(retained), TEST_CLEANUP_STATUS=str(cleanup_status))
    # A custom or inherited lock by itself must never admit host maintenance.
    environment['LEADPOET_DOCKER_OPERATION_LOCK_FILE'] = str(tmp_path / 'custom-docker.lock')
    if cold and build_status == 0:
        setup += '\ncleanup_cold_build_root\nCOLD_BUILD_ROOT=""\n'
    result = bash(setup + f'\nexit {build_status}\n', **environment)
    cleans_docker = managed and system == 'Linux' and (entry == 'local' or cold)
    expected = build_status or (cleanup_status if cleans_docker else 0)
    assert result.returncode == expected, result.stderr
    lines = log.read_text().splitlines() if log.exists() else []
    assert lines.count('reclaim') == int(cleans_docker)
    if entry == 'local':
        assert lines[-1] == 'unlock'
    assert ('cleanup failed' in result.stderr) == (cleans_docker and cleanup_status != 0)
    assert retained.read_bytes() == b'preserved'
    assert not cold_root.exists()


def test_role_temporary_cleanup_keeps_its_original_success_timing():
    source = Path('gateway/tee/build_role_enclaves.sh').read_text()
    archive = source.index('--archive')
    restore = source.index('--restore', archive)
    cleanup = source.index('    cleanup_cold_build_root', restore)
    assert archive < restore < cleanup < source.index('echo "Built role EIFs:"')
    assert source.count('reclaim_docker_storage_v2.sh') == 1
    start = source.index('    COLD_DOCKER_BUILD_STARTED=1')
    first_build = source.index('    sudo env', start)
    assert start < first_build
    assert source[start:first_build].strip() == 'COLD_DOCKER_BUILD_STARTED=1'


@pytest.mark.parametrize('domain', ['gateway', 'validator'])
@pytest.mark.parametrize('host', ['empty', 'occupied', 'registry-cache', 'production-identity'])
def test_isolated_workflow_builders_never_reclaim_a_shared_host(tmp_path, domain, host):
    import yaml

    workflow_source = Path('.github/workflows/attested-v2-release.yml').read_text()
    workflow = yaml.safe_load(workflow_source)
    builds = workflow['jobs']['builds']
    assert builds['strategy']['matrix']['include'] == [
        {'domain': 'gateway', 'runner': 'ubuntu-24.04'},
        {'domain': 'validator', 'runner': 'runs-on'},
    ]
    assert 'attested-v2-isolated' in builds['runs-on']
    assert any(step.get('run') == 'bash .github/scripts/build_attested_release.sh'
               for step in builds['steps'])
    source = Path('.github/scripts/build_attested_release.sh').read_text()
    assert 'reclaim_docker_storage_v2.sh' not in workflow_source + source
    # Exercise the actual builder boundary through both domain gates. Stub
    # only external account/Docker inventory and unrelated Nitro installation;
    # every Docker mutation is an error, and the real isolation checker runs.
    preflight = source[:source.index('docker version --format')]
    retained = tmp_path / 'retained-image-volume'
    retained.write_bytes(b'rollback-state')
    commands = tmp_path / 'docker-commands'
    stubs = r"""
uname() { echo x86_64; }
aws() {
  [ "$*" = 'sts get-caller-identity --query Account --output text' ] || return 90
  printf '%s\n' "$TEST_ACCOUNT"
}
python3() {
  if [ "$1" = .github/scripts/verify_attested_nitro_toolchain.py ]; then return 0; fi
  [ "$1" = .github/scripts/verify_attested_builder_isolation.py ] || return 91
  command python3 "$@"
}
docker() {
  if [ "$*" = 'ps -aq' ]; then
    echo ps >> "$TEST_COMMANDS"
    printf '%s' "$TEST_CONTAINERS"
  elif [ "$1" = info ] && [ "$2" = --format ] && [ "$#" = 3 ]; then
    echo info >> "$TEST_COMMANDS"
    printf '%s' "$TEST_MIRRORS"
  else
    echo unsafe-docker-command >&2
    return 92
  fi
}
"""
    result = subprocess.run(['bash', '-c', stubs + preflight + '\necho admitted\n'],
        env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path),
             'GATEWAY_V2_BUILDER_DOMAIN': domain, 'RUNNER_TEMP': str(tmp_path),
             'RUNNER_ENVIRONMENT': 'self-hosted' if host == 'production-identity' else 'github-hosted',
             'TEST_ACCOUNT': '493765492819' if host == 'production-identity' else '899952150628',
             'TEST_COMMANDS': str(commands),
             'TEST_CONTAINERS': 'retained-container' if host == 'occupied' else '',
             'TEST_MIRRORS': '["https://cache.example.invalid/"]' if host == 'registry-cache' else '[]'},
        text=True, capture_output=True)
    assert (result.returncode == 0) == (host == 'empty'), result.stderr
    assert ('admitted' in result.stdout) == (host == 'empty')
    calls = commands.read_text().splitlines() if commands.exists() else []
    assert calls == ([] if host == 'production-identity' else ['ps'] if host == 'occupied' else ['ps', 'info'])
    assert 'unsafe-docker-command' not in result.stderr
    assert retained.read_bytes() == b'rollback-state'


def test_release_builder_isolation_preserves_production_role_cleanup_guards():
    import yaml
    workflow = yaml.safe_load(Path('.github/workflows/attested-v2-release.yml').read_text())
    builds = workflow['jobs']['builds']
    assert builds['strategy']['matrix']['include'] == [
        {'domain': 'gateway', 'runner': 'ubuntu-24.04'},
        {'domain': 'validator', 'runner': 'runs-on'},
    ]
    assert '/runner=attested-v2-isolated/spot=false/retry=false' in builds['runs-on']
    assert any(step.get('run') == 'bash .github/scripts/build_attested_release.sh'
               for step in builds['steps'])
    release = Path('.github/scripts/build_attested_release.sh').read_text()
    build = release.index('/usr/bin/time -v bash gateway/tee/build_release_evidence_v2.sh')
    assert release.index('verify_attested_builder_isolation.py') < build
    assert release.index('test -z "$existing_containers"') < build
    assert release.index('leadpoet_acquire_docker_operation_lock_v2') < build
    assert "trap 'leadpoet_release_docker_operation_lock_v2' EXIT" in release
    assert 'reclaim_docker_storage_v2.sh' not in release

    # Managed production hosts retain their existing terminal cleanup trap.
    role = Path('gateway/tee/build_role_enclaves.sh').read_text()
    cleanup = function(role, 'cleanup_build')
    assert 'trap cleanup_build EXIT' in role
    assert '[ "$MANAGED_DOCKER_HOST" = 1 ] && [ "$COLD_DOCKER_BUILD_STARTED" = 1 ]' in cleanup
    assert 'VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1' in cleanup
    assert 'reclaim_docker_storage_v2.sh' in cleanup
