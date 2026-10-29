from pathlib import Path
import os
import stat

import pytest

from scripts import arena_runtime_cache_retirement as cleanup
from scripts.arena_ingress_config import commit_state, render
from tests.arena_runtime_cache_retirement_probe import fixture, archive_path


@pytest.fixture
def archive(tmp_path):
    kwargs = fixture(tmp_path/'case')
    try:
        yield kwargs, archive_path(kwargs)
    finally:
        for path in [kwargs['runtime_root'], *kwargs['runtime_root'].rglob('*')]:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)


def test_complete_plan_preserves_source_and_active_tree(archive):
    kwargs, root = archive
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mode)
              for p in (root/'module.py', root/'pkg/__init__.py')}
    result = cleanup.retire_runtime_caches(**kwargs)
    assert result['cache_paths'] == 3 and result['applied'] is False
    assert {p.relative_to(root): (p.read_bytes(), p.stat().st_mode)
            for p in (root/'module.py', root/'pkg/__init__.py')} == before
    assert (root/'pkg/__pycache__/__init__.cpython-311.pyc').exists()
    assert (kwargs['runtime_root']/f"slot1-{kwargs['sha']}"/'active.txt').read_text() == 'unchanged'


@pytest.mark.parametrize('case', ['unknown', 'unmatched_pyc', 'nested_cache_dir', 'cache_link',
                                 'cache_hardlink', 'source_changed', 'source_writable', 'active', 'mount',
                                 'root_link', 'wrong_sha', 'wrong_owner'])
@pytest.mark.parametrize('busy', [False, True])
def test_unproved_archive_or_cache_is_preserved(archive, case, tmp_path, busy):
    kwargs, root = archive
    cache = root/'pkg/__pycache__'
    if busy:
        process = kwargs['proc_root']/'123'; process.mkdir()
        (process/'cwd').symlink_to(root)
        (process/'fd').mkdir()
    if case == 'unknown':
        (cache/'unknown.txt').write_text('preserve')
    elif case == 'unmatched_pyc':
        (cache/'different.cpython-311.pyc').write_bytes(b'preserve')
    elif case == 'nested_cache_dir':
        (cache/'unknown').mkdir()
    elif case == 'cache_link':
        outside = tmp_path/'outside'; outside.write_text('preserve')
        (cache/'module.cpython-311.pyc').symlink_to(outside)
    elif case == 'cache_hardlink':
        outside = tmp_path/'outside'; outside.write_text('preserve')
        target = cache/'__init__.cpython-311.pyc'; target.unlink()
        os.link(outside, target)
    elif case == 'source_changed':
        (root/'module.py').chmod(0o644); (root/'module.py').write_text('changed')
        (root/'module.py').chmod(0o444)
    elif case == 'source_writable':
        (root/'module.py').chmod(0o644)
    elif case == 'active':
        kwargs['caddy_file'].write_text(render(arena_edge_port=8890, backend='closed', direct=True))
        commit_state(kwargs['ingress_state'], kwargs['caddy_file'], mode='dual', slot=0,
                     sha=kwargs['sha'], backend='closed')
    elif case == 'mount':
        kwargs['mountinfo'].write_text(f'1 0 0:1 / {cache} rw - rootfs rootfs rw\n')
    elif case == 'root_link':
        kwargs['runtime_root'] = tmp_path/'linked-runtime'
        kwargs['runtime_root'].symlink_to(root.parent, target_is_directory=True)
    elif case == 'wrong_sha':
        kwargs['sha'] = 'f'*40
    elif case == 'wrong_owner':
        kwargs['owner_uid'] += 1
    with pytest.raises((ValueError, RuntimeError, OSError)):
        cleanup.retire_runtime_caches(**kwargs, defer_if_busy=True)
    assert (cache/'__init__.cpython-311.pyc').read_bytes() == (b'preserve' if case == 'cache_hardlink' else b'disposable cache')
    if case in ('cache_link', 'cache_hardlink'):
        assert outside.read_text() == 'preserve'


def test_cache_source_requires_exact_tracked_source_name():
    assert cleanup._cache_source(Path('pkg/__pycache__/different.cpython-311.pyc')) == Path('pkg/different.py')
    assert cleanup._cache_source(Path('pkg/__pycache__/__init__.cpython-311.pyc')) == Path('pkg/__init__.py')
    assert cleanup._cache_source(Path('module.pyc')) == Path('module.py')
    assert cleanup._cache_source(Path('pkg/__pycache__/unknown.txt')) is None


def test_all_retirement_paths_use_narrow_helper():
    script = (Path(__file__).resolve().parents[1]/'gw_restart.sh').read_text()
    assert script.count('retire_arena_runtime_tree "') == 4
    assert 'chmod -R u+w "$old_tree"' not in script
    assert 'chmod -R u+w "$ARENA_CANDIDATE_TREE"' not in script
    assert 'chmod -R u+w "$ARENA_PREVIOUS_TREE"' not in script
    helper = script.split('retire_arena_runtime_tree() {', 1)[1].split('\n}', 1)[0]
    assert 'sudo env -i' in helper and '--owner-uid "$(id -u)"' in helper
    assert 'find "$tree" -type d -exec chmod u+w {} +' in helper
    assert helper.index('sudo env -i') < helper.index('find "$tree"') < helper.index('rm -rf')


def test_apply_checks_processes_with_own_archive_descriptor_closed(archive, monkeypatch):
    kwargs, root = archive
    opened = set()
    real_open, real_close = os.open, os.close
    checks = []
    def record_open(*args, **keywords):
        fd = real_open(*args, **keywords); opened.add(fd); return fd
    def record_close(fd):
        opened.discard(fd); return real_close(fd)
    root_identity = root.stat().st_dev, root.stat().st_ino
    def process_paths(_proc):
        assert all((os.fstat(fd).st_dev, os.fstat(fd).st_ino) != root_identity for fd in opened)
        checks.append('closed')
        return set()
    monkeypatch.setattr(cleanup.os, 'open', record_open)
    monkeypatch.setattr(cleanup.os, 'close', record_close)
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(cleanup, '_read_process_paths', process_paths)
    removed = []
    monkeypatch.setattr(cleanup.os, 'unlink', lambda name, **kw: removed.append(name))
    monkeypatch.setattr(cleanup.os, 'rmdir', lambda name, **kw: removed.append(name))
    assert cleanup.retire_runtime_caches(**kwargs, apply=True)['applied'] is True
    assert checks == ['closed']
    assert set(removed) == {'module.pyc', '__init__.cpython-311.pyc', '__pycache__'}
    assert (root/'module.py').read_text() == 'VALUE = 1\n'


def test_walk_permission_error_is_not_silently_skipped(archive, monkeypatch):
    kwargs, root = archive
    def unreadable(*args, **keywords):
        keywords['onerror'](PermissionError('denied fixture subtree'))
        yield None
    monkeypatch.setattr(cleanup.os, 'fwalk', unreadable)
    with pytest.raises(cleanup.ArenaRuntimeCacheRetirementError, match='inspect every path'):
        cleanup.retire_runtime_caches(**kwargs)
    assert (root/'pkg/__pycache__/__init__.cpython-311.pyc').exists()


def test_new_mount_before_deletion_is_rejected(archive, monkeypatch):
    kwargs, root = archive
    checks = []
    def mounts(_path):
        checks.append(True)
        return set() if len(checks) == 1 else {root/'pkg/__pycache__'}
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(cleanup, '_read_mount_points', mounts)
    with pytest.raises(cleanup.ArenaRuntimeCacheRetirementError, match='serving identity changed'):
        cleanup.retire_runtime_caches(**kwargs, apply=True)
    assert (root/'pkg/__pycache__/__init__.cpython-311.pyc').exists()


@pytest.mark.parametrize('privileged_status,report_status', [
    (0, 'cache_cleanup_complete'), (0, 'deferred_in_use'), (1, 'deferred_in_use'),
    (0, 'unknown'), (0, 'planned'), (0, 'malformed'), (0, 'wrong_archive'),
    (0, 'wrong_applied'), (0, 'wrong_cache_count'),
])
@pytest.mark.parametrize('retirement_path', [0, 1, 2, 3])
def test_shared_shell_helper_propagates_guard_denial_before_source_permissions(archive, tmp_path, privileged_status, report_status, retirement_path):
    import json
    import shlex
    import subprocess
    import sys
    kwargs, root = archive
    script = (Path(__file__).resolve().parents[1]/'gw_restart.sh').read_text()
    function = 'retire_arena_runtime_tree() {' + script.split('retire_arena_runtime_tree() {', 1)[1].split('\n}', 1)[0] + '\n}'
    capture = tmp_path/'privileged-arguments'
    harness = 'set -euo pipefail\n'
    for name, value in dict(GATEWAY_PREFLIGHT_TREE='/verified', LEADPOET_REPO_ROOT=kwargs['repo_root'],
                            GATEWAY_PYTHON_BIN=sys.executable, old_slot=0, old_tree=root,
                            PREPARED_GATEWAY_SHA=kwargs['sha'], old_sha=kwargs['sha'], ARENA_CANDIDATE_SLOT=0,
                            ARENA_CANDIDATE_TREE=root, ARENA_PREVIOUS_SLOT=0,
                            ARENA_PREVIOUS_SHA=kwargs['sha'], ARENA_PREVIOUS_TREE=root).items():
        harness += f'{name}={shlex.quote(str(value))}\n'
    harness += f'arena_runtime_tree() {{ printf "%s\\n" {shlex.quote(str(root))}; }}\n'
    payload = dict(archive=str(root), cache_paths=3, applied=report_status == 'cache_cleanup_complete',
                   status=report_status)
    if report_status.startswith('wrong_'):
        payload['status'] = 'deferred_in_use'
        payload[dict(wrong_archive='archive', wrong_applied='applied', wrong_cache_count='cache_paths')[report_status]] = dict(
            wrong_archive=str(root.parent), wrong_applied=True, wrong_cache_count=True)[report_status]
    report = 'not JSON' if report_status == 'malformed' else json.dumps(payload)
    harness += f'sudo() {{ printf "%s\\n" "$@" > {shlex.quote(str(capture))}; printf "%s\\n" {shlex.quote(report)}; return {privileged_status}; }}\n'
    calls = [line.strip() for line in script.splitlines() if line.strip().startswith('retire_arena_runtime_tree \"')]
    assert len(calls) == 4
    harness += function + '\n' + calls[retirement_path] + '\n'
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True, timeout=10)
    expected_status = 0 if privileged_status == 0 and report_status in ('cache_cleanup_complete', 'deferred_in_use') else 1
    assert result.returncode == expected_status, result.stderr
    arguments = capture.read_text().splitlines()
    assert arguments[:4] == ['env', '-i', 'PATH=/usr/bin:/bin', 'PYTHONPATH=/verified']
    assert arguments[-3:] == ['--owner-uid', str(os.getuid()), '--defer-if-busy']
    if expected_status or report_status == 'deferred_in_use':
        assert (root/'module.py').read_text() == 'VALUE = 1\n'
        assert stat.S_IMODE((root/'module.py').stat().st_mode) == 0o444
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
    else:
        assert not root.exists()


@pytest.mark.parametrize('reader_kind', ['cwd', 'fd', 'maps'])
@pytest.mark.parametrize('apply', [False, True])
def test_proved_busy_inactive_archive_is_deferred_untouched(archive, monkeypatch, reader_kind, apply):
    kwargs, root = archive
    process = kwargs['proc_root']/'123'; process.mkdir()
    (process/'fd').mkdir()
    target = root/'module.py'
    if reader_kind == 'cwd':
        (process/'cwd').symlink_to(root)
    elif reader_kind == 'fd':
        (process/'fd/7').symlink_to(target)
    else:
        (process/'maps').write_text(f'000000-001000 r--p 00000000 00:00 0 {target}\n')
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    before = {p.relative_to(root): (p.lstat().st_mode, p.read_bytes() if p.is_file() else None)
              for p in [root, *root.rglob('*')]}
    report = cleanup.retire_runtime_caches(**kwargs, apply=apply, defer_if_busy=True)
    assert report == dict(archive=str(root), cache_paths=3, applied=False, status='deferred_in_use')
    assert {p.relative_to(root): (p.lstat().st_mode, p.read_bytes() if p.is_file() else None)
            for p in [root, *root.rglob('*')]} == before
    assert target.read_text() == 'VALUE = 1\n'
    assert (process/'fd').is_dir()


def test_late_reader_is_deferred_without_cache_deletion(archive, monkeypatch):
    kwargs, root = archive
    def process_paths(_proc):
        return {root/'module.py'}
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(cleanup, '_read_process_paths', process_paths)
    report = cleanup.retire_runtime_caches(**kwargs, apply=True, defer_if_busy=True)
    assert report['status'] == 'deferred_in_use' and report['applied'] is False
    assert (root/'module.pyc').read_bytes() == b'legacy cache'
    assert stat.S_IMODE(root.stat().st_mode) == 0o555


def test_process_inspection_failure_stays_fatal(archive, monkeypatch):
    kwargs, root = archive
    def denied(_proc):
        raise PermissionError('process inspection denied')
    monkeypatch.setattr(cleanup, '_read_process_paths', denied)
    with pytest.raises(PermissionError, match='process inspection denied'):
        cleanup.retire_runtime_caches(**kwargs)
    assert (root/'module.pyc').read_bytes() == b'legacy cache'


def test_ingress_change_with_busy_reader_remains_fatal(archive, monkeypatch):
    kwargs, root = archive
    initial = cleanup.inspect_ingress(kwargs['ingress_state'], kwargs['caddy_file'])
    states = iter([initial, {**initial, 'slot': 0}])
    monkeypatch.setattr(cleanup, 'inspect_ingress', lambda *_args: next(states))
    monkeypatch.setattr(cleanup, '_read_process_paths', lambda _proc: {root})
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    with pytest.raises(cleanup.ArenaRuntimeCacheRetirementError, match='serving identity changed'):
        cleanup.retire_runtime_caches(**kwargs, apply=True, defer_if_busy=True)
    assert (root/'module.pyc').read_bytes() == b'legacy cache'


@pytest.mark.parametrize('apply', [False, True])
def test_busy_archive_requires_explicit_deferral_opt_in(archive, monkeypatch, apply):
    kwargs, root = archive
    monkeypatch.setattr(cleanup.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(cleanup, '_read_process_paths', lambda _proc: {root})
    with pytest.raises(cleanup.ArenaRuntimeCacheRetirementError, match='still in use'):
        cleanup.retire_runtime_caches(**kwargs, apply=apply)
    assert (root/'module.pyc').read_bytes() == b'legacy cache'
    assert stat.S_IMODE(root.stat().st_mode) == 0o555


@pytest.mark.parametrize('legacy_caller', [True, False])
@pytest.mark.parametrize('busy', [True, False])
def test_old_and_new_shell_callers_use_new_cli_without_deleting_busy_archive(archive, tmp_path, legacy_caller, busy):
    import shlex
    import subprocess
    import sys
    kwargs, root = archive
    repo = Path(__file__).resolve().parents[1]
    process = kwargs['proc_root']/'123'; process.mkdir()
    if busy:
        (process/'cwd').symlink_to(root)
    else:
        # No privileged unlink is needed for this ordinary source-only archive.
        root.chmod(0o755)
        (root/'pkg').chmod(0o755)
        (root/'module.pyc').unlink()
        (root/'pkg/__pycache__/__init__.cpython-311.pyc').unlink()
        (root/'pkg/__pycache__').rmdir()
        (root/'pkg').chmod(0o555)
        root.chmod(0o555)
    (process/'fd').mkdir()
    # Run the real CLI parser and function in a child process. Redirect only
    # production paths and the root UID check to this isolated, read-only case.
    runner = tmp_path/'fixture-cli.py'
    paths = {key: str(kwargs[key]) for key in ('ingress_state', 'caddy_file', 'proc_root', 'mountinfo')}
    runner.write_text(
        'from pathlib import Path\nimport sys\n'
        f'sys.path.insert(0, {str(repo)!r})\n'
        'from scripts import arena_runtime_cache_retirement as cleanup\n'
        f'cleanup.RUNTIME_ROOT = Path({str(kwargs["runtime_root"])!r})\n'
        'cleanup.os.geteuid = lambda: 0\n'
        'original = cleanup.retire_runtime_caches\n'
        'def fixture_retirement(**kwargs):\n'
        f'    kwargs.update({{key: Path(value) for key, value in {paths!r}.items()}})\n'
        '    return original(**kwargs)\n'
        'cleanup.retire_runtime_caches = fixture_retirement\n'
        'raise SystemExit(cleanup.main())\n'
    )
    script = (repo/'gw_restart.sh').read_text()
    current = 'retire_arena_runtime_tree() {' + script.split('retire_arena_runtime_tree() {', 1)[1].split('\n}', 1)[0] + '\n}'
    # The installed pre-capability controller relies only on the CLI's exit
    # status. It does not inspect the new JSON before chmod and source removal.
    legacy = '''retire_arena_runtime_tree() {
  local slot="$1" sha="$2" tree="$3"
  sudo env -i PATH=/usr/bin:/bin PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \\
    "$GATEWAY_PYTHON_BIN" -B "$GATEWAY_PREFLIGHT_TREE/scripts/arena_runtime_cache_retirement.py" \\
    --repo-root "$LEADPOET_REPO_ROOT" --slot "$slot" --sha "$sha" --owner-uid "$(id -u)" || return 1
  find "$tree" -type d -exec chmod u+w {} + || return 1
  rm -rf -- "$tree"
}'''
    harness = 'set -euo pipefail\n'
    for name, value in dict(GATEWAY_PREFLIGHT_TREE=repo, LEADPOET_REPO_ROOT=kwargs['repo_root'],
                            GATEWAY_PYTHON_BIN=sys.executable).items():
        harness += f'{name}={shlex.quote(str(value))}\n'
    harness += f'arena_runtime_tree() {{ printf "%s\\n" {shlex.quote(str(root))}; }}\n'
    harness += 'sudo() { while [ "$1" != --repo-root ]; do shift; done; '
    harness += f'{shlex.quote(sys.executable)} {shlex.quote(str(runner))} "$@"; }}\n'
    harness += (legacy if legacy_caller else current)
    harness += f'\nretire_arena_runtime_tree 0 {kwargs["sha"]} {shlex.quote(str(root))}\n'
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True, timeout=10)
    assert result.returncode == (1 if legacy_caller and busy else 0), result.stderr
    if busy:
        assert 'still in use' in (result.stderr if legacy_caller else result.stdout)
        assert (root/'module.pyc').read_bytes() == b'legacy cache'
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
        assert (process/'cwd').resolve() == root
    else:
        assert not root.exists()
