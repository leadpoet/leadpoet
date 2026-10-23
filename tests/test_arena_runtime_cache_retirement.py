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
                                 'cache_hardlink', 'source_changed', 'source_writable', 'active', 'mount', 'in_use',
                                 'root_link', 'wrong_sha', 'wrong_owner'])
def test_unproved_archive_or_cache_is_preserved(archive, case, tmp_path):
    kwargs, root = archive
    cache = root/'pkg/__pycache__'
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
    elif case == 'in_use':
        process = kwargs['proc_root']/'123'; process.mkdir()
        (process/'cwd').symlink_to(root)
        (process/'fd').mkdir()
    elif case == 'root_link':
        kwargs['runtime_root'] = tmp_path/'linked-runtime'
        kwargs['runtime_root'].symlink_to(root.parent, target_is_directory=True)
    elif case == 'wrong_sha':
        kwargs['sha'] = 'f'*40
    elif case == 'wrong_owner':
        kwargs['owner_uid'] += 1
    with pytest.raises((ValueError, RuntimeError, OSError)):
        cleanup.retire_runtime_caches(**kwargs)
    assert (cache/'__init__.cpython-311.pyc').read_bytes() == (b'preserve' if case == 'cache_hardlink' else b'disposable cache')
    if case in ('cache_link', 'cache_hardlink'):
        assert outside.read_text() == 'preserve'


def test_cache_source_requires_exact_tracked_source_name():
    assert cleanup._cache_source(Path('pkg/__pycache__/different.cpython-311.pyc')) == Path('pkg/different.py')
    assert cleanup._cache_source(Path('pkg/__pycache__/__init__.cpython-311.pyc')) == Path('pkg/__init__.py')
    assert cleanup._cache_source(Path('module.pyc')) == Path('module.py')
    assert cleanup._cache_source(Path('pkg/__pycache__/unknown.txt')) is None


def test_all_three_retirement_paths_use_narrow_helper():
    script = (Path(__file__).resolve().parents[1]/'gw_restart.sh').read_text()
    assert script.count('retire_arena_runtime_tree "') == 3
    assert 'chmod -R u+w "$old_tree"' not in script
    assert 'chmod -R u+w "$ARENA_CANDIDATE_TREE"' not in script
    assert 'chmod -R u+w "$ARENA_PREVIOUS_TREE"' not in script
    helper = script.split('retire_arena_runtime_tree() {', 1)[1].split('\n}', 1)[0]
    assert 'sudo env -i' in helper and '--owner-uid "$(id -u)"' in helper
    assert 'find "$tree" -type d -exec chmod u+w {} +' in helper
    assert helper.index('sudo env -i') < helper.index('find "$tree"') < helper.index('rm -rf')


def test_apply_rechecks_other_processes_with_own_archive_descriptor_closed(archive, monkeypatch):
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
    assert checks == ['closed', 'closed']
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


@pytest.mark.parametrize('privileged_status', [0, 1])
def test_shared_shell_helper_propagates_guard_denial_before_source_permissions(archive, tmp_path, privileged_status):
    import shlex
    import subprocess
    import sys
    kwargs, root = archive
    script = (Path(__file__).resolve().parents[1]/'gw_restart.sh').read_text()
    function = 'retire_arena_runtime_tree() {' + script.split('retire_arena_runtime_tree() {', 1)[1].split('\n}', 1)[0] + '\n}'
    capture = tmp_path/'privileged-arguments'
    harness = 'set -euo pipefail\n'
    for name, value in dict(GATEWAY_PREFLIGHT_TREE='/verified', LEADPOET_REPO_ROOT=kwargs['repo_root'],
                            GATEWAY_PYTHON_BIN=sys.executable).items():
        harness += f'{name}={shlex.quote(str(value))}\n'
    harness += f'arena_runtime_tree() {{ printf "%s\\n" {shlex.quote(str(root))}; }}\n'
    harness += f'sudo() {{ printf "%s\\n" "$@" > {shlex.quote(str(capture))}; return {privileged_status}; }}\n'
    harness += function + f'\nretire_arena_runtime_tree 0 {kwargs["sha"]} {shlex.quote(str(root))}\n'
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True, timeout=10)
    assert result.returncode == privileged_status, result.stderr
    arguments = capture.read_text().splitlines()
    assert arguments[:4] == ['env', '-i', 'PATH=/usr/bin:/bin', 'PYTHONPATH=/verified']
    assert arguments[-2:] == ['--owner-uid', str(os.getuid())]
    if privileged_status:
        assert (root/'module.py').read_text() == 'VALUE = 1\n'
        assert stat.S_IMODE((root/'module.py').stat().st_mode) == 0o444
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
    else:
        assert not root.exists()
