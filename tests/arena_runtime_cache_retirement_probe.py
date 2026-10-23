#!/usr/bin/env python3
"""Bounded root-owned cache regression probe; creates only fresh /tmp fixtures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import stat
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import arena_runtime_cache_retirement as cleanup
from scripts.arena_ingress_config import commit_state, render


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), '-c', 'safe.directory='+str(repo),
        '-c', 'core.hooksPath=/dev/null', *args], text=True, stderr=subprocess.PIPE).strip()


def fixture(base: Path, owner_uid: int | None = None) -> dict:
    owner_uid = os.getuid() if owner_uid is None else owner_uid
    base.mkdir(mode=0o755, parents=True)
    base.chmod(0o755)
    repo = base / 'repo'; repo.mkdir()
    git(repo, 'init', '-q'); git(repo, 'config', 'user.email', 'runtime-probe@example.invalid')
    git(repo, 'config', 'user.name', 'Runtime Probe')
    (repo/'module.py').write_text('VALUE = 1\n')
    (repo/'pkg').mkdir(); (repo/'pkg/__init__.py').write_text('VALUE = 2\n')
    git(repo, 'add', 'module.py', 'pkg/__init__.py'); git(repo, 'commit', '-qm', 'fixture')
    sha = git(repo, 'rev-parse', 'HEAD')
    root = base/'arena-runtime'; root.mkdir(mode=0o700)
    archive = root/f'slot0-{sha}'; archive.mkdir()
    shutil.copy2(repo/'module.py', archive/'module.py'); (archive/'pkg').mkdir()
    shutil.copy2(repo/'pkg/__init__.py', archive/'pkg/__init__.py')
    active = root/f'slot1-{sha}'; active.mkdir(); (active/'active.txt').write_text('unchanged')
    if os.geteuid() == 0:
        for path in [repo, *repo.rglob('*'), root, archive, *archive.rglob('*'), active, active/'active.txt']:
            os.chown(path, owner_uid, -1)
    cache = archive/'pkg/__pycache__'; cache.mkdir(mode=0o700)
    (cache/'__init__.cpython-311.pyc').write_bytes(b'disposable cache')
    (cache/'__init__.cpython-311.pyc').chmod(0o600)
    (archive/'module.pyc').write_bytes(b'legacy cache'); (archive/'module.pyc').chmod(0o600)
    for path in [archive, archive/'pkg']:
        path.chmod(0o555)
    (archive/'module.py').chmod(0o444); (archive/'pkg/__init__.py').chmod(0o444)
    state, caddy = base/'ingress.json', base/'Caddyfile'
    caddy.write_text(render(arena_edge_port=8891, backend='closed', direct=True))
    commit_state(state, caddy, mode='dual', slot=1, sha=sha, backend='closed')
    proc = base/'proc'; proc.mkdir()
    mountinfo = base/'mountinfo'; mountinfo.write_text('1 0 0:1 / / rw - rootfs rootfs rw\n')
    return dict(repo_root=repo, runtime_root=root, slot=0, sha=sha, owner_uid=owner_uid,
        ingress_state=state, caddy_file=caddy, proc_root=proc, mountinfo=mountinfo)


def archive_path(kwargs: dict) -> Path:
    return kwargs['runtime_root']/f"slot{kwargs['slot']}-{kwargs['sha']}"


def run_probe(owner_uid: int) -> dict:
    if os.geteuid() != 0 or owner_uid <= 0:
        raise SystemExit('run this isolated fixture as root with a non-root --owner-uid')
    owner_gid = pwd.getpwuid(owner_uid).pw_gid
    results = {}
    with tempfile.TemporaryDirectory(prefix='arena-cache-retirement-probe-') as temporary:
        root = Path(temporary); root.chmod(0o755)
        for case in ('unknown', 'root_source', 'link', 'active', 'root_cache'):
            kwargs = fixture(root/case, owner_uid); archive = archive_path(kwargs)
            cache = archive/'pkg/__pycache__'; cached = cache/'__init__.cpython-311.pyc'
            before = (archive/'module.py').read_bytes()
            active = kwargs['runtime_root']/f"slot1-{kwargs['sha']}"/'active.txt'
            if case == 'unknown':
                (cache/'unknown.txt').write_text('preserve')
            elif case == 'root_source':
                os.chown(archive/'module.py', 0, 0)
            elif case == 'link':
                outside = root/'outside'; outside.write_text('preserve outside')
                (cache/'other.pyc').symlink_to(outside)
            elif case == 'active':
                kwargs['caddy_file'].write_text(render(arena_edge_port=8890, backend='closed', direct=True))
                commit_state(kwargs['ingress_state'], kwargs['caddy_file'], mode='dual',
                             slot=0, sha=kwargs['sha'], backend='closed')
            if case != 'root_cache':
                try:
                    cleanup.retire_runtime_caches(**kwargs, apply=True)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError(f'{case} was not refused')
                assert cached.exists() and (archive/'module.py').read_bytes() == before
                assert active.read_text() == 'unchanged'
                if case == 'link':
                    assert outside.read_text() == 'preserve outside'
                results[case] = 'refused_without_deletion'
                continue
            assert cache.stat().st_uid == 0 and stat.S_IMODE(cache.stat().st_mode) == 0o700
            assert cached.stat().st_uid == 0 and stat.S_IMODE(cached.stat().st_mode) == 0o600
            old = subprocess.run(['chmod', '-R', 'u+w', str(archive)], user=owner_uid,
                group=owner_gid, extra_groups=[], capture_output=True, text=True)
            assert old.returncode != 0, 'old recursive chmod unexpectedly passed'
            assert '__pycache__' in old.stderr or 'module.pyc' in old.stderr
            assert cache.stat().st_uid == 0 and stat.S_IMODE(cache.stat().st_mode) == 0o700
            # Restore only this probe's source modes after the old-code replay.
            for path in (archive, archive/'pkg'):
                path.chmod(0o555)
            for path in (archive/'module.py', archive/'pkg/__init__.py'):
                path.chmod(0o444)
            kwargs['proc_root'] = Path('/proc')
            kwargs['mountinfo'] = Path('/proc/self/mountinfo')
            report = cleanup.retire_runtime_caches(**kwargs, apply=True)
            assert report['cache_paths'] == 3 and not cache.exists() and not (archive/'module.pyc').exists()
            assert (archive/'module.py').read_bytes() == before
            assert stat.S_IMODE((archive/'module.py').stat().st_mode) == 0o444
            assert (archive/'module.py').stat().st_uid == owner_uid
            subprocess.run(['find', str(archive), '-type', 'd', '-exec', 'chmod', 'u+w', '{}', '+'],
                user=owner_uid, group=owner_gid, extra_groups=[], check=True)
            subprocess.run(['rm', '-rf', '--', str(archive)], user=owner_uid,
                group=owner_gid, extra_groups=[], check=True)
            assert not archive.exists() and active.read_text() == 'unchanged'
            results[case] = 'old_chmod_failed_new_cache_cleanup_and_unprivileged_retirement_passed'
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-uid', required=True, type=int)
    print(json.dumps(run_probe(parser.parse_args().owner_uid), indent=2, sort_keys=True))
