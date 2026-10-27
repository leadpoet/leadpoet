#!/usr/bin/env python3
"""Remove only proved Python caches from an inactive exact Arena archive."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterator, Literal, TypedDict

from scripts.arena_ingress_config import inspect as inspect_ingress
from scripts.gateway_git_deploy import _candidate_blob_entries, verify_materialized_tree
from validator_tee.host.restart_artifact_cleanup_v2 import (
    _contains_nested_mount, _is_in_use, _read_mount_points, _read_process_paths,
)

RUNTIME_ROOT = Path('/home/ec2-user/.config/leadpoet/arena-runtime')
INGRESS_STATE = Path('/etc/caddy/arena-ingress-state.json')
CADDY_FILE = Path('/etc/caddy/Caddyfile')


class ArenaRuntimeCacheRetirementError(RuntimeError):
    """An archive or its disposable cache paths could not be proved."""


class ArenaRuntimeCacheRetirementResult(TypedDict):
    archive: str
    cache_paths: int
    applied: bool
    status: Literal['planned', 'cache_cleanup_complete', 'deferred_in_use']


def _no_link_ancestors(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for ancestor in (*reversed(path.parents), path):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise ArenaRuntimeCacheRetirementError('runtime ancestor is not a real directory')
    return path


@contextmanager
def _exact_git_directory(repo_root: Path) -> Iterator[None]:
    # Root only reads this exact already-verified deployment repository. Do not
    # change global Git trust or inherit additional injected Git configuration.
    prior = {key: value for key, value in os.environ.items()
             if key == 'GIT_CONFIG_COUNT' or key.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_'))}
    for key in prior:
        os.environ.pop(key)
    os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory',
                      GIT_CONFIG_VALUE_0=str(repo_root))
    try:
        yield
    finally:
        for key in list(os.environ):
            if key == 'GIT_CONFIG_COUNT' or key.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')):
                os.environ.pop(key)
        os.environ.update(prior)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid,
            value.st_nlink if stat.S_ISREG(value.st_mode) else 0)


def _cache_source(relative: Path) -> Path | None:
    if relative.suffix not in {'.pyc', '.pyo'}:
        return None
    if relative.parent.name == '__pycache__':
        try:
            return Path(importlib.util.source_from_cache(str(relative)))
        except ValueError:
            return None
    return relative.with_suffix('.py')



def _open_root_directory(path: Path) -> int:
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory(root_fd: int, relative: Path, identities: dict) -> int:
    descriptor = os.dup(root_fd)
    try:
        traversed = Path('.')
        for part in relative.parts:
            traversed /= part
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
            if _identity(os.fstat(descriptor)) != identities[traversed]:
                raise ArenaRuntimeCacheRetirementError('runtime directory identity changed')
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def retire_runtime_caches(
    *, repo_root: Path, runtime_root: Path, slot: int, sha: str, owner_uid: int,
    ingress_state: Path = INGRESS_STATE, caddy_file: Path = CADDY_FILE,
    proc_root: Path = Path('/proc'), mountinfo: Path = Path('/proc/self/mountinfo'),
    apply: bool = False,
) -> ArenaRuntimeCacheRetirementResult:
    """Validate everything first; unlink only disposable cache entries as root.

    Source removal remains the unprivileged caller's operation after its owned
    process drain. No source file permissions or ownership are changed here.
    """
    if (type(slot) is not int or slot not in (0, 1) or not isinstance(sha, str)
            or not re.fullmatch(r'[0-9a-f]{40}', sha)):
        raise ArenaRuntimeCacheRetirementError('runtime slot or exact SHA is invalid')
    if type(owner_uid) is not int or owner_uid < 0:
        raise ArenaRuntimeCacheRetirementError('runtime owner is invalid')
    runtime_root = _no_link_ancestors(runtime_root)
    archive = _no_link_ancestors(runtime_root / f'slot{slot}-{sha}')
    repo_root = _no_link_ancestors(repo_root)
    saved = archive.lstat()
    if (runtime_root.stat().st_uid != owner_uid or saved.st_uid != owner_uid
            or saved.st_mode & 0o222):
        raise ArenaRuntimeCacheRetirementError('runtime is not an owned immutable archive')
    ingress = inspect_ingress(ingress_state, caddy_file)
    if ingress['mode'] != 'legacy' and (ingress['slot'], ingress['sha']) == (slot, sha):
        raise ArenaRuntimeCacheRetirementError('active Arena archive must be preserved')
    if _contains_nested_mount(archive, _read_mount_points(mountinfo)):
        raise ArenaRuntimeCacheRetirementError('runtime contains a mounted path')
    with _exact_git_directory(repo_root):
        expected = _candidate_blob_entries(repo_root, sha)
        verify_materialized_tree(repo_root=repo_root, materialized_root=archive,
                                 target_sha=sha, strict_extras=True)
    tracked = {Path(path) for path in expected}
    directories = {Path('.')}
    for path in tracked:
        directories.update(path.parents)
    identities = {Path('.'): _identity(saved)}
    caches: dict[Path, tuple] = {}
    def traversal_error(error):
        raise ArenaRuntimeCacheRetirementError('runtime traversal could not inspect every path') from error

    root_fd = _open_root_directory(archive)
    try:
        if _identity(os.fstat(root_fd)) != _identity(saved):
            raise ArenaRuntimeCacheRetirementError('runtime root identity changed')
        for directory, names, files, descriptor in os.fwalk('.', dir_fd=root_fd, follow_symlinks=False, onerror=traversal_error):
            parent = Path(directory)
            for name in (*names, *files):
                relative = parent / name
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                is_directory = stat.S_ISDIR(metadata.st_mode)
                if (metadata.st_dev != saved.st_dev
                        or not (is_directory or stat.S_ISREG(metadata.st_mode))
                        or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)):
                    raise ArenaRuntimeCacheRetirementError('runtime contains a link or unfamiliar type')
                if relative in tracked or relative in directories:
                    if (metadata.st_uid != owner_uid or (relative in directories) != is_directory
                            or metadata.st_mode & 0o222):
                        raise ArenaRuntimeCacheRetirementError('runtime source ownership or immutable mode differs')
                elif (is_directory and relative.name == '__pycache__'
                      and relative.parent in directories):
                    if metadata.st_uid not in (0, owner_uid):
                        raise ArenaRuntimeCacheRetirementError('runtime cache owner is unfamiliar')
                    caches[relative] = _identity(metadata)
                elif (not is_directory and _cache_source(relative) in tracked
                      and (relative.parent in directories or relative.parent in caches)):
                    if metadata.st_uid not in (0, owner_uid):
                        raise ArenaRuntimeCacheRetirementError('runtime cache owner is unfamiliar')
                    caches[relative] = _identity(metadata)
                else:
                    raise ArenaRuntimeCacheRetirementError('runtime has unknown paths; preserved')
                if is_directory:
                    identities[relative] = _identity(metadata)
        if apply and os.geteuid() != 0:
            raise ArenaRuntimeCacheRetirementError('cache removal requires root')
        # Prove the entire inactive archive before allowing a deferred result.
        # Close our inspection FD before reading /proc, or the helper's own
        # descriptor would falsely prove that this archive is in use.
        os.close(root_fd)
        root_fd = None
        if (inspect_ingress(ingress_state, caddy_file) != ingress
                or _contains_nested_mount(archive, _read_mount_points(mountinfo))):
            raise ArenaRuntimeCacheRetirementError('runtime serving identity changed')
        _no_link_ancestors(archive)
        if _identity(archive.lstat()) != _identity(saved):
            raise ArenaRuntimeCacheRetirementError('runtime root identity changed')
        if _is_in_use(archive, _read_process_paths(proc_root)):
            # Diagnostic readers may retain an otherwise valid inactive tree.
            # Do not mutate it or let optional retirement stop a live release.
            return {'archive': str(archive), 'cache_paths': len(caches),
                    'applied': False, 'status': 'deferred_in_use'}
        if apply:
            root_fd = _open_root_directory(archive)
            if _identity(os.fstat(root_fd)) != _identity(saved):
                raise ArenaRuntimeCacheRetirementError('runtime root identity changed')
            for relative in sorted(caches, key=lambda path: len(path.parts), reverse=True):
                descriptor = _open_directory(root_fd, relative.parent, identities)
                try:
                    metadata = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
                    if _identity(metadata) != caches[relative]:
                        raise ArenaRuntimeCacheRetirementError('runtime cache identity changed')
                    if stat.S_ISDIR(metadata.st_mode):
                        os.rmdir(relative.name, dir_fd=descriptor)
                    else:
                        os.unlink(relative.name, dir_fd=descriptor)
                finally:
                    os.close(descriptor)
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return {'archive': str(archive), 'cache_paths': len(caches), 'applied': apply,
            'status': 'cache_cleanup_complete' if apply else 'planned'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--slot', required=True, type=int, choices=(0, 1))
    parser.add_argument('--sha', required=True)
    parser.add_argument('--owner-uid', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(retire_runtime_caches(repo_root=args.repo_root, runtime_root=RUNTIME_ROOT,
        slot=args.slot, sha=args.sha, owner_uid=args.owner_uid, apply=True), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
