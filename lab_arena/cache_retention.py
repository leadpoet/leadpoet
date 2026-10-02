"""Bounded cleanup of reproducible caches in retired Arena workspaces.

The runner holds a shared workspace lock while its caches can be used.  The
maintenance command takes the exclusive lock before inspecting or removing
entries.  It never removes a workspace or any run, sandbox, wallet, or journal.
"""

from __future__ import annotations

import fcntl
from contextlib import nullcontext
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from lab_arena.runtime_host import _open_directory_tree

LOCK_NAME = ".arena-cache-retention.lock"
MIN_AGE_DAYS = 7
_CACHE_NAMES = {
    "images": (re.compile(r"sha256-[0-9a-f]{64}\Z"), ".exported"),
    "sources": (re.compile(r"submission-[A-Za-z0-9._:-]{1,64}\Z"), ".ready"),
}
_EXTRA_IMAGE_ROOT = re.compile(r"judge-images-[a-z0-9][a-z0-9-]{0,63}\Z")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class CacheRetentionError(RuntimeError):
    pass


def workspace_cache_lock(
    workspace: Path, *, exclusive: bool, blocking: bool = False, create: bool = True,
) -> Any:
    """Lock a real workspace without following a symlink in any parent."""
    workspace = Path(workspace)
    if not workspace.is_absolute() or ".." in workspace.parts or workspace == Path("/"):
        raise CacheRetentionError("workspace path must be a dedicated absolute directory")
    directory = _open_directory_tree(workspace, create=False)
    try:
        descriptor = os.open(
            LOCK_NAME, os.O_RDWR | (os.O_CREAT if create else 0) | os.O_NOFOLLOW, 0o600,
            dir_fd=directory,
        )
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise CacheRetentionError("workspace lock is unsafe")
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(descriptor, mode | (0 if blocking else fcntl.LOCK_NB))
            return os.fdopen(descriptor, "r+b", closefd=True)
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(directory)


def _read_proc_file(path: Path) -> bytes | None:
    try:
        with path.open("rb") as stream:
            return stream.read(262145)
    except FileNotFoundError:
        return None  # Process exited during inspection.


def _mentions_workspace(value: bytes, workspace: bytes) -> bool:
    """Match an exact path or a path below it in process metadata."""
    start = 0
    while True:
        index = value.find(workspace, start)
        if index < 0:
            return False
        end = index + len(workspace)
        before = index == 0 or value[index - 1] in b"\0 =:'\"\t\n"
        after = end == len(value) or value[end] in b"/\0 =:'\"\t\n"
        if before and after:
            return True
        start = end


def workspace_has_runtime_process(workspace: Path, *, proc_root: Path = Path("/proc")) -> bool:
    """Fail closed if an Arena process may use this workspace.

    Reads process metadata only; it never logs command lines or environment.
    """
    if not proc_root.is_dir():
        raise CacheRetentionError("Linux process inspection is unavailable")
    target = os.fsencode(str(workspace))
    try:
        processes = tuple(path for path in proc_root.iterdir() if path.name.isdecimal())
    except OSError as exc:
        raise CacheRetentionError("process inspection failed") from exc
    for process in processes:
        if process.name == str(os.getpid()):
            continue
        possibly_relevant = True  # An unreadable process cannot be cleared as inactive.
        try:
            comm = _read_proc_file(process / "comm")
            if comm is None:
                continue
            possibly_relevant = bool(comm.strip())
            command = _read_proc_file(process / "cmdline")
            if command is None:
                continue
            if len(command) > 262144:
                raise CacheRetentionError("process metadata is too large")
            if not command:
                # Kernel threads have no command line or user workspace.
                continue
            try:
                cwd = os.readlink(process / "cwd")
            except FileNotFoundError:
                continue
            environment = _read_proc_file(process / "environ")
            if environment is None:
                continue
            if len(environment) > 262144:
                raise CacheRetentionError("process metadata is too large")
            if any(_mentions_workspace(value, target) for value in (command, os.fsencode(cwd), environment)):
                return True
        except PermissionError as exc:
            if possibly_relevant:
                raise CacheRetentionError("process metadata is unreadable") from exc
        except OSError as exc:
            if exc.errno not in (3,):  # ESRCH: process exited during inspection.
                raise CacheRetentionError("process inspection failed") from exc
    return False


def _path_has_mount(path: Path, mountinfo_path: Path) -> bool:
    """Refuse a path that contains any live mount, including bind mounts."""
    try:
        rows = mountinfo_path.read_bytes().splitlines()
    except OSError as exc:
        raise CacheRetentionError("mount inspection failed") from exc
    target = os.fsencode(str(path))
    for row in rows:
        fields = row.split(b" ")
        if len(fields) < 5:
            raise CacheRetentionError("mount metadata is invalid")
        mountpoint = fields[4]
        for encoded, decoded in ((b"\\040", b" "), (b"\\011", b"\t"), (b"\\012", b"\n"), (b"\\134", b"\\")):
            mountpoint = mountpoint.replace(encoded, decoded)
        if mountpoint == target or mountpoint.startswith(target + b"/") or _mentions_workspace(row, target):
            return True
    return False


def _allocated_bytes(path: Path) -> int:
    total = 0
    for directory, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            try:
                details = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            total += details.st_blocks * 512
    return total


def _cache_entries(workspace: Path, cache_name: str, cutoff: float) -> list[dict[str, Any]]:
    cache_root = workspace / cache_name
    if cache_root.is_symlink() or not cache_root.is_dir():
        return []
    image_cache = cache_name == "images" or bool(_EXTRA_IMAGE_ROOT.fullmatch(cache_name))
    pattern, marker_name = _CACHE_NAMES["images" if image_cache else "sources"]
    results = []
    for entry in cache_root.iterdir():
        if not pattern.fullmatch(entry.name) or entry.is_symlink() or not entry.is_dir():
            continue
        marker = entry / marker_name
        if marker.is_symlink() or not marker.is_file():
            continue
        if image_cache:
            try:
                if marker.read_text(encoding="utf-8").strip() != "sha256:" + entry.name.removeprefix("sha256-"):
                    continue
            except (OSError, UnicodeError):
                continue
            child = entry / "rootfs"
            if child.is_symlink() or not child.is_dir():
                continue
        else:
            if any((entry / name).is_symlink() or not (entry / name).exists() for name in ("source.tar.gz", "source", "deps")):
                continue
        entry_age = max(entry.lstat().st_mtime, marker.lstat().st_mtime)
        if entry_age >= cutoff:
            continue
        details = entry.lstat()
        results.append({
            "cache": cache_name, "entry": entry.name, "bytes": _allocated_bytes(entry),
            "_dev": details.st_dev, "_ino": details.st_ino,
        })
    return results


def prune_workspace_root(
    root: Path,
    *,
    active_workspaces: tuple[Path, ...],
    older_than_days: int = MIN_AGE_DAYS,
    apply: bool = False,
    now: float | None = None,
    proc_root: Path = Path("/proc"),
    mountinfo_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Inspect only direct children of ``root``; apply cache-only removals."""
    root = Path(root)
    if (
        not root.is_absolute()
        or str(root) in ("/", "/var", "/var/lib", "/home", "/root", "/tmp", "/var/tmp", "/usr", "/etc", "/opt")
        or ".." in root.parts
    ):
        raise CacheRetentionError("workspace root must be a dedicated absolute directory")
    if older_than_days < MIN_AGE_DAYS:
        raise CacheRetentionError("cache retention must be at least seven days")
    active = {str(path) for path in active_workspaces}
    if not active or not all(Path(path).is_absolute() and Path(path).parent == root for path in active):
        raise CacheRetentionError("active workspaces must be direct children of the workspace root")
    root_descriptor = _open_directory_tree(root, create=False)
    os.close(root_descriptor)
    cutoff = (time.time() if now is None else now) - older_than_days * 86400
    mountinfo_path = mountinfo_path or proc_root / "self" / "mountinfo"
    results: list[dict[str, Any]] = []
    for workspace in sorted(root.iterdir(), key=lambda path: path.name):
        if str(workspace) in active or workspace.is_symlink() or not workspace.is_dir():
            continue
        if workspace.name.endswith("wallets") or workspace.name.startswith("."):
            continue
        cache_names = [
            path.name for path in workspace.iterdir()
            if (path.name in _CACHE_NAMES or _EXTRA_IMAGE_ROOT.fullmatch(path.name))
            and path.is_dir() and not path.is_symlink()
        ]
        if not cache_names:
            continue
        workspace_details = workspace.lstat()
        activity_paths = [workspace, workspace / "runs", workspace / "sandboxes"]
        if any(path.exists() and path.lstat().st_mtime >= cutoff for path in activity_paths):
            results.append({"workspace": workspace.name, "status": "recent_workspace"})
            continue
        try:
            guard = workspace_cache_lock(workspace, exclusive=True, create=apply)
        except FileNotFoundError:
            if apply:
                raise
            guard = nullcontext()
        except BlockingIOError:
            results.append({"workspace": workspace.name, "status": "locked"})
            continue
        with guard:
            if workspace_has_runtime_process(workspace, proc_root=proc_root):
                results.append({"workspace": workspace.name, "status": "active_process"})
                continue
            if _path_has_mount(workspace, mountinfo_path):
                results.append({"workspace": workspace.name, "status": "mounted_workspace"})
                continue
            if any(path.exists() and path.lstat().st_mtime >= cutoff for path in activity_paths if path != workspace):
                results.append({"workspace": workspace.name, "status": "recent_workspace"})
                continue
            for cache_name in cache_names:
                for item in _cache_entries(workspace, cache_name, cutoff):
                    entry = workspace / cache_name / item["entry"]
                    public_item = {key: value for key, value in item.items() if not key.startswith("_")}
                    if _path_has_mount(entry, mountinfo_path):
                        results.append({"workspace": workspace.name, **public_item, "status": "mounted"})
                        continue
                    if apply:
                        workspace_descriptor = os.open(workspace, _DIR_FLAGS)
                        try:
                            current_workspace = os.fstat(workspace_descriptor)
                            if (current_workspace.st_dev, current_workspace.st_ino) != (workspace_details.st_dev, workspace_details.st_ino):
                                raise CacheRetentionError("workspace changed during pruning")
                            cache_descriptor = os.open(cache_name, _DIR_FLAGS, dir_fd=workspace_descriptor)
                            try:
                                details = os.stat(item["entry"], dir_fd=cache_descriptor, follow_symlinks=False)
                                if (
                                    not stat.S_ISDIR(details.st_mode)
                                    or (details.st_dev, details.st_ino) != (item["_dev"], item["_ino"])
                                ):
                                    raise CacheRetentionError("cache entry changed during pruning")
                                # Refuse implementations without symlink-safe traversal.
                                if not shutil.rmtree.avoids_symlink_attacks:
                                    raise CacheRetentionError("safe cache removal is unavailable")
                                if _path_has_mount(workspace, mountinfo_path):
                                    results.append({"workspace": workspace.name, **public_item, "status": "mounted_workspace"})
                                    continue
                                if _path_has_mount(entry, mountinfo_path):
                                    results.append({"workspace": workspace.name, **public_item, "status": "mounted"})
                                    continue
                                shutil.rmtree(item["entry"], dir_fd=cache_descriptor)
                            finally:
                                os.close(cache_descriptor)
                        finally:
                            os.close(workspace_descriptor)
                    results.append({"workspace": workspace.name, **public_item, "status": "removed" if apply else "eligible"})
    return results
