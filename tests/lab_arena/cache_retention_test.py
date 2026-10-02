from __future__ import annotations

import os
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena.cache_retention import (
    CacheRetentionError,
    prune_workspace_root,
    workspace_cache_lock,
    workspace_has_runtime_process,
)
from lab_arena.runner import Runner


NOW = 2_000_000_000
OLD = NOW - 8 * 86400


def _entry(workspace: Path, cache: str, name: str, *, recent: bool = False) -> Path:
    entry = workspace / cache / name
    entry.mkdir(parents=True)
    if cache == "sources":
        (entry / "source").mkdir()
        (entry / "deps").mkdir()
        (entry / "source.tar.gz").write_bytes(b"reproducible archive")
        marker = entry / ".ready"
        marker.write_text("ready")
    else:
        (entry / "rootfs").mkdir()
        (entry / "rootfs" / "content").write_bytes(b"reproducible image")
        marker = entry / ".exported"
        marker.write_text("sha256:" + name.removeprefix("sha256-"))
    stamp = NOW if recent else OLD
    os.utime(marker, (stamp, stamp))
    os.utime(entry, (stamp, stamp))
    return entry


def _fake_proc(tmp_path: Path) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "self").mkdir()
    (proc / "self" / "mountinfo").write_text("")
    return proc


def test_stale_completed_cache_entries_are_reclaimed_without_evidence_or_active_work(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    active_entry = _entry(active, "images", "sha256-" + "a" * 64)
    stale = root / "pydantic-competitive-20260913"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "b" * 64)
    judge = _entry(stale, "judge-images-intent-fit-hints", "sha256-" + "c" * 64)
    source = _entry(stale, "sources", "submission-source-1")
    recent = _entry(stale, "images", "sha256-" + "d" * 64, recent=True)
    incomplete = stale / "images" / ("sha256-" + "e" * 64)
    incomplete.mkdir()
    (stale / "results").mkdir()
    (stale / "results" / "accepted.json").write_text("evidence")
    (stale / "source-v1").mkdir()
    (stale / "source-v1" / "original.py").write_text("source")
    external = tmp_path / "outside"
    external.mkdir()
    (external / "keep").write_text("outside")
    (stale / "images" / ("sha256-" + "f" * 64)).symlink_to(external, target_is_directory=True)
    (root / "runner-wallets").mkdir()
    proc = _fake_proc(tmp_path)

    dry = prune_workspace_root(root, active_workspaces=(active,), now=NOW, proc_root=proc)
    assert {row["entry"] for row in dry} == {image.name, judge.name, source.name}
    assert all(row["status"] == "eligible" for row in dry)
    assert not (stale / ".arena-cache-retention.lock").exists()
    assert image.exists() and judge.exists() and source.exists()

    applied = prune_workspace_root(root, active_workspaces=(active,), now=NOW, proc_root=proc, apply=True)
    assert {row["entry"] for row in applied} == {image.name, judge.name, source.name}
    assert not image.exists() and not judge.exists() and not source.exists()
    assert active_entry.exists() and recent.exists() and incomplete.exists()
    assert (stale / "results" / "accepted.json").read_text() == "evidence"
    assert (stale / "source-v1" / "original.py").read_text() == "source"
    assert (external / "keep").read_text() == "outside"


def test_shared_runner_locks_block_pruner_until_both_runners_close(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)
    proc = _fake_proc(tmp_path)
    first = workspace_cache_lock(stale, exclusive=False)
    second = workspace_cache_lock(stale, exclusive=False)
    try:
        assert prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc) == [
            {"workspace": "old-test", "status": "locked"}
        ]
        first.close()
        assert image.exists()
        assert prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc)[0]["status"] == "locked"
    finally:
        first.close()
        second.close()
    assert prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc)[0]["status"] == "removed"
    assert not image.exists()


def test_runner_close_holds_cache_lock_until_pool_shutdown_succeeds(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock = workspace_cache_lock(workspace, exclusive=False)
    runner = object.__new__(Runner)
    runner._config = SimpleNamespace(workspace_cache_lock=lock)

    class BrokenPool:
        fail = True

        def shutdown(self, *, wait):
            if self.fail:
                raise RuntimeError("pool failure")

    runner._pool = BrokenPool()
    with pytest.raises(RuntimeError, match="pool failure"):
        runner.close()
    assert runner._config.workspace_cache_lock is lock
    with pytest.raises(BlockingIOError):
        workspace_cache_lock(workspace, exclusive=True)
    retained = weakref.ref(runner)
    del runner
    gc.collect()
    assert retained() is not None
    runner = retained()
    runner._pool.fail = False
    runner.close()
    assert runner._config.workspace_cache_lock is None
    with workspace_cache_lock(workspace, exclusive=True):
        pass


def test_active_legacy_process_and_symlinked_workspace_are_preserved(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)
    (root / "linked-test").symlink_to(stale, target_is_directory=True)
    proc = _fake_proc(tmp_path)
    process = proc / "123456"
    process.mkdir()
    (process / "comm").write_bytes(b"python3\n")
    (process / "cmdline").write_bytes(b"python3\0scripts/native_probe_pydantic.py\0--work-dir=" + os.fsencode(stale) + b"\0")
    (process / "environ").write_bytes(b"")
    (process / "cwd").symlink_to(tmp_path, target_is_directory=True)
    assert workspace_has_runtime_process(stale, proc_root=proc)
    rows = prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc)
    assert rows == [{"workspace": "old-test", "status": "active_process"}]
    assert image.exists()
    with pytest.raises((CacheRetentionError, OSError)):
        workspace_cache_lock(root / "linked-test", exclusive=True)


def test_retention_rejects_too_short_window_and_workspace_parent_symlink(tmp_path):
    root = tmp_path / "arena"
    (root / "runner").mkdir(parents=True)
    proc = _fake_proc(tmp_path)
    with pytest.raises(CacheRetentionError):
        prune_workspace_root(root, active_workspaces=(root / "runner",), older_than_days=1, proc_root=proc)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        prune_workspace_root(alias, active_workspaces=(alias / "runner",), proc_root=proc)


def test_parent_process_using_workspace_prevents_pruning(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)
    proc = _fake_proc(tmp_path)
    parent = proc / str(os.getppid())
    parent.mkdir()
    (parent / "comm").write_bytes(b"bash\n")
    (parent / "cmdline").write_bytes(b"bash\0")
    (parent / "environ").write_bytes(b"")
    (parent / "cwd").symlink_to(image / "rootfs", target_is_directory=True)
    assert prune_workspace_root(
        root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc,
    ) == [{"workspace": "old-test", "status": "active_process"}]
    assert image.exists()


def test_apply_never_continues_after_workspace_lock_disappears(tmp_path, monkeypatch):
    from lab_arena import cache_retention

    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)

    def missing_lock(*args, **kwargs):
        raise FileNotFoundError("workspace disappeared")

    monkeypatch.setattr(cache_retention, "workspace_cache_lock", missing_lock)
    with pytest.raises(FileNotFoundError):
        prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True)
    assert image.exists()


def test_mounted_cache_entry_is_preserved_even_under_exclusive_lock(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)
    proc = _fake_proc(tmp_path)
    (proc / "self" / "mountinfo").write_text("36 25 0:32 / %s/rootfs rw - tmpfs tmpfs rw\n" % image)
    rows = prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc)
    assert rows == [{"workspace": "old-test", "status": "mounted_workspace"}]
    assert image.exists()


def test_recent_run_activity_preserves_old_cache_until_workspace_is_retired(tmp_path):
    root = tmp_path / "arena"
    active = root / "runner"
    active.mkdir(parents=True)
    stale = root / "old-test"
    stale.mkdir()
    image = _entry(stale, "images", "sha256-" + "a" * 64)
    runs = stale / "runs"
    runs.mkdir()
    os.utime(runs, (NOW, NOW))
    proc = _fake_proc(tmp_path)
    rows = prune_workspace_root(root, active_workspaces=(active,), now=NOW, apply=True, proc_root=proc)
    assert rows == [{"workspace": "old-test", "status": "recent_workspace"}]
    assert image.exists()
