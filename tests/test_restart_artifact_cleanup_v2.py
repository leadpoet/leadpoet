from __future__ import annotations

import os
from pathlib import Path

import pytest

from validator_tee.host import restart_artifact_cleanup_v2 as cleanup
from validator_tee.host.restart_artifact_cleanup_v2 import (
    cleanup_restart_artifacts,
    verify_docker_lock_owner,
)


NOW = 2_000_000.0
OLD = NOW - 10_000


def _age(path: Path, timestamp: float = OLD) -> None:
    if path.is_dir() and not path.is_symlink():
        for child in path.rglob("*"):
            if not child.is_symlink():
                os.utime(child, (timestamp, timestamp))
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def _run(root: Path, **overrides):
    arguments = {
        "temporary_root": root,
        "temp_min_age_seconds": 100,
        "emergency_min_age_seconds": 100,
        "now_epoch": NOW,
        "apply": True,
        "usage_provider": lambda: [],
        "mount_points": set(),
    }
    arguments.update(overrides)
    return cleanup_restart_artifacts(**arguments)


def test_cleanup_removes_only_old_allowlisted_workspaces(tmp_path: Path) -> None:
    stale = tmp_path / "leadpoet-docker-normalize-stale"
    stale.mkdir()
    (stale / "orig.tar").write_bytes(b"old")
    _age(stale)
    recent = tmp_path / "pcr0_normalize_recent"
    recent.mkdir()
    unknown = tmp_path / "research-lab-unowned"
    unknown.mkdir()

    report = _run(tmp_path)

    assert not stale.exists()
    assert recent.exists()
    assert unknown.exists()
    assert [item["path"] for item in report["deleted"]] == [str(stale)]
    assert report["bytes_removed"] > 0


def test_cleanup_skips_open_workspace_and_symlink(tmp_path: Path) -> None:
    stale = tmp_path / "pcr0_normalize_open"
    stale.mkdir()
    open_file = stale / "orig.tar"
    open_file.write_bytes(b"open")
    _age(stale)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "leadpoet-image-normalize-link"
    link.symlink_to(outside, target_is_directory=True)
    _age(link)

    report = _run(tmp_path, usage_provider=lambda: [open_file])

    assert stale.exists()
    assert link.is_symlink()
    assert outside.exists()
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons[str(stale)] == "candidate is open or in use"
    assert reasons[str(link)] == "candidate is a symlink"


def test_cleanup_skips_nested_mount_and_obeys_oldest_first_bound(
    tmp_path: Path,
) -> None:
    oldest = tmp_path / "pcr0_normalize_oldest"
    second = tmp_path / "pcr0_normalize_second"
    mounted = tmp_path / "pcr0_normalize_mounted"
    for path, timestamp in ((oldest, OLD - 2), (second, OLD - 1), (mounted, OLD)):
        path.mkdir()
        _age(path, timestamp)

    report = _run(
        tmp_path,
        max_candidates=1,
        mount_points={mounted / "nested-mount"},
    )

    assert not oldest.exists()
    assert second.exists()
    assert mounted.exists()
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons[str(second)] == "per-run cleanup bound reached"
    assert reasons[str(mounted)] == "candidate contains a mount point"


def test_cleanup_restores_workspace_if_it_becomes_open_after_quarantine(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "leadpoet-docker-normalize-race"
    stale.mkdir()
    (stale / "orig.tar").write_bytes(b"race")
    _age(stale)
    calls = 0

    def usage_provider():
        nonlocal calls
        calls += 1
        quarantined = list(tmp_path.glob(stale.name + ".cleanup-*"))
        if calls >= 3 and quarantined:
            return [quarantined[0] / "orig.tar"]
        return []

    report = _run(tmp_path, usage_provider=usage_provider)

    assert stale.exists()
    assert not list(tmp_path.glob(stale.name + ".cleanup-*"))
    assert any(
        item["reason"] == "candidate became open during quarantine"
        for item in report["skipped"]
    )


def test_cleanup_restores_workspace_if_mount_appears_after_quarantine(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "leadpoet-docker-normalize-mount-race"
    stale.mkdir()
    (stale / "orig.tar").write_bytes(b"race")
    _age(stale)
    calls = 0

    def mount_provider():
        nonlocal calls
        calls += 1
        quarantined = list(tmp_path.glob(stale.name + ".cleanup-*"))
        if calls >= 2 and quarantined:
            return {quarantined[0] / "late-bind-mount"}
        return set()

    report = _run(tmp_path, mount_provider=mount_provider)

    assert stale.exists()
    assert not list(tmp_path.glob(stale.name + ".cleanup-*"))
    assert any(
        item["reason"] == "candidate gained a mount point during quarantine"
        for item in report["skipped"]
    )


def test_emergency_eif_cleanup_requires_verified_last_good_rollback_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emergency = tmp_path / "emergency-v2-20260718T010203Z"
    emergency.mkdir()
    (emergency / "tee-enclave-gateway_coordinator.eif").write_bytes(b"eif")
    _age(emergency)

    def reject(**_kwargs):
        raise RuntimeError("last-good missing")

    monkeypatch.setattr(cleanup, "_verify_emergency_rollback", reject)
    report = _run(
        tmp_path,
        emergency_backup_root=tmp_path,
        gateway_archive_root=tmp_path / "archive",
        gateway_last_good_manifest=tmp_path / "last-good.json",
    )

    assert emergency.exists()
    assert report["rollback_verification"] == {"verified": False, "required": True}
    assert any(
        "rollback verification failed" in item["reason"] for item in report["skipped"]
    )


def test_emergency_eif_cleanup_runs_after_complete_rollback_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emergency = tmp_path / "emergency-v2-20260718T010203Z"
    emergency.mkdir()
    (emergency / "tee-enclave-gateway_coordinator.eif").write_bytes(b"eif")
    _age(emergency)
    monkeypatch.setattr(
        cleanup,
        "_verify_emergency_rollback",
        lambda **_kwargs: {
            "verified": True,
            "last_good_commit_sha": "a" * 40,
            "current_release_hash": "sha256:" + "b" * 64,
            "verified_release_count": 3,
        },
    )

    report = _run(
        tmp_path,
        emergency_backup_root=tmp_path,
        gateway_archive_root=tmp_path / "archive",
        gateway_last_good_manifest=tmp_path / "last-good.json",
    )

    assert not emergency.exists()
    assert report["rollback_verification"]["verified"] is True


def test_emergency_rollback_verification_binds_last_good_role_pcr0s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.tee import release_archive_v2

    role_pcr0s = {
        "gateway_autoresearch": "1" * 96,
        "gateway_coordinator": "2" * 96,
        "gateway_scoring": "3" * 96,
    }
    calls = []
    monkeypatch.setattr(
        release_archive_v2,
        "load_last_good_release",
        lambda _path: {"commit_sha": "a" * 40, "role_pcr0s": role_pcr0s},
    )

    def verify_index(**kwargs):
        calls.append(kwargs)
        return {
            "current_release_hash": "sha256:" + "b" * 64,
            "releases": [{}, {}, {}],
        }

    monkeypatch.setattr(release_archive_v2, "verify_archive_index", verify_index)
    report = cleanup._verify_emergency_rollback(
        archive_root=tmp_path / "archive",
        last_good_manifest=tmp_path / "last-good.json",
    )

    assert calls == [
        {
            "archive_root": tmp_path / "archive",
            "required_commit_sha": "a" * 40,
            "required_role_pcr0s": role_pcr0s,
            "minimum_releases": 3,
            "maximum_releases": 3,
        }
    ]
    assert report["verified"] is True


def test_cleanup_dry_run_and_safety_scan_failure_never_mutate(tmp_path: Path) -> None:
    stale = tmp_path / "leadpoet-docker-normalize-dry-run"
    stale.mkdir()
    _age(stale)
    dry_run = _run(tmp_path, apply=False)
    assert stale.exists()
    assert dry_run["would_delete"][0]["path"] == str(stale)

    def unavailable():
        raise cleanup.RestartArtifactCleanupV2Error("process table is unavailable")

    failed = _run(tmp_path, usage_provider=unavailable)
    assert stale.exists()
    assert failed["safety_error"] == "process table is unavailable"


def test_apply_lock_identity_requires_live_owner_fd7(tmp_path: Path) -> None:
    lock_file = tmp_path / "docker-operation-v2.lock"
    lock_file.touch()
    proc_root = tmp_path / "proc"
    descriptor_root = proc_root / "123" / "fd"
    descriptor_root.mkdir(parents=True)
    (descriptor_root / "7").symlink_to(lock_file)

    verify_docker_lock_owner(
        lock_file=lock_file,
        owner_pid=123,
        proc_root=proc_root,
    )
    with pytest.raises(
        cleanup.RestartArtifactCleanupV2Error,
        match="does not own",
    ):
        verify_docker_lock_owner(
            lock_file=lock_file,
            owner_pid=124,
            proc_root=proc_root,
        )
