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


def test_cleanup_removes_only_old_gateway_restore_and_archive_staging(
    tmp_path: Path,
) -> None:
    eif_root = tmp_path / "tee"
    eif_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stale_restore = tmp_path / ".gateway-eif-restore.stale"
    stale_restore.mkdir()
    (stale_restore / "gateway-v2-release-manifest.json").write_bytes(b"old")
    stale_archive = archive_root / ".release.stale"
    stale_archive.mkdir()
    (stale_archive / "archive.json").write_bytes(b"old")
    recent_restore = tmp_path / ".gateway-eif-restore.recent"
    recent_restore.mkdir()
    recent_archive = archive_root / ".release.recent"
    recent_archive.mkdir()
    _age(stale_restore)
    _age(stale_archive)

    report = _run(
        tmp_path,
        gateway_eif_root=eif_root,
        gateway_archive_root=archive_root,
    )

    assert not stale_restore.exists()
    assert not stale_archive.exists()
    assert recent_restore.exists()
    assert recent_archive.exists()
    deleted = {item["path"]: item["kind"] for item in report["deleted"]}
    assert deleted[str(stale_restore)] == "gateway_eif_restore"
    assert deleted[str(stale_archive)] == "gateway_release_archive_prepare"


def test_cleanup_skips_open_gateway_restore_and_archive_staging(
    tmp_path: Path,
) -> None:
    eif_root = tmp_path / "tee"
    eif_root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stale_restore = tmp_path / ".gateway-eif-restore.open"
    stale_restore.mkdir()
    restore_file = stale_restore / "gateway-v2-release-manifest.json"
    restore_file.write_bytes(b"open")
    stale_archive = archive_root / ".release.open"
    stale_archive.mkdir()
    archive_file = stale_archive / "archive.json"
    archive_file.write_bytes(b"open")
    _age(stale_restore)
    _age(stale_archive)

    report = _run(
        tmp_path,
        gateway_eif_root=eif_root,
        gateway_archive_root=archive_root,
        usage_provider=lambda: [restore_file, archive_file],
    )

    assert stale_restore.exists()
    assert stale_archive.exists()
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons[str(stale_restore)] == "candidate is open or in use"
    assert reasons[str(stale_archive)] == "candidate is open or in use"


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


def _local_build(root: Path, commit: str = "a" * 40, *, role: str = "gateway_coordinator") -> Path:
    workspace = root / (commit + "-local")
    source = workspace / role / "source"
    for relative in ("gateway/tee/Dockerfile.enclave", "gateway/tee/stage_attested_runtime.sh",
                     "validator_tee/host/gateway_pcr0_builder.py"):
        marker = source / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("disposable archived build source\n")
    _age(workspace)
    return workspace


def _local_run(root: Path, **overrides):
    return _run(root / "tmp", gateway_build_root=root / "builds",
                gateway_build_commit="f" * 40, **overrides)


def test_local_build_layout_is_disposable_but_release_authority_is_preserved(tmp_path: Path):
    build_root = tmp_path / "builds"
    stale = _local_build(build_root)
    historical = _local_build(build_root, "b" * 40, role="gateway_scoring")
    current = _local_build(build_root, "f" * 40)
    recent = _local_build(build_root, "c" * 40)
    (recent / "gateway_coordinator" / "source" / "recent.txt").write_text("current work")
    foreign = build_root / "other-task"
    foreign.mkdir()
    unscoped = build_root / ("d" * 40)
    unscoped.mkdir()
    archive = tmp_path / "tee" / "releases-v2" / ("1" * 64)
    archive.mkdir(parents=True)
    immutable = archive / "gateway.eif"
    immutable.write_bytes(b"verified immutable release")
    runtime = tmp_path / "arena-runtime" / ("slot1-" + "a" * 40)
    runtime.mkdir(parents=True)
    runtime_file = runtime / "service.py"
    runtime_file.write_bytes(b"serving release")

    report = _local_run(tmp_path, gateway_eif_root=tmp_path / "tee",
                        gateway_archive_root=archive.parent)

    assert {item["path"] for item in report["deleted"]} == {str(stale), str(historical)}
    assert {item["kind"] for item in report["deleted"]} == {"gateway_local_build"}
    assert current.exists() and recent.exists() and foreign.exists() and unscoped.exists()
    assert immutable.read_bytes() == b"verified immutable release"
    assert runtime_file.read_bytes() == b"serving release"


@pytest.mark.parametrize("artifact", ["result.eif", "build-result.json", "other-role"])
def test_local_build_with_unclassified_or_retained_output_is_preserved(tmp_path: Path, artifact):
    stale = _local_build(tmp_path / "builds")
    (stale / "gateway_coordinator" / artifact).write_bytes(b"retained artifact")
    _age(stale)
    report = _local_run(tmp_path)
    assert stale.exists() and report["deleted"] == []
    assert report["skipped"][0]["reason"] == "local build workspace has unrecognized or retained artifacts"


@pytest.mark.parametrize("authority", ["gateway_eif_root", "gateway_archive_root", "gateway_last_good_manifest"])
def test_local_build_never_removes_an_actual_authority_path(tmp_path: Path, authority):
    stale = _local_build(tmp_path / "builds")
    protected = stale / "gateway_coordinator" / "source" / "authority"
    # Protect configured authority even if the referenced file is currently absent.
    report = _local_run(tmp_path, **{authority: protected})
    assert stale.exists() and report["deleted"] == []
    assert report["skipped"][0]["reason"] == "local build overlaps retained gateway authority"


def test_local_build_protects_last_good_and_fails_closed_for_invalid_record(tmp_path: Path):
    import json
    stale = _local_build(tmp_path / "builds")
    marker = tmp_path / "last-good.json"
    marker.write_text(json.dumps({"status": "succeeded", "target_sha": "a" * 40,
                                  "role_pcr0s": {"gateway_coordinator": "1" * 96}}))
    report = _local_run(tmp_path, gateway_last_good_manifest=marker)
    assert stale.exists() and report["deleted"] == []
    assert report["skipped"][0]["reason"] == "current or last-good gateway build is protected"
    marker.write_text("{}")
    report = _local_run(tmp_path, gateway_last_good_manifest=marker)
    assert stale.exists() and report["deleted"] == []
    assert report["skipped"][0]["reason"] == "gateway last-good build authority is invalid"


def test_local_build_requires_exact_current_commit(tmp_path: Path):
    stale = _local_build(tmp_path / "builds")
    report = _run(tmp_path / "tmp", gateway_build_root=tmp_path / "builds")
    assert stale.exists() and report["deleted"] == []
    assert report["skipped"][0]["reason"] == "current gateway build commit is required"
    with pytest.raises(cleanup.RestartArtifactCleanupV2Error, match="commit is invalid"):
        _run(tmp_path / "tmp", gateway_build_root=tmp_path / "builds", gateway_build_commit="main")


def test_local_build_open_path_mount_symlink_and_foreign_owner_are_preserved(tmp_path: Path, monkeypatch):
    open_build = _local_build(tmp_path / "builds", "a" * 40)
    mounted = _local_build(tmp_path / "builds", "b" * 40)
    foreign = _local_build(tmp_path / "builds", "c" * 40)
    link = tmp_path / "builds" / ("d" * 40 + "-local")
    link.symlink_to(open_build, target_is_directory=True)
    _age(link)
    original = Path.lstat

    def foreign_lstat(path):
        observed = original(path)
        if path == foreign:
            values = list(observed)
            values[4] = max(os.getuid() + 1, 1)
            return os.stat_result(values)
        return observed

    monkeypatch.setattr(Path, "lstat", foreign_lstat)
    report = _local_run(tmp_path, usage_provider=lambda: [open_build / "gateway_coordinator" / "source"],
                        mount_points={mounted / "nested-mount"})
    assert all(path.exists() for path in (open_build, mounted, foreign, link))
    assert report["deleted"] == []
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons[str(open_build)] == "candidate is open or in use"
    assert reasons[str(mounted)] == "candidate contains a mount point"
    assert reasons[str(foreign)] == "candidate owner is not allowed"
    assert reasons[str(link)] == "candidate is a symlink"


def test_local_build_inode_swap_during_quarantine_is_preserved(tmp_path: Path, monkeypatch):
    stale = _local_build(tmp_path / "builds")
    original = os.rename
    saved = stale.with_name("preserved-original")

    def replace_before_rename(source, destination):
        if Path(source) == stale and ".cleanup-" in str(destination):
            original(source, saved)
            replacement = _local_build(stale.parent)
            original(replacement, destination)
        else:
            original(source, destination)

    monkeypatch.setattr(os, "rename", replace_before_rename)
    report = _local_run(tmp_path)
    assert saved.exists() and stale.exists()
    assert report["deleted"] == []
    assert report["skipped"][0]["reason"] == "candidate identity changed during quarantine"


def test_local_build_dry_run_and_oldest_first_bound(tmp_path: Path):
    stale = _local_build(tmp_path / "builds", "a" * 40)
    second = _local_build(tmp_path / "builds", "b" * 40)
    _age(stale, OLD - 1)
    dry = _local_run(tmp_path, apply=False, max_candidates=1)
    assert [item["path"] for item in dry["would_delete"]] == [str(stale)]
    assert stale.exists() and second.exists()
    report = _local_run(tmp_path, max_candidates=1)
    assert not stale.exists() and second.exists()
    assert report["deleted"][0]["path"] == str(stale)


def test_local_build_preserves_authority_symlink_and_source_symlink(tmp_path: Path):
    protected = _local_build(tmp_path / "builds", "a" * 40)
    authority = tmp_path / "archive-link"
    authority.symlink_to(protected / "gateway_coordinator" / "source", target_is_directory=True)
    linked = _local_build(tmp_path / "builds", "b" * 40)
    source = linked / "gateway_coordinator" / "source"
    outside = tmp_path / "outside-source"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)
    _age(linked)
    report = _local_run(tmp_path, gateway_archive_root=authority)
    assert protected.exists() and linked.exists() and outside.exists()
    assert report["deleted"] == []
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons[str(protected)] == "local build overlaps retained gateway authority"
    assert reasons[str(linked)] == "local build workspace has unrecognized or retained artifacts"
