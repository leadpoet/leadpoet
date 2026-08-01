"""Bounded cleanup for repository-owned restart build artifacts.

The caller must hold the shared Docker operation lock.  This helper deletes
only explicitly named, old workspaces (including interrupted gateway archive
and EIF restore staging directories) and refuses to touch open paths,
symlinks, nested mounts, or legacy EIF backups without a verified rollback
set containing the durable last-good gateway commit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Callable, Iterable, Optional, Sequence
import uuid


REPORT_SCHEMA_VERSION = "leadpoet.restart_artifact_cleanup.v2"
DEFAULT_TEMP_MIN_AGE_SECONDS = 24 * 60 * 60
DEFAULT_EMERGENCY_MIN_AGE_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_CANDIDATES = 64
_EMERGENCY_BACKUP_RE = re.compile(r"^emergency-v2-[0-9]{8}T[0-9]{6}Z$")


class RestartArtifactCleanupV2Error(RuntimeError):
    """A cleanup boundary or safety observation is invalid."""


@dataclass(frozen=True)
class _Spec:
    root: Path
    kind: str
    min_age_seconds: int
    prefix: str = ""
    pattern: Optional[re.Pattern[str]] = None
    expected_type: str = "directory"

    def matches(self, name: str) -> bool:
        return bool(self.prefix and name.startswith(self.prefix)) or bool(
            self.pattern and self.pattern.fullmatch(name)
        )


@dataclass(frozen=True)
class _Candidate:
    path: Path
    kind: str
    expected_type: str
    min_age_seconds: int
    newest_mtime: float
    size_bytes: int
    device: int
    inode: int
    mode: int


def _normalized_path(value: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _path_is_within(candidate: Path, parent: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def _decode_mount_path(value: str) -> Path:
    decoded = value
    for encoded, character in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        decoded = decoded.replace(encoded, character)
    return _normalized_path(decoded)


def _read_mount_points(mountinfo_path: Path) -> set[Path]:
    try:
        lines = Path(mountinfo_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RestartArtifactCleanupV2Error("mount table is unavailable") from exc
    mount_points: set[Path] = set()
    for line in lines:
        fields = line.split(" - ", 1)[0].split()
        if len(fields) < 5:
            raise RestartArtifactCleanupV2Error("mount table contains an invalid row")
        mount_points.add(_decode_mount_path(fields[4]))
    return mount_points


def _link_target(path: Path) -> Optional[Path]:
    try:
        raw = os.readlink(path)
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise RestartArtifactCleanupV2Error(
            "process path inspection was denied"
        ) from exc
    except OSError:
        return None
    if raw.endswith(" (deleted)"):
        raw = raw[: -len(" (deleted)")]
    if not raw.startswith("/"):
        return None
    return _normalized_path(raw)


def _read_process_paths(proc_root: Path) -> set[Path]:
    root = Path(proc_root)
    if not root.is_dir():
        raise RestartArtifactCleanupV2Error("process table is unavailable")
    observed: set[Path] = set()
    for process in root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            for link_name in ("cwd", "root"):
                target = _link_target(process / link_name)
                if target is not None:
                    observed.add(target)
            fd_root = process / "fd"
            try:
                descriptors = list(fd_root.iterdir())
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise RestartArtifactCleanupV2Error(
                    "process descriptor inspection was denied"
                ) from exc
            for descriptor in descriptors:
                target = _link_target(descriptor)
                if target is not None:
                    observed.add(target)
            try:
                map_lines = (
                    (process / "maps")
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()
                )
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise RestartArtifactCleanupV2Error(
                    "process mapping inspection was denied"
                ) from exc
            for line in map_lines:
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    raw = fields[5]
                    if raw.endswith(" (deleted)"):
                        raw = raw[: -len(" (deleted)")]
                    observed.add(_normalized_path(raw))
        except FileNotFoundError:
            # Processes may exit at any point during a /proc walk.
            continue
    return observed


def verify_docker_lock_owner(
    *, lock_file: Path, owner_pid: int, proc_root: Path = Path("/proc")
) -> None:
    """Require --apply to be authorized by the live shared-lock owner."""

    expected = Path(lock_file)
    if not expected.is_absolute() or expected.is_symlink() or not expected.is_file():
        raise RestartArtifactCleanupV2Error("Docker operation lock path is invalid")
    if owner_pid < 1:
        raise RestartArtifactCleanupV2Error("Docker operation lock owner is invalid")
    observed = _link_target(Path(proc_root) / str(owner_pid) / "fd" / "7")
    if observed != _normalized_path(expected):
        raise RestartArtifactCleanupV2Error(
            "cleanup caller does not own the shared Docker operation lock"
        )


def _is_in_use(candidate: Path, observed_paths: Iterable[Path]) -> bool:
    normalized = _normalized_path(candidate)
    return any(
        _path_is_within(_normalized_path(observed), normalized)
        for observed in observed_paths
    )


def _contains_nested_mount(candidate: Path, mount_points: Iterable[Path]) -> bool:
    normalized = _normalized_path(candidate)
    return any(
        _path_is_within(_normalized_path(mount), normalized)
        for mount in mount_points
        if _normalized_path(mount) != normalized.parent
    )


def _measure_candidate(
    path: Path,
    *,
    kind: str,
    expected_type: str,
    min_age_seconds: int,
    allowed_owner_uids: set[int],
) -> _Candidate:
    try:
        root_stat = path.lstat()
    except OSError as exc:
        raise RestartArtifactCleanupV2Error("candidate is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise RestartArtifactCleanupV2Error("candidate is a symlink")
    if expected_type == "directory" and not stat.S_ISDIR(root_stat.st_mode):
        raise RestartArtifactCleanupV2Error("candidate is not a directory")
    if expected_type == "file" and not stat.S_ISREG(root_stat.st_mode):
        raise RestartArtifactCleanupV2Error("candidate is not a regular file")
    if root_stat.st_uid not in allowed_owner_uids:
        raise RestartArtifactCleanupV2Error("candidate owner is not allowed")

    newest_mtime = root_stat.st_mtime
    size_bytes = root_stat.st_size
    if stat.S_ISDIR(root_stat.st_mode):
        walk_error: list[OSError] = []

        def on_error(error: OSError) -> None:
            walk_error.append(error)

        for directory, child_directories, files in os.walk(
            path, topdown=True, followlinks=False, onerror=on_error
        ):
            for name in [*child_directories, *files]:
                child = Path(directory) / name
                try:
                    child_stat = child.lstat()
                except OSError as exc:
                    raise RestartArtifactCleanupV2Error(
                        "candidate contents changed during inspection"
                    ) from exc
                if child_stat.st_uid not in allowed_owner_uids:
                    raise RestartArtifactCleanupV2Error(
                        "candidate contains an unowned path"
                    )
                newest_mtime = max(newest_mtime, child_stat.st_mtime)
                size_bytes += child_stat.st_size
        if walk_error:
            raise RestartArtifactCleanupV2Error(
                "candidate contents could not be inspected"
            ) from walk_error[0]
    return _Candidate(
        path=path,
        kind=kind,
        expected_type=expected_type,
        min_age_seconds=min_age_seconds,
        newest_mtime=newest_mtime,
        size_bytes=size_bytes,
        device=root_stat.st_dev,
        inode=root_stat.st_ino,
        mode=root_stat.st_mode,
    )


def _candidate_identity_matches(candidate: _Candidate, path: Path) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return False
    return (
        observed.st_dev == candidate.device
        and observed.st_ino == candidate.inode
        and stat.S_IFMT(observed.st_mode) == stat.S_IFMT(candidate.mode)
        and not stat.S_ISLNK(observed.st_mode)
    )


def _restore_quarantine(quarantine: Path, original: Path) -> None:
    try:
        if quarantine.exists() and not original.exists():
            os.rename(quarantine, original)
    except OSError:
        pass


def _delete_candidate(
    candidate: _Candidate,
    *,
    now_epoch: float,
    allowed_owner_uids: set[int],
    usage_provider: Callable[[], Iterable[Path]],
    mount_provider: Callable[[], Iterable[Path]],
) -> Optional[str]:
    try:
        refreshed = _measure_candidate(
            candidate.path,
            kind=candidate.kind,
            expected_type=candidate.expected_type,
            min_age_seconds=candidate.min_age_seconds,
            allowed_owner_uids=allowed_owner_uids,
        )
    except RestartArtifactCleanupV2Error as exc:
        return str(exc)
    if (
        refreshed.device != candidate.device
        or refreshed.inode != candidate.inode
        or refreshed.newest_mtime > now_epoch - candidate.min_age_seconds
    ):
        return "candidate changed or became recent"
    try:
        if _is_in_use(candidate.path, usage_provider()):
            return "candidate is open or in use"
    except Exception as exc:
        return "process usage recheck failed: %s" % exc
    try:
        if _contains_nested_mount(candidate.path, mount_provider()):
            return "candidate contains a mount point during deletion recheck"
    except Exception as exc:
        return "mount table deletion recheck failed: %s" % exc

    quarantine = candidate.path.with_name(
        "%s.cleanup-%s" % (candidate.path.name, uuid.uuid4().hex)
    )
    try:
        os.rename(candidate.path, quarantine)
    except OSError as exc:
        return "candidate quarantine failed: %s" % exc
    if not _candidate_identity_matches(candidate, quarantine):
        _restore_quarantine(quarantine, candidate.path)
        return "candidate identity changed during quarantine"
    try:
        if _is_in_use(quarantine, usage_provider()):
            _restore_quarantine(quarantine, candidate.path)
            return "candidate became open during quarantine"
    except Exception as exc:
        _restore_quarantine(quarantine, candidate.path)
        return "process usage quarantine recheck failed: %s" % exc
    try:
        if _contains_nested_mount(quarantine, mount_provider()):
            _restore_quarantine(quarantine, candidate.path)
            return "candidate gained a mount point during quarantine"
    except Exception as exc:
        _restore_quarantine(quarantine, candidate.path)
        return "mount table quarantine recheck failed: %s" % exc
    try:
        if candidate.expected_type == "directory":
            shutil.rmtree(quarantine)
        else:
            quarantine.unlink()
    except OSError as exc:
        _restore_quarantine(quarantine, candidate.path)
        return "candidate deletion failed: %s" % exc
    return None


def _specifications(
    *,
    temporary_root: Path,
    gateway_build_root: Optional[Path],
    validator_build_root: Optional[Path],
    gateway_eif_root: Optional[Path],
    gateway_archive_root: Optional[Path],
    emergency_backup_root: Optional[Path],
    temp_min_age_seconds: int,
    emergency_min_age_seconds: int,
) -> list[_Spec]:
    specs = [
        _Spec(
            temporary_root,
            "docker_normalizer",
            temp_min_age_seconds,
            prefix="leadpoet-docker-normalize-",
        ),
        _Spec(
            temporary_root,
            "image_normalizer",
            temp_min_age_seconds,
            prefix="leadpoet-image-normalize-",
        ),
        _Spec(
            temporary_root,
            "pcr0_normalizer",
            temp_min_age_seconds,
            prefix="pcr0_normalize_",
        ),
        _Spec(
            temporary_root,
            "docker_build_log",
            temp_min_age_seconds,
            prefix="leadpoet-docker-build.",
            expected_type="file",
        ),
    ]
    if gateway_build_root is not None:
        normalizer_root = gateway_build_root / ".docker-image-normalizer-v2"
        specs.extend(
            [
                _Spec(
                    normalizer_root,
                    "docker_normalizer",
                    temp_min_age_seconds,
                    prefix="leadpoet-docker-normalize-",
                ),
                _Spec(
                    normalizer_root,
                    "image_normalizer",
                    temp_min_age_seconds,
                    prefix="leadpoet-image-normalize-",
                ),
            ]
        )
    if validator_build_root is not None:
        specs.append(
            _Spec(
                validator_build_root,
                "pcr0_normalizer",
                temp_min_age_seconds,
                prefix="pcr0_normalize_",
            )
        )
    if gateway_eif_root is not None:
        specs.append(
            _Spec(
                gateway_eif_root.parent,
                "gateway_eif_restore",
                temp_min_age_seconds,
                prefix=".gateway-eif-restore.",
            )
        )
    if gateway_archive_root is not None:
        specs.append(
            _Spec(
                gateway_archive_root,
                "gateway_release_archive_prepare",
                temp_min_age_seconds,
                prefix=".release.",
            )
        )
    if emergency_backup_root is not None:
        specs.append(
            _Spec(
                emergency_backup_root,
                "legacy_emergency_eif",
                emergency_min_age_seconds,
                pattern=_EMERGENCY_BACKUP_RE,
            )
        )
    return specs


def _validate_spec_root(root: Path) -> Optional[Path]:
    if not root.is_absolute():
        raise RestartArtifactCleanupV2Error("cleanup root must be absolute")
    if not root.exists():
        return None
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RestartArtifactCleanupV2Error("cleanup root is not a real directory")
    return root.resolve()


def _verify_emergency_rollback(
    *, archive_root: Path, last_good_manifest: Path
) -> dict[str, object]:
    from gateway.tee.release_archive_v2 import (
        load_last_good_release,
        verify_archive_index,
    )

    last_good = load_last_good_release(last_good_manifest)
    commit_sha = str(last_good["commit_sha"])
    index = verify_archive_index(
        archive_root=archive_root,
        required_commit_sha=commit_sha,
        required_role_pcr0s=last_good["role_pcr0s"],
        minimum_releases=3,
        maximum_releases=3,
    )
    return {
        "verified": True,
        "last_good_commit_sha": commit_sha,
        "current_release_hash": index["current_release_hash"],
        "verified_release_count": len(index["releases"]),
    }


def cleanup_restart_artifacts(
    *,
    temporary_root: Path,
    gateway_build_root: Optional[Path] = None,
    validator_build_root: Optional[Path] = None,
    gateway_eif_root: Optional[Path] = None,
    emergency_backup_root: Optional[Path] = None,
    gateway_archive_root: Optional[Path] = None,
    gateway_last_good_manifest: Optional[Path] = None,
    temp_min_age_seconds: int = DEFAULT_TEMP_MIN_AGE_SECONDS,
    emergency_min_age_seconds: int = DEFAULT_EMERGENCY_MIN_AGE_SECONDS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    apply: bool = False,
    now_epoch: Optional[float] = None,
    allowed_owner_uids: Optional[set[int]] = None,
    usage_provider: Optional[Callable[[], Iterable[Path]]] = None,
    proc_root: Path = Path("/proc"),
    mount_points: Optional[set[Path]] = None,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    mount_provider: Optional[Callable[[], Iterable[Path]]] = None,
) -> dict[str, object]:
    if temp_min_age_seconds < 1 or emergency_min_age_seconds < 1:
        raise RestartArtifactCleanupV2Error("cleanup ages must be positive")
    if max_candidates < 1:
        raise RestartArtifactCleanupV2Error("max candidates must be positive")
    now = float(time.time() if now_epoch is None else now_epoch)
    owners = set(allowed_owner_uids or {os.getuid()}) | {0}
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "apply" if apply else "dry_run",
        "now_epoch": now,
        "max_candidates": max_candidates,
        "deleted": [],
        "skipped": [],
        "bytes_removed": 0,
        "rollback_verification": {"verified": False, "required": False},
    }
    skipped = report["skipped"]
    assert isinstance(skipped, list)
    candidates: list[_Candidate] = []
    seen: set[Path] = set()
    specs = _specifications(
        temporary_root=Path(temporary_root),
        gateway_build_root=Path(gateway_build_root) if gateway_build_root else None,
        validator_build_root=(
            Path(validator_build_root) if validator_build_root else None
        ),
        gateway_eif_root=Path(gateway_eif_root) if gateway_eif_root else None,
        gateway_archive_root=(
            Path(gateway_archive_root) if gateway_archive_root else None
        ),
        emergency_backup_root=(
            Path(emergency_backup_root) if emergency_backup_root else None
        ),
        temp_min_age_seconds=temp_min_age_seconds,
        emergency_min_age_seconds=emergency_min_age_seconds,
    )
    try:
        observed_mounts = (
            {_normalized_path(path) for path in mount_points}
            if mount_points is not None
            else _read_mount_points(mountinfo_path)
        )
    except Exception as exc:
        report["safety_error"] = str(exc)
        return report

    for spec in specs:
        try:
            root = _validate_spec_root(spec.root)
        except RestartArtifactCleanupV2Error as exc:
            skipped.append(
                {"path": str(spec.root), "kind": spec.kind, "reason": str(exc)}
            )
            continue
        if root is None:
            continue
        try:
            children = list(root.iterdir())
        except OSError as exc:
            skipped.append(
                {
                    "path": str(root),
                    "kind": spec.kind,
                    "reason": "cleanup root cannot be listed: %s" % exc,
                }
            )
            continue
        for child in children:
            if not spec.matches(child.name):
                continue
            path = _normalized_path(child)
            if path in seen or path.parent != root:
                continue
            seen.add(path)
            if _contains_nested_mount(path, observed_mounts):
                skipped.append(
                    {
                        "path": str(path),
                        "kind": spec.kind,
                        "reason": "candidate contains a mount point",
                    }
                )
                continue
            try:
                candidate = _measure_candidate(
                    path,
                    kind=spec.kind,
                    expected_type=spec.expected_type,
                    min_age_seconds=spec.min_age_seconds,
                    allowed_owner_uids=owners,
                )
            except RestartArtifactCleanupV2Error as exc:
                skipped.append(
                    {"path": str(path), "kind": spec.kind, "reason": str(exc)}
                )
                continue
            if candidate.newest_mtime > now - candidate.min_age_seconds:
                skipped.append(
                    {
                        "path": str(path),
                        "kind": spec.kind,
                        "reason": "candidate is recent",
                    }
                )
                continue
            candidates.append(candidate)

    candidates.sort(key=lambda item: (item.newest_mtime, str(item.path)))
    for candidate in candidates[max_candidates:]:
        skipped.append(
            {
                "path": str(candidate.path),
                "kind": candidate.kind,
                "reason": "per-run cleanup bound reached",
            }
        )
    candidates = candidates[:max_candidates]

    emergency = [item for item in candidates if item.kind == "legacy_emergency_eif"]
    if emergency:
        report["rollback_verification"] = {"verified": False, "required": True}
        if gateway_archive_root is None or gateway_last_good_manifest is None:
            reason = "verified gateway rollback inputs are required"
            skipped.extend(
                {"path": str(item.path), "kind": item.kind, "reason": reason}
                for item in emergency
            )
            candidates = [
                item for item in candidates if item.kind != "legacy_emergency_eif"
            ]
        else:
            try:
                report["rollback_verification"] = _verify_emergency_rollback(
                    archive_root=Path(gateway_archive_root),
                    last_good_manifest=Path(gateway_last_good_manifest),
                )
            except Exception as exc:
                reason = "gateway rollback verification failed: %s" % exc
                skipped.extend(
                    {"path": str(item.path), "kind": item.kind, "reason": reason}
                    for item in emergency
                )
                candidates = [
                    item for item in candidates if item.kind != "legacy_emergency_eif"
                ]

    if not candidates:
        report["eligible_count"] = 0
        return report
    provider = usage_provider or (lambda: _read_process_paths(proc_root))
    live_mount_provider = mount_provider or (
        (lambda: {_normalized_path(path) for path in mount_points})
        if mount_points is not None
        else (lambda: _read_mount_points(mountinfo_path))
    )
    try:
        observed_usage = list(provider())
    except Exception as exc:
        report["safety_error"] = str(exc)
        skipped.extend(
            {"path": str(item.path), "kind": item.kind, "reason": str(exc)}
            for item in candidates
        )
        report["eligible_count"] = 0
        return report

    ready: list[_Candidate] = []
    for candidate in candidates:
        if _is_in_use(candidate.path, observed_usage):
            skipped.append(
                {
                    "path": str(candidate.path),
                    "kind": candidate.kind,
                    "reason": "candidate is open or in use",
                }
            )
        else:
            ready.append(candidate)
    report["eligible_count"] = len(ready)
    if not apply:
        report["would_delete"] = [
            {"path": str(item.path), "kind": item.kind, "size_bytes": item.size_bytes}
            for item in ready
        ]
        return report

    deleted = report["deleted"]
    assert isinstance(deleted, list)
    for candidate in ready:
        try:
            reason = _delete_candidate(
                candidate,
                now_epoch=now,
                allowed_owner_uids=owners,
                usage_provider=provider,
                mount_provider=live_mount_provider,
            )
        except RestartArtifactCleanupV2Error as exc:
            reason = str(exc)
        if reason:
            skipped.append(
                {"path": str(candidate.path), "kind": candidate.kind, "reason": reason}
            )
            continue
        deleted.append(
            {
                "path": str(candidate.path),
                "kind": candidate.kind,
                "size_bytes": candidate.size_bytes,
            }
        )
        report["bytes_removed"] = int(report["bytes_removed"]) + candidate.size_bytes
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporary-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--gateway-build-root", type=Path)
    parser.add_argument("--validator-build-root", type=Path)
    parser.add_argument("--gateway-eif-root", type=Path)
    parser.add_argument("--emergency-backup-root", type=Path)
    parser.add_argument("--gateway-archive-root", type=Path)
    parser.add_argument("--gateway-last-good-manifest", type=Path)
    parser.add_argument("--docker-lock-file", type=Path)
    parser.add_argument("--docker-lock-owner-pid", type=int)
    parser.add_argument(
        "--temp-min-age-seconds", type=int, default=DEFAULT_TEMP_MIN_AGE_SECONDS
    )
    parser.add_argument(
        "--emergency-min-age-seconds",
        type=int,
        default=DEFAULT_EMERGENCY_MIN_AGE_SECONDS,
    )
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--allowed-owner-uid", action="append", type=int, default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply:
        if args.docker_lock_file is None or args.docker_lock_owner_pid is None:
            raise RestartArtifactCleanupV2Error(
                "--apply requires the shared Docker operation lock identity"
            )
        verify_docker_lock_owner(
            lock_file=args.docker_lock_file,
            owner_pid=args.docker_lock_owner_pid,
        )
    owners = set(args.allowed_owner_uid) | {os.getuid(), 0}
    result = cleanup_restart_artifacts(
        temporary_root=args.temporary_root,
        gateway_build_root=args.gateway_build_root,
        validator_build_root=args.validator_build_root,
        gateway_eif_root=args.gateway_eif_root,
        emergency_backup_root=args.emergency_backup_root,
        gateway_archive_root=args.gateway_archive_root,
        gateway_last_good_manifest=args.gateway_last_good_manifest,
        temp_min_age_seconds=args.temp_min_age_seconds,
        emergency_min_age_seconds=args.emergency_min_age_seconds,
        max_candidates=args.max_candidates,
        apply=args.apply,
        allowed_owner_uids=owners,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
