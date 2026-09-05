"""Retain and verify complete gateway V2 EIF releases for explicit rollback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import (
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    historical_three_role_specs,
    validate_prior_release_manifest,
    validate_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.verify_release_artifacts_v2 import (
    ReleaseArtifactVerificationError,
    verify_release_artifacts,
)
from leadpoet_canonical.attested_v2 import sha256_json


ARCHIVE_SCHEMA_VERSION = "leadpoet.gateway_release_archive.v2"
ARCHIVE_INDEX_SCHEMA_VERSION = "leadpoet.gateway_release_archive_index.v2"
DEFAULT_RETAIN_RELEASES = 3
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")


class ReleaseArchiveV2Error(RuntimeError):
    """A gateway release archive is incomplete, mutable, or inconsistent."""


class ReleaseArchiveCacheMiss(ReleaseArchiveV2Error):
    """No verified archive exists for the exact approved release."""


def _path_exists_without_following(path: Path) -> bool:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReleaseArchiveV2Error("gateway release path cannot be inspected") from exc
    return True


def _path_without_symlink_ancestry(path: Path, field: str) -> Path:
    """Return an absolute path after rejecting every existing symlink component."""

    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ReleaseArchiveV2Error("%s cannot be inspected" % field) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseArchiveV2Error("%s has symlink ancestry" % field)
    return normalized


def _real_directory(path: Path, field: str) -> Path:
    normalized = _path_without_symlink_ancestry(path, field)
    try:
        metadata = normalized.lstat()
    except OSError as exc:
        raise ReleaseArchiveV2Error("%s is unavailable" % field) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseArchiveV2Error("%s is not a real directory" % field)
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseArchiveV2Error("%s is unavailable or invalid" % field) from exc
    if not isinstance(value, Mapping):
        raise ReleaseArchiveV2Error("%s must be an object" % field)
    return dict(value)


def _load_regular_json(path: Path, field: str) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReleaseArchiveV2Error("%s is unavailable or non-regular" % field)
    return _load_json(path, field)


def _normalize_role_pcr0s(value: Any, field: str) -> Dict[str, str]:
    historical_roles = set(
        historical_three_role_specs(
            expected_topology_hash=HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
        )
    )
    if not isinstance(value, Mapping) or set(value) not in (
        set(ROLE_SPECS),
        historical_roles,
    ):
        raise ReleaseArchiveV2Error("%s is incomplete" % field)
    normalized: Dict[str, str] = {}
    for role in sorted(value):
        pcr0 = str(value.get(role) or "").strip().lower()
        if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
            raise ReleaseArchiveV2Error("%s is invalid: %s" % (field, role))
        normalized[role] = pcr0
    return normalized


def load_last_good_release(path: Path) -> Dict[str, Any]:
    """Return the exact measured identity from a successful deployment record."""

    document = _load_regular_json(Path(path), "gateway last-good deployment")
    commit_sha = str(document.get("target_sha") or "").lower()
    if document.get("status") != "succeeded" or not _COMMIT_RE.fullmatch(commit_sha):
        raise ReleaseArchiveV2Error(
            "gateway last-good deployment is not a successful exact commit"
        )
    return {
        "commit_sha": commit_sha,
        "role_pcr0s": _normalize_role_pcr0s(
            document.get("role_pcr0s"),
            "gateway last-good deployment role PCR0s",
        ),
    }


def _release_role_pcr0s(release: Mapping[str, Any]) -> Dict[str, str]:
    roles = release.get("roles") if isinstance(release, Mapping) else None
    if not isinstance(roles, Mapping):
        raise ReleaseArchiveV2Error("gateway release roles are unavailable")
    return _normalize_role_pcr0s(
        {
            role: roles.get(role, {}).get("pcr0")
            if isinstance(roles.get(role), Mapping)
            else None
            for role in roles
        },
        "gateway release role PCR0s",
    )


def _archived_role_pcr0s(root: Path, item: Mapping[str, Any]) -> Dict[str, str]:
    release_hash = str(item.get("release_hash") or "").lower()
    if not _HASH_RE.fullmatch(release_hash):
        raise ReleaseArchiveV2Error("gateway release archive identity is invalid")
    release = validate_prior_release_manifest(
        _load_regular_json(
            root
            / release_hash.split(":", 1)[1]
            / "gateway-v2-release-manifest.json",
            "archived gateway release manifest",
        )
    )
    return _release_role_pcr0s(release)


def _verify_index_entry(root: Path, item: Mapping[str, Any]) -> Dict[str, Any]:
    expected_fields = {"release_hash", "commit_sha", "archive_hash", "archived_at"}
    if not isinstance(item, Mapping) or set(item) != expected_fields:
        raise ReleaseArchiveV2Error("gateway release archive index entry is invalid")
    release_hash = str(item.get("release_hash") or "").lower()
    commit_sha = str(item.get("commit_sha") or "").lower()
    if not _HASH_RE.fullmatch(release_hash) or not _COMMIT_RE.fullmatch(commit_sha):
        raise ReleaseArchiveV2Error("gateway release archive index identity is invalid")
    archive_path = root / release_hash.split(":", 1)[1]
    if not archive_path.is_dir() or archive_path.is_symlink():
        raise ReleaseArchiveV2Error("gateway indexed release archive is unavailable")
    archive_manifest = archive_path / "archive.json"
    if not archive_manifest.is_file() or archive_manifest.is_symlink():
        raise ReleaseArchiveV2Error("gateway release archive is unavailable")
    archived = verify_archive_directory(archive_path)
    for field in expected_fields:
        if archived.get(field) != item.get(field):
            raise ReleaseArchiveV2Error(
                "gateway release archive differs from its index entry"
            )
    return dict(item)


def _verify_archive_index_locked(
    *,
    root: Path,
    required_commit_sha: Optional[str] = None,
    required_role_pcr0s: Optional[Mapping[str, Any]] = None,
    minimum_releases: int = 1,
    maximum_releases: Optional[int] = None,
) -> Dict[str, Any]:
    index = _load_regular_json(root / "index.json", "gateway release archive index")
    if (
        set(index) != {"schema_version", "current_release_hash", "releases"}
        or index.get("schema_version") != ARCHIVE_INDEX_SCHEMA_VERSION
        or not isinstance(index.get("releases"), list)
    ):
        raise ReleaseArchiveV2Error("gateway release archive index schema is invalid")
    releases = list(index["releases"])
    if len(releases) < int(minimum_releases):
        raise ReleaseArchiveV2Error("gateway rollback archive set is incomplete")
    if maximum_releases is not None and len(releases) > int(maximum_releases):
        raise ReleaseArchiveV2Error("gateway rollback archive set is unbounded")
    verified = [_verify_index_entry(root, item) for item in releases]
    hashes = [str(item["release_hash"]) for item in verified]
    if len(set(hashes)) != len(hashes):
        raise ReleaseArchiveV2Error("gateway release archive index contains duplicates")
    if not hashes or index.get("current_release_hash") != hashes[0]:
        raise ReleaseArchiveV2Error(
            "gateway release archive current identity is inconsistent"
        )
    required = str(required_commit_sha or "").lower()
    if required or required_role_pcr0s is not None:
        if not _COMMIT_RE.fullmatch(required):
            raise ReleaseArchiveV2Error("required gateway rollback commit is invalid")
        required_pcr0s = _normalize_role_pcr0s(
            required_role_pcr0s,
            "required gateway rollback role PCR0s",
        )
        matching = [item for item in verified if item["commit_sha"] == required]
        if len(matching) != 1:
            raise ReleaseArchiveV2Error(
                "gateway last-good commit is not uniquely retained"
            )
        if _archived_role_pcr0s(root, matching[0]) != required_pcr0s:
            raise ReleaseArchiveV2Error(
                "gateway last-good role PCR0s are not retained"
            )
    return {
        "schema_version": index["schema_version"],
        "current_release_hash": index["current_release_hash"],
        "releases": verified,
    }


def verify_archive_index(
    *,
    archive_root: Path,
    required_commit_sha: Optional[str] = None,
    required_role_pcr0s: Optional[Mapping[str, Any]] = None,
    minimum_releases: int = 1,
    maximum_releases: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify the bounded archive index and every retained release under its lock."""

    root = _real_directory(Path(archive_root), "gateway release archive root")
    lock_path = root / ".archive.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ReleaseArchiveV2Error("gateway release archive lock is unavailable")
    try:
        lock_handle = lock_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReleaseArchiveV2Error(
            "gateway release archive lock is unavailable"
        ) from exc
    with lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _verify_archive_index_locked(
            root=root,
            required_commit_sha=required_commit_sha,
            required_role_pcr0s=required_role_pcr0s,
            minimum_releases=minimum_releases,
            maximum_releases=maximum_releases,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _measurement_pcr0(path: Path) -> str:
    try:
        output = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseArchiveV2Error("archived gateway measurement is unavailable") from exc
    candidates = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for offset, character in enumerate(output):
        if character != "{":
            continue
        try:
            candidates.append(json.loads(output[offset:]))
            break
        except json.JSONDecodeError:
            continue
    for value in candidates:
        measurements = value.get("Measurements") if isinstance(value, Mapping) else None
        pcr0 = (
            str(measurements.get("PCR0") or "").lower()
            if isinstance(measurements, Mapping)
            else ""
        )
        if len(pcr0) == 96 and all(character in "0123456789abcdef" for character in pcr0):
            return pcr0
    raise ReleaseArchiveV2Error("archived gateway measurement lacks PCR0")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_sources(
    *,
    release_manifest_path: Path,
    gateway_root: Path,
    eif_root: Path,
) -> Dict[str, Path]:
    sources = {
        "gateway-v2-release-manifest.json": release_manifest_path,
        "gateway-v2-local-verification.json": eif_root
        / "gateway-v2-local-verification.json",
    }
    for role in sorted(ROLE_SPECS):
        sources.update(
            {
                "tee-enclave-%s.eif" % role: eif_root
                / ("tee-enclave-%s.eif" % role),
                "enclave-build-%s.json" % role: eif_root
                / ("enclave-build-%s.json" % role),
                "enclave-image-%s.txt" % role: eif_root
                / ("enclave-image-%s.txt" % role),
                "build-identities/%s.json" % role: gateway_root
                / "_attested_runtime"
                / "gateway_enclave_build_identities"
                / (role + ".json"),
            }
        )
    return sources


def _copy_regular(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ReleaseArchiveV2Error(
            "gateway release artifact is missing or non-regular: %s" % source
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def verify_archive_directory(path: Path) -> Dict[str, Any]:
    root = _real_directory(Path(path), "gateway release archive")
    document = _load_regular_json(
        root / "archive.json", "gateway release archive"
    )
    expected_fields = {
        "schema_version",
        "release_hash",
        "commit_sha",
        "archived_at",
        "files",
        "archive_hash",
    }
    if set(document) != expected_fields or document.get(
        "schema_version"
    ) != ARCHIVE_SCHEMA_VERSION:
        raise ReleaseArchiveV2Error("gateway release archive schema is invalid")
    files = document.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ReleaseArchiveV2Error("gateway release archive inventory is empty")
    release = validate_prior_release_manifest(
        _load_regular_json(
            root / "gateway-v2-release-manifest.json",
            "archived gateway release manifest",
        )
    )
    expected_files = {
        "gateway-v2-release-manifest.json",
        "gateway-v2-local-verification.json",
    }
    for role in release["roles"]:
        expected_files.update(
            {
                "tee-enclave-%s.eif" % role,
                "enclave-build-%s.json" % role,
                "enclave-image-%s.txt" % role,
                "build-identities/%s.json" % role,
            }
        )
    if set(files) != expected_files:
        raise ReleaseArchiveV2Error("gateway release archive inventory is incomplete")
    for relative, metadata in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ReleaseArchiveV2Error("gateway release archive path is unsafe")
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "sha256",
            "size_bytes",
        }:
            raise ReleaseArchiveV2Error("gateway release inventory entry is invalid")
        candidate = root / relative_path
        if not candidate.is_file() or candidate.is_symlink():
            raise ReleaseArchiveV2Error("gateway release archive file is missing")
        if candidate.stat().st_size != metadata["size_bytes"]:
            raise ReleaseArchiveV2Error("gateway release archive size mismatch")
        if _sha256_file(candidate) != metadata["sha256"]:
            raise ReleaseArchiveV2Error("gateway release archive hash mismatch")
    body = {key: document[key] for key in expected_fields if key != "archive_hash"}
    if document.get("archive_hash") != sha256_json(body):
        raise ReleaseArchiveV2Error("gateway release archive manifest was altered")
    if (
        release["release_hash"] != document["release_hash"]
        or release["commit_sha"] != document["commit_sha"]
    ):
        raise ReleaseArchiveV2Error("gateway archived release identity mismatch")
    for role, expectation in release["roles"].items():
        image_id = (root / ("enclave-image-%s.txt" % role)).read_text(
            encoding="utf-8"
        ).strip()
        if image_id != expectation["normalized_image_hash"]:
            raise ReleaseArchiveV2Error("archived gateway image differs from release")
        observed_pcr0 = _measurement_pcr0(
            root / ("enclave-build-%s.json" % role)
        )
        if observed_pcr0 != expectation["pcr0"]:
            raise ReleaseArchiveV2Error("archived gateway PCR0 differs from release")
        identity = _load_json(
            root / "build-identities" / (role + ".json"),
            "archived gateway build identity",
        )
        if identity.get("identity_hash") != expectation["build_identity_hash"]:
            raise ReleaseArchiveV2Error(
                "archived gateway build identity differs from release"
            )
    verification = _load_json(
        root / "gateway-v2-local-verification.json",
        "archived local gateway verification",
    )
    if verification.get("release_hash") != release["release_hash"]:
        raise ReleaseArchiveV2Error("archived local verification is for another release")
    verification_roles = {
        item.get("physical_role"): item
        for item in verification.get("roles", [])
        if isinstance(item, Mapping)
    }
    if set(verification_roles) != set(release["roles"]):
        raise ReleaseArchiveV2Error("archived local verification roles are incomplete")
    for role, item in verification_roles.items():
        if item.get("eif_hash") != files["tee-enclave-%s.eif" % role]["sha256"]:
            raise ReleaseArchiveV2Error(
                "archived gateway EIF differs from local verification"
            )
        if item.get("pcr0") != release["roles"][role]["pcr0"]:
            raise ReleaseArchiveV2Error(
                "archived local PCR0 differs from release"
            )
    return document


def _restored_runtime_files() -> Sequence[str]:
    files = []
    for role in sorted(ROLE_SPECS):
        files.extend(
            (
                "tee-enclave-%s.eif" % role,
                "enclave-build-%s.json" % role,
                "enclave-image-%s.txt" % role,
            )
        )
    files.append("gateway-v2-local-verification.json")
    return tuple(files)


def _install_replace(source: Path, destination: Path) -> None:
    """Test seam for one atomic artifact replacement."""

    os.replace(source, destination)


def _copy_regular_for_rollback(source: Path, destination: Path) -> None:
    """Retain a rollback copy without requiring ownership of a live EIF inode."""

    _copy_regular(source, destination)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_restored_runtime(
    *,
    staged_root: Path,
    eif_root: Path,
    gateway_root: Path,
    release: Mapping[str, Any],
) -> Dict[str, Any]:
    """Transactionally replace the live artifacts and verify the installed set."""

    files = _restored_runtime_files()
    backup_root = staged_root / ".previous"
    backup_root.mkdir(mode=0o700)
    previous = set()
    for relative in files:
        target = eif_root / relative
        if not _path_exists_without_following(target):
            continue
        metadata = target.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseArchiveV2Error(
                "existing gateway EIF artifact is non-regular: %s" % target
            )
        backup = backup_root / relative
        try:
            _copy_regular_for_rollback(target, backup)
        except OSError as exc:
            raise ReleaseArchiveV2Error(
                "existing gateway EIF artifact cannot be retained atomically: %s"
                % target
            ) from exc
        previous.add(relative)

    installed = []
    try:
        for relative in files:
            source = staged_root / relative
            target = eif_root / relative
            _install_replace(source, target)
            installed.append(relative)
        _fsync_directory(eif_root)
        final_verification = verify_release_artifacts(
            release_manifest=release,
            gateway_root=gateway_root,
            eif_root=eif_root,
        )
        installed_verification = _load_regular_json(
            eif_root / "gateway-v2-local-verification.json",
            "installed local gateway verification",
        )
        if installed_verification != final_verification:
            raise ReleaseArchiveV2Error(
                "installed local gateway verification differs from live verification"
            )
        return final_verification
    except BaseException as exc:
        rollback_errors = []
        for relative in reversed(installed):
            target = eif_root / relative
            try:
                if relative in previous:
                    os.replace(backup_root / relative, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append("%s: %s" % (relative, rollback_exc))
        try:
            _fsync_directory(eif_root)
        except OSError as rollback_exc:
            rollback_errors.append("directory fsync: %s" % rollback_exc)
        if rollback_errors:
            raise ReleaseArchiveV2Error(
                "gateway EIF restore failed and rollback was incomplete: %s"
                % "; ".join(rollback_errors)
            ) from exc
        raise


def _promote_verified_release_locked(
    *, root: Path, index: Mapping[str, Any], release_hash: str
) -> Dict[str, Any]:
    """Make an installed exact release the bounded archive index head."""

    releases = list(index.get("releases") or ())
    matching = [item for item in releases if item.get("release_hash") == release_hash]
    if len(matching) != 1:
        raise ReleaseArchiveV2Error(
            "installed gateway release is not uniquely retained"
        )
    promoted = matching + [
        item for item in releases if item.get("release_hash") != release_hash
    ]
    updated = {
        "schema_version": ARCHIVE_INDEX_SCHEMA_VERSION,
        "current_release_hash": release_hash,
        "releases": promoted,
    }
    if (
        index.get("current_release_hash") != release_hash
        or releases != promoted
    ):
        _atomic_json(root / "index.json", updated)
        _fsync_directory(root)
    return updated


def restore_verified_release(
    *,
    release_manifest_path: Path,
    gateway_root: Path,
    eif_root: Path,
    archive_root: Path,
) -> Dict[str, Any]:
    """Restore only an exact, indexed release while holding the archive lock."""

    manifest_path = _path_without_symlink_ancestry(
        Path(release_manifest_path), "approved gateway release manifest"
    )
    release = validate_release_manifest(
        _load_regular_json(manifest_path, "approved gateway release manifest")
    )
    gateway = _real_directory(Path(gateway_root), "gateway root")
    destination = _real_directory(Path(eif_root), "gateway EIF root")
    root = _path_without_symlink_ancestry(
        Path(archive_root), "gateway release archive root"
    )
    if not _path_exists_without_following(root):
        raise ReleaseArchiveCacheMiss("exact gateway EIF archive is absent")
    root = _real_directory(root, "gateway release archive root")

    release_name = release["release_hash"].split(":", 1)[1]
    target = root / release_name
    index_path = root / "index.json"
    lock_path = root / ".archive.lock"
    if not _path_exists_without_following(index_path):
        unexpected = [
            item for item in root.iterdir() if item.name != ".archive.lock"
        ]
        if unexpected:
            raise ReleaseArchiveV2Error(
                "gateway release archive index is missing from a populated cache"
            )
        if _path_exists_without_following(lock_path):
            lock_metadata = lock_path.lstat()
            if not stat.S_ISREG(lock_metadata.st_mode):
                raise ReleaseArchiveV2Error(
                    "gateway release archive lock is unavailable"
                )
        raise ReleaseArchiveCacheMiss("exact gateway EIF archive is absent")
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ReleaseArchiveV2Error("gateway release archive lock is unavailable")
    try:
        lock_handle = lock_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReleaseArchiveV2Error(
            "gateway release archive lock is unavailable"
        ) from exc

    temporary: Optional[Path] = None
    with lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        index = _verify_archive_index_locked(
            root=root,
            maximum_releases=DEFAULT_RETAIN_RELEASES,
        )
        matches = [
            item
            for item in index["releases"]
            if item["release_hash"] == release["release_hash"]
        ]
        if not matches:
            if _path_exists_without_following(target):
                raise ReleaseArchiveV2Error(
                    "exact gateway EIF archive is present but unindexed"
                )
            if any(
                item["commit_sha"] == release["commit_sha"]
                for item in index["releases"]
            ):
                raise ReleaseArchiveV2Error(
                    "gateway EIF archive release differs for the exact commit"
                )
            raise ReleaseArchiveCacheMiss("exact gateway EIF archive is absent")
        if len(matches) != 1 or matches[0]["commit_sha"] != release["commit_sha"]:
            raise ReleaseArchiveV2Error(
                "gateway EIF archive exact release identity is inconsistent"
            )
        archive = _real_directory(target, "exact gateway EIF archive")
        if archive.name != release_name:
            raise ReleaseArchiveV2Error(
                "gateway EIF archive directory identity is inconsistent"
            )
        archived = _load_regular_json(
            archive / "archive.json", "gateway release archive"
        )
        if (
            archived.get("release_hash") != release["release_hash"]
            or archived.get("commit_sha") != release["commit_sha"]
        ):
            raise ReleaseArchiveV2Error(
                "gateway EIF archive differs from the approved exact release"
            )
        files = archived.get("files")
        if not isinstance(files, Mapping):
            raise ReleaseArchiveV2Error(
                "gateway release archive inventory is invalid"
            )
        for role in sorted(ROLE_SPECS):
            archived_identity = _load_regular_json(
                archive / "build-identities" / (role + ".json"),
                "archived gateway build identity",
            )
            staged_identity_path = _path_without_symlink_ancestry(
                gateway
                / "_attested_runtime"
                / "gateway_enclave_build_identities"
                / (role + ".json"),
                "staged gateway build identity",
            )
            staged_identity = _load_regular_json(
                staged_identity_path, "staged gateway build identity"
            )
            if archived_identity != staged_identity:
                raise ReleaseArchiveV2Error(
                    "%s cached build identity differs from the staged runtime" % role
                )

        # A same-release retry can already have the complete verified EIF set
        # installed. Prove that exact state before copying multi-gigabyte
        # archive bodies merely to reinstall themselves.
        try:
            live_verification = verify_release_artifacts(
                release_manifest=release,
                gateway_root=gateway,
                eif_root=destination,
            )
            persisted_live_verification = _load_regular_json(
                destination / "gateway-v2-local-verification.json",
                "installed local gateway verification",
            )
            if persisted_live_verification != live_verification:
                raise ReleaseArchiveV2Error(
                    "installed local gateway verification differs from live verification"
                )
        except (ReleaseArtifactVerificationError, ReleaseArchiveV2Error):
            live_verification = None
        if live_verification is not None:
            _promote_verified_release_locked(
                root=root,
                index=index,
                release_hash=release["release_hash"],
            )
            return {
                "status": "already_installed",
                "release_hash": release["release_hash"],
                "commit_sha": release["commit_sha"],
                "archive_hash": archived["archive_hash"],
                "archive_path": str(archive),
                "verification": live_verification,
            }

        temporary = Path(
            tempfile.mkdtemp(
                prefix=".gateway-eif-restore.", dir=str(destination.parent)
            )
        )
        os.chmod(temporary, 0o700)
        try:
            _copy_regular(archive / "archive.json", temporary / "archive.json")
            for relative in sorted(files):
                _copy_regular(archive / relative, temporary / relative)
            copied = verify_archive_directory(temporary)
            if copied != archived:
                raise ReleaseArchiveV2Error(
                    "copied gateway EIF archive differs from its locked source"
                )
            copied_release = validate_release_manifest(
                _load_regular_json(
                    temporary / "gateway-v2-release-manifest.json",
                    "copied gateway release manifest",
                )
            )
            if copied_release != release:
                raise ReleaseArchiveV2Error(
                    "cached gateway EIF release manifest differs from the approved release"
                )
            copied_verification = verify_release_artifacts(
                release_manifest=release,
                gateway_root=gateway,
                eif_root=temporary,
            )
            persisted_verification = _load_regular_json(
                temporary / "gateway-v2-local-verification.json",
                "copied local gateway verification",
            )
            if persisted_verification != copied_verification:
                raise ReleaseArchiveV2Error(
                    "copied local gateway verification differs from live verification"
                )
            installed_verification = _install_restored_runtime(
                staged_root=temporary,
                eif_root=destination,
                gateway_root=gateway,
                release=release,
            )
            _promote_verified_release_locked(
                root=root,
                index=index,
                release_hash=release["release_hash"],
            )
            return {
                "status": "restored",
                "release_hash": release["release_hash"],
                "commit_sha": release["commit_sha"],
                "archive_hash": archived["archive_hash"],
                "archive_path": str(archive),
                "verification": installed_verification,
            }
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def archive_verified_release(
    *,
    release_manifest_path: Path,
    gateway_root: Path,
    eif_root: Path,
    archive_root: Path,
    last_good_manifest_path: Optional[Path] = None,
    retain_releases: int = DEFAULT_RETAIN_RELEASES,
    archived_at: Optional[str] = None,
) -> Dict[str, Any]:
    if int(retain_releases) != DEFAULT_RETAIN_RELEASES:
        raise ReleaseArchiveV2Error(
            "gateway rollback archive must retain exactly current plus two predecessors"
        )
    manifest_path = _path_without_symlink_ancestry(
        Path(release_manifest_path), "approved gateway release manifest"
    )
    gateway = _real_directory(Path(gateway_root), "gateway root")
    eif = _real_directory(Path(eif_root), "gateway EIF root")
    release = validate_release_manifest(
        _load_regular_json(manifest_path, "approved gateway release manifest")
    )
    verification = verify_release_artifacts(
        release_manifest=release,
        gateway_root=gateway,
        eif_root=eif,
    )
    local_verification_path = eif / "gateway-v2-local-verification.json"
    observed_local = _load_regular_json(
        local_verification_path,
        "local gateway release verification",
    )
    if observed_local != verification:
        raise ReleaseArchiveV2Error(
            "persisted local gateway verification differs from live verification"
        )

    root = _path_without_symlink_ancestry(
        Path(archive_root), "gateway release archive root"
    )
    root.mkdir(parents=True, exist_ok=True)
    root = _real_directory(root, "gateway release archive root")
    os.chmod(root, 0o700)
    lock_path = root / ".archive.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ReleaseArchiveV2Error(
            "gateway release archive lock is unavailable"
        ) from exc
    try:
        lock_metadata = os.fstat(lock_descriptor)
    except OSError as exc:
        os.close(lock_descriptor)
        raise ReleaseArchiveV2Error(
            "gateway release archive lock is unavailable"
        ) from exc
    if not stat.S_ISREG(lock_metadata.st_mode):
        os.close(lock_descriptor)
        raise ReleaseArchiveV2Error(
            "gateway release archive lock is unavailable"
        )
    with os.fdopen(lock_descriptor, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        index_path = root / "index.json"
        if _path_exists_without_following(index_path):
            # Never carry an unverified historical entry into a new trusted
            # index.  This also proves every retained rollback body before any
            # archive or index mutation occurs.
            old_index = _verify_archive_index_locked(
                root=root,
                minimum_releases=1,
                maximum_releases=DEFAULT_RETAIN_RELEASES,
            )
            old_releases = list(old_index.get("releases") or [])
        else:
            old_releases = []

        last_good: Optional[Dict[str, Any]] = None
        if last_good_manifest_path is not None:
            marker = Path(last_good_manifest_path)
            try:
                marker.lstat()
            except FileNotFoundError:
                # A first deployment has no successful predecessor yet.
                pass
            except OSError as exc:
                raise ReleaseArchiveV2Error(
                    "gateway last-good deployment cannot be inspected"
                ) from exc
            else:
                last_good = load_last_good_release(marker)

        release_name = release["release_hash"].split(":", 1)[1]
        target = root / release_name

        pinned_last_good: Optional[Dict[str, Any]] = None
        if last_good is not None:
            current_is_last_good = release["commit_sha"] == last_good["commit_sha"]
            old_matching = [
                item
                for item in old_releases
                if isinstance(item, Mapping)
                and item.get("release_hash") != release["release_hash"]
                and str(item.get("commit_sha") or "").lower()
                == last_good["commit_sha"]
            ]
            if len(old_matching) + int(current_is_last_good) > 1:
                raise ReleaseArchiveV2Error(
                    "gateway last-good commit is not uniquely archived"
                )
            if current_is_last_good:
                if _release_role_pcr0s(release) != last_good["role_pcr0s"]:
                    raise ReleaseArchiveV2Error(
                        "gateway last-good role PCR0s are not archived"
                    )
            elif old_matching:
                pinned_last_good = _verify_index_entry(root, old_matching[0])
                if (
                    _archived_role_pcr0s(root, pinned_last_good)
                    != last_good["role_pcr0s"]
                ):
                    raise ReleaseArchiveV2Error(
                        "gateway last-good role PCR0s are not archived"
                    )

        if _path_exists_without_following(target):
            archived = verify_archive_directory(target)
            if (
                archived["release_hash"] != release["release_hash"]
                or archived["commit_sha"] != release["commit_sha"]
            ):
                raise ReleaseArchiveV2Error(
                    "existing gateway archive differs from the approved release"
                )
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".release.", dir=str(root)))
            try:
                sources = _expected_sources(
                    release_manifest_path=manifest_path,
                    gateway_root=gateway,
                    eif_root=eif,
                )
                inventory = {}
                for relative, source in sorted(sources.items()):
                    destination = temporary / relative
                    _copy_regular(source, destination)
                    inventory[relative] = {
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    }
                body = {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "release_hash": release["release_hash"],
                    "commit_sha": release["commit_sha"],
                    "archived_at": str(archived_at or _utc_now()),
                    "files": inventory,
                }
                _atomic_json(temporary / "archive.json", {**body, "archive_hash": sha256_json(body)})
                os.chmod(temporary, 0o700)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            archived = verify_archive_directory(target)
        entry = {
            "release_hash": archived["release_hash"],
            "commit_sha": archived["commit_sha"],
            "archive_hash": archived["archive_hash"],
            "archived_at": archived["archived_at"],
        }
        candidates = [entry] + [
            item
            for item in old_releases
            if isinstance(item, Mapping)
            and item.get("release_hash") != entry["release_hash"]
        ]
        retained = candidates[: int(retain_releases)]

        # A successful release must remain independently roll-backable even
        # after several later builds fail before activation.  Pin its complete,
        # reverified archive inside the bounded retention set.  A missing
        # archive is tolerated for legacy installations: in that state the
        # cleanup helper preserves the older emergency EIF backup instead.
        if last_good is not None and release["commit_sha"] == last_good["commit_sha"]:
            pinned_last_good = entry
        if pinned_last_good is not None and pinned_last_good not in retained:
            retained[-1] = pinned_last_good
        index_body = {
            "schema_version": ARCHIVE_INDEX_SCHEMA_VERSION,
            "current_release_hash": entry["release_hash"],
            "releases": retained,
        }
        _atomic_json(index_path, index_body)
        retained_names = {
            str(item["release_hash"]).split(":", 1)[1] for item in retained
        }
        for candidate in root.iterdir():
            if (
                candidate.is_dir()
                and not candidate.is_symlink()
                and re.fullmatch(r"[0-9a-f]{64}", candidate.name)
            ):
                if candidate.name not in retained_names:
                    shutil.rmtree(candidate)
        return {
            **entry,
            "archive_path": str(target),
            "retained_release_count": len(retained),
        }


def select_release_manifest(
    *, archive_root: Path, release_hash: str, output: Path
) -> Dict[str, Any]:
    normalized = str(release_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ReleaseArchiveV2Error("selected gateway release hash is invalid")
    archive = Path(archive_root) / normalized.split(":", 1)[1]
    document = verify_archive_directory(archive)
    release = _load_json(
        archive / "gateway-v2-release-manifest.json",
        "selected gateway release manifest",
    )
    _atomic_json(Path(output), release)
    return document


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", action="store_true")
    mode.add_argument("--restore", action="store_true")
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--select-release-hash")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--eif-root", type=Path)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--retain", type=int, default=DEFAULT_RETAIN_RELEASES)
    parser.add_argument("--last-good-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.archive:
        if not args.release_manifest or not args.gateway_root or not args.eif_root:
            raise ReleaseArchiveV2Error(
                "archiving requires release manifest, gateway root, and EIF root"
            )
        result = archive_verified_release(
            release_manifest_path=args.release_manifest,
            gateway_root=args.gateway_root,
            eif_root=args.eif_root,
            archive_root=args.archive_root,
            last_good_manifest_path=args.last_good_manifest,
            retain_releases=args.retain,
        )
    elif args.restore:
        if not args.release_manifest or not args.gateway_root or not args.eif_root:
            raise ReleaseArchiveV2Error(
                "restoring requires release manifest, gateway root, and EIF root"
            )
        try:
            result = restore_verified_release(
                release_manifest_path=args.release_manifest,
                gateway_root=args.gateway_root,
                eif_root=args.eif_root,
                archive_root=args.archive_root,
            )
        except ReleaseArchiveCacheMiss as exc:
            print(
                json.dumps(
                    {"status": "cache_miss", "reason": str(exc)},
                    sort_keys=True,
                    indent=2,
                )
            )
            return 3
    elif args.verify:
        result = verify_archive_directory(args.verify)
    else:
        if not args.output:
            raise ReleaseArchiveV2Error("release selection requires --output")
        result = select_release_manifest(
            archive_root=args.archive_root,
            release_hash=args.select_release_hash,
            output=args.output,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
