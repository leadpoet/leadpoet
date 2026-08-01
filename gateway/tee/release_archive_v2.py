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
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.verify_release_artifacts_v2 import verify_release_artifacts
from leadpoet_canonical.attested_v2 import sha256_json


ARCHIVE_SCHEMA_VERSION = "leadpoet.gateway_release_archive.v2"
ARCHIVE_INDEX_SCHEMA_VERSION = "leadpoet.gateway_release_archive_index.v2"
DEFAULT_RETAIN_RELEASES = 3
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")


class ReleaseArchiveV2Error(RuntimeError):
    """A gateway release archive is incomplete, mutable, or inconsistent."""


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
    if not isinstance(value, Mapping) or set(value) != set(ROLE_SPECS):
        raise ReleaseArchiveV2Error("%s is incomplete" % field)
    normalized: Dict[str, str] = {}
    for role in sorted(ROLE_SPECS):
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
            for role in ROLE_SPECS
        },
        "gateway release role PCR0s",
    )


def _archived_role_pcr0s(root: Path, item: Mapping[str, Any]) -> Dict[str, str]:
    release_hash = str(item.get("release_hash") or "").lower()
    if not _HASH_RE.fullmatch(release_hash):
        raise ReleaseArchiveV2Error("gateway release archive identity is invalid")
    release = validate_release_manifest(
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


def verify_archive_index(
    *,
    archive_root: Path,
    required_commit_sha: Optional[str] = None,
    required_role_pcr0s: Optional[Mapping[str, Any]] = None,
    minimum_releases: int = 1,
    maximum_releases: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify the bounded archive index and every retained release under its lock."""

    root = Path(archive_root)
    if not root.is_dir() or root.is_symlink():
        raise ReleaseArchiveV2Error("gateway release archive root is unavailable")
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
        index = _load_regular_json(
            root / "index.json", "gateway release archive index"
        )
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
    root = Path(path)
    document = _load_json(root / "archive.json", "gateway release archive")
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
    expected_files = {
        "gateway-v2-release-manifest.json",
        "gateway-v2-local-verification.json",
    }
    for role in ROLE_SPECS:
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
    release = validate_release_manifest(
        _load_json(
            root / "gateway-v2-release-manifest.json",
            "archived gateway release manifest",
        )
    )
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
    if set(verification_roles) != set(ROLE_SPECS):
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
    if int(retain_releases) < 3:
        raise ReleaseArchiveV2Error(
            "gateway rollback archive must retain current plus two predecessors"
        )
    release = validate_release_manifest(
        _load_json(release_manifest_path, "approved gateway release manifest")
    )
    verification = verify_release_artifacts(
        release_manifest=release,
        gateway_root=Path(gateway_root),
        eif_root=Path(eif_root),
    )
    local_verification_path = Path(eif_root) / "gateway-v2-local-verification.json"
    observed_local = _load_json(
        local_verification_path,
        "local gateway release verification",
    )
    if observed_local != verification:
        raise ReleaseArchiveV2Error(
            "persisted local gateway verification differs from live verification"
        )

    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock_path = root / ".archive.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        index_path = root / "index.json"
        if index_path.exists():
            old_index = _load_json(index_path, "gateway release archive index")
            if (
                set(old_index)
                != {"schema_version", "current_release_hash", "releases"}
                or old_index.get("schema_version") != ARCHIVE_INDEX_SCHEMA_VERSION
                or not isinstance(old_index.get("releases"), list)
            ):
                raise ReleaseArchiveV2Error(
                    "gateway release archive index schema is invalid"
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

        if target.exists():
            archived = verify_archive_directory(target)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".release.", dir=str(root)))
            try:
                sources = _expected_sources(
                    release_manifest_path=Path(release_manifest_path),
                    gateway_root=Path(gateway_root),
                    eif_root=Path(eif_root),
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
            if candidate.is_dir() and re.fullmatch(r"[0-9a-f]{64}", candidate.name):
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
