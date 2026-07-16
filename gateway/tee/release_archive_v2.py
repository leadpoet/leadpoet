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
        release_name = release["release_hash"].split(":", 1)[1]
        target = root / release_name
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
        entry = {
            "release_hash": archived["release_hash"],
            "commit_sha": archived["commit_sha"],
            "archive_hash": archived["archive_hash"],
            "archived_at": archived["archived_at"],
        }
        retained = [entry] + [
            item
            for item in old_releases
            if isinstance(item, Mapping)
            and item.get("release_hash") != entry["release_hash"]
        ]
        retained = retained[: int(retain_releases)]
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
