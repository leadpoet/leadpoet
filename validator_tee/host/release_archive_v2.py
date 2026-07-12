"""Retain complete verified validator V2 EIF releases for explicit rollback."""

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

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.release_v2 import (
    parse_pcr0,
    validate_validator_release,
    validate_validator_release_manifest,
)


ARCHIVE_SCHEMA_VERSION = "leadpoet.validator_release_archive.v2"
ARCHIVE_INDEX_SCHEMA_VERSION = "leadpoet.validator_release_archive_index.v2"
DEFAULT_RETAIN_RELEASES = 3
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ValidatorReleaseArchiveV2Error(RuntimeError):
    """A validator release archive is incomplete, mutable, or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorReleaseArchiveV2Error(
            "%s is unavailable or invalid" % field
        ) from exc
    if not isinstance(value, Mapping):
        raise ValidatorReleaseArchiveV2Error("%s must be an object" % field)
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary = Path(name)
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


def _copy_regular(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValidatorReleaseArchiveV2Error(
            "validator release artifact is missing or non-regular: %s" % source
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def verify_archive_directory(path: Path) -> Dict[str, Any]:
    root = Path(path)
    document = _load_json(root / "archive.json", "validator release archive")
    fields = {
        "schema_version",
        "release_hash",
        "commit_sha",
        "archived_at",
        "files",
        "archive_hash",
    }
    if set(document) != fields or document.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ValidatorReleaseArchiveV2Error("validator release archive schema is invalid")
    files = document.get("files")
    expected_files = {
        "validator-v2-release-manifest.json",
        "validator-v2-release.json",
        "validator-enclave.eif",
        "enclave_build_output.txt",
        "Dockerfile.enclave",
        "Dockerfile.base",
        "runtime-artifacts-v2.lock.json",
    }
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ValidatorReleaseArchiveV2Error("validator release archive inventory is incomplete")
    for relative, metadata in files.items():
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256", "size_bytes"}:
            raise ValidatorReleaseArchiveV2Error("validator release inventory entry is invalid")
        candidate = root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise ValidatorReleaseArchiveV2Error("validator release archive file is missing")
        if candidate.stat().st_size != metadata["size_bytes"]:
            raise ValidatorReleaseArchiveV2Error("validator release archive size mismatch")
        if _sha256_file(candidate) != metadata["sha256"]:
            raise ValidatorReleaseArchiveV2Error("validator release archive hash mismatch")
    body = {key: document[key] for key in fields if key != "archive_hash"}
    if document.get("archive_hash") != sha256_json(body):
        raise ValidatorReleaseArchiveV2Error("validator release archive manifest was altered")
    manifest = validate_validator_release_manifest(
        _load_json(
            root / "validator-v2-release-manifest.json",
            "archived validator release manifest",
        )
    )
    release = manifest["release"]
    local_release = validate_validator_release(
        _load_json(
            root / "validator-v2-release.json",
            "archived local validator release",
        )
    )
    if local_release != release:
        raise ValidatorReleaseArchiveV2Error("archived local validator release differs")
    if (
        release["release_hash"] != document["release_hash"]
        or release["commit_sha"] != document["commit_sha"]
    ):
        raise ValidatorReleaseArchiveV2Error("validator archived release identity mismatch")
    if files["validator-enclave.eif"]["sha256"] != release["eif_hash"]:
        raise ValidatorReleaseArchiveV2Error("archived validator EIF differs from release")
    if parse_pcr0((root / "enclave_build_output.txt").read_text(encoding="utf-8")) != release["pcr0"]:
        raise ValidatorReleaseArchiveV2Error("archived validator PCR0 differs from release")
    if files["Dockerfile.enclave"]["sha256"] != release["dockerfile_hash"]:
        raise ValidatorReleaseArchiveV2Error("archived validator Dockerfile differs")
    if files["Dockerfile.base"]["sha256"] != release["base_dockerfile_hash"]:
        raise ValidatorReleaseArchiveV2Error("archived validator base Dockerfile differs")
    return document


def archive_verified_release(
    *,
    release_manifest_path: Path,
    validator_tee_root: Path,
    archive_root: Path,
    retain_releases: int = DEFAULT_RETAIN_RELEASES,
    archived_at: Optional[str] = None,
) -> Dict[str, Any]:
    if int(retain_releases) < 3:
        raise ValidatorReleaseArchiveV2Error(
            "validator rollback archive must retain current plus two predecessors"
        )
    manifest = validate_validator_release_manifest(
        _load_json(release_manifest_path, "approved validator release manifest")
    )
    release = manifest["release"]
    tee_root = Path(validator_tee_root)
    local_release = validate_validator_release(
        _load_json(tee_root / "validator-v2-release.json", "local validator release")
    )
    if local_release != release:
        raise ValidatorReleaseArchiveV2Error(
            "local validator build differs from approved six-build release"
        )
    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with (root / ".archive.lock").open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        release_name = release["release_hash"].split(":", 1)[1]
        target = root / release_name
        if target.exists():
            archived = verify_archive_directory(target)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".release.", dir=str(root)))
            try:
                sources = {
                    "validator-v2-release-manifest.json": Path(release_manifest_path),
                    "validator-v2-release.json": tee_root / "validator-v2-release.json",
                    "validator-enclave.eif": tee_root / "validator-enclave.eif",
                    "enclave_build_output.txt": tee_root / "enclave_build_output.txt",
                    "Dockerfile.enclave": tee_root / "Dockerfile.enclave",
                    "Dockerfile.base": tee_root / "Dockerfile.base",
                    "runtime-artifacts-v2.lock.json": tee_root
                    / "runtime-artifacts-v2.lock.json",
                }
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
            index = _load_json(index_path, "validator release archive index")
            if (
                set(index) != {"schema_version", "current_release_hash", "releases"}
                or index.get("schema_version") != ARCHIVE_INDEX_SCHEMA_VERSION
                or not isinstance(index.get("releases"), list)
            ):
                raise ValidatorReleaseArchiveV2Error(
                    "validator release archive index schema is invalid"
                )
            old_releases = list(index["releases"])
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
            if isinstance(item, Mapping) and item.get("release_hash") != entry["release_hash"]
        ]
        retained = retained[: int(retain_releases)]
        _atomic_json(
            index_path,
            {
                "schema_version": ARCHIVE_INDEX_SCHEMA_VERSION,
                "current_release_hash": entry["release_hash"],
                "releases": retained,
            },
        )
        retained_names = {item["release_hash"].split(":", 1)[1] for item in retained}
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
        raise ValidatorReleaseArchiveV2Error("selected validator release hash is invalid")
    archive = Path(archive_root) / normalized.split(":", 1)[1]
    document = verify_archive_directory(archive)
    manifest = _load_json(
        archive / "validator-v2-release-manifest.json",
        "selected validator release manifest",
    )
    _atomic_json(Path(output), manifest)
    return document


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", action="store_true")
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--select-release-hash")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--validator-tee-root", type=Path)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--retain", type=int, default=DEFAULT_RETAIN_RELEASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.archive:
        if not args.release_manifest or not args.validator_tee_root:
            raise ValidatorReleaseArchiveV2Error(
                "archiving requires release manifest and validator TEE root"
            )
        result = archive_verified_release(
            release_manifest_path=args.release_manifest,
            validator_tee_root=args.validator_tee_root,
            archive_root=args.archive_root,
            retain_releases=args.retain,
        )
    elif args.verify:
        result = verify_archive_directory(args.verify)
    else:
        if not args.output:
            raise ValidatorReleaseArchiveV2Error("release selection requires --output")
        result = select_release_manifest(
            archive_root=args.archive_root,
            release_hash=args.select_release_hash,
            output=args.output,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
