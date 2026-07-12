"""Verify the offline gVisor artifact and write its measured rootfs marker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


LOCK_SCHEMA_VERSION = "leadpoet.runsc_runtime_lock.v2"
ROOTFS_SCHEMA_VERSION = "leadpoet.model_sandbox_rootfs.v2"
EXPECTED_PYTHON_VERSION = "3.9.24"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA512_RE = re.compile(r"^[0-9a-f]{128}$")


class SandboxRuntimeArtifactError(RuntimeError):
    """The gVisor artifact or measured rootfs marker differs from its lock."""


def load_runsc_lock(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxRuntimeArtifactError("runsc lock is unavailable") from exc
    fields = {
        "schema_version",
        "version",
        "architecture",
        "source_url",
        "artifact_filename",
        "install_path",
        "size_bytes",
        "sha256",
        "sha512",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SandboxRuntimeArtifactError("runsc lock fields are invalid")
    if value["schema_version"] != LOCK_SCHEMA_VERSION:
        raise SandboxRuntimeArtifactError("runsc lock schema is invalid")
    if value["architecture"] != "x86_64":
        raise SandboxRuntimeArtifactError("runsc architecture is unsupported")
    if not str(value["version"]).startswith("release-"):
        raise SandboxRuntimeArtifactError("runsc version is invalid")
    if not str(value["source_url"]).startswith(
        "https://storage.googleapis.com/gvisor/releases/release/"
    ):
        raise SandboxRuntimeArtifactError("runsc source is not an official release")
    if value["install_path"] != "/usr/local/bin/runsc":
        raise SandboxRuntimeArtifactError("runsc install path is invalid")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] <= 0:
        raise SandboxRuntimeArtifactError("runsc size is invalid")
    if not _HASH_RE.fullmatch(str(value["sha256"])):
        raise SandboxRuntimeArtifactError("runsc SHA-256 is invalid")
    if not _SHA512_RE.fullmatch(str(value["sha512"])):
        raise SandboxRuntimeArtifactError("runsc SHA-512 is invalid")
    return dict(value)


def verify_runsc_artifact(*, lock_path: Path, artifact_path: Path) -> Dict[str, Any]:
    lock = load_runsc_lock(lock_path)
    try:
        data = artifact_path.read_bytes()
    except OSError as exc:
        raise SandboxRuntimeArtifactError("offline runsc artifact is unavailable") from exc
    if len(data) != lock["size_bytes"]:
        raise SandboxRuntimeArtifactError("offline runsc artifact size differs")
    if sha256_bytes(data) != lock["sha256"]:
        raise SandboxRuntimeArtifactError("offline runsc artifact SHA-256 differs")
    if hashlib.sha512(data).hexdigest() != lock["sha512"]:
        raise SandboxRuntimeArtifactError("offline runsc artifact SHA-512 differs")
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "version": lock["version"],
        "artifact_filename": lock["artifact_filename"],
        "artifact_hash": lock["sha256"],
        "lock_hash": sha256_json(lock),
    }


def build_rootfs_manifest(
    *,
    lock_path: Path,
    requirements_lock_path: Path,
    python_version: str,
) -> Dict[str, Any]:
    lock = load_runsc_lock(lock_path)
    if str(python_version) != EXPECTED_PYTHON_VERSION:
        raise SandboxRuntimeArtifactError("sandbox Python version differs")
    try:
        requirements = requirements_lock_path.read_bytes()
    except OSError as exc:
        raise SandboxRuntimeArtifactError(
            "scoring dependency lock is unavailable"
        ) from exc
    return {
        "schema_version": ROOTFS_SCHEMA_VERSION,
        "python_version": EXPECTED_PYTHON_VERSION,
        "runsc_version": lock["version"],
        "runsc_sha256": lock["sha256"],
        "runsc_lock_hash": sha256_json(lock),
        "requirements_lock_hash": sha256_bytes(requirements),
        "rootfs_path": "/",
    }


def write_rootfs_manifest(
    *,
    lock_path: Path,
    requirements_lock_path: Path,
    python_version: str,
    output_path: Path,
) -> Dict[str, Any]:
    manifest = build_rootfs_manifest(
        lock_path=lock_path,
        requirements_lock_path=requirements_lock_path,
        python_version=python_version,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--lock", required=True, type=Path)
    verify.add_argument("--artifact", required=True, type=Path)
    marker = subparsers.add_parser("write-rootfs-manifest")
    marker.add_argument("--lock", required=True, type=Path)
    marker.add_argument("--requirements-lock", required=True, type=Path)
    marker.add_argument("--python-version", required=True)
    marker.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_runsc_artifact(
            lock_path=args.lock,
            artifact_path=args.artifact,
        )
    else:
        result = write_rootfs_manifest(
            lock_path=args.lock,
            requirements_lock_path=args.requirements_lock,
            python_version=args.python_version,
            output_path=args.output,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
