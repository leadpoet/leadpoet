"""Atomically install a preflighted current gateway release manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from leadpoet_canonical.attested_v2 import canonical_json


class GatewayReleaseStateInstallError(RuntimeError):
    pass


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        raise GatewayReleaseStateInstallError("prepared release state is invalid") from exc
    if not isinstance(value, Mapping):
        raise GatewayReleaseStateInstallError("prepared release state is invalid")
    return value


def _stage(parent: Path, name: str, value: Mapping[str, Any]) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", dir=parent)
    path = Path(raw)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write((canonical_json(dict(value)) + "\n").encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)
    return path


def install_gateway_release_state(
    *, prepared_manifest: Path, active_manifest: Path, expected_commit: str,
) -> Mapping[str, Any]:
    manifest = validate_release_manifest(_read(prepared_manifest))
    if manifest["commit_sha"] != expected_commit:
        raise GatewayReleaseStateInstallError("prepared manifest commit differs")
    parent = active_manifest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staged_manifest = _stage(parent, active_manifest.name, manifest)
    backups = []
    originally_absent = []
    try:
        if active_manifest.exists():
            if active_manifest.is_symlink() or not active_manifest.is_file():
                raise GatewayReleaseStateInstallError("active release state is unsafe")
            backup = active_manifest.with_name(active_manifest.name + ".previous")
            shutil.copyfile(active_manifest, backup)
            backup.chmod(0o600)
            backups.append((active_manifest, backup))
        else:
            originally_absent.append(active_manifest)
        os.replace(staged_manifest, active_manifest)
        # Prove the exact manifest after replacement. Retain the previous
        # manifest for bounded restart recovery and archive reconciliation.
        validate_release_manifest(_read(active_manifest))
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        for active, backup in backups:
            if backup.exists():
                os.replace(backup, active)
        for active in originally_absent:
            active.unlink(missing_ok=True)
        raise
    finally:
        staged_manifest.unlink(missing_ok=True)
    return {"commit_sha": expected_commit, "release_hash": manifest["release_hash"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(install_gateway_release_state(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
