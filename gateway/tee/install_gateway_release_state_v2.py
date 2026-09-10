"""Atomically install a preflighted gateway manifest and compact lineage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
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
    *, prepared_manifest: Path, prepared_lineage: Path,
    active_manifest: Path, active_lineage: Path, expected_commit: str,
) -> Mapping[str, Any]:
    manifest = validate_release_manifest(_read(prepared_manifest))
    if manifest["commit_sha"] != expected_commit:
        raise GatewayReleaseStateInstallError("prepared manifest commit differs")
    lineage = validate_compact_release_lineage_v2(
        _read(prepared_lineage), expected_current_commit=expected_commit,
        expected_current_gateway_release_hash=manifest["release_hash"],
    )
    if active_manifest.parent != active_lineage.parent:
        raise GatewayReleaseStateInstallError("active release paths must share a directory")
    parent = active_manifest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staged_manifest = _stage(parent, active_manifest.name, manifest)
    staged_lineage = _stage(parent, active_lineage.name, lineage)
    backups = []
    originally_absent = []
    try:
        for active in (active_manifest, active_lineage):
            if active.exists():
                if active.is_symlink() or not active.is_file():
                    raise GatewayReleaseStateInstallError("active release state is unsafe")
                backup = active.with_name(active.name + ".previous")
                shutil.copyfile(active, backup)
                backup.chmod(0o600)
                backups.append((active, backup))
            else:
                originally_absent.append(active)
        os.replace(staged_manifest, active_manifest)
        os.replace(staged_lineage, active_lineage)
        # Prove the exact pair after both replacements. Retain the previous pair
        # for bounded restart recovery and release-archive reconciliation.
        validate_release_manifest(_read(active_manifest))
        validate_compact_release_lineage_v2(
            _read(active_lineage), expected_current_commit=expected_commit,
            expected_current_gateway_release_hash=manifest["release_hash"],
        )
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
        staged_lineage.unlink(missing_ok=True)
    return {"commit_sha": expected_commit, "release_hash": manifest["release_hash"],
            "lineage_hash": lineage["lineage_hash"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-lineage", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--active-lineage", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(install_gateway_release_state(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
