"""Fail-closed validator V2 checks that run before production shutdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from validator_tee.enclave.hotkey_authority_v2 import (
    validate_hotkey_authority_configuration,
)
from validator_tee.host.hotkey_bootstrap_v2 import validate_hotkey_envelope
from validator_tee.host.release_v2 import validator_release_authority
from validator_tee.host.runtime_v2_bootstrap import build_runtime_configuration
from validator_tee.scripts.stage_runtime_artifacts_v2 import load_lock


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ValidatorRestartPreflightV2Error(RuntimeError):
    """The prepared validator release cannot safely replace production."""


def verify_host_hotkey_directory_empty_v2(path: Path) -> str:
    """Reject every parent-host hotkey entry, including backups and symlinks."""

    directory = Path(path)
    if directory.is_symlink():
        raise ValidatorRestartPreflightV2Error(
            "validator host hotkey directory must not be a symlink"
        )
    if not directory.exists():
        return str(directory)
    if not directory.is_dir():
        raise ValidatorRestartPreflightV2Error(
            "validator host hotkey path is not a directory"
        )
    try:
        first_entry = next(iter(directory.iterdir()), None)
    except OSError as exc:
        raise ValidatorRestartPreflightV2Error(
            "validator host hotkey directory cannot be inspected"
        ) from exc
    if first_entry is not None:
        raise ValidatorRestartPreflightV2Error(
            "usable validator hotkey material remains on the parent"
        )
    return str(directory)


def _load(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorRestartPreflightV2Error(
            "%s is unavailable or invalid" % field
        ) from exc
    if not isinstance(value, Mapping):
        raise ValidatorRestartPreflightV2Error("%s must be an object" % field)
    return dict(value)


def verify_validator_restart_preflight_v2(
    *,
    deploy_commit: str,
    validator_release_manifest: Mapping[str, Any],
    gateway_release_manifest: Mapping[str, Any],
    gateway_release_lineage: Mapping[str, Any],
    hotkey_configuration: Mapping[str, Any],
    hotkey_envelope: Mapping[str, Any],
    runtime_artifact_lock: Mapping[str, Any],
    host_hotkey_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    commit = str(deploy_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValidatorRestartPreflightV2Error(
            "validator deployment commit is invalid"
        )
    validator_manifest = dict(validator_release_manifest)
    validator = validator_release_authority(validator_manifest)
    gateway = validate_release_manifest(gateway_release_manifest)
    configuration = validate_hotkey_authority_configuration(hotkey_configuration)
    envelope = validate_hotkey_envelope(hotkey_envelope)
    lock = load_lock_document(runtime_artifact_lock)
    checked_hotkey_directory = ""
    if host_hotkey_directory is not None:
        checked_hotkey_directory = verify_host_hotkey_directory_empty_v2(
            Path(host_hotkey_directory)
        )
    if validator["commit_sha"] != commit:
        raise ValidatorRestartPreflightV2Error(
            "approved validator V2 release is for another commit"
        )
    if gateway["commit_sha"] != commit:
        raise ValidatorRestartPreflightV2Error(
            "approved gateway V2 release is for another commit"
        )
    build_runtime_configuration(
        validator_release=validator_manifest,
        gateway_release=gateway,
        gateway_release_lineage=gateway_release_lineage,
        hotkey_authority_config=configuration,
    )
    if (
        envelope["validator_hotkey"] != configuration["validator_hotkey"]
        or envelope["hotkey_public_key"] != configuration["hotkey_public_key"]
    ):
        raise ValidatorRestartPreflightV2Error(
            "validator hotkey envelope differs from measured configuration"
        )
    return {
        "schema_version": "leadpoet.validator_restart_preflight.v2",
        "status": "ready",
        "deploy_commit": commit,
        "validator_release_hash": validator["release_hash"],
        "validator_release_manifest_hash": validator_manifest[
            "release_manifest_hash"
        ],
        "gateway_release_hash": gateway["release_hash"],
        "gateway_release_lineage_hash": gateway_release_lineage["lineage_hash"],
        "validator_hotkey": configuration["validator_hotkey"],
        "host_hotkey_directory": checked_hotkey_directory,
        "runtime_artifact_count": len(lock["artifacts"]),
    }


def load_lock_document(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an already-loaded lock with the authoritative file validator."""

    import tempfile

    descriptor, name = tempfile.mkstemp(prefix="validator-runtime-lock.", suffix=".json")
    path = Path(name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(dict(value), handle, sort_keys=True)
        return load_lock(path)
    finally:
        path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-commit", required=True)
    parser.add_argument("--validator-release", type=Path, required=True)
    parser.add_argument("--gateway-release", type=Path, required=True)
    parser.add_argument("--gateway-release-lineage", type=Path, required=True)
    parser.add_argument("--hotkey-config", type=Path, required=True)
    parser.add_argument("--hotkey-envelope", type=Path, required=True)
    parser.add_argument("--runtime-artifact-lock", type=Path, required=True)
    parser.add_argument("--host-hotkey-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_validator_restart_preflight_v2(
        deploy_commit=args.deploy_commit,
        validator_release_manifest=_load(
            args.validator_release, "validator V2 release manifest"
        ),
        gateway_release_manifest=_load(
            args.gateway_release, "gateway V2 release manifest"
        ),
        gateway_release_lineage=_load(
            args.gateway_release_lineage, "gateway V2 release lineage"
        ),
        hotkey_configuration=_load(args.hotkey_config, "validator hotkey config"),
        hotkey_envelope=_load(args.hotkey_envelope, "validator hotkey envelope"),
        runtime_artifact_lock=_load(
            args.runtime_artifact_lock, "validator runtime artifact lock"
        ),
        host_hotkey_directory=args.host_hotkey_directory,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
