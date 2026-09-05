"""Create the small local runtime identity used by production restarts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import build_local_release_identity
from leadpoet_canonical.attested_v2 import canonical_json
from validator_tee.host.release_v2 import (
    build_local_validator_release_identity,
)


class LocalReleaseV2Error(RuntimeError):
    """The local build identity is incomplete or cannot be installed."""


def _load(path: Path, label: str) -> Any:
    descriptor = -1
    try:
        descriptor = os.open(
            str(Path(path)),
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 4 * 1024 * 1024:
            raise LocalReleaseV2Error(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(4 * 1024 * 1024 + 1)
        if not 0 < len(payload) <= 4 * 1024 * 1024:
            raise LocalReleaseV2Error(f"{label} is not a bounded regular file")
        return json.loads(payload)
    except LocalReleaseV2Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalReleaseV2Error(f"{label} is unavailable or invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((canonical_json(dict(value)) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-build-results", type=Path, required=True)
    parser.add_argument("--validator-release", type=Path, required=True)
    parser.add_argument("--gateway-output", type=Path, required=True)
    parser.add_argument("--validator-output", type=Path, required=True)
    args = parser.parse_args(argv)

    gateway_results = _load(
        args.gateway_build_results, "local gateway build results"
    )
    if not isinstance(gateway_results, list):
        raise LocalReleaseV2Error("local gateway build results must be a list")
    validator_release = _load(
        args.validator_release, "local validator build result"
    )
    if not isinstance(validator_release, Mapping):
        raise LocalReleaseV2Error("local validator build result must be an object")

    gateway = build_local_release_identity(gateway_results)
    validator = build_local_validator_release_identity(validator_release)
    if gateway["commit_sha"] != validator["release"]["commit_sha"]:
        raise LocalReleaseV2Error("local gateway and validator commits differ")
    _write(args.gateway_output, gateway)
    _write(args.validator_output, validator)
    print(f"local_release_commit={gateway['commit_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
