#!/usr/bin/env python3.11
"""Materialize immutable public artifacts used by the isolated restart."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import urllib.request


def _download(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_sha512: str | None = None,
    expected_size: int | None = None,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "leadpoet-local-restart-rehearsal/1"},
    )
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    sha256.update(chunk)
                    sha512.update(chunk)
                    size += len(chunk)
        if (
            sha256.hexdigest() != expected_sha256
            or (
                expected_sha512 is not None
                and sha512.hexdigest() != expected_sha512
            )
            or (expected_size is not None and size != expected_size)
        ):
            raise SystemExit(
                f"downloaded external artifact differs from its lock: {url}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _runsc(lock_path: Path, output_root: Path) -> Path:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {
        "artifact_filename",
        "sha256",
        "sha512",
        "size_bytes",
        "source_url",
    }
    if not isinstance(lock, dict) or not required.issubset(lock):
        raise SystemExit("runsc artifact lock is incomplete")
    expected_sha256 = str(lock["sha256"])
    if not expected_sha256.startswith("sha256:"):
        raise SystemExit("runsc SHA-256 lock is invalid")

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / str(lock["artifact_filename"])
    _download(
        url=str(lock["source_url"]),
        destination=destination,
        expected_sha256=expected_sha256.removeprefix("sha256:"),
        expected_sha512=str(lock["sha512"]),
        expected_size=int(lock["size_bytes"]),
    )
    destination.chmod(0o755)
    return destination


def _validator_runtime(lock_path: Path, output_root: Path) -> list[Path]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    artifacts = lock.get("artifacts") if isinstance(lock, dict) else None
    if (
        lock.get("schema_version")
        != "leadpoet.validator_runtime_artifacts.v2"
        or not isinstance(artifacts, dict)
        or not artifacts
    ):
        raise SystemExit("validator runtime artifact lock is incomplete")
    destinations = []
    for name, artifact in sorted(artifacts.items()):
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"filename", "sha256", "url"}
        ):
            raise SystemExit(f"validator runtime lock entry is invalid: {name}")
        filename = str(artifact["filename"])
        expected_sha256 = str(artifact["sha256"]).lower()
        url = str(artifact["url"])
        destination = output_root / filename
        if (
            destination.name != filename
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
            or not url.startswith("https://")
        ):
            raise SystemExit(f"validator runtime filename is invalid: {name}")
        _download(
            url=url,
            destination=destination,
            expected_sha256=expected_sha256,
        )
        destinations.append(destination)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runsc-lock", required=True, type=Path)
    parser.add_argument("--validator-runtime-lock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    destination = _runsc(args.runsc_lock, args.output_root)
    runtime = _validator_runtime(
        args.validator_runtime_lock,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "runsc": str(destination),
                "validator_runtime": [str(path) for path in runtime],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
