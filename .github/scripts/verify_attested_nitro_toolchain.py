#!/usr/bin/env python3
"""Check approved CLI and measured bootstrap bytes before executing the CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


LOCK = Path(__file__).with_name("attested-nitro-toolchain.lock.json")


def _verify_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"toolchain input must be a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"toolchain hash differs: {path.name}")


def verify_toolchain(*, nitro_cli: Path, blobs: Path, lock: dict) -> None:
    _verify_file(nitro_cli, lock["nitro_cli_sha256"])
    for name, digest in lock["blobs"].items():
        if Path(name).name != name:
            raise ValueError("invalid approved blob name")
        _verify_file(blobs / name, digest)
    # Do not execute a binary or helper until every approved input is checked.
    version = subprocess.run(
        [str(nitro_cli.resolve()), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    if version != lock["nitro_cli_version"]:
        raise ValueError("Nitro CLI version differs")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nitro-cli", required=True, type=Path)
    parser.add_argument("--blobs", required=True, type=Path)
    args = parser.parse_args(argv)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify_toolchain(nitro_cli=args.nitro_cli, blobs=args.blobs, lock=lock)
    print("Approved Nitro CLI 1.4.2 and bootstrap hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
