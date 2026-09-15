#!/usr/bin/env python3
"""Install the pinned Linux x64 Codex executable when building the Arena image."""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

VERSION = "0.154.0"
URL = "https://registry.npmjs.org/@openai/codex/-/codex-0.154.0-linux-x64.tgz"
SHA512 = "a4FI3A8sGtwGrOqltrPbrS2hajrHQG591EwmRfiRoLMb10VxdBtUGW4gu6IJVYENiYGA7k3P4jlRHEoCZU/s9Q=="
PREFIX = "package/vendor/x86_64-unknown-linux-musl/"


def install(destination: Path, *, archive: Path | None = None) -> None:
    """Verify npm's SHA-512, then install the native runtime without npm/Node."""

    with tempfile.TemporaryDirectory(prefix="arena-codex-install-") as directory:
        package = archive or Path(directory) / "codex.tgz"
        if archive is None:
            with urlopen(URL, timeout=120) as response, package.open("wb") as target:
                shutil.copyfileobj(response, target)
        digest = hashlib.sha512()
        with package.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if base64.b64encode(digest.digest()).decode() != SHA512:
            raise RuntimeError("Codex package checksum mismatch")
        with tarfile.open(package, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.name.startswith(PREFIX):
                    continue
                relative = Path(member.name.removeprefix(PREFIX))
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("invalid Codex package member")
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError("Codex package member is missing")
                target_path = destination / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                target_path.chmod(0o755 if member.mode & 0o111 else 0o644)
        if not (destination / "bin/codex").is_file():
            raise RuntimeError("Codex package executable is missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/opt/arena-codex"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    install(args.destination, archive=args.archive)
