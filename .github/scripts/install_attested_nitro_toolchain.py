#!/usr/bin/env python3
"""Extract only approved userspace files from pinned official Amazon RPMs.

This does not install an RPM, load a driver, reserve CPUs/RAM, or start services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

from verify_attested_nitro_toolchain import LOCK, _verify_file, verify_toolchain


MAX_PACKAGE_BYTES = 64 * 1024 * 1024


def download_package(package: dict, path: Path) -> None:
    total = 0
    with urllib.request.urlopen(package["url"], timeout=60) as response:
        with path.open("xb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > MAX_PACKAGE_BYTES:
                    raise ValueError("Nitro package exceeds size limit")
                output.write(chunk)
    _verify_file(path, package["sha256"])


def extract_file(package: Path, member: str, destination: Path, *, mode: int) -> None:
    # Read one fixed member to stdout. Never unpack an RPM into the filesystem
    # or execute its install hooks. Each RPM hash is verified before this call.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        subprocess.run(
            ["bsdtar", "-xOf", str(package), member],
            stdout=output,
            check=True,
            timeout=60,
        )
    destination.chmod(mode)


def install_toolchain(output: Path) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    output.mkdir(parents=False, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="nitro-rpms-", dir=output.parent) as temporary:
        packages = {}
        for package in lock["packages"]:
            path = Path(temporary) / (package["name"] + ".rpm")
            download_package(package, path)
            packages[package["name"]] = path
        binary = output / "bin/nitro-cli"
        extract_file(
            packages["aws-nitro-enclaves-cli"],
            "./usr/bin/nitro-cli", binary, mode=0o755,
        )
        blobs = output / "blobs"
        for name in lock["blobs"]:
            extract_file(
                packages["aws-nitro-enclaves-cli-devel"],
                "./usr/share/nitro_enclaves/blobs/" + name,
                blobs / name,
                mode=0o755 if name in {"init", "linuxkit"} else 0o644,
            )
        verify_toolchain(nitro_cli=binary, blobs=blobs, lock=lock)
    print("Approved userspace Nitro toolchain installed and verified")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    install_toolchain(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
