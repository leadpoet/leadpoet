"""Verify the offline CPython 3.9 scoring wheelhouse and installed runtime."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Tuple
import zipfile


EXPECTED_PYTHON = (3, 9, 24)
CRITICAL_IMPORTS = (
    "aiohttp",
    "bs4",
    "cbor2",
    "cryptography",
    "httpx",
    "lxml",
    "pydantic",
    "trafilatura",
)
_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")


class ScoringWheelhouseError(RuntimeError):
    """The scoring dependency closure differs from the reviewed lock."""


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "")).lower()


def _lock_records(path: Path) -> Dict[str, Tuple[str, str]]:
    records = {}
    pending_name = None
    pending_version = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pin = _PIN_RE.match(line)
        if pin:
            pending_name = _normalized_name(pin.group(1))
            pending_version = pin.group(2)
        digest = _HASH_RE.search(line)
        if digest and pending_name and pending_version:
            if pending_name in records:
                raise ScoringWheelhouseError("duplicate scoring lock package")
            records[pending_name] = (pending_version, digest.group(1))
            pending_name = None
            pending_version = None
    if pending_name is not None or not records:
        raise ScoringWheelhouseError("scoring lock is incomplete")
    return records


def _input_records(path: Path) -> Dict[str, str]:
    records = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pin = _PIN_RE.fullmatch(line)
        if not pin:
            raise ScoringWheelhouseError("scoring input contains a non-exact pin")
        name = _normalized_name(pin.group(1))
        if name in records:
            raise ScoringWheelhouseError("duplicate scoring input package")
        records[name] = pin.group(2)
    return records


def _wheel_metadata(path: Path) -> Tuple[str, str, Tuple[str, ...]]:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ScoringWheelhouseError("wheel metadata layout is invalid")
        metadata = archive.read(metadata_names[0]).decode("utf-8", "strict")
        wheel = archive.read(wheel_names[0]).decode("utf-8", "strict")
    name = ""
    package_version = ""
    for line in metadata.splitlines():
        if line.startswith("Name: "):
            name = _normalized_name(line[6:].strip())
        elif line.startswith("Version: "):
            package_version = line[9:].strip()
        if name and package_version:
            break
    tags = tuple(
        line[5:].strip()
        for line in wheel.splitlines()
        if line.startswith("Tag: ")
    )
    if not name or not package_version or not tags:
        raise ScoringWheelhouseError("wheel name, version, or tags are missing")
    return name, package_version, tags


def _tag_is_compatible(tag: str) -> bool:
    parts = tag.split("-", 2)
    if len(parts) != 3:
        return False
    python_tag, abi_tag, platform_tag = parts
    if platform_tag == "any":
        return python_tag in {"py2.py3", "py3"} and abi_tag == "none"
    if "x86_64" not in platform_tag or "manylinux" not in platform_tag:
        return False
    if python_tag == "cp39" and abi_tag in {"cp39", "abi3"}:
        return True
    return python_tag.startswith("cp3") and abi_tag == "abi3"


def verify_wheelhouse(*, input_path: Path, lock_path: Path, wheelhouse: Path) -> Dict[str, str]:
    inputs = _input_records(input_path)
    locked = _lock_records(lock_path)
    if inputs != {name: record[0] for name, record in locked.items()}:
        raise ScoringWheelhouseError("scoring input and lock pins differ")
    observed = {}
    wheel_paths = sorted(wheelhouse.glob("*.whl"))
    if len(wheel_paths) != len(locked):
        raise ScoringWheelhouseError("scoring wheelhouse file count differs from lock")
    for wheel_path in wheel_paths:
        name, package_version, tags = _wheel_metadata(wheel_path)
        if name in observed:
            raise ScoringWheelhouseError("scoring wheelhouse contains duplicate package")
        expected = locked.get(name)
        if expected is None or expected[0] != package_version:
            raise ScoringWheelhouseError("scoring wheel package or version is not locked")
        digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        if digest != expected[1]:
            raise ScoringWheelhouseError("scoring wheel hash differs from lock")
        if not any(_tag_is_compatible(tag) for tag in tags):
            raise ScoringWheelhouseError("scoring wheel has no CPython 3.9 x86_64-compatible tag")
        observed[name] = package_version
    if set(observed) != set(locked):
        raise ScoringWheelhouseError("scoring wheelhouse package set differs from lock")
    return observed


def verify_installed(*, lock_path: Path) -> Dict[str, str]:
    if tuple(sys.version_info[:3]) != EXPECTED_PYTHON:
        raise ScoringWheelhouseError(
            "scoring runtime requires Python %s" % ".".join(str(item) for item in EXPECTED_PYTHON)
        )
    locked = _lock_records(lock_path)
    observed = {}
    for name, (expected_version, _digest) in sorted(locked.items()):
        try:
            installed_version = version(name)
        except PackageNotFoundError as exc:
            raise ScoringWheelhouseError("locked scoring package is not installed") from exc
        if installed_version != expected_version:
            raise ScoringWheelhouseError("installed scoring package version differs from lock")
        observed[name] = installed_version
    for module_name in CRITICAL_IMPORTS:
        import_module(module_name)
    return observed


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wheel_parser = subparsers.add_parser("verify-wheelhouse")
    wheel_parser.add_argument("--input", required=True)
    wheel_parser.add_argument("--lock", required=True)
    wheel_parser.add_argument("--wheelhouse", required=True)
    installed_parser = subparsers.add_parser("verify-installed")
    installed_parser.add_argument("--lock", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "verify-wheelhouse":
        records = verify_wheelhouse(
            input_path=Path(args.input),
            lock_path=Path(args.lock),
            wheelhouse=Path(args.wheelhouse),
        )
    else:
        records = verify_installed(lock_path=Path(args.lock))
    print("verified scoring packages: %s" % len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
