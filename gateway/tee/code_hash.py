"""Deterministic gateway attestation code hash helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


ATTESTED_RUNTIME_DIR = "_attested_runtime"
ATTESTED_RUNTIME_PACKAGES = (
    "research_lab",
    "leadpoet_verifier",
    "schemas",
    "leadpoet_canonical",
    "qualification",
    "validator_models",
)

ROOT_FILES = ("main.py", "config.py", "pcr0_allowlist.json")
INCLUDE_DIRS = (
    "api",
    "tasks",
    "utils",
    "models",
    "tee",
    "middleware",
    "research_lab",
    "qualification",
    "fulfillment",
    "leadpoet_canonical",
    "validator_models",
    "miner_models",
    ATTESTED_RUNTIME_DIR,
)
HASH_SUFFIXES = (".py", ".json", ".txt", ".sh")
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "env",
    "logs",
    "node_modules",
    "secrets",
    "validation_artifacts",
    "venv",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".log", ".pem", ".key", ".jwk")
EXCLUDED_NAMES = {
    ".DS_Store",
    ".dockerignore",
    "gateway.log",
    "provision_pcrs.py",
    "verify_code_hash.py",
}


def _is_hashable(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS:
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    if path.name.startswith("."):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.suffix in HASH_SUFFIXES


def _iter_files(root: Path, logical_root: Path) -> Iterable[tuple[str, Path]]:
    if not root.exists():
        return
    if root.is_file():
        if _is_hashable(root):
            yield logical_root.as_posix(), root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and _is_hashable(path):
            yield (logical_root / path.relative_to(root)).as_posix(), path


def iter_gateway_code_hash_files(
    gateway_root: Path,
    *,
    runtime_fallback_root: Path | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return logical-path/file pairs included in the gateway TEE code hash.

    The enclave hashes ``gateway/_attested_runtime``. Local verifiers can pass
    ``runtime_fallback_root`` so a clean Git checkout hashes top-level
    ``research_lab/``, ``leadpoet_verifier/``, and ``schemas/`` with the same
    logical paths produced by the production staging script.
    """

    gateway_root = Path(gateway_root).resolve()
    files: dict[str, Path] = {}

    for filename in ROOT_FILES:
        path = gateway_root / filename
        if path.exists() and _is_hashable(path):
            files[filename] = path

    for dirname in INCLUDE_DIRS:
        root = gateway_root / dirname
        if root.exists():
            for logical_path, path in _iter_files(root, Path(dirname)):
                files[logical_path] = path

    attested_runtime = gateway_root / ATTESTED_RUNTIME_DIR
    if runtime_fallback_root is not None and not attested_runtime.exists():
        fallback_root = Path(runtime_fallback_root).resolve()
        for package in ATTESTED_RUNTIME_PACKAGES:
            source = fallback_root / package
            logical_root = Path(ATTESTED_RUNTIME_DIR) / package
            for logical_path, path in _iter_files(source, logical_root):
                files[logical_path] = path

    return tuple(sorted(files.items(), key=lambda item: item[0]))


def compute_gateway_code_hash(
    gateway_root: Path,
    *,
    runtime_fallback_root: Path | None = None,
    log_prefix: str = "[TEE]",
    verbose: bool = True,
) -> str:
    files_to_hash = iter_gateway_code_hash_files(
        gateway_root,
        runtime_fallback_root=runtime_fallback_root,
    )
    if verbose:
        print(f"{log_prefix} Hashing {len(files_to_hash)} attested gateway files", flush=True)

    hasher = hashlib.sha256()
    for index, (logical_path, file_path) in enumerate(files_to_hash):
        hasher.update(logical_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
        if verbose and (len(files_to_hash) <= 20 or index < 5):
            print(f"{log_prefix}    ✓ {logical_path}", flush=True)

    code_hash = hasher.hexdigest()
    if verbose:
        print(
            f"{log_prefix} Code hash computed from {len(files_to_hash)} files: "
            f"{code_hash[:32]}...{code_hash[-32:]}",
            flush=True,
        )
    return code_hash
