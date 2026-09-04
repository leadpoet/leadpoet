"""Deterministic gateway attestation code hash helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


ATTESTED_RUNTIME_DIR = "_attested_runtime"
ATTESTED_RUNTIME_PACKAGES = (
    "Leadpoet",
    "research_lab",
    "leadpoet_verifier",
    "schemas",
    "leadpoet_canonical",
    "qualification",
    "validator_models",
)
ATTESTED_RUNTIME_FILES = (
    "validator_tee/host/docker_operation_guard_v2.py",
)
ATTESTED_RUNTIME_GENERATED_FILES = (
    "gateway_enclave_build_identities/gateway_coordinator.json",
    "gateway_enclave_build_identities/gateway_scoring.json",
    "gateway_enclave_build_identity.json",
    "protected_workflows.json",
    "scoring_import_closure.json",
    "topology.json",
)
_ATTESTED_RUNTIME_ROLES = (
    "gateway_coordinator",
    "gateway_scoring",
)
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FALLBACK_COMMAND_TIMEOUT_SECONDS = 30

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
HASH_SUFFIXES = (".py", ".json", ".txt", ".dat", ".sh")
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


class GatewayCodeHashError(RuntimeError):
    """The clean-checkout runtime could not be reproduced exactly."""


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


def _fallback_environment(source_root: Path) -> dict[str, str]:
    return {
        "HOME": str(Path.home()),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(source_root),
    }


def _run_fallback_command(
    command: tuple[str, ...],
    *,
    source_root: Path,
) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(source_root),
            env=_fallback_environment(source_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=_FALLBACK_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GatewayCodeHashError(
            "attested runtime fallback command did not complete"
        ) from exc
    if completed.returncode != 0:
        raise GatewayCodeHashError("attested runtime fallback command failed")


def _fallback_commit(source_root: Path) -> str:
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .lower()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayCodeHashError(
            "attested runtime fallback commit is unavailable"
        ) from exc
    if not _FULL_COMMIT_RE.fullmatch(commit):
        raise GatewayCodeHashError("attested runtime fallback commit is invalid")
    return commit


def materialize_gateway_code_hash_runtime(
    *,
    gateway_root: Path,
    runtime_fallback_root: Path,
    destination_root: Path,
) -> Path:
    """Reproduce the exact source and generated runtime tree hashed in the EIF."""

    gateway_root = gateway_root.resolve()
    source_root = runtime_fallback_root.resolve()
    destination_root = destination_root.resolve()
    if destination_root.exists() or destination_root.is_symlink():
        raise GatewayCodeHashError(
            "attested runtime fallback destination already exists"
        )
    destination_root.mkdir(parents=True)

    for package in ATTESTED_RUNTIME_PACKAGES:
        source = source_root / package
        if not source.is_dir() or source.is_symlink():
            raise GatewayCodeHashError(
                "attested runtime fallback package is unavailable"
            )
        if any(path.is_symlink() for path in source.rglob("*")):
            raise GatewayCodeHashError(
                "attested runtime fallback package contains a symlink"
            )
        shutil.copytree(source, destination_root / package)

    protected_script = gateway_root / "tee/protected_workflows.py"
    scoring_manifest_script = gateway_root / "tee/scoring_import_closure.py"
    build_identity_script = gateway_root / "tee/build_identity.py"
    _run_fallback_command(
        (
            sys.executable,
            str(protected_script),
            "--root",
            str(source_root),
            "--stage-external-root",
            str(destination_root),
        ),
        source_root=source_root,
    )
    scoring_manifest = destination_root / "scoring_import_closure.json"
    _run_fallback_command(
        (
            sys.executable,
            str(scoring_manifest_script),
            "build",
            "--gateway-root",
            str(gateway_root),
            "--source-root",
            str(source_root),
            "--output",
            str(scoring_manifest),
        ),
        source_root=source_root,
    )

    protected_manifest = gateway_root / "tee/protected_workflows.json"
    topology_manifest = gateway_root / "tee/topology.json"
    shutil.copyfile(protected_manifest, destination_root / "protected_workflows.json")
    shutil.copyfile(topology_manifest, destination_root / "topology.json")

    commit = _fallback_commit(source_root)
    identities_root = destination_root / "gateway_enclave_build_identities"
    dependency_lock = gateway_root / "tee/requirements-scoring-py39.lock"
    for role in _ATTESTED_RUNTIME_ROLES:
        _run_fallback_command(
            (
                sys.executable,
                str(build_identity_script),
                "build",
                "--gateway-root",
                str(gateway_root),
                "--source-root",
                str(source_root),
                "--manifest",
                str(scoring_manifest),
                "--dependency-lock",
                str(dependency_lock),
                "--protected-manifest",
                str(protected_manifest),
                "--topology-manifest",
                str(topology_manifest),
                "--role",
                role,
                "--output",
                str(identities_root / (role + ".json")),
                "--commit",
                commit,
            ),
            source_root=source_root,
        )
    shutil.copyfile(
        identities_root / "gateway_coordinator.json",
        destination_root / "gateway_enclave_build_identity.json",
    )

    generated = {
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
        and path.relative_to(destination_root).as_posix()
        in ATTESTED_RUNTIME_GENERATED_FILES
    }
    if generated != set(ATTESTED_RUNTIME_GENERATED_FILES):
        raise GatewayCodeHashError(
            "attested runtime fallback generated inventory is incomplete"
        )
    return destination_root


def iter_gateway_code_hash_files(
    gateway_root: Path,
    *,
    runtime_fallback_root: Path | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return logical-path/file pairs included in the gateway TEE code hash.

    The enclave hashes ``gateway/_attested_runtime``. Local verifiers can pass
    ``runtime_fallback_root`` so a clean Git checkout hashes the exact
    top-level packages and protected host files copied by the production
    staging script with their enclave logical paths.
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
        for relative_path in ATTESTED_RUNTIME_FILES:
            source = fallback_root / relative_path
            logical_root = Path(ATTESTED_RUNTIME_DIR) / relative_path
            for logical_path, path in _iter_files(source, logical_root):
                files[logical_path] = path

    return tuple(sorted(files.items(), key=lambda item: item[0]))


def iter_gateway_code_hash_payloads(
    gateway_root: Path,
    *,
    runtime_fallback_root: Path | None = None,
) -> tuple[tuple[str, bytes], ...]:
    """Return the exact logical payloads hashed by the gateway enclave."""

    gateway_root = Path(gateway_root).resolve()
    attested_runtime = gateway_root / ATTESTED_RUNTIME_DIR
    if runtime_fallback_root is None or attested_runtime.exists():
        return tuple(
            (logical_path, path.read_bytes())
            for logical_path, path in iter_gateway_code_hash_files(gateway_root)
        )

    files = {
        logical_path: path.read_bytes()
        for logical_path, path in iter_gateway_code_hash_files(gateway_root)
    }
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-code-hash-runtime-"
    ) as temporary_root:
        runtime_root = materialize_gateway_code_hash_runtime(
            gateway_root=gateway_root,
            runtime_fallback_root=Path(runtime_fallback_root),
            destination_root=Path(temporary_root) / ATTESTED_RUNTIME_DIR,
        )
        for logical_path, path in _iter_files(
            runtime_root,
            Path(ATTESTED_RUNTIME_DIR),
        ):
            files[logical_path] = path.read_bytes()
    return tuple(sorted(files.items(), key=lambda item: item[0]))


def compute_gateway_code_hash(
    gateway_root: Path,
    *,
    runtime_fallback_root: Path | None = None,
    log_prefix: str = "[TEE]",
    verbose: bool = True,
) -> str:
    payloads_to_hash = iter_gateway_code_hash_payloads(
        gateway_root,
        runtime_fallback_root=runtime_fallback_root,
    )
    if verbose:
        print(
            f"{log_prefix} Hashing {len(payloads_to_hash)} attested gateway files",
            flush=True,
        )

    hasher = hashlib.sha256()
    for index, (logical_path, payload) in enumerate(payloads_to_hash):
        hasher.update(logical_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(payload)
        hasher.update(b"\0")
        if verbose and (len(payloads_to_hash) <= 20 or index < 5):
            print(f"{log_prefix}    ✓ {logical_path}", flush=True)

    code_hash = hasher.hexdigest()
    if verbose:
        print(
            f"{log_prefix} Code hash computed from {len(payloads_to_hash)} files: "
            f"{code_hash[:32]}...{code_hash[-32:]}",
            flush=True,
        )
    return code_hash
