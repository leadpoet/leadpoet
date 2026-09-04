#!/usr/bin/env python3.11
"""Create sanitized external approval inputs for the exact restart replica."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()




def _extract_candidate(*, source_repo: Path, commit: str, destination: Path) -> None:
    archive = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(source_repo),
            "archive",
            "--format=tar",
            commit,
        ],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit("candidate release archive path is unsafe")
        bundle.extractall(destination)


def _prepare_offline_root(
    *,
    source_root: Path,
    destination: Path,
    commit: str,
    scoring_wheelhouse_root: Path = Path(
        "/opt/leadpoet/scoring-wheelhouses"
    ),
    external_artifact_root: Path = Path(
        "/opt/leadpoet/external-artifacts"
    ),
) -> None:
    scoring_wheelhouse = scoring_wheelhouse_root / commit
    if not scoring_wheelhouse.is_dir():
        raise SystemExit(
            "exact scoring wheelhouse is unavailable for fixture commit"
        )
    shutil.copytree(
        scoring_wheelhouse,
        destination / "scoring-wheelhouse-py39",
    )
    external = external_artifact_root
    runsc_lock = json.loads(
        (source_root / "gateway/tee/runsc-runtime.lock.json").read_text(
            encoding="utf-8"
        )
    )
    runsc_filename = str(runsc_lock["artifact_filename"])
    shutil.copy2(external / runsc_filename, destination / runsc_filename)
    runtime_lock = json.loads(
        (source_root / "validator_tee/runtime-artifacts-v2.lock.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_root = destination / "validator-runtime"
    runtime_root.mkdir()
    for artifact in runtime_lock["artifacts"].values():
        filename = str(artifact["filename"])
        shutil.copy2(external / filename, runtime_root / filename)


def _materialize_validator_app(
    *,
    source_root: Path,
    offline_root: Path,
    destination: Path,
) -> tuple[str, str]:
    """Materialize the exact application COPY surface of Dockerfile.enclave."""

    artifact_root = source_root / ".validator-tee-artifacts"
    subprocess.run(
        [
            "/usr/bin/python3.11",
            str(
                source_root
                / "validator_tee/scripts/stage_runtime_artifacts_v2.py"
            ),
            "--lock",
            str(source_root / "validator_tee/runtime-artifacts-v2.lock.json"),
            "--output-dir",
            str(artifact_root),
            "--offline-artifact-root",
            str(offline_root / "validator-runtime"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    drand = Path(
        "/opt/leadpoet/drand-cabi-v2/libbittensor_drand_v2.so"
    )
    if not drand.is_file():
        raise SystemExit("measured local drand C ABI artifact is unavailable")
    shutil.copy2(drand, artifact_root / "libbittensor_drand_v2.so")

    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    directory_copies = (
        ("validator_tee/enclave", "validator_tee/enclave"),
        ("leadpoet_canonical", "leadpoet_canonical"),
        ("leadpoet_verifier", "leadpoet_verifier"),
        ("research_lab", "research_lab"),
        ("gateway/research_lab", "gateway/research_lab"),
        ("gateway/qualification/utils", "gateway/qualification/utils"),
        ("qualification/scoring", "qualification/scoring"),
    )
    file_copies = (
        ("validator_tee/runtime-artifacts-v2.lock.json", "validator_tee/runtime-artifacts-v2.lock.json"),
        (".validator-tee-artifacts/manifest.json", "validator_tee/runtime-artifacts-v2.manifest.json"),
        (".validator-tee-artifacts/libbittensor_drand_v2.so", "validator_tee/enclave/libbittensor_drand_v2.so"),
        ("gateway/__init__.py", "gateway/__init__.py"),
        ("gateway/qualification/__init__.py", "gateway/qualification/__init__.py"),
        ("gateway/qualification/models.py", "gateway/qualification/models.py"),
        ("gateway/qualification/config.py", "gateway/qualification/config.py"),
        ("gateway/db/__init__.py", "gateway/db/__init__.py"),
        ("gateway/db/client.py", "gateway/db/client.py"),
        ("gateway/tasks/__init__.py", "gateway/tasks/__init__.py"),
        ("gateway/tasks/icp_generator.py", "gateway/tasks/icp_generator.py"),
        ("qualification/__init__.py", "qualification/__init__.py"),
        ("neurons/validator.py", "neurons/validator.py"),
        ("validator_models/automated_checks.py", "validator_models/automated_checks.py"),
    )
    for source_name, destination_name in directory_copies:
        shutil.copytree(
            source_root / source_name,
            destination / destination_name,
        )
    for source_name, destination_name in file_copies:
        target = destination / destination_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / source_name, target)
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)

    from validator_tee.enclave.runtime_v2 import compute_app_manifest_hash

    app_manifest_hash = compute_app_manifest_hash(destination)
    dependency_files = (
        (
            "/app/validator_tee/enclave/requirements.txt",
            destination / "validator_tee/enclave/requirements.txt",
        ),
        (
            "/app/validator_tee/runtime-artifacts-v2.lock.json",
            destination / "validator_tee/runtime-artifacts-v2.lock.json",
        ),
        (
            "/app/validator_tee/runtime-artifacts-v2.manifest.json",
            destination / "validator_tee/runtime-artifacts-v2.manifest.json",
        ),
    )
    dependency_lock_hash = sha256_json(
        [
            {"path": canonical, "sha256": sha256_bytes(path.read_bytes())}
            for canonical, path in dependency_files
        ]
    )
    return app_manifest_hash, dependency_lock_hash


def _release_build_input(*, commit: str, destination: Path) -> dict:
    from artifact_identity import eif_hash, normalized_image_id, pcr0
    from gateway.tee.verify_release_artifacts_v2 import source_manifest_hash

    builder_root = destination / "release-builder"
    shutil.rmtree(builder_root, ignore_errors=True)
    source_root = builder_root / "source"
    offline_root = builder_root / "offline"
    source_root.mkdir(parents=True)
    offline_root.mkdir()
    _extract_candidate(
        source_repo=Path("/source"),
        commit=commit,
        destination=source_root,
    )
    _prepare_offline_root(
        source_root=source_root,
        destination=offline_root,
        commit=commit,
    )
    app_manifest_hash, dependency_lock_hash = _materialize_validator_app(
        source_root=source_root,
        offline_root=offline_root,
        destination=destination / "validator-app",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "ATTESTED_RUNTIME_COMMIT_SHA": commit,
            "ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE": "1",
            "GATEWAY_ROOT": str(source_root / "gateway"),
            "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": str(offline_root),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(source_root),
            "RESEARCH_LAB_RUNTIME_SOURCE_ROOT": str(source_root),
        }
    )
    subprocess.run(
        [
            "/bin/bash",
            str(source_root / "gateway/tee/stage_attested_runtime.sh"),
        ],
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
    )

    gateway_root = source_root / "gateway"
    generated_identities = (
        gateway_root
        / "_attested_runtime/gateway_enclave_build_identities"
    )
    attested_runtime_output = destination / "gateway-attested-runtime"
    shutil.rmtree(attested_runtime_output, ignore_errors=True)
    shutil.copytree(
        gateway_root / "_attested_runtime",
        attested_runtime_output,
    )
    identity_output = destination / "gateway-enclave-build-identities"
    shutil.rmtree(identity_output, ignore_errors=True)
    shutil.copytree(generated_identities, identity_output)
    source_hash = source_manifest_hash(gateway_root / "_enclave_source")
    dockerfile_hash = sha256_bytes(
        (gateway_root / "tee/Dockerfile.enclave").read_bytes()
    )
    topology = json.loads(
        (gateway_root / "tee/topology.json").read_text(encoding="utf-8")
    )
    topology_roles = topology.get("roles")
    if not isinstance(topology_roles, dict) or not topology_roles:
        raise RuntimeError("release fixture topology roles are invalid")
    identity_names = {
        path.stem for path in generated_identities.glob("*.json")
    }
    if identity_names != set(topology_roles):
        raise RuntimeError("release fixture role identities are incomplete")
    roles = {}
    for role in sorted(topology_roles):
        identity = json.loads(
            (
                gateway_root
                / "_attested_runtime/gateway_enclave_build_identities"
                / f"{role}.json"
            ).read_text(encoding="utf-8")
        )
        historical_role = role == "gateway_autoresearch"
        roles[role] = {
            "build_identity_hash": identity["identity_hash"],
            "commit_sha": commit,
            "dependency_lock_hash": identity["dependency_lock_hash"],
            "dockerfile_hash": dockerfile_hash,
            "eif_hash": (
                _hash(f"historical-eif:{commit}:{role}")
                if historical_role
                else eif_hash(commit, role)
            ),
            "execution_manifest_hash": identity["execution_manifest_hash"],
            "normalized_image_hash": (
                _hash(f"historical-image:{commit}:{role}")
                if historical_role
                else normalized_image_id(commit, role)
            ),
            "pcr0": pcr0(commit),
            "source_manifest_hash": source_hash,
            "service_role": identity["service_role"],
            "topology_hash": identity["topology_hash"],
        }
    value = {
        "commit_sha": commit,
        "gateway_roles": roles,
        "validator_app_manifest_hash": app_manifest_hash,
        "validator_dependency_lock_hash": dependency_lock_hash,
    }
    (destination / "release-build-input.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return value


def _release_build_input_without_scratch(
    *,
    commit: str,
    destination: Path,
) -> dict:
    """Retain only the immutable fixture outputs consumed by launchers."""

    builder_root = destination / "release-builder"
    try:
        value = _release_build_input(commit=commit, destination=destination)
    except BaseException:
        shutil.rmtree(builder_root, ignore_errors=True)
        raise
    shutil.rmtree(builder_root)
    if builder_root.exists():
        raise RuntimeError("release fixture scratch cleanup did not converge")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    release_input = _release_build_input_without_scratch(
        commit=args.candidate_sha,
        destination=Path(
            os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state")
        ),
    )
    print(
        json.dumps(
            {
                "release_build_commit": release_input["commit_sha"],
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
