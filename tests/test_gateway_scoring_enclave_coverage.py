from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gateway.tee.scoring_import_closure import (
    DYNAMIC_IMPORT_MODULES,
    ENTRYPOINT_MODULES,
    ScoringClosureError,
    build_manifest,
    verify_staged_manifest,
    write_manifest,
)
from gateway.tee.normalize_attested_runtime import normalize_runtime_tree
from gateway.tee.build_identity import build_identity, load_identity, write_identity


ROOT = Path(__file__).resolve().parents[1]


def _stage_manifest_files(manifest: dict, destination: Path) -> None:
    for item in manifest["files"]:
        module = item["module"]
        source = (
            ROOT / "gateway" / Path(*module.split(".")[1:]).with_suffix(".py")
            if module.startswith("gateway.")
            else ROOT / Path(*module.split(".")).with_suffix(".py")
        )
        if not source.exists():
            package_init = (
                ROOT / "gateway" / Path(*module.split(".")[1:]) / "__init__.py"
                if module.startswith("gateway.")
                else ROOT / Path(*module.split(".")) / "__init__.py"
            )
            source = package_init
        target = destination / item["staged_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def test_scoring_import_closure_contains_authority_modules():
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    modules = {item["module"] for item in manifest["files"]}

    assert set(ENTRYPOINT_MODULES) <= modules
    assert set(DYNAMIC_IMPORT_MODULES) <= modules
    assert "research_lab.eval.evaluator" in modules
    assert "research_lab.eval.baseline_summary" in modules
    assert "research_lab.eval.promotion_metric" in modules
    assert "research_lab.eval.private_runtime" in modules
    assert "qualification.scoring.intent_verification_three_stage" in modules
    assert "leadpoet_verifier.economics" in modules
    assert "gateway.tee.egress_policy" in modules
    assert "gateway.tee.egress_proxy" in modules
    assert "RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE" in manifest["environment_variables"]
    assert "QUALIFICATION_OPENROUTER_API_KEY" in manifest["environment_variables"]
    assert manifest["manifest_hash"].startswith("sha256:")


def test_staged_manifest_verifies_every_recorded_file(tmp_path: Path):
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    _stage_manifest_files(manifest, tmp_path)
    manifest_path = tmp_path / "_attested_runtime" / "scoring_import_closure.json"
    write_manifest(manifest, manifest_path)
    normalize_runtime_tree(tmp_path / "_attested_runtime")

    verified = verify_staged_manifest(gateway_root=tmp_path, manifest_path=manifest_path)
    assert verified["manifest_hash"] == manifest["manifest_hash"]


def test_staged_manifest_fails_closed_on_tampered_dependency(tmp_path: Path):
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    _stage_manifest_files(manifest, tmp_path)
    manifest_path = tmp_path / "_attested_runtime" / "scoring_import_closure.json"
    write_manifest(manifest, manifest_path)
    normalize_runtime_tree(tmp_path / "_attested_runtime")
    first_path = tmp_path / manifest["files"][0]["staged_path"]
    first_path.write_text(first_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ScoringClosureError, match="mismatch"):
        verify_staged_manifest(gateway_root=tmp_path, manifest_path=manifest_path)


def test_gateway_eif_build_enforces_scoring_manifest():
    dockerfile = (ROOT / "gateway" / "tee" / "Dockerfile.enclave").read_text(encoding="utf-8")
    stage_script = (ROOT / "gateway" / "tee" / "stage_attested_runtime.sh").read_text(encoding="utf-8")
    start_script = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(encoding="utf-8")

    assert "scoring_import_closure.py verify-staged" in dockerfile
    assert "scoring_import_closure.py\" build" in stage_script
    assert "scoring_import_closure.json" in stage_script
    assert "normalize_attested_runtime.py" in stage_script
    assert "build_identity.py" in stage_script
    assert "build_identity.py verify" in dockerfile
    assert 'COPY _enclave_source/ /app/gateway/' in dockerfile
    assert 'normalize_attested_runtime.py\" --root \"$BUILD_CONTEXT_TMP\"' in stage_script
    assert "--exclude='.source_commit'" in stage_script
    assert "ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE" in stage_script
    assert 'git -C "$CLEAN_SOURCE_ROOT" fetch -q --depth=1 origin "$ATTESTED_COMMIT_SHA"' in stage_script
    assert 'if [ "$RESOLVED_SOURCE_COMMIT" != "$ATTESTED_COMMIT_SHA" ]' in stage_script
    assert '"$SOURCE_GATEWAY_ROOT/" "$BUILD_CONTEXT_TMP/"' in stage_script
    assert '"$GATEWAY_ROOT/" "$BUILD_CONTEXT_TMP/"' not in stage_script
    assert 'pip download' in stage_script
    assert '--require-hashes' in stage_script
    assert '--no-index --find-links=/tmp/wheelhouse' in dockerfile
    assert 'requirements-scoring-py39.lock' in stage_script
    assert 'requirements-scoring-py39.lock' in dockerfile
    assert '--python-version 39' in stage_script
    assert '--abi cp39' in stage_script
    assert 'GATEWAY_ENCLAVE_CPU_COUNT 2' in start_script
    assert 'GATEWAY_ENCLAVE_MEMORY_MB 8192' in start_script
    assert '/home/ec2-user/.config/leadpoet/gateway.env' in start_script


def test_gateway_build_identity_resolve_command_returns_exact_commit(tmp_path: Path):
    commit = "a" * 40
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "gateway" / "tee" / "build_identity.py"),
            "resolve",
            "--gateway-root",
            str(tmp_path / "gateway"),
            "--source-root",
            str(tmp_path),
            "--commit",
            commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == commit


def test_gateway_build_identity_binds_commit_and_scoring_manifest(tmp_path: Path):
    gateway_root = tmp_path / "gateway"
    identity = build_identity(
        commit_sha="a" * 40,
        scoring_manifest_hash="sha256:" + "b" * 64,
    )
    write_identity(
        identity,
        gateway_root / "_attested_runtime" / "gateway_enclave_build_identity.json",
    )

    assert load_identity(gateway_root=gateway_root) == identity


def test_staged_manifest_fails_closed_on_metadata_drift(tmp_path: Path):
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    _stage_manifest_files(manifest, tmp_path)
    manifest_path = tmp_path / "_attested_runtime" / "scoring_import_closure.json"
    write_manifest(manifest, manifest_path)
    normalize_runtime_tree(tmp_path / "_attested_runtime")
    staged_item = next(
        item for item in manifest["files"]
        if item["staged_path"].startswith("_attested_runtime/")
    )
    staged_path = tmp_path / staged_item["staged_path"]
    staged_path.chmod(0o600)

    with pytest.raises(ScoringClosureError, match="mode mismatch"):
        verify_staged_manifest(gateway_root=tmp_path, manifest_path=manifest_path)
