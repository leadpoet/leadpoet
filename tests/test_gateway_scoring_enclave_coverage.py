from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gateway.tee.scoring_import_closure import (
    AUTORESEARCH_ENTRYPOINT_MODULES,
    DYNAMIC_IMPORT_MODULES,
    ENTRYPOINT_MODULES,
    MEASURED_DATA_PATHS,
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
    for item in manifest["data_files"]:
        source = ROOT / item["source_path"]
        target = destination / item["staged_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def test_scoring_import_closure_contains_authority_modules():
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    modules = {item["module"] for item in manifest["files"]}

    assert set(ENTRYPOINT_MODULES) <= modules
    assert set(AUTORESEARCH_ENTRYPOINT_MODULES) <= modules
    assert set(DYNAMIC_IMPORT_MODULES) <= modules
    assert manifest["autoresearch_entrypoint_modules"] == list(
        AUTORESEARCH_ENTRYPOINT_MODULES
    )
    assert "gateway.research_lab.worker_process" in modules
    assert "gateway.research_lab.worker" in modules
    assert "gateway.research_lab.autoresearch_runtime" in modules
    assert "gateway.research_lab.code_loop_engine" in modules
    assert "gateway.research_lab.code_build" in modules
    assert "gateway.research_lab.dev_eval_runner" in modules
    assert "gateway.research_lab.git_tree_evaluator" in modules
    assert "gateway.research_lab.git_tree_models" in modules
    assert "gateway.research_lab.git_tree_repository" in modules
    assert "gateway.research_lab.git_tree_scheduler" in modules
    assert "gateway.research_lab.git_tree_store" in modules
    assert "research_lab.code_editing" in modules
    assert "research_lab.eval.evaluator" in modules
    assert "research_lab.eval.baseline_summary" in modules
    assert "research_lab.eval.promotion_metric" in modules
    assert "research_lab.eval.private_runtime" in modules
    assert "qualification.scoring.intent_verification_three_stage" in modules
    assert "leadpoet_verifier.economics" in modules
    assert "leadpoet_canonical.attested_v2" in modules
    assert "gateway.tee.protected_workflows" in modules
    assert "gateway.tee.egress_policy" in modules
    assert "gateway.tee.egress_proxy" in modules
    assert "gateway.tee.autoresearch_executor_v2" in modules
    assert "gateway.tee.model_sandbox_v2" in modules
    assert "gateway.tee.provider_broker_v2" in modules
    assert "gateway.tee.scoring_executor_v2" in modules
    assert "gateway.tee.coordinator_active_model_source_v2" in modules
    assert "gateway.research_lab.active_model_authority_v2" in modules
    assert "leadpoet_canonical.allocation_handoff_v2" in modules
    coordinator_modules = {
        item["module"]
        for item in manifest["role_manifests"]["gateway_coordinator"]["files"]
    }
    assert {
        "gateway.tee.coordinator_active_model_source_v2",
        "gateway.tee.coordinator_allocation_source_v2",
        "gateway.tee.coordinator_chain_source_v2",
        "gateway.tee.coordinator_reward_source_v2",
        "gateway.tee.coordinator_source_add_v2",
        "gateway.tee.coordinator_weight_source_v2",
        "gateway.research_lab.active_model_authority_v2",
        "leadpoet_canonical.allocation_handoff_v2",
    } <= coordinator_modules
    assert {item["source_path"] for item in manifest["data_files"]} == set(
        MEASURED_DATA_PATHS
    )
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


def test_staged_manifest_fails_closed_on_tampered_runtime_data(tmp_path: Path):
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    _stage_manifest_files(manifest, tmp_path)
    manifest_path = tmp_path / "_attested_runtime" / "scoring_import_closure.json"
    write_manifest(manifest, manifest_path)
    normalize_runtime_tree(tmp_path / "_attested_runtime")
    staged = tmp_path / manifest["data_files"][0]["staged_path"]
    staged.write_bytes(staged.read_bytes() + b"\n")

    with pytest.raises(ScoringClosureError, match="mismatch"):
        verify_staged_manifest(gateway_root=tmp_path, manifest_path=manifest_path)


def test_staged_manifest_fails_closed_when_autoresearch_root_is_omitted(
    tmp_path: Path,
):
    manifest = build_manifest(gateway_root=ROOT / "gateway", source_root=ROOT)
    _stage_manifest_files(manifest, tmp_path)
    manifest["autoresearch_entrypoint_modules"] = manifest[
        "autoresearch_entrypoint_modules"
    ][1:]
    manifest_path = tmp_path / "_attested_runtime" / "scoring_import_closure.json"
    write_manifest(manifest, manifest_path)
    normalize_runtime_tree(tmp_path / "_attested_runtime")

    with pytest.raises(ScoringClosureError, match="auto-research.*entrypoints"):
        verify_staged_manifest(gateway_root=tmp_path, manifest_path=manifest_path)


def test_gateway_eif_build_enforces_scoring_manifest():
    dockerfile = (ROOT / "gateway" / "tee" / "Dockerfile.enclave").read_text(encoding="utf-8")
    stage_script = (ROOT / "gateway" / "tee" / "stage_attested_runtime.sh").read_text(encoding="utf-8")
    prepare_script = (
        ROOT / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
    ).read_text(encoding="utf-8")
    start_script = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(encoding="utf-8")

    assert "scoring_import_closure.py verify-staged" in dockerfile
    assert "scoring_import_closure.py\" build" in stage_script
    assert "scoring_import_closure.json" in stage_script
    assert "normalize_attested_runtime.py" in stage_script
    assert "build_identity.py" in stage_script
    assert "build_identity.py verify" in dockerfile
    assert "protected_workflows.py" in dockerfile
    assert "protected_workflows.py" in stage_script
    assert "--protected-manifest" in stage_script
    assert "--topology-manifest" in stage_script
    assert "gateway_scoring gateway_autoresearch" in stage_script
    assert "LEADPOET_ENCLAVE_ROLE" in dockerfile
    assert "topology.py" in dockerfile
    assert 'COPY _enclave_source/ /app/gateway/' in dockerfile
    assert 'normalize_attested_runtime.py\" --root \"$BUILD_CONTEXT_TMP\"' in stage_script
    assert "--exclude='.source_commit'" in stage_script
    assert "ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE" in stage_script
    assert 'git -C "$CLEAN_SOURCE_ROOT" fetch -q --depth=1 origin "$ATTESTED_COMMIT_SHA"' in stage_script
    assert 'if [ "$RESOLVED_SOURCE_COMMIT" != "$ATTESTED_COMMIT_SHA" ]' in stage_script
    assert '"$SOURCE_GATEWAY_ROOT/" "$BUILD_CONTEXT_TMP/"' in stage_script
    assert '"$GATEWAY_ROOT/" "$BUILD_CONTEXT_TMP/"' not in stage_script
    assert 'pip download' not in stage_script
    assert 'pip download' in prepare_script
    assert '--require-hashes' in prepare_script
    assert '--no-index --find-links=/tmp/wheelhouse' in dockerfile
    assert 'requirements-scoring-py39.lock' in stage_script
    assert 'requirements-scoring-py39.lock' in dockerfile
    assert '--python-version 39' in prepare_script
    assert '--abi cp39' in prepare_script
    assert 'GATEWAY_TEE_TOPOLOGY_MODE:-full' in start_script
    assert 'gateway_coordinator' in start_script
    assert 'gateway_scoring' in start_script
    assert 'gateway_autoresearch' in start_script
    assert 'TOTAL_CPUS' in start_script
    assert 'TOTAL_MEMORY_MIB' in start_script
    assert 'verify_topology.py' in start_script


def test_scoring_utility_import_does_not_require_bittensor():
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "bittensor" or name.startswith("bittensor."):
        raise ModuleNotFoundError("bittensor intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import Leadpoet.utils.utils_lead_extraction
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
def test_existing_restart_scripts_preserve_attested_build_paths():
    gateway_restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validator_restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    validator_build = (
        ROOT / "validator_tee" / "scripts" / "build_enclave.sh"
    ).read_text(encoding="utf-8")
    validator_dockerfile = (
        ROOT / "validator_tee" / "Dockerfile.enclave"
    ).read_text(encoding="utf-8")

    assert 'bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"' in gateway_restart
    assert 'build_role_enclaves.sh' in gateway_restart
    assert 'GATEWAY_TEE_SKIP_STAGE=1' in gateway_restart
    assert 'GATEWAY_ROOT="$GATEWAY_ROOT"' in gateway_restart
    assert 'GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT"' in gateway_restart
    assert "./start_enclave.sh" in gateway_restart
    role_builder = (
        ROOT / "gateway" / "tee" / "build_role_enclaves.sh"
    ).read_text(encoding="utf-8")
    assert 'docker build' in role_builder
    assert '--build-arg "LEADPOET_ENCLAVE_ROLE=${role}"' in role_builder
    assert (
        '"${GATEWAY_ROOT%/gateway}/validator_tee/host/'
        'docker_image_normalizer_v2.py"'
    ) in role_builder
    assert "validator_tee.host.docker_image_normalizer_v2" not in role_builder
    assert 'nitro-cli build-enclave' in role_builder
    assert 'bash validator_tee/scripts/build_enclave.sh' in validator_restart
    assert "nitro-cli run-enclave" in validator_restart
    assert '"gateway/research_lab"' in validator_build
    assert "COPY gateway/research_lab/ /app/gateway/research_lab/" in validator_dockerfile


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


def test_gateway_build_identity_binds_commit_role_and_execution_manifest(tmp_path: Path):
    gateway_root = tmp_path / "gateway"
    identity = build_identity(
        role="gateway_scoring",
        service_role="gateway_scoring",
        commit_sha="a" * 40,
        execution_manifest_hash="sha256:" + "b" * 64,
        dependency_lock_hash="sha256:" + "e" * 64,
        protected_manifest_hash="sha256:" + "c" * 64,
        topology_hash="sha256:" + "d" * 64,
    )
    write_identity(
        identity,
        gateway_root / "_attested_runtime" / "gateway_enclave_build_identity.json",
    )

    assert load_identity(
        gateway_root=gateway_root,
        expected_role="gateway_scoring",
    ) == identity
    with pytest.raises(Exception, match="role mismatch"):
        load_identity(
            gateway_root=gateway_root,
            expected_role="gateway_coordinator",
        )


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
