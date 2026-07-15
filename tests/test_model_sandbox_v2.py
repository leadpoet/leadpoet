from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
    ROOTFS_MANIFEST_NAME,
    ModelSandboxV2Error,
    RunscModelSandboxV2,
    RunscSandboxConfigV2,
)
from gateway.tee.sandbox_runtime_artifact import (
    build_rootfs_manifest,
    write_rootfs_manifest,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from research_lab.eval import build_local_private_artifact_manifest
from research_lab.eval.private_runtime import canonicalize_private_model_icp
from research_lab.eval.provider_evidence_cache import (
    build_evidence_cache_from_trace_entries,
    icp_evidence_cache_key,
)
from research_lab.eval.snapshot_store import SNAPSHOT_MISS_SENTINEL, SnapshotMiss


def _runtime(tmp_path: Path):
    runsc = tmp_path / "runsc"
    runsc.write_bytes(b"pinned-runsc-binary")
    runsc.chmod(0o755)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    marker = rootfs / ROOTFS_MANIFEST_NAME
    marker.write_text('{"rootfs":"pinned"}\n', encoding="utf-8")
    return RunscSandboxConfigV2(
        runsc_path=runsc,
        runsc_sha256=sha256_bytes(runsc.read_bytes()),
        rootfs_path=rootfs,
        rootfs_manifest_hash=sha256_bytes(marker.read_bytes()),
    )


def _request(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def adapter_metadata():\n    return {'version': '1'}\n",
        encoding="utf-8",
    )
    artifact = build_local_private_artifact_manifest(
        source_path=source,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/manifests/current.json",
        signature_ref="kms:signature",
        component_registry_version="1",
        scoring_adapter_version="1",
    )
    runtime_catalog = build_source_add_runtime_catalog_v2([])
    catalog_result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "provisioned_sources_hash": sha256_json([]),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": runtime_catalog,
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    return {
        "schema_version": MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
        "model_kind": "private",
        "operation": "metadata",
        "artifact": artifact,
        "source_bundle": build_source_bundle_v2(source),
        "module_name": "research_lab_adapter",
        "callable_name": "adapter_metadata",
        "input": {},
        "environment": {},
        "provider_evidence_cache": {},
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "live",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": sha256_json({"job": "model-job-1"}),
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": runtime_catalog,
        "provider_catalog_evidence": {
            "result": catalog_result,
            "root_receipt_hash": "sha256:" + "c" * 64,
        },
    }


def test_runsc_model_sandbox_builds_no_network_readonly_oci_bundle(tmp_path):
    observed = {}

    def runner(command, **kwargs):
        if "run" in command:
            bundle_arg = next(item for item in command if item.startswith("--bundle="))
            config = json.loads(
                (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
            )
            observed["command"] = list(command)
            observed["config"] = config
            observed["stdin"] = kwargs["input"]
            return SimpleNamespace(returncode=0, stdout='{"version":"1"}', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("metadata must not call a provider")
    )
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=runner,
    )
    try:
        result = sandbox.execute(
            _request(tmp_path),
            job_id="model-job-1",
            purpose="research_lab.private_model_run.v2",
            retry_policy_hashes={"openrouter": "sha256:" + "1" * 64},
            terminal_sink=lambda _attempt: None,
            artifact_sink=lambda _artifact: None,
        )
    finally:
        transport.restore()

    assert result["output"] == {"version": "1"}
    assert result["input_hash"] == sha256_json({})
    assert "--network=none" in observed["command"]
    config = observed["config"]
    assert config["root"]["readonly"] is True
    assert config["process"]["capabilities"]["effective"] == []
    assert config["process"]["noNewPrivileges"] is True
    assert {item["type"] for item in config["linux"]["namespaces"]} >= {
        "network",
        "user",
        "pid",
        "mount",
    }
    source_mount = next(
        item for item in config["mounts"] if item["destination"] == "/workspace/app"
    )
    assert "ro" in source_mount["options"]
    assert "/dev/nsm" in config["linux"]["maskedPaths"]
    assert observed["stdin"] == "{}"


def test_private_baseline_builds_exact_measured_provider_evidence_tape(tmp_path):
    raw_response = b'{"results":[{"title":"Measured"}]}'
    trace_entries = [
        {
            "phase": "call",
            "method": "GET",
            "url_redacted": "https://api.exa.ai/search?q=measured",
            "request_byte_len": 0,
            "request_body_b64": "",
            "response_status": 200,
            "response_body_b64": base64.b64encode(raw_response).decode(),
            "response_byte_len": len(raw_response),
            "truncated": False,
            "outcome": "success",
        }
    ]
    request = _request(tmp_path)
    icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    cache_ref = icp_evidence_cache_key(icp)
    request.update(
        {
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {"icp": icp, "context": {"mode": "private_baseline"}},
            "provider_evidence_cache_ref": cache_ref,
            "provider_evidence_mode": "record",
        }
    )
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="[]", stderr=""
        ),
        utc_day_supplier=lambda: "2026-07-10",
    )
    sandbox._run = lambda *_args, **_kwargs: ([], trace_entries)
    try:
        result = sandbox.execute(
            request,
            job_id="model-job-1",
            purpose="research_lab.private_model_run.v2",
            retry_policy_hashes={"exa": "sha256:" + "1" * 64},
            terminal_sink=lambda _attempt: None,
            artifact_sink=lambda _artifact: None,
        )
    finally:
        transport.restore()

    expected = {
        "schema_version": "1.1",
        "rolling_window_hash": "",
        "icp_ref": cache_ref,
        "utc_day": "2026-07-10",
        "entries": build_evidence_cache_from_trace_entries(trace_entries),
    }
    assert result["generated_provider_evidence_cache"] == expected
    assert result["generated_provider_evidence_cache_hash"] == sha256_json(expected)
    assert result["output"] == []


def test_candidate_model_never_claims_to_generate_baseline_tape(tmp_path):
    request = _request(tmp_path)
    icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    request.update(
        {
            "model_kind": "candidate",
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {"icp": icp, "context": {"mode": "candidate"}},
            "provider_evidence_cache_ref": icp_evidence_cache_key(icp),
        }
    )
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="[]", stderr=""
        ),
        utc_day_supplier=lambda: "2026-07-10",
    )
    sandbox._run = lambda *_args, **_kwargs: ([], [])
    try:
        result = sandbox.execute(
            request,
            job_id="model-job-2",
            purpose="research_lab.candidate_model_run.v2",
            retry_policy_hashes={"exa": "sha256:" + "1" * 64},
            terminal_sink=lambda _attempt: None,
            artifact_sink=lambda _artifact: None,
        )
    finally:
        transport.restore()
    assert result["generated_provider_evidence_cache"] == {}
    assert result["generated_provider_evidence_cache_hash"] == sha256_json({})


def test_runsc_dev_replay_has_snapshot_mount_and_no_live_provider_channel(tmp_path):
    observed = {}

    def runner(command, **kwargs):
        if "run" in command:
            bundle_arg = next(item for item in command if item.startswith("--bundle="))
            config = json.loads(
                (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
            )
            observed["command"] = list(command)
            observed["config"] = config
            observed["stdin"] = kwargs["input"]
            return SimpleNamespace(
                returncode=0,
                stdout='[{"company_name":"Measured Co"}]',
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "manifest.json").write_text("{}", encoding="utf-8")
    transport = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("dev replay must not call a live provider")
    )
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=runner,
    )
    try:
        result = sandbox.execute_dev_replay(
            artifact_doc=request["artifact"],
            source_bundle=request["source_bundle"],
            snapshot_root=snapshots,
            module_name="research_lab_adapter",
            callable_name="run_icp",
            icp={
                "industry": "Software",
                "intent_signal": "Hiring a platform engineer",
            },
            context={"dev_eval": True},
            environment={"RESEARCH_LAB_INCONTAINER_TRACE_MAX_BYTES": "1024"},
            credential_env_names=["EXA_API_KEY"],
            miss_policy="strict",
            timeout_seconds=30,
            job_id="dev-replay-job-1",
        )
    finally:
        transport.restore()

    assert result == [{"company_name": "Measured Co"}]
    assert "--network=none" in observed["command"]
    config = observed["config"]
    destinations = {item["destination"]: item for item in config["mounts"]}
    assert "/research_lab_dev_snapshots" in destinations
    assert "ro" in destinations["/research_lab_dev_snapshots"]["options"]
    assert "/run/leadpoet" not in destinations
    process_env = dict(item.split("=", 1) for item in config["process"]["env"])
    assert "LEADPOET_SANDBOX_PROVIDER_SOCKET" not in process_env
    assert process_env["EXA_API_KEY"] == "leadpoet-coordinator-managed-v2"
    assert process_env["RESEARCH_LAB_DEV_SNAPSHOT_DIR"] == (
        "/research_lab_dev_snapshots"
    )
    assert "dev_snapshot" in config["process"]["args"][2]
    assert json.loads(observed["stdin"])["context"] == {"dev_eval": True}


def test_runsc_dev_replay_propagates_typed_snapshot_miss(tmp_path):
    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=2,
                stdout="",
                stderr=SNAPSHOT_MISS_SENTINEL + "exa|GET|api.exa.ai/search|abc\n",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=runner,
    )
    try:
        with pytest.raises(SnapshotMiss, match="api.exa.ai/search"):
            sandbox.execute_dev_replay(
                artifact_doc=request["artifact"],
                source_bundle=request["source_bundle"],
                snapshot_root=snapshots,
                module_name="research_lab_adapter",
                callable_name="run_icp",
                icp={"industry": "Software", "intent_signal": "Hiring"},
                context={"dev_eval": True},
                environment={},
                credential_env_names=[],
                miss_policy="strict",
                timeout_seconds=30,
                job_id="dev-replay-miss",
            )
    finally:
        transport.restore()


def test_runsc_dev_replay_logs_cleanup_failure(tmp_path, caplog):
    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        raise RuntimeError("delete failed")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=runner,
    )
    try:
        with caplog.at_level(
            "WARNING", logger="gateway.tee.model_sandbox_v2"
        ):
            result = sandbox.execute_dev_replay(
                artifact_doc=request["artifact"],
                source_bundle=request["source_bundle"],
                snapshot_root=snapshots,
                module_name="research_lab_adapter",
                callable_name="run_icp",
                icp={"industry": "Software", "intent_signal": "Hiring"},
                context={"dev_eval": True},
                environment={},
                credential_env_names=[],
                miss_policy="strict",
                timeout_seconds=30,
                job_id="dev-replay-cleanup",
            )
    finally:
        transport.restore()

    assert result == []
    assert "research_lab_dev_replay_runsc_cleanup_failed" in caplog.text


def test_runsc_model_sandbox_rejects_runtime_binary_drift(tmp_path):
    config = _runtime(tmp_path)
    config.runsc_path.write_bytes(b"tampered")
    with pytest.raises(ModelSandboxV2Error, match="hash differs"):
        RunscModelSandboxV2(
            config=config,
            transport=BrokeredProviderTransportV2(lambda _request: {}),
        )


def test_runsc_model_sandbox_rejects_secret_environment_fields(tmp_path):
    request = _request(tmp_path)
    request["environment"] = {"PRIVATE_KEY": "must-not-enter"}
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="{}", stderr=""
        ),
    )
    try:
        with pytest.raises(ModelSandboxV2Error, match="secret field"):
            sandbox.execute(
                request,
                job_id="model-job-1",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"openrouter": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()


def test_runsc_model_sandbox_rejects_parent_supplied_provider_credentials(tmp_path):
    request = _request(tmp_path)
    request["environment"] = {"EXA_API_KEY": "parent-value"}
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="{}", stderr=""
        ),
    )
    try:
        with pytest.raises(ModelSandboxV2Error, match="parent-supplied credentials"):
            sandbox.execute(
                request,
                job_id="model-job-1",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"openrouter": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()


@pytest.mark.parametrize(
    "field,value",
    (
        ("provisioned_sources", "not-a-row-list"),
        ("provisioned_sources", [{"adapter_id": "valid-shape"}, "bad-row"]),
        ("private_registry_rows", {"not": "a-list"}),
    ),
)
def test_model_sandbox_rejects_malformed_provider_catalog_rows(
    tmp_path,
    field,
    value,
):
    request = _request(tmp_path)
    request["provider_catalog_evidence"]["result"] = dict(
        request["provider_catalog_evidence"]["result"]
    )
    request["provider_catalog_evidence"]["result"][field] = value
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="{}", stderr=""
        ),
    )
    try:
        with pytest.raises(
            ModelSandboxV2Error,
            match="provider catalog commitment differs",
        ):
            sandbox.execute(
                request,
                job_id="model-job-malformed-catalog",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"openrouter": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()


def test_measured_runtime_config_binds_runsc_python_and_dependency_lock(tmp_path):
    runsc = tmp_path / "runsc"
    runsc.write_bytes(b"pinned-runsc")
    runsc.chmod(0o555)
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("package==1 --hash=sha256:" + "a" * 64 + "\n")
    lock = {
        "schema_version": "leadpoet.runsc_runtime_lock.v2",
        "version": "release-test.0",
        "architecture": "x86_64",
        "source_url": "https://storage.googleapis.com/gvisor/releases/release/test/x86_64/runsc",
        "artifact_filename": "runsc-test-x86_64",
        "install_path": "/usr/local/bin/runsc",
        "size_bytes": len(runsc.read_bytes()),
        "sha256": sha256_bytes(runsc.read_bytes()),
        "sha512": __import__("hashlib").sha512(runsc.read_bytes()).hexdigest(),
    }
    lock_path = tmp_path / "runsc.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    marker = rootfs / ROOTFS_MANIFEST_NAME
    write_rootfs_manifest(
        lock_path=lock_path,
        requirements_lock_path=requirements,
        python_version="3.9.24",
        output_path=marker,
    )

    config = RunscSandboxConfigV2.from_measured_runtime(
        lock_path=lock_path,
        requirements_lock_path=requirements,
        rootfs_path=rootfs,
        runsc_path=runsc,
        python_version="3.9.24",
    )
    assert config.runsc_sha256 == lock["sha256"]
    assert config.rootfs_manifest_hash == sha256_bytes(marker.read_bytes())
    assert build_rootfs_manifest(
        lock_path=lock_path,
        requirements_lock_path=requirements,
        python_version="3.9.24",
    )["runsc_version"] == "release-test.0"

    marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelSandboxV2Error, match="marker differs"):
        RunscSandboxConfigV2.from_measured_runtime(
            lock_path=lock_path,
            requirements_lock_path=requirements,
            rootfs_path=rootfs,
            runsc_path=runsc,
            python_version="3.9.24",
        )
