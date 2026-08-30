from __future__ import annotations

import ast
import json
import os
import base64
import io
from pathlib import Path
import socket
import subprocess
import sys
import shutil
import tempfile
import threading
from types import SimpleNamespace

import pytest

from gateway.tee import model_sandbox_v2
from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
    MODEL_SANDBOX_BROKER_DIRECTORY,
    MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES,
    MODEL_SANDBOX_PYTHONPATH,
    MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
    MODEL_SANDBOX_SOURCE_DIRECTORY,
    MODEL_SANDBOX_VISIBLE_ROOT,
    ROOTFS_MANIFEST_NAME,
    ModelSandboxFailureProjectionV1,
    ModelSandboxV2Error,
    RunscModelSandboxV2,
    RunscSandboxConfigV2,
    _MEASURED_METADATA_BOOTSTRAP,
    _model_sandbox_process_timeout_seconds,
    _oci_config,
    _runsc_failure_evidence,
    _runsc_model_sandbox_error,
    _strip_private_runtime_failure_markers_v1,
    _sandbox_visible_workspace,
    model_source_import_bootstrap,
    prepare_model_sandbox_cgroup_v2,
    trusted_model_sandbox_import_bootstrap,
)
from gateway.tee.sandbox_runtime_artifact import (
    build_rootfs_manifest,
    write_rootfs_manifest,
)
from gateway.tee.provider_client_v2 import (
    BrokeredProviderTransportV2,
    ProviderClientV2Error,
)
from gateway.tee.sandbox_http_shim_v2 import (
    EVIDENCE_MISS_SENTINEL,
    SOCKET_ENV,
    SandboxHTTPShimV2Error,
    execute as execute_sandbox_http,
)
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)
from research_lab.eval import build_local_private_artifact_manifest
import research_lab.eval.private_runtime as private_runtime_module
from research_lab.eval.private_runtime import (
    PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    PRIVATE_RUNTIME_FAILURE_MARKER,
    PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION,
    canonicalize_private_model_icp,
    strip_incontainer_trace_lines,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    build_evidence_cache_from_trace_entries,
    icp_evidence_cache_key,
)
from research_lab.eval.snapshot_store import SNAPSHOT_MISS_SENTINEL, SnapshotMiss
from tests.test_sourcing_model_contract import _conforming_tree
import research_lab.sourcing_model_contract_check as compatibility


_SHORT_TEST_ROOTFS: list[Path] = []


@pytest.fixture(autouse=True)
def _cleanup_short_test_rootfs():
    start = len(_SHORT_TEST_ROOTFS)
    yield
    for rootfs in _SHORT_TEST_ROOTFS[start:]:
        shutil.rmtree(rootfs, ignore_errors=True)
    del _SHORT_TEST_ROOTFS[start:]


@pytest.fixture(autouse=True)
def _sandbox_source_admission_boundary(monkeypatch):
    """Adapt source admission only for sandbox launcher/mechanics unit tests."""

    calls = []

    def admit(
        root,
        *,
        manifest=None,
        source_tree_hash="",
        use_cache=False,
    ):
        source_root = Path(root)
        policy, policy_hash = compatibility.semantic_compatibility_policy_identity_v1()
        contract_path = source_root / policy["canonical_contract_path"]
        parity_path = source_root / policy["canonical_parity_path"]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest_document = compatibility._manifest_document(manifest)
        receipt = compatibility._semantic_compatibility_receipt(
            mode="semantic_v1",
            consumer_api_version=policy["consumer_api_version"],
            policy_hash=policy_hash,
            source_tree_hash=str(source_tree_hash),
            manifest=manifest_document,
            contract=contract,
            contract_hash=compatibility._snapshot_sha256(contract_path),
            parity_hash=compatibility._snapshot_sha256(parity_path),
            bindings={
                "adapter_version": "sourcing-model-research-lab-adapter:v7",
                "capability_contract_version": (
                    "sourcing-model-runtime-capabilities:v2"
                ),
                "component_registry_version": "sourcing-model-components:v2",
                "routing_compiler_version": "routing-compiler-v3",
                "scoring_adapter_version": "qualification-company-scorer:v1",
            },
        )
        calls.append(
            {
                "manifest_bound": bool(manifest_document),
                "source_admission_exercised": False,
                "use_cache": bool(use_cache),
                "receipt_hash": receipt["receipt_hash"],
            }
        )
        return receipt

    monkeypatch.setattr(
        private_runtime_module,
        "source_tree_compatibility_admission_v1",
        admit,
    )
    monkeypatch.setattr(
        model_sandbox_v2,
        "source_tree_compatibility_admission_v1",
        admit,
    )
    yield calls


def _runtime_receipt_stderr(stdin_payload: str) -> str:
    options = json.loads(stdin_payload)["context"]["runtime_options"]
    receipt = {
        "runtime_cap_seconds": options["runtime_cap_seconds"],
        "capability_contract": {
            "host_registered": [
                "deadline",
                "emit",
                "probe_origin",
                "resolve_host",
            ],
        },
        "industry_taxonomy": {
            "taxonomy_content_hash": "sha256:" + "a" * 64,
        },
        "firmographic_discovery": {"plan": {"target": 5}},
        "branches": [
            {
                "source": "news",
                "compiled_source": "news",
                "source_override": False,
                "route_tool_ids": ["intent.news", "intent.company_site"],
                "route_sources": ["news", "company_site"],
                "route_plan_sha256": "b" * 64,
                "route_policy_sha256": "c" * 64,
                "route_catalog_sha256": "d" * 64,
                "route_context_sha256": "e" * 64,
            }
        ],
    }
    return "sourcing_branch_receipt " + json.dumps(receipt) + "\n"


def test_model_sandbox_process_timeout_uses_committed_runtime_allocation():
    value = {
        "operation": "run_icp",
        "input": {
            "context": {
                "runtime_options": {"runtime_cap_seconds": 1500.0},
            }
        },
    }

    assert _model_sandbox_process_timeout_seconds(value) == 1503
    assert (
        _model_sandbox_process_timeout_seconds(
            {"operation": "metadata", "input": {}}
        )
        == 900
    )


@pytest.mark.parametrize("runtime_cap", [float("inf"), 1500.1, 9.9, "invalid"])
def test_model_sandbox_process_timeout_rejects_invalid_allocation(runtime_cap):
    with pytest.raises(
        ModelSandboxV2Error,
        match="model sandbox runtime allocation is invalid",
    ):
        _model_sandbox_process_timeout_seconds(
            {
                "operation": "run_icp",
                "input": {
                    "context": {
                        "runtime_options": {"runtime_cap_seconds": runtime_cap},
                    }
                },
            }
        )


def _runtime(tmp_path: Path):
    runsc = tmp_path / "runsc"
    runsc.write_bytes(b"pinned-runsc-binary")
    runsc.chmod(0o755)
    rootfs = Path(tempfile.mkdtemp(prefix="lpr-", dir="/tmp"))
    _SHORT_TEST_ROOTFS.append(rootfs)
    marker = rootfs / ROOTFS_MANIFEST_NAME
    marker.write_text('{"rootfs":"pinned"}\n', encoding="utf-8")
    visible_parent = rootfs / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
    visible_parent.mkdir(mode=0o711)
    visible_parent.chmod(0o711)
    sandbox_uid = os.getuid() or 65534
    sandbox_gid = os.getgid() or 65534
    return RunscSandboxConfigV2(
        runsc_path=runsc,
        runsc_sha256=sha256_bytes(runsc.read_bytes()),
        rootfs_path=rootfs,
        rootfs_manifest_hash=sha256_bytes(marker.read_bytes()),
        uid=sandbox_uid,
        gid=sandbox_gid,
    )


def _request(tmp_path: Path):
    source = tmp_path / "source"
    _conforming_tree(source)
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


def _metadata_request(tmp_path: Path):
    request = _request(tmp_path)
    request.update(
        {
            "provider_evidence_mode": "",
            "provider_cost_scope": "",
            "provider_runtime_catalog": {},
            "provider_catalog_evidence": {},
        }
    )
    return request


def _transport_failure_result(request):
    attempt = build_transport_attempt(
        request_id="f" * 32,
        logical_operation_id=request["logical_operation_id"],
        job_id=request["job_id"],
        purpose=request["purpose"],
        provider_id=request["provider_id"],
        attempt_number=request["attempt_number"],
        method=request["method"],
        destination_host="example.com",
        destination_port=443,
        path_hash="sha256:" + "1" * 64,
        nonsecret_headers_hash="sha256:" + "2" * 64,
        body_hash="sha256:" + "3" * 64,
        credential_ref_hash="sha256:" + "4" * 64,
        retry_policy_hash=request["retry_policy_hash"],
        timeout_ms=request["timeout_ms"],
        started_at="2026-07-10T00:00:00Z",
        terminal_status="transport_failure",
        http_status=None,
        response_hash=None,
        request_artifact_hash="sha256:" + "5" * 64,
        response_artifact_hash=None,
        tls_peer_chain_hash=None,
        tls_protocol=None,
        failure_code="timeout",
        completed_at="2026-07-10T00:00:01Z",
    )
    return {
        "terminal_status": "transport_failure",
        "failure_code": "timeout",
        "encrypted_request_artifact_id": "sha256:" + "5" * 64,
        "transport_attempt": attempt,
    }


def test_runsc_model_sandbox_builds_broker_free_metadata_oci_bundle(
    tmp_path, monkeypatch, _sandbox_source_admission_boundary
):
    observed = {}
    probe = {"measured": "probe"}

    def validate_metadata(metadata, **kwargs):
        observed["metadata_bindings"] = kwargs["expected_semantic_bindings"]
        return dict(metadata)

    monkeypatch.setattr(
        model_sandbox_v2,
        "validate_sourcing_adapter_metadata",
        validate_metadata,
    )
    monkeypatch.setattr(
        model_sandbox_v2,
        "_build_consumer_runtime_probe_from_observation_v1",
        lambda value, **_kwargs: (
            probe if value == {"invariants": {"hostile": "raw-observation"}} else {}
        ),
    )

    def runner(command, **kwargs):
        if "run" in command:
            observed["timeout"] = kwargs["timeout"]
            bundle_arg = next(item for item in command if item.startswith("--bundle="))
            config = json.loads(
                (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
            )
            observed["command"] = list(command)
            observed["config"] = config
            observed["stdin"] = kwargs["input"]
            process_env = dict(
                item.split("=", 1) for item in config["process"]["env"]
            )
            rootfs = Path(config["root"]["path"])
            source_root = rootfs / process_env[
                "LEADPOET_MODEL_SOURCE_ROOT"
            ].lstrip("/")
            observed["source_root"] = source_root
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "metadata": {"version": "1"},
                        "runtime_observation": {
                            "invariants": {"hostile": "raw-observation"},
                        },
                    }
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("metadata must not call a provider")
    )
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
        metadata_process_runner=runner,
    )
    monkeypatch.setattr(
        sandbox,
        "_create_provider_scope_v2",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata must not create a provider scope"
        ),
    )
    try:
        result = sandbox.execute(
            _metadata_request(tmp_path),
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
    assert result["compatibility_policy_hash"].startswith("sha256:")
    assert result["compatibility_admission_hash"].startswith("sha256:")
    assert result["consumer_runtime_probe"] == probe
    assert result["consumer_runtime_probe_hash"] == sha256_json(probe)
    assert observed["timeout"] == model_sandbox_v2.MODEL_SANDBOX_METADATA_TIMEOUT_SECONDS
    assert observed["metadata_bindings"] == {
        "adapter_version": "sourcing-model-research-lab-adapter:v7",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "component_registry_version": "sourcing-model-components:v2",
        "routing_compiler_version": "routing-compiler-v3",
        "scoring_adapter_version": "qualification-company-scorer:v1",
    }
    assert "--rootless=false" in observed["command"]
    assert "--rootless=true" not in observed["command"]
    assert "--network=none" in observed["command"]
    config = observed["config"]
    assert config["linux"]["cgroupsPath"].startswith("leadpoet-model/lp-")
    assert config["root"]["readonly"] is True
    assert config["process"]["cwd"] == "/tmp"
    process_args = config["process"]["args"]
    assert process_args[1:4] == ["-I", "-B", "-c"]
    assert "trusted_model_sandbox_import_bootstrap" not in process_args[4]
    assert "import gateway" not in process_args[4]
    assert "from gateway" not in process_args[4]
    process_env = dict(item.split("=", 1) for item in config["process"]["env"])
    assert process_env["PYTHONPATH"].split(":")[:2] == [
        "/app",
        MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
    ]
    assert process_env["PYTHONPATH"].split(":")[2] == process_env[
        "LEADPOET_MODEL_SOURCE_ROOT"
    ]
    assert MODEL_SANDBOX_PYTHONPATH.split(":") == [
        "/app",
        MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
    ]
    assert config["process"]["capabilities"]["effective"] == []
    assert config["process"]["noNewPrivileges"] is True
    assert {item["type"] for item in config["linux"]["namespaces"]} >= {
        "network",
        "user",
        "pid",
        "mount",
    }
    assert all(item["type"] != "bind" for item in config["mounts"])
    assert "/dev/nsm" in config["linux"]["maskedPaths"]
    assert set(json.loads(observed["stdin"])) == {"observation_plan"}
    assert "--host-uds=none" in observed["command"]
    assert "--host-uds=open" not in observed["command"]
    assert "LEADPOET_SANDBOX_PROVIDER_SOCKET" not in process_env
    assert "RESEARCH_LAB_PROVIDER_COST_SCOPE" not in process_env
    run_mount = next(
        item for item in config["mounts"] if item["destination"] == "/run"
    )
    assert run_mount["type"] == "tmpfs"
    assert "noexec" in run_mount["options"]
    assert observed["source_root"].is_relative_to(Path(config["root"]["path"]))
    assert observed["source_root"].name == MODEL_SANDBOX_SOURCE_DIRECTORY
    assert not (observed["source_root"].parent / MODEL_SANDBOX_BROKER_DIRECTORY).exists()
    assert "/dev/log" in config["linux"]["maskedPaths"]
    assert not observed["source_root"].exists()
    visible_parent = (
        Path(config["root"]["path"])
        / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
    )
    assert list(visible_parent.iterdir()) == []
    assert [
        item["manifest_bound"] for item in _sandbox_source_admission_boundary
    ] == [False, True]
    assert all(
        item["source_admission_exercised"] is False
        for item in _sandbox_source_admission_boundary
    )


def test_bounded_metadata_process_drains_stdout_and_stderr_concurrently():
    byte_count = 128 * 1024
    completed = model_sandbox_v2._run_bounded_metadata_process(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                f"payload = b'x' * {byte_count}\n"
                "os.write(1, payload)\n"
                "os.write(2, payload)\n"
            ),
        ],
        input_payload="{}",
        timeout_seconds=5,
        environment=dict(os.environ),
        stdout_limit=byte_count,
        stderr_limit=byte_count,
        termination_grace_seconds=0.5,
    )

    assert completed.returncode == 0
    assert len(completed.stdout.encode("utf-8")) == byte_count
    assert len(completed.stderr.encode("utf-8")) == byte_count


@pytest.mark.parametrize("extra_byte", (0, 1))
def test_bounded_metadata_process_enforces_dedicated_stdout_cap(extra_byte):
    byte_count = model_sandbox_v2.MAX_MODEL_METADATA_OUTPUT_BYTES + extra_byte

    def call():
        return model_sandbox_v2._run_bounded_metadata_process(
            [
                sys.executable,
                "-c",
                f"import os\nos.write(1, b'x' * {byte_count})\n",
            ],
            input_payload="{}",
            timeout_seconds=5,
            environment=dict(os.environ),
            termination_grace_seconds=0.5,
        )

    if extra_byte:
        with pytest.raises(ModelSandboxV2Error, match="output exceeds limit"):
            call()
    else:
        assert len(call().stdout.encode("utf-8")) == byte_count


def test_bounded_metadata_process_stops_hostile_stdout():
    with pytest.raises(
        ModelSandboxV2Error,
        match="model sandbox output exceeds limit",
    ):
        model_sandbox_v2._run_bounded_metadata_process(
            [
                sys.executable,
                "-c",
                "import os\nwhile True: os.write(1, b'x' * 65536)\n",
            ],
            input_payload="{}",
            timeout_seconds=5,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
            termination_grace_seconds=0.5,
        )


def test_bounded_metadata_process_stops_hostile_stderr_without_disclosure():
    secret = "metadata-diagnostic-secret"
    with pytest.raises(ModelSandboxV2Error) as raised:
        model_sandbox_v2._run_bounded_metadata_process(
            [
                sys.executable,
                "-c",
                (
                    "import os\n"
                    f"payload = {secret!r}.encode() * 4096\n"
                    "while True: os.write(2, payload)\n"
                ),
            ],
            input_payload="{}",
            timeout_seconds=5,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
            termination_grace_seconds=0.5,
        )

    message = str(raised.value)
    assert "diagnostic output exceeds limit" in message
    assert "stderr_prefix_hash=sha256:" in message
    assert secret not in message


def test_metadata_timeout_terminates_kills_and_force_deletes(
    tmp_path,
    _sandbox_source_admission_boundary,
):
    class StubbornProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("runsc", timeout)
            return self.returncode

    process = StubbornProcess()
    delete_commands = []

    def metadata_runner(command, **kwargs):
        return model_sandbox_v2._run_bounded_metadata_process(
            command,
            input_payload=kwargs["input"],
            timeout_seconds=0.01,
            environment=kwargs["env"],
            process_factory=lambda *_args, **_kwargs: process,
            stdout_limit=4096,
            stderr_limit=4096,
            termination_grace_seconds=0.01,
        )

    def cleanup_runner(command, **_kwargs):
        delete_commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=cleanup_runner,
        metadata_process_runner=metadata_runner,
    )
    try:
        with pytest.raises(ModelSandboxV2Error, match="model sandbox timed out"):
            sandbox.execute(
                _metadata_request(tmp_path),
                job_id="metadata-timeout-job",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"openrouter": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert len(delete_commands) == 1
    assert delete_commands[0][-3:-1] == ["delete", "--force"]
    assert delete_commands[0][-1].startswith("lp-")
    assert [
        item["manifest_bound"] for item in _sandbox_source_admission_boundary
    ] == [False, True]
    assert all(
        item["source_admission_exercised"] is False
        for item in _sandbox_source_admission_boundary
    )


def test_bounded_metadata_cleanup_closes_pipes_when_process_cannot_be_reaped():
    class UnreapedProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("runsc", timeout)

    process = UnreapedProcess()

    with pytest.raises(
        ModelSandboxV2Error,
        match="process could not be stopped",
    ):
        model_sandbox_v2._run_bounded_metadata_process(
            ["runsc", "run", "metadata"],
            input_payload="{}",
            timeout_seconds=0.01,
            environment={},
            process_factory=lambda *_args, **_kwargs: process,
            termination_grace_seconds=0.01,
        )

    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed


@pytest.mark.parametrize("failed_start", (2, 3))
def test_bounded_metadata_cleanup_survives_thread_start_failure(
    monkeypatch,
    failed_start,
):
    class StoppableProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_calls += 1
            self.returncode = -15

        def kill(self):
            self.kill_calls += 1
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("runsc", timeout)
            return self.returncode

    real_thread = threading.Thread
    created_threads = []
    start_count = 0

    class ControlledThread:
        def __init__(self, *args, **kwargs):
            self._thread = real_thread(*args, **kwargs)
            self.started = False
            created_threads.append(self)

        def start(self):
            nonlocal start_count
            start_count += 1
            if start_count == failed_start:
                raise RuntimeError("bounded metadata thread start failed")
            self.started = True
            self._thread.start()

        def join(self, timeout=None):
            self._thread.join(timeout)

        def is_alive(self):
            return self._thread.is_alive()

    process = StoppableProcess()
    monkeypatch.setattr(model_sandbox_v2, "Thread", ControlledThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        model_sandbox_v2._run_bounded_metadata_process(
            ["runsc", "run", "metadata"],
            input_payload="{}",
            timeout_seconds=1,
            environment={},
            process_factory=lambda *_args, **_kwargs: process,
            termination_grace_seconds=0.01,
        )

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert not any(
        thread.is_alive() for thread in created_threads if thread.started
    )


def test_runsc_model_sandbox_self_test_uses_production_launcher_and_broker(tmp_path):
    observed = {}

    def runner(command, **_kwargs):
        if "run" not in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        bundle_arg = next(item for item in command if item.startswith("--bundle="))
        config = json.loads(
            (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
        )
        process_env = dict(item.split("=", 1) for item in config["process"]["env"])
        rootfs = Path(config["root"]["path"])
        broker_root = rootfs / process_env[
            "LEADPOET_SANDBOX_PROVIDER_SOCKET"
        ].lstrip("/")
        broker_root = broker_root.parent
        source_root = rootfs / process_env["LEADPOET_MODEL_SOURCE_ROOT"].lstrip("/")
        observed["command"] = list(command)
        observed["config"] = config
        compile(config["process"]["args"][2], "<sandbox-self-test>", "exec")
        observed["source_token"] = (source_root / "self-test-token").read_text()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(str(broker_root / "provider.sock"))
            client.sendall(b"leadpoet-model-sandbox-self-test-request-v2")
            observed["response"] = client.recv(128)
        finally:
            client.close()
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"schema_version":"leadpoet.model_sandbox_self_test.v2",'
                '"status":"passed"}'
            ),
            stderr="",
        )

    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        result = sandbox.self_test()
    finally:
        transport.restore()

    assert result == {
        "schema_version": "leadpoet.model_sandbox_self_test.v2",
        "status": "passed",
    }
    assert "--rootless=false" in observed["command"]
    assert "--rootless=true" not in observed["command"]
    assert "--network=none" in observed["command"]
    assert "--host-uds=open" in observed["command"]
    assert "--platform=ptrace" in observed["command"]
    assert observed["source_token"] == "leadpoet-model-sandbox-self-test-v2\n"
    assert observed["response"] == b"leadpoet-model-sandbox-self-test-response-v2"
    assert all(
        item["type"] != "bind" for item in observed["config"]["mounts"]
    )
    process_env = dict(
        item.split("=", 1) for item in observed["config"]["process"]["env"]
    )
    source_visible = Path(process_env["LEADPOET_MODEL_SOURCE_ROOT"])
    socket_visible = Path(process_env["LEADPOET_SANDBOX_PROVIDER_SOCKET"])
    assert source_visible.parts[:2] == (
        "/",
        MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/"),
    )
    assert source_visible.name == MODEL_SANDBOX_SOURCE_DIRECTORY
    assert socket_visible.parent.name == MODEL_SANDBOX_BROKER_DIRECTORY
    assert source_visible.parent == socket_visible.parent.parent
    assert observed["config"]["process"]["user"] == {
        "uid": sandbox.config.uid,
        "gid": sandbox.config.gid,
    }
    assert observed["config"]["linux"]["uidMappings"] == [
        {"containerID": 0, "hostID": 0, "size": 1},
        {
            "containerID": sandbox.config.uid,
            "hostID": sandbox.config.uid,
            "size": 1,
        },
    ]
    assert observed["config"]["linux"]["gidMappings"] == [
        {"containerID": 0, "hostID": 0, "size": 1},
        {
            "containerID": sandbox.config.gid,
            "hostID": sandbox.config.gid,
            "size": 1,
        },
    ]
    assert observed["config"]["linux"]["resources"]["memory"]["limit"] == (
        sandbox.config.memory_limit_bytes
    )
    assert observed["config"]["linux"]["cgroupsPath"].startswith(
        "leadpoet-model/lp-self-test-"
    )


def test_prepare_model_sandbox_cgroup_delegates_required_controllers(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu io memory pids\n", encoding="ascii")
    (root / "cgroup.procs").write_text("101\n202\n", encoding="ascii")
    (root / "cgroup.subtree_control").write_text("", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/\n", encoding="ascii")

    def writer(path, value):
        if path.name == "cgroup.procs":
            current = (root / "cgroup.procs").read_text(encoding="ascii").split()
            (root / "cgroup.procs").write_text(
                "\n".join(item for item in current if item != value),
                encoding="ascii",
            )
            path.parent.mkdir(exist_ok=True)
            existing = path.read_text(encoding="ascii") if path.exists() else ""
            path.write_text(existing + value + "\n", encoding="ascii")
            return
        path.write_text(value.replace("+", ""), encoding="ascii")
        if path == root / "cgroup.subtree_control":
            jobs = root / "leadpoet-model"
            jobs.mkdir(exist_ok=True)
            (jobs / "cgroup.controllers").write_text(
                "cpu io memory pids\n", encoding="ascii"
            )
            (jobs / "cgroup.subtree_control").touch()

    parent = prepare_model_sandbox_cgroup_v2(
        cgroup_root=root,
        proc_self_cgroup_path=proc_cgroup,
        writer=writer,
    )

    assert parent == "leadpoet-model"
    assert (root / "cgroup.procs").read_text(encoding="ascii") == ""
    assert set(
        (root / "leadpoet-runtime" / "cgroup.procs")
        .read_text(encoding="ascii")
        .split()
    ) == {"101", "202"}
    assert set(
        (root / "cgroup.subtree_control").read_text(encoding="ascii").split()
    ) == {"cpu", "io", "memory", "pids"}
    assert set(
        (root / "leadpoet-model" / "cgroup.subtree_control")
        .read_text(encoding="ascii")
        .split()
    ) == {"cpu", "io", "memory", "pids"}


@pytest.mark.parametrize("proc_identity", ["", "0::\n"])
def test_prepare_model_sandbox_cgroup_resolves_v2_root_membership(
    tmp_path,
    proc_identity,
):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (root / "cgroup.procs").write_text(
        "%s\n" % os.getpid(), encoding="ascii"
    )
    (root / "cgroup.subtree_control").write_text("", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text(proc_identity, encoding="ascii")

    def writer(path, value):
        if path.name == "cgroup.procs":
            current = (root / "cgroup.procs").read_text(encoding="ascii").split()
            (root / "cgroup.procs").write_text(
                "\n".join(item for item in current if item != value),
                encoding="ascii",
            )
            path.parent.mkdir(exist_ok=True)
            existing = path.read_text(encoding="ascii") if path.exists() else ""
            path.write_text(existing + value + "\n", encoding="ascii")
            return
        path.write_text(value.replace("+", ""), encoding="ascii")
        if path == root / "cgroup.subtree_control":
            jobs = root / "leadpoet-model"
            jobs.mkdir(exist_ok=True)
            (jobs / "cgroup.controllers").write_text(
                "cpu memory pids\n", encoding="ascii"
            )
            (jobs / "cgroup.subtree_control").touch()

    assert (
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
            writer=writer,
        )
        == "leadpoet-model"
    )
    assert (root / "cgroup.procs").read_text(encoding="ascii") == ""
    assert (
        root / "leadpoet-runtime" / "cgroup.procs"
    ).read_text(encoding="ascii").split() == [str(os.getpid())]


def _write_nitro_cgroup_v1_layout(
    root: Path,
    proc_cgroup: Path,
    *,
    pid: int,
    relative: str = "/",
    include_current_controls: bool = False,
) -> None:
    lines = []
    for hierarchy, controller in enumerate(
        sorted(MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES),
        start=1,
    ):
        current = root / controller / relative.lstrip("/")
        current.mkdir(parents=True)
        (current / "tasks").write_text(f"{pid}\n", encoding="ascii")
        if include_current_controls:
            for filename in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES[controller]:
                (current / filename).write_text("1\n", encoding="ascii")
        lines.append(f"{hierarchy}:{controller}:{relative}")
    proc_cgroup.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_prepare_model_sandbox_cgroup_accepts_nitro_cgroup_v1(tmp_path):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(
        root,
        proc_cgroup,
        pid=os.getpid(),
        relative="/enclave/service",
    )

    assert (
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )
        == "leadpoet-model"
    )
    assert all(
        not (root / controller / "enclave/service" / filename).exists()
        for controller in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES
        for filename in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES[controller]
    )


def test_prepare_model_sandbox_cgroup_v1_rejects_missing_controller(tmp_path):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(root, proc_cgroup, pid=os.getpid())
    proc_cgroup.write_text(
        "1:cpu:/\n2:memory:/\n",
        encoding="ascii",
    )

    with pytest.raises(
        ModelSandboxV2Error,
        match="required cgroup controllers are unavailable",
    ):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_v1_rejects_unproven_membership(tmp_path):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(root, proc_cgroup, pid=999999)

    with pytest.raises(ModelSandboxV2Error, match="cgroup membership differs"):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_v1_accepts_nitro_root_without_child_limits(
    tmp_path,
):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(root, proc_cgroup, pid=os.getpid())

    assert (
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )
        == "leadpoet-model"
    )
    assert all(
        not (root / controller / filename).exists()
        for controller in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES
        for filename in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES[controller]
    )


def test_prepare_model_sandbox_cgroup_v1_rejects_redirected_controller(tmp_path):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(root, proc_cgroup, pid=os.getpid())
    outside = tmp_path / "outside-cpu"
    (root / "cpu").rename(outside)
    (root / "cpu").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ModelSandboxV2Error,
        match="cgroup v1 hierarchy is invalid",
    ):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_v1_rejects_duplicate_identity(tmp_path):
    root = tmp_path / "cgroup"
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write_nitro_cgroup_v1_layout(root, proc_cgroup, pid=os.getpid())
    proc_cgroup.write_text(
        proc_cgroup.read_text(encoding="ascii") + "4:cpu:/other\n",
        encoding="ascii",
    )

    with pytest.raises(ModelSandboxV2Error, match="cgroup identity is invalid"):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_rejects_unproven_nitro_membership(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (root / "cgroup.procs").write_text("999999\n", encoding="ascii")
    (root / "cgroup.subtree_control").write_text("", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("", encoding="ascii")

    with pytest.raises(
        ModelSandboxV2Error,
        match="cgroup identity is unavailable",
    ):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_rejects_malformed_proc_identity(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (root / "cgroup.procs").write_text(
        "%s\n" % os.getpid(), encoding="ascii"
    )
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::not-absolute\n", encoding="ascii")

    with pytest.raises(
        ModelSandboxV2Error,
        match="cgroup identity is invalid",
    ):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
        )


def test_prepare_model_sandbox_cgroup_returns_path_relative_to_nested_parent(
    tmp_path,
):
    root = tmp_path / "cgroup"
    parent = root / "nested" / "enclave"
    runtime = parent / "leadpoet-runtime"
    runtime.mkdir(parents=True)
    (root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (parent / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (parent / "cgroup.procs").write_text("", encoding="ascii")
    (parent / "cgroup.subtree_control").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    (runtime / "cgroup.procs").write_text("101\n", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text(
        "0::/nested/enclave/leadpoet-runtime\n", encoding="ascii"
    )

    def writer(path, value):
        path.write_text(value.replace("+", ""), encoding="ascii")
        if path == parent / "cgroup.subtree_control":
            jobs = parent / "leadpoet-model"
            jobs.mkdir(exist_ok=True)
            (jobs / "cgroup.controllers").write_text(
                "cpu memory pids\n", encoding="ascii"
            )
            (jobs / "cgroup.subtree_control").touch()

    cgroup_parent = prepare_model_sandbox_cgroup_v2(
        cgroup_root=root,
        proc_self_cgroup_path=proc_cgroup,
        writer=writer,
    )

    assert cgroup_parent == "leadpoet-model"


def test_prepare_model_sandbox_cgroup_fails_closed_on_busy_parent(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids\n", encoding="ascii")
    (root / "cgroup.procs").write_text("", encoding="ascii")
    (root / "cgroup.subtree_control").write_text("", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/\n", encoding="ascii")

    def writer(path, value):
        if path.name == "cgroup.subtree_control":
            raise OSError(16, "device or resource busy")
        path.write_text(value, encoding="ascii")

    with pytest.raises(
        ModelSandboxV2Error,
        match="cgroup controller delegation failed",
    ):
        prepare_model_sandbox_cgroup_v2(
            cgroup_root=root,
            proc_self_cgroup_path=proc_cgroup,
            writer=writer,
        )


def test_runsc_model_sandbox_self_test_redacts_launcher_stderr(tmp_path):
    secret = "provider-secret-must-not-escape"

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=128,
                stdout="",
                stderr=(
                    "running container: cannot set up cgroup for root: "
                    f"configuring cgroup: device busy {secret}"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        with pytest.raises(ModelSandboxV2Error) as raised:
            sandbox.self_test()
    finally:
        transport.restore()

    message = str(raised.value)
    assert "code=runsc_cgroup_setup" in message
    assert "stderr_hash=sha256:" in message
    assert secret not in message


@pytest.mark.parametrize(
    ("stderr", "expected"),
    (
        (
            "running container: cannot create gofer process: gofer: "
            "fork/exec /proc/self/exe: invalid argument",
            "runsc_gofer_exec",
        ),
        ("running container: Error setting up root FS: denied", "runsc_rootfs_setup"),
        ("running container: Failure to resolve mounts: denied", "runsc_mount_resolve"),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/workspace/app/self-test-token'",
            "runsc_source_mount_missing",
        ),
        (
            "RuntimeError: rootfs-visible source differs",
            "runsc_source_staging_missing",
        ),
        ("unexpected failure", "runsc_nonzero"),
    ),
)
def test_runsc_failure_evidence_is_bounded(stderr, expected):
    code, digest, exception_class_hash = _runsc_failure_evidence(
        stderr,
        returncode=1,
    )
    assert code == expected
    assert digest.startswith("sha256:")
    assert exception_class_hash is None


def _private_runtime_failure_marker(exception_class_hash, **extra):
    document = {
        "exception_class_hash": exception_class_hash,
        "schema_version": PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION,
        **extra,
    }
    return PRIVATE_RUNTIME_FAILURE_MARKER + " " + json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_runsc_failure_observation_uses_only_terminal_marker_and_wrapper_exit():
    spoofed_hash = "sha256:" + "1" * 64
    observed_hash = "sha256:" + "2" * 64
    stderr = "\n".join(
        (
            _private_runtime_failure_marker(spoofed_hash),
            "credential-secret-must-not-escape",
            _private_runtime_failure_marker(observed_hash),
        )
    ) + "\n"

    stripped, parsed_hash = _strip_private_runtime_failure_markers_v1(stderr)
    assert parsed_hash == observed_hash
    assert PRIVATE_RUNTIME_FAILURE_MARKER not in stripped
    assert "credential-secret-must-not-escape" in stripped

    launcher_code, stderr_hash, exception_class_hash = _runsc_failure_evidence(
        stderr,
        returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    )
    assert launcher_code == "runsc_nonzero"
    assert exception_class_hash == observed_hash
    assert stderr_hash == sha256_bytes(
        strip_incontainer_trace_lines(stripped).encode("utf-8")
    )

    error = _runsc_model_sandbox_error(
        stderr=stderr,
        returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    )
    assert isinstance(error.failure_projection, ModelSandboxFailureProjectionV1)
    assert error.failure_projection.exception_class_hash == observed_hash
    assert error.failure_projection.stderr_hash == stderr_hash
    assert "credential-secret-must-not-escape" not in str(error)


@pytest.mark.parametrize("returncode", (0, 1, 71, 137, -11, None))
def test_runsc_failure_marker_is_not_admitted_without_exact_wrapper_exit(
    returncode,
):
    observed_hash = "sha256:" + "3" * 64
    stderr = _private_runtime_failure_marker(observed_hash) + "\n"

    _launcher_code, _stderr_hash, exception_class_hash = (
        _runsc_failure_evidence(stderr, returncode=returncode)
    )

    assert exception_class_hash is None


def test_runsc_failure_marker_unknown_fields_fail_closed_and_are_stripped():
    earlier_hash = "sha256:" + "4" * 64
    stderr = "\n".join(
        (
            _private_runtime_failure_marker(earlier_hash),
            _private_runtime_failure_marker(
                "sha256:" + "5" * 64,
                unexpected="tampered",
            ),
        )
    ) + "\n"

    stripped, observed_hash = _strip_private_runtime_failure_markers_v1(stderr)

    assert observed_hash is None
    assert stripped == ""
    _code, stderr_hash, exception_class_hash = _runsc_failure_evidence(
        stderr,
        returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    )
    assert stderr_hash == sha256_bytes(b"")
    assert exception_class_hash is None


def test_runsc_failure_marker_without_terminal_lf_is_not_admitted():
    marker = _private_runtime_failure_marker("sha256:" + "5" * 64)

    stripped, observed_hash = _strip_private_runtime_failure_markers_v1(marker)

    assert stripped == ""
    assert observed_hash is None


@pytest.mark.parametrize(
    "stderr",
    (
        "ordinary diagnostic",
        "ordinary diagnostic\n",
        "ordinary diagnostic\r\n",
        "ordinary diagnostic\n\n\n",
        "first\r\nsecond\n\nthird",
    ),
)
def test_runsc_failure_marker_stripping_preserves_unmarked_stderr(stderr):
    stripped, observed_hash = _strip_private_runtime_failure_markers_v1(
        stderr
    )

    assert stripped == stderr
    assert observed_hash is None


def test_runsc_failure_marker_noncanonical_or_oversized_fails_closed():
    digest = "sha256:" + "6" * 64
    noncanonical = (
        PRIVATE_RUNTIME_FAILURE_MARKER
        + ' {"schema_version":"'
        + PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION
        + '","exception_class_hash":"'
        + digest
        + '"}'
    )
    oversized = _private_runtime_failure_marker(digest) + (" " * 600)

    for marker in (noncanonical, oversized):
        _stripped, observed_hash = _strip_private_runtime_failure_markers_v1(
            marker + "\n"
        )
        assert observed_hash is None


def test_metadata_style_nonwrapper_failure_keeps_class_observation_absent():
    spoofed_hash = "sha256:" + "7" * 64
    error = _runsc_model_sandbox_error(
        stderr=_private_runtime_failure_marker(spoofed_hash) + "\n",
        returncode=1,
    )

    assert error.failure_projection is not None
    assert error.failure_projection.exception_class_hash is None


def test_scoring_manager_runs_model_sandbox_self_test_before_accepting_jobs():
    service_path = Path(__file__).parents[1] / "gateway" / "tee" / "tee_service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_v2_scoring_job_manager"
    )
    self_test_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "self_test"
    ]
    executor_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ScoringExecutorV2"
    ]
    cgroup_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_model_sandbox_cgroup_v2"
    ]
    sandbox_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RunscModelSandboxV2"
    ]
    assert len(cgroup_calls) == 1
    assert len(sandbox_calls) == 1
    assert len(self_test_calls) == 1
    assert len(executor_calls) == 1
    assert cgroup_calls[0].lineno < sandbox_calls[0].lineno
    assert sandbox_calls[0].lineno < self_test_calls[0].lineno
    assert self_test_calls[0].lineno < executor_calls[0].lineno
    model_sandbox_keywords = [
        keyword
        for keyword in executor_calls[0].keywords
        if keyword.arg == "model_sandbox"
    ]
    assert len(model_sandbox_keywords) == 1
    assert isinstance(model_sandbox_keywords[0].value, ast.Name)
    assert model_sandbox_keywords[0].value.id == "model_sandbox"


def test_runsc_model_sandbox_rejects_redirected_broker_mount(tmp_path):
    config = _runtime(tmp_path)
    outside = tmp_path / "outside-broker"
    outside.mkdir()
    with _sandbox_visible_workspace(config) as workspace:
        redirected = workspace / MODEL_SANDBOX_BROKER_DIRECTORY
        redirected.symlink_to(outside, target_is_directory=True)
        source_root = workspace / MODEL_SANDBOX_SOURCE_DIRECTORY
        source_root.mkdir()
        with pytest.raises(
            ModelSandboxV2Error,
            match="provider broker identity is invalid",
        ):
            _oci_config(
                config=config,
                source_root=source_root,
                broker_root=redirected,
                process_args=[sys.executable, "-c", "pass"],
                environment={},
            )


def test_runsc_model_sandbox_rejects_source_outside_visible_root(tmp_path):
    config = _runtime(tmp_path)
    source_root = tmp_path / "outside-source"
    source_root.mkdir()
    with pytest.raises(
        ModelSandboxV2Error,
        match="source root is outside the visible root",
    ):
        _oci_config(
            config=config,
            source_root=source_root,
            broker_root=None,
            process_args=[sys.executable, "-c", "pass"],
            environment={},
        )


def test_runsc_model_sandbox_rejects_permissive_visible_parent(tmp_path):
    config = _runtime(tmp_path)
    parent = config.rootfs_path / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
    parent.chmod(0o755)
    with pytest.raises(
        ModelSandboxV2Error,
        match="visible root identity is invalid",
    ):
        with _sandbox_visible_workspace(config):
            pytest.fail("invalid visible parent must not be entered")


def test_visible_workspace_mode_is_independent_of_restrictive_umask(tmp_path):
    config = _runtime(tmp_path)
    previous = os.umask(0o077)
    try:
        with _sandbox_visible_workspace(config) as workspace:
            assert workspace.stat().st_mode & 0o777 == 0o711
    finally:
        os.umask(previous)


def test_runsc_model_sandbox_rejects_missing_visible_parent(tmp_path):
    config = _runtime(tmp_path)
    parent = config.rootfs_path / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
    parent.rmdir()
    with pytest.raises(
        ModelSandboxV2Error,
        match="visible root is unavailable",
    ):
        with _sandbox_visible_workspace(config):
            pytest.fail("missing measured visible parent must not be created")


def test_model_source_bootstrap_preserves_trusted_gateway_and_canonical_packages(
    tmp_path,
):
    trusted_root = tmp_path / "trusted"
    attested_root = tmp_path / "attested"
    source_root = tmp_path / "source"
    neutral_root = tmp_path / "neutral"
    neutral_root.mkdir()

    packages = {
        trusted_root / "gateway" / "__init__.py": "ORIGIN = 'trusted'\n",
        trusted_root / "gateway" / "tee" / "__init__.py": "ORIGIN = 'trusted'\n",
        attested_root / "leadpoet_canonical" / "__init__.py": "ORIGIN = 'trusted'\n",
        source_root / "gateway" / "__init__.py": "ORIGIN = 'source'\n",
        source_root / "gateway" / "tasks" / "__init__.py": "",
        source_root / "gateway" / "tasks" / "intent_taxonomy.py": (
            "ORIGIN = 'source'\n"
        ),
        source_root / "qualification" / "__init__.py": "ORIGIN = 'source'\n",
        source_root / "validator_models" / "__init__.py": "ORIGIN = 'source'\n",
    }
    for path, content in packages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    code = model_source_import_bootstrap(str(source_root)) + """
import gateway
import gateway.tasks.intent_taxonomy as model_task
import gateway.tee as trusted_tee
import leadpoet_canonical as trusted_canonical
import qualification as model_qualification
import validator_models as model_validator

print(gateway.ORIGIN)
print(trusted_tee.ORIGIN)
print(trusted_canonical.ORIGIN)
print(model_task.ORIGIN)
print(model_qualification.ORIGIN)
print(model_validator.ORIGIN)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=neutral_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(trusted_root), str(attested_root), str(source_root))
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "trusted",
        "trusted",
        "trusted",
        "source",
        "source",
        "source",
    ]


def test_model_source_cannot_shadow_trusted_provider_helper_imports(tmp_path):
    source_root = tmp_path / "source"
    qualification = source_root / "qualification"
    qualification.mkdir(parents=True)
    (qualification / "__init__.py").write_text(
        "ORIGIN = 'model-source'\n",
        encoding="utf-8",
    )
    code = (
        "import pathlib, sys\n"
        "from gateway.tee.sandbox_http_shim_v2 import "
        "_cached_terminal, _snapshot_terminal\n"
        + trusted_model_sandbox_import_bootstrap()
        + model_source_import_bootstrap(str(source_root))
        + """
import qualification

assert pathlib.Path(qualification.__file__).is_relative_to(
    pathlib.Path(_lp_source_root)
)
assert qualification.ORIGIN == 'model-source'
assert 'research_lab.eval.evaluator' not in sys.modules
assert _cached_terminal(
    method='GET',
    url='https://example.com',
    body=b'',
    mode='cache_live',
    cache={},
) is None
assert _snapshot_terminal(
    method='GET',
    url='https://example.com',
    body=b'',
) is None
assert 'research_lab.eval.evaluator' not in sys.modules
print('trusted-provider-helpers-with-model-qualification')
"""
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "trusted-provider-helpers-with-model-qualification"


@pytest.mark.parametrize(
    "forbidden_import",
    (
        "gateway.tee.model_sandbox_v2",
        "gateway.research_lab.model_authority_v2",
        "research_lab.sourcing_model_contract_check",
    ),
)
def test_metadata_observer_cannot_import_trusted_authority_modules(
    tmp_path,
    forbidden_import,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "research_lab_adapter.py").write_text(
        "import %s\n\ndef adapter_metadata():\n    return {}\n"
        % forbidden_import,
        encoding="utf-8",
    )
    observation_plan = {
        "schema_version": "leadpoet.consumer-runtime-observation-plan.v1",
        "runtime_invariants": None,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _MEASURED_METADATA_BOOTSTRAP,
            "research_lab_adapter",
            "adapter_metadata",
        ],
        input=json.dumps({"observation_plan": observation_plan}),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ.get("PATH", ""),
            "LEADPOET_MODEL_SOURCE_ROOT": str(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )

    assert completed.returncode != 0
    assert "metadata observer import is denied" in completed.stderr


@pytest.mark.parametrize(
    "harmless_import",
    ("xml.etree.ElementTree", "typing_extensions"),
)
def test_metadata_observer_allows_unlisted_harmless_dependencies(
    tmp_path,
    harmless_import,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "research_lab_adapter.py").write_text(
        "import %s\n\ndef adapter_metadata():\n    return {}\n"
        % harmless_import,
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _MEASURED_METADATA_BOOTSTRAP,
            "research_lab_adapter",
            "adapter_metadata",
        ],
        input=json.dumps(
            {
                "observation_plan": {
                    "schema_version": (
                        "leadpoet.consumer-runtime-observation-plan.v1"
                    ),
                    "runtime_invariants": None,
                }
            }
        ),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ.get("PATH", ""),
            "LEADPOET_MODEL_SOURCE_ROOT": str(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "metadata": {},
        "runtime_observation": {
            "invariants": {"profile": "legacy_exact"},
            "qualification_outcome_protocol": None,
        },
    }


@pytest.mark.parametrize(
    "dangerous_body",
    (
        "import socket\nsocket.socket()",
        "from pathlib import Path\nPath('observer-write').write_text('no')",
        "import subprocess\nsubprocess.run(['true'])",
    ),
)
def test_metadata_observer_denies_host_interaction(tmp_path, dangerous_body):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "research_lab_adapter.py").write_text(
        dangerous_body + "\n\ndef adapter_metadata():\n    return {}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _MEASURED_METADATA_BOOTSTRAP,
            "research_lab_adapter",
            "adapter_metadata",
        ],
        input=json.dumps(
            {
                "observation_plan": {
                    "schema_version": (
                        "leadpoet.consumer-runtime-observation-plan.v1"
                    ),
                    "runtime_invariants": None,
                }
            }
        ),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ.get("PATH", ""),
            "LEADPOET_MODEL_SOURCE_ROOT": str(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )

    assert completed.returncode != 0
    assert "metadata observer" in completed.stderr
    assert "is denied" in completed.stderr
    assert not (tmp_path / "observer-write").exists()


def test_trusted_parent_rejects_child_claimed_compatibility_decision():
    hostile_observation = {
        "callables": {},
        "constants": {},
        "imports": {},
        "capabilities": {},
        "invariants": {},
        "decision": "accepted",
    }

    with pytest.raises(
        ModelSandboxV2Error,
        match="runtime observation fields are invalid",
    ):
        model_sandbox_v2._build_consumer_runtime_probe_from_observation_v1(
            hostile_observation,
            compatibility_receipt={},
            metadata={},
            expected_source_tree_hash="sha256:" + "1" * 64,
            expected_manifest_hash="sha256:" + "2" * 64,
            expected_image_digest="example.invalid/model@sha256:" + "3" * 64,
            expected_module_name="research_lab_adapter",
            expected_callable_name="adapter_metadata",
        )


def test_eval_package_exports_are_lazy_and_backward_compatible(tmp_path):
    code = """
import sys
import research_lab.eval as evaluation

assert 'research_lab.eval.evaluator' not in sys.modules
assert 'research_lab.eval.private_runtime' not in sys.modules
assert evaluation.PrivateModelRuntimeError.__name__ == 'PrivateModelRuntimeError'
assert 'research_lab.eval.private_runtime' in sys.modules
assert 'research_lab.eval.evaluator' not in sys.modules
assert evaluation.RealEvaluatorRequired.__name__ == 'RealEvaluatorRequired'
assert 'research_lab.eval.evaluator' in sys.modules
assert set(evaluation.__all__) == set(evaluation.lazy_import_contract())
print('lazy-exports-ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "lazy-exports-ok"


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
        cgroup_parent="leadpoet-model",
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


@pytest.mark.parametrize(
    ("provider_error", "raises"),
    (
        ("attested transport failure: unexpected_eof", False),
        ("HTTPError: HTTP Error 500: Internal Server Error; status=500", False),
        ("HTTPError: too many requests; status=429", False),
        ("HTTPError: HTTP Error 402: Payment Required; status=402", True),
        ("HTTPError: HTTP Error 400: Bad Request; status=400; request quota exceeded", True),
        ("HTTPError: HTTP Error 404: Not Found; status=404", False),
    ),
)
def test_runsc_model_sandbox_distinguishes_transport_outage_from_empty_result(
    tmp_path,
    provider_error,
    raises,
):
    request = _request(tmp_path)
    icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    request.update(
        {
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {
                "icp": icp,
                "context": {
                    "mode": "private_baseline",
                    "runtime_options": {
                        "runtime_cap_seconds": 60.0,
                        "finalization_reserve_seconds": 6.0,
                        "agent_timeout_seconds": 54,
                    },
                },
            },
            "provider_evidence_cache_ref": icp_evidence_cache_key(icp),
        }
    )

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="[]",
                stderr=(
                    "research_lab_private_runtime_provider_error "
                    + provider_error
                    + "; url=https://api.exa.ai/search\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        if raises:
            with pytest.raises(
                ModelSandboxV2Error,
                match="provider-backed sourcing failed before returning companies",
            ):
                sandbox.execute(
                    request,
                    job_id="model-job-1",
                    purpose="research_lab.private_model_run.v2",
                    retry_policy_hashes={"exa": "sha256:" + "1" * 64},
                    terminal_sink=lambda _attempt: None,
                    artifact_sink=lambda _artifact: None,
                )
        else:
            result = sandbox.execute(
                request,
                job_id="model-job-1",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"exa": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
            assert result["output"] == []
    finally:
        transport.restore()


def test_runsc_candidate_model_keeps_transport_outage_terminal(tmp_path):
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

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="[]",
                stderr=(
                    "research_lab_private_runtime_provider_error "
                    "attested transport failure: unexpected_eof; "
                    "url=https://api.exa.ai/search\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        with pytest.raises(
            ModelSandboxV2Error,
            match="provider-backed sourcing failed before returning companies",
        ):
            sandbox.execute(
                request,
                job_id="candidate-job-1",
                purpose="research_lab.candidate_model_run.v2",
                retry_policy_hashes={"exa": "sha256:" + "1" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()


def test_model_sandbox_accepts_measured_transport_failure_after_model_fallback(
    tmp_path, monkeypatch
):
    terminals = []
    transport = BrokeredProviderTransportV2(_transport_failure_result)
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=lambda *_args, **_kwargs: None,
    )

    def run_model(*_args, **kwargs):
        monkeypatch.setenv(
            SOCKET_ENV,
            str(kwargs["broker_root"] / "provider.sock"),
        )
        terminal = execute_sandbox_http(
            method="GET",
            url="https://example.com/optional-evidence",
            headers={},
            body=b"",
            timeout_ms=1000,
        )
        assert terminal["terminal_status"] == "transport_failure"
        return {"version": "fallback"}, []

    sandbox._run = run_model
    request = _request(tmp_path)
    canonical_icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    request.update(
        {
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {"icp": canonical_icp, "context": {}},
            "provider_evidence_cache_ref": icp_evidence_cache_key(
                canonical_icp
            ),
        }
    )
    try:
        result = sandbox.execute(
            request,
            job_id="model-job-fallback",
            purpose="research_lab.private_model_run.v2",
            retry_policy_hashes={"public_web": "sha256:" + "6" * 64},
            terminal_sink=lambda attempt: terminals.append(dict(attempt)),
            artifact_sink=lambda _artifact: None,
        )
    finally:
        transport.restore()

    assert result["output"] == {"version": "fallback"}
    assert [item["terminal_status"] for item in terminals] == [
        "transport_failure"
    ]


def test_model_sandbox_still_rejects_missing_transport_terminal(
    tmp_path, monkeypatch
):
    def omit_terminal(_request):
        raise RuntimeError("coordinator omitted terminal")

    transport = BrokeredProviderTransportV2(omit_terminal)
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=lambda *_args, **_kwargs: None,
    )

    def run_model(*_args, **kwargs):
        monkeypatch.setenv(
            SOCKET_ENV,
            str(kwargs["broker_root"] / "provider.sock"),
        )
        with pytest.raises(SandboxHTTPShimV2Error, match="provider socket failed"):
            execute_sandbox_http(
                method="GET",
                url="https://example.com/optional-evidence",
                headers={},
                body=b"",
                timeout_ms=1000,
            )
        return {"version": "must-not-authorize"}, []

    sandbox._run = run_model
    request = _request(tmp_path)
    canonical_icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    request.update(
        {
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {"icp": canonical_icp, "context": {}},
            "provider_evidence_cache_ref": icp_evidence_cache_key(
                canonical_icp
            ),
        }
    )
    try:
        with pytest.raises(
            ProviderClientV2Error,
            match="missing a signed terminal record",
        ):
            sandbox.execute(
                request,
                job_id="model-job-missing-terminal",
                purpose="research_lab.private_model_run.v2",
                retry_policy_hashes={"public_web": "sha256:" + "6" * 64},
                terminal_sink=lambda _attempt: None,
                artifact_sink=lambda _artifact: None,
            )
    finally:
        transport.restore()


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
        cgroup_parent="leadpoet-model",
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
                stderr=_runtime_receipt_stderr(kwargs["input"]),
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
        cgroup_parent="leadpoet-model",
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
    assert "--host-uds=open" not in observed["command"]
    config = observed["config"]
    destinations = {item["destination"]: item for item in config["mounts"]}
    assert all(item["type"] != "bind" for item in config["mounts"])
    process_env = dict(item.split("=", 1) for item in config["process"]["env"])
    assert "LEADPOET_SANDBOX_PROVIDER_SOCKET" not in process_env
    assert process_env["EXA_API_KEY"] == "leadpoet-coordinator-managed-v2"
    assert process_env["RESEARCH_LAB_DEV_SNAPSHOT_DIR"].startswith(
        MODEL_SANDBOX_VISIBLE_ROOT + "/lp-job-"
    )
    assert process_env["RESEARCH_LAB_DEV_SNAPSHOT_DIR"].endswith(
        "/dev-snapshots"
    )
    assert "dev_snapshot" in config["process"]["args"][2]
    assert json.loads(observed["stdin"])["context"] == {
        "dev_eval": True,
        "runtime_options": {
            "runtime_cap_seconds": 27.0,
            "finalization_reserve_seconds": 2.7,
            "agent_timeout_seconds": 24,
        },
    }


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
        cgroup_parent="leadpoet-model",
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


def test_runsc_dev_provider_replay_preserves_typed_evidence_miss_after_marker(
    tmp_path,
):
    fingerprint = "exa|GET|api.exa.ai/search|abc"

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
                stdout="",
                stderr=(
                    EVIDENCE_MISS_SENTINEL
                    + fingerprint
                    + "\n\n"
                    + _private_runtime_failure_marker("sha256:" + "9" * 64)
                    + "\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request = _request(tmp_path)
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        with pytest.raises(
            SnapshotMiss,
            match="provider-evidence:" + fingerprint,
        ):
            sandbox.execute_dev_provider_replay(
                artifact_doc=request["artifact"],
                source_bundle=request["source_bundle"],
                module_name="research_lab_adapter",
                callable_name="run_icp",
                icp={"industry": "Software", "intent_signal": "Hiring"},
                context={"dev_eval": True},
                environment={},
                credential_env_names=[],
                provider_evidence_cache={
                    "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                    "entries": {},
                },
                snapshot_root=None,
                timeout_seconds=30,
                job_id="dev-provider-replay-miss",
            )
    finally:
        transport.restore()


def test_runsc_dev_provider_replay_propagates_typed_snapshot_miss(tmp_path):
    request_key = "exa|GET|api.exa.ai/search|abc"

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
                stdout="",
                stderr=SNAPSHOT_MISS_SENTINEL + request_key + "\n",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        with pytest.raises(SnapshotMiss, match="api.exa.ai/search"):
            sandbox.execute_dev_provider_replay(
                artifact_doc=request["artifact"],
                source_bundle=request["source_bundle"],
                module_name="research_lab_adapter",
                callable_name="run_icp",
                icp={"industry": "Software", "intent_signal": "Hiring"},
                context={"dev_eval": True},
                environment={},
                credential_env_names=[],
                provider_evidence_cache={
                    "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                    "entries": {},
                },
                snapshot_root=snapshots,
                timeout_seconds=30,
                job_id="dev-provider-replay-snapshot-miss",
            )
    finally:
        transport.restore()


@pytest.mark.parametrize("replay_kind", ("snapshot", "provider"))
def test_runsc_dev_non_sentinel_failure_is_secret_safe_and_projected(
    tmp_path,
    replay_kind,
):
    raw_secret = "credential-secret-must-not-escape"
    exception_hash = "sha256:" + "8" * 64

    def runner(command, **_kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
                stdout="",
                stderr=(
                    raw_secret
                    + "\n"
                    + _private_runtime_failure_marker(exception_hash)
                    + "\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
        process_runner=runner,
    )
    try:
        with pytest.raises(ModelSandboxV2Error) as raised:
            if replay_kind == "snapshot":
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
                    job_id="dev-replay-failure",
                )
            else:
                sandbox.execute_dev_provider_replay(
                    artifact_doc=request["artifact"],
                    source_bundle=request["source_bundle"],
                    module_name="research_lab_adapter",
                    callable_name="run_icp",
                    icp={"industry": "Software", "intent_signal": "Hiring"},
                    context={"dev_eval": True},
                    environment={},
                    credential_env_names=[],
                    provider_evidence_cache={
                        "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                        "entries": {},
                    },
                    snapshot_root=None,
                    timeout_seconds=30,
                    job_id="dev-provider-replay-failure",
                )
    finally:
        transport.restore()

    projection = raised.value.failure_projection
    assert raw_secret not in str(raised.value)
    assert projection is not None
    assert projection.launcher_code == "runsc_nonzero"
    assert projection.exception_class_hash == exception_hash
    assert projection.stderr_hash == sha256_bytes(raw_secret.encode("utf-8"))


def test_runsc_dev_replay_logs_cleanup_failure(tmp_path, caplog):
    def runner(command, **kwargs):
        if "run" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="[]",
                stderr=_runtime_receipt_stderr(kwargs["input"]),
            )
        raise RuntimeError("delete failed")

    request = _request(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
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
            cgroup_parent="leadpoet-model",
        )


def test_runsc_model_sandbox_rejects_secret_environment_fields(tmp_path):
    request = _request(tmp_path)
    request["environment"] = {"PRIVATE_KEY": "must-not-enter"}
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
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
        cgroup_parent="leadpoet-model",
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
    icp = canonicalize_private_model_icp(
        {"industry": "Software", "intent_signal": "Hiring"}
    )
    request.update(
        {
            "operation": "run_icp",
            "callable_name": "run_icp",
            "input": {"icp": icp, "context": {}},
            "provider_evidence_cache_ref": icp_evidence_cache_key(icp),
        }
    )
    request["provider_catalog_evidence"]["result"] = dict(
        request["provider_catalog_evidence"]["result"]
    )
    request["provider_catalog_evidence"]["result"][field] = value
    transport = BrokeredProviderTransportV2(lambda _request: {})
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
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
