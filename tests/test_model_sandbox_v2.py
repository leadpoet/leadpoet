from __future__ import annotations

import ast
import json
import os
import base64
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
    MODEL_SANDBOX_BROKER_DESTINATION,
    MODEL_SANDBOX_PYTHONPATH,
    MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
    MODEL_SANDBOX_SOURCE_ROOT,
    ROOTFS_MANIFEST_NAME,
    ModelSandboxV2Error,
    RunscModelSandboxV2,
    RunscSandboxConfigV2,
    _oci_config,
    _runsc_failure_evidence,
    model_source_import_bootstrap,
    prepare_model_sandbox_cgroup_v2,
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
from tests.private_model_artifact_fixtures import install_reviewed_consumer_snapshot


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


def _runtime(tmp_path: Path):
    runsc = tmp_path / "runsc"
    runsc.write_bytes(b"pinned-runsc-binary")
    runsc.chmod(0o755)
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    marker = rootfs / ROOTFS_MANIFEST_NAME
    marker.write_text('{"rootfs":"pinned"}\n', encoding="utf-8")
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
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def adapter_metadata():\n    return {'version': '1'}\n",
        encoding="utf-8",
    )
    install_reviewed_consumer_snapshot(source)
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
            process_env = dict(
                item.split("=", 1) for item in config["process"]["env"]
            )
            broker_mount = next(
                item
                for item in config["mounts"]
                if item["destination"] == MODEL_SANDBOX_BROKER_DESTINATION
            )
            broker_root = Path(broker_mount["source"])
            provider_socket = broker_root / "provider.sock"
            observed["broker_identity"] = (
                broker_root.stat().st_uid,
                broker_root.stat().st_gid,
                provider_socket.stat().st_uid,
                provider_socket.stat().st_gid,
                provider_socket.stat().st_mode & 0o777,
            )
            observed["broker_mount"] = broker_mount
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(str(provider_socket))
            finally:
                client.close()
            observed["broker_connected"] = True
            return SimpleNamespace(returncode=0, stdout='{"version":"1"}', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transport = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("metadata must not call a provider")
    )
    sandbox = RunscModelSandboxV2(
        config=_runtime(tmp_path),
        transport=transport,
        cgroup_parent="leadpoet-model",
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
    assert "--rootless=false" in observed["command"]
    assert "--rootless=true" not in observed["command"]
    assert "--network=none" in observed["command"]
    config = observed["config"]
    assert config["linux"]["cgroupsPath"].startswith("leadpoet-model/lp-")
    assert config["root"]["readonly"] is True
    assert config["process"]["cwd"] == "/tmp"
    process_env = dict(item.split("=", 1) for item in config["process"]["env"])
    assert process_env["PYTHONPATH"] == MODEL_SANDBOX_PYTHONPATH
    assert MODEL_SANDBOX_PYTHONPATH.split(":") == [
        "/app",
        MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
        MODEL_SANDBOX_SOURCE_ROOT,
    ]
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
    assert "--host-uds=open" in observed["command"]
    run_mount = next(
        item for item in config["mounts"] if item["destination"] == "/run"
    )
    assert run_mount["type"] == "tmpfs"
    assert "noexec" in run_mount["options"]
    sandbox_socket = Path(process_env["LEADPOET_SANDBOX_PROVIDER_SOCKET"])
    assert sandbox_socket == Path(MODEL_SANDBOX_BROKER_DESTINATION) / "provider.sock"
    broker_mount = observed["broker_mount"]
    assert broker_mount["type"] == "bind"
    assert set(broker_mount["options"]) >= {
        "rbind",
        "ro",
        "nosuid",
        "nodev",
        "noexec",
    }
    assert not Path(broker_mount["source"]).is_relative_to(
        Path(config["root"]["path"])
    )
    assert not (Path(config["root"]["path"]) / "leadpoet-model-broker").exists()
    assert "/dev/log" in config["linux"]["maskedPaths"]
    assert observed["broker_identity"] == (
        sandbox.config.uid,
        sandbox.config.gid,
        sandbox.config.uid,
        sandbox.config.gid,
        0o600,
    )
    assert observed["broker_connected"] is True


def test_runsc_model_sandbox_self_test_uses_production_launcher_and_broker(tmp_path):
    observed = {}

    def runner(command, **_kwargs):
        if "run" not in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        bundle_arg = next(item for item in command if item.startswith("--bundle="))
        config = json.loads(
            (Path(bundle_arg.split("=", 1)[1]) / "config.json").read_text()
        )
        destinations = {item["destination"]: item for item in config["mounts"]}
        broker_root = Path(destinations[MODEL_SANDBOX_BROKER_DESTINATION]["source"])
        source_root = Path(destinations[MODEL_SANDBOX_SOURCE_ROOT]["source"])
        observed["command"] = list(command)
        observed["config"] = config
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
    assert observed["config"]["process"]["user"] == {
        "uid": sandbox.config.uid,
        "gid": sandbox.config.gid,
    }
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
def test_prepare_model_sandbox_cgroup_resolves_nitro_root_membership(
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
        ("unexpected failure", "runsc_nonzero"),
    ),
)
def test_runsc_failure_evidence_is_bounded(stderr, expected):
    code, digest = _runsc_failure_evidence(stderr)
    assert code == expected
    assert digest.startswith("sha256:")


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
    redirected = tmp_path / "redirected-broker"
    redirected.symlink_to(outside, target_is_directory=True)
    source_root = tmp_path / "source-root"
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
