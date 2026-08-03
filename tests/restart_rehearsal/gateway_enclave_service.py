#!/usr/bin/env python3.11
"""Persistent role-isolated process for real gateway enclave RPC handlers."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

from sitecustomize import (
    _external_event,
    _gateway_enclave_socket_path,
    _handle_gateway_enclave_rpc,
    _sha256,
)


MAX_FRAME_BYTES = 128 * 1024 * 1024
ROLES = {
    "gateway_autoresearch",
    "gateway_coordinator",
    "gateway_scoring",
}


def _prepare_measured_cgroup_boundary(
    original_prepare: Any,
) -> str:
    """Exercise candidate cgroup delegation without writing host cgroups."""

    with tempfile.TemporaryDirectory(
        prefix="leadpoet-rehearsal-cgroup-"
    ) as raw_tmp:
        root = Path(raw_tmp)
        cgroup_root = root / "sys/fs/cgroup"
        parent = cgroup_root / "nested/enclave"
        runtime = parent / "leadpoet-runtime"
        runtime.mkdir(parents=True)
        controllers = "cpu io memory pids"
        (parent / "cgroup.controllers").write_text(
            controllers + "\n", encoding="ascii"
        )
        (parent / "cgroup.procs").write_text("", encoding="ascii")
        (parent / "cgroup.subtree_control").write_text(
            controllers + "\n", encoding="ascii"
        )
        (runtime / "cgroup.procs").write_text("101\n", encoding="ascii")
        proc_cgroup = root / "proc-self-cgroup"
        proc_cgroup.write_text(
            "0::/nested/enclave/leadpoet-runtime\n",
            encoding="ascii",
        )

        def write_cgroup(path: Path, value: str) -> None:
            path.write_text(value.replace("+", ""), encoding="ascii")
            if path == parent / "cgroup.subtree_control":
                jobs = parent / "leadpoet-model"
                jobs.mkdir(exist_ok=True)
                (jobs / "cgroup.controllers").write_text(
                    controllers + "\n", encoding="ascii"
                )
                (jobs / "cgroup.subtree_control").touch()

        delegated = original_prepare(
            cgroup_root=cgroup_root,
            proc_self_cgroup_path=proc_cgroup,
            writer=write_cgroup,
        )
        jobs = parent / "leadpoet-model"
        expected_controllers = set(controllers.split())
        parent_enabled = set(
            (parent / "cgroup.subtree_control")
            .read_text(encoding="ascii")
            .split()
        )
        jobs_enabled = set(
            (jobs / "cgroup.subtree_control")
            .read_text(encoding="ascii")
            .split()
        )
        if (
            delegated != "leadpoet-model"
            or parent_enabled != expected_controllers
            or jobs_enabled != expected_controllers
        ):
            raise ValueError("measured model cgroup boundary differs")
    _external_event(
        "nitro_enclaves",
        "measured_runtime_surface",
        phase="model_sandbox_cgroup",
        delegated_parent=delegated,
        controller_set=sorted(expected_controllers),
    )
    return delegated


class _MeasuredRunscBoundary:
    """Strict adapter for the privileged gVisor execution boundary."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._active: dict[str, str] = {}

    @staticmethod
    def _environment(document: dict[str, Any]) -> dict[str, str]:
        values: dict[str, str] = {}
        for item in document["process"]["env"]:
            name, separator, value = str(item).partition("=")
            if not separator or name in values:
                raise ValueError("model sandbox process environment differs")
            values[name] = value
        return values

    def _verify_oci_document(
        self,
        *,
        document: dict[str, Any],
        sandbox_id: str,
    ) -> tuple[Path, Path]:
        process = document.get("process")
        linux = document.get("linux")
        root = document.get("root")
        mounts = document.get("mounts")
        if (
            set(document)
            != {
                "hostname",
                "linux",
                "mounts",
                "ociVersion",
                "process",
                "root",
            }
            or document.get("ociVersion") != "1.0.2"
            or document.get("hostname") != "leadpoet-model-sandbox"
            or not isinstance(process, dict)
            or not isinstance(linux, dict)
            or not isinstance(root, dict)
            or not isinstance(mounts, list)
        ):
            raise ValueError("model sandbox OCI document differs")
        expected_resources = {
            "memory": {"limit": self._config.memory_limit_bytes},
            "cpu": {
                "quota": self._config.cpu_quota,
                "period": self._config.cpu_period,
            },
            "pids": {"limit": self._config.pids_limit},
        }
        namespace_types = {
            str(item.get("type"))
            for item in linux.get("namespaces") or ()
            if isinstance(item, dict)
        }
        expected_mapping = [
            {
                "containerID": self._config.uid,
                "hostID": self._config.uid,
                "size": 1,
            }
        ]
        capabilities = process.get("capabilities") or {}
        if (
            root
            != {"path": str(self._config.rootfs_path), "readonly": True}
            or process.get("user")
            != {"uid": self._config.uid, "gid": self._config.gid}
            or process.get("cwd") != "/tmp"
            or process.get("noNewPrivileges") is not True
            or any(capabilities.get(name) for name in capabilities)
            or linux.get("resources") != expected_resources
            or linux.get("cgroupsPath")
            != "leadpoet-model/" + sandbox_id
            or namespace_types
            != {"pid", "ipc", "uts", "mount", "network", "user"}
            or linux.get("uidMappings") != expected_mapping
            or linux.get("gidMappings") != expected_mapping
        ):
            raise ValueError("model sandbox OCI isolation differs")
        seccomp = linux.get("seccomp") or {}
        socket_rules = [
            item
            for item in seccomp.get("syscalls") or ()
            if isinstance(item, dict) and "socket" in (item.get("names") or ())
        ]
        if (
            seccomp.get("defaultAction") != "SCMP_ACT_ALLOW"
            or len(socket_rules) != 1
            or socket_rules[0].get("action") != "SCMP_ACT_ERRNO"
            or socket_rules[0].get("args")
            != [{"index": 0, "value": int(socket.AF_UNIX), "op": "SCMP_CMP_NE"}]
        ):
            raise ValueError("model sandbox network seccomp differs")

        by_destination = {
            str(item.get("destination")): item
            for item in mounts
            if isinstance(item, dict)
        }
        source_mount = by_destination.get("/workspace/app") or {}
        broker_mount = by_destination.get("/run/leadpoet") or {}
        run_mount = by_destination.get("/run") or {}
        source = Path(str(source_mount.get("source") or ""))
        broker = Path(str(broker_mount.get("source") or ""))
        required_bind_options = {"rbind", "ro", "nosuid", "nodev"}
        if (
            source_mount.get("type") != "bind"
            or not required_bind_options.issubset(
                set(source_mount.get("options") or ())
            )
            or broker_mount.get("type") != "bind"
            or not (required_bind_options | {"noexec"}).issubset(
                set(broker_mount.get("options") or ())
            )
            or run_mount.get("type") != "tmpfs"
            or not source.is_dir()
            or not broker.is_dir()
            or (source / "self-test-token").read_text(encoding="utf-8")
            != "leadpoet-model-sandbox-self-test-v2\n"
        ):
            raise ValueError("model sandbox measured mounts differ")
        environment = self._environment(document)
        expected_environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "/app:/app/gateway/_attested_runtime:/workspace/app",
            "LEADPOET_SANDBOX_PROVIDER_SOCKET": (
                "/run/leadpoet/provider.sock"
            ),
        }
        args = process.get("args") or []
        script = str(args[2]) if len(args) == 3 else ""
        if (
            environment != expected_environment
            or args[:2] != [self._config.python_path, "-c"]
            or "gateway.tee.sandbox_http_shim_v2" not in script
            or "leadpoet_canonical" not in script
            or "leadpoet-model-sandbox-self-test-request-v2" not in script
            or "leadpoet-model-sandbox-self-test-response-v2" not in script
        ):
            raise ValueError("model sandbox startup self-test differs")
        return source, broker

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        argv = [str(item) for item in command]
        if len(argv) == 5 and argv[2:4] == ["delete", "--force"]:
            sandbox_id = argv[4]
            expected_root = self._active.pop(sandbox_id, None)
            if (
                argv[0] != str(self._config.runsc_path)
                or not argv[1].startswith("--root=")
                or expected_root != argv[1]
            ):
                raise ValueError("model sandbox cleanup identity differs")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        if len(argv) != 9:
            raise ValueError("model sandbox runsc operation differs")
        root_arg, bundle_arg, sandbox_id = argv[1], argv[7], argv[8]
        if (
            argv[0] != str(self._config.runsc_path)
            or not root_arg.startswith("--root=")
            or argv[2:7]
            != [
                "--rootless=false",
                "--network=none",
                "--host-uds=open",
                "--platform=ptrace",
                "run",
            ]
            or not bundle_arg.startswith("--bundle=")
            or not sandbox_id.startswith("lp-self-test-")
            or sandbox_id in self._active
            or kwargs.get("input") != ""
            or kwargs.get("text") is not True
            or kwargs.get("capture_output") is not True
            or kwargs.get("check") is not False
            or int(kwargs.get("timeout") or 0) != 60
        ):
            raise ValueError("model sandbox runsc launch differs")
        bundle = Path(bundle_arg.removeprefix("--bundle="))
        runsc_root = Path(root_arg.removeprefix("--root="))
        if not bundle.is_dir() or not runsc_root.is_dir():
            raise ValueError("model sandbox runsc paths differ")
        document = json.loads(
            (bundle / "config.json").read_text(encoding="utf-8")
        )
        _source, broker = self._verify_oci_document(
            document=document,
            sandbox_id=sandbox_id,
        )
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(10)
            client.connect(str(broker / "provider.sock"))
            client.sendall(
                b"leadpoet-model-sandbox-self-test-request-v2"
            )
            response = client.recv(128)
        finally:
            client.close()
        if response != b"leadpoet-model-sandbox-self-test-response-v2":
            raise ValueError("model sandbox broker round trip differs")
        self._active[sandbox_id] = root_arg
        output = json.dumps(
            {
                "schema_version": "leadpoet.model_sandbox_self_test.v2",
                "status": "passed",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        _external_event(
            "nitro_enclaves",
            "measured_runtime_surface",
            phase="model_sandbox_self_test",
            command_hash=_sha256(json.dumps(argv, separators=(",", ":"))),
            oci_config_hash=_sha256(
                json.dumps(document, sort_keys=True, separators=(",", ":"))
            ),
            broker_round_trip=True,
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(min(64 * 1024, length - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    if len(output) != length:
        raise ValueError("persistent gateway enclave frame is incomplete")
    return bytes(output)


def _handle(role: str, gateway_root: Path, body: bytes) -> dict[str, Any]:
    request = json.loads(body)
    if (
        not isinstance(request, dict)
        or set(request) != {"method", "params"}
        or not isinstance(request["params"], dict)
    ):
        raise ValueError("gateway enclave request fields differ")
    expected_merkle = gateway_root / "tee/merkle.py"
    if not expected_merkle.is_file():
        raise ValueError(
            "persistent gateway enclave candidate root became unavailable"
        )
    # A real role enclave has one immutable /app filesystem. Keep the local
    # process pointed at its candidate-backed equivalent even if candidate
    # behavior updates other measured environment values between RPCs.
    os.environ["GATEWAY_ROOT"] = str(gateway_root)
    method = str(request["method"])
    try:
        result = _handle_gateway_enclave_rpc(
            role,
            method,
            request["params"],
        )
        response = {"status": "success", "result": result}
        diagnostic = {
            "status": "ok",
            "error_type": "",
            "error_hash": "",
        }
    except Exception as exc:
        response = {"status": "error", "error": str(exc)}
        diagnostic = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error_hash": _sha256(str(exc)),
        }
    return {"response": response, "diagnostic": diagnostic}


def _prepare_candidate_role_root(role: str) -> Path:
    if role not in ROLES:
        raise ValueError("persistent gateway enclave role differs")
    if "gateway" in sys.modules:
        raise ValueError(
            "gateway modules loaded before candidate role isolation"
        )
    candidate = str(os.environ.get("REHEARSAL_CANDIDATE_SHA") or "")
    configured_source = os.environ.get("REHEARSAL_GATEWAY_CANDIDATE_ROOT")
    if not configured_source:
        raise ValueError("persistent gateway enclave source is unavailable")
    source = Path(configured_source).resolve()
    state_root = Path(
        os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state")
    )
    identity_source = (
        state_root / "gateway-enclave-build-identities" / f"{role}.json"
    )
    attested_runtime_source = state_root / "gateway-attested-runtime"
    release_input = json.loads(
        (state_root / "release-build-input.json").read_text(encoding="utf-8")
    )
    expected_role = release_input.get("gateway_roles", {}).get(role)
    if (
        release_input.get("commit_sha") != candidate
        or not isinstance(expected_role, dict)
        or not source.is_dir()
        or not identity_source.is_file()
        or (
            os.environ.get("REHEARSAL_SCOPE") == "exact"
            and not attested_runtime_source.is_dir()
        )
    ):
        raise ValueError("persistent gateway enclave fixture differs")

    role_parent = state_root / "gateway-enclave-runtimes" / role
    gateway_root = role_parent / "gateway"
    if role_parent.exists():
        raise ValueError("persistent gateway enclave role root already exists")
    role_parent.mkdir(parents=True)
    shutil.copytree(source, gateway_root)
    if attested_runtime_source.is_dir():
        shutil.rmtree(
            gateway_root / "_attested_runtime",
            ignore_errors=True,
        )
        shutil.copytree(
            attested_runtime_source,
            gateway_root / "_attested_runtime",
        )
    identity_target = (
        gateway_root
        / "_attested_runtime/gateway_enclave_build_identity.json"
    )
    identity_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(identity_source, identity_target)

    os.environ["GATEWAY_ROOT"] = str(gateway_root)
    if attested_runtime_source.is_dir():
        source_parent = source.parent
        controller_source = Path("/source").resolve()
        sys.path[:] = [
            item
            for item in sys.path
            if item
            and Path(item).resolve() not in {source_parent, controller_source}
        ]
        sys.path.insert(0, str(gateway_root / "_attested_runtime"))
    sys.path.insert(0, str(role_parent))
    importlib.invalidate_caches()
    from gateway.tee.build_identity import load_identity

    identity = load_identity(
        gateway_root=gateway_root,
        expected_role=role,
    )
    expected = {
        "commit_sha": candidate,
        "identity_hash": expected_role.get("build_identity_hash"),
        "dependency_lock_hash": expected_role.get(
            "dependency_lock_hash"
        ),
        "execution_manifest_hash": expected_role.get(
            "execution_manifest_hash"
        ),
        "topology_hash": expected_role.get("topology_hash"),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("persistent gateway enclave identity differs")
    module_path = Path(
        str(sys.modules["gateway.tee.build_identity"].__file__)
    ).resolve()
    if not module_path.is_relative_to(gateway_root):
        raise ValueError("persistent gateway enclave code root differs")
    return gateway_root


def _install_measured_runtime_boundary(gateway_root: Path) -> None:
    if os.environ.get("REHEARSAL_SCOPE") != "exact":
        return
    canonical_root = Path(
        os.environ.get(
            "REHEARSAL_GATEWAY_CANONICAL_APP_ROOT",
            "/app/gateway",
        )
    ).resolve()
    from gateway.tee import model_sandbox_v2
    from gateway.tee.sandbox_runtime_artifact import (
        EXPECTED_PYTHON_VERSION,
        build_rootfs_manifest,
        verify_runsc_artifact,
    )

    first_line = (
        gateway_root / "tee/Dockerfile.enclave"
    ).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith(
        f"FROM python:{EXPECTED_PYTHON_VERSION}-"
    ):
        raise ValueError("measured enclave Python base differs")
    for relative in (
        "tee/runsc-runtime.lock.json",
        "tee/requirements-scoring-py39.lock",
    ):
        if (
            (gateway_root / relative).read_bytes()
            != (canonical_root / relative).read_bytes()
        ):
            raise ValueError("canonical measured runtime input differs")
    runsc_path = Path("/usr/local/bin/runsc")
    verify_runsc_artifact(
        lock_path=canonical_root / "tee/runsc-runtime.lock.json",
        artifact_path=runsc_path,
    )
    expected_manifest = build_rootfs_manifest(
        lock_path=canonical_root / "tee/runsc-runtime.lock.json",
        requirements_lock_path=(
            canonical_root / "tee/requirements-scoring-py39.lock"
        ),
        python_version=EXPECTED_PYTHON_VERSION,
    )
    observed_manifest = json.loads(
        Path("/leadpoet-model-rootfs.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if observed_manifest != expected_manifest:
        raise ValueError("measured model rootfs manifest differs")
    if (
        model_sandbox_v2.DEFAULT_RUNSC_LOCK_PATH
        != Path("/app/gateway/tee/runsc-runtime.lock.json")
        or model_sandbox_v2.DEFAULT_REQUIREMENTS_LOCK_PATH
        != Path("/app/gateway/tee/requirements-scoring-py39.lock")
    ):
        raise ValueError("candidate measured runtime paths differ")

    production_prepare_cgroup = (
        model_sandbox_v2.prepare_model_sandbox_cgroup_v2
    )
    production_sandbox = model_sandbox_v2.RunscModelSandboxV2

    def prepare_rehearsal_cgroup() -> str:
        return _prepare_measured_cgroup_boundary(
            production_prepare_cgroup
        )

    class RehearsalRunscModelSandboxV2(production_sandbox):
        def __init__(
            self,
            *,
            config: Any,
            transport: Any,
            cgroup_parent: str,
        ) -> None:
            self._rehearsal_runsc_boundary = _MeasuredRunscBoundary(
                config
            )
            super().__init__(
                config=config,
                transport=transport,
                cgroup_parent=cgroup_parent,
                process_runner=self._rehearsal_runsc_boundary,
            )

    # Physical Nitro/gVisor execution is an explicit local boundary. Candidate
    # validation still consumes its real lock, binary hash, rootfs marker, and
    # canonical /app paths before the developer-runtime adapter is installed.
    model_sandbox_v2.platform.python_version = (
        lambda: EXPECTED_PYTHON_VERSION
    )
    model_sandbox_v2.prepare_model_sandbox_cgroup_v2 = (
        prepare_rehearsal_cgroup
    )
    model_sandbox_v2.RunscModelSandboxV2 = (
        RehearsalRunscModelSandboxV2
    )
    _external_event(
        "nitro_enclaves",
        "measured_runtime_surface",
        candidate_gateway_root=str(gateway_root),
        canonical_gateway_root=str(canonical_root),
        python_version=EXPECTED_PYTHON_VERSION,
        runsc_sha256=expected_manifest["runsc_sha256"],
        rootfs_manifest_hash=_sha256(
            json.dumps(
                expected_manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )


def main() -> int:
    role = str(os.environ.get("REHEARSAL_GATEWAY_ENCLAVE_ROLE") or "")
    if os.environ.get("REHEARSAL_COMPONENT") != "gateway":
        raise SystemExit("persistent gateway enclave requires gateway scope")
    gateway_root = _prepare_candidate_role_root(role)
    _install_measured_runtime_boundary(gateway_root)
    socket_path = _gateway_enclave_socket_path(role)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        socket_path.chmod(0o600)
        server.listen(64)
        socket_path.with_suffix(".ready").write_text(
            "ready\n",
            encoding="ascii",
        )
        while True:
            connection, _address = server.accept()
            with connection:
                try:
                    prefix = _recv_exact(connection, 4)
                    size = int.from_bytes(prefix, "big")
                    if size < 2 or size > MAX_FRAME_BYTES:
                        raise ValueError(
                            "persistent gateway enclave request size differs"
                        )
                    body = _recv_exact(connection, size)
                    outer = _handle(role, gateway_root, body)
                except Exception as exc:
                    outer = {
                        "response": {
                            "status": "error",
                            "error": str(exc),
                        },
                        "diagnostic": {
                            "status": "rejected",
                            "error_type": type(exc).__name__,
                            "error_hash": _sha256(str(exc)),
                        },
                    }
                encoded = json.dumps(
                    outer,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                connection.sendall(
                    len(encoded).to_bytes(4, "big") + encoded
                )
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        socket_path.with_suffix(".ready").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
