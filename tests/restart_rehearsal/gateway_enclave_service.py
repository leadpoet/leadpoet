#!/usr/bin/env python3.11
"""Persistent role-isolated process for real gateway enclave RPC handlers."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
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
    *,
    required_controllers: frozenset[str],
    control_files: dict[str, tuple[str, ...]],
) -> str:
    """Exercise the candidate against Nitro's measured cgroup-v1 layout."""

    with tempfile.TemporaryDirectory(
        prefix="leadpoet-rehearsal-cgroup-"
    ) as raw_tmp:
        root = Path(raw_tmp)
        cgroup_root = root / "sys/fs/cgroup"
        current_pid = str(os.getpid())
        proc_cgroup = root / "proc-self-cgroup"
        proc_lines = []
        for hierarchy, controller in enumerate(
            sorted(required_controllers),
            start=1,
        ):
            controller_root = cgroup_root / controller
            controller_root.mkdir(parents=True)
            (controller_root / "tasks").write_text(
                current_pid + "\n", encoding="ascii"
            )
            proc_lines.append(f"{hierarchy}:{controller}:/")
        proc_cgroup.write_text("\n".join(proc_lines) + "\n", encoding="ascii")

        delegated = original_prepare(
            cgroup_root=cgroup_root,
            proc_self_cgroup_path=proc_cgroup,
        )
        if delegated != "leadpoet-model":
            raise ValueError("measured model cgroup boundary differs")
        if any(
            (cgroup_root / controller / filename).exists()
            for controller in sorted(required_controllers)
            for filename in control_files[controller]
        ):
            raise ValueError(
                "Nitro controller root unexpectedly exposes child limits"
            )
    _external_event(
        "nitro_enclaves",
        "measured_runtime_surface",
        phase="model_sandbox_cgroup",
        delegated_parent=delegated,
        cgroup_layout="nitro_v1_controller_root",
        controller_set=sorted(required_controllers),
    )
    return delegated


class _MeasuredRunscBoundary:
    """Strict adapter for the privileged gVisor execution boundary."""

    def __init__(self, config: Any, *, metadata_bootstrap: str) -> None:
        self._config = config
        self._metadata_bootstrap = str(metadata_bootstrap)
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

    def _verified_source_path(self, source_visible: str) -> Path:
        visible = Path(source_visible)
        parts = visible.parts
        if (
            not visible.is_absolute()
            or len(parts) != 4
            or parts[1] != "leadpoet-model-sandboxes"
            or re.fullmatch(r"lp-job-[0-9a-f]{16}", parts[2]) is None
            or parts[3] != "source"
        ):
            raise ValueError("model sandbox source path differs")
        rootfs = Path(self._config.rootfs_path).resolve(strict=True)
        expected = rootfs.joinpath(*parts[1:])
        try:
            source = expected.resolve(strict=True)
            source.relative_to(rootfs)
        except (OSError, ValueError) as exc:
            raise ValueError("model sandbox source path differs") from exc
        if source != expected or expected.is_symlink() or not source.is_dir():
            raise ValueError("model sandbox source path differs")
        return source

    def _verify_common_oci_document(
        self,
        *,
        document: dict[str, Any],
        sandbox_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
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
        expected_resources: dict[str, Any] = {
            "memory": {"limit": self._config.memory_limit_bytes},
            "cpu": {
                "quota": self._config.cpu_quota,
                "period": self._config.cpu_period,
            },
            "pids": {"limit": self._config.pids_limit},
        }
        expected_namespaces = [
            {"type": item}
            for item in ("pid", "ipc", "uts", "mount", "network", "user")
        ]
        expected_uid_mapping = [
            {
                "containerID": 0,
                "hostID": 0,
                "size": 1,
            },
            {
                "containerID": self._config.uid,
                "hostID": self._config.uid,
                "size": 1,
            }
        ]
        expected_gid_mapping = [
            {
                "containerID": 0,
                "hostID": 0,
                "size": 1,
            },
            {
                "containerID": self._config.gid,
                "hostID": self._config.gid,
                "size": 1,
            },
        ]
        expected_capabilities = {
            name: []
            for name in (
                "bounding",
                "effective",
                "inheritable",
                "permitted",
                "ambient",
            )
        }
        expected_mounts = [
            {"destination": "/proc", "type": "proc", "source": "proc"},
            {
                "destination": "/tmp",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "mode=1777", "size=1073741824"],
            },
            {
                "destination": "/run",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": [
                    "nosuid",
                    "nodev",
                    "noexec",
                    "mode=755",
                    "size=1048576",
                ],
            },
        ]
        expected_masked_paths = [
            "/dev/log",
            "/dev/nsm",
            "/proc/acpi",
            "/proc/keys",
            "/proc/kcore",
            "/proc/latency_stats",
            "/proc/timer_list",
            "/proc/timer_stats",
            "/sys/firmware",
        ]
        expected_readonly_paths = [
            "/proc/asound",
            "/proc/bus",
            "/proc/fs",
            "/proc/irq",
            "/proc/sys",
            "/proc/sysrq-trigger",
        ]
        if (
            root
            != {"path": str(self._config.rootfs_path), "readonly": True}
            or set(process)
            != {
                "args",
                "capabilities",
                "cwd",
                "env",
                "noNewPrivileges",
                "rlimits",
                "terminal",
                "user",
            }
            or set(linux)
            != {
                "cgroupsPath",
                "gidMappings",
                "maskedPaths",
                "namespaces",
                "readonlyPaths",
                "resources",
                "seccomp",
                "uidMappings",
            }
            or process.get("terminal") is not False
            or process.get("user")
            != {"uid": self._config.uid, "gid": self._config.gid}
            or process.get("cwd") != "/tmp"
            or process.get("noNewPrivileges") is not True
            or process.get("capabilities") != expected_capabilities
            or process.get("rlimits")
            != [
                {"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024},
                {
                    "type": "RLIMIT_NPROC",
                    "hard": self._config.pids_limit,
                    "soft": self._config.pids_limit,
                },
            ]
            or mounts != expected_mounts
            or linux.get("resources") != expected_resources
            or linux.get("cgroupsPath")
            != "leadpoet-model/" + sandbox_id
            or linux.get("namespaces") != expected_namespaces
            or linux.get("uidMappings") != expected_uid_mapping
            or linux.get("gidMappings") != expected_gid_mapping
            or linux.get("maskedPaths") != expected_masked_paths
            or linux.get("readonlyPaths") != expected_readonly_paths
        ):
            raise ValueError("model sandbox OCI isolation differs")
        seccomp = linux.get("seccomp") or {}
        if (
            set(seccomp) != {"architectures", "defaultAction", "syscalls"}
            or seccomp.get("defaultAction") != "SCMP_ACT_ALLOW"
            or seccomp.get("architectures") != ["SCMP_ARCH_X86_64"]
            or seccomp.get("syscalls")
            != [
                {
                    "names": ["socket"],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                    "args": [
                        {
                            "index": 0,
                            "value": int(socket.AF_UNIX),
                            "op": "SCMP_CMP_NE",
                        }
                    ],
                },
                {
                    "names": [
                        "mount",
                        "pivot_root",
                        "ptrace",
                        "bpf",
                        "keyctl",
                        "perf_event_open",
                    ],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                },
            ]
        ):
            raise ValueError("model sandbox network seccomp differs")
        return process, mounts, self._environment(document)

    def _verify_oci_document(
        self,
        *,
        document: dict[str, Any],
        sandbox_id: str,
    ) -> tuple[Path, Path]:
        process, mounts, environment = self._verify_common_oci_document(
            document=document,
            sandbox_id=sandbox_id,
        )

        by_destination = {
            str(item.get("destination")): item
            for item in mounts
            if isinstance(item, dict)
        }
        run_mount = by_destination.get("/run") or {}
        source_visible = str(environment.get("LEADPOET_MODEL_SOURCE_ROOT") or "")
        socket_visible = str(
            environment.get("LEADPOET_SANDBOX_PROVIDER_SOCKET") or ""
        )
        rootfs = Path(self._config.rootfs_path)
        source = self._verified_source_path(source_visible)
        socket_path = Path(socket_visible)
        if (
            socket_path.parts
            != (*Path(source_visible).parts[:-1], "broker", "provider.sock")
        ):
            raise ValueError("model sandbox provider socket path differs")
        broker_candidate = rootfs.resolve(strict=True).joinpath(
            *socket_path.parts[1:-1]
        )
        try:
            broker = broker_candidate.resolve(strict=True)
            broker.relative_to(rootfs.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ValueError("model sandbox provider socket path differs") from exc
        if (
            run_mount.get("type") != "tmpfs"
            or any(item.get("type") == "bind" for item in mounts)
            or broker != broker_candidate
            or not broker.is_dir()
            or (source / "self-test-token").read_text(encoding="utf-8")
            != "leadpoet-model-sandbox-self-test-v2\n"
        ):
            raise ValueError("model sandbox measured rootfs inputs differ")
        expected_environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": (
                "/app:/app/gateway/_attested_runtime:" + source_visible
            ),
            "LEADPOET_MODEL_SOURCE_ROOT": source_visible,
            "LEADPOET_SANDBOX_PROVIDER_SOCKET": socket_visible,
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

    def _verify_metadata_oci_document(
        self,
        *,
        document: dict[str, Any],
        sandbox_id: str,
    ) -> None:
        process, _mounts, environment = self._verify_common_oci_document(
            document=document,
            sandbox_id=sandbox_id,
        )
        source_visible = str(environment.get("LEADPOET_MODEL_SOURCE_ROOT") or "")
        source = self._verified_source_path(source_visible)
        expected_environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": (
                "/app:/app/gateway/_attested_runtime:" + source_visible
            ),
            "LEADPOET_MODEL_SOURCE_ROOT": source_visible,
        }
        args = process.get("args") or []
        if (
            environment != expected_environment
            or args
            != [
                self._config.python_path,
                "-I",
                "-B",
                "-c",
                self._metadata_bootstrap,
                *args[5:7],
            ]
            or len(args) != 7
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", str(args[5]))
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(args[6]))
        ):
            raise ValueError("model sandbox metadata OCI isolation differs")

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
        if argv[2:7] == [
            "--rootless=false",
            "--network=none",
            "--host-uds=none",
            "--platform=ptrace",
            "run",
        ]:
            bundle = Path(bundle_arg.removeprefix("--bundle="))
            document = json.loads(
                (bundle / "config.json").read_text(encoding="utf-8")
            )
            self._verify_metadata_oci_document(
                document=document,
                sandbox_id=sandbox_id,
            )
            environment = self._environment(document)
            process = document.get("process") or {}
            process_args = process.get("args") or []
            payload = json.loads(str(kwargs.get("input") or ""))
            if (
                argv[0] != str(self._config.runsc_path)
                or not root_arg.startswith("--root=")
                or not bundle_arg.startswith("--bundle=")
                or not sandbox_id.startswith("lp-")
                or sandbox_id in self._active
                or kwargs.get("text") is not True
                or kwargs.get("capture_output") is not True
                or kwargs.get("check") is not False
                or int(kwargs.get("timeout") or 0) != 120
                or set(payload) != {"observation_plan"}
                or "LEADPOET_SANDBOX_PROVIDER_SOCKET" in environment
                or "RESEARCH_LAB_PROVIDER_COST_SCOPE" in environment
                or process_args[1:4] != ["-I", "-B", "-c"]
                or document.get("root", {}).get("readonly") is not True
                or not any(
                    item.get("destination") == "/run"
                    and item.get("type") == "tmpfs"
                    for item in document.get("mounts") or ()
                )
            ):
                raise ValueError("model sandbox metadata boundary differs")
            self._active[sandbox_id] = root_arg
            return SimpleNamespace(
                returncode=125,
                stdout="",
                stderr="rehearsal_metadata_requires_external_runsc_evidence",
            )
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
            production_prepare_cgroup,
            required_controllers=model_sandbox_v2.MODEL_SANDBOX_REQUIRED_CONTROLLERS,
            control_files=model_sandbox_v2.MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES,
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
                config,
                metadata_bootstrap=model_sandbox_v2._MEASURED_METADATA_BOOTSTRAP,
            )
            super().__init__(
                config=config,
                transport=transport,
                cgroup_parent=cgroup_parent,
                process_runner=self._rehearsal_runsc_boundary,
                metadata_process_runner=self._rehearsal_runsc_boundary,
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
                try:
                    connection.sendall(
                        len(encoded).to_bytes(4, "big") + encoded
                    )
                except (
                    BrokenPipeError,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ):
                    # A restarted worker can disappear after the enclave has
                    # accepted its request. Isolate that client lifecycle so
                    # the persistent role remains available to the replacement
                    # worker; all request execution failures still fail closed
                    # through the structured error response above.
                    continue
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        socket_path.with_suffix(".ready").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
