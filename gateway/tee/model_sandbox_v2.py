"""Fresh, fail-closed gVisor sandbox for V2 private/candidate model jobs."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.sandbox_provider_socket_v2 import SandboxProviderSocketServerV2
from gateway.tee.sandbox_http_shim_v2 import EVIDENCE_MISS_SENTINEL
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import (
    source_add_placeholder_environment_v2,
    source_add_runtime_retry_hashes_v2,
    validate_source_add_runtime_catalog_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from research_lab.eval import (
    PrivateModelRuntimeError,
    PrivateModelArtifactManifest,
    ensure_private_model_outputs,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import (
    _DOCKER_ADAPTER_BOOTSTRAP,
    _DOCKER_METADATA_BOOTSTRAP,
    _raise_on_empty_provider_error,
    SOURCING_MODEL_MAX_RUNTIME_CAP_SECONDS,
    canonicalize_private_model_icp,
    context_with_runtime_options,
    parse_incontainer_trace_lines,
    parse_sourcing_runtime_lines,
    strip_incontainer_trace_lines,
    validate_sourcing_runtime_receipt,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    build_evidence_cache_from_trace_entries,
    icp_evidence_cache_key,
    merge_evidence_caches,
)
from research_lab.eval.snapshot_store import (
    SNAPSHOT_MISS_SENTINEL,
    SnapshotMiss,
    container_replay_env,
    dev_replay_bootstrap,
)


logger = logging.getLogger(__name__)


MODEL_SANDBOX_REQUEST_SCHEMA_VERSION = "leadpoet.model_sandbox_request.v2"
MODEL_SANDBOX_RESULT_SCHEMA_VERSION = "leadpoet.model_sandbox_result.v2"
PROVIDER_EVIDENCE_TAPE_INPUT_SCHEMA_VERSION = (
    "leadpoet.provider_evidence_tape_input.v2"
)
ROOTFS_MANIFEST_NAME = "leadpoet-model-rootfs.manifest.json"
DEFAULT_RUNSC_LOCK_PATH = Path("/app/gateway/tee/runsc-runtime.lock.json")
DEFAULT_REQUIREMENTS_LOCK_PATH = Path(
    "/app/gateway/tee/requirements-scoring-py39.lock"
)
MAX_MODEL_INPUT_BYTES = 16 * 1024 * 1024
MODEL_SANDBOX_TIMEOUT_SECONDS = 900
MODEL_SANDBOX_TIMEOUT_GRACE_SECONDS = 3
MAX_MODEL_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_PROVIDER_EVIDENCE_CACHE_BYTES = 32 * 1024 * 1024
MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT = "/app/gateway/_attested_runtime"
MODEL_SANDBOX_VISIBLE_ROOT = "/leadpoet-model-sandboxes"
MODEL_SANDBOX_SOURCE_DIRECTORY = "source"
MODEL_SANDBOX_BROKER_DIRECTORY = "broker"
MODEL_SANDBOX_SELF_TEST_SCHEMA_VERSION = "leadpoet.model_sandbox_self_test.v2"
MODEL_SANDBOX_SELF_TEST_TIMEOUT_SECONDS = 60
MODEL_SANDBOX_CGROUP_ROOT = Path("/sys/fs/cgroup")
MODEL_SANDBOX_RUNTIME_CGROUP_NAME = "leadpoet-runtime"
MODEL_SANDBOX_JOB_CGROUP_NAME = "leadpoet-model"
MODEL_SANDBOX_REQUIRED_CONTROLLERS = frozenset({"cpu", "memory", "pids"})
MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES = {
    "cpu": ("cpu.cfs_quota_us", "cpu.cfs_period_us"),
    "memory": ("memory.limit_in_bytes",),
    "pids": ("pids.max",),
}
MODEL_SANDBOX_PYTHONPATH = ":".join(
    (
        "/app",
        MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT,
    )
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_CALLABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "DEEPLINE_API_KEY",
        "EXA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
    }
)
_MEASURED_CREDENTIAL_PLACEHOLDER = "leadpoet-coordinator-managed-v2"
_MODEL_SANDBOX_CGROUP_LOCK = Lock()


class ModelSandboxV2Error(RuntimeError):
    """A model runtime, bundle, provider path, or output failed validation."""


def _runsc_failure_evidence(stderr: str) -> tuple[str, str]:
    """Return a fixed launcher code and hash without exposing sandbox output."""

    sanitized = strip_incontainer_trace_lines(str(stderr or ""))
    lowered = sanitized.lower()
    patterns = (
        (
            "no such file or directory: '/workspace/app/self-test-token'",
            "runsc_source_mount_missing",
        ),
        (
            "rootfs-visible source differs",
            "runsc_source_staging_missing",
        ),
        ("gofer: fork/exec /proc/self/exe: invalid argument", "runsc_gofer_exec"),
        ("cannot create gofer process", "runsc_gofer_create"),
        ("cannot set up cgroup", "runsc_cgroup_setup"),
        ("configuring cgroup", "runsc_cgroup_setup"),
        ("error setting up root fs", "runsc_rootfs_setup"),
        ("error setting up fs", "runsc_mount_setup"),
        ("failure to resolve mounts", "runsc_mount_resolve"),
        ("failed to chroot", "runsc_gofer_chroot"),
        ("permission denied", "runsc_permission_denied"),
    )
    failure_code = "runsc_nonzero"
    for marker, candidate in patterns:
        if marker in lowered:
            failure_code = candidate
            break
    return failure_code, sha256_bytes(sanitized.encode("utf-8"))


def _runsc_run_command(
    *,
    config: "RunscSandboxConfigV2",
    runsc_root: Path,
    bundle: Path,
    sandbox_id: str,
    host_uds: bool,
) -> list[str]:
    command = [
        str(config.runsc_path),
        "--root=%s" % runsc_root,
        # The measured enclave starts runsc as root and the OCI document
        # supplies an explicit 65534:65534 user namespace. gVisor rootless
        # mode instead remaps the caller to container root and is incompatible
        # with that keep-ID contract.
        "--rootless=false",
        "--network=none",
    ]
    if host_uds:
        command.append("--host-uds=open")
    return [
        *command,
        "--platform=ptrace",
        "run",
        "--bundle=%s" % bundle,
        sandbox_id,
    ]


def _write_cgroup_value(path: Path, value: str) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(value)


def _pid_is_direct_cgroup_member(
    cgroup: Path,
    pid: int,
    *,
    membership_file: str = "cgroup.procs",
) -> bool:
    if membership_file not in {"cgroup.procs", "tasks"}:
        raise ModelSandboxV2Error("model sandbox cgroup membership file is invalid")
    try:
        members = (cgroup / membership_file).read_text(encoding="ascii").split()
    except OSError as exc:
        raise ModelSandboxV2Error(
            "model sandbox cgroup membership is unavailable"
        ) from exc
    return str(pid) in members


def _normalized_cgroup_relative_path(value: str) -> str:
    if not value.startswith("/") or "\x00" in value:
        raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
    relative = Path(value).as_posix().lstrip("/")
    if ".." in Path(relative).parts:
        raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
    return relative


def _current_cgroup_path(
    proc_self_cgroup_path: Path,
    *,
    cgroup_root: Path,
    pid: Optional[int] = None,
) -> str:
    try:
        lines = proc_self_cgroup_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ModelSandboxV2Error("model sandbox cgroup identity is unavailable") from exc
    unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(unified) > 1:
        raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
    if unified and unified[0].startswith("/"):
        return _normalized_cgroup_relative_path(unified[0])
    if unified and unified[0]:
        raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")

    # Docker-to-EIF boots a private cgroup-v2 hierarchy but may omit the
    # conventional ``0::/`` identity line from procfs. In that environment the
    # enclave service is a direct member of the namespace root. Verify that
    # membership from cgroupfs itself instead of treating the absent proc line
    # as evidence that cgroup-v2 is unavailable.
    root = cgroup_root.resolve()
    if (root / "cgroup.controllers").is_file() and _pid_is_direct_cgroup_member(
        root, int(os.getpid() if pid is None else pid)
    ):
        return ""
    raise ModelSandboxV2Error("model sandbox cgroup identity is unavailable")


def _current_cgroup_v1_paths(proc_self_cgroup_path: Path) -> Dict[str, str]:
    try:
        lines = proc_self_cgroup_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ModelSandboxV2Error("model sandbox cgroup identity is unavailable") from exc

    controller_paths: Dict[str, str] = {}
    for line in lines:
        if not line:
            continue
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit():
            raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
        hierarchy, raw_controllers, raw_path = fields
        if not raw_controllers:
            continue
        controllers = raw_controllers.split(",")
        if (
            int(hierarchy) <= 0
            or not controllers
            or any(
                not re.fullmatch(r"[A-Za-z0-9_.=-]+", controller)
                for controller in controllers
            )
        ):
            raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
        relative = _normalized_cgroup_relative_path(raw_path)
        for controller in controllers:
            if controller in controller_paths:
                raise ModelSandboxV2Error("model sandbox cgroup identity is invalid")
            controller_paths[controller] = relative
    if not MODEL_SANDBOX_REQUIRED_CONTROLLERS.issubset(controller_paths):
        raise ModelSandboxV2Error(
            "model sandbox required cgroup controllers are unavailable"
        )
    return controller_paths


def _prepare_model_sandbox_cgroup_v1(
    *,
    cgroup_root: Path,
    proc_self_cgroup_path: Path,
    pid: Optional[int] = None,
) -> str:
    controller_paths = _current_cgroup_v1_paths(proc_self_cgroup_path)
    current_pid = int(os.getpid() if pid is None else pid)
    for controller in sorted(MODEL_SANDBOX_REQUIRED_CONTROLLERS):
        mount_entry = cgroup_root / controller
        mount = mount_entry.resolve()
        if mount_entry.is_symlink() or mount.parent != cgroup_root:
            raise ModelSandboxV2Error(
                "model sandbox cgroup v1 hierarchy is invalid"
            )
        current = (mount / controller_paths[controller]).resolve()
        try:
            current.relative_to(mount)
        except ValueError as exc:
            raise ModelSandboxV2Error(
                "model sandbox cgroup identity is invalid"
            ) from exc
        if not mount.is_dir() or not current.is_dir():
            raise ModelSandboxV2Error(
                "model sandbox cgroup v1 hierarchy is unavailable"
            )
        if not _pid_is_direct_cgroup_member(
            current,
            current_pid,
            membership_file="tasks",
        ):
            raise ModelSandboxV2Error("model sandbox cgroup membership differs")
    # On cgroup v1, rootful runsc resolves this relative path beneath each
    # controller path of the current process, creates a unique job child, and
    # applies the OCI CPU, memory, and PID limits there. Nitro's enclave init
    # places the service in each controller root, where Linux does not expose
    # every child-only limit file. The mandatory runsc startup self-test is the
    # fail-closed proof that those files exist on the created child and accept
    # the measured OCI limits.
    return MODEL_SANDBOX_JOB_CGROUP_NAME


def prepare_model_sandbox_cgroup_v2(
    *,
    cgroup_root: Path = MODEL_SANDBOX_CGROUP_ROOT,
    proc_self_cgroup_path: Path = Path("/proc/self/cgroup"),
    writer: Callable[[Path, str], None] = _write_cgroup_value,
) -> str:
    """Prepare the measured cgroup hierarchy for rootful gVisor jobs."""

    with _MODEL_SANDBOX_CGROUP_LOCK:
        root = cgroup_root.resolve()
        if not (root / "cgroup.controllers").is_file():
            return _prepare_model_sandbox_cgroup_v1(
                cgroup_root=root,
                proc_self_cgroup_path=proc_self_cgroup_path,
            )
        current_relative = _current_cgroup_path(
            proc_self_cgroup_path,
            cgroup_root=root,
        )
        current = root / current_relative
        if current.name == MODEL_SANDBOX_RUNTIME_CGROUP_NAME:
            parent = current.parent
        else:
            parent = current
        runtime = parent / MODEL_SANDBOX_RUNTIME_CGROUP_NAME
        jobs = parent / MODEL_SANDBOX_JOB_CGROUP_NAME
        required_paths = (
            parent / "cgroup.controllers",
            parent / "cgroup.procs",
            parent / "cgroup.subtree_control",
        )
        if not parent.is_dir() or any(not path.is_file() for path in required_paths):
            raise ModelSandboxV2Error("model sandbox cgroup v2 hierarchy is unavailable")
        available = set(
            (parent / "cgroup.controllers").read_text(encoding="ascii").split()
        )
        if not MODEL_SANDBOX_REQUIRED_CONTROLLERS.issubset(available):
            raise ModelSandboxV2Error(
                "model sandbox required cgroup controllers are unavailable"
            )
        runtime.mkdir(mode=0o755, exist_ok=True)
        if current == parent:
            for _attempt in range(3):
                direct_pids = [
                    item
                    for item in (parent / "cgroup.procs")
                    .read_text(encoding="ascii")
                    .split()
                    if item.isdigit()
                ]
                if not direct_pids:
                    break
                for pid in direct_pids:
                    try:
                        writer(runtime / "cgroup.procs", pid)
                    except OSError as exc:
                        if not Path("/proc").joinpath(pid).exists():
                            continue
                        raise ModelSandboxV2Error(
                            "model sandbox runtime cgroup migration failed"
                        ) from exc
            if (parent / "cgroup.procs").read_text(encoding="ascii").split():
                raise ModelSandboxV2Error(
                    "model sandbox cgroup parent still owns processes"
                )
        elif current != runtime:
            raise ModelSandboxV2Error("model sandbox runtime cgroup differs")

        # runsc's cgroup-v2 implementation enables every controller exposed by
        # the namespace root while creating the OCI cgroup path. Delegate that
        # same set here so its later write cannot fail on an undelegated
        # controller; the required subset remains the resource-limit contract.
        controller_value = " ".join("+" + item for item in sorted(available))
        try:
            writer(parent / "cgroup.subtree_control", controller_value)
        except OSError as exc:
            raise ModelSandboxV2Error(
                "model sandbox cgroup controller delegation failed"
            ) from exc
        enabled = set(
            (parent / "cgroup.subtree_control")
            .read_text(encoding="ascii")
            .replace("+", "")
            .split()
        )
        if not available.issubset(enabled):
            raise ModelSandboxV2Error(
                "model sandbox cgroup controller delegation differs"
            )
        jobs.mkdir(mode=0o755, exist_ok=True)
        job_available = set(
            (jobs / "cgroup.controllers").read_text(encoding="ascii").split()
        )
        if not available.issubset(job_available):
            raise ModelSandboxV2Error(
                "model sandbox job cgroup controllers differ"
            )
        try:
            writer(jobs / "cgroup.subtree_control", controller_value)
        except OSError as exc:
            raise ModelSandboxV2Error(
                "model sandbox job cgroup delegation failed"
            ) from exc
        job_enabled = set(
            (jobs / "cgroup.subtree_control")
            .read_text(encoding="ascii")
            .replace("+", "")
            .split()
        )
        if not available.issubset(job_enabled):
            raise ModelSandboxV2Error(
                "model sandbox job cgroup delegation differs"
            )
        # Relative OCI cgroup paths are resolved from the parent of runsc's
        # current cgroup. The service now lives in the sibling runtime cgroup,
        # so return the job subtree relative to their shared parent.
        return jobs.relative_to(parent).as_posix()


def model_sandbox_job_cgroup_path(parent: str, sandbox_id: str) -> str:
    if (
        not parent
        or parent.startswith("/")
        or ".." in Path(parent).parts
        or not re.fullmatch(r"lp-[a-z0-9-]{1,120}", str(sandbox_id or ""))
    ):
        raise ModelSandboxV2Error("model sandbox job cgroup identity is invalid")
    return str(Path(parent) / sandbox_id)


def model_source_import_bootstrap(source_root: Optional[str] = None) -> str:
    """Activate model-owned packages after trusted enclave imports are pinned."""

    normalized_root = str(source_root or "").rstrip("/")
    if source_root is not None and (
        not normalized_root.startswith("/") or "\x00" in normalized_root
    ):
        raise ModelSandboxV2Error("model sandbox source root is invalid")
    root_expression = (
        repr(normalized_root)
        if source_root is not None
        else "_lp_os.environ['LEADPOET_MODEL_SOURCE_ROOT']"
    )
    root_guard = (
        ""
        if source_root is not None
        else """
if not _lp_source_root.startswith('/leadpoet-model-sandboxes/'):
    raise RuntimeError('model sandbox source root differs')
"""
    )
    return f"""
import gateway as _lp_gateway
import gateway.tee as _lp_trusted_gateway_tee
import leadpoet_canonical as _lp_trusted_canonical
import os as _lp_os
import sys as _lp_sys

_lp_source_root = {root_expression}
{root_guard}
_lp_source_gateway = _lp_source_root + '/gateway'
while _lp_source_root in _lp_sys.path:
    _lp_sys.path.remove(_lp_source_root)
_lp_sys.path.insert(0, _lp_source_root)
if _lp_source_gateway not in _lp_gateway.__path__:
    _lp_gateway.__path__.insert(0, _lp_source_gateway)
"""


def provider_evidence_tape_input_root(cache_ref: str, cache_hash: str) -> str:
    normalized_ref = str(cache_ref or "").lower()
    normalized_hash = str(cache_hash or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_ref):
        raise ModelSandboxV2Error("provider evidence cache ref is invalid")
    if not _HASH_RE.fullmatch(normalized_hash):
        raise ModelSandboxV2Error("provider evidence cache hash is invalid")
    return sha256_json(
        {
            "schema_version": PROVIDER_EVIDENCE_TAPE_INPUT_SCHEMA_VERSION,
            "provider_evidence_cache_ref": normalized_ref,
            "provider_evidence_cache_hash": normalized_hash,
        }
    )


@dataclass(frozen=True)
class RunscSandboxConfigV2:
    runsc_path: Path
    runsc_sha256: str
    rootfs_path: Path
    rootfs_manifest_hash: str
    python_path: str = "/usr/local/bin/python3"
    uid: int = 65534
    gid: int = 65534
    memory_limit_bytes: int = 8 * 1024 * 1024 * 1024
    cpu_quota: int = 200000
    cpu_period: int = 100000
    pids_limit: int = 512

    @classmethod
    def from_measured_runtime(
        cls,
        *,
        lock_path: Path = DEFAULT_RUNSC_LOCK_PATH,
        requirements_lock_path: Path = DEFAULT_REQUIREMENTS_LOCK_PATH,
        rootfs_path: Path = Path("/"),
        runsc_path: Optional[Path] = None,
        python_version: Optional[str] = None,
    ) -> "RunscSandboxConfigV2":
        from gateway.tee.sandbox_runtime_artifact import (
            build_rootfs_manifest,
            load_runsc_lock,
        )

        lock = load_runsc_lock(lock_path)
        observed_python = str(python_version or platform.python_version())
        expected_marker = build_rootfs_manifest(
            lock_path=lock_path,
            requirements_lock_path=requirements_lock_path,
            python_version=observed_python,
        )
        marker_path = rootfs_path / ROOTFS_MANIFEST_NAME
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelSandboxV2Error(
                "measured model rootfs marker is unavailable"
            ) from exc
        if marker != expected_marker:
            raise ModelSandboxV2Error("measured model rootfs marker differs")
        config = cls(
            runsc_path=runsc_path or Path(str(lock["install_path"])),
            runsc_sha256=str(lock["sha256"]),
            rootfs_path=rootfs_path,
            rootfs_manifest_hash=sha256_bytes(marker_path.read_bytes()),
        )
        config.validate()
        return config

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunscSandboxConfigV2":
        fields = {
            "runsc_path",
            "runsc_sha256",
            "rootfs_path",
            "rootfs_manifest_hash",
            "python_path",
            "uid",
            "gid",
            "memory_limit_bytes",
            "cpu_quota",
            "cpu_period",
            "pids_limit",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ModelSandboxV2Error("runsc sandbox configuration fields are invalid")
        return cls(
            runsc_path=Path(str(value["runsc_path"])),
            runsc_sha256=str(value["runsc_sha256"]),
            rootfs_path=Path(str(value["rootfs_path"])),
            rootfs_manifest_hash=str(value["rootfs_manifest_hash"]),
            python_path=str(value["python_path"]),
            uid=int(value["uid"]),
            gid=int(value["gid"]),
            memory_limit_bytes=int(value["memory_limit_bytes"]),
            cpu_quota=int(value["cpu_quota"]),
            cpu_period=int(value["cpu_period"]),
            pids_limit=int(value["pids_limit"]),
        )

    def document(self) -> Dict[str, Any]:
        return {
            "runsc_path": str(self.runsc_path),
            "runsc_sha256": self.runsc_sha256,
            "rootfs_path": str(self.rootfs_path),
            "rootfs_manifest_hash": self.rootfs_manifest_hash,
            "python_path": self.python_path,
            "uid": self.uid,
            "gid": self.gid,
            "memory_limit_bytes": self.memory_limit_bytes,
            "cpu_quota": self.cpu_quota,
            "cpu_period": self.cpu_period,
            "pids_limit": self.pids_limit,
        }

    def validate(self) -> None:
        for value, field in (
            (self.runsc_sha256, "runsc_sha256"),
            (self.rootfs_manifest_hash, "rootfs_manifest_hash"),
        ):
            if not _HASH_RE.fullmatch(str(value or "").lower()):
                raise ModelSandboxV2Error("%s is invalid" % field)
        if not self.runsc_path.is_file() or not os.access(self.runsc_path, os.X_OK):
            raise ModelSandboxV2Error("measured runsc executable is unavailable")
        if sha256_bytes(self.runsc_path.read_bytes()) != self.runsc_sha256:
            raise ModelSandboxV2Error("runsc executable hash differs")
        if not self.rootfs_path.is_dir():
            raise ModelSandboxV2Error("measured model rootfs is unavailable")
        marker = self.rootfs_path / ROOTFS_MANIFEST_NAME
        if not marker.is_file() or sha256_bytes(marker.read_bytes()) != self.rootfs_manifest_hash:
            raise ModelSandboxV2Error("model rootfs manifest hash differs")
        if not self.python_path.startswith("/") or ".." in Path(self.python_path).parts:
            raise ModelSandboxV2Error("sandbox Python path is invalid")
        if self.uid <= 0 or self.gid <= 0:
            raise ModelSandboxV2Error("sandbox identity must be unprivileged")
        if (
            self.memory_limit_bytes < 256 * 1024 * 1024
            or self.cpu_quota <= 0
            or self.cpu_period <= 0
            or self.pids_limit < 16
        ):
            raise ModelSandboxV2Error("sandbox resource limits are invalid")


def _request(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "model_kind",
        "operation",
        "artifact",
        "source_bundle",
        "module_name",
        "callable_name",
        "input",
        "environment",
        "provider_evidence_cache",
        "provider_evidence_cache_ref",
        "provider_evidence_mode",
        "provider_snapshot_bundle",
        "provider_snapshot_tree_hash",
        "provider_snapshot_manifest_hash",
        "provider_cost_scope",
        "provider_cost_cap_microusd",
        "provider_call_cap",
        "provider_runtime_catalog",
        "provider_catalog_evidence",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelSandboxV2Error("model sandbox request fields are invalid")
    if value.get("schema_version") != MODEL_SANDBOX_REQUEST_SCHEMA_VERSION:
        raise ModelSandboxV2Error("model sandbox request schema is invalid")
    if value.get("model_kind") not in {"private", "candidate"}:
        raise ModelSandboxV2Error("model sandbox kind is invalid")
    if value.get("operation") not in {"metadata", "run_icp"}:
        raise ModelSandboxV2Error("model sandbox operation is invalid")
    module_name = str(value.get("module_name") or "")
    callable_name = str(value.get("callable_name") or "")
    if not _MODULE_RE.fullmatch(module_name) or not _CALLABLE_RE.fullmatch(callable_name):
        raise ModelSandboxV2Error("model adapter entrypoint is invalid")
    environment = value.get("environment")
    if not isinstance(environment, Mapping):
        raise ModelSandboxV2Error("model sandbox environment must be an object")
    normalized_environment = {}
    for name, item in environment.items():
        normalized_name = str(name)
        normalized_value = str(item)
        if (
            not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", normalized_name)
            or "\x00" in normalized_value
            or len(normalized_value.encode("utf-8")) > 16 * 1024
        ):
            raise ModelSandboxV2Error("model sandbox environment value is invalid")
        lowered = normalized_name.lower()
        if any(marker in lowered for marker in ("secret", "password", "private_key")):
            raise ModelSandboxV2Error("model sandbox environment contains a secret field")
        normalized_environment[normalized_name] = normalized_value
    if any(name in _CREDENTIAL_ENV_NAMES for name in normalized_environment):
        raise ModelSandboxV2Error(
            "model sandbox environment contains parent-supplied credentials"
        )
    evidence_cache = value.get("provider_evidence_cache")
    if not isinstance(evidence_cache, Mapping):
        raise ModelSandboxV2Error("provider evidence cache must be an object")
    normalized_evidence_cache = dict(evidence_cache)
    encoded_evidence_cache = canonical_json(normalized_evidence_cache).encode("utf-8")
    if len(encoded_evidence_cache) > MAX_PROVIDER_EVIDENCE_CACHE_BYTES:
        raise ModelSandboxV2Error("provider evidence cache exceeds limit")
    if normalized_evidence_cache and (
        normalized_evidence_cache.get("schema_version") != "1.1"
        or not isinstance(normalized_evidence_cache.get("entries"), Mapping)
    ):
        raise ModelSandboxV2Error("provider evidence cache is invalid")
    cache_ref = str(value.get("provider_evidence_cache_ref") or "").lower()
    evidence_mode = str(value.get("provider_evidence_mode") or "").strip().lower()
    if evidence_mode not in {"live", "cache_live", "record", "frozen"}:
        raise ModelSandboxV2Error("provider evidence mode is invalid")
    if evidence_mode == "frozen" and not normalized_evidence_cache:
        if not value.get("provider_snapshot_bundle"):
            raise ModelSandboxV2Error("frozen provider evidence sources are empty")
    snapshot_bundle = value.get("provider_snapshot_bundle")
    snapshot_tree_hash = str(value.get("provider_snapshot_tree_hash") or "")
    snapshot_manifest_hash = str(
        value.get("provider_snapshot_manifest_hash") or ""
    )
    if snapshot_bundle:
        if (
            not isinstance(snapshot_bundle, Mapping)
            or not _HASH_RE.fullmatch(snapshot_tree_hash)
            or not _HASH_RE.fullmatch(snapshot_manifest_hash)
            or snapshot_bundle.get("source_tree_hash") != snapshot_tree_hash
        ):
            raise ModelSandboxV2Error("provider snapshot commitment is invalid")
        normalized_snapshot_bundle: Dict[str, Any] = dict(snapshot_bundle)
    else:
        if snapshot_tree_hash or snapshot_manifest_hash:
            raise ModelSandboxV2Error("provider snapshot commitment is incomplete")
        normalized_snapshot_bundle = {}
    if value.get("operation") == "run_icp":
        raw_input = value.get("input")
        if not isinstance(raw_input, Mapping) or not isinstance(
            raw_input.get("icp"), Mapping
        ):
            raise ModelSandboxV2Error("model run input fields are invalid")
        expected_cache_ref = icp_evidence_cache_key(
            canonicalize_private_model_icp(raw_input["icp"])
        )
        if cache_ref != expected_cache_ref:
            raise ModelSandboxV2Error("provider evidence cache ref differs from ICP")
    elif cache_ref:
        raise ModelSandboxV2Error("metadata request has provider evidence cache ref")
    encoded_input = canonical_json(value.get("input")).encode("utf-8")
    if len(encoded_input) > MAX_MODEL_INPUT_BYTES:
        raise ModelSandboxV2Error("model sandbox input exceeds limit")
    scope = str(value.get("provider_cost_scope") or "").lower()
    if not _HASH_RE.fullmatch(scope):
        raise ModelSandboxV2Error("provider cost scope is invalid")
    cost_cap_microusd = value.get("provider_cost_cap_microusd")
    provider_call_cap = value.get("provider_call_cap")
    if (
        isinstance(cost_cap_microusd, bool)
        or not isinstance(cost_cap_microusd, int)
        or cost_cap_microusd < 0
        or cost_cap_microusd > 500_000
        or isinstance(provider_call_cap, bool)
        or not isinstance(provider_call_cap, int)
        or provider_call_cap < 0
        or provider_call_cap > 32
    ):
        raise ModelSandboxV2Error("provider tree evaluation caps are invalid")
    if evidence_mode in {"record", "frozen"}:
        if value.get("model_kind") == "candidate":
            if cost_cap_microusd <= 0 or provider_call_cap <= 0:
                raise ModelSandboxV2Error(
                    "provider tree evaluation caps are required"
                )
        elif cost_cap_microusd or provider_call_cap:
            raise ModelSandboxV2Error(
                "provider tree evaluation caps are out of scope"
            )
    elif cost_cap_microusd or provider_call_cap:
        raise ModelSandboxV2Error("provider tree evaluation caps are out of scope")
    try:
        provider_runtime_catalog = validate_source_add_runtime_catalog_v2(
            value.get("provider_runtime_catalog") or {}
        )
    except Exception as exc:
        raise ModelSandboxV2Error(
            "model sandbox provider runtime catalog is invalid"
        ) from exc
    catalog_evidence = value.get("provider_catalog_evidence")
    if not isinstance(catalog_evidence, Mapping) or set(catalog_evidence) != {
        "result",
        "root_receipt_hash",
    }:
        raise ModelSandboxV2Error(
            "model sandbox provider catalog evidence is invalid"
        )
    catalog_result = catalog_evidence.get("result")
    root_receipt_hash = str(catalog_evidence.get("root_receipt_hash") or "")
    provisioned_sources = (
        catalog_result.get("provisioned_sources")
        if isinstance(catalog_result, Mapping)
        else None
    )
    private_registry_rows = (
        catalog_result.get("private_registry_rows")
        if isinstance(catalog_result, Mapping)
        else None
    )
    if (
        not isinstance(catalog_result, Mapping)
        or not _HASH_RE.fullmatch(root_receipt_hash)
        or catalog_result.get("schema_version")
        != "leadpoet.source_add_catalog_snapshot.v2"
        or not isinstance(provisioned_sources, list)
        or any(not isinstance(item, Mapping) for item in provisioned_sources)
        or not isinstance(private_registry_rows, list)
        or any(not isinstance(item, Mapping) for item in private_registry_rows)
        or catalog_result.get("provisioned_sources_hash")
        != sha256_json([dict(item) for item in provisioned_sources])
        or catalog_result.get("private_registry_rows_hash")
        != sha256_json([dict(item) for item in private_registry_rows])
        or catalog_result.get("runtime_catalog_hash")
        != provider_runtime_catalog["catalog_hash"]
        or catalog_result.get("runtime_catalog") != provider_runtime_catalog
    ):
        raise ModelSandboxV2Error(
            "model sandbox provider catalog commitment differs"
        )
    return {
        **dict(value),
        "artifact": dict(value["artifact"]),
        "source_bundle": dict(value["source_bundle"]),
        "module_name": module_name,
        "callable_name": callable_name,
        "environment": dict(sorted(normalized_environment.items())),
        "provider_evidence_cache": normalized_evidence_cache,
        "provider_evidence_cache_ref": cache_ref,
        "provider_evidence_mode": evidence_mode,
        "provider_snapshot_bundle": normalized_snapshot_bundle,
        "provider_snapshot_tree_hash": snapshot_tree_hash,
        "provider_snapshot_manifest_hash": snapshot_manifest_hash,
        "provider_cost_scope": scope,
        "provider_cost_cap_microusd": cost_cap_microusd,
        "provider_call_cap": provider_call_cap,
        "provider_runtime_catalog": provider_runtime_catalog,
        "provider_catalog_evidence": {
            "result": dict(catalog_result),
            "root_receipt_hash": root_receipt_hash,
        },
    }


def _model_sandbox_process_timeout_seconds(value: Mapping[str, Any]) -> int:
    """Derive the bounded runsc deadline from the committed model allocation."""

    if value.get("operation") != "run_icp":
        return MODEL_SANDBOX_TIMEOUT_SECONDS
    raw_input = value.get("input")
    context = raw_input.get("context") if isinstance(raw_input, Mapping) else None
    options = context.get("runtime_options") if isinstance(context, Mapping) else None
    if not isinstance(options, Mapping):
        return MODEL_SANDBOX_TIMEOUT_SECONDS
    try:
        runtime_cap = float(options.get("runtime_cap_seconds"))
    except (TypeError, ValueError) as exc:
        raise ModelSandboxV2Error(
            "model sandbox runtime allocation is invalid"
        ) from exc
    if (
        not math.isfinite(runtime_cap)
        or runtime_cap < 10.0
        or runtime_cap > SOURCING_MODEL_MAX_RUNTIME_CAP_SECONDS
    ):
        raise ModelSandboxV2Error("model sandbox runtime allocation is invalid")
    return int(math.ceil(runtime_cap)) + MODEL_SANDBOX_TIMEOUT_GRACE_SECONDS


def _normalize_source_permissions(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _sandbox_visible_parent(config: RunscSandboxConfigV2) -> tuple[Path, Path]:
    configured_rootfs = Path(config.rootfs_path)
    try:
        configured_rootfs_stat = configured_rootfs.lstat()
        rootfs = configured_rootfs.resolve(strict=True)
        rootfs_stat = rootfs.lstat()
    except OSError as exc:
        raise ModelSandboxV2Error("model sandbox rootfs is unavailable") from exc
    if (
        not rootfs.is_dir()
        or configured_rootfs.is_symlink()
        or configured_rootfs_stat.st_dev != rootfs_stat.st_dev
        or configured_rootfs_stat.st_ino != rootfs_stat.st_ino
    ):
        raise ModelSandboxV2Error("model sandbox rootfs identity is invalid")
    parent = rootfs / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
    try:
        parent_stat = parent.lstat()
    except OSError as exc:
        raise ModelSandboxV2Error(
            "model sandbox visible root is unavailable"
        ) from exc
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent_stat.st_uid != rootfs_stat.st_uid
        or parent_stat.st_gid != rootfs_stat.st_gid
        or parent_stat.st_mode & 0o777 != 0o711
    ):
        raise ModelSandboxV2Error(
            "model sandbox visible root identity is invalid"
        )
    return rootfs, parent


@contextmanager
def _sandbox_visible_workspace(
    config: RunscSandboxConfigV2,
) -> Iterator[Path]:
    rootfs, parent = _sandbox_visible_parent(config)
    workspace: Path | None = None
    for _attempt in range(4):
        candidate = parent / ("lp-job-" + secrets.token_hex(8))
        try:
            candidate.mkdir(mode=0o711)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ModelSandboxV2Error(
                "model sandbox visible workspace is unavailable"
            ) from exc
        candidate.chmod(0o711)
        workspace = candidate
        break
    if workspace is None:
        raise ModelSandboxV2Error(
            "model sandbox visible workspace is unavailable"
        )
    workspace_stat = workspace.lstat()
    rootfs_stat = rootfs.lstat()
    if (
        workspace.is_symlink()
        or workspace.parent != parent
        or workspace_stat.st_uid != rootfs_stat.st_uid
        or workspace_stat.st_gid != rootfs_stat.st_gid
        or workspace_stat.st_mode & 0o777 != 0o711
    ):
        shutil.rmtree(workspace, ignore_errors=True)
        raise ModelSandboxV2Error(
            "model sandbox visible workspace identity is invalid"
        )
    try:
        yield workspace
    finally:
        try:
            for item in sorted(workspace.rglob("*"), reverse=True):
                if item.is_symlink():
                    continue
                if item.is_dir():
                    item.chmod(0o700)
                elif item.is_file():
                    item.chmod(0o600)
            workspace.chmod(0o700)
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "model_sandbox_visible_workspace_cleanup_failed "
                "workspace_hash=%s error_type=%s",
                sha256_bytes(str(workspace).encode("utf-8")),
                type(exc).__name__,
            )


def _sandbox_visible_path(
    config: RunscSandboxConfigV2,
    path: Path,
    *,
    field: str,
) -> str:
    rootfs, parent = _sandbox_visible_parent(config)
    try:
        resolved = Path(path).resolve(strict=True)
        resolved.relative_to(parent)
        relative = resolved.relative_to(rootfs)
    except (OSError, ValueError) as exc:
        raise ModelSandboxV2Error(
            "model sandbox %s is outside the visible root" % field
        ) from exc
    if Path(path).is_symlink() or not resolved.is_dir():
        raise ModelSandboxV2Error(
            "model sandbox %s identity is invalid" % field
        )
    return "/" + relative.as_posix()


def _copy_readonly_visible_tree(source: Path, destination: Path) -> None:
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_dir():
        raise ModelSandboxV2Error("sandbox read-only tree is invalid")
    for item in source_path.rglob("*"):
        if item.is_symlink():
            raise ModelSandboxV2Error("sandbox read-only tree contains a symlink")
    try:
        shutil.copytree(source_path, destination)
    except OSError as exc:
        raise ModelSandboxV2Error("sandbox read-only tree copy failed") from exc
    _normalize_source_permissions(destination)


def _oci_config(
    *,
    config: RunscSandboxConfigV2,
    source_root: Path,
    broker_root: Optional[Path],
    process_args: list[str],
    environment: Mapping[str, str],
    cgroups_path: Optional[str] = None,
) -> Dict[str, Any]:
    source_path = _sandbox_visible_path(
        config,
        source_root,
        field="source root",
    )
    process_env = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        **dict(environment),
        "LEADPOET_MODEL_SOURCE_ROOT": source_path,
        "PYTHONPATH": MODEL_SANDBOX_PYTHONPATH + ":" + source_path,
    }
    if broker_root is not None:
        broker_path = Path(broker_root)
        try:
            broker_stat = broker_path.lstat()
        except OSError as exc:
            raise ModelSandboxV2Error(
                "model sandbox provider broker is unavailable"
            ) from exc
        if (
            not broker_path.is_dir()
            or broker_path.is_symlink()
            or broker_stat.st_uid != config.uid
            or broker_stat.st_gid != config.gid
        ):
            raise ModelSandboxV2Error(
                "model sandbox provider broker identity is invalid"
            )
        broker_visible_path = _sandbox_visible_path(
            config,
            broker_path,
            field="provider broker",
        )
        process_env["LEADPOET_SANDBOX_PROVIDER_SOCKET"] = (
            broker_visible_path + "/provider.sock"
        )
    mounts = [
        {"destination": "/proc", "type": "proc", "source": "proc"},
        {
            "destination": "/tmp",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "nodev", "mode=1777", "size=1073741824"],
        },
    ]
    if broker_root is not None:
        # Hide ordinary runtime sockets. The only reachable broker lives under
        # this job's unpredictable, rootfs-visible workspace.
        mounts.append(
            {
                "destination": "/run",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "noexec", "mode=755", "size=1048576"],
            }
        )
    linux = {
        "namespaces": [
            {"type": "pid"},
            {"type": "ipc"},
            {"type": "uts"},
            {"type": "mount"},
            {"type": "network"},
            {"type": "user"},
        ],
        # Rootful runsc starts its measured gofer as namespace root while the
        # untrusted workload remains the explicit nobody identity below.  Both
        # identities must be mapped or the gofer cannot re-exec after applying
        # its setup capabilities (fork/exec /proc/self/exe returns EINVAL).
        "uidMappings": [
            {"containerID": 0, "hostID": 0, "size": 1},
            {"containerID": config.uid, "hostID": config.uid, "size": 1},
        ],
        "gidMappings": [
            {"containerID": 0, "hostID": 0, "size": 1},
            {"containerID": config.gid, "hostID": config.gid, "size": 1},
        ],
        "resources": {
            "memory": {"limit": config.memory_limit_bytes},
            "cpu": {"quota": config.cpu_quota, "period": config.cpu_period},
            "pids": {"limit": config.pids_limit},
        },
        "maskedPaths": [
            "/dev/log",
            "/dev/nsm",
            "/proc/acpi",
            "/proc/keys",
            "/proc/kcore",
            "/proc/latency_stats",
            "/proc/timer_list",
            "/proc/timer_stats",
            "/sys/firmware",
        ],
        "readonlyPaths": [
            "/proc/asound",
            "/proc/bus",
            "/proc/fs",
            "/proc/irq",
            "/proc/sys",
            "/proc/sysrq-trigger",
        ],
        "seccomp": {
            "defaultAction": "SCMP_ACT_ALLOW",
            "architectures": ["SCMP_ARCH_X86_64"],
            "syscalls": [
                {
                    "names": ["socket"],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                    "args": [
                        {
                            "index": 0,
                            "value": int(socket_af_unix()),
                            "op": "SCMP_CMP_NE",
                        }
                    ],
                },
                {
                    "names": ["mount", "pivot_root", "ptrace", "bpf", "keyctl", "perf_event_open"],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                },
            ],
        },
    }
    if cgroups_path is not None:
        normalized_cgroups_path = str(cgroups_path or "")
        if (
            not normalized_cgroups_path
            or normalized_cgroups_path.startswith("/")
            or ".." in Path(normalized_cgroups_path).parts
        ):
            raise ModelSandboxV2Error("model sandbox job cgroup path is invalid")
        linux["cgroupsPath"] = normalized_cgroups_path
    return {
        "ociVersion": "1.0.2",
        "process": {
            "terminal": False,
            "user": {"uid": config.uid, "gid": config.gid},
            "args": process_args,
            "env": ["%s=%s" % item for item in sorted(process_env.items())],
            # A model may ship a compatibility ``gateway`` package. Starting
            # from its source directory would shadow the measured
            # ``gateway.tee`` package before the trusted HTTP shim is loaded.
            "cwd": "/tmp",
            "capabilities": {
                "bounding": [],
                "effective": [],
                "inheritable": [],
                "permitted": [],
                "ambient": [],
            },
            "rlimits": [
                {"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024},
                {"type": "RLIMIT_NPROC", "hard": config.pids_limit, "soft": config.pids_limit},
            ],
            "noNewPrivileges": True,
        },
        "root": {"path": str(config.rootfs_path), "readonly": True},
        "hostname": "leadpoet-model-sandbox",
        "mounts": mounts,
        "linux": linux,
    }


def socket_af_unix() -> int:
    import socket

    return int(socket.AF_UNIX)


def _completed_process_runner(command: list[str], **kwargs: Any):
    return subprocess.run(command, **kwargs)


class RunscModelSandboxV2:
    def __init__(
        self,
        *,
        config: RunscSandboxConfigV2,
        transport: BrokeredProviderTransportV2,
        cgroup_parent: str,
        process_runner: Callable[..., Any] = _completed_process_runner,
        utc_day_supplier: Callable[[], str] = lambda: time.strftime(
            "%Y-%m-%d", time.gmtime()
        ),
    ) -> None:
        config.validate()
        model_sandbox_job_cgroup_path(cgroup_parent, "lp-constructor-check")
        self.config = config
        self.cgroup_parent = cgroup_parent
        self._transport = transport
        self._process_runner = process_runner
        self._utc_day_supplier = utc_day_supplier

    @staticmethod
    def _create_provider_scope_v2(
        transport: BrokeredProviderTransportV2,
        *,
        job_id: str,
        purpose: str,
        retry_policy_hashes: Mapping[str, str],
        terminal_sink: Callable[[Mapping[str, Any]], None],
        artifact_sink: Callable[[str], None],
        dynamic_provider_catalog: Mapping[str, Any],
    ) -> Any:
        """Bind model-owned fallback behavior to complete measured terminals."""

        return transport.create_scope(
            job_id=job_id,
            purpose=purpose,
            logical_operation_id=job_id,
            retry_policy_hashes={
                **dict(retry_policy_hashes),
                **source_add_runtime_retry_hashes_v2(dynamic_provider_catalog),
            },
            terminal_sink=terminal_sink,
            artifact_sink=artifact_sink,
            allow_transport_failures=True,
            dynamic_provider_catalog=dynamic_provider_catalog,
        )

    def self_test(self) -> Dict[str, Any]:
        """Exercise the measured launcher and broker boundary before job intake."""

        request_token = b"leadpoet-model-sandbox-self-test-request-v2"
        response_token = b"leadpoet-model-sandbox-self-test-response-v2"
        expected = {
            "schema_version": MODEL_SANDBOX_SELF_TEST_SCHEMA_VERSION,
            "status": "passed",
        }
        with tempfile.TemporaryDirectory(
            prefix="lp-model-self-test-", dir="/tmp"
        ) as tmp, _sandbox_visible_workspace(self.config) as visible_root:
            tmp_root = Path(tmp)
            source_root = visible_root / MODEL_SANDBOX_SOURCE_DIRECTORY
            source_root.mkdir(mode=0o755)
            probe_path = source_root / "self-test-token"
            probe_path.write_text("leadpoet-model-sandbox-self-test-v2\n", encoding="utf-8")
            _normalize_source_permissions(source_root)

            broker_root = visible_root / MODEL_SANDBOX_BROKER_DIRECTORY
            broker_root.mkdir(mode=0o700)
            try:
                os.chown(broker_root, self.config.uid, self.config.gid)
            except OSError as exc:
                raise ModelSandboxV2Error(
                    "model sandbox self-test broker identity is unavailable"
                ) from exc
            socket_path = broker_root / "provider.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.settimeout(MODEL_SANDBOX_SELF_TEST_TIMEOUT_SECONDS)
            listener.bind(str(socket_path))
            listener.listen(1)
            try:
                os.chown(socket_path, self.config.uid, self.config.gid)
                socket_path.chmod(0o600)
            except OSError as exc:
                listener.close()
                raise ModelSandboxV2Error(
                    "model sandbox self-test socket identity is unavailable"
                ) from exc

            served = Event()
            server_errors: list[str] = []

            def serve_once() -> None:
                try:
                    connection, _ = listener.accept()
                    with connection:
                        received = b""
                        while len(received) < len(request_token):
                            chunk = connection.recv(len(request_token) - len(received))
                            if not chunk:
                                break
                            received += chunk
                        if received != request_token:
                            server_errors.append("request_mismatch")
                            return
                        connection.sendall(response_token)
                        served.set()
                except (OSError, TimeoutError):
                    server_errors.append("socket_unavailable")

            server_thread = Thread(
                target=serve_once,
                name="model-sandbox-self-test-broker",
                daemon=True,
            )

            bundle = tmp_root / "bundle"
            bundle.mkdir(mode=0o700)
            runsc_root = tmp_root / "runsc"
            runsc_root.mkdir(mode=0o700)
            sandbox_id = "lp-self-test-%s" % secrets.token_hex(8)
            script = """
import json
import os
from pathlib import Path
import socket
import gateway.tee.sandbox_http_shim_v2
import leadpoet_canonical

source_root = Path(os.environ['LEADPOET_MODEL_SOURCE_ROOT'])
if (source_root / 'self-test-token').read_text(encoding='utf-8') != 'leadpoet-model-sandbox-self-test-v2\\n':
    raise RuntimeError('rootfs-visible source differs')
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    client.settimeout(10)
    client.connect(os.environ['LEADPOET_SANDBOX_PROVIDER_SOCKET'])
    client.sendall(b'leadpoet-model-sandbox-self-test-request-v2')
    response = client.recv(128)
finally:
    client.close()
if response != b'leadpoet-model-sandbox-self-test-response-v2':
    raise RuntimeError('broker response differs')
print(json.dumps({'schema_version': 'leadpoet.model_sandbox_self_test.v2', 'status': 'passed'}, sort_keys=True, separators=(',', ':')))
"""
            config_doc = _oci_config(
                config=self.config,
                source_root=source_root,
                broker_root=broker_root,
                process_args=[self.config.python_path, "-c", script],
                environment={},
                cgroups_path=model_sandbox_job_cgroup_path(
                    self.cgroup_parent, sandbox_id
                ),
            )
            (bundle / "config.json").write_text(
                canonical_json(config_doc), encoding="utf-8"
            )
            command = _runsc_run_command(
                config=self.config,
                runsc_root=runsc_root,
                bundle=bundle,
                sandbox_id=sandbox_id,
                host_uds=True,
            )
            completed = None
            server_thread.start()
            try:
                completed = self._process_runner(
                    command,
                    input="",
                    text=True,
                    capture_output=True,
                    timeout=MODEL_SANDBOX_SELF_TEST_TIMEOUT_SECONDS,
                    env={
                        "HOME": str(tmp_root),
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                    },
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelSandboxV2Error(
                    "model sandbox self-test timed out"
                ) from exc
            finally:
                try:
                    self._process_runner(
                        [
                            str(self.config.runsc_path),
                            "--root=%s" % runsc_root,
                            "delete",
                            "--force",
                            sandbox_id,
                        ],
                        text=True,
                        capture_output=True,
                        timeout=30,
                        env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                        check=False,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "model_sandbox_self_test_runsc_cleanup_failed "
                        "sandbox_id=%s error_type=%s",
                        sandbox_id,
                        type(cleanup_exc).__name__,
                    )
                listener.close()
                server_thread.join(timeout=5)

            if completed is None or int(completed.returncode) != 0:
                stderr = "" if completed is None else str(completed.stderr or "")
                failure_code, stderr_hash = _runsc_failure_evidence(stderr)
                returncode = "unknown" if completed is None else str(completed.returncode)
                raise ModelSandboxV2Error(
                    "model sandbox self-test failed code=%s returncode=%s stderr_hash=%s"
                    % (failure_code, returncode, stderr_hash)
                )
            if not served.is_set() or server_errors:
                raise ModelSandboxV2Error(
                    "model sandbox self-test failed code=broker_round_trip"
                )
            try:
                result = json.loads(str(completed.stdout or ""))
            except json.JSONDecodeError as exc:
                raise ModelSandboxV2Error(
                    "model sandbox self-test output is invalid"
                ) from exc
            if result != expected:
                raise ModelSandboxV2Error(
                    "model sandbox self-test result differs"
                )
            return expected

    def execute(
        self,
        request: Mapping[str, Any],
        *,
        job_id: str,
        purpose: str,
        retry_policy_hashes: Mapping[str, str],
        terminal_sink: Callable[[Mapping[str, Any]], None],
        artifact_sink: Callable[[str], None],
    ) -> Dict[str, Any]:
        value = _request(request)
        artifact = PrivateModelArtifactManifest.from_mapping(value["artifact"])
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise ModelSandboxV2Error("model artifact is invalid: " + "; ".join(errors))
        # AF_UNIX paths are limited to roughly 108 bytes on Linux. The measured
        # short parent leaves sufficient room for an unpredictable job path.
        with tempfile.TemporaryDirectory(
            prefix="lp-model-v2-", dir="/tmp"
        ) as tmp, _sandbox_visible_workspace(self.config) as visible_root:
            tmp_root = Path(tmp)
            source_root = visible_root / MODEL_SANDBOX_SOURCE_DIRECTORY
            source_evidence = extract_source_bundle_v2(
                value["source_bundle"],
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            _normalize_source_permissions(source_root)
            provider_snapshot_root: Path | None = None
            provider_snapshot_archive_hash = sha256_json({})
            if value["provider_snapshot_bundle"]:
                provider_snapshot_root = visible_root / "provider-snapshot"
                snapshot_evidence = extract_source_bundle_v2(
                    value["provider_snapshot_bundle"],
                    destination=provider_snapshot_root,
                    expected_source_tree_hash=value[
                        "provider_snapshot_tree_hash"
                    ],
                )
                snapshot_store = ProviderSnapshotStore(
                    str(provider_snapshot_root),
                    mode=MODE_REPLAY,
                )
                manifest = snapshot_store.load_manifest()
                verification = snapshot_store.verify_manifest(manifest)
                if (
                    manifest is None
                    or not verification.get("passed")
                    or manifest.get("manifest_hash")
                    != value["provider_snapshot_manifest_hash"]
                ):
                    raise ModelSandboxV2Error(
                        "provider snapshot manifest verification failed"
                    )
                provider_snapshot_archive_hash = str(
                    snapshot_evidence["archive_sha256"]
                )
                _normalize_source_permissions(provider_snapshot_root)
            broker_root = visible_root / MODEL_SANDBOX_BROKER_DIRECTORY
            try:
                broker_root.mkdir(mode=0o700)
                os.chown(broker_root, self.config.uid, self.config.gid)
            except OSError as exc:
                raise ModelSandboxV2Error(
                    "model sandbox provider broker identity is unavailable"
                ) from exc
            provider_scope = self._create_provider_scope_v2(
                self._transport,
                job_id=job_id,
                purpose=purpose,
                retry_policy_hashes=retry_policy_hashes,
                terminal_sink=terminal_sink,
                artifact_sink=artifact_sink,
                dynamic_provider_catalog=value["provider_runtime_catalog"],
            )
            server = SandboxProviderSocketServerV2(
                socket_path=broker_root / "provider.sock",
                transport=self._transport,
                execution_scope=provider_scope,
            )
            server.start()
            try:
                try:
                    os.chown(
                        server.socket_path,
                        self.config.uid,
                        self.config.gid,
                    )
                    server.socket_path.chmod(0o600)
                except OSError as exc:
                    raise ModelSandboxV2Error(
                        "model sandbox provider socket identity is unavailable"
                    ) from exc
                result, trace_entries = self._run(
                    value,
                    artifact=artifact,
                    source_root=source_root,
                    broker_root=broker_root,
                    tmp_root=tmp_root,
                    job_id=job_id,
                    provider_snapshot_root=provider_snapshot_root,
                )
            finally:
                server.close()
            provider_scope.assert_accepted_result_is_complete()
        output_hash = sha256_json(result)
        generated_evidence_cache = {}
        if value["operation"] == "run_icp" and value["provider_evidence_mode"] == "record":
            utc_day = str(self._utc_day_supplier() or "")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", utc_day):
                raise ModelSandboxV2Error("provider evidence cache UTC day is invalid")
            existing_entries = dict(
                dict(value["provider_evidence_cache"]).get("entries") or {}
            )
            generated_evidence_cache = {
                "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                "rolling_window_hash": "",
                "icp_ref": value["provider_evidence_cache_ref"],
                "utc_day": utc_day,
                "entries": merge_evidence_caches(
                    existing_entries,
                    build_evidence_cache_from_trace_entries(trace_entries),
                ),
            }
        return {
            "schema_version": MODEL_SANDBOX_RESULT_SCHEMA_VERSION,
            "model_kind": value["model_kind"],
            "operation": value["operation"],
            "model_artifact_hash": artifact.model_artifact_hash,
            "model_manifest_hash": artifact.manifest_hash,
            "compatibility_image_digest": artifact.image_digest,
            "source_bundle_hash": source_evidence["archive_sha256"],
            "runtime_config_hash": sha256_json(self.config.document()),
            "input_hash": sha256_json(value["input"]),
            "provider_evidence_cache_hash": sha256_json(
                value["provider_evidence_cache"]
            ),
            "provider_evidence_cache_ref": value["provider_evidence_cache_ref"],
            "provider_evidence_mode": value["provider_evidence_mode"],
            "provider_snapshot_archive_hash": provider_snapshot_archive_hash,
            "provider_snapshot_tree_hash": (
                value["provider_snapshot_tree_hash"] or sha256_json({})
            ),
            "provider_snapshot_manifest_hash": (
                value["provider_snapshot_manifest_hash"] or sha256_json({})
            ),
            "provider_cost_cap_microusd": value["provider_cost_cap_microusd"],
            "provider_call_cap": value["provider_call_cap"],
            "provider_runtime_catalog_hash": value[
                "provider_runtime_catalog"
            ]["catalog_hash"],
            "generated_provider_evidence_cache_hash": sha256_json(
                generated_evidence_cache
            ),
            "trace_entries_hash": sha256_json(trace_entries),
            "output_hash": output_hash,
            "output": result,
            "trace_entries": trace_entries,
            "generated_provider_evidence_cache": generated_evidence_cache,
        }

    def execute_dev_replay(
        self,
        *,
        artifact_doc: Mapping[str, Any],
        source_bundle: Mapping[str, Any],
        snapshot_root: Path,
        module_name: str,
        callable_name: str,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
        environment: Mapping[str, str],
        credential_env_names: Sequence[str],
        miss_policy: str,
        timeout_seconds: int,
        job_id: str,
    ) -> list[Mapping[str, Any]]:
        """Run one frozen-snapshot dev ICP with no live provider channel."""
        artifact = PrivateModelArtifactManifest.from_mapping(artifact_doc)
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise ModelSandboxV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        normalized_module = str(module_name or "")
        normalized_callable = str(callable_name or "")
        if not _MODULE_RE.fullmatch(normalized_module) or not _CALLABLE_RE.fullmatch(
            normalized_callable
        ):
            raise ModelSandboxV2Error("model adapter entrypoint is invalid")
        normalized_timeout = int(timeout_seconds)
        if normalized_timeout < 10:
            raise ModelSandboxV2Error("dev replay timeout is outside limit")
        normalized_environment: dict[str, str] = {}
        for name, item in environment.items():
            normalized_name = str(name)
            normalized_value = str(item)
            if (
                not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", normalized_name)
                or normalized_name in _CREDENTIAL_ENV_NAMES
                or "\x00" in normalized_value
                or len(normalized_value.encode("utf-8")) > 16 * 1024
            ):
                raise ModelSandboxV2Error("dev replay environment is invalid")
            normalized_environment[normalized_name] = normalized_value
        normalized_credential_names = sorted(
            {str(name) for name in credential_env_names}
        )
        if any(name not in _CREDENTIAL_ENV_NAMES for name in normalized_credential_names):
            raise ModelSandboxV2Error("dev replay credential name is invalid")
        if not Path(snapshot_root).is_dir():
            raise ModelSandboxV2Error("dev replay snapshot root is unavailable")
        stdin_payload = {
            "icp": canonicalize_private_model_icp(icp),
            "context": context_with_runtime_options(
                context,
                outer_timeout_seconds=normalized_timeout,
            ),
        }
        if len(canonical_json(stdin_payload).encode("utf-8")) > MAX_MODEL_INPUT_BYTES:
            raise ModelSandboxV2Error("dev replay input exceeds limit")

        with tempfile.TemporaryDirectory(
            prefix="lp-dev-replay-v2-", dir="/tmp"
        ) as tmp, _sandbox_visible_workspace(self.config) as visible_root:
            tmp_root = Path(tmp)
            source_root = visible_root / MODEL_SANDBOX_SOURCE_DIRECTORY
            extract_source_bundle_v2(
                source_bundle,
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            _normalize_source_permissions(source_root)
            visible_snapshot_root = visible_root / "dev-snapshots"
            _copy_readonly_visible_tree(
                Path(snapshot_root),
                visible_snapshot_root,
            )
            return self._run_dev_replay(
                source_root=source_root,
                module_name=normalized_module,
                callable_name=normalized_callable,
                stdin_payload=stdin_payload,
                environment={
                    **normalized_environment,
                    **{
                        name: _MEASURED_CREDENTIAL_PLACEHOLDER
                        for name in normalized_credential_names
                    },
                    **container_replay_env(
                        _sandbox_visible_path(
                            self.config,
                            visible_snapshot_root,
                            field="dev snapshot root",
                        ),
                        miss_policy=miss_policy,
                    ),
                },
                timeout_seconds=normalized_timeout,
                tmp_root=tmp_root,
                job_id=job_id,
            )

    def execute_dev_provider_replay(
        self,
        *,
        artifact_doc: Mapping[str, Any],
        source_bundle: Mapping[str, Any],
        module_name: str,
        callable_name: str,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
        environment: Mapping[str, str],
        credential_env_names: Sequence[str],
        provider_evidence_cache: Mapping[str, Any],
        snapshot_root: Path | None,
        timeout_seconds: int,
        job_id: str,
    ) -> list[Mapping[str, Any]]:
        """Replay one ICP from a frozen measured provider-evidence overlay."""

        artifact = PrivateModelArtifactManifest.from_mapping(artifact_doc)
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise ModelSandboxV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        normalized_module = str(module_name or "")
        normalized_callable = str(callable_name or "")
        if not _MODULE_RE.fullmatch(normalized_module) or not _CALLABLE_RE.fullmatch(
            normalized_callable
        ):
            raise ModelSandboxV2Error("model adapter entrypoint is invalid")
        normalized_timeout = int(timeout_seconds)
        if normalized_timeout < 10:
            raise ModelSandboxV2Error("dev provider replay timeout is outside limit")
        cache = dict(provider_evidence_cache)
        if (
            cache.get("schema_version") != EVIDENCE_CACHE_SCHEMA_VERSION
            or not isinstance(cache.get("entries"), Mapping)
        ):
            raise ModelSandboxV2Error("dev provider evidence cache is invalid")
        normalized_environment: dict[str, str] = {}
        for name, item in environment.items():
            normalized_name = str(name)
            normalized_value = str(item)
            if (
                not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", normalized_name)
                or normalized_name in _CREDENTIAL_ENV_NAMES
                or "\x00" in normalized_value
                or len(normalized_value.encode("utf-8")) > 16 * 1024
            ):
                raise ModelSandboxV2Error("dev provider replay environment is invalid")
            normalized_environment[normalized_name] = normalized_value
        normalized_credential_names = sorted(
            {str(name) for name in credential_env_names}
        )
        if any(name not in _CREDENTIAL_ENV_NAMES for name in normalized_credential_names):
            raise ModelSandboxV2Error(
                "dev provider replay credential name is invalid"
            )
        stdin_payload = {
            "icp": canonicalize_private_model_icp(icp),
            "context": context_with_runtime_options(
                context,
                outer_timeout_seconds=normalized_timeout,
            ),
        }
        if len(canonical_json(stdin_payload).encode("utf-8")) > MAX_MODEL_INPUT_BYTES:
            raise ModelSandboxV2Error("dev provider replay input exceeds limit")

        with tempfile.TemporaryDirectory(
            prefix="lp-dev-provider-replay-v2-", dir="/tmp"
        ) as tmp, _sandbox_visible_workspace(self.config) as visible_root:
            tmp_root = Path(tmp)
            source_root = visible_root / MODEL_SANDBOX_SOURCE_DIRECTORY
            extract_source_bundle_v2(
                source_bundle,
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            _normalize_source_permissions(source_root)
            evidence_root = visible_root / "provider-evidence"
            evidence_root.mkdir(mode=0o700)
            cache_path = evidence_root / "provider-evidence-cache.json"
            cache_path.write_text(canonical_json(cache), encoding="utf-8")
            os.chown(cache_path, self.config.uid, self.config.gid)
            cache_path.chmod(0o400)
            os.chown(evidence_root, self.config.uid, self.config.gid)
            evidence_root.chmod(0o500)
            visible_snapshot_root: Path | None = None
            if snapshot_root is not None:
                visible_snapshot_root = visible_root / "dev-snapshots"
                _copy_readonly_visible_tree(
                    Path(snapshot_root),
                    visible_snapshot_root,
                )
            return self._run_dev_provider_replay(
                source_root=source_root,
                evidence_root=evidence_root,
                module_name=normalized_module,
                callable_name=normalized_callable,
                stdin_payload=stdin_payload,
                environment={
                    **normalized_environment,
                    **{
                        name: _MEASURED_CREDENTIAL_PLACEHOLDER
                        for name in normalized_credential_names
                    },
                    "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": (
                        _sandbox_visible_path(
                            self.config,
                            evidence_root,
                            field="provider evidence root",
                        )
                        + "/provider-evidence-cache.json"
                    ),
                    "RESEARCH_LAB_PROVIDER_EVIDENCE_MODE": "frozen",
                    **(
                        {
                            "RESEARCH_LAB_DEV_SNAPSHOT_DIR": _sandbox_visible_path(
                                self.config,
                                visible_snapshot_root,
                                field="dev snapshot root",
                            )
                        }
                        if snapshot_root is not None
                        else {}
                    ),
                },
                timeout_seconds=normalized_timeout,
                tmp_root=tmp_root,
                job_id=job_id,
            )

    def _run_dev_provider_replay(
        self,
        *,
        source_root: Path,
        evidence_root: Path,
        module_name: str,
        callable_name: str,
        stdin_payload: Mapping[str, Any],
        environment: Mapping[str, str],
        timeout_seconds: int,
        tmp_root: Path,
        job_id: str,
    ) -> list[Mapping[str, Any]]:
        bundle = tmp_root / "bundle"
        bundle.mkdir(mode=0o700)
        runsc_root = tmp_root / "runsc"
        runsc_root.mkdir(mode=0o700)
        sandbox_id = "lp-dev-provider-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        bootstrap = (
            "from gateway.tee.sandbox_http_shim_v2 import install as _lp_install;"
            "_lp_install();\n"
            + model_source_import_bootstrap()
            + _DOCKER_ADAPTER_BOOTSTRAP
        )
        config_doc = _oci_config(
            config=self.config,
            source_root=source_root,
            broker_root=None,
            process_args=[
                self.config.python_path,
                "-c",
                bootstrap,
                module_name,
                callable_name,
            ],
            environment=environment,
            cgroups_path=model_sandbox_job_cgroup_path(
                self.cgroup_parent, sandbox_id
            ),
        )
        (bundle / "config.json").write_text(
            canonical_json(config_doc), encoding="utf-8"
        )
        command = _runsc_run_command(
            config=self.config,
            runsc_root=runsc_root,
            bundle=bundle,
            sandbox_id=sandbox_id,
            host_uds=False,
        )
        try:
            completed = self._process_runner(
                command,
                input=canonical_json(stdin_payload),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={"HOME": str(tmp_root), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                check=False,
            )
        finally:
            try:
                self._process_runner(
                    [
                        str(self.config.runsc_path),
                        "--root=%s" % runsc_root,
                        "delete",
                        "--force",
                        sandbox_id,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                    check=False,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_dev_provider_replay_cleanup_failed "
                    "sandbox_id=%s error=%s",
                    sandbox_id,
                    str(exc)[:240],
                )
        if int(completed.returncode) != 0:
            stderr = str(completed.stderr or "")
            if EVIDENCE_MISS_SENTINEL in stderr:
                fingerprint = stderr.rsplit(EVIDENCE_MISS_SENTINEL, 1)[-1].splitlines()[0]
                raise SnapshotMiss("provider-evidence:" + fingerprint.strip())
            raise ModelSandboxV2Error(
                "dev provider replay adapter failed with code %s stderr_hash=%s"
                % (
                    completed.returncode,
                    sha256_bytes(stderr.encode("utf-8")),
                )
            )
        if len(str(completed.stdout).encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
            raise ModelSandboxV2Error("dev provider replay adapter output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error(
                "dev provider replay adapter output is invalid JSON"
            ) from exc
        if not isinstance(decoded, list):
            raise ModelSandboxV2Error(
                "dev provider replay adapter must return a JSON array"
            )
        try:
            validate_sourcing_runtime_receipt(
                str(completed.stderr or ""),
                expected_runtime_options=dict(stdin_payload["context"])[
                    "runtime_options"
                ],
            )
        except PrivateModelRuntimeError as exc:
            raise ModelSandboxV2Error(
                "dev provider replay sourcing runtime receipt is invalid"
            ) from exc
        return list(
            ensure_private_model_outputs(
                decoded,
                context_label="V2 dev provider replay model sandbox",
                require_non_empty=False,
            )
        )

    def _run_dev_replay(
        self,
        *,
        source_root: Path,
        module_name: str,
        callable_name: str,
        stdin_payload: Mapping[str, Any],
        environment: Mapping[str, str],
        timeout_seconds: int,
        tmp_root: Path,
        job_id: str,
    ) -> list[Mapping[str, Any]]:
        bundle = tmp_root / "bundle"
        bundle.mkdir(mode=0o700)
        runsc_root = tmp_root / "runsc"
        runsc_root.mkdir(mode=0o700)
        sandbox_id = "lp-dev-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        config_doc = _oci_config(
            config=self.config,
            source_root=source_root,
            broker_root=None,
            process_args=[
                self.config.python_path,
                "-c",
                dev_replay_bootstrap()
                + model_source_import_bootstrap()
                + _DOCKER_ADAPTER_BOOTSTRAP,
                module_name,
                callable_name,
            ],
            environment=environment,
            cgroups_path=model_sandbox_job_cgroup_path(
                self.cgroup_parent, sandbox_id
            ),
        )
        (bundle / "config.json").write_text(
            canonical_json(config_doc), encoding="utf-8"
        )
        command = _runsc_run_command(
            config=self.config,
            runsc_root=runsc_root,
            bundle=bundle,
            sandbox_id=sandbox_id,
            host_uds=False,
        )
        try:
            completed = self._process_runner(
                command,
                input=canonical_json(stdin_payload),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={
                    "HOME": str(tmp_root),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                },
                check=False,
            )
        finally:
            try:
                self._process_runner(
                    [
                        str(self.config.runsc_path),
                        "--root=%s" % runsc_root,
                        "delete",
                        "--force",
                        sandbox_id,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                    check=False,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_dev_replay_runsc_cleanup_failed sandbox_id=%s error=%s",
                    sandbox_id,
                    str(exc)[:240],
                )
        if int(completed.returncode) != 0:
            stderr = str(completed.stderr or "")
            if SNAPSHOT_MISS_SENTINEL in stderr:
                request_key = stderr.rsplit(SNAPSHOT_MISS_SENTINEL, 1)[-1].splitlines()[0]
                raise SnapshotMiss(request_key.strip())
            raise ModelSandboxV2Error(
                "dev replay adapter failed with code %s: %s"
                % (completed.returncode, stderr[-1200:])
            )
        if len(str(completed.stdout).encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
            raise ModelSandboxV2Error("dev replay adapter output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error("dev replay adapter output is invalid JSON") from exc
        if not isinstance(decoded, list):
            raise ModelSandboxV2Error("dev replay adapter must return a JSON array")
        try:
            validate_sourcing_runtime_receipt(
                str(completed.stderr or ""),
                expected_runtime_options=dict(stdin_payload["context"])[
                    "runtime_options"
                ],
            )
        except PrivateModelRuntimeError as exc:
            raise ModelSandboxV2Error(
                "dev replay sourcing runtime receipt is invalid"
            ) from exc
        return list(
            ensure_private_model_outputs(
                decoded,
                context_label="V2 dev replay model sandbox",
                require_non_empty=False,
            )
        )

    def _run(
        self,
        value: Mapping[str, Any],
        *,
        artifact: PrivateModelArtifactManifest,
        source_root: Path,
        broker_root: Path,
        tmp_root: Path,
        job_id: str,
        provider_snapshot_root: Path | None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        operation = value["operation"]
        if operation == "run_icp":
            raw_input = value["input"]
            if not isinstance(raw_input, Mapping) or set(raw_input) != {"icp", "context"}:
                raise ModelSandboxV2Error("model run input fields are invalid")
            stdin_payload = {
                "icp": canonicalize_private_model_icp(raw_input["icp"]),
                "context": dict(raw_input["context"]),
            }
            bootstrap = (
                "from gateway.tee.sandbox_http_shim_v2 import install as _lp_install;"
                "_lp_install();\n"
                + model_source_import_bootstrap()
                + _DOCKER_ADAPTER_BOOTSTRAP
            )
        else:
            stdin_payload = {}
            bootstrap = model_source_import_bootstrap() + _DOCKER_METADATA_BOOTSTRAP
        encoded_input = canonical_json(stdin_payload)
        process_timeout_seconds = _model_sandbox_process_timeout_seconds(value)
        bundle = tmp_root / "bundle"
        bundle.mkdir(mode=0o700)
        runsc_root = tmp_root / "runsc"
        runsc_root.mkdir(mode=0o700)
        sandbox_id = "lp-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        environment = {
            **dict(value["environment"]),
            "RESEARCH_LAB_PROVIDER_COST_SCOPE": value["provider_cost_scope"],
            "RESEARCH_LAB_PROVIDER_EVIDENCE_MODE": value["provider_evidence_mode"],
            **(
                {
                    "RESEARCH_LAB_PROVIDER_COST_CAP_MICROUSD": str(
                        value["provider_cost_cap_microusd"]
                    ),
                    "RESEARCH_LAB_PROVIDER_CALL_CAP": str(
                        value["provider_call_cap"]
                    ),
                }
                if value["provider_cost_cap_microusd"]
                and value["provider_call_cap"]
                else {}
            ),
            **{
                name: _MEASURED_CREDENTIAL_PLACEHOLDER
                for name in sorted(_CREDENTIAL_ENV_NAMES)
            },
            **source_add_placeholder_environment_v2(
                value["provider_runtime_catalog"]
            ),
        }
        evidence_cache = dict(value["provider_evidence_cache"])
        if evidence_cache:
            evidence_cache_path = broker_root / "provider-evidence-cache.json"
            evidence_cache_path.write_text(
                canonical_json(evidence_cache),
                encoding="utf-8",
            )
            evidence_cache_path.chmod(0o444)
            environment["RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH"] = (
                _sandbox_visible_path(
                    self.config,
                    broker_root,
                    field="provider broker",
                )
                + "/provider-evidence-cache.json"
            )
        if provider_snapshot_root is not None:
            environment["RESEARCH_LAB_DEV_SNAPSHOT_DIR"] = _sandbox_visible_path(
                self.config,
                provider_snapshot_root,
                field="provider snapshot root",
            )
        config_doc = _oci_config(
            config=self.config,
            source_root=source_root,
            broker_root=broker_root,
            process_args=[
                self.config.python_path,
                "-c",
                bootstrap,
                value["module_name"],
                value["callable_name"],
            ],
            environment=environment,
            cgroups_path=model_sandbox_job_cgroup_path(
                self.cgroup_parent, sandbox_id
            ),
        )
        (bundle / "config.json").write_text(
            canonical_json(config_doc), encoding="utf-8"
        )
        command = _runsc_run_command(
            config=self.config,
            runsc_root=runsc_root,
            bundle=bundle,
            sandbox_id=sandbox_id,
            host_uds=True,
        )
        try:
            completed = self._process_runner(
                command,
                input=encoded_input,
                text=True,
                capture_output=True,
                timeout=process_timeout_seconds,
                env={
                    "HOME": str(tmp_root),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelSandboxV2Error("model sandbox timed out") from exc
        finally:
            try:
                self._process_runner(
                    [
                        str(self.config.runsc_path),
                        "--root=%s" % runsc_root,
                        "delete",
                        "--force",
                        sandbox_id,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                    check=False,
                )
            except Exception as cleanup_exc:
                logger.warning(
                    "model_sandbox_runsc_cleanup_failed "
                    "sandbox_id=%s error_type=%s",
                    sandbox_id,
                    type(cleanup_exc).__name__,
                )
        if int(completed.returncode) != 0:
            stripped_stderr = strip_incontainer_trace_lines(
                str(completed.stderr or "")
            )
            failure_code, error_hash = _runsc_failure_evidence(stripped_stderr)
            raise ModelSandboxV2Error(
                "model sandbox failed code=%s returncode=%s stderr_hash=%s"
                % (failure_code, completed.returncode, error_hash)
            )
        if len(str(completed.stdout).encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
            raise ModelSandboxV2Error("model sandbox output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error("model sandbox output is invalid JSON") from exc
        if operation == "run_icp":
            stderr = str(completed.stderr or "")
            raw_input = value.get("input")
            context = raw_input.get("context") if isinstance(raw_input, Mapping) else None
            defer_retryable_errors = bool(
                value.get("model_kind") == "private"
                and isinstance(context, Mapping)
                and context.get("mode") == "private_baseline"
            )
            try:
                _raise_on_empty_provider_error(
                    decoded,
                    stderr,
                    context_label="V2 model sandbox",
                    defer_retryable_errors=defer_retryable_errors,
                )
            except PrivateModelRuntimeError as exc:
                raise ModelSandboxV2Error(str(exc)) from exc
            output = list(
                ensure_private_model_outputs(
                    decoded,
                    context_label="V2 model sandbox",
                    require_non_empty=False,
                )
            )
            return output, [
                *parse_incontainer_trace_lines(stderr),
                *parse_sourcing_runtime_lines(stderr),
            ]
        if not isinstance(decoded, Mapping):
            raise ModelSandboxV2Error("model metadata output must be an object")
        return dict(decoded), parse_incontainer_trace_lines(
            str(completed.stderr or "")
        )
