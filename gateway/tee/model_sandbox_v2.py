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
    compute_private_source_tree_hash,
    PrivateModelRuntimeError,
    PrivateModelArtifactManifest,
    ensure_private_model_outputs,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import (
    _DOCKER_ADAPTER_BOOTSTRAP,
    _docker_adapter_bootstrap_for_qualify_compatibility,
    _docker_adapter_bootstrap_for_qualification_protocol_v2,
    _raise_on_empty_provider_error,
    PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    PRIVATE_RUNTIME_FAILURE_MARKER,
    PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION,
    QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2,
    QUALIFICATION_OUTCOME_ENTRYPOINT_V2,
    QUALIFICATION_OUTCOME_PROTOCOL_PROBE_MODE_V1,
    QUALIFICATION_OUTCOME_PROTOCOL_PROBE_SCHEMA_V1,
    QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2,
    qualification_outcome_required_route_terminal_satisfies_v2,
    QUALIFICATION_OUTCOME_REQUIRED_PROBE_CASES_V1,
    qualification_outcome_contract_sha256_v2,
    qualification_outcome_probe_nonce_valid_v1,
    SOURCING_MODEL_MAX_RUNTIME_CAP_SECONDS,
    canonicalize_private_model_icp,
    context_with_runtime_options,
    parse_incontainer_trace_lines,
    parse_sourcing_runtime_lines,
    strip_incontainer_trace_lines,
    validate_sourcing_adapter_metadata,
    validate_sourcing_runtime_receipt,
    validate_qualification_outcome_protocol_metadata_v2,
    validate_qualification_outcome_protocol_probe_cases_v1,
    validate_qualification_outcome_envelope_v2,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    build_evidence_cache_from_trace_entries,
    canonical_request_fingerprint,
    icp_evidence_cache_key,
    merge_evidence_caches,
)
from research_lab.eval.snapshot_store import (
    MODE_REPLAY,
    ProviderSnapshotStore,
    SNAPSHOT_MISS_SENTINEL,
    SnapshotMiss,
    container_replay_env,
    dev_replay_bootstrap,
)
from research_lab.sourcing_model_contract_check import (
    SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
    semantic_compatibility_policy_identity_v1,
    source_tree_compatibility_admission as source_tree_compatibility_admission_v1,
    validate_source_tree_compatibility_receipt,
)


logger = logging.getLogger(__name__)


MODEL_SANDBOX_REQUEST_SCHEMA_VERSION = "leadpoet.model_sandbox_request.v2"
MODEL_SANDBOX_RESULT_SCHEMA_VERSION = "leadpoet.model_sandbox_result.v2"
NATIVE_QUALIFY_RELEASE_IDENTITY_V1 = {
    "source_tree_hash": (
        "sha256:491d6e76adf629b60d913062005191673f962db3cd5cd77223a68cf6262ac60f"
    ),
    "git_commit_sha": "e55e57f2be0ddadcc6b9c92c18b932dc2c354d21",
    "manifest_hash": (
        "sha256:af68f0fbd29c77f9ffe686dcbddbc1e5dd1cab6c8725c7c9669de367bd592928"
    ),
    "image_digest": (
        "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
        "sourcing-model@sha256:f1ae9bc0ba2cd55450e4c1b1bbdb0030514dbf5afd380f29a09d5e95bdb0ade5"
    ),
}
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
MODEL_SANDBOX_METADATA_TIMEOUT_SECONDS = 120
MODEL_SANDBOX_TIMEOUT_GRACE_SECONDS = 3
MAX_MODEL_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_MODEL_METADATA_OUTPUT_BYTES = 1024 * 1024
MAX_MODEL_METADATA_DIAGNOSTIC_BYTES = 1024 * 1024
MODEL_SANDBOX_PIPE_CHUNK_BYTES = 64 * 1024
MAX_PROVIDER_EVIDENCE_CACHE_BYTES = 32 * 1024 * 1024
MODEL_SANDBOX_ATTESTED_RUNTIME_ROOT = "/app/gateway/_attested_runtime"
MODEL_SANDBOX_VISIBLE_ROOT = "/leadpoet-model-sandboxes"
MODEL_SANDBOX_SOURCE_DIRECTORY = "source"
MODEL_SANDBOX_BROKER_DIRECTORY = "broker"
MODEL_SANDBOX_SELF_TEST_SCHEMA_VERSION = "leadpoet.model_sandbox_self_test.v2"
CONSUMER_RUNTIME_PROBE_SCHEMA_VERSION = (
    "leadpoet.sourcing-model-consumer-runtime-probe.v1"
)
CONSUMER_RUNTIME_OBSERVATION_PLAN_SCHEMA_V1 = (
    "leadpoet.consumer-runtime-observation-plan.v1"
)
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
PRIVATE_RUNTIME_FAILURE_MARKER_MAX_BYTES = 512
RUNSC_FAILURE_CODES_V1 = frozenset(
    {
        "runsc_cgroup_setup",
        "runsc_gofer_chroot",
        "runsc_gofer_create",
        "runsc_gofer_exec",
        "runsc_mount_resolve",
        "runsc_mount_setup",
        "runsc_nonzero",
        "runsc_permission_denied",
        "runsc_rootfs_setup",
        "runsc_source_mount_missing",
        "runsc_source_staging_missing",
    }
)


@dataclass(frozen=True)
class ModelSandboxFailureProjectionV1:
    """Bounded diagnostic observation from one failed sandbox process.

    Every field is an untrusted diagnostic observation derived from a
    model-controlled process.  Signing binds what was observed; it does not
    make the observation scoring, retry, admission, or trust authority.
    """

    launcher_code: str
    stderr_hash: str
    exception_class_hash: Optional[str] = None


class ModelSandboxV2Error(RuntimeError):
    """A model runtime, bundle, provider path, or output failed validation."""

    def __init__(
        self,
        message: str,
        *,
        failure_projection: Optional[ModelSandboxFailureProjectionV1] = None,
    ) -> None:
        super().__init__(message)
        self.failure_projection = (
            validate_model_sandbox_failure_projection_v1(failure_projection)
            if failure_projection is not None
            else None
        )


def validate_model_sandbox_failure_projection_v1(
    value: ModelSandboxFailureProjectionV1,
) -> ModelSandboxFailureProjectionV1:
    if type(value) is not ModelSandboxFailureProjectionV1:
        raise ModelSandboxV2Error("model sandbox failure projection is invalid")
    if (
        type(value.launcher_code) is not str
        or value.launcher_code not in RUNSC_FAILURE_CODES_V1
    ):
        raise ModelSandboxV2Error("model sandbox launcher failure code is invalid")
    if type(value.stderr_hash) is not str or not _HASH_RE.fullmatch(
        value.stderr_hash
    ):
        raise ModelSandboxV2Error("model sandbox stderr hash is invalid")
    if value.exception_class_hash is not None and (
        type(value.exception_class_hash) is not str
        or not _HASH_RE.fullmatch(value.exception_class_hash)
    ):
        raise ModelSandboxV2Error(
            "model sandbox exception-class observation is invalid"
        )
    return value


def _validated_private_runtime_failure_marker_v1(line: str) -> Optional[str]:
    prefix = PRIVATE_RUNTIME_FAILURE_MARKER + " "
    if not line.startswith(prefix):
        return None
    if len(line.encode("utf-8")) > PRIVATE_RUNTIME_FAILURE_MARKER_MAX_BYTES:
        return None
    payload = line[len(prefix) :]
    try:
        document = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(document, Mapping)
        or set(document) != {"exception_class_hash", "schema_version"}
        or document.get("schema_version")
        != PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION
        or not isinstance(document.get("exception_class_hash"), str)
        or not _HASH_RE.fullmatch(document["exception_class_hash"])
        or payload != canonical_json(dict(document))
    ):
        return None
    return str(document["exception_class_hash"])


def _strip_private_runtime_failure_markers_v1(
    stderr: str,
) -> tuple[str, Optional[str]]:
    """Strip marker lines and accept only one exact terminal observation."""

    raw_stderr = str(stderr or "")
    lines = raw_stderr.splitlines(keepends=True)
    terminal = lines[-1] if lines else ""
    terminal_line = (
        terminal[:-1]
        if terminal.endswith("\n") and not terminal.endswith("\r\n")
        else None
    )
    observed_hash = (
        _validated_private_runtime_failure_marker_v1(terminal_line)
        if terminal_line is not None
        else None
    )
    prefix = PRIVATE_RUNTIME_FAILURE_MARKER + " "
    stripped = "".join(line for line in lines if not line.startswith(prefix))
    return stripped, observed_hash


def _runsc_failure_evidence(
    stderr: str,
    *,
    returncode: Any = None,
) -> tuple[str, str, Optional[str]]:
    """Return bounded safe observations without exposing sandbox output."""

    marker_stripped, observed_exception_class_hash = (
        _strip_private_runtime_failure_markers_v1(stderr)
    )
    try:
        wrapper_exit = int(returncode) == PRIVATE_RUNTIME_FAILURE_EXIT_CODE
    except (TypeError, ValueError):
        wrapper_exit = False
    exception_class_hash = (
        observed_exception_class_hash if wrapper_exit else None
    )
    sanitized = strip_incontainer_trace_lines(marker_stripped)
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
    return (
        failure_code,
        sha256_bytes(sanitized.encode("utf-8")),
        exception_class_hash,
    )


def _runsc_model_sandbox_error(
    *,
    stderr: str,
    returncode: Any,
) -> ModelSandboxV2Error:
    launcher_code, stderr_hash, exception_class_hash = _runsc_failure_evidence(
        stderr,
        returncode=returncode,
    )
    return ModelSandboxV2Error(
        "model sandbox failed code=%s returncode=%s stderr_hash=%s"
        % (launcher_code, returncode, stderr_hash),
        failure_projection=ModelSandboxFailureProjectionV1(
            launcher_code=launcher_code,
            stderr_hash=stderr_hash,
            exception_class_hash=exception_class_hash,
        ),
    )


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
    command.append("--host-uds=open" if host_uds else "--host-uds=none")
    return [
        *command,
        "--platform=ptrace",
        "run",
        "--bundle=%s" % bundle,
        sandbox_id,
    ]


def _force_delete_runsc_sandbox(
    *,
    process_runner: Callable[..., Any],
    config: "RunscSandboxConfigV2",
    runsc_root: Path,
    sandbox_id: str,
    failure_event: str,
) -> None:
    try:
        process_runner(
            [
                str(config.runsc_path),
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
            "%s sandbox_id=%s error_type=%s",
            failure_event,
            sandbox_id,
            type(exc).__name__,
        )
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


def trusted_model_sandbox_import_bootstrap() -> str:
    """Pin host-owned provider helpers before model packages become visible."""

    return """
import research_lab.eval.provider_evidence_cache as _lp_trusted_evidence_cache
import research_lab.eval.snapshot_store as _lp_trusted_snapshot_store
"""


def _model_adapter_bootstrap_for_compatibility_receipt_v1(
    compatibility_receipt: Mapping[str, Any],
    *,
    artifact: PrivateModelArtifactManifest,
) -> str:
    """Select host-owned adapter bytes from an exact artifact-bound receipt."""

    if compatibility_receipt.get("admission_mode") == "qualification_protocol_v2":
        try:
            validate_source_tree_compatibility_receipt(
                compatibility_receipt,
                manifest=artifact,
                source_tree_hash=artifact.model_artifact_hash,
            )
        except (TypeError, ValueError) as exc:
            raise ModelSandboxV2Error(
                "qualification protocol receipt differs from signed artifact"
            ) from exc
        return _docker_adapter_bootstrap_for_qualification_protocol_v2()

    try:
        validated_receipt = validate_source_tree_compatibility_receipt(
            compatibility_receipt,
            manifest=artifact,
            source_tree_hash=artifact.model_artifact_hash,
        )
    except (TypeError, ValueError) as exc:
        raise ModelSandboxV2Error(
            "qualify compatibility receipt differs from signed artifact"
        ) from exc
    release_identity = {
        "source_tree_hash": artifact.model_artifact_hash,
        "git_commit_sha": artifact.git_commit_sha,
        "manifest_hash": artifact.manifest_hash,
        "image_digest": artifact.image_digest,
    }
    native_source_tree = NATIVE_QUALIFY_RELEASE_IDENTITY_V1[
        "source_tree_hash"
    ]
    if (
        validated_receipt.get("source_tree_hash") == native_source_tree
        and release_identity != NATIVE_QUALIFY_RELEASE_IDENTITY_V1
    ):
        raise ModelSandboxV2Error(
            "native qualify source differs from its reviewed release identity"
        )
    return _docker_adapter_bootstrap_for_qualify_compatibility(
        preserve_native_qualify=(
            validated_receipt.get("source_tree_hash")
            == artifact.model_artifact_hash
            and release_identity == NATIVE_QUALIFY_RELEASE_IDENTITY_V1
        )
    )


def _validate_qualification_terminal_observation_v1(
    envelope: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Join producer completion to payload-free host terminal observations."""

    document = dict(observation)
    fields = {
        "schema_version",
        "request_intent_count",
        "terminal_count",
        "latest_operation_count",
        "accepted_latest_terminal_count",
        "successful_latest_terminal_count",
        "failed_latest_terminal_count",
        "unresolved_latest_terminal_count",
        "latest_terminal_attempt_hashes",
        "successful_latest_terminal_attempt_hashes",
        "required_route_commitments",
        "required_route_count",
        "successful_required_route_count",
        "unresolved_required_route_count",
        "required_route_terminals",
    }
    count_fields = {
        "request_intent_count",
        "terminal_count",
        "latest_operation_count",
        "accepted_latest_terminal_count",
        "successful_latest_terminal_count",
        "failed_latest_terminal_count",
        "unresolved_latest_terminal_count",
        "required_route_count",
        "successful_required_route_count",
        "unresolved_required_route_count",
    }
    counts = {name: document.get(name) for name in count_fields}
    latest_hashes = document.get("latest_terminal_attempt_hashes")
    successful_hashes = document.get(
        "successful_latest_terminal_attempt_hashes"
    )
    required_commitments = document.get("required_route_commitments")
    required_terminals = document.get("required_route_terminals")
    receipt = dict(envelope["route_completion_receipt"])
    disposition = str(receipt.get("disposition") or "")
    if (
        set(document) != fields
        or document.get("schema_version")
        != "leadpoet.provider-terminal-observation.v1"
        or any(type(item) is not int or item < 0 for item in counts.values())
        or document["request_intent_count"] != document["terminal_count"]
        or document["latest_operation_count"]
        != document["accepted_latest_terminal_count"]
        + document["failed_latest_terminal_count"]
        or document["successful_latest_terminal_count"]
        > document["accepted_latest_terminal_count"]
        or document["unresolved_latest_terminal_count"]
        != document["latest_operation_count"]
        - document["successful_latest_terminal_count"]
        or not isinstance(latest_hashes, list)
        or latest_hashes != sorted(latest_hashes)
        or len(set(latest_hashes)) != len(latest_hashes)
        or len(latest_hashes) != document["latest_operation_count"]
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(item or "")) is None
            for item in latest_hashes
        )
        or not isinstance(successful_hashes, list)
        or successful_hashes != sorted(successful_hashes)
        or len(set(successful_hashes)) != len(successful_hashes)
        or len(successful_hashes)
        != document["successful_latest_terminal_count"]
        or not set(successful_hashes) <= set(latest_hashes)
        or not isinstance(required_commitments, list)
        or required_commitments != sorted(required_commitments)
        or len(set(required_commitments)) != len(required_commitments)
        or len(required_commitments)
        > QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(item or "")) is None
            for item in required_commitments
        )
        or not isinstance(required_terminals, list)
        or len(required_terminals) != len(required_commitments)
        or any(
            not isinstance(item, Mapping)
            or set(item)
            != {
                "route_commitment",
                "attempt_hash",
                "terminal_status",
                "http_status",
            }
            or item.get("route_commitment") != required_commitments[index]
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(item.get("attempt_hash") or ""),
            )
            is None
            or item.get("terminal_status")
            not in {
                "authenticated_response",
                "attested_local_response",
                "transport_failure",
            }
            or (
                item.get("terminal_status") == "transport_failure"
                and item.get("http_status") is not None
            )
            or (
                item.get("terminal_status") != "transport_failure"
                and (
                    type(item.get("http_status")) is not int
                    or not 100 <= item["http_status"] <= 599
                )
            )
            for index, item in enumerate(required_terminals)
        )
        or document["required_route_count"] != len(required_commitments)
        or document["successful_required_route_count"]
        != sum(
            1
            for item in required_terminals
            if item["terminal_status"]
            in {"authenticated_response", "attested_local_response"}
            and 200 <= item["http_status"] <= 299
        )
        or document["unresolved_required_route_count"]
        != document["required_route_count"]
        - document["successful_required_route_count"]
    ):
        raise ModelSandboxV2Error(
            "qualification outcome differs from measured provider terminals"
        )
    bound_outcomes = receipt.get("extensions", {}).get(
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2
    )
    bound_commitments = (
        [item.get("commitment") for item in bound_outcomes]
        if isinstance(bound_outcomes, list)
        and all(isinstance(item, Mapping) for item in bound_outcomes)
        else []
    )
    if (
        not isinstance(bound_outcomes, list)
        or bound_commitments != required_commitments
        or receipt["route_summary"]["attempted"]
        != len(required_commitments)
        or any(
            outcome.get("commitment") != terminal.get("route_commitment")
            or (
                outcome.get("state") in {"completed", "confirmed_empty"}
                and not qualification_outcome_required_route_terminal_satisfies_v2(
                    outcome.get("state"),
                    terminal.get("terminal_status"),
                    terminal.get("http_status"),
                )
            )
            for outcome, terminal in zip(bound_outcomes, required_terminals)
        )
    ):
        raise ModelSandboxV2Error(
            "qualification outcome required routes differ from host observation"
        )
    if disposition.startswith("complete_") and (
        not required_commitments
        or any(
            outcome.get("state") not in {"completed", "confirmed_empty"}
            or not qualification_outcome_required_route_terminal_satisfies_v2(
                outcome.get("state"),
                terminal.get("terminal_status"),
                terminal.get("http_status"),
            )
            for outcome, terminal in zip(bound_outcomes, required_terminals)
        )
    ):
        raise ModelSandboxV2Error(
            "complete outcome lacks complete required-route authority"
        )
    return document


def _runtime_invariant_policy_v1(
    compatibility_receipt: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    receipt = dict(compatibility_receipt)
    if receipt.get("admission_mode") == "qualification_protocol_v2":
        if (
            receipt.get("consumer_api_version")
            != "research-lab-qualification-consumer-api:v2"
            or receipt.get("decision")
            != SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION
            or receipt.get("policy_hash")
            != f"sha256:{qualification_outcome_contract_sha256_v2()}"
        ):
            raise ModelSandboxV2Error(
                "qualification runtime receipt differs from protected policy"
            )
        # Protocol-v2 behavior is selected by its nonce probes below. It does
        # not inherit semantic-v1's exact source-function equality policy.
        return {"profile": "qualification_protocol_v2"}
    policy, policy_hash = semantic_compatibility_policy_identity_v1()
    if (
        receipt.get("consumer_api_version") != policy["consumer_api_version"]
        or receipt.get("decision") != SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION
        or receipt.get("policy_hash") != policy_hash
    ):
        raise ModelSandboxV2Error(
            "consumer runtime receipt differs from compatibility policy"
        )
    if receipt.get("admission_mode") == "legacy_exact":
        return None
    if receipt.get("admission_mode") != "semantic_v1":
        raise ModelSandboxV2Error("consumer runtime admission mode is invalid")
    return dict(policy.get("runtime_invariants") or {})


def _runtime_probe_expected_invariants(
    invariant_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if invariant_policy is None:
        return {"profile": "legacy_exact"}
    policy = dict(invariant_policy)
    if policy == {"profile": "qualification_protocol_v2"}:
        return dict(policy)
    company = dict(policy.get("company_fit") or {})
    capabilities = dict(policy.get("runtime_capabilities") or {})
    names = sorted(str(item) for item in capabilities.get("names") or ())
    capability_metadata = {
        "capability_contract_version": capabilities.get("contract_version"),
        "capabilities": names,
        "host_registered": [],
    }
    registered_metadata = {
        **capability_metadata,
        "host_registered": names,
    }
    sentinel = dict(capabilities.get("sentinel") or {})
    return {
        "profile": "semantic_v1",
        "schema_version": policy.get("schema_version"),
        "adapter_dependencies": {
            "build_query_returns_string": True,
            "first_party_industry_run_is_context_manager": True,
            "flow_mode_is_supported": True,
        },
        "company_fit": {
            "identity": dict(company.get("identity") or {}),
            "strict_boolean": {
                str(case["case_id"]): case.get("expected")
                for case in company.get("strict_boolean_cases") or ()
            },
            "reconcile": {
                str(case["case_id"]): case.get("expected")
                for case in company.get("reconcile_cases") or ()
            },
            "aggregate": {
                str(case["case_id"]): case.get("expected")
                for case in company.get("aggregate_cases") or ()
            },
        },
        "runtime_capabilities": {
            "initial_metadata": capability_metadata,
            "unknown_registration_rejected": True,
            "noncallable_registration_rejected": True,
            "registered_metadata": registered_metadata,
            "dispatch": {
                "deadline": sentinel.get("deadline"),
                "emit_events": [dict(sentinel.get("emit_event") or {})],
                "http_fetch_result_is_sentinel": True,
                "http_fetch_calls": [
                    {
                        "url": sentinel.get("http_url"),
                        "timeout": sentinel.get("http_timeout"),
                        "max_bytes": sentinel.get("http_max_bytes"),
                        "accept": sentinel.get("http_accept"),
                    }
                ],
                "resolve_calls": [sentinel.get("resolve_input")],
                "resolve_result": dict(
                    capabilities.get("host_resolution") or {}
                ).get("TIMEOUT"),
                "probe_calls": [sentinel.get("probe_input")],
                "probe_result": dict(
                    capabilities.get("origin_reachability") or {}
                ).get("UNKNOWN"),
            },
            "after_reset_metadata": capability_metadata,
            "defaults": {
                "deadline_is_none": True,
                "emit_succeeded": True,
                "http_fetch_identity": True,
                "resolve_host_identity": True,
                "probe_origin_result": dict(
                    capabilities.get("origin_reachability") or {}
                ).get("UNKNOWN"),
                "may_attempt_unknown": True,
                "timeout_is_terminal": False,
                "nxdomain_is_terminal": True,
            },
        },
    }


def _runtime_probe_observation_plan_v1(
    compatibility_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    invariant_policy = _runtime_invariant_policy_v1(compatibility_receipt)
    document = {
        "schema_version": CONSUMER_RUNTIME_OBSERVATION_PLAN_SCHEMA_V1,
        "runtime_invariants": invariant_policy,
    }
    if invariant_policy == {"profile": "qualification_protocol_v2"}:
        document.update(
            {
                "qualification_outcome_entrypoint": QUALIFICATION_OUTCOME_ENTRYPOINT_V2,
                "qualification_outcome_probe_mode": (
                    QUALIFICATION_OUTCOME_PROTOCOL_PROBE_MODE_V1
                ),
                "qualification_outcome_probe_schema_version": (
                    QUALIFICATION_OUTCOME_PROTOCOL_PROBE_SCHEMA_V1
                ),
                "qualification_outcome_probes": [
            {
                "case_id": case_id,
                "nonce": secrets.token_hex(24),
            }
            for case_id in QUALIFICATION_OUTCOME_REQUIRED_PROBE_CASES_V1
                ],
            }
        )
    return document


def _consumer_runtime_probe_v1(
    *,
    compatibility_receipt: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_module_name: str,
    expected_callable_name: str,
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONSUMER_RUNTIME_PROBE_SCHEMA_VERSION,
        "compatibility_receipt": dict(compatibility_receipt),
        "entrypoint": {
            "module": expected_module_name,
            "callable": expected_callable_name,
        },
        "metadata_identity": {
            **{
                name: metadata.get(name)
                for name in (
                    "adapter_version",
                    "capability_contract_version",
                    "component_registry_version",
                    "scoring_adapter_version",
                )
            },
            "runtime_capabilities": sorted(
                str(item) for item in metadata.get("runtime_capabilities") or ()
            ),
        },
        "invariants": dict(invariants),
    }


def validate_consumer_runtime_probe_v1(
    value: Mapping[str, Any],
    *,
    compatibility_receipt: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_source_tree_hash: str,
    expected_manifest_hash: str,
    expected_image_digest: str,
    expected_module_name: str,
    expected_callable_name: str,
) -> dict[str, Any]:
    document = dict(value)
    receipt = dict(compatibility_receipt)
    invariant_policy = _runtime_invariant_policy_v1(receipt)
    raw_invariants = document.get("invariants")
    if not isinstance(raw_invariants, Mapping):
        raise ModelSandboxV2Error(
            "consumer runtime probe differs from host admission"
        )
    observed_invariants = dict(raw_invariants)
    observed_qualification = observed_invariants.pop(
        "qualification_outcome_protocol", None
    )
    expected_invariants = _runtime_probe_expected_invariants(invariant_policy)
    expected = _consumer_runtime_probe_v1(
        compatibility_receipt=receipt,
        metadata=metadata,
        expected_module_name=expected_module_name,
        expected_callable_name=expected_callable_name,
        invariants={
            **expected_invariants,
            **(
                {"qualification_outcome_protocol": observed_qualification}
                if observed_qualification is not None
                else {}
            ),
        },
    )
    metadata_protocol = metadata.get("qualification_outcome_protocol")
    admitted_protocol_v2 = (
        receipt.get("admission_mode") == "qualification_protocol_v2"
    )
    if (metadata_protocol is not None) != admitted_protocol_v2:
        raise ModelSandboxV2Error(
            "consumer qualification protocol differs from source admission"
        )
    if (metadata_protocol is None) != (observed_qualification is None):
        raise ModelSandboxV2Error(
            "consumer qualification protocol observation is missing"
        )
    if metadata_protocol is not None:
        try:
            validate_qualification_outcome_protocol_metadata_v2(
                metadata_protocol
            )
            if (
                not isinstance(observed_qualification, Mapping)
                or set(observed_qualification) != {"cases", "nonce_sha256s"}
                or not isinstance(observed_qualification.get("cases"), Mapping)
                or not isinstance(
                    observed_qualification.get("nonce_sha256s"), Mapping
                )
            ):
                raise PrivateModelRuntimeError(
                    "qualification outcome probe observation is invalid"
                )
            validate_qualification_outcome_protocol_probe_cases_v1(
                observed_qualification["cases"],
                expected_nonce_sha256s=observed_qualification["nonce_sha256s"],
            )
        except PrivateModelRuntimeError as exc:
            raise ModelSandboxV2Error(
                "consumer qualification protocol observation differs"
            ) from exc
    if (
        document != expected
        or observed_invariants != expected_invariants
        or receipt.get("source_tree_hash") != expected_source_tree_hash
        or receipt.get("manifest_hash") != expected_manifest_hash
        or receipt.get("image_digest") != expected_image_digest
    ):
        raise ModelSandboxV2Error(
            "consumer runtime probe differs from host admission"
        )
    return document


def _build_consumer_runtime_probe_from_observation_v1(
    observation: Mapping[str, Any],
    *,
    compatibility_receipt: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_source_tree_hash: str,
    expected_manifest_hash: str,
    expected_image_digest: str,
    expected_module_name: str,
    expected_callable_name: str,
    observation_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = dict(observation)
    metadata_document = dict(metadata)
    expected_observation_fields = {
        "invariants",
        "qualification_outcome_protocol",
    }
    if set(raw) != expected_observation_fields or not isinstance(
        raw.get("invariants"), Mapping
    ):
        raise ModelSandboxV2Error(
            "consumer runtime observation fields are invalid"
        )
    raw_qualification = raw.get("qualification_outcome_protocol")
    normalized_qualification = None
    if metadata_document.get("qualification_outcome_protocol") is not None:
        plan = dict(observation_plan or {})
        cases = (
            dict(raw_qualification).get("cases")
            if isinstance(raw_qualification, Mapping)
            else None
        )
        probe_plan = plan.get("qualification_outcome_probes")
        if (
            not isinstance(probe_plan, list)
            or not isinstance(cases, Mapping)
            or any(
                not isinstance(item, Mapping)
                or not qualification_outcome_probe_nonce_valid_v1(
                    item.get("nonce")
                )
                for item in probe_plan
            )
        ):
            raise ModelSandboxV2Error(
                "consumer qualification protocol probe plan is invalid"
            )
        nonce_hashes = {
            str(item["case_id"]): hashlib.sha256(
                str(item["nonce"]).encode("ascii")
            ).hexdigest()
            for item in probe_plan
        }
        try:
            validate_qualification_outcome_protocol_metadata_v2(
                metadata_document["qualification_outcome_protocol"]
            )
            normalized_cases = (
                validate_qualification_outcome_protocol_probe_cases_v1(
                    cases,
                    expected_nonce_sha256s=nonce_hashes,
                )
            )
        except PrivateModelRuntimeError as exc:
            raise ModelSandboxV2Error(
                "consumer qualification protocol probe differs"
            ) from exc
        normalized_qualification = {
            "cases": normalized_cases,
            "nonce_sha256s": dict(sorted(nonce_hashes.items())),
        }
    elif raw_qualification is not None:
        raise ModelSandboxV2Error(
            "consumer qualification protocol observation is unexpected"
        )
    probe = _consumer_runtime_probe_v1(
        compatibility_receipt=compatibility_receipt,
        metadata=metadata_document,
        expected_module_name=expected_module_name,
        expected_callable_name=expected_callable_name,
        invariants={
            **dict(raw["invariants"]),
            **(
                {"qualification_outcome_protocol": normalized_qualification}
                if normalized_qualification is not None
                else {}
            ),
        },
    )
    return validate_consumer_runtime_probe_v1(
        probe,
        compatibility_receipt=compatibility_receipt,
        metadata=metadata_document,
        expected_source_tree_hash=expected_source_tree_hash,
        expected_manifest_hash=expected_manifest_hash,
        expected_image_digest=expected_image_digest,
        expected_module_name=expected_module_name,
        expected_callable_name=expected_callable_name,
    )


_MEASURED_METADATA_BOOTSTRAP = r"""
import contextlib as _lp_contextlib
import importlib as _lp_importlib
import json as _lp_json
import os as _lp_os
import sys as _lp_sys
from pathlib import Path as _lp_Path

_lp_payload = _lp_json.load(_lp_sys.stdin)
_lp_plan = _lp_payload.get("observation_plan")
if (
    not isinstance(_lp_plan, dict)
    or set(_lp_plan)
    not in (
        {"schema_version", "runtime_invariants"},
        {
            "schema_version",
            "runtime_invariants",
            "qualification_outcome_entrypoint",
            "qualification_outcome_probe_mode",
            "qualification_outcome_probe_schema_version",
            "qualification_outcome_probes",
        },
    )
    or _lp_plan.get("schema_version")
    != "leadpoet.consumer-runtime-observation-plan.v1"
):
    raise RuntimeError("metadata observation plan is invalid")
_lp_blocked_imports = (
    "__main__",
    "gateway.research_lab",
    "gateway.tee",
    "research_lab.sourcing_model_contract_check",
)
_lp_denied_events = {
    "ctypes.dlopen", "os.chdir", "os.chmod", "os.chown", "os.link",
    "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.symlink",
    "os.system", "os.truncate", "socket.__new__", "subprocess.Popen",
}
def _lp_audit(event, args):
    if event in _lp_denied_events:
        raise RuntimeError("metadata observer operation is denied")
    if event == "open":
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        if (
            isinstance(mode, str)
            and any(item in mode for item in ("+", "a", "w", "x"))
        ) or (isinstance(flags, int) and flags & 0o3301):
            raise RuntimeError("metadata observer write is denied")
    if event == "import" and args:
        name = str(args[0] or "")
        if any(
            name == item or name.startswith(item + ".")
            for item in _lp_blocked_imports
        ):
            raise RuntimeError("metadata observer import is denied")
_lp_sys.addaudithook(_lp_audit)

_lp_entry_module, _lp_callable_name = _lp_sys.argv[1:3]
_lp_source_root = _lp_Path(
    _lp_os.environ["LEADPOET_MODEL_SOURCE_ROOT"]
).resolve(strict=True)
if not _lp_source_root.is_dir():
    raise RuntimeError("metadata source root is invalid")
_lp_sys.path.insert(0, str(_lp_source_root))
_lp_adapter_module = _lp_importlib.import_module(_lp_entry_module)
_lp_metadata = getattr(_lp_adapter_module, _lp_callable_name)()
_lp_qualification_outcome_observation = None
if isinstance(_lp_metadata, dict) and _lp_metadata.get(
    "qualification_outcome_protocol"
) is not None:
    _lp_qualification_protocol = _lp_metadata[
        "qualification_outcome_protocol"
    ]
    _lp_outcome_entrypoint_name = _lp_plan.get(
        "qualification_outcome_entrypoint"
    )
    _lp_probe_mode = _lp_plan.get("qualification_outcome_probe_mode")
    _lp_probe_schema_version = _lp_plan.get(
        "qualification_outcome_probe_schema_version"
    )
    if (
        not isinstance(_lp_qualification_protocol, dict)
        or not isinstance(_lp_outcome_entrypoint_name, str)
        or _lp_qualification_protocol.get("entrypoint")
        != _lp_outcome_entrypoint_name
        or not isinstance(_lp_probe_mode, str)
        or not isinstance(_lp_probe_schema_version, str)
    ):
        raise RuntimeError("qualification outcome protocol plan is invalid")
    _lp_qualification_route = _lp_importlib.import_module(
        "sourcing_model.qualification_route"
    )
    if not callable(getattr(_lp_qualification_route, "transport_headers", None)):
        raise RuntimeError("qualification route transport hook is invalid")
    _lp_probe_cases = _lp_plan.get("qualification_outcome_probes")
    if (
        not isinstance(_lp_probe_cases, list)
        or len(_lp_probe_cases) != 2
        or any(
            not isinstance(_lp_case, dict)
            or set(_lp_case) != {"case_id", "nonce"}
            for _lp_case in _lp_probe_cases
        )
    ):
        raise RuntimeError("qualification outcome probe plan is invalid")
    _lp_outcome_entrypoint = getattr(
        _lp_adapter_module, _lp_outcome_entrypoint_name
    )
    _lp_observed_cases = {}
    for _lp_case in _lp_probe_cases:
        with _lp_contextlib.redirect_stdout(_lp_sys.stderr):
            _lp_observed_cases[_lp_case["case_id"]] = _lp_outcome_entrypoint(
                {},
                {
                    "mode": _lp_probe_mode,
                    "probe": {
                        "schema_version": _lp_probe_schema_version,
                        "case_id": _lp_case["case_id"],
                        "nonce": _lp_case["nonce"],
                    },
                },
            )
    _lp_qualification_outcome_observation = {
        "cases": _lp_observed_cases,
    }

_lp_invariant_policy = _lp_plan["runtime_invariants"]
if _lp_invariant_policy == {"profile": "qualification_protocol_v2"}:
    _lp_invariants = {"profile": "qualification_protocol_v2"}
elif _lp_invariant_policy is None:
    _lp_invariants = {"profile": "legacy_exact"}
else:
    _lp_company_policy = dict(_lp_invariant_policy.get("company_fit") or {})
    _lp_capability_policy = dict(
        _lp_invariant_policy.get("runtime_capabilities") or {}
    )
    _lp_company_fit = _lp_importlib.import_module(
        "qualification.scoring.company_fit_decision"
    )
    _lp_discovery = _lp_importlib.import_module("sourcing_model.discovery")
    _lp_orchestrator = _lp_importlib.import_module("sourcing_model.orchestrator")
    _lp_validation = _lp_importlib.import_module("sourcing_model.validation")
    _lp_capabilities = _lp_importlib.import_module(
        "sourcing_model.runtime_capabilities"
    )
    def _lp_capability_metadata():
        value = dict(_lp_capabilities.capability_metadata())
        return {
            "capability_contract_version": value.get(
                "capability_contract_version"
            ),
            "capabilities": sorted(
                str(item) for item in value.get("capabilities") or ()
            ),
            "host_registered": sorted(
                str(item) for item in value.get("host_registered") or ()
            ),
        }
    _lp_company = {
        "identity": dict(_lp_company_fit.company_fit_decision_contract_identity()),
        "strict_boolean": {
            str(case["case_id"]): _lp_company_fit.strict_company_fit_boolean(
                case.get("input")
            )
            for case in _lp_company_policy.get("strict_boolean_cases") or ()
        },
        "reconcile": {
            str(case["case_id"]): _lp_company_fit.reconcile_company_fit_decisions(
                list(case.get("decisions") or ())
            )
            for case in _lp_company_policy.get("reconcile_cases") or ()
        },
        "aggregate": {
            str(case["case_id"]): _lp_company_fit.aggregate_company_fit_decisions(
                dict(case.get("decisions") or {}),
                stage_required=bool(case.get("stage_required")),
            )
            for case in _lp_company_policy.get("aggregate_cases") or ()
        },
    }
    _lp_dependency_policy = dict(
        _lp_invariant_policy.get("adapter_dependencies") or {}
    )
    _lp_build_query_probe = dict(
        _lp_dependency_policy.get("build_query") or {}
    )
    _lp_industry_run = _lp_validation.first_party_industry_run()
    _lp_flow_mode = _lp_orchestrator.flow_mode()
    _lp_build_query = _lp_discovery.build_query(
        dict(_lp_build_query_probe.get("icp") or {}),
        str(_lp_build_query_probe.get("source") or ""),
    )
    _lp_adapter_dependencies = {
        "build_query_returns_string": isinstance(_lp_build_query, str),
        "first_party_industry_run_is_context_manager": (
            callable(getattr(_lp_industry_run, "__enter__", None))
            and callable(getattr(_lp_industry_run, "__exit__", None))
        ),
        "flow_mode_is_supported": _lp_flow_mode in set(
            str(item) for item in _lp_dependency_policy.get("flow_modes") or ()
        ),
    }
    _lp_sentinel = dict(_lp_capability_policy.get("sentinel") or {})
    _lp_http_token = object()
    _lp_emit_events = []
    _lp_http_calls = []
    _lp_resolve_calls = []
    _lp_probe_calls = []
    def _lp_http_fetch(url, **kwargs):
        _lp_http_calls.append({"url": url, **dict(kwargs)})
        return _lp_http_token
    def _lp_resolve(name):
        _lp_resolve_calls.append(name)
        return _lp_capabilities.HostResolution.TIMEOUT
    def _lp_probe(host):
        _lp_probe_calls.append(host)
        return _lp_capabilities.OriginReachability.UNKNOWN
    _lp_initial_metadata = _lp_capability_metadata()
    _lp_unknown_rejected = False
    _lp_noncallable_rejected = False
    _lp_capabilities.reset()
    try:
        try:
            _lp_capabilities.register("__consumer_probe_unknown__", lambda: None)
        except KeyError:
            _lp_unknown_rejected = True
        try:
            _lp_capabilities.register("emit", None)
        except TypeError:
            _lp_noncallable_rejected = True
        _lp_capabilities.register(
            "deadline", lambda: _lp_sentinel.get("deadline")
        )
        _lp_capabilities.register(
            "emit", lambda event: _lp_emit_events.append(dict(event))
        )
        _lp_capabilities.register("http_fetch", _lp_http_fetch)
        _lp_capabilities.register("resolve_host", _lp_resolve)
        _lp_capabilities.register("probe_origin", _lp_probe)
        _lp_registered_metadata = _lp_capability_metadata()
        _lp_deadline = _lp_capabilities.deadline()
        _lp_capabilities.emit(dict(_lp_sentinel.get("emit_event") or {}))
        _lp_http_result = _lp_capabilities.http_fetch(
            str(_lp_sentinel.get("http_url") or ""),
            timeout=_lp_sentinel.get("http_timeout"),
            max_bytes=_lp_sentinel.get("http_max_bytes"),
            accept=_lp_sentinel.get("http_accept"),
        )
        _lp_resolve_result = _lp_capabilities.resolve_host(
            str(_lp_sentinel.get("resolve_input") or "")
        )
        _lp_probe_result = _lp_capabilities.probe_origin(
            str(_lp_sentinel.get("probe_input") or "")
        )
    finally:
        _lp_capabilities.reset()
    _lp_after_reset_metadata = _lp_capability_metadata()
    _lp_capabilities.default_emit({"consumer_probe": True})
    _lp_default_probe = _lp_capabilities.default_probe_origin("invalid.example")
    _lp_invariants = {
        "profile": "semantic_v1",
        "schema_version": _lp_invariant_policy.get("schema_version"),
        "adapter_dependencies": _lp_adapter_dependencies,
        "company_fit": _lp_company,
        "runtime_capabilities": {
            "initial_metadata": _lp_initial_metadata,
            "unknown_registration_rejected": _lp_unknown_rejected,
            "noncallable_registration_rejected": _lp_noncallable_rejected,
            "registered_metadata": _lp_registered_metadata,
            "dispatch": {
                "deadline": _lp_deadline,
                "emit_events": _lp_emit_events,
                "http_fetch_result_is_sentinel": (
                    _lp_http_result is _lp_http_token
                ),
                "http_fetch_calls": _lp_http_calls,
                "resolve_calls": _lp_resolve_calls,
                "resolve_result": getattr(
                    _lp_resolve_result, "value", _lp_resolve_result
                ),
                "probe_calls": _lp_probe_calls,
                "probe_result": getattr(
                    _lp_probe_result, "value", _lp_probe_result
                ),
            },
            "after_reset_metadata": _lp_after_reset_metadata,
            "defaults": {
                "deadline_is_none": _lp_capabilities.default_deadline() is None,
                "emit_succeeded": True,
                "http_fetch_identity": (
                    _lp_capabilities.capability("http_fetch")
                    is _lp_capabilities.default_http_fetch
                ),
                "resolve_host_identity": (
                    _lp_capabilities.capability("resolve_host")
                    is _lp_capabilities.default_resolve_host
                ),
                "probe_origin_result": getattr(
                    _lp_default_probe, "value", _lp_default_probe
                ),
                "may_attempt_unknown": _lp_capabilities.may_attempt(
                    _lp_capabilities.OriginReachability.UNKNOWN
                ),
                "timeout_is_terminal": _lp_capabilities.is_terminally_unresolvable(
                    _lp_capabilities.HostResolution.TIMEOUT
                ),
                "nxdomain_is_terminal": _lp_capabilities.is_terminally_unresolvable(
                    _lp_capabilities.HostResolution.NXDOMAIN
                ),
            },
        },
    }

_lp_runtime_observation = {
    "invariants": _lp_invariants,
    "qualification_outcome_protocol": _lp_qualification_outcome_observation,
}
_lp_sys.stdout.write(_lp_json.dumps(
    {
        "metadata": _lp_metadata,
        "runtime_observation": _lp_runtime_observation,
    },
    sort_keys=True,
    separators=(",", ":"),
))
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
    if value.get("operation") == "metadata" and normalized_environment:
        raise ModelSandboxV2Error("model metadata environment must be empty")
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
    if value.get("operation") == "metadata":
        if evidence_mode:
            raise ModelSandboxV2Error(
                "model metadata provider evidence mode must be empty"
            )
    elif evidence_mode not in {"live", "cache_live", "record", "frozen"}:
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
    elif (
        value.get("input") != {}
        or cache_ref
        or normalized_evidence_cache
        or normalized_snapshot_bundle
    ):
        raise ModelSandboxV2Error(
            "metadata request has input, provider evidence, or snapshot state"
        )
    encoded_input = canonical_json(value.get("input")).encode("utf-8")
    if len(encoded_input) > MAX_MODEL_INPUT_BYTES:
        raise ModelSandboxV2Error("model sandbox input exceeds limit")
    scope = str(value.get("provider_cost_scope") or "").lower()
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
    if value.get("operation") == "metadata":
        if scope or cost_cap_microusd or provider_call_cap:
            raise ModelSandboxV2Error(
                "model metadata provider cost state must be empty"
            )
    elif not _HASH_RE.fullmatch(scope):
        raise ModelSandboxV2Error("provider cost scope is invalid")
    elif evidence_mode in {"record", "frozen"}:
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
    catalog_evidence = value.get("provider_catalog_evidence")
    if value.get("operation") == "metadata":
        if value.get("provider_runtime_catalog") != {} or catalog_evidence != {}:
            raise ModelSandboxV2Error(
                "model metadata provider catalog state must be empty"
            )
        provider_runtime_catalog: dict[str, Any] = {}
        normalized_catalog_evidence: dict[str, Any] = {}
    else:
        try:
            provider_runtime_catalog = validate_source_add_runtime_catalog_v2(
                value.get("provider_runtime_catalog") or {}
            )
        except Exception as exc:
            raise ModelSandboxV2Error(
                "model sandbox provider runtime catalog is invalid"
            ) from exc
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
        normalized_catalog_evidence = {
            "result": dict(catalog_result),
            "root_receipt_hash": root_receipt_hash,
        }
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
        "provider_catalog_evidence": normalized_catalog_evidence,
    }


def _local_provider_replay_resolver_v2(
    *,
    evidence_mode: str,
    evidence_cache: Mapping[str, Any],
    evidence_cache_hash: str,
    snapshot_root: Path | None,
    snapshot_manifest_hash: str,
) -> Callable[[Mapping[str, Any], str], Mapping[str, Any]]:
    """Resolve sandbox replay hints from exact host-bound evidence only."""

    normalized_mode = str(evidence_mode or "")
    if normalized_mode not in {"live", "cache_live", "record", "frozen"}:
        raise ModelSandboxV2Error("provider replay mode is invalid")
    cache = dict(evidence_cache)
    if evidence_cache_hash != sha256_json(cache):
        raise ModelSandboxV2Error("provider replay cache authority differs")
    if snapshot_root is None:
        if snapshot_manifest_hash:
            raise ModelSandboxV2Error("provider replay snapshot authority differs")
        snapshot_store = None
    else:
        if not _HASH_RE.fullmatch(str(snapshot_manifest_hash or "")):
            raise ModelSandboxV2Error("provider replay snapshot authority is invalid")
        snapshot_store = ProviderSnapshotStore(
            str(snapshot_root),
            mode=MODE_REPLAY,
        )
        snapshot_manifest = snapshot_store.load_manifest()
        snapshot_verification = snapshot_store.verify_manifest(
            snapshot_manifest
        )
        if (
            not isinstance(snapshot_manifest, Mapping)
            or snapshot_manifest.get("manifest_hash")
            != snapshot_manifest_hash
            or not snapshot_verification.get("passed")
        ):
            raise ModelSandboxV2Error(
                "provider replay snapshot authority differs"
            )

    def resolve(
        request: Mapping[str, Any],
        replay_kind: str,
    ) -> Mapping[str, Any]:
        method = str(request.get("method") or "").upper()
        url = str(request.get("url") or "")
        body = request.get("body")
        if not isinstance(body, bytes):
            raise ModelSandboxV2Error("provider replay request body is invalid")
        selected_kind = ""
        selected: dict[str, Any] | None = None
        authority_hash = ""
        if snapshot_store is not None:
            try:
                response = snapshot_store.replay(method, url, body=body)
            except SnapshotMiss:
                response = None
            if response is not None:
                selected_kind = "snapshot"
                selected = {
                    "terminal_status": "attested_local_response",
                    "http_status": int(response.get("status") or 0),
                    "headers": dict(response.get("headers") or {}),
                    "body_b64": base64.b64encode(
                        str(response.get("body_text") or "").encode("utf-8")
                    ).decode("ascii"),
                    "failure_code": None,
                }
                authority_hash = str(snapshot_manifest_hash)
        if selected is None and normalized_mode != "live":
            fingerprint = canonical_request_fingerprint(method, url, body)
            entries = cache.get("entries")
            record = (
                entries.get(fingerprint)
                if isinstance(entries, Mapping)
                else None
            )
            if isinstance(record, Mapping):
                status = record.get("status")
                encoded_body = record.get("body_b64")
                if (
                    isinstance(status, bool)
                    or not isinstance(status, int)
                    or not 100 <= status <= 599
                    or not isinstance(encoded_body, str)
                ):
                    raise ModelSandboxV2Error(
                        "provider replay cache record is invalid"
                    )
                try:
                    base64.b64decode(encoded_body, validate=True)
                except Exception as exc:
                    raise ModelSandboxV2Error(
                        "provider replay cache body is invalid"
                    ) from exc
                selected_kind = "provider_evidence_cache"
                selected = {
                    "terminal_status": "attested_local_response",
                    "http_status": status,
                    "headers": {"content-type": "application/json"},
                    "body_b64": encoded_body,
                    "failure_code": None,
                }
                authority_hash = str(evidence_cache_hash)
        if selected is None or selected_kind != str(replay_kind):
            raise ModelSandboxV2Error(
                "provider replay hint has no matching host authority"
            )
        return {
            **selected,
            "local_authority_sha256": authority_hash,
        }

    return resolve


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
        {
            "destination": "/run",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "nodev", "noexec", "mode=755", "size=1048576"],
        },
    ]
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


@dataclass
class _BoundedPipeCapture:
    limit: int
    chunks: list[bytes]
    captured_bytes: int = 0

    def append(self, chunk: bytes) -> bool:
        remaining = max(0, self.limit - self.captured_bytes)
        if remaining:
            bounded = chunk[:remaining]
            self.chunks.append(bounded)
            self.captured_bytes += len(bounded)
        return len(chunk) > remaining

    def value(self) -> bytes:
        return b"".join(self.chunks)


def _drain_bounded_pipe(
    stream: Any,
    *,
    capture: _BoundedPipeCapture,
    overflow: Event,
    io_failure: Event,
) -> None:
    try:
        while True:
            chunk = stream.read(MODEL_SANDBOX_PIPE_CHUNK_BYTES)
            if not chunk:
                break
            raw = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
            if capture.append(raw):
                overflow.set()
    except Exception:
        io_failure.set()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _write_process_input(
    stream: Any,
    payload: bytes,
    *,
    io_failure: Event,
) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except BrokenPipeError:
        pass
    except Exception:
        io_failure.set()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _stop_bounded_process(process: Any, *, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ModelSandboxV2Error(
            "model sandbox process could not be stopped"
        ) from exc


def _run_bounded_metadata_process(
    command: list[str],
    *,
    input_payload: str,
    timeout_seconds: float,
    environment: Mapping[str, str],
    process_factory: Callable[..., Any] = subprocess.Popen,
    stdout_limit: int = MAX_MODEL_METADATA_OUTPUT_BYTES,
    stderr_limit: int = MAX_MODEL_METADATA_DIAGNOSTIC_BYTES,
    termination_grace_seconds: float = MODEL_SANDBOX_TIMEOUT_GRACE_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run metadata while retaining at most the declared output byte caps."""

    if (
        not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or stdout_limit <= 0
        or stderr_limit <= 0
        or termination_grace_seconds <= 0
    ):
        raise ModelSandboxV2Error("model sandbox process bounds are invalid")
    process = process_factory(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env=dict(environment),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _stop_bounded_process(
            process, grace_seconds=termination_grace_seconds
        )
        raise ModelSandboxV2Error("model sandbox process pipes are unavailable")

    stdout_capture = _BoundedPipeCapture(stdout_limit, [])
    stderr_capture = _BoundedPipeCapture(stderr_limit, [])
    stdout_overflow = Event()
    stderr_overflow = Event()
    io_failure = Event()
    stdout_thread = Thread(
        target=_drain_bounded_pipe,
        kwargs={
            "stream": process.stdout,
            "capture": stdout_capture,
            "overflow": stdout_overflow,
            "io_failure": io_failure,
        },
        daemon=True,
        name="leadpoet-metadata-stdout",
    )
    stderr_thread = Thread(
        target=_drain_bounded_pipe,
        kwargs={
            "stream": process.stderr,
            "capture": stderr_capture,
            "overflow": stderr_overflow,
            "io_failure": io_failure,
        },
        daemon=True,
        name="leadpoet-metadata-stderr",
    )
    input_thread = Thread(
        target=_write_process_input,
        args=(process.stdin, input_payload.encode("utf-8")),
        kwargs={"io_failure": io_failure},
        daemon=True,
        name="leadpoet-metadata-stdin",
    )
    deadline = time.monotonic() + float(timeout_seconds)
    monitor_wait = Event()
    timed_out = False
    stopped_early = False
    threads = (stdout_thread, stderr_thread, input_thread)
    started_threads: list[Thread] = []
    execution_error: BaseException | None = None
    stop_error: BaseException | None = None
    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        while process.poll() is None:
            if (
                stdout_overflow.is_set()
                or stderr_overflow.is_set()
                or io_failure.is_set()
            ):
                stopped_early = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                stopped_early = True
                break
            monitor_wait.wait(min(0.02, remaining))
    except BaseException as exc:
        execution_error = exc
        stopped_early = True
    finally:
        try:
            if stopped_early:
                _stop_bounded_process(
                    process, grace_seconds=termination_grace_seconds
                )
            else:
                process.wait()
        except BaseException as exc:
            stop_error = exc
        for thread in started_threads:
            thread.join(termination_grace_seconds)
        live_threads = [thread for thread in started_threads if thread.is_alive()]
        if live_threads:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except Exception:
                    pass
            for thread in live_threads:
                thread.join(termination_grace_seconds)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except Exception:
                pass
    if any(thread.is_alive() for thread in started_threads):
        raise ModelSandboxV2Error(
            "model sandbox process pipes did not close"
        ) from (stop_error or execution_error)
    if stop_error is not None:
        raise stop_error from execution_error
    if execution_error is not None:
        raise execution_error

    stdout_bytes = stdout_capture.value()
    stderr_bytes = stderr_capture.value()
    if stdout_overflow.is_set():
        raise ModelSandboxV2Error("model sandbox output exceeds limit")
    if stderr_overflow.is_set():
        raise ModelSandboxV2Error(
            "model sandbox diagnostic output exceeds limit stderr_prefix_hash=%s"
            % sha256_bytes(stderr_bytes)
        )
    if io_failure.is_set():
        raise ModelSandboxV2Error("model sandbox process pipe failed")
    if timed_out:
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=stdout_bytes,
            stderr=stderr_bytes,
        )
    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelSandboxV2Error(
            "model sandbox process output is not valid UTF-8"
        ) from exc
    return subprocess.CompletedProcess(
        command,
        int(process.returncode),
        stdout=stdout,
        stderr=stderr,
    )


def _bounded_metadata_process_runner(command: list[str], **kwargs: Any):
    if (
        kwargs.get("text") is not True
        or kwargs.get("capture_output") is not True
        or kwargs.get("check") is not False
    ):
        raise ModelSandboxV2Error("metadata process runner contract is invalid")
    return _run_bounded_metadata_process(
        command,
        input_payload=str(kwargs.get("input") or ""),
        timeout_seconds=float(kwargs["timeout"]),
        environment=dict(kwargs.get("env") or {}),
    )


class RunscModelSandboxV2:
    def __init__(
        self,
        *,
        config: RunscSandboxConfigV2,
        transport: BrokeredProviderTransportV2,
        cgroup_parent: str,
        process_runner: Callable[..., Any] = _completed_process_runner,
        metadata_process_runner: Optional[Callable[..., Any]] = None,
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
        self._metadata_process_runner = (
            metadata_process_runner or _bounded_metadata_process_runner
        )
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
                _force_delete_runsc_sandbox(
                    process_runner=self._process_runner,
                    config=self.config,
                    runsc_root=runsc_root,
                    sandbox_id=sandbox_id,
                    failure_event="model_sandbox_self_test_runsc_cleanup_failed",
                )
                listener.close()
                server_thread.join(timeout=5)

            if completed is None or int(completed.returncode) != 0:
                stderr = "" if completed is None else str(completed.stderr or "")
                failure_code, stderr_hash, _exception_class_hash = (
                    _runsc_failure_evidence(
                        stderr,
                        returncode=(
                            None if completed is None else completed.returncode
                        ),
                    )
                )
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
        provider_terminal_observation: dict[str, Any] = {}
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
            try:
                compatibility_receipt = source_tree_compatibility_admission_v1(
                    source_root,
                    manifest=artifact,
                    source_tree_hash=artifact.model_artifact_hash,
                    use_cache=True,
                )
            except ValueError as exc:
                raise ModelSandboxV2Error(str(exc)) from exc
            if compute_private_source_tree_hash(source_root) != artifact.model_artifact_hash:
                raise ModelSandboxV2Error(
                    "model source changed during measured compatibility admission"
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
            if value["operation"] == "metadata":
                result, trace_entries = self._run_metadata_compatibility(
                    value,
                    artifact=artifact,
                    source_root=source_root,
                    tmp_root=tmp_root,
                    job_id=job_id,
                    compatibility_receipt=compatibility_receipt,
                )
            else:
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
                    local_replay_resolver=_local_provider_replay_resolver_v2(
                        evidence_mode=value["provider_evidence_mode"],
                        evidence_cache=value["provider_evidence_cache"],
                        evidence_cache_hash=sha256_json(
                            value["provider_evidence_cache"]
                        ),
                        snapshot_root=provider_snapshot_root,
                        snapshot_manifest_hash=value[
                            "provider_snapshot_manifest_hash"
                        ],
                    ),
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
                        compatibility_receipt=compatibility_receipt,
                    )
                finally:
                    server.close()
                provider_scope.assert_accepted_result_is_complete()
                provider_terminal_observation = (
                    provider_scope.completion_observation()
                )
                if compatibility_receipt.get("admission_mode") == (
                    "qualification_protocol_v2"
                ):
                    _validate_qualification_terminal_observation_v1(
                        result,
                        provider_terminal_observation,
                    )
        consumer_runtime_probe: dict[str, Any] = {}
        if value["operation"] == "metadata":
            if not isinstance(result, Mapping) or set(result) != {
                "metadata",
                "consumer_runtime_probe",
            }:
                raise ModelSandboxV2Error("model metadata result fields are invalid")
            if not isinstance(result["metadata"], Mapping) or not isinstance(
                result["consumer_runtime_probe"], Mapping
            ):
                raise ModelSandboxV2Error("model metadata result is invalid")
            output: Any = dict(result["metadata"])
            consumer_runtime_probe = dict(result["consumer_runtime_probe"])
        else:
            output = result
        output_hash = sha256_json(output)
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
            "compatibility_policy_hash": compatibility_receipt["policy_hash"],
            "compatibility_admission_hash": compatibility_receipt[
                "receipt_hash"
            ],
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
            "provider_runtime_catalog_hash": (
                sha256_json({})
                if value["operation"] == "metadata"
                else value["provider_runtime_catalog"]["catalog_hash"]
            ),
            "generated_provider_evidence_cache_hash": sha256_json(
                generated_evidence_cache
            ),
            "trace_entries_hash": sha256_json(trace_entries),
            "output_hash": output_hash,
            "output": output,
            "trace_entries": trace_entries,
            "generated_provider_evidence_cache": generated_evidence_cache,
            **(
                {
                    "provider_terminal_observation": (
                        provider_terminal_observation
                    ),
                    "provider_terminal_observation_hash": sha256_json(
                        provider_terminal_observation
                    ),
                }
                if value["operation"] == "run_icp"
                else {}
            ),
            **(
                {
                    "consumer_runtime_probe": consumer_runtime_probe,
                    "consumer_runtime_probe_hash": sha256_json(
                        consumer_runtime_probe
                    ),
                }
                if value["operation"] == "metadata"
                else {}
            ),
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
            _force_delete_runsc_sandbox(
                process_runner=self._process_runner,
                config=self.config,
                runsc_root=runsc_root,
                sandbox_id=sandbox_id,
                failure_event="research_lab_dev_provider_replay_cleanup_failed",
            )
        if int(completed.returncode) != 0:
            stderr = str(completed.stderr or "")
            if EVIDENCE_MISS_SENTINEL in stderr:
                fingerprint = stderr.rsplit(EVIDENCE_MISS_SENTINEL, 1)[-1].splitlines()[0]
                raise SnapshotMiss("provider-evidence:" + fingerprint.strip())
            raise _runsc_model_sandbox_error(
                stderr=stderr,
                returncode=completed.returncode,
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
            _force_delete_runsc_sandbox(
                process_runner=self._process_runner,
                config=self.config,
                runsc_root=runsc_root,
                sandbox_id=sandbox_id,
                failure_event="research_lab_dev_replay_runsc_cleanup_failed",
            )
        if int(completed.returncode) != 0:
            stderr = str(completed.stderr or "")
            if SNAPSHOT_MISS_SENTINEL in stderr:
                request_key = stderr.rsplit(SNAPSHOT_MISS_SENTINEL, 1)[-1].splitlines()[0]
                raise SnapshotMiss(request_key.strip())
            raise _runsc_model_sandbox_error(
                stderr=stderr,
                returncode=completed.returncode,
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

    def _run_metadata_compatibility(
        self,
        value: Mapping[str, Any],
        *,
        artifact: PrivateModelArtifactManifest,
        source_root: Path,
        tmp_root: Path,
        job_id: str,
        compatibility_receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        stdin_payload = {
            "observation_plan": _runtime_probe_observation_plan_v1(
                compatibility_receipt
            )
        }
        bundle = tmp_root / "bundle"
        bundle.mkdir(mode=0o700)
        runsc_root = tmp_root / "runsc"
        runsc_root.mkdir(mode=0o700)
        sandbox_id = "lp-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        config_doc = _oci_config(
            config=self.config,
            source_root=source_root,
            broker_root=None,
            process_args=[
                self.config.python_path,
                "-I",
                "-B",
                "-c",
                _MEASURED_METADATA_BOOTSTRAP,
                value["module_name"],
                value["callable_name"],
            ],
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
            host_uds=False,
        )
        try:
            completed = self._metadata_process_runner(
                command,
                input=canonical_json(stdin_payload),
                text=True,
                capture_output=True,
                timeout=MODEL_SANDBOX_METADATA_TIMEOUT_SECONDS,
                env={
                    "HOME": str(tmp_root),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelSandboxV2Error("model sandbox timed out") from exc
        finally:
            _force_delete_runsc_sandbox(
                process_runner=self._process_runner,
                config=self.config,
                runsc_root=runsc_root,
                sandbox_id=sandbox_id,
                failure_event="model_sandbox_runsc_cleanup_failed",
            )
        if int(completed.returncode) != 0:
            raise _runsc_model_sandbox_error(
                stderr=str(completed.stderr or ""),
                returncode=completed.returncode,
            )
        if (
            len(str(completed.stdout).encode("utf-8"))
            > MAX_MODEL_METADATA_OUTPUT_BYTES
        ):
            raise ModelSandboxV2Error("model sandbox output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error("model sandbox output is invalid JSON") from exc
        if not isinstance(decoded, Mapping) or set(decoded) != {
            "metadata",
            "runtime_observation",
        }:
            raise ModelSandboxV2Error("model metadata output fields are invalid")
        raw_metadata = decoded.get("metadata")
        raw_observation = decoded.get("runtime_observation")
        if not isinstance(raw_metadata, Mapping) or not isinstance(
            raw_observation, Mapping
        ):
            raise ModelSandboxV2Error("model metadata output is invalid")
        try:
            metadata = validate_sourcing_adapter_metadata(
                raw_metadata,
                expected_semantic_bindings=dict(
                    compatibility_receipt.get("bindings") or {}
                ),
                require_company_fit_contract=(
                    compatibility_receipt.get("admission_mode") == "semantic_v1"
                ),
            )
        except PrivateModelRuntimeError as exc:
            raise ModelSandboxV2Error(
                "model metadata differs from measured compatibility admission"
            ) from exc
        probe = _build_consumer_runtime_probe_from_observation_v1(
            raw_observation,
            compatibility_receipt=compatibility_receipt,
            metadata=metadata,
            expected_source_tree_hash=artifact.model_artifact_hash,
            expected_manifest_hash=artifact.manifest_hash,
            expected_image_digest=artifact.image_digest,
            expected_module_name=value["module_name"],
            expected_callable_name=value["callable_name"],
            observation_plan=stdin_payload["observation_plan"],
        )
        return {"metadata": metadata, "consumer_runtime_probe": probe}, []

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
        compatibility_receipt: Mapping[str, Any],
    ) -> tuple[Any, list[dict[str, Any]]]:
        if value["operation"] != "run_icp":
            raise ModelSandboxV2Error("model sandbox run operation is invalid")
        raw_input = value["input"]
        if not isinstance(raw_input, Mapping) or set(raw_input) != {"icp", "context"}:
            raise ModelSandboxV2Error("model run input fields are invalid")
        stdin_payload = {
            "icp": canonicalize_private_model_icp(raw_input["icp"]),
            "context": dict(raw_input["context"]),
        }
        bootstrap = (
            "from gateway.tee.sandbox_http_shim_v2 import install as _lp_install;\n"
            + trusted_model_sandbox_import_bootstrap()
            + "_lp_install();\n"
            + model_source_import_bootstrap()
            + _model_adapter_bootstrap_for_compatibility_receipt_v1(
                compatibility_receipt,
                artifact=artifact,
            )
        )
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
            **(
                {"LEADPOET_QUALIFICATION_PROTOCOL_V2": "1"}
                if compatibility_receipt.get("admission_mode")
                == "qualification_protocol_v2"
                else {}
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
            _force_delete_runsc_sandbox(
                process_runner=self._process_runner,
                config=self.config,
                runsc_root=runsc_root,
                sandbox_id=sandbox_id,
                failure_event="model_sandbox_runsc_cleanup_failed",
            )
        if int(completed.returncode) != 0:
            raise _runsc_model_sandbox_error(
                stderr=str(completed.stderr or ""),
                returncode=completed.returncode,
            )
        if len(str(completed.stdout).encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
            raise ModelSandboxV2Error("model sandbox output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error("model sandbox output is invalid JSON") from exc
        stderr = str(completed.stderr or "")
        if compatibility_receipt.get("admission_mode") == (
            "qualification_protocol_v2"
        ):
            try:
                envelope = validate_qualification_outcome_envelope_v2(
                    decoded
                )
            except PrivateModelRuntimeError as exc:
                raise ModelSandboxV2Error(
                    "model qualification outcome envelope is invalid"
                ) from exc
            if envelope["route_completion_receipt"].get("probe") is not None:
                raise ModelSandboxV2Error(
                    "model qualification runtime returned probe authority"
                )
            expected_invocation_sha256 = hashlib.sha256(
                canonical_json(stdin_payload).encode("utf-8")
            ).hexdigest()
            if (
                envelope["route_completion_receipt"].get(
                    "invocation_sha256"
                )
                != expected_invocation_sha256
            ):
                raise ModelSandboxV2Error(
                    "model qualification outcome invocation differs"
                )
            return envelope, [
                *parse_incontainer_trace_lines(stderr),
                *parse_sourcing_runtime_lines(stderr),
            ]
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
