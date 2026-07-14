"""Fresh, fail-closed gVisor sandbox for V2 private/candidate model jobs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.sandbox_provider_socket_v2 import SandboxProviderSocketServerV2
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import (
    source_add_placeholder_environment_v2,
    source_add_runtime_retry_hashes_v2,
    validate_source_add_runtime_catalog_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from research_lab.eval import (
    PrivateModelArtifactManifest,
    ensure_private_model_outputs,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import (
    _DOCKER_ADAPTER_BOOTSTRAP,
    _DOCKER_METADATA_BOOTSTRAP,
    canonicalize_private_model_icp,
    parse_incontainer_trace_lines,
    strip_incontainer_trace_lines,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    build_evidence_cache_from_trace_entries,
    icp_evidence_cache_key,
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
MAX_MODEL_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_PROVIDER_EVIDENCE_CACHE_BYTES = 32 * 1024 * 1024
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


class ModelSandboxV2Error(RuntimeError):
    """A model runtime, bundle, provider path, or output failed validation."""


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
        "provider_cost_scope",
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
        "provider_cost_scope": scope,
        "provider_runtime_catalog": provider_runtime_catalog,
        "provider_catalog_evidence": {
            "result": dict(catalog_result),
            "root_receipt_hash": root_receipt_hash,
        },
    }


def _normalize_source_permissions(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _oci_config(
    *,
    config: RunscSandboxConfigV2,
    source_root: Path,
    broker_root: Optional[Path],
    process_args: list[str],
    environment: Mapping[str, str],
    readonly_mounts: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    process_env = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "/app:/workspace/app",
        **dict(environment),
    }
    if broker_root is not None:
        process_env["LEADPOET_SANDBOX_PROVIDER_SOCKET"] = (
            "/run/leadpoet/provider.sock"
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
            "destination": "/workspace/app",
            "type": "bind",
            "source": str(source_root),
            "options": ["rbind", "ro", "nosuid", "nodev"],
        },
    ]
    if broker_root is not None:
        mounts.append(
            {
                "destination": "/run/leadpoet",
                "type": "bind",
                "source": str(broker_root),
                "options": ["rbind", "ro", "nosuid", "nodev", "noexec"],
            }
        )
    for destination, source in sorted((readonly_mounts or {}).items()):
        if (
            not str(destination).startswith("/")
            or ".." in Path(str(destination)).parts
            or not Path(source).is_dir()
        ):
            raise ModelSandboxV2Error("sandbox read-only mount is invalid")
        mounts.append(
            {
                "destination": str(destination),
                "type": "bind",
                "source": str(Path(source).resolve()),
                "options": ["rbind", "ro", "nosuid", "nodev"],
            }
        )
    return {
        "ociVersion": "1.0.2",
        "process": {
            "terminal": False,
            "user": {"uid": config.uid, "gid": config.gid},
            "args": process_args,
            "env": ["%s=%s" % item for item in sorted(process_env.items())],
            "cwd": "/workspace/app",
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
        "linux": {
            "namespaces": [
                {"type": "pid"},
                {"type": "ipc"},
                {"type": "uts"},
                {"type": "mount"},
                {"type": "network"},
                {"type": "user"},
            ],
            "uidMappings": [{"containerID": config.uid, "hostID": config.uid, "size": 1}],
            "gidMappings": [{"containerID": config.gid, "hostID": config.gid, "size": 1}],
            "resources": {
                "memory": {"limit": config.memory_limit_bytes},
                "cpu": {"quota": config.cpu_quota, "period": config.cpu_period},
                "pids": {"limit": config.pids_limit},
            },
            "maskedPaths": [
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
        },
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
        process_runner: Callable[..., Any] = _completed_process_runner,
        utc_day_supplier: Callable[[], str] = lambda: time.strftime(
            "%Y-%m-%d", time.gmtime()
        ),
    ) -> None:
        config.validate()
        self.config = config
        self._transport = transport
        self._process_runner = process_runner
        self._utc_day_supplier = utc_day_supplier

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
        # AF_UNIX paths are limited to roughly 108 bytes on Linux.  A fixed
        # short parent also keeps the mounted broker socket path reproducible.
        with tempfile.TemporaryDirectory(prefix="lp-model-v2-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            source_root = tmp_root / "source"
            source_evidence = extract_source_bundle_v2(
                value["source_bundle"],
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            _normalize_source_permissions(source_root)
            broker_root = tmp_root / "broker"
            broker_root.mkdir(mode=0o700)
            provider_scope = self._transport.create_scope(
                job_id=job_id,
                purpose=purpose,
                logical_operation_id=job_id,
                retry_policy_hashes={
                    **dict(retry_policy_hashes),
                    **source_add_runtime_retry_hashes_v2(
                        value["provider_runtime_catalog"]
                    ),
                },
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
                result, trace_entries = self._run(
                    value,
                    artifact=artifact,
                    source_root=source_root,
                    broker_root=broker_root,
                    tmp_root=tmp_root,
                    job_id=job_id,
                )
            finally:
                server.close()
        output_hash = sha256_json(result)
        generated_evidence_cache = {}
        if (
            value["model_kind"] == "private"
            and value["operation"] == "run_icp"
            and str(dict(value["input"].get("context") or {}).get("mode") or "")
            == "private_baseline"
        ):
            utc_day = str(self._utc_day_supplier() or "")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", utc_day):
                raise ModelSandboxV2Error("provider evidence cache UTC day is invalid")
            generated_evidence_cache = {
                "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                "rolling_window_hash": "",
                "icp_ref": value["provider_evidence_cache_ref"],
                "utc_day": utc_day,
                "entries": build_evidence_cache_from_trace_entries(trace_entries),
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
            "context": dict(context),
        }
        if len(canonical_json(stdin_payload).encode("utf-8")) > MAX_MODEL_INPUT_BYTES:
            raise ModelSandboxV2Error("dev replay input exceeds limit")

        with tempfile.TemporaryDirectory(prefix="lp-dev-replay-v2-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            source_root = tmp_root / "source"
            extract_source_bundle_v2(
                source_bundle,
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            _normalize_source_permissions(source_root)
            return self._run_dev_replay(
                source_root=source_root,
                snapshot_root=Path(snapshot_root),
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
                        "/research_lab_dev_snapshots",
                        miss_policy=miss_policy,
                    ),
                },
                timeout_seconds=normalized_timeout,
                tmp_root=tmp_root,
                job_id=job_id,
            )

    def _run_dev_replay(
        self,
        *,
        source_root: Path,
        snapshot_root: Path,
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
        config_doc = _oci_config(
            config=self.config,
            source_root=source_root,
            broker_root=None,
            process_args=[
                self.config.python_path,
                "-c",
                dev_replay_bootstrap() + _DOCKER_ADAPTER_BOOTSTRAP,
                module_name,
                callable_name,
            ],
            environment=environment,
            readonly_mounts={
                "/research_lab_dev_snapshots": snapshot_root,
            },
        )
        (bundle / "config.json").write_text(
            canonical_json(config_doc), encoding="utf-8"
        )
        sandbox_id = "lp-dev-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        command = [
            str(self.config.runsc_path),
            "--root=%s" % runsc_root,
            "--rootless=true",
            "--network=none",
            "--platform=ptrace",
            "run",
            "--bundle=%s" % bundle,
            sandbox_id,
        ]
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
                "_lp_install();\n" + _DOCKER_ADAPTER_BOOTSTRAP
            )
        else:
            stdin_payload = {}
            bootstrap = _DOCKER_METADATA_BOOTSTRAP
        encoded_input = canonical_json(stdin_payload)
        bundle = tmp_root / "bundle"
        bundle.mkdir(mode=0o700)
        runsc_root = tmp_root / "runsc"
        runsc_root.mkdir(mode=0o700)
        environment = {
            **dict(value["environment"]),
            "RESEARCH_LAB_PROVIDER_COST_SCOPE": value["provider_cost_scope"],
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
                "/run/leadpoet/provider-evidence-cache.json"
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
        )
        (bundle / "config.json").write_text(
            canonical_json(config_doc), encoding="utf-8"
        )
        sandbox_id = "lp-%s-%s" % (
            hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16],
            secrets.token_hex(8),
        )
        command = [
            str(self.config.runsc_path),
            "--root=%s" % runsc_root,
            "--rootless=true",
            "--network=none",
            "--platform=ptrace",
            "run",
            "--bundle=%s" % bundle,
            sandbox_id,
        ]
        try:
            completed = self._process_runner(
                command,
                input=encoded_input,
                text=True,
                capture_output=True,
                timeout=900,
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
            except Exception:
                pass
        if int(completed.returncode) != 0:
            stripped_stderr = strip_incontainer_trace_lines(
                str(completed.stderr or "")
            )
            error_hash = sha256_bytes(stripped_stderr.encode("utf-8"))
            raise ModelSandboxV2Error(
                "model sandbox failed with code %s stderr_hash=%s"
                % (completed.returncode, error_hash)
            )
        if len(str(completed.stdout).encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
            raise ModelSandboxV2Error("model sandbox output exceeds limit")
        try:
            decoded = json.loads(str(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ModelSandboxV2Error("model sandbox output is invalid JSON") from exc
        if operation == "run_icp":
            output = list(
                ensure_private_model_outputs(
                    decoded,
                    context_label="V2 model sandbox",
                    require_non_empty=False,
                )
            )
            return output, parse_incontainer_trace_lines(
                str(completed.stderr or "")
            )
        if not isinstance(decoded, Mapping):
            raise ModelSandboxV2Error("model metadata output must be an object")
        return dict(decoded), parse_incontainer_trace_lines(
            str(completed.stderr or "")
        )
