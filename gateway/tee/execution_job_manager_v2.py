"""Bounded V2-only enclave job queue with signed receipts and transitions."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import inspect
import json
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    ROLE_PURPOSES,
    build_execution_receipt_body,
    build_transition_command_body,
    canonical_json,
    create_signed_execution_receipt,
    create_signed_transition_command,
    host_operation_root,
    merkle_root,
    sha256_bytes,
    transport_root,
    validate_boot_identity,
    validate_receipt_graph,
    validate_transport_attempt,
)


JOB_SCHEMA_VERSION = "leadpoet.enclave_execution_job.v2"
PARENT_RECEIPT_GRAPHS_FIELD = "_v2_parent_receipt_graphs"
MAX_JOB_COUNT = 256
MAX_QUEUED_JOBS = 64
MIN_TERMINAL_EVICTION_AGE_SECONDS = 300
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
MAX_RESULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_RESULT_CHUNK_BYTES = 512 * 1024
MAX_EXTERNAL_RECEIPT_GRAPHS = 64
MAX_EXTERNAL_RECEIPT_GRAPH_BYTES = 64 * 1024 * 1024
TERMINAL_STATES = frozenset({"cancelled", "failed", "succeeded"})
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


class ExecutionJobV2Error(RuntimeError):
    """A V2 job is malformed, unmeasured, duplicated, or unavailable."""


@dataclass(frozen=True)
class TransitionSpecV2:
    operation: str
    target: str
    idempotency_key: str
    expected_state_hash: str
    payload_hash: str
    ttl_seconds: int = 300


@dataclass
class ExecutionResultV2:
    output: Mapping[str, Any]
    receipt_output: Optional[Mapping[str, Any]] = None
    transport_attempts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    artifact_hashes: Sequence[str] = field(default_factory=tuple)
    transitions: Sequence[TransitionSpecV2] = field(default_factory=tuple)


@dataclass(frozen=True)
class StageReceiptSpecV2:
    purpose: str
    input_root: str
    output_root: str
    artifact_hashes: Sequence[str] = field(default_factory=tuple)


@dataclass
class ExecutionContextV2:
    job_id: str
    purpose: str
    epoch_id: int
    parent_receipt_hashes: tuple = field(default_factory=tuple)
    provider_credential_profile: str = "default"
    provider_credential_ref_hashes: dict = field(default_factory=dict)
    transport_attempts: list = field(default_factory=list)
    artifact_hashes: list = field(default_factory=list)
    stage_receipts: list = field(default_factory=list)
    external_receipt_graphs: list = field(default_factory=list)
    allowed_failed_receipt_hashes: set = field(default_factory=set)
    host_operation_channel: Any = None
    allowed_purposes: frozenset = frozenset()
    _transport_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _external_receipt_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def record_transport(self, attempt: Mapping[str, Any]) -> None:
        validate_transport_attempt(attempt)
        if attempt["job_id"] != self.job_id or attempt["purpose"] != self.purpose:
            raise ExecutionJobV2Error("transport attempt differs from execution scope")
        expected_credential = self.provider_credential_ref_hashes.get(
            str(attempt.get("provider_id") or "")
        )
        if (
            expected_credential is not None
            and attempt.get("credential_ref_hash") != expected_credential
        ):
            raise ExecutionJobV2Error(
                "transport credential differs from the attested job profile"
            )
        expected_proxy = self.provider_credential_ref_hashes.get("egress_proxy")
        if (
            expected_proxy is not None
            and attempt.get("egress_proxy_ref_hash") != expected_proxy
        ):
            raise ExecutionJobV2Error(
                "transport proxy differs from the attested job profile"
            )
        with self._transport_lock:
            if any(
                item["attempt_hash"] == attempt["attempt_hash"]
                for item in self.transport_attempts
            ):
                raise ExecutionJobV2Error("transport attempt is duplicated")
            self.transport_attempts.append(dict(attempt))

    def record_artifact(self, artifact_hash: str) -> None:
        digest = str(artifact_hash or "").lower()
        if not _HASH_RE.fullmatch(digest):
            raise ExecutionJobV2Error("execution artifact hash is invalid")
        if digest not in self.artifact_hashes:
            self.artifact_hashes.append(digest)

    def record_external_receipt_graph(
        self,
        graph: Mapping[str, Any],
        *,
        allowed_failed_receipt_hashes: Iterable[str] = (),
    ) -> str:
        """Bind a validated nested enclave execution into this job's ancestry."""
        allowed_failed = {
            _hash(value, "allowed failed receipt hash")
            for value in allowed_failed_receipt_hashes
        }
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=allowed_failed,
        )
        encoded = _canonical_bytes(graph)
        if len(encoded) > MAX_EXTERNAL_RECEIPT_GRAPH_BYTES:
            raise ExecutionJobV2Error("external receipt graph exceeds size limit")
        normalized = json.loads(encoded.decode("utf-8"))
        root_hash = _hash(
            normalized.get("root_receipt_hash"),
            "external root receipt hash",
        )
        with self._external_receipt_lock:
            existing = {
                str(item["root_receipt_hash"]): item
                for item in self.external_receipt_graphs
            }
            if root_hash in existing:
                if existing[root_hash] != normalized:
                    raise ExecutionJobV2Error(
                        "external receipt graph conflicts with existing root"
                    )
                return root_hash
            if len(self.external_receipt_graphs) >= MAX_EXTERNAL_RECEIPT_GRAPHS:
                raise ExecutionJobV2Error("external receipt graph count exceeds limit")
            self.external_receipt_graphs.append(normalized)
            self.allowed_failed_receipt_hashes.update(allowed_failed)
        return root_hash

    def external_receipt_roots(self) -> tuple:
        with self._external_receipt_lock:
            return tuple(
                sorted(
                    str(item["root_receipt_hash"])
                    for item in self.external_receipt_graphs
                )
            )

    def execute_host_operation(
        self,
        *,
        operation: str,
        payload: Mapping[str, Any],
        expected_state_hash: str,
        timeout_seconds: int,
        response_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if self.host_operation_channel is None:
            raise ExecutionJobV2Error("host operations are unavailable for this role")
        return self.host_operation_channel.execute(
            operation=operation,
            payload=payload,
            expected_state_hash=expected_state_hash,
            timeout_seconds=timeout_seconds,
            response_validator=response_validator,
        )

    def host_operation_records(self) -> Sequence[Mapping[str, Any]]:
        if self.host_operation_channel is None:
            return ()
        return self.host_operation_channel.complete_ledger()

    def record_stage(
        self,
        *,
        purpose: str,
        input_root: str,
        output_root: str,
        artifact_hashes: Sequence[str] = (),
    ) -> None:
        normalized_purpose = str(purpose or "")
        if normalized_purpose not in self.allowed_purposes:
            raise ExecutionJobV2Error("stage receipt purpose is not authorized")
        normalized_artifacts = tuple(
            _hash(item, "stage artifact hash") for item in artifact_hashes
        )
        self.stage_receipts.append(
            StageReceiptSpecV2(
                purpose=normalized_purpose,
                input_root=_hash(input_root, "stage input root"),
                output_root=_hash(output_root, "stage output root"),
                artifact_hashes=normalized_artifacts,
            )
        )


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ExecutionJobV2Error("%s is invalid" % field)
    return normalized


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ExecutionJobV2Error("%s is invalid" % field)
    return normalized


def _manifest(
    value: Mapping[str, Any],
    *,
    role: str,
    operations: Mapping[str, Iterable[str]],
) -> Dict[str, Any]:
    required = {
        "schema_version",
        "job_id",
        "operation",
        "purpose",
        "epoch_id",
        "sequence",
        "payload_sha256",
        "payload_size_bytes",
        "parent_receipt_hashes",
        "input_artifact_hashes",
        "provider_credential_profile",
        "provider_credential_ref_hashes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ExecutionJobV2Error("V2 job manifest fields are invalid")
    if value["schema_version"] != JOB_SCHEMA_VERSION:
        raise ExecutionJobV2Error("V2 job manifest schema is invalid")
    operation = _identifier(value["operation"], "operation")
    purpose = _identifier(value["purpose"], "purpose")
    allowed = set(operations.get(operation, ()))
    if purpose not in allowed or purpose not in ROLE_PURPOSES.get(role, ()):
        raise ExecutionJobV2Error("operation purpose is not authorized for role")
    epoch_id = value["epoch_id"]
    sequence = value["sequence"]
    size = value["payload_size_bytes"]
    if not isinstance(epoch_id, int) or epoch_id < 0:
        raise ExecutionJobV2Error("epoch_id must be non-negative")
    if not isinstance(sequence, int) or sequence < 0:
        raise ExecutionJobV2Error("sequence must be non-negative")
    if not isinstance(size, int) or size < 2 or size > MAX_INPUT_BYTES:
        raise ExecutionJobV2Error("payload size is outside limit")
    parents = value["parent_receipt_hashes"]
    artifacts = value["input_artifact_hashes"]
    provider_credentials = value["provider_credential_ref_hashes"]
    provider_profile = _identifier(
        value["provider_credential_profile"],
        "provider credential profile",
    )
    if not isinstance(parents, list) or not isinstance(artifacts, list):
        raise ExecutionJobV2Error("job receipt/artifact roots must be arrays")
    if not isinstance(provider_credentials, Mapping):
        raise ExecutionJobV2Error("job provider credential references must be an object")
    normalized_provider_credentials = {}
    for provider_id, digest in provider_credentials.items():
        normalized_provider_credentials[
            _identifier(provider_id, "provider credential provider_id")
        ] = _hash(digest, "provider credential reference")
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": _identifier(value["job_id"], "job_id"),
        "operation": operation,
        "purpose": purpose,
        "epoch_id": epoch_id,
        "sequence": sequence,
        "payload_sha256": _hash(value["payload_sha256"], "payload_sha256"),
        "payload_size_bytes": size,
        "parent_receipt_hashes": sorted(
            {_hash(item, "parent_receipt_hash") for item in parents}
        ),
        "input_artifact_hashes": sorted(
            {_hash(item, "input_artifact_hash") for item in artifacts}
        ),
        "provider_credential_profile": provider_profile,
        "provider_credential_ref_hashes": dict(
            sorted(normalized_provider_credentials.items())
        ),
    }


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class ExecutionJobManagerV2:
    def __init__(
        self,
        *,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        sign_digest: Callable[[bytes], Any],
        operations: Mapping[str, Iterable[str]],
        executor: Callable[[str, Mapping[str, Any], ExecutionContextV2], Any],
        worker_count: int,
        configured_worker_count: Optional[int] = None,
        host_operation_channel_factory: Optional[
            Callable[[str, str], Any]
        ] = None,
        failed_parent_graph_policy: Optional[
            Callable[
                [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
                Iterable[str],
            ]
        ] = None,
        retention_seconds: int = 3600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        boot = dict(boot_identity_supplier())
        validate_boot_identity(boot)
        self.boot_identity = boot
        self.role = str(boot["role"])
        self._boot_identity_supplier = boot_identity_supplier
        self._sign_digest = sign_digest
        self._operations = {
            str(operation): frozenset(str(item) for item in purposes)
            for operation, purposes in operations.items()
        }
        self._executor = executor
        self._host_operation_channel_factory = host_operation_channel_factory
        self._failed_parent_graph_policy = failed_parent_graph_policy
        self._retention_seconds = max(60, int(retention_seconds))
        self._clock = clock
        self._jobs = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self._active = set()
        self._workers = []
        self._terminal_eviction_count = 0
        self._configured_worker_count = (
            int(worker_count)
            if configured_worker_count is None
            else int(configured_worker_count)
        )
        if not 0 <= self._configured_worker_count <= 500:
            raise ValueError("configured worker count is invalid")
        for index in range(max(1, int(worker_count))):
            worker = threading.Thread(
                target=self._worker_loop,
                name="enclave-v2-executor-%s" % (index + 1),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            self._purge_locked()
            counts = {}
            for job in self._jobs.values():
                counts[job["state"]] = counts.get(job["state"], 0) + 1
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "authority": "v2_only",
            "role": self.role,
            "physical_role": self.boot_identity["physical_role"],
            "boot_identity_hash": self.boot_identity["boot_identity_hash"],
            "worker_count": len(self._workers),
            "configured_worker_count": self._configured_worker_count,
            "workers_alive": all(worker.is_alive() for worker in self._workers),
            "queue_depth": self._queue.qsize(),
            "active_job_ids": sorted(self._active),
            "job_counts": counts,
            "terminal_eviction_count": self._terminal_eviction_count,
            "supported_operations": sorted(self._operations),
        }

    def submit(self, manifest: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = _manifest(
            manifest,
            role=self.role,
            operations=self._operations,
        )
        manifest_hash = sha256_bytes(_canonical_bytes(normalized))
        now = self._clock()
        with self._lock:
            self._purge_locked()
            existing = self._jobs.get(normalized["job_id"])
            if existing is not None:
                if existing["manifest_hash"] != manifest_hash:
                    raise ExecutionJobV2Error(
                        "job_id already exists with another manifest"
                    )
                return self._summary(existing)
            if len(self._jobs) >= MAX_JOB_COUNT:
                self._evict_oldest_terminal_locked()
            if len(self._jobs) >= MAX_JOB_COUNT:
                raise ExecutionJobV2Error("V2 job capacity is full")
            job = {
                "manifest": normalized,
                "manifest_hash": manifest_hash,
                "state": "uploading",
                "input": bytearray(),
                "result": b"",
                "result_hash": None,
                "receipt": None,
                "receipts": [],
                "transitions": [],
                "transport_attempts": [],
                "artifact_hashes": [],
                "host_operations": [],
                "external_receipt_graphs": [],
                "host_operation_channel": None,
                "error_code": None,
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[normalized["job_id"]] = job
            return self._summary(job)

    def put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data_b64: str,
        chunk_sha256: str,
    ) -> Dict[str, Any]:
        job_id = _identifier(job_id, "job_id")
        if not isinstance(offset, int) or offset < 0:
            raise ExecutionJobV2Error("chunk offset is invalid")
        try:
            chunk = base64.b64decode(str(data_b64), validate=True)
        except Exception as exc:
            raise ExecutionJobV2Error("chunk is invalid base64") from exc
        if not chunk or len(chunk) > MAX_CHUNK_BYTES:
            raise ExecutionJobV2Error("chunk size is outside limit")
        if sha256_bytes(chunk) != _hash(chunk_sha256, "chunk_sha256"):
            raise ExecutionJobV2Error("chunk hash mismatch")
        with self._lock:
            job = self._job(job_id)
            if job["state"] != "uploading":
                raise ExecutionJobV2Error("job does not accept chunks")
            if offset != len(job["input"]):
                raise ExecutionJobV2Error("chunk offset differs from uploaded length")
            if offset + len(chunk) > job["manifest"]["payload_size_bytes"]:
                raise ExecutionJobV2Error("chunk exceeds declared payload")
            job["input"].extend(chunk)
            job["updated_at"] = self._clock()
            return self._summary(job)

    def seal(self, job_id: str) -> Dict[str, Any]:
        job_id = _identifier(job_id, "job_id")
        with self._lock:
            job = self._job(job_id)
            if job["state"] in {"queued", "running"} | TERMINAL_STATES:
                return self._summary(job)
            payload = bytes(job["input"])
            if len(payload) != job["manifest"]["payload_size_bytes"]:
                raise ExecutionJobV2Error("payload size differs from manifest")
            if sha256_bytes(payload) != job["manifest"]["payload_sha256"]:
                raise ExecutionJobV2Error("payload hash differs from manifest")
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                raise ExecutionJobV2Error("payload must be canonical UTF-8 JSON") from exc
            if not isinstance(decoded, Mapping) or _canonical_bytes(decoded) != payload:
                raise ExecutionJobV2Error("payload must be a canonical JSON object")
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                raise ExecutionJobV2Error("V2 execution queue is full") from exc
            job["state"] = "queued"
            job["updated_at"] = self._clock()
            return self._summary(job)

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._summary(self._job(_identifier(job_id, "job_id")))

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] in TERMINAL_STATES:
                return self._summary(job)
            job["cancel_requested"] = True
            channel = job.get("host_operation_channel")
            if channel is not None:
                channel.close(failure_code="cancelled")
            if job["state"] in {"uploading", "queued"}:
                job["state"] = "cancelled"
                job["input"] = bytearray()
            job["updated_at"] = self._clock()
            return self._summary(job)

    def result_chunk(
        self,
        *,
        job_id: str,
        offset: int = 0,
        max_bytes: int = DEFAULT_RESULT_CHUNK_BYTES,
    ) -> Dict[str, Any]:
        if not isinstance(offset, int) or offset < 0:
            raise ExecutionJobV2Error("result offset is invalid")
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_RESULT_CHUNK_BYTES:
            raise ExecutionJobV2Error("result chunk size is outside limit")
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in {"succeeded", "failed", "cancelled"} or not job[
                "result_hash"
            ]:
                raise ExecutionJobV2Error("job result is unavailable")
            result = job["result"]
            if offset > len(result):
                raise ExecutionJobV2Error("result offset exceeds size")
            chunk = result[offset : offset + max_bytes]
            return {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(chunk).decode("ascii"),
                "chunk_sha256": sha256_bytes(chunk),
                "result_sha256": job["result_hash"],
                "total_size_bytes": len(result),
                "eof": offset + len(chunk) >= len(result),
            }

    def receipt(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            receipt = self._job(_identifier(job_id, "job_id"))["receipt"]
            if receipt is None:
                raise ExecutionJobV2Error("job receipt is unavailable")
            return dict(receipt)

    def receipts(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["receipt"] is None:
                raise ExecutionJobV2Error("job receipts are unavailable")
            return tuple(dict(item) for item in job["receipts"])

    def transitions(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] != "succeeded":
                raise ExecutionJobV2Error("job transitions are unavailable")
            return tuple(dict(item) for item in job["transitions"])

    def transport_attempts(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job transport attempts are unavailable")
            return tuple(dict(item) for item in job["transport_attempts"])

    def artifact_hashes(self, job_id: str) -> Sequence[str]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job artifact hashes are unavailable")
            return tuple(str(item) for item in job["artifact_hashes"])

    def next_host_operation(
        self, *, job_id: str, wait_ms: int = 0
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            channel = job.get("host_operation_channel")
        if channel is None:
            return None
        return channel.next_command(wait_ms=wait_ms)

    def complete_host_operation(
        self,
        *,
        job_id: str,
        request_hash: str,
        terminal_status: str,
        response: Optional[Mapping[str, Any]],
        failure_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            channel = job.get("host_operation_channel")
        if channel is None:
            raise ExecutionJobV2Error("job has no host operation channel")
        return channel.complete(
            request_hash=request_hash,
            terminal_status=terminal_status,
            response=response,
            failure_code=failure_code,
        )

    def host_operations(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job host operations are unavailable")
            return tuple(dict(item) for item in job["host_operations"])

    def external_receipt_graphs(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error(
                    "job external receipt graphs are unavailable"
                )
            return tuple(
                json.loads(_canonical_bytes(item).decode("utf-8"))
                for item in job["external_receipt_graphs"]
            )

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            except Exception as exc:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None and job["state"] not in TERMINAL_STATES:
                        job["state"] = "failed"
                        job["error_code"] = "receipt_unavailable"
                        job["input"] = bytearray()
                        job["updated_at"] = self._clock()
                print(
                    "[TEE] V2 execution worker failed closed job_id=%s type=%s"
                    % (job_id, type(exc).__name__),
                    flush=True,
                )
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] == "cancelled":
                return
            if job["cancel_requested"]:
                job["state"] = "cancelled"
                return
            job["state"] = "running"
            job["updated_at"] = self._clock()
            self._active.add(job_id)
            manifest = dict(job["manifest"])
            payload_bytes = bytes(job["input"])
        context = ExecutionContextV2(
            job_id=job_id,
            purpose=manifest["purpose"],
            epoch_id=manifest["epoch_id"],
            parent_receipt_hashes=tuple(manifest["parent_receipt_hashes"]),
            provider_credential_profile=str(
                manifest["provider_credential_profile"]
            ),
            provider_credential_ref_hashes=dict(
                manifest["provider_credential_ref_hashes"]
            ),
            artifact_hashes=list(manifest["input_artifact_hashes"]),
            allowed_purposes=frozenset(ROLE_PURPOSES[self.role]),
        )
        if self._host_operation_channel_factory is not None:
            context.host_operation_channel = self._host_operation_channel_factory(
                job_id,
                manifest["purpose"],
            )
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current["host_operation_channel"] = context.host_operation_channel
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            parent_graphs = payload.pop(PARENT_RECEIPT_GRAPHS_FIELD, None)
            if parent_graphs is None:
                parent_graphs = []
            if not isinstance(parent_graphs, list) or any(
                not isinstance(graph, Mapping) for graph in parent_graphs
            ):
                raise ExecutionJobV2Error("job parent receipt graphs are invalid")
            parent_roots = []
            for graph in parent_graphs:
                allowed_failed = ()
                if self._failed_parent_graph_policy is not None:
                    allowed_failed = tuple(
                        self._failed_parent_graph_policy(manifest, payload, graph)
                    )
                parent_roots.append(
                    context.record_external_receipt_graph(
                        graph,
                        allowed_failed_receipt_hashes=allowed_failed,
                    )
                )
            if sorted(parent_roots) != sorted(manifest["parent_receipt_hashes"]):
                raise ExecutionJobV2Error(
                    "job parent receipt graphs differ from manifest ancestry"
                )
            value = self._executor(manifest["operation"], payload, context)
            if inspect.isawaitable(value):
                value = asyncio.run(value)
            result = value if isinstance(value, ExecutionResultV2) else ExecutionResultV2(value)
            if not isinstance(result.output, Mapping):
                raise ExecutionJobV2Error("executor output must be an object")
            for attempt in result.transport_attempts:
                context.record_transport(attempt)
            for artifact_hash in result.artifact_hashes:
                context.record_artifact(artifact_hash)
            result_bytes = _canonical_bytes(dict(result.output))
            if len(result_bytes) > MAX_OUTPUT_BYTES:
                raise ExecutionJobV2Error("executor output exceeds size limit")
            receipt_output = result.receipt_output or result.output
            if not isinstance(receipt_output, Mapping):
                raise ExecutionJobV2Error("receipt output must be an object")
            receipt_output_bytes = _canonical_bytes(dict(receipt_output))
            if len(receipt_output_bytes) > MAX_OUTPUT_BYTES:
                raise ExecutionJobV2Error("receipt output exceeds size limit")
            stage_receipts = self._stage_receipts(
                manifest=manifest,
                context=context,
            )
            root_manifest = dict(manifest)
            root_parents = list(manifest["parent_receipt_hashes"])
            if stage_receipts:
                root_parents = [stage_receipts[-1]["receipt_hash"]]
            root_parents.extend(context.external_receipt_roots())
            root_manifest["parent_receipt_hashes"] = sorted(set(root_parents))
            receipt = self._receipt(
                manifest=root_manifest,
                context=context,
                output_root=sha256_bytes(receipt_output_bytes),
                status="succeeded",
                failure_code=None,
            )
            transitions = self._transitions(receipt, result.transitions)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job["cancel_requested"]:
                    cancelled_bytes = _canonical_bytes(
                        {"status": "failed", "failure_code": "cancelled"}
                    )
                    cancelled_receipt = self._receipt(
                        manifest=root_manifest,
                        context=context,
                        output_root=sha256_bytes(cancelled_bytes),
                        status="failed",
                        failure_code="cancelled",
                    )
                    job["state"] = "cancelled"
                    job["result"] = cancelled_bytes
                    job["result_hash"] = sha256_bytes(cancelled_bytes)
                    job["receipt"] = cancelled_receipt
                    job["receipts"] = [cancelled_receipt]
                    job["transitions"] = []
                    job["transport_attempts"] = list(context.transport_attempts)
                    job["artifact_hashes"] = list(context.artifact_hashes)
                    job["host_operations"] = list(
                        context.host_operation_records()
                    )
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                else:
                    job["state"] = "succeeded"
                    job["result"] = result_bytes
                    job["result_hash"] = sha256_bytes(result_bytes)
                    job["receipt"] = receipt
                    job["receipts"] = list(stage_receipts) + [receipt]
                    job["transitions"] = transitions
                    job["transport_attempts"] = list(context.transport_attempts)
                    job["artifact_hashes"] = list(context.artifact_hashes)
                    job["host_operations"] = list(context.host_operation_records())
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                job["input"] = bytearray()
                job["updated_at"] = self._clock()
        except Exception as exc:
            failure_code = "execution_%s" % type(exc).__name__.lower()[:80]
            failure_bytes = _canonical_bytes(
                {"status": "failed", "failure_code": failure_code}
            )
            try:
                failure_manifest = dict(manifest)
                failure_manifest["parent_receipt_hashes"] = sorted(
                    set(manifest["parent_receipt_hashes"])
                    | set(context.external_receipt_roots())
                )
                receipt = self._receipt(
                    manifest=failure_manifest,
                    context=context,
                    output_root=sha256_bytes(failure_bytes),
                    status="failed",
                    failure_code=failure_code,
                )
            except Exception:
                receipt = None
                failure_code = "receipt_unavailable"
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["state"] = "failed"
                    job["error_code"] = failure_code
                    job["result"] = failure_bytes
                    job["result_hash"] = sha256_bytes(failure_bytes)
                    job["receipt"] = receipt
                    job["receipts"] = [receipt] if receipt is not None else []
                    job["transport_attempts"] = list(context.transport_attempts)
                    job["artifact_hashes"] = list(context.artifact_hashes)
                    try:
                        job["host_operations"] = list(
                            context.host_operation_records()
                        )
                    except Exception:
                        job["host_operations"] = []
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                    job["input"] = bytearray()
                    job["updated_at"] = self._clock()
        finally:
            with self._lock:
                self._active.discard(job_id)

    def _receipt(
        self,
        *,
        manifest: Mapping[str, Any],
        context: ExecutionContextV2,
        output_root: str,
        status: str,
        failure_code: Optional[str],
    ) -> Dict[str, Any]:
        current_boot = dict(self._boot_identity_supplier())
        if current_boot != self.boot_identity:
            raise ExecutionJobV2Error("enclave boot identity changed during execution")
        body = build_execution_receipt_body(
            role=self.role,
            purpose=manifest["purpose"],
            job_id=manifest["job_id"],
            epoch_id=manifest["epoch_id"],
            sequence=manifest["sequence"],
            commit_sha=self.boot_identity["commit_sha"],
            pcr0=self.boot_identity["pcr0"],
            build_manifest_hash=self.boot_identity["build_manifest_hash"],
            dependency_lock_hash=self.boot_identity["dependency_lock_hash"],
            config_hash=self.boot_identity["config_hash"],
            boot_identity_hash=self.boot_identity["boot_identity_hash"],
            input_root=manifest["payload_sha256"],
            output_root=output_root,
            transport_root_hash=(
                transport_root(context.transport_attempts)
                if context.transport_attempts
                else EMPTY_TRANSPORT_ROOT
            ),
            host_operation_root_hash=(
                host_operation_root(context.host_operation_records())
                if context.host_operation_channel is not None
                else EMPTY_HOST_OPERATION_ROOT
            ),
            artifact_root=(
                merkle_root(context.artifact_hashes, domain="leadpoet-artifact-v2")
                if context.artifact_hashes
                else EMPTY_ARTIFACT_ROOT
            ),
            parent_receipt_hashes=manifest["parent_receipt_hashes"],
            status=status,
            failure_code=failure_code,
            issued_at=_utc_timestamp(self._clock()),
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=self.boot_identity["signing_pubkey"],
            sign_digest=self._sign_digest,
        )

    def _stage_receipts(
        self,
        *,
        manifest: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Sequence[Dict[str, Any]]:
        if not context.stage_receipts:
            return ()
        issued_at = _utc_timestamp(self._clock())
        parent_hashes = list(manifest["parent_receipt_hashes"])
        output = []
        root_fragment = manifest["payload_sha256"].split(":", 1)[1][:24]
        for index, spec in enumerate(context.stage_receipts):
            body = build_execution_receipt_body(
                role=self.role,
                purpose=spec.purpose,
                job_id="stage:%s:%s" % (root_fragment, index),
                epoch_id=manifest["epoch_id"],
                sequence=index,
                commit_sha=self.boot_identity["commit_sha"],
                pcr0=self.boot_identity["pcr0"],
                build_manifest_hash=self.boot_identity["build_manifest_hash"],
                dependency_lock_hash=self.boot_identity["dependency_lock_hash"],
                config_hash=self.boot_identity["config_hash"],
                boot_identity_hash=self.boot_identity["boot_identity_hash"],
                input_root=spec.input_root,
                output_root=spec.output_root,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=(
                    merkle_root(
                        spec.artifact_hashes,
                        domain="leadpoet-artifact-v2",
                    )
                    if spec.artifact_hashes
                    else EMPTY_ARTIFACT_ROOT
                ),
                parent_receipt_hashes=parent_hashes,
                status="succeeded",
                failure_code=None,
                issued_at=issued_at,
            )
            receipt = create_signed_execution_receipt(
                body=body,
                enclave_pubkey=self.boot_identity["signing_pubkey"],
                sign_digest=self._sign_digest,
            )
            output.append(receipt)
            parent_hashes = [receipt["receipt_hash"]]
        return tuple(output)

    def _transitions(
        self,
        receipt: Mapping[str, Any],
        specs: Sequence[TransitionSpecV2],
    ) -> Sequence[Dict[str, Any]]:
        issued = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        output = []
        for spec in specs:
            expires = issued + timedelta(seconds=max(1, int(spec.ttl_seconds)))
            body = build_transition_command_body(
                operation=spec.operation,
                target=spec.target,
                idempotency_key=spec.idempotency_key,
                expected_state_hash=spec.expected_state_hash,
                payload_hash=spec.payload_hash,
                receipt_hash=receipt["receipt_hash"],
                issued_at=issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at=expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            output.append(
                create_signed_transition_command(
                    body=body,
                    enclave_pubkey=self.boot_identity["signing_pubkey"],
                    sign_digest=self._sign_digest,
                )
            )
        return tuple(output)

    def _summary(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": job["manifest"]["job_id"],
            "operation": job["manifest"]["operation"],
            "purpose": job["manifest"]["purpose"],
            "state": job["state"],
            "manifest_hash": job["manifest_hash"],
            "uploaded_bytes": len(job["input"]),
            "expected_bytes": job["manifest"]["payload_size_bytes"],
            "result_sha256": job["result_hash"],
            "result_size_bytes": len(job["result"]),
            "receipt_hash": (job["receipt"] or {}).get("receipt_hash"),
            "receipt_count": len(job["receipts"]),
            "transition_count": len(job["transitions"]),
            "transport_attempt_count": len(job["transport_attempts"]),
            "artifact_hash_count": len(job["artifact_hashes"]),
            "host_operation_count": len(job["host_operations"]),
            "external_receipt_graph_count": len(job["external_receipt_graphs"]),
            "error_code": job["error_code"],
            "cancel_requested": bool(job["cancel_requested"]),
        }

    def _job(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise ExecutionJobV2Error("V2 job was not found")
        return job

    def _purge_locked(self) -> None:
        cutoff = self._clock() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["state"] in TERMINAL_STATES and job["updated_at"] < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _evict_oldest_terminal_locked(self) -> Optional[str]:
        cutoff = self._clock() - MIN_TERMINAL_EVICTION_AGE_SECONDS
        candidates = [
            (
                float(job["updated_at"]),
                float(job["created_at"]),
                job_id,
            )
            for job_id, job in self._jobs.items()
            if (
                job["state"] in TERMINAL_STATES
                and job_id not in self._active
                and float(job["updated_at"]) <= cutoff
            )
        ]
        if not candidates:
            return None
        _, _, job_id = min(candidates)
        del self._jobs[job_id]
        self._terminal_eviction_count += 1
        return job_id
