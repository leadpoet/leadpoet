"""Bounded, nonblocking scoring jobs for the existing gateway enclave RPC.

RPC handlers only validate/copy bounded chunks and enqueue sealed jobs. A
dedicated worker thread performs scoring, so existing event/checkpoint calls
remain responsive even when a scoring provider is slow.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_receipts import (
    SCORING_ROLE,
    build_receipt_body,
    create_signed_receipt,
)

from gateway.tee.scoring_executor import (
    ScoringExecutionResult,
    SUPPORTED_OPERATIONS,
    configuration_hash,
    execute_scoring_operation,
    purpose_allowed_for_operation,
)


JOB_SCHEMA_VERSION = "leadpoet.gateway_scoring_job.v1"
MODES = frozenset({"off", "shadow", "required"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
MAX_JOB_COUNT = 64
MAX_QUEUED_JOBS = 16
MAX_SCORING_WORKERS = 5
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
DEFAULT_RESULT_CHUNK_BYTES = 512 * 1024
MAX_RESULT_CHUNK_BYTES = 1024 * 1024
DEFAULT_RETENTION_SECONDS = 3600

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ScoringJobError(ValueError):
    """Raised for invalid, conflicting, or unavailable scoring jobs."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScoringJobError("value is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_hash(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise ScoringJobError("%s must be sha256:<64 lowercase hex>" % field)
    return text


def _require_identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ScoringJobError("%s is not a valid identifier" % field)
    return text


def _normalized_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "schema_version",
        "job_id",
        "operation",
        "purpose",
        "epoch_id",
        "commit_sha",
        "config_hash",
        "payload_sha256",
        "payload_size_bytes",
        "evidence_roots",
        "parent_receipt_hashes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoringJobError("scoring job manifest fields do not match the schema")
    if value.get("schema_version") != JOB_SCHEMA_VERSION:
        raise ScoringJobError("unsupported scoring job schema")
    operation = str(value.get("operation") or "")
    if operation not in SUPPORTED_OPERATIONS:
        raise ScoringJobError("unsupported scoring operation")
    purpose = str(value.get("purpose") or "")
    if not purpose_allowed_for_operation(operation, purpose):
        raise ScoringJobError("purpose is not valid for scoring operation")
    epoch_id = value.get("epoch_id")
    if not isinstance(epoch_id, int) or epoch_id < 0:
        raise ScoringJobError("epoch_id must be a non-negative integer")
    commit_sha = str(value.get("commit_sha") or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit_sha):
        raise ScoringJobError("commit_sha must be a full Git object id")
    size = value.get("payload_size_bytes")
    if not isinstance(size, int) or size < 2 or size > MAX_INPUT_BYTES:
        raise ScoringJobError("payload_size_bytes is outside the allowed range")
    evidence_value = value.get("evidence_roots")
    if not isinstance(evidence_value, Mapping):
        raise ScoringJobError("evidence_roots must be an object")
    evidence = {
        _require_identifier(name, "evidence root name"): _require_hash(digest, "evidence root")
        for name, digest in sorted(evidence_value.items())
    }
    parent_value = value.get("parent_receipt_hashes")
    if not isinstance(parent_value, list):
        raise ScoringJobError("parent_receipt_hashes must be a list")
    parents = sorted({_require_hash(item, "parent receipt hash") for item in parent_value})
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": _require_identifier(value.get("job_id"), "job_id"),
        "operation": operation,
        "purpose": purpose,
        "epoch_id": epoch_id,
        "commit_sha": commit_sha,
        "config_hash": _require_hash(value.get("config_hash"), "config_hash"),
        "payload_sha256": _require_hash(value.get("payload_sha256"), "payload_sha256"),
        "payload_size_bytes": size,
        "evidence_roots": evidence,
        "parent_receipt_hashes": parents,
    }


class ScoringJobManager:
    def __init__(
        self,
        *,
        build_manifest_hash: str,
        commit_sha: str,
        signer: Callable[[bytes], Any],
        public_key_supplier: Callable[[], str],
        attestation_supplier: Callable[[Mapping[str, Any]], str],
        executor: Callable[[str, Mapping[str, Any]], Any] = execute_scoring_operation,
        config_hash_supplier: Callable[[], str] = configuration_hash,
        mode: Optional[str] = None,
        worker_count: Optional[int] = None,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        configured_mode = str(mode or os.getenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", "off")).strip().lower()
        if configured_mode not in MODES:
            raise ScoringJobError("invalid attested scoring mode")
        self.mode = configured_mode
        self.build_manifest_hash = _require_hash(build_manifest_hash, "build_manifest_hash")
        self.commit_sha = str(commit_sha or "").strip().lower()
        if not _COMMIT_RE.fullmatch(self.commit_sha):
            raise ScoringJobError("scoring manager commit_sha must be a full Git object id")
        self._signer = signer
        self._public_key_supplier = public_key_supplier
        self._attestation_supplier = attestation_supplier
        self._executor = executor
        self._config_hash_supplier = config_hash_supplier
        self._retention_seconds = max(60, int(retention_seconds))
        self._clock = clock
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        configured_workers = worker_count
        if configured_workers is None:
            try:
                configured_workers = int(os.getenv("RESEARCH_LAB_ATTESTED_SCORING_WORKERS", "1"))
            except ValueError:
                configured_workers = 1
        self.worker_count = max(1, min(MAX_SCORING_WORKERS, int(configured_workers)))
        self._active_job_ids = set()
        self._workers = []
        if self.mode != "off":
            for index in range(self.worker_count):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name="gateway-enclave-scoring-%s" % (index + 1),
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            self._purge_expired_locked()
            counts = {}
            for job in self._jobs.values():
                state = str(job["state"])
                counts[state] = counts.get(state, 0) + 1
            active_job_ids = sorted(self._active_job_ids)
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "mode": self.mode,
            "enabled": self.mode != "off",
            "worker_alive": bool(self._workers) and all(worker.is_alive() for worker in self._workers),
            "worker_count": len(self._workers),
            "queue_depth": self._queue.qsize(),
            "active_job_id": active_job_ids[0] if active_job_ids else None,
            "active_job_ids": active_job_ids,
            "job_counts": counts,
            "build_manifest_hash": self.build_manifest_hash,
            "commit_sha": self.commit_sha,
            "config_hash": _require_hash(self._config_hash_supplier(), "config_hash"),
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "limits": {
                "max_jobs": MAX_JOB_COUNT,
                "max_queued_jobs": MAX_QUEUED_JOBS,
                "max_input_bytes": MAX_INPUT_BYTES,
                "max_output_bytes": MAX_OUTPUT_BYTES,
                "max_chunk_bytes": MAX_CHUNK_BYTES,
            },
        }

    def submit(self, manifest: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_enabled()
        normalized = _normalized_manifest(manifest)
        current_config_hash = _require_hash(self._config_hash_supplier(), "config_hash")
        if normalized["config_hash"] != current_config_hash:
            raise ScoringJobError("scoring config hash does not match enclave configuration")
        if normalized["commit_sha"] != self.commit_sha:
            raise ScoringJobError("scoring commit does not match enclave build identity")
        manifest_hash = _sha256(_canonical_json(normalized))
        now = self._clock()
        with self._lock:
            self._purge_expired_locked()
            existing = self._jobs.get(normalized["job_id"])
            if existing is not None:
                if existing["manifest_hash"] != manifest_hash:
                    raise ScoringJobError("job_id already exists with a different manifest")
                return self._summary_locked(existing)
            if len(self._jobs) >= MAX_JOB_COUNT:
                raise ScoringJobError("scoring job capacity is full")
            job = {
                "manifest": normalized,
                "manifest_hash": manifest_hash,
                "state": "uploading",
                "input": bytearray(),
                "result": b"",
                "result_hash": None,
                "receipt": None,
                "error_code": None,
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[normalized["job_id"]] = job
            return self._summary_locked(job)

    def put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data_b64: str,
        chunk_sha256: str,
    ) -> Dict[str, Any]:
        self._require_enabled()
        job_id = _require_identifier(job_id, "job_id")
        if not isinstance(offset, int) or offset < 0:
            raise ScoringJobError("offset must be a non-negative integer")
        try:
            chunk = base64.b64decode(str(data_b64), validate=True)
        except Exception as exc:
            raise ScoringJobError("chunk is not valid base64") from exc
        if not chunk or len(chunk) > MAX_CHUNK_BYTES:
            raise ScoringJobError("chunk size is outside the allowed range")
        if _sha256(chunk) != _require_hash(chunk_sha256, "chunk_sha256"):
            raise ScoringJobError("chunk hash mismatch")
        with self._lock:
            job = self._get_job_locked(job_id)
            if job["state"] != "uploading":
                raise ScoringJobError("job no longer accepts chunks")
            if offset != len(job["input"]):
                raise ScoringJobError("chunk offset does not match uploaded length")
            expected_size = job["manifest"]["payload_size_bytes"]
            if offset + len(chunk) > expected_size:
                raise ScoringJobError("chunk exceeds declared payload size")
            job["input"].extend(chunk)
            job["updated_at"] = self._clock()
            return self._summary_locked(job)

    def seal(self, job_id: str) -> Dict[str, Any]:
        self._require_enabled()
        job_id = _require_identifier(job_id, "job_id")
        with self._lock:
            job = self._get_job_locked(job_id)
            if job["state"] in {"queued", "running"} | TERMINAL_STATES:
                return self._summary_locked(job)
            if job["state"] != "uploading":
                raise ScoringJobError("job is not uploadable")
            payload = bytes(job["input"])
            if len(payload) != job["manifest"]["payload_size_bytes"]:
                raise ScoringJobError("uploaded payload size does not match manifest")
            if _sha256(payload) != job["manifest"]["payload_sha256"]:
                raise ScoringJobError("uploaded payload hash does not match manifest")
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScoringJobError("payload must be a UTF-8 JSON object") from exc
            if not isinstance(decoded, Mapping):
                raise ScoringJobError("payload must be a JSON object")
            if _canonical_json(decoded) != payload:
                raise ScoringJobError("payload must use canonical JSON encoding")
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                raise ScoringJobError("scoring queue is full") from exc
            job["state"] = "queued"
            job["updated_at"] = self._clock()
            return self._summary_locked(job)

    def status(self, job_id: str) -> Dict[str, Any]:
        job_id = _require_identifier(job_id, "job_id")
        with self._lock:
            return self._summary_locked(self._get_job_locked(job_id))

    def cancel(self, job_id: str) -> Dict[str, Any]:
        self._require_enabled()
        job_id = _require_identifier(job_id, "job_id")
        with self._lock:
            job = self._get_job_locked(job_id)
            if job["state"] in TERMINAL_STATES:
                return self._summary_locked(job)
            job["cancel_requested"] = True
            if job["state"] in {"uploading", "queued"}:
                job["state"] = "cancelled"
                job["input"] = bytearray()
            job["updated_at"] = self._clock()
            return self._summary_locked(job)

    def result_chunk(self, *, job_id: str, offset: int = 0, max_bytes: int = DEFAULT_RESULT_CHUNK_BYTES) -> Dict[str, Any]:
        job_id = _require_identifier(job_id, "job_id")
        if not isinstance(offset, int) or offset < 0:
            raise ScoringJobError("offset must be a non-negative integer")
        if not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > MAX_RESULT_CHUNK_BYTES:
            raise ScoringJobError("max_bytes is outside the allowed range")
        with self._lock:
            job = self._get_job_locked(job_id)
            if job["state"] != "succeeded":
                raise ScoringJobError("job result is not available")
            result = job["result"]
            if offset > len(result):
                raise ScoringJobError("result offset exceeds result size")
            chunk = result[offset: offset + max_bytes]
            return {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(chunk).decode("ascii"),
                "chunk_sha256": _sha256(chunk),
                "result_sha256": job["result_hash"],
                "total_size_bytes": len(result),
                "eof": offset + len(chunk) >= len(result),
            }

    def receipt(self, job_id: str) -> Dict[str, Any]:
        job_id = _require_identifier(job_id, "job_id")
        with self._lock:
            job = self._get_job_locked(job_id)
            if job["receipt"] is None:
                raise ScoringJobError("job receipt is not available")
            return dict(job["receipt"])

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute_job(job_id)
            except Exception as exc:
                print(
                    "[TEE] attested scoring worker internal error type=%s" % type(exc).__name__,
                    flush=True,
                )
            finally:
                self._queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] == "cancelled":
                return
            if job["cancel_requested"]:
                job["state"] = "cancelled"
                job["input"] = bytearray()
                job["updated_at"] = self._clock()
                return
            job["state"] = "running"
            job["updated_at"] = self._clock()
            self._active_job_ids.add(job_id)
            manifest = dict(job["manifest"])
            payload_bytes = bytes(job["input"])
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            result = self._executor(manifest["operation"], payload)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            derived_evidence_roots = {}
            if isinstance(result, ScoringExecutionResult):
                derived_evidence_roots = dict(result.evidence_roots)
                result = dict(result.result)
            if not isinstance(result, Mapping):
                raise ScoringJobError("scoring executor result must be an object")
            result_bytes = _canonical_json(dict(result))
            if len(result_bytes) > MAX_OUTPUT_BYTES:
                raise ScoringJobError("scoring result exceeds maximum size")
            receipt = self._create_receipt(
                manifest=manifest,
                output_root=_sha256(result_bytes),
                status="succeeded",
                failure_code=None,
                derived_evidence_roots=derived_evidence_roots,
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job["cancel_requested"]:
                    job["state"] = "cancelled"
                    job["result"] = b""
                    job["result_hash"] = None
                    job["receipt"] = None
                else:
                    job["state"] = "succeeded"
                    job["result"] = result_bytes
                    job["result_hash"] = _sha256(result_bytes)
                    job["receipt"] = receipt
                job["input"] = bytearray()
                job["updated_at"] = self._clock()
        except Exception as exc:
            failure_code = "execution_error"
            try:
                failure_output = _canonical_json({"status": "failed", "failure_code": failure_code})
                receipt = self._create_receipt(
                    manifest=manifest,
                    output_root=_sha256(failure_output),
                    status="failed",
                    failure_code=failure_code,
                    derived_evidence_roots=None,
                )
            except Exception:
                receipt = None
            print(
                "[TEE] attested scoring job failed job_id=%s type=%s" % (job_id, type(exc).__name__),
                flush=True,
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["state"] = "failed"
                    job["error_code"] = failure_code
                    job["receipt"] = receipt
                    job["input"] = bytearray()
                    job["updated_at"] = self._clock()
        finally:
            with self._lock:
                self._active_job_ids.discard(job_id)

    def _create_receipt(
        self,
        *,
        manifest: Mapping[str, Any],
        output_root: str,
        status: str,
        failure_code: Optional[str],
        derived_evidence_roots: Optional[Mapping[str, str]],
    ) -> Dict[str, Any]:
        evidence_roots = dict(manifest["evidence_roots"])
        for name, digest in sorted(dict(derived_evidence_roots or {}).items()):
            normalized_name = _require_identifier(name, "derived evidence root name")
            normalized_digest = _require_hash(digest, "derived evidence root")
            existing = evidence_roots.get(normalized_name)
            if existing is not None and existing != normalized_digest:
                raise ScoringJobError("derived evidence root conflicts with job manifest")
            evidence_roots[normalized_name] = normalized_digest
        body = build_receipt_body(
            role=SCORING_ROLE,
            purpose=manifest["purpose"],
            job_id=manifest["job_id"],
            epoch_id=manifest["epoch_id"],
            commit_sha=manifest["commit_sha"],
            build_manifest_hash=self.build_manifest_hash,
            config_hash=manifest["config_hash"],
            input_root=manifest["payload_sha256"],
            output_root=output_root,
            evidence_roots=evidence_roots,
            parent_receipt_hashes=manifest["parent_receipt_hashes"],
            status=status,
            failure_code=failure_code,
            issued_at=datetime.utcfromtimestamp(self._clock()).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return create_signed_receipt(
            body=body,
            enclave_pubkey=self._public_key_supplier(),
            attestation_document_b64=self._attestation_supplier(manifest),
            sign_digest=self._signer,
        )

    def _summary_locked(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        manifest = job["manifest"]
        return {
            "job_id": manifest["job_id"],
            "operation": manifest["operation"],
            "purpose": manifest["purpose"],
            "state": job["state"],
            "manifest_hash": job["manifest_hash"],
            "uploaded_bytes": len(job["input"]),
            "expected_bytes": manifest["payload_size_bytes"],
            "result_sha256": job["result_hash"],
            "result_size_bytes": len(job["result"]),
            "receipt_hash": (job["receipt"] or {}).get("receipt_hash"),
            "error_code": job["error_code"],
            "cancel_requested": bool(job["cancel_requested"]),
        }

    def _get_job_locked(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise ScoringJobError("scoring job was not found")
        return job

    def _purge_expired_locked(self) -> None:
        cutoff = self._clock() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["state"] in TERMINAL_STATES and job["updated_at"] < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _require_enabled(self) -> None:
        if self.mode == "off":
            raise ScoringJobError("attested scoring is disabled")
