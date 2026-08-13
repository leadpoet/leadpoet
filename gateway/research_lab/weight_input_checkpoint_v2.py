"""Crash-safe gateway checkpoint for one exact V2 weight-input request.

The checkpoint stores public, attested receipt material only. It is not a
cache authority: callers must still pass the result through the validator
enclave. Its purpose is to make an already completed measured reconstruction
survive gateway process replacement without another Supabase source read. A
checkpoint from an earlier release can be replayed only when its attested
receipt set identifies that exact producer release. The validator enclave
still validates all receipt and release proofs before it uses the result.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
)
from gateway.research_lab.weight_input_authorization_v2 import (
    validate_gateway_weight_release_identity_v2,
)


CHECKPOINT_SCHEMA_VERSION = "leadpoet.gateway_weight_input_checkpoint.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESULT_FIELDS = {
    "input_receipt_hashes",
    "gateway_authority_event_hash",
    "upstream_receipt_set",
    "compact_ancestry",
}
_RECEIPT_SET_FIELDS = {
    "boot_identities",
    "receipts",
    "transport_attempts",
    "host_operations",
}
_STORE_LOCK_NAME = ".weight-input-store-v2.lock"
_DEFAULT_MIN_FREE_BYTES = 256 * 1024 * 1024


class WeightInputCheckpointV2Error(RuntimeError):
    """A local checkpoint is incomplete, conflicting, or corrupt."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise WeightInputCheckpointV2Error("%s is invalid" % field)
    return normalized


def _canonical_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WeightInputCheckpointV2Error("%s is invalid" % field)
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise WeightInputCheckpointV2Error(
            "%s is not canonical JSON" % field
        ) from exc
    if not isinstance(normalized, dict):
        raise WeightInputCheckpointV2Error("%s is invalid" % field)
    return normalized


def _validate_result(value: Any) -> Dict[str, Any]:
    result = _canonical_object(value, "checkpoint result")
    if set(result) != _RESULT_FIELDS:
        raise WeightInputCheckpointV2Error("checkpoint result fields are invalid")
    input_hashes = result.get("input_receipt_hashes")
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    ):
        raise WeightInputCheckpointV2Error(
            "checkpoint input receipt categories are incomplete"
        )
    normalized_hashes = {
        category: _hash(input_hashes[category], "%s receipt hash" % category)
        for category in sorted(input_hashes)
    }
    if len(set(normalized_hashes.values())) != len(normalized_hashes):
        raise WeightInputCheckpointV2Error(
            "checkpoint input receipt hashes are not unique"
        )
    receipt_set = result.get("upstream_receipt_set")
    if (
        not isinstance(receipt_set, Mapping)
        or set(receipt_set) != _RECEIPT_SET_FIELDS
        or any(not isinstance(receipt_set[field], list) for field in receipt_set)
    ):
        raise WeightInputCheckpointV2Error(
            "checkpoint receipt set is invalid"
        )
    compact = result.get("compact_ancestry")
    if compact is not None:
        if not isinstance(compact, Mapping) or set(compact) != {
            "upstream_ancestry_proofs",
            "upstream_transport_attempts",
        }:
            raise WeightInputCheckpointV2Error(
                "checkpoint compact ancestry is invalid"
            )
        proofs = compact.get("upstream_ancestry_proofs")
        attempts = compact.get("upstream_transport_attempts")
        if (
            not isinstance(proofs, Mapping)
            or set(proofs) != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
            or not isinstance(attempts, list)
        ):
            raise WeightInputCheckpointV2Error(
                "checkpoint compact ancestry is incomplete"
            )
    result["input_receipt_hashes"] = normalized_hashes
    result["gateway_authority_event_hash"] = _hash(
        result.get("gateway_authority_event_hash"),
        "gateway authority event hash",
    )
    return result


def _producer_release_is_in_receipt_set(
    *,
    release_identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    """Require cross-release replay to identify its original measured build."""

    producer = validate_gateway_weight_release_identity_v2(release_identity)
    receipt_set = result.get("upstream_receipt_set")
    identities = (
        receipt_set.get("boot_identities")
        if isinstance(receipt_set, Mapping)
        else None
    )
    if not isinstance(identities, list):
        return False
    expected = {
        "role": producer["service_role"],
        "physical_role": producer["physical_role"],
        "commit_sha": producer["commit_sha"],
        "pcr0": producer["pcr0"],
        "build_manifest_hash": producer["build_manifest_hash"],
        "dependency_lock_hash": producer["dependency_lock_hash"],
    }
    return any(
        isinstance(identity, Mapping)
        and all(identity.get(field) == value for field, value in expected.items())
        for identity in identities
    )


def validate_weight_input_checkpoint_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_fields = {
        "schema_version",
        "release_identity",
        "request_hash",
        "netuid",
        "epoch_id",
        "allocation_hash",
        "calculation_snapshot_hash",
        "leaderboard_window_start",
        "leaderboard_window_end",
        "result",
        "result_hash",
        "created_at",
        "checkpoint_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise WeightInputCheckpointV2Error("checkpoint fields are invalid")
    if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise WeightInputCheckpointV2Error("checkpoint schema is invalid")
    release_identity = validate_gateway_weight_release_identity_v2(
        value.get("release_identity")
    )
    request_hash = _hash(value.get("request_hash"), "request hash")
    allocation_hash = _hash(value.get("allocation_hash"), "allocation hash")
    calculation_hash = _hash(
        value.get("calculation_snapshot_hash"),
        "calculation snapshot hash",
    )
    netuid = value.get("netuid")
    epoch_id = value.get("epoch_id")
    if (
        not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid <= 0
        or not isinstance(epoch_id, int)
        or isinstance(epoch_id, bool)
        or epoch_id < 0
    ):
        raise WeightInputCheckpointV2Error("checkpoint epoch scope is invalid")
    for field in ("leaderboard_window_start", "leaderboard_window_end", "created_at"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise WeightInputCheckpointV2Error(
                "checkpoint %s is invalid" % field.replace("_", " ")
            )
    result = _validate_result(value.get("result"))
    result_hash = _hash(value.get("result_hash"), "result hash")
    if result_hash != sha256_json(result):
        raise WeightInputCheckpointV2Error("checkpoint result hash differs")
    body = {key: value[key] for key in expected_fields if key != "checkpoint_hash"}
    body.update(
        {
            "release_identity": release_identity,
            "request_hash": request_hash,
            "allocation_hash": allocation_hash,
            "calculation_snapshot_hash": calculation_hash,
            "result": result,
            "result_hash": result_hash,
        }
    )
    checkpoint_hash = _hash(value.get("checkpoint_hash"), "checkpoint hash")
    if checkpoint_hash != sha256_json(body):
        raise WeightInputCheckpointV2Error("checkpoint hash differs")
    return {**body, "checkpoint_hash": checkpoint_hash}


class GatewayWeightInputCheckpointStoreV2:
    """Atomically retain immutable checkpoints keyed by signed request hash."""

    def __init__(
        self,
        directory: Path,
        *,
        max_files: Optional[int] = None,
        max_bytes: Optional[int] = None,
        min_free_bytes: int = _DEFAULT_MIN_FREE_BYTES,
    ) -> None:
        self.directory = Path(directory).expanduser()
        for field, value in {
            "max_files": max_files,
            "max_bytes": max_bytes,
        }.items():
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise WeightInputCheckpointV2Error(
                    "checkpoint %s must be positive" % field.replace("_", " ")
                )
        if (
            not isinstance(min_free_bytes, int)
            or isinstance(min_free_bytes, bool)
            or min_free_bytes <= 0
        ):
            raise WeightInputCheckpointV2Error(
                "checkpoint minimum free bytes must be positive"
            )
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self._lock = threading.RLock()
        self._store_lock_local = threading.local()

    @contextmanager
    def _exclusive_store_lock(self):
        """Serialize writers from old and new gateway processes."""

        depth = getattr(self._store_lock_local, "depth", 0)
        if depth:
            self._store_lock_local.depth = depth + 1
            try:
                yield
            finally:
                self._store_lock_local.depth -= 1
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        lock_path = self.directory / _STORE_LOCK_NAME
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._store_lock_local.depth = 1
            yield
        finally:
            self._store_lock_local.depth = 0
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _ensure_storage_capacity(self, encoded_size: int) -> None:
        """Fail closed at configured bounds; never remove authority records."""

        files = []
        for path in self.directory.iterdir():
            if path.name == _STORE_LOCK_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                raise WeightInputCheckpointV2Error(
                    "checkpoint storage inventory contains an invalid entry"
                )
            try:
                files.append(path.lstat().st_size)
            except OSError as exc:
                raise WeightInputCheckpointV2Error(
                    "checkpoint storage inventory failed"
                ) from exc
        if (
            (self.max_files is not None and len(files) + 1 > self.max_files)
            or (
                self.max_bytes is not None
                and sum(files) + encoded_size > self.max_bytes
            )
            or shutil.disk_usage(self.directory).free
            < self.min_free_bytes + encoded_size
        ):
            raise WeightInputCheckpointV2Error(
                "checkpoint storage capacity is insufficient"
            )

    def _path(self, request_hash: str) -> Path:
        digest = _hash(request_hash, "request hash").removeprefix("sha256:")
        return self.directory / (digest + ".json")

    def load(
        self,
        *,
        release_identity: Mapping[str, Any],
        request_hash: str,
        netuid: int,
        epoch_id: int,
        allocation_hash: str,
        calculation_snapshot_hash: str,
        leaderboard_window_start: str,
        leaderboard_window_end: str,
    ) -> Optional[Dict[str, Any]]:
        path = self._path(request_hash)
        with self._lock:
            if not path.exists():
                return None
            try:
                checkpoint = validate_weight_input_checkpoint_v2(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError) as exc:
                raise WeightInputCheckpointV2Error(
                    "checkpoint cannot be read"
                ) from exc
            expected = {
                "request_hash": _hash(request_hash, "request hash"),
                "netuid": int(netuid),
                "epoch_id": int(epoch_id),
                "allocation_hash": _hash(allocation_hash, "allocation hash"),
                "calculation_snapshot_hash": _hash(
                    calculation_snapshot_hash,
                    "calculation snapshot hash",
                ),
                "leaderboard_window_start": str(leaderboard_window_start),
                "leaderboard_window_end": str(leaderboard_window_end),
            }
            if any(checkpoint[field] != expected[field] for field in expected):
                raise WeightInputCheckpointV2Error(
                    "checkpoint differs from the signed request scope"
                )
            current_release = validate_gateway_weight_release_identity_v2(
                release_identity
            )
            if (
                checkpoint["release_identity"] != current_release
                and not _producer_release_is_in_receipt_set(
                    release_identity=checkpoint["release_identity"],
                    result=checkpoint["result"],
                )
            ):
                raise WeightInputCheckpointV2Error(
                    "checkpoint producer release is absent from receipt ancestry"
                )
            return checkpoint

    def persist(
        self,
        *,
        release_identity: Mapping[str, Any],
        request_hash: str,
        netuid: int,
        epoch_id: int,
        allocation_hash: str,
        calculation_snapshot_hash: str,
        leaderboard_window_start: str,
        leaderboard_window_end: str,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        normalized_result = _validate_result(result)
        body = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "release_identity": validate_gateway_weight_release_identity_v2(
                release_identity
            ),
            "request_hash": _hash(request_hash, "request hash"),
            "netuid": int(netuid),
            "epoch_id": int(epoch_id),
            "allocation_hash": _hash(allocation_hash, "allocation hash"),
            "calculation_snapshot_hash": _hash(
                calculation_snapshot_hash,
                "calculation snapshot hash",
            ),
            "leaderboard_window_start": str(leaderboard_window_start),
            "leaderboard_window_end": str(leaderboard_window_end),
            "result": normalized_result,
            "result_hash": sha256_json(normalized_result),
            "created_at": _timestamp(),
        }
        candidate = validate_weight_input_checkpoint_v2(
            {**body, "checkpoint_hash": sha256_json(body)}
        )
        path = self._path(request_hash)
        with self._lock, self._exclusive_store_lock():
            existing = None
            if path.exists():
                existing = self.load(
                    release_identity=release_identity,
                    request_hash=request_hash,
                    netuid=netuid,
                    epoch_id=epoch_id,
                    allocation_hash=allocation_hash,
                    calculation_snapshot_hash=calculation_snapshot_hash,
                    leaderboard_window_start=leaderboard_window_start,
                    leaderboard_window_end=leaderboard_window_end,
                )
            if existing is not None:
                if existing["result"] == candidate["result"]:
                    return existing
                raise WeightInputCheckpointV2Error(
                    "checkpoint conflicts with an existing result"
                )
            encoded = json.dumps(
                candidate,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self._ensure_storage_capacity(len(encoded))
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".%s." % path.name,
                dir=str(self.directory),
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(str(temporary), str(path))
                os.chmod(path, 0o600)
                directory_fd = os.open(str(self.directory), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as exc:
                raise WeightInputCheckpointV2Error(
                    "checkpoint atomic write failed"
                ) from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            readback = self.load(
                release_identity=release_identity,
                request_hash=request_hash,
                netuid=netuid,
                epoch_id=epoch_id,
                allocation_hash=allocation_hash,
                calculation_snapshot_hash=calculation_snapshot_hash,
                leaderboard_window_start=leaderboard_window_start,
                leaderboard_window_end=leaderboard_window_end,
            )
            # Store objects are short-lived per request. A concurrent request
            # can win the atomic replace with the same verified result and a
            # different creation timestamp. Treat that exact result as the
            # durable winner instead of failing a valid submission retry.
            if readback is None or readback["result"] != candidate["result"]:
                raise WeightInputCheckpointV2Error(
                    "checkpoint durable readback differs"
                )
            return readback
