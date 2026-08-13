"""Crash-safe validator journal for one exact V2 weight-input plan.

The validator writes the complete calculation plan before contacting the
gateway. A restart therefore reuses the same block-bound snapshot and request
instead of creating a new request that cannot use the gateway's durable
checkpoint. The journal contains public authority material only.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
)


JOURNAL_SCHEMA_VERSION = "leadpoet.validator_weight_input_journal.v2"
RELEASE_IDENTITY_FIELDS = frozenset(
    {
        "commit_sha",
        "pcr0",
        "build_manifest_hash",
        "dependency_lock_hash",
        "config_hash",
        "boot_identity_hash",
    }
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_FLOAT_BITS_RE = re.compile(r"^[0-9a-f]{16}$")
_STORE_LOCK_NAME = ".weight-input-journal-v2.lock"
_DEFAULT_MIN_FREE_BYTES = 256 * 1024 * 1024
_JOURNAL_WRITE_RESERVE_BYTES = MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES


class WeightInputJournalV2Error(RuntimeError):
    """The durable weight-input plan is missing, corrupt, or conflicting."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise WeightInputJournalV2Error("%s is invalid" % field)
    return normalized


def _commit(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise WeightInputJournalV2Error("weight input release commit is invalid")
    return normalized


def _validator_hotkey(value: Any) -> str:
    hotkey = str(value or "").strip()
    if (
        not hotkey
        or len(hotkey) > 128
        or any(character.isspace() for character in hotkey)
    ):
        raise WeightInputJournalV2Error(
            "weight input validator hotkey is invalid"
        )
    return hotkey


def _scope_integer(value: Any, field: str, *, positive: bool) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise WeightInputJournalV2Error("weight input %s is invalid" % field)
    if (positive and value <= 0) or (not positive and value < 0):
        raise WeightInputJournalV2Error("weight input %s is invalid" % field)
    return value


def validate_weight_input_release_identity_v2(
    value: Mapping[str, Any],
) -> Dict[str, str]:
    """Validate the complete measured validator release used by the journal."""

    if not isinstance(value, Mapping) or set(value) != RELEASE_IDENTITY_FIELDS:
        raise WeightInputJournalV2Error(
            "weight input release identity fields are invalid"
        )
    pcr0 = str(value.get("pcr0") or "").strip().lower()
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise WeightInputJournalV2Error(
            "weight input release PCR0 is invalid"
        )
    return {
        "commit_sha": _commit(value.get("commit_sha")),
        "pcr0": pcr0,
        "build_manifest_hash": _hash(
            value.get("build_manifest_hash"), "release build manifest hash"
        ),
        "dependency_lock_hash": _hash(
            value.get("dependency_lock_hash"), "release dependency lock hash"
        ),
        "config_hash": _hash(
            value.get("config_hash"), "release config hash"
        ),
        "boot_identity_hash": _hash(
            value.get("boot_identity_hash"), "release boot identity hash"
        ),
    }


def _stable_release_identity_v2(value: Mapping[str, Any]) -> Dict[str, str]:
    """Return release fields that remain fixed across enclave boot rotation."""

    release = validate_weight_input_release_identity_v2(value)
    return {
        key: release[key]
        for key in sorted(release)
        if key != "boot_identity_hash"
    }


def _canonical_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WeightInputJournalV2Error("%s is invalid" % field)
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise WeightInputJournalV2Error(
            "%s is not canonical JSON" % field
        ) from exc
    if not isinstance(normalized, dict):
        raise WeightInputJournalV2Error("%s is invalid" % field)
    return normalized


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(value)).encode("utf-8")


def _canonical_bytes_b64(value: Mapping[str, Any]) -> str:
    return base64.b64encode(_canonical_bytes(value)).decode("ascii")


def _validate_canonical_bytes(
    *,
    value: Mapping[str, Any],
    encoded: Any,
    expected_hash: Any,
    field: str,
) -> tuple[str, str]:
    if not isinstance(encoded, str) or not encoded:
        raise WeightInputJournalV2Error("%s canonical bytes are invalid" % field)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise WeightInputJournalV2Error(
            "%s canonical bytes are invalid" % field
        ) from exc
    if decoded != _canonical_bytes(value):
        raise WeightInputJournalV2Error(
            "%s canonical bytes differ" % field
        )
    normalized_hash = _hash(expected_hash, "%s hash" % field)
    if normalized_hash != sha256_bytes(decoded):
        raise WeightInputJournalV2Error("%s hash differs" % field)
    return base64.b64encode(decoded).decode("ascii"), normalized_hash


def _float_bits(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise WeightInputJournalV2Error("host weights are invalid")
    result = []
    for value in values:
        try:
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError("host weight is not finite")
            bits = struct.pack("!d", normalized).hex()
        except (TypeError, ValueError, OverflowError) as exc:
            raise WeightInputJournalV2Error("host weight is invalid") from exc
        if not _FLOAT_BITS_RE.fullmatch(bits):
            raise WeightInputJournalV2Error("host weight bits are invalid")
        result.append(bits)
    return result


def decode_host_weight_bits_v2(values: Any) -> list[float]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not _FLOAT_BITS_RE.fullmatch(value)
        for value in values
    ):
        raise WeightInputJournalV2Error("host weight bits are invalid")
    decoded = [struct.unpack("!d", bytes.fromhex(value))[0] for value in values]
    if any(not math.isfinite(value) for value in decoded):
        raise WeightInputJournalV2Error("host weight is invalid")
    return decoded


def _metagraph_hash(values: Any) -> str:
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
    ):
        raise WeightInputJournalV2Error("metagraph hotkeys are invalid")
    return sha256_json(list(values))


def _validate_gateway_inputs(value: Any) -> Dict[str, Any]:
    normalized = _canonical_object(value, "gateway inputs")
    compact_fields = {
        "input_receipt_hashes",
        "gateway_authority_event_hash",
        "request_authorization",
        "upstream_ancestry_proofs",
        "upstream_transport_attempts",
    }
    full_fields = {
        "input_receipt_hashes",
        "gateway_authority_event_hash",
        "request_authorization",
        "upstream_receipt_set",
    }
    if set(normalized) not in (compact_fields, full_fields):
        raise WeightInputJournalV2Error("gateway input fields are invalid")
    hashes = normalized.get("input_receipt_hashes")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    ):
        raise WeightInputJournalV2Error(
            "gateway input receipt categories are incomplete"
        )
    normalized_hashes = {
        category: _hash(hashes[category], "%s receipt hash" % category)
        for category in sorted(hashes)
    }
    if len(set(normalized_hashes.values())) != len(normalized_hashes):
        raise WeightInputJournalV2Error(
            "gateway input receipt hashes are not unique"
        )
    normalized["input_receipt_hashes"] = normalized_hashes
    normalized["gateway_authority_event_hash"] = _hash(
        normalized.get("gateway_authority_event_hash"),
        "gateway authority event hash",
    )
    if not isinstance(normalized.get("request_authorization"), Mapping):
        raise WeightInputJournalV2Error(
            "gateway input request authorization is missing"
        )
    if set(normalized) == compact_fields:
        if (
            not isinstance(normalized.get("upstream_ancestry_proofs"), Mapping)
            or set(normalized["upstream_ancestry_proofs"])
            != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
            or not isinstance(normalized.get("upstream_transport_attempts"), list)
        ):
            raise WeightInputJournalV2Error(
                "gateway compact input ancestry is incomplete"
            )
    elif not isinstance(normalized.get("upstream_receipt_set"), Mapping):
        raise WeightInputJournalV2Error("gateway input receipt set is missing")
    return normalized


def validate_weight_input_journal_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "state",
        "revision",
        "release_identity",
        "validator_hotkey",
        "netuid",
        "epoch_id",
        "plan",
        "plan_canonical_bytes_b64",
        "plan_hash",
        "gateway_inputs",
        "gateway_inputs_canonical_bytes_b64",
        "gateway_inputs_hash",
        "created_at",
        "updated_at",
        "journal_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WeightInputJournalV2Error("weight input journal fields are invalid")
    if value.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise WeightInputJournalV2Error("weight input journal schema is invalid")
    state = value.get("state")
    if state not in {"planned", "inputs_verified"}:
        raise WeightInputJournalV2Error("weight input journal state is invalid")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise WeightInputJournalV2Error("weight input journal revision is invalid")
    if (state == "planned" and revision != 0) or (
        state == "inputs_verified" and revision != 1
    ):
        raise WeightInputJournalV2Error(
            "weight input journal revision differs from state"
        )
    release_identity = validate_weight_input_release_identity_v2(
        value.get("release_identity")
    )
    hotkey = _validator_hotkey(value.get("validator_hotkey"))
    netuid = _scope_integer(value.get("netuid"), "netuid", positive=True)
    epoch_id = _scope_integer(value.get("epoch_id"), "epoch", positive=False)
    plan = _canonical_object(value.get("plan"), "weight input plan")
    expected_plan_fields = {
        "calculation_snapshot",
        "calculation_snapshot_hash",
        "host_uids",
        "host_weight_float_bits",
        "allocation_hash",
        "leaderboard_window_start",
        "leaderboard_window_end",
        "metagraph_hash",
    }
    if set(plan) != expected_plan_fields:
        raise WeightInputJournalV2Error("weight input plan fields are invalid")
    calculation = _canonical_object(
        plan.get("calculation_snapshot"),
        "calculation snapshot",
    )
    if (
        calculation.get("netuid") != netuid
        or calculation.get("epoch_id") != epoch_id
        or calculation.get("commit_sha") != release_identity["commit_sha"]
    ):
        raise WeightInputJournalV2Error(
            "calculation snapshot differs from journal scope"
        )
    calculation_hash = _hash(
        plan.get("calculation_snapshot_hash"),
        "calculation snapshot hash",
    )
    if calculation_hash != sha256_json(calculation):
        raise WeightInputJournalV2Error("calculation snapshot hash differs")
    metagraph_hash = _hash(plan.get("metagraph_hash"), "metagraph hash")
    if metagraph_hash != _metagraph_hash(calculation.get("metagraph_hotkeys")):
        raise WeightInputJournalV2Error("metagraph hash differs")
    host_uids = plan.get("host_uids")
    if (
        not isinstance(host_uids, list)
        or any(
            not isinstance(uid, int) or isinstance(uid, bool) or uid < 0
            for uid in host_uids
        )
        or len(host_uids) != len(set(host_uids))
    ):
        raise WeightInputJournalV2Error("host UIDs are invalid")
    weights = decode_host_weight_bits_v2(plan.get("host_weight_float_bits"))
    if len(weights) != len(host_uids):
        raise WeightInputJournalV2Error("host vector lengths differ")
    allocation_hash = _hash(plan.get("allocation_hash"), "allocation hash")
    for field in ("leaderboard_window_start", "leaderboard_window_end"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise WeightInputJournalV2Error(
                "weight input %s is invalid" % field.replace("_", " ")
            )
    normalized_plan = {
        **plan,
        "calculation_snapshot": calculation,
        "calculation_snapshot_hash": calculation_hash,
        "host_uids": host_uids,
        "host_weight_float_bits": _float_bits(weights),
        "allocation_hash": allocation_hash,
        "metagraph_hash": metagraph_hash,
    }
    plan_bytes, plan_hash = _validate_canonical_bytes(
        value=normalized_plan,
        encoded=value.get("plan_canonical_bytes_b64"),
        expected_hash=value.get("plan_hash"),
        field="weight input plan",
    )
    gateway_inputs = value.get("gateway_inputs")
    gateway_inputs_bytes = value.get("gateway_inputs_canonical_bytes_b64")
    gateway_inputs_hash = value.get("gateway_inputs_hash")
    if state == "planned":
        if any(
            item is not None
            for item in (
                gateway_inputs,
                gateway_inputs_bytes,
                gateway_inputs_hash,
            )
        ):
            raise WeightInputJournalV2Error(
                "planned journal contains verified gateway inputs"
            )
        normalized_inputs = None
        normalized_inputs_bytes = None
        normalized_inputs_hash = None
    else:
        normalized_inputs = _validate_gateway_inputs(gateway_inputs)
        normalized_inputs_bytes, normalized_inputs_hash = _validate_canonical_bytes(
            value=normalized_inputs,
            encoded=gateway_inputs_bytes,
            expected_hash=gateway_inputs_hash,
            field="gateway inputs",
        )
    for field in ("created_at", "updated_at"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise WeightInputJournalV2Error(
                "weight input journal %s is invalid" % field.replace("_", " ")
            )
    body = {
        key: value[key]
        for key in fields
        if key != "journal_hash"
    }
    body.update(
        {
            "release_identity": release_identity,
            "validator_hotkey": hotkey,
            "netuid": netuid,
            "epoch_id": epoch_id,
            "plan": normalized_plan,
            "plan_canonical_bytes_b64": plan_bytes,
            "plan_hash": plan_hash,
            "gateway_inputs": normalized_inputs,
            "gateway_inputs_canonical_bytes_b64": normalized_inputs_bytes,
            "gateway_inputs_hash": normalized_inputs_hash,
        }
    )
    journal_hash = _hash(value.get("journal_hash"), "journal hash")
    if journal_hash != sha256_json(body):
        raise WeightInputJournalV2Error("weight input journal hash differs")
    return {**body, "journal_hash": journal_hash}


def require_weight_input_metagraph_match_v2(
    journal: Mapping[str, Any],
    current_metagraph_hotkeys: Any,
) -> Dict[str, Any]:
    """Fail unless verified inputs still use the finalized metagraph order."""

    record = validate_weight_input_journal_v2(journal)
    if record["state"] != "inputs_verified":
        raise WeightInputJournalV2Error(
            "weight input journal has no verified gateway inputs"
        )
    return require_weight_input_plan_metagraph_match_v2(
        record,
        current_metagraph_hotkeys,
    )


def require_weight_input_plan_metagraph_match_v2(
    journal: Mapping[str, Any],
    current_metagraph_hotkeys: Any,
) -> Dict[str, Any]:
    """Fail unless a durable plan still uses the current metagraph order."""

    record = validate_weight_input_journal_v2(journal)
    if record["plan"]["metagraph_hash"] != _metagraph_hash(
        current_metagraph_hotkeys
    ):
        raise WeightInputJournalV2Error(
            "current metagraph differs from the prepared weight input plan"
        )
    return record


class AuthoritativeWeightInputJournalV2:
    """Persist one immutable plan per epoch and exact validator release."""

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
                raise WeightInputJournalV2Error(
                    "weight input journal %s must be positive"
                    % field.replace("_", " ")
                )
        if (
            not isinstance(min_free_bytes, int)
            or isinstance(min_free_bytes, bool)
            or min_free_bytes <= 0
        ):
            raise WeightInputJournalV2Error(
                "weight input journal minimum free bytes must be positive"
            )
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self._lock = threading.RLock()
        self._store_lock_local = threading.local()

    @contextmanager
    def _exclusive_store_lock(self):
        """Serialize exact-scope journal transitions across processes."""

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

    def _ensure_storage_capacity(
        self,
        *,
        encoded_size: int,
        replacing_path: Path,
    ) -> None:
        """Fail closed at configured bounds; never prune authority journals."""

        files = []
        for path in self.directory.iterdir():
            if path.name == _STORE_LOCK_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                raise WeightInputJournalV2Error(
                    "weight input journal storage inventory contains an invalid entry"
                )
            try:
                files.append((path, path.lstat().st_size))
            except OSError as exc:
                raise WeightInputJournalV2Error(
                    "weight input journal storage inventory failed"
                ) from exc
        replacing_size = next(
            (size for path, size in files if path == replacing_path),
            0,
        )
        replacing_exists = any(path == replacing_path for path, _ in files)
        projected_files = len(files) + (0 if replacing_exists else 1)
        projected_bytes = (
            sum(size for _, size in files) - replacing_size + encoded_size
        )
        if (
            (
                self.max_files is not None
                and projected_files > self.max_files
            )
            or (
                self.max_bytes is not None
                and projected_bytes > self.max_bytes
            )
            or shutil.disk_usage(self.directory).free
            < self.min_free_bytes + encoded_size
        ):
            raise WeightInputJournalV2Error(
                "weight input journal storage capacity is insufficient"
            )

    def verify_storage_ready(self) -> None:
        """Require room for the next complete exact-scope journal."""

        with self._lock, self._exclusive_store_lock():
            self._ensure_storage_capacity(
                encoded_size=_JOURNAL_WRITE_RESERVE_BYTES,
                replacing_path=self.directory / ".next-journal-reservation",
            )

    def _path(
        self,
        *,
        release_identity: Mapping[str, Any],
        validator_hotkey: str,
        netuid: int,
        epoch_id: int,
    ) -> Path:
        scope = {
            "release_identity": _stable_release_identity_v2(
                release_identity
            ),
            "validator_hotkey": _validator_hotkey(validator_hotkey),
            "netuid": _scope_integer(netuid, "netuid", positive=True),
            "epoch_id": _scope_integer(epoch_id, "epoch", positive=False),
        }
        digest = sha256_json(scope).removeprefix("sha256:")
        return self.directory / (
            "%d-%d-%s.json" % (scope["netuid"], scope["epoch_id"], digest)
        )

    def load_epoch(
        self,
        *,
        release_identity: Mapping[str, Any],
        validator_hotkey: str,
        netuid: int,
        epoch_id: int,
    ) -> Optional[Dict[str, Any]]:
        release = validate_weight_input_release_identity_v2(release_identity)
        hotkey = _validator_hotkey(validator_hotkey)
        normalized_netuid = _scope_integer(netuid, "netuid", positive=True)
        normalized_epoch = _scope_integer(epoch_id, "epoch", positive=False)
        path = self._path(
            release_identity=release,
            validator_hotkey=hotkey,
            netuid=normalized_netuid,
            epoch_id=normalized_epoch,
        )
        with self._lock:
            if not path.exists():
                return None
            record = self._load_path(path)
            if (
                _stable_release_identity_v2(record["release_identity"])
                != _stable_release_identity_v2(release)
                or record["validator_hotkey"] != hotkey
                or record["netuid"] != normalized_netuid
                or record["epoch_id"] != normalized_epoch
            ):
                raise WeightInputJournalV2Error(
                    "weight input journal differs from requested scope"
                )
            return record

    def record_plan(
        self,
        *,
        release_identity: Mapping[str, Any],
        validator_hotkey: str,
        netuid: int,
        epoch_id: int,
        calculation_snapshot: Mapping[str, Any],
        host_uids: Any,
        host_weights: Any,
        allocation_hash: str,
        leaderboard_window_start: str,
        leaderboard_window_end: str,
    ) -> Dict[str, Any]:
        release = validate_weight_input_release_identity_v2(release_identity)
        hotkey = _validator_hotkey(validator_hotkey)
        normalized_netuid = _scope_integer(netuid, "netuid", positive=True)
        normalized_epoch = _scope_integer(epoch_id, "epoch", positive=False)
        calculation = _canonical_object(
            calculation_snapshot,
            "calculation snapshot",
        )
        plan = {
            "calculation_snapshot": calculation,
            "calculation_snapshot_hash": sha256_json(calculation),
            "host_uids": list(host_uids),
            "host_weight_float_bits": _float_bits(host_weights),
            "allocation_hash": _hash(allocation_hash, "allocation hash"),
            "leaderboard_window_start": str(leaderboard_window_start),
            "leaderboard_window_end": str(leaderboard_window_end),
            "metagraph_hash": _metagraph_hash(
                calculation.get("metagraph_hotkeys")
            ),
        }
        plan_bytes = _canonical_bytes_b64(plan)
        plan_hash = sha256_bytes(base64.b64decode(plan_bytes))
        now = _timestamp()
        body = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "state": "planned",
            "revision": 0,
            "release_identity": release,
            "validator_hotkey": hotkey,
            "netuid": normalized_netuid,
            "epoch_id": normalized_epoch,
            "plan": plan,
            "plan_canonical_bytes_b64": plan_bytes,
            "plan_hash": plan_hash,
            "gateway_inputs": None,
            "gateway_inputs_canonical_bytes_b64": None,
            "gateway_inputs_hash": None,
            "created_at": now,
            "updated_at": now,
        }
        candidate = validate_weight_input_journal_v2(
            {**body, "journal_hash": sha256_json(body)}
        )
        path = self._path(
            release_identity=release,
            validator_hotkey=hotkey,
            netuid=normalized_netuid,
            epoch_id=normalized_epoch,
        )
        with self._lock, self._exclusive_store_lock():
            existing = self.load_epoch(
                release_identity=release,
                validator_hotkey=hotkey,
                netuid=normalized_netuid,
                epoch_id=normalized_epoch,
            )
            if existing is not None:
                if existing["plan_hash"] == plan_hash:
                    return existing
                raise WeightInputJournalV2Error(
                    "another weight input plan already exists for this scope"
                )
            self._write(path, candidate)
            return self._readback(path, candidate["journal_hash"])

    def record_gateway_inputs(
        self,
        *,
        release_identity: Mapping[str, Any],
        validator_hotkey: str,
        netuid: int,
        epoch_id: int,
        plan_hash: str,
        gateway_inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        release = validate_weight_input_release_identity_v2(release_identity)
        hotkey = _validator_hotkey(validator_hotkey)
        normalized_netuid = _scope_integer(netuid, "netuid", positive=True)
        normalized_epoch = _scope_integer(epoch_id, "epoch", positive=False)
        expected_plan_hash = _hash(plan_hash, "plan hash")
        normalized_inputs = _validate_gateway_inputs(gateway_inputs)
        inputs_bytes = _canonical_bytes_b64(normalized_inputs)
        inputs_hash = sha256_bytes(base64.b64decode(inputs_bytes))
        path = self._path(
            release_identity=release,
            validator_hotkey=hotkey,
            netuid=normalized_netuid,
            epoch_id=normalized_epoch,
        )
        with self._lock, self._exclusive_store_lock():
            current = self.load_epoch(
                release_identity=release,
                validator_hotkey=hotkey,
                netuid=normalized_netuid,
                epoch_id=normalized_epoch,
            )
            if current is None:
                raise WeightInputJournalV2Error(
                    "weight input plan is unavailable for gateway readback"
                )
            if current["plan_hash"] != expected_plan_hash:
                raise WeightInputJournalV2Error(
                    "weight input plan hash differs from durable scope"
                )
            if current["state"] == "inputs_verified":
                if (
                    current["gateway_inputs"] == normalized_inputs
                    and current["gateway_inputs_canonical_bytes_b64"]
                    == inputs_bytes
                    and current["gateway_inputs_hash"] == inputs_hash
                ):
                    return current
                raise WeightInputJournalV2Error(
                    "gateway inputs conflict with durable readback"
                )
            body = {
                key: current[key]
                for key in current
                if key != "journal_hash"
            }
            body.update(
                {
                    "state": "inputs_verified",
                    "revision": 1,
                    "gateway_inputs": normalized_inputs,
                    "gateway_inputs_canonical_bytes_b64": inputs_bytes,
                    "gateway_inputs_hash": inputs_hash,
                    "updated_at": _timestamp(),
                }
            )
            candidate = validate_weight_input_journal_v2(
                {**body, "journal_hash": sha256_json(body)}
            )
            self._write(path, candidate)
            return self._readback(path, candidate["journal_hash"])

    def _load_path(self, path: Path) -> Dict[str, Any]:
        try:
            return validate_weight_input_journal_v2(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            raise WeightInputJournalV2Error(
                "weight input journal cannot be read"
            ) from exc

    def _readback(self, path: Path, expected_hash: str) -> Dict[str, Any]:
        value = self._load_path(path)
        if value["journal_hash"] != expected_hash:
            raise WeightInputJournalV2Error(
                "weight input journal durable readback differs"
            )
        return value

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        encoded = canonical_json(dict(value)).encode("utf-8")
        self._ensure_storage_capacity(
            encoded_size=len(encoded),
            replacing_path=path,
        )
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
            raise WeightInputJournalV2Error(
                "weight input journal atomic write failed"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify durable validator weight-input storage readiness."
    )
    parser.add_argument("--verify-storage-ready", action="store_true")
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if not args.verify_storage_ready:
        parser.error("--verify-storage-ready is required")
    directory = Path(args.directory).expanduser()
    if not directory.is_absolute():
        print("ERROR: validator weight-input storage path must be absolute", file=sys.stderr)
        return 1
    try:
        AuthoritativeWeightInputJournalV2(
            directory
        ).verify_storage_ready()
    except (OSError, WeightInputJournalV2Error) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print("validator weight-input storage is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
