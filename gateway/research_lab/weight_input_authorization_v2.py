"""Crash-safe authorization for one exact gateway V2 weight-input build.

The gateway records the signed request before any measured source read.  A
process replacement may therefore recognize the exact request without relying
on process-local TTL state.  This record is authorization only; it never makes
an incomplete input reconstruction usable after the source cutoff.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
    validate_weight_inputs_request_v2,
)


AUTHORIZATION_SCHEMA_VERSION = "leadpoet.gateway_weight_input_authorization.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_RELEASE_IDENTITY_FIELDS = {
    "physical_role",
    "service_role",
    "commit_sha",
    "pcr0",
    "build_manifest_hash",
    "dependency_lock_hash",
    "build_identity_hash",
    "release_hash",
}
_STORE_LOCK_NAME = ".weight-input-store-v2.lock"
_DEFAULT_MIN_FREE_BYTES = 256 * 1024 * 1024
_CHECKPOINT_RESERVE_BYTES = MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES


class WeightInputAuthorizationV2Error(RuntimeError):
    """The durable signed request is unavailable, corrupt, or conflicting."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise WeightInputAuthorizationV2Error("%s is invalid" % field)
    return normalized


def validate_gateway_weight_release_identity_v2(
    value: Mapping[str, Any],
) -> Dict[str, str]:
    """Validate the independently reproduced coordinator release identity."""

    if not isinstance(value, Mapping) or set(value) != _RELEASE_IDENTITY_FIELDS:
        raise WeightInputAuthorizationV2Error(
            "gateway weight release identity fields are invalid"
        )
    normalized = {key: str(value[key] or "").strip().lower() for key in value}
    if (
        normalized["physical_role"] != "gateway_coordinator"
        or normalized["service_role"] != "gateway_coordinator"
        or not _COMMIT_RE.fullmatch(normalized["commit_sha"])
        or not _PCR0_RE.fullmatch(normalized["pcr0"])
        or normalized["pcr0"] == "0" * 96
    ):
        raise WeightInputAuthorizationV2Error(
            "gateway weight release identity is invalid"
        )
    for field in (
        "build_manifest_hash",
        "dependency_lock_hash",
        "build_identity_hash",
        "release_hash",
    ):
        normalized[field] = _hash(normalized[field], field.replace("_", " "))
    return normalized


def load_gateway_weight_release_identity_v2() -> Dict[str, str]:
    """Load the exact approved coordinator identity used by this gateway."""

    from gateway.tee.release_manifest_v2 import (
        role_expectation,
        validate_release_manifest,
    )

    path = Path(
        os.environ.get(
            "GATEWAY_V2_RELEASE_MANIFEST",
            "/home/ec2-user/tee/gateway-v2-release-manifest.json",
        )
    ).expanduser()
    try:
        manifest = validate_release_manifest(
            json.loads(path.read_text(encoding="utf-8"))
        )
        identity = role_expectation(manifest, "gateway_coordinator")
    except Exception as exc:
        raise WeightInputAuthorizationV2Error(
            "approved gateway weight release identity is unavailable"
        ) from exc
    return validate_gateway_weight_release_identity_v2(identity)


def _canonical_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WeightInputAuthorizationV2Error("%s is invalid" % field)
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise WeightInputAuthorizationV2Error(
            "%s is not canonical JSON" % field
        ) from exc
    if not isinstance(normalized, dict):
        raise WeightInputAuthorizationV2Error("%s is invalid" % field)
    return normalized


def validate_weight_input_authorization_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "state",
        "release_identity",
        "request",
        "calculation_snapshot",
        "validator_hotkey_signature",
        "source_cutoff_block",
        "created_at",
        "authorization_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization fields are invalid"
        )
    if (
        value.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or value.get("state") != "authorized"
    ):
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization schema is invalid"
        )
    release_identity = validate_gateway_weight_release_identity_v2(
        value.get("release_identity")
    )
    try:
        request = validate_weight_inputs_request_v2(value.get("request"))
    except Exception as exc:
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization request is invalid"
        ) from exc
    calculation = _canonical_object(
        value.get("calculation_snapshot"), "calculation snapshot"
    )
    if (
        sha256_json(calculation) != request["calculation_snapshot_hash"]
        or calculation.get("netuid") != request["netuid"]
        or calculation.get("epoch_id") != request["epoch_id"]
        or calculation.get("block") != request["block"]
    ):
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization snapshot differs from request"
        )
    signature = str(value.get("validator_hotkey_signature") or "")
    if not signature or len(signature) > 1024 or any(
        character.isspace() for character in signature
    ):
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization signature is invalid"
        )
    source_cutoff_block = value.get("source_cutoff_block")
    if (
        not isinstance(source_cutoff_block, int)
        or isinstance(source_cutoff_block, bool)
        or source_cutoff_block < 0
    ):
        raise WeightInputAuthorizationV2Error(
            "gateway weight source cutoff is invalid"
        )
    if not isinstance(value.get("created_at"), str) or not value["created_at"]:
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization timestamp is invalid"
        )
    body = {
        key: value[key] for key in fields if key != "authorization_hash"
    }
    body.update(
        {
            "release_identity": release_identity,
            "request": request,
            "calculation_snapshot": calculation,
            "validator_hotkey_signature": signature,
            "source_cutoff_block": source_cutoff_block,
        }
    )
    authorization_hash = _hash(
        value.get("authorization_hash"), "authorization hash"
    )
    if authorization_hash != sha256_json(body):
        raise WeightInputAuthorizationV2Error(
            "gateway weight authorization hash differs"
        )
    return {**body, "authorization_hash": authorization_hash}


class GatewayWeightInputAuthorizationStoreV2:
    """Atomically retain signed input authorization across process changes."""

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
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization %s must be positive"
                    % field.replace("_", " ")
                )
        if (
            not isinstance(min_free_bytes, int)
            or isinstance(min_free_bytes, bool)
            or min_free_bytes <= 0
        ):
            raise WeightInputAuthorizationV2Error(
                "gateway weight authorization minimum free bytes must be positive"
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

    def _ensure_storage_capacity(
        self,
        encoded_size: int,
        *,
        additional_files: int = 1,
    ) -> None:
        """Fail closed at configured bounds; never remove authority records."""

        files = []
        for path in self.directory.iterdir():
            if path.name == _STORE_LOCK_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization storage inventory contains "
                    "an invalid entry"
                )
            try:
                files.append(path.lstat().st_size)
            except OSError as exc:
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization storage inventory failed"
                ) from exc
        if (
            (
                self.max_files is not None
                and len(files) + additional_files > self.max_files
            )
            or (
                self.max_bytes is not None
                and sum(files) + encoded_size > self.max_bytes
            )
            or shutil.disk_usage(self.directory).free
            < self.min_free_bytes + encoded_size
        ):
            raise WeightInputAuthorizationV2Error(
                "gateway weight authorization storage capacity is insufficient"
            )

    def _path(
        self,
        request_hash: str,
        release_identity: Mapping[str, Any],
    ) -> Path:
        digest = _hash(request_hash, "request hash").removeprefix("sha256:")
        release = validate_gateway_weight_release_identity_v2(release_identity)
        release_digest = release["release_hash"].removeprefix("sha256:")
        return self.directory / (
            digest + "." + release_digest + ".authorized.json"
        )

    def verify_storage_ready(self) -> None:
        """Require room for the next authorization and checkpoint pair."""

        with self._lock, self._exclusive_store_lock():
            self._ensure_storage_capacity(
                _CHECKPOINT_RESERVE_BYTES,
                additional_files=2,
            )

    def load(
        self,
        *,
        release_identity: Mapping[str, Any],
        request: Mapping[str, Any],
        calculation_snapshot: Mapping[str, Any],
        source_cutoff_block: int,
    ) -> Optional[Dict[str, Any]]:
        normalized_request = validate_weight_inputs_request_v2(request)
        normalized_release = validate_gateway_weight_release_identity_v2(
            release_identity
        )
        path = self._path(
            normalized_request["request_hash"], normalized_release
        )
        # A read for a never-initialized store must stay side-effect free. This
        # is important at the source cutoff: a rejected new request must not
        # create a lock file or make an empty directory look like durable
        # authorization. persist() rechecks under the cross-process lock, so a
        # concurrent first writer remains safe after this fast miss.
        with self._lock:
            if not self.directory.exists():
                return None
        with self._lock, self._exclusive_store_lock():
            if not path.exists():
                return None
            try:
                record = validate_weight_input_authorization_v2(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError) as exc:
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization cannot be read"
                ) from exc
            expected = {
                "release_identity": normalized_release,
                "request": normalized_request,
                "calculation_snapshot": _canonical_object(
                    calculation_snapshot, "calculation snapshot"
                ),
                "source_cutoff_block": int(source_cutoff_block),
            }
            if any(record[field] != expected[field] for field in expected):
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization differs from request scope"
                )
            return record

    def persist(
        self,
        *,
        release_identity: Mapping[str, Any],
        request: Mapping[str, Any],
        calculation_snapshot: Mapping[str, Any],
        validator_hotkey_signature: str,
        source_cutoff_block: int,
    ) -> Dict[str, Any]:
        normalized_request = validate_weight_inputs_request_v2(request)
        body = {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "state": "authorized",
            "release_identity": validate_gateway_weight_release_identity_v2(
                release_identity
            ),
            "request": normalized_request,
            "calculation_snapshot": _canonical_object(
                calculation_snapshot, "calculation snapshot"
            ),
            "validator_hotkey_signature": str(validator_hotkey_signature),
            "source_cutoff_block": int(source_cutoff_block),
            "created_at": _timestamp(),
        }
        candidate = validate_weight_input_authorization_v2(
            {**body, "authorization_hash": sha256_json(body)}
        )
        path = self._path(normalized_request["request_hash"], release_identity)
        with self._lock, self._exclusive_store_lock():
            existing = None
            if path.exists():
                existing = self.load(
                    release_identity=release_identity,
                    request=normalized_request,
                    calculation_snapshot=calculation_snapshot,
                    source_cutoff_block=source_cutoff_block,
                )
            checkpoint_path = self.directory / (
                normalized_request["request_hash"].removeprefix("sha256:")
                + ".json"
            )
            if existing is not None:
                if not checkpoint_path.exists():
                    # Reserve the bounded logical payload before any source
                    # read. A full store must fail here, not after expensive
                    # reconstruction has already completed.
                    self._ensure_storage_capacity(
                        _CHECKPOINT_RESERVE_BYTES,
                        additional_files=1,
                    )
                return existing
            encoded = json.dumps(
                candidate, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            self._ensure_storage_capacity(
                len(encoded)
                + (0 if checkpoint_path.exists() else _CHECKPOINT_RESERVE_BYTES),
                additional_files=1 if checkpoint_path.exists() else 2,
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
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization atomic write failed"
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
                request=normalized_request,
                calculation_snapshot=calculation_snapshot,
                source_cutoff_block=source_cutoff_block,
            )
            # Another request can win the atomic replace with a different
            # valid sr25519 signature for the same canonical request. load()
            # has already verified the complete release-bound scope and the
            # stored record hash, so that concurrent winner is equivalent.
            if readback is None:
                raise WeightInputAuthorizationV2Error(
                    "gateway weight authorization durable readback differs"
                )
            return readback


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify durable gateway weight-input storage readiness."
    )
    parser.add_argument("--verify-storage-ready", action="store_true")
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if not args.verify_storage_ready:
        parser.error("--verify-storage-ready is required")
    directory = Path(args.directory).expanduser()
    if not directory.is_absolute():
        print("ERROR: gateway weight-input storage path must be absolute", file=sys.stderr)
        return 1
    try:
        GatewayWeightInputAuthorizationStoreV2(
            directory
        ).verify_storage_ready()
    except (OSError, WeightInputAuthorizationV2Error) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print("gateway weight-input storage is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
