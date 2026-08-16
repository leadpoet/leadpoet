"""Single-record durable cache for a fully verified auditor weight authority."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json


SCHEMA_VERSION = "leadpoet.auditor_latest_verified_weight_bundle.v1"
IDENTITY_CACHE_SCHEMA_VERSION = "leadpoet.independent_pcr0_identities.v2"
SIGNATURE_DOMAIN = "leadpoet.auditor_latest_verified_weight_bundle.v1:"
MAX_RECORD_BYTES = 128 * 1024 * 1024


class AuditorLatestVerifiedBundleV2Error(ValueError):
    """A cached auditor authority is absent, corrupt, or unauthenticated."""


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditorLatestVerifiedBundleV2Error("%s is invalid" % field)
    normalized = int(value)
    if normalized < minimum:
        raise AuditorLatestVerifiedBundleV2Error("%s is invalid" % field)
    return normalized


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if (
        len(normalized) != 71
        or not normalized.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in normalized[7:])
    ):
        raise AuditorLatestVerifiedBundleV2Error("%s is invalid" % field)
    return normalized


def _compact_weights_hash(value: Any) -> str:
    """Normalize the compact verifier's canonical raw digest for storage."""

    normalized = str(value or "").lower()
    if len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    ):
        normalized = "sha256:" + normalized
    return _hash(normalized, "weights hash")


def _no_duplicate_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise AuditorLatestVerifiedBundleV2Error(
                "cached auditor authority contains duplicate JSON keys"
            )
        value[key] = child
    return value


def verified_bundle_projection_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Project only the exact submission vector and its verified identity."""

    if not isinstance(value, Mapping):
        raise AuditorLatestVerifiedBundleV2Error(
            "verified auditor authority must be an object"
        )
    uids = value.get("uids")
    weights = value.get("weights_u16")
    if (
        not isinstance(uids, list)
        or not isinstance(weights, list)
        or not uids
        or len(uids) != len(weights)
    ):
        raise AuditorLatestVerifiedBundleV2Error(
            "verified auditor weight vector is invalid"
        )
    normalized_uids = [
        _integer(uid, "verified auditor UID", minimum=0) for uid in uids
    ]
    normalized_weights = [
        _integer(weight, "verified auditor u16 weight", minimum=0)
        for weight in weights
    ]
    if (
        len(set(normalized_uids)) != len(normalized_uids)
        or any(uid > 65535 for uid in normalized_uids)
        or any(weight > 65535 for weight in normalized_weights)
        or not any(normalized_weights)
    ):
        raise AuditorLatestVerifiedBundleV2Error(
            "verified auditor weight vector is invalid"
        )
    stage = str(value.get("authority_stage") or "finalized")
    if stage not in {"published", "finalized"}:
        raise AuditorLatestVerifiedBundleV2Error(
            "verified auditor authority stage is invalid"
        )
    validator_hotkey = str(value.get("validator_hotkey") or "")
    if not validator_hotkey or len(validator_hotkey) > 64:
        raise AuditorLatestVerifiedBundleV2Error(
            "verified validator hotkey is invalid"
        )
    return {
        "epoch_id": _integer(value.get("epoch_id"), "source epoch", minimum=1),
        "netuid": _integer(value.get("netuid"), "netuid", minimum=1),
        "block": _integer(value.get("block"), "bundle block", minimum=1),
        "uids": normalized_uids,
        "weights_u16": normalized_weights,
        "weights_hash": _compact_weights_hash(value.get("weights_hash")),
        "bundle_hash": _hash(value.get("bundle_hash"), "bundle hash"),
        "authority_stage": stage,
        "validator_hotkey": validator_hotkey,
        "receipt_graph_hash": str(value.get("receipt_graph_hash") or ""),
    }


def _record_body(
    *,
    auditor_hotkey: str,
    authority: Mapping[str, Any],
    identity_cache: Mapping[str, Any],
    verified_bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    projection = verified_bundle_projection_v2(verified_bundle)
    hotkey = str(auditor_hotkey or "")
    if not hotkey or len(hotkey) > 64:
        raise AuditorLatestVerifiedBundleV2Error("auditor hotkey is invalid")
    if not isinstance(authority, Mapping):
        raise AuditorLatestVerifiedBundleV2Error("cached authority is invalid")
    if (
        not isinstance(identity_cache, Mapping)
        or identity_cache.get("schema_version") != IDENTITY_CACHE_SCHEMA_VERSION
        or not isinstance(identity_cache.get("entries"), list)
        or not identity_cache.get("entries")
    ):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached independent release identities are invalid"
        )
    normalized_authority = json.loads(canonical_json(dict(authority)))
    normalized_cache = json.loads(canonical_json(dict(identity_cache)))
    return {
        "schema_version": SCHEMA_VERSION,
        "auditor_hotkey": hotkey,
        "source_epoch_id": projection["epoch_id"],
        "netuid": projection["netuid"],
        "authority": normalized_authority,
        "identity_cache": normalized_cache,
        "verified_bundle": projection,
        "authority_hash": sha256_json(normalized_authority),
        "identity_cache_hash": sha256_json(normalized_cache),
        "verified_bundle_hash": sha256_json(projection),
    }


def record_signature_message_v2(record_hash: str) -> bytes:
    return (SIGNATURE_DOMAIN + _hash(record_hash, "record hash")).encode("ascii")


def build_latest_verified_bundle_record_v2(
    *,
    auditor_hotkey: str,
    authority: Mapping[str, Any],
    identity_cache: Mapping[str, Any],
    verified_bundle: Mapping[str, Any],
    sign: Callable[[bytes], bytes],
) -> Dict[str, Any]:
    body = _record_body(
        auditor_hotkey=auditor_hotkey,
        authority=authority,
        identity_cache=identity_cache,
        verified_bundle=verified_bundle,
    )
    record_hash = sha256_json(body)
    signature = bytes(sign(record_signature_message_v2(record_hash)))
    if len(signature) != 64:
        raise AuditorLatestVerifiedBundleV2Error(
            "auditor cache signature is invalid"
        )
    return {
        **body,
        "record_hash": record_hash,
        "auditor_signature": signature.hex(),
    }


def validate_latest_verified_bundle_record_v2(
    value: Mapping[str, Any],
    *,
    expected_hotkey: str,
    expected_netuid: int,
    verify_signature: Callable[[bytes, bytes], bool],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "auditor_hotkey",
        "source_epoch_id",
        "netuid",
        "authority",
        "identity_cache",
        "verified_bundle",
        "authority_hash",
        "identity_cache_hash",
        "verified_bundle_hash",
        "record_hash",
        "auditor_signature",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority fields are invalid"
        )
    body = _record_body(
        auditor_hotkey=str(value.get("auditor_hotkey") or ""),
        authority=value.get("authority"),
        identity_cache=value.get("identity_cache"),
        verified_bundle=value.get("verified_bundle"),
    )
    if body["schema_version"] != value.get("schema_version"):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority schema is invalid"
        )
    if body["auditor_hotkey"] != str(expected_hotkey):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority belongs to another hotkey"
        )
    if body["netuid"] != _integer(expected_netuid, "expected netuid", minimum=1):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority targets another subnet"
        )
    if body["source_epoch_id"] != _integer(
        value.get("source_epoch_id"), "source epoch", minimum=1
    ):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor source epoch differs"
        )
    for field in (
        "authority_hash",
        "identity_cache_hash",
        "verified_bundle_hash",
    ):
        if body[field] != _hash(value.get(field), field.replace("_", " ")):
            raise AuditorLatestVerifiedBundleV2Error(
                "cached auditor authority integrity differs"
            )
    record_hash = _hash(value.get("record_hash"), "record hash")
    if record_hash != sha256_json(body):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority record hash differs"
        )
    signature_text = str(value.get("auditor_signature") or "").lower()
    try:
        signature = bytes.fromhex(signature_text)
    except ValueError as exc:
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority signature is invalid"
        ) from exc
    if len(signature) != 64 or not bool(
        verify_signature(record_signature_message_v2(record_hash), signature)
    ):
        raise AuditorLatestVerifiedBundleV2Error(
            "cached auditor authority signature is invalid"
        )
    return {**body, "record_hash": record_hash, "auditor_signature": signature_text}


class LatestVerifiedBundleStoreV2:
    """Atomically retain exactly one authenticated verified authority."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(
        self,
        *,
        expected_hotkey: str,
        expected_netuid: int,
        verify_signature: Callable[[bytes, bytes], bool],
    ) -> Dict[str, Any]:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError as exc:
            raise AuditorLatestVerifiedBundleV2Error(
                "no cached verified auditor authority exists"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > MAX_RECORD_BYTES
        ):
            raise AuditorLatestVerifiedBundleV2Error(
                "cached auditor authority file is unsafe"
            )
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
        except AuditorLatestVerifiedBundleV2Error:
            raise
        except Exception as exc:
            raise AuditorLatestVerifiedBundleV2Error(
                "cached auditor authority is malformed"
            ) from exc
        return validate_latest_verified_bundle_record_v2(
            value,
            expected_hotkey=expected_hotkey,
            expected_netuid=expected_netuid,
            verify_signature=verify_signature,
        )

    def replace_if_newer(
        self,
        record: Mapping[str, Any],
        *,
        expected_hotkey: str,
        expected_netuid: int,
        verify_signature: Callable[[bytes, bytes], bool],
    ) -> bool:
        normalized = validate_latest_verified_bundle_record_v2(
            record,
            expected_hotkey=expected_hotkey,
            expected_netuid=expected_netuid,
            verify_signature=verify_signature,
        )
        try:
            previous = self.load(
                expected_hotkey=expected_hotkey,
                expected_netuid=expected_netuid,
                verify_signature=verify_signature,
            )
        except AuditorLatestVerifiedBundleV2Error:
            previous = None
        if previous is not None:
            previous_epoch = int(previous["source_epoch_id"])
            next_epoch = int(normalized["source_epoch_id"])
            if next_epoch < previous_epoch:
                return False
            if next_epoch == previous_epoch:
                if normalized["record_hash"] != previous["record_hash"]:
                    previous_bundle = dict(previous["verified_bundle"])
                    next_bundle = dict(normalized["verified_bundle"])
                    previous_stage = previous_bundle.pop("authority_stage")
                    next_stage = next_bundle.pop("authority_stage")
                    if previous_bundle != next_bundle:
                        raise AuditorLatestVerifiedBundleV2Error(
                            "conflicting verified authorities exist for one epoch"
                        )
                    if not (
                        previous_stage == "published"
                        and next_stage == "finalized"
                    ):
                        return False
                else:
                    return False

        encoded = (canonical_json(normalized) + "\n").encode("ascii")
        if len(encoded) > MAX_RECORD_BYTES:
            raise AuditorLatestVerifiedBundleV2Error(
                "cached auditor authority exceeds the size bound"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % self.path.name,
            dir=str(self.path.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            if self.path.exists() and self.path.is_symlink():
                raise AuditorLatestVerifiedBundleV2Error(
                    "cached auditor authority destination is unsafe"
                )
            os.replace(str(temporary), str(self.path))
            directory = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return True


__all__ = [
    "AuditorLatestVerifiedBundleV2Error",
    "LatestVerifiedBundleStoreV2",
    "MAX_RECORD_BYTES",
    "SCHEMA_VERSION",
    "build_latest_verified_bundle_record_v2",
    "record_signature_message_v2",
    "validate_latest_verified_bundle_record_v2",
    "verified_bundle_projection_v2",
]
