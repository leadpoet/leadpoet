"""Hardware-only V2 boot identity for the existing validator enclave."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import (
    build_boot_attestation_user_data,
    build_boot_identity_body,
    canonical_json,
    create_boot_identity,
    sha256_bytes,
    sha256_json,
)


VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION = (
    "leadpoet.validator_runtime_config.v4"
)
GATEWAY_RELEASE_LINEAGE_SCHEMA_VERSION = "leadpoet.attested_release_lineage.v1"
VALIDATOR_PHYSICAL_ROLE = "validator_weights"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_APPROVED_RELEASE_ROLES = frozenset(
    {
        "gateway_coordinator",
        "gateway_scoring",
        "gateway_autoresearch",
        "validator_weights",
    }
)
_STATEFUL_EPOCH_MODE = "stateful_v1"
_EPOCH_SCHEME = "bittensor.subnet_epoch_index.v1"
_CUTOVER_SCHEMA_VERSION = "leadpoet.subnet_epoch_cutover.v1"


class ValidatorRuntimeV2Error(RuntimeError):
    """The validator enclave cannot prove its hardware/runtime identity."""


def _validate_epoch_authority(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "mode",
        "cutover_manifest",
    }:
        raise ValidatorRuntimeV2Error("validator epoch authority fields are invalid")
    if value.get("mode") != _STATEFUL_EPOCH_MODE:
        raise ValidatorRuntimeV2Error("validator epoch authority mode is invalid")
    manifest = value.get("cutover_manifest")
    fields = {
        "schema_version",
        "epoch_scheme",
        "network_genesis_hash",
        "netuid",
        "cutover_block",
        "cutover_block_hash",
        "first_subnet_epoch_index",
        "first_settlement_epoch_id",
        "last_legacy_epoch_id",
        "mapping_hash",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ValidatorRuntimeV2Error("validator epoch cutover fields are invalid")
    normalized = dict(manifest)
    if normalized.get("schema_version") != _CUTOVER_SCHEMA_VERSION:
        raise ValidatorRuntimeV2Error("validator epoch cutover schema is invalid")
    if normalized.get("epoch_scheme") != _EPOCH_SCHEME:
        raise ValidatorRuntimeV2Error("validator epoch scheme is invalid")
    for field in (
        "netuid",
        "cutover_block",
        "first_subnet_epoch_index",
        "first_settlement_epoch_id",
        "last_legacy_epoch_id",
    ):
        raw = normalized.get(field)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ValidatorRuntimeV2Error(
                "validator epoch cutover %s is invalid" % field
            )
    if normalized["netuid"] <= 0:
        raise ValidatorRuntimeV2Error("validator epoch cutover netuid is invalid")
    for field in ("network_genesis_hash", "cutover_block_hash"):
        raw_hash = str(normalized.get(field) or "").lower()
        if not re.fullmatch(r"0x[0-9a-f]{64}", raw_hash):
            raise ValidatorRuntimeV2Error(
                "validator epoch cutover %s is invalid" % field
            )
        normalized[field] = raw_hash
    if normalized["first_settlement_epoch_id"] != (
        normalized["last_legacy_epoch_id"] + 1
    ):
        raise ValidatorRuntimeV2Error(
            "validator epoch settlement mapping is not monotonic"
        )
    body = {
        key: normalized[key]
        for key in (
            "schema_version",
            "epoch_scheme",
            "network_genesis_hash",
            "netuid",
            "cutover_block",
            "cutover_block_hash",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
        )
    }
    if normalized.get("mapping_hash") != sha256_json(body):
        raise ValidatorRuntimeV2Error("validator epoch cutover hash mismatch")
    return {
        "mode": _STATEFUL_EPOCH_MODE,
        "cutover_manifest": {**body, "mapping_hash": normalized["mapping_hash"]},
    }


def compute_app_manifest_hash(app_root: Path = Path("/app")) -> str:
    root = Path(app_root)
    if not root.is_dir():
        raise ValidatorRuntimeV2Error("validator enclave app root is unavailable")
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "mode": path.stat().st_mode & 0o777,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    if not entries:
        raise ValidatorRuntimeV2Error("validator enclave app manifest is empty")
    return sha256_json(entries)


def dependency_lock_hash(
    paths: Any = None,
) -> str:
    locked_paths = list(
        paths
        or (
            Path("/app/validator_tee/enclave/requirements.txt"),
            Path("/app/validator_tee/runtime-artifacts-v2.lock.json"),
            Path("/app/validator_tee/runtime-artifacts-v2.manifest.json"),
        )
    )
    entries = []
    try:
        for path in locked_paths:
            normalized = Path(path)
            entries.append(
                {
                    "path": normalized.as_posix(),
                    "sha256": sha256_bytes(normalized.read_bytes()),
                }
            )
    except OSError as exc:
        raise ValidatorRuntimeV2Error("validator dependency lock is unavailable") from exc
    if len(entries) != len(locked_paths) or not entries:
        raise ValidatorRuntimeV2Error("validator dependency lock is incomplete")
    return sha256_json(entries)


def _validate_configuration(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "build_manifest_hash",
        "dependency_lock_hash",
        "gateway_release_hash",
        "gateway_release_lineage",
        "hotkey_authority_config_hash",
        "epoch_authority",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidatorRuntimeV2Error("validator V2 runtime fields are invalid")
    schema_version = value.get("schema_version")
    if schema_version != VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION:
        raise ValidatorRuntimeV2Error("validator V2 runtime schema is invalid")
    epoch_authority = _validate_epoch_authority(value["epoch_authority"])
    commit = str(value.get("commit_sha") or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValidatorRuntimeV2Error("validator V2 commit is invalid")
    for field in (
        "build_manifest_hash",
        "dependency_lock_hash",
        "gateway_release_hash",
        "hotkey_authority_config_hash",
    ):
        if not _HASH_RE.fullmatch(str(value.get(field) or "")):
            raise ValidatorRuntimeV2Error("validator V2 %s is invalid" % field)
    lineage = validate_gateway_release_lineage(value["gateway_release_lineage"])
    if lineage["current_commit_sha"] != commit:
        raise ValidatorRuntimeV2Error(
            "validator current gateway lineage commit differs"
        )
    if lineage["current_gateway_release_hash"] != str(
        value["gateway_release_hash"]
    ).lower():
        raise ValidatorRuntimeV2Error(
            "validator current gateway lineage release differs"
        )
    normalized = {
        "schema_version": schema_version,
        "commit_sha": commit,
        "build_manifest_hash": str(value["build_manifest_hash"]).lower(),
        "dependency_lock_hash": str(value["dependency_lock_hash"]).lower(),
        "gateway_release_hash": str(value["gateway_release_hash"]).lower(),
        "hotkey_authority_config_hash": str(
            value["hotkey_authority_config_hash"]
        ).lower(),
        "gateway_release_lineage": lineage,
        "epoch_authority": epoch_authority,
    }
    return normalized


def validate_gateway_release_lineage(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "current_commit_sha",
        "current_gateway_release_hash",
        "releases",
        "lineage_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidatorRuntimeV2Error("validator gateway release lineage is invalid")
    if value.get("schema_version") != GATEWAY_RELEASE_LINEAGE_SCHEMA_VERSION:
        raise ValidatorRuntimeV2Error(
            "validator gateway release lineage schema is invalid"
        )
    current_commit = str(value.get("current_commit_sha") or "").lower()
    current_release_hash = str(
        value.get("current_gateway_release_hash") or ""
    ).lower()
    releases = value.get("releases")
    if (
        not _COMMIT_RE.fullmatch(current_commit)
        or not _HASH_RE.fullmatch(current_release_hash)
        or not isinstance(releases, Mapping)
        or not 1 <= len(releases) <= 512
    ):
        raise ValidatorRuntimeV2Error("validator gateway release lineage is invalid")
    normalized_releases = {}
    for release_commit, release in releases.items():
        commit = str(release_commit or "").lower()
        release_fields = {"channel_hash", "gateway_release_hash", "roles"}
        if (
            not _COMMIT_RE.fullmatch(commit)
            or not isinstance(release, Mapping)
            or set(release) != release_fields
            or not _HASH_RE.fullmatch(str(release.get("channel_hash") or ""))
            or not _HASH_RE.fullmatch(
                str(release.get("gateway_release_hash") or "")
            )
        ):
            raise ValidatorRuntimeV2Error(
                "validator gateway release lineage entry is invalid"
            )
        roles = release.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != _APPROVED_RELEASE_ROLES:
            raise ValidatorRuntimeV2Error(
                "validator approved release lineage roles are incomplete"
            )
        normalized_roles = {}
        for role, expectation in roles.items():
            expectation_fields = {
                "commit_sha",
                "pcr0",
                "build_manifest_hash",
                "dependency_lock_hash",
            }
            if (
                not isinstance(expectation, Mapping)
                or set(expectation) != expectation_fields
            ):
                raise ValidatorRuntimeV2Error(
                    "validator gateway release expectation is invalid"
                )
            expected_commit = str(expectation.get("commit_sha") or "").lower()
            expected_pcr0 = str(expectation.get("pcr0") or "").lower()
            expected_manifest = str(
                expectation.get("build_manifest_hash") or ""
            ).lower()
            expected_lock = str(
                expectation.get("dependency_lock_hash") or ""
            ).lower()
            if (
                expected_commit != commit
                or not _PCR0_RE.fullmatch(expected_pcr0)
                or expected_pcr0 == "0" * 96
                or not _HASH_RE.fullmatch(expected_manifest)
                or not _HASH_RE.fullmatch(expected_lock)
            ):
                raise ValidatorRuntimeV2Error(
                    "validator gateway release expectation is invalid"
                )
            normalized_roles[str(role)] = {
                "commit_sha": expected_commit,
                "pcr0": expected_pcr0,
                "build_manifest_hash": expected_manifest,
                "dependency_lock_hash": expected_lock,
            }
        normalized_releases[commit] = {
            "channel_hash": str(release["channel_hash"]).lower(),
            "gateway_release_hash": str(
                release["gateway_release_hash"]
            ).lower(),
            "roles": {
                role: normalized_roles[role] for role in sorted(normalized_roles)
            },
        }
    current = normalized_releases.get(current_commit)
    if (
        current is None
        or current["gateway_release_hash"] != current_release_hash
    ):
        raise ValidatorRuntimeV2Error(
            "validator current gateway release is absent from lineage"
        )
    body = {
        "schema_version": GATEWAY_RELEASE_LINEAGE_SCHEMA_VERSION,
        "current_commit_sha": current_commit,
        "current_gateway_release_hash": current_release_hash,
        "releases": {
            release_commit: normalized_releases[release_commit]
            for release_commit in sorted(normalized_releases)
        },
    }
    if value.get("lineage_hash") != sha256_json(body):
        raise ValidatorRuntimeV2Error(
            "validator gateway release lineage hash mismatch"
        )
    return {**body, "lineage_hash": value["lineage_hash"]}


def _nsm_attest(*, user_data: bytes, public_key: bytes) -> bytes:
    try:
        from validator_tee.enclave.nsm_lib import get_attestation_document

        response = get_attestation_document(
            user_data=bytes(user_data),
            public_key=bytes(public_key),
        )
        document = response["Attestation"]["document"]
    except Exception as exc:
        raise ValidatorRuntimeV2Error("hardware Nitro attestation is unavailable") from exc
    if not isinstance(document, (bytes, bytearray)) or not document:
        raise ValidatorRuntimeV2Error("hardware Nitro attestation is empty")
    return bytes(document)


def _nsm_pcr0() -> str:
    try:
        from validator_tee.enclave.nsm_lib import get_pcr_measurements

        pcr0 = str(get_pcr_measurements()["PCR0"]).lower()
    except Exception as exc:
        raise ValidatorRuntimeV2Error("hardware PCR0 is unavailable") from exc
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise ValidatorRuntimeV2Error("hardware PCR0 is invalid")
    return pcr0


class ValidatorRuntimeIdentityV2:
    def __init__(
        self,
        *,
        signing_pubkey_supplier: Callable[[], str],
        app_manifest_supplier: Callable[[], str] = compute_app_manifest_hash,
        dependency_lock_supplier: Callable[[], str] = dependency_lock_hash,
        pcr0_supplier: Callable[[], str] = _nsm_pcr0,
        attestation_supplier: Callable[..., bytes] = _nsm_attest,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._signing_pubkey_supplier = signing_pubkey_supplier
        self._app_manifest_supplier = app_manifest_supplier
        self._dependency_lock_supplier = dependency_lock_supplier
        self._pcr0_supplier = pcr0_supplier
        self._attestation_supplier = attestation_supplier
        self._clock = clock
        self._configuration = None  # type: Optional[Dict[str, Any]]
        self._boot_identity = None  # type: Optional[Dict[str, Any]]
        self._lock = threading.Lock()

    def configure(
        self,
        configuration: Mapping[str, Any],
        *,
        expected_config_hash: str,
    ) -> Dict[str, Any]:
        normalized = _validate_configuration(configuration)
        observed_manifest = self._app_manifest_supplier()
        observed_lock = self._dependency_lock_supplier()
        if normalized["build_manifest_hash"] != observed_manifest:
            raise ValidatorRuntimeV2Error("validator app manifest differs from release")
        if normalized["dependency_lock_hash"] != observed_lock:
            raise ValidatorRuntimeV2Error("validator dependency lock differs from release")
        config_hash = sha256_json(normalized)
        if config_hash != str(expected_config_hash or "").lower():
            raise ValidatorRuntimeV2Error("validator runtime config hash mismatch")
        with self._lock:
            if self._boot_identity is not None:
                if self._configuration != normalized:
                    raise ValidatorRuntimeV2Error("validator runtime config is immutable")
                return dict(self._boot_identity)
            pcr0 = str(self._pcr0_supplier() or "").lower()
            if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
                raise ValidatorRuntimeV2Error("validator hardware PCR0 is invalid")
            pubkey = str(self._signing_pubkey_supplier() or "").lower()
            issued = self._clock().astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            provisional = {
                "role": "validator_weights",
                "physical_role": VALIDATOR_PHYSICAL_ROLE,
                "commit_sha": normalized["commit_sha"],
                "pcr0": pcr0,
                "build_manifest_hash": observed_manifest,
                "dependency_lock_hash": observed_lock,
                "config_hash": config_hash,
                "boot_nonce": os.urandom(16).hex(),
                "signing_pubkey": pubkey,
                "transport_pubkey": pubkey,
                "transport_certificate_hash": sha256_bytes(
                    b"validator-v2-transport:" + bytes.fromhex(pubkey)
                ),
                "issued_at": issued,
            }
            user_data = build_boot_attestation_user_data(provisional)
            user_data_bytes = canonical_json(user_data).encode("utf-8")
            if len(user_data_bytes) > 512:
                raise ValidatorRuntimeV2Error("validator boot user_data exceeds NSM limit")
            body = build_boot_identity_body(
                **provisional,
                attestation_user_data_hash=sha256_json(user_data),
            )
            document = self._attestation_supplier(
                user_data=user_data_bytes,
                public_key=bytes.fromhex(pubkey),
            )
            identity = create_boot_identity(
                body=body,
                attestation_document_b64=base64.b64encode(document).decode("ascii"),
            )
            self._configuration = normalized
            self._boot_identity = identity
            return dict(identity)

    def boot_identity(self) -> Dict[str, Any]:
        with self._lock:
            if self._boot_identity is None:
                raise ValidatorRuntimeV2Error("validator V2 runtime is not configured")
            return dict(self._boot_identity)

    def gateway_release_lineage(self) -> Dict[str, Any]:
        with self._lock:
            if self._configuration is None:
                raise ValidatorRuntimeV2Error("validator V2 runtime is not configured")
            return {
                commit: {
                    **release,
                    "roles": {
                        role: dict(expectation)
                        for role, expectation in release["roles"].items()
                    },
                }
                for commit, release in self._configuration[
                    "gateway_release_lineage"
                ]["releases"].items()
            }

    def epoch_authority(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._configuration is None:
                raise ValidatorRuntimeV2Error("validator V2 runtime is not configured")
            value = self._configuration.get("epoch_authority")
            if value is None:
                return None
            return {
                "mode": value["mode"],
                "cutover_manifest": dict(value["cutover_manifest"]),
            }

    def hotkey_authority_config_hash(self) -> str:
        with self._lock:
            if self._configuration is None:
                raise ValidatorRuntimeV2Error(
                    "validator V2 runtime is not configured"
                )
            return str(self._configuration["hotkey_authority_config_hash"])
