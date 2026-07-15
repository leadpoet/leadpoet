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


VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION = "leadpoet.validator_runtime_config.v2"
VALIDATOR_PHYSICAL_ROLE = "validator_weights"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_GATEWAY_ROLES = frozenset(
    {
        "gateway_coordinator",
        "gateway_scoring",
        "gateway_autoresearch",
    }
)


class ValidatorRuntimeV2Error(RuntimeError):
    """The validator enclave cannot prove its hardware/runtime identity."""


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
        "gateway_role_expectations",
        "hotkey_authority_config_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidatorRuntimeV2Error("validator V2 runtime fields are invalid")
    if value.get("schema_version") != VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION:
        raise ValidatorRuntimeV2Error("validator V2 runtime schema is invalid")
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
    expectations = value.get("gateway_role_expectations")
    if not isinstance(expectations, Mapping) or set(expectations) != _GATEWAY_ROLES:
        raise ValidatorRuntimeV2Error("validator V2 gateway role set is incomplete")
    normalized_expectations = {}
    for role, expectation in expectations.items():
        if not isinstance(expectation, Mapping) or set(expectation) != {
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
        }:
            raise ValidatorRuntimeV2Error("validator V2 gateway expectation is invalid")
        expected_commit = str(expectation.get("commit_sha") or "").lower()
        expected_pcr0 = str(expectation.get("pcr0") or "").lower()
        expected_manifest = str(expectation.get("build_manifest_hash") or "").lower()
        if (
            not _COMMIT_RE.fullmatch(expected_commit)
            or not _PCR0_RE.fullmatch(expected_pcr0)
            or expected_pcr0 == "0" * 96
            or not _HASH_RE.fullmatch(expected_manifest)
        ):
            raise ValidatorRuntimeV2Error("validator V2 gateway expectation is invalid")
        normalized_expectations[str(role)] = {
            "commit_sha": expected_commit,
            "pcr0": expected_pcr0,
            "build_manifest_hash": expected_manifest,
        }
    return {
        "schema_version": VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION,
        "commit_sha": commit,
        "build_manifest_hash": str(value["build_manifest_hash"]).lower(),
        "dependency_lock_hash": str(value["dependency_lock_hash"]).lower(),
        "gateway_release_hash": str(value["gateway_release_hash"]).lower(),
        "hotkey_authority_config_hash": str(
            value["hotkey_authority_config_hash"]
        ).lower(),
        "gateway_role_expectations": {
            role: normalized_expectations[role]
            for role in sorted(normalized_expectations)
        },
    }


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

    def gateway_expectations(self) -> Dict[str, Any]:
        with self._lock:
            if self._configuration is None:
                raise ValidatorRuntimeV2Error("validator V2 runtime is not configured")
            return {
                role: dict(value)
                for role, value in self._configuration[
                    "gateway_role_expectations"
                ].items()
            }

    def hotkey_authority_config_hash(self) -> str:
        with self._lock:
            if self._configuration is None:
                raise ValidatorRuntimeV2Error(
                    "validator V2 runtime is not configured"
                )
            return str(self._configuration["hotkey_authority_config_hash"])
