"""Fail-closed V2 runtime identity created inside a gateway enclave."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.tee.build_identity import load_identity
from gateway.tee.mtls_identity import generate_ephemeral_tls_identity
from gateway.tee.topology import role_spec
from leadpoet_canonical.attested_v2 import (
    build_boot_attestation_user_data,
    build_boot_identity_body,
    canonical_json,
    create_boot_identity,
    sha256_json,
)


RUNTIME_CONFIG_SCHEMA_VERSION = "leadpoet.enclave_runtime_config.v2"
BOOTSTRAP_SCHEMA_VERSION = "leadpoet.gateway_v2_bootstrap.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_SECRET_MARKERS = (
    "api_key",
    "credential",
    "password",
    "private_key",
    "secret",
    "seed",
    "service_role",
    "token",
)


class RuntimeIdentityV2Error(RuntimeError):
    """The measured runtime cannot establish a genuine immutable V2 boot."""


def _issued_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_public_configuration(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(configuration, Mapping):
        raise RuntimeIdentityV2Error("V2 runtime configuration must be an object")
    normalized = json.loads(canonical_json(dict(configuration)))
    encoded = canonical_json(normalized).encode("utf-8")
    if len(encoded) > 256 * 1024:
        raise RuntimeIdentityV2Error("V2 runtime configuration exceeds size limit")
    for key in normalized:
        lowered = str(key).lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            if not lowered.endswith(("_hash", "_ref", "_ref_hash")):
                raise RuntimeIdentityV2Error(
                    "V2 runtime configuration contains a secret-shaped field"
                )
    serialized = canonical_json(normalized).lower()
    if any(marker in serialized for marker in ("sk-or-", "aws_secret_access_key", "sb_secret")):
        raise RuntimeIdentityV2Error("V2 runtime configuration contains secret material")
    return normalized


def _validate_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RuntimeIdentityV2Error("%s is invalid" % field)
    return normalized


def _validate_release_configuration(
    configuration: Mapping[str, Any],
    *,
    physical_role: str,
    build_identity: Mapping[str, Any],
) -> None:
    expected_fields = {
        "bootstrap_schema_version",
        "release_hash",
        "release_commit_sha",
        "own_build_identity_hash",
        "release_roles",
        "peer_releases",
        "provider_ref_hashes",
        "job_lease_slot_ref_hashes",
        "provider_retry_policy_hashes",
        "provider_registry_hash",
        "protected_workflow_manifest_hash",
        "encrypted_artifact_policy",
        "encrypted_artifact_policy_hash",
        "artifact_master_key_ref_hash",
        "research_lab_execution_config",
        "research_lab_execution_config_hash",
        "execution_worker_count",
        "configured_worker_count",
    }
    if set(configuration) != expected_fields:
        raise RuntimeIdentityV2Error("V2 runtime release fields are incomplete")
    if configuration.get("bootstrap_schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        raise RuntimeIdentityV2Error("V2 bootstrap schema is invalid")
    _validate_hash(configuration.get("release_hash"), "release_hash")
    if configuration.get("release_commit_sha") != build_identity.get("commit_sha"):
        raise RuntimeIdentityV2Error("V2 release commit differs from measured build")
    if configuration.get("own_build_identity_hash") != build_identity.get("identity_hash"):
        raise RuntimeIdentityV2Error("V2 release build identity differs from measured build")
    if (
        configuration.get("protected_workflow_manifest_hash")
        != build_identity.get("protected_manifest_hash")
    ):
        raise RuntimeIdentityV2Error(
            "V2 protected workflow manifest differs from measured build"
        )
    for field in (
        "provider_registry_hash",
        "protected_workflow_manifest_hash",
        "encrypted_artifact_policy_hash",
        "artifact_master_key_ref_hash",
        "research_lab_execution_config_hash",
    ):
        _validate_hash(configuration.get(field), field)
    from gateway.tee.research_lab_runtime_config_v2 import (
        research_lab_execution_config_hash,
        validate_research_lab_execution_config,
    )

    research_lab_config = configuration.get("research_lab_execution_config")
    if not isinstance(research_lab_config, Mapping):
        raise RuntimeIdentityV2Error(
            "V2 Research Lab execution configuration is missing"
        )
    try:
        normalized_research_lab_config = validate_research_lab_execution_config(
            research_lab_config
        )
    except Exception as exc:
        raise RuntimeIdentityV2Error(
            "V2 Research Lab execution configuration is invalid"
        ) from exc
    if dict(research_lab_config) != normalized_research_lab_config:
        raise RuntimeIdentityV2Error(
            "V2 Research Lab execution configuration is not normalized"
        )
    if research_lab_execution_config_hash(normalized_research_lab_config) != (
        configuration.get("research_lab_execution_config_hash")
    ):
        raise RuntimeIdentityV2Error(
            "V2 Research Lab execution configuration hash mismatch"
        )
    from gateway.tee.artifact_persistence_v2 import validate_artifact_policy

    artifact_policy = configuration.get("encrypted_artifact_policy")
    if not isinstance(artifact_policy, Mapping):
        raise RuntimeIdentityV2Error("V2 encrypted artifact policy is missing")
    normalized_artifact_policy = validate_artifact_policy(artifact_policy)
    if dict(artifact_policy) != normalized_artifact_policy:
        raise RuntimeIdentityV2Error("V2 encrypted artifact policy is not normalized")
    if sha256_json(normalized_artifact_policy) != configuration.get(
        "encrypted_artifact_policy_hash"
    ):
        raise RuntimeIdentityV2Error("V2 encrypted artifact policy hash mismatch")
    for field in (
        "provider_ref_hashes",
        "job_lease_slot_ref_hashes",
        "provider_retry_policy_hashes",
    ):
        value = configuration.get(field)
        if not isinstance(value, Mapping) or not value:
            raise RuntimeIdentityV2Error("%s is incomplete" % field)
        for name, digest in value.items():
            if not str(name or "").strip():
                raise RuntimeIdentityV2Error("%s contains an empty name" % field)
            _validate_hash(digest, "%s.%s" % (field, name))
    from gateway.tee.provider_broker_v2 import (
        expected_job_credential_slot_ref_hashes,
    )

    if dict(configuration["job_lease_slot_ref_hashes"]) != (
        expected_job_credential_slot_ref_hashes()
    ):
        raise RuntimeIdentityV2Error(
            "V2 job credential slots differ from measured policy"
        )

    release_roles = configuration.get("release_roles")
    expected_release_roles = {
        "gateway_coordinator",
        "gateway_scoring",
        "gateway_autoresearch",
    }
    if not isinstance(release_roles, Mapping) or set(release_roles) != expected_release_roles:
        raise RuntimeIdentityV2Error("V2 release role set is incomplete")
    for release_role, expectation in release_roles.items():
        role_spec(str(release_role))
        if not isinstance(expectation, Mapping) or set(expectation) != {
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        }:
            raise RuntimeIdentityV2Error("V2 release role fields are invalid")
        if not re.fullmatch(
            r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
            str(expectation.get("commit_sha") or ""),
        ):
            raise RuntimeIdentityV2Error("V2 release role commit is invalid")
        pcr0 = str(expectation.get("pcr0") or "")
        if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
            raise RuntimeIdentityV2Error("V2 release role PCR0 is invalid")
        _validate_hash(
            expectation.get("build_manifest_hash"),
            "release role build_manifest_hash",
        )
        _validate_hash(
            expectation.get("dependency_lock_hash"),
            "release role dependency_lock_hash",
        )
    own_release = release_roles.get(physical_role)
    if (
        own_release.get("commit_sha") != build_identity.get("commit_sha")
        or own_release.get("build_manifest_hash")
        != build_identity.get("execution_manifest_hash")
        or own_release.get("dependency_lock_hash")
        != build_identity.get("dependency_lock_hash")
    ):
        raise RuntimeIdentityV2Error("V2 own release role differs from measured build")

    expected_peers = (
        {
            "gateway_scoring",
            "gateway_autoresearch",
        }
        if physical_role == "gateway_coordinator"
        else {"gateway_coordinator"}
    )
    peer_releases = configuration.get("peer_releases")
    if not isinstance(peer_releases, Mapping) or set(peer_releases) != expected_peers:
        raise RuntimeIdentityV2Error("V2 peer release set differs from topology")
    for peer_role, expectation in peer_releases.items():
        role_spec(str(peer_role))
        if not isinstance(expectation, Mapping) or set(expectation) != {
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
        }:
            raise RuntimeIdentityV2Error("V2 peer release fields are invalid")
        if not re.fullmatch(
            r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
            str(expectation.get("commit_sha") or ""),
        ):
            raise RuntimeIdentityV2Error("V2 peer release commit is invalid")
        pcr0 = str(expectation.get("pcr0") or "")
        if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
            raise RuntimeIdentityV2Error("V2 peer release PCR0 is invalid")
        _validate_hash(
            expectation.get("build_manifest_hash"),
            "peer build_manifest_hash",
        )

    execution_worker_count = configuration.get("execution_worker_count")
    configured_worker_count = configuration.get("configured_worker_count")
    if physical_role == "gateway_coordinator":
        valid_worker_counts = (
            execution_worker_count == 0 and configured_worker_count == 0
        )
    elif physical_role == "gateway_scoring":
        valid_worker_counts = (
            execution_worker_count == 10
            and isinstance(configured_worker_count, int)
            and 0 < configured_worker_count <= 500
        )
    else:
        valid_worker_counts = (
            isinstance(configured_worker_count, int)
            and 0 < configured_worker_count <= 500
            and execution_worker_count == configured_worker_count
        )
    if not valid_worker_counts:
        raise RuntimeIdentityV2Error("V2 worker count differs from approved topology")


def nsm_attestation_document(*, user_data: bytes, signing_pubkey: bytes) -> bytes:
    """Request a real COSE_Sign1 document; there is no development fallback."""

    try:
        from nsm_lib import get_attestation_document

        response = get_attestation_document(
            user_data=bytes(user_data),
            public_key=bytes(signing_pubkey),
        )
        document = response["Attestation"]["document"]
    except Exception as exc:
        raise RuntimeIdentityV2Error("hardware Nitro attestation is unavailable") from exc
    if not isinstance(document, (bytes, bytearray)) or not document:
        raise RuntimeIdentityV2Error("hardware Nitro attestation document is empty")
    return bytes(document)


class RuntimeIdentityV2:
    """One immutable signing/TLS/config identity for an enclave boot."""

    def __init__(
        self,
        *,
        gateway_root: Path,
        physical_role: str,
        signing_pubkey_supplier: Callable[[], str],
        pcr0_supplier: Callable[[], str],
        attestation_supplier: Callable[..., bytes] = nsm_attestation_document,
        tls_identity_supplier: Callable[..., Mapping[str, Any]] = generate_ephemeral_tls_identity,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._gateway_root = Path(gateway_root)
        self._physical_role = str(physical_role or "")
        self._build_identity = load_identity(
            gateway_root=self._gateway_root,
            expected_role=self._physical_role,
        )
        topology_role = role_spec(self._physical_role)
        if topology_role["service_role"] != self._build_identity["service_role"]:
            raise RuntimeIdentityV2Error("build identity service role mismatch")
        self._service_role = str(topology_role["service_role"])
        self._signing_pubkey_supplier = signing_pubkey_supplier
        self._pcr0_supplier = pcr0_supplier
        self._attestation_supplier = attestation_supplier
        self._tls_identity_supplier = tls_identity_supplier
        self._clock = clock
        self._lock = threading.Lock()
        self._runtime_configuration = None  # type: Optional[Dict[str, Any]]
        self._tls_identity = None  # type: Optional[Dict[str, Any]]
        self._boot_identity = None  # type: Optional[Dict[str, Any]]

    def configure(
        self,
        *,
        configuration: Mapping[str, Any],
        expected_config_hash: str,
    ) -> Dict[str, Any]:
        normalized = _validate_public_configuration(configuration)
        _validate_release_configuration(
            normalized,
            physical_role=self._physical_role,
            build_identity=self._build_identity,
        )
        config_document = {
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "physical_role": self._physical_role,
            "service_role": self._service_role,
            "configuration": normalized,
        }
        config_hash = sha256_json(config_document)
        if config_hash != str(expected_config_hash or "").strip().lower():
            raise RuntimeIdentityV2Error("V2 runtime configuration hash mismatch")
        with self._lock:
            if self._boot_identity is not None:
                if self._boot_identity["config_hash"] != config_hash:
                    raise RuntimeIdentityV2Error(
                        "V2 runtime configuration is immutable for enclave lifetime"
                    )
                return self.public_status()

            pcr0 = str(self._pcr0_supplier() or "").strip().lower()
            if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
                raise RuntimeIdentityV2Error("hardware PCR0 is unavailable")
            own_release = normalized["release_roles"][self._physical_role]
            if pcr0 != own_release["pcr0"]:
                raise RuntimeIdentityV2Error(
                    "hardware PCR0 differs from verified V2 release role"
                )
            signing_pubkey = str(self._signing_pubkey_supplier() or "").strip().lower()
            tls_identity = dict(
                self._tls_identity_supplier(service_role=self._service_role)
            )
            issued_at = _issued_at(self._clock)
            provisional = {
                "role": self._service_role,
                "physical_role": self._physical_role,
                "commit_sha": self._build_identity["commit_sha"],
                "pcr0": pcr0,
                "build_manifest_hash": self._build_identity[
                    "execution_manifest_hash"
                ],
                "dependency_lock_hash": self._build_identity[
                    "dependency_lock_hash"
                ],
                "config_hash": config_hash,
                "boot_nonce": secrets.token_hex(16),
                "signing_pubkey": signing_pubkey,
                "transport_pubkey": tls_identity["transport_pubkey"],
                "transport_certificate_hash": tls_identity[
                    "certificate_sha256"
                ],
                "issued_at": issued_at,
            }
            user_data = build_boot_attestation_user_data(provisional)
            user_data_bytes = canonical_json(user_data).encode("utf-8")
            if len(user_data_bytes) > 512:
                raise RuntimeIdentityV2Error("V2 boot attestation user_data is too large")
            body = build_boot_identity_body(
                **provisional,
                attestation_user_data_hash=sha256_json(user_data),
            )
            document = self._attestation_supplier(
                user_data=user_data_bytes,
                signing_pubkey=bytes.fromhex(signing_pubkey),
            )
            boot_identity = create_boot_identity(
                body=body,
                attestation_document_b64=base64.b64encode(document).decode("ascii"),
            )
            self._runtime_configuration = config_document
            self._tls_identity = tls_identity
            self._boot_identity = boot_identity
            return self.public_status()

    def public_status(self) -> Dict[str, Any]:
        if self._boot_identity is None:
            return {
                "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
                "status": "not_configured",
                "physical_role": self._physical_role,
                "service_role": self._service_role,
            }
        return {
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "status": "ready",
            "physical_role": self._physical_role,
            "service_role": self._service_role,
            "commit_sha": self._boot_identity["commit_sha"],
            "pcr0": self._boot_identity["pcr0"],
            "config_hash": self._boot_identity["config_hash"],
            "boot_identity_hash": self._boot_identity["boot_identity_hash"],
            "transport_certificate_hash": self._boot_identity[
                "transport_certificate_hash"
            ],
        }

    def boot_identity(self) -> Dict[str, Any]:
        with self._lock:
            if self._boot_identity is None:
                raise RuntimeIdentityV2Error("V2 runtime identity is not configured")
            return dict(self._boot_identity)

    def transport_certificate_pem(self) -> bytes:
        with self._lock:
            if self._tls_identity is None:
                raise RuntimeIdentityV2Error("V2 runtime identity is not configured")
            return bytes(self._tls_identity["certificate_pem"])

    def transport_private_key_pem(self) -> bytes:
        """Internal-only accessor for the enclave TLS listener/client."""

        with self._lock:
            if self._tls_identity is None:
                raise RuntimeIdentityV2Error("V2 runtime identity is not configured")
            return bytes(self._tls_identity["private_key_pem"])

    def tls_identity(self) -> Dict[str, Any]:
        """Internal TLS material for the measured channel implementation."""

        with self._lock:
            if self._tls_identity is None:
                raise RuntimeIdentityV2Error("V2 runtime identity is not configured")
            return dict(self._tls_identity)

    def peer_release_expectation(self, physical_role: str) -> Dict[str, str]:
        configuration = self.runtime_configuration()["configuration"]
        releases = configuration.get("peer_releases")
        if not isinstance(releases, Mapping):
            raise RuntimeIdentityV2Error("V2 peer release map is unavailable")
        value = releases.get(str(physical_role or ""))
        if not isinstance(value, Mapping) or set(value) != {
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
        }:
            raise RuntimeIdentityV2Error("V2 peer release expectation is invalid")
        return {
            "commit_sha": str(value["commit_sha"]),
            "pcr0": str(value["pcr0"]),
            "build_manifest_hash": str(value["build_manifest_hash"]),
        }

    def release_role_expectation(self, physical_role: str) -> Dict[str, str]:
        configuration = self.runtime_configuration()["configuration"]
        releases = configuration.get("release_roles")
        if not isinstance(releases, Mapping):
            raise RuntimeIdentityV2Error("V2 release role map is unavailable")
        value = releases.get(str(physical_role or ""))
        if not isinstance(value, Mapping) or set(value) != {
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        }:
            raise RuntimeIdentityV2Error("V2 release role expectation is invalid")
        return {str(name): str(item) for name, item in value.items()}

    def expected_peer_roles(self) -> tuple[str, ...]:
        configuration = self.runtime_configuration()["configuration"]
        releases = configuration.get("peer_releases")
        if not isinstance(releases, Mapping):
            raise RuntimeIdentityV2Error("V2 peer release map is unavailable")
        return tuple(sorted(str(role) for role in releases))

    def runtime_configuration(self) -> Dict[str, Any]:
        with self._lock:
            if self._runtime_configuration is None:
                raise RuntimeIdentityV2Error("V2 runtime identity is not configured")
            return json.loads(canonical_json(self._runtime_configuration))

    def research_lab_config(self):
        from gateway.tee.research_lab_runtime_config_v2 import (
            research_lab_config_from_document,
        )

        configuration = self.runtime_configuration()["configuration"]
        return research_lab_config_from_document(
            configuration["research_lab_execution_config"]
        )

    def apply_research_lab_behavior_environment(self) -> None:
        from gateway.tee.research_lab_runtime_config_v2 import (
            apply_behavior_environment,
        )

        configuration = self.runtime_configuration()["configuration"]
        apply_behavior_environment(
            configuration["research_lab_execution_config"]
        )
