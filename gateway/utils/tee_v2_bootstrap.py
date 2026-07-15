"""Host-side bootstrap for the fixed three-enclave V2 gateway topology.

The parent supplies public configuration and relays boot documents, certificates,
and ciphertext.  Every enclave verifies peers against the independently built
release manifest before mutually authenticated TLS is started.  This module
never receives provider plaintext credentials.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from gateway.tee.runtime_identity_v2 import RUNTIME_CONFIG_SCHEMA_VERSION
from gateway.tee.topology import ROLE_SPECS
from gateway.utils.tee_client import TEEClient
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    verify_boot_identity_nitro,
)


BOOTSTRAP_SCHEMA_VERSION = "leadpoet.gateway_v2_bootstrap.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TEEV2BootstrapError(RuntimeError):
    """The complete enclave release cannot be configured or mutually attested."""


def load_release_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TEEV2BootstrapError("V2 release manifest is unavailable") from exc
    return validate_release_manifest(value)


def _hash_map(value: Mapping[str, Any], field: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TEEV2BootstrapError("%s must be a non-empty object" % field)
    normalized = {
        str(name): str(digest or "").strip().lower()
        for name, digest in value.items()
    }
    if any(not name or not _HASH_RE.fullmatch(digest) for name, digest in normalized.items()):
        raise TEEV2BootstrapError("%s contains an invalid hash" % field)
    return dict(sorted(normalized.items()))


def runtime_configuration_documents(
    *,
    release_manifest: Mapping[str, Any],
    provider_ref_hashes: Mapping[str, str],
    provider_retry_policy_hashes: Mapping[str, str],
    provider_registry_hash: str,
    protected_workflow_manifest_hash: str,
    encrypted_artifact_policy: Mapping[str, Any],
    artifact_master_key_ref_hash: str,
    research_lab_execution_config: Mapping[str, Any],
    configured_worker_counts: Mapping[str, int],
) -> Dict[str, Dict[str, Any]]:
    from gateway.tee.provider_broker_v2 import (
        expected_job_credential_slot_ref_hashes,
        expected_provider_credential_slots,
        measured_retry_policy_hashes,
        provider_registry_hash as measured_provider_registry_hash,
    )

    release = validate_release_manifest(release_manifest)
    provider_refs = _hash_map(provider_ref_hashes, "provider_ref_hashes")
    retry_hashes = _hash_map(
        provider_retry_policy_hashes, "provider_retry_policy_hashes"
    )
    for name, digest in (
        ("provider_registry_hash", provider_registry_hash),
        ("protected_workflow_manifest_hash", protected_workflow_manifest_hash),
        ("artifact_master_key_ref_hash", artifact_master_key_ref_hash),
    ):
        if not _HASH_RE.fullmatch(str(digest or "").lower()):
            raise TEEV2BootstrapError("%s is invalid" % name)
    if str(provider_registry_hash).lower() != measured_provider_registry_hash():
        raise TEEV2BootstrapError(
            "provider registry hash differs from measured broker routes"
        )
    if set(provider_refs) != set(expected_provider_credential_slots()):
        raise TEEV2BootstrapError(
            "provider credential references differ from measured routes"
        )
    expected_retry_hashes = measured_retry_policy_hashes(
        str(protected_workflow_manifest_hash).lower()
    )
    if retry_hashes != expected_retry_hashes:
        raise TEEV2BootstrapError(
            "provider retry hashes differ from protected workflow code"
        )
    from gateway.tee.artifact_persistence_v2 import validate_artifact_policy
    from gateway.tee.research_lab_runtime_config_v2 import (
        research_lab_execution_config_hash,
        validate_research_lab_execution_config,
    )

    normalized_artifact_policy = validate_artifact_policy(encrypted_artifact_policy)
    encrypted_artifact_policy_hash = sha256_json(normalized_artifact_policy)
    normalized_research_lab_config = validate_research_lab_execution_config(
        research_lab_execution_config
    )
    normalized_research_lab_config_hash = research_lab_execution_config_hash(
        normalized_research_lab_config
    )
    normalized_worker_counts = {
        str(role): int(count) for role, count in configured_worker_counts.items()
    }
    if set(normalized_worker_counts) != {
        "gateway_scoring",
        "gateway_autoresearch",
    } or any(
        count <= 0 or count > 500 for count in normalized_worker_counts.values()
    ):
        raise TEEV2BootstrapError("configured worker counts are invalid")

    release_expectations = {
        role: role_expectation(release, role) for role in sorted(ROLE_SPECS)
    }
    documents = {}
    for role in sorted(ROLE_SPECS):
        peers = (
            [candidate for candidate in sorted(ROLE_SPECS) if candidate != role]
            if role == "gateway_coordinator"
            else ["gateway_coordinator"]
        )
        peer_releases = {
            peer: {
                "commit_sha": release_expectations[peer]["commit_sha"],
                "pcr0": release_expectations[peer]["pcr0"],
                "build_manifest_hash": release_expectations[peer][
                    "build_manifest_hash"
                ],
            }
            for peer in peers
        }
        execution_worker_count = 0
        configured_worker_count = 0
        if role == "gateway_scoring":
            execution_worker_count = 10
            configured_worker_count = normalized_worker_counts[role]
        elif role == "gateway_autoresearch":
            execution_worker_count = normalized_worker_counts[role]
            configured_worker_count = normalized_worker_counts[role]
        configuration = {
            "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "release_hash": release["release_hash"],
            "release_commit_sha": release["commit_sha"],
            "own_build_identity_hash": release_expectations[role][
                "build_identity_hash"
            ],
            "release_roles": {
                release_role: {
                    "commit_sha": release_expectations[release_role]["commit_sha"],
                    "pcr0": release_expectations[release_role]["pcr0"],
                    "build_manifest_hash": release_expectations[release_role][
                        "build_manifest_hash"
                    ],
                    "dependency_lock_hash": release_expectations[release_role][
                        "dependency_lock_hash"
                    ],
                }
                for release_role in sorted(ROLE_SPECS)
            },
            "peer_releases": peer_releases,
            "provider_ref_hashes": provider_refs,
            "job_lease_slot_ref_hashes": (
                expected_job_credential_slot_ref_hashes()
            ),
            "provider_retry_policy_hashes": retry_hashes,
            "provider_registry_hash": str(provider_registry_hash).lower(),
            "protected_workflow_manifest_hash": str(
                protected_workflow_manifest_hash
            ).lower(),
            "encrypted_artifact_policy": normalized_artifact_policy,
            "encrypted_artifact_policy_hash": encrypted_artifact_policy_hash,
            "artifact_master_key_ref_hash": str(
                artifact_master_key_ref_hash
            ).lower(),
            "research_lab_execution_config": normalized_research_lab_config,
            "research_lab_execution_config_hash": (
                normalized_research_lab_config_hash
            ),
            "execution_worker_count": execution_worker_count,
            "configured_worker_count": configured_worker_count,
        }
        config_document = {
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "configuration": json.loads(canonical_json(configuration)),
        }
        documents[role] = {
            "configuration": config_document["configuration"],
            "configuration_hash": sha256_json(config_document),
        }
    return documents


def _default_clients() -> Dict[str, TEEClient]:
    return {
        role: TEEClient(cid=int(spec["cid"]))
        for role, spec in ROLE_SPECS.items()
    }


async def bootstrap_gateway_enclaves_v2(
    *,
    release_manifest: Mapping[str, Any],
    runtime_documents: Mapping[str, Mapping[str, Any]],
    clients: Optional[Mapping[str, Any]] = None,
    boot_verifier: Callable[..., Mapping[str, Any]] = verify_boot_identity_nitro,
) -> Dict[str, Any]:
    """Configure, independently verify, pair, and health-check all three roles."""

    release = validate_release_manifest(release_manifest)
    if set(runtime_documents) != set(ROLE_SPECS):
        raise TEEV2BootstrapError("runtime documents do not cover every role")
    role_clients = dict(clients or _default_clients())
    if set(role_clients) != set(ROLE_SPECS):
        raise TEEV2BootstrapError("enclave clients do not cover every role")

    for role in sorted(ROLE_SPECS):
        document = runtime_documents[role]
        if not isinstance(document, Mapping) or set(document) != {
            "configuration",
            "configuration_hash",
        }:
            raise TEEV2BootstrapError("runtime document fields are invalid")
        status = await role_clients[role].v2_configure_runtime(
            configuration=dict(document["configuration"]),
            configuration_hash=str(document["configuration_hash"]),
        )
        if status.get("status") != "ready" or status.get("physical_role") != role:
            raise TEEV2BootstrapError("%s runtime configuration failed" % role)

    boots = {}
    certificates = {}
    for role in sorted(ROLE_SPECS):
        boot = await role_clients[role].v2_get_boot_identity()
        expectation = role_expectation(release, role)
        if boot.get("physical_role") != role:
            raise TEEV2BootstrapError("%s returned another physical role" % role)
        if boot.get("commit_sha") != expectation["commit_sha"]:
            raise TEEV2BootstrapError("%s boot commit differs from release" % role)
        if boot.get("build_manifest_hash") != expectation["build_manifest_hash"]:
            raise TEEV2BootstrapError("%s boot manifest differs from release" % role)
        if boot.get("dependency_lock_hash") != expectation["dependency_lock_hash"]:
            raise TEEV2BootstrapError("%s dependency lock differs from release" % role)
        boot_verifier(boot, expected_pcr0=expectation["pcr0"])
        boots[role] = dict(boot)
        certificates[role] = await role_clients[role].v2_get_transport_certificate()

    for role in sorted(ROLE_SPECS):
        peers = (
            [candidate for candidate in sorted(ROLE_SPECS) if candidate != role]
            if role == "gateway_coordinator"
            else ["gateway_coordinator"]
        )
        for peer in peers:
            registered = await role_clients[role].v2_register_peer(
                boot_identity=boots[peer],
                certificate_pem=certificates[peer],
            )
            if registered.get("physical_role") != peer:
                raise TEEV2BootstrapError("%s did not register peer %s" % (role, peer))

    for role in sorted(ROLE_SPECS):
        started = await role_clients[role].v2_start_tls_service()
        if started.get("status") not in {"started", "already_running"}:
            raise TEEV2BootstrapError("%s TLS service did not start" % role)

    channels = []
    for runner_role in (
        "gateway_scoring",
        "gateway_autoresearch",
    ):
        runner_to_coordinator = await role_clients[runner_role].v2_call_peer_health(
            "gateway_coordinator"
        )
        coordinator_to_runner = await role_clients[
            "gateway_coordinator"
        ].v2_call_peer_health(runner_role)
        for source, target, result in (
            (runner_role, "gateway_coordinator", runner_to_coordinator),
            ("gateway_coordinator", runner_role, coordinator_to_runner),
        ):
            if result.get("status") != "healthy":
                raise TEEV2BootstrapError(
                    "attested TLS channel failed: %s -> %s" % (source, target)
                )
            channels.append({"source": source, "target": target})
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "status": "ready",
        "release_hash": release["release_hash"],
        "commit_sha": release["commit_sha"],
        "boot_identity_hashes": {
            role: boots[role]["boot_identity_hash"] for role in sorted(boots)
        },
        "channels": channels,
    }


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    from gateway.tee.provider_broker_v2 import (
        measured_retry_policy_hashes,
        provider_registry_hash,
    )
    from gateway.utils.tee_kms_provision_v2 import (
        load_provider_envelopes,
        provider_reference_hashes_from_envelopes,
    )
    from gateway.tee.research_lab_runtime_config_v2 import (
        build_research_lab_execution_config,
    )
    from gateway.research_lab.provider_profiles_v2 import (
        verify_required_worker_proxy_profiles_v2,
    )

    release = load_release_manifest(args.release_manifest)
    envelopes = load_provider_envelopes(args.credential_envelope)
    provider_refs = provider_reference_hashes_from_envelopes(envelopes)
    artifact_envelopes = [
        item
        for item in envelopes
        if item["credential_slot"] == "artifact_master_key"
    ]
    if len(artifact_envelopes) != 1:
        raise TEEV2BootstrapError(
            "exactly one artifact master-key envelope is required"
        )
    try:
        protected_manifest = json.loads(
            args.protected_workflow_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TEEV2BootstrapError(
            "protected workflow manifest is unavailable"
        ) from exc
    protected_hash = str(protected_manifest.get("manifest_hash") or "").lower()
    if not _HASH_RE.fullmatch(protected_hash):
        raise TEEV2BootstrapError(
            "protected workflow manifest hash is invalid"
        )
    profile_set = verify_required_worker_proxy_profiles_v2(
        config_dir=args.config_dir
    )
    documents = runtime_configuration_documents(
        release_manifest=release,
        provider_ref_hashes=provider_refs,
        provider_retry_policy_hashes=measured_retry_policy_hashes(protected_hash),
        provider_registry_hash=provider_registry_hash(),
        protected_workflow_manifest_hash=protected_hash,
        encrypted_artifact_policy=json.loads(
            args.encrypted_artifact_policy.read_text(encoding="utf-8")
        ),
        artifact_master_key_ref_hash=str(
            artifact_envelopes[0]["credential_ref_hash"]
        ),
        research_lab_execution_config=build_research_lab_execution_config(),
        configured_worker_counts=profile_set["worker_counts"],
    )
    return await bootstrap_gateway_enclaves_v2(
        release_manifest=release,
        runtime_documents=documents,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument(
        "--credential-envelope",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--protected-workflow-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--encrypted-artifact-policy", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(_main_async(args))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
