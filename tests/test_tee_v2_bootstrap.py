import base64

import pytest

from gateway.tee.provider_broker_v2 import (
    measured_retry_policy_hashes,
    provider_registry_hash,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from gateway.utils.tee_v2_bootstrap import (
    TEEV2BootstrapError,
    bootstrap_gateway_enclaves_v2,
    runtime_configuration_documents,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    create_boot_identity,
    sha256_json,
)


def _hash(character):
    return "sha256:" + character * 64


def _release():
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        values = {
            "commit_sha": "1" * 40,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **values,
                    }
                )
    return build_release_manifest(
        rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


class _Client:
    def __init__(self, role, release):
        self.role = role
        self.release = release
        self.config_hash = None
        self.registered = []

    async def v2_configure_runtime(self, *, configuration, configuration_hash):
        self.config_hash = configuration_hash
        return {"status": "ready", "physical_role": self.role}

    async def v2_get_boot_identity(self):
        summary = self.release["roles"][self.role]
        body = build_boot_identity_body(
            role=summary["service_role"],
            physical_role=self.role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=self.config_hash,
            boot_nonce=("a" * 32),
            signing_pubkey=("b" * 64),
            transport_pubkey=("c" * 64),
            transport_certificate_hash=_hash("d"),
            attestation_user_data_hash=_hash("e"),
            issued_at="2026-07-10T00:00:00Z",
        )
        return create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"attestation").decode(),
        )

    async def v2_get_transport_certificate(self):
        return ("certificate-" + self.role).encode()

    async def v2_register_peer(self, *, boot_identity, certificate_pem):
        self.registered.append(boot_identity["physical_role"])
        return {"physical_role": boot_identity["physical_role"]}

    async def v2_start_tls_service(self):
        return {"status": "started"}

    async def v2_call_peer_health(self, role):
        return {"status": "healthy", "role": role}


def _documents(release):
    protected_hash = _hash("5")
    return runtime_configuration_documents(
        release_manifest=release,
        provider_ref_hashes={
            "openrouter": _hash("1"),
            "exa": _hash("2"),
            "scrapingdog": _hash("3"),
            "deepline": _hash("4"),
            "supabase_service_role": _hash("7"),
            "truelist": _hash("8"),
        },
        provider_retry_policy_hashes=measured_retry_policy_hashes(protected_hash),
        provider_registry_hash=provider_registry_hash(),
        protected_workflow_manifest_hash=protected_hash,
        encrypted_artifact_policy={
            "schema_version": "leadpoet.encrypted_artifact_policy.v2",
            "bucket_host": "immutable.example.s3.us-east-1.amazonaws.com",
            "key_prefix": "/attested-v2/artifacts/",
            "minimum_retention_days": 365,
        },
        artifact_master_key_ref_hash=_hash("6"),
        research_lab_execution_config=build_research_lab_execution_config(
            environment={}
        ),
        configured_worker_counts={
            "gateway_scoring": 25,
            "gateway_autoresearch": 10,
        },
    )


@pytest.mark.asyncio
async def test_bootstrap_configures_and_checks_all_four_tls_directions():
    release = _release()
    clients = {role: _Client(role, release) for role in ROLE_SPECS}
    result = await bootstrap_gateway_enclaves_v2(
        release_manifest=release,
        runtime_documents=_documents(release),
        clients=clients,
        boot_verifier=lambda identity, **_: identity,
    )
    assert result["status"] == "ready"
    assert result["release_hash"] == release["release_hash"]
    assert len(result["channels"]) == 4
    assert sorted(clients["gateway_coordinator"].registered) == sorted(
        role for role in ROLE_SPECS if role != "gateway_coordinator"
    )
    for role in ROLE_SPECS:
        if role != "gateway_coordinator":
            assert clients[role].registered == ["gateway_coordinator"]


def test_runtime_documents_bind_release_and_preserve_10_worker_pool():
    release = _release()
    documents = _documents(release)
    assert documents["gateway_scoring"]["configuration"]["execution_worker_count"] == 10
    assert documents["gateway_scoring"]["configuration"]["configured_worker_count"] == 25
    assert documents["gateway_autoresearch"]["configuration"]["execution_worker_count"] == 10
    assert documents["gateway_autoresearch"]["configuration"]["configured_worker_count"] == 10
    assert documents["gateway_coordinator"]["configuration"]["execution_worker_count"] == 0
    assert documents["gateway_coordinator"]["configuration"]["configured_worker_count"] == 0
    assert all(
        document["configuration"]["release_hash"] == release["release_hash"]
        for document in documents.values()
    )
    assert all(
        set(document["configuration"]["release_roles"]) == set(ROLE_SPECS)
        for document in documents.values()
    )
    config_hashes = {
        document["configuration"]["research_lab_execution_config_hash"]
        for document in documents.values()
    }
    assert len(config_hashes) == 1


@pytest.mark.asyncio
async def test_bootstrap_rejects_boot_commit_not_in_release():
    release = _release()
    clients = {role: _Client(role, release) for role in ROLE_SPECS}
    original = clients["gateway_scoring"].v2_get_boot_identity

    async def wrong_boot():
        value = await original()
        value["commit_sha"] = "f" * 40
        return value

    clients["gateway_scoring"].v2_get_boot_identity = wrong_boot
    with pytest.raises(TEEV2BootstrapError, match="boot commit"):
        await bootstrap_gateway_enclaves_v2(
            release_manifest=release,
            runtime_documents=_documents(release),
            clients=clients,
            boot_verifier=lambda identity, **_: identity,
        )
