import base64

import pytest

from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.utils.tee_v2_bootstrap import (
    COORDINATOR_ROLE,
    TEEV2BootstrapError,
    bootstrap_gateway_enclaves_v2,
    runtime_configuration_documents,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    create_boot_identity,
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

def _documents(release):
    protected_hash = _hash("5")
    return runtime_configuration_documents(
        release_manifest=release,
        protected_workflow_manifest_hash=protected_hash,
    )


@pytest.mark.asyncio
async def test_bootstrap_configures_and_checks_coordinator_identity():
    release = _release()
    clients = {COORDINATOR_ROLE: _Client(COORDINATOR_ROLE, release)}
    result = await bootstrap_gateway_enclaves_v2(
        release_manifest=release,
        runtime_documents=_documents(release),
        clients=clients,
        boot_verifier=lambda identity, **_: identity,
    )
    assert result["status"] == "ready"
    assert result["release_hash"] == release["release_hash"]
    assert set(result["boot_identity_hashes"]) == {COORDINATOR_ROLE}


def test_runtime_documents_contain_only_measured_identity_configuration():
    release = _release()
    documents = _documents(release)
    expected_fields = {
        "bootstrap_schema_version",
        "release_hash",
        "release_commit_sha",
        "own_build_identity_hash",
        "release_roles",
        "peer_releases",
        "protected_workflow_manifest_hash",
    }
    assert all(
        set(document["configuration"]) == expected_fields
        for document in documents.values()
    )
    assert all(
        document["configuration"]["release_hash"] == release["release_hash"]
        for document in documents.values()
    )
    assert set(documents) == {COORDINATOR_ROLE}
    configuration = documents[COORDINATOR_ROLE]["configuration"]
    assert set(configuration["release_roles"]) == {COORDINATOR_ROLE}
    assert configuration["peer_releases"] == {}


@pytest.mark.asyncio
async def test_bootstrap_rejects_boot_commit_not_in_release():
    release = _release()
    clients = {COORDINATOR_ROLE: _Client(COORDINATOR_ROLE, release)}
    original = clients[COORDINATOR_ROLE].v2_get_boot_identity

    async def wrong_boot():
        value = await original()
        value["commit_sha"] = "f" * 40
        return value

    clients[COORDINATOR_ROLE].v2_get_boot_identity = wrong_boot
    with pytest.raises(TEEV2BootstrapError, match="boot commit"):
        await bootstrap_gateway_enclaves_v2(
            release_manifest=release,
            runtime_documents=_documents(release),
            clients=clients,
            boot_verifier=lambda identity, **_: identity,
        )
