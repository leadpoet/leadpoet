from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.build_identity import build_identity, write_identity
from gateway.tee.runtime_identity_v2 import (
    RUNTIME_CONFIG_SCHEMA_VERSION,
    RuntimeIdentityV2,
    RuntimeIdentityV2Error,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_attestation_user_data,
    canonical_json,
    sha256_json,
    validate_boot_identity,
)

HASH = "sha256:" + "a" * 64


def _manager(tmp_path: Path, *, pcr0: str = "b" * 96):
    gateway_root = tmp_path / "gateway"
    identity = build_identity(
        role="gateway_coordinator",
        service_role="gateway_coordinator",
        commit_sha="c" * 40,
        execution_manifest_hash=HASH,
        dependency_lock_hash="sha256:" + "d" * 64,
        protected_manifest_hash="sha256:" + "e" * 64,
        topology_hash="sha256:" + "f" * 64,
    )
    write_identity(
        identity,
        gateway_root / "_attested_runtime" / "gateway_enclave_build_identity.json",
    )
    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    observed = {}

    def _attest(*, user_data, signing_pubkey):
        observed["user_data"] = bytes(user_data)
        observed["signing_pubkey"] = bytes(signing_pubkey)
        return b"real-nitro-document"

    return (
        RuntimeIdentityV2(
            gateway_root=gateway_root,
            physical_role="gateway_coordinator",
            signing_pubkey_supplier=lambda: signing_pubkey,
            pcr0_supplier=lambda: pcr0,
            attestation_supplier=_attest,
        ),
        observed,
        signing_pubkey,
    )


def _configuration():
    release_roles = {
        "gateway_coordinator": {
            "commit_sha": "c" * 40,
            "pcr0": "b" * 96,
            "build_manifest_hash": HASH,
            "dependency_lock_hash": "sha256:" + "d" * 64,
        }
    }
    return {
        "bootstrap_schema_version": "leadpoet.gateway_v2_bootstrap.v2",
        "release_hash": HASH,
        "release_commit_sha": "c" * 40,
        "own_build_identity_hash": build_identity(
            role="gateway_coordinator",
            service_role="gateway_coordinator",
            commit_sha="c" * 40,
            execution_manifest_hash=HASH,
            dependency_lock_hash="sha256:" + "d" * 64,
            protected_manifest_hash="sha256:" + "e" * 64,
            topology_hash="sha256:" + "f" * 64,
        )["identity_hash"],
        "release_roles": release_roles,
        "peer_releases": {},
        "protected_workflow_manifest_hash": "sha256:" + "e" * 64,
    }


def _configuration_hash(configuration):
    return sha256_json(
        {
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "physical_role": "gateway_coordinator",
            "service_role": "gateway_coordinator",
            "configuration": configuration,
        }
    )


def test_runtime_boot_identity_binds_role_build_config_tls_and_nitro(tmp_path: Path):
    manager, observed, signing_pubkey = _manager(tmp_path)
    configuration = _configuration()
    assert set(configuration) == {
        "bootstrap_schema_version",
        "release_hash",
        "release_commit_sha",
        "own_build_identity_hash",
        "release_roles",
        "peer_releases",
        "protected_workflow_manifest_hash",
    }
    assert configuration["peer_releases"] == {}
    status = manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
    assert status["status"] == "ready"
    boot = manager.boot_identity()
    validate_boot_identity(boot)
    assert boot["physical_role"] == "gateway_coordinator"
    assert boot["role"] == "gateway_coordinator"
    assert boot["signing_pubkey"] == signing_pubkey
    assert boot["dependency_lock_hash"] == "sha256:" + "d" * 64
    assert json.loads(observed["user_data"]) == build_boot_attestation_user_data(boot)
    assert observed["signing_pubkey"] == bytes.fromhex(signing_pubkey)
    assert manager.transport_certificate_pem().startswith(b"-----BEGIN CERTIFICATE-----")


def test_runtime_configuration_is_immutable_for_boot(tmp_path: Path):
    manager, _, _ = _manager(tmp_path)
    configuration = _configuration()
    first = manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
    assert manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    ) == first
    changed = copy.deepcopy(configuration)
    changed["release_hash"] = "sha256:" + "8" * 64
    with pytest.raises(RuntimeIdentityV2Error, match="immutable"):
        manager.configure(
            configuration=changed,
            expected_config_hash=_configuration_hash(changed),
        )


def test_runtime_identity_rejects_zero_pcr_and_secret_material(tmp_path: Path):
    manager, _, _ = _manager(tmp_path, pcr0="0" * 96)
    configuration = _configuration()
    with pytest.raises(RuntimeIdentityV2Error, match="PCR0"):
        manager.configure(
            configuration=configuration,
            expected_config_hash=_configuration_hash(configuration),
        )

    manager, _, _ = _manager(tmp_path / "secret")
    unsafe = {"openrouter_api_key": "sk-or-secret"}
    with pytest.raises(RuntimeIdentityV2Error, match="secret"):
        manager.configure(
            configuration=unsafe,
            expected_config_hash=_configuration_hash(unsafe),
        )


def test_config_hash_is_canonical_and_order_independent(tmp_path: Path):
    manager, _, _ = _manager(tmp_path)
    configuration = _configuration()
    reversed_configuration = dict(reversed(list(configuration.items())))
    assert canonical_json(configuration) == canonical_json(reversed_configuration)
    manager.configure(
        configuration=reversed_configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
