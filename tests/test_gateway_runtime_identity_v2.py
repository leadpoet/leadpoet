from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.build_identity import build_identity, write_identity
from gateway.tee.provider_broker_v2 import (
    expected_job_credential_slot_ref_hashes,
    provider_registry_hash,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
    research_lab_execution_config_hash,
)
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
from tests.v2_epoch_test_utils import epoch_test_environment


HASH = "sha256:" + "a" * 64


def _manager(tmp_path: Path, *, pcr0: str = "b" * 96):
    gateway_root = tmp_path / "gateway"
    identity = build_identity(
        role="gateway_scoring",
        service_role="gateway_scoring",
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
            physical_role="gateway_scoring",
            signing_pubkey_supplier=lambda: signing_pubkey,
            pcr0_supplier=lambda: pcr0,
            attestation_supplier=_attest,
        ),
        observed,
        signing_pubkey,
    )


def _configuration():
    artifact_policy = {
        "schema_version": "leadpoet.encrypted_artifact_policy.v2",
        "bucket_host": "immutable.example.s3.us-east-1.amazonaws.com",
        "key_prefix": "/attested-v2/artifacts/",
        "minimum_retention_days": 365,
    }
    role_pcr0s = {
        "gateway_coordinator": "1" * 96,
        "gateway_scoring": "b" * 96,
    }
    release_roles = {
        role: {
            "commit_sha": "c" * 40,
            "pcr0": role_pcr0s[role],
            "build_manifest_hash": HASH,
            "dependency_lock_hash": "sha256:" + "d" * 64,
        }
        for role in (
            "gateway_coordinator",
            "gateway_scoring",
        )
    }
    lineage_roles = {
        **release_roles,
        "validator_weights": {
            "commit_sha": "c" * 40,
            "pcr0": "5" * 96,
            "build_manifest_hash": HASH,
            "dependency_lock_hash": "sha256:" + "d" * 64,
        },
    }
    lineage_body = {
        "schema_version": "leadpoet.attested_release_lineage.v1",
        "current_commit_sha": "c" * 40,
        "current_gateway_release_hash": HASH,
        "releases": {
            "c" * 40: {
                "channel_hash": "sha256:" + "9" * 64,
                "gateway_release_hash": HASH,
                "roles": lineage_roles,
            }
        },
    }
    research_lab_config = build_research_lab_execution_config(
        environment=epoch_test_environment()
    )
    return {
        "bootstrap_schema_version": "leadpoet.gateway_v2_bootstrap.v2",
        "release_hash": HASH,
        "release_commit_sha": "c" * 40,
        "own_build_identity_hash": build_identity(
            role="gateway_scoring",
            service_role="gateway_scoring",
            commit_sha="c" * 40,
            execution_manifest_hash=HASH,
            dependency_lock_hash="sha256:" + "d" * 64,
            protected_manifest_hash="sha256:" + "e" * 64,
            topology_hash="sha256:" + "f" * 64,
        )["identity_hash"],
        "release_roles": release_roles,
        "gateway_release_lineage": {
            **lineage_body,
            "lineage_hash": sha256_json(lineage_body),
        },
        "peer_releases": {
            "gateway_coordinator": {
                "commit_sha": "c" * 40,
                "pcr0": "1" * 96,
                "build_manifest_hash": HASH,
            }
        },
        "provider_ref_hashes": {"openrouter": "sha256:" + "1" * 64},
        "job_lease_slot_ref_hashes": (
            expected_job_credential_slot_ref_hashes()
        ),
        "provider_retry_policy_hashes": {
            "openrouter": "sha256:" + "2" * 64
        },
        "provider_registry_hash": provider_registry_hash(
            execution_config=research_lab_config
        ),
        "protected_workflow_manifest_hash": "sha256:" + "e" * 64,
        "encrypted_artifact_policy": artifact_policy,
        "encrypted_artifact_policy_hash": sha256_json(artifact_policy),
        "artifact_master_key_ref_hash": "sha256:" + "4" * 64,
        "research_lab_execution_config": research_lab_config,
        "research_lab_execution_config_hash": (
            research_lab_execution_config_hash(research_lab_config)
        ),
        "execution_worker_count": 10,
        "configured_worker_count": 25,
    }


def _configuration_hash(configuration):
    return sha256_json(
        {
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "physical_role": "gateway_scoring",
            "service_role": "gateway_scoring",
            "configuration": configuration,
        }
    )


def test_runtime_boot_identity_binds_role_build_config_tls_and_nitro(tmp_path: Path):
    manager, observed, signing_pubkey = _manager(tmp_path)
    configuration = _configuration()
    status = manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
    assert status["status"] == "ready"
    boot = manager.boot_identity()
    validate_boot_identity(boot)
    assert boot["physical_role"] == "gateway_scoring"
    assert boot["role"] == "gateway_scoring"
    assert boot["signing_pubkey"] == signing_pubkey
    assert boot["dependency_lock_hash"] == "sha256:" + "d" * 64
    assert json.loads(observed["user_data"]) == build_boot_attestation_user_data(boot)
    assert observed["signing_pubkey"] == bytes.fromhex(signing_pubkey)
    assert manager.transport_certificate_pem().startswith(b"-----BEGIN CERTIFICATE-----")


def test_runtime_verifies_checkpoint_boot_against_hash_bound_release_lineage(
    tmp_path: Path, monkeypatch
):
    from gateway.tee import release_lineage_v2

    observed = []
    monkeypatch.setattr(
        release_lineage_v2,
        "verify_boot_identity_nitro",
        lambda identity, **kwargs: observed.append(kwargs) or identity,
    )
    manager, _, _ = _manager(tmp_path)
    configuration = _configuration()
    manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
    boot = manager.boot_identity()
    assert manager.verify_release_lineage_boot(boot) == boot
    assert observed[0]["expected_pcr0"] == boot["pcr0"]

    with pytest.raises(Exception, match="pcr0"):
        manager.verify_release_lineage_boot({**boot, "pcr0": "9" * 96})


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
    changed = {**configuration, "configured_worker_count": 24}
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


def test_runtime_reconstructs_exact_measured_research_lab_config(tmp_path: Path):
    manager, _, _ = _manager(tmp_path)
    configuration = _configuration()
    fields = configuration["research_lab_execution_config"]["fields"]
    fields["source_add_leg1_alpha_percent"] = 0.35
    fields["lab_champion_threshold_points"] = 2.75
    configuration["research_lab_execution_config_hash"] = (
        research_lab_execution_config_hash(
            configuration["research_lab_execution_config"]
        )
    )
    manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
    assert manager.research_lab_config().source_add_leg1_alpha_percent == 0.35
    assert manager.research_lab_config().lab_champion_threshold_points == 2.75


def test_runtime_configuration_accepts_the_fixed_512_release_bound(tmp_path: Path):
    manager, _, _ = _manager(tmp_path)
    configuration = _configuration()
    lineage = configuration["gateway_release_lineage"]
    template = next(iter(lineage["releases"].values()))
    releases = {}
    for index in range(511):
        commit = f"{index:040x}"
        entry = copy.deepcopy(template)
        entry["roles"] = {
            role: {**expectation, "commit_sha": commit}
            for role, expectation in entry["roles"].items()
        }
        releases[commit] = entry
    releases["c" * 40] = template
    body = {
        "schema_version": "leadpoet.attested_release_lineage.v1",
        "current_commit_sha": "c" * 40,
        "current_gateway_release_hash": HASH,
        "releases": {commit: releases[commit] for commit in sorted(releases)},
    }
    configuration["gateway_release_lineage"] = {
        **body,
        "lineage_hash": sha256_json(body),
    }
    assert len(canonical_json(configuration).encode("utf-8")) < 1024 * 1024
    manager.configure(
        configuration=configuration,
        expected_config_hash=_configuration_hash(configuration),
    )
