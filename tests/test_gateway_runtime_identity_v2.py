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
    lineage_roles = dict(release_roles)
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
        "gateway_release_lineage": {
            **lineage_body,
            "lineage_hash": sha256_json(lineage_body),
        },
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
        "gateway_release_lineage",
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
    changed = copy.deepcopy(configuration)
    changed_release_hash = "sha256:" + "8" * 64
    changed["release_hash"] = changed_release_hash
    changed_lineage = changed["gateway_release_lineage"]
    changed_lineage["current_gateway_release_hash"] = changed_release_hash
    changed_lineage["releases"]["c" * 40][
        "gateway_release_hash"
    ] = changed_release_hash
    changed_lineage_body = {
        key: value for key, value in changed_lineage.items() if key != "lineage_hash"
    }
    changed_lineage["lineage_hash"] = sha256_json(changed_lineage_body)
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
