from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from validator_tee.enclave.runtime_v2 import (
    VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
    ValidatorRuntimeIdentityV2,
    ValidatorRuntimeV2Error,
    compute_app_manifest_hash,
)


def _hash(character):
    return "sha256:" + character * 64


def _configuration():
    commit = "a" * 40
    historical_commit = "e" * 40

    def roles(release_commit, characters):
        return {
            role: {
                "commit_sha": release_commit,
                "pcr0": character * 96,
                "build_manifest_hash": _hash(character),
                "dependency_lock_hash": _hash("9"),
            }
            for role, character in zip(
                (
                    "gateway_autoresearch",
                    "gateway_coordinator",
                    "gateway_scoring",
                ),
                characters,
            )
        }

    releases = {
        commit: {
            "channel_hash": _hash("1"),
            "gateway_release_hash": _hash("d"),
            "roles": roles(commit, "124"),
        },
        historical_commit: {
            "channel_hash": _hash("2"),
            "gateway_release_hash": _hash("e"),
            "roles": roles(historical_commit, "567"),
        },
    }
    lineage_body = {
        "schema_version": "leadpoet.attested_release_lineage.v1",
        "current_commit_sha": commit,
        "current_gateway_release_hash": _hash("d"),
        "releases": releases,
    }
    cutover_body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": "0x" + "1" * 64,
        "netuid": 71,
        "cutover_block": 8_637_156,
        "cutover_block_hash": "0x" + "2" * 64,
        "first_subnet_epoch_index": 23_927,
        "first_settlement_epoch_id": 23_992,
        "last_legacy_epoch_id": 23_991,
    }
    return {
        "schema_version": VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
        "commit_sha": commit,
        "build_manifest_hash": _hash("b"),
        "dependency_lock_hash": _hash("c"),
        "gateway_release_hash": _hash("d"),
        "hotkey_authority_config_hash": _hash("f"),
        "gateway_release_lineage": {
            **lineage_body,
            "lineage_hash": sha256_json(lineage_body),
        },
        "epoch_authority": {
            "mode": "stateful_v1",
            "cutover_manifest": {
                **cutover_body,
                "mapping_hash": sha256_json(cutover_body),
            },
        },
    }


def _stateful_configuration():
    return _configuration()


def _runtime(*, pcr0="f" * 96):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    observed = {}

    def attest(*, user_data, public_key):
        observed["user_data"] = bytes(user_data)
        observed["public_key"] = bytes(public_key)
        return b"nitro"

    return (
        ValidatorRuntimeIdentityV2(
            signing_pubkey_supplier=lambda: pubkey,
            app_manifest_supplier=lambda: _hash("b"),
            dependency_lock_supplier=lambda: _hash("c"),
            pcr0_supplier=lambda: pcr0,
            attestation_supplier=attest,
        ),
        observed,
        pubkey,
    )


def test_validator_boot_binds_hardware_release_and_gateway_lineage():
    runtime, observed, pubkey = _runtime()
    configuration = _configuration()
    boot = runtime.configure(
        configuration,
        expected_config_hash=sha256_json(configuration),
    )
    assert boot["role"] == "validator_weights"
    assert boot["physical_role"] == "validator_weights"
    assert boot["signing_pubkey"] == pubkey
    assert observed["public_key"] == bytes.fromhex(pubkey)
    lineage = runtime.gateway_release_lineage()
    assert lineage["a" * 40]["roles"]["gateway_coordinator"]["pcr0"] == "2" * 96
    assert lineage["e" * 40]["roles"]["gateway_scoring"]["pcr0"] == "7" * 96


def test_validator_runtime_rejects_zero_pcr_and_manifest_drift():
    configuration = _configuration()
    runtime, _, _ = _runtime(pcr0="0" * 96)
    with pytest.raises(ValidatorRuntimeV2Error, match="PCR0"):
        runtime.configure(
            configuration,
            expected_config_hash=sha256_json(configuration),
        )

    runtime, _, _ = _runtime()
    changed = {**configuration, "build_manifest_hash": _hash("9")}
    with pytest.raises(ValidatorRuntimeV2Error, match="app manifest"):
        runtime.configure(changed, expected_config_hash=sha256_json(changed))


def test_stateful_runtime_binds_cutover_and_returns_defensive_copy():
    runtime, _, _ = _runtime()
    configuration = _stateful_configuration()
    runtime.configure(
        configuration,
        expected_config_hash=sha256_json(configuration),
    )

    authority = runtime.epoch_authority()
    assert authority == configuration["epoch_authority"]
    authority["cutover_manifest"]["netuid"] = 1
    assert runtime.epoch_authority()["cutover_manifest"]["netuid"] == 71


def test_stateful_runtime_rejects_tampered_cutover_mapping():
    runtime, _, _ = _runtime()
    configuration = _stateful_configuration()
    configuration["epoch_authority"]["cutover_manifest"]["mapping_hash"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(ValidatorRuntimeV2Error, match="hash mismatch"):
        runtime.configure(
            configuration,
            expected_config_hash=sha256_json(configuration),
        )


def test_runtime_rejects_tampered_or_incomplete_gateway_release_lineage():
    runtime, _, _ = _runtime()
    configuration = _configuration()
    configuration["gateway_release_lineage"]["releases"]["e" * 40]["roles"][
        "gateway_scoring"
    ]["pcr0"] = "8" * 96
    with pytest.raises(ValidatorRuntimeV2Error, match="lineage hash mismatch"):
        runtime.configure(
            configuration,
            expected_config_hash=sha256_json(configuration),
        )

    runtime, _, _ = _runtime()
    configuration = _configuration()
    del configuration["gateway_release_lineage"]["releases"]["e" * 40]["roles"][
        "gateway_scoring"
    ]
    with pytest.raises(ValidatorRuntimeV2Error, match="roles are incomplete"):
        runtime.configure(
            configuration,
            expected_config_hash=sha256_json(configuration),
        )


def test_app_manifest_hash_covers_paths_modes_and_bytes(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    target = tmp_path / "pkg" / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    first = compute_app_manifest_hash(tmp_path)
    target.write_text("value = 2\n", encoding="utf-8")
    assert compute_app_manifest_hash(tmp_path) != first
