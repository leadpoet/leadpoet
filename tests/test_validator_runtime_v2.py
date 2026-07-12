from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from validator_tee.enclave.runtime_v2 import (
    VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION,
    ValidatorRuntimeIdentityV2,
    ValidatorRuntimeV2Error,
    compute_app_manifest_hash,
)


def _hash(character):
    return "sha256:" + character * 64


def _configuration():
    return {
        "schema_version": VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION,
        "commit_sha": "a" * 40,
        "build_manifest_hash": _hash("b"),
        "dependency_lock_hash": _hash("c"),
        "gateway_release_hash": _hash("d"),
        "hotkey_authority_config_hash": _hash("f"),
        "gateway_role_expectations": {
            role: {
                "commit_sha": "e" * 40,
                "pcr0": character * 96,
                "build_manifest_hash": _hash(character),
            }
            for role, character in (
                ("gateway_coordinator", "1"),
                ("gateway_scoring_a", "2"),
                ("gateway_scoring_b", "3"),
                ("gateway_autoresearch", "4"),
            )
        },
    }


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


def test_validator_boot_binds_hardware_release_and_gateway_roles():
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
    assert runtime.gateway_expectations()["gateway_scoring_b"]["pcr0"] == "3" * 96


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


def test_app_manifest_hash_covers_paths_modes_and_bytes(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    target = tmp_path / "pkg" / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    first = compute_app_manifest_hash(tmp_path)
    target.write_text("value = 2\n", encoding="utf-8")
    assert compute_app_manifest_hash(tmp_path) != first
