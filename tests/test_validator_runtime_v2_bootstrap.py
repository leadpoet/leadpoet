from __future__ import annotations

import copy
import json

import pytest

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    SubnetEpochCutover,
    SubnetEpochError,
)

from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)
from validator_tee.host.runtime_v2_bootstrap import (
    ValidatorRuntimeBootstrapV2Error,
    build_runtime_configuration,
    configure_validator_runtime_v2,
)
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
)
from validator_tee.enclave.runtime_v2 import (
    VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
)


def _hash(character):
    return "sha256:" + character * 64


def _gateway_release(commit="1" * 40):
    evidence = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "2345"[index]
        deterministic = {
            "commit_sha": commit,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("6"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("7"),
            "dockerfile_hash": _hash("8"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                evidence.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return build_release_manifest(
        evidence, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


def _validator_release(commit="1" * 40):
    release = build_validator_release(
        commit_sha=commit,
        pcr0="9" * 96,
        app_manifest_hash=_hash("a"),
        dependency_lock_hash=_hash("b"),
        normalized_image_hash=_hash("c"),
        eif_hash=_hash("d"),
        dockerfile_hash=_hash("e"),
        base_dockerfile_hash=_hash("f"),
    )
    return build_validator_release_manifest(
        [
            build_validator_build_evidence(
                release,
                builder_domain=domain,
                builder_id=domain + "-parent",
                build_ordinal=ordinal,
            )
            for domain in ("gateway", "validator")
            for ordinal in (1, 2, 3)
        ]
    )


def _hotkey_config():
    return {
        "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
        "validator_hotkey": "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK",
        "hotkey_public_key": "1" * 64,
        "chain_signing_profile_hash": _hash("2"),
        "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
        "drand_library_sha256": "3" * 64,
    }


@pytest.fixture(autouse=True)
def _official_epoch_authority(monkeypatch):
    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=8_637_156,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=23_927,
        first_settlement_epoch_id=23_992,
        last_legacy_epoch_id=23_991,
    )
    monkeypatch.setenv(CUTOVER_JSON_ENV, json.dumps(cutover.to_dict()))
    monkeypatch.delenv(CUTOVER_PATH_ENV, raising=False)


def test_runtime_configuration_requires_cutover_authority(monkeypatch):
    monkeypatch.delenv(CUTOVER_JSON_ENV, raising=False)
    monkeypatch.delenv(CUTOVER_PATH_ENV, raising=False)
    with pytest.raises(SubnetEpochError, match="requires a cutover manifest"):
        build_runtime_configuration(
            validator_release=_validator_release(),
            gateway_release=_gateway_release(),
            hotkey_authority_config=_hotkey_config(),
        )


def test_stateful_runtime_configuration_measures_cutover(monkeypatch):
    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=8_637_156,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=23_927,
        first_settlement_epoch_id=23_992,
        last_legacy_epoch_id=23_991,
    )
    monkeypatch.setenv(CUTOVER_JSON_ENV, json.dumps(cutover.to_dict()))
    monkeypatch.delenv(CUTOVER_PATH_ENV, raising=False)

    config = build_runtime_configuration(
        validator_release=_validator_release(),
        gateway_release=_gateway_release(),
        hotkey_authority_config=_hotkey_config(),
    )

    assert (
        config["schema_version"]
        == VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION
    )
    assert config["epoch_authority"] == {
        "mode": "stateful_v1",
        "cutover_manifest": cutover.to_dict(),
    }
    assert set(config["gateway_role_expectations"]) == set(ROLE_SPECS)


def test_runtime_configuration_rejects_asymmetric_commits():
    with pytest.raises(ValidatorRuntimeBootstrapV2Error, match="different commits"):
        build_runtime_configuration(
            validator_release=_validator_release("1" * 40),
            gateway_release=_gateway_release("2" * 40),
            hotkey_authority_config=_hotkey_config(),
        )


def test_bootstrap_verifies_nitro_and_exact_readback():
    validator_release = _validator_release()
    authorized_release = validator_release["release"]
    gateway_release = _gateway_release()
    configuration = build_runtime_configuration(
        validator_release=validator_release,
        gateway_release=gateway_release,
        hotkey_authority_config=_hotkey_config(),
    )
    boot = {
        "role": "validator_weights",
        "physical_role": "validator_weights",
        "commit_sha": authorized_release["commit_sha"],
        "pcr0": authorized_release["pcr0"],
        "build_manifest_hash": authorized_release["app_manifest_hash"],
        "dependency_lock_hash": authorized_release["dependency_lock_hash"],
        "config_hash": sha256_json(configuration),
        "boot_identity_hash": _hash("1"),
    }

    class Client:
        def configure_authoritative_v2(self, observed_configuration, observed_hash):
            assert observed_configuration == configuration
            assert observed_hash == sha256_json(configuration)
            return dict(boot)

        def get_authoritative_v2_boot_identity(self):
            return dict(boot)

    verified = []

    def verify(identity, *, expected_pcr0=None):
        assert identity == boot
        assert expected_pcr0 == authorized_release["pcr0"]
        verified.append(True)
        return {"verified": True}

    result = configure_validator_runtime_v2(
        validator_release=validator_release,
        gateway_release=gateway_release,
        hotkey_authority_config=_hotkey_config(),
        client=Client(),
        boot_verifier=verify,
    )
    assert verified == [True]
    assert result["boot_identity"] == boot


def test_bootstrap_rejects_readback_substitution():
    validator_release = _validator_release()
    authorized_release = validator_release["release"]
    gateway_release = _gateway_release()
    configuration = build_runtime_configuration(
        validator_release=validator_release,
        gateway_release=gateway_release,
        hotkey_authority_config=_hotkey_config(),
    )
    boot = {
        "role": "validator_weights",
        "physical_role": "validator_weights",
        "commit_sha": authorized_release["commit_sha"],
        "pcr0": authorized_release["pcr0"],
        "build_manifest_hash": authorized_release["app_manifest_hash"],
        "dependency_lock_hash": authorized_release["dependency_lock_hash"],
        "config_hash": sha256_json(configuration),
        "boot_identity_hash": _hash("1"),
    }

    class Client:
        def configure_authoritative_v2(self, *_args):
            return dict(boot)

        def get_authoritative_v2_boot_identity(self):
            changed = copy.deepcopy(boot)
            changed["boot_identity_hash"] = _hash("2")
            return changed

    with pytest.raises(ValidatorRuntimeBootstrapV2Error, match="readback"):
        configure_validator_runtime_v2(
            validator_release=validator_release,
            gateway_release=gateway_release,
            hotkey_authority_config=_hotkey_config(),
            client=Client(),
            boot_verifier=lambda *_args, **_kwargs: {"verified": True},
        )
