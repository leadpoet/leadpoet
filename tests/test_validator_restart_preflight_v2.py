from __future__ import annotations

import base64
import copy
import json

import pytest

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    SubnetEpochCutover,
)
from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
)
from validator_tee.host.hotkey_bootstrap_v2 import (
    HOTKEY_ENVELOPE_SCHEMA_VERSION,
)
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)
from validator_tee.host.restart_preflight_v2 import (
    ValidatorRestartPreflightV2Error,
    verify_host_hotkey_directory_empty_v2,
    verify_validator_restart_preflight_v2,
)
from validator_tee.scripts.stage_runtime_artifacts_v2 import SCHEMA_VERSION


COMMIT = "1" * 40
HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
PUBLIC_KEY = "a6bfe69c29bf9e4db65c63ac6f6d1e23c252ca871744afb6edc5623d9bc39004"


def _hash(character):
    return "sha256:" + character * 64


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


def _gateway_release(commit=COMMIT):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "2345"[index]
        values = {
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


def _validator_release(commit=COMMIT):
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


def _gateway_lineage(commit=COMMIT):
    return build_release_lineage_v2(
        [
            build_release_channel_v2(
                gateway_release_manifest=_gateway_release(commit),
                validator_release_manifest=_validator_release(commit),
            )
        ],
        current_commit=commit,
    )


def _hotkey_config():
    return {
        "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": PUBLIC_KEY,
        "chain_signing_profile_hash": _hash("2"),
        "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
        "drand_library_sha256": "3" * 64,
    }


def _hotkey_envelope():
    ciphertext = b"kms-encrypted-validator-hotkey"
    context = {"service": "leadpoet", "purpose": "validator-hotkey-v2"}
    return {
        "schema_version": HOTKEY_ENVELOPE_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": PUBLIC_KEY,
        "ciphertext_blob_b64": base64.b64encode(ciphertext).decode("ascii"),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id_hash": _hash("4"),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
    }


def _runtime_lock():
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "sr25519_cp37": {
                "filename": "sr25519.whl",
                "sha256": "5" * 64,
                "url": "https://files.pythonhosted.org/sr25519.whl",
            }
        },
    }


def _verify(**overrides):
    values = {
        "deploy_commit": COMMIT,
        "validator_release_manifest": _validator_release(),
        "gateway_release_manifest": _gateway_release(),
        "gateway_release_lineage": _gateway_lineage(),
        "hotkey_configuration": _hotkey_config(),
        "hotkey_envelope": _hotkey_envelope(),
        "runtime_artifact_lock": _runtime_lock(),
    }
    values.update(overrides)
    return verify_validator_restart_preflight_v2(**values)


def test_validator_preflight_accepts_exact_cross_host_release_and_hotkey():
    result = _verify()
    assert result["status"] == "ready"
    assert result["deploy_commit"] == COMMIT
    assert result["validator_hotkey"] == HOTKEY
    assert result["runtime_artifact_count"] == 1


def test_validator_preflight_rejects_release_for_another_commit():
    with pytest.raises(
        ValidatorRestartPreflightV2Error,
        match="validator V2 release is for another commit",
    ):
        _verify(deploy_commit="2" * 40)


def test_validator_preflight_rejects_gateway_release_for_another_commit():
    with pytest.raises(
        ValidatorRestartPreflightV2Error,
        match="gateway V2 release is for another commit",
    ):
        _verify(gateway_release_manifest=_gateway_release("2" * 40))


def test_validator_preflight_rejects_hotkey_envelope_substitution():
    envelope = copy.deepcopy(_hotkey_envelope())
    envelope["hotkey_public_key"] = "6" * 64
    with pytest.raises(
        ValidatorRestartPreflightV2Error,
        match="differs from measured configuration",
    ):
        _verify(hotkey_envelope=envelope)


def test_validator_preflight_rejects_any_parent_hotkey_entry(tmp_path):
    hotkey_directory = tmp_path / "wallet" / "hotkeys"
    hotkey_directory.mkdir(parents=True)
    (hotkey_directory / "default.backup").write_text("secret", encoding="utf-8")
    with pytest.raises(
        ValidatorRestartPreflightV2Error,
        match="usable validator hotkey material remains on the parent",
    ):
        verify_host_hotkey_directory_empty_v2(hotkey_directory)


def test_validator_preflight_accepts_missing_or_empty_hotkey_directory(tmp_path):
    missing = tmp_path / "missing-hotkeys"
    assert verify_host_hotkey_directory_empty_v2(missing) == str(missing)
    missing.mkdir()
    assert verify_host_hotkey_directory_empty_v2(missing) == str(missing)


def test_validator_preflight_rejects_hotkey_directory_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "hotkeys"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        ValidatorRestartPreflightV2Error,
        match="must not be a symlink",
    ):
        verify_host_hotkey_directory_empty_v2(link)
