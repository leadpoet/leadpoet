import json
from pathlib import Path

import pytest

from gateway.research_lab import attested_coordinator_v2
from gateway.tee.coordinator_executor_v2 import (
    COORDINATOR_OPERATIONS_V2,
    OP_RESEARCH_LAB_ALLOCATION,
    coordinator_receipt_output_v2,
)
from gateway.tee.release_channel_v2 import (
    ReleaseChannelV2Error,
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.release_lineage_v2 import (
    ReleaseLineageV2Error,
    load_approved_release_lineage_v2,
)
from scripts import bootstrap_temporary_testnet_weights_host as bootstrap
from tests.test_release_channel_v2 import (
    _gateway_manifest,
    _local_gateway_manifest,
    _local_validator_manifest,
)


def _private_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


@pytest.mark.asyncio
async def test_coordinator_bridge_uses_strict_coordinator_role(monkeypatch):
    observed = {}

    async def execute(**kwargs):
        observed.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(attested_coordinator_v2, "execute_scoring_v2", execute)
    result = await attested_coordinator_v2.execute_coordinator_v2(
        operation=OP_RESEARCH_LAB_ALLOCATION,
        purpose="research_lab.allocation.v2",
        epoch_id=9,
        sequence=1,
        payload={"epoch": 9},
        client=object(),
    )
    assert result == {"status": "succeeded"}
    assert observed["operation_registry"] == COORDINATOR_OPERATIONS_V2
    assert observed["physical_role_override"] == "gateway_coordinator"
    assert observed["expected_service_role"] == "gateway_coordinator"
    assert observed["rpc_namespace"] == "coordinator_v2"
    assert observed["receipt_output_projector"] is coordinator_receipt_output_v2
    assert observed["allow_persistence_bound_artifact_descriptors"] is True
    assert observed["provider_profile_loader"]("default") == {
        "profile": "default",
        "credential_ref_hashes": {},
        "envelopes": [],
    }


@pytest.mark.asyncio
async def test_coordinator_bridge_preserves_explicit_provider_profile_loader(
    monkeypatch,
):
    observed = {}

    async def execute(**kwargs):
        observed.update(kwargs)
        return {"status": "succeeded"}

    def load_profile(profile, **_kwargs):
        return {
            "profile": profile,
            "credential_ref_hashes": {"openrouter": "sha256:" + ("a" * 64)},
            "envelopes": [],
        }

    monkeypatch.setattr(attested_coordinator_v2, "execute_scoring_v2", execute)
    await attested_coordinator_v2.execute_coordinator_v2(
        operation=OP_RESEARCH_LAB_ALLOCATION,
        purpose="research_lab.allocation.v2",
        epoch_id=9,
        sequence=1,
        payload={"epoch": 9},
        provider_profile_loader=load_profile,
        client=object(),
    )

    assert observed["provider_profile_loader"] is load_profile


@pytest.mark.asyncio
async def test_normal_testnet401_coordinator_uses_installed_release_and_local_lineage(
    monkeypatch, tmp_path
):
    prior_commit = "1" * 40
    current_commit = "2" * 40
    prior = build_release_channel_v2(
        gateway_release_manifest=_local_gateway_manifest(prior_commit),
        validator_release_manifest=_local_validator_manifest(prior_commit),
    )
    current_gateway = _local_gateway_manifest(current_commit)
    current_validator = _local_validator_manifest(current_commit)
    current = build_release_channel_v2(
        gateway_release_manifest=current_gateway,
        validator_release_manifest=current_validator,
    )
    lineage = build_release_lineage_v2(
        [prior, current], current_commit=current_commit
    )
    staged_gateway = tmp_path / "gateway-release.json"
    canonical_gateway = tmp_path / "tee" / "gateway-v2-release-manifest.json"
    prior_path = tmp_path / "prior-release-channel-v2.json"
    validator_path = tmp_path / "validator-release.json"
    lineage_path = tmp_path / "gateway-lineage.json"
    for path, value in (
        (staged_gateway, current_gateway),
        (prior_path, prior),
        (validator_path, current_validator),
        (lineage_path, lineage),
    ):
        _private_json(path, value)
    installed = bootstrap._install_canonical_gateway_release(
        {
            "candidate_sha": current_commit,
            "gateway": {"release_manifest": str(staged_gateway)},
        },
        destination=canonical_gateway,
    )
    assert bootstrap._install_canonical_gateway_release(
        {
            "candidate_sha": current_commit,
            "gateway": {"release_manifest": str(staged_gateway)},
        },
        destination=canonical_gateway,
    ) == installed
    monkeypatch.setattr(
        attested_coordinator_v2,
        "DEFAULT_RELEASE_MANIFEST_PATH",
        canonical_gateway,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_PRIOR_CHANNEL",
        prior_path,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_CURRENT_VALIDATOR_RELEASE",
        validator_path,
    )
    monkeypatch.setattr(
        attested_coordinator_v2,
        "_TEMPORARY_TESTNET401_RELEASE_LINEAGE",
        lineage_path,
    )
    monkeypatch.setitem(
        attested_coordinator_v2.execute_coordinator_v2.__kwdefaults__,
        "release_manifest_path",
        canonical_gateway,
    )
    monkeypatch.setenv(
        "LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS", "true"
    )
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setenv("BITTENSOR_NETUID", "401")
    monkeypatch.setenv("ALLOWED_NETUIDS", "401")

    observed = {}

    async def execute(**kwargs):
        assert kwargs["release_manifest"] is None
        assert kwargs["release_manifest_path"] == canonical_gateway
        loader = kwargs["release_channel_loader"]
        observed["loader"] = loader
        assert loader(prior_commit) == prior
        assert loader(current_commit) == current
        with pytest.raises(RuntimeError, match="not approved"):
            loader("3" * 40)
        return {"status": "succeeded"}

    monkeypatch.setattr(attested_coordinator_v2, "execute_scoring_v2", execute)
    result = await attested_coordinator_v2.execute_coordinator_v2(
        operation=OP_RESEARCH_LAB_ALLOCATION,
        purpose="research_lab.allocation.v2",
        epoch_id=22_058,
        sequence=0,
        payload={"epoch": 22_058, "netuid": 401},
        client=object(),
    )
    assert result == {"status": "succeeded"}
    assert canonical_gateway.stat().st_mode & 0o777 == 0o600
    approved = load_approved_release_lineage_v2(
        current_release=current_gateway,
        parent_graphs=[
            {
                "boot_identities": [
                    {"commit_sha": prior_commit, "physical_role": "gateway_coordinator"},
                    {"commit_sha": prior_commit, "physical_role": "validator_weights"},
                    {"commit_sha": current_commit, "physical_role": "validator_weights"},
                ]
            }
        ],
        release_channel_loader=observed["loader"],
    )
    assert approved[prior_commit] == {
        "gateway_release_manifest": prior["gateway_release_manifest"],
        "validator_release_manifest": prior["validator_release_manifest"],
    }
    assert approved[current_commit] == {
        "gateway_release_manifest": current["gateway_release_manifest"],
        "validator_release_manifest": current["validator_release_manifest"],
    }

    tampered_lineage = {**lineage, "lineage_hash": "sha256:" + "0" * 64}
    _private_json(lineage_path, tampered_lineage)
    with pytest.raises(ReleaseLineageV2Error):
        attested_coordinator_v2._temporary_testnet401_release_channel_loader(
            current_release_path=canonical_gateway
        )
    _private_json(lineage_path, lineage)
    _private_json(validator_path, _local_validator_manifest("3" * 40))
    with pytest.raises(ReleaseChannelV2Error):
        attested_coordinator_v2._temporary_testnet401_release_channel_loader(
            current_release_path=canonical_gateway
        )


@pytest.mark.asyncio
async def test_temporary_release_loader_rejects_wrong_network(monkeypatch):
    monkeypatch.setenv(
        "LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS", "true"
    )
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "401")
    monkeypatch.setenv("ALLOWED_NETUIDS", "401")
    with pytest.raises(RuntimeError, match="scope is invalid"):
        await attested_coordinator_v2.execute_coordinator_v2(
            operation=OP_RESEARCH_LAB_ALLOCATION,
            purpose="research_lab.allocation.v2",
            epoch_id=22_058,
            sequence=0,
            payload={"epoch": 22_058, "netuid": 401},
            client=object(),
        )


def test_canonical_release_installer_rejects_an_existing_different_release(
    tmp_path,
):
    source = tmp_path / "source.json"
    destination = tmp_path / "tee" / "gateway-v2-release-manifest.json"
    _private_json(source, _gateway_manifest("1" * 40))
    bootstrap._install_canonical_gateway_release(
        {
            "candidate_sha": "1" * 40,
            "gateway": {"release_manifest": str(source)},
        },
        destination=destination,
    )
    _private_json(source, _gateway_manifest("2" * 40))
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError,
        match="manifest differs",
    ):
        bootstrap._install_canonical_gateway_release(
            {
                "candidate_sha": "2" * 40,
                "gateway": {"release_manifest": str(source)},
            },
            destination=destination,
        )
