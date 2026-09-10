"""Endpoint policy checks retained by the Arena chain signer."""
import json
from pathlib import Path
import pytest
from leadpoet_canonical.chain_source_v2 import ChainSourceV2Error, chain_source_boundary_for_profile_v2


@pytest.mark.parametrize(
    "endpoint",
    (
        "wss://entrypoint-finney.opentensor.ai:8443",
        "wss://entrypoint-finney.opentensor.ai:not-a-port",
        "wss://user:password@entrypoint-finney.opentensor.ai:443",
        "wss://entrypoint-finney.opentensor.ai:443/rpc",
        "wss://entrypoint-finney.opentensor.ai:443?query=value",
        "wss://entrypoint-finney.opentensor.ai:443#fragment",
    ),
)
def test_chain_source_boundary_rejects_endpoint_authority_suffixes(endpoint):
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile["chain_endpoint"] = endpoint

    with pytest.raises(ChainSourceV2Error, match="outside measured policy"):
        chain_source_boundary_for_profile_v2(profile)


def test_chain_source_boundary_retains_measured_test_network_support():
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "validator_tee/enclave/chain_signing_profile_test_v2.json"
        ).read_text(encoding="utf-8")
    )

    boundary = chain_source_boundary_for_profile_v2(profile)

    assert boundary["chain_host"] == "test.finney.opentensor.ai"
    assert boundary["chain_archive_host"] == "test.finney.opentensor.ai"
