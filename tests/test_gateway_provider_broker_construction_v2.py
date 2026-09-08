import importlib
import json
from pathlib import Path

from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, SubnetEpochCutover
from gateway.tee import provider_broker_v2, rpc_authority
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from tests.v2_epoch_test_utils import epoch_test_environment


def _testnet401_environment():
    cutover = SubnetEpochCutover(
        network_genesis_hash=(
            "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        ),
        netuid=401,
        cutover_block=7_700_000,
        cutover_block_hash="0x" + "4" * 64,
        first_subnet_epoch_index=1,
        first_settlement_epoch_id=1,
        last_legacy_epoch_id=0,
    )
    return {
        "BITTENSOR_NETWORK": "test",
        "BITTENSOR_NETUID": "401",
        CUTOVER_JSON_ENV: json.dumps(cutover.to_dict()),
    }


def test_testnet401_expected_chain_uses_validated_runtime_profile(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    tee_service = importlib.import_module("gateway.tee.tee_service")
    execution_config = build_research_lab_execution_config(
        environment=_testnet401_environment()
    )

    assert tee_service._v2_expected_chain(
        {"research_lab_execution_config": execution_config}
    ) == "wss://test.finney.opentensor.ai:443"


def test_coordinator_provider_broker_serializes_request_scoped_direct_transport(
    monkeypatch,
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    tee_service = importlib.import_module("gateway.tee.tee_service")
    captured = {}
    execution_config = build_research_lab_execution_config(
        environment=epoch_test_environment()
    )

    class RuntimeIdentity:
        @staticmethod
        def runtime_configuration():
            return {
                "configuration": {
                    "provider_ref_hashes": {},
                    "provider_retry_policy_hashes": {},
                    "provider_registry_hash": (
                        provider_broker_v2.provider_registry_hash()
                    ),
                    "job_lease_slot_ref_hashes": {},
                    "research_lab_execution_config": execution_config,
                }
            }

    class ArtifactVault:
        @staticmethod
        def seal(*_args, **_kwargs):
            raise AssertionError("artifact sealing is outside broker construction")

    class Transport:
        def __init__(self, **kwargs):
            captured["transport"] = dict(kwargs)

    class Broker:
        def __init__(self, **kwargs):
            captured["broker"] = dict(kwargs)

    class Proxy:
        @staticmethod
        def ensure_running():
            return {"status": "running"}

    monkeypatch.setattr(tee_service, "v2_provider_broker", None)
    monkeypatch.setattr(
        tee_service,
        "get_v2_runtime_identity",
        lambda: RuntimeIdentity(),
    )
    monkeypatch.setattr(tee_service, "get_provider_egress_proxy", lambda: Proxy())
    monkeypatch.setattr(tee_service, "get_v2_artifact_vault", lambda: ArtifactVault())
    monkeypatch.setattr(rpc_authority, "active_enclave_role", lambda: "gateway_coordinator")
    monkeypatch.setattr(provider_broker_v2, "HTTPXProviderTransport", Transport)
    monkeypatch.setattr(provider_broker_v2, "ProviderBrokerV2", Broker)

    broker = tee_service.get_v2_provider_broker()

    assert isinstance(broker, Broker)
    transport_options = dict(captured["transport"])
    ensure_egress_ready = transport_options.pop("ensure_egress_ready")
    assert callable(ensure_egress_ready)
    assert ensure_egress_ready() == {"status": "running"}
    assert transport_options == {
        "allow_authenticated_complete_body_eof": True,
        "reuse_direct_connections": False,
        "reuse_upstream_proxy_connections": False,
    }
    assert captured["broker"]["transport"].__class__ is Transport
    assert captured["broker"]["routes"]["supabase"].hosts == (
        "qplwoislplkcegvdmbim.supabase.co",
    )


def test_coordinator_provider_broker_singleton_is_restart_safe(
    monkeypatch,
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    tee_service = importlib.import_module("gateway.tee.tee_service")
    execution_config = build_research_lab_execution_config(
        environment=epoch_test_environment()
    )

    class RuntimeIdentity:
        @staticmethod
        def runtime_configuration():
            return {
                "configuration": {
                    "provider_ref_hashes": {},
                    "provider_retry_policy_hashes": {},
                    "provider_registry_hash": provider_broker_v2.provider_registry_hash(),
                    "job_lease_slot_ref_hashes": {},
                    "research_lab_execution_config": execution_config,
                }
            }

    class ArtifactVault:
        @staticmethod
        def seal(*_args, **_kwargs):
            raise AssertionError("artifact sealing is outside broker construction")

    class Transport:
        def __init__(self, **_kwargs):
            pass

    class Broker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Proxy:
        @staticmethod
        def ensure_running():
            return {"status": "running"}

    monkeypatch.setattr(tee_service, "v2_provider_broker", None)
    monkeypatch.setattr(tee_service, "get_v2_runtime_identity", lambda: RuntimeIdentity())
    monkeypatch.setattr(tee_service, "get_provider_egress_proxy", lambda: Proxy())
    monkeypatch.setattr(tee_service, "get_v2_artifact_vault", lambda: ArtifactVault())
    monkeypatch.setattr(rpc_authority, "active_enclave_role", lambda: "gateway_coordinator")
    monkeypatch.setattr(provider_broker_v2, "HTTPXProviderTransport", Transport)
    monkeypatch.setattr(provider_broker_v2, "ProviderBrokerV2", Broker)

    first = tee_service.get_v2_provider_broker()
    second = tee_service.get_v2_provider_broker()
    assert first is second
    assert "routing_authorization_validator" not in first.kwargs
