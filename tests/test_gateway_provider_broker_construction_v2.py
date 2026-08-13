import importlib
from pathlib import Path

from gateway.tee import provider_broker_v2, rpc_authority


def test_coordinator_provider_broker_uses_raw_isolated_transport(
    monkeypatch,
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    tee_service = importlib.import_module("gateway.tee.tee_service")
    captured = {"transports": []}

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
                }
            }

    class ArtifactVault:
        @staticmethod
        def seal(*_args, **_kwargs):
            raise AssertionError("artifact sealing is outside broker construction")

    class Transport:
        def __init__(self, **kwargs):
            self.options = dict(kwargs)
            captured["transports"].append(self)

    class Broker:
        def __init__(self, **kwargs):
            captured["broker"] = dict(kwargs)

    monkeypatch.setattr(tee_service, "v2_provider_broker", None)
    monkeypatch.setattr(
        tee_service,
        "get_v2_runtime_identity",
        lambda: RuntimeIdentity(),
    )
    monkeypatch.setattr(tee_service, "get_provider_egress_proxy", lambda: object())
    monkeypatch.setattr(tee_service, "get_v2_artifact_vault", lambda: ArtifactVault())
    monkeypatch.setattr(rpc_authority, "active_enclave_role", lambda: "gateway_coordinator")
    monkeypatch.setattr(provider_broker_v2, "HTTPXProviderTransport", Transport)
    monkeypatch.setattr(provider_broker_v2, "ProviderBrokerV2", Broker)

    broker = tee_service.get_v2_provider_broker()

    assert isinstance(broker, Broker)
    assert len(captured["transports"]) == 2
    assert all(
        transport.options
        == {
            "allow_authenticated_complete_body_eof": True,
            "reuse_direct_connections": False,
        }
        for transport in captured["transports"]
    )
    assert captured["broker"]["transport"] is captured["transports"][0]
    assert (
        captured["broker"]["reserved_weight_transport"]
        is captured["transports"][1]
    )
    assert captured["broker"]["reserved_weight_transport_slots"] == 1
