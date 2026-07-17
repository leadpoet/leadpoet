from __future__ import annotations

from validator_tee.enclave import tee_service
from validator_tee.host.vsock_client import ValidatorEnclaveClient


def test_validator_rpc_configures_authority_once_and_disables_legacy(monkeypatch):
    calls = {}

    class FakeRuntime:
        def __init__(self, *, signing_pubkey_supplier):
            calls["pubkey"] = signing_pubkey_supplier()
            self._boot = {"boot_identity_hash": "sha256:" + "1" * 64}

        def configure(self, configuration, *, expected_config_hash):
            calls["configuration"] = dict(configuration)
            calls["expected_config_hash"] = expected_config_hash
            return dict(self._boot)

        def boot_identity(self):
            return dict(self._boot)

        def gateway_expectations(self):
            return {"gateway_coordinator": {"pcr0": "2" * 96}}

    class FakeAuthority:
        def __init__(self, **kwargs):
            calls["authority_kwargs"] = set(kwargs)

        def compute(self, request):
            calls["weight_request"] = dict(request)
            return {
                "weight_snapshot": {"epoch_id": 7},
                "weight_result": {"weights_hash": "3" * 64},
                "weights_signature": "4" * 128,
                "receipt_graph": {"root_receipt_hash": "sha256:" + "5" * 64},
                "boot_identity": {"boot_identity_hash": "sha256:" + "1" * 64},
            }

    class FakeHotkeyAuthority:
        def register_weight_result(self, result):
            calls["registered_weight_result"] = dict(result)
            return "sha256:" + "9" * 64

    monkeypatch.setattr(
        "validator_tee.enclave.runtime_v2.ValidatorRuntimeIdentityV2",
        FakeRuntime,
    )
    monkeypatch.setattr(
        "validator_tee.enclave.weight_authority_v2.ValidatorWeightAuthorityV2",
        FakeAuthority,
    )
    monkeypatch.setattr(tee_service, "validator_runtime_v2", None)
    monkeypatch.setattr(tee_service, "validator_weight_authority_v2", None)
    monkeypatch.setattr(
        tee_service,
        "validator_hotkey_authority_v2",
        FakeHotkeyAuthority(),
    )
    monkeypatch.setattr(tee_service, "get_public_key", lambda: "6" * 64)

    response = tee_service.handle_request(
        {
            "command": "configure_authoritative_v2",
            "configuration": {"schema_version": "fixture"},
            "expected_config_hash": "sha256:" + "7" * 64,
        }
    )
    assert response["status"] == "ok"
    assert response["boot_identity"]["boot_identity_hash"].endswith("1" * 64)
    assert calls["pubkey"] == "6" * 64

    computed = tee_service.handle_request(
        {
            "command": "compute_authoritative_weights_v2",
            "weight_request": {"fixture": True},
        }
    )
    assert computed["status"] == "ok"
    assert computed["weight_result"]["weights_hash"] == "3" * 64
    assert computed["weight_authorization_id"] == "sha256:" + "9" * 64

    for command, extra in (
        ("sign_weights", {"weights_hash": "8" * 64}),
        ("get_attestation", {"epoch_id": 7}),
        ("compute_weights_v2", {"snapshot": {}}),
    ):
        rejected = tee_service.handle_request({"command": command, **extra})
        assert rejected["status"] == "error"
        assert "permanently removed" in rejected["error"]


def test_validator_rpc_rejects_authoritative_compute_before_configuration(monkeypatch):
    monkeypatch.setattr(tee_service, "validator_runtime_v2", None)
    monkeypatch.setattr(tee_service, "validator_weight_authority_v2", None)
    monkeypatch.setattr(tee_service, "validator_hotkey_authority_v2", None)
    response = tee_service.handle_request(
        {
            "command": "compute_authoritative_weights_v2",
            "weight_request": {"fixture": True},
        }
    )
    assert response["status"] == "error"
    assert "not configured" in response["error"]


def test_validator_host_client_uses_authoritative_rpc_shapes(monkeypatch):
    client = ValidatorEnclaveClient(enclave_cid=16)
    requests = []

    def send(request, *, timeout_seconds=30):
        requests.append((dict(request), timeout_seconds))
        command = request["command"]
        if command == "configure_authoritative_v2":
            return {"status": "ok", "boot_identity": {"boot": True}}
        if command == "get_authoritative_v2_boot_identity":
            return {"status": "ok", "boot_identity": {"boot": True}}
        return {
            "status": "ok",
            "weight_snapshot": {"snapshot": True},
            "weight_result": {"weights": True},
            "weights_signature": "a" * 128,
            "receipt_graph": {"graph": True},
            "boot_identity": {"boot": True},
            "weight_authorization_id": "sha256:" + "c" * 64,
            "source_artifacts": [{"artifact_hash": "sha256:" + "d" * 64}],
        }

    monkeypatch.setattr(client, "_send_request", send)
    assert client.configure_authoritative_v2({"release": True}, "sha256:" + "b" * 64) == {
        "boot": True
    }
    assert client.get_authoritative_v2_boot_identity() == {"boot": True}
    response = client.compute_authoritative_weights_v2({"request": True})
    assert response["receipt_graph"] == {"graph": True}
    assert response["weight_authorization_id"] == "sha256:" + "c" * 64
    assert response["source_artifacts"] == [
        {"artifact_hash": "sha256:" + "d" * 64}
    ]
    assert [item[0]["command"] for item in requests] == [
        "configure_authoritative_v2",
        "get_authoritative_v2_boot_identity",
        "compute_authoritative_weights_v2",
    ]
    assert requests[-1][1] == 180


def test_validator_host_client_exposes_explicit_epoch_boundary_capture(monkeypatch):
    client = ValidatorEnclaveClient(enclave_cid=16)
    observed = {}

    def send(request, *, timeout_seconds=30):
        observed["request"] = dict(request)
        observed["timeout"] = timeout_seconds
        return {
            "status": "ok",
            "capture_result": {
                "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1"
            },
        }

    monkeypatch.setattr(client, "_send_request", send)
    result = client.capture_subnet_epoch_boundary_v2(
        cutover_manifest={"mapping_hash": "sha256:" + "1" * 64},
        settlement_epoch_id=100,
    )
    assert result["schema_version"] == "leadpoet.subnet_epoch_boundary_capture.v1"
    assert observed["request"] == {
        "command": "capture_subnet_epoch_boundary_v2",
        "capture_request": {
            "cutover_manifest": {"mapping_hash": "sha256:" + "1" * 64},
            "settlement_epoch_id": 100,
        },
    }
    assert observed["timeout"] == 180


def test_validator_rpc_exposes_only_structured_hotkey_operations(monkeypatch):
    calls = []

    class FakeHotkeyAuthority:
        def recipient_request(self):
            calls.append(("recipient", None))
            return {"recipient": True}

        def provision_seed(self, **kwargs):
            calls.append(("provision", kwargs))
            return {"provisioned": True}

        def public_state(self):
            calls.append(("state", None))
            return {"provisioned": True}

        def sign_application_message(self, **kwargs):
            calls.append(("application", kwargs))
            return {"purpose": "validator.gateway_binding.v2"}

        def prepare_weight_commit(self, **kwargs):
            calls.append(("prepare", kwargs))
            return {"commit_authorization_id": "sha256:" + "1" * 64}

        def sign_weight_extrinsic(self, **kwargs):
            calls.append(("extrinsic", kwargs))
            return {"authorization_hash": "sha256:" + "2" * 64}

        def recover_weight_publication(self, **kwargs):
            calls.append(("recover", kwargs))
            return {"weight_authorization_id": "sha256:" + "4" * 64}

    monkeypatch.setattr(
        tee_service,
        "validator_hotkey_authority_v2",
        FakeHotkeyAuthority(),
    )

    assert tee_service.handle_request(
        {"command": "get_hotkey_recipient_v2"}
    )["recipient_request"] == {"recipient": True}
    assert tee_service.handle_request(
        {
            "command": "provision_hotkey_v2",
            "ciphertext_for_recipient_b64": "ciphertext",
        }
    )["hotkey_state"] == {"provisioned": True}
    assert tee_service.handle_request(
        {"command": "get_hotkey_state_v2"}
    )["hotkey_state"] == {"provisioned": True}
    assert tee_service.handle_request(
        {
            "command": "sign_application_message_v2",
            "message_hex": "00",
            "parent_receipt_hash": "sha256:" + "3" * 64,
        }
    )["signature_result"]["purpose"] == "validator.gateway_binding.v2"
    assert tee_service.handle_request(
        {
            "command": "prepare_weight_commit_v2",
            "commit_request": {"weight_authorization_id": "authorization"},
        }
    )["commit_result"]["commit_authorization_id"].startswith("sha256:")
    assert tee_service.handle_request(
        {
            "command": "sign_weight_extrinsic_v2",
            "signature_request": {"commit_authorization_id": "commit"},
        }
    )["signature_result"]["authorization_hash"].startswith("sha256:")
    assert tee_service.handle_request(
        {
            "command": "recover_weight_publication_v2",
            "published_bundle": {"bundle": True},
            "weight_submission_event_hash": "sha256:" + "3" * 64,
            "extrinsic_signature_results": [],
        }
    )["recovery_result"]["weight_authorization_id"].startswith("sha256:")
    assert calls == [
        ("recipient", None),
        ("provision", {"ciphertext_for_recipient_b64": "ciphertext"}),
        ("state", None),
        (
            "application",
            {
                "message_hex": "00",
                "parent_receipt_hash": "sha256:" + "3" * 64,
            },
        ),
        ("prepare", {"weight_authorization_id": "authorization"}),
        ("extrinsic", {"commit_authorization_id": "commit"}),
        (
            "recover",
            {
                "published_bundle": {"bundle": True},
                "weight_submission_event_hash": "sha256:" + "3" * 64,
                "extrinsic_signature_results": [],
            },
        ),
    ]


def test_validator_host_client_uses_hotkey_rpc_shapes(monkeypatch):
    client = ValidatorEnclaveClient(enclave_cid=16)
    requests = []

    def send(request, *, timeout_seconds=30):
        requests.append((dict(request), timeout_seconds))
        command = request["command"]
        if command == "get_hotkey_recipient_v2":
            return {"status": "ok", "recipient_request": {"recipient": True}}
        if command in {
            "configure_hotkey_authority_v2",
            "provision_hotkey_v2",
            "get_hotkey_state_v2",
        }:
            return {"status": "ok", "hotkey_state": {"provisioned": True}}
        if command == "sign_application_message_v2":
            return {"status": "ok", "signature_result": {"application": True}}
        if command == "prepare_weight_commit_v2":
            return {"status": "ok", "commit_result": {"commit": True}}
        if command == "recover_weight_publication_v2":
            return {
                "status": "ok",
                "recovery_result": {"weight_authorization_id": "sha256:" + "3" * 64},
            }
        return {"status": "ok", "signature_result": {"extrinsic": True}}

    monkeypatch.setattr(client, "_send_request", send)
    assert client.configure_hotkey_authority_v2(
        {"hotkey": True}, "sha256:" + "1" * 64
    )["provisioned"] is True
    assert client.get_hotkey_recipient_v2() == {"recipient": True}
    assert client.provision_hotkey_v2("ciphertext")["provisioned"] is True
    assert client.get_hotkey_state_v2()["provisioned"] is True
    assert client.sign_application_message_v2(
        b"message", parent_receipt_hash="sha256:" + "2" * 64
    ) == {"application": True}
    assert client.prepare_weight_commit_v2({"weight": True}) == {"commit": True}
    assert client.recover_weight_publication_v2(
        published_bundle={"bundle": True},
        weight_submission_event_hash="sha256:" + "4" * 64,
        extrinsic_signature_results=[],
    )["weight_authorization_id"] == "sha256:" + "3" * 64
    assert client.sign_weight_extrinsic_v2({"payload": True}) == {
        "extrinsic": True
    }
    assert [item[0]["command"] for item in requests] == [
        "configure_hotkey_authority_v2",
        "get_hotkey_recipient_v2",
        "provision_hotkey_v2",
        "get_hotkey_state_v2",
        "sign_application_message_v2",
        "prepare_weight_commit_v2",
        "recover_weight_publication_v2",
        "sign_weight_extrinsic_v2",
    ]
