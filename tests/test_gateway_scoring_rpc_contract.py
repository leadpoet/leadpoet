import base64
import importlib
import json
from pathlib import Path


def _tee_service(monkeypatch):
    tee_dir = Path(__file__).resolve().parents[1] / "gateway" / "tee"
    monkeypatch.syspath_prepend(str(tee_dir))
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_scoring_a")
    return importlib.import_module("gateway.tee.tee_service")


def test_v1_scoring_rpc_is_not_authorized_for_v2_role(monkeypatch):
    service = _tee_service(monkeypatch)
    for method in (
        "scoring_configure_runtime",
        "scoring_health",
        "scoring_submit_job",
        "scoring_get_status",
        "scoring_get_result",
    ):
        response = service.handle_rpc(method, {})
        assert response == {
            "error": "RPC method is not authorized for enclave role gateway_scoring_a"
        }
    assert service.handle_rpc("unknown", {}) == {
        "error": "RPC method is not authorized for enclave role gateway_scoring_a"
    }


def test_v1_scoring_runtime_service_is_absent(monkeypatch):
    service = _tee_service(monkeypatch)
    assert not hasattr(service, "get_scoring_job_manager")
    assert not hasattr(service, "configure_scoring_runtime")
    assert not hasattr(service, "handle_scoring_rpc")
    assert not hasattr(service, "scoring_runtime_configuration")


def test_coordinator_signs_and_hash_chains_transparency_events_in_enclave(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    tee_dir = Path(__file__).resolve().parents[1] / "gateway" / "tee"
    monkeypatch.syspath_prepend(str(tee_dir))
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    service = importlib.import_module("gateway.tee.tee_service")
    from gateway.tee import enclave_signer

    enclave_signer._reset_for_testing()
    service.event_signer_initialization = None
    service.event_buffer.clear()
    service.sequence_counter = 0
    monkeypatch.setattr(service, "compute_code_hash", lambda: "1" * 64)

    def fake_attestation(code_hash):
        assert code_hash == "1" * 64
        enclave_signer._ATTESTATION_DOCUMENT = b"nitro-document"
        return enclave_signer._ATTESTATION_DOCUMENT

    monkeypatch.setattr(
        enclave_signer,
        "generate_attestation_document",
        fake_attestation,
    )

    initialized = service.handle_rpc(
        "initialize_event_signer",
        {"prev_log_tip_hash": "a" * 64},
    )["result"]
    restart = initialized["restart_log_entry"]
    assert restart["signed_event"]["prev_event_hash"] == "a" * 64
    assert initialized["identity"]["attestation_document_b64"] == base64.b64encode(
        b"nitro-document"
    ).decode("ascii")

    payload = {"epoch_id": 42, "validator_hotkey": "validator"}
    payload_hash = __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    signed = service.handle_rpc(
        "sign_transparency_event",
        {
            "event_type": "WEIGHT_SUBMISSION_V2",
            "payload": payload,
            "payload_hash": payload_hash,
        },
    )["result"]
    entry = signed["log_entry"]
    assert entry["signed_event"]["prev_event_hash"] == restart["event_hash"]
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(entry["enclave_pubkey"])).verify(
        bytes.fromhex(entry["enclave_signature"]),
        bytes.fromhex(entry["event_hash"]),
    )
    assert len(service.event_buffer) == 2

    enclave_signer._reset_for_testing()
    service.event_signer_initialization = None
    service.event_buffer.clear()
