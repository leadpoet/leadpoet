import base64
import importlib
from pathlib import Path

import pytest

from gateway.db import client as database_client
from gateway.utils import logger as gateway_logger
from gateway.utils import tee_client as tee_client_module


class _Coordinator:
    def __init__(self) -> None:
        self.initial_tip = "unset"
        self.events = []

    async def initialize_event_signer(self, previous_tip):
        self.initial_tip = previous_tip
        return {
            "identity": {
                "purpose": "gateway_event_signing",
                "enclave_pubkey": "c" * 64,
                "attestation_document_b64": "attestation",
            },
            "restart_log_entry": {"event_hash": "a" * 64},
            "buffer": {"sequence": 1, "buffer_size": 1},
        }

    async def sign_transparency_event(self, *, event_type, payload, payload_hash):
        self.events.append(
            {
                "event_type": event_type,
                "payload": payload,
                "payload_hash": payload_hash,
            }
        )
        return {
            "log_entry": {
                "signed_event": {
                    "event_type": event_type,
                    "timestamp": "2026-07-12T00:00:00Z",
                    "boot_id": "00000000-0000-0000-0000-000000000001",
                    "monotonic_seq": 2,
                    "prev_event_hash": "a" * 64,
                    "payload": payload,
                },
                "event_hash": "b" * 64,
                "enclave_pubkey": "c" * 64,
                "enclave_signature": "d" * 128,
            },
            "buffer": {"sequence": 9, "buffer_size": 10},
        }


def _deny_database_access(monkeypatch) -> None:
    def unexpected_database_access(*args, **kwargs):
        raise AssertionError("retired relational audit database was accessed")

    monkeypatch.setattr(
        database_client,
        "get_read_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_write_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_async_read_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_async_write_client",
        unexpected_database_access,
    )


@pytest.mark.asyncio
async def test_signer_boot_starts_without_a_relational_log_tip(monkeypatch):
    coordinator = _Coordinator()
    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", coordinator)

    identity = await gateway_logger.initialize_enclave_event_signing()

    assert identity["purpose"] == "gateway_event_signing"
    assert coordinator.initial_tip is None


@pytest.mark.asyncio
async def test_dictionary_event_is_signed_and_buffered_by_coordinator(monkeypatch):
    coordinator = _Coordinator()
    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", coordinator)

    result = await gateway_logger.log_event(
        {
            "event_type": "ICP_SET_ACTIVATED",
            "actor_hotkey": "system",
            "payload": {"set_id": 20260710, "icp_count": 20},
        }
    )

    assert coordinator.events == [
        {
            "event_type": "ICP_SET_ACTIVATED",
            "payload": {
                "actor_hotkey": "system",
                "set_id": 20260710,
                "icp_count": 20,
            },
            "payload_hash": gateway_logger.compute_payload_hash(
                {
                    "actor_hotkey": "system",
                    "set_id": 20260710,
                    "icp_count": 20,
                }
            ),
        }
    ]
    assert result["event_hash"] == "b" * 64
    assert result["tee_buffered"] is True
    assert result["tee_sequence"] == 9
    assert result["sequence"] == 9
    assert result["buffer_size"] == 10


@pytest.mark.asyncio
async def test_typed_event_is_signed_and_buffered_by_coordinator(monkeypatch):
    coordinator = _Coordinator()
    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", coordinator)
    payload = {"validator_hotkey": "validator", "epoch_id": 42}

    result = await gateway_logger.log_event("WEIGHT_SUBMISSION_V2", payload)

    assert coordinator.events[0]["event_type"] == "WEIGHT_SUBMISSION_V2"
    assert coordinator.events[0]["payload"] == payload
    assert coordinator.events[0]["payload_hash"] == gateway_logger.compute_payload_hash(payload)
    assert result["tee_buffer_size"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_result",
    [
        {
            "log_entry": {"signed_event": {"monotonic_seq": 2}},
            "buffer": {"sequence": 9, "buffer_size": 10},
        },
        {
            "log_entry": {"signed_event": {}, "event_hash": "b" * 64},
            "buffer": {"sequence": 9, "buffer_size": 10},
        },
        {
            "log_entry": {
                "signed_event": {"monotonic_seq": 2},
                "event_hash": "b" * 64,
            },
            "buffer": {"buffer_size": 10},
        },
        {
            "log_entry": {
                "signed_event": {"monotonic_seq": 2},
                "event_hash": "b" * 64,
            },
            "buffer": {"sequence": 9},
        },
    ],
)
async def test_malformed_coordinator_result_fails_closed(
    monkeypatch,
    tmp_path,
    malformed_result,
):
    class _MalformedCoordinator:
        async def sign_transparency_event(self, **kwargs):
            return malformed_result

    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", _MalformedCoordinator())
    monkeypatch.setattr(gateway_logger, "FALLBACK_LOG_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="Failed to sign and buffer event"):
        await gateway_logger.log_event("TEST_EVENT", {"value": 1})


@pytest.mark.asyncio
async def test_hourly_checkpoint_uses_real_enclave_signer_and_direct_payload(monkeypatch):
    from gateway.tasks.hourly_batch import build_arweave_checkpoint_log_event
    from gateway.tee import enclave_signer

    tee_dir = Path(__file__).resolve().parents[1] / "gateway" / "tee"
    monkeypatch.syspath_prepend(str(tee_dir))
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    service = importlib.import_module("gateway.tee.tee_service")
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

    class _InProcessCoordinator:
        async def initialize_event_signer(self, previous_tip):
            return service.initialize_event_signer(previous_tip)

        async def sign_transparency_event(self, **kwargs):
            return service.sign_transparency_event(**kwargs)

    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", _InProcessCoordinator())

    try:
        identity = await gateway_logger.initialize_enclave_event_signing()
        restart_entry = service.event_buffer[0]["signed_log_entry"]
        checkpoint_event = build_arweave_checkpoint_log_event(
            tx_id="arweave-tx-123",
            header={
                "checkpoint_number": 42,
                "event_count": 7,
                "merkle_root": "a" * 64,
                "time_range": {
                    "start": "2026-07-03T12:00:00Z",
                    "end": "2026-07-03T15:00:00Z",
                },
            },
            compressed_size_bytes=2048,
        )
        result = await gateway_logger.log_event(checkpoint_event)

        assert identity["purpose"] == "gateway_event_signing"
        assert identity["attestation_document_b64"] == base64.b64encode(
            b"nitro-document"
        ).decode("ascii")
        assert restart_entry["signed_event"]["boot_id"] == identity["signer_state"][
            "boot_id"
        ]
        assert restart_entry["signed_event"]["payload"][
            "prev_log_tip_event_hash"
        ] is None
        assert result["signed_event"]["prev_event_hash"] == restart_entry["event_hash"]
        assert result["signed_event"]["payload"]["arweave_tx_id"] == "arweave-tx-123"
        assert "payload" not in result["signed_event"]["payload"]
        assert result["signed_event"]["payload"]["actor_hotkey"] == "system"
        assert result["sequence"] == 1
        assert result["buffer_size"] == 2
        assert len(service.event_buffer) == 2
    finally:
        enclave_signer._reset_for_testing()
        service.event_signer_initialization = None
        service.event_buffer.clear()
        service.sequence_counter = 0
