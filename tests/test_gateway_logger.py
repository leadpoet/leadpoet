import pytest

from gateway.utils import logger as gateway_logger
from gateway.utils import tee_client as tee_client_module


class _FakeExecute:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self):
        return {"ok": True}


class _FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def insert(self, row):
        self.rows.append(row)
        return _FakeExecute(self.rows)


class _FakeSupabase:
    def __init__(self):
        self.rows = []

    def table(self, name):
        assert name == "transparency_log"
        return _FakeTable(self.rows)


@pytest.mark.asyncio
async def test_legacy_log_event_supplies_non_null_signature(monkeypatch):
    fake = _FakeSupabase()

    async def fake_get_supabase_async():
        return fake

    monkeypatch.setattr(gateway_logger, "_get_supabase_async", fake_get_supabase_async)

    result = await gateway_logger.log_event(
        {
            "event_type": "ICP_SET_ACTIVATED",
            "actor_hotkey": "system",
            "nonce": "nonce-1",
            "ts": "2026-07-10T00:00:00+00:00",
            "payload": {"set_id": 20260710, "icp_count": 20},
        }
    )

    assert result["status"] == "buffered"
    assert fake.rows
    assert fake.rows[0]["signature"] == ""
    assert fake.rows[0]["payload_hash"]


@pytest.mark.asyncio
async def test_signed_log_event_is_created_and_buffered_by_coordinator(monkeypatch):
    fake = _FakeSupabase()
    payload = {"validator_hotkey": "validator", "epoch_id": 42}

    async def fake_get_supabase_async():
        return fake

    class _Coordinator:
        async def sign_transparency_event(self, *, event_type, payload, payload_hash):
            assert event_type == "WEIGHT_SUBMISSION_V2"
            assert payload == {
                "validator_hotkey": "validator",
                "epoch_id": 42,
            }
            assert payload_hash == gateway_logger.compute_payload_hash(payload)
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

    monkeypatch.setattr(gateway_logger, "_get_supabase_async", fake_get_supabase_async)
    monkeypatch.setattr(tee_client_module, "tee_client", _Coordinator())

    result = await gateway_logger.log_event("WEIGHT_SUBMISSION_V2", payload)

    assert result["event_hash"] == "b" * 64
    assert result["tee_buffered"] is True
    assert result["tee_sequence"] == 9
    assert fake.rows[0]["signature"] == "d" * 128
    assert fake.rows[0]["signed_log_entry"]["event_hash"] == "b" * 64
    assert fake.rows[0]["tee_sequence"] == 9
