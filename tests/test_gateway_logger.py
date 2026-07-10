import pytest

from gateway.utils import logger as gateway_logger


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

