from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from Leadpoet.utils import cloud_db
from gateway.fulfillment import api as fulfillment_api


class _Query:
    def __init__(self, rows): self.rows = rows
    def select(self, _fields): return self
    def order(self, _field): return self
    def range(self, _start, _end): return self
    def execute(self): return SimpleNamespace(data=self.rows)


class _Supabase:
    def __init__(self, rows): self.rows = rows
    def table(self, name):
        assert name == "banned_hotkeys"
        return _Query(self.rows)


class _Response:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): return None
    def json(self): return self.payload


def test_banned_hotkey_snapshot_is_sorted_unique_and_complete(monkeypatch):
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: _Supabase([{"hotkey": "5B"}, {"hotkey": "5A"}]))
    assert fulfillment_api._collect_banned_hotkeys_sync() == {"banned_hotkeys": ["5A", "5B"], "banned_lookup_ok": True}


@pytest.mark.asyncio
async def test_banned_hotkey_endpoint_fails_closed_on_source_error(monkeypatch):
    async def run_inline(function, *_args): return function()
    monkeypatch.setattr(fulfillment_api, "run_db", run_inline)
    monkeypatch.setattr(fulfillment_api, "_collect_banned_hotkeys_sync", lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")))
    with pytest.raises(HTTPException) as error:
        await fulfillment_api.get_banned_hotkeys()
    assert error.value.status_code == 503


def test_validator_fetches_canonical_gateway_ban_snapshot(monkeypatch):
    monkeypatch.setattr(cloud_db.requests, "get", lambda *_args, **_kwargs: _Response({"banned_hotkeys": ["5A", "5B"], "banned_lookup_ok": True}))
    assert cloud_db.gateway_get_banned_hotkeys_snapshot(object()) == {"banned_hotkeys": ["5A", "5B"], "banned_lookup_ok": True}


def test_validator_rejects_noncanonical_gateway_ban_snapshot(monkeypatch):
    monkeypatch.setattr(cloud_db.requests, "get", lambda *_args, **_kwargs: _Response({"banned_hotkeys": ["5B", "5A"], "banned_lookup_ok": True}))
    monkeypatch.setattr(cloud_db.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        cloud_db.gateway_get_banned_hotkeys_snapshot(object())
