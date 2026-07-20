from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from Leadpoet.utils import cloud_db
from gateway.fulfillment import api as fulfillment_api


class _Query:
    def __init__(self, pages):
        self._pages = pages
        self._range = (0, 999)

    def select(self, _fields):
        return self

    def order(self, _field):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        page = self._range[0] // 1000
        return SimpleNamespace(data=self._pages[page] if page < len(self._pages) else [])


class _Supabase:
    def __init__(self, pages):
        self._pages = pages

    def table(self, name):
        assert name == "banned_hotkeys"
        return _Query(self._pages)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_banned_hotkey_snapshot_is_sorted_unique_and_complete(monkeypatch):
    monkeypatch.setattr(
        fulfillment_api,
        "_get_supabase",
        lambda: _Supabase([[{"hotkey": "5B"}, {"hotkey": "5A"}]]),
    )

    assert fulfillment_api._collect_banned_hotkeys_sync() == {
        "banned_hotkeys": ["5A", "5B"],
        "banned_lookup_ok": True,
    }


@pytest.mark.asyncio
async def test_banned_hotkey_endpoint_fails_closed_on_source_error(monkeypatch):
    async def run_inline(function, *_args):
        return function()

    monkeypatch.setattr(fulfillment_api, "run_db", run_inline)
    monkeypatch.setattr(
        fulfillment_api,
        "_collect_banned_hotkeys_sync",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(HTTPException) as error:
        await fulfillment_api.get_banned_hotkeys()

    assert error.value.status_code == 503
    assert error.value.detail == "Authoritative banned hotkey snapshot is unavailable"


def test_validator_fetches_canonical_gateway_ban_snapshot(monkeypatch):
    monkeypatch.setattr(
        cloud_db.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "banned_hotkeys": ["5A", "5B"],
                "banned_lookup_ok": True,
            }
        ),
    )

    assert cloud_db.gateway_get_banned_hotkeys_snapshot(object()) == {
        "banned_hotkeys": ["5A", "5B"],
        "banned_lookup_ok": True,
    }


def test_validator_rejects_noncanonical_gateway_ban_snapshot(monkeypatch):
    monkeypatch.setattr(
        cloud_db.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "banned_hotkeys": ["5B", "5A"],
                "banned_lookup_ok": True,
            }
        ),
    )
    monkeypatch.setattr(cloud_db.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        cloud_db.gateway_get_banned_hotkeys_snapshot(object())
