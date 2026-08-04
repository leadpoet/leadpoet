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


class _RewardsQuery:
    def __init__(self, pages):
        self._pages = pages
        self._range = (0, 999)
        self.selected = None
        self.ordered = None

    def select(self, fields):
        self.selected = fields
        return self

    @property
    def not_(self):
        return self

    def is_(self, _field, _value):
        return self

    def gt(self, _field, _value):
        return self

    def order(self, field):
        self.ordered = field
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        page = self._range[0] // 1000
        data = self._pages[page] if page < len(self._pages) else []
        return SimpleNamespace(data=data)


class _RewardsSupabase:
    def __init__(self, pages):
        self.query = _RewardsQuery(pages)

    def table(self, name):
        assert name == "fulfillment_score_consensus"
        return self.query


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


def test_active_rewards_use_consensus_order_and_return_sorted_hotkeys(monkeypatch):
    supabase = _RewardsSupabase(
        [[
            {"consensus_id": "1", "miner_hotkey": "5B", "reward_pct": 0.2},
            {"consensus_id": "2", "miner_hotkey": "5A", "reward_pct": 0.1},
            {"consensus_id": "3", "miner_hotkey": "5B", "reward_pct": 0.05},
        ]]
    )
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: supabase)

    result = fulfillment_api._collect_active_rewards_sync(100)

    assert result == {
        "rewards": {"5A": 0.1, "5B": 0.25},
        "total_active_rows": 3,
    }
    assert supabase.query.selected == (
        "consensus_id, miner_hotkey, reward_pct, reward_expires_epoch"
    )
    assert supabase.query.ordered == "consensus_id"


def test_active_rewards_fail_closed_at_pagination_limit(monkeypatch):
    full_page = [
        {"consensus_id": str(index), "miner_hotkey": "5A", "reward_pct": 0.0}
        for index in range(1000)
    ]
    supabase = _RewardsSupabase([full_page] * 50)
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: supabase)

    with pytest.raises(RuntimeError, match="pagination limit"):
        fulfillment_api._collect_active_rewards_sync(100)


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


def test_champion_ban_check_uses_canonical_gateway_snapshot(monkeypatch):
    from neurons.validator import Validator

    wallet = object()
    validator = object.__new__(Validator)
    validator.wallet = wallet
    calls = []

    def snapshot(requested_wallet):
        calls.append(requested_wallet)
        return {
            "banned_hotkeys": ["5A", "5B"],
            "banned_lookup_ok": True,
        }

    monkeypatch.setattr(cloud_db, "gateway_get_banned_hotkeys_snapshot", snapshot)

    assert validator._is_champion_hotkey_banned("5B") is True
    assert validator._is_champion_hotkey_banned("5C") is False
    assert calls == [wallet, wallet]


def test_champion_ban_check_fails_closed_when_snapshot_is_unavailable(
    monkeypatch,
):
    from neurons.validator import Validator

    validator = object.__new__(Validator)
    validator.wallet = object()
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_banned_hotkeys_snapshot",
        lambda _wallet: (_ for _ in ()).throw(RuntimeError("gateway unavailable")),
    )

    with pytest.raises(
        RuntimeError,
        match="canonical champion ban snapshot is unavailable",
    ):
        validator._is_champion_hotkey_banned("5A")
