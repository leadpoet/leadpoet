"""Late publication must not replace a newer completed day's authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena.service import ArenaService, ServiceError


def _row(day: str, *, scope: str = "finney", promoted: bool = False) -> dict:
    return {
        "round_id": f"arena-{day}",
        "evaluation_date": day,
        "status": "published",
        "arena_network_name": scope,
        "arena_netuid": 71,
        "configuration_doc": {"mode": "live", "rewards_enabled": True},
        "publication_doc": {"king_decision": {"outcome": "crowned"}},
        "promotion_required": True,
        "baseline_promoted_at": "done" if promoted else None,
        "reward_activated_at": None,
    }


class _Store:
    def __init__(self, rows: list[dict], pending: list[dict] | None = None):
        self.rows = rows
        self.pending = pending or []
        self.calls = []

    def list_rounds(self, **kwargs):
        self.calls.append(kwargs)
        scoped = [
            row for row in self.rows
            if row["arena_network_name"] == kwargs["network_name"]
            and row["arena_netuid"] == kwargs["netuid"]
            and row["status"] == kwargs["status"]
        ]
        scoped.sort(key=lambda row: row["evaluation_date"], reverse=True)
        start = kwargs.get("offset") or 0
        return scoped[start:start + kwargs["limit"]]

    def latest_published_day(self, **kwargs):
        self.calls.append(kwargs)
        scoped = [
            row for row in self.rows
            if row["arena_network_name"] == kwargs["network_name"]
            and row["arena_netuid"] == kwargs["netuid"]
            and row["status"] == "published"
        ]
        return max(scoped, key=lambda row: row["evaluation_date"], default=None)

    def pending_promotions(self, **kwargs):
        start = kwargs.get("offset") or 0
        return self.pending[start:start + kwargs["limit"]]

    def get_round(self, round_id):
        return next((row for row in self.rows if row["round_id"] == round_id), None)


def _service(rows: list[dict], pending: list[dict] | None = None) -> ArenaService:
    service = object.__new__(ArenaService)
    service._store = _Store(rows, pending)
    service._config = SimpleNamespace(
        mode="live", pinned_round_id=None, baseline_promoter_factory=None,
    )
    service._chain_scope = lambda: ("finney", 71)
    service._round = lambda round_id: next(
        row for row in rows if row["round_id"] == round_id
    )
    return service


def test_newer_published_day_skips_old_promotion_and_reward():
    old, newer = _row("2026-10-01"), _row("2026-10-02", promoted=True)
    service = _service([old, newer])

    assert service.promote_baseline(old["round_id"]) == {"status": "superseded"}
    assert service.activate_reward(old["round_id"]) == {"status": "superseded"}
    assert not service._superseded_by_published_day(newer)
    assert all(call["network_name"] == "finney" and call["netuid"] == 71
               for call in service._store.calls)


def test_same_day_and_other_chain_do_not_supersede():
    old = _row("2026-10-01")
    same_day = {**_row("2026-10-01"), "round_id": "arena-2026-10-01-retry"}
    other_chain = _row("2026-10-03", scope="testnet")
    service = _service([old, same_day, other_chain])

    assert not service._superseded_by_published_day(old)
    assert not service._superseded_by_published_day(_row("2026-10-04"))


def test_old_pending_rows_do_not_hide_new_day_after_first_page():
    old = _row("2026-10-01")
    newer = _row("2026-10-02")
    pending = [{"round_id": old["round_id"]} for _ in range(100)]
    pending.append({"round_id": newer["round_id"]})
    service = _service([old, newer], pending)
    calls = []
    service.promote_baseline = lambda round_id: (
        calls.append(round_id) or {"status": "promoted" if round_id == newer["round_id"] else "superseded"}
    )

    assert service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    assert calls[-1] == newer["round_id"]


def test_superseded_promotion_does_not_block_new_baseline_or_reward():
    old, newer = _row("2026-10-01"), _row("2026-10-02", promoted=True)
    service = _service([old, newer], [{"round_id": old["round_id"]}])
    assert not service._pending_promotion_blocks()

    calls = []
    service.activate_reward = lambda round_id: (
        calls.append(round_id) or {
            "status": "activated" if round_id == newer["round_id"] else "superseded"
        }
    )
    assert service.activate_pending_rewards() == {"status": "ok", "activated": 1}
    assert calls == [old["round_id"], newer["round_id"]]


def test_pinned_live_service_can_ignore_other_superseded_round():
    old, newer = _row("2026-10-01"), _row("2026-10-02")
    service = _service([old, newer], [{"round_id": old["round_id"]}])
    service._config.pinned_round_id = newer["round_id"]
    service._round = lambda round_id: (
        newer if round_id == newer["round_id"] else (_ for _ in ()).throw(
            ServiceError("round_scope_mismatch", 409)
        )
    )
    assert not service._pending_promotion_blocks()


@pytest.mark.parametrize("value", [None, "2026-13-01", "2026-W40-4"])
def test_invalid_frozen_evaluation_date_fails_closed(value):
    with pytest.raises(ServiceError):
        ArenaService._evaluation_day({"evaluation_date": value})
