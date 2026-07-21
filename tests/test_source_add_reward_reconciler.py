"""SOURCE_ADD rewards must stop paying once their full obligation is delivered.

The allocator only counts a reward down while allocation snapshots keep
ticking; when the epoch counter froze during the stateful-epoch switch,
finished rewards kept collecting every epoch. The reconciler appends a
``stopped_forward`` event for any reward whose snapshot-settled total
already covers ``alpha_percent x reward_epochs``, and leaves everything
still owed untouched.
"""

import pytest

from gateway.research_lab import maintenance


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


def _reward_row(ref, status="active", alpha=1.0, epochs=20, seq=3):
    return {
        "reward_ref": ref,
        "miner_hotkey": "hk-" + ref[-4:],
        "current_reward_status": status,
        "desired_alpha_percent": alpha,
        "epoch_count": epochs,
        "current_event_seq": seq,
    }


@pytest.mark.asyncio
async def test_fully_delivered_rewards_stop_and_owed_ones_survive(monkeypatch):
    rows = [
        _reward_row("source_add_reward:aaaa"),  # paid 26 of 20 -> stop
        _reward_row("source_add_reward:bbbb"),  # paid 12 of 20 -> keep
    ]
    monkeypatch.setattr(maintenance, "select_all", _async_value(rows))
    monkeypatch.setattr(
        maintenance, "_resolve_maintenance_epoch", _async_value(24070)
    )
    import gateway.research_lab.allocations as allocations

    monkeypatch.setattr(
        allocations,
        "_source_add_paid_alpha_to_date",
        _async_value(
            {"source_add_reward:aaaa": 26.0, "source_add_reward:bbbb": 12.0}
        ),
    )
    inserted = []

    async def fake_insert(table, row):
        inserted.append((table, dict(row)))
        return dict(row)

    import gateway.research_lab.store as store

    monkeypatch.setattr(store, "insert_row", fake_insert)

    plan = await maintenance.reconcile_source_add_reward_statuses(dry_run=True)
    assert plan["dry_run"] is True
    assert [p["reward_ref"] for p in plan["planned"]] == ["source_add_reward:aaaa"]
    assert plan["planned"][0]["next_seq"] == 4
    assert inserted == []  # dry run writes nothing

    result = await maintenance.reconcile_source_add_reward_statuses(dry_run=False)
    assert result["ok"] is True
    assert result["stopped_count"] == 1
    table, row = inserted[0]
    assert table == "research_lab_source_add_reward_events"
    assert row["reward_ref"] == "source_add_reward:aaaa"
    assert row["seq"] == 4
    assert row["reward_status"] == "stopped_forward"


@pytest.mark.asyncio
async def test_nothing_planned_when_all_rewards_still_owed(monkeypatch):
    rows = [_reward_row("source_add_reward:cccc")]
    monkeypatch.setattr(maintenance, "select_all", _async_value(rows))
    monkeypatch.setattr(
        maintenance, "_resolve_maintenance_epoch", _async_value(24070)
    )
    import gateway.research_lab.allocations as allocations

    monkeypatch.setattr(
        allocations,
        "_source_add_paid_alpha_to_date",
        _async_value({"source_add_reward:cccc": 19.99}),
    )
    plan = await maintenance.reconcile_source_add_reward_statuses(dry_run=True)
    assert plan["planned"] == []
