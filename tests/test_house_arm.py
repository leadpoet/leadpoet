"""Tests for the house-funded baseline arm scheduler (§9.3 / fablefollowup 4.3).

Covers, with fake store rows (no live Supabase):
  * policy clamp — the daily budget is clamped hard at the policy max ($500)
    regardless of arguments, and today's recorded house spend is subtracted;
  * dedupe — an open (fresh) house loop consumes the one-loop-per-hotkey slot
    so no new loop is opened;
  * dry-run — plans and writes nothing, even with the master flag enabled;
  * master flag — dry_run=False without RESEARCH_LAB_HOUSE_ARM_ENABLED never
    writes (both must be flipped to spend);
  * tag shape — ticket_doc.arm/house_policy_ref/house_lane, funded/queued
    events, capacity doc on the queued event, synthetic house payment ref;
  * idempotency — re-running the same pass opens nothing new; a paid-but-
    unqueued orphan is repaired without a second payment;
  * round-robin — the next lane follows the most recent house ticket's lane;
  * comparison math — matched-budget yield-per-dollar via
    counterfactual_gate.build_matched_budget_comparison, and the
    insufficient-data path;
  * raw-key rejection — a raw-looking key ref is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

import pytest

import gateway.research_lab.house_arm as house_arm
from gateway.research_lab.house_arm import (
    DEFAULT_PLANNER_LANES,
    HOUSE_ARM_TAG,
    HOUSE_POLICY_REF,
    build_house_arm_comparison_report,
    house_arm_status,
    next_house_lane,
    open_house_loops,
)


HOUSE_HOTKEY = "5HouseHotkey000000000000000000000000000000000000"
MINER_HOTKEY = "5MinerHotkey000000000000000000000000000000000000"
HOUSE_KEY_REF = "encrypted_ref:openrouter:house-key-1"
NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def _matches(row: Mapping[str, Any], filters: Any) -> bool:
    for spec in tuple(filters or ()):
        if len(spec) == 2:
            fname, op, value = spec[0], "eq", spec[1]
        else:
            fname, op, value = spec
        raw = row.get(fname)
        if op == "eq":
            if str(raw) != str(value):
                return False
        elif op == "gte":
            if not (str(raw) >= str(value)):
                return False
        elif op == "lt":
            if not (str(raw) < str(value)):
                return False
        else:  # pragma: no cover - unexpected operator in tests
            raise AssertionError(f"unsupported filter op {op}")
    return True


@dataclass
class FakeHouseStore:
    """Table-name-dispatched fakes for the store functions house_arm uses."""

    tickets: list[dict[str, Any]] = field(default_factory=list)
    ticket_events: list[dict[str, Any]] = field(default_factory=list)
    queue_events: list[dict[str, Any]] = field(default_factory=list)
    queue_current: list[dict[str, Any]] = field(default_factory=list)
    payments: list[dict[str, Any]] = field(default_factory=list)
    key_refs: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    score_bundles: list[dict[str, Any]] = field(default_factory=list)
    public_projections: list[str] = field(default_factory=list)

    def _table_rows(self, table: str) -> list[dict[str, Any]]:
        return {
            "research_loop_tickets": self.tickets,
            "research_loop_ticket_current": self.tickets,
            "research_loop_ticket_events": self.ticket_events,
            "research_loop_run_queue_current": self.queue_current,
            "research_loop_start_payments": self.payments,
            "research_lab_openrouter_key_refs": self.key_refs,
            "research_lab_candidate_evaluation_current": self.candidates,
            "research_evaluation_score_bundle_current": self.score_bundles,
        }[table]

    async def select_one(self, table: str, **kwargs: Any) -> dict[str, Any] | None:
        for row in self._table_rows(table):
            if _matches(row, kwargs.get("filters")):
                return dict(row)
        return None

    async def select_many(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self._table_rows(table) if _matches(row, kwargs.get("filters"))]
        return rows[: int(kwargs.get("limit") or 100)]

    async def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self._table_rows(table) if _matches(row, kwargs.get("filters"))]
        return rows[: int(kwargs.get("max_rows") or 10000)]

    async def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        self._table_rows(table).append(dict(row))
        return dict(row)

    async def create_ticket_event(self, **kwargs: Any) -> dict[str, Any]:
        seq = len([e for e in self.ticket_events if e.get("ticket_id") == kwargs.get("ticket_id")])
        event = {"event_id": str(uuid4()), "seq": seq, **kwargs}
        self.ticket_events.append(event)
        return dict(event)

    async def create_queue_event(self, **kwargs: Any) -> dict[str, Any]:
        event = {"event_id": str(uuid4()), **kwargs}
        self.queue_events.append(event)
        self.queue_current = [row for row in self.queue_current if row.get("run_id") != kwargs.get("run_id")]
        self.queue_current.append(
            {
                "run_id": kwargs.get("run_id"),
                "ticket_id": kwargs.get("ticket_id"),
                "current_queue_status": kwargs.get("event_type"),
                "current_status_at": NOW.isoformat(),
            }
        )
        return dict(event)

    async def create_loop_start_payment(self, **kwargs: Any) -> dict[str, Any]:
        verification_doc = {
            "call_function": kwargs.get("payment_info", {}).get("call_function"),
            "payment_kind": kwargs.get("payment_kind"),
            "run_id": kwargs.get("run_id"),
            "compute_budget_usd": kwargs.get("compute_budget_usd"),
            **(kwargs.get("extra_verification_doc") or {}),
        }
        row = {
            "payment_id": str(uuid4()),
            "ticket_id": kwargs.get("ticket_id"),
            "payment_ref": kwargs.get("payment_ref"),
            "block_hash": kwargs.get("block_hash"),
            "extrinsic_index": kwargs.get("extrinsic_index"),
            "miner_hotkey": kwargs.get("miner_hotkey"),
            "required_usd": kwargs.get("required_usd"),
            "payment_status": "verified",
            "verification_doc": verification_doc,
            "created_at": NOW.isoformat(),
        }
        self.payments.append(row)
        return dict(row)

    async def payment_ref_exists(self, block_hash: str, extrinsic_index: int) -> bool:
        return any(
            row.get("block_hash") == block_hash and int(row.get("extrinsic_index") or 0) == int(extrinsic_index)
            for row in self.payments
        )

    async def safe_project_public_loop_activity(self, ticket_id: str, **kwargs: Any) -> None:
        self.public_projections.append(str(ticket_id))


def _fake_config() -> Any:
    return SimpleNamespace(
        allowed_research_islands=("generalist",),
        loop_start_fee_usd=0.2,
        default_compute_budget_usd=5.0,
        default_auto_research_model_tier="default",
        resolve_auto_research_model=lambda tier: ("default", "model-id", {}),
        active_loop_stale_after_seconds=7200,
        miner_openrouter_key_ref_env_map_json="",
        miner_openrouter_key_env_var="OPENROUTER_API_KEY",
        improvement_threshold_points=1.0,
        max_active_autoresearch_loops_per_hotkey=1,
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeHouseStore:
    fake = FakeHouseStore()
    fake.key_refs.append(
        {"key_ref": HOUSE_KEY_REF, "miner_hotkey": HOUSE_HOTKEY, "preflight_status": "passed"}
    )
    monkeypatch.setattr(house_arm, "select_one", fake.select_one)
    monkeypatch.setattr(house_arm, "select_many", fake.select_many)
    monkeypatch.setattr(house_arm, "select_all", fake.select_all)
    monkeypatch.setattr(house_arm, "insert_row", fake.insert_row)
    monkeypatch.setattr(house_arm, "create_ticket_event", fake.create_ticket_event)
    monkeypatch.setattr(house_arm, "create_queue_event", fake.create_queue_event)
    monkeypatch.setattr(house_arm, "create_loop_start_payment", fake.create_loop_start_payment)
    monkeypatch.setattr(house_arm, "payment_ref_exists", fake.payment_ref_exists)
    monkeypatch.setattr(house_arm, "safe_project_public_loop_activity", fake.safe_project_public_loop_activity)

    config = _fake_config()
    monkeypatch.setattr(
        house_arm,
        "ResearchLabGatewayConfig",
        SimpleNamespace(from_env=lambda: config),
    )
    monkeypatch.setattr(house_arm, "_bittensor_network_netuid", lambda: ("test", 71))

    async def _not_paused() -> dict[str, Any]:
        return {"paused": False}

    monkeypatch.setattr(house_arm, "get_autoresearch_maintenance_state", _not_paused)
    monkeypatch.setattr(
        house_arm,
        "autoresearch_queue_capacity_doc",
        lambda cfg: {
            "autoresearch_capacity_policy": "proxy_worker_capacity:v1",
            "autoresearch_capacity": 5,
            "active_loop_stale_after_seconds": 7200,
        },
    )

    monkeypatch.setenv(house_arm.HOUSE_HOTKEY_ENV, HOUSE_HOTKEY)
    monkeypatch.setenv(house_arm.HOUSE_OPENROUTER_KEY_REF_ENV, HOUSE_KEY_REF)
    monkeypatch.delenv(house_arm.HOUSE_ARM_ENABLED_ENV, raising=False)
    return fake


def _write_counts(store: FakeHouseStore) -> tuple[int, int, int, int]:
    return (
        len(store.tickets),
        len(store.ticket_events),
        len(store.queue_events),
        len(store.payments),
    )


def _house_payment_row(*, spend_compute: float = 5.0, fee: float = 0.2, run_id: str | None = None) -> dict[str, Any]:
    ref = f"house_arm:{run_id or uuid4()}"
    return {
        "payment_id": str(uuid4()),
        "payment_ref": f"{ref}:0",
        "block_hash": ref,
        "extrinsic_index": 0,
        "miner_hotkey": HOUSE_HOTKEY,
        "required_usd": fee,
        "payment_status": "verified",
        "verification_doc": {"compute_budget_usd": spend_compute, "run_id": run_id or str(uuid4())},
        "created_at": NOW.isoformat(),
    }


# ---------------------------------------------------------------------------
# Clamp enforcement
# ---------------------------------------------------------------------------


async def test_budget_clamped_hard_at_policy_max_regardless_of_args(store):
    result = await open_house_loops(budget_usd_today=10_000.0, max_open_loops=3, dry_run=True, now=NOW)
    assert result["ok"] is True
    assert result["budget"]["policy_daily_max_usd"] == 500.0
    assert result["budget"]["effective_budget_usd"] == 500.0
    assert result["budget"]["requested_usd"] == 10_000.0


async def test_todays_house_spend_reduces_remaining_budget(store):
    for _ in range(3):
        store.payments.append(_house_payment_row())
    result = await open_house_loops(budget_usd_today=500.0, max_open_loops=3, dry_run=True, now=NOW)
    assert result["budget"]["todays_spend_usd"] == pytest.approx(15.6)
    assert result["budget"]["remaining_usd"] == pytest.approx(484.4)
    assert result["budget"]["per_loop_usd"] == pytest.approx(5.2)


async def test_budget_below_per_loop_cost_opens_nothing(store):
    store.payments.append(_house_payment_row(spend_compute=7.0, fee=0.5))  # 7.5 spent
    result = await open_house_loops(budget_usd_today=10.0, max_open_loops=3, dry_run=True, now=NOW)
    # remaining 2.5 < per-loop 5.2
    assert result["to_open"] == 0
    assert result["planned"] == []
    assert any("per-loop" in str(item) for item in result["skipped"])


async def test_miner_spend_does_not_count_against_house_budget(store):
    miner_payment = _house_payment_row()
    miner_payment["miner_hotkey"] = MINER_HOTKEY
    store.payments.append(miner_payment)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=True, now=NOW)
    assert result["budget"]["todays_spend_usd"] == 0.0


# ---------------------------------------------------------------------------
# Dedupe against open loops (one active loop per hotkey)
# ---------------------------------------------------------------------------


def _open_house_ticket_and_queue_row(store: FakeHouseStore, *, lane: str = "query_construction") -> str:
    ticket_id = str(uuid4())
    store.tickets.append(
        {
            "ticket_id": ticket_id,
            "miner_hotkey": HOUSE_HOTKEY,
            "island": "generalist",
            "created_at": NOW.isoformat(),
            "ticket_doc": {"arm": HOUSE_ARM_TAG, "house_policy_ref": HOUSE_POLICY_REF, "house_lane": lane},
        }
    )
    run_id = house_arm._house_run_id(ticket_id)
    store.queue_current.append(
        {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "current_queue_status": "started",
            "current_status_at": NOW.isoformat(),
        }
    )
    store.payments.append(_house_payment_row(run_id=run_id))
    return ticket_id


async def test_open_house_loop_dedupes_against_running_loop(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    _open_house_ticket_and_queue_row(store)
    before = _write_counts(store)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=3, dry_run=False, now=NOW)
    assert result["ok"] is True
    assert result["open_house_loops"] == 1
    assert result["to_open"] == 0
    assert result["opened"] == []
    assert any("active loop" in str(item) for item in result["skipped"])
    assert _write_counts(store) == before


async def test_non_house_loops_do_not_count_as_open_house_loops(store):
    ticket_id = str(uuid4())
    store.tickets.append(
        {
            "ticket_id": ticket_id,
            "miner_hotkey": MINER_HOTKEY,
            "created_at": NOW.isoformat(),
            "ticket_doc": {},
        }
    )
    store.queue_current.append(
        {
            "run_id": str(uuid4()),
            "ticket_id": ticket_id,
            "current_queue_status": "started",
            "current_status_at": NOW.isoformat(),
        }
    )
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=True, now=NOW)
    assert result["open_house_loops"] == 0
    assert result["to_open"] == 1


# ---------------------------------------------------------------------------
# Dry-run writes nothing; master flag gates spending
# ---------------------------------------------------------------------------


async def test_dry_run_plans_but_writes_nothing(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    before = _write_counts(store)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=True, now=NOW)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["to_open"] == 1
    assert len(result["planned"]) == 1
    assert result["planned"][0]["action"] == "open_house_loop"
    assert result["planned"][0]["lane"] == DEFAULT_PLANNER_LANES[0]
    assert result["opened"] == []
    assert _write_counts(store) == before
    assert store.public_projections == []


async def test_no_dry_run_without_master_flag_refuses_and_writes_nothing(store):
    before = _write_counts(store)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is False
    assert house_arm.HOUSE_ARM_ENABLED_ENV in str(result["error"])
    assert _write_counts(store) == before


async def test_dry_run_allowed_without_master_flag(store):
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=True, now=NOW)
    assert result["ok"] is True
    assert result["house_arm_enabled"] is False
    assert len(result["planned"]) == 1


async def test_maintenance_pause_blocks_opening(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")

    async def _paused() -> dict[str, Any]:
        return {"paused": True, "reason": "maintenance", "status_at": NOW.isoformat()}

    monkeypatch.setattr(house_arm, "get_autoresearch_maintenance_state", _paused)
    before = _write_counts(store)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is False
    assert "paused" in str(result["error"])
    assert _write_counts(store) == before


async def test_raw_looking_key_ref_is_rejected(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    monkeypatch.setenv(house_arm.HOUSE_OPENROUTER_KEY_REF_ENV, "sk-or-v1-rawkeymaterial")
    before = _write_counts(store)
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is False
    assert "raw" in str(result["error"]).lower()
    assert _write_counts(store) == before


async def test_unregistered_encrypted_key_ref_is_rejected(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    monkeypatch.setenv(house_arm.HOUSE_OPENROUTER_KEY_REF_ENV, "encrypted_ref:openrouter:not-registered")
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is False
    assert "not registered" in str(result["error"])


# ---------------------------------------------------------------------------
# Tag shape + reused miner path artifacts
# ---------------------------------------------------------------------------


async def test_open_house_loop_writes_miner_shaped_rows_with_house_tags(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is True
    assert len(result["opened"]) == 1
    opened = result["opened"][0]
    assert opened["status"] == "queued"
    assert opened["lane"] == DEFAULT_PLANNER_LANES[0]

    # Ticket row: miner-shaped with §9.3 arm tags inside ticket_doc.
    assert len(store.tickets) == 1
    ticket = store.tickets[0]
    assert ticket["miner_hotkey"] == HOUSE_HOTKEY
    assert ticket["island"] == "generalist"
    assert ticket["miner_openrouter_key_ref"] == HOUSE_KEY_REF
    assert ticket["miner_openrouter_key_handling"] == "encrypted_ref"
    doc = ticket["ticket_doc"]
    assert doc["arm"] == HOUSE_ARM_TAG
    assert doc["house_policy_ref"] == HOUSE_POLICY_REF
    assert doc["house_lane"] == DEFAULT_PLANNER_LANES[0]
    assert doc["source"] == "gateway_research_lab_house_arm"
    assert doc["requested_compute_budget_usd"] == 5.0
    assert "sk-or-" not in str(store.tickets)

    # Ticket hash discipline matches store.create_ticket (hash over payload with hash="").
    unhashed = dict(ticket)
    unhashed["ticket_hash"] = ""
    assert ticket["ticket_hash"] == house_arm.canonical_hash(unhashed)

    # Deterministic run id — one run per house ticket.
    run_id = house_arm._house_run_id(str(ticket["ticket_id"]))
    assert opened["run_id"] == run_id

    # Payment row: synthetic house ref through the same payments table.
    assert len(store.payments) == 1
    payment = store.payments[0]
    assert payment["payment_ref"] == f"house_arm:{run_id}:0"
    assert payment["required_usd"] == 0.2
    assert payment["verification_doc"]["arm"] == HOUSE_ARM_TAG
    assert payment["verification_doc"]["payment_kind"] == "loop_start"
    assert payment["verification_doc"]["compute_budget_usd"] == 5.0

    # Ticket events: opened -> funded -> queued, same sequence miners get.
    event_types = [event["event_type"] for event in store.ticket_events]
    assert event_types == ["opened", "funded", "queued"]

    # Queue event: capacity doc (scripts/43/54 admission), key refs, arm tags.
    assert len(store.queue_events) == 1
    queue_doc = store.queue_events[0]["event_doc"]
    assert store.queue_events[0]["event_type"] == "queued"
    assert queue_doc["arm"] == HOUSE_ARM_TAG
    assert queue_doc["house_policy_ref"] == HOUSE_POLICY_REF
    assert queue_doc["autoresearch_capacity"] == 5
    assert queue_doc["payment_kind"] == "loop_start"
    assert queue_doc["miner_openrouter_key_ref"] == HOUSE_KEY_REF
    assert queue_doc["miner_openrouter_key_handling"] == "encrypted_ref"
    assert queue_doc["requested_compute_budget_usd"] == 5.0
    assert queue_doc["max_compute_budget_usd"] == 5.0

    # Public card projection through the same safe path.
    assert store.public_projections == [str(ticket["ticket_id"])]


async def test_reopening_same_pass_is_idempotent(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    first = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert len(first["opened"]) == 1
    after_first = _write_counts(store)
    second = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert second["opened"] == []
    assert second["to_open"] == 0
    assert _write_counts(store) == after_first


async def test_orphaned_paid_opening_is_repaired_without_second_payment(store, monkeypatch):
    monkeypatch.setenv(house_arm.HOUSE_ARM_ENABLED_ENV, "true")
    # Open normally, then simulate a crash that lost the queue write.
    await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    store.queue_current.clear()
    store.queue_events.clear()
    payments_before = len(store.payments)
    tickets_before = len(store.tickets)

    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=False, now=NOW)
    assert result["ok"] is True
    repaired = [item for item in result["opened"] if item["action"] == "repaired_orphaned_opening"]
    assert len(repaired) == 1
    assert len(store.payments) == payments_before  # no double funding
    assert len(store.tickets) == tickets_before  # no new ticket
    assert len(store.queue_events) == 1  # queue write restored
    # The repaired orphan consumed the single hotkey slot: no extra loop opened.
    assert [item for item in result["opened"] if item["action"] == "opened_house_loop"] == []


# ---------------------------------------------------------------------------
# Round-robin lane rotation
# ---------------------------------------------------------------------------


def test_next_house_lane_rotates_and_wraps():
    lanes = ("a", "b", "c")
    assert next_house_lane(None, lanes) == "a"
    assert next_house_lane("a", lanes) == "b"
    assert next_house_lane("c", lanes) == "a"
    assert next_house_lane("unknown", lanes) == "a"


def test_planner_lanes_fallback_matches_pinned_copy():
    assert house_arm.planner_lanes() == DEFAULT_PLANNER_LANES


async def test_round_robin_follows_most_recent_house_ticket_lane(store):
    store.tickets.append(
        {
            "ticket_id": str(uuid4()),
            "miner_hotkey": HOUSE_HOTKEY,
            "created_at": "2026-07-01T00:00:00+00:00",
            "ticket_doc": {"arm": HOUSE_ARM_TAG, "house_lane": DEFAULT_PLANNER_LANES[1]},
        }
    )
    result = await open_house_loops(budget_usd_today=200.0, max_open_loops=1, dry_run=True, now=NOW)
    planned = [item for item in result["planned"] if item["action"] == "open_house_loop"]
    assert planned[0]["lane"] == DEFAULT_PLANNER_LANES[2]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


async def test_house_arm_status_reports_spend_and_remaining_clamp(store):
    store.payments.append(_house_payment_row())  # 5.2 spent today
    _open_house_ticket_and_queue_row(store)  # + one open loop (adds its own payment)
    result = await house_arm_status(now=NOW)
    assert result["ok"] is True
    assert result["policy_daily_max_usd"] == 500.0
    assert result["todays_spend_usd"] == pytest.approx(10.4)
    assert result["remaining_clamp_usd"] == pytest.approx(489.6)
    assert result["open_house_loops"] == 1


async def test_house_arm_status_requires_hotkey(store, monkeypatch):
    monkeypatch.delenv(house_arm.HOUSE_HOTKEY_ENV, raising=False)
    result = await house_arm_status(now=NOW)
    assert result["ok"] is False
    assert house_arm.HOUSE_HOTKEY_ENV in str(result["error"])


# ---------------------------------------------------------------------------
# Matched-budget comparison math
# ---------------------------------------------------------------------------


def _candidate(hotkey: str, *, status: str = "scored", bundle_id: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": f"candidate:{uuid4()}",
        "run_id": str(uuid4()),
        "ticket_id": str(uuid4()),
        "miner_hotkey": hotkey,
        "created_at": "2026-06-15T00:00:00+00:00",
        "current_candidate_status": status,
        "current_reason": status,
        "current_score_bundle_id": bundle_id,
    }


def _bundle(bundle_id: str, mean_delta: float) -> dict[str, Any]:
    return {
        "score_bundle_id": bundle_id,
        "score_bundle_doc": {"aggregates": {"mean_delta": mean_delta}},
    }


def _range_payment(hotkey: str, *, fee: float = 0.2, compute: float = 5.0) -> dict[str, Any]:
    row = _house_payment_row(fee=fee, spend_compute=compute)
    row["miner_hotkey"] = hotkey
    row["created_at"] = "2026-06-15T12:00:00+00:00"
    return row


async def test_comparison_math_matched_budget_yield_per_dollar(store):
    # House: 1 payment ($5.2), 1 scored candidate with delta 2.0 (keep).
    store.payments.append(_range_payment(HOUSE_HOTKEY))
    store.score_bundles.append(_bundle("bundle-house", 2.0))
    store.candidates.append(_candidate(HOUSE_HOTKEY, bundle_id="bundle-house"))
    # Miner: 2 payments ($10.4), deltas 1.5 (keep) and 0.5, one failed candidate.
    store.payments.append(_range_payment(MINER_HOTKEY))
    store.payments.append(_range_payment(MINER_HOTKEY))
    store.score_bundles.append(_bundle("bundle-m1", 1.5))
    store.score_bundles.append(_bundle("bundle-m2", 0.5))
    store.candidates.append(_candidate(MINER_HOTKEY, bundle_id="bundle-m1"))
    store.candidates.append(_candidate(MINER_HOTKEY, bundle_id="bundle-m2"))
    store.candidates.append(_candidate(MINER_HOTKEY, status="failed"))

    result = await build_house_arm_comparison_report(
        start_date="2026-06-01", end_date="2026-06-30", config=_fake_config()
    )
    assert result["ok"] is True
    assert result["status"] == "compared"

    arms = result["arms"]
    assert arms["house"]["candidates_scored"] == 1
    assert arms["house"]["keeps"] == 1
    assert arms["house"]["deltas"] == [2.0]
    assert arms["house"]["spend_usd"] == pytest.approx(5.2)
    assert arms["miner"]["candidates_total"] == 3
    assert arms["miner"]["candidates_scored"] == 2
    assert arms["miner"]["keeps"] == 1
    assert arms["miner"]["verified_points"] == pytest.approx(2.0)
    assert arms["miner"]["spend_usd"] == pytest.approx(10.4)

    # Matched budget = min(520, 1040) = 520 cents; yields are per-$1000.
    assert result["matched_budget_usd"] == pytest.approx(5.2)
    summary = result["summary"]
    # miner: 2.0 pts on $10.4 -> scaled 1.0 pt on $5.2 -> 192.307692 pts/$1000
    assert summary["miner_yield_points_per_1000_usd"] == pytest.approx(192.307692, abs=1e-6)
    # house: 2.0 pts on $5.2 -> 384.615385 pts/$1000
    assert summary["house_yield_points_per_1000_usd"] == pytest.approx(384.615385, abs=1e-6)
    assert summary["delta_points_per_1000_usd"] == pytest.approx(-192.307693, abs=1e-5)
    # House out-yielded miners at matched budget -> the honesty gate fails.
    assert summary["miners_add_signal"] is False

    comparison = result["comparison"]
    assert comparison["matched_budget_cents"] == 520
    assert comparison["miner_budget_cents"] == comparison["allocator_budget_cents"] == 520
    assert comparison["state"] == "local_stub"


async def test_comparison_reports_insufficient_data_without_house_spend(store):
    store.payments.append(_range_payment(MINER_HOTKEY))
    result = await build_house_arm_comparison_report(
        start_date="2026-06-01", end_date="2026-06-30", config=_fake_config()
    )
    assert result["ok"] is True
    assert result["status"] == "insufficient_data"
    assert "comparison" not in result


async def test_comparison_rejects_bad_date_range(store):
    result = await build_house_arm_comparison_report(
        start_date="2026-06-30", end_date="2026-06-01", config=_fake_config()
    )
    assert result["ok"] is False
    assert "end_date" in str(result["error"])


async def test_comparison_is_read_only(store):
    store.payments.append(_range_payment(HOUSE_HOTKEY))
    store.payments.append(_range_payment(MINER_HOTKEY))
    before = _write_counts(store)
    await build_house_arm_comparison_report(
        start_date="2026-06-01", end_date="2026-06-30", config=_fake_config()
    )
    assert _write_counts(store) == before


# ---------------------------------------------------------------------------
# Policy record posture
# ---------------------------------------------------------------------------


def test_house_policy_record_validates_with_hard_clamps():
    policy = house_arm.build_house_arm_operating_policy(_fake_config())
    assert policy.daily_budget_min_cents == 20_000  # $200/day floor
    assert policy.daily_budget_max_cents == 50_000  # $500/day ceiling
    # The record stays in its inert P1.9 shape: spending is gated by the env
    # flag + dry_run, never by record fields.
    assert policy.spend_enabled is False
    assert policy.scheduler_enabled is False
    assert policy.grant_eligible is False
