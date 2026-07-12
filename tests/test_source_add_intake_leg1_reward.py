from types import SimpleNamespace

import pytest

from gateway.research_lab import allocations
from gateway.research_lab import api
from gateway.research_lab import v2_authority
from gateway.research_lab.source_add_provenance import PRECHECK_MANUAL, PRECHECK_PASSED, PRECHECK_REJECTED


def _config(**overrides):
    config = {
        "source_add_rewards_enabled": True,
        "source_add_leg1_alpha_percent": 1.0,
        "source_add_leg1_max_per_utc_day": 10,
        "lab_reward_epochs": 20,
        "evaluation_epoch": 0,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _record(**overrides):
    record = {
        "submission_id": "source_add_submission:abc123",
        "adapter_id": "adapter:credible-api",
        "miner_hotkey": "5MinerHotkey111111111111111111111111111111111",
    }
    record.update(overrides)
    return SimpleNamespace(**record)


def _provenance_graph():
    return {"root_receipt_hash": "sha256:" + "a" * 64}


@pytest.fixture(autouse=True)
def _measured_reward_authority(monkeypatch):
    async def authorize(**kwargs):
        assert kwargs["decision_kind"] == "source_add_leg1"
        assert kwargs["artifact_kind"] == "source_add_reward_decision"
        assert kwargs["parent_graphs"] == (_provenance_graph(),)
        return {"status": "matched"}

    monkeypatch.setattr(v2_authority, "authorize_reward_decision_v2", authorize)


@pytest.mark.asyncio
async def test_precheck_pass_creates_leg1_reward_with_null_catalog(monkeypatch):
    writes = []

    async def fake_select_many(*_args, **_kwargs):
        return []

    async def fake_insert_row(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    async def fake_epoch(_configured=0):
        return 700, None, "test"

    monkeypatch.setattr(api, "select_many", fake_select_many)
    monkeypatch.setattr(api, "insert_row", fake_insert_row)
    monkeypatch.setattr(api, "resolve_research_lab_evaluation_epoch", fake_epoch)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=PRECHECK_PASSED,
        provenance_graph=_provenance_graph(),
        config=_config(),
    )

    assert result["source_add_leg1_reward_status"] == "created"
    obligation = writes[0][1]
    event = writes[1][1]
    assert writes[0][0] == "research_lab_source_add_reward_obligations"
    assert obligation["catalog_id"] is None
    assert obligation["adapter_id"] == "adapter:credible-api"
    assert obligation["miner_hotkey"] == "5MinerHotkey111111111111111111111111111111111"
    assert obligation["leg"] == 1
    assert obligation["reward_kind"] == "source_acceptance"
    assert obligation["alpha_percent"] == pytest.approx(1.0)
    assert obligation["reward_epochs"] == 20
    assert obligation["start_epoch"] == 701
    assert obligation["trigger_evidence_doc"]["reward_trigger"] == "provenance_precheck_passed"
    assert writes[1][0] == "research_lab_source_add_reward_events"
    assert event["reward_status"] == "active"
    assert event["reason"] == "leg1_provenance_precheck_passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [PRECHECK_MANUAL, PRECHECK_REJECTED, ""])
async def test_non_pass_precheck_does_not_create_leg1_reward(monkeypatch, status):
    async def fail_insert_row(*_args, **_kwargs):
        raise AssertionError("reward rows must not be inserted")

    monkeypatch.setattr(api, "insert_row", fail_insert_row)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=status,
        provenance_graph=_provenance_graph(),
        config=_config(),
    )

    assert result["source_add_leg1_reward_status"] == "skipped_precheck_not_passed"


@pytest.mark.asyncio
async def test_rewards_disabled_does_not_create_leg1_reward(monkeypatch):
    async def fail_insert_row(*_args, **_kwargs):
        raise AssertionError("reward rows must not be inserted")

    monkeypatch.setattr(api, "insert_row", fail_insert_row)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=PRECHECK_PASSED,
        provenance_graph=_provenance_graph(),
        config=_config(source_add_rewards_enabled=False),
    )

    assert result["source_add_leg1_reward_status"] == "disabled"


@pytest.mark.asyncio
async def test_existing_leg1_reward_is_idempotent(monkeypatch):
    async def fake_select_many(*_args, **_kwargs):
        return [{"adapter_id": "adapter:credible-api", "leg": 1, "reward_ref": "source_add_reward:existing123456"}]

    async def fail_insert_row(*_args, **_kwargs):
        raise AssertionError("existing Leg 1 must not be inserted again")

    async def fake_epoch(_configured=0):
        return 700, None, "test"

    monkeypatch.setattr(api, "select_many", fake_select_many)
    monkeypatch.setattr(api, "insert_row", fail_insert_row)
    monkeypatch.setattr(api, "resolve_research_lab_evaluation_epoch", fake_epoch)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=PRECHECK_PASSED,
        provenance_graph=_provenance_graph(),
        config=_config(),
    )

    assert result["source_add_leg1_reward_status"] == "already_created"


@pytest.mark.asyncio
async def test_leg1_daily_cap_blocks_new_reward(monkeypatch):
    async def fake_select_many(table, **_kwargs):
        if table == "research_lab_source_add_reward_current":
            return []
        if table == "research_lab_source_add_reward_events":
            return [{"reward_ref": f"source_add_reward:{i:016x}"} for i in range(2)]
        return []

    async def fail_insert_row(*_args, **_kwargs):
        raise AssertionError("daily cap must prevent insert")

    async def fake_epoch(_configured=0):
        return 700, None, "test"

    monkeypatch.setattr(api, "select_many", fake_select_many)
    monkeypatch.setattr(api, "insert_row", fail_insert_row)
    monkeypatch.setattr(api, "resolve_research_lab_evaluation_epoch", fake_epoch)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=PRECHECK_PASSED,
        provenance_graph=_provenance_graph(),
        config=_config(source_add_leg1_max_per_utc_day=2),
    )

    assert result["source_add_leg1_reward_status"] == "daily_cap_reached"
    assert result["daily_cap"] == 2


@pytest.mark.asyncio
async def test_duplicate_insert_is_treated_as_already_created(monkeypatch):
    async def fake_select_many(*_args, **_kwargs):
        return []

    async def duplicate_insert(*_args, **_kwargs):
        raise RuntimeError("duplicate key value violates unique constraint 23505")

    async def fake_epoch(_configured=0):
        return 700, None, "test"

    monkeypatch.setattr(api, "select_many", fake_select_many)
    monkeypatch.setattr(api, "insert_row", duplicate_insert)
    monkeypatch.setattr(api, "resolve_research_lab_evaluation_epoch", fake_epoch)

    result = await api._maybe_create_source_add_leg1_reward_for_precheck(
        record=_record(),
        precheck_status=PRECHECK_PASSED,
        provenance_graph=_provenance_graph(),
        config=_config(),
    )

    assert result["source_add_leg1_reward_status"] == "already_created"


@pytest.mark.asyncio
async def test_allocation_reads_active_source_add_reward_without_catalog(monkeypatch):
    async def fake_select_all(table, *, filters=(), **_kwargs):
        assert table == "research_lab_source_add_reward_current"
        if ("current_reward_status", "active") not in filters:
            return []
        return [
            {
                "reward_ref": "source_add_reward:" + "1" * 16,
                "adapter_id": "adapter:credible-api",
                "catalog_id": None,
                "miner_hotkey": "5MinerHotkey111111111111111111111111111111111",
                "leg": 1,
                "reward_kind": "source_acceptance",
                "current_reward_status": "active",
                "desired_alpha_percent": 1.0,
                "start_epoch": 700,
                "epoch_count": 20,
            }
        ]

    monkeypatch.setattr(allocations, "select_all", fake_select_all)

    rows = await allocations._active_source_add_reward_rows(701)

    assert len(rows) == 1
    assert rows[0]["miner_hotkey"] == "5MinerHotkey111111111111111111111111111111111"
    assert rows[0]["desired_alpha_percent"] == pytest.approx(1.0)
    assert rows[0]["reward_kind"] == "source_acceptance"
