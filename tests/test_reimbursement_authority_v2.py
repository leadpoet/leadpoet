from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.research_lab import attested_v2_store, reimbursement_awards, v2_authority
from gateway.research_lab.tee_protocol import ResearchLabTeeProtocolError


def _policy():
    return {
        "policy_id": "reimbursement-policy:test",
        "enabled": True,
        "min_rebate_rate": 0.25,
        "base_rebate_rate": 0.5,
        "max_rebate_rate": 0.75,
        "high_participation_target": 10,
        "reimbursement_epochs": 20,
        "max_usd_per_run": 1000,
        "max_usd_per_hotkey_day": 5000,
        "max_usd_per_island_day": 10000,
        "global_budget_usd": 100000,
        "include_loop_start_fee_in_base": False,
        "material_spend_ratio": 0.0,
        "default_island": "generalist",
        "usd_per_0_1_percent_epoch": 1.0,
        "distinct_funded_hotkey_weight": 1,
        "paid_loop_weight": 1,
        "unique_brief_weight": 1,
    }


def _config():
    return SimpleNamespace(
        reimbursements_enabled=True,
        shadow_reimbursements_enabled=False,
        reimbursement_default_island="generalist",
        loop_start_fee_usd=0.0,
        evaluation_epoch=0,
        reimbursement_policy_doc=lambda enabled: {**_policy(), "enabled": enabled},
    )


def _snapshot():
    return {
        "snapshot_id": "participation:generalist:2026-07-10",
        "island": "generalist",
        "lookback_start": "2026-07-03T00:00:00+00:00",
        "lookback_end": "2026-07-10T00:00:00+00:00",
        "distinct_funded_hotkeys": 1,
        "paid_loop_count": 1,
        "unique_brief_count": 1,
    }


@pytest.mark.asyncio
async def test_reimbursement_receipt_precedes_every_business_write(monkeypatch):
    order = []
    graph = {"root_receipt_hash": "sha256:" + "a" * 64}

    async def load_graph(**kwargs):
        assert kwargs == {
            "artifact_kind": "autoresearch_run",
            "artifact_ref": "run-1",
        }
        return graph

    async def authorize(**kwargs):
        assert order == []
        assert kwargs["decision_kind"] == "reimbursement"
        assert kwargs["artifact_kind"] == "reimbursement_decision"
        assert kwargs["parent_graphs"] == (graph,)
        assert kwargs["expected_result"] is None
        assert kwargs["artifact_ref"] == ""
        assert kwargs["decision_payload"]["autoresearch_result"]["status"] == "completed"
        order.append("authorized")
        award = {
            "award_id": "reimbursement_award:sha256:" + "1" * 64,
            "run_id": "run-1",
            "miner_hotkey": "hotkey-1",
            "island": "generalist",
            "run_day": "2026-07-10",
            "status": "awarded",
            "award_status": "awarded",
            "participation_score": 3.0,
            "participation_fraction": 0.3,
            "rebate_rate": 0.5,
            "eligible_cost_microusd": 10_000_000,
            "target_reimbursement_microusd": 5_000_000,
            "target_reimbursement_usd": 5.0,
            "reimbursement_epochs": 20,
            "loop_start_fee_included": False,
            "input_hash": "sha256:" + "2" * 64,
        }
        schedule = {
            "schedule_id": "reimbursement_schedule:sha256:" + "3" * 64,
            "award_id": award["award_id"],
            "status": "scheduled",
            "schedule_status": "scheduled",
            "start_epoch": 251,
            "epoch_count": 20,
            "total_microusd": 5_000_000,
            "entries": [],
        }
        return {
            "status": "matched",
            "result": {
                "decision_kind": "reimbursement",
                "award": award,
                "schedule": schedule,
                "source_state": {
                    "run_cost": {
                        "run_id": "run-1",
                        "actual_openrouter_cost_usd": 10.0,
                    },
                    "participation_snapshot": _snapshot(),
                    "policy": _policy(),
                    "cap_usage": {},
                    "start_epoch": 251,
                },
            },
        }

    async def persist_snapshot(**_kwargs):
        order.append("snapshot")
        return {"participation_snapshot_id": "snapshot-row-1"}

    async def create_award(**kwargs):
        order.append("award")
        return dict(kwargs["award"]), {"event_type": "awarded"}

    async def create_schedule(**kwargs):
        order.append("schedule")
        return {"schedule_id": kwargs["schedule"]["schedule_id"]}

    async def resolve_epoch(_configured):
        return 250, 90000, "test"

    monkeypatch.setattr(
        reimbursement_awards,
        "_build_participation_snapshot",
        lambda *_args, **_kwargs: _async(_snapshot()),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "_reimbursement_cap_usage",
        lambda *_args, **_kwargs: _async(
            {
                "hotkey_day_awarded_usd": 0.0,
                "island_day_awarded_usd": 0.0,
                "global_awarded_usd": 0.0,
            }
        ),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "_persist_participation_snapshot",
        persist_snapshot,
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "resolve_research_lab_evaluation_epoch",
        resolve_epoch,
    )
    monkeypatch.setattr(reimbursement_awards, "create_reimbursement_award", create_award)
    monkeypatch.setattr(
        reimbursement_awards,
        "create_reimbursement_schedule",
        create_schedule,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(v2_authority, "authorize_reward_decision_v2", authorize)

    result = await reimbursement_awards.create_reimbursement_decision(
        _config(),
        run_id="run-1",
        ticket_id="ticket-1",
        ticket={"miner_hotkey": "hotkey-1", "island": "generalist"},
        payment={"payment_id": "payment-1"},
        receipt_id="receipt-1",
        budget_context={"requested_compute_budget_usd": 20.0},
        cost_evidence={
            "trusted_cost_ledger": True,
            "actual_openrouter_cost_usd": 10.0,
            "cost_ledger": {"actual_openrouter_cost_usd": 10.0},
        },
        source="test",
        miner_openrouter_key_ref="encrypted_ref:openrouter:test",
        autoresearch_result={
            "status": "completed",
            "actual_openrouter_cost_microusd": 10_000_000,
        },
    )

    assert result["status"] == "awarded"
    assert order == ["authorized", "snapshot", "award", "schedule"]


@pytest.mark.asyncio
async def test_reimbursement_authority_failure_prevents_business_writes(monkeypatch):
    writes = []

    async def reject(**_kwargs):
        raise RuntimeError("measured authority unavailable")

    async def no_write(**_kwargs):
        writes.append(True)
        raise AssertionError("business persistence must not run")

    monkeypatch.setattr(
        reimbursement_awards,
        "_build_participation_snapshot",
        lambda *_args, **_kwargs: _async(_snapshot()),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "_reimbursement_cap_usage",
        lambda *_args, **_kwargs: _async({}),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "resolve_research_lab_evaluation_epoch",
        lambda *_args, **_kwargs: _async((250, 90000, "test")),
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        lambda **_kwargs: _async({"root_receipt_hash": "sha256:" + "a" * 64}),
    )
    monkeypatch.setattr(v2_authority, "authorize_reward_decision_v2", reject)
    monkeypatch.setattr(reimbursement_awards, "_persist_participation_snapshot", no_write)
    monkeypatch.setattr(reimbursement_awards, "create_reimbursement_award", no_write)
    monkeypatch.setattr(reimbursement_awards, "create_reimbursement_schedule", no_write)

    with pytest.raises(RuntimeError, match="measured authority"):
        await reimbursement_awards.create_reimbursement_decision(
            _config(),
            run_id="run-1",
            ticket_id="ticket-1",
            ticket={"miner_hotkey": "hotkey-1", "island": "generalist"},
            payment={"payment_id": "payment-1"},
            receipt_id="receipt-1",
            budget_context={"requested_compute_budget_usd": 20.0},
            cost_evidence={
                "trusted_cost_ledger": True,
                "actual_openrouter_cost_usd": 10.0,
                "cost_ledger": {"actual_openrouter_cost_usd": 10.0},
            },
            source="test",
            miner_openrouter_key_ref="encrypted_ref:openrouter:test",
            autoresearch_result={
                "status": "completed",
                "actual_openrouter_cost_microusd": 10_000_000,
            },
        )
    assert writes == []


@pytest.mark.asyncio
async def test_legacy_reimbursement_is_rejected_before_any_write(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")
    order = []

    async def persist_snapshot(**_kwargs):
        order.append("snapshot")
        return {"participation_snapshot_id": "snapshot-row-1"}

    async def create_award(**kwargs):
        order.append("award")
        return dict(kwargs["award"]), {"event_type": "awarded"}

    async def create_schedule(**kwargs):
        order.append("schedule")
        return {"schedule_id": kwargs["schedule"]["schedule_id"]}

    monkeypatch.setattr(
        reimbursement_awards,
        "_build_participation_snapshot",
        lambda *_args, **_kwargs: _async(_snapshot()),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "_reimbursement_cap_usage",
        lambda *_args, **_kwargs: _async(
            {
                "hotkey_day_awarded_usd": 0.0,
                "island_day_awarded_usd": 0.0,
                "global_awarded_usd": 0.0,
            }
        ),
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "_persist_participation_snapshot",
        persist_snapshot,
    )
    monkeypatch.setattr(
        reimbursement_awards,
        "resolve_research_lab_evaluation_epoch",
        lambda *_args, **_kwargs: _async((250, 90000, "test")),
    )
    monkeypatch.setattr(reimbursement_awards, "create_reimbursement_award", create_award)
    monkeypatch.setattr(
        reimbursement_awards,
        "create_reimbursement_schedule",
        create_schedule,
    )
    with pytest.raises(ResearchLabTeeProtocolError, match="V1 authority is retired"):
        await reimbursement_awards.create_reimbursement_decision(
            _config(),
            run_id="run-legacy",
            ticket_id="ticket-legacy",
            ticket={"miner_hotkey": "hotkey-1", "island": "generalist"},
            payment={"payment_id": "payment-1"},
            receipt_id="receipt-1",
            budget_context={"requested_compute_budget_usd": 20.0},
            cost_evidence={
                "trusted_cost_ledger": True,
                "actual_openrouter_cost_usd": 10.0,
                "cost_ledger": {"actual_openrouter_cost_usd": 10.0},
            },
            source="test",
            miner_openrouter_key_ref="encrypted_ref:openrouter:test",
        )
    assert order == []


async def _async(value):
    return value
