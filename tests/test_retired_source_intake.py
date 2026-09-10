"""Retired incentive systems have no executable producer or allocation path."""
from pathlib import Path

import pytest

from gateway.tee.coordinator_executor_v2 import COORDINATOR_OPERATIONS_V2
from gateway.tee.supabase_source_v2 import QUERY_POLICIES


def test_retired_incentive_producers_are_absent():
    root = Path(__file__).resolve().parents[1]
    for path in (
        "research_lab/source_add.py",
        "research_lab/source_add_rewards.py",
        "research_lab/validator_integration.py",
        "gateway/research_lab/allocations.py",
        "gateway/research_lab/maintenance.py",
        "gateway/research_lab/champion_settlement_v2.py",
        "gateway/tee/reward_executor_v2.py",
        "gateway/tee/coordinator_allocation_source_v2.py",
        "leadpoet_verifier/economics.py",
        "gateway/api/weights.py",
        "gateway/fulfillment/rewards.py",
        "neurons/auditor_validator.py",
    ):
        assert not (root / path).exists(), path


def test_qualification_cannot_execute_retired_incentive_operations():
    assert set(COORDINATOR_OPERATIONS_V2) == {
        "attest_artifact_persistence", "attest_qualification_admission",
    }
    assert set(QUERY_POLICIES) == {
        "qualification_epoch_assignment", "qualification_leads_by_ids", "banned_hotkeys",
    }


@pytest.mark.asyncio
async def test_fulfillment_preserves_delivered_winners_without_emission_writes(monkeypatch):
    from gateway.fulfillment import lifecycle

    writes = []
    enriched = []

    class Query:
        def update(self, fields):
            writes.append({"fields": fields, "keys": {}})
            return self

        def eq(self, field, value):
            writes[-1]["keys"][field] = value
            return self

        def execute(self):
            return None

    class Database:
        def table(self, name):
            assert name == "fulfillment_score_consensus"
            return Query()

    async def enrich(database, request_id, winners):
        enriched.extend(winners)

    monkeypatch.setattr(lifecycle, "_get_supabase", Database)
    monkeypatch.setattr(lifecycle, "_attach_intent_details_for_winners", enrich)
    first = {"request_id": "earlier", "submission_id": "a", "lead_id": "lead"}
    tied = {"request_id": "current", "submission_id": "b", "lead_id": "lead"}
    result = await lifecycle._finalize_chain_winners(
        "current", [first], [("lead", [first, tied])],
    )
    assert result == {"lead"}
    assert [row["keys"]["request_id"] for row in writes] == ["earlier", "current"]
    assert all(row["fields"] == {"is_winner": True} for row in writes)
    assert len(enriched) == 2
