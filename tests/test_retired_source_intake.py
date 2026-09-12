"""Retired incentive systems have no executable producer or allocation path."""
from pathlib import Path

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
        "gateway/fulfillment",
        "gateway/api/role_translate.py",
        "gateway/utils/role_translate.py",
        "gateway/utils/hotkey_bucket.py",
        "qualification/scoring/fulfillment_scorer.py",
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
