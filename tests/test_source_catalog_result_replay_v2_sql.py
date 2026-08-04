from pathlib import Path

from gateway.research_lab.attested_v2_store import (
    replayable_execution_result_v2,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
)


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "142-research-lab-source-catalog-result-replay.sql"
).read_text(encoding="utf-8")


def test_source_catalog_result_replay_is_exactly_scoped_and_validated():
    assert replayable_execution_result_v2(
        operation="source_add_catalog_snapshot_v2",
        purpose="research_lab.source_add_catalog_snapshot.v2",
    )
    assert not replayable_execution_result_v2(
        operation="source_add_catalog_snapshot_v2",
        purpose="research_lab.allocation.v2",
    )
    for value in (
        "source_add_catalog_snapshot_v2",
        "research_lab.source_add_catalog_snapshot.v2",
        "research_lab_source_catalog_replay_contract_v2",
    ):
        assert value in SQL
    for constraint in (
        "research_lab_attested_execution_results_v2_operation_check",
        "research_lab_attested_execution_results_v2_purpose_check",
        "research_lab_attested_exec_results_v2_op_purpose_check",
    ):
        assert f"VALIDATE CONSTRAINT\n        {constraint}" in SQL


def test_source_catalog_replay_migration_preserves_existing_authorities():
    for value in (
        "research_lab_allocation",
        "allocation_settlement_frontier_bootstrap_v2",
        "attest_weight_input",
        "attest_active_private_model",
        "observe_chain_realized_weights_v1",
        "attest_chain_realized_settlement_v1",
        "research_lab.allocation.v2",
        "research_lab.active_private_model.v2",
        "research_lab.chain_weight_observation.v1",
        "research_lab.chain_realized_epoch_settlement.v1",
    ):
        assert value in SQL
    assert "NOTIFY pgrst, 'reload schema'" in SQL
    assert (
        "scripts/142-research-lab-source-catalog-result-replay.sql",
        "research_lab_source_catalog_replay_contract_v2",
    ) in REQUIRED_SUPABASE_V2_RPCS
