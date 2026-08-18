from pathlib import Path
import re

from leadpoet_canonical.attested_v2 import COORDINATOR_ROLE, ROLE_PURPOSES
from leadpoet_canonical.weight_authority_v2 import WEIGHT_INPUT_PURPOSES


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "99-research-lab-v2-champion-settlement.sql"
).read_text(encoding="utf-8")
COMPAT_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "102-research-lab-legacy-allocation-netuid-compat.sql"
).read_text(encoding="utf-8")
NONFINALIZATION_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "103-research-lab-legacy-allocation-nonfinalization.sql"
).read_text(encoding="utf-8")
FENCE_REPAIR_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "106-repair-stateful-epoch-fence-trigger-coverage.sql"
).read_text(encoding="utf-8")
EPOCH_INDEX_REPAIR_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "108-repair-legacy-nonfinalization-epoch-index.concurrent.sql"
).read_text(encoding="utf-8")
CHAIN_REALIZED_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "126-research-lab-chain-realized-settlement.sql"
).read_text(encoding="utf-8")
CHAIN_UNATTRIBUTED_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "127-research-lab-chain-unattributed-settlement.sql"
).read_text(encoding="utf-8")
CHAIN_TRANSPORT_PURPOSE_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "128-research-lab-chain-settlement-transport-purposes.sql"
).read_text(encoding="utf-8")
CHAMPION_LIFETIME_CREDIT_SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "132-research-lab-champion-lifetime-credit.sql"
).read_text(encoding="utf-8")


def test_finalized_allocation_view_requires_bundle_publication_and_finalization():
    assert "research_lab_attested_weight_bundles_v2 b" in SQL
    assert "research_lab_attested_publication_events_v2 p" in SQL
    assert "research_lab_attested_weight_finalizations_v2 f" in SQL
    assert "f.weight_submission_event_hash = p.weight_submission_event_hash" in SQL


def test_finalized_allocation_view_is_service_role_only():
    assert "WITH (security_invoker = true)" in SQL
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "TO service_role" in SQL


def test_legacy_settlement_migration_is_append_only_and_receipt_backed():
    assert "research_lab_legacy_finalized_allocation_migrations_v2" in SQL
    assert "PRIMARY KEY (netuid, epoch_id)" in SQL
    assert "REFERENCES public.research_lab_attested_execution_receipts_v2" in SQL
    assert "prevent_research_lab_attested_v2_mutation" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "GRANT SELECT, INSERT" in SQL


def test_historical_allocation_netuid_is_optional_but_must_match_when_present():
    expected_guard = "WHEN NOT (allocation_doc ? 'netuid') THEN TRUE"
    expected_match = "(allocation_doc->>'netuid')::NUMERIC = netuid"
    assert expected_guard in SQL
    assert expected_match in SQL
    assert expected_guard in COMPAT_SQL
    assert expected_match in COMPAT_SQL
    assert "UPDATE public." not in COMPAT_SQL
    assert "DELETE FROM public." not in COMPAT_SQL
    assert (
        "VALIDATE CONSTRAINT research_lab_legacy_allocation_doc_netuid_check"
        in COMPAT_SQL
    )


def test_legacy_nonfinalization_is_append_only_and_creates_no_payment_view():
    assert "research_lab_legacy_allocation_nonfinalizations_v2" in (
        NONFINALIZATION_SQL
    )
    assert "leadpoet.legacy_allocation_nonfinalization.v2" in (
        NONFINALIZATION_SQL
    )
    assert "PRIMARY KEY (netuid, epoch_id)" in NONFINALIZATION_SQL
    assert "finding_receipt_hash" in NONFINALIZATION_SQL
    assert "prevent_research_lab_attested_v2_mutation" in (
        NONFINALIZATION_SQL
    )
    assert "ENABLE ROW LEVEL SECURITY" in NONFINALIZATION_SQL
    assert "GRANT SELECT, INSERT" in NONFINALIZATION_SQL
    assert "research_lab_finalized_allocation_epochs_v2" not in (
        NONFINALIZATION_SQL
    )


def test_legacy_nonfinalization_installs_and_repairs_stateful_epoch_fence():
    trigger = "enforce_research_lab_stateful_epoch_fence_v1"
    table = "research_lab_legacy_allocation_nonfinalizations_v2"
    for sql in (NONFINALIZATION_SQL, FENCE_REPAIR_SQL):
        assert trigger in sql
        assert table in sql
        assert "BEFORE INSERT OR UPDATE" in sql
    assert "trigger_meta.tgenabled <> 'D'" in FENCE_REPAIR_SQL
    assert "a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')" in (
        FENCE_REPAIR_SQL
    )


def test_legacy_nonfinalization_epoch_identity_is_indexed_for_cutover():
    index_name = "idx_research_lab_legacy_nonfinalization_epoch_v2"
    index_target = (
        "public.research_lab_legacy_allocation_nonfinalizations_v2"
        "(epoch_id DESC)"
    )
    assert index_name in NONFINALIZATION_SQL
    assert index_target in NONFINALIZATION_SQL
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in EPOCH_INDEX_REPAIR_SQL
    assert index_name in EPOCH_INDEX_REPAIR_SQL
    assert index_target in EPOCH_INDEX_REPAIR_SQL
    assert "CREATE TABLE" not in EPOCH_INDEX_REPAIR_SQL
    assert "ALTER TABLE" not in EPOCH_INDEX_REPAIR_SQL
    for contract_fragment in (
        "access_method.amname = 'btree'",
        "index_meta.indisvalid",
        "index_meta.indisready",
        "index_meta.indislive",
        "index_meta.indpred IS NULL",
        "index_meta.indexprs IS NULL",
        "index_meta.indkey[0] = epoch_column.attnum",
        "index_meta.indoption[0] = 3",
        "operator_class.opcdefault",
    ):
        assert contract_fragment in EPOCH_INDEX_REPAIR_SQL


def test_deployed_receipt_allowlist_accepts_measured_legacy_settlement():
    assert "DROP CONSTRAINT %I" in SQL
    assert "research_lab_attested_execution_receipts_v2_role_purpose_check" in SQL
    assert "research_lab.legacy_finalized_allocation.v2" in SQL
    assert (
        "VALIDATE CONSTRAINT "
        "research_lab_attested_execution_receipts_v2_role_purpose_check"
    ) in SQL


def test_chain_realized_settlement_tables_are_append_only_and_service_role_only():
    assert "research_lab_chain_realized_epoch_settlements_v1" in (
        CHAIN_REALIZED_SQL
    )
    assert "research_lab_chain_realized_obligation_credits_v1" in (
        CHAIN_REALIZED_SQL
    )
    assert "leadpoet.research_lab_chain_realized_epoch_settlement.v1" in (
        CHAIN_REALIZED_SQL
    )
    assert "leadpoet.research_lab_chain_realized_obligation_credit.v1" in (
        CHAIN_REALIZED_SQL
    )
    assert "PRIMARY KEY (netuid, epoch_id)" in CHAIN_REALIZED_SQL
    assert re.search(
        r"PRIMARY KEY\s*\(\s*netuid,\s*epoch_id,\s*"
        r"obligation_kind,\s*obligation_source_id\s*\)",
        CHAIN_REALIZED_SQL,
    )
    assert "prevent_research_lab_attested_v2_mutation" in CHAIN_REALIZED_SQL
    assert "ENABLE ROW LEVEL SECURITY" in CHAIN_REALIZED_SQL
    assert "FROM PUBLIC, anon, authenticated" in CHAIN_REALIZED_SQL
    assert "GRANT SELECT, INSERT" not in CHAIN_REALIZED_SQL
    assert "persist_research_lab_chain_realized_settlement_v1" in (
        CHAIN_REALIZED_SQL
    )
    assert "GRANT EXECUTE" in CHAIN_REALIZED_SQL


def test_chain_realized_migration_extends_replay_contract_exactly():
    assert re.search(
        r"pg_get_constraintdef\(oid\)\s*~\s*'\\moperation\\M'",
        CHAIN_REALIZED_SQL,
    )
    assert re.search(
        r"pg_get_constraintdef\(oid\)\s*~\s*'\\mpurpose\\M'",
        CHAIN_REALIZED_SQL,
    )
    for value in (
        "observe_chain_realized_weights_v1",
        "attest_chain_realized_settlement_v1",
        "research_lab.chain_weight_observation.v1",
        "research_lab.chain_realized_epoch_settlement.v1",
    ):
        assert value in CHAIN_REALIZED_SQL
    for constraint in (
        "research_lab_attested_execution_results_v2_operation_check",
        "research_lab_attested_execution_results_v2_purpose_check",
        "research_lab_attested_execution_results_v2_operation_purpose_check",
    ):
        assert f"VALIDATE CONSTRAINT\n        {constraint}" in (
            CHAIN_REALIZED_SQL
        )
    assert re.search(
        r"operation\s*=\s*'observe_chain_realized_weights_v1'\s*"
        r"AND purpose\s*=\s*"
        r"'research_lab\.chain_weight_observation\.v1'",
        CHAIN_REALIZED_SQL,
    )
    assert re.search(
        r"operation\s*=\s*'attest_chain_realized_settlement_v1'\s*"
        r"AND purpose\s*=\s*"
        r"'research_lab\.chain_realized_epoch_settlement\.v1'",
        CHAIN_REALIZED_SQL,
    )
    weight_input_match = re.search(
        r"operation\s*=\s*'attest_weight_input'\s*"
        r"AND purpose IN \((.*?)\n\s*\)\s*\n\s*\)\s*\n\s*OR",
        CHAIN_REALIZED_SQL,
        re.DOTALL,
    )
    assert weight_input_match is not None
    migrated_weight_input_purposes = set(
        re.findall(r"'([^']+)'", weight_input_match.group(1))
    )
    canonical_weight_input_purposes = {
        purpose
        for role, purpose in WEIGHT_INPUT_PURPOSES.values()
        if role == COORDINATOR_ROLE
    }
    assert (
        migrated_weight_input_purposes
        == canonical_weight_input_purposes
    )


def test_chain_realized_receipt_allowlist_matches_canonical_contract_exactly():
    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            CHAIN_REALIZED_SQL,
            re.DOTALL,
        )
        assert match is not None, role
        migrated_purposes = set(re.findall(r"'([^']+)'", match.group(1)))
        expected_at_126 = set(expected_purposes)
        if role == "gateway_coordinator":
            expected_at_126.discard(
                "research_lab.ancestry_checkpoint_bootstrap.v2"
            )
            expected_at_126.discard(
                "research_lab.allocation_settlement_frontier_bootstrap.v2"
            )
        if role == "gateway_scoring":
            expected_at_126.difference_update(
                {
                    "research_lab.candidate_hybrid_test.v2",
                    "research_lab.candidate_hybrid_discovery.v2",
                    "research_lab.model_compatibility.v2",
                }
            )
        assert migrated_purposes == expected_at_126, role


def test_chain_settlement_transport_purposes_are_explicitly_admitted():
    assert (
        "research_lab_attested_transport_attempts_v2_purpose_check"
        in CHAIN_TRANSPORT_PURPOSE_SQL
    )
    assert "purpose ~ '\\.v2$'" in CHAIN_TRANSPORT_PURPOSE_SQL
    for purpose in (
        "research_lab.chain_weight_observation.v1",
        "research_lab.chain_realized_epoch_settlement.v1",
    ):
        assert purpose in CHAIN_TRANSPORT_PURPOSE_SQL
    assert (
        "research_lab_attested_transport_purpose_contract_v2"
        in CHAIN_TRANSPORT_PURPOSE_SQL
    )
    assert "constraint_row.convalidated" in CHAIN_TRANSPORT_PURPOSE_SQL
    assert "pg_catalog.pg_get_constraintdef" in CHAIN_TRANSPORT_PURPOSE_SQL
    assert "FROM PUBLIC, anon, authenticated" in CHAIN_TRANSPORT_PURPOSE_SQL
    assert "TO service_role" in CHAIN_TRANSPORT_PURPOSE_SQL
    assert "UPDATE public." not in CHAIN_TRANSPORT_PURPOSE_SQL
    assert "DELETE FROM public." not in CHAIN_TRANSPORT_PURPOSE_SQL


def test_chain_realized_credit_rows_require_complete_epoch_marker():
    assert "settlement_hash" in CHAIN_REALIZED_SQL
    assert re.search(
        r"REFERENCES\s+"
        r"public\.research_lab_chain_realized_epoch_settlements_v1\s*\(",
        CHAIN_REALIZED_SQL,
    )
    assert "jsonb_typeof(settlement_doc->'credit_hashes') = 'array'" in (
        CHAIN_REALIZED_SQL
    )
    assert "credited_alpha_percent <= lab_attributed_alpha_percent" in (
        CHAIN_REALIZED_SQL
    )
    assert "lab_attributed_alpha_percent <= observed_chain_alpha_percent" in (
        CHAIN_REALIZED_SQL
    )


def test_unattributed_chain_marker_is_zero_credit_and_receipt_backed():
    assert "research_lab_chain_realized_epoch_settlement.v2" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "persist_research_lab_chain_realized_unattributed_v2" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "jsonb_array_length(requested_credits) <> 0" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "settlement_doc->'credit_hashes' = '[]'::JSONB" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "research_lab_attested_execution_receipts_v2" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "research_lab.chain_realized_epoch_settlement.v1" in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "INSERT INTO public.research_lab_chain_realized_obligation" not in (
        CHAIN_UNATTRIBUTED_SQL
    )
    assert "GRANT EXECUTE" in CHAIN_UNATTRIBUTED_SQL


def test_chain_realized_rpc_enforces_activation_contiguity_and_receipt():
    assert (
        "chain_realized_settlement_activation_invalid"
        in CHAIN_REALIZED_SQL
    )
    assert (
        "chain_realized_settlement_predecessor_missing"
        in CHAIN_REALIZED_SQL
    )
    assert "predecessor.epoch_id = settlement_epoch - 1" in (
        CHAIN_REALIZED_SQL
    )
    assert "chain_realized_settlement_receipt_invalid" in (
        CHAIN_REALIZED_SQL
    )
    for receipt_contract in (
        "receipt.role = 'gateway_coordinator'",
        "receipt.purpose =",
        "'research_lab.chain_realized_epoch_settlement.v1'",
        "receipt.epoch_id = settlement_epoch",
        "receipt.output_root = requested_settlement_hash",
        "receipt.receipt_status = 'succeeded'",
    ):
        assert receipt_contract in CHAIN_REALIZED_SQL
    assert re.search(
        r"pg_advisory_xact_lock\s*\(\s*"
        r"pg_catalog\.hashtext\('chain_realized_settlement_v1'\),\s*"
        r"settlement_netuid\s*\)",
        CHAIN_REALIZED_SQL,
    )


def test_champion_lifetime_credit_migration_is_additive_and_fail_closed():
    assert "UPDATE public." not in CHAMPION_LIFETIME_CREDIT_SQL
    assert "DELETE FROM public." not in CHAMPION_LIFETIME_CREDIT_SQL
    assert "champion_credit_policy TEXT" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "accelerated_lifetime_cap_v1" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "scheduled_bonus_v1" in CHAMPION_LIFETIME_CREDIT_SQL
    assert (
        "leadpoet.research_lab_chain_realized_epoch_settlement.v3"
        in CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert (
        "leadpoet.research_lab_chain_realized_obligation_credit.v2"
        in CHAMPION_LIFETIME_CREDIT_SQL
    )
    for constraint in (
        "research_lab_chain_settlement_schema_check",
        "research_lab_chain_settlement_champion_policy_check",
        "research_lab_chain_credit_schema_policy_check",
        "research_lab_chain_credit_policy_amount_check",
    ):
        assert f"VALIDATE CONSTRAINT {constraint}" in (
            " ".join(CHAMPION_LIFETIME_CREDIT_SQL.split())
        )


def test_champion_lifetime_credit_rpc_is_atomic_idempotent_and_receipt_backed():
    rpc = "persist_research_lab_chain_realized_lifetime_settlement_v2"
    assert rpc in CHAMPION_LIFETIME_CREDIT_SQL
    assert "pg_advisory_xact_lock" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "ON CONFLICT DO NOTHING" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "chain_realized_lifetime_credit_conflict" in (
        CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert "chain_realized_lifetime_credit_set_conflict" in (
        CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert "research_lab_attested_execution_receipts_v2" in (
        CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert "receipt_status = 'succeeded'" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "GRANT EXECUTE" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "TO service_role" in CHAMPION_LIFETIME_CREDIT_SQL


def test_champion_lifetime_credit_contract_exposes_validated_schema():
    assert "research_lab_champion_lifetime_credit_contract_v1" in (
        CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert "convalidated" in CHAMPION_LIFETIME_CREDIT_SQL
    assert "pg_catalog.pg_get_constraintdef" in (
        CHAMPION_LIFETIME_CREDIT_SQL
    )
    assert "NOTIFY pgrst, 'reload schema'" in CHAMPION_LIFETIME_CREDIT_SQL


def test_migration_99_allowlist_matches_canonical_contract_before_migration_101():
    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        migrated_purposes = set(re.findall(r"'([^']+)'", match.group(1)))
        expected_at_99 = set(expected_purposes)
        if role == "gateway_coordinator":
            expected_at_99.discard(
                "research_lab.ancestry_checkpoint_bootstrap.v2"
            )
            expected_at_99.discard(
                "research_lab.allocation_settlement_frontier_bootstrap.v2"
            )
            expected_at_99.discard("research_lab.subnet_epoch_cutover.v2")
            expected_at_99.discard(
                "research_lab.chain_realized_epoch_settlement.v1"
            )
            expected_at_99.discard(
                "research_lab.chain_realized_obligation_credit.v1"
            )
            expected_at_99.discard(
                "research_lab.chain_weight_observation.v1"
            )
        if role == "gateway_scoring":
            expected_at_99.difference_update(
                {
                    "research_lab.candidate_hybrid_test.v2",
                    "research_lab.candidate_hybrid_discovery.v2",
                    "research_lab.model_compatibility.v2",
                }
            )
        if role == "validator_weights":
            expected_at_99.discard("validator.subnet_epoch_snapshot.v2")
        assert migrated_purposes == expected_at_99, role
