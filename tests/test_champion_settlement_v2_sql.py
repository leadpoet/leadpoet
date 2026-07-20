from pathlib import Path
import re

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


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
            expected_at_99.discard("research_lab.subnet_epoch_cutover.v2")
        if role == "validator_weights":
            expected_at_99.discard("validator.subnet_epoch_snapshot.v2")
        assert migrated_purposes == expected_at_99, role
