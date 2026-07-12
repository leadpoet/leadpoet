from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts" / "92-validator-sourcing-attested-v2.sql").read_text(
    encoding="utf-8"
)


def test_sourcing_v2_table_is_additive_append_only_and_service_role_only():
    assert "CREATE TABLE IF NOT EXISTS public.validator_sourcing_epoch_inputs_v2" in SQL
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "prevent_research_lab_attested_v2_mutation" in SQL
    assert "REVOKE ALL" in SQL
    assert "FROM anon, authenticated" in SQL
    assert "TO service_role" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "ALTER TABLE public.research_lab" not in SQL
    assert "Apply script 86 before script 92" in SQL
    assert "DROP TRIGGER" not in SQL
    assert "DROP POLICY" not in SQL
    assert SQL.index("ENABLE ROW LEVEL SECURITY") < SQL.index(
        "CREATE INDEX IF NOT EXISTS"
    )


def test_sourcing_v2_row_binds_canonical_document_and_scoring_receipt():
    assert "leadpoet.sourcing_epoch.v2" in SQL
    assert "qualification.sourcing_epoch.v2" in SQL
    assert "receipt_doc->>'role' = 'gateway_scoring'" in SQL
    assert "source_doc->>'epoch_hash' = epoch_hash" in SQL
    assert "source_doc->>'decision_root' = decision_root" in SQL
    assert "research_lab_attested_execution_receipts_v2(receipt_hash)" in SQL


def test_v2_authority_schema_allows_measured_qualification_purposes():
    authority_sql = (
        ROOT / "scripts" / "86-research-lab-attested-v2-authority.sql"
    ).read_text(encoding="utf-8")
    assert "'qualification.lead_decision.v2'" in authority_sql
    assert "'qualification.sourcing_epoch.v2'" in authority_sql
