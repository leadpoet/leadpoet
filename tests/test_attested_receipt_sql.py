from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "scripts" / "80-research-lab-attested-execution-sidecars.sql"


def test_attested_receipt_sql_is_additive_append_only_and_private():
    text = SQL.read_text(encoding="utf-8")
    assert "research_lab_attested_execution_receipts" in text
    assert "research_lab_attested_artifact_links" in text
    assert "research_lab_attested_weight_bundles" in text
    assert "BEFORE UPDATE OR DELETE" in text
    assert "ENABLE ROW LEVEL SECURITY" in text
    assert "GRANT SELECT, INSERT" in text
    assert "GRANT UPDATE" not in text
    assert "GRANT DELETE" not in text
    assert "research_lab_source_add" not in text
    assert "ALTER TABLE public.research_lab_" not in text.replace(
        "ALTER TABLE public.research_lab_attested_execution_receipts", ""
    ).replace("ALTER TABLE public.research_lab_attested_artifact_links", "").replace(
        "ALTER TABLE public.research_lab_attested_weight_bundles", ""
    )
