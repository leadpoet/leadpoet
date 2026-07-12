from pathlib import Path


SQL_PATH = Path("scripts/89-research-lab-provider-evidence-cache-v2.sql")


def test_provider_evidence_cache_migration_is_append_only_and_private() -> None:
    sql = SQL_PATH.read_text(encoding="utf-8")

    assert "research_lab_provider_evidence_cache_v2" in sql
    assert "PRIMARY KEY (utc_day, request_fingerprint)" in sql
    assert "prevent_research_lab_attested_v2_mutation" in sql
    assert "BEFORE UPDATE OR DELETE" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FROM anon, authenticated" in sql
    assert "GRANT SELECT, INSERT" in sql
    assert "TO service_role" in sql
    assert "encrypted_cache_doc" in sql
    assert "leadpoet.encrypted_artifact.v2" in sql
