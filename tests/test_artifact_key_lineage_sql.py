from pathlib import Path


SQL = Path("scripts/125-research-lab-artifact-key-lineage.sql").read_text()


def test_artifact_key_lineage_preserves_rows_and_scopes_uniqueness():
    assert "DELETE FROM" not in SQL
    assert "TRUNCATE" not in SQL
    assert "artifact_master_key_ref_hash" in SQL
    assert "research_lab_provider_evidence_cache_v2_key_day_request_key" in SQL
    assert "research_lab_provider_evidence_cache_v2_legacy_day_request_key" in SQL
    assert "research_lab_provider_outcome_checkpoints_v2_key_day_sequence_key" in SQL
    assert "research_lab_provider_outcome_checkpoints_v2_legacy_day_sequence_key" in SQL
    assert SQL.count("WHERE artifact_master_key_ref_hash IS NULL") == 2
    assert "UPDATE public." not in SQL
