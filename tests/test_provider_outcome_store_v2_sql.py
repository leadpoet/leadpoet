from pathlib import Path


SQL = Path("scripts/90-research-lab-provider-outcome-checkpoints-v2.sql")


def test_provider_outcome_checkpoint_migration_is_append_only_and_private() -> None:
    text = SQL.read_text()
    assert "research_lab_provider_outcome_checkpoints_v2" in text
    assert "UNIQUE (utc_day, sequence)" in text
    assert "BEFORE UPDATE OR DELETE" in text
    assert "ENABLE ROW LEVEL SECURITY" in text
    assert "GRANT SELECT, INSERT" in text
    assert "FROM anon, authenticated" in text
    assert "GRANT UPDATE" not in text
    assert "GRANT DELETE" not in text


def test_provider_outcome_checkpoint_migration_stores_ciphertext_not_plaintext() -> None:
    text = SQL.read_text().lower()
    assert "encrypted_checkpoint_doc" in text
    for forbidden in (
        "request_body",
        "response_body",
        "provider_output",
        "openrouter_api_key",
        "scrapingdog_api_key",
        "exa_api_key",
    ):
        assert forbidden not in text
