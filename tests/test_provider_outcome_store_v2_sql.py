from pathlib import Path


SQL = Path("scripts/90-research-lab-provider-outcome-checkpoints-v2.sql")
APPEND_SQL = Path("scripts/131-research-lab-provider-outcome-backpressure.sql")
CONTENTION_STATUS_SQL = Path(
    "scripts/133-research-lab-provider-outcome-contention-status.sql"
)


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


def test_provider_outcome_checkpoint_append_is_atomic_and_private() -> None:
    text = APPEND_SQL.read_text()
    assert "append_research_lab_provider_outcome_checkpoint_v2" in text
    assert "pg_try_advisory_xact_lock" in text
    assert "provider outcome checkpoint append is busy" in text
    assert "pg_advisory_xact_lock(" not in text
    assert "ORDER BY c.sequence DESC" in text
    assert "checkpoint_sequence <> current_sequence + 1" in text
    assert "incoming_previous_hash <> current_checkpoint_hash" in text
    assert "ERRCODE = '40001'" in text
    assert "GRANT EXECUTE" in text
    assert "TO service_role" in text
    assert "FROM PUBLIC, anon, authenticated" in text
    assert "UPDATE public.research_lab_provider_outcome_checkpoints_v2" not in text
    assert "DELETE FROM public.research_lab_provider_outcome_checkpoints_v2" not in text


def test_provider_outcome_expected_contention_returns_measured_statuses() -> None:
    text = CONTENTION_STATUS_SQL.read_text()
    assert "pg_try_advisory_xact_lock" in text
    assert "'status', 'busy'" in text
    assert "'status', 'conflict'" in text
    assert "research_lab_provider_outcome_contention_contract_v2" in text
    assert "leadpoet.provider_outcome_contention_contract.v2" in text
    assert "GRANT EXECUTE" in text
    assert "TO service_role" in text
    assert "FROM PUBLIC, anon, authenticated" in text
    assert "RAISE EXCEPTION 'provider outcome checkpoint fields are invalid'" in text
    assert "RAISE EXCEPTION 'provider outcome checkpoint identity is invalid'" in text
    assert "RAISE EXCEPTION 'provider outcome checkpoint durable insert differs'" in text
    assert "UPDATE public.research_lab_provider_outcome_checkpoints_v2" not in text
    assert "DELETE FROM public.research_lab_provider_outcome_checkpoints_v2" not in text
