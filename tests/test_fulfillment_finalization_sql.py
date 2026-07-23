from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "scripts" / "117-fulfillment-finalization-fence.sql"


def test_score_rpc_and_finalization_use_same_request_row_lock() -> None:
    text = SQL.read_text(encoding="utf-8")

    assert "CREATE OR REPLACE FUNCTION public.fulfillment_upsert_scores" in text
    assert "CREATE OR REPLACE FUNCTION public.fulfillment_claim_finalization" in text
    assert text.count("FOR UPDATE;") == 2
    assert "FULFILLMENT_SCORE_WINDOW_CLOSED" in text
    assert "v_request_status NOT IN ('scoring', 'partially_fulfilled')" in text
    assert "SET status = 'finalizing'" in text


def test_request_status_constraint_allows_retryable_finalization() -> None:
    text = SQL.read_text(encoding="utf-8")

    assert "DROP CONSTRAINT IF EXISTS fulfillment_requests_status_check" in text
    assert "ADD CONSTRAINT fulfillment_requests_status_check" in text
    assert "'finalizing'" in text


def test_finalization_claim_is_service_role_only() -> None:
    text = SQL.read_text(encoding="utf-8")

    assert "FROM PUBLIC, anon, authenticated;" in text and "TO service_role;" in text


def test_finalization_claim_checks_count_and_latest_score_timestamp() -> None:
    text = SQL.read_text(encoding="utf-8")

    assert "p_expected_score_count bigint" in text
    assert "p_expected_latest_scored_at timestamptz" in text
    assert "SELECT count(*), max(scored_at)" in text
    assert "v_score_count IS DISTINCT FROM p_expected_score_count" in text
    assert "v_latest_scored_at IS DISTINCT FROM p_expected_latest_scored_at" in text


def test_score_rpc_persists_all_post_rpc_compatibility_fields() -> None:
    text = SQL.read_text(encoding="utf-8")

    for column in (
        "failure_detail",
        "intent_signals_detail",
        "attribute_verification",
    ):
        assert f"EXCLUDED.{column}" in text

    assert "clock_timestamp()" in text
    assert "COALESCE(" in text
