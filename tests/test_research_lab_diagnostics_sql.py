from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts" / "77-research-lab-ranked-diagnostics-ledger-noise.sql").read_text(
    encoding="utf-8"
)


def test_results_ledger_serving_stamp_migration_is_additive_and_sanitized():
    for column in (
        "serving_model_version_hash",
        "serving_model_manifest_hash",
        "serving_model_artifact_hash",
        "private_model_version_id",
        "candidate_id",
        "score_bundle_id",
        "serving_model_version_doc",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in SQL

    assert "research_lab_results_ledger_serving_model_doc_safe_check" in SQL
    assert "serving_model_version_doc::TEXT !~*" in SQL
    assert "openrouter_api_key" in SQL
    assert "service_role" in SQL
    assert "provider_output" in SQL


def test_daily_noise_budget_view_projects_completed_benchmark_docs_only():
    assert "CREATE OR REPLACE VIEW public.research_lab_daily_noise_budget_report_current" in SQL
    assert "public.research_lab_private_model_benchmark_current" in SQL
    assert "score_summary_doc->'daily_noise_budget'" in SQL
    assert "b.benchmark_quality = 'passed'" in SQL
    assert "b.current_benchmark_status = 'completed'" in SQL
    for field in (
        "mean_icp_score",
        "sample_sd",
        "standard_error",
        "confidence_band_95_lower",
        "confidence_band_95_upper",
        "zero_score_count",
        "high_volatility",
        "observability_only",
    ):
        assert field in SQL

    assert "GRANT SELECT ON TABLE public.research_lab_daily_noise_budget_report_current TO service_role" in SQL
