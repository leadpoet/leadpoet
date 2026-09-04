"""Focused tests for Research Lab sales-LLM capture metadata."""

from __future__ import annotations

from pathlib import Path

def test_sales_llm_capture_migration_is_additive_service_role_only():
    sql = Path("scripts/76-research-lab-sales-llm-capture.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS public.research_lab_company_label_examples" in sql
    assert "CREATE OR REPLACE VIEW public.research_lab_sales_llm_corpus_metadata_current" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "GRANT SELECT, INSERT ON TABLE public.research_lab_company_label_examples TO service_role" in sql
    assert "eligible_for_training BOOLEAN    NOT NULL DEFAULT FALSE" in sql
    assert "CHECK (eligible_for_training IS FALSE)" in sql
    assert "training_approved" not in sql
    assert "prevent_research_lab_company_label_examples_mutation" in sql
    assert "dedup_key            TEXT        NOT NULL UNIQUE" in sql
    for forbidden in (
        "sk-or-",
        "openrouter_api_key",
        "service_role",
        "page_content",
        "raw_content",
        "private_repo",
        "proxy[_-]?url",
    ):
        assert forbidden in sql
