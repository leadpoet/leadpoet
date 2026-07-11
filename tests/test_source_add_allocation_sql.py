from pathlib import Path


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "87-research-lab-source-add-allocation-containment.sql"
).read_text(encoding="utf-8")


def test_source_add_snapshot_column_and_total_cap_constraint_are_additive():
    assert "ADD COLUMN IF NOT EXISTS source_add_alpha_percent" in SQL
    assert "research_lab_emission_allocation_source_add_cap_check" in SQL
    assert "source_add_alpha_percent\n        + reimbursement_alpha_percent" in SQL
    assert "VALIDATE CONSTRAINT research_lab_emission_allocation_source_add_cap_check" in SQL
    assert "UPDATE public.research_lab_emission_allocation_snapshots" not in SQL
    assert "DELETE FROM public.research_lab_emission_allocation_snapshots" not in SQL


def test_current_view_exposes_source_add_without_widening_permissions():
    assert "CREATE OR REPLACE VIEW public.research_lab_emission_allocation_current" in SQL
    assert "source_add_alpha_percent\nFROM public.research_lab_emission_allocation_snapshots" in SQL
    assert "REVOKE ALL ON TABLE public.research_lab_emission_allocation_current FROM anon, authenticated" in SQL
    assert "GRANT SELECT ON TABLE public.research_lab_emission_allocation_current TO service_role" in SQL


def test_new_score_bundles_reject_raw_private_runtime_pointers():
    assert "research_evaluation_score_bundles_public_containment_check" in SQL
    assert "image[_-]?digest" in SQL
    assert "manifest[_-]?uri" in SQL
    assert "image[_-]?repository" in SQL
    assert "\\.dkr\\.ecr\\." in SQL
    assert "private_model_manifest_doc" in SQL
    assert "candidate_patch_manifest" in SQL
    assert ") NOT VALID;" in SQL
    assert "VALIDATE CONSTRAINT research_evaluation_score_bundles_public_containment_check" not in SQL
