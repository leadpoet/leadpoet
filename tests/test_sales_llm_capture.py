"""Focused tests for Research Lab sales-LLM capture metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest
from research_lab.canonical import sha256_json


def _manifest(tag: str) -> dict:
    return {
        "schema_version": "1.0",
        "model_artifact_hash": sha256_json({"artifact": tag}),
        "manifest_hash": sha256_json({"manifest": tag}),
    }


def _candidate_request(**overrides):
    parent = _manifest("parent")
    candidate = _manifest("candidate")
    payload = {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "island": "generalist",
        "private_model_manifest": parent,
        "candidate_patch_manifest": {
            "parent_artifact_hash": parent["model_artifact_hash"],
            "candidate_artifact_hash": candidate["model_artifact_hash"],
        },
        "candidate_model_manifest": candidate,
        "candidate_source_diff_hash": sha256_json({"diff": "safe"}),
        "candidate_build_doc": {
            "source_diff_artifact_uri": "file:///tmp/research-lab-fixture-source-diff.json"
        },
    }
    payload.update(overrides)
    return ResearchLabCandidateArtifactCreateRequest(**payload)


def test_image_build_candidate_requires_persisted_source_diff_artifact_uri():
    request = _candidate_request()
    reparsed = ResearchLabCandidateArtifactCreateRequest.model_validate(
        request.model_dump(mode="json")
    )
    assert reparsed.candidate_build_doc["source_diff_artifact_uri"].startswith("file://")

    with pytest.raises(ValueError, match="source_diff_artifact_uri"):
        _candidate_request(candidate_build_doc={"source_diff_hash": sha256_json({"diff": "safe"})})


def test_image_build_candidate_rejects_failed_source_diff_persistence():
    with pytest.raises(ValueError, match="source diff artifact persistence failed"):
        _candidate_request(
            candidate_build_doc={
                "source_diff_artifact_uri": "s3://trace/source-diff.json",
                "source_diff_artifact_error": "s3 put failed",
            }
        )


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
