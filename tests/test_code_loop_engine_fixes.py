"""Tests for the code_loop_engine fixes from fableanalysis.md.

Covers: bug #5 (checkpoint rehydration + resume restore), bug #16/#17/#20/#21
kill-switch flags, §6.2-5 within-run memory sanitization, and node-id
uniqueness (the S3 diff-key overwrite half of bug #20).
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from gateway.research_lab import code_loop_engine as engine
from gateway.research_lab.code_build import CodeEditBuildResult
from research_lab.code_editing import CodeEditDraft
from research_lab.eval import PrivateModelArtifactManifest


def _draft(**overrides):
    payload = dict(
        failure_mode="weak sourcing recall",
        mechanism="widen provider query fan-out",
        expected_improvement="+2 companies per ICP",
        risk="slower sourcing",
        lane="provider",
        target_files=("sourcing_model.py",),
        unified_diff="--- a/sourcing_model.py\n+++ b/sourcing_model.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        redacted_summary="widen fan-out",
        test_plan="run adapter smoke",
        rollback_plan="revert diff",
        predicted_delta=1.5,
        plan_path_id="path-1",
        plan_alignment={"aligned": True},
    )
    payload.update(overrides)
    return CodeEditDraft(**payload)


def _manifest():
    return PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "a" * 64,
        git_commit_sha="b" * 40,
        image_digest="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        component_registry_version="1.0",
        scoring_adapter_version="1.0",
        manifest_uri="s3://test-bucket/research-lab/sourcing-model/manifest.json",
        manifest_hash="sha256:" + "e" * 64,
        signature_ref="kms://sig",
        build_id="build-1",
    )


def _build_result():
    return CodeEditBuildResult(
        candidate_model_manifest=_manifest(),
        code_edit_manifest={"target_files": ["sourcing_model.py"], "kind": "code_edit"},
        source_diff_hash="sha256:" + "f" * 64,
        build_doc={"build_doc_hash": "sha256:" + "1" * 64, "source_diff_artifact_uri": "s3://test-bucket/diff.json"},
    )


class _FakeS3:
    def __init__(self, store):
        self.store = store

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        body = self.store[(Bucket, Key)]
        return {"Body": types.SimpleNamespace(read=lambda: body)}


@pytest.fixture()
def fake_boto3(monkeypatch):
    store: dict = {}
    fake = types.ModuleType("boto3")
    fake.client = lambda name: _FakeS3(store)
    monkeypatch.setitem(sys.modules, "boto3", fake)
    return store


# --- bug #5: rehydration artifact write + restore round-trip ---


def test_write_and_rehydrate_candidate_round_trip(fake_boto3):
    doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id="node-abc",
            iteration=2,
            draft=_draft(),
            build=_build_result(),
        )
    )
    assert doc["loop_candidate_artifact_uri"].startswith("s3://test-bucket/")
    assert doc["loop_candidate_artifact_hash"]

    (_bucket, _key), body = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    candidate = engine._rehydrated_candidate_from_artifact_payload(payload)
    assert candidate.node_id == "node-abc"
    assert candidate.iteration == 2
    assert candidate.draft == _draft()
    assert candidate.build.source_diff_hash == "sha256:" + "f" * 64
    assert candidate.build.candidate_model_manifest == _manifest()


def test_rehydrate_rejects_hash_mismatch(fake_boto3):
    asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id="node-abc",
            iteration=1,
            draft=_draft(),
            build=_build_result(),
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    payload["source_diff_hash"] = "sha256:tampered"
    with pytest.raises(ValueError):
        engine._rehydrated_candidate_from_artifact_payload(payload)


def test_write_skips_non_s3_manifest():
    manifest = PrivateModelArtifactManifest.from_mapping(
        {**_manifest().to_dict(), "manifest_uri": "file:///local/manifest.json"}
    )
    doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=manifest,
            run_id="run-1",
            node_id="node",
            iteration=0,
            draft=_draft(),
            build=_build_result(),
        )
    )
    assert doc == {"loop_candidate_artifact_skipped": "manifest_uri_not_s3"}


def _engine_instance():
    return engine.CodeEditLoopEngine(
        settings=object(),
        call_openrouter=None,
        event_sink=None,
        builder=None,
    )


def test_restore_selected_from_resume(fake_boto3):
    write_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id="node-abc",
            iteration=3,
            draft=_draft(),
            build=_build_result(),
        )
    )
    resume = {
        "selected_candidates": [
            {
                "node_id": "node-abc",
                "rehydration_artifact_uri": write_doc["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": write_doc["loop_candidate_artifact_hash"],
            },
            # Legacy summary without rehydration refs degrades silently.
            {"node_id": "node-legacy"},
        ]
    }
    restored = asyncio.run(
        _engine_instance()._restore_selected_from_resume(
            resume=resume,
            run_id="run-1",
            artifact=_manifest(),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert len(restored) == 1
    assert restored[0].node_id == "node-abc"
    assert restored[0].iteration == 3
    assert restored[0].rehydration_artifact_uri == write_doc["loop_candidate_artifact_uri"]


def test_restore_skips_hash_mismatch(fake_boto3):
    write_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id="node-abc",
            iteration=1,
            draft=_draft(),
            build=_build_result(),
        )
    )
    resume = {
        "selected_candidates": [
            {
                "node_id": "node-abc",
                "rehydration_artifact_uri": write_doc["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": "sha256:not-the-right-hash",
            }
        ]
    }
    restored = asyncio.run(
        _engine_instance()._restore_selected_from_resume(
            resume=resume,
            run_id="run-1",
            artifact=_manifest(),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert restored == []


def test_restore_tolerates_missing_selected_candidates():
    restored = asyncio.run(
        _engine_instance()._restore_selected_from_resume(
            resume={},
            run_id="run-1",
            artifact=_manifest(),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert restored == []


# --- kill-switch flags (bugs 5, 16, 17, 20, 21; §6.2-5) ---


@pytest.mark.parametrize(
    "helper,env",
    [
        (engine._resume_restore_selected_enabled, "RESEARCH_LAB_LOOP_RESUME_RESTORE_SELECTED"),
        (engine._planner_parse_retry_enabled, "RESEARCH_LAB_LOOP_PLANNER_PARSE_RETRY"),
        (engine._stage_error_containment_enabled, "RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT"),
        (engine._stop_at_candidate_cap_enabled, "RESEARCH_LAB_LOOP_STOP_AT_CANDIDATE_CAP"),
        (engine._judge_parse_soft_skip_enabled, "RESEARCH_LAB_LOOP_JUDGE_PARSE_SOFT_SKIP"),
        (engine._within_run_memory_enabled, "RESEARCH_LAB_LOOP_WITHIN_RUN_MEMORY"),
        (engine._min_runtime_skip_when_selected_enabled, "RESEARCH_LAB_LOOP_MIN_RUNTIME_SKIP_WHEN_SELECTED"),
        (engine._build_heartbeat_enabled, "RESEARCH_LAB_LOOP_BUILD_HEARTBEAT"),
    ],
)
def test_fix_flags_default_on_and_disableable(helper, env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    assert helper() is True
    monkeypatch.setenv(env, "false")
    assert helper() is False


# --- §6.2-5: within-run memory text sanitization ---


def test_memory_safe_text_truncates_and_passes_clean_text():
    text = engine._memory_safe_text("judge rejected: diff does not compile " + "x" * 500)
    assert len(text) <= 280
    assert text.startswith("judge rejected")


def test_memory_safe_text_redacts_secret_shaped_reasons():
    assert "sk-or-" not in engine._memory_safe_text("failed auth with sk-or-abc123")
    assert engine._memory_safe_text("error mentioning service_role denied") == (
        "[redacted secret-like diagnostic text]"
    )


# --- bug #20: node ids unique per build counter (S3 key overwrite fix) ---


def test_node_id_unique_per_candidate_index():
    draft = _draft()
    first = engine._node_id("run-1", 1, 0, draft)
    second = engine._node_id("run-1", 1, 1, draft)
    assert first != second
    assert first == engine._node_id("run-1", 1, 0, draft)
