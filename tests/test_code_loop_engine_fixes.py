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
from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    CodeEditPatchApplyError,
    ParentImageSourceContext,
)
from gateway.research_lab.loop_engine import AutoResearchLoopSettings, OpenRouterCallResult
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


def _source_context(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "sourcing_model.py").write_text("x = 1\n", encoding="utf-8")
    return ParentImageSourceContext(
        source_root=source_root,
        source_mode="extracted",
        parent_image_digest_hash="sha256:" + "9" * 64,
        source_tree_hash="sha256:" + "8" * 64,
        top_level_paths=("sourcing_model.py",),
        editable_files=("sourcing_model.py",),
        file_previews=(),
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


# --- reimbursement cost evidence: failed/no-candidate exits keep miner spend ---


class _NoCandidateBuilder:
    def __init__(self, source_context):
        self._source_context = source_context
        self.config = types.SimpleNamespace(
            loop_planner_enabled=False,
            loop_planner_max_tokens=2000,
            loop_alignment_judge_enabled=False,
            loop_alignment_judge_max_tokens=800,
            loop_novelty_strict=False,
            code_edit_source_inspection_rounds=1,
            code_edit_source_inspection_max_files=4,
            code_edit_source_inspection_file_bytes=4000,
            code_edit_source_inspection_total_bytes=16000,
            code_edit_source_inspection_search_matches=5,
            code_edit_patch_repair_attempts=0,
        )

    def prepare_parent_source_context(self, *, parent_artifact, workspace_dir):
        return self._source_context

    def validate_draft_against_source_context(self, draft, source_context, *, read_paths, require_read):
        return []

    def check_patch_applies(self, *, draft, parent_artifact, source_context):
        return None

    def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context):
        raise AssertionError("no-candidate test should not build")


class _PlannerNoCandidateBuilder(_NoCandidateBuilder):
    def __init__(self, source_context):
        super().__init__(source_context)
        self.config.loop_planner_enabled = True


def _loop_direction_plan_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "miner_focus_interpretation": "improve recall without weakening ICP fit",
        "loop_goal": "route to one safe implementation path",
        "required_lane": "source_routing",
        "required_mechanism": "add an alternate discovery surface after completed-empty primary result",
        "generalization_claim": "helps future sealed ICPs with sparse primary-provider coverage",
        "target_behavior": ["preserve ICP constraints", "recover completed-empty primary results"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": ["source_routing"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["do not weaken ICP constraints"],
        "success_criteria": ["patch touches an editable runtime file"],
        "novelty_requirements": ["not a duplicate of prior attempts"],
        "anti_overfit_checks": ["preserves multiple qualified company outputs"],
        "ranked_paths": [
            {
                "path_id": "alternate_discovery_surface",
                "lane": "source_routing",
                "mechanism": "alternate discovery after completed-empty result",
            }
        ],
        "selected_path_id": "alternate_discovery_surface",
    }
    payload.update(overrides)
    return payload


async def test_loop_direction_no_new_safe_path_preserves_stop_reason(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        assert stage == "loop_planner"
        return OpenRouterCallResult(
            content=json.dumps(
                _loop_direction_plan_payload(
                    no_new_safe_path=True,
                    reason="ticket names a concrete provider path not present in editable_files",
                )
            ),
            provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
            cost_microusd=1000,
        )

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-no-new-safe-path",
        ticket={
            "ticket_id": "ticket-no-new-safe-path",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "port a missing provider-specific path"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.iterations_completed == 0
    assert result.stop_reason == "loop_direction_no_new_safe_path"
    assert [event.event_type for event in events].count("no_viable_patch") == 1
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.event_doc["run_summary"]["stop_reason"] == "loop_direction_no_new_safe_path"


async def test_unimplementable_binding_plan_stops_after_one_draft(tmp_path):
    events = []
    calls = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        required_lane="provider_fallback",
                        required_mechanism="salvage malformed Sonar provider responses",
                        allowed_lanes=["provider_fallback"],
                        disallowed_lanes=["query_construction"],
                        ranked_paths=[
                            {
                                "path_id": "sonar_malformed_200_parse_salvage_recall",
                                "lane": "provider_fallback",
                                "mechanism": "add Sonar parse salvage",
                            }
                        ],
                        selected_path_id="sonar_malformed_200_parse_salvage_recall",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "required file is not present in editable_files; no call_sonar implementation exists",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-unimplementable-binding-plan",
        ticket={
            "ticket_id": "ticket-unimplementable-binding-plan",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "salvage malformed Sonar responses"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.iterations_completed == 1
    assert result.stop_reason == "binding_plan_unimplementable"
    assert calls.count("code_edit_draft") == 1
    no_viable = [event for event in events if event.event_type == "no_viable_patch"][-1]
    assert no_viable.event_doc["terminal"] is True
    assert no_viable.event_doc["stop_reason"] == "binding_plan_unimplementable"
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.event_doc["run_summary"]["stop_reason"] == "binding_plan_unimplementable"


async def test_no_candidate_loop_failed_carries_final_cost_ledger(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content='{"no_viable_patch":true,"reason":"loop found no safe candidate"}',
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_NoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-no-candidate",
        ticket={
            "ticket_id": "ticket-no-candidate",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "test"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.selected_candidates == ()
    assert result.actual_openrouter_cost_microusd == 3000
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.cost_ledger["actual_openrouter_cost_microusd"] == 3000
    assert terminal.event_doc["run_summary"]["cost_ledger"]["actual_openrouter_cost_microusd"] == 3000


class _RepairFailureBuilder(_NoCandidateBuilder):
    def __init__(self, source_context):
        super().__init__(source_context)
        self.config.code_edit_patch_repair_attempts = 1

    def check_patch_applies(self, *, draft, parent_artifact, source_context):
        raise CodeEditPatchApplyError("patch does not apply")


class _CostedRepairError(RuntimeError):
    provider_usage = {
        "provider": "openrouter",
        "response_id": "repair-failed",
        "raw_trace_ref": {"s3_ref": "s3://bucket/repair.json", "sha256": "sha256:" + "1" * 64},
    }
    cost_microusd = 4321


async def test_repair_call_failure_records_exception_cost_in_running_ledger(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        assert stage == "code_edit_repair"
        raise _CostedRepairError("repair model failed after provider spend")

    async def event_sink(event):
        events.append(event)

    loop = engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_RepairFailureBuilder(_source_context(tmp_path)),
    )

    repaired, openrouter_calls, estimated_cost, actual_cost_microusd, budget_exhausted = (
        await loop._ensure_patch_applies_or_repair(
            draft=_draft(),
            run_id="run-repair-failed",
            node_id="node-repair-failed",
            iteration=1,
            settings=loop.settings,
            artifact=_manifest(),
            source_context=_source_context(tmp_path),
            source_inspection_context={},
            read_paths=("sourcing_model.py",),
            budget_context={"requested_compute_budget_usd": 5.0},
            budget_limit_microusd=5_000_000,
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
            provider_usage=[],
        )
    )

    assert repaired is None
    assert openrouter_calls == 0
    assert estimated_cost == 0.0
    assert actual_cost_microusd == 4321
    assert budget_exhausted is False
    repair_failure = [
        event
        for event in events
        if event.event_type == "code_edit_repair_failed"
        and event.event_doc.get("stage") == "code_edit_repair_call_failed"
    ][-1]
    assert repair_failure.event_type == "code_edit_repair_failed"
    assert repair_failure.cost_ledger["actual_openrouter_cost_microusd"] == 4321
    assert repair_failure.provider_usage[0]["call_outcome"] == "contained_failure"
    assert repair_failure.provider_usage[0]["raw_trace_ref"]["s3_ref"] == "s3://bucket/repair.json"
    exhausted = [
        event
        for event in events
        if event.event_type == "candidate_repair_exhausted"
        and event.event_doc.get("stage") == "code_edit_repair_call_failed"
    ][-1]
    assert exhausted.cost_ledger["actual_openrouter_cost_microusd"] == 4321
