"""Tests for §6.2-8 multi-candidate drafts and the §6.3-1/§6.3-4 dev-eval wiring
in ``gateway.research_lab.code_loop_engine``.

Covers: multi-candidate parse → N built with the bug-20 cap respected (unique
node ids / S3 rehydration keys, within-run dedupe applied per draft, one draft
LLM call counted once), flag-off single-draft byte-compatibility, dev-score
attachment through the ``dev_evaluator`` seam (ranking, within-run memory feed,
event/checkpoint emission), checkpoint/rehydration round-trips with and without
dev fields, the §6.3-4-lite plateau stop boundaries, and dev-eval failure
containment. Fake patterns follow tests/test_code_loop_engine_fixes.py and
tests/test_learning_flywheel.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab import code_loop_engine as engine
from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    ParentImageSourceContext,
)
from gateway.research_lab.loop_engine import AutoResearchLoopSettings
from research_lab.canonical import sha256_json
from research_lab.code_editing import CodeEditDraft
from research_lab.eval import PrivateModelArtifactManifest


PARENT_HASH = "sha256:" + "a" * 64
S3_MANIFEST_URI = "s3://test-bucket/research-lab/sourcing-model/manifest.json"
DEV_SCORE_VERSION = "research-lab-dev-eval-mechanical-v1"

_DEV_ENVS = (
    "RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS",
    "RESEARCH_LAB_LOOP_DRAFTS_PER_CALL",
    "RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED",
    "RESEARCH_LAB_DEV_SNAPSHOT_URI",
    "RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP",
    "RESEARCH_LAB_LOOP_DEV_PLATEAU_WINDOW",
    "RESEARCH_LAB_LOOP_DEV_PLATEAU_MIN_DELTA",
)


@pytest.fixture(autouse=True)
def _clean_dev_envs(monkeypatch):
    for name in _DEV_ENVS:
        monkeypatch.delenv(name, raising=False)


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


def _manifest(manifest_uri: str = "file:///local/manifest.json") -> PrivateModelArtifactManifest:
    return PrivateModelArtifactManifest(
        model_artifact_hash=PARENT_HASH,
        git_commit_sha="b" * 40,
        image_digest="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        component_registry_version="1.0",
        scoring_adapter_version="1.0",
        manifest_uri=manifest_uri,
        manifest_hash="sha256:" + "e" * 64,
        signature_ref="kms://sig",
        build_id="build-1",
    )


def _registry_doc() -> dict[str, Any]:
    return {
        "manifest_version": "sourcing-model-components:v1",
        "champion_base": "sourcing-model-research-lab-adapter:v1",
        "eval_version": "research-lab-private-evaluator:v1",
        "entries": [],
        "source_receipt_refs": ["receipt:test"],
    }


def _diff(index: int) -> str:
    return (
        "diff --git a/sourcing_model/pipeline.py b/sourcing_model/pipeline.py\n"
        "--- a/sourcing_model/pipeline.py\n"
        "+++ b/sourcing_model/pipeline.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        f"+x = {100 + index}\n"
    )


def _candidate_doc(index: int, *, diff: str | None = None, summary: str | None = None) -> dict[str, Any]:
    return {
        "lane": "provider",
        "hypothesis": {
            "failure_mode": f"weak sourcing recall {index}",
            "mechanism": f"widen provider query fan-out {index}",
            "expected_improvement": "+2 companies per ICP",
            "risk": "slower sourcing",
        },
        "code_edit": {
            "target_files": ["sourcing_model/pipeline.py"],
            "unified_diff": diff if diff is not None else _diff(index),
            "redacted_summary": summary if summary is not None else f"widen fan-out {index}",
            "test_plan": "run adapter smoke",
            "rollback_plan": "revert diff",
        },
    }


def _draft_response(candidate_docs: list[dict[str, Any]]) -> str:
    return json.dumps({"candidates": candidate_docs})


class _FakeCaller:
    """Scripted caller: one draft response per code_edit_draft call (last repeats)."""

    def __init__(self, draft_responses):
        self.calls: list[tuple[str, Any]] = []
        self._draft_responses = list(draft_responses)
        self._draft_call_index = 0

    async def __call__(self, messages, timeout_seconds, max_tokens, stage):
        self.calls.append((stage, messages))
        if stage == "source_inspection":
            return json.dumps(
                {"requests": [{"operation": "read_file", "path": "sourcing_model/pipeline.py"}]}
            )
        if stage == "code_edit_draft":
            index = min(self._draft_call_index, len(self._draft_responses) - 1)
            self._draft_call_index += 1
            return self._draft_responses[index]
        raise AssertionError(f"unexpected stage: {stage}")

    def draft_call_count(self) -> int:
        return sum(1 for stage, _messages in self.calls if stage == "code_edit_draft")

    def draft_message_texts(self) -> list[str]:
        return [
            "\n".join(str(message.get("content") or "") for message in messages)
            for stage, messages in self.calls
            if stage == "code_edit_draft"
        ]


class _FakeBuilder:
    def __init__(self, source_context, *, validate_reject_marker: str | None = None):
        self._source_context = source_context
        self._validate_reject_marker = validate_reject_marker
        self.build_calls: list[dict[str, Any]] = []
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
        if self._validate_reject_marker and self._validate_reject_marker in draft.redacted_summary:
            return ["synthetic source-context rejection"]
        return []

    def check_patch_applies(self, *, draft, parent_artifact, source_context):
        return None

    def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context):
        self.build_calls.append(
            {
                "run_id": run_id,
                "candidate_index": candidate_index,
                "unified_diff": draft.unified_diff,
            }
        )
        return CodeEditBuildResult(
            candidate_model_manifest=_manifest(S3_MANIFEST_URI),
            code_edit_manifest={"target_files": list(draft.target_files), "kind": "code_edit"},
            source_diff_hash=sha256_json({"unified_diff": draft.unified_diff}),
            build_doc={"build_doc_hash": "sha256:" + "1" * 64},
        )


class _ScriptedDevEvaluator:
    """Returns one scripted outcome per call: float score, Exception, or garbage."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    async def __call__(self, candidate):
        self.calls.append(candidate.node_id)
        if not self.outcomes:
            raise RuntimeError("no scripted dev outcome left")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "garbage":
            return "not-a-mapping"
        if outcome is None:
            return {"note": "result without a score"}
        return {
            "dev_score_version": DEV_SCORE_VERSION,
            "aggregate_dev_score": float(outcome),
            "ranking_only": True,
        }


def _source_context(tmp_path: Path) -> ParentImageSourceContext:
    source_root = tmp_path / "source"
    (source_root / "sourcing_model").mkdir(parents=True, exist_ok=True)
    (source_root / "sourcing_model" / "pipeline.py").write_text("x = 1\n", encoding="utf-8")
    return ParentImageSourceContext(
        source_root=source_root,
        source_mode="extracted",
        parent_image_digest_hash="sha256:" + "9" * 64,
        source_tree_hash="sha256:" + "8" * 64,
        top_level_paths=("sourcing_model",),
        editable_files=("sourcing_model/pipeline.py",),
        file_previews=(),
    )


def _run_engine(
    tmp_path: Path,
    *,
    draft_responses,
    max_candidates: int = 1,
    min_iterations: int = 1,
    max_iterations: int = 1,
    dev_evaluator=None,
    manifest_uri: str = "file:///local/manifest.json",
    validate_reject_marker: str | None = None,
    resume_state=None,
):
    caller = _FakeCaller(draft_responses)
    events: list[Any] = []

    async def _sink(event):
        events.append(event)

    builder = _FakeBuilder(_source_context(tmp_path), validate_reject_marker=validate_reject_marker)
    loop_engine = engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=300,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            draft_timeout_seconds=30,
            reflection_timeout_seconds=30,
            estimated_iteration_cost_usd=0.01,
            max_candidates=max_candidates,
        ),
        call_openrouter=caller,
        event_sink=_sink,
        builder=builder,
        dev_evaluator=dev_evaluator,
    )
    result = asyncio.run(
        loop_engine.run(
            run_id="run-mc-1",
            ticket={
                "ticket_id": "ticket-1",
                "island": "generalist",
                "miner_hotkey": "hotkey-1",
                "brief_sanitized_ref": "brief:1",
                "ticket_doc": {"brief_public_summary": "improve sourcing recall"},
            },
            artifact=_manifest(manifest_uri),
            component_registry=_registry_doc(),
            benchmark_public_summary={"aggregate": 12.0},
            model_id="test-model",
            budget_context={"requested_compute_budget_usd": 5.0},
            requested_loop_count=1,
            resume_state=resume_state,
        )
    )
    return result, events, caller, builder


def _events_of(events, event_type):
    return [event for event in events if event.event_type == event_type]


# ---------------------------------------------------------------------------
# Flag defaults / env parsing
# ---------------------------------------------------------------------------


def test_multi_candidate_flag_defaults_on(monkeypatch):
    assert engine._multi_candidate_drafts_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS", "false")
    assert engine._multi_candidate_drafts_enabled() is False


def test_dev_eval_flag_defaults_on_and_disableable(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", raising=False)
    assert engine._dev_eval_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "false")
    assert engine._dev_eval_enabled() is False


def test_dev_plateau_flag_defaults_off_and_enableable(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP", raising=False)
    assert engine._dev_plateau_stop_enabled() is False
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP", "true")
    assert engine._dev_plateau_stop_enabled() is True


@pytest.mark.parametrize(
    "raw,expected",
    [("", 3), ("5", 5), ("0", 1), ("-2", 1), ("abc", 3)],
)
def test_drafts_per_call_limit_parsing(raw, expected, monkeypatch):
    if raw:
        monkeypatch.setenv("RESEARCH_LAB_LOOP_DRAFTS_PER_CALL", raw)
    assert engine._drafts_per_call_limit() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("", 2), ("1", 1), ("0", 1), ("x", 2)],
)
def test_dev_plateau_window_parsing(raw, expected, monkeypatch):
    if raw:
        monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_WINDOW", raw)
    assert engine._dev_plateau_window() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("", 0.5), ("1.25", 1.25), ("-1", 0.0), ("x", 0.5)],
)
def test_dev_plateau_min_delta_parsing(raw, expected, monkeypatch):
    if raw:
        monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_MIN_DELTA", raw)
    assert engine._dev_plateau_min_delta() == expected


# ---------------------------------------------------------------------------
# §6.2-8 multi-candidate drafts
# ---------------------------------------------------------------------------


def test_multi_candidate_parse_builds_n_with_cap_and_unique_keys(tmp_path, fake_boto3):
    response = _draft_response([_candidate_doc(0), _candidate_doc(1), _candidate_doc(2)])
    result, events, caller, builder = _run_engine(
        tmp_path,
        draft_responses=[response],
        max_candidates=2,
        manifest_uri=S3_MANIFEST_URI,
    )
    assert result.status == "completed"
    # The parse limit is min(remaining slots=2, drafts-per-call=3): two built, the
    # third candidate is never parsed/built — no build-and-discard.
    assert len(result.selected_candidates) == 2
    assert len(builder.build_calls) == 2
    diffs = [candidate.draft.unified_diff for candidate in result.selected_candidates]
    assert diffs == [_diff(0), _diff(1)]
    # One draft LLM call counted once (source inspection + draft = 2 calls total).
    assert caller.draft_call_count() == 1
    assert result.openrouter_call_count == 2
    # Unique node ids and per-build candidate indices (unique S3 diff keys, bug 20).
    node_ids = [candidate.node_id for candidate in result.selected_candidates]
    assert len(set(node_ids)) == 2
    assert [call["candidate_index"] for call in builder.build_calls] == [0, 1]
    # Unique S3 rehydration artifact keys, one per built candidate.
    rehydration_keys = [key for (_bucket, key) in fake_boto3 if "loop-candidates" in key]
    assert len(rehydration_keys) == len(set(rehydration_keys)) == 2
    assert len(_events_of(events, "candidate_build_passed")) == 2
    # Two per-draft reflections (one per build outcome), exactly once per draft.
    assert len(_events_of(events, "reflection_recorded")) == 2
    checkpoints = _events_of(events, "checkpoint_saved")
    assert checkpoints[-1].event_doc["checkpoint"]["built_candidate_count"] == 2


def test_multi_candidate_within_run_dedupe_applies_per_draft(tmp_path):
    # Candidate 0 is rejected by source validation (its diff hash is recorded);
    # candidate 1 repeats the same diff and must be dedupe-skipped before any
    # further spend; candidate 2 is distinct and builds.
    shared_diff = _diff(0)
    response = _draft_response(
        [
            _candidate_doc(0, diff=shared_diff, summary="REJECT me"),
            _candidate_doc(1, diff=shared_diff, summary="same diff again"),
            _candidate_doc(2),
        ]
    )
    result, events, _caller, builder = _run_engine(
        tmp_path,
        draft_responses=[response],
        max_candidates=3,
        validate_reject_marker="REJECT",
    )
    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].draft.unified_diff == _diff(2)
    assert [call["unified_diff"] for call in builder.build_calls] == [_diff(2)]
    dedupe_events = [
        event
        for event in _events_of(events, "code_edit_validation_failed")
        if event.event_doc.get("stage") == "within_run_duplicate_rejected_diff"
    ]
    assert len(dedupe_events) == 1
    assert dedupe_events[0].event_doc["unified_diff_hash"] == sha256_json(
        {"unified_diff": shared_diff}
    )


def test_multi_candidate_flag_off_single_draft_byte_compatible(tmp_path, monkeypatch):
    parse_limits: list[int] = []
    real_parse = engine.parse_code_edit_response

    def _recording_parse(raw, *, max_candidates=1):
        parse_limits.append(max_candidates)
        return real_parse(raw, max_candidates=max_candidates)

    monkeypatch.setattr(engine, "parse_code_edit_response", _recording_parse)
    response = _draft_response([_candidate_doc(0), _candidate_doc(1), _candidate_doc(2)])

    monkeypatch.setenv("RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS", "false")
    result_off, events_off, caller_off, builder_off = _run_engine(
        tmp_path, draft_responses=[response]
    )
    monkeypatch.setenv("RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS", "true")
    result_on, events_on, caller_on, builder_on = _run_engine(
        tmp_path, draft_responses=[response]
    )

    # At prod config (max_candidates=1) the flag is inert: same parse limit,
    # same single built draft (the first), same event stream.
    assert parse_limits == [1, 1]
    assert len(result_off.selected_candidates) == len(result_on.selected_candidates) == 1
    assert (
        result_off.selected_candidates[0].draft.unified_diff
        == result_on.selected_candidates[0].draft.unified_diff
        == _diff(0)
    )
    assert len(builder_off.build_calls) == len(builder_on.build_calls) == 1
    assert [event.event_type for event in events_off] == [
        event.event_type for event in events_on
    ]
    # The prompt asked for exactly one candidate both ways.
    for caller in (caller_off, caller_on):
        assert '"max_candidates":1' in caller.draft_message_texts()[0]


def test_multi_candidate_flag_off_preserves_flat_settings_limit(tmp_path, monkeypatch):
    parse_limits: list[int] = []
    real_parse = engine.parse_code_edit_response

    def _recording_parse(raw, *, max_candidates=1):
        parse_limits.append(max_candidates)
        return real_parse(raw, max_candidates=max_candidates)

    monkeypatch.setattr(engine, "parse_code_edit_response", _recording_parse)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS", "false")
    response = _draft_response([_candidate_doc(0), _candidate_doc(1), _candidate_doc(2)])
    result, _events, _caller, _builder = _run_engine(
        tmp_path, draft_responses=[response], max_candidates=2
    )
    # Legacy behavior: the parser gets settings.max_candidates flat.
    assert parse_limits == [2]
    assert len(result.selected_candidates) == 2


def test_multi_candidate_second_call_bounded_by_remaining_slots(tmp_path, monkeypatch):
    parse_limits: list[int] = []
    real_parse = engine.parse_code_edit_response

    def _recording_parse(raw, *, max_candidates=1):
        parse_limits.append(max_candidates)
        return real_parse(raw, max_candidates=max_candidates)

    monkeypatch.setattr(engine, "parse_code_edit_response", _recording_parse)
    first = _draft_response([_candidate_doc(0), _candidate_doc(1)])
    second = _draft_response([_candidate_doc(2), _candidate_doc(3)])
    result, _events, _caller, builder = _run_engine(
        tmp_path,
        draft_responses=[first, second],
        max_candidates=3,
        max_iterations=2,
    )
    # First call: min(3 remaining, 3 per-call) = 3; after two builds only one
    # slot remains, so the second call asks for and parses exactly 1.
    assert parse_limits == [3, 1]
    assert len(result.selected_candidates) == 3
    assert len(builder.build_calls) == 3
    assert result.stop_reason in {"candidate_limit_reached", "max_iterations"}


def test_drafts_per_call_env_bounds_parse_limit(tmp_path, monkeypatch):
    parse_limits: list[int] = []
    real_parse = engine.parse_code_edit_response

    def _recording_parse(raw, *, max_candidates=1):
        parse_limits.append(max_candidates)
        return real_parse(raw, max_candidates=max_candidates)

    monkeypatch.setattr(engine, "parse_code_edit_response", _recording_parse)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DRAFTS_PER_CALL", "1")
    response = _draft_response([_candidate_doc(0), _candidate_doc(1), _candidate_doc(2)])
    result, _events, _caller, _builder = _run_engine(
        tmp_path, draft_responses=[response], max_candidates=3
    )
    assert parse_limits == [1]
    assert len(result.selected_candidates) == 1


# ---------------------------------------------------------------------------
# §6.3-1 dev-eval: attachment, memory feed, events, checkpoints
# ---------------------------------------------------------------------------


def _enable_dev_eval(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_LAB_DEV_SNAPSHOT_URI", "s3://test-bucket/dev-snapshots")


def test_dev_score_attached_and_emitted(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    evaluator = _ScriptedDevEvaluator([41.5])
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    candidate = result.selected_candidates[0]
    assert candidate.dev_score == 41.5
    assert candidate.dev_score_version == DEV_SCORE_VERSION
    assert evaluator.calls == [candidate.node_id]
    build_passed = _events_of(events, "candidate_build_passed")[0]
    assert build_passed.event_doc["dev_score"] == 41.5
    assert build_passed.event_doc["dev_score_version"] == DEV_SCORE_VERSION
    assert build_passed.event_doc["dev_score_ranking_only"] is True
    selected_event = _events_of(events, "candidate_selected")[0]
    assert selected_event.event_doc["dev_score"] == 41.5
    checkpoint = _events_of(events, "checkpoint_saved")[-1].event_doc["checkpoint"]
    assert checkpoint["selected_candidates"][0]["dev_score"] == 41.5
    assert checkpoint["selected_candidates"][0]["dev_score_version"] == DEV_SCORE_VERSION


def test_dev_score_feeds_within_run_memory(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    evaluator = _ScriptedDevEvaluator([33.25, 12.0])
    result, _events, caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
        ],
        max_candidates=2,
        max_iterations=2,
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    draft_texts = caller.draft_message_texts()
    assert len(draft_texts) == 2
    # First draft call: no scores yet.
    assert "best_dev_score" not in draft_texts[0]
    # Second draft call sees the first candidate's ranking-only score.
    assert "dev_scores" in draft_texts[1]
    assert "best_dev_score" in draft_texts[1]
    assert "33.25" in draft_texts[1]


def test_dev_eval_off_keeps_docs_dev_free(tmp_path):
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=_ScriptedDevEvaluator([41.5]),
    )
    assert result.status == "completed"
    assert result.selected_candidates[0].dev_score is None
    serialized = json.dumps(
        [event.event_doc for event in events], default=str
    )
    assert "dev_score" not in serialized


def test_dev_eval_skipped_without_snapshot_uri(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "true")
    evaluator = _ScriptedDevEvaluator([41.5])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert result.selected_candidates[0].dev_score is None
    assert evaluator.calls == []


def test_dev_eval_unwired_seam_skips_silently(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=None,
    )
    assert result.status == "completed"
    assert result.selected_candidates[0].dev_score is None
    assert "dev_score" not in json.dumps(
        _events_of(events, "candidate_build_passed")[0].event_doc, default=str
    )


# ---------------------------------------------------------------------------
# §6.3-1 ranking (scored beat unscored; desc; tie stability)
# ---------------------------------------------------------------------------


def _built_candidate(index: int, dev_score: float | None = None) -> engine.BuiltCodeEditCandidate:
    draft = CodeEditDraft(
        failure_mode="weak sourcing recall",
        mechanism=f"mechanism {index}",
        expected_improvement="+2 companies per ICP",
        risk="slower sourcing",
        lane="provider",
        target_files=("sourcing_model/pipeline.py",),
        unified_diff=_diff(index),
        redacted_summary=f"widen fan-out {index}",
        test_plan="run adapter smoke",
        rollback_plan="revert diff",
        predicted_delta=1.5,
        plan_path_id="path-1",
        plan_alignment={},
    )
    build = CodeEditBuildResult(
        candidate_model_manifest=_manifest(S3_MANIFEST_URI),
        code_edit_manifest={"target_files": ["sourcing_model/pipeline.py"], "kind": "code_edit"},
        source_diff_hash=sha256_json({"unified_diff": draft.unified_diff}),
        build_doc={"build_doc_hash": "sha256:" + "1" * 64},
    )
    return engine.BuiltCodeEditCandidate(
        draft=draft,
        build=build,
        node_id=f"node:code-edit:{index:016d}",
        iteration=index + 1,
        dev_score=dev_score,
        dev_score_version=DEV_SCORE_VERSION if dev_score is not None else "",
    )


def test_rank_orders_scored_desc_then_unscored_in_build_order():
    unscored_a = _built_candidate(0)
    low = _built_candidate(1, dev_score=5.0)
    unscored_b = _built_candidate(2)
    high = _built_candidate(3, dev_score=9.0)
    ranked = engine._rank_selected_by_dev_score([unscored_a, low, unscored_b, high])
    assert [c.node_id for c in ranked] == [
        high.node_id,
        low.node_id,
        unscored_a.node_id,
        unscored_b.node_id,
    ]


def test_rank_is_stable_for_ties():
    tie_first = _built_candidate(0, dev_score=7.0)
    top = _built_candidate(1, dev_score=9.0)
    tie_second = _built_candidate(2, dev_score=7.0)
    ranked = engine._rank_selected_by_dev_score([tie_first, top, tie_second])
    assert [c.node_id for c in ranked] == [top.node_id, tie_first.node_id, tie_second.node_id]


def test_rank_noop_with_zero_scores():
    unscored = _built_candidate(0)
    trailing = _built_candidate(2)
    assert engine._rank_selected_by_dev_score([]) == []
    assert engine._rank_selected_by_dev_score([unscored, trailing]) == [unscored, trailing]


def test_rank_single_scored_outranks_unscored():
    # The per-iteration cap truncation keeps only the head of this list, so a
    # lone scored candidate must move ahead of earlier unscored builds — an
    # unscored build must never displace the only build with evidence.
    unscored = _built_candidate(0)
    scored = _built_candidate(1, dev_score=7.0)
    trailing = _built_candidate(2)
    ranked = engine._rank_selected_by_dev_score([unscored, scored, trailing])
    assert [c.node_id for c in ranked] == [
        scored.node_id,
        unscored.node_id,
        trailing.node_id,
    ]


def test_run_ranks_selected_by_dev_score_desc(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    evaluator = _ScriptedDevEvaluator([10.0, 50.0])
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
        ],
        max_candidates=2,
        max_iterations=2,
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert [candidate.dev_score for candidate in result.selected_candidates] == [50.0, 10.0]
    # candidate_selected emission order matches the ranked result order.
    selected_events = _events_of(events, "candidate_selected")
    assert [event.event_doc["dev_score"] for event in selected_events] == [50.0, 10.0]
    assert [event.event_doc["candidate_index"] for event in selected_events] == [0, 1]
    assert selected_events[0].node_id == result.selected_candidates[0].node_id


def test_run_scored_candidates_rank_ahead_of_unscored(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    # Second build's evaluation crashes → unscored; two other builds carry scores.
    evaluator = _ScriptedDevEvaluator([5.0, RuntimeError("dev harness down"), 7.0])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
        ],
        max_candidates=3,
        max_iterations=3,
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert [candidate.dev_score for candidate in result.selected_candidates] == [7.0, 5.0, None]
    assert result.selected_candidates[2].draft.unified_diff == _diff(1)


def test_run_cap_one_lone_scored_build_survives_truncation(tmp_path, monkeypatch):
    # Build-and-discard shape (cap=1, stop-at-cap off, min-runtime gates keep
    # the loop searching): the first build's evaluation fails (unscored), the
    # second scores. The per-iteration cap truncation must keep the scored
    # build — the run's only candidate with evidence — not the earlier
    # unscored one.
    _enable_dev_eval(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STOP_AT_CANDIDATE_CAP", "false")
    evaluator = _ScriptedDevEvaluator([RuntimeError("dev harness down"), 90.0])
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
        ],
        max_candidates=1,
        min_iterations=2,
        max_iterations=2,
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    # Both drafts were built before the cap truncation picked the finalist.
    assert len(_events_of(events, "candidate_build_passed")) == 2
    assert len(result.selected_candidates) == 1
    finalist = result.selected_candidates[0]
    assert finalist.dev_score == 90.0
    assert finalist.draft.unified_diff == _diff(1)


def test_run_cap_one_all_unscored_keeps_build_order(tmp_path, monkeypatch):
    # Same shape but every evaluation fails: with zero scored builds the
    # ranking must stay a no-op and the first build remains the finalist.
    _enable_dev_eval(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STOP_AT_CANDIDATE_CAP", "false")
    evaluator = _ScriptedDevEvaluator(
        [RuntimeError("dev harness down"), RuntimeError("dev harness down")]
    )
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
        ],
        max_candidates=1,
        min_iterations=2,
        max_iterations=2,
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert len(_events_of(events, "candidate_build_passed")) == 2
    finalist = result.selected_candidates[0]
    assert finalist.dev_score is None
    assert finalist.draft.unified_diff == _diff(0)


# ---------------------------------------------------------------------------
# Checkpoint / rehydration round-trip with and without dev fields
# ---------------------------------------------------------------------------


def _round_trip_draft() -> CodeEditDraft:
    return _built_candidate(0).draft


def _round_trip_build() -> CodeEditBuildResult:
    return _built_candidate(0).build


def test_rehydration_round_trip_with_dev_fields(fake_boto3):
    doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-1",
            node_id="node-dev",
            iteration=2,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
            dev_score=41.5,
            dev_score_version=DEV_SCORE_VERSION,
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    assert payload["dev_score"] == 41.5
    assert payload["dev_score_version"] == DEV_SCORE_VERSION
    candidate = engine._rehydrated_candidate_from_artifact_payload(payload)
    assert candidate.dev_score == 41.5
    assert candidate.dev_score_version == DEV_SCORE_VERSION
    assert doc["loop_candidate_artifact_hash"]


def test_rehydration_round_trip_without_dev_fields(fake_boto3):
    asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-1",
            node_id="node-plain",
            iteration=1,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    # Unscored candidates keep the exact pre-dev-eval payload shape.
    assert "dev_score" not in payload
    assert "dev_score_version" not in payload
    candidate = engine._rehydrated_candidate_from_artifact_payload(payload)
    assert candidate.dev_score is None
    assert candidate.dev_score_version == ""


def test_rehydration_hash_guard_covers_dev_fields(fake_boto3):
    asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-1",
            node_id="node-dev",
            iteration=1,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
            dev_score=41.5,
            dev_score_version=DEV_SCORE_VERSION,
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    payload["dev_score"] = 99.9
    with pytest.raises(ValueError):
        engine._rehydrated_candidate_from_artifact_payload(payload)


def test_restore_selected_from_resume_restores_dev_fields(fake_boto3):
    scored_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-1",
            node_id="node-dev",
            iteration=3,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
            dev_score=41.5,
            dev_score_version=DEV_SCORE_VERSION,
        )
    )
    plain_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-1",
            node_id="node-plain",
            iteration=1,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
        )
    )
    loop_engine = engine.CodeEditLoopEngine(
        settings=object(), call_openrouter=None, event_sink=None, builder=None
    )
    restored = asyncio.run(
        loop_engine._restore_selected_from_resume(
            resume={
                "selected_candidates": [
                    {
                        "node_id": "node-dev",
                        "rehydration_artifact_uri": scored_doc["loop_candidate_artifact_uri"],
                        "rehydration_artifact_hash": scored_doc["loop_candidate_artifact_hash"],
                        # Checkpoint summaries may carry dev fields; restore tolerates them.
                        "dev_score": 41.5,
                        "dev_score_version": DEV_SCORE_VERSION,
                    },
                    {
                        "node_id": "node-plain",
                        "rehydration_artifact_uri": plain_doc["loop_candidate_artifact_uri"],
                        "rehydration_artifact_hash": plain_doc["loop_candidate_artifact_hash"],
                    },
                ]
            },
            run_id="run-1",
            artifact=_manifest(S3_MANIFEST_URI),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert len(restored) == 2
    by_node = {candidate.node_id: candidate for candidate in restored}
    assert by_node["node-dev"].dev_score == 41.5
    assert by_node["node-dev"].dev_score_version == DEV_SCORE_VERSION
    assert by_node["node-plain"].dev_score is None


def test_resume_seeds_dev_scores_into_memory_and_ranking(tmp_path, fake_boto3, monkeypatch):
    _enable_dev_eval(monkeypatch)
    scored_a = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-mc-1",
            node_id="node-resume-a",
            iteration=1,
            draft=_round_trip_draft(),
            build=_round_trip_build(),
            dev_score=41.5,
            dev_score_version=DEV_SCORE_VERSION,
        )
    )
    scored_b = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(S3_MANIFEST_URI),
            run_id="run-mc-1",
            node_id="node-resume-b",
            iteration=2,
            draft=_built_candidate(1).draft,
            build=_built_candidate(1).build,
            dev_score=40.0,
            dev_score_version=DEV_SCORE_VERSION,
        )
    )
    resume_state = {
        "iterations_completed": 2,
        "built_candidate_count": 2,
        "selected_candidates": [
            {
                "node_id": "node-resume-a",
                "rehydration_artifact_uri": scored_a["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": scored_a["loop_candidate_artifact_hash"],
            },
            {
                "node_id": "node-resume-b",
                "rehydration_artifact_uri": scored_b["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": scored_b["loop_candidate_artifact_hash"],
            },
        ],
    }
    evaluator = _ScriptedDevEvaluator([50.0])
    result, events, caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(2)])],
        max_candidates=3,
        max_iterations=3,
        dev_evaluator=evaluator,
        manifest_uri=S3_MANIFEST_URI,
        resume_state=resume_state,
    )
    assert result.status == "completed"
    assert _events_of(events, "loop_resumed")
    # Restored scores were re-seeded into within-run memory before the new draft.
    first_draft_text = caller.draft_message_texts()[0]
    assert "best_dev_score" in first_draft_text
    assert "41.5" in first_draft_text
    # Final ranking spans restored + newly built candidates, dev-score desc.
    assert [candidate.dev_score for candidate in result.selected_candidates] == [
        50.0,
        41.5,
        40.0,
    ]
    assert result.selected_candidates[1].node_id == "node-resume-a"


# ---------------------------------------------------------------------------
# §6.3-4 lite plateau stop
# ---------------------------------------------------------------------------


def _enable_plateau(monkeypatch):
    _enable_dev_eval(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP", "true")


def test_plateau_stop_triggers_after_window_of_no_improvement(tmp_path, monkeypatch):
    _enable_plateau(monkeypatch)
    # Window 2, min delta 0.5 (defaults): 10.3 and 10.6 never beat the running
    # best by more than 0.5 → two consecutive stale scores → stop.
    evaluator = _ScriptedDevEvaluator([10.0, 10.3, 10.6, 99.0])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
            _draft_response([_candidate_doc(3)]),
        ],
        max_candidates=6,
        max_iterations=6,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "dev_score_plateau"
    assert result.iterations_completed == 3
    assert len(result.selected_candidates) == 3
    assert len(evaluator.calls) == 3


def test_plateau_boundary_improvement_of_exactly_delta_is_stale(tmp_path, monkeypatch):
    _enable_plateau(monkeypatch)
    # Improvements of exactly min_delta (0.5) do NOT reset the window: "no
    # improvement > MIN_DELTA" counts the boundary as plateau.
    evaluator = _ScriptedDevEvaluator([10.0, 10.5, 11.0, 99.0])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
            _draft_response([_candidate_doc(3)]),
        ],
        max_candidates=6,
        max_iterations=6,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "dev_score_plateau"
    assert result.iterations_completed == 3


def test_plateau_does_not_trigger_while_improving(tmp_path, monkeypatch):
    _enable_plateau(monkeypatch)
    evaluator = _ScriptedDevEvaluator([10.0, 10.6, 11.2])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
        ],
        max_candidates=6,
        max_iterations=3,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "max_iterations"
    assert result.iterations_completed == 3
    assert len(result.selected_candidates) == 3


def test_plateau_stale_then_improving_resets_window(tmp_path, monkeypatch):
    _enable_plateau(monkeypatch)
    # stale (10.2), improving (11.0 > 10.2 + 0.5), stale (11.1) → never two
    # consecutive stale scores → no plateau stop.
    evaluator = _ScriptedDevEvaluator([10.0, 10.2, 11.0, 11.1])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
            _draft_response([_candidate_doc(3)]),
        ],
        max_candidates=6,
        max_iterations=4,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "max_iterations"
    assert result.iterations_completed == 4


def test_plateau_flag_off_never_stops_early(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    # Plateaued scores, but RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP stays default off.
    evaluator = _ScriptedDevEvaluator([10.0, 10.0, 10.0])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
        ],
        max_candidates=6,
        max_iterations=3,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "max_iterations"
    assert result.iterations_completed == 3


def test_plateau_window_env_override(tmp_path, monkeypatch):
    _enable_plateau(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_PLATEAU_WINDOW", "1")
    evaluator = _ScriptedDevEvaluator([10.0, 10.1, 99.0])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[
            _draft_response([_candidate_doc(0)]),
            _draft_response([_candidate_doc(1)]),
            _draft_response([_candidate_doc(2)]),
        ],
        max_candidates=6,
        max_iterations=6,
        dev_evaluator=evaluator,
    )
    assert result.stop_reason == "dev_score_plateau"
    assert result.iterations_completed == 2


# ---------------------------------------------------------------------------
# Dev-eval failure containment
# ---------------------------------------------------------------------------


def test_dev_eval_exception_never_fails_the_run(tmp_path, monkeypatch):
    _enable_dev_eval(monkeypatch)
    evaluator = _ScriptedDevEvaluator([RuntimeError("replay store unreachable")])
    result, events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].dev_score is None
    assert len(_events_of(events, "candidate_build_passed")) == 1
    assert "dev_score" not in json.dumps(
        _events_of(events, "candidate_build_passed")[0].event_doc, default=str
    )


@pytest.mark.parametrize("outcome", ["garbage", None, float("nan")])
def test_dev_eval_malformed_result_leaves_candidate_unscored(tmp_path, monkeypatch, outcome):
    _enable_dev_eval(monkeypatch)
    evaluator = _ScriptedDevEvaluator([outcome])
    result, _events, _caller, _builder = _run_engine(
        tmp_path,
        draft_responses=[_draft_response([_candidate_doc(0)])],
        dev_evaluator=evaluator,
    )
    assert result.status == "completed"
    assert result.selected_candidates[0].dev_score is None
    assert result.selected_candidates[0].dev_score_version == ""


def test_maybe_dev_eval_helper_contains_evaluator_crash(monkeypatch):
    _enable_dev_eval(monkeypatch)

    async def _boom(candidate):
        raise ValueError("scripted failure")

    loop_engine = engine.CodeEditLoopEngine(
        settings=object(),
        call_openrouter=None,
        event_sink=None,
        builder=None,
        dev_evaluator=_boom,
    )
    candidate = _built_candidate(0)
    evaluated = asyncio.run(
        loop_engine._maybe_dev_eval_candidate(candidate, run_id="run-1")
    )
    assert evaluated == candidate
