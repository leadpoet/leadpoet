"""Tests for the §9 learning-flywheel wiring (fablefollowup 3.5 / 4.4 / 4.5).

Covers: the mechanical ``reflection_recorded`` emission in the code-edit lane
(default ON, pure capture, DB-CHECK/corpus-marker safe), the §9.5 lesson store
(retrieval over reflection events with staleness demotion, flag default OFF),
and the §9.4 allocator priors (results-ledger cells through the real
meta_allocator deterministic seeded Thompson selection with a uniform
exploration floor, flag default OFF). Includes a full CodeEditLoopEngine.run()
pass over a real ParentImageSourceContext to verify the injection points and
that lesson/prior fetch failures never fail a paid run.
"""

from __future__ import annotations

import asyncio
import json
import re
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from gateway.research_lab import allocator_priors, lesson_store
from gateway.research_lab import code_loop_engine as engine
from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    CodeEditImageBuildError,
    ParentImageSourceContext,
)
from gateway.research_lab.loop_engine import AutoResearchLoopSettings
from research_lab.code_editing import CodeEditDraft
from research_lab.engine_v1 import ReflectionRecord
from research_lab.eval import PrivateModelArtifactManifest
from research_lab.meta_allocator import validate_cell_yield_prior_record


PARENT_HASH = "sha256:" + "a" * 64

# The scripts/34 event_doc CHECK constraint, replicated verbatim: a reflection
# event whose doc matches this regex would fail the INSERT (and related markers
# poison the §8 trajectory-corpus record).
_EVENT_DOC_CHECK = re.compile(
    r"(?i)(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext)"
)


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


def _draft(**overrides: Any) -> CodeEditDraft:
    payload = dict(
        failure_mode="weak sourcing recall",
        mechanism="widen provider query fan-out",
        expected_improvement="+2 companies per ICP",
        risk="slower sourcing",
        lane="provider",
        target_files=("sourcing_model/pipeline.py",),
        unified_diff=(
            "diff --git a/sourcing_model/pipeline.py b/sourcing_model/pipeline.py\n"
            "--- a/sourcing_model/pipeline.py\n"
            "+++ b/sourcing_model/pipeline.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        ),
        redacted_summary="widen fan-out",
        test_plan="run adapter smoke",
        rollback_plan="revert diff",
        predicted_delta=1.5,
        plan_path_id="path-1",
        plan_alignment={},
    )
    payload.update(overrides)
    return CodeEditDraft(**payload)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_reflection_emission_flag_defaults_on(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_REFLECTION_EMISSION_ENABLED", raising=False)
    assert engine._reflection_emission_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_REFLECTION_EMISSION_ENABLED", "false")
    assert engine._reflection_emission_enabled() is False


def test_lesson_retrieval_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(lesson_store.LESSON_RETRIEVAL_ENABLED_ENV, raising=False)
    assert lesson_store.lesson_retrieval_enabled() is False
    monkeypatch.setenv(lesson_store.LESSON_RETRIEVAL_ENABLED_ENV, "true")
    assert lesson_store.lesson_retrieval_enabled() is True


def test_allocator_priors_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(allocator_priors.ALLOCATOR_PRIORS_ENABLED_ENV, raising=False)
    assert allocator_priors.allocator_priors_enabled() is False
    monkeypatch.setenv(allocator_priors.ALLOCATOR_PRIORS_ENABLED_ENV, "1")
    assert allocator_priors.allocator_priors_enabled() is True


# ---------------------------------------------------------------------------
# Reflection sanitization + mechanical reflection doc (3.5)
# ---------------------------------------------------------------------------


def test_reflection_safe_text_redacts_protected_markers():
    poisoned = "stage saw llm response and judge_prompt plus llm_response and page content"
    cleaned = engine._reflection_safe_text(poisoned)
    lowered = cleaned.lower()
    assert "llm response" not in lowered
    assert "llm_response" not in lowered
    assert "judge_prompt" not in lowered
    assert "page content" not in lowered
    assert "[protected-material-redacted]" in cleaned


def test_reflection_safe_text_keeps_clean_text_and_truncates():
    clean = engine._reflection_safe_text("hunk did not apply at sourcing_model/pipeline.py " + "y" * 500)
    assert clean.startswith("hunk did not apply")
    assert len(clean) <= 280
    assert engine._reflection_safe_text("failed auth with sk-or-abc123") == (
        "failed auth with [redacted-openrouter-key]"
    )


def test_mechanical_reflection_doc_build_passed_shape():
    doc = engine._mechanical_reflection_doc(
        run_id="run-1",
        node_id="node:code-edit:abc",
        iteration=2,
        outcome="candidate_build_passed",
        detail="image built and private tests passed",
        draft=_draft(),
        artifact=_manifest(),
        component_registry=_registry_doc(),
    )
    assert doc["outcome"] == "candidate_build_passed"
    assert doc["iteration"] == 2
    assert doc["reflection_source"] == "mechanical"
    assert doc["lane"] == "provider"
    assert doc["target_files"] == ["sourcing_model/pipeline.py"]
    reflection = doc["reflection"]
    for field in ("worked", "failed", "why", "next_question"):
        assert reflection[field]
    assert reflection["champion_base"] == PARENT_HASH
    assert reflection["component"] == "sourcing_model/pipeline.py"
    assert reflection["eval_version"] == "research-lab-private-evaluator:v1"
    assert reflection["engine_authored"] is True
    assert reflection["stale_basis"] is False
    # engine_v1.ReflectionRecord round-trip (build_reflection_record semantics).
    record = ReflectionRecord.from_mapping(reflection)
    assert record.lesson_id.startswith("lesson:")
    assert record.basis_patch_seq == 0


def test_mechanical_reflection_doc_is_event_doc_check_safe():
    doc = engine._mechanical_reflection_doc(
        run_id="run-1",
        node_id="node:code-edit:abc",
        iteration=1,
        outcome="candidate_image_build_failed",
        detail="docker pull denied for service_role image; judge_prompt leaked; llm response dumped",
        draft=_draft(),
        artifact=_manifest(),
        component_registry=_registry_doc(),
    )
    serialized = json.dumps(doc, default=str)
    assert not _EVENT_DOC_CHECK.search(serialized)
    assert doc["reflection"]["failed"]


def test_mechanical_reflection_doc_without_draft_uses_lane_fallback():
    doc = engine._mechanical_reflection_doc(
        run_id="run-1",
        node_id="",
        iteration=1,
        outcome="candidate_build_failed",
        detail="",
        draft=None,
        artifact=_manifest(),
        component_registry={},
    )
    assert doc["reflection"]["component"] == "code_edit"
    assert doc["reflection"]["eval_version"] == "unversioned"
    for field in ("worked", "failed", "why", "next_question"):
        assert doc["reflection"][field]


def test_emit_reflection_recorded_is_best_effort():
    async def _failing_sink(event):
        raise RuntimeError("event store down")

    loop_engine = engine.CodeEditLoopEngine(
        settings=object(), call_openrouter=None, event_sink=_failing_sink, builder=None
    )
    # Must swallow the sink failure entirely.
    asyncio.run(
        loop_engine._emit_reflection_recorded(
            run_id="run-1",
            node_id="node-1",
            iteration=1,
            draft=_draft(),
            outcome="candidate_build_passed",
            detail="ok",
            artifact=_manifest(),
            component_registry=_registry_doc(),
            elapsed=lambda: 1.0,
            openrouter_calls=1,
            estimated_cost=0.01,
            actual_cost_microusd=100,
        )
    )


def test_emit_reflection_recorded_respects_kill_switch(monkeypatch):
    events: list[Any] = []

    async def _sink(event):
        events.append(event)

    monkeypatch.setenv("RESEARCH_LAB_REFLECTION_EMISSION_ENABLED", "false")
    loop_engine = engine.CodeEditLoopEngine(
        settings=object(), call_openrouter=None, event_sink=_sink, builder=None
    )
    asyncio.run(
        loop_engine._emit_reflection_recorded(
            run_id="run-1",
            node_id="node-1",
            iteration=1,
            draft=_draft(),
            outcome="candidate_build_passed",
            detail="ok",
            artifact=_manifest(),
            component_registry=_registry_doc(),
            elapsed=lambda: 1.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert events == []


# ---------------------------------------------------------------------------
# Lesson store (4.5)
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal stand-in for gateway.research_lab.store select helpers."""

    def __init__(self, rows_by_table: Mapping[str, list[dict[str, Any]]]):
        self.rows_by_table = {name: list(rows) for name, rows in rows_by_table.items()}
        self.queries: list[tuple[str, Any]] = []

    async def select_many(self, table, *, columns="*", filters, order_by=(), limit=100):
        self.queries.append((table, list(filters)))
        rows = [dict(row) for row in self.rows_by_table.get(table, [])]
        for spec in filters:
            field, value = spec[0], spec[-1]
            rows = [row for row in rows if str(row.get(field)) == str(value)]
        for field, desc in reversed(list(order_by or ())):
            rows.sort(key=lambda row: str(row.get(field) or ""), reverse=bool(desc))
        return rows[:limit]


def _reflection_event_row(
    *,
    created_at: str,
    lane: str = "provider",
    component: str = "sourcing_model/pipeline.py",
    champion_base: str = PARENT_HASH,
    outcome: str = "candidate_build_failed",
    lesson_id: str | None = None,
    failed: str = "candidate build failed: hunk mismatch",
) -> dict[str, Any]:
    lesson_id = lesson_id or f"lesson:{created_at}"
    return {
        "event_id": f"event-{created_at}",
        "run_id": "run-1",
        "node_id": f"node-{created_at}",
        "event_type": "reflection_recorded",
        "created_at": created_at,
        "event_doc": {
            "schema_version": "1.0",
            "iteration": 1,
            "outcome": outcome,
            "reflection_source": "mechanical",
            "lane": lane,
            "target_files": [component],
            "reflection": {
                "lesson_id": lesson_id,
                "node_id": f"node-{created_at}",
                "worked": "draft parsed",
                "failed": failed,
                "why": "the build stage rejected the candidate",
                "next_question": "what minimal change avoids the failure?",
                "champion_base": champion_base,
                "component": component,
                "eval_version": "research-lab-private-evaluator:v1",
                "basis_patch_seq": 0,
                "stale_basis": False,
                "engine_authored": True,
                "contradicted_by": None,
            },
        },
    }


def test_fetch_recent_lessons_filters_and_demotes_stale():
    stale_hash = "sha256:" + "0" * 64
    store = _FakeStore(
        {
            lesson_store.LOOP_EVENTS_TABLE: [
                _reflection_event_row(created_at="2026-07-01T00:00:00Z", champion_base=stale_hash),
                _reflection_event_row(created_at="2026-07-02T00:00:00Z"),
                _reflection_event_row(created_at="2026-07-03T00:00:00Z", lane="intent"),
                {"event_type": "reflection_recorded", "event_doc": {"no_reflection": True}},
                {"event_type": "loop_started", "event_doc": {}},
            ]
        }
    )
    lessons = asyncio.run(
        lesson_store.fetch_recent_lessons(
            lane="provider",
            components=(),
            active_parent_hash=PARENT_HASH,
            limit=5,
            store=store,
        )
    )
    # lane=intent filtered out; fresh lesson ranks before the stale one.
    assert [lesson["stale_basis"] for lesson in lessons] == [False, True]
    assert lessons[0]["recorded_at"] == "2026-07-02T00:00:00Z"
    assert lessons[1]["champion_base"] == stale_hash
    # store was queried for reflection events only.
    assert store.queries[0][1] == [("event_type", "reflection_recorded")]


def test_fetch_recent_lessons_component_filter_and_limit():
    store = _FakeStore(
        {
            lesson_store.LOOP_EVENTS_TABLE: [
                _reflection_event_row(
                    created_at=f"2026-07-0{index}T00:00:00Z",
                    component=("sourcing_model/pipeline.py" if index % 2 else "gateway/api.py"),
                    lesson_id=f"lesson:{index}",
                )
                for index in range(1, 6)
            ]
        }
    )
    lessons = asyncio.run(
        lesson_store.fetch_recent_lessons(
            components=("sourcing_model/pipeline.py",),
            active_parent_hash=PARENT_HASH,
            limit=2,
            store=store,
        )
    )
    assert len(lessons) == 2
    assert all(lesson["component"] == "sourcing_model/pipeline.py" for lesson in lessons)
    # newest first
    assert lessons[0]["recorded_at"] > lessons[1]["recorded_at"]


def test_fetch_recent_lessons_sanitizes_poisoned_text():
    store = _FakeStore(
        {
            lesson_store.LOOP_EVENTS_TABLE: [
                _reflection_event_row(
                    created_at="2026-07-02T00:00:00Z",
                    failed="judge_prompt leaked with llm response body",
                )
            ]
        }
    )
    lessons = asyncio.run(lesson_store.fetch_recent_lessons(store=store))
    assert lessons
    lowered = lessons[0]["failed"].lower()
    assert "judge_prompt" not in lowered
    assert "llm response" not in lowered


def test_build_lesson_prompt_context_none_when_empty_and_caps_chars():
    store = _FakeStore({lesson_store.LOOP_EVENTS_TABLE: []})
    assert (
        asyncio.run(lesson_store.build_lesson_prompt_context(store=store)) is None
    )

    many = _FakeStore(
        {
            lesson_store.LOOP_EVENTS_TABLE: [
                _reflection_event_row(
                    created_at=f"2026-07-02T00:00:0{index}Z", lesson_id=f"lesson:{index}"
                )
                for index in range(5)
            ]
        }
    )
    doc = asyncio.run(
        lesson_store.build_lesson_prompt_context(
            active_parent_hash=PARENT_HASH, limit=5, max_chars=900, store=many
        )
    )
    assert doc is not None
    assert doc["schema_version"] == "1.0"
    assert 1 <= doc["lesson_count"] < 5
    assert len(json.dumps(doc, default=str)) <= 900 or doc["lesson_count"] == 1


# ---------------------------------------------------------------------------
# Allocator priors (4.4)
# ---------------------------------------------------------------------------


def _ledger_row(
    *,
    ledger_row_id: str,
    island: str = "generalist",
    lane: str = "provider",
    status: str = "discard",
    delta: float | None = None,
    cost_usd: float = 1.0,
    created_at: str = "2026-07-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "ledger_row_id": ledger_row_id,
        "island": island,
        "targeted_metric": "candidate_delta_vs_daily_baseline",
        "status": status,
        "delta_vs_parent": delta,
        "cost_usd": cost_usd,
        "description": (
            f"CODE_EDIT on {lane} targeted candidate_delta_vs_daily_baseline; "
            f"decision={status}; delta=n/a."
        ),
        "created_at": created_at,
    }


def _ledger_rows() -> list[dict[str, Any]]:
    rows = []
    # Cell A: provider lane, 5 keeps of 6 attempts (dominant yield).
    for index in range(6):
        rows.append(
            _ledger_row(
                ledger_row_id=f"aaaa000{index}-0000-4000-8000-00000000000{index}",
                lane="provider",
                status="keep" if index < 5 else "discard",
                delta=1.5 if index < 5 else -0.5,
            )
        )
    # Cell B: intent lane, 0 keeps of 6 attempts (crashes and discards).
    for index in range(6):
        rows.append(
            _ledger_row(
                ledger_row_id=f"bbbb000{index}-0000-4000-8000-00000000000{index}",
                lane="intent",
                status="crash" if index % 2 else "discard",
            )
        )
    return rows


def test_build_cell_yield_prior_records_aggregates_and_validates():
    priors, cells = allocator_priors.build_cell_yield_prior_records(_ledger_rows())
    assert len(priors) == 2
    for prior in priors:
        assert validate_cell_yield_prior_record(prior) == []
        assert prior.prior_id.startswith("cell_yield_prior:")
        assert prior.cell_ref.startswith("map_cell:generalist:")
        assert prior.patch_type == "CODE_EDIT"
        assert prior.uses_local_fixtures is False
        assert prior.results_ledger_input_ready is True
        assert all(ref.startswith("results_ledger:") for ref in prior.source_results_ledger_refs)
    by_lane = {prior.target_component: prior for prior in priors}
    provider = by_lane["provider"]
    assert provider.observed_attempts == 6
    assert provider.kept_patches == 5
    # Laplace keep-rate: (5+1)/(6+2)
    assert provider.posterior_mean_delta == pytest.approx(0.75)
    intent_stats = cells["map_cell:generalist:intent:candidate_delta_vs_daily_baseline"]
    assert intent_stats["crash"] == 3
    assert intent_stats["keep"] == 0


def test_build_cell_yield_priors_ranks_deterministically_with_floor():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    doc_one = asyncio.run(
        allocator_priors.build_cell_yield_priors(store=store, day="2026-07-02")
    )
    doc_two = asyncio.run(
        allocator_priors.build_cell_yield_priors(store=store, day="2026-07-02")
    )
    assert doc_one == doc_two  # deterministic seeded Thompson: workers agree
    assert doc_one is not None
    assert doc_one["schema_version"] == "1.0"
    assert doc_one["selection_id"].startswith("meta_allocator_selection:cell-yield:")
    assert doc_one["exploration_floor"] == pytest.approx(0.10)
    cells = doc_one["ranked_cells"]
    assert [cell["lane"] for cell in cells] == ["provider", "intent"]
    weights = [cell["weight"] for cell in cells]
    assert sum(weights) == pytest.approx(1.0, abs=1e-4)
    # Epsilon floor: no cell is ever zeroed out (uniform 10% exploration mass).
    floor_share = allocator_priors.EXPLORATION_FLOOR / len(cells)
    assert all(weight >= floor_share - 1e-9 for weight in weights)
    assert cells[0]["keep"] == 5
    # A different day derives a different seed (window unchanged).
    other_day = asyncio.run(
        allocator_priors.build_cell_yield_priors(store=store, day="2026-07-03")
    )
    assert other_day is not None
    assert other_day["seed"] != doc_one["seed"]


def test_build_cell_yield_priors_empty_ledger_returns_none():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: []})
    assert asyncio.run(allocator_priors.build_cell_yield_priors(store=store)) is None


def test_selection_seed_deterministic():
    seed = allocator_priors.selection_seed("2026-07-02", "sha256:" + "f" * 64)
    assert seed == allocator_priors.selection_seed("2026-07-02", "sha256:" + "f" * 64)
    assert seed != allocator_priors.selection_seed("2026-07-03", "sha256:" + "f" * 64)


def test_floored_weights_uniform_when_no_signal():
    weights = allocator_priors._floored_weights([0.0, 0.0], floor=0.1)
    assert weights == [pytest.approx(0.5), pytest.approx(0.5)]
    assert allocator_priors._floored_weights([]) == []


# ---------------------------------------------------------------------------
# Full-loop integration: emission + injection points (3.5 + 4.4 + 4.5 wiring)
# ---------------------------------------------------------------------------


_DRAFT_RESPONSE = json.dumps(
    {
        "candidates": [
            {
                "lane": "provider",
                "hypothesis": {
                    "failure_mode": "weak sourcing recall",
                    "mechanism": "widen provider query fan-out",
                    "expected_improvement": "+2 companies per ICP",
                    "risk": "slower sourcing",
                },
                "code_edit": {
                    "target_files": ["sourcing_model/pipeline.py"],
                    "unified_diff": (
                        "diff --git a/sourcing_model/pipeline.py b/sourcing_model/pipeline.py\n"
                        "--- a/sourcing_model/pipeline.py\n"
                        "+++ b/sourcing_model/pipeline.py\n"
                        "@@ -1 +1 @@\n"
                        "-x = 1\n"
                        "+x = 2\n"
                    ),
                    "redacted_summary": "widen fan-out",
                    "test_plan": "run adapter smoke",
                    "rollback_plan": "revert diff",
                },
            }
        ]
    }
)


class _FakeCaller:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    async def __call__(self, messages, timeout_seconds, max_tokens, stage):
        self.calls.append((stage, messages))
        if stage == "loop_planner":
            raise RuntimeError("planner unavailable in test")
        if stage == "source_inspection":
            return json.dumps(
                {"requests": [{"operation": "read_file", "path": "sourcing_model/pipeline.py"}]}
            )
        if stage == "code_edit_draft":
            return _DRAFT_RESPONSE
        raise AssertionError(f"unexpected stage: {stage}")

    def messages_for(self, stage: str) -> str:
        return json.dumps(
            [messages for called_stage, messages in self.calls if called_stage == stage],
            default=str,
        )


class _FakeBuilder:
    def __init__(self, source_context, *, build_error: Exception | None = None):
        self._source_context = source_context
        self._build_error = build_error
        self.config = types.SimpleNamespace(
            loop_planner_enabled=True,
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
        if self._build_error is not None:
            raise self._build_error
        return CodeEditBuildResult(
            candidate_model_manifest=_manifest(),
            code_edit_manifest={"target_files": list(draft.target_files), "kind": "code_edit"},
            source_diff_hash="sha256:" + "f" * 64,
            build_doc={"build_doc_hash": "sha256:" + "1" * 64},
        )


def _source_context(tmp_path: Path) -> ParentImageSourceContext:
    source_root = tmp_path / "source"
    (source_root / "sourcing_model").mkdir(parents=True)
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


def _run_engine(tmp_path: Path, *, build_error: Exception | None = None):
    caller = _FakeCaller()
    events: list[Any] = []

    async def _sink(event):
        events.append(event)

    loop_engine = engine.CodeEditLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=300,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=30,
            reflection_timeout_seconds=30,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=caller,
        event_sink=_sink,
        builder=_FakeBuilder(_source_context(tmp_path), build_error=build_error),
    )
    result = asyncio.run(
        loop_engine.run(
            run_id="run-flywheel-1",
            ticket={
                "ticket_id": "ticket-1",
                "island": "generalist",
                "miner_hotkey": "hotkey-1",
                "brief_sanitized_ref": "brief:1",
                "ticket_doc": {"brief_public_summary": "improve sourcing recall"},
            },
            artifact=_manifest(),
            component_registry=_registry_doc(),
            benchmark_public_summary={"aggregate": 12.0},
            model_id="test-model",
            budget_context={"requested_compute_budget_usd": 5.0},
            requested_loop_count=1,
        )
    )
    return result, events, caller


def test_run_emits_build_passed_reflection(tmp_path):
    result, events, _caller = _run_engine(tmp_path)
    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    reflections = [event for event in events if event.event_type == "reflection_recorded"]
    assert len(reflections) == 1
    event = reflections[0]
    assert event.node_id == result.selected_candidates[0].node_id
    doc = event.event_doc
    assert doc["outcome"] == "candidate_build_passed"
    assert doc["reflection"]["champion_base"] == PARENT_HASH
    assert doc["reflection"]["component"] == "sourcing_model/pipeline.py"
    for field in ("worked", "failed", "why", "next_question"):
        assert doc["reflection"][field]
    assert not _EVENT_DOC_CHECK.search(json.dumps(doc, default=str))


def test_run_emits_build_failure_reflection(tmp_path):
    result, events, _caller = _run_engine(
        tmp_path, build_error=CodeEditImageBuildError("docker build exited 1")
    )
    assert result.status == "failed"
    reflections = [event for event in events if event.event_type == "reflection_recorded"]
    assert len(reflections) == 1
    doc = reflections[0].event_doc
    assert doc["outcome"] == "candidate_image_build_failed"
    assert "docker build exited 1" in doc["reflection"]["failed"]
    assert not _EVENT_DOC_CHECK.search(json.dumps(doc, default=str))


def test_run_reflection_kill_switch_suppresses_emission(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_REFLECTION_EMISSION_ENABLED", "false")
    result, events, _caller = _run_engine(tmp_path)
    assert result.status == "completed"
    assert not [event for event in events if event.event_type == "reflection_recorded"]


def test_run_injects_lessons_and_priors_when_flags_on(tmp_path, monkeypatch):
    monkeypatch.setenv(lesson_store.LESSON_RETRIEVAL_ENABLED_ENV, "true")
    monkeypatch.setenv(allocator_priors.ALLOCATOR_PRIORS_ENABLED_ENV, "true")

    async def _fake_lessons(**kwargs):
        assert kwargs["active_parent_hash"] == PARENT_HASH
        return {
            "schema_version": "1.0",
            "note": "lessons",
            "lesson_count": 1,
            "lessons": [{"lesson_id": "lesson:test", "failed": "hunk mismatch", "stale_basis": False}],
        }

    async def _fake_priors(**kwargs):
        return {
            "schema_version": "1.0",
            "selection_id": "meta_allocator_selection:cell-yield:test",
            "ranked_cells": [{"cell_ref": "map_cell:generalist:provider:m", "weight": 0.9}],
        }

    monkeypatch.setattr(lesson_store, "build_lesson_prompt_context", _fake_lessons)
    monkeypatch.setattr(allocator_priors, "build_cell_yield_priors", _fake_priors)

    result, _events, caller = _run_engine(tmp_path)
    assert result.status == "completed"
    planner_messages = caller.messages_for("loop_planner")
    assert "retrieved_lessons" in planner_messages
    assert "cell_yield_priors" in planner_messages
    draft_messages = caller.messages_for("code_edit_draft")
    assert "retrieved_lessons" in draft_messages
    # Priors are a planner-context ordering hint only.
    assert "cell_yield_priors" not in draft_messages


def test_run_defaults_do_not_inject_lessons_or_priors(tmp_path, monkeypatch):
    monkeypatch.delenv(lesson_store.LESSON_RETRIEVAL_ENABLED_ENV, raising=False)
    monkeypatch.delenv(allocator_priors.ALLOCATOR_PRIORS_ENABLED_ENV, raising=False)
    result, _events, caller = _run_engine(tmp_path)
    assert result.status == "completed"
    all_messages = json.dumps([messages for _stage, messages in caller.calls], default=str)
    assert "retrieved_lessons" not in all_messages
    assert "cell_yield_priors" not in all_messages


def test_run_survives_lesson_and_prior_fetch_failures(tmp_path, monkeypatch):
    monkeypatch.setenv(lesson_store.LESSON_RETRIEVAL_ENABLED_ENV, "true")
    monkeypatch.setenv(allocator_priors.ALLOCATOR_PRIORS_ENABLED_ENV, "true")

    async def _boom(**kwargs):
        raise RuntimeError("store unreachable")

    monkeypatch.setattr(lesson_store, "build_lesson_prompt_context", _boom)
    monkeypatch.setattr(allocator_priors, "build_cell_yield_priors", _boom)

    result, events, caller = _run_engine(tmp_path)
    # A lesson/prior fetch failure must never fail a paid run.
    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    all_messages = json.dumps([messages for _stage, messages in caller.calls], default=str)
    assert "retrieved_lessons" not in all_messages
    assert "cell_yield_priors" not in all_messages
    assert [event for event in events if event.event_type == "reflection_recorded"]
