"""Pinning tests for research_lab/axis_provenance.py (trajectoryimprovements.md P11).

Every stage's emitter value is pinned so changing axis semantics is a
deliberate, reviewed decision — a silent flip here would poison Stage-3
curation filters (the ORO failure the field exists to prevent).
"""

from __future__ import annotations

import asyncio

from research_lab.axis_provenance import (
    AXIS_A,
    AXIS_B,
    STAGE_PROVENANCE,
    axis_rollup,
    call_episode,
    current_call_episode,
    episode_id,
    provenance_for_stage,
)


# stage → (call_emitter, teacher_model_flag) — the audited truth table.
PINNED_STAGE_VALUES = {
    "loop_planner": ("code", False),
    "plan_alignment_judge": ("code", True),
    "code_edit_draft": ("code", False),
    "code_edit_repair": ("code", False),
    "source_inspection": ("model", False),
    "loop_iteration": ("code", False),
    "scorer_judgment": ("code", True),
    "operator_repair": ("code", False),
    "incontainer_model_runtime": ("code", False),
}


def test_every_stage_value_is_pinned():
    assert set(STAGE_PROVENANCE) == set(PINNED_STAGE_VALUES)
    for stage, (emitter, teacher) in PINNED_STAGE_VALUES.items():
        provenance = provenance_for_stage(stage)
        assert provenance["call_emitter"] == emitter, stage
        assert provenance["teacher_model_flag"] is teacher, stage
        assert provenance["purpose"], stage
        assert provenance["component"], stage


def test_unknown_stage_defaults_conservative_code_emitter():
    provenance = provenance_for_stage("some_new_stage")
    assert provenance["call_emitter"] == "code"
    assert provenance["teacher_model_flag"] is False
    assert provenance["purpose"] == "some_new_stage"  # never empty


def test_source_inspection_is_the_only_model_emitted_stage():
    model_stages = [
        stage
        for stage, entry in STAGE_PROVENANCE.items()
        if entry["call_emitter"] == "model"
    ]
    assert model_stages == ["source_inspection"]


def test_incontainer_champion_pipeline_is_axis_b_by_construction():
    # v5 §8.3: the champion pipeline is axis-B; §8.6 trains on it directly.
    assert provenance_for_stage("incontainer_model_runtime")["call_emitter"] == "code"


def test_axis_rollup_conjunction_semantics():
    model_call = {"stage": "source_inspection", "call_emitter": "model"}
    code_call = {"stage": "code_edit_draft", "call_emitter": "code"}
    # All control-flow-driving calls model-emitted → axis-A.
    assert axis_rollup([model_call, model_call]) == AXIS_A
    # Mixed → axis-B (conjunction).
    assert axis_rollup([model_call, code_call]) == AXIS_B
    # Pure classifier pipeline → axis-B.
    assert axis_rollup([code_call]) == AXIS_B
    # Empty trace → axis-B, never a false axis-A.
    assert axis_rollup([]) == AXIS_B


def test_axis_rollup_ignores_non_control_flow_calls():
    # scorer judgments annotate; they do not drive control flow.
    scorer = {"stage": "scorer_judgment", "call_emitter": "code"}
    inspection = {"stage": "source_inspection", "call_emitter": "model"}
    assert axis_rollup([inspection, scorer]) == AXIS_A


def test_call_episode_scoping_and_id():
    assert current_call_episode() == {}
    with call_episode(run_id="run-1", iteration=3, inspection_round=2):
        episode = current_call_episode()
        assert episode == {"run_id": "run-1", "iteration": 3, "inspection_round": 2}
        assert episode_id(episode) == "run-1:i3:r2"
    assert current_call_episode() == {}


def test_call_episode_propagates_into_to_thread():
    """The worker records raw traces inside asyncio.to_thread — the episode
    contextvar must survive that hop."""

    async def _run() -> dict:
        with call_episode(run_id="run-2", iteration=1, inspection_round=1):
            return await asyncio.to_thread(current_call_episode)

    episode = asyncio.run(_run())
    assert episode["run_id"] == "run-2"
