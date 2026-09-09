"""Tests for current OpenRouter telemetry provenance."""

from __future__ import annotations

from research_lab.axis_provenance import (
    STAGE_PROVENANCE,
    provenance_for_stage,
)


# stage → (call_emitter, teacher_model_flag) — the audited truth table.
PINNED_STAGE_VALUES = {
    "scorer_judgment": ("code", True),
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


def test_no_current_stage_is_marked_model_emitted():
    model_stages = [
        stage
        for stage, entry in STAGE_PROVENANCE.items()
        if entry["call_emitter"] == "model"
    ]
    assert model_stages == []
