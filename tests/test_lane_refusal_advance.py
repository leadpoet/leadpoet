"""Lane-refusal advance: after a source-grounded refusal the ranked-path
fallback prefers a different lane and never re-selects refused/attempted
paths; the kill switch restores the old re-ask behavior."""

from __future__ import annotations

from gateway.research_lab.code_loop_engine import (
    _ranked_path_fallback_plan,
    _refusal_lane_advance_enabled,
)


def _base_plan() -> dict:
    return {
        "schema_version": "1.0",
        "loop_goal": "improve sourcing recall",
        "required_lane": "source_routing",
        "required_mechanism": "retry exa then alternate surface",
        "selected_path_id": "p1",
        "ranked_paths": [
            {"path_id": "p1", "lane": "source_routing", "mechanism": "exa fallback"},
            {"path_id": "p2", "lane": "source_routing", "mechanism": "exa retry tuning"},
            {"path_id": "p3", "lane": "prompt_shaping", "mechanism": "tighten discovery prompt"},
        ],
    }


def test_fallback_prefers_different_lane_after_refusal():
    plan = _ranked_path_fallback_plan(
        _base_plan(),
        attempted_path_ids={"p1"},
        max_paths=3,
        fallback_index=1,
        refused_lanes={"source_routing"},
    )
    assert plan is not None
    # p2 shares the refused lane; p3 (different lane) must win.
    assert plan["selected_path_id"] == "p3"
    assert plan["required_lane"] == "prompt_shaping"


def test_fallback_same_lane_allowed_when_no_alternative():
    base = _base_plan()
    base["ranked_paths"] = [
        {"path_id": "p1", "lane": "source_routing", "mechanism": "exa fallback"},
        {"path_id": "p2", "lane": "source_routing", "mechanism": "exa retry tuning"},
    ]
    plan = _ranked_path_fallback_plan(
        base,
        attempted_path_ids={"p1"},
        max_paths=3,
        fallback_index=1,
        refused_lanes={"source_routing"},
    )
    assert plan is not None
    assert plan["selected_path_id"] == "p2"  # same lane is the only option left


def test_fallback_order_unchanged_without_refusals():
    plan = _ranked_path_fallback_plan(
        _base_plan(),
        attempted_path_ids={"p1"},
        max_paths=3,
        fallback_index=1,
    )
    assert plan is not None
    assert plan["selected_path_id"] == "p2"  # ranked order preserved


def test_fallback_never_reselects_attempted_paths():
    plan = _ranked_path_fallback_plan(
        _base_plan(),
        attempted_path_ids={"p1", "p2", "p3"},
        max_paths=3,
        fallback_index=2,
        refused_lanes={"source_routing"},
    )
    assert plan is None


def test_refusal_lane_advance_default_on(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_LOOP_REFUSAL_LANE_ADVANCE", raising=False)
    assert _refusal_lane_advance_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_LOOP_REFUSAL_LANE_ADVANCE", "false")
    assert _refusal_lane_advance_enabled() is False
