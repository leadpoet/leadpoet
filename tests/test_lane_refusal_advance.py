"""Lane-refusal advance: after a source-grounded refusal the ranked-path
fallback prefers a different lane and never re-selects refused/attempted
paths; the kill switch restores the old re-ask behavior."""

from __future__ import annotations

import types

from gateway.research_lab.code_loop_engine import (
    _candidate_edit_constraints,
    _plan_source_feasibility_errors,
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


def _complete_path(path_id, lane, mechanism, validation_mode="runtime_checks", validation_paths=None):
    return {
        "path_id": path_id,
        "lane": lane,
        "mechanism": mechanism,
        "target_behavior": [f"behavior for {path_id}"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": [lane],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": [f"avoid old {path_id} mechanism"],
        "success_criteria": [f"validate {path_id}"],
        "novelty_requirements": [f"novel {path_id}"],
        "anti_overfit_checks": [f"generalize {path_id}"],
        "generalization_claim": f"generalization {path_id}",
        "novelty_contrast": f"contrast {path_id}",
        "validation_mode": validation_mode,
        "validation_paths": list(validation_paths or []),
    }


def test_v1_1_ranked_fallback_replaces_all_path_local_fields():
    first = _complete_path("p1", "source_routing", "first mechanism")
    second = _complete_path("p2", "query_construction", "second mechanism")
    base = {
        "schema_version": "1.1",
        "required_lane": first["lane"],
        "required_mechanism": first["mechanism"],
        "selected_path_id": "p1",
        "ranked_paths": [first, second],
        **{key: value for key, value in first.items() if key not in {"path_id", "lane", "mechanism"}},
    }
    plan = _ranked_path_fallback_plan(
        base,
        attempted_path_ids={"p1"},
        max_paths=3,
        fallback_index=1,
    )
    assert plan is not None
    assert plan["selected_path_id"] == "p2"
    assert plan["required_lane"] == second["lane"]
    assert plan["required_mechanism"] == second["mechanism"]
    for key in (
        "target_behavior",
        "must_inspect",
        "allowed_lanes",
        "disallowed_lanes",
        "must_not_try",
        "success_criteria",
        "novelty_requirements",
        "anti_overfit_checks",
        "generalization_claim",
        "novelty_contrast",
        "validation_mode",
        "validation_paths",
    ):
        assert plan[key] == second[key]


def test_ranked_fallback_skips_malformed_legacy_alternative_without_inheritance():
    first = _complete_path("p1", "source_routing", "first mechanism")
    third = _complete_path("p3", "query_construction", "third mechanism")
    base = {
        "schema_version": "1.0",
        "required_lane": first["lane"],
        "required_mechanism": first["mechanism"],
        "selected_path_id": "p1",
        "ranked_paths": [
            first,
            {"path_id": "p2", "lane": "provider_fallback"},
            third,
        ],
        **{key: value for key, value in first.items() if key not in {"path_id", "lane", "mechanism"}},
    }
    plan = _ranked_path_fallback_plan(
        base,
        attempted_path_ids={"p1"},
        max_paths=3,
        fallback_index=1,
    )
    assert plan is not None
    assert plan["selected_path_id"] == "p3"
    assert plan["must_inspect"] == third["must_inspect"]


def test_candidate_constraints_do_not_advertise_missing_test_files():
    source_context = types.SimpleNamespace(
        editable_files=("sourcing_model/discovery.py", "gateway/tests/test_route.py")
    )
    constraints = _candidate_edit_constraints(
        source_context,
        config=types.SimpleNamespace(private_test_cmd="python -m smoke"),
        dev_evaluator_configured=True,
    )
    assert constraints["new_files_allowed"] is False
    assert constraints["editable_test_paths"] == ["gateway/tests/test_route.py"]
    assert constraints["allowed_validation_modes"] == ["runtime_checks", "existing_test_files"]
    assert "python -m smoke" not in str(constraints)


def test_plan_feasibility_rejects_unavailable_validation_path():
    path = _complete_path(
        "p1",
        "query_construction",
        "bounded query variant",
        validation_mode="existing_test_files",
        validation_paths=["tests/test_missing.py"],
    )
    plan = {
        "schema_version": "1.1",
        "required_lane": path["lane"],
        "required_mechanism": path["mechanism"],
        "selected_path_id": path["path_id"],
        "ranked_paths": [path],
        **{key: value for key, value in path.items() if key not in {"path_id", "lane", "mechanism"}},
    }
    source_context = types.SimpleNamespace(
        editable_files=("sourcing_model/discovery.py",),
        planner_source_index={
            "files": [{"path": "sourcing_model/discovery.py", "symbols": []}]
        },
    )
    errors, _missing = _plan_source_feasibility_errors(
        plan,
        source_context=source_context,
        candidate_edit_constraints={"editable_test_paths": []},
    )
    assert "loop_direction_plan_existing_tests_unavailable" in errors
    assert any(error.endswith("tests/test_missing.py") for error in errors)


def test_plan_feasibility_rejects_non_indexed_prose_reference():
    path = _complete_path("p1", "query_construction", "bounded query variant")
    path["must_inspect"] = ["inspect the discovery implementation"]
    plan = {
        "schema_version": "1.1",
        "required_lane": path["lane"],
        "required_mechanism": path["mechanism"],
        "selected_path_id": path["path_id"],
        "ranked_paths": [path],
        **{key: value for key, value in path.items() if key not in {"path_id", "lane", "mechanism"}},
    }
    source_context = types.SimpleNamespace(
        editable_files=("sourcing_model/discovery.py",),
        planner_source_index={
            "files": [{"path": "sourcing_model/discovery.py", "symbols": []}]
        },
    )
    errors, missing = _plan_source_feasibility_errors(
        plan,
        source_context=source_context,
        candidate_edit_constraints={"editable_test_paths": []},
    )
    assert "loop_direction_plan_reference_not_exact" in errors
    assert missing == ()
