"""Offline integrity controls for frozen semantic product plans.

These controls check accounting and authority boundaries. They do not claim
that a model's semantic decomposition is accurate.
"""
from dataclasses import asdict

import pytest

from qualification.scoring import product_coverage as pc


CRITERION = "Acme provides clinic scheduling and either voice or chat intake"
MODEL = "offline-semantic-planner"
URL = "https://acme.example/products"
QUOTE = "Acme provides scheduling, voice intake, and chat intake for clinics."


def leaf(i, text):
    return {"id": (i if isinstance(i, str) else f"x{i}"),
            "op": "leaf", "text": text, "children": []}


def plan_for(nodes, *, criterion=CRITERION, model=MODEL):
    raw = {"root": "root", "nodes": [
        {"id": "root", "op": "all", "text": "Clinic scheduling and intake", "children": [n["id"] for n in nodes]},
        *nodes,
    ]}
    return pc.validate_semantic_plan(raw, criterion=criterion, model=model)


def matrix(plan, statuses):
    return {"plan_id": plan.plan_id, "claims": [
        {"id": node["id"], "status": statuses[node["id"]],
         "activity_role": "supplier_operator",
         "evidence": ([{"url": URL, "quote": QUOTE}]
                      if statuses[node["id"]] != "UNPROVEN" else [])}
        for node in plan.nodes if node["op"] == "leaf"
    ]}


def validate(raw, plan, *, criterion=CRITERION, whole="VERIFIED"):
    return pc.validate_coverage(raw, criterion=criterion, product_plan=plan,
                                whole_criterion_status=whole,
                                citation_rejection=lambda _url, _quote: None)


def test_and_with_one_unknown_cannot_pass():
    plan = plan_for([leaf(1, "Clinic scheduling"), leaf(2, "Intake")])
    result = validate(matrix(plan, {"p1": "VERIFIED", "p2": "UNPROVEN"}), plan,
                      whole="UNPROVEN")
    assert result["status"] == "UNPROVEN"
    assert "p2" in result["unresolved_ids"]


def test_or_success_does_not_hide_unknown_shared_qualifier():
    plan = pc.validate_semantic_plan({"root": "all", "nodes": [
        {"id": "all", "op": "all", "text": "Clinics get intake", "children": ["qual", "choice"]},
        leaf("qual", "For clinics"),
        {"id": "choice", "op": "any", "text": "Voice or chat intake", "children": ["voice", "chat"]},
        leaf("voice", "Voice intake"), leaf("chat", "Chat intake"),
    ]}, criterion=CRITERION, model=MODEL)
    result = validate(matrix(plan, {"p1": "UNPROVEN", "p3": "VERIFIED", "p4": "UNPROVEN"}),
                      plan, whole="UNPROVEN")
    assert result["status"] == "UNPROVEN"
    assert "p1" in result["unresolved_ids"]


def test_all_leaves_verified_still_requires_whole_criterion_verification():
    plan = plan_for([leaf(1, "Clinic scheduling"), leaf(2, "Intake")])
    result = validate(matrix(plan, {"p1": "VERIFIED", "p2": "VERIFIED"}), plan,
                      whole="UNPROVEN")
    assert result["status"] == "UNPROVEN"
    assert result["whole_criterion_gap"] == CRITERION


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_plan"])
def test_claim_id_and_plan_binding_fail_closed(mutation):
    plan = plan_for([leaf(1, "Clinic scheduling"), leaf(2, "Intake")])
    raw = matrix(plan, {"p1": "VERIFIED", "p2": "VERIFIED"})
    if mutation == "missing":
        raw["claims"].pop()
    elif mutation == "extra":
        raw["claims"].append({**raw["claims"][0], "id": "p99"})
    elif mutation == "duplicate":
        raw["claims"][1]["id"] = raw["claims"][0]["id"]
    else:
        raw["plan_id"] = "0" * 64
    with pytest.raises(pc.CoverageError):
        validate(raw, plan)


def test_serialized_plan_fields_cannot_forge_native_authority():
    plan = plan_for([leaf(1, "Clinic scheduling"), leaf(2, "Intake")])
    with pytest.raises(pc.CoverageError):
        validate(matrix(plan, {"p1": "VERIFIED", "p2": "VERIFIED"}), asdict(plan))


def test_nodes_property_returns_a_mutable_copy():
    plan = plan_for([leaf(1, "Clinic scheduling"), leaf(2, "Intake")])
    changed = plan.nodes
    changed[0]["text"] = "forged"
    changed.clear()
    assert plan.nodes[0]["text"] == "Clinic scheduling and intake"
    assert pc.plan_matches(plan, CRITERION)


def test_cache_is_exact_keyed_and_bounded(monkeypatch):
    cache = pc._PLAN_CACHE
    previous = cache.copy()
    cache.clear()
    monkeypatch.setattr(pc, "MAX_CACHED_PLANS", 2)
    try:
        leaves = [leaf(1, "Clinic scheduling"), leaf(2, "Intake")]
        first = plan_for(leaves)
        other_criterion = plan_for(leaves, criterion=CRITERION + ".")
        other_model = plan_for(leaves, model=MODEL + "-2")
        pc._store_trusted_plan(first)
        assert pc.get_cached_plan(CRITERION, model=MODEL) == first
        assert pc.get_cached_plan(CRITERION + " ", model=MODEL) is None
        assert pc.get_cached_plan(CRITERION, model=MODEL + "-2") is None
        monkeypatch.setattr(pc, "PLAN_VERSION", pc.PLAN_VERSION + "-next")
        assert pc.get_cached_plan(CRITERION, model=MODEL) is None
        monkeypatch.setattr(pc, "PLAN_VERSION", first.version)
        pc._store_trusted_plan(other_criterion)
        pc._store_trusted_plan(other_model)
        assert len(cache) == 2
        assert pc.get_cached_plan(CRITERION, model=MODEL) is None
    finally:
        cache.clear()
        cache.update(previous)
