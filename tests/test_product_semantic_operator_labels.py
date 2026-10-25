"""Operator labels are display text; leaf text carries plan obligations."""
import pytest

from qualification.scoring import product_coverage as pc


CRITERION = (
    "Designs specialized AI accelerator chips and server hardware that data centers "
    "and AI labs use to run large model inference faster and cheaper"
)
MODEL = "openai/gpt-6-luna"
CAPTURED_LEAVES = (
    (
        "Designs specialized AI accelerator chips.",
        "Designs server hardware.",
        "The hardware is for use by data centers and AI labs.",
        "Supports running inference for large models.",
        "Enables large-model inference to run faster.",
        "Enables large-model inference to run more cheaply.",
    ),
    (
        "Designs specialized AI accelerator chips.",
        "Designs server hardware.",
        "The chips and server hardware are for use by data centers and AI labs.",
        "Enables running inference for large models.",
        "Enables faster large-model inference.",
        "Enables lower-cost large-model inference.",
    ),
)


@pytest.mark.parametrize("leaves", CAPTURED_LEAVES)
def test_captured_empty_all_root_retains_six_obligations(leaves):
    # Physical planner responses 5793 and 5801 from the MatX canary.
    ids = [f"n{i}" for i in range(6)]
    raw = {"root": "r", "nodes": [
        {"id": "r", "op": "all", "text": "", "children": ids},
        *({"id": key, "op": "leaf", "text": value, "children": []}
          for key, value in zip(ids, leaves)),
    ]}
    plan = pc.validate_semantic_plan(raw, criterion=CRITERION, model=MODEL)
    assert [node["text"] for node in plan.nodes if node["op"] == "leaf"] == list(leaves)
    assert plan.nodes[0]["text"] == ""
    assert pc.plan_matches(plan, CRITERION)


@pytest.mark.parametrize("text", ["", "  \t  "])
def test_empty_or_whitespace_leaf_still_rejected(text):
    raw = {"root": "r", "nodes": [{"id": "r", "op": "leaf", "text": text, "children": []}]}
    with pytest.raises(pc.CoverageError, match="invalid semantic product node"):
        pc.validate_semantic_plan(raw, criterion=CRITERION, model=MODEL)


def test_one_leaf_root_remains_valid():
    raw = {"root": "r", "nodes": [{"id": "r", "op": "leaf", "text": "Designs server hardware.", "children": []}]}
    assert pc.validate_semantic_plan(raw, criterion=CRITERION, model=MODEL).nodes[0]["text"] == "Designs server hardware."


def test_nested_any_can_have_empty_operator_label_without_losing_leaves():
    raw = {"root": "r", "nodes": [
        {"id": "r", "op": "all", "text": " ", "children": ["qualifier", "alternatives"]},
        {"id": "qualifier", "op": "leaf", "text": "Used by data centers.", "children": []},
        {"id": "alternatives", "op": "any", "text": "", "children": ["chips", "servers"]},
        {"id": "chips", "op": "leaf", "text": "Designs accelerator chips.", "children": []},
        {"id": "servers", "op": "leaf", "text": "Designs server hardware.", "children": []},
    ]}
    plan = pc.validate_semantic_plan(raw, criterion=CRITERION, model=MODEL)
    assert [node["op"] for node in plan.nodes] == ["all", "leaf", "any", "leaf", "leaf"]
    assert [node["text"] for node in plan.nodes] == [
        "", "Used by data centers.", "", "Designs accelerator chips.", "Designs server hardware.",
    ]


def test_operator_only_scope_cannot_replace_whole_criterion_judgment():
    # The Grid Dynamics retry at physical request 5810 put this scope only in
    # the ALL display label. The host cannot infer missing semantics from text.
    criterion = (
        "Provides managed IT infrastructure, cloud migration, and mainframe modernization "
        "services that large enterprises outsource to run and transform their technology environments"
    )
    raw = {"root": "r", "nodes": [
        {"id": "r", "op": "all", "text": "For large enterprises outsourcing operation and transformation.",
         "children": ["infra", "cloud", "mainframe"]},
        {"id": "infra", "op": "leaf", "text": "Provides managed IT infrastructure services.", "children": []},
        {"id": "cloud", "op": "leaf", "text": "Provides cloud migration services.", "children": []},
        {"id": "mainframe", "op": "leaf", "text": "Provides mainframe modernization services.", "children": []},
    ]}
    plan = pc.validate_semantic_plan(raw, criterion=criterion, model=MODEL)
    wire = {"plan_id": plan.plan_id, "claims": [
        {"id": node["id"], "status": "VERIFIED", "activity_role": "supplier_operator",
         "evidence": [{"url": "https://example.test/products", "quote": "Example supplies the service."}]}
        for node in plan.nodes if node["op"] == "leaf"
    ]}
    result = pc.validate_coverage(
        wire, criterion=criterion, product_plan=plan, whole_criterion_status="UNPROVEN",
        citation_rejection=lambda _url, _quote: None,
    )
    assert result["status"] == "UNPROVEN"
    assert result["whole_criterion_gap"] == criterion
