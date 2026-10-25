"""Offline accounting controls, not an accuracy claim about model semantics."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring.product_coverage import (
    CoverageError, PRIVATE_KEY, compile_plan, coverage_matches, validate_coverage,
)
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer


CRITERION = "Provides voice and workflow software for clinics"
URL = "https://acme.example/voice"
SECOND = "https://acme.example/workflow"
QUOTE = "Acme provides voice software for clinics."
SECOND_QUOTE = "Acme provides workflow software for clinics."


def node(i, span, op="leaf", children=()):
    return {"id": f"p{i}", "span": span, "op": op, "children": list(children)}


def matrix():
    nodes = [node(0, CRITERION, "all", ["p1", "p2", "p3", "p4"]),
             node(1, "Provides"), node(2, "voice"),
             node(3, "workflow software"), node(4, "for clinics")]
    return {"nodes": nodes, "claims": [claim(i) for i in range(1, 5)]}


def claim(i, status="VERIFIED", role="supplier_operator", url=URL, quote=QUOTE):
    return {"id": f"p{i}", "status": status, "activity_role": role,
            "evidence": [{"url": url, "quote": quote}] if status != "UNPROVEN" else []}


def validate(raw, criterion=CRITERION, **kwargs):
    return validate_coverage(raw, criterion=criterion,
                             citation_rejection=lambda u, q: None, **kwargs)


def test_missing_and_clause_cannot_pass_and_split_pages_can_complete_it():
    raw = matrix()
    raw["claims"][2] = claim(3, "UNPROVEN")
    proof = validate(raw)
    assert proof["status"] == "UNPROVEN" and proof["unresolved_ids"] == ["p3"]
    raw["claims"][2] = claim(3, url=SECOND, quote=SECOND_QUOTE)
    complete = validate(raw, locked_nodes=proof["nodes"])
    assert complete["status"] == "VERIFIED"
    assert {c["evidence"][0]["url"] for c in complete["claims"]} == {URL, SECOND}


@pytest.mark.parametrize("mutation", [
    lambda r: r["claims"].pop(),
    lambda r: r["claims"].append(claim(99)),
    lambda r: r["nodes"][2].update(span="invented capability"),
    lambda r: r.update(nodes=[node(0, CRITERION)], claims=[claim(0)]),
    lambda r: r["nodes"][0].update(children=["p1", "p2", "p4"]),
    lambda r: r["nodes"][0].update(op="any"),
])
def test_omission_forgery_and_whole_compound_leaf_are_rejected(mutation):
    raw = matrix()
    mutation(raw)
    with pytest.raises(CoverageError):
        validate(raw)


def test_repair_cannot_change_the_compiled_tree():
    raw = matrix()
    locked = compile_plan(CRITERION, raw)
    raw["nodes"][2]["span"] = "voice and workflow software"
    with pytest.raises(CoverageError):
        validate(raw, locked_nodes=locked)


@pytest.mark.parametrize("field,value", [("op", []), ("id", []), ("status", {}), ("activity_role", [])])
def test_malformed_inner_json_gets_scoped_validation_error(field, value):
    raw = matrix()
    (raw["nodes"][0] if field == "op" else raw["claims"][0])[field] = value
    with pytest.raises(CoverageError):
        validate(json.dumps(raw))


def alternatives():
    text = "Provides solar thermal or storage hardware for businesses"
    nodes = [node(0, text, "all", ["p1", "p2", "p3"]), node(1, "Provides"),
             node(2, "solar thermal or storage", "any", ["p4", "p5"]),
             node(3, "hardware for businesses"), node(4, "solar thermal"), node(5, "storage")]
    return text, {"nodes": nodes, "claims": [claim(i) for i in (1, 3, 4, 5)]}


def test_or_one_branch_and_all_shared_scope_and_contradiction_logic():
    text, raw = alternatives()
    raw["claims"][-1] = claim(5, "UNPROVEN")
    assert validate(raw, text)["status"] == "VERIFIED"
    raw["claims"][1] = claim(3, "UNPROVEN")
    assert validate(raw, text)["status"] == "UNPROVEN"
    raw["claims"][1] = claim(3)
    raw["claims"][2] = claim(4, "CONTRADICTED")
    assert validate(raw, text)["status"] == "UNPROVEN"
    raw["claims"][3] = claim(5, "CONTRADICTED")
    assert validate(raw, text)["status"] == "CONTRADICTED"


def test_support_product_requires_all_workflows_and_all_declared_channels():
    text = "Provides software to manage tickets, customer conversations, and account workflows across Slack, email, and chat"
    spans = ["Provides software to manage", "tickets", "customer conversations",
             "account workflows", "across Slack", "email", "chat"]
    raw = {"nodes": [node(0, text, "all", [f"p{i}" for i in range(1, 8)]),
                     *[node(i, span) for i, span in enumerate(spans, 1)]],
           "claims": [claim(i) for i in range(1, 8)]}
    assert validate(raw, text)["status"] == "VERIFIED"
    for missing in (4, 5, 6, 7):
        partial = json.loads(json.dumps(raw))
        partial["claims"][missing - 1] = claim(missing, "UNPROVEN")
        assert validate(partial, text)["status"] == "UNPROVEN"


@pytest.mark.parametrize("text, first, second", [
    ("Provides software or licenses hardware", "Provides software", "licenses hardware"),
    ("Software for banks or hardware", "Software for banks", "hardware"),
])
def test_valid_asymmetric_or_does_not_invent_shared_qualifiers(text, first, second):
    raw = {"nodes": [node(0, text, "any", ["p1", "p2"]), node(1, first), node(2, second)],
           "claims": [claim(1), claim(2, "UNPROVEN")]}
    assert validate(raw, text)["status"] == "VERIFIED"


def test_examples_are_annotations_but_cannot_swallow_trailing_relative_scope():
    text = "Provides workflow software such as scheduling and intake"
    raw = {"nodes": [node(0, text, "all", ["p1", "p2"]),
                     node(1, "Provides workflow software"),
                     node(2, "scheduling and intake", "example")], "claims": [claim(1)]}
    assert validate(raw, text)["status"] == "VERIFIED"
    text += " that serves clinics through partners"
    raw["nodes"][0]["span"] = text
    raw["nodes"][2]["span"] += " that serves clinics through partners"
    with pytest.raises(CoverageError):
        validate(raw, text)


def test_wrong_role_unbound_quote_and_json_matrix_have_no_pass_authority():
    raw = matrix()
    raw["claims"][1]["activity_role"] = "customer_user"
    assert validate(raw)["status"] == "UNPROVEN"
    rejected = validate_coverage(matrix(), criterion=CRITERION,
                                citation_rejection=lambda u, q: "wrong entity or absent exact quote")
    assert rejected["status"] == "UNPROVEN"
    proof = validate(matrix())
    assert coverage_matches(proof, CRITERION)
    assert not coverage_matches(json.loads(json.dumps(proof)), CRITERION)
    assert not coverage_matches(proof, CRITERION + " worldwide")


def finding(raw):
    return {"target": "industry", "status": "VERIFIED", "observed_industry": "Software",
            "observed_subindustry": "Practice software", "activity_role": "supplier_operator",
            "evidence_url": URL, "evidence_quote": QUOTE,
            "product_coverage": raw, "reason": "Direct supplied software."}


def actual_validate(raw, pages):
    return inv._validated_findings(
        {"findings": [finding(raw)]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls={u: u for u in pages}, first_party_domains={"acme.example"},
        identity_names={"acme"}, requested_product_service=CRITERION,
        identity_anchor={"submitted_domain": "acme.example", "observed_domain": "acme.example",
                         "verified_domain": "acme.example", "submitted_name": "Acme"},
    )["industry"]


def test_actual_industry_admission_uses_bound_multi_page_quotes_and_normalizes_failure():
    raw = matrix()
    raw["claims"][2] = claim(3, url=SECOND, quote=SECOND_QUOTE)
    missing = actual_validate(raw, {URL: QUOTE})
    assert missing["status"] == "UNPROVEN"
    assert missing["evidence_url"] == missing["evidence_quote"] == ""
    assert missing["activity_role"] == "unresolved"
    complete = actual_validate(raw, {URL: QUOTE, SECOND: SECOND_QUOTE})
    assert complete["status"] == "VERIFIED"
    assert coverage_matches(complete["product_coverage"], CRITERION)


def result(proof=None):
    return scorer.company_fit_match("original positive", details={
        "dimension_decisions": {"industry": "match", "employee_size": "match", "geography": "match"},
        "identity_decision": "match", "required_attribute_decision": "match",
        "dimension_evidence": {"industry": {"url": URL, "quote": QUOTE,
                                              **({"product_coverage": proof} if proof is not None else {})}},
    })


def test_final_gate_preserves_legacy_and_requires_host_proof_for_arena():
    icp = SimpleNamespace(product_service=CRITERION)
    original = result()
    assert scorer._final_product_coverage_result(original, icp, enabled=False) is original
    assert scorer._final_product_coverage_result(original, icp, enabled=True).decision == "unavailable"
    proof = validate(matrix())
    assert scorer._final_product_coverage_result(result(proof), icp, enabled=True).decision == "match"
    assert scorer._final_product_coverage_result(result(dict(proof)), icp, enabled=True).decision == "unavailable"
    assert scorer._final_product_coverage_result(original, SimpleNamespace(product_service=""), enabled=True) is original


def test_product_only_positive_uses_existing_review_route_before_final_gate():
    icp = SimpleNamespace(product_service=CRITERION, required_attribute="")
    original = result()
    assert scorer._positive_semantic_review_needed(original, icp, ())
    assert not scorer._positive_semantic_review_needed(
        original, SimpleNamespace(product_service="", required_attribute=""), (),
    )
    # Route selection sees the unchanged industry match, including an active
    # attribute positive. The final wrapper runs only after that route returns.
    assert scorer._positive_semantic_review_needed(
        original, SimpleNamespace(product_service=CRITERION, required_attribute="A"), (),
    )


def test_unrelated_projection_preserves_proof_and_final_wrapper_covers_early_returns():
    proof = validate(matrix())
    assert scorer._project_investigator_industry({PRIVATE_KEY: proof}, None)[PRIVATE_KEY] is proof
    calls = []

    @scorer._final_product_coverage_gate
    async def existing_route(company, icp, **kwargs):
        calls.append(kwargs)
        return result()

    icp = SimpleNamespace(product_service=CRITERION)
    assert asyncio.run(existing_route(None, icp, evidence_investigator=False,
                                     require_company_fit_dimensions=True)).decision == "match"
    assert asyncio.run(existing_route(None, icp, evidence_investigator=True,
                                     require_company_fit_dimensions=True)).decision == "unavailable"
    assert len(calls) == 2


@pytest.mark.parametrize("resolved", [True, False])
def test_existing_loop_targets_missing_ids_with_unchanged_call_limits(monkeypatch, resolved):
    requests, queries = [], []
    partial = matrix()
    partial["claims"][2] = claim(3, "UNPROVEN")
    complete = matrix()
    complete["claims"][2] = claim(3, url=SECOND, quote=SECOND_QUOTE)

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn in (1, 3):
            name, arguments = "search_web", {"query": "Acme generic software"}
        elif turn == 4:
            name, arguments = "fetch_page", {"url": SECOND}
        else:
            value = finding(partial if turn == 2 or not resolved else complete)
            value["product_coverage"] = json.dumps(value["product_coverage"], separators=(",", ":"))
            if turn == 2 or not resolved:
                value.update(status="UNPROVEN", evidence_url="", evidence_quote="",
                             activity_role="unresolved", reason="Incomplete supplied capability.")
            name, arguments = "submit_findings", {"findings": [value]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"c{turn}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def search(_session, query, *, key):
        queries.append(query)
        return {"results": [{"url": SECOND, "title": "Workflow software"}]}

    async def fetch(_session, url, **kwargs):
        return {"ok": True, "url": url, "final_url": url, "text": SECOND_QUOTE}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    output = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("industry",), requested_industry="Software", requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "Acme", "registrable_dns_domain": "acme.example"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))
    assert output.get("claims", {}).get("industry", {}).get("status") == (
        "VERIFIED" if resolved else "UNPROVEN"
    ), (output, len(requests), queries)
    assert output["_completed_submit"] is True
    assert queries == ["Acme generic software", "Acme workflow software"]
    assert output["usage"]["search_calls"] == 2
    assert output["usage"]["fetch_calls"] == 1
    assert output["usage"]["reasoning_turns"] == 5
    assert all(r["max_tokens"] == 3000 for r in requests)
    assert inv.MAX_REASONING_TURNS == 8
    request_document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert request_document["requested_product_service"] == CRITERION
    assert "product_coverage_instructions" not in request_document
    assert "Acme provides voice software for clinics." in str(request_document["prefetched_sources"])
