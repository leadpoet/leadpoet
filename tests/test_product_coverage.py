"""Frozen semantic obligations over admitted evidence; no live model calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import product_coverage as pc
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from tests.product_coverage_fixtures import coverage, trusted_plan

CRITERION = "Provides voice and workflow software for clinics"
URL = "https://acme.example/voice"
SECOND = "https://acme.example/workflow"
QUOTE = "Acme provides voice and workflow software for clinics."
SECOND_QUOTE = "Acme provides workflow software for clinics."
NODES = [
    {"id": "p0", "span": CRITERION, "op": "all", "children": ["p1", "p2", "p3"]},
    {"id": "p1", "span": "Provides voice software", "op": "leaf", "children": []},
    {"id": "p2", "span": "Provides workflow software", "op": "leaf", "children": []},
    {"id": "p3", "span": "For clinics", "op": "leaf", "children": []},
]


def matrix():
    return json.loads(coverage(CRITERION, URL, QUOTE, nodes=NODES))


def finding(raw):
    return {"target": "industry", "status": "VERIFIED", "observed_industry": "Software",
            "observed_subindustry": "Practice software", "activity_role": "supplier_operator",
            "evidence_url": URL, "evidence_quote": QUOTE,
            "product_coverage": raw, "reason": "Supplied software satisfies the original criterion."}


def validate(raw, *, whole="VERIFIED"):
    return pc.validate_coverage(raw, criterion=CRITERION,
        product_plan=pc.get_cached_plan(CRITERION, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL),
        whole_criterion_status=whole, citation_rejection=lambda u, q: None)


def test_missing_clause_and_whole_criterion_do_not_pass():
    raw = matrix()
    raw["claims"][1].update(status="UNPROVEN", evidence=[])
    assert validate(raw)["status"] == "UNPROVEN"
    complete = matrix()
    proof = validate(complete, whole="UNPROVEN")
    assert proof["status"] == "UNPROVEN" and "p0" in proof["unresolved_ids"]
    assert proof["whole_criterion_gap"] == CRITERION
    assert validate(complete)["status"] == "VERIFIED"


def actual_validate(raw, pages):
    return inv._validated_findings(
        {"findings": [finding(raw)]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls={u: u for u in pages}, first_party_domains={"acme.example"},
        identity_names={"acme"}, requested_product_service=CRITERION,
        product_plan=pc.get_cached_plan(CRITERION, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL),
        identity_anchor={"submitted_domain": "acme.example", "observed_domain": "acme.example",
                         "verified_domain": "acme.example", "submitted_name": "Acme"},
    )["industry"]


def test_bound_split_page_evidence_and_wrong_entity_or_role():
    raw = matrix()
    raw["claims"][1]["evidence"] = [{"url": SECOND, "quote": SECOND_QUOTE}]
    missing = actual_validate(raw, {URL: QUOTE})
    assert missing["status"] == "UNPROVEN"
    assert missing["evidence_url"] == missing["evidence_quote"] == ""
    assert missing["activity_role"] == "unresolved"
    complete = actual_validate(raw, {URL: QUOTE, SECOND: SECOND_QUOTE})
    assert complete["status"] == "VERIFIED"
    assert pc.coverage_matches(complete["product_coverage"], CRITERION)
    raw["claims"][1]["evidence"][0]["url"] = "https://unbound.example/product"
    assert actual_validate(raw, {URL: QUOTE, "https://unbound.example/product": SECOND_QUOTE})["status"] == "UNPROVEN"
    raw = matrix()
    raw["claims"][1]["activity_role"] = "customer_user"
    assert actual_validate(raw, {URL: QUOTE})["status"] == "UNPROVEN"


@pytest.mark.parametrize("case", ["bound", "unbound", "wrong_quote", "customer", "missing_clause"])
def test_native_joint_quotes_keep_company_role_and_whole_criterion_gates(case):
    raw = matrix()
    raw["claims"][1]["evidence"] = [
        {"url": URL, "quote": QUOTE},
        {"url": SECOND, "quote": SECOND_QUOTE},
    ]
    pages = {URL: QUOTE, SECOND: SECOND_QUOTE}
    if case == "unbound":
        raw["claims"][1]["evidence"][1]["url"] = "https://unbound.example/product"
        pages["https://unbound.example/product"] = SECOND_QUOTE
    elif case == "wrong_quote":
        raw["claims"][1]["evidence"][1]["quote"] = "A quote absent from the fetched page."
    elif case == "customer":
        raw["claims"][1]["activity_role"] = "customer_user"
    elif case == "missing_clause":
        raw["claims"][2].update(status="UNPROVEN", evidence=[])

    validated = actual_validate(raw, pages)
    if case == "bound":
        assert validated["status"] == "VERIFIED"
        assert pc.coverage_matches(validated["product_coverage"], CRITERION)
    else:
        assert validated["status"] == "UNPROVEN"
        assert validated["evidence_url"] == validated["evidence_quote"] == ""


def result(proof=None):
    return scorer.company_fit_match("original positive", details={
        "dimension_decisions": {"industry": "match", "employee_size": "match", "geography": "match"},
        "identity_decision": "match", "required_attribute_decision": "match",
        "dimension_evidence": {"industry": {"url": URL, "quote": QUOTE,
                                              **({"product_coverage": proof} if proof is not None else {})}},
    })


def test_final_gate_preserves_legacy_and_requires_native_current_proof():
    raw = matrix()
    proof = validate(raw)
    icp = SimpleNamespace(product_service=CRITERION)
    original = result()
    assert scorer._final_product_coverage_result(original, icp, enabled=False) is original
    assert scorer._final_product_coverage_result(original, icp, enabled=True).decision == "unavailable"
    assert scorer._final_product_coverage_result(result(proof), icp, enabled=True).decision == "match"
    assert scorer._final_product_coverage_result(result(json.loads(json.dumps(proof))), icp, enabled=True).decision == "unavailable"
    assert scorer._project_investigator_industry({pc.PRIVATE_KEY: proof}, None)[pc.PRIVATE_KEY] is proof
    assert scorer._positive_semantic_review_needed(original, SimpleNamespace(product_service=CRITERION, required_attribute=""), ())


def test_exact_criterion_cache_and_final_gate_agree_on_boundary_whitespace():
    criterion = " \n" + CRITERION + "\n "
    plan = trusted_plan(criterion, NODES)
    raw = json.loads(coverage(criterion, URL, QUOTE, nodes=NODES))
    proof = pc.validate_coverage(raw, criterion=criterion, product_plan=plan,
                               whole_criterion_status="VERIFIED", citation_rejection=lambda u, q: None)
    assert pc.coverage_matches(proof, criterion)
    assert not pc.coverage_matches(proof, CRITERION)
    assert scorer._final_product_coverage_result(result(proof), SimpleNamespace(product_service=criterion), enabled=True).decision == "match"


@pytest.mark.parametrize("resolved", [True, False])
@pytest.mark.parametrize("query,expected", [
    ("site:acme.example workflow orchestration", "site:acme.example workflow orchestration"),
    ("ACME workflow orchestration", "ACME workflow orchestration"),
    ("workflow orchestration", "Acme workflow orchestration"),
    ("site:notacme.example workflow", "Acme site:notacme.example workflow"),
    ("site:acme.example.evil workflow", "Acme site:acme.example.evil workflow"),
])
def test_frozen_claims_targeted_research_preserves_query_and_budgets(monkeypatch, resolved, query, expected):
    requests, queries = [], []
    complete = matrix()
    complete["claims"][1]["evidence"] = [{"url": SECOND, "quote": SECOND_QUOTE}]
    partial = matrix()
    partial["claims"][1].update(status="UNPROVEN", evidence=[])

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Acme generic software"}
        elif turn == 3:
            feedback = json.loads(payload["messages"][-1]["content"])
            assert {c["id"] for c in feedback["product_coverage"]["unresolved_clauses"]} == {"p0", "p2"}
            assert feedback["product_coverage"]["plan_id"] == complete["plan_id"]
            name, arguments = "search_web", {"query": query}
        elif turn == 4:
            name, arguments = "fetch_page", {"url": SECOND}
        else:
            value = finding(json.dumps(partial if turn == 2 or not resolved else complete))
            if turn == 2 or not resolved:
                value.update(status="UNPROVEN", evidence_url="", evidence_quote="", activity_role="unresolved")
            name, arguments = "submit_findings", {"findings": [value]}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": str(turn), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    async def search(_session, query, **kwargs):
        queries.append(query)
        return {"results": [{"url": SECOND, "title": "Workflow platform"}]}

    async def fetch(_session, url, **kwargs):
        return {"ok": True, "url": url, "final_url": url, "text": SECOND_QUOTE}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    output = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"}, targets=("industry",),
        requested_industry="Software", requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "Acme", "registrable_dns_domain": "acme.example"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))
    assert output["_completed_submit"] is True
    assert output["claims"]["industry"]["status"] == ("VERIFIED" if resolved else "UNPROVEN")
    assert queries == ["Acme generic software", expected]
    assert output["usage"]["reasoning_turns"] == 5
    assert output["usage"]["search_calls"] == 2 and output["usage"]["fetch_calls"] == 1
    assert output["usage"]["product_plan"]["cache_hit"] is True
    assert inv.MAX_REASONING_TURNS == 8 and inv.ADMISSION_DEADLINE_SECONDS == 110.0
