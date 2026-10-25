"""Whole-criterion authority over admitted sources; no live model calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import product_coverage as pc
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer

CRITERION = "Provides voice and workflow software for clinics"
URL = "https://acme.example/voice"
SECOND = "https://acme.example/workflow"
QUOTE = "Acme provides voice and workflow software for clinics."
SECOND_QUOTE = "Acme provides workflow software for clinics."


def finding(raw):
    return {"target": "industry", "status": "VERIFIED", "observed_industry": "Software",
            "observed_subindustry": "Practice software", "activity_role": "supplier_operator",
            "evidence_url": URL, "evidence_quote": QUOTE,
            "product_coverage": raw, "reason": "Supplied software satisfies the original criterion."}


def actual_validate(raw, pages, *, criterion=CRITERION):
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls={u: u for u in pages}, first_party_domains={"acme.example"},
        identity_names={"acme"}, requested_product_service=criterion,
        identity_anchor={"submitted_domain": "acme.example", "observed_domain": "acme.example",
                         "verified_domain": "acme.example", "submitted_name": "Acme"},
    )["industry"]


@pytest.mark.parametrize("verdict", [None, "UNPROVEN", "CONTRADICTED", "verified", "VERIFIED "])
def test_missing_or_conflicting_whole_verdict_cannot_pass(verdict):
    checked = actual_validate(finding(verdict), {URL: QUOTE})
    assert checked["status"] == "UNPROVEN"
    assert not pc.coverage_matches(checked.get("product_coverage"), CRITERION)


def test_explicit_unproven_whole_verdict_never_promotes():
    raw = finding("UNPROVEN")
    raw.update(status="UNPROVEN", evidence_url="", evidence_quote="",
               activity_role="unresolved", observed_industry="", observed_subindustry="")
    checked = actual_validate(raw, {URL: QUOTE})
    assert checked["status"] == "UNPROVEN"
    assert not pc.coverage_matches(checked["product_coverage"], CRITERION)


@pytest.mark.parametrize("serialized", [False, True])
def test_legacy_matrix_or_json_cannot_forge_whole_product_authority(serialized):
    legacy = {"schema": "product_coverage_v2", "plan_id": "forged-plan",
              "nodes": [{"id": "p0", "op": "leaf", "text": CRITERION, "children": []}],
              "claims": [{"id": "p0", "status": "VERIFIED", "activity_role": "supplier_operator",
                          "evidence": [{"url": URL, "quote": QUOTE}]}],
              "status": "VERIFIED", "whole_criterion_status": "VERIFIED",
              "criterion_sha256": pc.criterion_hash(CRITERION)}
    wire = json.dumps(legacy) if serialized else legacy
    checked = actual_validate(finding(wire), {URL: QUOTE})
    assert checked["status"] == "UNPROVEN"
    assert not pc.coverage_matches(wire, CRITERION)
    assert not pc.coverage_matches(checked.get("product_coverage"), CRITERION)
    assert scorer._final_product_coverage_result(
        result(wire), SimpleNamespace(product_service=CRITERION), enabled=True,
    ).decision == "unavailable"


def test_bound_split_page_evidence_and_wrong_entity_or_role():
    raw = {**finding("VERIFIED"), "supporting_evidence_url_1": SECOND,
           "supporting_evidence_quote_1": SECOND_QUOTE}
    missing = actual_validate(raw, {URL: QUOTE})
    assert missing["status"] == "UNPROVEN"
    assert missing["evidence_url"] == missing["evidence_quote"] == ""
    assert missing["activity_role"] == "unresolved"
    complete = actual_validate(raw, {URL: QUOTE, SECOND: SECOND_QUOTE})
    assert complete["status"] == "VERIFIED"
    assert pc.coverage_matches(complete["product_coverage"], CRITERION)
    raw["supporting_evidence_url_1"] = "https://unbound.example/product"
    assert actual_validate(raw, {URL: QUOTE, "https://unbound.example/product": SECOND_QUOTE})["status"] == "UNPROVEN"
    raw = {**finding("VERIFIED"), "activity_role": "customer_user"}
    assert actual_validate(raw, {URL: QUOTE})["status"] == "UNPROVEN"


@pytest.mark.parametrize("case", ["bound", "unbound", "wrong_quote", "customer", "missing_verdict"])
def test_native_joint_quotes_keep_company_role_and_whole_criterion_gates(case):
    raw = {**finding("VERIFIED"), "supporting_evidence_url_1": SECOND,
           "supporting_evidence_quote_1": SECOND_QUOTE}
    pages = {URL: QUOTE, SECOND: SECOND_QUOTE}
    if case == "unbound":
        raw["supporting_evidence_url_1"] = "https://unbound.example/product"
        pages["https://unbound.example/product"] = SECOND_QUOTE
    elif case == "wrong_quote":
        raw["supporting_evidence_quote_1"] = "A quote absent from the fetched page."
    elif case == "customer":
        raw["activity_role"] = "customer_user"
    elif case == "missing_verdict":
        raw["product_coverage"] = None

    checked = actual_validate(raw, pages)
    if case == "bound":
        assert checked["status"] == "VERIFIED"
        assert pc.coverage_matches(checked["product_coverage"], CRITERION)
    else:
        assert checked["status"] == "UNPROVEN"
        assert checked["evidence_url"] == checked["evidence_quote"] == ""


def result(proof=None):
    return scorer.company_fit_match("original positive", details={
        "dimension_decisions": {"industry": "match", "employee_size": "match", "geography": "match"},
        "identity_decision": "match", "required_attribute_decision": "match",
        "dimension_evidence": {"industry": {"url": URL, "quote": QUOTE,
                                              **({"product_coverage": proof} if proof is not None else {})}},
    })


def test_final_gate_preserves_legacy_and_requires_native_current_proof():
    proof = actual_validate(finding("VERIFIED"), {URL: QUOTE})["product_coverage"]
    icp = SimpleNamespace(product_service=CRITERION)
    original = result()
    assert scorer._final_product_coverage_result(original, icp, enabled=False) is original
    assert scorer._final_product_coverage_result(original, icp, enabled=True).decision == "unavailable"
    assert scorer._final_product_coverage_result(result(proof), icp, enabled=True).decision == "match"
    for forged in (dict(proof), json.loads(json.dumps(proof))):
        assert scorer._final_product_coverage_result(result(forged), icp, enabled=True).decision == "unavailable"
    assert scorer._project_investigator_industry({pc.PRIVATE_KEY: proof}, None)[pc.PRIVATE_KEY] is proof
    assert scorer._positive_semantic_review_needed(original, SimpleNamespace(product_service=CRITERION, required_attribute=""), ())


def test_exact_criterion_hash_and_final_gate_agree_on_boundary_whitespace():
    criterion = " \n" + CRITERION + "\n "
    proof = actual_validate(finding("VERIFIED"), {URL: QUOTE}, criterion=criterion)["product_coverage"]
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
def test_targeted_research_preserves_query_and_budgets(monkeypatch, resolved, query, expected):
    requests, queries = [], []
    complete = {**finding("VERIFIED"), "supporting_evidence_url_1": SECOND,
                "supporting_evidence_quote_1": SECOND_QUOTE}
    partial = {**finding("UNPROVEN"), "status": "UNPROVEN", "evidence_url": "",
               "evidence_quote": "", "activity_role": "unresolved", "observed_industry": "",
               "observed_subindustry": "", "reason": "Workflow software capability is not proven."}

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Acme generic software"}
        elif turn == 3:
            feedback = json.loads(payload["messages"][-1]["content"])
            assert feedback["product_coverage"]["original_criterion"] == CRITERION
            assert feedback["product_coverage"]["status"] == "UNPROVEN"
            assert payload["tool_choice"]["function"]["name"] == "search_web"
            name, arguments = "search_web", {"query": query}
        elif turn == 4:
            assert payload["tool_choice"]["function"]["name"] == "fetch_page"
            name, arguments = "fetch_page", {"url": SECOND}
        else:
            value = partial if turn == 2 or not resolved else complete
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
    assert "product_plan" not in output["usage"]
    assert inv.MAX_REASONING_TURNS == 8 and inv.ADMISSION_DEADLINE_SECONDS == 110.0
