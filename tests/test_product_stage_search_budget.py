"""One mandatory stage search must leave two product-review searches."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from tests.product_coverage_fixtures import coverage, trusted_plan


CRITERION = "Provides managed infrastructure and cloud migration"
NODES = [
    {"id": "p0", "span": CRITERION, "op": "all", "children": ["p1", "p2"]},
    {"id": "p1", "span": "Managed infrastructure", "op": "leaf", "children": []},
    {"id": "p2", "span": "Cloud migration", "op": "leaf", "children": []},
]


@pytest.mark.parametrize("stage,product,base_limit,expected_limit", [
    (True, True, 2, 3), (True, False, 2, 2), (False, True, 2, 2),
    (True, True, 0, 0),
])
def test_stage_discovery_and_product_searches_share_explicit_bound(
    monkeypatch, stage, product, base_limit, expected_limit,
):
    monkeypatch.setattr(inv, "MAX_SEARCH_CALLS", base_limit)
    if product:
        trusted_plan(CRITERION, NODES)
    requests, searches = [], []

    async def post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn <= 3:
            name, arguments = "search_web", {"query": f"Acme unresolved capability {turn}"}
        else:
            raw = {"target": "industry", "status": "UNPROVEN", "evidence_url": "",
                   "evidence_quote": "", "reason": "No loaded proof of the full criterion."}
            if product:
                wire = json.loads(coverage(CRITERION, "https://acme.example/product", "unused", nodes=NODES))
                for claim in wire["claims"]:
                    claim.update(status="UNPROVEN", activity_role="unresolved", evidence=[])
                raw["product_coverage"] = json.dumps(wire)
            findings = [raw]
            if stage:
                findings.insert(0, {"target": "stage", "status": "UNPROVEN", "observed_value": None,
                                    "evidence_url": "", "evidence_quote": "", "reason": "Current listing unproven."})
            name, arguments = "submit_findings", {"findings": findings}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": str(turn), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    async def search(_session, query, **kwargs):
        searches.append(query)
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=(("stage", "industry") if stage else ("industry",)),
        requested_stage="Public" if stage else "", requested_industry="IT services",
        requested_product_service=CRITERION if product else "",
    ))
    initial = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert "the stated search cap" in requests[0]["messages"][0]["content"]
    assert initial["investigation_limits"]["search_calls"] == expected_limit
    assert initial["investigation_limits"].get("remaining_search_calls", expected_limit) == (
        expected_limit - int(stage and base_limit > 0)
    )
    assert result["usage"]["search_calls"] == len(searches) == expected_limit
    assert result["usage"]["fetch_calls"] == 0
    assert len(requests) == 4
    last_tool = json.loads(requests[-1]["messages"][-1]["content"])
    assert last_tool["error"] == "search_budget_exhausted"
    if product:
        submit_tool = next(tool["function"] for tool in requests[0]["tools"]
                           if tool["function"]["name"] == "submit_findings")
        description = json.dumps(submit_tool["parameters"])
        assert "one weakest mandatory unresolved leaf" in description
        assert "prefer a specific" in description
        assert "uncaptured original-criterion gap" in description
        assert "never upgrade automatically" in description
        assert "Supplier/operator proof" in description
