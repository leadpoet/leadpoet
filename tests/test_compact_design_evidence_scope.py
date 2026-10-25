"""Prompt placement controls; offline checks cannot prove model interpretation."""

import asyncio

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv


def test_product_proof_instructions_preserve_design_intended_use_inference():
    prompt = " ".join(inv._PRODUCT_COVERAGE_INSTRUCTIONS.split())
    assert "For design/develop criteria, company-bound functional/performance facts" in prompt
    assert "intended use, target customers and unquantified performance/cost benefits by reasonable inference" in prompt
    assert "Do not turn a descriptive intended-use clause in a design/develop criterion into an extra shipment or named-customer requirement" in prompt
    assert "Do not add shipments, named deployments, public prices or benchmarks unless requested" in prompt
    assert prompt.index("return UNPROVEN") < prompt.index("For design/develop criteria")
    assert prompt.index("For design/develop criteria") < prompt.index("Search one weakest")


def test_design_inference_does_not_waive_capabilities_customer_scope_or_actual_facts():
    prompt = " ".join(inv._PRODUCT_COVERAGE_INSTRUCTIONS.split())
    assert "Still prove every core capability and customer/market restriction" in prompt
    assert "compatibility, enabling technology or adjacent products cannot fill a missing capability" in prompt
    assert "Explicit manufacture, sale, actual-use, deployment or milestone conditions need corresponding facts; never invent them" in prompt
    assert "at the proof level the criterion requests" in prompt
    assert "require actual use only when the criterion requires it" in prompt
    assert "Each citation must independently pass" in prompt
    assert "normal source, identity, role and quote checks" in prompt
    assert "If any condition lacks proof" in prompt
    assert "discovery highlights alone are not proof" in prompt


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("targets", [("industry",), ("industry", "required_attribute")])
def test_adjacent_design_rule_reaches_product_submit_schema(monkeypatch, positive, targets):
    requests = []

    async def post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        raise RuntimeError("offline tool-description capture complete")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=targets, requested_industry="Equipment",
        requested_product_service="Designs equipment for specialist laboratories",
        requested_attribute="Provides a commercial offering" if "required_attribute" in targets else "",
        positive_semantic_review=positive,
    ))
    assert len(requests) == 1
    submit = next(t["function"] for t in requests[0]["tools"]
                  if t["function"]["name"] == "submit_findings")
    field = submit["parameters"]["properties"]["findings"]["items"]["properties"]["product_coverage"]
    assert field["description"] == inv._PRODUCT_COVERAGE_INSTRUCTIONS
    assert "For design/develop criteria" in field["description"]
    assert "actual-use, deployment or milestone conditions" in field["description"]
    assert field["type"] == ["string", "null"]
    assert field["maxLength"] == 16
