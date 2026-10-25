"""Prompt transport and source guards; offline tests do not prove model semantics."""

import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring.product_coverage import coverage_matches


CRITERION = "Supplies workflow and reconciliation software used by merchants and institutions"
URL = "https://acme.example/services"
QUOTE = "Acme supplies workflow and reconciliation software to merchants and institutions."


def test_connected_evidence_does_not_invent_universal_customer_capability_scope():
    prompt = inv._PRODUCT_JOINT_EVIDENCE_SUFFIX
    assert "Prove every mandatory capability and customer relationship" in prompt
    assert "at the scope the criterion requests" in prompt
    assert "A list of capabilities and customer groups does not alone require separate proof" in prompt
    assert "every group uses every capability" in prompt
    assert "Preserve any requested link between a capability, user group, offering, or shared scope" in prompt
    assert "unless the criterion requires that shared scope" in prompt
    assert "contract or use case" in prompt
    assert "connected company-supplied portfolio permitted by the criterion" in prompt
    assert "An upstream supplier or partner alone does not establish" in prompt
    assert "a required customer or user relationship" in prompt
    assert "actual use by every" not in prompt


def test_quantifier_guidance_retains_mandatory_conditions_and_evidence_constraints():
    prompt = inv._PRODUCT_JOINT_EVIDENCE_SUFFIX
    assert "Do not combine disconnected offerings to invent a missing capability" in prompt
    assert "customer-owned capabilities" in prompt
    assert "including shared market and use" in prompt
    assert "every source/identity check still applies" in prompt
    assert "discovery highlights are locators only, never proof" in prompt
    coverage_prompt = inv._PRODUCT_COVERAGE_INSTRUCTIONS
    assert "Each citation must independently pass" in coverage_prompt
    assert "customer/market restriction and shared scope" in coverage_prompt
    assert "If any condition lacks proof" in coverage_prompt
    assert "return UNPROVEN" in coverage_prompt


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("product", [False, True])
def test_scope_guidance_reaches_only_product_review_system_messages(
    monkeypatch, positive, product,
):
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        raise RuntimeError("offline prompt capture complete")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        requested_product_service=CRITERION if product else "",
        positive_semantic_review=positive,
    ))
    assert len(requests) == 1
    system_messages = [message["content"] for message in requests[0]["messages"]
                       if message["role"] == "system"]
    assert system_messages == ([inv._SYSTEM_PROMPT, inv._PRODUCT_JOINT_EVIDENCE_SUFFIX]
                               if product else [inv._SYSTEM_PROMPT])


@pytest.mark.parametrize("invalid", [
    "valid", "unfetched", "fabricated_quote", "customer_role", "third_party_role",
    "unbound_source", "unbound_redirect",
])
def test_customer_scope_guidance_cannot_bypass_source_identity_or_role_guards(invalid):
    url = "https://unbound.example/services" if invalid == "unbound_source" else URL
    quote = "Acme supplies an invented service absent from this page." if invalid == "fabricated_quote" else QUOTE
    role = {"customer_role": "customer_user", "third_party_role": "third_party"}.get(
        invalid, "supplier_operator",
    )
    final_url = "https://unbound.example/services" if invalid == "unbound_redirect" else url
    claim = {
        "target": "industry", "status": "VERIFIED", "observed_industry": "Software",
        "observed_subindustry": "Workflow software", "activity_role": role,
        "evidence_url": url, "evidence_quote": quote, "product_coverage": "VERIFIED",
        "reason": "Model-declared judgment for source guard control only.",
    }
    result = inv._validated_findings(
        {"findings": [claim]}, targets=("industry",),
        fetched_pages={} if invalid == "unfetched" else {url: QUOTE},
        fetched_final_urls={url: final_url}, first_party_domains={"acme.example"},
        identity_names={"acme"}, requested_product_service=CRITERION,
        identity_anchor={"submitted_name": "Acme", "submitted_domain": "acme.example",
                         "observed_domain": "acme.example", "verified_domain": "acme.example"},
    )["industry"]
    if invalid == "valid":
        assert result["status"] == "VERIFIED"
        assert result["evidence_url"] == URL
        assert result["evidence_quote"] == QUOTE
        assert coverage_matches(result["product_coverage"], CRITERION)
    else:
        assert result["status"] == "UNPROVEN"
        assert result["evidence_url"] == result["evidence_quote"] == ""
        assert not coverage_matches(result.get("product_coverage"), CRITERION)
