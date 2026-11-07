"""Blank observations retain an independently verified first-party identity.

GoodData's archived identity is the fixture. Its historical raw LinkedIn
observation is unavailable; these whitespace cases prove a current handoff bug,
not the cause of its historical required-attribute result.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_company_evidence_investigator import _company, _finding, _icp


HOME = "https://www.gooddata.ai/"
PRODUCT = HOME + "platform/embedded-analytics/"
PRICING = HOME + "pricing/"
LINKEDIN = "https://www.linkedin.com/company/gooddata"
PRODUCT_QUOTE = "GoodData.AI provides analytics and dashboards."
COMMERCIAL_QUOTE = "Annual contracts include the operated Cloud SaaS offering."
IDENTITY = {
    "normalized_name": "gooddataai",
    "registrable_dns_domain": "gooddata.ai",
    "linkedin_company_slug": "gooddata",
    "verified_brand_name": "GoodData.AI",
    "verified_legal_name_aliases": ["GoodData Corporation"],
}
MISSING = object()


def _run(monkeypatch, raw_linkedin, *, identity=None, body=COMMERCIAL_QUOTE,
         support_url=PRICING, final_url=PRICING, status="VERIFIED"):
    company = _company(name="GoodData.AI", website=HOME, linkedin=LINKEDIN)
    attribute = "Sells subscription analytics software."
    verdict = {
        "observed_company_name": "GoodData.AI",
        "observed_company_website": HOME,
    }
    if raw_linkedin is not MISSING:
        verdict["observed_company_linkedin"] = raw_linkedin
    anchor = dict(IDENTITY if identity is None else identity)
    receipt = lead_scorer._web_identity_receipt(
        company, verdict, verified_homepage_identity=anchor,
        verified_homepage_transport_domain="gooddata.ai", company_quality=True,
    )
    captured = {}
    investigate = investigator.investigate_company_evidence

    async def capture(**kwargs):
        captured["handoff"] = kwargs["prior_observations"].get(
            "observed_company_linkedin",
        )
        captured["result"] = await investigate(**kwargs)
        raise RuntimeError("stop after native investigation")

    async def post(*_args, **_kwargs):
        finding = _finding(
            "required_attribute", status=status, evidence_url=PRODUCT,
            evidence_quote=PRODUCT_QUOTE, activity_role="supplier_operator",
            supporting_evidence_url_1=support_url,
            supporting_evidence_quote_1=COMMERCIAL_QUOTE,
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "offline", "type": "function", "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [finding]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", capture)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock(
        side_effect=AssertionError("All test sources are already fetched"),
    ))
    sources = {
        PRODUCT: {"status": "fetched", "final_url": PRODUCT, "text": PRODUCT_QUOTE},
        support_url: {"status": "fetched", "final_url": final_url, "text": body},
    }
    with pytest.raises(RuntimeError, match="stop after native investigation"):
        asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=_icp(required_attribute=attribute), verdict=verdict,
            investigation_targets=("required_attribute",), icp_attribute=attribute,
            icp_stage="", verified_identity=anchor,
            verified_transport_domain="gooddata.ai",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False, company_quality=True,
            review_positive_semantics=True,
            required_attribute_source_cache=sources,
        ))
    return receipt, captured


@pytest.mark.parametrize("blank", ["", " ", "\n\t", " \r\n "])
def test_blank_observation_keeps_verified_identity_and_exact_support(monkeypatch, blank):
    receipt, captured = _run(monkeypatch, blank)
    assert receipt["decision"] == "match"
    assert captured["handoff"] == LINKEDIN
    finding = captured["result"]["claims"]["required_attribute"]
    assert finding["status"] == "VERIFIED"
    assert finding["supporting_evidence"] == [{"url": PRICING, "quote": COMMERCIAL_QUOTE}]
    assert captured["result"]["usage"]["fetch_calls"] == 0


@pytest.mark.parametrize("invalid,expected_identity", [
    (MISSING, "unavailable"), (None, "unavailable"),
    ("not-a-url", "match"), ("https://www.linkedin.com/company/wrong", "unavailable"),
])
def test_absent_null_and_nonblank_observations_are_not_replaced(monkeypatch, invalid, expected_identity):
    receipt, captured = _run(monkeypatch, invalid)
    # Existing homepage-backed identity handling can accept a malformed model
    # locator. This change must still leave that nonblank observation intact.
    assert receipt["decision"] == expected_identity
    assert captured["handoff"] == (None if invalid is MISSING else invalid)
    assert captured["result"]["claims"]["required_attribute"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("identity", [{}, {**IDENTITY, "linkedin_company_slug": "wrong"},
                                      {**IDENTITY, "registrable_dns_domain": "other.example"}])
def test_blank_cannot_supply_missing_or_conflicting_homepage_proof(monkeypatch, identity):
    receipt, captured = _run(monkeypatch, " \n", identity=identity)
    assert receipt["decision"] != "match"
    assert captured["handoff"] == " \n"
    assert captured["result"]["claims"]["required_attribute"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("kwargs", [
    {"body": "The cited commercial sentence is absent."},
    {"support_url": "https://other.example/pricing/", "final_url": "https://other.example/pricing/"},
    {"final_url": "https://other.example/pricing/"},
])
def test_identity_handoff_does_not_replace_quote_or_source_proof(monkeypatch, kwargs):
    receipt, captured = _run(monkeypatch, " \n", **kwargs)
    assert receipt["decision"] == "match"
    assert captured["result"]["claims"]["required_attribute"]["status"] == "UNPROVEN"


def test_matching_nonblank_observation_is_unchanged(monkeypatch):
    receipt, captured = _run(monkeypatch, LINKEDIN + "/")
    assert receipt["decision"] == "match"
    assert captured["handoff"] == LINKEDIN + "/"
    assert captured["result"]["claims"]["required_attribute"]["status"] == "VERIFIED"


def test_blank_handoff_does_not_turn_an_unproven_finding_into_a_pass(monkeypatch):
    receipt, captured = _run(monkeypatch, " \n", status="UNPROVEN")
    assert receipt["decision"] == "match"
    assert captured["result"]["claims"]["required_attribute"]["status"] == "UNPROVEN"
