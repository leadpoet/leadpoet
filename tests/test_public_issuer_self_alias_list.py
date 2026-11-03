"""Issuer self-reference lists preserve exact quote and current-listing guards."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer as lead
from qualification.scoring.competition import _normalized_company, _normalized_icp


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
                     "ttd_issuer_self_alias_seq9.json").read_text())


def validate(findings, *, pages=None, name="The Trade Desk", domain="thetradedesk.com"):
    pages = FIXTURE["pages"] if pages is None else pages
    anchor = {"submitted_name": name, "submitted_domain": domain}
    if name == "The Trade Desk":
        anchor.update(
            submitted_linkedin_slug="the-trade-desk", observed_name="The Trade Desk, Inc.",
            observed_domain=domain, observed_linkedin_slug="the-trade-desk",
            verified_domain=domain, verified_name="thetradedesk",
            verified_linkedin_slug="the-trade-desk", verified_structured_alias_slug="",
        )
    return investigator._validated_findings(
        {"findings": findings}, targets=tuple(f["target"] for f in findings),
        fetched_pages={u: p["text"] for u, p in pages.items()},
        fetched_final_urls={u: p["final_url"] for u, p in pages.items()},
        first_party_domains={domain},
        identity_names={"".join(c for c in name.casefold() if c.isalnum())},
        identity_anchor=anchor,
        requested_stage="Public",
    )


def test_exact_captured_issuer_alias_list_validates_original_positive_packet():
    findings = validate(FIXTURE["findings"])
    assert {k: v["status"] for k, v in findings.items()} == {
        "industry": "VERIFIED", "stage": "VERIFIED", "required_attribute": "VERIFIED",
    }
    stage = findings["stage"]
    quote = stage["evidence_quote"]
    assert investigator._quote_occurs(quote, FIXTURE["pages"][stage["evidence_url"]]["text"])
    assert lead._stage_quote_supports_observation("public", quote)
    assert lead._public_quote_has_bound_market_locator(quote, ("The Trade Desk",))
    assert lead._stage_evidence_supports_observation(
        "public", quote, evidence_url=stage["evidence_url"],
        evidence_final_url=stage["evidence_url"],
        first_party_domains=("thetradedesk.com",), identity_names=("The Trade Desk",),
    )
    assert lead._decision_from_observed_stage(
        {"observed_company_stage": "Public", "stage_matches": True,
         "stage_evidence_url": stage["evidence_url"], "stage_evidence_quote": quote},
        "public", company=CompanyOutput(**_normalized_company(FIXTURE["company"])),
    ) == lead.COMPANY_FIT_MATCH


def test_original_packet_projects_to_full_fit_with_held_non_target_receipts(monkeypatch):
    """Hold actual non-target receipts; do not claim a complete live lead replay."""
    def no_fetch(*args, **kwargs):
        raise AssertionError("captured pages must suffice; no external fetch")

    monkeypatch.setattr(lead, "_fetch_bounded_html", no_fetch)
    findings = validate(FIXTURE["findings"])
    verdict = lead._project_investigator_industry(
        FIXTURE["prior_observations"], findings["industry"],
    )
    verdict = lead._project_investigator_stage(verdict, findings["stage"], icp_stage="public")
    verdict = lead._project_investigator_required_attribute(verdict, findings["required_attribute"])
    source_cache = {u: {"status": "fetched", **p} for u, p in FIXTURE["pages"].items()}
    verdict, _ = asyncio.run(lead._ground_required_attribute_evidence(
        verdict, active_attribute=True, source_cache=source_cache,
    ))
    icp = ICPPrompt(**_normalized_icp(FIXTURE["icp"]))
    result = lead._reverify_decision(
        verdict, icp.required_attribute, lead._normalize_company_stage(icp.company_stage),
        icp=icp, company=CompanyOutput(**_normalized_company(FIXTURE["company"])),
        verified_homepage_identity=FIXTURE["verified_homepage_identity"],
        verified_homepage_transport_domain="thetradedesk.com",
        structured_employee_size_evidence=FIXTURE["structured_employee_size_evidence"],
        validated_stage_finding=findings["stage"],
    )
    assert result.decision == lead.COMPANY_FIT_MATCH, result.details
    assert set(result.details["dimension_decisions"].values()) == {lead.COMPANY_FIT_MATCH}
    assert result.details["required_attribute_decision"] == lead.COMPANY_FIT_MATCH
    assert result.details["required_attribute_grounding"]["status"] == "grounded"


def test_actual_positive_packet_finishes_investigator_without_quote_repair(monkeypatch):
    # The native trajectory discovered this URL after the initial prefetch.
    dsp_url = "https://www.thetradedesk.com/our-demand-side-platform"
    fetch_response = {"choices": [{"message": {"tool_calls": [{
        "id": "captured-dsp-fetch", "type": "function", "function": {
            "name": "fetch_page", "arguments": json.dumps({"url": dsp_url}),
        },
    }]}}]}
    response = {"choices": [{"message": {"tool_calls": [{
        "id": "captured-seq9", "type": "function", "function": {
            "name": "submit_findings", "arguments": json.dumps({"findings": FIXTURE["findings"]}),
        },
    }]}}]}
    post = AsyncMock(side_effect=[(200, fetch_response), (200, response)])
    fetch = AsyncMock(return_value={"ok": True, "url": dsp_url, **FIXTURE["pages"][dsp_url]})
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator=FIXTURE["company_locator"],
        targets=("industry", "stage", "required_attribute"), requested_stage="Public",
        requested_industry=FIXTURE["icp"]["industry"],
        requested_product_service=FIXTURE["icp"]["product_service"],
        requested_attribute=FIXTURE["icp"]["required_attribute"],
        prior_observations=FIXTURE["prior_observations"],
        verified_homepage_identity=FIXTURE["verified_homepage_identity"],
        prefetched_pages=FIXTURE["pages"],
    ))
    assert {k: v["status"] for k, v in result["claims"].items()} == {
        "industry": "VERIFIED", "stage": "VERIFIED", "required_attribute": "VERIFIED",
    }
    assert result["_completed_submit"] is True
    assert post.await_count == 2  # Original fetch, then original submit; no repair.
    fetch.assert_awaited_once()
    assert fetch.await_args.args[1] == dsp_url


def stage_finding(quote, *, name="Acme", url="https://acme.example/investors", final_url=None):
    return validate([{
        "target": "stage", "status": "VERIFIED", "observed_value": "Public",
        "evidence_url": url, "evidence_quote": quote,
    }], pages={url: {"text": quote, "final_url": final_url or url}},
        name=name, domain="acme.example")["stage"]


@pytest.mark.parametrize("aliases", [
    '("Acme")', '("Acme" or the "Company")',
    '(“Acme,” the “Company” or “we”)', '("Acme", the "Company" or "we")',
    '(“Acme”, “Company”, “we”)', '("Acme" or "we")',
])
def test_only_explicit_self_reference_lists_are_ignored_before_bound_locator(aliases):
    quote = f"Acme, Inc. {aliases} (NASDAQ: ACM) announced current quarterly results."
    assert stage_finding(quote)["status"] == "VERIFIED"


@pytest.mark.parametrize("quote", [
    'OtherCo (“OtherCo,” the “Company” or “we”) (NASDAQ: OTHR) supplies Acme.',
    'Acme received funding from OtherCo (“OtherCo,” the “Company” or “we”) (NASDAQ: OTHR).',
    'NASDAQ: ACM 13.00 Pricing delayed by 20 minutes.',
    'Acme (“Acme,” the “Company” or “we”) (OTC: ACM) announced results.',
    'Acme (“Acme,” the “Company” or “we”) (NASDAQ: ACM) plans to list next year.',
    'Acme (“Acme,” the “Company” or “we”) (NASDAQ: ACM) was delisted yesterday.',
    'Acme (“Acme,” the “Company” or “we”) (NASDAQ: ACM) was taken private yesterday.',
    'Acme (“Acme,” the “Company” or “we”) (NASDAQ: ACM) completed its IPO in 2018.',
    'Acme (“formerly Acme,” the “Company” or “we”) (NASDAQ: ACM) announced results.',
    'Acme (“Acme,” the “OtherCo” or “we”) (NASDAQ: ACM) announced results.',
    'Acme (“Acme,” the “Company” or “we” or “OtherCo”) (NASDAQ: ACM) announced results.',
    'Acme (“Acme,” the “Company” or “we” or “Company”) (NASDAQ: ACM) announced results.',
    'Acme (“Acme,” the “Company” or “we”) (NASDAQ: 123) announced results.',
])
def test_alias_list_does_not_bypass_entity_market_or_chronology_guards(quote):
    assert stage_finding(quote)["status"] == "UNPROVEN"


def test_alias_list_keeps_existing_explicit_issuer_authority_on_other_publishers():
    # An explicit named issuer locator already works on third-party sources.
    quote = 'Acme (“Acme,” the “Company” or “we”) (NASDAQ: ACM) announced results.'
    assert stage_finding(quote, url="https://other.example/release")["status"] == "VERIFIED"
    foreign = quote.replace("Acme", "OtherCo")
    assert stage_finding(foreign, url="https://other.example/release")["status"] == "UNPROVEN"
    anonymous = 'Our company (“Acme,” the “Company” or “we”) (NASDAQ: ACM) announced results.'
    assert stage_finding(anonymous, final_url="https://other.example/release")["status"] == "UNPROVEN"


@pytest.mark.parametrize("alter", [
    lambda q: q.replace("buyers of advertising", "buyers of digital advertising"),
    lambda q: q.replace("a provider of a global technology platform ", ""),
    lambda q: q.replace("TTD", "OTHER"),
])
def test_exact_alias_support_does_not_make_altered_or_spliced_quotes_grounded(alter):
    original = next(f for f in FIXTURE["findings"] if f["target"] == "stage")
    assert validate([{**original, "evidence_quote": alter(original["evidence_quote"])}])["stage"]["status"] == "UNPROVEN"
