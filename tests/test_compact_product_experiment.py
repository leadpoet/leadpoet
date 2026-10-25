"""Offline controls for the experimental single-verdict product route."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.product_coverage import (
    ValidatedWholeProductCoverage, coverage_matches,
)


CRITERION = "Provides voice and workflow software for clinics"
PRIMARY = "https://acme.example/voice"
SUPPORT = "https://acme.example/workflow"
PRIMARY_QUOTE = "Acme supplies voice software for clinics."
SUPPORT_QUOTE = "Acme supplies workflow software for clinics."
ANCHOR = {"submitted_domain": "acme.example", "observed_domain": "acme.example",
          "verified_domain": "acme.example", "submitted_name": "Acme"}


def _finding(*, status="VERIFIED", product="VERIFIED", role="supplier_operator",
             support_url=SUPPORT, support_quote=SUPPORT_QUOTE):
    return {
        "target": "industry", "status": status, "product_coverage": product,
        "observed_value": None, "observed_country": "", "observed_state": "",
        "observed_industry": "Software" if status != "UNPROVEN" else "",
        "observed_subindustry": "Practice software" if status != "UNPROVEN" else "",
        "activity_role": role, "evidence_url": PRIMARY if status != "UNPROVEN" else "",
        "evidence_quote": PRIMARY_QUOTE if status != "UNPROVEN" else "",
        "supporting_evidence_url_1": support_url,
        "supporting_evidence_quote_1": support_quote,
        "supporting_evidence_url_2": "", "supporting_evidence_quote_2": "",
        "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "Acme supplies both required capabilities to clinics." if status == "VERIFIED"
                  else "The original criterion remains unproven.",
    }


def _validate(raw, *, pages=None, finals=None):
    pages = pages if pages is not None else {PRIMARY: PRIMARY_QUOTE, SUPPORT: SUPPORT_QUOTE}
    finals = finals if finals is not None else {url: url for url in pages}
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls=finals, first_party_domains={"acme.example"},
        identity_names={"acme"}, identity_anchor=ANCHOR,
        requested_product_service=CRITERION,
    )["industry"]


def test_native_whole_verdict_uses_two_independently_bound_quotes():
    finding = _validate(_finding())
    proof = finding["product_coverage"]
    assert finding["status"] == "VERIFIED"
    assert isinstance(proof, ValidatedWholeProductCoverage)
    assert [part["url"] for part in proof["evidence"]] == [PRIMARY, SUPPORT]
    assert coverage_matches(proof, CRITERION)
    assert not coverage_matches(proof, CRITERION + " for hospitals")


def test_missing_industry_observation_keeps_precise_correction_reason():
    raw = _finding()
    raw["observed_industry"] = ""
    finding = _validate(raw)
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == inv.MISSING_INDUSTRY_OBSERVATION_REASON
    assert finding["product_coverage_error"] == inv.MISSING_INDUSTRY_OBSERVATION_REASON
    assert not coverage_matches(finding.get("product_coverage"), CRITERION)
    assert not finding.get("product_positive_citation_rejected")


@pytest.mark.parametrize("case", [
    "unknown_source", "quote_absent", "other_entity", "redirected_final",
    "redirected_primary", "partial_pair", "customer_role", "missing_verdict",
    "conflicting_verdict",
])
def test_native_whole_verdict_rejects_bad_source_or_protocol(case):
    raw = _finding()
    pages = {PRIMARY: PRIMARY_QUOTE, SUPPORT: SUPPORT_QUOTE}
    finals = {url: url for url in pages}
    if case == "unknown_source":
        raw["supporting_evidence_url_1"] = "https://acme.example/unfetched"
    elif case == "quote_absent":
        raw["supporting_evidence_quote_1"] = "Acme supplies an absent service."
    elif case == "other_entity":
        raw["supporting_evidence_url_1"] = "https://other.example/workflow"
        raw["supporting_evidence_quote_1"] = "Other supplies workflow software for clinics."
        pages[raw["supporting_evidence_url_1"]] = raw["supporting_evidence_quote_1"]
        finals[raw["supporting_evidence_url_1"]] = raw["supporting_evidence_url_1"]
    elif case == "redirected_final":
        finals[SUPPORT] = "https://other.example/workflow"
    elif case == "redirected_primary":
        finals[PRIMARY] = "https://other.example/voice"
    elif case == "partial_pair":
        raw["supporting_evidence_quote_1"] = ""
    elif case == "customer_role":
        raw["activity_role"] = "customer_user"
    elif case == "missing_verdict":
        raw["product_coverage"] = None
    else:
        raw["product_coverage"] = "UNPROVEN"
    finding = _validate(raw, pages=pages, finals=finals)
    assert finding["status"] == "UNPROVEN"
    assert not coverage_matches(finding.get("product_coverage"), CRITERION)


def test_unproven_and_contradicted_whole_criterion_never_promote():
    raw = _finding(status="UNPROVEN", product="UNPROVEN", role="unresolved",
                   support_url="", support_quote="")
    finding = _validate(raw)
    assert finding["status"] == "UNPROVEN"
    assert not coverage_matches(finding["product_coverage"], CRITERION)
    contradicted = _validate(_finding(
        status="CONTRADICTED", product="CONTRADICTED", role="customer_user",
        support_url="", support_quote="",
    ))
    assert contradicted["status"] == "CONTRADICTED"
    assert not coverage_matches(contradicted["product_coverage"], CRITERION)
    # This tests the model's negative submission. The host does not parse
    # English to prove that a VERIFIED judgment covers every requirement.


def test_final_gate_requires_native_criterion_bound_receipt():
    proof = _validate(_finding())["product_coverage"]
    semantic = {"url": PRIMARY, "quote": PRIMARY_QUOTE}
    assert scorer._industry_evidence_decision(
        "Software", "Practice software", "Software", True,
        semantic_evidence=semantic, industry_activity_role="supplier_operator",
        requested_product_service=CRITERION, product_coverage=proof,
    ) == "match"
    assert scorer._industry_evidence_decision(
        "Software", "Practice software", "Software", True,
        semantic_evidence=semantic, industry_activity_role="supplier_operator",
        requested_product_service=CRITERION, product_coverage=dict(proof),
    ) == "unavailable"
    details = {
        "dimension_decisions": {"industry": "match", "employee_size": "match",
                                "geography": "match"},
        "identity_decision": "match", "required_attribute_decision": "match",
        "dimension_evidence": {"industry": {"url": PRIMARY, "quote": PRIMARY_QUOTE,
                                             "product_coverage": proof}},
    }
    icp = SimpleNamespace(product_service=CRITERION)
    native = scorer.company_fit_match("reviewed", details=details)
    assert scorer._final_product_coverage_result(native, icp, enabled=True).decision == "match"
    for false_proof in (None, dict(proof), json.loads(json.dumps(proof))):
        details["dimension_evidence"]["industry"]["product_coverage"] = false_proof
        result = scorer._final_product_coverage_result(
            scorer.company_fit_match("untrusted", details=details), icp, enabled=True,
        )
        assert result.decision == "unavailable"
        assert result.details["dimension_decisions"]["employee_size"] == "match"
        assert result.details["dimension_decisions"]["geography"] == "match"


def test_native_investigator_uses_original_criterion_without_planner(monkeypatch):
    requests = []

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        requests.append(payload)
        validate_operation_request("openrouter.chat", payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "one", "type": "function", "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [_finding()]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("industry",), requested_industry="Software",
        requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [PRIMARY, SUPPORT]},
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"},
        prefetched_pages={PRIMARY: {"final_url": PRIMARY, "text": PRIMARY_QUOTE},
                          SUPPORT: {"final_url": SUPPORT, "text": SUPPORT_QUOTE}},
    ))
    assert len(requests) == 1, result
    assert requests[0]["model"] == inv.POSITIVE_SEMANTIC_REVIEW_MODEL
    assert CRITERION in str(requests[0]["messages"])
    assert "server_frozen_product_plan" not in str(requests[0]["messages"])
    from lab_arena.operations import validate_operation_request
    validate_operation_request("openrouter.chat", requests[0])
    assert result.get("_completed_submit") is True, result
    assert coverage_matches(result["claims"]["industry"]["product_coverage"], CRITERION)
    assert result["usage"]["reasoning_turns"] == 1
