"""Structured employee ranges stay bound to verified company identity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details


def _inputs(
    *,
    name="CrowdStrike",
    domain="crowdstrike.com",
    slug="crowdstrike",
    employee_count="5,001-10,000",
):
    signal_url = f"https://{domain}/news/platform"
    company = SimpleNamespace(
        company_name=name,
        company_website=f"https://www.{domain}/",
        company_linkedin=f"https://www.linkedin.com/company/{slug}",
        intent_details=f"{name} has more than 5,000 employees.",
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0,
            description=f"{name} launched its platform.",
            date="2026-09-01",
            url=signal_url,
        )],
    )
    icp = SimpleNamespace(
        prompt="Find large software companies with a recent product launch.",
        product_service="Enterprise software",
        intent_signals=["Product launch"],
        employee_count=[employee_count],
    )
    results = [{
        "after_decay": 50.0,
        "matched_icp_signal": 0,
        "evidence_urls": [signal_url],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": "2026-09-01",
            "authoritative_date_basis": "event",
            "verification_trace": {"intent_verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "same_entity_check": "pass",
                "supporting_quotes": [f"{name} launched its platform."],
                "evidence_urls_used": [signal_url],
            }]}},
        },
    }]
    fit = {
        "gate": "company_fit",
        "decision": "match",
        "dimension_evidence": {
            "identity": {
                "decision": "match",
                "web_identity_receipt": {
                    "decision": "match",
                    "observed_domain": domain,
                    "observed_linkedin_slug": slug,
                },
            },
            "employee_size": {
                "decision": "match",
                "submitted_decision": "match",
                "observed_decision": "match",
                "web_evidence": {
                    "employee_count": employee_count,
                    "provider": "harvestapi_get_company",
                    "source_field": "employeeCountRange",
                    "url": f"https://www.linkedin.com/company/{slug}",
                    "website": f"https://{domain}/",
                },
            },
        },
    }
    return company, icp, results, fit


def _structured_sources(document):
    return [
        source for source in document["admitted_evidence"]
        if source["evidence_kind"] == "structured_provider_observation"
    ]


def test_crowdstrike_structured_range_is_a_bound_typed_fact():
    company, icp, results, fit = _inputs()

    document = intent_details.review_evidence(company, icp, results, fit)

    sources = _structured_sources(document)
    assert sources == [{
        "source_index": sources[0]["source_index"],
        "evidence_kind": "structured_provider_observation",
        "company_dimension": "employee_size",
        "source_url": "https://www.linkedin.com/company/crowdstrike",
        "admitted_text": [
            "Structured provider observation (employeeCountRange): "
            "5,001-10,000"
        ],
    }]
    source = sources[0]
    assert document["verified_company_evidence"]["employee_size"][
        "evidence_source_indexes"
    ] == [source["source_index"]]
    quote = source["admitted_text"][0]
    grounding = [{
        "unit_id": 0,
        "contains_factual_claim": True,
        "status": "VERIFIED",
        "evidence": [{"source_index": source["source_index"], "quote": quote}],
    }]
    assert intent_details._validate_unit_grounding(grounding, document) == (
        True, True, {},
    )
    assert "not a verbatim webpage quotation" in (
        intent_details._STRUCTURED_EMPLOYEE_RANGE_SYSTEM_APPENDIX
    )
    assert "company stage" in intent_details._STRUCTURED_EMPLOYEE_RANGE_SYSTEM_APPENDIX
    assert "observed_dates" not in source


def test_generic_canonical_range_above_5000_is_admitted():
    company, icp, results, fit = _inputs(
        name="Example Systems",
        domain="example-systems.com",
        slug="example-systems",
        employee_count="10,001+",
    )

    document = intent_details.review_evidence(company, icp, results, fit)

    assert _structured_sources(document)[0]["admitted_text"] == [
        "Structured provider observation (employeeCountRange): 10,001+"
    ]


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("provider", "other_provider"),
        ("source_field", "employeeCount"),
        ("employee_count", "5001-10000"),
        ("url", "https://www.linkedin.com/company/other-company"),
        ("website", "https://other.example/"),
        ("extra", "untrusted"),
        ("missing", None),
        ("company_decision", "mismatch"),
        ("identity_decision", "mismatch"),
        ("identity_receipt_decision", "mismatch"),
        ("employee_decision", "mismatch"),
        ("employee_submitted_decision", "mismatch"),
        ("employee_observed_decision", "mismatch"),
        ("identity_domain", "other.example"),
        ("identity_slug", "other-company"),
    ],
)
def test_unbound_or_noncanonical_structured_range_is_not_admitted(mutation, value):
    company, icp, results, fit = _inputs()
    fit = deepcopy(fit)
    dimensions = fit["dimension_evidence"]
    employee = dimensions["employee_size"]
    evidence = employee["web_evidence"]
    if mutation in {"provider", "source_field", "employee_count", "url", "website"}:
        evidence[mutation] = value
    elif mutation == "extra":
        evidence["submitted_value"] = value
    elif mutation == "missing":
        del evidence["source_field"]
    elif mutation == "company_decision":
        fit["decision"] = value
    elif mutation == "identity_decision":
        dimensions["identity"]["decision"] = value
    elif mutation == "identity_receipt_decision":
        dimensions["identity"]["web_identity_receipt"]["decision"] = value
    elif mutation == "employee_decision":
        employee["decision"] = value
    elif mutation == "employee_submitted_decision":
        employee["submitted_decision"] = value
    elif mutation == "employee_observed_decision":
        employee["observed_decision"] = value
    elif mutation == "identity_domain":
        dimensions["identity"]["web_identity_receipt"]["observed_domain"] = value
    elif mutation == "identity_slug":
        dimensions["identity"]["web_identity_receipt"][
            "observed_linkedin_slug"
        ] = value

    document = intent_details.review_evidence(company, icp, results, fit)

    assert _structured_sources(document) == []


def test_submitted_employee_value_alone_is_not_admitted():
    company, icp, results, fit = _inputs()
    employee = fit["dimension_evidence"]["employee_size"]
    employee.pop("web_evidence")
    employee["submitted_employee_count"] = "5,001-10,000"

    document = intent_details.review_evidence(company, icp, results, fit)

    assert _structured_sources(document) == []
