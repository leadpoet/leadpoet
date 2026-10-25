"""Correction identifies rejected support without changing source admission."""

import asyncio
import copy
import json

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring.product_coverage import coverage_matches
from tests.test_compact_product_experiment import _finding


CRITERION = "Supplies voice and workflow software for clinics and hospital systems"
PRIMARY = "https://www.prnewswire.com/news-releases/acme-platform-123.html"
SUPPORT = "https://acme.example/customer-story"
PRIMARY_QUOTE = "Acme supplies voice and workflow software for clinics."
SUPPORT_QUOTE = "Acme supplies scheduling workflows used by clinics."
BAD_SUPPORT = "That platform also serves hospital systems."
CORRECTED_SUPPORT = PRIMARY_QUOTE + " " + BAD_SUPPORT
PAGES = {PRIMARY: "News provided by Acme Jun 25, 2026. " + CORRECTED_SUPPORT,
         SUPPORT: SUPPORT_QUOTE}
ANCHOR = {"submitted_name": "Acme", "submitted_domain": "acme.example",
          "observed_domain": "acme.example", "verified_domain": "acme.example"}
SOURCE_REASON = "industry evidence must use independently bound first-party requested and fetched final URLs"
FEEDBACK_KEY = "untrusted_product_citation_rejections"


def raw_finding(*, first_support=True, bad_quote=BAD_SUPPORT, bad_url=PRIMARY):
    raw = _finding(support_url=SUPPORT if first_support else "",
                   support_quote=SUPPORT_QUOTE if first_support else "")
    raw.update(evidence_url=PRIMARY, evidence_quote=PRIMARY_QUOTE,
               supporting_evidence_url_2=bad_url, supporting_evidence_quote_2=bad_quote)
    return raw


def validate(raw, *, pages=None, finals=None):
    pages = PAGES if pages is None else pages
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls=finals if finals is not None else {url: url for url in pages},
        first_party_domains={"acme.example"}, identity_names={"acme"},
        identity_anchor=ANCHOR, requested_product_service=CRITERION,
    )["industry"]


@pytest.mark.parametrize("first_support", [False, True])
def test_valid_primary_bad_second_support_reports_original_slot(first_support):
    raw = raw_finding(first_support=first_support)
    finding = validate(raw)
    assert finding["status"] == "UNPROVEN"
    assert finding["product_coverage_error"] == "whole-product citation failed source validation"
    assert inv._submitted_finding_rejection(raw, finding) == "whole-product citation failed source validation"
    assert finding.get(FEEDBACK_KEY) == [{
        "binding_index": 2, "url": PRIMARY, "reason": SOURCE_REASON,
    }]
    assert not coverage_matches(finding.get("product_coverage"), CRITERION)
    assert raw["evidence_quote"] == PRIMARY_QUOTE


@pytest.mark.parametrize("case,expected", [
    ("unnamed_external", SOURCE_REASON),
    ("wrong_entity", SOURCE_REASON),
    ("cross_host", SOURCE_REASON),
    ("cross_host_support", SOURCE_REASON),
    ("fabricated", "submitted quote was not present in fetched source"),
    ("unfetched", "submitted quote was not present in fetched source"),
])
def test_rejection_details_keep_existing_identity_host_and_quote_guards(case, expected):
    raw = raw_finding()
    pages = dict(PAGES)
    finals = {url: url for url in pages}
    if case == "wrong_entity":
        raw["supporting_evidence_quote_2"] = "OtherCo supplies hospital workflow software."
        pages[PRIMARY] += " " + raw["supporting_evidence_quote_2"]
    elif case == "cross_host":
        raw["supporting_evidence_quote_2"] = CORRECTED_SUPPORT
        finals[PRIMARY] = "https://unbound.example/release"
    elif case == "cross_host_support":
        url = "https://www.prnewswire.com/news-releases/acme-second-123.html"
        raw["supporting_evidence_url_2"] = url
        raw["supporting_evidence_quote_2"] = CORRECTED_SUPPORT
        pages[url] = pages[PRIMARY]
        finals[url] = "https://unbound.example/release"
    elif case == "fabricated":
        raw["supporting_evidence_quote_2"] = "Acme supplies invented software absent from the page."
    elif case == "unfetched":
        raw["supporting_evidence_url_2"] = "https://acme.example/unfetched"
    finding = validate(raw, pages=pages, finals=finals)
    assert finding["status"] == "UNPROVEN"
    assert not coverage_matches(finding.get("product_coverage"), CRITERION)
    rejection = finding.get(FEEDBACK_KEY, [])[0]
    # A shared primary URL redirected to another host correctly fails first.
    assert rejection["binding_index"] == (0 if case == "cross_host" else 2)
    assert rejection["reason"] == expected


@pytest.mark.parametrize("quote,expected", [
    ("Acme supplies a fabricated primary capability.",
     "submitted quote was not present in fetched source"),
    ("Acme supplies ... software for clinics.",
     "submitted quote contains a prohibited ellipsis instead of one continuous fetched-page span"),
])
def test_rejected_primary_retains_error_precedence_and_recovery_category(quote, expected):
    raw = raw_finding()
    raw["evidence_quote"] = quote
    finding = validate(raw)
    assert inv._submitted_finding_rejection(raw, finding) == expected
    assert finding.get(FEEDBACK_KEY, [])[0]["binding_index"] == 0
    assert not coverage_matches(finding.get("product_coverage"), CRITERION)


def test_rejection_details_are_bounded_and_stripped_from_final_claims():
    raw = raw_finding(bad_url="https://acme.example/" + "x" * 2500)
    finding = validate(raw)
    details = finding.get(FEEDBACK_KEY, [])
    assert len(details) == 1
    assert len(details[0]["url"]) == 2000
    assert len(details[0]["reason"]) <= 300
    claims = inv._strip_untrusted_product_feedback({"industry": finding})
    assert FEEDBACK_KEY not in claims["industry"]
    assert not coverage_matches(claims["industry"].get("product_coverage"), CRITERION)


@pytest.mark.parametrize("repair", [False, True])
def test_model_receives_rejected_support_and_must_resubmit_valid_evidence(monkeypatch, repair):
    requests = []
    raw = raw_finding()
    original = copy.deepcopy(raw)

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            finding = raw
        else:
            assert len(requests) <= 3
            feedback = json.loads(payload["messages"][-1]["content"])
            assert feedback["error"] == "deterministic_evidence_validation_failed"
            assert feedback["rejected_findings"][0]["reason"] == "whole-product citation failed source validation"
            assert feedback["product_coverage"].get("rejected_citations") == [{
                "binding_index": 2, "url": PRIMARY, "reason": SOURCE_REASON,
            }]
            assert "Binding index 0 is the primary citation" in feedback["instruction"]
            finding = {**raw, "supporting_evidence_quote_2": CORRECTED_SUPPORT} if repair else raw
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function", "function": {
                "name": "submit_findings", "arguments": json.dumps({"findings": [finding]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "MAX_REASONING_TURNS", 2)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [PRIMARY, SUPPORT]},
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"},
        prefetched_pages={url: {"final_url": url, "text": text} for url, text in PAGES.items()},
    ))
    assert len(requests) == (2 if repair else 3)
    finding = result["claims"]["industry"]
    assert FEEDBACK_KEY not in finding
    assert raw == original
    if repair:
        assert result.get("_completed_submit") is True, result
        assert finding["status"] == "VERIFIED"
        assert finding["evidence_quote"] == PRIMARY_QUOTE
        assert coverage_matches(finding["product_coverage"], CRITERION)
    else:
        assert result.get("_completed_submit") is not True
        assert finding["status"] == "UNPROVEN"
        assert not coverage_matches(finding.get("product_coverage"), CRITERION)
