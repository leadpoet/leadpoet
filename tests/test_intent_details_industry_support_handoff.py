"""Preserve bounded host-validated citations, not the industry's model verdict."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details


CASES = json.loads((Path(__file__).parent / "fixtures" /
    "intent_details_industry_support_handoffs.json").read_text())["cases"]


def _inputs(case=None):
    case = deepcopy(case or CASES[0])
    company = SimpleNamespace(
        company_name=case["company"]["name"],
        company_website=case["company"]["website"],
        intent_details=case["paragraph"],
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0, description="The company announced a contract.",
            url="https://source.example/contract",
        )],
    )
    icp = SimpleNamespace(
        prompt=case["icp"]["prompt"],
        product_service=case["icp"]["product_service"],
        intent_signals=["New contract"],
    )
    receipt = {"decision": "match", "dimension_evidence": {
        "identity": {"decision": "match", "web_identity_receipt": case["identity"]},
        "industry": {"decision": "match", "web_evidence": {
            **case["industry_primary"],
            "supporting_evidence": case["industry_support"],
        }},
    }}
    # The signal is only scaffolding for the real review_evidence entry point.
    # Tests assert source projection/binding, never semantic model accuracy.
    quote = "The company announced a contract."
    signals = [{"after_decay": 54.0, "matched_icp_signal": 0,
        "evidence_urls": ["https://source.example/contract"],
        "judge_verdict": {"decision": "verified", "client_ready": True,
            "verification_trace": {"intent_verdict": {"signal_evaluations": [{
                "signal_status": "supported", "same_entity_check": "pass",
                "supporting_quotes": [quote],
                "evidence_urls_used": ["https://source.example/contract"],
            }]}}}}]
    return company, icp, signals, receipt


def _industry_sources(document):
    return [document["admitted_evidence"][index] for index in
            document["verified_company_evidence"]["industry"]["evidence_source_indexes"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["company"]["name"])
def test_captured_mainframe_citations_survive_real_review_evidence(monkeypatch, case):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "anthropic/claude-sonnet-5.5")
    document = intent_details.review_evidence(*_inputs(case))
    sources = _industry_sources(document)
    primary = case["industry_primary"]
    assert any(source["source_url"] == primary["url"] and
               primary["quote"] in source["admitted_text"] for source in sources)
    for pair, digest in zip(case["industry_support"], case["provenance"]["support_quote_sha256"]):
        assert hashlib.sha256(pair["quote"].encode()).hexdigest() == digest
        assert any(source["source_url"] == pair["url"] and
                   pair["quote"] in source["admitted_text"] for source in sources)
        assert pair["url"] in document["verified_company_evidence"]["industry"]["source_urls"]


def test_support_and_primary_duplicates_are_not_added_twice(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs()
    web = inputs[3]["dimension_evidence"]["industry"]["web_evidence"]
    web["supporting_evidence"] += [deepcopy(web["supporting_evidence"][0]),
                                 {"url": web["url"], "quote": web["quote"]}]
    sources = _industry_sources(intent_details.review_evidence(*inputs))
    pairs = [(s["source_url"], text) for s in sources for text in s["admitted_text"]]
    assert len(pairs) == len(set(pairs)) == 2


@pytest.mark.parametrize("guard", ["policy_disabled", "fit_not_match", "industry_not_match",
                                  "identity_not_match", "missing_observed_domain", "missing_observed_name"])
def test_existing_policy_and_match_identity_guards(monkeypatch, guard):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs();receipt = inputs[3]
    if guard == "policy_disabled":monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    elif guard == "fit_not_match":receipt["decision"] = "unavailable"
    elif guard == "industry_not_match":receipt["dimension_evidence"]["industry"]["decision"] = "mismatch"
    elif guard == "identity_not_match":receipt["dimension_evidence"]["identity"]["web_identity_receipt"]["decision"] = "mismatch"
    elif guard == "missing_observed_domain":receipt["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_domain"] = ""
    else:receipt["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_name"] = ""
    document = intent_details.review_evidence(*inputs)
    texts = [t for s in document["admitted_evidence"] for t in s.get("admitted_text", [])]
    assert inputs[3]["dimension_evidence"]["industry"]["web_evidence"]["supporting_evidence"][0]["quote"] not in texts


@pytest.mark.parametrize("invalid", ["bad_url", "empty_quote", "too_long", "too_many", "wrong_type"])
def test_invalid_support_never_becomes_bindable_evidence(monkeypatch, invalid):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs();web = inputs[3]["dimension_evidence"]["industry"]["web_evidence"]
    support = web["supporting_evidence"]
    if invalid == "bad_url":support[0]["url"] = "javascript:untrusted"
    elif invalid == "empty_quote":support[0]["quote"] = ""
    elif invalid == "too_long":support[0]["quote"] = "x" * 2001
    elif invalid == "too_many":web["supporting_evidence"] = support * 9
    else:web["supporting_evidence"] = "not source pairs"
    document = intent_details.review_evidence(*inputs)
    assert len(_industry_sources(document)) == 1


def test_legacy_without_support_has_identical_canonical_document(monkeypatch):
    inputs = _inputs()
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    old = intent_details.review_evidence(*inputs)
    inputs[3]["dimension_evidence"]["industry"]["web_evidence"].pop("supporting_evidence")
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    new = intent_details.review_evidence(*inputs)
    assert json.dumps(old, sort_keys=True, separators=(",", ":")) == json.dumps(new, sort_keys=True, separators=(",", ":"))


def test_existing_primary_url_projection_is_byte_compatible():
    url = "https://supplier.example/services"
    document = {"verified_signals": [], "verified_company_evidence": {
        "industry": {"url": url, "quote": "Managed infrastructure.",
                     "evidence_url": url, "evidence_quote": "Cloud migration."},
    }}
    intent_details._project_admitted_evidence(document)
    assert document["verified_company_evidence"]["industry"]["source_urls"] == [url, url]
    assert [source["admitted_text"] for source in document["admitted_evidence"]] == [
        ["Managed infrastructure."], ["Cloud migration."],
    ]


def test_support_already_in_selected_body_is_not_duplicated(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs();pair = inputs[3]["dimension_evidence"]["industry"]["web_evidence"]["supporting_evidence"][0]
    document = intent_details.review_evidence(*inputs, company_source_contexts=[{
        "dimension": "industry", "url": pair["url"], "text": pair["quote"],
    }])
    sources = _industry_sources(document)
    assert sum(intent_details._quote_is_bound(pair["quote"], source["admitted_text"])
               for source in sources) == 1
    assert any(s["evidence_kind"] == "verified_company_source_context" for s in sources)


def test_support_remains_bound_when_optional_body_window_does_not_contain_it(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs();web = inputs[3]["dimension_evidence"]["industry"]["web_evidence"]
    document = intent_details.review_evidence(*inputs, company_source_contexts=[{
        "dimension": "industry", "url": web["url"],
        "text": web["quote"] + " Unrelated navigation." * 1000,
    }])
    pair = web["supporting_evidence"][0]
    sources = _industry_sources(document)
    assert any(s["source_url"] == pair["url"] and s["admitted_text"] == [pair["quote"]]
               for s in sources)
    body_bytes = sum(len(text.encode()) for s in document["admitted_evidence"]
                     if s["evidence_kind"] == "verified_company_source_context"
                     for text in s["admitted_text"])
    assert body_bytes <= intent_details._MAX_SOURCE_CONTEXT_BYTES


def test_new_bindings_use_existing_citation_repair_without_regrading_other_checks(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    document = intent_details.review_evidence(*_inputs())
    pair = CASES[0]["industry_support"][0]
    index = next(s["source_index"] for s in _industry_sources(document)
                 if s["admitted_text"] == [pair["quote"]])
    response = {
        "unit_grounding": [{"unit_id": unit["unit_id"], "contains_factual_claim": unit["unit_id"] == 0,
            "status": "VERIFIED", "evidence": ([{"source_index": index, "quote": "An invented unsupported claim."}]
                if unit["unit_id"] == 0 else [])} for unit in document["intent_details_units"]],
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{key: key != "connects_icp" for key in intent_details._CHECKS},
    }
    with pytest.raises(intent_details._BoundedReviewRepairNeeded) as error:
        intent_details._validate_review_response(json.dumps(response), document)
    assert error.value.issues == {0: ("nonexact_quote",)}
    assert error.value.semantic_unit_ids == set()
    merged = intent_details._merge_citation_repairs(json.dumps({"repairs": [{
        "unit_id": 0, "evidence": [{"source_index": index, "quote": pair["quote"]}],
    }]}), error.value.held_response, {0})
    checks = intent_details._validate_review_response(json.dumps(merged), document, initial_review=False)
    assert checks["facts_supported"] is True  # Binding contract only; no model accuracy claim.
    assert checks["connects_icp"] is False


def test_support_over_existing_document_limit_is_unavailable_not_silently_dropped(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs = _inputs()
    supported = intent_details.review_evidence(*inputs)
    primary_only = deepcopy(inputs)
    primary_only[3]["dimension_evidence"]["industry"]["web_evidence"].pop("supporting_evidence")
    original = intent_details.review_evidence(*primary_only)
    bound = (len(json.dumps(original, ensure_ascii=False)) + len(json.dumps(supported, ensure_ascii=False))) // 2
    monkeypatch.setattr(intent_details, "_MAX_REVIEW_DOCUMENT_CHARACTERS", bound)
    with pytest.raises(ValueError, match="exceeds its bound"):
        intent_details.review_evidence(*inputs)
