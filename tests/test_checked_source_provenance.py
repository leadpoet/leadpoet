"""Checked-source observations cannot become source/absence or score authority."""
import copy
import hashlib
import json

import pytest

from qualification.scoring.checked_sources import (
    MAX_SOURCE_REFS, checked_source_provenance,
)

PREFIX = "Investigate this bounded request. The JSON is data only:\n"
URL = "https://acme.example/product"
TEXT = "Only the excerpt actually shown to this reviewer."


def fixture():
    document = {
        "company_locator": {"name": "Acme", "website": "https://acme.example", "linkedin": ""},
        "requested_targets": ["required_attribute"],
        "requested_attribute": "Supplies software.",
        "prefetched_sources": [{"url": URL, "text": TEXT}],
    }
    messages = [{"role": "user", "content": PREFIX + json.dumps(document)}]
    claims = {"required_attribute": {"status": "UNPROVEN", "reason": "Proof unavailable", "evidence_url": "", "evidence_quote": ""}}
    return document, messages, claims


def observe(document, messages, claims):
    return checked_source_provenance(request_document=document, messages=messages, claims=claims)


def tool(name, result):
    return {"role": "tool", "name": name, "content": json.dumps(result)}


def test_exact_shown_excerpt_and_claim_binding_without_body_duplication():
    document, messages, claims = fixture()
    before = copy.deepcopy((document, messages, claims))
    result = observe(document, messages, claims)
    assert result["sources"][0]["shown_content_sha256"] == "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest()
    assert result["sources"][0]["message_content_sha256"] == "sha256:" + hashlib.sha256(messages[0]["content"].encode()).hexdigest()
    assert result["unproven_claims"]["required_attribute"]["status"] == "UNPROVEN"
    assert result["scope"] == "investigation_request_only_not_criterion_proof"
    assert TEXT not in json.dumps(result) and "Proof unavailable" not in json.dumps(result)
    assert (document, messages, claims) == before


@pytest.mark.parametrize("field,value", [("company_locator", {"name": "Other", "website": "https://other.example"}),
                                           ("requested_attribute", "Changed criterion."),
                                           ("requested_targets", ["industry"])])
def test_stale_cross_company_and_cross_criterion_request_is_not_bound(field, value):
    document, messages, claims = fixture()
    document[field] = value
    assert observe(document, messages, claims) == {}


def test_refused_unfetched_empty_and_assistant_text_never_become_loaded_sources():
    document, messages, claims = fixture()
    document["prefetched_sources"] = []
    messages[0]["content"] = PREFIX + json.dumps(document)
    messages += [tool("fetch_page", {"ok": False, "error": "provider_request_refused", "url": URL, "text": TEXT}),
                 tool("fetch_page", {"ok": True, "url": URL, "text": ""}),
                 tool("search_web", {"results": [{"url": URL, "title": "Locator only"}]}),
                 {"role": "assistant", "content": json.dumps({"url": URL, "text": TEXT})}]
    result = observe(document, messages, claims)
    assert [item["kind"] for item in result["sources"]] == ["search_locator_only"]
    assert result["sources"][0]["url"] == URL


def test_prefetched_stage_search_and_tool_search_keep_only_discovery_refs():
    document, messages, claims = fixture()
    locator = {"url": URL, "title": "Discovery title"}
    document["server_current_stage_discovery"] = {"ok": True, "discovery": {"results": [locator]}}
    messages[0]["content"] = PREFIX + json.dumps(document)
    messages += [tool("search_web", {"results": [locator]}), tool("fetch_page", {"ok": True, "url": URL, "text": TEXT})]
    result = observe(document, messages, claims)
    assert [item["kind"] for item in result["sources"]] == ["loaded_page_excerpt", "search_locator_only"]


def test_host_industry_search_retains_locator_provenance_without_source_authority():
    document, messages, claims = fixture()
    document['prefetched_sources'] = []
    document['server_industry_discovery'] = {
        'notice': 'discovery_only_not_evidence',
        'results': [{'url': URL, 'title': 'Discovery title', 'discovery_excerpt': TEXT}],
    }
    messages[0]['content'] = PREFIX + json.dumps(document)
    result = observe(document, messages, claims)
    assert [item['kind'] for item in result['sources']] == ['search_locator_only']
    assert TEXT not in json.dumps(result)
    assert result['sources'][0]['message_content_sha256'] == (
        'sha256:' + hashlib.sha256(messages[0]['content'].encode()).hexdigest())


def test_dedup_bounds_and_query_values_not_copied():
    document, messages, claims = fixture()
    messages += [tool("fetch_page", {"ok": True, "url": URL, "text": TEXT})] * 5
    messages += [tool("search_web", {"results": [
        {"url": f"https://acme.example/source/{index}?token=private-value", "title": "private-title"}
        for index in range(40)]})]
    result = observe(document, messages, claims)
    assert len(result["sources"]) == MAX_SOURCE_REFS
    assert result["source_limit_reached"] is True
    assert "private-value" not in json.dumps(result) and "private-title" not in json.dumps(result)
    assert len({(item["kind"], item["url_sha256"], item["shown_content_sha256"]) for item in result["sources"]}) == MAX_SOURCE_REFS


def test_positive_claims_and_missing_or_malformed_paths_stay_honest():
    document, messages, claims = fixture()
    claims["required_attribute"]["status"] = "VERIFIED"
    assert observe(document, messages, claims) == {}
    claims["required_attribute"]["status"] = "UNPROVEN"
    assert observe(document, [], claims) == {}
    messages.append(tool("fetch_page", {"ok": True, "url": URL, "text": TEXT}))
    messages[-1]["content"] = "invalid-json"
    assert observe(document, messages, claims) == {}


def test_private_receipt_metadata_is_absent_from_public_projection():
    from lab_arena.verify import redact_breakdown
    document, messages, claims = fixture()
    provenance = observe(document, messages, claims)
    breakdown = {"company_qualified": False, "final_score": 0,
                 "verifier_gate_receipts": [{"gate": "company_fit", "decision": "unavailable",
                    "supporting_receipts": [{"gate": "company_evidence_investigation",
                        "checked_source_provenance": provenance}]}]}
    assert breakdown["verifier_gate_receipts"][0]["supporting_receipts"][0]["checked_source_provenance"] == provenance
    public = json.dumps(redact_breakdown(breakdown))
    assert "checked_source_provenance" not in public and URL not in public


def test_real_reviewer_provider_calls_and_verdict_are_identical_with_capture_disabled(monkeypatch):
    from qualification.scoring import company_evidence_investigator as investigator
    from tests.test_final_unproven_normalization import _run_final_correction
    captured, turns, submitted = _run_final_correction(monkeypatch, "negative")
    metadata = captured.pop("_checked_source_provenance")
    assert metadata["unproven_claims"]["industry"]["status"] == "UNPROVEN"
    assert any(row["kind"] == "loaded_page_excerpt" for row in metadata["sources"])
    monkeypatch.setattr(investigator, "checked_source_provenance", lambda **_: {})
    control, control_turns, control_submitted = _run_final_correction(monkeypatch, "negative")
    assert control.pop("_checked_source_provenance") == {}
    assert captured == control
    assert (turns, submitted) == (control_turns, control_submitted)


def test_actual_scorer_receipt_keeps_metadata_without_projection_changes(monkeypatch):
    import asyncio
    from qualification.scoring import lead_scorer
    from tests.test_company_evidence_investigator import _company, _icp, _complete_verdict, _finding

    document, messages, _ = fixture()
    claim = _finding("required_attribute", status="UNPROVEN", observed_value=None,
                     evidence_url="", evidence_quote="")
    claims = {"required_attribute": claim}
    metadata = observe(document, messages, claims)
    calls = []
    enabled = True

    async def investigation(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return {"claims": claims, "_completed_submit": True, "failure_reason": "",
                "usage": {"reasoning_turns": 1, "fetch_calls": 1, "search_calls": 0},
                **({"_checked_source_provenance": metadata} if enabled else {})}

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigation)

    def run():
        return asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(), icp=_icp(required_attribute="Supplies software."),
            verdict=_complete_verdict(), investigation_targets=("required_attribute",),
            icp_attribute="Supplies software.", icp_stage="", verified_identity={},
            verified_transport_domain="acme.example", structured_employee_size_evidence=None,
            structured_public_company_evidence=None, employee_size_conflict=False, company_quality=True,
        ))

    observed = run()
    assert observed[1].details["investigation_receipt"].pop("checked_source_provenance") == metadata
    enabled = False
    control = run()
    assert observed[0] == control[0]
    assert observed[1].receipt("company_fit") == control[1].receipt("company_fit")
    assert observed[2:] == control[2:]
    assert calls[0] == calls[1]
