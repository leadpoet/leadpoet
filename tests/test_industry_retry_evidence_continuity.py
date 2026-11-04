"""Multi-source industry evidence must survive the existing bounded retry path."""
from copy import deepcopy
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_matched_company_retry_source_continuity import (
    IDENTITY, PLATFORM_URL, PLATFORM_TEXT, _fit_result,
)


def supported_result():
    result = _fit_result()
    pages = {PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT}}
    support = []
    for index in range(2):
        url = f"https://unibuddy.com/workflows/{index}"
        quote = f"Unibuddy supplies connected workflow capability {index} to universities."
        support.append({"url": url, "quote": quote})
        pages[url] = {"final_url": url, "text": quote}
    result.details["dimension_evidence"]["industry"]["web_evidence"]["supporting_evidence"] = support
    return result, pages


def retain(result, pages):
    sink, retry = {}, {}
    lead_scorer._retain_matched_investigator_source_contexts(
        sink, result, {investigator.PRIVATE_FETCHED_PAGES_KEY: pages},
    )
    lead_scorer._retain_matched_company_retry_sources(retry, sink, result)
    return sink, lead_scorer._matched_company_retry_prefetched_pages(retry, IDENTITY)


def test_supporting_industry_pages_survive_retention_and_retry(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    before = deepcopy(result.details)
    sink, restored = retain(result, pages)
    assert sink == restored == pages
    assert result.details == before
    assert investigator.MAX_PREFETCHED_PAGES == 3
    assert investigator.MAX_RETURNED_FETCHED_PAGES == 7


def test_old_policy_does_not_admit_new_support_fields(monkeypatch):
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    result, pages = supported_result()
    sink, restored = retain(result, pages)
    assert sink == restored == {PLATFORM_URL: pages[PLATFORM_URL]}


def test_primary_dimension_contexts_keep_their_prior_order(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    dimensions = result.details["dimension_evidence"]
    dimensions["stage"] = {
        "decision": "match",
        "web_evidence": deepcopy(dimensions["industry"]["web_evidence"]),
    }
    contexts = lead_scorer._matched_investigator_source_contexts(result, pages)
    assert [(c["dimension"], c["url"]) for c in contexts[:2]] == [
        ("industry", PLATFORM_URL), ("stage", PLATFORM_URL),
    ]


def test_support_does_not_displace_primary_stage(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    stage_url = "https://unibuddy.com/funding"
    stage_quote = "Unibuddy raised a Series B."
    pages[stage_url] = {"final_url": stage_url, "text": stage_quote}
    result.details["dimension_evidence"]["stage"] = {
        "decision": "match", "web_evidence": {"url": stage_url, "quote": stage_quote},
    }
    sink, restored = retain(result, pages)
    assert len(sink) == 4
    assert restored == pages
    assert PLATFORM_URL in restored and stage_url in restored


def test_all_cited_sources_reach_industry_retry_without_expanding_shared_prefetch(monkeypatch):
    from tests.test_matched_company_retry_source_continuity import _company, _icp, _verdict
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    for index in range(2, 6):
        url = f"https://unibuddy.com/workflows/{index}"
        quote = f"Unibuddy supplies connected workflow capability {index} to universities."
        pages[url] = {"final_url": url, "text": quote}
        result.details["dimension_evidence"]["industry"]["web_evidence"]["supporting_evidence"].append({"url": url, "quote": quote})
    sink, restored = retain(result, pages)
    assert sink == restored == pages
    retry = {}
    lead_scorer._retain_matched_company_retry_sources(retry, sink, result)
    captured = []

    async def capture(**kwargs):
        captured.append(kwargs)
        return {"claims": {}, "failure_reason": "source unavailable"}

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", capture)
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=_company(), icp=_icp(), verdict=_verdict(),
        investigation_targets=("industry",), icp_attribute=_icp().required_attribute,
        icp_stage="", verified_identity=IDENTITY, verified_transport_domain="unibuddy.com",
        structured_employee_size_evidence=None, structured_public_company_evidence=None,
        employee_size_conflict=False, company_quality=True,
        matched_company_retry_source_cache=retry, review_positive_semantics=True,
    ))
    assert len(captured) == 1
    assert len(captured[0]["prefetched_pages"]) == investigator.MAX_PREFETCHED_PAGES == 3
    assert captured[0]["retained_industry_pages"] == pages
    assert lead_scorer._matched_company_retry_prefetched_pages(
        retry, {**IDENTITY, "registrable_dns_domain": "other.example"}
    ) == {}


def test_maximum_citations_and_funding_context_replace_stale_optional_pages(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    dimensions = result.details["dimension_evidence"]
    for index in range(2, 8):
        url = f"https://unibuddy.com/workflows/{index}"
        quote = f"Unibuddy supplies workflow capability {index}."
        pages[url] = {"final_url": url, "text": quote}
        dimensions["industry"]["web_evidence"]["supporting_evidence"].append({"url": url, "quote": quote})
    for dimension in ("employee_size", "geography", "stage"):
        url = f"https://unibuddy.com/{dimension}"
        quote = f"Unibuddy current {dimension} evidence."
        pages[url] = {"final_url": url, "text": quote}
        dimensions[dimension] = {"decision": "match", "web_evidence": {"url": url, "quote": quote}}
    sink, restored = retain(result, pages)
    assert sink == restored == pages
    auxiliary = "https://unibuddy.com/funding-context"
    sink[auxiliary] = {"final_url": auxiliary, "text": "Unibuddy current funding context."}
    stale = {f"https://unibuddy.com/stale/{i}": {"final_url": f"https://unibuddy.com/stale/{i}", "text": "Old optional source"}
             for i in range(investigator.max_matched_retry_pages())}
    retry = {"verified_identity": IDENTITY, "pages": stale}
    lead_scorer._retain_matched_company_retry_sources(retry, sink, result,
        funding_contexts=[{"dimension": "first_party_context", "url": auxiliary, "text": sink[auxiliary]["text"]}])
    assert lead_scorer._matched_company_retry_prefetched_pages(retry, IDENTITY) == sink
    overflow = deepcopy(retry)
    overflow["pages"]["https://unibuddy.com/extra"] = {"final_url": "https://unibuddy.com/extra", "text": "Unibuddy extra source"}
    assert lead_scorer._matched_company_retry_prefetched_pages(overflow, IDENTITY) == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_native_retry_reclassifies_retained_sources_with_existing_guards(monkeypatch, enabled):
    import json
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import URL, FEATURES
    from tests.test_company_evidence_investigator import _finding
    finding = _finding("industry", status="VERIFIED", evidence_url=URL,
                       evidence_quote=FEATURES, activity_role="supplier_operator",
                       observed_industry="Software")
    run, calls, clock, reviews = install(monkeypatch, finding=finding)
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol" if enabled else "")
    pages = {
        f"https://www.coreworks.ai/capability/{index}": {
            "final_url": f"https://www.coreworks.ai/capability/{index}",
            "text": f"Coreworks supplies connected workflow capability {index}.",
        } for index in range(5)
    }
    wrong = "https://other.example/product"
    pages[wrong] = {"final_url": wrong, "text": "Coreworks capability not on its bound domain."}
    redirect = "https://www.coreworks.ai/redirect"
    pages[redirect] = {"final_url": wrong, "text": "Cross-domain content cannot supply proof."}
    pages[URL] = {"final_url": URL, "text": "Old body must not replace a freshly fetched body."}
    result = asyncio.run(run(retained_industry_pages=pages,
        prefetched_pages={URL: {"final_url": URL, "text": FEATURES}},
        prior_observations={"submitted_source_urls": [URL]}))
    assert result["_completed_submit"] is True
    retained = result[investigator.PRIVATE_FETCHED_PAGES_KEY]
    assert retained[URL]["text"] == FEATURES
    assert wrong not in retained and redirect not in retained
    valid = {u for u in pages if "/capability/" in u}
    assert valid.issubset(retained) is enabled
    if enabled:
        final = reviews[-1]
        seen = {p["source_url"] for m in final["messages"][2:]
                for p in json.loads(m["content"])["untrusted_source_pages"]}
        assert valid.issubset(seen)
        assert result["claims"]["industry"]["status"] == "VERIFIED"
        selector = json.dumps(calls['model'][0]['messages'])
        assert all(pages[url]['text'] in selector for url in valid)
        assert pages[wrong]['text'] not in selector
        assert pages[redirect]['text'] not in selector
    assert result["usage"]["prefetched_pages"] <= 3
    assert result["usage"]["fetch_calls"] <= investigator.MAX_FETCH_CALLS
    assert result["usage"]["search_calls"] <= investigator.MAX_SEARCH_CALLS


@pytest.mark.parametrize("fault", ["absent_quote", "cross_domain", "unmatched", "missing_identity"])
def test_support_does_not_bypass_source_or_identity_checks(monkeypatch, fault):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    result, pages = supported_result()
    support_url = next(url for url in pages if url != PLATFORM_URL)
    if fault == "absent_quote":
        pages[support_url]["text"] = "An unrelated fact on the same website."
    elif fault == "cross_domain":
        pages[support_url]["final_url"] = "https://other.example/workflows"
    elif fault == "unmatched":
        result.details["dimension_evidence"]["industry"]["decision"] = "unavailable"
    else:
        result.details["dimension_evidence"]["identity"]["web_identity_receipt"].pop("observed_domain")
    sink, restored = retain(result, pages)
    assert support_url not in sink and support_url not in restored
    if fault in {"unmatched", "missing_identity"}:
        assert sink == restored == {}


def test_supporting_sources_follow_existing_paragraph_context_contract(monkeypatch):
    """Multi-source industry findings must not make a valid paragraph malformed."""
    from qualification.scoring import intent_details
    from qualification.scoring.company_fit_decision import company_fit_match
    from tests.test_arena_intent_details_grounding import inputs

    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, icp, signals, receipt = inputs()
    receipt["dimension_evidence"]["identity"] = {
        "decision": "match", "web_identity_receipt": {
            "decision": "match", "evidence_source": "company_web_reverification",
            "observed_name": "Acme", "observed_domain": "acme.example",
            "observed_linkedin_slug": "acme",
        },
    }
    primary = receipt["dimension_evidence"]["industry"]["web_evidence"]
    support = [
        {"url": f"https://acme.example/product/{i}",
         "quote": "Acme supplies reporting software, platform workflows, product expansion and geographic reach."}
        for i in range(2)
    ]
    primary["supporting_evidence"] = support
    pages = {p["url"]: {"final_url": p["url"], "text": p["quote"]}
             for p in [primary, *support]}
    fit = company_fit_match(details={"dimension_evidence": receipt["dimension_evidence"]})
    retained = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, fit, {investigator.PRIVATE_FETCHED_PAGES_KEY: pages}, company=company,
    )
    assert retained == pages  # Industry evidence and retry storage are unchanged.
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, retained, paragraph=company.intent_details, company=company,
    )
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"), company_source_contexts=contexts,
    )
    # Select the relevant cited support as the single industry context. The
    # original primary quote remains admitted through the final fit receipt.
    assert len(contexts) == 1
    assert contexts[0]["dimension"] == "industry"
    assert contexts[0]["url"] in {p["url"] for p in support}
    admitted = [item for item in document["admitted_evidence"]
                if item["evidence_kind"] == "verified_company_source_context"]
    assert {item["source_url"] for item in admitted} == {c["url"] for c in contexts}
    assert any(item["source_url"] == primary["url"]
               and primary["quote"] in item.get("admitted_text", [])
               for item in document["admitted_evidence"])
    assert sum(len(text.encode()) for item in document["admitted_evidence"]
               if item["evidence_kind"] in {"verified_source_context", "verified_company_source_context"}
               for text in item.get("admitted_text", [])) <= intent_details._MAX_SOURCE_CONTEXT_BYTES
