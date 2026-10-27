"""Multi-source industry evidence must survive the existing bounded retry path."""
from copy import deepcopy

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


def test_support_does_not_displace_primary_stage_or_expand_retry_cap(monkeypatch):
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
    assert len(restored) == investigator.MAX_PREFETCHED_PAGES
    assert PLATFORM_URL in restored and stage_url in restored


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
