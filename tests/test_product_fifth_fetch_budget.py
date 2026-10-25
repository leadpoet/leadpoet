"""A discovered fifth product source uses the existing bounded research route."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer
from tests.product_coverage_fixtures import (
    initial_investigation_content, native_product_post, validated_industry_finding,
)
from tests.test_company_evidence_investigator import _finding

BASE = "https://acme.example"
CRITERION = "Supplies workflow software to businesses"
QUOTE = "Acme supplies workflow software to businesses."
URLS = [f"{BASE}/source-{index}" for index in range(6)]


def _trace(monkeypatch, *, criterion=CRITERION, final_status="UNPROVEN",
           deny_sixth=True, reserve_fifth=False, target="industry",
           failure=None):
    requests, fetched, fallbacks = [], [], []
    clock = [0.0]
    actions = [("search_web", {"query": "Acme workflow software"})]
    actions += [("fetch_page", {"url": url}) for url in URLS[:6 if deny_sixth else 5]]
    actions += [("submit_findings", {"findings": [_finding(
        target, status=final_status, activity_role="supplier_operator",
        observed_industry="Software", evidence_url=URLS[4] if final_status == "VERIFIED" else "",
        evidence_quote=QUOTE if final_status == "VERIFIED" else "",
    )]})]

    async def model(_session, _url, *, headers, payload):
        requests.append(payload)
        name, arguments = actions[len(requests) - 1]
        if reserve_fifth and name == "fetch_page" and arguments["url"] == URLS[4]:
            clock[0] = inv.ADMISSION_DEADLINE_SECONDS - inv.JUDGMENT_ADMISSION_RESERVE_SECONDS
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            },
        }]}}]}

    async def search(_session, _query, *, key, max_results=5):
        # The exact fifth URL is visible before the model selects it. These
        # locators do not supply proof and the model must still fetch the URL.
        return {"results": [{"url": url, "title": "Source"} for url in URLS]}

    async def fetch(_session, url, *, stealth_mode=False):
        fetched.append(url)
        if not stealth_mode and failure == "fourth_transient" and url == URLS[3]:
            return {"ok": False, "error": "http_503"}
        if not stealth_mode and failure in {"fifth_transient", "fifth_stealth"} and url == URLS[4]:
            return {"ok": False, "error": "http_400" if failure == "fifth_stealth" else "http_503",
                    "_retry_with_stealth": failure == "fifth_stealth"}
        return {"ok": True, "url": url, "final_url": url,
                "text": QUOTE if url == URLS[4] else "Acme background only."}

    async def fallback(_session, url):
        fallbacks.append(url)
        assert failure == "fourth_transient" and url == URLS[3]
        return {"ok": True, "url": url, "final_url": url, "text": "Acme background only."}

    saved = {f"{BASE}/saved-{index}": {"final_url": f"{BASE}/saved-{index}",
                                      "text": "Acme background only."}
             for index in range(3)}
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline")
    monkeypatch.setenv("EXA_API_KEY", "offline")
    monkeypatch.setattr(inv, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(inv, "_post_json", native_product_post(model))
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    monkeypatch.setattr(inv, "_fetch_exa_exact_page", fallback)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": BASE}, targets=(target,),
        requested_industry="Software", requested_product_service=criterion,
        requested_attribute=criterion if target == "required_attribute" else "",
        prior_observations={"submitted_source_urls": list(saved)}, prefetched_pages=saved,
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example"},
    ))
    return result, fetched, requests, fallbacks


@pytest.mark.parametrize("target,criterion,cap", [
    ("industry", CRITERION, 5), ("industry", "", 4), ("industry", "   ", 4),
    ("required_attribute", CRITERION, 4),
])
def test_discovered_fifth_url_is_dispatched_only_for_nonempty_product_review(monkeypatch, target, criterion, cap):
    result, fetched, requests, _fallbacks = _trace(monkeypatch, criterion=criterion, target=target)
    assert fetched == URLS[:cap]
    initial = json.loads(initial_investigation_content(requests[0]).split("\n", 1)[1])
    assert initial["investigation_limits"]["fetch_calls"] == cap
    assert initial["investigation_limits"]["remaining_fetch_calls"] == cap
    assert result["usage"]["fetch_calls"] == cap
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["reasoning_turns"] == 8
    assert json.loads(requests[-1]["messages"][-1]["content"])["error"] == "fetch_budget_exhausted"
    # A denied optional request does not erase an honest terminal negative.
    assert result["claims"][target]["status"] == "UNPROVEN"
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert URLS[5] not in result[inv.PRIVATE_FETCHED_PAGES_KEY]
    if cap == 5:
        refs = result["_checked_source_provenance"]["sources"]
        assert any(ref.get("url") == URLS[4] and ref["kind"] == "loaded_page_excerpt" for ref in refs)


def test_eighth_loaded_page_reaches_native_quote_and_private_cache_gates(monkeypatch):
    result, fetched, _requests, _fallbacks = _trace(monkeypatch, final_status="VERIFIED", deny_sixth=False)
    assert fetched == URLS[:5]
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    pages = result[inv.PRIVATE_FETCHED_PAGES_KEY]
    assert len(pages) == 8
    loaded, finals = inv._validated_returned_fetched_pages(pages, submitted_source_urls=list(pages))
    assert loaded[URLS[4]] == QUOTE and finals[URLS[4]] == URLS[4]
    cache = {URLS[4]: {"status": "source_unavailable", "text": "", "final_url": ""}}
    lead_scorer._hydrate_required_attribute_source_cache(cache, result)
    assert cache[URLS[4]]["text"] == QUOTE
    assert lead_scorer._MAX_RETAINED_INVESTIGATOR_CONTEXTS == 6
    overflow = {**pages, f"{BASE}/extra": {"final_url": f"{BASE}/extra", "text": QUOTE}}
    assert inv._validated_returned_fetched_pages(overflow, submitted_source_urls=list(overflow)) == ({}, {})

    # Raising a transport budget grants no quote, entity, source or role authority.
    for mutation in ("fabricated_quote", "wrong_entity", "foreign_final", "unfetched", "customer_role"):
        finding = _finding("industry", status="VERIFIED", activity_role="supplier_operator",
                           observed_industry="Software", evidence_url=URLS[4], evidence_quote=QUOTE)
        text, final = QUOTE, URLS[4]
        if mutation == "fabricated_quote":
            finding["evidence_quote"] = "Acme supplies invented capabilities."
        elif mutation == "wrong_entity":
            text = finding["evidence_quote"] = "Otherco supplies workflow software to businesses."
        elif mutation == "foreign_final":
            final = "https://foreign.example/source"
        elif mutation == "unfetched":
            finding["evidence_url"] = URLS[5]
        else:
            finding["activity_role"] = "customer_user"
        claim = validated_industry_finding(
            finding, CRITERION, fetched_pages={URLS[4]: text},
            fetched_final_urls={URLS[4]: final}, first_party_domains={"acme.example"},
            identity_names={"acme"},
        )
        assert claim["status"] != "VERIFIED", mutation


def test_fifth_source_does_not_extend_judgment_reserve(monkeypatch):
    result, fetched, requests, _fallbacks = _trace(monkeypatch, deny_sixth=False, reserve_fifth=True)
    assert fetched == URLS[:4]
    assert result["usage"]["fetch_calls"] == 4
    assert json.loads(requests[-1]["messages"][-1]["content"])["error"] == "judgment_time_reserved"
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["_completed_submit"] is True


@pytest.mark.parametrize("failure,expected_fetched,expected_fallbacks,expected_calls", [
    ("fourth_transient", URLS[:4], [URLS[3]], 5),
    ("fifth_transient", URLS[:5], [], 5),
    ("fifth_stealth", [*URLS[:5], URLS[4]], [], 6),
])
def test_product_retries_share_cap_without_admitting_a_sixth_source(
    monkeypatch, failure, expected_fetched, expected_fallbacks, expected_calls,
):
    result, fetched, _requests, fallbacks = _trace(monkeypatch, failure=failure)
    assert fetched == expected_fetched
    assert fallbacks == expected_fallbacks
    assert result["usage"]["fetch_calls"] == expected_calls
    assert URLS[5] not in fetched
    # Existing same-URL stealth settlement may exceed the attempt cap once;
    # it does not add a source URL or admit another fallback after the fifth.
    assert result["failure_reason"] == ("provider_error" if failure == "fifth_transient" else "")


def test_fetch_prompt_defers_to_per_request_limit_without_old_four_fetch_rule():
    prompt = " ".join(inv._SYSTEM_PROMPT.split())
    assert "stated fresh fetch cap" in prompt
    assert "four fresh fetch" not in prompt
    assert "4 fresh fetch" not in prompt
    assert "four-call budget" not in prompt
    assert (inv.MAX_REASONING_TURNS, inv.MAX_SEARCH_CALLS, inv.MAX_PREFETCHED_PAGES,
            inv.MAX_PAGE_CHARACTERS, inv.ADMISSION_DEADLINE_SECONDS,
            inv.JUDGMENT_ADMISSION_RESERVE_SECONDS) == (8, 2, 3, 24_000, 110.0, 30.0)
