"""Replay Nov20's retained Qwilr source/quote through the real investigator."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from qualification.scoring import linkedin_company_size as profile

CASE = json.loads((Path(__file__).parent / "fixtures/company_evidence/qwilr_headquarters_alias.json").read_text())
URL = CASE["finding"]["evidence_url"]
HQ = {
    "name": "Qwilr", "website": "https://qwilr.com/", "url": URL,
    "provider": profile.STRUCTURED_PROFILE_PROVIDER,
    "source_field": profile.STRUCTURED_PROFILE_HEADQUARTERS_SOURCE_FIELD,
    "country": "Australia", "state": "New South Wales",
    "text": CASE["finding"]["evidence_quote"],
}


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("source_fault", ["", "wrong-domain", "wrong-profile"])
def test_retained_quote_survives_full_investigator_validation(monkeypatch, retry, source_fault):
    sources = {p["url"]: {"status": "fetched", "final_url": p["url"], "text": p["text"]}
               for p in CASE["prefetched_sources"]}
    # The non-www capture reached the exact company profile via a verified redirect.
    sources[URL.replace("www.linkedin.com", "linkedin.com")]["final_url"] = URL
    retry_pages = {"https://qwilr.com/" + p: {"final_url": "https://qwilr.com/" + p,
                   "text": "Qwilr offers proposal software."} for p in ("", "pricing/", "about/")} if retry else {}
    hq = dict(HQ)
    if source_fault == "wrong-domain": hq["website"] = "https://other.example/"
    if source_fault == "wrong-profile": hq["url"] = "https://www.linkedin.com/company/other"
    # Do not let the earlier retained alias body carry a valid projection in negative controls.
    if source_fault:
        for p in sources.values(): p["text"] = p["text"].replace(HQ["text"], "")
    pages = scorer._investigator_prefetched_pages(
        sources, list(dict.fromkeys([*sources, *retry_pages])),
        matched_company_retry_pages=retry_pages,
        structured_profile_headquarters_evidence=hq,
        verified_identity=CASE["identity"], include_structured_headquarters=True,
    )
    assert len(pages) <= inv.MAX_PREFETCHED_PAGES
    if source_fault:
        assert all(HQ["text"] not in p["text"] for p in pages.values())
        if retry: assert pages == retry_pages
        return
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{"id": "retained-hq",
            "type": "function", "function": {"name": "submit_findings",
            "arguments": json.dumps({"findings": [CASE["finding"]]})}}]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(inv, "_post_json", post)
    fetch = AsyncMock(side_effect=AssertionError("No paid source fetch is needed"))
    search = AsyncMock(side_effect=AssertionError("No paid search is needed"))
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    monkeypatch.setattr(inv, "_search_web", search)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator=CASE["company_locator"], targets=("geography",),
        requested_geography=CASE["requested_geography"],
        prior_observations={"submitted_source_urls": list(pages)},
        verified_homepage_identity=CASE["identity"],
        prefetched_pages=pages, structured_headquarters_evidence=hq,
    ))
    assert result["claims"]["geography"]["status"] == "VERIFIED", result
    assert result["claims"]["geography"]["evidence_quote"] == HQ["text"]
    assert len(requests) == 1
    assert "complete bounded JSON" in "".join(m["content"] for m in requests[0]["messages"] if m["role"] == "system")
    assert fetch.await_count == search.await_count == 0
