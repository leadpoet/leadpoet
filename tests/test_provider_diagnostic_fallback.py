"""Known diagnostic pages use the existing exact-source fallback budget."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _finding

URL = "https://acme.example/product"
QUOTE = "Acme supplies workflow software for customers."
# Exact canonical body captured from host-funded Firecrawl requests.
DIAGNOSTIC = "Provider account capacity details redacted."


@pytest.mark.parametrize("case", ["valid", "wrong_url", "empty", "diagnostic", "fallback_failure", "fallback_timeout",
    "401", "403", "429", "deadline", "missing_key", "502", "503", "primary_timeout"])
def test_actual_diagnostic_and_exact_source_recovery(monkeypatch, case):
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    primary, fallback = [], []
    success = case in {"valid", "502", "503", "primary_timeout"}

    async def fetched(_session, url, **kwargs):
        target = kwargs["params"]["url"]
        primary.append(target)
        if case == "deadline": clock.now = c.ADMISSION_DEADLINE_SECONDS
        if case == "missing_key": monkeypatch.delenv("EXA_API_KEY")
        if case == "primary_timeout": raise asyncio.TimeoutError()
        return (int(case) if case in {"401", "403", "429", "502", "503"} else 200), target, DIAGNOSTIC

    async def post(_session, url, *, headers, payload):
        if url == "https://api.exa.ai/contents":
            assert payload["ids"] == [URL]
            assert payload["maxAgeHours"] == 0
            fallback.append(payload["ids"][0])
            if case == "fallback_timeout": raise asyncio.TimeoutError()
            if case == "fallback_failure": return 503, {}
            return 200, {"results": [{"url": "https://other.example/product" if case == "wrong_url" else URL,
                "text": "" if case == "empty" else DIAGNOSTIC if case == "diagnostic" else QUOTE}]}
        forced = payload.get("tool_choice")
        if isinstance(forced, dict) and forced["function"]["name"] == "search_web":
            name, arguments = "search_web", {"query": "Acme workflow software"}
        else:
            name, arguments = "submit_findings", {"findings": [_finding("industry", observed_value=None,
                observed_industry="Software", observed_subindustry="SaaS", activity_role="supplier_operator",
                status="VERIFIED" if success else "UNPROVEN", evidence_url=URL if success else "",
                evidence_quote=QUOTE if success else "", reason="Reviewed the requested company source.")]}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": "test-call", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    async def search(*args, **kwargs): return {"results": []}
    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(c, "_fetch_bounded_html", fetched)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_search_web", search)
    monkeypatch.setattr(c, "time", clock)
    result = asyncio.run(c.investigate_company_evidence(company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("industry",), requested_industry="Software", positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example", "linkedin_company_slug": "acme"}))
    assert primary == [URL]
    allowed = case not in {"401", "403", "429", "deadline", "missing_key"}
    assert fallback == ([URL] if allowed else [])
    assert result.get("_completed_submit") is success, result
    if success:
        assert result["claims"]["industry"]["status"] == "VERIFIED"
        assert result["claims"]["industry"]["evidence_quote"] == QUOTE
        assert result["usage"]["fetch_calls"] == 2
        outcomes = result["usage"]["fetch_outcomes"]
        assert outcomes[0]["url_sha256"] == outcomes[1]["url_sha256"]
        assert outcomes[0]["ok"] is False and outcomes[1]["ok"] is True
    else:
        assert result["failure_reason"] == ("admission_budget_interrupted" if case == "deadline" else "provider_error")


def _tool(name, arguments, index):
    return 200, {"choices": [{"message": {"tool_calls": [{"id": str(index), "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def test_diagnostic_on_final_admitted_fetch_cannot_exceed_existing_cap(monkeypatch):
    calls, fallbacks = [], []
    turn = 0
    async def post(_session, _url, *, headers, payload):
        nonlocal turn
        turn += 1
        if turn <= 4: return _tool("fetch_page", {"url": f"https://acme.example/source{turn}"}, turn)
        if turn == 5: return _tool("search_web", {"query": "Acme workflow software"}, turn)
        return _tool("submit_findings", {"findings": [_finding("industry", status="UNPROVEN", evidence_url="", evidence_quote="")]}, turn)
    async def fetch(_session, url, **kwargs):
        calls.append(url)
        return {"ok": False, "error": "provider_diagnostic_body"} if len(calls) == 4 else {"ok": True, "url": url, "final_url": url, "text": QUOTE}
    async def fallback(*args, **kwargs):
        fallbacks.append(args)
        raise AssertionError("Fallback cannot exceed four fetch calls")
    async def search(*args, **kwargs): return {"results": []}
    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY"): monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    monkeypatch.setattr(c, "_fetch_exa_exact_page", fallback)
    monkeypatch.setattr(c, "_search_web", search)
    result = asyncio.run(c.investigate_company_evidence(company_locator={"name": "Acme", "website": "https://acme.example/"}, targets=("industry",)))
    assert len(calls) == c.MAX_FETCH_CALLS == result["usage"]["fetch_calls"] == 4
    assert not fallbacks and result["failure_reason"] == "provider_error"


def test_linkedin_profile_keeps_existing_specialized_transport(monkeypatch):
    profile = "https://www.linkedin.com/company/acme"
    turn, fallback = 0, []
    async def current(url, **kwargs):
        assert url == profile
        kwargs["source_text_sink"].update(url=profile, text=DIAGNOSTIC)
        return {"url": profile, "employee_count": "11-50", "quote": "11-50 employees"}
    async def post(_session, url, *, headers, payload):
        nonlocal turn
        turn += 1
        if turn == 1: return _tool("fetch_page", {"url": profile}, turn)
        if turn == 2: return _tool("search_web", {"query": "Acme software"}, turn)
        return _tool("submit_findings", {"findings": [_finding("industry", status="UNPROVEN", evidence_url="", evidence_quote="")]}, turn)
    async def no_fallback(*args, **kwargs): fallback.append(args); raise AssertionError("Specialized LinkedIn path")
    async def search(*args, **kwargs): return {"results": []}
    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY"): monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(c, "fetch_current_linkedin_company_size", current)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_fetch_exa_exact_page", no_fallback)
    monkeypatch.setattr(c, "_search_web", search)
    result = asyncio.run(c.investigate_company_evidence(company_locator={"name": "Acme", "website": "https://acme.example/"}, targets=("industry",)))
    assert not fallback and result["usage"]["fetch_calls"] == 1
    assert result["failure_reason"] == "provider_error"


def test_archive_context_does_not_use_plain_contents_fallback(monkeypatch):
    homepage, archive = "https://example.com/", "https://example.com/news"
    article, investor = archive + "/investment-in-acme", "https://investor.example/acme-series-b"
    calls, fallbacks = [], []
    turn = 0
    async def fetch(_session, url, *, archive_targets=(), archive_company_names=()):
        calls.append((url, tuple(archive_targets)))
        if url == archive:
            assert archive_targets == (article,)
            return {"ok": False, "error": "provider_diagnostic_body"}
        return {"ok": True, "url": url, "final_url": url, "text": "Acme received a minority investment."}
    async def post(_session, _url, *, headers, payload):
        nonlocal turn
        turn += 1
        if turn <= 2: return _tool("fetch_page", {"url": (article, archive)[turn - 1]}, turn)
        return _tool("submit_findings", {"findings": [_finding("stage", status="UNPROVEN", evidence_url="", evidence_quote="")]}, turn)
    async def search(*args, **kwargs): return {"results": [{"url": article}]}
    async def fallback(*args, **kwargs): fallbacks.append(args); raise AssertionError("Archive evidence needs its existing context")
    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY"): monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(c, "_fetch_page", fetch)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_search_web", search)
    monkeypatch.setattr(c, "_fetch_exa_exact_page", fallback)
    result = asyncio.run(c.investigate_company_evidence(company_locator={"name": "Acme", "website": homepage},
        targets=("stage",), requested_stage="Series B",
        prior_observations={"submitted_source_urls": [homepage, investor]}, verified_homepage_identity={},
        verified_navigation_identity={"normalized_name": "Acme", "registrable_dns_domain": "example.com", "linkedin_company_slug": "acme"},
        homepage_navigation_locators=[{"url": archive, "label": "News"}],
        prefetched_pages={homepage: {"final_url": homepage, "text": "Acme platform."},
            investor: {"final_url": investor, "text": "Acme completed its Series B in December 2025."}}))
    assert calls == [(article, ()), (archive, (article,))]
    assert not fallbacks and result["usage"]["fetch_calls"] == 2
    assert result["failure_reason"] == "provider_error"
