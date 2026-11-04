"""Typed request refusals reuse the existing bounded exact-source recovery.

All provider/model replies are hermetic controls. These tests do not assert that
Exa can retrieve the live Coreworks page or measure a real model's accuracy.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from lab_arena import company_judgments, scoring as arena_scoring
from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import (
    _attribute_finding, _competition_company, _complete_verdict, _finding, _icp,
)
from tests.test_company_evidence_judgment_reserve import _Clock, _response


URL = "https://www.coreworks.ai/pricing"
FEATURES = "Coreworks supplies analysis and reporting software to customers."
SUBSCRIPTION = "Coreworks Pro includes analysis and reporting for $49 per month."
REFUSAL = json.dumps({"error": {"code": "provider_request_refused"}})
IDENTITY = {"normalized_name": "coreworks", "registrable_dns_domain": "coreworks.ai",
            "linkedin_company_slug": "coreworks-ai"}


def _install_loop(monkeypatch, *, status=403, body=REFUSAL, contents=None,
                  finding=None, urls=(URL,), case="", target="required_attribute",
                  company_name="Coreworks", website="https://www.coreworks.ai/",
                  identity=None):
    """Patch transports, not fetch/contents/validation or the investigator loop."""
    clock = _Clock()
    calls = {"primary": [], "contents": [], "model": [], "search": []}
    pending = list(urls)
    finding = finding or _attribute_finding()

    async def raw(_session, _url, **kwargs):
        url = kwargs.get("params", {}).get("url", _url)
        calls["primary"].append(url)
        if url != urls[-1]:
            return 200, url, FEATURES
        if case == "deadline":
            clock.now = c.ADMISSION_DEADLINE_SECONDS
        if case == "missing_key":
            monkeypatch.delenv("EXA_API_KEY")
        if case == "broker_exception":
            raise c._ArenaProviderFetchError("typed broker refusal", broker_status=403,
                                            broker_error_code="provider_request_refused")
        return status, url, body

    async def post(_session, url, *, headers, payload):
        if url == "https://api.exa.ai/contents":
            calls["contents"].append(payload)
            assert payload == {"ids": [urls[-1]],
                               "text": {"maxCharacters": c.MAX_PAGE_CHARACTERS},
                               "maxAgeHours": 0, "livecrawlTimeout": 20_000}
            if isinstance(contents, BaseException):
                raise contents
            return contents if contents is not None else (200, {
                "results": [{"url": urls[-1], "text": SUBSCRIPTION}]})
        calls["model"].append(payload)
        if pending:
            return 200, _response("fetch_page", {"url": pending.pop(0)})
        forced = payload.get("tool_choice")
        if isinstance(forced, dict) and forced["function"]["name"] == "search_web":
            return 200, _response("search_web", {"query": f"{company_name} software pricing"})
        return 200, _response("submit_findings", {
            "findings": finding if isinstance(finding, list) else [finding]})

    async def search(_session, query, **kwargs):
        calls["search"].append(query)
        # Force the ordinary final judgment reserve, without another loop.
        clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
        return {"results": [], "notice": "discovery_only_not_evidence"}

    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    if case == "no_scrapingdog_key":
        monkeypatch.delenv("SCRAPINGDOG_API_KEY")
        monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(c, "_fetch_bounded_html", raw)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_search_web", search)
    monkeypatch.setattr(c, "time", clock)

    async def run(**kwargs):
        return await c.investigate_company_evidence(
            company_locator={"name": company_name, "website": website},
            targets=(target,), requested_industry="Software",
            requested_attribute="Supplies analysis and reporting software AND sells recurring subscriptions.",
            verified_homepage_identity=identity or IDENTITY,
            positive_semantic_review=True, **kwargs)

    return run, calls, clock


@pytest.mark.parametrize("case", ["", "broker_exception"])
def test_coreworks_shaped_third_fetch_refusal_uses_last_slot_and_normal_validation(monkeypatch, case):
    urls = ("https://www.coreworks.ai/product", "https://www.coreworks.ai/about", URL)
    run, calls, _ = _install_loop(monkeypatch, urls=urls, case=case,
        finding=_attribute_finding("VERIFIED", url=URL, quote=SUBSCRIPTION))
    result = asyncio.run(run())
    assert calls["primary"] == list(urls)
    assert len(calls["contents"]) == 1
    assert result["_completed_submit"] is True
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["evidence_quote"] == SUBSCRIPTION
    assert result["usage"]["fetch_calls"] == c.MAX_FETCH_CALLS == 4
    outcomes = result["usage"]["fetch_outcomes"]
    assert [(o["ok"], o["error_class"]) for o in outcomes] == [
        (True, ""), (True, ""), (False, "provider_request_refused"), (True, "")]
    assert outcomes[-2]["url_sha256"] == outcomes[-1]["url_sha256"] == hashlib.sha256(URL.encode()).hexdigest()
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == SUBSCRIPTION


@pytest.mark.parametrize("case", ["features_only", "unsupported_pricing", "wrong_company", "quote_absent"])
def test_retrieval_does_not_supply_missing_company_or_subscription_proof(monkeypatch, case):
    url = "https://evidence.example/coreworks-pricing" if case == "wrong_company" else URL
    text = (
        "Other Company supplies analysis software for $49 per month." if case == "wrong_company"
        else FEATURES + " Coreworks sells a one-time report for $49." if case == "unsupported_pricing"
        else FEATURES)
    quote = text if case == "wrong_company" else SUBSCRIPTION
    # Semantic negative replies remain negative even after successful retrieval.
    # Invalid positive replies must fail the ordinary identity and quote checks.
    finding = (_attribute_finding() if case in {"features_only", "unsupported_pricing"}
               else _attribute_finding("VERIFIED", url=url, quote=quote))
    run, calls, _ = _install_loop(monkeypatch, urls=(url,), finding=finding,
        contents=(200, {"results": [{"url": url, "text": text}]}))
    result = asyncio.run(run())
    assert len(calls["contents"]) == 1
    claim = result["claims"]["required_attribute"]
    assert claim["status"] == "UNPROVEN"
    assert claim["evidence_url"] == claim["evidence_quote"] == ""
    if case in {"features_only", "unsupported_pricing"}:
        assert result["_completed_submit"] is True
        assert result["failure_reason"] == ""
    else:
        assert result["_completed_submit"] is False
        assert result["failure_reason"] == "malformed_response"
        assert claim["reason"] == ("source quote did not identify the investigated company"
                                  if case == "wrong_company" else "submitted quote was not present in fetched source")


@pytest.mark.parametrize("status,body,expected", [
    (403, "Forbidden", "http_error"),
    (403, '{"error":', "http_error"),
    (403, '{"error":"provider_request_refused"}', "http_error"),
    (403, '{"error":{"code":"provider_request_refused_extra"}}', "http_error"),
    (401, REFUSAL, "http_error"), (429, REFUSAL, "http_error"),
    (404, REFUSAL, "source_not_found"), (410, REFUSAL, "source_gone"),
])
def test_other_statuses_and_unvalidated_refusals_do_not_gain_fallback(monkeypatch, status, body, expected):
    run, calls, _ = _install_loop(monkeypatch, status=status, body=body)
    result = asyncio.run(run())
    assert calls["primary"] == [URL]
    assert calls["contents"] == []
    assert result["usage"]["fetch_outcomes"][0]["error_class"] == expected
    assert result["_completed_submit"] is (status in {404, 410})


def test_refusal_body_without_provider_credentials_is_not_trusted(monkeypatch):
    run, calls, _ = _install_loop(monkeypatch, case="no_scrapingdog_key")
    result = asyncio.run(run())
    assert calls["primary"] == [URL] and not calls["contents"]
    assert result["usage"]["fetch_outcomes"][0]["error_class"] == "http_error"
    assert result["failure_reason"] == "provider_error"


@pytest.mark.parametrize("case", ["missing_key", "deadline", "cap"])
def test_refusal_fallback_respects_existing_admission_limits(monkeypatch, case):
    urls = tuple(f"https://www.coreworks.ai/source{i}" for i in range(3)) + (URL,) if case == "cap" else (URL,)
    run, calls, _ = _install_loop(monkeypatch, case=case, urls=urls)
    result = asyncio.run(run())
    assert calls["primary"] == list(urls)
    assert not calls["contents"]
    assert result["usage"]["fetch_calls"] == len(urls) <= c.MAX_FETCH_CALLS
    assert result["usage"]["fetch_outcomes"][-1]["error_class"] == "provider_request_refused"
    if case == "cap":
        assert result["_completed_submit"] is True


@pytest.mark.parametrize("contents,error", [
    ((401, {}), "http_error"), ((403, {}), "http_error"), ((429, {}), "http_error"),
    ((503, {}), "http_error"), (asyncio.TimeoutError(), "fetch_failed"),
    ((200, {"results": [{"url": "https://other.example/pricing", "text": SUBSCRIPTION}]}), "fetch_failed"),
    ((200, {"results": [{"url": URL, "text": SUBSCRIPTION}] * 2}), "fetch_failed"),
    ((200, {}), "fetch_failed"), ((200, {"results": []}), "fetch_failed"),
    ((200, {"error": "unavailable"}), "fetch_failed"),
    ((200, {"results": [{"url": URL, "text": ""}]}), "fetch_failed"),
    ((200, {"results": [{"url": URL, "text": "Provider account capacity details redacted."}]}), "fetch_failed"),
    ((200, {"results": [{"url": URL, "text": "%PDF-1.7 binary content"}]}), "unsupported_binary_content"),
    ((200, {"results": [{"url": URL, "error": "crawl failed", "text": SUBSCRIPTION}]}), "fetch_failed"),
])
def test_failed_contents_remains_latest_source_failure_and_does_not_recurse(monkeypatch, contents, error):
    run, calls, _ = _install_loop(monkeypatch, contents=contents)
    result = asyncio.run(run())
    assert calls["primary"] == [URL] and len(calls["contents"]) == 1
    assert result["usage"]["fetch_calls"] == 2
    outcomes = result["usage"]["fetch_outcomes"]
    assert [o["error_class"] for o in outcomes] == ["provider_request_refused", error]
    assert outcomes[0]["url_sha256"] == outcomes[1]["url_sha256"]
    assert result["failure_reason"] == "provider_error"
    assert result["_completed_submit"] is False
    assert URL not in result[c.PRIVATE_FETCHED_PAGES_KEY]


def test_exact_target_crawl_denial_stays_company_local_after_refusal(monkeypatch):
    run, calls, _ = _install_loop(monkeypatch, contents=(200, {"results": [], "statuses": [{
        "id": URL, "status": "error", "error": {"httpStatusCode": 403, "tag": "CRAWL_HTTP_403"}}]}))
    result = asyncio.run(run())
    assert len(calls["contents"]) == 1
    assert result["usage"]["fetch_outcomes"][-1]["error_class"] == "source_access_denied"
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""


def test_linkedin_refusal_keeps_specialized_transport(monkeypatch):
    profile = "https://www.linkedin.com/company/coreworks-ai"
    run, calls, _ = _install_loop(monkeypatch, urls=(profile,))

    # A validated transport refusal is injected at the shared fetch boundary;
    # the actual loop must still exclude strict LinkedIn company profile URLs.
    async def refused(*_args, **_kwargs):
        calls["primary"].append(profile)
        return {"ok": False, "error": "provider_request_refused"}

    monkeypatch.setattr(c, "_fetch_page", refused)
    result = asyncio.run(run())
    assert calls["primary"] == [profile]
    assert not calls["contents"]
    assert result["usage"]["fetch_calls"] == 1


def test_archive_refusal_preserves_visible_card_recovery(monkeypatch):
    homepage, archive = "https://example.com/", "https://example.com/news"
    article, investor = archive + "/investment-in-acme", "https://investor.example/acme-series-b"
    _, calls, _ = _install_loop(monkeypatch, urls=(article, archive), target="stage",
        finding=_finding("stage", status="UNPROVEN", observed_value=None,
                         evidence_url="", evidence_quote=""))

    async def fetch(_session, url, *, archive_targets=(), archive_company_names=()):
        calls["primary"].append((url, tuple(archive_targets)))
        if url == archive:
            assert archive_targets == (article,)
            return {"ok": False, "error": "provider_request_refused"}
        return {"ok": True, "url": url, "final_url": url,
                "text": "Acme received a minority investment."}

    monkeypatch.setattr(c, "_fetch_page", fetch)
    async def search(*_args, **_kwargs):
        return {"results": [{"url": article}]}
    monkeypatch.setattr(c, "_search_web", search)
    # Reuse the actual archive-selection path used by the existing diagnostic
    # controls. Its loaded archive context cannot be replaced by plain contents.
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={"name": "Acme", "website": homepage}, targets=("stage",),
        requested_stage="Series B", prior_observations={"submitted_source_urls": [homepage, investor]},
        verified_homepage_identity={},
        verified_navigation_identity={"normalized_name": "Acme", "registrable_dns_domain": "example.com", "linkedin_company_slug": "acme"},
        homepage_navigation_locators=[{"url": archive, "label": "News"}],
        prefetched_pages={homepage: {"final_url": homepage, "text": "Acme platform."},
            investor: {"final_url": investor, "text": "Acme completed its Series B in December 2025."}}))
    assert calls["primary"] == [(article, ()), (archive, (article,))]
    assert not calls["contents"]
    assert result["usage"]["fetch_calls"] == 2


@pytest.mark.parametrize("contents", [
    (401, {}), (403, {}), (429, {}), (503, {}), asyncio.TimeoutError(),
    (200, {"results": [{"url": "https://other.example/pricing", "text": SUBSCRIPTION}]}),
    (200, {"results": []}),
    (200, {"results": [{"url": URL, "text": ""}]}),
    (200, {"results": [{"url": URL, "text": "%PDF-1.7 binary content"}]}),
])
def test_real_scorer_retries_failed_fallback_without_publishing_cached_zero(monkeypatch, contents):
    product = "https://acme.example/student-plans"
    product_quote = "Acme supplies schools with student planning software."
    investigations, published = [], []
    real_investigate = c.investigate_company_evidence

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match("homepage identity verified", details={
            "identity": {"decision": "match", "evidence_source": "company_homepage",
                         "observed_name": "acme", "observed_domain": "acme.example",
                         "observed_linkedin_slug": "acme"},
            "verified_homepage_transport_domain": "acme.example"})

    async def provider(**_kwargs):
        return _complete_verdict(
            observed_industry="Education", observed_subindustry="school planning software",
            industry_evidence_url=product, industry_evidence_quote=product_quote,
            attribute_satisfied=True, required_attribute_evidence_url=product,
            required_attribute_evidence_quote=product_quote), ""

    async def grounding(_session, url):
        assert url == product
        return 200, url, product_quote

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    async def investigate(**kwargs):
        assert kwargs["targets"] == ("industry", "required_attribute")
        _, calls, _ = _install_loop(monkeypatch, contents=contents, finding=[
            _finding("industry", status="UNPROVEN", observed_value=None,
                     evidence_url="", evidence_quote=""), _attribute_finding()])
        result = await real_investigate(**kwargs)
        investigations.append(result)
        assert calls["primary"] == [URL] and len(calls["contents"]) == 1
        assert result["_completed_submit"] is False
        assert result["failure_reason"] == "provider_error"
        assert result["usage"]["fetch_outcomes"][-1]["ok"] is False
        return result

    async def no_intent(*_args, **_kwargs):
        raise AssertionError("Provider failure cannot become a scored negative or positive")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", grounding)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", keep_observation)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", no_intent)
    policy = arena_scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2", company_quality=True)
    scorer = arena_scoring.lab_scorer(policy)
    icp = _icp(industry="Education", sub_industry="school planning software",
        product_service="student planning software used by schools",
        required_attribute="Sells student planning software to schools.",
        intent_signals=["Announced a completed funding event"]).model_dump(mode="json")
    companies = [{**_competition_company(), "industry": "Education"}]
    document = arena_scoring.build_scoring_input(
        scored_run_id="refusal-fallback-failed", icp=icp, companies=companies,
        policy=policy, evaluation_date="2026-10-22")
    refs = company_judgments.build_company_scopes(
        scoring_input=document, round_id="arena-2026-10-22", network_name="finney", netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1", company_quality_policy="company_quality_v1")
    lease = {"schema_version": company_judgments.LEASE_SCHEMA_VERSION, "hits": [], "misses": [{
        "company_index": ref["company_index"], "cache_key": ref["cache_key"],
        "company_input_hash": ref["company_input_hash"], "authority_slot": 0,
    } for ref in refs]}
    with pytest.raises(arena_scoring.ScoringError, match="judge_exhausted"):
        arena_scoring.score_quality_work_item(
            {"scored_run_id": "refusal-fallback-failed"}, icp=icp, companies=companies,
            scorer=scorer, cache_context=lease, on_company_judgments=published.append)
    assert len(investigations) == arena_scoring.MAX_JUDGE_RETRIES
    assert published == []
