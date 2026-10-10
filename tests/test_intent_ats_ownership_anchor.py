"""Employer ownership needs an observed careers relationship, not name overlap."""
import asyncio
from unittest import mock

import pytest

from qualification.scoring import intent_verification_three_stage as intent

ASHBY = "https://jobs.ashbyhq.com/fluency/c6aceb26-9658-48c2-ac07-6cdf8bbb38df"
GH = "https://job-boards.greenhouse.io/customerio/jobs/8055585"
IDENTITY = {"observed_domain": "usefluency.com", "observed_linkedin_slug": "usefluency"}


def page(url, html):
    return {"url": url, "text": "Fetched page text", "meta": {
        "observed_ownership_links": intent._observed_ownership_links(html, url)}}


@pytest.mark.parametrize("target, board", [
    (ASHBY, "https://jobs.ashbyhq.com/fluency"),
    (GH, "https://job-boards.greenhouse.io/customerio"),
    ("https://jobs.lever.co/acme/c6aceb26-9658-48c2-ac07-6cdf8bbb38df", "https://jobs.lever.co/acme"),
])
def test_official_property_observed_careers_link_binds_exact_board(target, board):
    result = page("https://usefluency.com/", f'<a href="{board}">Careers</a>')
    anchors = intent._ats_ownership_anchors([result], [target], IDENTITY)
    assert [a["posting_url"] for a in anchors] == [target]
    assert anchors[0]["kind"] == "official_careers_link"


@pytest.mark.parametrize("url, html", [
    (ASHBY, '<p>Fluency uses usefluency.com software.</p>'),
    (ASHBY, '<a href="https://usefluency.com/docs">Customer docs</a>'),
    (ASHBY, '<a href="https://usefluency.com/">Powered by Fluency</a>'),
    ("https://unrelated.example/", '<a href="https://jobs.ashbyhq.com/fluency">Careers</a>'),
    ("https://usefluency.com/", '<a href="https://jobs.ashbyhq.com/another">Careers</a>'),
    ("https://usefluency.com/", '<a href="https://jobs.ashbyhq.com/fluency">Powered by</a>'),
    ("https://usefluency.com/", '<p>Our tenant is fluency.</p>'),
    ("https://usefluency.com/", '<a hidden href="https://jobs.ashbyhq.com/fluency">Careers</a>'),
])
def test_mentions_tenant_wrong_property_and_unrelated_links_are_not_ownership(url, html):
    assert intent._ats_ownership_anchors([page(url, html)], [ASHBY], IDENTITY) == []


def test_job_body_direct_official_careers_link_is_supported():
    result = page(GH, '<p>Benefits at Customer.io:</p><a href="https://customer.io/careers#benefits">Benefits</a>')
    anchors = intent._ats_ownership_anchors([result], [GH], {"observed_domain": "customer.io"})
    assert anchors[0]["kind"] == "job_official_careers_link"


def test_observed_links_are_bounded_and_invalid_urls_are_not_admitted():
    body = '<a href="javascript:alert(1)">Careers</a>' + ''.join(
        f'<a href="https://usefluency.com/docs/{i}">Docs</a>' for i in range(100))
    links = intent._observed_ownership_links(body, ASHBY)
    assert len(links) == 80
    assert all(link["url"].startswith("https://") for link in links)


def test_reuses_fetched_official_evidence_without_another_request():
    contents = {"results": [page("https://usefluency.com/", '<a href="https://jobs.ashbyhq.com/fluency">Careers</a>')], "statuses": []}
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock()) as fetch:
        receipt = asyncio.run(intent._resolve_ats_employer_ownership(contents, [ASHBY], IDENTITY, [ASHBY, "https://usefluency.com/"]))
    fetch.assert_not_awaited()
    assert receipt["resolved"] is True
    assert receipt["source_count"] == 2


def test_follows_only_observed_official_careers_link_within_three_sources():
    home = page("https://usefluency.com/", '<a href="/careers">Careers</a>')
    careers = page("https://usefluency.com/careers", '<a href="https://jobs.ashbyhq.com/fluency">Open positions</a>')
    responses = [{"results": [home], "statuses": []}, {"results": [careers], "statuses": []}]
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock(side_effect=responses)) as fetch:
        receipt = asyncio.run(intent._resolve_ats_employer_ownership({"results": [], "statuses": []}, [ASHBY], IDENTITY, [ASHBY]))
    assert fetch.await_args_list == [mock.call(["https://usefluency.com/"]), mock.call(["https://usefluency.com/careers"])]
    assert receipt["resolved"] is True
    assert receipt["source_count"] == 3


def test_existing_three_source_allowance_cannot_be_extended():
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock()) as fetch:
        receipt = asyncio.run(intent._resolve_ats_employer_ownership({"results": [], "statuses": []}, [ASHBY], IDENTITY, [ASHBY, "https://usefluency.com/about", "https://usefluency.com/news"]))
    fetch.assert_not_awaited()
    assert receipt == {"anchors": [], "source_count": 3, "resolved": False}


def test_absent_link_does_not_guess_career_route_or_follow_other_property():
    home = page("https://usefluency.com/", '<a href="https://other.example/careers">Careers</a>')
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock(return_value={"results": [home], "statuses": []})) as fetch:
        receipt = asyncio.run(intent._resolve_ats_employer_ownership({"results": [], "statuses": []}, [ASHBY], IDENTITY, [ASHBY]))
    fetch.assert_awaited_once_with(["https://usefluency.com/"])
    assert receipt["resolved"] is False
    assert receipt["source_count"] == 2


def test_reuses_fetched_homepage_observed_careers_route():
    home = page("https://usefluency.com/", '<a href="/careers">Careers</a>')
    careers = page("https://usefluency.com/careers", '<a href="https://jobs.ashbyhq.com/fluency">Open positions</a>')
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock(return_value={"results": [careers], "statuses": []})) as fetch:
        receipt = asyncio.run(intent._resolve_ats_employer_ownership({"results": [home], "statuses": []}, [ASHBY], IDENTITY, [ASHBY, "https://usefluency.com/"]))
    fetch.assert_awaited_once_with(["https://usefluency.com/careers"])
    assert receipt["resolved"] is True
    assert receipt["source_count"] == 3


def test_fetched_markdown_careers_links_are_reused():
    result = {"url": "https://usefluency.com/careers", "text": "[Open positions](https://jobs.ashbyhq.com/fluency)"}
    assert intent._ats_ownership_anchors([result], [ASHBY], IDENTITY)


@pytest.mark.parametrize("board", [
    "https://boards.greenhouse.io/customerio",
    "https://job-boards.greenhouse.io/customerio",
])
def test_existing_greenhouse_host_variants_keep_same_observed_board(board):
    official = page("https://customer.io/careers", f'<a href="{board}">Open roles</a>')
    assert intent._ats_ownership_anchors([official], [GH], {"observed_domain": "customer.io"})


def test_greenhouse_host_normalization_does_not_merge_different_tenants():
    official = page("https://customer.io/careers", '<a href="https://boards.greenhouse.io/another">Open roles</a>')
    assert intent._ats_ownership_anchors([official], [GH], {"observed_domain": "customer.io"}) == []


def test_greenhouse_regions_are_separate_board_namespaces():
    official = page("https://customer.io/careers", '<a href="https://job-boards.eu.greenhouse.io/customerio">Open roles</a>')
    assert intent._ats_ownership_anchors([official], [GH], {"observed_domain": "customer.io"}) == []
    eu_target = GH.replace("job-boards.greenhouse", "job-boards.eu.greenhouse")
    assert intent._ats_ownership_anchors([official], [eu_target], {"observed_domain": "customer.io"})


def test_hidden_markdown_inside_html_is_not_an_observed_link():
    result = page("https://usefluency.com/", '<script>"[Careers](https://jobs.ashbyhq.com/fluency)"</script>')
    assert intent._ats_ownership_anchors([result], [ASHBY], IDENTITY) == []


def test_powered_by_label_is_not_job_employer_ownership():
    result = page(ASHBY, '<a href="https://usefluency.com/careers">Powered by Fluency</a>')
    assert intent._ats_ownership_anchors([result], [ASHBY], IDENTITY) == []


def test_retained_customerio_text_fragment_is_not_part_of_resource_identity():
    href = 'https://customer.io/careers#:~:text=our%20collective%20success.-,BENEFITS,-Our%C2%A0'
    result = page(GH, f'<a href="{href}">See full benefits here</a>')
    assert result["meta"]["observed_ownership_links"][0]["url"] == "https://customer.io/careers"
    assert intent._ats_ownership_anchors([result], [GH], {"observed_domain": "customer.io"})


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("snapshot_source, expected", [
    ("https://usefluency.com/", True),
    ("https://unrelated.example/", False),
    ("https://usefluency.com/%0Aunsafe", False),
])
def test_wayback_ownership_links_require_exact_official_snapshot_provenance(snapshot_source, expected, relative):
    import httpx
    snapshot = "https://web.archive.org/web/20260721211653/" + snapshot_source
    href = ("" if relative else "http://web.archive.org") + "/web/20260721211653id_/https://jobs.ashbyhq.com/fluency"
    body = '<html><body><p>' + ('Company information. ' * 40) + f'</p><a href="{href}">Careers</a></body></html>'
    def reply(request):
        if request.url.host == "archive.org":
            return httpx.Response(200, json={"archived_snapshots": {"closest": {"url": snapshot}}})
        return httpx.Response(200, text=body)
    original = httpx.AsyncClient
    with mock.patch.object(intent.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(reply), **kwargs)):
        fetched = asyncio.run(intent._try_wayback("https://usefluency.com/"))
    assert fetched["ok"] is True
    result = {"url": "https://usefluency.com/", "text": fetched["content"], "meta": fetched["meta"]}
    anchors = intent._ats_ownership_anchors([result], [ASHBY], IDENTITY)
    assert bool(anchors) is expected
    if expected:
        assert anchors[0]["archive_url"] == snapshot
        assert anchors[0]["linked_url"] == "https://jobs.ashbyhq.com/fluency"


def test_normal_page_cannot_unwrap_wayback_link_into_ownership():
    result = page("https://usefluency.com/", '<a href="https://web.archive.org/web/20260721211653/https://jobs.ashbyhq.com/fluency">Careers</a>')
    assert intent._ats_ownership_anchors([result], [ASHBY], IDENTITY) == []


@pytest.mark.parametrize("citation, expected", [
    ("https://usefluency.com/careers/platform-engineer", "approve"),
    (ASHBY, "review"),
    ("https://news.example/hiring", "review"),
    ("https://www.linkedin.com/jobs/view/1234567890/", "approve"),
])
def test_unused_ats_does_not_block_other_grounded_job_evidence(citation, expected):
    official = "https://usefluency.com/careers/platform-engineer"
    alternate = citation if citation not in {ASHBY, official} else official
    urls = [ASHBY, alternate]
    contents = {"results": [{"url": url, "text": "We're hiring a platform engineer. Responsibilities: build and operate our platform.",
        "meta": {"kind": "linkedin_job"} if "linkedin.com/jobs/" in url else {}}
        for url in urls], "statuses": []}
    response = {"model": "test", "usage": {}, "answer": {"overall_verdict": "qualified", "overall_confidence": "high", "signal_evaluations": [{
        "signal_status": "supported", "confidence": "high", "same_entity_check": "pass", "verification_mode": "source_grounded",
        "evidence_urls_used": [citation], "claim_matches_miner_date": "no_date_in_content", "source_accessibility": "accessible",
        "supporting_quotes": ["We're hiring a platform engineer."], "contradicting_quotes": [], "unsupported_parts": [], "risk_notes": []}]}}
    async def fetch(requested, **_kwargs):
        return contents if requested == urls else {"results": [], "statuses": []}
    with mock.patch.object(intent, "_fetch_sd_then_exa", mock.AsyncMock(side_effect=fetch)), mock.patch.object(intent, "_call_openrouter", mock.AsyncMock(return_value=response)):
        result = asyncio.run(intent.verify_three_stage(None, company_name="Fluency", company_website="https://usefluency.com", company_linkedin="https://www.linkedin.com/company/usefluency",
            source_url=ASHBY, miner_claim="We're hiring a platform engineer.", target_signal_text="Hiring platform engineers", evidence_type="HIRING", integrity_policy=True, company_quality=False, stage1_soft_reject=True,
            verified_company_identity={"decision": "match", "observed_name": "fluency", "observed_domain": "usefluency.com", "observed_linkedin_slug": "usefluency", "evidence_source": "company_web_reverification"},
            evidence_bundle=[{"url": url, "description": "We're hiring a platform engineer.", "snippet": "We're hiring a platform engineer."} for url in urls]))
    assert result["decision"] == expected
    assert result["client_ready"] is (expected == "approve")
