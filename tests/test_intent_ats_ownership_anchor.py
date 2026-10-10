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
