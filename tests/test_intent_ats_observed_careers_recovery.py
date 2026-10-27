"""A third source must be an observed listing, not another navigation portal.

The homepage link fixture comes from retained October 26 bytes. The listing
fixture reflects October 27 rendered DOM. These tests prove bounded retrieval
and ownership, not historical hiring qualification or a model judgment.
"""

import asyncio
import html
import json
from pathlib import Path

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.verification_helpers import visible_html_link_labels, visible_html_text


FIXTURE = json.loads((Path(__file__).parent / "fixtures/baringa_ats_observed_links.json").read_text())
TARGET = FIXTURE["submitted_posting_url"]
HOME = FIXTURE["homepage_capture"]["request_url"]
LISTING = FIXTURE["observed_listing_url"]
IDENTITY = {"observed_name": "Baringa", "observed_domain": "baringa.com",
            "observed_linkedin_slug": "baringa"}


def home_html():
    links = "".join(
        f'<a href="{html.escape(row["href"], quote=True)}">{html.escape(row["label"])}</a>'
        for row in FIXTURE["homepage_visible_links"]
    )
    return '<html><body>' + links + '<main><p>' + (
        'Baringa provides management consulting services to corporate clients. ' * 12
    ) + '</p></main></body></html>'


def listing_html(*, tenant="baringa", region="eu", hidden=False):
    links = "".join(
        f'<a {"hidden" if hidden else ""} href="{row["href"].replace("/baringa/", "/" + tenant + "/").replace(".eu.", "." + region + ".")}" '
        f'aria-label="{html.escape(row["aria_label"], quote=True)}"><svg></svg></a>'
        for row in FIXTURE["rendered_links"]
    )
    return '<html><body><h1>Open positions</h1>' + links + '<p>' + (
        'We are recruiting consultants to deliver transformation programmes. ' * 32
    ) + '</p></body></html>'


def test_retained_homepage_does_not_lose_careers_after_eighty_other_links():
    links = intent._observed_ownership_links(home_html(), HOME)
    assert len(links) <= 80
    results = [{"url": HOME, "text": "", "meta": {"observed_ownership_links": links}}]
    assert intent._observed_official_careers_url(results, IDENTITY, [intent._normalize_url(TARGET), intent._normalize_url(HOME)]) == LISTING


def test_visible_accessible_link_label_is_available_without_becoming_body_text():
    raw = '<a href="' + TARGET + '" aria-label="Apply for Consultant position"><svg></svg></a>'
    assert visible_html_link_labels(raw) == ((TARGET, "Apply for Consultant position"),)
    assert visible_html_text(raw) == ""


@pytest.mark.parametrize("wrapper", [
    '<div hidden>{}</div>', '<div aria-hidden="true">{}</div>',
    '<div style="display:none">{}</div>', '<script>{}</script>',
])
def test_hidden_accessible_labels_are_never_discovery_or_ownership(wrapper):
    link = '<a href="' + TARGET + '" aria-label="Apply for Consultant position"></a>'
    assert visible_html_link_labels(wrapper.format(link)) == ()


def test_visible_label_cannot_be_overridden_by_an_accessible_apply_claim():
    raw = '<a href="' + TARGET + '" aria-label="Apply for Consultant">Powered by</a>'
    assert visible_html_link_labels(raw) == ((TARGET, "Powered by"),)


def test_markdown_routes_are_prioritized_before_the_same_eighty_link_cap():
    body = "\n".join(f"[Docs](https://baringa.com/docs/{i})" for i in range(100))
    body += f"\n[Open positions]({LISTING})"
    links = intent._observed_ownership_links(body, HOME)
    assert len(links) == 80 and links[0]["url"] == LISTING
    too_late = "\n".join(f"[Docs](https://baringa.com/docs/{i})" for i in range(500))
    assert all(row["url"] != LISTING for row in intent._observed_ownership_links(
        too_late + f"\n[Open positions]({LISTING})", HOME,
    ))


@pytest.mark.parametrize("url,label", [
    ("https://unrelated.example/careers/open", "Open positions"),
    ("https://baringa.com/news/hiring", "Hiring"),
    ("https://baringa.com/careers/benefits", "Benefits"),
    ("https://baringa.com/careers/community", "Alumni"),
])
def test_new_discovery_does_not_treat_every_careers_path_or_label_as_a_listing(url, label):
    results = [{"url": HOME, "text": "", "meta": {"observed_ownership_links": [
        {"url": url, "label": label},
    ]}}]
    assert intent._observed_official_careers_url(results, IDENTITY, []) == ""
    assert intent._ats_ownership_anchors(results, [TARGET], IDENTITY) == []


@pytest.mark.parametrize("links", [
    None, [{"url": TARGET, "label": "Apply", "asserted_owner": True}],
    [{"url": "javascript:alert(1)", "label": "Apply"}],
    [{"url": TARGET, "label": "x" * 301}],
    [{"url": TARGET, "label": "Apply"}] * 81,
])
def test_careers_receipt_ownership_metadata_is_closed_and_bounded(links):
    assert intent._careers_index_receipt({
        "kind": "first_party_careers_index", "observed_job_link_count": 1,
        "observed_ownership_links": links,
    }) is None


@pytest.mark.parametrize("fault", ["", "wrong_tenant", "wrong_region", "hidden"])
def test_real_fetch_path_uses_only_three_sources_and_keeps_ownership_guards(monkeypatch, fault):
    requests = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            assert url == "https://api.scrapingdog.com/scrape"
            params = kwargs["params"]
            target = params["url"]
            requests.append({"url": target, "dynamic": params.get("dynamic", "false")})
            if target == HOME:
                return httpx.Response(200, text=home_html())
            assert target == LISTING, "No guessed or intermediate careers source is allowed"
            if fault != "hidden":
                assert params.get("dynamic") == "true"
            return httpx.Response(200, text=listing_html(
                tenant="another" if fault == "wrong_tenant" else "baringa",
                region="uk" if fault == "wrong_region" else "eu",
                hidden=fault == "hidden",
            ))

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: Client())
    # Unexpected fallback is a test failure, never a real provider request.
    async def no_fallback(*_args, **_kwargs):
        return {"ok": False, "content": "", "stage": "offline-disabled"}
    monkeypatch.setattr(intent, "_scrape_exa", no_fallback)
    monkeypatch.setattr(intent, "_try_wayback", no_fallback)
    contents = {"results": [{"url": TARGET, "text": "Baringa Transformation, Change and Delivery Consultant Sofia, Bulgaria", "meta": {"kind": "greenhouse_job"}}], "statuses": []}
    receipt = asyncio.run(intent._resolve_ats_employer_ownership(contents, [TARGET], IDENTITY, [TARGET]))
    assert receipt["source_count"] == 3
    assert {row["url"] for row in requests} == {HOME, LISTING}
    assert receipt["resolved"] is (not fault)
    if not fault:
        assert len(requests) == 2
        assert receipt["anchors"][0]["kind"] == "official_careers_link"
        assert receipt["anchors"][0]["source_url"] == LISTING
        assert receipt["anchors"][0]["posting_url"] == TARGET


def test_full_verifier_threads_observed_ownership_for_the_exact_baseline_job(monkeypatch):
    from tests.test_intent_first_party_careers_dynamic import _identity, _verdict

    requests, prompts = [], []
    captured = FIXTURE["submitted_posting_capture"]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            assert url == "https://api.scrapingdog.com/scrape"
            params = kwargs["params"]
            requests.append(dict(params))
            if params["url"] == captured["request_url"]:
                return httpx.Response(200, json=captured["payload"])
            if params["url"] == HOME:
                return httpx.Response(200, text=home_html())
            assert params["url"] == LISTING
            assert params["dynamic"] == "true"
            return httpx.Response(200, text=listing_html())

    async def judge(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(TARGET, "supported", "Transformation, Change and Delivery Consultant")

    async def no_fallback(*_args, **_kwargs):
        return {"ok": False, "content": "", "stage": "offline-disabled"}

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    monkeypatch.setattr(intent, "_scrape_exa", no_fallback)
    monkeypatch.setattr(intent, "_try_wayback", no_fallback)
    result = asyncio.run(intent.verify_three_stage(
        None, company_name="Baringa", company_website=HOME,
        company_linkedin="https://www.linkedin.com/company/baringa",
        source_url=TARGET,
        miner_claim="Baringa has an open Transformation, Change and Delivery Consultant role in Sofia, Bulgaria. This activity may help Baringa expand its management and digital-transformation consulting.",
        target_signal_text="Actively hiring for consulting, delivery, or practice-leadership roles, per current job postings or the careers page",
        evidence_type="HIRING", declared_source="job_board", stage1_soft_reject=True,
        company_quality=True, integrity_policy=True,
        verified_company_identity=_identity("Baringa", "baringa.com"),
    ))
    assert result["decision"] == "approve"
    assert result["client_ready"] is True
    assert result["ats_employer_ownership"]["source_count"] == 3
    assert result["ats_employer_ownership"]["resolved"] is True
    assert result["verified_job_source_urls"] == [TARGET]
    assert len(prompts) == 1 and '"resolved": true' in prompts[0]
    assert [r["url"] for r in requests] == [captured["request_url"], HOME, LISTING]

