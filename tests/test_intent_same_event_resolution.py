from datetime import date
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.arena_integrity import (
    source_dates_from_verdict,
    source_grounded_date_verdict,
)


MAX_INDEX = "https://maxretail.com/press/"
MAX_EVENT = "https://maxretail.com/retailer-resources/expands-point-of-sale/"
MAX_CLAIM = (
    "Max Retail announces a new point-of-sale integration with RICS Software, "
    "helping footwear retailers streamline aged inventory management."
)
MAX_QUOTE = (
    "On March 3, 2026, Max Retail announced its new point-of-sale integration "
    "with RICS Software, available immediately."
)
SOKIN_ARTICLE = (
    "https://sokin.com/news/"
    "sokin-raises-50m-series-b-following-100-year-on-year-growth"
)
SOKIN_INDEX = "https://sokin.com/news"
SOKIN_CLAIM = (
    "Sokin announced that it secured $50 million in Series B funding for "
    "global expansion, financial infrastructure, and product capabilities."
)
SOKIN_QUOTE = (
    "Sokin today announced it has secured $50 million in Series B funding "
    "to accelerate its global expansion and product capabilities."
)
SOKIN_CARD = (
    "Sokin raises $50M Series B following 100% year-on-year growth "
    "The investment will power Sokin’s next phase of global expansion, "
    "platform enhancement and cross-border payments innovation. "
    "Company news Dec 1, 2025"
)


def _sokin_archive_result(*, card=SOKIN_CARD, href=SOKIN_ARTICLE,
                          extra_cards=()):
    return {
        "url": SOKIN_INDEX,
        "text": "Earlier news. " + card + " Later news Dec 2, 2025.",
        "source_publication_date": "",
        "meta": {"same_host_event_links": [
            {"url": href, "label": card}, *extra_cards,
        ]},
    }


def test_archive_card_survives_article_focused_text_projection():
    raw_index = (
        '<html><body><main class="blog-index"><h1>News</h1>'
        f'<a href="{SOKIN_ARTICLE}">{SOKIN_CARD}</a>'
        '</main></body></html>'
    )
    links = verifier._same_host_event_links(raw_index, SOKIN_INDEX)
    row = {"company": "Sokin", "claim": SOKIN_CLAIM,
           "claimed_source_urls": [SOKIN_ARTICLE]}
    assert verifier._bound_article_archive_card([{
        "url": SOKIN_INDEX,
        "text": "News index summary without the card",
        "meta": {"same_host_event_links": links},
    }], archive_url=SOKIN_INDEX, article_url=SOKIN_ARTICLE, row=row) == {
        "text": SOKIN_CARD, "source_publication_date": "2025-12-01",
    }


@pytest.mark.parametrize("wrapper", [
    '<div hidden>{card}</div>',
    '<section aria-hidden="true">{card}</section>',
    '<div style="display: none">{card}</div>',
    '<div style="visibility: hidden">{card}</div>',
    '<div class="hidden-card">{card}</div>',
    '<div id="hidden-card">{card}</div>',
    '<a href="{article}" hidden>{label}</a>',
    '<a href="{article}" aria-hidden="true">{label}</a>',
    '<a href="{article}" style="display:none">{label}</a>',
    '<script>{card}</script>',
    '<template>{card}</template>',
])
def test_archive_link_in_hidden_ancestor_cannot_supply_date(wrapper):
    card = f'<a href="{SOKIN_ARTICLE}">{SOKIN_CARD}</a>'
    html = (
        '<style>.hidden-card, #hidden-card { display: none }</style>'
        + wrapper.format(card=card, article=SOKIN_ARTICLE, label=SOKIN_CARD)
        + '<nav><a href="/news">News</a></nav>'
    )
    links = verifier._same_host_event_links(html, SOKIN_INDEX)
    assert all(link["url"] != SOKIN_ARTICLE for link in links)


def test_visible_archive_index_and_navigation_links_remain_available():
    html = (
        '<nav><a href="/news">News</a></nav>'
        '<section class="blog-index">'
        f'<a href="{SOKIN_ARTICLE}">{SOKIN_CARD}</a>'
        '</section>'
    )
    links = verifier._same_host_event_links(html, SOKIN_ARTICLE)
    assert [link["url"] for link in links] == [SOKIN_INDEX]
    assert verifier._same_host_event_links(html, SOKIN_INDEX) == [{
        "url": SOKIN_ARTICLE, "label": SOKIN_CARD,
    }]


def test_sokin_shaped_archive_card_binds_original_article_and_one_date():
    article_html = (
        '<a href="/news">News Learn more what’s happening</a>'
        '<a href="/news/in-an-agent-rush-own-the-infrastructure">'
        'Company news Jul 30, 2026 In an agent rush, own the infrastructure'
        '</a>'
    )
    contents = {"results": [{"url": SOKIN_ARTICLE, "meta": {
        "same_host_event_links": verifier._same_host_event_links(
            article_html, SOKIN_ARTICLE,
        ),
    }}]}
    row = {"company": "Sokin", "claim": SOKIN_CLAIM,
           "claimed_source_urls": [SOKIN_ARTICLE]}
    assert verifier._article_archive_locator(contents, row) == SOKIN_INDEX
    assert verifier._same_event_link_candidates(contents, row)[0]["url"] != (
        SOKIN_INDEX
    )
    assert verifier._bound_article_archive_card(
        [_sokin_archive_result()], archive_url=SOKIN_INDEX,
        article_url=SOKIN_ARTICLE, row=row,
    ) == {
        "text": SOKIN_CARD, "source_publication_date": "2025-12-01",
    }


@pytest.mark.parametrize("mutation", [
    "adjacent_date", "wrong_href", "wrong_title", "wrong_company",
    "conflicting_card_dates", "conflicting_dates_in_card",
])
def test_archive_card_rejects_unbound_or_ambiguous_dates(mutation):
    row = {"company": "Sokin", "claim": SOKIN_CLAIM,
           "claimed_source_urls": [SOKIN_ARTICLE]}
    card = SOKIN_CARD
    href = SOKIN_ARTICLE
    extras = ()
    if mutation == "adjacent_date":
        card = card.replace(" Dec 1, 2025", "")
    elif mutation == "wrong_href":
        href = "https://sokin.com/news/other-series-b"
    elif mutation == "wrong_title":
        card = "Sokin opens a new office. Company news Dec 1, 2025"
    elif mutation == "wrong_company":
        card = card.replace("Sokin", "OtherCo")
    elif mutation == "conflicting_card_dates":
        extras = ({"url": href, "label": card.replace(
            "Dec 1, 2025", "Jan 2, 2026",
        )},)
    elif mutation == "conflicting_dates_in_card":
        card += " Updated Jan 2, 2026"
    assert verifier._bound_article_archive_card(
        [_sokin_archive_result(card=card, href=href, extra_cards=extras)],
        archive_url=SOKIN_INDEX, article_url=SOKIN_ARTICLE, row=row,
    ) is None


@pytest.mark.asyncio
async def test_archive_card_uses_existing_one_fetch_one_judge_and_date_gate(
    monkeypatch,
):
    calls = AsyncMock(side_effect=[
        _verdict(claim=SOKIN_CLAIM, url=SOKIN_ARTICLE, quote=SOKIN_QUOTE),
        _verdict(
            claim=SOKIN_CLAIM, url=SOKIN_INDEX, quote=SOKIN_CARD,
            risk_notes=["same_event_as_submitted:verified",
                        "source_publication_date:2025-12-01"],
        ),
    ])

    async def fetch(urls, *args, **kwargs):
        if urls == [SOKIN_ARTICLE]:
            return {"results": [{
                "url": SOKIN_ARTICLE, "text": SOKIN_QUOTE,
                "source_publication_date": "",
                "meta": {"same_host_event_links": [
                    {"url": SOKIN_INDEX, "label": "News"},
                    {"url": "https://sokin.com/news/in-an-agent-rush-own-the-infrastructure",
                     "label": "Company news Jul 30, 2026 In an agent rush, own the infrastructure"},
                ]},
            }], "statuses": []}
        assert urls == [SOKIN_INDEX]
        raw_index = (
            '<html><body><main class="blog-index"><h1>News</h1>'
            f'<a href="{SOKIN_ARTICLE}">{SOKIN_CARD}</a>'
            '</main></body></html>'
        )
        return {"results": [{
            "url": SOKIN_INDEX,
            "text": "News index summary without the dated card",
            "source_publication_date": "",
            "meta": {"same_host_event_links": verifier._same_host_event_links(
                raw_index, SOKIN_INDEX,
            )},
        }], "statuses": []}

    fetched = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetched)
    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    result = await verifier.verify_three_stage(
        object(), company_name="Sokin", company_linkedin="",
        company_website="https://sokin.com/", source_url=SOKIN_ARTICLE,
        miner_claim=SOKIN_CLAIM,
        miner_signal_date="2026-06-18",
        target_signal_text="Raised Series B funding in the last 12 months.",
        evidence_type="FUNDING", stage1_soft_reject=True,
        integrity_policy=True, buyer_max_age_days=365,
    )
    assert fetched.await_count == 2  # original + one resolution fetch
    assert calls.await_count == 2  # initial + one resolution judge
    assert result["source_resolution"]["selected_urls"] == [SOKIN_INDEX]
    assert result["source_resolution"]["status"] == "verified"
    assert result["source_publication_dates"] == ["2025-12-01"]
    assert [item["text"] for item in result["verified_source_context"]] == [
        SOKIN_QUOTE, SOKIN_CARD,
    ]
    assert "not by itself the date" in calls.await_args_list[1].args[2]
    assert "one visible same-host archive card" in calls.await_args_list[1].args[2]
    assert "does not establish company ownership" in calls.await_args_list[1].args[2]
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"],
    )
    freshness = source_grounded_date_verdict(
        event_date=event, publication_dates=publications,
        buyer_cap_days=365, evaluated_on=date(2026, 10, 1),
    )
    assert event is None
    assert freshness.verdict == "in_window"
    assert freshness.authoritative_date == "2025-12-01"
    assert freshness.basis == "publication_date"
    assert freshness.age_days == 304


def test_generic_archive_card_and_future_out_of_window_dates():
    article = "https://acme.test/blog/acme-unveils-atlas-routing"
    index = "https://acme.test/blog"
    claim = "Acme unveils Atlas routing for business workflows."
    row = {"company": "Acme", "claim": claim,
           "claimed_source_urls": [article]}
    assert verifier._article_archive_locator({"results": [{"url": article, "meta": {
        "same_host_event_links": [{"url": index, "label": "Blog"}],
    }}]}, row) == index
    for label, expected in (
        ("Acme unveils Atlas routing. Blog Mar 3, 2026", "in_window"),
        ("Acme unveils Atlas routing. Blog Mar 3, 2024", "out_of_window"),
        ("Acme unveils Atlas routing. Blog Mar 3, 2027", "uncertain"),
    ):
        card = verifier._bound_article_archive_card([{
            "url": index, "text": label,
            "meta": {"same_host_event_links": [{"url": article, "label": label}]},
        }], archive_url=index, article_url=article, row=row)
        assert card is not None
        assert source_grounded_date_verdict(
            event_date=None,
            publication_dates=[card["source_publication_date"]],
            buyer_cap_days=365, evaluated_on=date(2026, 10, 1),
        ).verdict == expected


def test_archive_card_cannot_be_promoted_to_actual_event_date():
    verdict = _verdict(
        claim=SOKIN_CLAIM, url=SOKIN_INDEX, quote=SOKIN_CARD,
        risk_notes=["same_event_as_submitted:verified",
                    "source_event_date:2025-12-01"],
    )["answer"]
    result = _sokin_archive_result()
    result["text"] = SOKIN_CARD
    result["source_publication_date"] = "2025-12-01"
    assert verifier._same_event_resolution_outcome(
        verdict, [result], [SOKIN_INDEX],
        submitted_claim=SOKIN_CLAIM, archive_card_date="2025-12-01",
    ) == "unproven"


def _stage1_review():
    return {
        "answer": {"signal_evaluations": [{
            "signal_status": "unable_to_verify",
            "same_entity_check": "unclear",
            "confidence": "medium",
        }]},
        "model": "test-stage1",
        "usage": {},
    }


def _verdict(
    *,
    claim: str,
    url: str,
    status: str = "supported",
    quote: str = "",
    risk_notes=None,
    date_match: str = "no_date_in_content",
):
    supporting = [quote] if status == "supported" and quote else []
    contradicting = [quote] if status == "contradicted" and quote else []
    return {
        "answer": {
            "overall_verdict": (
                "qualified" if status == "supported" else "disqualified"
            ),
            "overall_confidence": "high",
            "summary": "bounded test verdict",
            "missing_or_risks": [],
            "signal_evaluations": [{
                "signal_id": "signal-1",
                "claim": claim,
                "verification_mode": "source_grounded",
                "signal_status": status,
                "source_urls_supplied": [url],
                "evidence_urls_used": [url],
                "source_accessibility": "accessible",
                "same_entity_check": "pass",
                "entity_match_reason": "exact company",
                "supporting_quotes": supporting,
                "contradicting_quotes": contradicting,
                "unsupported_parts": [],
                "source_quality": "official_first_party",
                "risk_notes": list(risk_notes or []),
                "confidence": "high",
                "claim_matches_miner_date": date_match,
                "author_type": "n/a",
                "author_employer_matches_lead": "n/a",
                "author_role_matches_spec": "n/a",
                "author_satisfies_role_spec": "n/a",
            }],
        },
        "model": "test-stage3",
        "usage": {},
    }


def test_same_host_event_links_keep_cross_path_and_drop_external_links():
    body = f'''<a href="/retailer-resources/expands-point-of-sale/">
      RICS point-of-sale integration</a>
      <a href="https://example.com/newer-story">External story</a>
      <a href="{MAX_INDEX}#top">Current page</a>'''

    assert verifier._same_host_event_links(body, MAX_INDEX) == [{
        "url": MAX_EVENT,
        "label": "RICS point-of-sale integration",
    }]


def test_one_distinctive_event_term_reaches_selection_but_nav_does_not():
    contents = {"results": [{
        "meta": {"same_host_event_links": [
            {"url": MAX_EVENT, "label": "RICS integration"},
            {"url": "https://maxretail.com/about/", "label": "About us"},
        ]},
    }]}
    row = {
        "claim": "RICS partnership",
        "_target_signal_text": "Launched a major capability",
    }

    assert verifier._same_event_link_candidates(contents, row) == [{
        "url": MAX_EVENT,
        "label": "RICS integration",
    }]


@pytest.mark.asyncio
async def test_generic_fetch_preserves_observed_same_host_links(monkeypatch):
    async def no_special_route(_url):
        return {"routed": False, "ok": False, "content": ""}

    monkeypatch.setattr(verifier, "_scrape_ashby_job", no_special_route)
    monkeypatch.setattr(verifier, "_scrape_greenhouse_job", no_special_route)
    monkeypatch.setattr(verifier, "_scrape_workday_cxs", no_special_route)
    monkeypatch.setattr(verifier, "_scrape_sd_hardened", AsyncMock(return_value={
        "ok": True,
        "content": MAX_CLAIM,
        "source_publication_date": "2024-06-05",
        "stage": "sd:baseline",
        "meta": {"same_host_event_links": [{
            "url": MAX_EVENT,
            "label": "RICS integration",
        }]},
    }))

    fetched = await verifier._fetch_sd_then_exa([MAX_INDEX])

    assert fetched["results"][0]["meta"]["same_host_event_links"] == [{
        "url": MAX_EVENT,
        "label": "RICS integration",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_bundle", [False, True])
@pytest.mark.parametrize("summarized", [False, True])
async def test_max_retail_index_follows_only_selected_same_event_link(monkeypatch, with_bundle, summarized):
    calls = AsyncMock(side_effect=[
        _verdict(
            claim=MAX_CLAIM,
            url=MAX_INDEX,
            quote=MAX_CLAIM,
            risk_notes=["source_publication_date:2024-06-05"],
        ),
        _verdict(
            claim=("Max Retail released its RICS point-of-sale integration." if summarized else MAX_CLAIM),
            url=MAX_EVENT,
            quote=f"“{MAX_QUOTE}”" if summarized else MAX_QUOTE,
            risk_notes=["same_event_as_submitted:verified", "source_event_date:2026-03-03"],
        ),
    ])

    async def fetch(urls, *args, **kwargs):
        if urls == [MAX_INDEX]:
            return {
                "results": [{
                    "url": MAX_INDEX,
                    "title": "Press",
                    "text": MAX_CLAIM,
                    "source_publication_date": "2024-06-05",
                    "meta": {"same_host_event_links": [{
                        "url": MAX_EVENT,
                        "label": "Max Retail announces RICS point-of-sale integration",
                    }]},
                }],
                "statuses": [{"url": MAX_INDEX, "source": "scrapingdog"}],
            }
        assert urls == [MAX_EVENT]
        return {
            "results": [{
                "url": MAX_EVENT,
                "title": "RICS integration",
                "text": MAX_QUOTE,
                "source_publication_date": "2026-03-03",
                "meta": {},
            }],
            "statuses": [{"url": MAX_EVENT, "source": "scrapingdog"}],
        }

    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(side_effect=fetch))

    result = await verifier.verify_three_stage(
        object(),
        company_name="Max Retail",
        company_linkedin="",
        company_website="https://maxretail.com/",
        source_url=MAX_INDEX,
        evidence_bundle=([{
            "url": MAX_INDEX, "description": MAX_CLAIM, "snippet": MAX_CLAIM,
            "date": None,
        }] if with_bundle else None),
        miner_claim=MAX_CLAIM,
        target_signal_text=(
            "Launched a new product or major capability in the last 12 months."
        ),
        evidence_type="PRODUCT_LAUNCH",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=365,
    )

    assert result["client_ready"] is True
    assert result["source_resolution"] == {
        "attempted": True,
        "status": "verified",
        "selected_urls": [MAX_EVENT],
        "reason": "ranked_visible_same_host_links",
    }
    assert result["source_publication_dates"] == ["2026-03-03"]
    assert result["verified_source_context"][0]["url"] == MAX_EVENT
    assert calls.await_count == 2
    assert "ONE-HOP SAME-EVENT SOURCE RESOLUTION" in (
        calls.await_args_list[1].args[2]
    )


@pytest.mark.asyncio
async def test_unproven_link_fetch_preserves_original_stale_date(monkeypatch):
    calls = AsyncMock(side_effect=[
        _verdict(
            claim=MAX_CLAIM,
            url=MAX_INDEX,
            quote=MAX_CLAIM,
            risk_notes=["source_publication_date:2024-06-05"],
        ),
    ])

    async def fetch(urls, *args, **kwargs):
        if urls == [MAX_INDEX]:
            return {
                "results": [{
                    "url": MAX_INDEX,
                    "text": MAX_CLAIM,
                    "source_publication_date": "2024-06-05",
                    "meta": {"same_host_event_links": [{
                        "url": MAX_EVENT,
                        "label": "RICS integration",
                    }]},
                }],
                "statuses": [],
            }
        return {
            "results": [],
            "statuses": [{
                "url": MAX_EVENT,
                "source": "none",
                "stage": "transport_failure",
            }],
        }

    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(side_effect=fetch))

    result = await verifier.verify_three_stage(
        object(),
        company_name="Max Retail",
        company_linkedin="",
        company_website="https://maxretail.com/",
        source_url=MAX_INDEX,
        miner_claim=MAX_CLAIM,
        target_signal_text="Launched a major capability in the last 12 months.",
        evidence_type="PRODUCT_LAUNCH",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=365,
    )

    assert result["source_resolution"]["status"] == "unproven"
    assert result["source_publication_dates"] == ["2024-06-05"]
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 22),
    )
    assert freshness.verdict == "out_of_window"
    assert freshness.authoritative_date == "2024-06-05"
    assert calls.await_count == 1


@pytest.mark.asyncio
async def test_phia_older_event_month_overrides_newer_article_metadata(monkeypatch):
    source = "https://www.globenewswire.com/news-release/phia-series-a.html"
    claim = (
        "Phia, the shopping agent launched by co-founders Phoebe Gates and "
        "Sophia Kianni in April 2025"
    )
    calls = AsyncMock(side_effect=[
        _verdict(
            claim=claim,
            url=source,
            quote=claim,
            risk_notes=["source_publication_date:2026-01-27"],
            date_match="contradicted",
        ),
    ])
    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [{
            "url": source,
            "title": "Phia raises Series A",
            "text": claim,
            "source_publication_date": "2026-01-27",
            "meta": {},
        }],
        "statuses": [],
    }))

    result = await verifier.verify_three_stage(
        object(),
        company_name="Phia",
        company_linkedin="",
        company_website="https://phia.com/",
        source_url=source,
        miner_claim=claim,
        target_signal_text="Launched a new product in the last 12 months.",
        miner_signal_date="2026-01-27",
        evidence_type="PRODUCT_LAUNCH",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=365,
    )

    item = result["verdict"]["signal_evaluations"][0]
    assert item["risk_notes"] == ["source_event_month:2025-04"]
    assert "source_event_month:YYYY-MM" in calls.await_args_list[0].args[2]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 22),
    )
    assert event == "2025-04"
    assert freshness.verdict == "out_of_window"
    assert freshness.authoritative_date == "2025-04-30"
    assert freshness.basis == "event_month_latest_bound"


@pytest.mark.parametrize(
    ("company", "claim", "target", "source_text"),
    [
        (
            "Buywander",
            "Buywander opened a fourth warehouse store in Salt Lake City.",
            "Launched a new product or major capability in the last 12 months.",
            "Buywander opened a fourth warehouse store in Salt Lake City.",
        ),
        (
            "MontyCloud",
            "MontyCloud raised Series A funding and plans to double its India team.",
            "Opened a new office, delivery center, or regional hub in the last 12 months.",
            "MontyCloud raised Series A funding and plans to double its India team.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_nonmatching_event_controls_remain_rejected(
    monkeypatch, company, claim, target, source_text
):
    source = "https://example.test/submitted-event"
    calls = AsyncMock(side_effect=[
        _verdict(
            claim=claim,
            url=source,
            status="contradicted",
            quote=source_text,
            risk_notes=["source_event_date:2025-10-07"],
        ),
    ])
    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [{
            "url": source,
            "title": "Submitted event",
            "text": source_text,
            "source_publication_date": "2025-10-07",
            "meta": {},
        }],
        "statuses": [],
    }))

    result = await verifier.verify_three_stage(
        object(),
        company_name=company,
        company_linkedin="",
        company_website="https://example.test/",
        source_url=source,
        miner_claim=claim,
        target_signal_text=target,
        evidence_type="PRODUCT_LAUNCH",
        stage1_soft_reject=True,
        integrity_policy=True,
    )

    assert result["client_ready"] is False
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"
    assert calls.await_count == 1
    if company == "Buywander":
        assert "warehouse, store, office, facility" in calls.await_args_list[0].args[2]


def test_approximate_event_month_overlapping_window_stays_unproven():
    event, publications = source_dates_from_verdict({
        "risk_notes": [
            "source_event_month:2026-09",
            "source_publication_date:2026-09-01",
        ]
    }, ["2026-09-01"])
    result = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=30,
        evaluated_on=date(2026, 9, 22),
    )

    assert event == "2026-09"
    assert result.verdict == "uncertain"
    assert result.basis == "event_month_overlaps_window"


def test_unrelated_grounded_date_does_not_override_publication_metadata():
    claim = "Acme launched Control Copilot."
    source = (
        "Acme launched Control Copilot. The company was founded in April 2020."
    )
    item = {
        "claim": claim,
        "supporting_quotes": [
            "Acme launched Control Copilot.",
            "The company was founded in April 2020.",
        ],
        "risk_notes": ["source_publication_date:2026-08-01"],
    }

    assert verifier._bind_approximate_event_month(item, source, claim) == ""
    assert item["risk_notes"] == ["source_publication_date:2026-08-01"]


def test_conflicting_grounded_event_months_preserve_date_uncertainty():
    claim = "Acme launched Alpha in April 2025 and Beta in May 2025."
    item = {
        "claim": claim,
        "supporting_quotes": [claim],
        "risk_notes": ["source_publication_date:2026-08-01"],
    }

    assert verifier._bind_approximate_event_month(item, claim, claim) == ""
    assert item["risk_notes"] == ["source_event_date_conflict"]


def test_hallucinated_linked_event_date_cannot_verify_source_resolution():
    event_quote = "Acme launched Control Copilot for policy automation."
    linked_text = event_quote + " Unrelated archive item: March 3, 2026."
    verdict = _verdict(
        claim="Acme launched Control Copilot.",
        url="https://acme.test/control-copilot",
        quote=event_quote,
        risk_notes=[
            "source_event_date:2026-03-03",
            "source_publication_date:2026-03-03",
        ],
    )["answer"]

    outcome = verifier._same_event_resolution_outcome(
        verdict,
        [{
            "url": "https://acme.test/control-copilot",
            "text": linked_text,
            "source_publication_date": "2026-03-03",
        }],
        ["https://acme.test/control-copilot"],
        submitted_claim="Acme launched Control Copilot.",
    )

    assert outcome == "unproven"


def test_same_entity_newer_event_cannot_replace_the_submitted_claim():
    url = "https://acme.test/new-launch"
    quote = "On March 3, 2026, Acme launched New Copilot."
    verdict = _verdict(
        claim="Acme launched New Copilot.", url=url, quote=quote,
        risk_notes=["source_event_date:2026-03-03"],
    )["answer"]
    assert verifier._same_event_resolution_outcome(
        verdict, [{"url": url, "text": quote}], [url],
        submitted_claim="Acme launched Original Copilot.",
    ) == "unproven"


@pytest.mark.parametrize("wrapper", ['"{}"', "'{}'", '“{}”', '‘{}’', '{}'])
def test_copied_quote_wrappers_are_formatting(wrapper):
    text = "Acme launched Control Copilot on March 3, 2026."
    assert verifier._grounded_exact_text(text, wrapper.format(text))


@pytest.mark.parametrize("quote", [
    "Acme launched Control Copilot on March 4, 2026.",
    "Acme ... Control Copilot on March 3, 2026.",
    "Control Copilot launched Acme on March 3, 2026.",
    '""', "", "Acme launched Control Copilot",
])
def test_quote_recovery_does_not_invent_or_remove_source_words(quote):
    text = "Acme has not launched Control Copilot on March 3, 2026."
    assert not verifier._grounded_exact_text(text, quote)


def test_linked_review_can_summarize_explicitly_verified_same_event():
    url = "https://acme.test/release"
    text = "On March 3, 2026, Acme launched Control Copilot for policy automation."
    verdict = _verdict(
        claim="Acme released its policy automation product, Control Copilot.",
        url=url, quote=f'“{text}”',
        risk_notes=["same_event_as_submitted:verified", "source_event_date:2026-03-03"],
    )["answer"]
    assert verifier._same_event_resolution_outcome(
        verdict, [{"url": url, "text": text}], [url],
        submitted_claim="Acme launched Control Copilot.",
    ) == "verified"


@pytest.mark.parametrize("mutation", [
    {"signal_id": "another-signal"},
    {"same_entity_check": "fail"},
    {"signal_status": "partially_supported"},
    {"confidence": "medium"},
    {"unsupported_parts": ["The claimed product was not launched."]},
    {"contradicting_quotes": ["Launch is planned, not completed."]},
    {"evidence_urls_used": ["https://unrelated.test/other-event"]},
    {"risk_notes": ["same_event_as_submitted:verified", "source_event_date:2026-03-04"]},
])
def test_linked_review_still_requires_signal_entity_event_source_and_date(mutation):
    url = "https://acme.test/release"
    text = "On March 3, 2026, Acme launched Control Copilot for policy automation."
    verdict = _verdict(
        claim="Acme released Control Copilot for policy automation.",
        url=url, quote=text,
        risk_notes=["same_event_as_submitted:verified", "source_event_date:2026-03-03"],
    )["answer"]
    verdict["signal_evaluations"][0].update(mutation)
    assert verifier._same_event_resolution_outcome(
        verdict, [{"url": url, "text": text}], [url],
        submitted_claim="Acme launched Control Copilot.",
    ) == "unproven"


def test_supported_medium_summary_can_reach_clarification_not_acceptance():
    source = "Acme launched Control Copilot for policy automation."
    verdict = _verdict(
        claim="Acme released a policy automation product.",
        url="https://acme.test/release", quote=source,
    )["answer"]
    item = verdict["signal_evaluations"][0]
    verdict["overall_confidence"] = item["confidence"] = "medium"
    assert verifier._supported_medium_needs_clarification(verdict, item, source)
    assert verifier._decision(verdict, company_quality=True) == "review"
