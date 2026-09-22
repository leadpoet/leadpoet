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
async def test_max_retail_index_follows_only_selected_same_event_link(monkeypatch):
    calls = AsyncMock(side_effect=[
        _stage1_review(),
        _verdict(
            claim=MAX_CLAIM,
            url=MAX_INDEX,
            quote=MAX_CLAIM,
            risk_notes=["source_publication_date:2024-06-05"],
        ),
        _verdict(
            claim=MAX_CLAIM,
            url=MAX_EVENT,
            quote=MAX_QUOTE,
            risk_notes=["source_publication_date:2026-03-03"],
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
    assert calls.await_count == 3
    assert "ONE-HOP SAME-EVENT SOURCE RESOLUTION" in (
        calls.await_args_list[2].args[2]
    )


@pytest.mark.asyncio
async def test_unproven_link_fetch_preserves_original_stale_date(monkeypatch):
    calls = AsyncMock(side_effect=[
        _stage1_review(),
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
    assert calls.await_count == 2


@pytest.mark.asyncio
async def test_phia_older_event_month_overrides_newer_article_metadata(monkeypatch):
    source = "https://www.globenewswire.com/news-release/phia-series-a.html"
    claim = (
        "Phia, the shopping agent launched by co-founders Phoebe Gates and "
        "Sophia Kianni in April 2025"
    )
    calls = AsyncMock(side_effect=[
        _stage1_review(),
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
    assert "source_event_month:YYYY-MM" in calls.await_args_list[1].args[2]
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
        _stage1_review(),
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
    assert calls.await_count == 2
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
