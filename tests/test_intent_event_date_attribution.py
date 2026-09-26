"""Event timing stays bound to the event, not a page timestamp."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.arena_integrity import (
    source_dates_from_verdict,
    source_grounded_date_verdict,
)


IDP_URL = "https://careers.idp.com/news/idp-digital-campus-launch"
OLD_SOURCE = (
    "https://investors.idp.com/FormBuilder/_Resource/_module/"
    "v1AiEHYL20-_Rje11PzkYA/FY20%20H1%20Investor%20Presentation.pdf"
)
IDP_CLAIM = "IDP Education has officially launched its new Digital Campus in Chennai, India."
IDP_BODY = (
    "IDP Education 28 October 2025 IDP Education cements position as "
    "edu-tech leader with new Digital Campus in Chennai. " + IDP_CLAIM
    + " IDP's digital transformation efforts, driven from Chennai, were "
    "recognised recently when IDP was awarded global Education Agency of "
    "the Year at the PIEoneer Awards in London (September 2019)."
)


def _answer(*, claim, url, quote, notes=(), status="supported"):
    return {"answer": {
        "overall_verdict": "qualified" if status == "supported" else "needs_review",
        "overall_confidence": "high",
        "signal_evaluations": [{
            "signal_id": "signal-1",
            "claim": claim,
            "verification_mode": "source_grounded",
            "signal_status": status,
            "source_urls_supplied": [url],
            "evidence_urls_used": [url] if status == "supported" else [],
            "source_accessibility": "accessible",
            "same_entity_check": "pass",
            "entity_match_reason": "exact company",
            "supporting_quotes": [quote] if quote else [],
            "contradicting_quotes": [],
            "unsupported_parts": [],
            "source_quality": "official_first_party",
            "risk_notes": list(notes),
            "confidence": "high",
            "claim_matches_miner_date": "consistent",
            "author_type": "n/a",
            "author_employer_matches_lead": "n/a",
            "author_role_matches_spec": "n/a",
            "author_satisfies_role_spec": "n/a",
        }],
        "missing_or_risks": [],
    }, "model": "test-stage3", "usage": {}}


async def _run(monkeypatch, responses, *, body, publication_date="", search_urls=()):
    calls = AsyncMock(side_effect=responses)

    async def fetch(urls, **kwargs):
        if urls == [IDP_URL]:
            return {"results": [{
                "url": IDP_URL,
                "text": body,
                "source_publication_date": publication_date,
                "meta": {"same_host_event_links": []},
            }], "statuses": []}
        assert urls == list(search_urls)
        return {"results": [{
            "url": OLD_SOURCE,
            "text": "The Chennai Digital Campus launched in November 2019.",
            "source_publication_date": "2020-02-01",
            "meta": {},
        }], "statuses": []}

    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(
        verifier, "_bounded_same_event_date_search",
        AsyncMock(return_value=list(search_urls)),
    )
    result = await verifier.verify_three_stage(
        object(),
        company_name="IDP Education",
        company_linkedin="",
        company_website="https://idp.com/",
        source_url=IDP_URL,
        miner_claim=IDP_CLAIM,
        target_signal_text="Opened a new campus in the last 12 months.",
        miner_signal_date="2025-10-28",
        evidence_type="MARKET_EXPANSION",
        declared_source="job_board",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=365,
    )
    return result, calls


@pytest.mark.asyncio
async def test_republished_campus_dateline_recovers_explicit_old_event(monkeypatch):
    initial = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote="IDP Education 28 October 2025 IDP Education cements position as edu-tech leader with new Digital Campus in Chennai.",
        notes=["source_event_date:2025-10-28"],
    )
    clarified = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote=IDP_CLAIM,
        notes=["source_event_date_disputed"],
    )
    recovered = _answer(
        claim=IDP_CLAIM,
        url=OLD_SOURCE,
        quote="The Chennai Digital Campus launched in November 2019.",
        notes=[
            "same_event_as_submitted:verified",
            "source_event_month:2019-11",
        ],
    )
    result, calls = await _run(
        monkeypatch,
        [initial, clarified, recovered],
        body=IDP_BODY,
        search_urls=[OLD_SOURCE],
    )

    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 3
    assert result["source_resolution"]["status"] == "verified"
    assert event == "2019-11"
    assert freshness.verdict == "out_of_window"
    assert freshness.basis == "event_month_latest_bound"


@pytest.mark.asyncio
async def test_original_current_announcement_keeps_bound_publication_date(monkeypatch):
    current = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote=IDP_CLAIM,
        notes=[
            "source_publication_date:2026-08-20",
            "source_event_publication_binding:verified",
        ],
    )
    result, calls = await _run(
        monkeypatch, [current], body=IDP_BODY, publication_date="2026-08-20"
    )
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 1
    assert "date_attribution_clarification" not in result
    assert result["client_ready"] is True
    assert freshness.verdict == "in_window"
    assert freshness.basis == "publication_date"


@pytest.mark.asyncio
async def test_unbound_publication_date_does_not_become_event_freshness(monkeypatch):
    unbound = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote=IDP_CLAIM,
        notes=["source_publication_date:2026-08-20"],
    )
    result, calls = await _run(
        monkeypatch, [unbound], body=IDP_CLAIM, publication_date="2026-08-20"
    )
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 1
    assert result["client_ready"] is True
    assert item["risk_notes"] == []
    assert freshness.verdict == "uncertain"


@pytest.mark.asyncio
async def test_explicit_date_dispute_blocks_new_publication_fallback(monkeypatch):
    disputed = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote=IDP_CLAIM,
        notes=[
            "source_publication_date:2026-08-20",
            "source_event_publication_binding:verified",
            "source_event_date_disputed",
        ],
    )
    result, calls = await _run(
        monkeypatch, [disputed], body=IDP_BODY, publication_date="2026-08-20"
    )
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 1
    assert result["client_ready"] is True
    assert item["risk_notes"] == ["source_event_date_disputed"]
    assert freshness.verdict == "uncertain"


@pytest.mark.asyncio
async def test_unrelated_page_date_becomes_honest_uncertainty(monkeypatch):
    initial = _answer(
        claim=IDP_CLAIM,
        url=IDP_URL,
        quote="IDP Education board meeting: October 28, 2025.",
        notes=["source_event_date:2025-10-28"],
    )
    clarified = _answer(
        claim=IDP_CLAIM, url=IDP_URL, quote=IDP_CLAIM, notes=[]
    )
    body = IDP_CLAIM + " IDP Education board meeting: October 28, 2025."
    result, calls = await _run(monkeypatch, [initial, clarified], body=body)
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 2
    assert result["client_ready"] is True
    assert event is None
    assert freshness.verdict == "uncertain"


@pytest.mark.asyncio
async def test_clean_missing_date_keeps_existing_uncertainty_without_extra_call(monkeypatch):
    missing = _answer(
        claim=IDP_CLAIM, url=IDP_URL, quote=IDP_CLAIM, notes=[]
    )
    result, calls = await _run(monkeypatch, [missing], body=IDP_CLAIM)
    item = result["verdict"]["signal_evaluations"][0]
    event, publications = source_dates_from_verdict(
        item, result["source_publication_dates"]
    )
    freshness = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=365,
        evaluated_on=date(2026, 9, 25),
    )

    assert calls.await_count == 1
    assert result["client_ready"] is True
    assert freshness.verdict == "uncertain"
