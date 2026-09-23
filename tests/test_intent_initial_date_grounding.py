"""Fetched date grounding can trigger recovery without deleting judge dates."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.arena_integrity import (
    source_dates_from_verdict,
    source_grounded_date_verdict,
)


INDEX_URL = "https://acme.example/press/"
EVENT_URL = "https://acme.example/news/atlas-launch/"
CLAIM = "Acme announced the Atlas workflow integration."


def _response(
    *, url=INDEX_URL, claim=CLAIM, quote=CLAIM, notes=(),
    status="supported", date_match="no_date_in_content",
):
    return {"answer": {
        "overall_verdict": "qualified" if status == "supported" else "uncertain",
        "overall_confidence": "high", "summary": "bounded date test",
        "missing_or_risks": [], "signal_evaluations": [{
            "signal_id": "signal-1", "claim": claim,
            "verification_mode": "source_grounded", "signal_status": status,
            "source_urls_supplied": [url], "evidence_urls_used": [url],
            "source_accessibility": "accessible", "same_entity_check": "pass",
            "entity_match_reason": "exact company",
            "supporting_quotes": [quote] if quote else [],
            "contradicting_quotes": [], "unsupported_parts": [],
            "source_quality": "official_first_party", "risk_notes": list(notes),
            "confidence": "high", "claim_matches_miner_date": date_match,
            "author_type": "n/a", "author_employer_matches_lead": "n/a",
            "author_role_matches_spec": "n/a", "author_satisfies_role_spec": "n/a",
        }],
    }, "model": "test-stage3", "usage": {}}


async def _verify(monkeypatch, responses, *, body, links=()):
    calls = AsyncMock(side_effect=responses)

    async def fetch(urls, *_args, **_kwargs):
        if urls == [INDEX_URL]:
            return {"results": [{
                "url": INDEX_URL, "text": body,
                "meta": {"same_host_event_links": list(links)},
            }], "statuses": []}
        assert urls == [EVENT_URL]
        return {"results": [{
            "url": EVENT_URL,
            "text": "On March 3, 2026, Acme launched Atlas.",
            "source_publication_date": "2026-03-03", "meta": {},
        }], "statuses": []}

    monkeypatch.setattr(verifier, "_call_openrouter", calls)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    result = await verifier.verify_three_stage(
        object(), company_name="Acme", company_linkedin="",
        company_website="https://acme.example/", source_url=INDEX_URL,
        miner_claim=CLAIM,
        target_signal_text="Launched a major capability in the last 12 months",
        evidence_type="PRODUCT_LAUNCH", stage1_soft_reject=True,
        integrity_policy=True, buyer_max_age_days=365,
    )
    return result, calls


@pytest.mark.parametrize("source_date", [
    "March 3, 2026", "Mar. 3, 2026", "3 March 2026", "3 Mar 2026",
    "Sept. 15, 2026", "15 Sept 2026",
])
def test_common_body_date_formats_suppress_unneeded_recovery(source_date):
    canonical = "2026-09-15" if "Sept" in source_date else "2026-03-03"
    item = _response(notes=[f"source_event_date:{canonical}"])["answer"][
        "signal_evaluations"
    ][0]
    assert verifier._has_grounded_source_event_date(
        item, f"Published {source_date}. {CLAIM}"
    )
    assert item["risk_notes"] == [f"source_event_date:{canonical}"]


@pytest.mark.asyncio
async def test_old_no_comma_date_remains_out_of_window_without_followup(monkeypatch):
    old_note = "source_event_date:2024-09-12"
    result, calls = await _verify(
        monkeypatch, [_response(notes=[old_note], date_match="consistent")],
        body="Acme announced Atlas on September 12 2024.",
    )
    item = result["verdict"]["signal_evaluations"][0]
    event_date, publications = source_dates_from_verdict(item)
    freshness = source_grounded_date_verdict(
        event_date=event_date, publication_dates=publications,
        buyer_cap_days=365, evaluated_on=date(2026, 9, 23),
    )
    assert calls.await_count == 1
    assert "source_resolution" not in result
    assert item["risk_notes"] == [old_note]
    assert freshness.verdict == "out_of_window"


@pytest.mark.asyncio
async def test_unfamiliar_valid_source_date_remains_supported(monkeypatch):
    note = "source_event_date:2024-09-12"
    result, calls = await _verify(
        monkeypatch, [_response(notes=[note], date_match="consistent")],
        body="Acme announced Atlas on the 12th of September, 2024.",
    )
    assert calls.await_count == 1
    assert result["decision"] == "approve"
    assert result["verdict"]["signal_evaluations"][0]["risk_notes"] == [note]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_status", ["supported", "partially_supported", "unable_to_verify"]
)
async def test_text_ungrounded_date_can_attempt_bounded_recovery(
    monkeypatch, initial_status,
):
    result, calls = await _verify(
        monkeypatch, [
            _response(
                notes=["source_event_date:2026-03-03"],
                status=initial_status, date_match="consistent",
            ),
            _response(
                url=EVENT_URL, quote="On March 3, 2026, Acme launched Atlas.",
                notes=["same_event_as_submitted:verified",
                       "source_event_date:2026-03-03"],
            ),
        ], body=CLAIM,
        links=[{"url": EVENT_URL, "label": "Atlas launch"}],
    )
    assert calls.await_count == 2
    assert result["source_resolution"]["status"] == "verified"


@pytest.mark.asyncio
async def test_unproven_link_keeps_original_old_date_and_verdict(monkeypatch):
    old_note = "source_event_date:2024-09-12"
    result, calls = await _verify(
        monkeypatch, [
            _response(notes=[old_note], date_match="consistent"),
            _response(
                url=EVENT_URL, claim="Acme launched an unrelated product.",
                quote="On March 3, 2026, Acme launched Atlas.",
                notes=["source_event_date:2026-03-03"],
            ),
        ], body="Acme announced Atlas on September 12 2024.",
        links=[{"url": EVENT_URL, "label": "Atlas launch"}],
    )
    item = result["verdict"]["signal_evaluations"][0]
    assert calls.await_count == 2
    assert result["source_resolution"]["status"] == "unproven"
    assert item["risk_notes"] == [old_note]
    assert source_dates_from_verdict(item) == ("2024-09-12", [])


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["wrong_entity", "contradiction"])
async def test_wrong_entity_or_contradiction_cannot_start_recovery(
    monkeypatch, blocked_by,
):
    initial = _response(status="contradicted", quote="")
    item = initial["answer"]["signal_evaluations"][0]
    if blocked_by == "wrong_entity":
        item["signal_status"] = "wrong_entity"
        item["same_entity_check"] = "fail"
    else:
        item["contradicting_quotes"] = ["Acme did not launch Atlas."]
    result, calls = await _verify(
        monkeypatch, [initial], body=CLAIM,
        links=[{"url": EVENT_URL, "label": "Atlas launch"}],
    )

    assert calls.await_count == 1
    assert "source_resolution" not in result
