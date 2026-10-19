"""One unavailable source can discover one alternative for the original event."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier


ORIGINAL = "https://www.instagram.com/p/DYliXGhoHbm/"
ALTERNATIVE = (
    "https://www.linkedin.com/posts/e27-optimatic-_echelonsg-echelonsg2026-"
    "activity-7463076223317860352-kklS"
)
CLAIM = "PixVerse was publicly named as an AI Competition Partner for Echelon Singapore 2026."
QUOTE = (
    "Echelon Singapore 2026 on May 20, 2026: organizer e27 welcomes PixVerse "
    "as our AI Competition Partner."
)


def missing_source():
    return {"results": [], "statuses": [{
        "url": ORIGINAL, "source": "none",
        "sd_stage": "all_tiers_exhausted:http_422", "sd_error": "http_422",
        "exa_stage": "exa_no_results", "exa_error": "[]",
    }]}


def fetched_source(text=QUOTE, url=ALTERNATIVE):
    return {"results": [{"url": url, "text": text, "meta": {
        # A tempting third hop must never be fetched on this path.
        "same_host_event_links": [{"url": "https://www.linkedin.com/third-hop",
                                   "label": "PixVerse Echelon 2026"}],
    }}], "statuses": []}


def judgment(*, status="supported", entity="pass", quote=QUOTE, notes=None,
             url=ALTERNATIVE, confidence="high"):
    return {"model": "test", "usage": {}, "answer": {
        "overall_verdict": "qualified", "overall_confidence": confidence,
        "signal_evaluations": [{
            "signal_id": "signal-1", "claim": CLAIM,
            "verification_mode": "source_grounded", "signal_status": status,
            "confidence": confidence, "same_entity_check": entity,
            "source_urls_supplied": [ORIGINAL, url], "evidence_urls_used": [url],
            "source_quality": "official_event_organizer",
            "supporting_quotes": [quote] if quote else [],
            "contradicting_quotes": [], "unsupported_parts": [],
            "claim_matches_miner_date": "consistent",
            "risk_notes": (["same_event_as_submitted:verified",
                            "source_event_date:2026-05-20"] if notes is None else notes),
        }],
    }}


async def run_verifier(monkeypatch, *, original=None, alternative=None,
                       response=None, search=None, **kwargs):
    fetch = AsyncMock(side_effect=[
        missing_source() if original is None else original,
        fetched_source() if alternative is None else alternative,
    ])
    discover = AsyncMock(return_value={
        "status": "selected", "reason": "bounded_same_event_search",
        "selected_urls": [ALTERNATIVE],
    } if search is None else search)
    judge = AsyncMock(return_value=judgment() if response is None else response)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_find_same_event_source", discover)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    result = await verifier.verify_three_stage(
        object(), company_name="PixVerse", company_website="https://pixverse.ai/",
        company_linkedin="", source_url=ORIGINAL, miner_claim=CLAIM,
        target_signal_text="Publicly named as a partner at a technology event",
        miner_signal_date="2026-05-20", evidence_type="PARTNERSHIP",
        stage1_soft_reject=True, integrity_policy=True, **kwargs,
    )
    return result, fetch, discover, judge


@pytest.mark.asyncio
async def test_healthy_alternative_uses_one_ordinary_judge_and_retains_body(monkeypatch):
    result, fetch, search, judge = await run_verifier(monkeypatch)
    assert result["client_ready"]
    assert result["source_resolution"]["status"] == "verified"
    assert fetch.await_count == 2
    assert search.await_count == judge.await_count == 1
    prompt = judge.await_args.args[2]
    assert CLAIM in prompt and ORIGINAL in prompt and ALTERNATIVE in prompt
    assert "official company, event organizer" in prompt
    assert QUOTE in prompt
    assert result["verified_source_context"][0]["url"] == ALTERNATIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "unrelated_company", "newer_event", "conflicting_date", "ungrounded_snippet",
    "missing_same_event_marker", "medium_confidence", "missing_event_date",
    "publication_date_only", "unproved_source_authority",
])
async def test_alternative_does_not_automatically_qualify(monkeypatch, case):
    response = judgment()
    body = QUOTE
    item = response["answer"]["signal_evaluations"][0]
    if case == "unrelated_company":
        body = QUOTE.replace("PixVerse", "OtherCo")
        item.update(signal_status="wrong_entity", same_entity_check="fail")
    elif case == "newer_event":
        body = QUOTE.replace("2026", "2027")
        item.update(signal_status="unable_to_verify", risk_notes=[])
    elif case == "conflicting_date":
        body = QUOTE.replace("May 20, 2026", "May 20, 2027")
    elif case == "ungrounded_snippet":
        body = "Organizer e27. Read our upcoming events and general news."
    elif case == "missing_same_event_marker":
        item["risk_notes"] = ["source_event_date:2026-05-20"]
    elif case == "medium_confidence":
        item["confidence"] = "medium"
        response["answer"]["overall_confidence"] = "medium"
    elif case == "missing_event_date":
        item["risk_notes"] = ["same_event_as_submitted:verified"]
    elif case == "publication_date_only":
        item["risk_notes"] = ["same_event_as_submitted:verified",
                              "source_publication_date:2026-05-20"]
        alternative = fetched_source()
        alternative["results"][0]["source_publication_date"] = "2026-05-20"
    elif case == "unproved_source_authority":
        item["signal_status"] = "unable_to_verify"
        item["unsupported_parts"] = ["Unknown publisher's authority is unproved"]
    result, fetch, search, judge = await run_verifier(
        monkeypatch, alternative=(alternative if case == "publication_date_only"
                                 else fetched_source(body)), response=response,
    )
    assert not result["client_ready"]
    assert fetch.await_count == 2
    assert search.await_count == judge.await_count == 1


@pytest.mark.asyncio
async def test_same_event_grounded_date_correction_uses_ordinary_policy(monkeypatch):
    body = QUOTE.replace("May 20, 2026", "May 19, 2026")
    response = judgment(quote=body, notes=[
        "same_event_as_submitted:verified", "source_event_date:2026-05-19",
    ])
    result, *_ = await run_verifier(
        monkeypatch, alternative=fetched_source(body), response=response,
    )
    assert result["source_resolution"]["status"] == "verified"
    assert "source_event_date:2026-05-19" in result["verdict"]["signal_evaluations"][0]["risk_notes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [
    {"sd_stage": "all_tiers_exhausted:http_401", "sd_error": "http_401"},
    {"sd_stage": "all_tiers_exhausted:http_402", "sd_error": "http_402"},
    {"sd_stage": "all_tiers_exhausted:http_403", "sd_error": "http_403"},
    {"sd_stage": "all_tiers_exhausted:http_400", "sd_error": "http_400"},
    {"sd_stage": "all_tiers_exhausted:http_429", "sd_error": "http_429"},
    {"sd_stage": "all_tiers_exhausted:http_502", "sd_error": "http_502"},
    {"sd_stage": "all_tiers_exhausted:client_deadline:baseline"},
    {"exa_stage": "exa_transient_exhausted"},
    {"exa_stage": "exa_http_error", "exa_error": "HTTP 401"},
    {"exa_stage": "exa_http_error", "exa_error": "HTTP 402"},
    {"exa_error": '[{"status":"pending"}]'},
    {"exa_error": '[{"error":{"httpStatusCode":429}}]'},
    {"exa_error": "unknown"},
])
async def test_provider_failure_never_becomes_source_absence(monkeypatch, mutation):
    original = missing_source()
    original["statuses"][0].update(mutation)
    result, fetch, search, judge = await run_verifier(monkeypatch, original=original)
    assert result["decision"] == "unavailable"
    assert fetch.await_count == 1
    search.assert_not_awaited()
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_healthy_original_does_no_discovery_or_extra_fetch(monkeypatch):
    result, fetch, search, judge = await run_verifier(
        monkeypatch, original=fetched_source(url=ORIGINAL),
        response=judgment(url=ORIGINAL),
    )
    assert result["client_ready"]
    assert "source_resolution" not in result
    assert fetch.await_count == judge.await_count == 1
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_alternative_is_bounded_and_retryable(monkeypatch):
    alternative = missing_source()
    alternative["statuses"][0].update(url=ALTERNATIVE, exa_stage="exa_transient_exhausted")
    result, fetch, search, judge = await run_verifier(monkeypatch, alternative=alternative)
    assert result["decision"] == "unavailable"
    assert fetch.await_count == 2 and search.await_count == 1
    judge.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unavailable", "unproven"])
async def test_failed_discovery_is_bounded(monkeypatch, status):
    result, fetch, search, judge = await run_verifier(
        monkeypatch, search={"status": status, "reason": "test"},
    )
    assert result["decision"] == "unavailable"
    assert fetch.await_count == search.await_count == 1
    judge.assert_not_awaited()
    if status == "unavailable":
        from qualification.scoring.competition import _intent_detail_has_source_local_failure

        assert not _intent_detail_has_source_local_failure({"judge_verdict": {
            "decision": "rejected_verifier_error", "pipeline_decision": "unavailable",
            "rejection_reason": "evidence_fetch_failed", "verification_trace": {
                "provider_attempts": result["scrape"]["statuses"],
            },
        }})


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("budget exhausted"),
                                       asyncio.TimeoutError(), ValueError("pending")])
async def test_existing_search_provider_budget_timeout_and_pending_remain_unavailable(
    monkeypatch, error,
):
    from qualification.scoring import company_evidence_investigator as investigator

    monkeypatch.setenv("EXA_API_KEY", "unit-test")
    search = AsyncMock(side_effect=error)
    monkeypatch.setattr(investigator, "_search_web", search)
    row = {"company": "pixverse.ai", "claim": CLAIM,
           "signal_date": "2026-05-20", "claimed_source_urls": [ORIGINAL]}
    result = await verifier._find_same_event_source(row)
    assert result == {"status": "unavailable", "reason": "search_provider_unavailable"}
    assert search.await_count == 1
    assert "budget" not in str(result)


@pytest.mark.asyncio
async def test_existing_search_uses_only_locators_selects_one_and_has_deadline(monkeypatch):
    from qualification.scoring import company_evidence_investigator as investigator

    monkeypatch.setenv("EXA_API_KEY", "unit-test")
    search = AsyncMock(return_value={"results": [
        {"url": ORIGINAL, "title": CLAIM},
        {"url": ALTERNATIVE, "title": CLAIM,
         "text": "Unfetched snippet must never be used as evidence"},
        {"url": "https://example.org/other", "title": "Other news"},
    ]})
    session_factory = investigator.aiohttp.ClientSession
    sessions = []

    def session(**kwargs):
        sessions.append(kwargs)
        return session_factory(**kwargs)

    monkeypatch.setattr(investigator.aiohttp, "ClientSession", session)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = await verifier._find_same_event_source({
        "company": "pixverse.ai", "claim": CLAIM, "signal_date": "2026-05-20",
        "claimed_source_urls": [ORIGINAL],
    })
    assert result["selected_urls"] == [ALTERNATIVE]
    assert "snippet" not in str(result)
    assert search.await_count == 1
    assert sessions[0]["timeout"].total == 30
