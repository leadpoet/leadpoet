"""Source-grounded judging omits only the nonbinding search precheck."""

from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as intent


URL = "https://news.example/acme-atlas"
CLAIM = "Acme launched Atlas."
DENIAL = "Acme has not launched Atlas."


def _response(status="supported"):
    return {
        "model": "test-reviewer",
        "usage": {},
        "answer": {
            "signal_evaluations": [{
                "signal_status": status,
                "confidence": "high",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "source_accessibility": "accessible",
                "evidence_urls_used": [URL],
                "supporting_quotes": [CLAIM] if status == "supported" else [],
                "contradicting_quotes": [DENIAL] if status == "contradicted" else [],
                "unsupported_parts": [],
                "risk_notes": [],
                "claim_matches_miner_date": "no_date_in_content",
            }],
        },
    }


async def _verify(monkeypatch, response, *, text=CLAIM, soft=True, fetch_ok=True):
    events = []

    async def fetch(_urls):
        events.append("fetch")
        return {
            "results": [{"url": URL, "title": "Acme news", "text": text}] if fetch_ok else [],
            "statuses": [{"url": URL, "source": "scrapingdog", "stage": "ok" if fetch_ok else "provider_error"}],
        }

    async def review(_client, model, prompt, **_kwargs):
        events.append("review")
        if soft:
            assert model == intent.ARENA_EVIDENCE_MODEL
            assert text in prompt
        return response

    call = AsyncMock(side_effect=review)
    fetch_call = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(intent, "_call_openrouter", call)
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch_call)
    result = await intent.verify_three_stage(
        None,
        company_name="Acme",
        company_website="https://acme.com",
        company_linkedin="https://www.linkedin.com/company/acme",
        source_url=URL,
        miner_claim=CLAIM,
        target_signal_text="Launched a product",
        stage1_soft_reject=soft,
        integrity_policy=True,
    )
    return result, call, fetch_call, events


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "text", "decision"), [
    ("supported", CLAIM, "approve"),
    ("contradicted", DENIAL, "reject"),
])
async def test_source_grounded_decision_requires_fetch_and_final_review(
    monkeypatch, status, text, decision,
):
    result, call, fetch, events = await _verify(monkeypatch, _response(status), text=text)

    assert result["decision"] == decision
    assert result["client_ready"] is (decision == "approve")
    assert events == ["fetch", "review"]
    call.assert_awaited_once()
    fetch.assert_awaited_once()
    assert result["stage1"]["status"] == "skipped_source_grounded"
    assert result["stage1"]["model"] is None
    assert result["stage1"]["usage"] == {}


@pytest.mark.asyncio
async def test_missing_source_content_cannot_skip_to_approval(monkeypatch):
    result, call, fetch, events = await _verify(monkeypatch, _response(), fetch_ok=False)

    assert result["decision"] == "unavailable"
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "evidence_fetch_failed"
    assert events == ["fetch"]
    call.assert_not_awaited()
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_review_failure_remains_unavailable(monkeypatch):
    result, call, fetch, events = await _verify(monkeypatch, {"_error": "invalid_json_content"})

    assert result["decision"] == "unavailable"
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "stage3_llm_error:invalid_json_content"
    assert events == ["fetch", "review"]
    call.assert_awaited_once()
    fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "decision"), [
    ("supported", "approve"), ("contradicted", "reject"),
])
async def test_standalone_mode_preserves_first_pass_decision(monkeypatch, status, decision):
    result, call, fetch, events = await _verify(monkeypatch, _response(status), soft=False)

    assert result["decision"] == decision
    assert events == ["review"]
    call.assert_awaited_once()
    assert call.await_args.args[1] == intent.STAGE1_MODEL
    fetch.assert_not_awaited()
    assert result["stage3"] is None


LONG_URL = (
    "https://news.example/fort-worths-acme-group-acquires-california-based-"
    "plasma-ruggedized-solutions/"
)
MISCOPIED_URL = LONG_URL.replace("california", "calcium")


async def _verify_miscopied_citation(monkeypatch, **mode):
    response = _response()
    response["answer"]["signal_evaluations"][0]["evidence_urls_used"] = [MISCOPIED_URL]

    async def fetch(_urls):
        return {
            "results": [{"url": LONG_URL, "title": "Acme news", "text": CLAIM}],
            "statuses": [{"url": LONG_URL, "source": "scrapingdog", "stage": "ok"}],
        }

    monkeypatch.setattr(intent, "_call_openrouter", AsyncMock(return_value=response))
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", AsyncMock(side_effect=fetch))
    return await intent.verify_three_stage(
        None,
        company_name="Acme",
        company_website="https://acme.com",
        company_linkedin="https://www.linkedin.com/company/acme",
        source_url=LONG_URL,
        miner_claim=CLAIM,
        target_signal_text="Launched a product",
        stage1_soft_reject=True,
        **mode,
    )


@pytest.mark.asyncio
async def test_source_only_judge_miscopying_the_supplied_url_still_approves(monkeypatch):
    result = await _verify_miscopied_citation(monkeypatch, integrity_policy=True)

    item = result["verdict"]["signal_evaluations"][0]
    assert result["decision"] == "approve"
    assert item["signal_status"] == "supported"
    assert item["evidence_urls_used"] == [LONG_URL]
    assert item["risk_notes"] == ["Restored a miscopied supplied evidence URL."]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [
    {"integrity_policy": False},
    {"integrity_policy": True, "evidence_type": "SOCIAL_POSTING"},
])
async def test_searching_judge_citing_a_near_identical_url_is_not_approved(monkeypatch, mode):
    result = await _verify_miscopied_citation(monkeypatch, **mode)

    assert result["decision"] != "approve"
    assert result["client_ready"] is False
