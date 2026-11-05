"""Fresh full-source fingerprints gate prospective native-verdict reuse."""
import copy
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_evidence_reuse as reuse
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
from tests.test_retained_source_native_retry import _fixture


async def _run(monkeypatch, *, callback=None, fixture=None, contents=None,
               answer=None, claim=None, extra=None, provider_error=None):
    fixture = copy.deepcopy(fixture or _fixture())
    url = fixture["source_url"]
    fetch = AsyncMock(return_value=contents or {
        "results": [{"url": url, "title": "EPAM and Wiz", "text": fixture["source_text"],
                     "meta": {"final_url": url}, "source_publication_date": "2026-08-25"}],
        "statuses": [{"url": url, "source": "scrapingdog", "stage": "ok"}],
    })
    judge = AsyncMock(return_value={"_error": provider_error} if provider_error else {
        "model": verifier.ARENA_EVIDENCE_MODEL, "usage": {},
        "answer": copy.deepcopy(fixture["native_retry_answer"] if answer is None else answer)})
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)

    async def verify():
        with use_evaluation_date("2026-10-24"):
            return await verifier.verify_three_stage(
                None, **fixture["company"], source_url=url,
                miner_claim=fixture["miner_claim"] if claim is None else claim,
                target_signal_text=fixture["target_signal_text"],
                buyer_request_context=fixture["buyer_request_context"],
                miner_signal_date=fixture["miner_date"],
                **{"evidence_type": "PARTNERSHIP", "stage1_soft_reject": True,
                   "integrity_policy": True, "company_quality": True,
                   "verified_company_identity": fixture["identity"], "buyer_max_age_days": 365,
                   **(extra or {})},
            )
    if callback is None:
        result = await verify()
    else:
        with reuse.install_evidence_reuse(callback):
            result = await verify()
    return result, fetch, judge


@pytest.mark.asyncio
async def test_legacy_path_does_not_build_or_exchange_reuse_snapshot(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy scorer must not build prospective evidence")
    monkeypatch.setattr(reuse, "snapshot", forbidden)
    result, fetch, judge = await _run(monkeypatch)
    assert result["decision"] == "approve"
    assert fetch.await_count == judge.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_claim", [False, True])
async def test_actual_native_positive_and_false_acquisition_outcomes_are_unchanged(monkeypatch, wrong_claim):
    fixture, events = _fixture(), []
    async def exchange(request):
        events.append(copy.deepcopy(request))
        return {"status": "miss"}
    result, fetch, judge = await _run(monkeypatch, fixture=fixture, callback=exchange,
        claim=fixture["wrong_acquisition_claim"] if wrong_claim else None,
        answer=fixture["wrong_acquisition_answer"] if wrong_claim else None)
    assert result["decision"] == ("reject" if wrong_claim else "approve")
    assert fetch.await_count == judge.await_count == 1
    assert [item["operation"] for item in events] == ["lookup", "complete"]
    assert events[1]["judgment"] == result
    assert events[0]["evidence_input"] == events[1]["evidence_input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_claim", [False, True])
@pytest.mark.parametrize("evidence_model", [None, "openai/gpt-oss-120b"])
async def test_exact_text_hit_fetches_again_and_preserves_native_result(monkeypatch, wrong_claim, evidence_model):
    if evidence_model is not None:
        monkeypatch.setenv("ARENA_INTENT_EVIDENCE_MODEL", evidence_model)
    storage, events = {}, []
    fixture = _fixture()
    inputs = {"claim": fixture["wrong_acquisition_claim"],
              "answer": fixture["wrong_acquisition_answer"]} if wrong_claim else {}
    async def exchange(request):
        events.append(request["operation"])
        key = (reuse.evidence_fingerprint(request["evidence_input"]),
               json.dumps(request["claim"], sort_keys=True))
        if request["operation"] == "complete":
            storage[key] = request["judgment"]
        if key in storage:
            return {"status": "hit", "judgment": storage[key]}
        return {"status": "miss"}
    first, first_fetch, first_judge = await _run(monkeypatch, callback=exchange, **inputs)
    second, second_fetch, second_judge = await _run(monkeypatch, callback=exchange, **inputs)
    assert first == second
    assert first_fetch.await_count == second_fetch.await_count == first_judge.await_count == 1
    second_judge.assert_not_awaited()
    assert events == ["lookup", "complete", "lookup"]


@pytest.mark.asyncio
async def test_frozen_evidence_model_change_has_distinct_cache_identity(monkeypatch):
    captured = []
    async def exchange(request):
        if request["operation"] == "lookup":
            captured.append(request["evidence_input"])
        return {"status": "miss"}
    for model in ("openai/gpt-6-luna", "openai/gpt-oss-120b"):
        monkeypatch.setenv("ARENA_INTENT_EVIDENCE_MODEL", model)
        result, _, judge = await _run(monkeypatch, callback=exchange)
        assert result["decision"] == "approve"
        assert judge.call_args.args[1] == model
    assert [item["model"] for item in captured] == ["openai/gpt-6-luna", "openai/gpt-oss-120b"]
    assert reuse.evidence_fingerprint(captured[0]) != reuse.evidence_fingerprint(captured[1])
    captured.clear()
    await _run(monkeypatch, callback=exchange, extra={"stage3_model": "openai/gpt-6-luna"})
    assert not captured  # An explicit different model is still outside the frozen evidence role.


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["text", "source_publication_date", "meta", "title"])
async def test_same_url_changed_source_or_metadata_changes_fingerprint(monkeypatch, field):
    fixture, captured = _fixture(), []
    async def exchange(request):
        if request["operation"] == "lookup":
            captured.append(request["evidence_input"])
        return {"status": "miss"}
    await _run(monkeypatch, callback=exchange)
    page = {"url": fixture["source_url"], "title": "EPAM and Wiz", "text": fixture["source_text"],
            "meta": {"final_url": fixture["source_url"]}, "source_publication_date": "2026-08-25"}
    page[field] = {**page[field], "is_closed": True} if field == "meta" else (
        "2026-08-26" if field == "source_publication_date" else page[field] + " Changed source content")
    await _run(monkeypatch, callback=exchange, contents={"results": [page],
        "statuses": [{"url": fixture["source_url"], "source": "scrapingdog", "stage": "ok"}]})
    assert len(captured) == 2
    assert reuse.evidence_fingerprint(captured[0]) != reuse.evidence_fingerprint(captured[1])


@pytest.mark.asyncio
async def test_snapshot_retains_full_body_instead_of_truncated_receipt(monkeypatch):
    fixture, captured = _fixture(), []
    fixture["source_text"] += "x" * 15_000 + " full source tail"
    async def exchange(request):
        captured.append(request)
        return {"status": "miss"}
    result, _, _ = await _run(monkeypatch, fixture=fixture, callback=exchange)
    assert captured[0]["evidence_input"]["contents"]["results"][0]["text"] == fixture["source_text"]
    assert "full source tail" not in result["verified_source_context"][0]["text"]


def test_wording_is_separate_but_dates_are_exact_evidence_identity():
    row = {"claim": "Original", "signal_date": "2026-08-25", "_evidence_bundle": [
        {"url": "https://example.com", "description": "Original", "snippet": "Quote", "date": "2026-08-25"}]}
    kwargs = {"verified_identity_context": {}, "model": "native", "company_quality": True,
              "decision_context": {}}
    first, claim = reuse.snapshot(row, {"results": []}, **kwargs)
    changed = copy.deepcopy(row)
    changed["claim"] = changed["_evidence_bundle"][0]["description"] = "Equivalent wording"
    changed["_evidence_bundle"][0]["snippet"] = "Other wording"
    second, new_claim = reuse.snapshot(changed, {"results": []}, **kwargs)
    assert first == second and claim != new_claim
    changed["_evidence_bundle"][0]["date"] = "2026-08-26"
    third, _ = reuse.snapshot(changed, {"results": []}, **kwargs)
    assert reuse.evidence_fingerprint(first) != reuse.evidence_fingerprint(third)
    with use_evaluation_date("2026-10-24"):
        today, _ = reuse.snapshot(row, {"results": []}, **kwargs)
    with use_evaluation_date("2026-10-25"):
        tomorrow, _ = reuse.snapshot(row, {"results": []}, **kwargs)
    assert reuse.evidence_fingerprint(today) != reuse.evidence_fingerprint(tomorrow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["social", "podcast", "search_model", "archive_fetch", "funding_context", "retained", "timeout", "review"])
async def test_unsupported_or_incomplete_paths_do_not_complete_candidate(monkeypatch, case):
    fixture, events, extra = _fixture(), [], {}
    answer, contents = None, None
    if case in {"social", "podcast"}:
        extra["evidence_type"] = "SOCIAL_POSTING" if case == "social" else "PODCAST_APPEARANCE"
    elif case == "search_model":
        extra["stage3_model"] = "perplexity/sonar-pro"
    elif case == "archive_fetch":
        contents = {"results": [{"url": fixture["source_url"], "text": fixture["source_text"]}],
                    "statuses": [{"url": fixture["source_url"], "source": "scrapingdog", "stage": "wayback_ok"}]}
    elif case == "funding_context":
        extra["verified_funding_stage_context"] = [{"dimension": "stage", "url": fixture["source_url"], "text": fixture["source_text"]}]
    elif case == "retained":
        contents = {"results": [], "statuses": [{"url": fixture["source_url"], "source": "none",
            "sd_stage": "all_tiers_exhausted:http_502", "exa_stage": "exa_transient_exhausted"}]}
        extra["verified_source_pages"] = {fixture["source_url"]: {
            "final_url": fixture["source_url"], "text": fixture["source_text"]}}
    elif case == "review":
        answer = copy.deepcopy(fixture["native_retry_answer"])
        answer["signal_evaluations"][0]["signal_status"] = "unable_to_verify"
    async def exchange(request):
        events.append(request)
        return {"status": "miss"}
    result, fetch, judge = await _run(monkeypatch, callback=exchange, fixture=fixture,
                                    answer=answer, contents=contents, extra=extra,
                                    provider_error="timeout" if case == "timeout" else None)
    assert fetch.await_count == judge.await_count == 1
    assert not any(event["operation"] == "complete" for event in events)
    if case in {"social", "podcast", "search_model", "archive_fetch", "funding_context", "retained"}:
        assert not events


@pytest.mark.asyncio
async def test_invalid_hit_is_infrastructure_failure(monkeypatch):
    async def exchange(request):
        return {"status": "hit", "judgment": {"decision": "approve", "client_ready": True}}
    with pytest.raises(SignalJudgmentCacheError, match="not terminal"):
        await _run(monkeypatch, callback=exchange)


def test_context_is_local_and_restored():
    async def exchange(request):
        return None
    assert reuse.current_callback() is None
    with reuse.install_evidence_reuse(exchange):
        assert reuse.current_callback() is exchange
    assert reuse.current_callback() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["search", "linked", "archive", "dynamic", "ats"])
async def test_existing_complete_native_recovery_paths_bypass_reuse(monkeypatch, path):
    events = []
    async def exchange(request):
        events.append(request)
        return {"status": "miss"}
    with reuse.install_evidence_reuse(exchange):
        if path == "search":
            from tests.test_intent_unavailable_same_event_source import run_verifier
            result, fetch, search, judge = await run_verifier(monkeypatch)
            assert result["client_ready"] is True
            assert fetch.await_count == 2 and search.await_count == judge.await_count == 1
        elif path == "linked":
            from tests.test_intent_same_event_resolution import test_max_retail_index_follows_only_selected_same_event_link
            await test_max_retail_index_follows_only_selected_same_event_link(monkeypatch, False, False)
        elif path == "archive":
            from tests.test_intent_same_event_resolution import test_archive_card_uses_existing_one_fetch_one_judge_and_date_gate
            await test_archive_card_uses_existing_one_fetch_one_judge_and_date_gate(monkeypatch)
        elif path == "dynamic":
            from tests.test_intent_first_party_careers_dynamic import test_full_verifier_preserves_visible_cards_after_dynamic_failure
            await test_full_verifier_preserves_visible_cards_after_dynamic_failure(monkeypatch, True)
        else:
            from tests.test_intent_ats_provider_contract import test_encoded_greenhouse_job_reaches_stage3
            await test_encoded_greenhouse_job_reaches_stage3(monkeypatch)
    assert not events
