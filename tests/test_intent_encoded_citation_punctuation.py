"""A URL's encoded apostrophe is punctuation, not a different numeric ID."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.intent_evidence_reuse import terminal_result


def _fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/reviewer_seams/"
        "thunes_encoded_apostrophe_nov16.json").read_text())
    assert hashlib.sha256(fixture["source"]["text"].encode()).hexdigest() == fixture["source_text_sha256"]
    raw = json.dumps(fixture["raw_verdict"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture["raw_verdict_sha256"]
    return fixture


def _guard(fixture, verdict=None, *, supplied=None, contents=None, source_only=True):
    return intent._apply_guardrails(
        {"claimed_source_urls": supplied or [fixture["source"]["url"]]},
        deepcopy(verdict or fixture["raw_verdict"]),
        restore_miscopied_citations=source_only,
        contents={"results": [fixture["source"]]} if contents is None else contents,
    )["signal_evaluations"][0]


@pytest.mark.asyncio
async def test_captured_thunes_source_and_raw_reply_restore_only_supplied_url(monkeypatch):
    fixture = _fixture()
    original = deepcopy(fixture)
    judge = AsyncMock(return_value={"answer": deepcopy(fixture["raw_verdict"]), "usage": {}})
    fetch = AsyncMock(return_value={"results": [deepcopy(fixture["source"])], "statuses": []})
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    result = await intent.verify_three_stage(
        None, company_name="Thunes", company_website="https://thunes.com",
        company_linkedin="https://www.linkedin.com/company/thunespayments/",
        source_url=fixture["source"]["url"], miner_claim=fixture["miner_claim"],
        miner_signal_date=fixture["miner_signal_date"],
        target_signal_text=fixture["target_icp_signal"], evidence_type="MARKET_EXPANSION",
        stage1_soft_reject=True, integrity_policy=True,
    )
    evaluation = result["verdict"]["signal_evaluations"][0]
    assert result["decision"] == "approve"
    assert evaluation["evidence_urls_used"] == [fixture["source"]["url"]]
    assert evaluation["supporting_quotes"] == original["raw_verdict"]["signal_evaluations"][0]["supporting_quotes"]
    assert terminal_result(result, contents=fetch.return_value, source_url=fixture["source"]["url"])
    assert fixture == original and fetch.return_value["results"] == [original["source"]]
    judge.assert_awaited_once()
    fetch.assert_awaited_once()


@pytest.mark.parametrize("apostrophe", ["%E2%80%99", "%e2%80%99", "%E2%80%98", "%27"])
def test_word_apostrophe_encoding_does_not_create_numeric_url_identifiers(apostrophe):
    fixture = _fixture()
    verdict = deepcopy(fixture["raw_verdict"])
    verdict["signal_evaluations"][0]["evidence_urls_used"][0] = (
        verdict["signal_evaluations"][0]["evidence_urls_used"][0].replace("%E2%80%99", apostrophe))
    result = _guard(fixture, verdict)
    assert result["signal_status"] == "supported"
    assert result["evidence_urls_used"] == [fixture["source"]["url"]]


@pytest.mark.parametrize("encoded_supplied", [False, True])
def test_actual_numeric_identifiers_and_exact_query_survive_punctuation_comparison(encoded_supplied):
    fixture = _fixture()
    verdict = deepcopy(fixture["raw_verdict"])
    evaluation = verdict["signal_evaluations"][0]
    supplied = fixture["source"]["url"].replace("/news/", "/news/2026/11/16/42/")
    copied = evaluation["evidence_urls_used"][0].replace("/news/", "/news/2026/11/16/42/")
    if encoded_supplied:
        supplied, copied = copied, supplied
    query = "?edition=42&label=nation%E2%80%99s"
    supplied, copied = supplied + query, copied + query
    contents = {"results": [{**fixture["source"], "url": supplied}]}
    evaluation["evidence_urls_used"] = [copied]
    result = _guard(fixture, verdict, supplied=[supplied], contents=contents)
    assert result["signal_status"] == "supported"
    assert result["evidence_urls_used"] == [supplied]
    evaluation["evidence_urls_used"] = [copied.replace("/42/", "/43/")]
    assert _guard(fixture, verdict, supplied=[supplied], contents=contents)["signal_status"] == "unable_to_verify"


@pytest.mark.parametrize("unsafe", [
    "offsite_near_copy", "scheme", "port", "query", "numeric_id", "encoded_digit", "encoded_slash",
    "double_encoded", "numeric_apostrophe", "invented_quote", "extra_invented_quote",
    "wrong_source", "empty_source", "ambiguous", "searching",
])
def test_punctuation_repair_keeps_exact_source_and_url_safety_boundaries(unsafe):
    fixture = _fixture()
    verdict = deepcopy(fixture["raw_verdict"])
    evaluation = verdict["signal_evaluations"][0]
    url = evaluation["evidence_urls_used"][0]
    supplied, contents = [fixture["source"]["url"]], {"results": [deepcopy(fixture["source"])]}
    if unsafe == "offsite_near_copy":
        url = url.replace("www.thunes.com", "www.thunex.com")
    elif unsafe == "scheme":
        url = url.replace("https://", "http://")
    elif unsafe == "port":
        url = url.replace("www.thunes.com", "www.thunes.com:8443")
    elif unsafe == "query":
        url += "?event=nation%E2%80%99s"
    elif unsafe == "numeric_id":
        url = url.replace("%E2%80%99", "2027")
    elif unsafe == "encoded_digit":
        url = url.replace("%E2%80%99", "%32")
    elif unsafe == "encoded_slash":
        url = url.replace("%E2%80%99", "%2F")
    elif unsafe == "double_encoded":
        url = url.replace("%E2%80%99", "%25E2%2580%2599")
    elif unsafe == "numeric_apostrophe":
        url = url.replace("nation%E2%80%99s", "nation1%E2%80%99s")
        supplied[0] = supplied[0].replace("nations", "nation1s")
        contents["results"][0]["url"] = supplied[0]
    elif unsafe == "invented_quote":
        evaluation["supporting_quotes"] = ["Thunes expanded its Direct Global Network into a different country."]
    elif unsafe == "extra_invented_quote":
        evaluation["supporting_quotes"].append("Thunes expanded into another country in 2027.")
    elif unsafe == "wrong_source":
        contents["results"][0]["url"] = "https://www.thunes.com/news/different-event/"
    elif unsafe == "empty_source":
        contents["results"][0]["text"] = ""
    elif unsafe == "ambiguous":
        supplied.append(supplied[0].replace("nations", "nation"))
    evaluation["evidence_urls_used"] = [url]
    result = _guard(fixture, verdict, supplied=supplied, contents=contents, source_only=unsafe != "searching")
    assert result["signal_status"] == "unable_to_verify"
    assert result["evidence_urls_used"] == [url]


@pytest.mark.parametrize("paragraph", [False, True])
def test_encoded_word_punctuation_restoration_reuses_only_grounded_terminal_result(monkeypatch, paragraph):
    from tests.lab_arena.evidence_cache_scorer_e2e_test import setup_native, metrics

    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch, paragraph=paragraph)
    item = fixture["native_retry_answer"]["signal_evaluations"][0]
    item["evidence_urls_used"] = [fixture["source_url"].replace("partners-with", "partner%E2%80%99s-with")]
    baseline, baseline_company = run()
    miner, miner_company = run(run_id="miner-identical")
    before = metrics(baseline, baseline_company, policy, icp)
    after = metrics(miner, miner_company, policy, icp)
    assert before == after and before["qualified"] is True and before["score"] > 0
    assert judge.await_count == 1 and fetch.await_count == 2
    compare.assert_not_awaited()
    assert len(host.signal_rows) == 1
