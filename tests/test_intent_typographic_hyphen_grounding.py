"""Typographic word hyphens must not change an otherwise exact source quote."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.intent_evidence_reuse import (
    reusable_evidence_terminal_judgment, terminal_result,
)
from qualification.scoring.signal_judgment_cache import reusable_terminal_judgment


def _fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/reviewer_seams/"
        "pkf_citation_hyphen_nov15.json").read_text())
    assert hashlib.sha256(fixture["source"]["text"].encode()).hexdigest() == fixture["source_text_sha256"]
    raw = json.dumps(fixture["raw_verdict"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture["raw_verdict_sha256"]
    return fixture


@pytest.mark.parametrize("hyphen", ["\u2010", "\u2011"])
@pytest.mark.parametrize("reverse", [False, True])
def test_typographic_in_word_hyphens_preserve_exact_visible_quote(hyphen, reverse):
    source = "The Florida-based firm joined the accounting practice."
    quote = source.replace("-", hyphen)
    if reverse:
        source, quote = quote, source
    assert intent._grounded_exact_text(source, quote, whole_words=True, min_words=5)


@pytest.mark.parametrize("source,quote", [
    ("The firm acquired 10-20 companies.", "The firm acquired 10\u201120 companies."),
    ("The firm reports -10 percent growth.", "The firm reports \u201110 percent growth."),
    ("The firm reports -ten percent growth.", "The firm reports \u2011ten percent growth."),
    ("The firm reports ten-fold growth.", "The firm reports ten\u2212fold growth."),
    ("The Florida-based firm joined.", "The Florida\u2013based firm joined."),
    ("The Florida-based firm joined.", "The Florida\u2014based firm joined."),
    ("The Florida-based firm joined.", "The Florida based firm joined."),
    ("The Florida-based firm joined.", "The Georgia\u2011based firm joined."),
    ("The Florida-based firm did not join.", "The Florida\u2011based firm did join."),
    ("The Florida-based firm joined in 2026.", "The Florida\u2011based firm joined in 2027."),
    ("The Florida-based firm joined.", "The Florida\u2011based joined firm."),
])
def test_formatting_comparison_preserves_substantive_text_and_numeric_punctuation(source, quote):
    assert not intent._grounded_exact_text(source, quote, whole_words=True)


@pytest.mark.asyncio
async def test_saved_pkf_reply_repairs_its_url_without_rewriting_raw_quote_or_source(monkeypatch):
    fixture = _fixture()
    original = deepcopy(fixture)
    answer = {"answer": deepcopy(fixture["raw_verdict"]), "usage": {}}
    judge = AsyncMock(return_value=answer)
    fetch = AsyncMock(return_value={"results": [deepcopy(fixture["source"])], "statuses": []})
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    result = await intent.verify_three_stage(
        None, company_name="PKF O'Connor Davies", company_website="https://www.pkfod.com/",
        company_linkedin="https://www.linkedin.com/company/pkf-oconnor-davies/",
        source_url=fixture["source"]["url"], miner_claim=fixture["miner_claim"],
        miner_signal_date=fixture["miner_signal_date"],
        target_signal_text=fixture["target_icp_signal"], evidence_type="ACQUISITION",
        stage1_soft_reject=True, integrity_policy=True,
    )
    evaluation = result["verdict"]["signal_evaluations"][0]
    assert result["decision"] == "approve"
    assert evaluation["evidence_urls_used"] == [fixture["source"]["url"]]
    assert evaluation["supporting_quotes"] == fixture["raw_verdict"]["signal_evaluations"][0]["supporting_quotes"]
    assert "Restored a miscopied supplied evidence URL." in evaluation["risk_notes"]
    assert terminal_result(result, contents=fetch.return_value, source_url=fixture["source"]["url"])
    assert fixture == original
    assert fetch.return_value["results"] == [original["source"]]
    judge.assert_awaited_once()
    fetch.assert_awaited_once()


@pytest.mark.parametrize("unsafe", ["offsite", "digits", "query", "different_fact", "searching"])
def test_saved_pkf_quote_does_not_weaken_citation_safety(unsafe):
    fixture = _fixture()
    verdict = deepcopy(fixture["raw_verdict"])
    evaluation = verdict["signal_evaluations"][0]
    if unsafe == "offsite":
        evaluation["evidence_urls_used"] = ["https://unrelated.example/acquisition"]
    elif unsafe == "digits":
        evaluation["evidence_urls_used"][0] += "2027"
    elif unsafe == "query":
        evaluation["evidence_urls_used"][0] += "?event=other"
    elif unsafe == "different_fact":
        evaluation["supporting_quotes"] = [evaluation["supporting_quotes"][0].replace("Topping Kessler", "Different Target")]
    result = intent._apply_guardrails(
        {"claimed_source_urls": [fixture["source"]["url"]]}, verdict,
        restore_miscopied_citations=unsafe != "searching", contents={"results": [fixture["source"]]},
    )
    assert result["signal_evaluations"][0]["signal_status"] == "unable_to_verify"
    assert intent._decision(result) != "approve"


def test_saved_accepted_pkf_quote_uses_same_v1_and_v2_admission_without_mutation():
    fixture = _fixture()
    raw = deepcopy(fixture["accepted_signal"])
    raw.pop("counted_in_aggregate")
    evidence = {"schema_version": "leadpoet.intent_evidence_input.v1",
        "row": {"claimed_source_urls": raw["evidence_urls"], "signal_date": fixture["miner_signal_date"]},
        "contents": {"results": [deepcopy(fixture["source"])]}}
    expected = {"expected_urls": raw["evidence_urls"], "expected_index": 0,
        "expected_evidence_type": "ACQUISITION"}
    original, original_evidence = deepcopy(raw), deepcopy(evidence)
    assert reusable_terminal_judgment(raw, **expected)
    assert reusable_evidence_terminal_judgment(raw, evidence_input=evidence, **expected)
    assert raw == original and evidence == original_evidence
    # A second, invented quote cannot hide behind the harmless hyphen variant.
    raw["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"][0]["supporting_quotes"].append(
        "The firm acquired an unrelated company in 2027."
    )
    assert not reusable_evidence_terminal_judgment(raw, evidence_input=evidence, **expected)


@pytest.mark.parametrize("paragraph", [False, True])
def test_native_repaired_quote_reuses_the_original_signal_after_a_fresh_fetch(monkeypatch, paragraph):
    from tests.lab_arena.evidence_cache_scorer_e2e_test import setup_native, metrics

    fixture, fetch, judge, compare, host, calls, policy, icp, run = setup_native(monkeypatch, paragraph=paragraph)
    item = fixture["native_retry_answer"]["signal_evaluations"][0]
    item["evidence_urls_used"] = [fixture["source_url"].replace("partners-with", "partnes-with")]
    item["supporting_quotes"] = [re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "\u2011", quote)
        for quote in item["supporting_quotes"]]
    expected_quotes = deepcopy(item["supporting_quotes"])
    baseline, baseline_company = run()
    miner, miner_company = run(run_id="miner-identical")
    before = metrics(baseline, baseline_company, policy, icp)
    after = metrics(miner, miner_company, policy, icp)
    assert before == after and before["qualified"] is True and before["score"] > 0
    assert judge.await_count == 1 and fetch.await_count == 2
    compare.assert_not_awaited()
    assert len(host.signal_rows) == 1
    saved = next(iter(host.signal_rows.values()))["judgment"]
    assert saved["verifier_result"]["verdict"]["signal_evaluations"][0]["supporting_quotes"] == expected_quotes
