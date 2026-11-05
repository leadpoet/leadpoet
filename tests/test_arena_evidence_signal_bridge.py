"""Real signal verifier/date/scoring paths reuse only fresh admitted evidence."""
import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt, IntentSignal
from lab_arena.contracts import document_hash
from qualification.scoring import lead_scorer, verification_helpers
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.competition import _normalized_icp, count_penalizable_false_positives
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.signal_judgment_cache import install_exchange, SignalJudgmentCacheError
from tests.test_retained_source_native_retry import _fixture


class HostTransport:
    """Only the existing host transport is fake; no fake judge/cache response."""

    def __init__(self):
        self.rows = {}
        self.requests = []
        self.uses = []
        self.run = "baseline"
        self.fail_release = False
        self.fail_complete = False

    async def __call__(self, request):
        request = deepcopy(request)
        self.requests.append(request)
        assert request["evidence_judgment_policy"] == "evidence_judgment_v2"
        assert request["judgment_kind"] == "signal_evidence_v2"
        assert request["evidence_hash"] == document_hash(request["evidence_input"])
        assert request["evidence_input"]["contents"]["results"][0]["text"]
        operation = request["operation"]
        if operation == "candidates":
            return {"candidates": [{"reference": row["reference"], "judgment": row["judgment"]}
                for row in self.rows.values() if row["evidence_hash"] == request["evidence_hash"]]}
        key = document_hash({"evidence": request["evidence_hash"], "claim": request["claim"]})
        if operation == "claim":
            reference = request.get("candidate_reference") or {"cache_key": key, "authority_slot": 0}
            row = self.rows.get(reference["cache_key"])
            if row is not None:
                assert row["evidence_hash"] == request["evidence_hash"]
                if row["judgment"]["claim"] != request["claim"]:
                    proof = request["equivalence_receipt"]
                    assert proof["source_claim_hash"] == document_hash(row["judgment"]["claim"])
                    assert proof["current_claim_hash"] == document_hash(request["claim"])
                self.uses.append((self.run, row["source_run"], reference))
                return {"status": "hit", "reference": reference, "judgment": row["judgment"]}
            return {"status": "miss", "reference": reference}
        assert operation == "complete"
        if request["judgment"] is None:
            if self.fail_release:
                raise RuntimeError("test release unavailable")
            return {"status": "released", "reference": request["reference"]}
        if self.fail_complete:
            raise RuntimeError("test complete unavailable")
        reference = request["reference"]
        self.rows[reference["cache_key"]] = {"reference": reference,
            "evidence_hash": request["evidence_hash"], "judgment": request["judgment"],
            "source_run": self.run}
        self.uses.append((self.run, self.run, reference))
        return {"status": "stored", "reference": reference}


def _setup(monkeypatch, *, comparison="equivalent"):
    fixture = _fixture()
    monkeypatch.setenv("ARENA_EVIDENCE_JUDGMENT_POLICY", "evidence_judgment_v2")
    fetch = AsyncMock(return_value={"results": [{"url": fixture["source_url"],
        "title": "EPAM and Wiz", "text": fixture["source_text"],
        "meta": {"final_url": fixture["source_url"]}, "source_publication_date": "2026-08-25"}],
        "statuses": [{"url": fixture["source_url"], "source": "scrapingdog", "stage": "ok"}]})
    async def native_answer(_client, _model, prompt, **kwargs):
        assert fixture["source_text"] in prompt
        assert "exact supplied source extraction" in prompt
        wrong = "completed its acquisition of Wiz" in prompt
        return {"model": verifier.ARENA_EVIDENCE_MODEL, "usage": {}, "answer": deepcopy(
            fixture["wrong_acquisition_answer"] if wrong else fixture["native_retry_answer"])}
    judge = AsyncMock(side_effect=native_answer)
    compare = AsyncMock(return_value=json.dumps({"decision": comparison,
        "material_differences": [] if comparison == "equivalent" else ["Different event"]}))
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", compare)
    return fixture, fetch, judge, compare, HostTransport()


async def _score(fixture, transport, *, claim=None, snippet=None, source="news", run="baseline", no_time_decay=True):
    claim = fixture["miner_claim"] if claim is None else claim
    signal = IntentSignal(source=source, description=claim, snippet=claim if snippet is None else snippet,
        url=fixture["source_url"], date=fixture["miner_date"], matched_icp_signal=0)
    company = CompanyOutput.model_construct(**fixture["company"], intent_signals=[signal])
    icp = ICPPrompt.model_validate(_normalized_icp(fixture["icp"]))
    transport.run = run
    with use_evaluation_date("2026-10-24"), install_exchange(transport):
        return await lead_scorer.score_company_competition_intent_signal(
            company, icp, integrity_policy=True, company_quality=True,
            verified_company_identity=fixture["identity"],
            no_time_decay=no_time_decay,
        )


@pytest.mark.asyncio
async def test_baseline_and_miners_share_native_judgment_after_fresh_fetch_and_strict_comparison(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    baseline = await _score(fixture, host)
    exact = await _score(fixture, host, run="miner-exact")
    compare.assert_not_awaited()
    rewritten = await _score(fixture, host, run="miner-rewrite",
        claim="EPAM partners with Wiz to help enterprises reduce cloud risk.")
    assert fetch.await_count == 3 and judge.await_count == compare.await_count == 1
    assert baseline == exact == rewritten
    assert baseline[1] > 0 and len(host.rows) == 1
    assert [(run, source) for run, source, _ in host.uses] == [
        ("baseline", "baseline"), ("miner-exact", "baseline"), ("miner-rewrite", "baseline")]
    assert all(row.get("signal_judgment_reference") for result in (baseline, exact, rewritten) for row in result[-1])
    assert compare.await_args.kwargs["response_format"]["json_schema"]["name"] == "arena_claim_equivalence_v1"


@pytest.mark.asyncio
async def test_changed_same_url_source_runs_unchanged_native_judge_and_stores_new_variant(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    first = await _score(fixture, host)
    fetch.return_value["results"][0]["text"] += " Additional current source text."
    second = await _score(fixture, host, run="miner-changed-source")
    assert first[1] == second[1] > 0
    assert fetch.await_count == judge.await_count == 2
    compare.assert_not_awaited()
    assert len(host.rows) == 2


@pytest.mark.asyncio
async def test_false_acquisition_cannot_reuse_partnership_and_remains_penalizable_primary(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch, comparison="different")
    approved = await _score(fixture, host)
    rejected = await _score(fixture, host, claim=fixture["wrong_acquisition_claim"], run="miner-false")
    assert approved[1] > 0 and rejected[1] == 0
    assert fetch.await_count == judge.await_count == 2
    assert rejected[-1][0]["judge_verdict"]["decision"] == "rejected_three_stage"
    assert count_penalizable_false_positives([{"intent_signals_detail": rejected[-1]}],
        icp_has_intent_signals=True) == (0, 1)
    assert len(host.rows) == 2


@pytest.mark.asyncio
async def test_grounded_negative_reuses_without_changing_native_zero(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    first = await _score(fixture, host, claim=fixture["wrong_acquisition_claim"])
    second = await _score(fixture, host, claim=fixture["wrong_acquisition_claim"], run="miner")
    assert first == second and first[1] == 0
    assert fetch.await_count == 2 and judge.await_count == 1
    compare.assert_not_awaited()
    assert len(host.rows) == 1 and second[-1][0].get("signal_judgment_reference")


@pytest.mark.asyncio
async def test_equivalent_false_claim_reuses_negative_but_corrected_true_claim_runs_native(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    original = fixture["wrong_acquisition_claim"]
    first = await _score(fixture, host, claim=original)
    rewritten = await _score(fixture, host,
        claim="On August 25, 2026, EPAM completed its acquisition of Wiz.", run="miner-rewrite")
    assert first == rewritten and rewritten[1] == 0
    assert judge.await_count == compare.await_count == 1 and fetch.await_count == 2
    corrected = await _score(fixture, host, run="miner-corrected")
    assert corrected[1] > 0 and judge.await_count == 2 and fetch.await_count == 3
    assert len(host.rows) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("comparison_failure", ["uncertain", "timeout"])
async def test_uncertain_or_failed_comparison_runs_native_without_reusing_or_zeroing(monkeypatch, comparison_failure):
    fixture, fetch, judge, compare, host = _setup(monkeypatch, comparison="uncertain")
    first = await _score(fixture, host)
    if comparison_failure == "timeout":
        compare.side_effect = asyncio.TimeoutError("test comparison timeout")
    second = await _score(fixture, host,
        claim="EPAM partners with Wiz to help enterprises reduce cloud risk.", run="miner")
    assert first[1] == second[1] > 0 and fetch.await_count == judge.await_count == 2
    assert compare.await_count == 1 and len(host.rows) == 2


@pytest.mark.asyncio
async def test_verifier_timeout_releases_then_retries_instead_of_caching_zero(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    judge.side_effect = None
    judge.return_value = {"_error": "timeout"}
    first = await _score(fixture, host)
    assert first[1] == 0 and first[-1][0]["judge_verdict"]["decision"] == "rejected_verifier_error"
    assert not host.rows and host.requests[-1]["operation"] == "complete" and host.requests[-1]["judgment"] is None
    assert count_penalizable_false_positives([{"intent_signals_detail": first[-1]}],
        icp_has_intent_signals=True) == (0, 0)
    judge.return_value = {"model": verifier.ARENA_EVIDENCE_MODEL, "usage": {}, "answer": fixture["native_retry_answer"]}
    second = await _score(fixture, host, run="retry")
    assert second[1] > 0 and fetch.await_count == judge.await_count == 2 and len(host.rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_release", [False, True])
async def test_cancellation_releases_owned_reservation_and_preserves_original(monkeypatch, fail_release):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    host.fail_release = fail_release
    judge.side_effect = asyncio.CancelledError("native judge cancelled")
    with pytest.raises(asyncio.CancelledError, match="native judge cancelled"):
        await _score(fixture, host)
    assert host.requests[-1]["operation"] == "complete" and host.requests[-1]["judgment"] is None
    assert not host.rows


@pytest.mark.asyncio
async def test_unsupported_source_runs_native_without_legacy_early_claim(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    fetch.return_value["statuses"][0]["stage"] = "wayback_ok"
    first = await _score(fixture, host)
    second = await _score(fixture, host, run="miner")
    assert first[1] == second[1] > 0 and fetch.await_count == judge.await_count == 2
    assert not host.requests and not host.rows
    assert not first[-1][0].get("signal_judgment_reference")


@pytest.mark.asyncio
async def test_no_exchange_preserves_native_path_even_with_prospective_env(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    signal = IntentSignal(source="news", description=fixture["miner_claim"], snippet=fixture["miner_claim"],
        url=fixture["source_url"], date=fixture["miner_date"], matched_icp_signal=0)
    with use_evaluation_date("2026-10-24"):
        result = await lead_scorer.score_company_competition_intent_signal(
            CompanyOutput.model_construct(**fixture["company"], intent_signals=[signal]),
            ICPPrompt.model_validate(_normalized_icp(fixture["icp"])), integrity_policy=True,
            company_quality=True, verified_company_identity=fixture["identity"])
    assert result[1] > 0 and fetch.await_count == judge.await_count == 1
    assert not result[-1][0].get("signal_judgment_reference")


@pytest.mark.asyncio
async def test_postprocessing_error_releases_without_changing_original_exception(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    def fail_decay(*args, **kwargs):
        raise RuntimeError("test decay failed")
    monkeypatch.setattr(lead_scorer, "_apply_signal_time_decay", fail_decay)
    with pytest.raises(RuntimeError, match="test decay failed"):
        await _score(fixture, host, no_time_decay=False)
    assert host.requests[-1]["operation"] == "complete" and host.requests[-1]["judgment"] is None
    assert not host.rows


@pytest.mark.asyncio
async def test_failed_final_store_releases_owned_reservation(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    host.fail_complete = True
    with pytest.raises(SignalJudgmentCacheError, match="exchange unavailable"):
        await _score(fixture, host)
    assert host.requests[-1]["operation"] == "complete" and host.requests[-1]["judgment"] is None
    assert not host.rows


@pytest.mark.asyncio
async def test_large_multibyte_source_snapshot_skips_lookup_and_keeps_native_score(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    fetch.return_value["results"][0]["text"] += "界" * 55_000
    fetch.return_value["results"][0]["meta"]["large_source_metadata"] = "界" * 120_000
    result = await _score(fixture, host)
    assert result[1] > 0 and fetch.await_count == judge.await_count == 1
    assert not host.requests and not host.rows
    assert not result[-1][0].get("signal_judgment_reference")


@pytest.mark.asyncio
async def test_large_multibyte_native_reply_releases_before_oversized_complete(monkeypatch):
    fixture, fetch, judge, compare, host = _setup(monkeypatch)
    answer = deepcopy(fixture["native_retry_answer"])
    answer["summary"] = "界" * 200_000
    judge.side_effect = None
    judge.return_value = {"model": verifier.ARENA_EVIDENCE_MODEL, "usage": {}, "answer": answer}
    result = await _score(fixture, host)
    assert result[1] > 0 and fetch.await_count == judge.await_count == 1
    assert [request["operation"] for request in host.requests] == ["candidates", "claim", "complete"]
    assert host.requests[-1]["judgment"] is None and not host.rows
    assert not result[-1][0].get("signal_judgment_reference")
