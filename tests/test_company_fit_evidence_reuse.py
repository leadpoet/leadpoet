"""Fit authority is adopted only after fresh exact source equality."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import company_fit_evidence_reuse as reuse, lead_scorer
from qualification.scoring.company_fit_decision import company_fit_unavailable
from tests.lab_arena.verified_fit_checkpoint_test import native_fit
from tests.test_arena_intent_retry_cache import _company_model, _icp_model, _identity

HOME = "https://acme.example"
PAGE = "https://acme.example/about"
TEXT = "Acme provides software."


def fit_result(negative=False):
    fit = native_fit()
    identity = fit.details["dimension_evidence"]["identity"]
    identity.update(identity=_identity(), web_identity_receipt=_identity(), homepage_identity_decision="match")
    if negative:
        fit.decision = "mismatch"
        fit.details["company_fit_decision"] = "mismatch"
        fit.details["company_fit_dimensions"]["industry"] = "mismatch"
        fit.details["dimension_evidence"]["industry"]["decision"] = "mismatch"
    return fit


def record_sources(text=TEXT):
    reuse.record_source("homepage", {"url": HOME}, {"status": 200, "final_url": HOME, "text": text})
    reuse.record_source("investigator_page", {"url": PAGE, "stealth_mode": False,
        "archive_targets": [], "archive_company_names": []}, {"result": {
            "ok": True, "url": PAGE, "final_url": PAGE, "text": text,
        }, "contexts": {}})


class Authority:
    def __init__(self):
        self.rows = {}
        self.operations = []

    async def exchange(self, request):
        self.operations.append(request["operation"])
        if request["operation"] == "candidates":
            return {"candidates": [{"reference": {"variant": key}, "judgment": deepcopy(value)}
                                    for key, value in reversed(list(self.rows.items()))][:5]}
        key = request["source_snapshot_hash"]
        if request["operation"] == "claim":
            if key in self.rows:
                return {"status": "hit", "reference": {"variant": key}, "judgment": deepcopy(self.rows[key])}
            return {"status": "miss", "reference": {"variant": key}}
        assert request["operation"] == "complete"
        self.rows.setdefault(key, deepcopy(request["judgment"]))
        return {"status": "stored"}


def harness(monkeypatch, *, negative=False):
    authority = Authority()
    current = {"text": TEXT, "calls": 0, "timeout": False, "unavailable": False}

    async def read(entry):
        if current["timeout"]:
            raise TimeoutError("fresh source unavailable")
        if entry["kind"] == "homepage":
            reuse.record_source("homepage", entry["request"], {"status": 200, "final_url": HOME, "text": current["text"]})
        else:
            reuse.record_source("investigator_page", entry["request"], {"result": {
                "ok": True, "url": PAGE, "final_url": PAGE, "text": current["text"],
            }, "contexts": {}})
    monkeypatch.setattr(reuse, "_read_source", read)

    async def native(company):
        current["calls"] += 1
        assert all(signal.description == "" for signal in company.intent_signals)
        record_sources(current["text"])
        if current["unavailable"]:
            return company_fit_unavailable("source unavailable", details={"failure_class": "provider_error"})
        return fit_result(negative)

    async def score(company=None):
        with reuse.install_fit_evidence_reuse(authority.exchange):
            return await reuse.fit_with_evidence_reuse(native, company=company or _company_model(), linkedin={}, matched={}, required={})
    return current, authority, score


@pytest.mark.parametrize("negative", [False, True])
def test_rewrite_revalidates_sources_before_reusing_exact_native_receipt(monkeypatch, negative):
    current, authority, score = harness(monkeypatch, negative=negative)
    original = asyncio.run(score())
    rewritten = _company_model()
    rewritten.intent_signals[0].description = "Different intent claim wording"
    observed = asyncio.run(score(rewritten))
    assert current["calls"] == 1
    assert observed.receipt("company_fit") == original.receipt("company_fit")
    assert len(authority.rows) == 1
    assert authority.operations == ["candidates", "claim", "complete", "candidates", "claim"]


def test_changed_source_appends_variant_without_overwriting_original(monkeypatch):
    current, authority, score = harness(monkeypatch)
    asyncio.run(score())
    first = deepcopy(authority.rows)
    current["text"] = TEXT + " New company evidence."
    asyncio.run(score())
    assert current["calls"] == 2
    assert len(authority.rows) == 2
    assert all(authority.rows[key] == value for key, value in first.items())
    asyncio.run(score())
    assert current["calls"] == 2


def test_fresh_timeout_never_adopts_old_receipt_or_caches_provider_unavailable(monkeypatch):
    current, authority, score = harness(monkeypatch)
    asyncio.run(score())
    current.update(timeout=True, unavailable=True)
    observed = asyncio.run(score())
    assert current["calls"] == 2 and observed.decision == "unavailable"
    assert len(authority.rows) == 1
    assert authority.operations[-1] == "candidates"


def test_incomplete_evidence_and_unproven_negative_stay_native():
    with reuse.capture_fit_evidence() as capture:
        reuse.record_source("homepage", {"url": HOME}, {"status": 200, "final_url": HOME, "text": TEXT})
    assert reuse.build_fit_snapshot(fit_result(), {}, {}, {}, capture) is None
    assert reuse.build_fit_snapshot(company_fit_unavailable("not proven"), {}, {}, {}, capture) is None


def test_no_source_copy_or_input_mutation_when_observation_is_disabled():
    company = _company_model()
    before = company.model_dump(mode="json")
    canonical = reuse.canonical_fit_company(company)
    assert company.model_dump(mode="json") == before
    assert canonical.intent_signals[0].url == company.intent_signals[0].url
    assert canonical.intent_signals[0].date == company.intent_signals[0].date
    assert canonical.company_stage_evidence == company.company_stage_evidence
    assert all(signal.description == "" for signal in canonical.intent_signals)


def test_observer_preserves_success_failure_and_source_projection():
    @reuse.observed_source("linkedin_contents", ("profile_url", "expected_company_identity"), ("source_text_sink",))
    async def fetch(profile_url, *, expected_company_identity=None, source_text_sink=None):
        if source_text_sink is not None:
            source_text_sink.update(url=profile_url, text=TEXT)
        return {"outcome": "verified", "url": profile_url}
    with reuse.capture_fit_evidence() as captured:
        result = asyncio.run(fetch(PAGE, source_text_sink={}))
    assert result["outcome"] == "verified" and captured.complete
    entry = captured.manifest()["sources"][0]
    assert entry["request"]["capture_sinks"] == ["source_text_sink"]
    @reuse.observed_source("investigator_exa", ("url",))
    async def failed(url):
        raise TimeoutError("unchanged native exception")
    with reuse.capture_fit_evidence() as captured:
        with pytest.raises(TimeoutError): asyncio.run(failed(PAGE))
    assert not captured.complete


@pytest.mark.parametrize("mutation", [
    lambda value: value["company_fit_receipt"].update(decision="unavailable"),
    lambda value: value["company_fit_receipt"]["company_fit_dimensions"].update(industry="unavailable"),
    lambda value: value["evidence_manifest"]["sources"][0].update(response_hash="sha256:" + "z" * 64),
    lambda value: value.update(source_snapshot_hash="sha256:" + "0" * 64),
])
def test_invalid_snapshot_cannot_be_adopted(mutation):
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    snapshot = reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured)
    assert snapshot is not None
    mutation(snapshot)
    with pytest.raises(reuse.FitEvidenceReuseError): reuse.validate_fit_snapshot(snapshot)


def test_v2_free_prechecks_do_not_adopt_a_cached_duplicate(monkeypatch):
    async def native(*_args, **_kwargs):
        raise AssertionError("duplicate should return before native paid fit")
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", native)
    requests = []
    async def exchange(request):
        requests.append(request)
        raise AssertionError("free duplicate guard must precede candidates")
    company = _company_model()
    with reuse.install_fit_evidence_reuse(exchange):
        result = asyncio.run(lead_scorer.score_company_competition_intent(
            company, _icp_model(), 0, 0, {company.company_name.lower().strip()},
            integrity_policy=True, fit_evidence_reuse_enabled=True,
        ))
    assert result.final_score == 0 and requests == []


@pytest.mark.parametrize("negative", [False, True])
def test_native_company_score_reuses_fit_without_changing_score_or_receipt(monkeypatch, negative):
    current, authority, _ = harness(monkeypatch, negative=negative)
    async def verify(*_args, **_kwargs):
        current["calls"] += 1
        record_sources(current["text"])
        return fit_result(negative)
    async def signal(*_args, **_kwargs):
        return 80, 80, 1, 100, False, []
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    async def score():
        with reuse.install_fit_evidence_reuse(authority.exchange):
            return await lead_scorer.score_company_competition_intent(
                _company_model(), _icp_model(), 0, 0, set(), integrity_policy=True,
                fit_evidence_reuse_enabled=True,
            )
    first, second = asyncio.run(score()), asyncio.run(score())
    assert current["calls"] == 1
    assert first.final_score == second.final_score == (0 if negative else 80)
    assert first.verifier_gate_receipts == second.verifier_gate_receipts


def test_source_manifest_hash_uses_gateway_canonical_unicode_encoding():
    from lab_arena import contracts
    with reuse.capture_fit_evidence() as captured:
        record_sources(TEXT + " café ソフトウェア")
    snapshot = reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured)
    assert snapshot is not None
    assert snapshot["source_snapshot_hash"] == contracts.document_hash(snapshot["evidence_manifest"])
    assert reuse._hash({"text": "café ソフトウェア"}) == contracts.document_hash({"text": "café ソフトウェア"})


def test_real_revalidation_dispatch_compares_original_transport_projection(monkeypatch):
    from qualification.scoring import company_verification as cv, company_evidence_investigator as cei
    current = {"text": TEXT}
    async def homepage(_session, url):
        return 200, url, current["text"]
    @reuse.observed_source("investigator_page", ("url", "stealth_mode", "archive_targets", "archive_company_names"))
    async def page(_session, url, *, stealth_mode=False, archive_targets=(), archive_company_names=()):
        return {"ok": True, "url": url, "final_url": url, "text": current["text"]}
    monkeypatch.setattr(cv, "_fetch_bounded_html", homepage)
    monkeypatch.setattr(cei, "_fetch_page", page)
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    snapshot = reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured)
    assert asyncio.run(reuse.revalidate_fit_snapshot(snapshot))
    current["text"] += " Changed body."
    assert not asyncio.run(reuse.revalidate_fit_snapshot(snapshot))


def test_invocation_bound_completed_fit_does_not_change_during_retry(monkeypatch):
    calls = []
    async def verify(*_args, **_kwargs):
        calls.append("fit")
        record_sources()
        return fit_result()
    async def signal(*_args, **_kwargs):
        return 80, 80, 1, 100, False, []
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    authority, local_fit = Authority(), {}
    async def score():
        with reuse.install_fit_evidence_reuse(authority.exchange):
            return await lead_scorer.score_company_competition_intent(
                _company_model(), _icp_model(), 0, 0, set(), integrity_policy=True,
                fit_evidence_reuse_enabled=True, verified_company_fit_retry_cache=local_fit,
            )
    first = asyncio.run(score())
    operations = list(authority.operations)
    second = asyncio.run(score())
    assert calls == ["fit"] and first.verifier_gate_receipts == second.verifier_gate_receipts
    assert authority.operations == operations


def test_original_paragraph_selects_auxiliary_context_after_fit_reuse():
    company = _company_model()
    company.intent_details = "Acme builds security tooling for software buyers."
    auxiliary = "https://acme.example/products"
    body = "Acme builds security tooling for software buyers."
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        reuse.record_source("investigator_page", {"url": auxiliary, "stealth_mode": False,
            "archive_targets": [], "archive_company_names": []}, {"result": {
                "ok": True, "url": auxiliary, "final_url": auxiliary, "text": body,
            }, "contexts": {}})
    matched = {}
    reuse._restore_paragraph_source_context(company, fit_result(), captured, matched)
    assert matched[auxiliary] == {"final_url": auxiliary, "text": body}
