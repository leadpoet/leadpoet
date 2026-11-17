"""Native fit projections may omit support; cache proof must retain all of it."""
import asyncio
from copy import deepcopy

import pytest

from lab_arena import contracts
from qualification.scoring import company_fit_evidence_reuse as reuse, lead_scorer
from tests.test_company_fit_evidence_reuse import PAGE, TEXT, fit_result, record_sources

SUPPORT = PAGE + "/support"
SUPPORT_TEXT = "Acme provides verified supporting evidence."
TARGETS = {"required_attribute": "required_attribute", "geography": "geography",
           "employee_size": "headcount", "stage": "stage", "industry": "industry"}


def supported_fit():
    fit = fit_result()
    review = fit.details["supporting_receipts"][0]
    review["claims"]["required_attribute"] = {
        "target": "required_attribute", "status": "VERIFIED",
        "evidence_url": PAGE, "evidence_quote": TEXT,
    }
    review["targets"].append("required_attribute")
    fit.details["dimension_evidence"]["required_attribute"] = {"decision": "match"}
    for dimension, target in TARGETS.items():
        finding = review["claims"][target]
        finding["supporting_evidence"] = (
            [{"url": SUPPORT, "quote": SUPPORT_TEXT}]
            if dimension in {"required_attribute", "stage", "industry"} else [])
        project(fit, dimension)
    return fit


def project(fit, dimension):
    finding = fit.details["supporting_receipts"][0]["claims"][TARGETS[dimension]]
    fit.details["dimension_evidence"][dimension]["web_evidence"] = lead_scorer._dimension_web_evidence(
        {"dimension_evidence": {dimension: finding}}, dimension)


def recorded_support(text=SUPPORT_TEXT):
    reuse.record_source("investigator_page", {
        "url": SUPPORT, "stealth_mode": False, "archive_targets": [], "archive_company_names": [],
    }, {"result": {"ok": True, "url": SUPPORT, "final_url": SUPPORT, "text": text}, "contexts": {}})


def captured_sources(text=SUPPORT_TEXT):
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        recorded_support(text)
    return captured


@pytest.mark.parametrize("dimension", TARGETS)
def test_completed_finding_uses_native_projection_and_grounds_all_support(dimension):
    fit = supported_fit()
    receipt = fit.receipt("company_fit")
    web = receipt["dimension_evidence"][dimension]["web_evidence"]
    assert bool(web.get("supporting_evidence")) == (dimension in {"stage", "industry"})
    assert reuse._grounded_finding(receipt, dimension, web, {PAGE: TEXT, SUPPORT: SUPPORT_TEXT})


def test_complete_snapshot_preserves_full_finding_and_source_hashes():
    fit, captured = supported_fit(), captured_sources()
    snapshot = reuse.build_fit_snapshot(fit, {}, {}, {}, captured)
    assert snapshot is not None
    assert reuse.validate_fit_snapshot(snapshot) == snapshot
    assert snapshot["company_fit_receipt"] == fit.receipt("company_fit")
    assert snapshot["source_snapshot_hash"] == contracts.document_hash(captured.manifest())
    changed = deepcopy(snapshot)
    changed["company_fit_receipt"]["supporting_receipts"][0]["claims"]["required_attribute"]["supporting_evidence"] = []
    assert contracts.document_hash(changed) != contracts.document_hash(snapshot)


@pytest.mark.parametrize("dimension,limit", [("required_attribute", 2), ("stage", 2), ("industry", 8)])
def test_native_support_limit_remains_admissible(dimension, limit):
    fit = supported_fit()
    fit.details["supporting_receipts"][0]["claims"][dimension]["supporting_evidence"] *= limit
    project(fit, dimension)
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured_sources()) is not None


@pytest.mark.parametrize("dimension", ["required_attribute", "geography", "employee_size"])
@pytest.mark.parametrize("invalid", ["too_many", "extra_field", "long_quote", "invalid_url"])
def test_unprojected_support_must_have_native_shape_and_bounds(dimension, invalid):
    fit = supported_fit()
    finding = fit.details["supporting_receipts"][0]["claims"][TARGETS[dimension]]
    support = [{"url": SUPPORT, "quote": SUPPORT_TEXT}]
    if invalid == "too_many":
        support *= 3 if dimension == "required_attribute" else 1
    elif invalid == "extra_field":
        support[0]["unexpected"] = "unvalidated metadata"
    elif invalid == "long_quote":
        support[0]["quote"] = "Acme" + " " * 2001 + "provides verified supporting evidence."
    else:
        support[0]["url"] = "http://acme.example/about/support"
    finding["supporting_evidence"] = support
    project(fit, dimension)
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        reuse.record_source("investigator_page", {
            "url": support[0]["url"], "stealth_mode": False,
            "archive_targets": [], "archive_company_names": [],
        }, {"result": {"ok": True, "url": support[0]["url"],
                       "final_url": support[0]["url"], "text": SUPPORT_TEXT}, "contexts": {}})
    receipt = fit.receipt("company_fit")
    assert not reuse._grounded_finding(receipt, dimension,
        receipt["dimension_evidence"][dimension]["web_evidence"],
        {PAGE: TEXT, support[0]["url"]: SUPPORT_TEXT})
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None


@pytest.mark.parametrize("source", [None, {}, {"url": SUPPORT}, {"quote": SUPPORT_TEXT},
    {"url": 7, "quote": SUPPORT_TEXT}, {"url": SUPPORT, "quote": None},
    {"url": SUPPORT, "quote": 7}])
def test_required_attribute_support_entries_keep_native_types_and_fields(source):
    fit = supported_fit()
    fit.details["supporting_receipts"][0]["claims"]["required_attribute"]["supporting_evidence"] = [source]
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured_sources()) is None


def test_required_attribute_support_accepts_native_raw_quote_limit():
    fit = supported_fit()
    support = fit.details["supporting_receipts"][0]["claims"]["required_attribute"]["supporting_evidence"]
    support[0]["quote"] = SUPPORT_TEXT.ljust(2000)
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured_sources()) is not None


@pytest.mark.parametrize("dimension", TARGETS)
@pytest.mark.parametrize("support", [{}, "", 0, False, None, {"url": SUPPORT}])
def test_malformed_support_is_not_hidden_by_an_empty_projection(dimension, support):
    fit = supported_fit()
    fit.details["supporting_receipts"][0]["claims"][TARGETS[dimension]]["supporting_evidence"] = support
    project(fit, dimension)
    receipt = fit.receipt("company_fit")
    web = receipt["dimension_evidence"][dimension]["web_evidence"]
    assert not reuse._grounded_finding(receipt, dimension, web, {PAGE: TEXT, SUPPORT: SUPPORT_TEXT})
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured_sources()) is None


@pytest.mark.parametrize("dimension,limit", [("stage", 2), ("industry", 8)])
@pytest.mark.parametrize("invalid", ["too_many", "invalid_url", "empty_quote", "long_quote", "extra_field"])
def test_retained_projection_never_admits_dropped_or_changed_support(dimension, limit, invalid):
    fit = supported_fit()
    support = fit.details["supporting_receipts"][0]["claims"][dimension]["supporting_evidence"]
    if invalid == "too_many":
        support *= limit + 1
    elif invalid == "invalid_url":
        support[0]["url"] = "not-a-url"
    elif invalid == "empty_quote":
        support[0]["quote"] = ""
    elif invalid == "long_quote":
        support[0]["quote"] = "x" * 2001
    else:
        support[0]["unexpected"] = True
    project(fit, dimension)
    receipt = fit.receipt("company_fit")
    web = receipt["dimension_evidence"][dimension]["web_evidence"]
    # Even a matching body cannot legitimize a native-dropped support record.
    pages = {PAGE: TEXT, support[0]["url"]: support[0]["quote"]}
    assert not reuse._grounded_finding(receipt, dimension, web, pages)


@pytest.mark.parametrize("mutation", [
    "missing_finding", "unverified", "incomplete", "primary_url", "primary_quote",
    "support_url", "support_quote", "stage_projection", "industry_projection",
    "industry_proof", "identity", "geography", "employee_size", "stage",
])
def test_incomplete_or_conflicting_native_proof_still_refuses_snapshot(mutation):
    fit = supported_fit()
    review = fit.details["supporting_receipts"][0]
    finding = review["claims"]["required_attribute"]
    evidence = fit.details["dimension_evidence"]
    if mutation == "missing_finding":
        review["claims"].pop("required_attribute")
    elif mutation == "unverified":
        finding["status"] = "UNAVAILABLE"
    elif mutation == "incomplete":
        review["completed_submitted_findings"] = False
    elif mutation.startswith("primary_"):
        finding["evidence_" + mutation.removeprefix("primary_")] = "Unproved primary evidence"
    elif mutation.startswith("support_"):
        finding["supporting_evidence"][0][mutation.removeprefix("support_")] = "Unproved supporting evidence"
    elif mutation.endswith("_projection"):
        evidence[mutation.removesuffix("_projection")]["web_evidence"].pop("supporting_evidence")
    elif mutation == "industry_proof":
        review["claims"]["industry"].pop("_industry_review")
    elif mutation == "identity":
        evidence["identity"]["web_identity_receipt"]["observed_name"] = "Another company"
    else:
        field = {"geography": "observed_country", "employee_size": "observed_value", "stage": "observed_value"}[mutation]
        review["claims"][TARGETS[mutation]][field] = "Different native fact"
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured_sources()) is None


@pytest.mark.parametrize("change", ["removed", "changed"])
def test_all_original_support_pages_remain_required(change):
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        if change == "changed":
            recorded_support("The supporting fact was removed.")
    assert captured.complete
    assert reuse.build_fit_snapshot(supported_fit(), {}, {}, {}, captured) is None


def test_held_fit_receipt_reuses_through_native_aggregation_and_authority_protocol(monkeypatch):
    """Real aggregation/bridge/admission, held verifier replies and memory storage."""
    from lab_arena import signal_cache_control
    from qualification.scoring.evidence_cache_bridge import install_company_fit_exchange
    from tests.lab_arena.evidence_cache_scorer_e2e_test import FitAndSignalTransport
    from tests.lab_arena.evidence_judgments_v2_test import document
    from tests.test_company_fit_evidence_reuse import HOME
    from tests.test_arena_intent_retry_cache import _company_model, _icp_model

    host = FitAndSignalTransport()
    host.current_input = document()
    current = {"text": SUPPORT_TEXT, "native": 0, "reads": 0}

    async def read(entry):
        current["reads"] += 1
        url = entry["request"]["url"]
        if url == SUPPORT:
            recorded_support(current["text"])
        elif entry["kind"] == "homepage":
            reuse.record_source("homepage", entry["request"], {"status": 200, "final_url": HOME, "text": TEXT})
        else:
            reuse.record_source("investigator_page", entry["request"], {"result": {
                "ok": True, "url": PAGE, "final_url": PAGE, "text": TEXT}, "contexts": {}})

    async def verify(*_args, **_kwargs):
        current["native"] += 1
        record_sources()
        recorded_support(current["text"])
        return supported_fit()

    async def signal(*_args, **_kwargs):
        return 80, 80, 1, 100, False, []

    async def exchange(request):
        return await host(signal_cache_control.validate_request({**request, "company_index": 0}))

    async def score(run_id):
        host.run = run_id
        host.current_input["scored_run_id"] = run_id
        company = _company_model()
        company.intent_signals[0].description = "Current claim for " + run_id
        host.current_input["companies"][0]["intent_signals"][0]["description"] = company.intent_signals[0].description
        with install_company_fit_exchange(exchange):
            return await lead_scorer.score_company_competition_intent(
                company, _icp_model(), 0, 0, set(), integrity_policy=True,
                fit_evidence_reuse_enabled=True)

    monkeypatch.setattr(reuse, "_read_source", read)
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    first = asyncio.run(score("baseline"))
    assert len(host.fit_rows) == 1
    original = deepcopy(host.fit_rows)
    second = asyncio.run(score("miner-rewrite"))
    assert current["native"] == 1 and current["reads"] == 3
    assert first.final_score == second.final_score == 80
    assert first.verifier_gate_receipts == second.verifier_gate_receipts
    assert host.fit_rows == original
    assert [(run, source) for run, source, _reference in host.uses] == [
        ("baseline", "baseline"), ("miner-rewrite", "baseline")]
    stored = next(iter(host.fit_rows.values()))["judgment"]
    assert stored["company_fit_receipt"]["supporting_receipts"] == supported_fit().details["supporting_receipts"]
    current["text"] += " A new source revision."
    third = asyncio.run(score("miner-new-source"))
    assert current["native"] == 2 and len(host.fit_rows) == 2
    assert third.final_score == 80
    assert all(host.fit_rows[key] == row for key, row in original.items())
    current["text"] = "The supporting fact was removed."
    asyncio.run(score("miner-missing-support"))
    assert current["native"] == 3 and len(host.fit_rows) == 2
