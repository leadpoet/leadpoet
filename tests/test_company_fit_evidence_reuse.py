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
    fit.details["dimension_evidence"]["web_observations"] = {
        "observed_employee_count": "51-200", "observed_hq_country": "United States",
        "observed_hq_state": "", "observed_company_stage": "Series A",
        "observed_industry": "Software", "observed_subindustry": "",
        "industry_activity_role": "supplier_operator",
    }
    claims = {target: {"target": target, "status": "VERIFIED", "evidence_url": PAGE,
                       "evidence_quote": TEXT}
              for target in ("industry", "headcount", "geography", "stage")}
    claims["headcount"]["observed_value"] = "51-200"
    claims["geography"].update(observed_country="United States", observed_state="")
    claims["stage"]["observed_value"] = "Series A"
    claims["industry"].update(observed_industry="Software", observed_subindustry="",
                              activity_role="supplier_operator", status="CONTRADICTED" if negative else "VERIFIED",
                              _industry_review={"schema": "industry_evidence_review_v1", "model": "native-model",
                                  "packet_sha256": "a" * 64, "decision": "mismatch" if negative else "match",
                                  "evidence": [{"url": PAGE, "quote": TEXT}]})
    fit.details["supporting_receipts"] = [{"gate": "company_evidence_investigation",
        "completed_submitted_findings": True, "targets": list(claims), "claims": claims,
        "failure_reason": ""}]
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
                                    for key, value in reversed(list(self.rows.items()))][:8]}
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


def test_eight_candidates_use_the_same_bound_as_gateway(monkeypatch):
    current, authority, score = harness(monkeypatch)
    asyncio.run(score())
    reads = []
    read = reuse._read_source
    async def counted(entry):
        reads.append((entry["kind"], entry["request"]))
        return await read(entry)
    monkeypatch.setattr(reuse, "_read_source", counted)
    first = deepcopy(next(iter(authority.rows.values())))
    # Older variants can precede the currently matching source. All eight
    # candidates are valid protocol input; a changed page is only a miss.
    for index in range(7):
        other = deepcopy(first)
        other["evidence_manifest"]["sources"][0]["response_hash"] = "sha256:" + str(index) * 64
        other["source_snapshot_hash"] = reuse._hash(other["evidence_manifest"])
        authority.rows[other["source_snapshot_hash"]] = other
    asyncio.run(score())
    assert current["calls"] == 1
    assert len(reads) == 2  # Eight variants must not pay for each source eight times.
    asyncio.run(score())
    assert len(reads) == 4  # A separate submission must still fetch fresh evidence.


@pytest.mark.parametrize("search_kind", ["web", "industry"])
def test_unretained_search_prevents_reuse_of_unchanged_cited_pages(monkeypatch, search_kind):
    from qualification.scoring import company_evidence_investigator as investigator
    calls = []
    async def post(*args, **kwargs):
        calls.append(kwargs["payload"])
        return 200, {"results": [], "choices": [{"message": {"content": "New funding round", "annotations": []}}]}
    monkeypatch.setattr(investigator, "_post_json", post)
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        if search_kind == "web":
            asyncio.run(investigator._search_web(None, "latest round", key="test-only"))
        else:
            asyncio.run(investigator._search_industry_locators(
                None, "latest company facts", key="test-only", company_locator={"name": "Acme"},
                include_domains=["acme.example"]))
    assert calls and not captured.complete
    assert reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured) is None


def test_unrecorded_optional_context_cannot_enter_or_reenter_fit_cache(monkeypatch):
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    stale = {"https://acme.example/unrecorded": {
        "final_url": "https://acme.example/unrecorded", "text": "Old unsupported evidence"}}
    assert reuse.build_fit_snapshot(fit_result(), {}, stale, {}, captured) is None
    snapshot = reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured)
    snapshot["matched_company_sources"] = stale
    async def read(entry):
        record_sources()
    monkeypatch.setattr(reuse, "_read_source", read)
    assert not asyncio.run(reuse.revalidate_fit_snapshot(snapshot))


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


def test_quote_occurrence_without_independent_findings_does_not_close_discovery():
    fit = fit_result()
    fit.details.pop("supporting_receipts")
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None


def test_homepage_quote_uses_native_plain_projection_but_fingerprint_keeps_html(monkeypatch):
    from qualification.scoring.company_evidence_investigator import _plain_text
    html = "<html><title>Acme</title><body><p>Acme provides</p><p>software.</p></body></html>"
    plain = _plain_text(html)
    assert plain not in html
    fit = fit_result()
    for item in fit.details["dimension_evidence"].values():
        if "web_evidence" in item:
            item["web_evidence"].update(url=HOME, quote=plain)
    for finding in fit.details["supporting_receipts"][0]["claims"].values():
        finding.update(evidence_url=HOME, evidence_quote=plain)
    fit.details["supporting_receipts"][0]["claims"]["industry"]["_industry_review"]["evidence"] = [
        {"url": HOME, "quote": plain}]
    def capture_home(body):
        reuse.record_source("homepage", {"url": HOME}, {"status": 200, "final_url": HOME, "text": body})
    with reuse.capture_fit_evidence() as captured:
        capture_home(html)
    snapshot = reuse.build_fit_snapshot(fit, {}, {}, {}, captured)
    assert snapshot is not None
    current = {"html": html}
    async def read(_entry): capture_home(current["html"])
    monkeypatch.setattr(reuse, "_read_source", read)
    assert asyncio.run(reuse.revalidate_fit_snapshot(snapshot))
    current["html"] = html.replace("<body>", '<body class="changed">')
    assert _plain_text(current["html"]) == plain
    assert not asyncio.run(reuse.revalidate_fit_snapshot(snapshot))


def test_exact_native_context_with_link_metadata_is_bound_without_making_metadata_evidence():
    from qualification.scoring.company_evidence_investigator import _plain_text, _quote_occurs
    html = ('<html><body><p>' + TEXT + '</p><a href="https://acme.example/products">Products</a></body></html>')
    body = _plain_text(html)
    assert "[[SERVER_VISIBLE_LINK_DESTINATIONS_FOR_IDENTITY_ONLY]]" in body
    assert not _quote_occurs(body, body)  # Factual quotes must still exclude this metadata.
    with reuse.capture_fit_evidence() as captured:
        reuse.record_source("homepage", {"url": HOME}, {"status": 200, "final_url": HOME, "text": html})
        reuse.record_source("investigator_page", {"url": PAGE, "stealth_mode": False,
            "archive_targets": [], "archive_company_names": []},
            {"result": {"ok": True, "url": PAGE, "final_url": PAGE, "text": body}, "contexts": {}})
    contexts = {url: {"final_url": url, "text": body} for url in (HOME, PAGE)}
    assert reuse.build_fit_snapshot(fit_result(), {}, contexts, {}, captured) is not None
    contexts[PAGE]["text"] = body.replace("https://acme.example/products", "https://acme.example/forged")
    assert reuse.build_fit_snapshot(fit_result(), {}, contexts, {}, captured) is None
    assert not _quote_occurs(body, body)


@pytest.mark.parametrize("mutation", [
    lambda fit: fit.details["supporting_receipts"][0].update(completed_submitted_findings=False),
    lambda fit: fit.details["supporting_receipts"][0].update(failure_reason="provider_error"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["industry"].pop("_industry_review"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["industry"]["_industry_review"].update(decision="unavailable"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["industry"]["_industry_review"].update(evidence=[{"url": PAGE, "quote": "Not present"}]),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["industry"].update(activity_role="customer_user"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["stage"].update(observed_value="Series B"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["headcount"].update(observed_value="201-500"),
    lambda fit: fit.details["supporting_receipts"][0]["claims"]["geography"].update(observed_country="Canada"),
])
def test_incomplete_or_different_independent_fact_stays_native(mutation):
    fit = fit_result()
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    mutation(fit)
    before = deepcopy(fit.receipt("company_fit"))
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None
    assert fit.receipt("company_fit") == before


def test_native_initial_lookup_can_discover_sources_without_becoming_proof(monkeypatch):
    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def json(self):
            return {"choices": [{"message": {"content": '{"observed_industry":"Software"}'}}]}
    class Session(Response):
        def __init__(self, **_kwargs): pass
        def post(self, *_args, **_kwargs): return Response()
    monkeypatch.setattr(lead_scorer.aiohttp, "ClientSession", Session)
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        verdict, error = asyncio.run(lead_scorer._request_company_reverify_json(
            key="test-only", prompt="unchanged native lookup", telemetry_purpose="test"))
    assert verdict == {"observed_industry": "Software"} and error == "" and captured.complete
    assert reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured) is not None
    incomplete = fit_result()
    incomplete.details.pop("supporting_receipts")
    assert reuse.build_fit_snapshot(incomplete, {}, {}, {}, captured) is None


def test_fresh_revalidation_rejects_old_quote_only_snapshot(monkeypatch):
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    snapshot = reuse.build_fit_snapshot(fit_result(), {}, {}, {}, captured)
    snapshot["company_fit_receipt"].pop("supporting_receipts")
    async def read(_entry): record_sources()
    monkeypatch.setattr(reuse, "_read_source", read)
    assert not asyncio.run(reuse.revalidate_fit_snapshot(snapshot))


def test_captured_current_profile_size_closes_native_numeric_observation():
    fit = fit_result()
    profile = "https://www.linkedin.com/company/acme"
    quote = "Company size\n51-200 employees"
    fit.details["dimension_evidence"]["employee_size"]["web_evidence"] = {"url": profile, "quote": quote}
    fit.details["supporting_receipts"][0]["claims"].pop("headcount")
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        reuse.record_source("linkedin_contents", {"profile_url": profile, "expected_company_identity": {}, "capture_sinks": []},
                            {"result": {"url": profile, "quote": quote, "employee_count": "51-200"}, "contexts": {}})
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is not None
    fit.details["dimension_evidence"]["web_observations"]["observed_employee_count"] = "201-500"
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None


def test_captured_structured_size_is_exact_typed_evidence_not_model_quote():
    from qualification.scoring.linkedin_company_size import project_structured_linkedin_company_size
    fit = fit_result()
    profile = "https://www.linkedin.com/company/acme"
    native = project_structured_linkedin_company_size("acme.example", profile, {"status": 200, "element": {
        "website": HOME, "linkedinUrl": profile, "employeeCountRange": {"start": 51, "end": 200}}})
    assert native
    fit.details["dimension_evidence"]["employee_size"]["web_evidence"] = native
    fit.details["supporting_receipts"][0]["claims"].pop("headcount")
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        reuse.record_source("linkedin_structured", {"requested_domain": "acme.example", "profile_url": profile,
            "company_identity_observed_profile_url": "", "expected_company_name": "Acme", "preserve_identity_website": False,
            "capture_sinks": []}, {"result": deepcopy(native), "contexts": {}})
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is not None
    native["employee_count"] = "201-500"
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None


def test_captured_structured_headquarters_closes_native_country_observation():
    from qualification.scoring.linkedin_company_size import (
        project_structured_linkedin_company_headquarters, project_structured_linkedin_company_size,
    )
    fit = fit_result()
    profile = "https://www.linkedin.com/company/acme"
    payload = {
        "status": 200, "element": {"name": "Acme", "website": HOME, "linkedinUrl": profile,
            "employeeCountRange": {"start": 51, "end": 200},
            "locations": [{"headquarter": True, "country": "France", "city": "Paris"}]}}
    native = project_structured_linkedin_company_headquarters("acme.example", profile, "Acme", payload)
    assert native
    fit.details["dimension_evidence"]["geography"]["web_evidence"] = {"url": profile, "quote": native["text"]}
    fit.details["dimension_evidence"]["web_observations"].update(observed_hq_country="France", observed_hq_state="")
    fit.details["supporting_receipts"][0]["claims"].pop("geography")
    with reuse.capture_fit_evidence() as captured:
        record_sources()
        reuse.record_source("linkedin_structured", {"requested_domain": "acme.example", "profile_url": profile,
            "company_identity_observed_profile_url": "", "expected_company_name": "Acme", "preserve_identity_website": False,
            "capture_sinks": ["company_headquarters_evidence"]},
            {"result": project_structured_linkedin_company_size("acme.example", profile, payload),
             "contexts": {"company_headquarters_evidence": native}})
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is not None
    fit.details["dimension_evidence"]["web_observations"]["observed_hq_country"] = "Germany"
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None


def test_required_attribute_needs_its_own_completed_grounded_finding():
    fit = fit_result()
    fit.details["dimension_evidence"]["required_attribute"] = {
        "decision": "match", "web_evidence": {"url": PAGE, "quote": TEXT}}
    with reuse.capture_fit_evidence() as captured:
        record_sources()
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is None
    review = fit.details["supporting_receipts"][0]
    review["targets"].append("required_attribute")
    review["claims"]["required_attribute"] = {"target": "required_attribute", "status": "VERIFIED",
                                              "evidence_url": PAGE, "evidence_quote": TEXT}
    assert reuse.build_fit_snapshot(fit, {}, {}, {}, captured) is not None


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
