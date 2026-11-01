"""Returned investigator evidence keeps its separate, existing page bound."""

import asyncio
from copy import deepcopy

import pytest

from gateway.qualification.models import CompanyOutput
from lab_arena import scoring as arena_scoring
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_UNAVAILABLE
from tests.test_company_evidence_investigator import _company, _finding, _competition_company
from tests.test_linkedin_headcount_conflict_resolution import (
    IDENTITY, PROFILE, STRUCTURED, _investigation,
)
from tests.test_matched_company_retry_source_continuity import (
    IDENTITY as PLATFORM_IDENTITY,
    PLATFORM_QUOTE, PLATFORM_TEXT, PLATFORM_URL, _fit_result,
    _company as _platform_company, _icp as _platform_icp,
    _verdict as _platform_verdict, _finding as _industry_finding,
)


def _pages(count, *, url=PLATFORM_URL, text=PLATFORM_TEXT):
    pages = {url: {"final_url": url, "text": text}}
    for index in range(count - 1):
        unrelated = f"https://unibuddy.com/uncited-{index}/"
        pages[unrelated] = {"final_url": unrelated, "text": "Uncited source."}
    return pages


@pytest.mark.parametrize("count", [3, 4, 7])
def test_matched_source_survives_other_returned_pages(count):
    sink = {}
    result = _fit_result(decision=COMPANY_FIT_UNAVAILABLE)
    lead_scorer._retain_matched_investigator_source_contexts(
        sink, result,
        {investigator.PRIVATE_FETCHED_PAGES_KEY: _pages(count)},
    )
    retry = {}
    lead_scorer._retain_matched_company_retry_sources(retry, sink, result)
    assert sink == {PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT}}
    assert retry["pages"] == sink
    # Extra uncited pages do not become retry context.
    assert investigator.MAX_PREFETCHED_PAGES == 3
    assert investigator.MAX_RETURNED_FETCHED_PAGES == 7


def test_returned_count_is_checked_before_cited_source_selection():
    pages = _pages(8)
    assert investigator._validated_returned_fetched_pages(
        pages, submitted_source_urls=(PLATFORM_URL,),
    ) == ({}, {})
    sink = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        sink, _fit_result(), {investigator.PRIVATE_FETCHED_PAGES_KEY: pages},
    )
    assert sink == {}


@pytest.mark.parametrize("count", [4, 7])
def test_returned_bound_does_not_change_incoming_prefetch_bound(count):
    pages = _pages(count)
    assert investigator._validated_returned_fetched_pages(
        pages, submitted_source_urls=tuple(pages),
    )[0]
    assert investigator._validated_prefetched_pages(
        pages, submitted_source_urls=tuple(pages),
    ) == ({}, {})
    assert lead_scorer._investigator_prefetched_pages(
        {}, tuple(pages), verified_homepage_pages=pages,
    ) == {}


@pytest.mark.parametrize("override", [
    {"text": ""},
    {"text": "x" * (investigator.MAX_PAGE_CHARACTERS + 1)},
    {"text": "%PDF-1.7 decoded binary"},
    {"text": "Provider account capacity details redacted."},
    {"final_url": "http://127.0.0.1/private"},
    {"final_url": "https://user:password@unibuddy.com/private"},
    {"final_url": "https://unibuddy.com:444/private"},
    {"final_url": "HTTPS://UNIBUDDY.COM/platform/"},
])
def test_returned_pages_keep_existing_source_security_checks(override):
    pages = _pages(4)
    pages[PLATFORM_URL].update(override)
    assert investigator._validated_returned_fetched_pages(
        pages, submitted_source_urls=(PLATFORM_URL,),
    ) == ({}, {})


@pytest.mark.parametrize("unsafe", ["unproven", "quote_absent", "cross_domain"])
def test_returned_pages_cannot_make_unmatched_or_unbound_source_reusable(unsafe):
    result = _fit_result(decision=COMPANY_FIT_UNAVAILABLE)
    pages = _pages(4)
    if unsafe == "unproven":
        result.details["dimension_evidence"]["industry"]["decision"] = (
            COMPANY_FIT_UNAVAILABLE
        )
    elif unsafe == "quote_absent":
        pages[PLATFORM_URL]["text"] = "Unibuddy unrelated page."
    else:
        pages[PLATFORM_URL]["final_url"] = "https://other.example/platform/"
    sink = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        sink, result, {investigator.PRIVATE_FETCHED_PAGES_KEY: pages},
    )
    retry = {}
    lead_scorer._retain_matched_company_retry_sources(retry, sink, result)
    assert sink == retry == {}


@pytest.mark.parametrize("unsafe", [
    "identity_mismatch", "receipt_mismatch", "missing_name", "missing_domain",
    "missing_linkedin", "untrusted_identity", "unproven_dimension", "quote_absent",
])
def test_targeted_projection_requires_independent_identity_and_matched_quote(unsafe):
    result = lead_scorer._reverify_decision(
        _platform_verdict(), "", "", icp=_platform_icp(),
        company=_platform_company(), verified_homepage_identity=PLATFORM_IDENTITY,
        verified_homepage_transport_domain="unibuddy.com", company_quality=True,
    )
    details = result.details
    if unsafe == "identity_mismatch":
        details["identity_decision"] = lead_scorer.COMPANY_FIT_MISMATCH
    elif unsafe == "receipt_mismatch":
        details["identity_receipt"]["decision"] = lead_scorer.COMPANY_FIT_MISMATCH
    elif unsafe.startswith("missing_"):
        field = {
            "missing_name": "observed_name", "missing_domain": "observed_domain",
            "missing_linkedin": "observed_linkedin_slug",
        }[unsafe]
        details["identity_receipt"].pop(field)
    elif unsafe == "untrusted_identity":
        details["identity_receipt"]["evidence_source"] = "model_submitted"
    elif unsafe == "unproven_dimension":
        details["dimension_decisions"]["industry"] = COMPANY_FIT_UNAVAILABLE
    else:
        details["dimension_evidence"]["industry"]["quote"] = "An absent exact quote."
    original = deepcopy(result.details)
    projection = lead_scorer._investigator_retention_projection(result)
    sink = {}
    if projection is not None:
        lead_scorer._retain_matched_investigator_source_contexts(
            sink, projection,
            {investigator.PRIVATE_FETCHED_PAGES_KEY: _pages(4)},
        )
    assert sink == {}
    assert result.details == original


@pytest.mark.parametrize("count", [4, 7, 8])
def test_reopened_stage_requires_trigger_proof_with_other_returned_pages(count):
    trigger = "https://news.example/doctronic-raises-series-b"
    base = _company(name="Doctronic", website="https://doctronic.ai")
    company = CompanyOutput.model_validate({
        **base.model_dump(),
        "intent_signals": [{
            "description": "Doctronic announced an update.",
            "source": "news", "url": trigger, "date": "2026-03-24",
            "snippet": "Doctronic announced an update.",
        }],
    })
    finding = _finding("stage", observed_value="Series A")
    pages = _pages(count, url=trigger, text="Doctronic completed Series A.")
    output = {investigator.PRIVATE_FETCHED_PAGES_KEY: pages}
    assert lead_scorer._reopened_stage_dispute_resolved(
        company, "Series A", finding, output,
    ) is (count <= investigator.MAX_RETURNED_FETCHED_PAGES)
    del pages[trigger]
    assert not lead_scorer._reopened_stage_dispute_resolved(
        company, "Series A", finding, output,
    )


@pytest.mark.parametrize("count", [4, 7, 8])
def test_headcount_requires_exact_profile_proof_with_other_returned_pages(count):
    output = deepcopy(_investigation())
    finding = output["claims"]["headcount"]
    pages = _pages(count, url=PROFILE, text=finding["evidence_quote"])
    output[investigator.PRIVATE_FETCHED_PAGES_KEY] = pages
    context = {"structured_profile_evidence": STRUCTURED}
    assert lead_scorer._fetched_linkedin_headcount_resolves_conflict(
        finding, output, context, IDENTITY,
    ) is (count <= investigator.MAX_RETURNED_FETCHED_PAGES)
    pages[PROFILE]["final_url"] = "https://www.linkedin.com/company/other"
    assert not lead_scorer._fetched_linkedin_headcount_resolves_conflict(
        finding, output, context, IDENTITY,
    )


@pytest.mark.parametrize("count", [4, 7])
def test_outer_retry_reuses_matched_source_and_rejudges_same_quality(monkeypatch, count):
    calls = []
    decisions = []
    company, icp = _platform_company(), _platform_icp()

    async def investigate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            assert kwargs["prefetched_pages"] == {
                PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
            }
        return {
            "claims": {"industry": _industry_finding(
                status="VERIFIED", role="supplier_operator",
                reason="The same fetched quote proves supplier activity.",
            )},
            "_completed_submit": True, "failure_reason": "",
            investigator.PRIVATE_FETCHED_PAGES_KEY: _pages(count),
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)

    def scoped_runner(batch, _icp, _reference, retry_evidence_scope,
                      *, original_company_indexes):
        assert original_company_indexes == [0]
        retry = retry_evidence_scope.setdefault("same-frozen-company", {})
        sink = {}
        _, result, *_ = asyncio.run(
            lead_scorer._run_targeted_company_evidence_investigation(
                company=company, icp=icp, verdict=_platform_verdict(),
                investigation_targets=("industry",),
                icp_attribute="", icp_stage="",
                verified_identity=PLATFORM_IDENTITY,
                verified_transport_domain="unibuddy.com",
                structured_employee_size_evidence=None,
                structured_public_company_evidence=None,
                employee_size_conflict=False, company_quality=True,
                matched_company_source_sink=sink,
                matched_company_retry_source_cache=retry,
            )
        )
        decisions.append(result.details["dimension_decisions"]["industry"])
        assert sink == {
            PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
        }
        # Build the final receipt with the same observed decisions and source
        # evidence; no UNPROVEN dimension becomes matched in this handoff.
        retention_projection = lead_scorer._investigator_retention_projection(result)
        assert retention_projection is not None
        final_fit = lead_scorer._complete_company_fit_result(
            result.decision, result.reason,
            dimensions=result.details["dimension_decisions"],
            dimension_evidence=retention_projection.details["dimension_evidence"],
            stage_required=False,
        )
        lead_scorer._retain_matched_company_retry_sources(retry, sink, final_fit)
        # A later gate is unavailable on the first attempt. The matched
        # source remains untrusted input to the repeated independent review.
        return [{
            "final_score": 0.0 if len(calls) == 1 else 54.0,
            "failure_reason": (
                "Intent verification unavailable: verifier provider error"
                if len(calls) == 1 else ""
            ),
        }]

    def scorer(*_args):
        raise AssertionError("The private scoped runner must be used")

    setattr(scorer, arena_scoring._SCOPED_RETRY_EVIDENCE_RUNNER, scoped_runner)
    rows = arena_scoring.score_work_item(
        {"scored_run_id": "returned-page-retry"},
        icp=icp.model_dump(mode="json"),
        companies=[company.model_dump(mode="json")], scorer=scorer,
        max_retries=2,
    )
    assert decisions == [lead_scorer.COMPANY_FIT_MATCH] * 2
    assert len(calls) == 2
    assert rows == [{"final_score": 54.0, "failure_reason": ""}]


@pytest.mark.parametrize("count", [4, 7])
def test_native_integrity_adapter_reuses_returned_source_after_intent_timeout(monkeypatch, count):
    investigations = []
    intent_calls = []
    icp = _platform_icp().model_copy(update={"required_attribute": ""})

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("Fixture prechecks passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match("Fixture homepage independently verified", details={
            "identity": {
                "decision": "match", "evidence_source": "company_homepage",
                "observed_name": "Unibuddy", "observed_domain": "unibuddy.com",
                "observed_linkedin_slug": "unibuddy",
            },
            "verified_homepage_transport_domain": "unibuddy.com",
        })

    async def investigate(**kwargs):
        investigations.append(kwargs)
        return {
            "claims": {"industry": _industry_finding(
                status="VERIFIED", role="supplier_operator",
                reason="The same fetched quote proves supplier activity.",
            )},
            "_completed_submit": True, "failure_reason": "",
            investigator.PRIVATE_FETCHED_PAGES_KEY: _pages(count),
        }

    async def web_boundary(company, actual_icp, **kwargs):
        verdict = _platform_verdict()
        verdict["geography_evidence_quote"] = (
            "Unibuddy headquarters are in the United Kingdom."
        )
        _, result, *_ = await lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=actual_icp, verdict=verdict,
            investigation_targets=("industry",), icp_attribute="", icp_stage="",
            verified_identity=PLATFORM_IDENTITY, verified_transport_domain="unibuddy.com",
            structured_employee_size_evidence=None, structured_public_company_evidence=None,
            employee_size_conflict=False, company_quality=kwargs["company_quality"],
            matched_company_source_sink=kwargs["matched_company_source_sink"],
            matched_company_retry_source_cache=kwargs["matched_company_retry_source_cache"],
        )
        assert result.details["dimension_decisions"]["industry"] == "match"
        return result

    async def intent_boundary(*_args, **_kwargs):
        intent_calls.append(True)
        if len(intent_calls) == 1:
            raise TimeoutError("Fixture intent provider timeout")
        return 54, 54, 1.0, 1.0, False, [{
            "raw": 54, "after_decay": 54, "matched_icp_signal": 0,
            "judge_verdict": {
                "decision": "verified", "pipeline_decision": "accept",
                "verification_trace": {"intent_verdict": {
                    "signal_evaluations": [{"signal_status": "supported"}],
                }},
            },
        }]

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web_boundary)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", intent_boundary)
    scorer = arena_scoring.lab_scorer(arena_scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
    ))
    company = _competition_company()
    submitted = _platform_company().model_dump(mode="json")
    for field in (
        "company_name", "company_website", "company_linkedin", "industry",
        "employee_count", "country", "state",
    ):
        company[field] = submitted[field]
    company["fit_evidence_urls"] = [PLATFORM_URL]
    rows = arena_scoring.score_work_item(
        {"scored_run_id": "native-returned-source-retry"},
        icp=icp.model_dump(mode="json"),
        companies=[company],
        scorer=scorer, max_retries=2,
    )
    # Fit and its exact source evidence completed before the intent timeout.
    # Only the unfinished intent stage is retried.
    assert len(investigations) == 1
    assert len(intent_calls) == 2
    assert rows[0]["company_qualified"] is True
    assert rows[0]["final_score"] == 54
