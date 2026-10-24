"""Server-fetched exact submitted pages remain available to the ordinary intent judge."""
import asyncio
import copy
from unittest.mock import AsyncMock

import pytest
from gateway.qualification.models import CompanyOutput, ICPPrompt, IntentSignal
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.company_evidence_investigator import PRIVATE_FETCHED_PAGES_KEY
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.evaluation_clock import use_evaluation_date

URL = "https://www.amdocs.com/press-release/veon-and-amdocs-enter-10-year-strategic-agreement-advance-digital-services"
# Exact contiguous paragraph from the captured 2026-10-24 fit request's server-prefetched page.
TEXT = 'Dubai, Almaty and Jersey City, [October 7, 2026] – VEON Ltd. ("VEON") (NASDAQ: VEON), a global digital operator, and Amdocs (NASDAQ: DOX), a leading provider of software and services to communications and media companies, today announced a 10-year strategic agreement. Under the agreement, Amdocs will run and continuously modernize VEON\'s core technology operations, accelerate its strategy as a digital operator, and provide long-term cost predictability across its IT operations in Eurasia.'
IDENTITY = {"decision": "match", "evidence_source": "company_web_reverification",
            "observed_name": "amdocs", "observed_domain": "amdocs.com", "observed_linkedin_slug": "amdocs"}
CLAIM = "VEON and Amdocs announced a 10-year strategic agreement to modernize technology operations."
PAGES = {URL: {"final_url": URL, "text": TEXT}}


def _inputs():
    signal = IntentSignal(source="news", description=CLAIM, url=URL,
                          date="2026-10-07", snippet=CLAIM, matched_icp_signal=0)
    company = CompanyOutput.model_construct(company_name="Amdocs", company_website="https://amdocs.com/",
        company_linkedin="", intent_signals=[signal], intent_details=CLAIM, required_attribute=None)
    icp = ICPPrompt(icp_id="retained-source", prompt="Find technology services providers with strategic partnerships",
        industry="Information Technology", sub_industry="IT Infrastructure and Managed Services", employee_count="10,001+",
        company_stage="Public", geography="United States", country="United States", product_service="Managed technology services",
        intent_signals=["Announced a strategic partnership in the last year"], intent_signal_evidence_types=["PARTNERSHIP"], intent_max_age_days=365)
    fit = company_fit_match(details={"dimension_evidence": {
        "identity": {"web_identity_receipt": IDENTITY},
        "industry": {"decision": "match", "web_evidence": {"url": URL, "quote": TEXT[TEXT.index("Under the agreement,"):]}},
    }})
    return company, icp, fit


def _answer(*, status="supported", event_date="2026-10-07", quote=TEXT):
    return {"model": "held-response", "usage": {}, "answer": {
        "overall_verdict": "qualified", "overall_confidence": "high",
        "signal_evaluations": [{"signal_id": "signal-1", "claim": CLAIM,
            "verification_mode": "source_grounded", "signal_status": status,
            "confidence": "high", "same_entity_check": "pass", "evidence_urls_used": [URL],
            "supporting_quotes": [quote], "contradicting_quotes": [], "unsupported_parts": [],
            "claim_matches_miner_date": "consistent", "risk_notes": ["source_event_date:" + event_date]}]}}


def _missing(url=URL):
    return {"results": [], "statuses": [{"url": url, "source": "none", "sd_stage": "all_tiers_exhausted:http_502",
        "sd_error": "http_502", "exa_stage": "exa_transient_exhausted", "exa_error": "HTTP 502"}]}


def _run(monkeypatch, *, pages=None, answer=None, paragraph_match=True, fresh_source=False):
    company, icp, fit = _inputs()
    async def matched_fit(*args, **kwargs):
        lead_scorer._retain_matched_investigator_source_contexts(kwargs["matched_company_source_sink"], fit,
            {PRIVATE_FETCHED_PAGES_KEY: copy.deepcopy(PAGES if pages is None else pages)}, company=company)
        return fit
    fetch = AsyncMock(return_value={"results": [{"url": URL, "text": TEXT}], "statuses": []} if fresh_source else _missing())
    judge = AsyncMock(return_value=_answer() if answer is None else answer)
    paragraph_documents = []
    async def review(co, ic, signals, receipt, **kwargs):
        paragraph_documents.append(intent_details.review_evidence(co, ic, signals, receipt,
            company_source_contexts=kwargs.get("company_source_contexts")))
        return {"gate": "intent_details", "decision": "match" if paragraph_match else "mismatch", "details": {}}
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", matched_fit)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setattr(intent_details, "review_intent_details", review)
    with use_evaluation_date("2026-10-24"):
        result = asyncio.run(lead_scorer.score_company_competition_intent(company, icp, 0, 0, set(), integrity_policy=True, company_quality=True))
    return result, fetch, judge, paragraph_documents


def test_captured_exact_source_reaches_full_scorer_judge_date_and_paragraph(monkeypatch):
    result, fetch, judge, paragraphs = _run(monkeypatch)
    assert result.final_score > 0
    assert fetch.await_count == 1
    assert judge.await_count == 1 and TEXT in judge.await_args.args[2]
    verdict = result.intent_signals_detail[0]["judge_verdict"]
    assert verdict["authoritative_date"] == "2026-10-07" and verdict["date_verdict"] == "in_window"
    assert paragraphs and any(item.get("source_url") == URL and TEXT in item.get("admitted_text", [])
                              for item in paragraphs[0]["admitted_evidence"])


@pytest.mark.parametrize("change", ["unfetched", "wrong_url", "wrong_company", "cross_domain_redirect", "same_domain_other_page", "oversized"])
def test_ineligible_retained_page_preserves_original_provider_failure(monkeypatch, change):
    pages = copy.deepcopy(PAGES)
    if change == "unfetched":
        pages = {}
    elif change == "wrong_url":
        pages = {URL + "-other": pages.pop(URL)}
    elif change == "wrong_company":
        pages[URL]["text"] = TEXT.replace("Amdocs", "OtherCo")
    elif change == "cross_domain_redirect":
        pages[URL]["final_url"] = "https://other.example/release"
    elif change == "same_domain_other_page":
        pages[URL]["final_url"] = "https://www.amdocs.com/press-release/other-event"
    elif change == "extra_fields":
        pages[URL]["summary"] = "Model-authored summary"
    else:
        pages[URL]["text"] = TEXT + "x" * 24_000
    result, fetch, judge, paragraphs = _run(monkeypatch, pages=pages)
    assert result.final_score == 0 and "unavailable" in result.failure_reason.lower()
    assert fetch.await_count == 1
    judge.assert_not_awaited()
    assert not paragraphs


def test_exact_page_does_not_bypass_paragraph_rejection(monkeypatch):
    result, fetch, judge, paragraphs = _run(monkeypatch, paragraph_match=False)
    assert result.final_score == 0 and "paragraph contract" in result.failure_reason
    assert fetch.await_count == 1
    assert judge.await_count == len(paragraphs) == 1


def test_actual_source_date_wins_over_submitted_newer_date(monkeypatch):
    old_text = TEXT.replace("October 7, 2026", "October 7, 2024")
    result, fetch, judge, paragraphs = _run(monkeypatch,
        pages={URL: {"final_url": URL, "text": old_text}},
        answer=_answer(event_date="2024-10-07", quote=old_text))
    assert result.final_score == 0
    verdict = result.intent_signals_detail[0]["judge_verdict"]
    assert verdict["authoritative_date"] == "2024-10-07"
    assert verdict["date_verdict"] == "out_of_window"
    assert fetch.await_count == 1
    assert judge.await_count == 1 and not paragraphs


@pytest.mark.parametrize("case", ["different_event", "wrong_entity", "judge_unavailable"])
def test_retained_source_does_not_replace_normal_event_and_date_judgment(monkeypatch, case):
    response = _answer()
    item = response["answer"]["signal_evaluations"][0]
    if case == "different_event":
        item.update(signal_status="unable_to_verify", unsupported_parts=["The submitted event was not proved"])
    elif case == "wrong_entity":
        item.update(signal_status="wrong_entity", same_entity_check="fail")
    elif case == "ungrounded_quote":
        item["supporting_quotes"] = ["Amdocs announced an entirely different event"]
    elif case == "unbound_date":
        item["risk_notes"] = ["source_event_date:2026-10-08"]
    else:
        response = {"_error": "HTTP 502"}
    result, fetch, judge, paragraphs = _run(monkeypatch, answer=response)
    assert result.final_score == 0
    assert fetch.await_count == 1
    assert judge.await_count == 1 and not paragraphs
    if case == "judge_unavailable":
        assert "unavailable" in result.failure_reason.lower()


def test_private_source_projection_binds_identity_and_requested_resource():
    project = verifier._verified_submitted_source_pages
    assert project(PAGES, [URL], IDENTITY, company_name="Amdocs") == PAGES
    assert project({URL: {**PAGES[URL], "summary": "untrusted"}}, [URL], IDENTITY, company_name="Amdocs") == {}
    assert project(PAGES, [URL + "?other=1"], IDENTITY, company_name="Amdocs") == {}
    assert project(PAGES, [URL], {**IDENTITY, "observed_domain": "other.example"}, company_name="Amdocs") == {}
    assert project(PAGES, [URL], IDENTITY, company_name="OtherCo") == {}
    assert project(PAGES, [URL], {**IDENTITY, "observed_linkedin_slug": ""}, company_name="Amdocs") == {}


def test_hiring_retains_existing_fresh_fetch_with_listing_metadata(monkeypatch):
    company, icp, _ = _inputs()
    fetch = AsyncMock(return_value=_missing())
    judge = AsyncMock()
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    result = asyncio.run(verifier.verify_three_stage(object(), company_name="Amdocs", company_website="https://amdocs.com/",
        company_linkedin="", source_url=URL, miner_claim="Amdocs is hiring cloud engineers", target_signal_text="Hiring cloud engineers",
        miner_signal_date="2026-10-07", evidence_type="HIRING", stage1_soft_reject=True, integrity_policy=True,
        verified_company_identity=IDENTITY, verified_source_pages=PAGES))
    assert result["decision"] == "unavailable" and fetch.await_count == 1
    judge.assert_not_awaited()


def test_source_bytes_enter_local_retry_identity_and_optional_absence_is_stable():
    company, icp, _ = _inputs()
    args = ("round-context", IDENTITY, company.intent_signals, icp.intent_signals[0], "PARTNERSHIP", 365, {})
    old = lead_scorer._intent_terminal_retry_key(*args)
    assert old == lead_scorer._intent_terminal_retry_key(*args, verified_source_pages={})
    first = lead_scorer._intent_terminal_retry_key(*args, verified_source_pages=PAGES)
    changed = copy.deepcopy(PAGES)
    changed[URL]["text"] += " Additional source text."
    assert first != old and first != lead_scorer._intent_terminal_retry_key(*args, verified_source_pages=changed)


def test_existing_signal_cache_control_and_scope_bind_optional_source_bytes():
    from lab_arena import scoring, signal_cache_control, signal_judgments
    from tests.test_arena_score_integrity import _icp, _public_company
    request = {"operation": "claim", "company_index": 0, "signal_index": 0,
               "verified_company_identity": IDENTITY, "verified_funding_stage_context": [], "verified_source_pages": PAGES}
    assert signal_cache_control.validate_request(request)["verified_source_pages"] == PAGES
    with pytest.raises(ValueError):
        signal_cache_control.validate_request({**request, "verified_source_pages": [PAGES]})
    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True)
    policy["signal_judgment_policy"] = signal_judgments.POLICY
    company = _public_company("Amdocs", domain="amdocs.com")
    company["intent_signals"][0]["url"] = URL
    document = scoring.build_scoring_input(scored_run_id="execution", icp=_icp(), companies=[company], policy=policy, evaluation_date="2026-10-24")
    args = dict(scoring_input=document, company_index=0, signal_index=0, verified_company_identity=IDENTITY,
        verified_funding_stage_context=[], round_id="round", network_name="finney", netuid=71,
        scorer_image_digest="sha256:" + "a" * 64, scorer_image_reference="registry/scorer@sha256:" + "a" * 64)
    old = signal_judgments.build_scope(**args)
    assert old == signal_judgments.build_scope(**args, verified_source_pages={})
    scoped = signal_judgments.build_scope(**args, verified_source_pages=PAGES)
    changed = copy.deepcopy(PAGES)
    changed[URL]["text"] += " Additional source text."
    assert scoped["cache_key"] != old["cache_key"]
    assert scoped["cache_key"] != signal_judgments.build_scope(**args, verified_source_pages=changed)["cache_key"]
    assert scoped["expected_context_urls"] == old["expected_context_urls"] == []
    # Training export reconstructs the field from its saved scope, without widening permitted evidence URLs.
    assert scoped == signal_judgments.build_scope(**args, verified_source_pages=scoped["verified_source_pages"])


def test_actual_scorer_sends_only_effective_pages_to_existing_signal_exchange(monkeypatch):
    from qualification.scoring.signal_judgment_cache import install_exchange
    requests = []
    async def exchange(request):
        requests.append(copy.deepcopy(request))
        return {"status": "miss", "reference": {"cache_key": "test"}} if request["operation"] == "claim" else {"status": "stored", "reference": request["reference"]}
    with install_exchange(exchange):
        result, fetch, judge, _ = _run(monkeypatch)
    assert result.final_score > 0 and len(requests) == 2
    assert requests[0]["verified_source_pages"] == requests[1]["verified_source_pages"] == PAGES
    assert fetch.await_count == 1
    assert judge.await_count == 1


@pytest.mark.parametrize("case", ["ungrounded_quote", "unbound_date"])
def test_retained_fallback_requires_grounding_without_changing_fresh_policy(monkeypatch, case):
    # Existing fresh-source policy trusts these held model assertions. A text-only
    # fallback requires literal quote/date grounding to recover an unavailable read.
    response = _answer()
    item = response["answer"]["signal_evaluations"][0]
    if case == "ungrounded_quote":
        item["supporting_quotes"] = ["Amdocs announced an entirely different event"]
    else:
        item["risk_notes"] = ["source_event_date:2026-10-08"]
    retained, *_ = _run(monkeypatch, answer=response)
    fresh, *_ = _run(monkeypatch, pages={}, answer=response, fresh_source=True)
    assert retained.final_score == 0 and "unavailable" in retained.failure_reason.lower()
    assert fresh.final_score > 0


@pytest.mark.parametrize("case", ["undated", "conflicting_dates", "publication_only", "future"])
def test_retained_text_without_one_grounded_event_date_preserves_unavailable(monkeypatch, case):
    body = TEXT
    response = _answer()
    item = response["answer"]["signal_evaluations"][0]
    if case == "undated":
        body = body.replace("[October 7, 2026] – ", "")
        item["risk_notes"] = []
    elif case == "conflicting_dates":
        body += " VEON and Amdocs also announced a separate agreement on October 8, 2026."
        item["risk_notes"].append("source_event_date:2026-10-08")
    elif case == "publication_only":
        item["risk_notes"] = ["source_publication_date:2026-10-07"]
    else:
        body = body.replace("October 7, 2026", "October 7, 2027")
        item["risk_notes"] = ["source_event_date:2027-10-07"]
    item["supporting_quotes"] = [body]
    result, fetch, judge, paragraphs = _run(monkeypatch, pages={URL: {"final_url": URL, "text": body}}, answer=response)
    assert result.final_score == 0 and "unavailable" in result.failure_reason.lower()
    assert fetch.await_count == judge.await_count == 1 and not paragraphs


@pytest.mark.parametrize("publication", ["2026-04-20", "2024-04-20"])
def test_successful_adobe_head_publication_metadata_is_unchanged(monkeypatch, publication):
    from tests import test_publication_date_head_metadata as adobe
    adobe._install_source(monkeypatch)
    scraped = asyncio.run(verifier._scrape_sd_hardened(adobe.URL))
    assert scraped["source_publication_date"] == adobe.PUBLICATION
    fetch = AsyncMock(return_value={"results": [{"url": adobe.URL, "text": scraped["content"],
        "source_title": adobe.TITLE, "source_publication_date": publication}], "statuses": []})
    response = adobe._provider()
    response["answer"]["signal_evaluations"][0]["risk_notes"] = ["source_publication_date:" + publication]
    judge = AsyncMock(return_value=response)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    identity = {**IDENTITY, "observed_name": "adobe", "observed_domain": "adobe.com", "observed_linkedin_slug": "adobe"}
    signal = IntentSignal(source="news", description=adobe.QUOTE, url=adobe.URL, date="2026-10-24", snippet=adobe.QUOTE, matched_icp_signal=0)
    _, icp, _ = _inputs()
    icp = icp.model_copy(update={"intent_signal_evidence_types": ["PRODUCT_LAUNCH"]})
    outcomes = []
    with use_evaluation_date("2026-10-24"):
        for pages in (None, {adobe.URL: {"final_url": adobe.URL, "text": scraped["content"]}}):
            receipts = []
            score = asyncio.run(lead_scorer._score_single_intent_signal(signal, icp, None, "Adobe", "https://adobe.com/",
                stage1_soft_reject=True, llm_only_intent_gate=True, integrity_policy=True, company_quality=True,
                verified_company_identity=identity, verified_source_pages=pages, verdict_out=receipts, evidence_signals=[signal]))
            outcomes.append((score[0], receipts[-1]["authoritative_date"], receipts[-1]["date_verdict"]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] == publication
    assert outcomes[0][2] == ("in_window" if publication.startswith("2026") else "out_of_window")
    assert fetch.await_count == judge.await_count == 2


def test_unproved_retained_fallback_remains_retryable_and_is_not_cached(monkeypatch):
    from lab_arena import scoring
    from qualification.scoring.signal_judgment_cache import install_exchange
    from tests.test_same_event_source_local_finality import _score_to_finality
    requests = []
    async def exchange(request):
        requests.append(copy.deepcopy(request))
        if request["operation"] == "claim":
            return {"status": "miss", "reference": {"cache_key": "test"}}
        assert request["judgment"] is None
        return {"status": "released", "reference": request["reference"]}
    with install_exchange(exchange):
        result, fetch, judge, _ = _run(monkeypatch, answer=_answer(status="unable_to_verify"))
    assert "unavailable" in result.failure_reason.lower()
    assert fetch.await_count == judge.await_count == 1 and len(requests) == 2
    row = result.model_dump(mode="json")
    row.update(company_index=0, company_qualified=False, duplicate_company=False,
        company_identity_key="domain:scored0.example.com", company_identity_alias_keys=["domain:scored0.example.com"])
    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=3"):
        _score_to_finality(row)


@pytest.mark.parametrize("case", ["confirmed_absence", "multi_source"])
def test_existing_absence_and_bundle_paths_are_not_replaced(monkeypatch, case):
    source = _missing()
    kwargs = {}
    if case == "confirmed_absence":
        source["statuses"][0].update(sd_stage="genuine_404", sd_error="http_404", exa_stage="exa_target_not_found", exa_error="target_not_found",
            exa_target_absence={"id_matches_requested_url": True, "status": "error", "error_tag": "CRAWL_NOT_FOUND", "error_http_status": 404, "confirmed_attempts": 2})
    else:
        other = URL + "?second=1"
        source["statuses"].append({**source["statuses"][0], "url": other})
        kwargs["evidence_bundle"] = [{"url": url, "description": CLAIM, "date": "2026-10-07", "snippet": CLAIM} for url in (URL, other)]
    fetch = AsyncMock(return_value=source)
    judge = AsyncMock()
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setattr(verifier, "_find_same_event_source", AsyncMock(return_value={"status": "unproven", "reason": "no_same_event_locator"}))
    result = asyncio.run(verifier.verify_three_stage(object(), company_name="Amdocs", company_website="https://amdocs.com/", company_linkedin="",
        source_url=URL, miner_claim=CLAIM, target_signal_text="Announced a strategic partnership", miner_signal_date="2026-10-07", evidence_type="PARTNERSHIP",
        stage1_soft_reject=True, integrity_policy=True, verified_company_identity=IDENTITY, verified_source_pages=PAGES, **kwargs))
    assert result["decision"] == ("reject" if case == "confirmed_absence" else "unavailable")
    assert fetch.await_count == 1
    judge.assert_not_awaited()


def test_final_query_is_part_of_exact_resource_binding():
    assert verifier._verified_submitted_source_pages({URL: {"final_url": URL + "?different=1", "text": TEXT}}, [URL], IDENTITY, company_name="Amdocs") == {}


@pytest.mark.parametrize("shape", ["declared", "url"])
def test_job_board_metadata_paths_are_not_replaced_for_other_intent_types(monkeypatch, shape):
    url = URL if shape == "declared" else "https://jobs.lever.co/amdocs/role-123"
    fetch = AsyncMock(return_value=_missing(url))
    judge = AsyncMock()
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    # Exact first-party-shaped text cannot override the existing source-type gate.
    if shape == "url":
        monkeypatch.setattr(verifier, "_verified_submitted_source_pages", lambda *args, **kwargs: {url: {"final_url": url, "text": TEXT}})
    result = asyncio.run(verifier.verify_three_stage(object(), company_name="Amdocs", company_website="https://amdocs.com/", company_linkedin="",
        source_url=url, miner_claim=CLAIM, target_signal_text="Announced a strategic partnership", miner_signal_date="2026-10-07", evidence_type="PARTNERSHIP",
        declared_source="job_board" if shape == "declared" else "news", stage1_soft_reject=True, integrity_policy=True,
        verified_company_identity=IDENTITY, verified_source_pages={url: {"final_url": url, "text": TEXT}}))
    assert result["decision"] == "unavailable" and result["rejection_reason"] == "evidence_fetch_failed"
    assert fetch.await_count == 1
    judge.assert_not_awaited()


@pytest.mark.parametrize("overall", ["disqualified", "needs_review", "unverified"])
def test_retained_fallback_requires_affirmative_overall_judgment(monkeypatch, overall):
    answer = _answer()
    answer["answer"]["overall_verdict"] = overall
    result, fetch, judge, paragraphs = _run(monkeypatch, answer=answer)
    assert result.final_score == 0 and "unavailable" in result.failure_reason.lower()
    assert result.intent_signals_detail[0]["judge_verdict"]["pipeline_decision"] == "unavailable"
    assert fetch.await_count == judge.await_count == 1 and not paragraphs


def test_new_source_context_crosses_actual_runner_socket_and_host_http_client(monkeypatch):
    from tests.lab_arena import signal_cache_control_test as controls
    request = {**controls.request(), "verified_source_pages": PAGES}
    monkeypatch.setattr(controls, "request", lambda: copy.deepcopy(request))
    # Exercises WorkerSocketServer -> host validate_request -> HttpArenaApiClient
    # -> host validate_request again -> mock gateway, with no provider debit.
    controls.test_socket_to_gateway_has_no_provider_debit(monkeypatch)
