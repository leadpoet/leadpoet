"""Source/publication facts survive nonqualifying verdicts without granting credit."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.evaluation_clock import use_evaluation_date

RECAP = "https://news.example/aurasell-funding-report"
ORIGINAL = "https://aurasell.ai/original-funding"
PRODUCT = "https://aurasell.ai/new-product"
REPORT_QUOTE = "Aurasell announced a $30 million Seed round alongside the product launch."
OLD_QUOTE = "On August 26, 2025, Aurasell announced its $30 million Seed round."
PRODUCT_QUOTE = "Aurasell launched its agent builder on June 24, 2026."
IDENTITY = {"decision":"match","evidence_source":"company_web_reverification",
            "observed_name":"aurasell","observed_domain":"aurasell.ai","observed_linkedin_slug":"aurasell"}


def _inputs(*, evidence_type="FUNDING"):
    company = CompanyOutput.model_construct(
        company_name="Aurasell",company_website="https://aurasell.ai/",
        company_linkedin="https://www.linkedin.com/company/aurasell",
        intent_details="A report dated February 14, 2026 says Aurasell announced Seed funding. Aurasell launched its agent builder in June 2026.",
        intent_signals=[
            type("Signal", (), {"description":REPORT_QUOTE,"url":RECAP,"date":"2026-02-14","matched_icp_signal":1})(),
            type("Signal", (), {"description":PRODUCT_QUOTE,"url":PRODUCT,"date":"2026-06-24","matched_icp_signal":0})(),
        ],
    )
    icp = ICPPrompt(icp_id="nonqualifying-context",prompt="Find software firms with a new product",
        industry="Software",sub_industry="SaaS",employee_count="51-200",geography="United States",
        country="United States",product_service="Software",company_stage="Seed",
        intent_signals=["Launched a new product", "Raised funding"],
        intent_signal_evidence_types=["PRODUCT_LAUNCH",evidence_type],intent_max_age_days=365)
    product = {"after_decay":51,"matched_icp_signal":0,"evidence_type":"PRODUCT_LAUNCH","evidence_urls":[PRODUCT],
        "judge_verdict":{"decision":"verified","client_ready":True,"verification_trace":{
            "intent_verdict":{"signal_evaluations":[{"signal_status":"supported","same_entity_check":"pass",
                "supporting_quotes":[PRODUCT_QUOTE],"evidence_urls_used":[PRODUCT]}]},
            "verified_source_context":[{"url":PRODUCT,"text":PRODUCT_QUOTE,"source_publication_date":"2026-06-24"}]}}}
    negative = {"after_decay":0,"matched_icp_signal":1,"evidence_type":evidence_type,"evidence_urls":[RECAP],
        "judge_verdict":{"decision":"rejected_three_stage","client_ready":False,"verification_trace":{
            "intent_verdict":{"signal_evaluations":[{"signal_status":"contradicted","verification_mode":"source_grounded",
                "same_entity_check":"pass","supporting_quotes":[REPORT_QUOTE,OLD_QUOTE],"contradicting_quotes":[OLD_QUOTE],
                "evidence_urls_used":[RECAP,ORIGINAL],"source_urls_supplied":[RECAP,ORIGINAL],
                "risk_notes":["source_event_date:2025-08-26"]}]},
            "verified_source_context":[
                {"url":RECAP,"text":REPORT_QUOTE,"source_publication_date":"2026-02-14"},
                {"url":ORIGINAL,"text":OLD_QUOTE,"source_publication_date":"2025-08-26"},
            ]}}}
    fit = company_fit_match(details={"dimension_evidence":{"identity":{"web_identity_receipt":IDENTITY}}}).receipt("company_fit")
    return company,icp,[product,negative],fit


def _negative_evaluation(rows):
    return rows[-1]["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"][0]


def _document(*, mutate=None, evidence_type="FUNDING"):
    company,icp,rows,fit = _inputs(evidence_type=evidence_type)
    if mutate: mutate(rows[-1],_negative_evaluation(rows))
    document = intent_details.review_evidence(company,icp,rows,fit)
    return document,rows


def test_rejected_funding_preserves_report_publication_and_original_quote_source():
    document,rows = _document()
    sources = document["admitted_evidence"]
    report = next(x for x in sources if x.get("source_url")==RECAP and REPORT_QUOTE in x.get("admitted_text",[]))
    original = next(x for x in sources if x.get("source_url")==ORIGINAL and OLD_QUOTE in x.get("admitted_text",[]))
    assert report["observed_dates"] == [{"date":"2026-02-14","basis":"source_publication_date"}]
    assert original["observed_dates"] == [{"date":"2025-08-26","basis":"source_publication_date"}]
    assert OLD_QUOTE not in report["admitted_text"]
    assert {r["matched_icp_signal"] for r in document["verified_signals"]} == {0}
    assert document["non_qualifying_signals"][0]["matched_icp_signal"] == 1
    assert rows[-1]["after_decay"] == 0
    assert not any(d.get("basis")=="event_date" for x in sources for d in x.get("observed_dates",[]))


def test_publication_and_quote_share_source_for_existing_recent_report_repair():
    document,_ = _document()
    report = next(x for x in document["admitted_evidence"] if x.get("source_url")==RECAP
                  and REPORT_QUOTE in x.get("admitted_text",[]))
    repair = {"evidence":[{"source_index":report["source_index"],"quote":REPORT_QUOTE},
                          {"source_index":report["source_index"],"quote":"2026-02-14"}]}
    assert intent_details._relative_time_repair_has_source_bound_date(repair,document)
    original = next(x for x in document["admitted_evidence"] if x.get("source_url")==ORIGINAL)
    repair["evidence"][1]["source_index"] = original["source_index"]
    assert not intent_details._relative_time_repair_has_source_bound_date(repair,document)


@pytest.mark.parametrize("mutation", ["wrong_entity","uncited","not_supplied","no_context"])
def test_unbound_negative_source_never_gains_publication_facts(mutation):
    def mutate(row,evaluation):
        if mutation=="wrong_entity": evaluation["same_entity_check"]="fail"
        elif mutation=="uncited": evaluation["evidence_urls_used"]=[RECAP]
        elif mutation=="not_supplied": evaluation["source_urls_supplied"]=[RECAP]
        else: row["judge_verdict"]["verification_trace"].pop("verified_source_context")
    document,_ = _document(mutate=mutate)
    negative_sources=[x for x in document["admitted_evidence"] if x.get("matched_icp_signal")==1]
    if mutation in {"wrong_entity","no_context"}:
        assert not any(x.get("observed_dates") for x in negative_sources)
        assert all(not x.get("source_url") for x in negative_sources)
    else:
        assert not any(x.get("source_url")==ORIGINAL for x in negative_sources)
        # Do not incorrectly assign the old primary quote to sole remaining recap.
        assert not any(OLD_QUOTE in x.get("admitted_text",[]) and x.get("source_url")==RECAP for x in negative_sources)


def test_model_and_submitted_dates_do_not_become_source_publication_metadata():
    def mutate(row,evaluation):
        evaluation["risk_notes"]=["source_publication_date:2026-10-18","source_event_date:2026-10-18"]
        for context in row["judge_verdict"]["verification_trace"]["verified_source_context"]:
            context.pop("source_publication_date")
    document,_ = _document(mutate=mutate)
    negative=[x for x in document["admitted_evidence"] if x.get("matched_icp_signal")==1]
    assert not any(x.get("observed_dates") for x in negative)
    assert "2026-10-18" not in json.dumps(negative)
    # The submitted signal's February date also cannot substitute for a fetch.
    assert "2026-02-14" not in json.dumps(negative)


@pytest.mark.parametrize("published", ["2026-02-30","not-a-date","2026-02-14 event completed",None])
def test_malformed_fetched_publication_date_does_not_enter_fact_projection(published):
    def mutate(row,evaluation):
        for context in row["judge_verdict"]["verification_trace"]["verified_source_context"]:
            context["source_publication_date"]=published
    document,_ = _document(mutate=mutate)
    assert not any(x.get("observed_dates") for x in document["admitted_evidence"] if x.get("matched_icp_signal")==1)


def test_ambiguous_or_unmatched_quotes_remain_explicitly_url_unassigned():
    def mutate(row,evaluation):
        evaluation["supporting_quotes"]=[REPORT_QUOTE,"A legacy unbound model quote."]
        row["judge_verdict"]["verification_trace"]["verified_source_context"][1]["text"] = "A duplicate source repeats: " + REPORT_QUOTE
    document,_ = _document(mutate=mutate)
    source=next(x for x in document["admitted_evidence"] if REPORT_QUOTE in x.get("admitted_text",[]))
    assert not source.get("source_url")
    assert "A legacy unbound model quote." in source["admitted_text"]
    assert not source.get("observed_dates")


def test_nonfunding_publication_observation_is_also_not_event_proof():
    document,_ = _document(evidence_type="JOB_CHANGE")
    source=next(x for x in document["admitted_evidence"] if x.get("source_url")==RECAP)
    assert source["observed_dates"][0]["basis"]=="source_publication_date"
    assert {r["matched_icp_signal"] for r in document["verified_signals"]}=={0}


def test_nonqualifying_context_does_not_consume_positive_page_budget_or_expand_limits():
    company,icp,rows,fit = _inputs()
    rows[0]["judge_verdict"]["verification_trace"]["verified_source_context"][0]["text"] += " x"*10000
    baseline=intent_details.review_evidence(company,icp,rows[:1],fit)
    rows[-1]["judge_verdict"]["verification_trace"]["verified_source_context"][0]["text"] += " y"*20000
    document=intent_details.review_evidence(company,icp,[rows[0],*[copy.deepcopy(rows[-1]) for _ in range(4)]],fit)
    assert len(document["non_qualifying_signals"])==3
    positive=lambda doc:next(x["admitted_text"] for x in doc["admitted_evidence"] if x.get("source_url")==PRODUCT)
    assert positive(document)==positive(baseline)
    assert len(json.dumps(document,ensure_ascii=False))<=intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
    for source in document["admitted_evidence"]:
        if source["evidence_kind"].startswith("non_qualifying_"):
            assert len(source.get("admitted_text",[]))<=2
            assert all(len(q)<=700 for q in source.get("admitted_text",[]))


@pytest.mark.parametrize("status,entity,retained", [
    ("contradicted","pass",True), ("partially_supported","pass",True),
    ("wrong_entity","fail",True), ("contradicted","fail",False),
    ("unable_to_verify","pass",False),
])
def test_completed_grounded_source_context_survives_real_scorer_trace(monkeypatch,status,entity,retained):
    from tests import test_funding_stage_source_reuse as funding
    answer = funding._answer(None,urls=[funding.SECONDARY,funding.PRIMARY])
    evaluation=answer["answer"]["signal_evaluations"][0]
    evaluation.update(signal_status=status,same_entity_check=entity)
    monkeypatch.setattr(funding,"_answer",lambda *args,**kwargs:copy.deepcopy(answer))
    result,receipts,fetch,provider=funding._run(
        monkeypatch,context=[funding.CONTEXT],publication_date="2026-02-14",
        body=funding.RECAP_TEXT + " extra"*10000,
    )
    contexts=receipts[-1]["verification_trace"].get("verified_source_context") or []
    assert bool(contexts) is retained
    assert result[0]==0
    if status == "wrong_entity":
        # Retain exact negative proof for judgment reuse. It remains rejected;
        # the separate paragraph projection still withholds its company facts.
        traced = receipts[-1]["verification_trace"]["intent_verdict"]["signal_evaluations"][0]
        assert traced["signal_status"] == "wrong_entity"
        assert traced["same_entity_check"] == "fail"
        assert receipts[-1]["client_ready"] is False
    if retained:
        assert all(len(c["text"])<=700 for c in contexts)
        assert any(c["url"]==funding.SECONDARY and c["source_publication_date"]=="2026-02-14" for c in contexts)
    assert fetch.await_count==1


def test_provider_metadata_and_model_date_tags_cannot_supply_unfetched_publication(monkeypatch):
    from tests import test_funding_stage_source_reuse as funding
    answer=funding._answer(None,urls=[funding.SECONDARY,funding.PRIMARY])
    answer["answer"]["signal_evaluations"][0].update(
        signal_status="contradicted",risk_notes=["source_publication_date:2026-02-14"],
    )
    monkeypatch.setattr(funding,"_answer",lambda *args,**kwargs:copy.deepcopy(answer))
    _,receipts,_,_=funding._run(monkeypatch,context=[funding.CONTEXT],
        metadata={"source_publication_date":"2026-02-14"})
    assert all(not c.get("source_publication_date") for c in receipts[-1]["verification_trace"]["verified_source_context"])


@pytest.mark.parametrize("kind", ["flattened_header","strict_dateline","footer"])
def test_literal_source_date_survives_compact_trace_without_event_inference(monkeypatch,kind):
    from tests import test_funding_stage_source_reuse as funding
    header = "Aurasell launches. " + "intro "*40 + "San Francisco August 26, 2025 Since the launch. "
    if kind=="strict_dateline": header="Aurasell launches.\nSan Francisco August 26, 2025 - An announcement.\n"
    if kind=="footer": header="Aurasell launches. No dated header. "
    text=header + "details "*500 + funding.QUOTE
    if kind=="footer": text += " footer August 26, 2025"
    answer=funding._answer(None,urls=[funding.SECONDARY,funding.PRIMARY])
    answer["answer"]["signal_evaluations"][0]["signal_status"]="contradicted"
    monkeypatch.setattr(funding,"_answer",lambda *args,**kwargs:copy.deepcopy(answer))
    _,receipts,_,_=funding._run(monkeypatch,context=[{**funding.CONTEXT,"text":text}],publication_date="2026-02-14")
    trace=receipts[-1]["verification_trace"]
    source=next(c for c in trace["verified_source_context"] if c["url"]==funding.PRIMARY)
    assert len(source["text"])<=700
    assert ("August 26, 2025" in source["text"])==(kind!="footer")
    assert source.get("source_body_dateline_dates",[])==(["2025-08-26"] if kind=="strict_dateline" else [])
    company,icp,rows,fit=_inputs()
    company.intent_signals[0].url=funding.SECONDARY
    rows[-1]["evidence_urls"]=[funding.SECONDARY]
    rows[-1]["judge_verdict"]["verification_trace"]=trace
    document=intent_details.review_evidence(company,icp,rows,fit)
    sources=[s for s in document["admitted_evidence"] if s.get("source_url")==funding.PRIMARY]
    assert any("August 26, 2025" in q for s in sources for q in s.get("admitted_text",[]))==(kind!="footer")
    assert not any(d["basis"]=="event_date" for s in sources for d in s.get("observed_dates",[]))
    assert sum(len(s.get("admitted_text",[])) for s in sources if s["evidence_kind"]=="non_qualifying_supporting_quotes")<=2


def test_dated_excerpt_never_displaces_uniquely_bound_substantive_quotes():
    def mutate(row,evaluation):
        second="Aurasell completed another reported action."
        evaluation["supporting_quotes"]=[REPORT_QUOTE,second]
        contexts=row["judge_verdict"]["verification_trace"]["verified_source_context"]
        contexts[0]["text"]+=" "+second
        contexts[1]["text"]="San Francisco August 26, 2025 Since the announcement."
    document,_=_document(mutate=mutate)
    sources=[s for s in document["admitted_evidence"] if s.get("matched_icp_signal")==1]
    quoted=[q for s in sources for q in s.get("admitted_text",[])]
    assert REPORT_QUOTE in quoted and "Aurasell completed another reported action." in quoted
    assert not any("San Francisco" in q for q in quoted)
