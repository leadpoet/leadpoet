"""Publisher premiums use observed ownership and the exact cited posting."""

import asyncio
import copy
import socket

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_arena_score_integrity import _company_model, _icp_model


OWNED = "https://boards.greenhouse.io/acme/jobs/12345"
UNOWNED = "https://boards.greenhouse.io/contoso/jobs/45678"
HOME = "https://acme.com/"
IDENTITY = {
    "decision": "match", "observed_name": "acme", "observed_domain": "acme.com",
    "observed_linkedin_slug": "acme", "evidence_source": "company_web_reverification",
}
BODY = (
    "Acme Software Engineer job. Responsibilities: build Acme's data platform. "
    "Acme uses Python in production. Qualifications: Python engineering. Apply now."
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("ATS publisher tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def _exercise(
    monkeypatch, *, link="direct", entity="pass", company_quality=False,
    evidence_type="TECHSTACK", bundle=False, cite_owned=True,
    full_source_allowance=False, with_identity=True,
):
    primary = UNOWNED if bundle and not cite_owned else OWNED
    urls = [primary]
    if bundle:
        urls.append(OWNED if primary == UNOWNED else UNOWNED)
    if full_source_allowance:
        urls.extend(["https://news.example/one", "https://news.example/two"])
    post_body = BODY if entity == "pass" else BODY.replace("Acme", "Contoso")
    if link == "direct":
        post_body += "\n[Careers](https://acme.com/careers)"
    pages = {}
    for url in urls:
        text = post_body if url == OWNED else BODY.replace("Acme", "Contoso")
        pages[url] = {"url": url, "text": text, "meta": {
            "kind": "greenhouse_job" if url in (OWNED, UNOWNED) else "news",
            "observed_ownership_links": intent._observed_ownership_links(text, url),
        }}
    home_url = "https://contoso.com/" if link == "wrong_property" else HOME
    board = "https://boards.greenhouse.io/contoso" if link == "wrong_board" else "https://boards.greenhouse.io/acme"
    label = "Powered by" if link == "powered_by" else "Careers"
    home_body = "No observed careers link" if link == "absent" else f'<a href="{board}">{label}</a>'
    home = {"url": home_url, "text": home_body, "meta": {
        "observed_ownership_links": intent._observed_ownership_links(home_body, home_url),
    }}
    fetches = []

    async def fetch(requested, **kwargs):
        fetches.append(requested)
        if requested == urls:
            return {"results": [copy.deepcopy(pages[url]) for url in urls], "statuses": []}
        assert requested == [HOME]
        if link == "unavailable":
            return {"results": [], "statuses": [{"url": HOME, "source": "exa", "error": "timeout"}]}
        return {"results": [copy.deepcopy(home)], "statuses": []}

    async def judge(*args, **kwargs):
        return {"answer": {
            "overall_verdict": "qualified" if entity == "pass" else "not_qualified",
            "overall_confidence": "high", "signal_evaluations": [{
                "signal_status": "supported" if entity == "pass" else "wrong_entity",
                "verification_mode": "source_grounded", "same_entity_check": entity,
                "confidence": "high", "evidence_urls_used": [primary],
                "claim_matches_miner_date": "no_date_in_content", "risk_notes": [],
            }],
        }, "model": "offline-control", "usage": {}}

    target = "Actively hiring Software Engineers" if evidence_type == "HIRING" else "Uses Python in its production tech stack"
    claim = "Acme is hiring Software Engineers" if evidence_type == "HIRING" else "Acme uses Python in production"
    signals = _company_model([{
        "source": "job_board", "description": claim, "url": url,
        "date": None, "snippet": "Production Python engineering", "matched_icp_signal": 0,
    } for url in urls]).intent_signals
    icp = _icp_model().model_copy(update={
        "intent_signals": [target], "intent_signal_evidence_types": [evidence_type],
        "intent_max_age_days": 365,
    })
    receipts = []
    verifier = intent.verify_three_stage

    async def retain_result(*args, **kwargs):
        result = await verifier(*args, **kwargs)
        receipts.append(result)
        return result

    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    monkeypatch.setattr(intent, "verify_three_stage", retain_result)
    verdicts = []
    with use_evaluation_date("2026-10-16"):
        score = asyncio.run(lead_scorer._score_single_intent_signal(
            signals[0], icp, None, "Acme", HOME, trust_signal_date=True,
            stage1_soft_reject=True, llm_only_intent_gate=True,
            integrity_policy=True, company_quality=company_quality,
            verified_company_identity=IDENTITY if with_identity else None,
            evidence_signals=signals if len(signals) > 1 else None,
            verdict_out=verdicts,
        ))
    return score[0], receipts[0], verdicts[-1], fetches


@pytest.mark.parametrize("company_quality", [False, True])
@pytest.mark.parametrize("evidence_type", ["TECHSTACK", "NEWS"])
def test_fetched_official_careers_backlink_earns_premium_without_extra_fetch(
    monkeypatch, company_quality, evidence_type,
):
    score, receipt, verdict, fetches = _exercise(
        monkeypatch, company_quality=company_quality, evidence_type=evidence_type,
    )
    assert score == 60
    assert receipt["client_ready"] is True
    assert receipt["ats_employer_ownership"]["source_count"] == 1
    assert receipt["ats_employer_ownership"]["anchors"][0]["kind"] == "job_official_careers_link"
    assert receipt["verified_job_source_urls"] == [OWNED]
    assert verdict["source_multiplier"] == 1.0
    assert fetches == [[OWNED]]


@pytest.mark.parametrize("company_quality", [False, True])
def test_observed_official_homepage_link_binds_exact_ats_board(monkeypatch, company_quality):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link="home", company_quality=company_quality)
    assert score == 60
    assert receipt["job_publisher_relationship"] == "verified"
    assert receipt["ats_employer_ownership"]["source_count"] == 2
    assert receipt["verified_job_source_urls"] == [OWNED]
    assert fetches == [[OWNED], [HOME]]


@pytest.mark.parametrize("link", ["absent", "wrong_property", "wrong_board", "powered_by", "unavailable"])
def test_unproven_owner_retains_ordinary_techstack_credit(monkeypatch, link):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link=link)
    assert score == 54
    assert receipt["client_ready"] is True
    assert receipt["job_publisher_relationship"] == "unverified"
    assert receipt["ats_employer_ownership"]["resolved"] is False
    assert receipt["verified_job_source_urls"] == []
    assert verdict["source_multiplier"] == 0.9
    assert fetches == [[OWNED], [HOME]]


@pytest.mark.parametrize("company_quality", [False, True])
def test_ownership_anchor_never_overrides_wrong_employer(monkeypatch, company_quality):
    score, receipt, verdict, fetches = _exercise(monkeypatch, entity="fail", company_quality=company_quality)
    assert score == 0
    assert receipt["decision"] == "reject"
    assert receipt["client_ready"] is False
    assert receipt["stage3"]["same_entity_check"] == "fail"
    assert fetches == [[OWNED]]


@pytest.mark.parametrize("cite_owned", [False, True])
def test_bundle_premium_belongs_only_to_the_cited_owned_posting(monkeypatch, cite_owned):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link="home", bundle=True, cite_owned=cite_owned)
    assert receipt["client_ready"] is True
    assert receipt["verified_job_source_urls"] == [OWNED]
    assert receipt["verdict"]["signal_evaluations"][0]["evidence_urls_used"] == [OWNED if cite_owned else UNOWNED]
    assert score == (60 if cite_owned else 54)
    assert verdict["source_multiplier"] == (1.0 if cite_owned else 0.9)
    assert len(fetches) == 2


def test_publisher_lookup_keeps_existing_three_source_allowance(monkeypatch):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link="home", full_source_allowance=True)
    assert score == 54
    assert receipt["ats_employer_ownership"] == {"anchors": [], "source_count": 3, "resolved": False}
    assert verdict["source_multiplier"] == 0.9
    assert len(fetches) == 1
    assert len(fetches[0]) == 3


def test_hiring_still_requires_independent_ownership(monkeypatch):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link="absent", evidence_type="HIRING")
    assert score == 0
    assert receipt["decision"] == "review"
    assert receipt["rejection_reason"] == "stage3_identity_unresolved"
    assert receipt["client_ready"] is False


def test_candidate_domain_without_verified_receipt_cannot_authorize_lookup(monkeypatch):
    score, receipt, verdict, fetches = _exercise(monkeypatch, link="home", with_identity=False)
    assert score == 54
    assert receipt["ats_employer_ownership"]["resolved"] is False
    assert receipt["verified_job_source_urls"] == []
    assert fetches == [[OWNED]]
