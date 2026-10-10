"""Exact ATS retrieval must not turn tenant resemblance into employer proof."""

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from qualification.scoring import intent_verification_three_stage as intent


SOURCE = "https://jobs.ashbyhq.com/fluency/c6aceb26-9658-48c2-ac07-6cdf8bbb38df"
BODY = (Path(__file__).parent / "fixtures" / "fluency_ashby_ai_platform.txt").read_text()
CLAIM = (
    "Fluency is hiring a Software Engineer, AI Platform to build and operate "
    "its data platform and related infrastructure."
)
QUOTE = (
    "We're hiring a full-time Software Engineer, AI Platform to own the data "
    "platform, ETL pipelines, and agent infrastructure that everything else "
    "at the company runs on."
)


def _verify(
    *, entity, domain="fluency.inc", slug="fluencyinc", source=SOURCE,
    meta=None, confidence="high", stage1_soft_reject=True, official_links=(),
    status="supported",
):
    prompts = []
    verdict = {
        "answer": {
            "overall_verdict": "qualified" if entity == "pass" else "not_qualified",
            "overall_confidence": "high",
            "signal_evaluations": [{
                "signal_status": status,
                "verification_mode": "source_grounded",
                "same_entity_check": entity,
                "confidence": confidence,
                "claim": CLAIM,
                "claim_matches_miner_date": "consistent",
                "evidence_urls_used": [source],
                "source_accessibility": "accessible",
                "supporting_quotes": [QUOTE],
                "contradicting_quotes": [],
                "risk_notes": [],
                "unsupported_parts": [],
            }],
        },
        "model": "test-model",
        "usage": {},
    }

    async def judge(_client, _model, prompt, **_kwargs):
        prompts.append(prompt)
        return verdict

    contents = {
        "results": [{"url": source, "text": BODY, "meta": meta or {"kind": "ashby_job"}}],
        "statuses": [],
    }
    async def fetched(urls, **kwargs):
        if urls == [source]:
            return contents
        return {"results": [{"url": urls[0], "text": "Official company page",
                "meta": {"observed_ownership_links": [
                    {"url": url, "label": "Careers"} for url in official_links]}}],
                "statuses": []}
    with mock.patch.object(intent, "_call_openrouter", judge), mock.patch.object(
        intent, "_fetch_sd_then_exa", mock.AsyncMock(side_effect=fetched)
    ) as fetch:
        result = asyncio.run(intent.verify_three_stage(
            None,
            company_name="Fluency",
            company_website="https://" + domain,
            company_linkedin="https://www.linkedin.com/company/" + slug,
            source_url=source,
            miner_claim=CLAIM,
            target_signal_text="Actively hiring platform or integration engineers.",
            miner_signal_date="2026-05-16",
            evidence_type="HIRING",
            declared_source="news",
            stage1_soft_reject=stage1_soft_reject,
            integrity_policy=True,
            company_quality=False,
            verified_company_identity={
                "decision": "match",
                "observed_name": "fluency",
                "observed_domain": domain,
                "observed_linkedin_slug": slug,
                "evidence_source": "company_web_reverification",
            },
            buyer_max_age_days=365,
        ))
    assert fetch.await_count == (2 if intent._is_recognized_ats_posting(source) else 1)
    assert all("MODEL-OWNED EXACT HIRING EMPLOYER BINDING" not in p for p in prompts)
    assert all("ATS tenant slug or URL resemblance is only a lookup hint" in p for p in prompts)
    item = (result.get("verdict", {}).get("signal_evaluations") or [{}])[0]
    assert "normalized_exact_hiring_employer_binding" not in item.get("risk_notes", [])
    return result


@pytest.mark.parametrize(("entity", "confidence", "decision"), [
    ("unclear", "high", "review"),
    ("unclear", "medium", "review"),
    ("fail", "high", "reject"),
])
def test_same_name_fluency_posting_does_not_override_employer_verdict(entity, confidence, decision):
    # The lead is the Vermont advertising company; the saved ATS body belongs
    # to usefluency.com's distinct San Francisco workflow software company.
    assert len(BODY) == 4449
    assert "San Francisco" in BODY
    assert "process conformance" in BODY
    result = _verify(entity=entity, confidence=confidence)

    assert result["decision"] == decision
    assert result["client_ready"] is False
    assert result["stage3"]["same_entity_check"] == entity


def test_ats_hiring_cannot_approve_ambiguous_employer_at_stage1():
    result = _verify(entity="unclear", stage1_soft_reject=False)

    assert result["stage1"]["decision"] == "review"
    assert result["decision"] == "review"
    assert result["client_ready"] is False


@pytest.mark.parametrize("source", [
    SOURCE + "?utm_source=careers",
    "https://job-boards.greenhouse.io/fluency/jobs/1234567890",
    "https://jobs.lever.co/fluency/c6aceb26-9658-48c2-ac07-6cdf8bbb38df",
])
def test_recognized_ats_hiring_needs_employer_pass_including_query_urls(source):
    result = _verify(entity="unclear", source=source)

    assert result["decision"] == "review"
    assert result["rejection_reason"] == "stage3_identity_unresolved"
    assert result["client_ready"] is False


def test_non_ats_historical_decision_is_unchanged():
    result = _verify(entity="unclear", source="https://news.example/fluency-hiring")

    assert result["decision"] == "approve"
    assert result["stage3"]["same_entity_check"] == "unclear"


def test_independently_grounded_ats_employer_pass_is_preserved():
    # usefluency.com's first-party Careers link points to this exact board.
    result = _verify(entity="pass", domain="usefluency.com", slug="usefluency",
                     official_links=["https://jobs.ashbyhq.com/fluency"])

    assert result["decision"] == "approve"
    assert result["client_ready"] is True
    assert result["stage3"]["same_entity_check"] == "pass"


@pytest.mark.parametrize("source", [
    "https://fluency.inc/careers",
    "https://www.linkedin.com/company/fluencyinc/posts/",
    "https://www.linkedin.com/jobs/view/1234567890/",
])
def test_existing_official_and_linkedin_employer_passes_are_preserved(source):
    meta = {"kind": "linkedin_job"} if "/jobs/view/" in source else {}
    result = _verify(entity="pass", source=source, meta=meta)

    assert result["decision"] == "approve"
    assert result["client_ready"] is True


@pytest.mark.parametrize(("url", "text", "kind"), [
    (SOURCE + "?unrelated=1", BODY, "ashby_job"),
    (SOURCE.replace("c6aceb26", "a6aceb26"), BODY, "ashby_job"),
    (SOURCE, "Fluency", "ashby_job"),
    (SOURCE, BODY, "greenhouse_job"),
])
def test_existing_exact_ats_source_shape_controls_remain_fail_closed(url, text, kind):
    assert intent._exact_ats_result_binds_company(
        source_url=SOURCE,
        contents={"results": [{"url": url, "text": text, "meta": {"kind": kind}}]},
        company_domain="fluency.inc",
        company_name="Fluency",
    ) is False


def test_unsupported_model_pass_cannot_replace_employer_ownership():
    result = _verify(entity="pass")
    assert result["decision"] == "review"
    assert result["rejection_reason"] == "stage3_identity_unresolved"
    assert result["ats_employer_ownership"]["resolved"] is False
    assert result["job_publisher_relationship"] == "unverified"
    assert result["verified_job_source_urls"] == []


def test_stage1_model_pass_cannot_skip_fetched_ownership():
    result = _verify(entity="pass", stage1_soft_reject=False)
    assert result["stage1"]["decision"] == "review"
    assert result["decision"] == "review"


def test_owned_employer_does_not_promote_partial_criterion():
    result = _verify(entity="pass", status="partially_supported",
                     domain="usefluency.com", slug="usefluency",
                     official_links=["https://jobs.ashbyhq.com/fluency"])
    assert result["decision"] == "review"
    assert result["stage3"]["same_entity_check"] == "pass"
    assert result["ats_employer_ownership"]["resolved"] is True
