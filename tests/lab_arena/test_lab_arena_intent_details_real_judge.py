"""Exercise Intent Details through the real Arena scorer subprocess."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

import pytest

from lab_arena import intent_details_policy, output, scoring
from qualification.scoring import intent_details
from tests.lab_arena import test_lab_arena_real_judge as real_judge


LAUNCH_URL = real_judge.SITE + "/news/fleet-platform"
LAUNCH_SENTENCE = (
    "Acme Robotics launched its Fleet Control platform on August 20, 2026, "
    "for logistics operating teams."
)
FUNDING_DESCRIPTION = "Acme Robotics announced a $40 million Series B funding round."
LAUNCH_DESCRIPTION = "Acme Robotics launched its Fleet Control platform."
PARAGRAPH = (
    "Acme Robotics announced a $40 million Series B funding round on July 15, "
    "2026, which may support its growth. It also launched its Fleet Control "
    "platform for logistics operating teams on August 20, 2026. Together, the "
    "funding and product launch make Acme Robotics relevant to the ICP for "
    "software companies with recent funding and product expansion."
)


def _icp() -> Dict[str, Any]:
    icp = real_judge.arena_icp()
    icp.update(
        prompt=(
            "Find software companies with recent funding and product expansion"
        ),
        product_service="Fleet-management software for logistics operating teams",
        intent_signals=[
            "Announced a Series A or later funding round in the last 12 months",
            "Launched a software product in the last 12 months",
        ],
    )
    return icp


def _company() -> Dict[str, Any]:
    return {
        "company_name": real_judge.COMPANY,
        "company_website": real_judge.SITE,
        "company_linkedin": "https://www.linkedin.com/company/acme-robotics",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "country": "United States",
        "state": "California",
        "intent_details": PARAGRAPH,
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": FUNDING_DESCRIPTION,
                "date": "2026-07-15",
                "url": real_judge.NEWS_URL,
            },
            {
                "matched_icp_signal": 1,
                "description": LAUNCH_DESCRIPTION,
                "date": "2026-08-20",
                "url": LAUNCH_URL,
            },
        ],
    }


def _scoring_input() -> Dict[str, Any]:
    icp = _icp()
    companies = output.validate_companies(
        [_company()], schema_version=intent_details_policy.OUTPUT_SCHEMA
    )
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        intent_details=True,
    )
    assert "company_quality_policy" not in policy
    return scoring.build_scoring_input(
        scored_run_id="intent-details-real-judge",
        icp=icp,
        companies=companies,
        policy=policy,
        evaluation_date=real_judge.EVALUATION_DATE,
    )


def _source_results() -> Dict[str, Any]:
    return {
        "requestId": "intent-details-evidence",
        "results": [
            {
                "id": real_judge.NEWS_URL,
                "url": real_judge.NEWS_URL,
                "title": "Acme Robotics raises Series B",
                "publishedDate": "2026-07-15T00:00:00.000Z",
                "author": "Acme Robotics",
                "text": real_judge.PAGE_TEXT,
                "highlights": [real_judge.SENTENCE],
                "highlightScores": [0.99],
                "score": 0.99,
            },
            {
                "id": LAUNCH_URL,
                "url": LAUNCH_URL,
                "title": "Acme Robotics launches Fleet Control",
                "publishedDate": "2026-08-20T00:00:00.000Z",
                "author": "Acme Robotics",
                "text": LAUNCH_SENTENCE,
                "highlights": [LAUNCH_SENTENCE],
                "highlightScores": [0.98],
                "score": 0.98,
            },
        ],
        "searchTime": 4.0,
    }


def _stage3_reply(prompt: str) -> Dict[str, Any]:
    is_launch = LAUNCH_DESCRIPTION in prompt
    claim = LAUNCH_DESCRIPTION if is_launch else FUNDING_DESCRIPTION
    url = LAUNCH_URL if is_launch else real_judge.NEWS_URL
    quote = LAUNCH_SENTENCE if is_launch else real_judge.SENTENCE
    event_date = "2026-08-20" if is_launch else "2026-07-15"
    return {
        "overall_verdict": "qualified",
        "overall_confidence": "high",
        "summary": "The supplied first-party source directly supports the claim.",
        "signal_evaluations": [
            {
                "signal_id": "signal-1",
                "claim": claim,
                "verification_mode": "source_grounded",
                "signal_status": "supported",
                "source_urls_supplied": [url],
                "evidence_urls_used": [url],
                "source_accessibility": "accessible",
                "same_entity_check": "pass",
                "entity_match_reason": "The first-party page names Acme Robotics.",
                "supporting_quotes": [quote],
                "contradicting_quotes": [],
                "unsupported_parts": [],
                "source_quality": "first_party",
                "risk_notes": ["source_event_date:" + event_date],
                "confidence": "high",
                "claim_matches_miner_date": "consistent",
                "author_type": "n/a",
                "author_employer_matches_lead": "n/a",
                "author_role_matches_spec": "n/a",
                "author_satisfies_role_spec": "n/a",
            }
        ],
        "missing_or_risks": [],
    }


def _company_fit_reply() -> Dict[str, Any]:
    about_url = real_judge.SITE + "/about"
    return {
        "observed_company_name": real_judge.COMPANY,
        "observed_company_website": real_judge.SITE,
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/acme-robotics"
        ),
        "observed_employee_count": "51-200",
        "employee_size_matches": True,
        "employee_size_evidence_url": about_url,
        "employee_size_evidence_quote": "Acme Robotics has 120 employees.",
        "observed_industry": "Software",
        "observed_subindustry": "Robotics fleet-management software",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": about_url,
        "industry_evidence_quote": (
            "Acme Robotics sells a robotics fleet-management platform."
        ),
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": about_url,
        "geography_evidence_quote": (
            "Acme Robotics is headquartered in San Francisco, California."
        ),
        "observed_company_stage": "Series B",
        "stage_matches": True,
        "stage_evidence_url": real_judge.NEWS_URL,
        "stage_evidence_quote": real_judge.SENTENCE,
        "attribute_satisfied": True,
        "required_attribute_evidence_url": about_url,
        "required_attribute_evidence_quote": (
            "Acme Robotics sells a robotics fleet-management platform."
        ),
        "reason": "Independent sources support every active fit dimension.",
    }


def _local_responder(
    review_checks: Mapping[str, bool] | None,
    review_documents: list[Dict[str, Any]],
):
    """Return a fake provider that honors the two strict judge contracts."""

    def respond(
        operation_id: str, normalized: Mapping[str, Any]
    ) -> Tuple[int, Dict[str, str], bytes]:
        headers = {"content-type": "application/json"}
        if operation_id == "openrouter.chat":
            messages = normalized.get("messages") or []
            prompt = " ".join(str(item.get("content") or "") for item in messages)
            response_format = normalized.get("response_format") or {}
            schema = response_format.get("json_schema") or {}
            schema_name = schema.get("name")
            if schema_name == "arena_intent_details_review":
                assert schema.get("strict") is True
                declared = schema.get("schema") or {}
                assert declared.get("additionalProperties") is False
                assert set(declared.get("required") or []) == {
                    *intent_details._CHECKS,
                    "signal_coverage",
                    "unsupported_factual_clause",
                    "unsupported_factual_reason",
                }
                review_documents.append(json.loads(str(messages[-1]["content"])))
                content = json.dumps(
                    {
                        **dict(review_checks),
                        "signal_coverage": [
                            {"matched_icp_signal": index, "covered": True}
                            for index in (0, 1)
                        ],
                        "unsupported_factual_clause": (
                            PARAGRAPH if not review_checks["facts_supported"] else ""
                        ),
                        "unsupported_factual_reason": (
                            "The supplied evidence does not support this factual clause."
                            if not review_checks["facts_supported"] else ""
                        ),
                    } if review_checks is not None else {}
                )
            elif schema_name == "verification":
                content = json.dumps(_stage3_reply(prompt))
            else:
                content = json.dumps(_company_fit_reply())
            body = real_judge.chat_completion(
                str(normalized.get("model")), content
            )
            return 200, headers, json.dumps(body).encode()
        if operation_id in ("exa.search", "exa.contents"):
            return 200, headers, json.dumps(_source_results()).encode()
        if operation_id == "deepline.execute":
            envelope = {
                "job_id": "job-intent-details",
                "status": "completed",
                "result": {"data": _source_results()},
                "billing": {"credits_charged": 1, "cost_usd": 0.01},
            }
            return 200, headers, json.dumps(envelope).encode()
        if operation_id.startswith("scrapingdog.scrape"):
            html = (
                "<html><head><title>Acme Robotics</title></head><body><p>"
                + real_judge.PAGE_TEXT
                + "</p><p>"
                + LAUNCH_SENTENCE
                + "</p></body></html>"
            )
            return 200, {"content-type": "text/html; charset=utf-8"}, html.encode()
        return real_judge.respond(operation_id, normalized)

    return respond


@pytest.mark.parametrize(
    "failed_check,expected_decision",
    [(None, "match"), ("facts_supported", "mismatch")],
)
def test_real_judge_reviews_one_paragraph_against_two_verified_signals(
    tmp_path, monkeypatch, failed_check: str | None, expected_decision: str
) -> None:
    checks = {
        name: name != failed_check for name in intent_details._CHECKS
    }
    review_documents: list[Dict[str, Any]] = []
    monkeypatch.setattr(
        real_judge, "respond", _local_responder(checks, review_documents)
    )

    document = _scoring_input()
    result = real_judge.run_real_judge(tmp_path, document)
    summary = real_judge.describe(result)

    assert result["exit_code"] == 0 and result["output"] is not None, summary
    assert "failure" not in result["output"], summary
    assert not result["rejected"], summary
    assert len(result["output"]["breakdowns"]) == 1, summary
    breakdown = result["output"]["breakdowns"][0]
    receipt = next(
        item
        for item in breakdown["verifier_gate_receipts"]
        if item.get("gate") == "intent_details"
    )
    assert receipt["decision"] == expected_decision, summary
    assert receipt["checks"] == checks, summary
    assert breakdown["final_score"] > 0 if failed_check is None else breakdown[
        "final_score"
    ] == 0

    assert len(review_documents) == 1, summary
    review = review_documents[0]
    assert review["intent_details"] == PARAGRAPH
    assert [item["matched_icp_signal"] for item in review["verified_signals"]] == [
        0,
        1,
    ]
    assert [item["supporting_quotes"] for item in review["verified_signals"]] == [
        [real_judge.SENTENCE],
        [LAUNCH_SENTENCE],
    ]
    assert [item["authoritative_date"] for item in review["verified_signals"]] == [
        "2026-07-15",
        "2026-08-20",
    ]
    assert document["scorer_policy"]["intent_details_policy"] == (
        intent_details_policy.POLICY
    )
    assert "company_quality_policy" not in document["scorer_policy"]
    assert document["scorer_policy"]["scoring_adapter_version"] == (
        "qualification_integrity_v2"
    )


def test_real_judge_retries_malformed_intent_details_review(
    tmp_path, monkeypatch
) -> None:
    review_documents: list[Dict[str, Any]] = []
    monkeypatch.setattr(
        real_judge, "respond", _local_responder(None, review_documents)
    )

    result = real_judge.run_real_judge(tmp_path, _scoring_input())
    summary = real_judge.describe(result)

    assert result["exit_code"] == 0 and result["output"] is not None, summary
    assert result["output"]["failure"] == "judge_error", summary
    assert result["output"]["reason"] == "malformed_response", summary
    assert len(review_documents) == scoring.MAX_JUDGE_RETRIES, summary
    assert all(len(item["verified_signals"]) == 2 for item in review_documents)
