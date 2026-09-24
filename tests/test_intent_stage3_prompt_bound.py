"""Regression tests for the native OpenRouter Stage-3 message bound."""

import asyncio
import json
from unittest import mock

import pytest

from lab_arena import operations
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.prompts import _common


SOURCE_URL = (
    "https://jobs.ashbyhq.com/example/"
    "9e8d2540-4a54-4346-bd92-d78fac75e6ff"
)
CLAIM = "Example has an open enterprise account executive position."
TARGET_ICP_SIGNAL = "Company has an active enterprise sales job posting."
SIGNAL_DATE = "2026-09-05"


def _row(*, exact_binding=False):
    return {
        "id": "signal-1",
        "company": "example.com",
        "website": "https://example.com",
        "company_linkedin": "https://www.linkedin.com/company/example",
        "contact_linkedin": "",
        "claim": CLAIM,
        "signal_date": SIGNAL_DATE,
        "signal_type": "intent",
        "claimed_source_urls": [SOURCE_URL],
        "_target_signal_text": TARGET_ICP_SIGNAL,
        "_declared_source": "job_board",
        "_evidence_type": "HIRING",
        "_exact_hiring_employer_binding": exact_binding,
    }


def _long_job_body():
    return (
        "SOURCE-START Example enterprise account executive opening.\n"
        + ("Responsibilities, qualifications, and company context. " * 530)
        + "\nSOURCE-END Applications are closed for this position."
    )


def _openrouter_parameters(prompt):
    return {
        "model": "perplexity/sonar-pro",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": intent._SYS_MESSAGE},
            *intent._openrouter_user_messages(prompt),
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verification",
                "strict": True,
                "schema": intent._SCHEMA,
            },
        },
        "provider": {"data_collection": "deny", "zdr": True},
        "include_reasoning": True,
    }


def _verified_identity():
    return {
        "decision": "match",
        "observed_name": "example",
        "observed_domain": "example.com",
        "observed_linkedin_slug": "example",
        "verified_legal_name_aliases": ["Example Technologies LLC"],
        "evidence_source": "company_web_reverification",
    }


def test_conversion_sized_prompt_preserves_required_context_and_validates():
    body = _long_job_body()
    assert len(body) > 26_000
    prompt = intent._build_final_judge_prompt(
        _row(exact_binding=True),
        {
            "results": [{"url": SOURCE_URL, "title": "Open role", "text": body}],
            "statuses": [],
        },
    )

    assert len(prompt) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    assert SOURCE_URL in prompt
    assert CLAIM in prompt
    assert TARGET_ICP_SIGNAL in prompt
    assert SIGNAL_DATE in prompt
    assert "SOURCE-START" in prompt
    assert "SOURCE-END Applications are closed" in prompt
    assert body in prompt
    assert _common._SOURCE_OMISSION_MARKER.strip() not in prompt
    assert "MODEL-OWNED EXACT HIRING EMPLOYER BINDING" in prompt
    assert "Use only the exact source extraction above" in prompt
    operations.validate_operation_request(
        "openrouter.chat", _openrouter_parameters(prompt)
    )


def test_three_long_sources_each_keep_url_and_both_ends():
    sources = []
    for index in range(3):
        sources.append({
            "url": f"https://evidence.example/source-{index}",
            "title": f"Source {index}",
            "text": (
                f"SOURCE-{index}-START\n"
                + (f"source-{index}-body " * 4_000)
                + f"\nSOURCE-{index}-END"
            ),
        })

    prompt = intent._build_final_judge_prompt(
        _row(), {"results": sources, "statuses": []}
    )

    assert len(prompt) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    for index, source in enumerate(sources):
        assert source["url"] in prompt
        assert f"SOURCE-{index}-START" in prompt
        assert f"SOURCE-{index}-END" in prompt
    assert prompt.count(_common._SOURCE_OMISSION_MARKER.strip()) == 3
    operations.validate_operation_request(
        "openrouter.chat", _openrouter_parameters(prompt)
    )


def test_fixed_metadata_overflow_fails_instead_of_truncating_required_facts():
    row = _row()
    row["claim"] = "required-claim-token " * 2_000

    with mock.patch.object(
        _common, "FINAL_JUDGE_PROMPT_MAX_CHARS", 2_000
    ):
        with pytest.raises(
            ValueError,
            match="fixed context exceeds verifier transport limit",
        ):
            intent._build_final_judge_prompt(
                row,
                {
                    "results": [{
                        "url": SOURCE_URL,
                        "title": "Open role",
                        "text": _long_job_body(),
                    }],
                    "statuses": [],
                },
            )


def test_long_evidence_still_requires_stage3_terminal_verdict():
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        operations.validate_operation_request(
            "openrouter.chat", _openrouter_parameters(prompt)
        )
        return {
            "answer": {
                "signal_evaluations": [{
                    "signal_status": "contradicted",
                    "verification_mode": "source_grounded",
                    "same_entity_check": "pass",
                    "confidence": "high",
                    "evidence_urls_used": [SOURCE_URL],
                    "claim_matches_miner_date": "no_date_in_content",
                    "risk_notes": [],
                    "unsupported_parts": [],
                    "supporting_quotes": [
                        "Applications are closed for this position."
                    ],
                }]
            },
            "model": "perplexity/sonar-pro",
            "usage": {},
        }

    async def fetch(_urls):
        return {
            "results": [{"url": SOURCE_URL, "title": "Open role", "text": _long_job_body()}],
            "statuses": [],
        }

    with mock.patch.object(intent, "_call_openrouter", call_openrouter), mock.patch.object(
        intent, "_fetch_sd_then_exa", fetch
    ):
        result = asyncio.run(intent.verify_three_stage(
            None,
            company_name="Example",
            company_linkedin="https://www.linkedin.com/company/example",
            company_website="https://example.com",
            source_url=SOURCE_URL,
            miner_claim=CLAIM,
            target_signal_text=TARGET_ICP_SIGNAL,
            miner_signal_date=SIGNAL_DATE,
            evidence_type="HIRING",
            stage1_soft_reject=True,
        ))

    assert len(prompts) == 1
    assert len(prompts[0]) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"


def _exact_ats_stage_verdict(
    status,
    *,
    confidence="high",
    same_entity="pass",
    claim=CLAIM,
    supporting_quote="Enterprise Account Executive",
):
    return {
        "answer": {
            "overall_verdict": "qualified" if status == "supported" else "not_qualified",
            "overall_confidence": confidence,
            "signal_evaluations": [{
                "signal_status": status,
                "verification_mode": "source_grounded",
                "same_entity_check": same_entity,
                "confidence": confidence,
                "evidence_urls_used": [SOURCE_URL],
                "claim_matches_miner_date": "no_date_in_content",
                "source_accessibility": "accessible",
                "claim": claim,
                "supporting_quotes": [supporting_quote],
                "contradicting_quotes": [],
                "risk_notes": [],
                "unsupported_parts": [],
            }],
        },
        "model": "perplexity/sonar-pro",
        "usage": {},
    }


def _exact_ats_contents(
    *,
    title="Enterprise Account Executive",
    text=(
        "Enterprise Account Executive\n"
        "Responsibilities: own the sales cycle. Qualifications: five years "
        "of sales experience. Apply for this job. Employment type: full-time."
    ),
):
    return {
        "results": [{
            "url": SOURCE_URL,
            "title": title,
            "text": text,
            "meta": {"kind": "ashby_job"},
        }],
        "statuses": [],
    }


def _exact_ats_result(
    stage3_verdict,
    *,
    claim=CLAIM,
    target_signal=TARGET_ICP_SIGNAL,
    contents=None,
):
    with mock.patch.object(
        intent,
        "_call_openrouter",
        mock.AsyncMock(return_value=stage3_verdict),
    ), mock.patch.object(
        intent,
        "_fetch_sd_then_exa",
        mock.AsyncMock(return_value=contents or _exact_ats_contents()),
    ):
        return asyncio.run(intent.verify_three_stage(
            None,
            company_name="Example",
            company_linkedin="https://www.linkedin.com/company/example",
            company_website="https://example.com",
            source_url=SOURCE_URL,
            miner_claim=claim,
            target_signal_text=target_signal,
            miner_signal_date=SIGNAL_DATE,
            evidence_type="HIRING",
            stage1_soft_reject=True,
        ))


def test_exact_ats_can_resolve_identity_for_semantically_supported_claim():
    result = _exact_ats_result(
        _exact_ats_stage_verdict(
            "supported",
            confidence="medium",
            same_entity="unclear",
        )
    )

    assert result["client_ready"] is True
    assert result["decision"] == "approve"
    assert result["stage3"]["status"] == "supported"
    assert result["stage3"]["same_entity_check"] == "pass"


@pytest.mark.parametrize(
    ("status", "expected_decision", "expected_reason"),
    [
        ("partially_supported", "review", "stage3_review"),
        ("wrong_entity", "reject", "stage3_wrong_entity"),
    ],
)
def test_exact_ats_does_not_promote_non_supported_semantic_verdicts(
    status, expected_decision, expected_reason
):
    result = _exact_ats_result(_exact_ats_stage_verdict(status))

    assert result["client_ready"] is False
    assert result["decision"] == expected_decision
    assert result["rejection_reason"] == expected_reason
    assert result["stage3"]["status"] == status


def test_exact_ats_does_not_promote_true_but_semantically_wrong_role():
    claim = "Example has an open backend engineer position."
    target = "Company has an active enterprise sales job posting."
    result = _exact_ats_result(
        _exact_ats_stage_verdict(
            "contradicted",
            claim=claim,
            supporting_quote="Backend Engineer",
        ),
        claim=claim,
        target_signal=target,
        contents=_exact_ats_contents(
            title="Backend Engineer",
            text=(
                "Backend Engineer\nResponsibilities: build APIs. Qualifications: "
                "five years of Python. Apply for this job. Employment type: full-time."
            ),
        ),
    )

    assert result["client_ready"] is False
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"
    assert result["stage3"]["status"] == "contradicted"


@pytest.mark.parametrize("source_count", [1, 3])
def test_integrity_bundle_and_exact_binding_preserve_prompt_bound(source_count):
    row = _row(exact_binding=source_count == 1)
    row["_integrity_policy"] = True
    row["_evidence_bundle"] = [
        {"url": SOURCE_URL + "?evidence=" + str(index), "description": CLAIM,
         "date": SIGNAL_DATE, "snippet": "An enterprise sales opening"}
        for index in range(source_count)
    ]
    row["claimed_source_urls"] = [item["url"] for item in row["_evidence_bundle"]]
    sources = [{"url": item["url"], "text": _long_job_body()}
               for item in row["_evidence_bundle"]]
    prompt = intent._build_final_judge_prompt(row, {"results": sources, "statuses": []})
    assert len(prompt) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    assert "COMBINED CRITERION EVIDENCE" in prompt
    assert prompt.count("SOURCE-START") == source_count
    assert prompt.count("SOURCE-END Applications are closed") == source_count
    operations.validate_operation_request("openrouter.chat", _openrouter_parameters(prompt))


@pytest.mark.parametrize(
    ("source_count", "exact_ats"),
    [(1, False), (2, False), (3, False), (1, True)],
)
def test_full_verifier_budgets_verified_identity_and_exact_ats_suffixes(
    source_count, exact_ats
):
    urls = [
        SOURCE_URL
        if exact_ats
        else f"https://example.com/evidence/{index}"
        for index in range(source_count)
    ]
    body = (
        "SOURCE-START Example enterprise account executive opening.\n"
        + ("Responsibilities, qualifications, and company context. " * 1_200)
        + "\nSOURCE-END Applications are open for this position."
    )
    evidence_bundle = [
        {
            "url": url,
            "description": CLAIM,
            "date": SIGNAL_DATE,
            "snippet": "Example has an enterprise sales opening.",
        }
        for url in urls
    ]
    contents = {
        "results": [
            {
                "url": url,
                "title": "Enterprise Account Executive",
                "text": body,
                "meta": {"kind": "ashby_job"} if exact_ats else {},
            }
            for url in urls
        ],
        "statuses": [],
    }
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        operations.validate_operation_request(
            "openrouter.chat", _openrouter_parameters(prompt)
        )
        return {
            "answer": {
                "overall_verdict": "qualified",
                "overall_confidence": "high",
                "signal_evaluations": [{
                    "signal_status": "supported",
                    "verification_mode": "source_grounded",
                    "same_entity_check": "pass",
                    "confidence": "high",
                    "evidence_urls_used": [urls[0]],
                    "claim_matches_miner_date": "no_date_in_content",
                    "supporting_quotes": [
                        "Example enterprise account executive opening."
                    ],
                    "risk_notes": [],
                    "unsupported_parts": [],
                }],
            },
            "model": "perplexity/sonar-pro",
            "usage": {},
        }

    with mock.patch.object(
        intent, "_call_openrouter", call_openrouter
    ), mock.patch.object(
        intent, "_fetch_sd_then_exa", mock.AsyncMock(return_value=contents)
    ):
        result = asyncio.run(intent.verify_three_stage(
            None,
            company_name="Example",
            company_linkedin="https://www.linkedin.com/company/example",
            company_website="https://example.com",
            source_url=urls[0],
            miner_claim=CLAIM,
            target_signal_text=TARGET_ICP_SIGNAL,
            miner_signal_date=SIGNAL_DATE,
            evidence_type="HIRING" if exact_ats else "PRODUCT_LAUNCH",
            declared_source="job_board" if exact_ats else "news",
            stage1_soft_reject=True,
            integrity_policy=True,
            verified_company_identity=_verified_identity(),
            evidence_bundle=evidence_bundle,
        ))

    assert len(prompts) == 1
    stage3_prompt = prompts[0]
    assert len(stage3_prompt) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    assert "COMPANY IDENTITY ATTRIBUTION" in stage3_prompt
    if source_count > 1:
        assert _common._SOURCE_OMISSION_MARKER.strip() in stage3_prompt
    else:
        assert body[:_common.MAX_SCRAPED_CHARS] in stage3_prompt
    assert (
        "MODEL-OWNED EXACT HIRING EMPLOYER BINDING" in stage3_prompt
    ) is exact_ats
    assert result["company_check"] is True
    assert result["decision"] == "approve"


def test_single_bounded_page_keeps_its_middle_event_date():
    body = "INTRO " * 5000 + "Example completed its funding in March 2026." + " FOOTER" * 4200
    assert len(body) <= _common.MAX_SCRAPED_CHARS
    prompt = intent._build_final_judge_prompt(_row(), {
        "results": [{"url": SOURCE_URL, "title": "Case study", "text": body}],
        "statuses": [],
    })
    assert body in prompt
    messages = intent._openrouter_user_messages(prompt)
    assert len(messages) > 1
    assert "".join(message["content"] for message in messages) == prompt
    assert all(len(message["content"]) < operations.OPENROUTER_MAX_CONTENT_CHARS for message in messages)
    operations.validate_operation_request("openrouter.chat", _openrouter_parameters(prompt))


def test_unicode_prompt_transport_stays_bounded_without_losing_text():
    prompt = "資料😀" * (_common.FINAL_JUDGE_PROMPT_MAX_CHARS // 3)
    parameters = _openrouter_parameters(prompt)
    assert "".join(m["content"] for m in parameters["messages"][1:]) == prompt
    assert len(json.dumps(parameters).encode()) < 1_000_000
    operations.validate_operation_request("openrouter.chat", parameters)


def test_completed_acquisition_rule_accepts_current_control_but_not_future_deals():
    rule = _common.ACQUISITION_BLOCK
    assert "buyer now owns or controls" in rule
    assert "active post-acquisition integration" in rule
    assert "expected future closing do" in rule
    assert "not prove completion" in rule
