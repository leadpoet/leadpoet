"""Regression tests for the native OpenRouter Stage-3 message bound."""

import asyncio
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
            {"role": "user", "content": prompt},
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

    assert len(prompt) <= operations.OPENROUTER_MAX_CONTENT_CHARS
    assert SOURCE_URL in prompt
    assert CLAIM in prompt
    assert TARGET_ICP_SIGNAL in prompt
    assert SIGNAL_DATE in prompt
    assert "SOURCE-START" in prompt
    assert "SOURCE-END Applications are closed" in prompt
    assert _common._SOURCE_OMISSION_MARKER.strip() in prompt
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
                + (f"source-{index}-body " * 1_500)
                + f"\nSOURCE-{index}-END"
            ),
        })

    prompt = intent._build_final_judge_prompt(
        _row(), {"results": sources, "statuses": []}
    )

    assert len(prompt) <= operations.OPENROUTER_MAX_CONTENT_CHARS
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
        if len(prompts) == 1:
            return {
                "answer": {
                    "signal_evaluations": [{
                        "signal_status": "unable_to_verify",
                        "verification_mode": "source_grounded",
                        "same_entity_check": "pass",
                        "confidence": "medium",
                    }]
                },
                "model": "perplexity/sonar",
                "usage": {},
            }
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

    assert len(prompts) == 2
    assert len(prompts[1]) <= operations.OPENROUTER_MAX_CONTENT_CHARS
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
    stage1 = {
        "answer": {
            "signal_evaluations": [{
                "signal_status": "unable_to_verify",
                "verification_mode": "source_grounded",
                "same_entity_check": "unclear",
                "confidence": "medium",
            }]
        },
        "model": "perplexity/sonar",
        "usage": {},
    }
    with mock.patch.object(
        intent,
        "_call_openrouter",
        mock.AsyncMock(side_effect=[stage1, stage3_verdict]),
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
    assert len(prompt) <= operations.OPENROUTER_MAX_CONTENT_CHARS
    assert "COMBINED CRITERION EVIDENCE" in prompt
    assert prompt.count("SOURCE-START") == source_count
    assert prompt.count("SOURCE-END Applications are closed") == source_count
    operations.validate_operation_request("openrouter.chat", _openrouter_parameters(prompt))
