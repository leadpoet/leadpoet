"""Career-site news reaches event review without weakening hiring gates."""

import pytest

from qualification.scoring import competition
from qualification.scoring import intent_verification_three_stage as intent


@pytest.mark.asyncio
@pytest.mark.parametrize("company_quality", [False, True])
@pytest.mark.parametrize("domain,path", [
    ("idp.com", "/news/fy26research"),
    ("example.com", "/announcements/new-market"),
])
@pytest.mark.parametrize("verdict", ["supported", "contradicted", "unable_to_verify"])
async def test_non_hiring_event_on_careers_host_gets_independent_review(
    monkeypatch, domain, path, verdict, company_quality,
):
    source_url = "https://careers." + domain + path
    claim = "Launched new Student Placement destinations in Malaysia and UAE."
    fetched_text = "On August 20, 2026, IDP launched " + claim
    prompts = []

    async def fetch(urls, **kwargs):
        assert urls == [source_url]
        assert not kwargs.get("prefer_dynamic_job_index")
        return {"results": [{"url": source_url, "text": fetched_text}], "statuses": []}

    async def judge(client, model, prompt, **kwargs):
        prompts.append(prompt)
        return {"answer": {
            "overall_verdict": "qualified" if verdict == "supported" else "not_qualified",
            "overall_confidence": "high",
            "signal_evaluations": [{
                "claim": claim,
                "signal_status": verdict,
                "confidence": "high",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "source_accessibility": "accessible",
                "evidence_urls_used": [source_url],
                "supporting_quotes": [fetched_text] if verdict == "supported" else [],
                "contradicting_quotes": [],
                "unsupported_parts": [],
                    "risk_notes": (
                        [
                            "source_event_date:2026-08-20",
                            "source_event_date_binding:verified",
                        ]
                    if verdict == "supported" else []
                ),
                "claim_matches_miner_date": "consistent",
            }],
        }, "model": model, "usage": {}}

    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = await intent.verify_three_stage(
        None,
        company_name="IDP Education" if domain == "idp.com" else "Example",
        company_website="https://" + domain,
        company_linkedin="",
        source_url=source_url,
        miner_claim=claim,
        target_signal_text="Company expanded into a new geographic market.",
        miner_signal_date="2026-08-20",
        evidence_type="MARKET_EXPANSION",
        declared_source=competition._evidence_source(
            source_url, company_website="https://" + domain,
        ),
        stage1_soft_reject=True,
        integrity_policy=True,
        company_quality=company_quality,
        verified_company_identity={
            "decision": "match",
            "evidence_source": "company_homepage",
            "observed_name": "idpeducation" if domain == "idp.com" else "example",
            "observed_domain": domain,
            "observed_linkedin_slug": (
                "idp-education-pty-ltd" if domain == "idp.com" else "example"
            ),
        },
    )

    assert len(prompts) == 1
    assert result["stage3"] is not None
    assert result["client_ready"] is (verdict == "supported")
    if verdict == "supported":
        assert result["verified_source_context"][0]["url"] == source_url


@pytest.mark.asyncio
@pytest.mark.parametrize("company_quality", [False, True])
@pytest.mark.parametrize("evidence_type", ["HIRING", None])
async def test_empty_job_shell_still_fails_for_hiring(
    monkeypatch, evidence_type, company_quality,
):
    source_url = "https://careers.example.com/jobs/123"

    async def fetch(urls, **kwargs):
        return {"results": [{
            "url": source_url, "text": "Example. Join our team.",
        }], "statuses": []}

    async def judge(*args, **kwargs):
        pytest.fail("An empty hiring page must not reach the semantic reviewer")

    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = await intent.verify_three_stage(
        None,
        company_name="Example",
        company_website="https://example.com",
        company_linkedin="",
        source_url=source_url,
        miner_claim="Example has an active software engineer job opening.",
        target_signal_text="Company is actively hiring software engineers.",
        evidence_type=evidence_type,
        declared_source="job_board",
        stage1_soft_reject=True,
        integrity_policy=True,
        company_quality=company_quality,
        verified_company_identity={
            "decision": "match",
            "evidence_source": "company_homepage",
            "observed_name": "example",
            "observed_domain": "example.com",
            "observed_linkedin_slug": "example",
        },
    )
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"
