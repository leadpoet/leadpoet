"""A retained API posting must use the same bound source path as its human URL."""
import json
import sys
import types

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent


@pytest.fixture(params=["default", "frozen_runtime"], autouse=True)
def source_variant(request, monkeypatch):
    if request.param == "default":
        return
    from scripts.arena_reviewer_greenhouse_api_source_seam import convert_greenhouse_api_source_seam
    from tests.reviewer_frozen_sources import frozen_source
    source = convert_greenhouse_api_source_seam(frozen_source(
        "archive_https", "qualification/scoring/intent_verification_three_stage.py",
    ))
    module = types.ModuleType("greenhouse_frozen_runtime_control")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "frozen_greenhouse_control.py", "exec"), module.__dict__)
    monkeypatch.setattr(sys.modules[__name__], "intent", module)

API = "https://boards-api.greenhouse.io/v1/boards/equalexperts/jobs/8425346002"
HUMAN = "https://job-boards.greenhouse.io/equalexperts/jobs/8425346002"
# Retained source fields; the external role-description locator is substituted.
POSTING = {
    "id": 8425346002, "absolute_url": HUMAN, "company_name": "Equal Experts",
    "title": "Senior Delivery Lead", "first_published": "2026-02-19T10:31:08-05:00",
    "updated_at": "2026-09-30T01:01:22-04:00", "location": {"name": "South Africa"},
    "departments": [{"name": "Delivery"}],
    "content": '&lt;p&gt;&lt;a href=&quot;https://example.test/role&quot;&gt;Role Description&lt;/a&gt;&lt;/p&gt;',
}


def transport(monkeypatch, payload=POSTING):
    calls = []

    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None
        async def get(self, url, *, headers=None, params, **_kwargs):
            calls.append(params["url"])
            if params["url"] == "https://equalexperts.com/":
                return httpx.Response(200, text="<html><body><p>" +
                                      "Equal Experts company information. " * 120 + "</p></body></html>")
            return httpx.Response(200, json=payload)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: Client())
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [API, API + "?content=true", HUMAN])
async def test_exact_api_and_human_posting_keep_source_identity(monkeypatch, url):
    calls = transport(monkeypatch)
    result = await intent._fetch_sd_then_exa([url])
    assert calls == [API + "?content=true"]
    assert len(result["results"]) == 1
    source = result["results"][0]
    assert source["url"] == url
    assert "Equal Experts\nSenior Delivery Lead\n2026-02-19" in source["text"]
    assert source["source_publication_date"] == "2026-02-19"
    assert source["meta"]["kind"] == "greenhouse_job"


@pytest.mark.parametrize("url", [
    API.replace("boards-api.greenhouse.io", "boards-api.greenhouse.io.example.test"),
    API.replace("boards-api.greenhouse.io", "example.test"),
    API.replace("/v1/", "/v2/"), API.replace("/jobs/", "/jobs/../jobs/"),
    API + "?content=false", API + "?content=true&content=true",
    API + "?gh_jid=8425346002", API + "/extra",
    API.replace("8425346002", "not-numeric"),
])
def test_api_identity_rejects_unbound_locator(url):
    assert intent._greenhouse_posting_identity(url) is None


def test_api_and_us_human_board_share_only_the_existing_namespace():
    assert intent._ats_board_identity(API) == intent._ats_board_identity(HUMAN)
    assert intent._ats_board_identity(API) != intent._ats_board_identity(
        HUMAN.replace("job-boards.greenhouse", "job-boards.eu.greenhouse")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [
    {"id": 8425346003},
    {"absolute_url": HUMAN.replace("equalexperts", "other-employer")},
    {"absolute_url": HUMAN.replace("8425346002", "8425346003")},
])
async def test_api_source_keeps_returned_posting_binding(monkeypatch, replacement):
    calls = transport(monkeypatch, {**POSTING, **replacement})
    result = await intent._scrape_greenhouse_job(API)
    assert not result["ok"]
    assert result["error"] == "posting_invalid"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,owner,expected", [
    ("supported", "equalexperts.com", "approve"),
    ("contradicted", "equalexperts.com", "reject"),
    ("supported", "other-employer.example", "review"),
    ("supported", None, "review"),
])
async def test_api_source_still_obeys_stage3_verdict(monkeypatch, status, owner, expected):
    # A substantive body is a separate control. The retained link-only body
    # below must still fail the existing job-body gate.
    content = "<h2>About the role</h2><p>Senior Delivery Lead</p>"
    if owner:
        content += f'<a href="https://{owner}/careers">Careers</a>'
    calls = transport(monkeypatch, {**POSTING, "content": content})
    prompts = []

    async def judge(_client, _model, prompt):
        prompts.append(prompt)
        return {"model": "perplexity/sonar-pro", "usage": {}, "answer": {
            "overall_verdict": "qualified" if status == "supported" else "not_qualified",
            "overall_confidence": "high", "signal_evaluations": [{
                "signal_status": status, "verification_mode": "source_grounded",
                "same_entity_check": "pass", "confidence": "high",
                "evidence_urls_used": [API], "claim_matches_miner_date": "no_date_in_content",
                "source_accessibility": "accessible", "claim": "Equal Experts is hiring a Senior Delivery Lead.",
                "supporting_quotes": ["Senior Delivery Lead"], "contradicting_quotes": [],
                "risk_notes": [], "unsupported_parts": [],
            }],
        }}

    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = await intent.verify_three_stage(
        None, company_name="Equal Experts", company_linkedin="",
        company_website="https://equalexperts.com",
        verified_company_identity={"decision": "match", "observed_name": "equalexperts",
            "observed_domain": "equalexperts.com", "observed_linkedin_slug": "",
            "evidence_source": "company_web_reverification"},
        source_url=API, miner_claim="Equal Experts is hiring a Senior Delivery Lead.",
        target_signal_text="Actively hiring for consulting, delivery, or practice-lead roles.",
        evidence_type="HIRING", declared_source="job_board", stage1_soft_reject=True,
    )
    assert len(prompts) == 1, result
    assert "Senior Delivery Lead" in json.dumps(prompts[0])
    assert result["stage3"]["status"] == status
    assert result["decision"] == expected, result
    assert result["client_ready"] is (expected == "approve")
    if owner == "equalexperts.com":
        assert calls == [API + "?content=true"]
        assert result["ats_employer_ownership"]["resolved"] is True
    elif status == "supported":
        assert result["ats_employer_ownership"]["resolved"] is False


@pytest.mark.asyncio
async def test_retained_link_only_posting_keeps_existing_job_body_gate(monkeypatch):
    calls = transport(monkeypatch)
    async def no_judge(*_args, **_kwargs):
        pytest.fail("a link-only posting must not buy a model verdict")
    monkeypatch.setattr(intent, "_call_openrouter", no_judge)
    result = await intent.verify_three_stage(
        None, company_name="Equal Experts", company_linkedin="",
        company_website="https://equalexperts.com",
        verified_company_identity={"decision": "match", "observed_name": "equalexperts",
            "observed_domain": "equalexperts.com", "observed_linkedin_slug": "",
            "evidence_source": "company_web_reverification"},
        source_url=API, miner_claim="Equal Experts is hiring a Senior Delivery Lead.",
        target_signal_text="Actively hiring for consulting, delivery, or practice-lead roles.",
        evidence_type="HIRING", declared_source="job_board", stage1_soft_reject=True,
    )
    assert calls == [API + "?content=true"]
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"
    assert result["stage3"] is None
