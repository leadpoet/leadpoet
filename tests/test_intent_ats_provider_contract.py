"""ATS fetches stay inside the trusted Arena provider contract."""

from __future__ import annotations

import json

import httpx
import pytest

from lab_arena import operations, scoring_provider_compat, shim
from qualification.scoring import intent_verification_three_stage as intent


ASHBY_URL = (
    "https://jobs.ashbyhq.com/acme/"
    "12345678-1234-1234-1234-123456789abc"
)
GREENHOUSE_URL = "https://boards.greenhouse.io/acme/jobs/12345"
GREENHOUSE_CLAIM = "Acme is hiring a power electronics engineer."
GREENHOUSE_TARGET = "Company is actively hiring power electronics engineers."
WORKDAY_URL = (
    "https://acme.wd5.myworkdayjobs.com/en-US/Careers/job/"
    "San-Francisco/Software-Engineer_R123"
)


def _ashby_payload() -> dict:
    return {
        "jobs": [{
            "id": "12345678-1234-1234-1234-123456789abc",
            "jobUrl": ASHBY_URL,
            "isListed": True,
            "title": "Software Engineer",
            "descriptionPlain": "Build reliable industrial control systems.",
        }]
    }


def _greenhouse_payload() -> dict:
    return {
        "id": 12345,
        "absolute_url": GREENHOUSE_URL,
        "title": "Software Engineer",
        "company_name": "Acme",
        "content": "<p>Build reliable industrial control systems.</p>",
    }


def _encoded_greenhouse_payload() -> dict:
    return {
        "id": 12345,
        "absolute_url": GREENHOUSE_URL,
        "title": "Power Electronics Engineer",
        "company_name": "Acme",
        "content": (
            "&lt;section&gt;&lt;h2&gt;What You&#39;ll Do&lt;/h2&gt;"
            "&lt;p&gt;Design autonomous systems for industrial sites.&lt;/p&gt;&lt;/section&gt;"
        ),
    }


def _workday_payload() -> dict:
    return {
        "jobPostingInfo": {
            "title": "Software Engineer",
            "jobReqId": "R123",
            "location": "San Francisco",
            "jobDescription": "Build reliable industrial control systems.",
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper", "source_url", "transport_url", "payload", "adapter", "stage"),
    [
        (
            intent._scrape_ashby_job,
            ASHBY_URL,
            "https://api.ashbyhq.com/posting-api/job-board/acme",
            _ashby_payload(),
            "generic_ats_json:ashby",
            "sd:ashby_api:1",
        ),
        (
            intent._scrape_greenhouse_job,
            GREENHOUSE_URL,
            "https://boards-api.greenhouse.io/v1/boards/acme/jobs/12345?content=true",
            _greenhouse_payload(),
            "generic_ats_json:greenhouse",
            "sd:greenhouse_api:1",
        ),
        (
            intent._scrape_workday_cxs,
            WORKDAY_URL,
            (
                "https://acme.wd5.myworkdayjobs.com/wday/cxs/acme/Careers/"
                "job/San-Francisco/Software-Engineer_R123"
            ),
            _workday_payload(),
            "generic_ats_json:workday",
            "sd:workday_cxs:1",
        ),
    ],
)
async def test_specialized_ats_fetch_uses_shim_contract_and_identity_adapter(
    monkeypatch, helper, source_url, transport_url, payload, adapter, stage
):
    calls = []

    class RoutedClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers, params):
            request = httpx.Request("GET", url, headers=headers, params=params)
            operation_id, normalized = operations.match_request(
                request.method, str(request.url), request.content, request.headers
            )
            frame = shim.build_operation_frame(operation_id, normalized, 60_000)
            assert shim.decode_operation_frame(frame) == (
                operation_id,
                normalized,
                60_000,
            )
            route = scoring_provider_compat.route_for(
                kind="score",
                funding_source="miner_key",
                round_id="arena-2026-09-13",
                operation_id=operation_id,
                parameters=normalized,
            )
            assert route is not None
            provider_body = json.dumps({
                "status": "completed",
                "result": {"data": payload},
            }).encode()
            status, response_headers, body, response_url = (
                scoring_provider_compat.adapt_response_with_trusted_url(
                    route,
                    status=200,
                    headers={},
                    body=provider_body,
                )
            )
            calls.append({
                "operation_id": operation_id,
                "parameters": normalized,
                "adapter": route.adapter,
                "response_url": response_url,
            })
            return httpx.Response(
                status,
                headers=response_headers,
                content=body,
                request=request,
            )

    monkeypatch.setenv(
        "SCRAPINGDOG_API_KEY", operations.SCRAPINGDOG_RUNTIME_HANDLE
    )
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: RoutedClient())

    result = await helper(source_url)

    assert result["routed"] is True
    assert result["ok"] is True
    assert result["stage"] == stage
    assert "Software Engineer" in result["content"]
    assert calls == [{
        "operation_id": "scrapingdog.scrape",
        "parameters": {
            "url": transport_url,
            "dynamic": False,
        },
        "adapter": adapter,
        "response_url": "",
    }]


class _GreenhouseClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, *, headers, params):
        self.calls += 1
        request = httpx.Request("GET", url, headers=headers, params=params)
        return httpx.Response(200, json=self.payload, request=request)


def _verdict(status: str) -> dict:
    supported = status == "supported"
    return {
        "answer": {
            "overall_verdict": "qualified" if supported else "not_qualified",
            "overall_confidence": "high",
            "signal_evaluations": [{
                "signal_status": status,
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "evidence_urls_used": [GREENHOUSE_URL],
                "claim_matches_miner_date": "no_date_in_content",
                "source_accessibility": "accessible",
                "claim": GREENHOUSE_CLAIM,
                "supporting_quotes": ["Design autonomous systems for industrial sites."],
                "contradicting_quotes": [],
                "risk_notes": [],
                "unsupported_parts": [],
            }],
        },
        "model": "perplexity/sonar-pro",
        "usage": {},
    }


async def _run_greenhouse_verification(monkeypatch, payload, stage3_status):
    client = _GreenhouseClient(payload)
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(stage3_status)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)
    result = await intent.verify_three_stage(
        None,
        company_name="Acme",
        company_linkedin="",
        company_website="https://acme.com",
        source_url=GREENHOUSE_URL,
        miner_claim=GREENHOUSE_CLAIM,
        target_signal_text=GREENHOUSE_TARGET,
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
    )
    return result, prompts, client


@pytest.mark.asyncio
async def test_greenhouse_decodes_encoded_job_heading_before_body_gate(monkeypatch):
    payload = _encoded_greenhouse_payload()
    client = _GreenhouseClient(payload)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    # This is the exact old failure condition: the HTML extractor preserves
    # the entity while the apostrophe is still encoded, so no anchor matches.
    from qualification.scoring.verification_helpers import extract_article_body

    assert not intent._looks_like_job_body(extract_article_body(payload["content"]))
    result = await intent._scrape_greenhouse_job(GREENHOUSE_URL)

    assert result["ok"] is True
    assert result["stage"] == "sd:greenhouse_api:1"
    assert "What You'll Do" in result["content"]
    assert intent._looks_like_job_body(result["content"])
    assert client.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("date_fields", "expected"),
    [
        ({
            "first_published": "2026-08-20T15:28:09-04:00",
            "updated_at": "2026-09-12T09:00:00-04:00",
        }, "2026-08-20"),
        ({"updated_at": "2026-09-12T09:00:00-04:00"}, ""),
        ({
            "first_published": "August 20, 2026",
            "updated_at": "2026-09-12T09:00:00-04:00",
        }, "2026-08-20"),
    ],
)
async def test_greenhouse_publication_date_uses_only_first_published(
    monkeypatch, date_fields, expected
):
    client = _GreenhouseClient({**_greenhouse_payload(), **date_fields})
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    fetched = await intent._fetch_sd_then_exa([GREENHOUSE_URL])

    assert fetched["results"][0]["source_publication_date"] == expected
    assert client.calls == 1


@pytest.mark.asyncio
async def test_encoded_greenhouse_job_reaches_stage3(monkeypatch):
    result, prompts, client = await _run_greenhouse_verification(
        monkeypatch, _encoded_greenhouse_payload(), "supported"
    )

    assert len(prompts) == 1
    assert "What You'll Do" in json.dumps(prompts[0])
    assert client.calls == 1
    assert result["stage3"]["status"] == "supported"
    assert result["decision"] == "approve"
    assert result["client_ready"] is True


@pytest.mark.asyncio
async def test_encoded_greenhouse_job_still_obeys_negative_stage3(monkeypatch):
    result, prompts, client = await _run_greenhouse_verification(
        monkeypatch, _encoded_greenhouse_payload(), "contradicted"
    )

    assert len(prompts) == 1
    assert client.calls == 1
    assert result["stage3"]["status"] == "contradicted"
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {**_greenhouse_payload(), "content": None},
        {**_greenhouse_payload(), "content": ""},
        {**_greenhouse_payload(), "company_name": ""},
        {**_greenhouse_payload(), "id": 54321},
        {
            **_greenhouse_payload(),
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/54321",
        },
    ],
)
async def test_greenhouse_rejects_malformed_or_wrong_posting(monkeypatch, payload):
    client = _GreenhouseClient(payload)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-runtime-handle")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    result = await intent._scrape_greenhouse_job(GREENHOUSE_URL)

    assert result["ok"] is False
    assert result["stage"] == "greenhouse_api_exhausted"
    assert result["error"] == "posting_invalid"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_greenhouse_shell_still_stops_before_stage3(monkeypatch):
    payload = {
        **_greenhouse_payload(),
        "content": "<main><p>Acme careers portal.</p></main>",
    }
    result, prompts, client = await _run_greenhouse_verification(
        monkeypatch, payload, "supported"
    )

    assert len(prompts) == 0
    assert client.calls == 1
    assert result["stage3"] is None
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"
