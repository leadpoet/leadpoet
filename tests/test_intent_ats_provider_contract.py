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
