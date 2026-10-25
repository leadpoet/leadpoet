"""Real investigator→reviewer transport with no network or paid model calls."""
import asyncio
import json as jsonlib

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from tests.test_product_coverage import CRITERION, URL, QUOTE, finding, matrix


@pytest.mark.parametrize("positive,product", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("exhausted", [False, True])
def test_effort_survives_real_transport_and_generic_output_length_retry(
    monkeypatch, positive, product, exhausted,
):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return self.body

    class Session:
        def __init__(self):
            self.requests = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, url, *, headers, json):
            assert url == "https://openrouter.ai/api/v1/chat/completions"
            validate_operation_request("openrouter.chat", json)
            self.requests.append(json)
            if exhausted and len(self.requests) == 1:
                return Response({"choices": [{"finish_reason": "length", "message": {"content": None}}],
                                 "usage": {"completion_tokens": 3000,
                                           "completion_tokens_details": {"reasoning_tokens": 3000}}})
            raw = finding(jsonlib.dumps(matrix()) if product else None)
            return Response({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
                "id": "submit", "type": "function", "function": {
                    "name": "submit_findings",
                    "arguments": jsonlib.dumps({"findings": [raw]}),
                },
            }]}}]})

    session = Session()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv.aiohttp, "ClientSession", lambda **kwargs: session)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("industry",), requested_industry="Software",
        requested_product_service=CRITERION if product else "",
        positive_semantic_review=positive,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "Acme", "registrable_dns_domain": "acme.example"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))

    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert len(session.requests) == (2 if exhausted else 1)
    for request in session.requests:
        assert request["tool_choice"] == "required"  # Ordinary turn, including generic retry.
        assert request["max_tokens"] == 3000
        assert request["model"] == (inv.POSITIVE_SEMANTIC_REVIEW_MODEL if positive else inv.INVESTIGATOR_MODEL)
        if positive and product:
            assert request["reasoning"] == {"effort": "low"}
        else:
            assert "reasoning" not in request
    if exhausted:
        assert "currently allowed tools" in session.requests[1]["messages"][-1]["content"]
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    assert inv.MAX_REASONING_TURNS == 8
    assert inv.ADMISSION_DEADLINE_SECONDS == 110.0
