"""Real investigator→reviewer transport with no network or paid model calls."""
import asyncio
import json as jsonlib
from types import SimpleNamespace

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from tests.test_product_coverage import CRITERION, URL, QUOTE, finding, matrix


JOINT_EVIDENCE_SUFFIX = (
    "For industry product coverage, a leaf may use up to three exact fetched URL/quote bindings. "
    "Read complementary evidence together for the same verified company and the qualifying "
    "offering or connected company-supplied portfolio permitted by the criterion. Do not require "
    "one quote to repeat every part of a leaf, or one product, platform, or contract for separate "
    "required capabilities unless the criterion requires them to share that scope. Do not combine "
    "disconnected offerings to invent a missing capability, or treat customer-owned capabilities "
    "or hypothetical facts as the company's supplied activity. Every mandatory condition, "
    "including shared market and use, and every source/identity check still applies."
)


def test_joint_evidence_prompt_allows_portfolio_only_when_criterion_permits_it():
    assert "connected company-supplied portfolio permitted by the criterion" in JOINT_EVIDENCE_SUFFIX
    assert "unless the criterion requires them to share that scope" in JOINT_EVIDENCE_SUFFIX
    assert "including shared market and use" in JOINT_EVIDENCE_SUFFIX
    assert "Do not combine disconnected offerings to invent a missing capability" in JOINT_EVIDENCE_SUFFIX


@pytest.mark.parametrize("positive,product", [(True, True), (True, False), (False, True), (False, False)])
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
            normalized = validate_operation_request("openrouter.chat", json)
            assert normalized["messages"] == json["messages"]
            self.requests.append(json)
            if exhausted and len(self.requests) == 1:
                return Response({"choices": [{"finish_reason": "length", "message": {"content": None}}],
                                 "usage": {"completion_tokens": json["max_tokens"],
                                           "completion_tokens_details": {"reasoning_tokens": json["max_tokens"]}}})
            raw = finding(jsonlib.dumps(matrix()) if product else None)
            return Response({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
                "id": "submit", "type": "function", "function": {
                    "name": "submit_findings",
                    "arguments": jsonlib.dumps({"findings": [raw]}),
                },
            }]}}]})

    session = Session()
    matrix()  # Isolate company transport; planner transport is tested separately.
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

    assert result.get("_completed_submit") is True, result
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert len(session.requests) == (2 if exhausted else 1)
    for index, request in enumerate(session.requests):
        assert request["messages"][0] == {"role": "system", "content": inv._SYSTEM_PROMPT}
        if product:
            assert request["messages"][1] == {"role": "system", "content": JOINT_EVIDENCE_SUFFIX}
            assert sum(message["role"] == "system" for message in request["messages"]) == 2
        else:
            assert sum(message["role"] == "system" for message in request["messages"]) == 1
        assert request["tool_choice"] == "required"  # Ordinary turn, including generic retry.
        ordinary_product = positive and product and index == 0
        assert request["max_tokens"] == (4096 if ordinary_product else 3000)
        assert request["model"] == (inv.POSITIVE_SEMANTIC_REVIEW_MODEL if positive or product else inv.INVESTIGATOR_MODEL)
        if product:
            assert request["reasoning"] == {"effort": "medium" if ordinary_product else "low"}
        else:
            assert "reasoning" not in request
    if exhausted:
        assert "currently allowed tools" in session.requests[1]["messages"][-1]["content"]
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    assert inv.MAX_REASONING_TURNS == 8
    assert inv.ADMISSION_DEADLINE_SECONDS == 110.0


@pytest.mark.parametrize("mode", [
    "generic_at_cap", "generic_below_cap", "serialization_at_cap",
    "serialization_below_cap", "forced_final", "late_final", "deadline_exhausted", "repeated_exhaustion",
])
def test_per_request_budget_drives_repairs_and_final_admission(monkeypatch, mode):
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    requests = []

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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, url, *, headers, json):
            validate_operation_request("openrouter.chat", json)
            requests.append(json)
            if len(requests) == 1 and mode in {"forced_final", "late_final"}:
                message = {"tool_calls": [{"id": "search", "type": "function", "function": {
                    "name": "search_web", "arguments": '{"query":"Acme workflow software"}',
                }}]}
            elif len(requests) == 1 or mode == "repeated_exhaustion":
                if mode == "deadline_exhausted":
                    clock.now = inv.ADMISSION_DEADLINE_SECONDS
                used_tokens = 3000 if mode.endswith("below_cap") else json["max_tokens"]
                message = ({"tool_calls": [{"id": "partial", "type": "function", "function": {
                    "name": "submit_findings", "arguments": '{"findings":[',
                }}]} if mode.startswith("serialization") else {"content": None})
                # No finish_reason=length shortcut: usage must be compared to
                # this request's cap (4096 initially, 3000 on the repair).
                return Response({"choices": [{"finish_reason": "stop", "message": message}],
                                 "usage": {"completion_tokens": used_tokens}})
            else:
                raw = finding(jsonlib.dumps(matrix()))
                message = {"tool_calls": [{"id": "submit", "type": "function", "function": {
                    "name": "submit_findings", "arguments": jsonlib.dumps({"findings": [raw]}),
                }}]}
            return Response({"choices": [{"finish_reason": "tool_calls", "message": message}]})

    async def search(*args, **kwargs):
        clock.now = (inv.ADMISSION_DEADLINE_SECONDS if mode == "late_final" else
                     inv.ADMISSION_DEADLINE_SECONDS - inv.JUDGMENT_ADMISSION_RESERVE_SECONDS)
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv.aiohttp, "ClientSession", lambda **kwargs: Session())
    monkeypatch.setattr(inv, "time", clock)
    monkeypatch.setattr(inv, "_search_web", search)
    matrix()  # Precompile the native plan for company-only budget controls.
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("industry",), requested_industry="Software", requested_product_service=CRITERION,
        positive_semantic_review=True, prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "Acme", "registrable_dns_domain": "acme.example"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))
    expected_complete = mode in {"generic_at_cap", "serialization_at_cap", "forced_final", "late_final"}
    assert result.get("_completed_submit", False) is expected_complete
    assert requests[0]["max_tokens"] == 4096
    assert requests[0]["reasoning"] == {"effort": "medium"}
    assert len(requests) == (1 if mode.endswith("below_cap") or mode == "deadline_exhausted" else 2)
    if len(requests) == 2:
        assert requests[1]["max_tokens"] == 3000
        assert requests[1]["reasoning"] == {"effort": "low"}
        if mode in {"serialization_at_cap", "forced_final", "late_final"}:
            assert requests[1]["tool_choice"] == {
                "type": "function", "function": {"name": "submit_findings"},
            }
        if mode == "late_final":
            assert [tool["function"]["name"] for tool in requests[1]["tools"]] == ["submit_findings"]
    assert inv.PRODUCT_REVIEW_MAX_TOKENS == 4096
    assert inv.REASONING_MAX_TOKENS == 3000
    assert inv.MAX_REASONING_TURNS == 8
    assert inv.ADMISSION_DEADLINE_SECONDS == 110.0
