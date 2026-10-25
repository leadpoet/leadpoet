"""Bounded planner transport and admission controls; all provider IO is fake."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import product_coverage as pc


@pytest.mark.parametrize("mode,attempts,accepted", [
    ("repair", 2, True), ("invalid", 2, False),
    ("window_expired", 1, False), ("late_settlement", 1, True),
    ("errors_envelope", 2, False), ("failed_envelope", 2, False),
])
def test_planner_attempts_and_settlement_are_bounded_without_cancelling(monkeypatch, mode, attempts, accepted):
    pc._PLAN_CACHE.clear()
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    criterion = "Designs useful software for clinics. " + mode
    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            if mode in {"window_expired", "late_settlement"}:
                clock.now = 45.0
            valid = mode in {"late_settlement", "errors_envelope", "failed_envelope"} or (mode == "repair" and len(requests) == 2)
            plan = ({"root": "model-root", "nodes": [{"id": "model-root", "op": "leaf",
                     "text": "Designs clinic software", "children": []}]} if valid else
                    {"root": "r", "nodes": []})
            body = {"choices": [{"message": {"tool_calls": [{"id": str(len(requests)),
                "type": "function", "function": {"name": "submit_product_plan",
                    "arguments": json.dumps({"plan_json": json.dumps(plan)})}}]}}]}
            if mode == "errors_envelope":
                body["errors"] = [{"code": "provider_failed"}]
            elif mode == "failed_envelope":
                body["status"] = "failed"
            return body

    class Session:
        def post(self, url, *, headers, json):
            validate_operation_request("openrouter.chat", json)
            requests.append(json)
            return Response()

    monkeypatch.setattr(inv, "time", clock)
    plan, receipt = asyncio.run(inv._get_or_compile_product_plan(
        Session(), criterion=criterion, headers={}, started=0.0,
    ))
    assert (plan is not None) is accepted
    assert len(requests) == len(receipt["attempts"]) == attempts
    assert receipt["outcome"] == ("accepted" if accepted else "unavailable")
    assert all(json.loads(r["messages"][1]["content"]) == {"requested_product_service": criterion} for r in requests)
    assert all(r["model"] == inv.POSITIVE_SEMANTIC_REVIEW_MODEL and r["max_tokens"] == 4096 for r in requests)
    assert inv.MAX_PRODUCT_PLAN_ATTEMPTS == 2 and inv.PRODUCT_PLAN_WINDOW_SECONDS == 30.0
    if accepted:
        assert pc.get_cached_plan(criterion, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL) is plan
        cached, cached_receipt = asyncio.run(inv._get_or_compile_product_plan(
            Session(), criterion=criterion, headers={}, started=clock.now,
        ))
        assert cached is plan and cached_receipt["cache_hit"] is True
        assert cached_receipt["attempts"] == [] and len(requests) == attempts
    else:
        assert pc.get_cached_plan(criterion, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL) is None


def test_planner_failure_is_an_unfinished_infrastructure_result_not_company_nonfit(monkeypatch):
    receipt = {"outcome": "unavailable", "phase": "product_plan", "attempts": [1, 2],
               "failure_reason": inv.PRODUCT_PLAN_FAILURE_REASON}
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_get_or_compile_product_plan", AsyncMock(return_value=(None, receipt)))
    diagnostic = {}
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "UnavailableCo", "website": "https://unavailable.example"},
        targets=("industry",), requested_industry="Software", requested_product_service="Provides software",
        diagnostic=diagnostic,
    ))
    assert result["claims"] == {} and result["_completed_submit"] is False
    assert result["failure_reason"] == inv.MALFORMED_RESPONSE_FAILURE_REASON
    assert result["product_plan_failure_reason"] == inv.PRODUCT_PLAN_FAILURE_REASON
    assert result["usage"]["product_plan"] == receipt
    assert result["usage"]["reasoning_turns"] == result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
