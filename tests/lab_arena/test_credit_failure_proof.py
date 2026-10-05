"""Durable, sanitized proof for settled no-charge provider credit refusals."""

from __future__ import annotations

import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, CONTEXT, FakeTransport, HOST_KEYS, make_broker,
)


DEEPLINE_REFUSAL = {
    "code": "INSUFFICIENT_CREDITS",
    "error": "Insufficient credits",
    "billing": {
        "kind": "insufficient_credits",
        "required_credits": 5,
        "balance_credits": 4.14,
        "needed_credits": 0.86,
    },
}


def _request(provider):
    return {
        "openrouter": ("openrouter.chat", CHAT),
        "deepline": (
            "deepline.execute",
            {"tool": "exa_search", "payload": {"query": "x"}},
        ),
        "scrapingdog": (
            "scrapingdog.scrape", {"url": "https://example.com/about"},
        ),
    }[provider]


def _broker(response, *, funding_source):
    return make_broker(
        transport=FakeTransport([response]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: funding_source,
    )


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
@pytest.mark.parametrize(
    "provider,response",
    [
        ("openrouter", (402, {"error": {"code": 402, "message": "quota"}})),
        ("deepline", (402, DEEPLINE_REFUSAL)),
        ("scrapingdog", (402, {"error": "payment required"})),
    ],
)
def test_direct_credit_refusal_persists_proof_and_replays_without_dispatch(
    provider, response, funding_source,
):
    current, store, transport = _broker(response, funding_source=funding_source)
    operation_id, parameters = _request(provider)
    kwargs = dict(
        operation_id=operation_id, parameters=parameters,
        action_sequence=0, timeout_ms=5000,
    )

    first = current.execute(CONTEXT, **kwargs)
    terminal = store.calls[first.call["call_identity"]]["terminal"]
    proof = {
        "schema_version": "leadpoet.lab_arena.credit_failure_proof.v1",
        "provider": provider,
        "reason": "out_of_credit",
        "provider_status": 402,
        "actual_microusd": 0,
    }
    assert first.call["outcome"] == "settled"
    assert first.call["actual_microusd"] == 0
    assert first.call["credit_failure_proof"] == proof
    assert terminal["credit_failure_proof"] == proof
    assert terminal["call_succeeded"] is False
    assert "Insufficient credits" not in json.dumps(proof)

    sent = len(transport.sent)
    replay = current.execute(CONTEXT, **kwargs)
    assert len(transport.sent) == sent
    assert replay.call["idempotent"] is True
    assert replay.call["credit_failure_proof"] == proof
    assert replay.call["actual_microusd"] == 0


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_openrouter_http_200_embedded_402_needs_zero_charge_proof(funding_source):
    response = {
        "object": "response",
        "status": "failed",
        "error_type": "payment_required",
        "error": {"code": "server_error", "message": "quota"},
        "usage": {"cost": 0},
    }
    current, store, _ = _broker((200, response), funding_source=funding_source)
    result = current.execute(
        CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
        action_sequence=1, timeout_ms=5000,
    )
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert result.call["outcome"] == "settled"
    assert result.call["credit_failure_proof"]["provider_status"] == 402
    assert terminal["status"] in (402, 502)
    assert terminal["credit_failure_proof"] == result.call["credit_failure_proof"]


@pytest.mark.parametrize("cost", [0.01, None])
def test_embedded_openrouter_402_charged_or_unknown_is_not_safe(cost):
    response = {
        "object": "response", "status": "failed",
        "error_type": "payment_required",
        "error": {"code": "server_error", "message": "quota"},
    }
    if cost is not None:
        response["usage"] = {"cost": cost}
    current, store, _ = _broker((200, response), funding_source="miner_key")
    result = current.execute(
        CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
        action_sequence=2, timeout_ms=5000,
    )
    assert "credit_failure_proof" not in result.call
    call = store.calls[result.call["call_identity"]]
    assert "credit_failure_proof" not in call.get("terminal", {})
    if cost is None:
        assert result.call["outcome"] == "uncertain"
    else:
        assert result.call["outcome"] == "settled"
        assert result.call["actual_microusd"] > 0


def test_direct_openrouter_402_with_unpriced_generation_has_no_proof(monkeypatch):
    monkeypatch.setattr(br, "_openrouter_generation_readback", lambda **_kwargs: None)
    current, store, _ = _broker(
        (402, {"id": "gen-unpriced", "error": {"code": 402}}),
        funding_source="miner_key",
    )
    result = current.execute(
        CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
        action_sequence=3, timeout_ms=5000,
    )
    assert "credit_failure_proof" not in result.call
    assert "credit_failure_proof" not in store.calls[result.call["call_identity"]]["terminal"]


def test_direct_openrouter_402_with_malformed_charge_has_no_proof():
    current, store, _ = _broker(
        (402, {"error": {"code": 402}, "usage": {"cost": "unknown"}}),
        funding_source="miner_key",
    )
    result = current.execute(
        CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
        action_sequence=8, timeout_ms=5000,
    )
    assert "credit_failure_proof" not in result.call
    assert "credit_failure_proof" not in store.calls[result.call["call_identity"]]["terminal"]


@pytest.mark.parametrize(
    "provider,response",
    [
        (
            "openrouter",
            (402, {"error": {"code": 402}, "usage": {"cost": "0.01"}}),
        ),
        (
            "deepline",
            (402, {
                "job_id": "charged-refusal", "status": "failed",
                "code": "INSUFFICIENT_CREDITS", "error": "Insufficient credits",
                "billing": {"credits_charged": 0.02},
            }),
        ),
    ],
)
def test_charged_402_never_marks_credit_recovery(provider, response):
    current, store, _ = _broker(response, funding_source="miner_key")
    operation_id, parameters = _request(provider)
    result = current.execute(
        CONTEXT, operation_id=operation_id, parameters=parameters,
        action_sequence=7, timeout_ms=5000,
    )
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] > 0
    assert "credit_failure_proof" not in result.call
    assert "credit_failure_proof" not in store.calls[result.call["call_identity"]]["terminal"]


def test_deepline_402_without_exact_zero_remains_uncertain():
    current, store, _ = _broker(
        (402, {"code": "INSUFFICIENT_CREDITS", "billing": {"credits_charged": "unknown"}}),
        funding_source="miner_key",
    )
    result = current.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters=_request("deepline")[1],
        action_sequence=4, timeout_ms=5000,
    )
    assert result.call["outcome"] == "uncertain"
    assert "credit_failure_proof" not in result.call
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"


@pytest.mark.parametrize("status", [401, 403, 429])
@pytest.mark.parametrize("provider", ["openrouter", "deepline", "scrapingdog"])
def test_other_account_failures_never_mark_credit_recovery(provider, status):
    current, store, _ = _broker(
        (status, {"error": {"code": status}}), funding_source="miner_key",
    )
    operation_id, parameters = _request(provider)
    result = current.execute(
        CONTEXT, operation_id=operation_id, parameters=parameters,
        action_sequence=5, timeout_ms=5000,
    )
    assert "credit_failure_proof" not in result.call
    assert all(
        "credit_failure_proof" not in call.get("terminal", {})
        for call in store.calls.values()
    )


def test_transport_timeout_never_marks_credit_recovery():
    current, store, _ = make_broker(transport=FakeTransport(fail=True))
    result = current.execute(
        CONTEXT, operation_id="scrapingdog.scrape",
        parameters=_request("scrapingdog")[1],
        action_sequence=6, timeout_ms=5000,
    )
    assert result.call["outcome"] == "uncertain"
    assert "credit_failure_proof" not in result.call
    assert all("terminal" not in call for call in store.calls.values())
