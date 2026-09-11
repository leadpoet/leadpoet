"""Broker state machine, cost rules, and error genericness (labarena.md 7.3-7.5, 18.3, 18.4)."""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import quote

import pytest

from lab_arena import broker as br
from lab_arena import contracts, operations

KEY = "sk-or-v1-" + "k" * 40
DL_KEY = "dl_secret_" + "e" * 30
DOG_KEY = "dogsecret" + "d" * 30
HOST_KEYS = {"openrouter": KEY, "deepline": DL_KEY, "scrapingdog": DOG_KEY}
ECHO_KEY = "synthetic+/=query-key"
ENCODED_ECHO_KEY = quote(ECHO_KEY, safe="")
LOWERCASE_ENCODED_ECHO_KEY = ENCODED_ECHO_KEY.replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")


def price_table():
    return br.validate_price_table({
        "schema_version": br.PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": "2026-09-02T00:00:00Z",
        "source": br.OPENROUTER_MODELS_URL,
        "models": {
            "openai/gpt-4o-mini": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"},
            "anthropic/claude-3.5-haiku": {"prompt": "0.0000008", "completion": "0.000004", "request": "0.00001", "image": "0", "web_search": "0", "internal_reasoning": "0.000004"},
        },
    })


class FakeLedgerStore:
    """In-memory model of the section 7.5 ledger functions and their statuses."""

    def __init__(self, *, per_icp_quota=30, openrouter_capacity=10_000_000, budget_busy_responses=0):
        self.per_icp_quota = per_icp_quota
        self.openrouter_capacity = openrouter_capacity
        self.budget_busy_responses = budget_busy_responses
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.stale = False
        self.lock = threading.Lock()
        self.log: List[str] = []

    def _view(self, call):
        return {"status": {"reservation": "reserved", "dispatch": "dispatched", "settlement": "settled", "uncertain": "uncertain", "recovery": "recovered", "refusal": "refused"}[call["kind"]], "idempotent": True, "call_identity": call["identity"], "amount_microusd": call["amount"], "terminal_response": call.get("terminal"), "reason": call.get("reason")}

    def _consumed(self, provider):
        return sum(1 for c in self.calls.values() if c.get("provider") == provider and c["kind"] in ("reservation", "dispatch", "settlement", "uncertain"))

    def reserve_call(self, *, run_id, lease_token_hash, call_identity, operation_id, provider, funding_source, amount_microusd, call_doc, lease_ttl_seconds):
        with self.lock:
            self.log.append("reserve")
            if self.stale:
                return {"status": "stale"}
            existing = self.calls.get(call_identity)
            if existing:
                return self._view(existing)
            assert funding_source in ("host", "miner_key")
            reserve_remaining = call_doc.get("reserve_remaining_budget") is True
            if reserve_remaining and self.budget_busy_responses:
                self.budget_busy_responses -= 1
                return {"status": "budget_busy"}
            dynamic_inflight = any(
                call.get("reserve_remaining") and call["kind"] in ("reservation", "dispatch")
                for call in self.calls.values()
            )
            if reserve_remaining and dynamic_inflight:
                return {"status": "budget_busy"}
            if reserve_remaining:
                amount_microusd = max(0, self.openrouter_capacity)
            reason = None
            if self._consumed(provider) >= self.per_icp_quota:
                reason = "per_icp_quota"
            elif self.openrouter_capacity < amount_microusd:
                reason = "provider_cost_cap"
            if reason:
                self.calls[call_identity] = {"kind": "refusal", "identity": call_identity, "amount": 0, "reason": reason}
                return {"status": "refused", "idempotent": False, "reason": reason, "call_identity": call_identity}
            self.openrouter_capacity -= amount_microusd
            self.calls[call_identity] = {"kind": "reservation", "identity": call_identity, "amount": amount_microusd, "provider": provider}
            self.calls[call_identity]["funding_source"] = funding_source
            self.calls[call_identity]["reserve_remaining"] = reserve_remaining
            return {"status": "reserved", "idempotent": False, "call_identity": call_identity, "amount_microusd": amount_microusd}

    def mark_dispatched(self, *, run_id, lease_token_hash, call_identity):
        with self.lock:
            self.log.append("dispatch")
            if self.stale:
                return {"status": "stale"}
            call = self.calls.get(call_identity)
            if call is None:
                return {"status": "not_reserved"}
            if call["kind"] != "reservation":
                return self._view(call)
            call["kind"] = "dispatch"
            return {"status": "dispatched", "idempotent": False, "call_identity": call_identity, "amount_microusd": call["amount"]}

    def settle_call(self, *, run_id, lease_token_hash, call_identity, actual_microusd, terminal_response, lease_ttl_seconds):
        with self.lock:
            self.log.append("settle")
            if self.stale:
                return {"status": "stale"}
            call = self.calls[call_identity]
            if call["kind"] != "dispatch":
                return self._view(call)
            self.openrouter_capacity += call["amount"]  # outstanding released
            self.openrouter_capacity -= actual_microusd
            call.update({"kind": "settlement", "terminal": terminal_response, "actual": actual_microusd})
            return {"status": "settled", "idempotent": False, "actual_microusd": actual_microusd, "released_microusd": max(0, call["amount"] - actual_microusd), "terminal_response": terminal_response}

    def mark_uncertain(self, *, run_id, lease_token_hash, call_identity, call_doc, lease_ttl_seconds):
        with self.lock:
            self.log.append("uncertain")
            if self.stale:
                return {"status": "stale"}
            call = self.calls[call_identity]
            if call["kind"] != "dispatch":
                return self._view(call)
            call["kind"] = "uncertain"
            call["uncertain_doc"] = dict(call_doc)
            return {"status": "uncertain", "idempotent": False, "amount_microusd": call["amount"]}

class FakeTransport:
    def __init__(self, responses=None, *, fail=False):
        self.responses = list(responses or [])
        self.fail = fail
        self.sent: List[Dict[str, Any]] = []
        self.synthetic_deepline_history = None

    def send(self, *, method, url, headers, body, timeout_seconds):
        self.sent.append({"method": method, "url": url, "headers": dict(headers), "body": body, "timeout": timeout_seconds})
        if self.fail:
            raise br.ProviderTransportError("ReadTimeout")
        if (
            url == br.DEEPLINE_BILLING_HISTORY_URL
            and not self.responses
            and self.synthetic_deepline_history is not None
        ):
            status, payload = 200, self.synthetic_deepline_history
            self.synthetic_deepline_history = None
        else:
            status, payload = self.responses.pop(0) if self.responses else (200, {"data": []})
        if (
            "code.deepline.com" in url
            and url != br.DEEPLINE_BILLING_HISTORY_URL
            and 200 <= status < 300
            and isinstance(payload, dict)
        ):
            payload = dict(payload)
            payload.setdefault("billing", {"credits_charged": 0})
            payload.setdefault("job_id", "fake-job-%d" % len(self.sent))
            payload.setdefault("status", "completed")
            try:
                operation = json.loads(body)["operation"]
                credits = payload["billing"]["credits_charged"]
            except (KeyError, TypeError, ValueError):
                pass
            else:
                self.synthetic_deepline_history = {
                    "recent": {
                        "entries": [
                            {
                                "request_id": payload["job_id"],
                                "operation": operation,
                                "provider": "test_provider",
                                "credits": float(credits),
                                "charge_state": "posted",
                            }
                        ],
                        "has_more": False,
                        "next_cursor": None,
                    }
                }
        raw = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
        return br.ProviderResponse(status, {"content-type": "application/json", "x-ratelimit-remaining": "3", "set-cookie": "s=1"}, raw)


def make_broker(store=None, transport=None, **kwargs):
    store = store or FakeLedgerStore()
    transport = transport or FakeTransport()
    broker = br.Broker(
        store=store,
        key_for=lambda provider: HOST_KEYS[provider],
        price_table=price_table(),
        transport=transport,
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
        **kwargs,
    )
    return broker, store, transport


def deepline_history_entry(
    request_id,
    operation,
    credits,
    *,
    charge_state="posted",
    provider="exa",
):
    """Return the verified live billing-history entry shape."""

    return {
        "request_id": request_id,
        "operation": operation,
        "provider": provider,
        "credits": credits,
        "charge_state": charge_state,
        "status": "completed",
        "billing_mode": "metered",
        "outcome": "success",
        "delta": credits,
    }


def deepline_history(*entries, has_more=False, next_offset=None):
    return {
        "recent": {
            "entries": list(entries),
            "has_more": has_more,
            "next_offset": next_offset,
        }
    }


def test_default_http_transport_does_not_inherit_proxy_environment():
    transport = br.HttpxProviderTransport()
    try:
        assert transport._client._trust_env is False
    finally:
        transport.close()


CONTEXT = br.RunContext(run_id="r1", assignment_id="arena-2026-09-02:s1:1:0", icp_position=0, lease_token_hash=contracts.document_hash("lease"), miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", submission_id="s1", stage=1)
CHAT = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "find fintech companies"}], "max_tokens": 200}


def test_deepline_call_uses_the_host_key_and_settles():
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": [{"url": "https://a.example"}]})]))
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "fintech"}}, action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and json.loads(result.body)["results"][0]["url"] == "https://a.example"
    assert set(result.headers) == {"content-type", "content-length"}
    assert store.log == ["reserve", "dispatch", "settle"]
    call = result.call
    assert call["outcome"] == "settled" and call["reserved_microusd"] == 10_000_000
    assert call["actual_microusd"] == 0 and call["funding_source"] == "host"
    assert call["reservation_basis"] == "remaining_budget_dynamic_deepline"
    sent = transport.sent[0]
    assert sent["url"] == "https://code.deepline.com/api/v2/integrations/exa_search/execute" and sent["method"] == "POST"
    assert sent["headers"]["authorization"] == "Bearer " + DL_KEY and "x-api-key" not in sent["headers"]
    # The body and header pinned from Deepline's official client (test_lab_arena_deepline_contract).
    assert json.loads(sent["body"]) == {"provider": "exa", "operation": "exa_search", "payload": {"query": "fintech"}}
    assert sent["headers"]["x-deepline-execute-response-intent"] == "raw"
    assert store.calls[call["call_identity"]]["provider"] == "deepline"


def test_deepline_per_call_billing_settles_and_person_entities_are_dropped():
    envelope = {
        "job_id": "iad1::x", "status": "completed",
        "result": {"data": {"requestId": "r", "results": [{"id": "u", "url": "https://a.example", "text": "t", "entities": [{"type": "person", "properties": {"name": "Jane Roe"}}]}]}},
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, envelope)])
    )
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_contents", "payload": {"urls": ["https://a.example"]}}, action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and b"Jane Roe" not in result.body
    assert json.loads(result.body)["result"]["data"]["results"][0]["entities"] == []
    call = result.call
    assert call["reserved_microusd"] == 10_000_000 and call["actual_microusd"] == 2_000 and call["outcome"] == "settled"
    assert call["cost_basis"] == "deepline_billing_credits_charged_x_0.10_usd"
    assert store.calls[call["call_identity"]]["actual"] == 2_000
    terminal = store.calls[call["call_identity"]]["terminal"]
    assert terminal["provider_cost"] == {
        "basis": "deepline_billing_credits_charged_x_0.10_usd",
        "units": "0.02",
        "unit_name": "credits",
        "operation": "exa_contents",
        "request_id": "iad1::x",
    }
    assert terminal["provider_cost"]["units"] == str(envelope["billing"]["credits_charged"])
    replay = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_contents", "payload": {"urls": ["https://a.example"]}}, action_sequence=0, timeout_ms=5000)
    assert replay.status == 200 and replay.call["outcome"] == "settled"
    assert store.calls[call["call_identity"]]["terminal"]["provider_cost"] == terminal["provider_cost"]
    assert len(transport.sent) == 1
    assert json.loads(transport.sent[0]["body"])["operation"] == "exa_contents"


def test_scrapingdog_credential_goes_in_the_query_and_never_in_the_model_response():
    broker, store, transport = make_broker(transport=FakeTransport([(200, b"<html>hi</html>")]))
    result = broker.execute(CONTEXT, operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=5000)
    assert result.status == 200
    sent = transport.sent[0]
    assert "api_key=" + DOG_KEY in sent["url"] and sent["url"].startswith("https://api.scrapingdog.com/scrape?")
    assert "premium=" not in sent["url"]  # the judge's premium tiers are declared fields, no longer pinned off
    assert DOG_KEY not in result.body.decode() and DOG_KEY not in json.dumps(result.call)
    assert result.call["reserved_microusd"] == 250 and result.call["actual_microusd"] == 250
    assert result.call["cost_basis"] == "scrapingdog_legacy_endpoint_map"
    assert store.calls[result.call["call_identity"]]["provider"] == "scrapingdog"
    assert store.calls[result.call["call_identity"]]["terminal"]["provider_cost"] == {
        "basis": "scrapingdog_legacy_endpoint_map", "units": "5",
        "unit_name": "credits", "operation": "scrapingdog.scrape",
    }


@pytest.mark.parametrize(
    ("provider_body", "response_header"),
    (
        (("<html>request api_key=%s</html>" % ECHO_KEY).encode(), ""),
        (("<html>request api_key=%s</html>" % ENCODED_ECHO_KEY).encode(), ""),
        (("<html>request api_key=%s</html>" % LOWERCASE_ENCODED_ECHO_KEY).encode(), ""),
        (b'{"echo":"synthetic\\u002b\\u002f\\u003dquery-key"}', ""),
        (b'{"echo":"synthetic\\u00252B\\u00252F\\u00253Dquery-key"}', ""),
        (b"<html>ordinary provider content</html>", LOWERCASE_ENCODED_ECHO_KEY),
    ),
    ids=("literal", "url_encoded", "url_encoded_lowercase", "json_escaped", "json_escaped_url_encoded", "header"),
)
def test_scrapingdog_credential_echo_is_blocked_before_return_or_persistence(provider_body, response_header):

    class EchoTransport(FakeTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            if not response_header:
                return response
            return br.ProviderResponse(
                response.status,
                {**response.headers, "x-request-echo": response_header},
                response.body,
            )

    broker, store, transport = make_broker(
        transport=EchoTransport([(200, provider_body)]),
        credential_for=lambda _context, _provider: ECHO_KEY,
    )

    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert "api_key=" + ENCODED_ECHO_KEY in transport.sent[0]["url"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    persisted_body = base64.b64decode(terminal["body_b64"])
    assert ECHO_KEY.encode() not in persisted_body
    assert ENCODED_ECHO_KEY.encode() not in persisted_body
    assert not response_header or response_header not in json.dumps(terminal)


@pytest.mark.parametrize(
    "provider_body",
    (b"<html>ordinary provider content</html>", b'{"text":"\\ud800"}'),
    ids=("ordinary", "json_lone_surrogate"),
)
def test_benign_scrapingdog_text_is_returned_unchanged(provider_body):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, provider_body)]),
        credential_for=lambda _context, _provider: ECHO_KEY,
    )

    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 200 and result.body == provider_body
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert base64.b64decode(terminal["body_b64"]) == provider_body


def test_openrouter_reserves_maximum_cost_and_settles_reported_actual():
    usage = {"prompt_tokens": 20, "completion_tokens": 50, "cost": "0.000033"}
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"id": "gen", "model": "openai/gpt-4o-mini", "choices": [], "usage": usage})]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    expected_max = br.max_openrouter_cost_microusd(price_table(), "openai/gpt-4o-mini", dict(CHAT), max_output_tokens=200)
    expected_actual = br.actual_openrouter_cost_microusd(price_table(), "openai/gpt-4o-mini", {"usage": usage})
    assert result.call["reserved_microusd"] == expected_max and result.call["actual_microusd"] == expected_actual < expected_max
    sent = transport.sent[0]
    assert sent["headers"]["authorization"] == "Bearer " + KEY
    body = json.loads(sent["body"])
    assert body["provider"] == {"allow_fallbacks": False, "data_collection": "deny", "zdr": True} and body["stream"] is False
    assert store.openrouter_capacity == 10_000_000 - expected_actual


def test_openrouter_reported_model_alias_does_not_erase_actual_billing():
    payload = {
        "model": "openai/gpt-4o-mini-2024-07-18",
        "choices": [],
        "usage": {"cost": "0.00000345"},
    }
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    assert result.call["actual_microusd"] == 4
    assert store.calls[result.call["call_identity"]]["actual"] == 4


def test_openrouter_exact_actual_may_exceed_the_conservative_reservation():
    payload = {
        "model": "openai/gpt-4o-mini",
        "choices": [],
        "usage": {"cost": "1.25"},
    }
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    assert result.call["actual_microusd"] == 1_250_000
    assert result.call["actual_microusd"] > result.call["reserved_microusd"]
    assert store.calls[result.call["call_identity"]]["actual"] == 1_250_000


def test_openrouter_input_reservation_counts_utf8_bytes():
    ascii_cost = br.max_openrouter_cost_microusd(
        price_table(), "openai/gpt-4o-mini", {"messages": [{"role": "user", "content": "a" * 100}]}, max_output_tokens=1
    )
    unicode_cost = br.max_openrouter_cost_microusd(
        price_table(), "openai/gpt-4o-mini", {"messages": [{"role": "user", "content": "雪" * 100}]}, max_output_tokens=1
    )
    assert unicode_cost > ascii_cost


def test_openrouter_http_200_error_status_is_normalized_and_replayed_without_a_second_send():
    payload = {"error": {"code": 429, "message": "upstream temporarily rate-limited"}}
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )

    first = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    second = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)

    expected_max = br.max_openrouter_cost_microusd(price_table(), "openai/gpt-4o-mini", dict(CHAT), max_output_tokens=200)
    assert first.status == second.status == 502
    assert json.loads(first.body) == json.loads(second.body) == {"error": {"code": "provider_unavailable"}}
    assert first.call["provider_status"] == 429 and first.call["actual_microusd"] == 0
    assert second.call["idempotent"] is True
    assert store.openrouter_capacity == 10_000_000
    assert [sent["method"] for sent in transport.sent].count("POST") == 1
    assert store.log.count("settle") == 1
    assert "provider_cost" not in store.calls[first.call["call_identity"]]["terminal"]


@pytest.mark.parametrize("provider_status", [500, 503, 599])
def test_openrouter_http_200_server_error_envelope_is_infrastructure(provider_status):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, {"error": {"code": provider_status, "message": "upstream failure"}})])
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["provider_status"] == provider_status
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize("provider_status", [401, 402, 403])
def test_openrouter_http_200_credential_error_envelope_uses_miner_key_error(provider_status):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, {"error": {"code": provider_status, "message": "credential rejected"}})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 402 and json.loads(result.body) == {"error": {"code": "miner_credentials_unavailable"}}
    assert result.call["provider_status"] == provider_status
    assert result.call["actual_microusd"] == 0 < result.call["reserved_microusd"]
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("provider_status", [400, 404])
def test_openrouter_http_200_caller_error_envelope_uses_effective_status(provider_status):
    payload = {"error": {"code": provider_status, "message": "request rejected"}}
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == provider_status and json.loads(result.body) == payload
    assert result.call["provider_status"] == provider_status
    assert result.call["actual_microusd"] == 0 < result.call["reserved_microusd"]
    assert store.log == ["reserve", "dispatch", "settle"]


def test_openrouter_timeout_without_billing_is_uncertain():
    payload = {"error": {"code": 408, "message": "request timeout"}}
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and result.call["provider_status"] == 408
    assert result.call["outcome"] == "uncertain"
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_error_with_known_usage_settles_the_exact_charge():
    payload = {"error": {"code": 400, "message": "request rejected"}, "usage": {"cost": "0.0000091"}}
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 400
    assert result.call["actual_microusd"] == 10
    assert store.calls[result.call["call_identity"]]["actual"] == 10


def test_openrouter_http_200_error_finished_choice_is_normalized():
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": "partial output"},
            "finish_reason": "error",
            "error": {"code": 502, "message": "provider disconnected"},
        }]
    }
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["provider_status"] == 502
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_http_200_top_level_error_applies_to_error_finished_choice_without_nested_error():
    payload = {
        "error": {"code": 429, "message": "upstream temporarily rate-limited"},
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "error"}],
    }
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["provider_status"] == 429
    assert result.call["actual_microusd"] == 0 < result.call["reserved_microusd"]
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("payload", [
    {"error": {"message": "missing status"}},
    {"error": {"code": "429", "message": "string status"}},
    {"error": {"code": 399, "message": "non-error status"}},
    {"choices": [{"finish_reason": "error", "error": {"code": 600}}]},
    {"error": {"code": 429}, "choices": [{"finish_reason": "error", "error": {"code": 502}}]},
])
def test_openrouter_http_200_malformed_error_envelope_fails_closed(payload):
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["outcome"] == "uncertain"
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_http_200_normal_completion_is_unchanged():
    payload = {
        "id": "gen",
        "model": "openai/gpt-4o-mini",
        "error": None,
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": "0.0000021"},
    }
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 200 and json.loads(result.body) == payload
    assert result.call["actual_microusd"] < result.call["reserved_microusd"]
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("payload", [
    {"choices": []},
    {"usage": {"prompt_tokens": "x", "completion_tokens": 1}},
    {"usage": {"prompt_tokens": -1, "completion_tokens": 1}},
    {"model": "other/model", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    {"usage": {"prompt_tokens": 10 ** 9, "completion_tokens": 10 ** 9}},
])
def test_missing_malformed_or_wrong_model_usage_marks_the_call_uncertain(payload):
    broker, store, transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "provider_status": 200,
        "body_bytes": len(json.dumps(payload).encode("utf-8")),
        "body_is_mapping": True,
        "usage_present": "usage" in payload,
        "billing_present": False,
    }


def test_any_priced_model_is_allowed_and_an_unpriced_model_is_refused():
    response = {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": "0.000001"}}
    broker, store, transport = make_broker(transport=FakeTransport([(200, response), (200, response)]))
    huge = dict(CHAT, max_tokens=4096)
    broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=huge, action_sequence=0, timeout_ms=1000)
    assert json.loads(transport.sent[0]["body"])["max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS
    priced = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=dict(CHAT, model="anthropic/claude-3.5-haiku"), action_sequence=1, timeout_ms=1000)
    assert priced.status == 200 and len(transport.sent) == 2
    other = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=dict(CHAT, model="openai/gpt-5"), action_sequence=2, timeout_ms=1000)
    assert other.status == 400 and json.loads(other.body) == {"error": {"code": "model_not_allowed"}}
    assert store.log.count("reserve") == 2
    unknown = broker.execute(CONTEXT, operation_id="deepline.play", parameters={}, action_sequence=3, timeout_ms=1000)
    assert unknown.status == 400


def test_budget_refusal_is_generic_and_recorded_under_the_identity():
    broker, store, transport = make_broker(store=FakeLedgerStore(per_icp_quota=0))
    refused = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert refused.status == 402 and json.loads(refused.body) == {"error": {"code": "budget_refused"}}
    assert refused.call["outcome"] == "refused" and refused.call["reason"] == "per_icp_quota"
    assert transport.sent == []
    again = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert again.status == 402 and store.log == ["reserve", "reserve"]


def test_transport_failure_after_send_marks_uncertain_and_keeps_full_reservation():
    broker, store, transport = make_broker(transport=FakeTransport(fail=True))
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert result.call["outcome"] == "uncertain" and result.call["actual_microusd"] == 10_000_000
    assert store.openrouter_capacity == 0
    # A later identical request neither re-sends nor releases the reservation.
    late = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert late.status == 409 and json.loads(late.body) == {"error": {"code": "call_uncertain"}} and len(transport.sent) == 1


def test_successful_deepline_reply_without_billing_is_uncertain():
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, b'{"results":[]}')])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "provider_status": 200,
        "body_bytes": len(b'{"results":[]}'),
        "body_is_mapping": True,
        "usage_present": False,
        "billing_present": False,
    }


@pytest.mark.parametrize("status", [{"message": "not a status"}, ["completed"], "secret-shaped-unrecognized-status"])
def test_missing_cost_diagnostics_never_store_arbitrary_job_status(status):
    body = json.dumps({"status": status}).encode()
    diagnostic = br._missing_provider_cost_call_doc(
        br.ProviderResponse(200, {}, body), {"status": status}
    )
    assert diagnostic["provider_status"] == 200
    assert "top_level_job_status" not in diagnostic
    assert "not a status" not in json.dumps(diagnostic)
    assert "secret-shaped" not in json.dumps(diagnostic)


def test_real_deepline_free_company_search_without_billing_settles_known_zero():
    envelope = {
        "job_id": "iad1::free-company-search",
        "result": {"data": [{"domain": "example.com", "name": "Example"}]},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "free_simple_company_search",
            "payload": {"sql": "SELECT * FROM companies LIMIT 1"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and json.loads(result.body) == envelope
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_free_simple_company_search_completed_zero"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


def test_real_deepline_hunter_discover_without_billing_settles_verified_zero():
    envelope = {
        "job_id": "iad1::hunter-discover",
        "result": {"data": [{"domain": "example.com"}]},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "hunter_discover", "payload": {"query": "fintech"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and json.loads(result.body) == envelope
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_hunter_discover_completed_zero"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


def test_real_deepline_free_company_search_keeps_reported_billing():
    envelope = {
        "job_id": "iad1::free-company-search",
        "result": {"data": []},
        "status": "completed",
        "billing": {"credits_charged": "0.02"},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "free_simple_company_search", "payload": {"sql": "SELECT 1"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 2_000
    assert result.call["cost_basis"] == "deepline_billing_credits_charged_x_0.10_usd"
    assert store.calls[result.call["call_identity"]]["actual"] == 2_000
    assert len(transport.sent) == 1


def test_real_deepline_free_company_search_malformed_billing_does_not_fall_back():
    envelope = {
        "job_id": "iad1::free-company-search",
        "result": {"data": []},
        "status": "completed",
        "billing": {"credits_charged": "not-a-number"},
    }
    raw = json.dumps(envelope).encode("utf-8")
    broker, store, _transport = make_broker(transport=FakeTransport([(200, raw)]))
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "free_simple_company_search", "payload": {"sql": "SELECT 1"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "provider_status": 200,
        "body_bytes": len(raw),
        "body_is_mapping": True,
        "usage_present": False,
        "billing_present": True,
        "top_level_job_status": "completed",
        "deepline_job_id": "iad1::free-company-search",
        "deepline_operation": "free_simple_company_search",
    }


def test_real_deepline_paid_fixed_tool_recovers_zero_from_billing_history():
    envelope = {
        "job_id": "iad1::paid-harvest-job",
        "result": {"data": {"id": "job-1"}},
        "status": "completed",
    }
    raw = json.dumps(envelope).encode("utf-8")
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (200, raw),
                (
                    200,
                    deepline_history(
                        deepline_history_entry(
                            "iad1::paid-harvest-job",
                            "harvestapi_get_job",
                            0,
                            provider="harvestapi",
                        )
                    ),
                ),
            ]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "harvestapi_get_job",
            "payload": {"url": "https://www.linkedin.com/jobs/view/1"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == 1_000
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_billing_history_credits_x_0.10_usd"
    assert store.calls[result.call["call_identity"]]["actual"] == 0
    assert len(transport.sent) == 2
    assert 0 < transport.sent[1]["timeout"] <= 1.0


def test_deepline_missing_native_billing_recovers_full_positive_history_charge():
    job_id = "iad1::positive-history"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (200, json.dumps(envelope).encode()),
                (
                    200,
                    deepline_history(
                        deepline_history_entry(job_id, "exa_search", 0.14)
                    ),
                ),
            ]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 14_000
    assert result.call["cost_basis"] == "deepline_billing_history_credits_x_0.10_usd"
    assert store.calls[result.call["call_identity"]]["actual"] == 14_000
    assert 0 < transport.sent[1]["timeout"] <= 30.0


def test_deepline_completed_native_billing_is_scoped_to_one_grouped_request():
    wrapper_job_id = "iad1::wrapper-job"
    envelope = {
        "job_id": wrapper_job_id,
        "result": {"data": []},
        "status": "completed",
        # The per-call charge is smaller than the shared history group total.
        "billing": {"credits_charged": 0.56, "cost_usd": 0.056},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode())])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "predictleads_company_financing_events",
            "payload": {"company_id_or_domain": "example.com"},
        },
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == 56_000
    assert result.call["actual_microusd"] == 56_000
    assert result.call["cost_basis"] == "deepline_billing_credits_charged_x_0.10_usd"
    assert store.calls[result.call["call_identity"]]["actual"] == 56_000
    assert store.calls[result.call["call_identity"]]["terminal"]["provider_cost"] == {
        "basis": "deepline_billing_credits_charged_x_0.10_usd",
        "units": "0.56",
        "unit_name": "credits",
        "operation": "predictleads_company_financing_events",
        "request_id": wrapper_job_id,
    }
    assert len(transport.sent) == 1


@pytest.mark.parametrize(
    ("provider_status", "envelope"),
    [
        (
            200,
            {
                "status": "completed",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
        ),
        (
            200,
            {
                "job_id": "iad1::pending-job",
                "status": "pending",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
        ),
        (
            201,
            {
                "job_id": "iad1::created-job",
                "status": "completed",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
        ),
    ],
)
def test_deepline_native_billing_requires_http_200_completed_job(
    provider_status, envelope
):
    broker, store, _transport = make_broker(
        transport=FakeTransport(
            [(provider_status, json.dumps(envelope).encode("utf-8"))]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "predictleads_company_financing_events",
            "payload": {"company_id_or_domain": "example.com"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"


def test_deepline_shared_history_cost_never_settles_one_request():
    wrapper_job_id = "iad1::wrapper-job"
    internal_job_id = "iad1::internal-job"
    envelope = {
        "job_id": wrapper_job_id,
        "result": {"data": []},
        "status": "completed",
    }
    history_entry = {
        **deepline_history_entry(
            internal_job_id,
            "predictleads_company_financing_events",
            1.12,
            provider="predictleads",
        ),
        "metadata": {"chargeGroupIds": [internal_job_id, wrapper_job_id]},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (200, json.dumps(envelope).encode()),
                (200, deepline_history(history_entry)),
            ]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "predictleads_company_financing_events",
            "payload": {"company_id_or_domain": "example.com"},
        },
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert len(transport.sent) == 2


def test_deepline_billing_history_follows_forward_offset_without_poll_delay(monkeypatch):
    monkeypatch.setattr(
        br.time,
        "sleep",
        lambda _seconds: pytest.fail("cursor pagination must not wait for a poll"),
    )
    job_id = "iad1::second-page"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
    }
    next_offset = 50
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (200, json.dumps(envelope).encode()),
                (
                    200,
                    deepline_history(
                        deepline_history_entry("unrelated", "exa_search", 0),
                        has_more=True,
                        next_offset=next_offset,
                    ),
                ),
                (
                    200,
                    deepline_history(
                        deepline_history_entry(job_id, "exa_search", 0.14)
                    ),
                ),
            ]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["actual_microusd"] == 14_000
    assert transport.sent[2]["url"] == (
        br.DEEPLINE_BILLING_HISTORY_URL
        + "&recent_offset=50"
    )
    assert store.calls[result.call["call_identity"]]["actual"] == 14_000


def test_deepline_billing_history_keeps_polling_after_old_four_read_window(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    job_id = "iad1::lagged-history"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
    }

    class DelayedFullHistory(FakeTransport):
        def send(self, **kwargs):
            if kwargs["url"].startswith(br.DEEPLINE_BILLING_HISTORY_URL):
                self.sent.append({
                    "method": kwargs["method"], "url": kwargs["url"],
                    "headers": dict(kwargs["headers"]), "body": kwargs["body"],
                    "timeout": kwargs["timeout_seconds"],
                })
                if kwargs["url"] == br.DEEPLINE_BILLING_HISTORY_URL and elapsed[0] >= 6.0:
                    history = deepline_history(
                        deepline_history_entry(job_id, "exa_search", 0.03),
                        has_more=True, next_offset=50,
                    )
                else:
                    offset = int(kwargs["url"].rsplit("=", 1)[-1]) if "recent_offset=" in kwargs["url"] else 0
                    history = deepline_history(
                        deepline_history_entry("older", "exa_search", 0.01),
                        has_more=True, next_offset=offset + 50,
                    )
                return br.ProviderResponse(200, {}, json.dumps(history).encode())
            return super().send(**kwargs)

    broker, store, transport = make_broker(
        transport=DelayedFullHistory([(200, json.dumps(envelope).encode())])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["actual_microusd"] == 3_000
    assert elapsed[0] == 6.0
    assert [sent["url"] for sent in transport.sent[1:]].count(
        br.DEEPLINE_BILLING_HISTORY_URL
    ) == 4
    assert store.calls[result.call["call_identity"]]["actual"] == 3_000


def test_deepline_billing_history_stops_on_equal_nonadvancing_offset():
    job_id = "iad1::missing-after-equal-offset"
    envelope = {"job_id": job_id, "result": {"data": []}, "status": "completed"}
    broker, store, transport = make_broker(transport=FakeTransport([
        (200, json.dumps(envelope).encode()),
        (200, deepline_history(has_more=True, next_offset=50)),
        (200, deepline_history(has_more=True, next_offset=50)),
    ]))
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert [request["url"] for request in transport.sent[1:]] == [
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=50",
    ]
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_deepline_billing_history_scans_deeper_than_old_offset_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(br.time, "sleep", sleeps.append)
    job_id = "iad1::new-charge-arrived"
    envelope = {"job_id": job_id, "result": {"data": []}, "status": "completed"}
    older_pages = [
        (200, deepline_history(has_more=True, next_offset=50)),
        (200, deepline_history(has_more=True, next_offset=100)),
        (200, deepline_history(has_more=True, next_offset=50)),
        (200, deepline_history(has_more=True, next_offset=150)),
        (200, deepline_history(has_more=True, next_offset=50)),
    ]
    broker, store, transport = make_broker(transport=FakeTransport([
        (200, json.dumps(envelope).encode()),
        *older_pages,
        (200, deepline_history(
            deepline_history_entry(job_id, "exa_search", 0.03),
            has_more=True, next_offset=50,
        )),
    ]))
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 200
    assert result.call["actual_microusd"] == 3_000
    assert store.log == ["reserve", "dispatch", "settle"]
    assert [request["url"] for request in transport.sent[1:]] == [
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=50",
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=100",
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=150",
    ]
    assert sleeps == [2.0, 2.0]


def test_deepline_nonterminal_older_match_resets_offset_before_newest_refresh(monkeypatch):
    sleeps = []
    monkeypatch.setattr(br.time, "sleep", sleeps.append)
    job_id = "iad1::becomes-terminal-on-newest"
    nonterminal = deepline_history_entry(job_id, "exa_search", 0.03, charge_state="temporary_hold")
    envelope = {"job_id": job_id, "result": {"data": []}, "status": "completed"}
    broker, store, transport = make_broker(transport=FakeTransport([
        (200, json.dumps(envelope).encode()),
        (200, deepline_history(has_more=True, next_offset=50)),
        (200, deepline_history(nonterminal, has_more=True, next_offset=100)),
        (200, deepline_history(has_more=True, next_offset=50)),
        (200, deepline_history(deepline_history_entry(job_id, "exa_search", 0.03))),
    ]))
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["actual_microusd"] == 3_000
    assert [request["url"] for request in transport.sent[1:]] == [
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=50",
        br.DEEPLINE_BILLING_HISTORY_URL,
        br.DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=50",
    ]
    assert sleeps == [2.0]
    assert store.log == ["reserve", "dispatch", "settle"]


def test_deepline_billing_history_accepts_exact_match_without_exhausting_pages():
    job_id = "iad1::match-on-full-account"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (200, json.dumps(envelope).encode()),
                (
                    200,
                    deepline_history(
                        deepline_history_entry(job_id, "exa_search", 0.14),
                        has_more=True,
                        next_offset=50,
                    ),
                ),
            ]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 200 and result.call["actual_microusd"] == 14_000
    assert len(transport.sent) == 2
    assert store.calls[result.call["call_identity"]]["actual"] == 14_000


@pytest.mark.parametrize("request_seconds", [2.0, 20.0, 45.0])
def test_deepline_final_history_refresh_keeps_original_time_bound(monkeypatch, request_seconds):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))

    class SlowHistory:
        def __init__(self):
            self.timeouts = []

        def send(self, **kwargs):
            timeout = kwargs["timeout_seconds"]
            self.timeouts.append(timeout)
            elapsed[0] += min(1.5, timeout)
            next_offset = len(self.timeouts) * 50
            return br.ProviderResponse(200, {}, json.dumps(deepline_history(
                deepline_history_entry("older", "exa_search", 0.01),
                has_more=True, next_offset=next_offset,
            )).encode())

    transport = SlowHistory()
    cost = br._deepline_billing_readback(
        transport=transport, secret="test-key", request_id="new-job",
        operation="exa_search", request_deadline=request_seconds,
    )
    assert cost is None
    assert elapsed[0] == min(request_seconds, 30.0)
    assert 1 <= len(transport.timeouts) <= br._DEEPLINE_BILLING_MAX_ATTEMPTS
    assert all(0 < value <= min(request_seconds, 30.0) for value in transport.timeouts)


@pytest.mark.parametrize(
    "entries",
    [
        [
            deepline_history_entry("iad1::invalid-history", "exa_search", 0.1),
            deepline_history_entry("iad1::invalid-history", "exa_search", 0.1),
        ],
        [deepline_history_entry("iad1::invalid-history", "exa_contents", 0.1)],
        [
            {
                **deepline_history_entry(
                    "iad1::invalid-history", "exa_search", 0.1
                ),
                "provider": {"secret-shaped": "do-not-store-this"},
            }
        ],
        [deepline_history_entry("iad1::invalid-history", "exa_search", True)],
        [deepline_history_entry("iad1::invalid-history", "exa_search", float("inf"))],
        [
            deepline_history_entry(
                "iad1::invalid-history", "exa_search", 0.1, charge_state="free"
            )
        ],
        [
            *[
                deepline_history_entry("unrelated-%d" % index, "exa_search", 0)
                for index in range(51)
            ]
        ],
    ],
    ids=(
        "duplicate",
        "wrong_operation",
        "malformed_provider",
        "boolean_credits",
        "non_finite_credits",
        "nonzero_free_charge",
        "more_than_recent_limit",
    ),
)
def test_deepline_invalid_or_conflicting_history_fails_closed(entries):
    job_id = "iad1::invalid-history"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
        "billing": {"credits_charged": "not-a-number"},
    }
    sentinel = "history-private-content-do-not-store"
    history = deepline_history(*entries)
    history["untrusted"] = sentinel
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode()), (200, history)])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert len(transport.sent) == 2
    persisted = json.dumps(store.calls[result.call["call_identity"]])
    assert sentinel not in persisted
    assert "do-not-store-this" not in persisted
    assert sentinel.encode() not in result.body


def test_deepline_pending_history_exhausts_bounded_reads_then_fails_closed(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda seconds: None)
    job_id = "iad1::pending-history"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
        "billing": {"credits_charged": "not-a-number"},
    }
    pending = deepline_history(
        deepline_history_entry(
            job_id, "exa_search", 0, charge_state="pending"
        )
    )
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(200, json.dumps(envelope).encode())]
            + [(200, pending)] * br._DEEPLINE_BILLING_MAX_ATTEMPTS
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert len(transport.sent) == 1 + br._DEEPLINE_BILLING_MAX_ATTEMPTS
    assert all(
        sent["url"] == br.DEEPLINE_BILLING_HISTORY_URL
        for sent in transport.sent[1:]
    )
    assert "actual" not in store.calls[result.call["call_identity"]]


def test_deepline_billing_history_transport_timeout_polls_then_fails_closed(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda seconds: None)
    job_id = "iad1::history-timeout"

    class HistoryTimeoutTransport(FakeTransport):
        def send(self, **kwargs):
            if kwargs["url"] == br.DEEPLINE_BILLING_HISTORY_URL:
                self.sent.append(
                    {
                        "method": kwargs["method"],
                        "url": kwargs["url"],
                        "headers": dict(kwargs["headers"]),
                        "body": kwargs["body"],
                        "timeout": kwargs["timeout_seconds"],
                    }
                )
                raise br.ProviderTransportError("ReadTimeout")
            return super().send(**kwargs)

    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
        "billing": {"credits_charged": "not-a-number"},
    }
    broker, store, transport = make_broker(
        transport=HistoryTimeoutTransport([(200, json.dumps(envelope).encode())])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert len(transport.sent) == 1 + br._DEEPLINE_BILLING_MAX_ATTEMPTS
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert diagnostic["reason"] == "missing_provider_cost"
    assert diagnostic["deepline_job_id"] == job_id
    assert diagnostic["deepline_operation"] == "exa_search"


def test_deepline_credential_in_job_id_is_not_persisted():
    envelope = {
        "job_id": "iad1::" + DL_KEY,
        "result": {"data": {"results": []}},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, envelope)])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    persisted = store.calls[result.call["call_identity"]]
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert "deepline_job_id" not in persisted["uncertain_doc"]
    assert DL_KEY not in json.dumps(persisted)
    assert len(transport.sent) == 1


def test_deepline_billing_history_credential_echo_is_never_exposed_or_persisted():
    job_id = "iad1::history-key-echo"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
        "billing": {"credits_charged": "not-a-number"},
    }
    echo = json.dumps({"recent": {"entries": []}, "echo": DL_KEY}).encode()
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode()), (200, echo)])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert DL_KEY not in json.dumps(result.call)
    assert DL_KEY not in json.dumps(store.calls[result.call["call_identity"]])
    assert store.calls[result.call["call_identity"]].get("terminal") is None
    assert len(transport.sent) == 2


def test_dynamic_deepline_uses_the_store_authoritative_remaining_budget():
    store = FakeLedgerStore(openrouter_capacity=12_345)
    response = {"results": [], "billing": {"credits_charged": "0.01"}}
    broker, store, _transport = make_broker(store=store, transport=FakeTransport([(200, response)]))
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert result.status == 200 and result.call["reserved_microusd"] == 12_345
    assert result.call["actual_microusd"] == 1_000
    assert result.call["reservation_basis"] == "remaining_budget_dynamic_deepline"
    assert store.openrouter_capacity == 11_345


def test_unknown_dynamic_deepline_billing_holds_the_authoritative_reservation():
    store = FakeLedgerStore(openrouter_capacity=54_321)
    broker, store, _transport = make_broker(
        store=store, transport=FakeTransport([(200, b'{"results":[]}')])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 54_321
    assert store.openrouter_capacity == 0


def test_dynamic_deepline_retries_transient_budget_busy(monkeypatch):
    store = FakeLedgerStore(openrouter_capacity=8_765, budget_busy_responses=1)
    monkeypatch.setattr(br.time, "sleep", lambda _seconds: None)
    response = {"results": [], "billing": {"credits_charged": 0}}
    broker, store, transport = make_broker(store=store, transport=FakeTransport([(200, response)]))
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert result.status == 200 and result.call["reserved_microusd"] == 8_765
    assert store.log[:2] == ["reserve", "reserve"]
    assert [sent["method"] for sent in transport.sent] == ["POST"]


def test_dynamic_deepline_budget_busy_stops_at_the_reserve_deadline(monkeypatch):
    store = FakeLedgerStore(openrouter_capacity=8_765, budget_busy_responses=1000)
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    broker, store, transport = make_broker(store=store)
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=2000,
    )
    assert result.status == 502 and result.call["outcome"] == "not_dispatched"
    assert result.call["reason"] == "budget_busy" and transport.sent == []
    assert store.calls == {} and store.openrouter_capacity == 8_765
    assert elapsed[0] == pytest.approx(operations.BUDGET_ADMISSION_MAX_SECONDS)


def test_dynamic_deepline_admission_wait_does_not_consume_operation_window(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    store = FakeLedgerStore(openrouter_capacity=8_765, budget_busy_responses=65)
    job_id = "iad1::queued-window"

    class SlowExecutionThenLaggedHistory(FakeTransport):
        def send(self, **kwargs):
            if kwargs["method"] == "POST":
                elapsed[0] += 20.0
            return super().send(**kwargs)

    broker, store, transport = make_broker(
        store=store,
        transport=SlowExecutionThenLaggedHistory([
            (200, {"job_id": job_id, "status": "completed", "result": {"data": []}, "billing": None}),
            (200, deepline_history()),
            (200, deepline_history()),
            (200, deepline_history()),
            (200, deepline_history(
                deepline_history_entry(job_id, "exa_search", 0.1)
            )),
        ]),
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert result.status == 200 and result.call["actual_microusd"] == 10_000
    assert store.log.count("reserve") == 66
    assert store.log[65:67] == ["reserve", "dispatch"]
    assert elapsed[0] == pytest.approx(39.0)
    assert transport.sent[0]["method"] == "POST"
    assert transport.sent[0]["timeout"] == pytest.approx(60.0)
    assert transport.sent[1]["timeout"] == pytest.approx(30.0)
    assert transport.sent[-1]["timeout"] == pytest.approx(24.0)


def test_fault_injection_points_produce_single_accounting_results():
    # After reservation (crash before dispatch): resuming the identity dispatches exactly once.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    identity_args = dict(operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "crash"}}, action_sequence=5, timeout_ms=1000)
    request_hash = contracts.document_hash(operations.validate_operation_request("deepline.execute", {"tool": "exa_search", "payload": {"query": "crash"}}))
    identity = contracts.provider_call_identity(attempt=1, assignment_id=CONTEXT.assignment_id, icp_position=0, action_sequence=5, operation_id="deepline.execute", request_hash=request_hash)
    store.reserve_call(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="host", amount_microusd=0, call_doc={}, lease_ttl_seconds=420)
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 200
    assert [sent["method"] for sent in transport.sent].count("POST") == 1
    assert [c for c in store.calls.values() if c["kind"] == "settlement"]
    # After the dispatch marker (crash before send): the repeat never sends and reports uncertain.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    store.reserve_call(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="host", amount_microusd=0, call_doc={}, lease_ttl_seconds=420)
    store.mark_dispatched(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity)
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 409 and transport.sent == []
    # After settlement / HTTP response loss: the repeat returns the stored terminal response without a send.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": [1]})]))
    first = broker.execute(CONTEXT, **identity_args)
    second = broker.execute(CONTEXT, **identity_args)
    assert second.status == first.status and second.body == first.body and second.call["idempotent"] is True
    assert [sent["method"] for sent in transport.sent].count("POST") == 1
    assert store.log.count("settle") == 1
    # Stage closed between reservation and dispatch: the marker fails and nothing is sent.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    original = store.mark_dispatched

    def closing(**kwargs):
        store.stale = True
        return original(**kwargs)

    store.mark_dispatched = closing
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 409 and json.loads(result.body) == {"error": {"code": "lease_stale"}} and transport.sent == []


def test_two_broker_instances_cause_at_most_one_dispatch_per_identity():
    store = FakeLedgerStore()
    responses = [(200, {"results": ["a"]}), (200, {"results": ["b"]})]
    transports = [FakeTransport([responses[0]]), FakeTransport([responses[1]])]
    brokers = [make_broker(store=store, transport=transports[i])[0] for i in range(2)]
    results = []
    barrier = threading.Barrier(2)

    def worker(index):
        barrier.wait(timeout=10)
        results.append(brokers[index].execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "race"}}, action_sequence=9, timeout_ms=1000))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    sends = sum(
        sent["method"] == "POST"
        for transport in transports
        for sent in transport.sent
    )
    assert sends == 1
    settled = [r for r in results if r.status == 200]
    assert len(settled) >= 1
    assert [c for c in store.calls.values() if c["kind"] == "settlement"]


def test_errors_and_responses_carry_no_provider_account_or_credential_detail():
    broker, store, transport = make_broker(transport=FakeTransport([(401, {"error": {"message": "invalid api key " + DL_KEY}})]))
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert DL_KEY not in result.body.decode() and "x-ratelimit-remaining" not in result.headers and "set-cookie" not in result.headers
    document = result.to_document()
    assert set(document) == {"status", "headers", "body_b64", "call"}
    for value in (json.dumps(document), repr(broker)):
        assert DL_KEY not in value and DOG_KEY not in value and KEY not in value


def test_price_table_parsing_and_validation():
    response = {"data": [
        {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"}},
        {"id": "openai/catalog/id/that-is-not-an-arena-model-id", "pricing": {"prompt": "0.1", "completion": "0.1"}},
        {"id": "other/model", "pricing": {"prompt": "1"}},
    ]}
    table = br.price_table_from_models_response(response, fetched_at="2026-09-02T00:00:00Z")
    assert table["models"]["openai/gpt-4o-mini"]["completion"] == "0.0000006"
    assert "openai/catalog/id/that-is-not-an-arena-model-id" not in table["models"]
    assert "other/model" not in table["models"]
    with pytest.raises(contracts.ArenaContractError):
        br.price_table_from_models_response(response, ["missing/model"], fetched_at="2026-09-02T00:00:00Z")
    with pytest.raises(contracts.ArenaContractError):
        br.validate_price_table(dict(table, models={}))
    with pytest.raises(contracts.ArenaContractError):
        br.validate_price_table(dict(table, extra="x"))
    with pytest.raises(contracts.ArenaContractError):
        br.Broker(store=FakeLedgerStore(), key_for=lambda provider: HOST_KEYS[provider], price_table=table, judge_models=["anthropic/claude-3.5-haiku"], transport=FakeTransport())
    cost = br.max_openrouter_cost_microusd(price_table(), "anthropic/claude-3.5-haiku", {"messages": [{"role": "user", "content": "hi"}]}, max_output_tokens=100)
    assert cost > 0
    parsed = br.parse_broker_document({"status": 200, "headers": {"content-type": "application/json"}, "body_b64": base64.b64encode(b"{}").decode(), "call": {"a": 1}})
    assert parsed.body == b"{}" and parsed.call == {"a": 1}
    with_url = br.parse_broker_document({
        "status": 200,
        "headers": {
            "content-type": "text/html",
            operations.TRUSTED_RESPONSE_URL_HEADER: "https://www.example.com/final",
        },
        "body_b64": base64.b64encode(b"<html></html>").decode(),
        "call": {"a": 1},
    })
    assert with_url.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == "https://www.example.com/final"
    assert set(with_url.to_document()) == {"status", "headers", "body_b64", "call"}
    for invalid in (
        {**with_url.to_document(), "response_url": "https://www.example.com/final"},
        {**with_url.to_document(), "unexpected": "https://attacker.example/"},
    ):
        with pytest.raises(contracts.ArenaContractError):
            br.parse_broker_document(invalid)


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 503])
def test_host_account_or_provider_failure_is_infrastructure_for_scoring_and_execution(status):

    scoring_context = br.RunContext(**{**CONTEXT.__dict__, "kind": "score"})
    broker, store, transport = make_broker(transport=FakeTransport([(status, {"error": {"message": "invalid api key " + DL_KEY}})]))
    result = broker.execute(scoring_context, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "acme"}}, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["error_code"] == "provider_unavailable" and result.call["outcome"] == "uncertain"
    assert store.openrouter_capacity == 0
    assert DL_KEY.encode() not in result.body
    broker, store, transport = make_broker(transport=FakeTransport([(status, {"error": {"message": "invalid api key"}})]))
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "acme"}}, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable" and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    if status in (401, 402, 403, 429):
        assert result.call["outcome"] == "settled" and result.call["actual_microusd"] == 0
        assert store.openrouter_capacity == 10_000_000
    else:
        assert result.call["outcome"] == "uncertain"
        assert store.openrouter_capacity == 0


def test_true_caller_400_remains_visible_to_the_bundle():
    broker, _store, _transport = make_broker(transport=FakeTransport([(400, {"error": {"message": "invalid request"}})]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 400 and result.call.get("error_code") is None


def test_a_reply_the_sanitizer_refuses_after_dispatch_settles_as_uncertain_not_dispatched_forever():
    """A non-JSON reply on a JSON operation (a provider's HTML error page) must not strand the call."""

    broker, store, transport = make_broker(transport=FakeTransport([(200, b"<html>Cloudflare error</html>")]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["outcome"] == "uncertain" and result.call["error_code"] == "provider_unavailable"
    assert store.log[-1] == "uncertain" and "settle" not in store.log[-1:]  # the reservation is consumed, the head is terminal
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "settle_failure",
        "failure_stage": "response_adaptation",
        "error_class": "OperationResponseError",
    }
    assert b"Cloudflare" not in result.body


def test_a_store_that_rejects_the_settlement_leaves_the_call_uncertain():
    """If the ledger refuses the terminal response, the reservation is still consumed rather than left dispatched."""

    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    original = store.settle_call

    def refusing_settle(**kwargs):
        raise br.ArenaContractError("terminal response rejected")

    store.settle_call = refusing_settle
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "acme"}}, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and result.call["outcome"] == "uncertain" and store.log[-1] == "uncertain"
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "settle_failure",
        "failure_stage": "settlement",
        "error_class": "ArenaContractError",
    }
    store.settle_call = original


def test_response_adaptation_failure_retains_only_safe_class_and_stage(monkeypatch):
    secret = "api-key-must-not-be-stored"

    class AdapterFailure(RuntimeError):
        pass

    def fail_adaptation(*_args, **_kwargs):
        raise AdapterFailure(secret)

    monkeypatch.setattr(
        br.scoring_provider_compat,
        "adapt_response_with_trusted_url",
        fail_adaptation,
    )
    scoring_context = br.RunContext(**{**CONTEXT.__dict__, "kind": "score"})
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, {"result": {"data": {}}})]),
        funding_source_for=lambda _context: "miner_key",
        credential_for=lambda _context, provider: HOST_KEYS[provider],
    )

    result = broker.execute(
        scoring_context,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com"},
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.calls[result.call["call_identity"]]["actual"] == 0
    assert secret not in json.dumps(result.to_document())


def test_known_raw_cost_survives_response_sanitization_failure(monkeypatch):
    def refuse_response(*_args, **_kwargs):
        raise operations.OperationResponseError("invalid_response")

    monkeypatch.setattr(operations, "sanitize_response", refuse_response)
    payload = {"model": "openai/gpt-4o-mini", "choices": [], "usage": {"cost": "0.0000123"}}
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 13
    assert store.calls[result.call["call_identity"]]["actual"] == 13
    assert store.calls[result.call["call_identity"]]["terminal"]["provider_cost"] == {
        "basis": "openrouter_usage_cost",
        "units": "0.0000123",
        "unit_name": "usd",
        "operation": "openrouter.chat",
    }
