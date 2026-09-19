"""Broker state machine, cost rules, and error genericness (labarena.md 7.3-7.5, 18.3, 18.4)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import quote

import httpx
import pytest

from lab_arena import broker as br
from lab_arena import contracts, operations
from lab_arena.store import ArenaStoreUnavailable

KEY = "sk-or-v1-" + "k" * 40
DL_KEY = "dl_secret_" + "e" * 30
DOG_KEY = "dogsecret" + "d" * 30
HOST_KEYS = {"openrouter": KEY, "deepline": DL_KEY, "scrapingdog": DOG_KEY}
ECHO_KEY = "synthetic+/=query-key"
ENCODED_ECHO_KEY = quote(ECHO_KEY, safe="")
LOWERCASE_ENCODED_ECHO_KEY = ENCODED_ECHO_KEY.replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")


def assert_unpriced_uncertain(result, *, reserved=None):
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    if reserved is None:
        assert result.call["reserved_microusd"] > 0
    else:
        assert result.call["reserved_microusd"] == reserved


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


def luna_price_table():
    table = price_table()
    table["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL] = {
        "prompt": "0.0000002",
        "completion": "0.0000012",
        "request": "0",
        "image": "0",
        "web_search": "0",
        "internal_reasoning": "0",
    }
    return br.validate_price_table(table)


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
        return {
            "status": {"reservation": "reserved", "dispatch": "dispatched", "settlement": "settled", "uncertain": "uncertain", "recovery": "recovered", "refusal": "refused"}[call["kind"]],
            "idempotent": True,
            "call_identity": call["identity"],
            "amount_microusd": (
                call["actual"] if call["kind"] == "settlement" else call["amount"]
            ),
            "terminal_response": call.get("terminal"),
            "reason": call.get("reason"),
            "account_failure_evidence": (
                call.get("uncertain_doc", {}).get("account_failure_evidence")
            ),
        }

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
            self.calls[call_identity] = {
                "kind": "reservation",
                "identity": call_identity,
                "amount": amount_microusd,
                "run_id": run_id,
                "operation_id": operation_id,
                "provider": provider,
                "call_doc": dict(call_doc),
            }
            self.calls[call_identity]["funding_source"] = funding_source
            self.calls[call_identity]["reserve_remaining"] = reserve_remaining
            return {"status": "reserved", "idempotent": False, "call_identity": call_identity, "amount_microusd": amount_microusd}

    def list_ledger(self, *, call_identity=None, limit=None, **_kwargs):
        with self.lock:
            call = self.calls.get(call_identity)
            if call is None:
                return []
            reservation = {
                "entry_id": 1,
                "entry_kind": "reservation",
                "run_id": call["run_id"],
                "call_identity": call["identity"],
                "operation_id": call["operation_id"],
                "provider": call["provider"],
                "funding_source": call["funding_source"],
                "amount_microusd": call["amount"],
                "entry_doc": dict(call["call_doc"]),
            }
            if call["kind"] == "reservation":
                return [reservation]
            head = {
                **reservation,
                "entry_id": 2,
                "entry_kind": call["kind"],
                "amount_microusd": call.get("actual", call["amount"]),
            }
            return [reservation, head]

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

    def reconcile_openrouter_cost(
        self,
        *,
        round_id,
        run_id,
        call_identity,
        uncertain_entry_id,
        generation_id,
        credential_fingerprint,
        actual_microusd,
        cost_units,
    ):
        with self.lock:
            self.log.append("reconcile")
            call = self.calls[call_identity]
            if call["kind"] == "settlement":
                return {
                    "status": "settled",
                    "idempotent": True,
                    "actual_microusd": call["actual"],
                }
            assert call["kind"] == "uncertain"
            assert call["uncertain_doc"]["openrouter_generation_id"] == generation_id
            assert call["uncertain_doc"]["credential_fingerprint"] == credential_fingerprint
            call.update(
                {
                    "kind": "settlement",
                    "actual": actual_microusd,
                    "cost_units": cost_units,
                }
            )
            return {
                "status": "settled",
                "idempotent": False,
                "actual_microusd": actual_microusd,
            }


class ZeroReservationLedgerStore(FakeLedgerStore):
    """Model confirmed-cost admission retaining identity without a money hold."""

    def reserve_call(self, **kwargs):
        result = super().reserve_call(**kwargs)
        if result.get("status") != "reserved" or result.get("idempotent") is True:
            return result
        call = self.calls[kwargs["call_identity"]]
        self.openrouter_capacity += call["amount"]
        call["amount"] = 0
        result["amount_microusd"] = 0
        return result


class FakeTransport:
    def __init__(self, responses=None, *, fail=False):
        self.responses = list(responses or [])
        self.fail = fail
        self.sent: List[Dict[str, Any]] = []
        self.synthetic_deepline_history = None

    def send(
        self,
        *,
        method,
        url,
        headers,
        body,
        timeout_seconds,
        max_response_bytes=None,
    ):
        self.sent.append({"method": method, "url": url, "headers": dict(headers), "body": body, "timeout": timeout_seconds})
        response_headers = {}
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
            selected = self.responses.pop(0) if self.responses else (200, {"data": []})
            if len(selected) == 3:
                status, payload, response_headers = selected
            else:
                status, payload = selected
                response_headers = {}
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
        return br.ProviderResponse(
            status,
            {
                "content-type": "application/json",
                "x-ratelimit-remaining": "3",
                "set-cookie": "s=1",
                **response_headers,
            },
            raw,
        )


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


def test_champion_account_failure_retries_four_provider_attempts_then_marks_fallback():
    marked = []
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(429, {"error": {"code": 429}}) for _ in range(4)]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=7,
        timeout_ms=5000,
    )

    assert result.status == 402
    assert result.call["provider_fallback_required"] is True
    assert result.call["provider_fallback_marked"] is True
    assert result.call["champion_credential_attempts"] == 4
    assert len(transport.sent) == 4
    assert len(store.calls) == 4
    base_identities = {
        call["call_doc"]["base_call_identity"]
        for call in store.calls.values()
    }
    assert len(base_identities) == 1
    assert {
        call["call_doc"]["provider_attempt"]
        for call in store.calls.values()
    } == {1, 2, 3, 4}
    assert all(
        call["call_doc"]["action_sequence"] == 7
        for call in store.calls.values()
    )
    assert all(
        call["terminal"]["account_failure_evidence"]["provider_status"]
        == 429
        for call in store.calls.values()
    )
    assert marked == [
        (
            CONTEXT.run_id,
            "scrapingdog",
            {
                "error_class": "account_credential_failure",
                "provider_status": 429,
                "action_sequence": 7,
                "provider_attempts": 4,
                "base_call_identity": result.call["base_call_identity"],
            },
        )
    ]


def test_champion_provider_outage_does_not_retry_or_mark_fallback():
    marked = []
    broker, store, transport = make_broker(
        transport=FakeTransport([(503, {"error": {"code": 503}})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context, provider, evidence))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert len(transport.sent) == 1
    assert len(store.calls) == 1
    assert marked == []


def test_champion_openrouter_upstream_429_does_not_fallback_account():
    marked = []
    payload = {
        "error": {
            "code": 429,
            "message": "upstream temporarily rate-limited",
            "metadata": {"provider_name": "upstream-provider"},
        }
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(429, payload)]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context, provider, evidence))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=7,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["provider_status"] == 429
    assert len(transport.sent) == len(store.calls) == 1
    assert marked == []


@pytest.mark.parametrize(
    ("provider_status", "payload"),
    [
        (
            403,
            {
                "error": {
                    "code": 403,
                    "metadata": {"error_type": "content_policy_violation"},
                }
            },
        ),
        (
            403,
            {
                "error": {
                    "code": 403,
                    "metadata": {"error_type": "refusal"},
                }
            },
        ),
        (
            403,
            {
                "error_type": "content_policy_violation",
                "error": {"code": "image_content_policy_violation"},
            },
        ),
        (
            403,
            {
                "error_type": "refusal",
                "error": {"code": "invalid_prompt"},
            },
        ),
        (
            200,
            {
                "object": "response",
                "status": "failed",
                "error_type": "content_policy_violation",
                "error": {
                    "code": "image_content_policy_violation",
                    "message": "content policy violation",
                },
            },
        ),
        (
            200,
            {
                "object": "response",
                "status": "failed",
                "error_type": "refusal",
                "error": {
                    "code": "invalid_prompt",
                    "message": "request refused",
                },
            },
        ),
        (
            403,
            {
                "error": {
                    "code": 403,
                    "metadata": {"patterns": ["blocked-pattern"]},
                }
            },
        ),
    ],
    ids=(
        "chat_content_policy",
        "chat_refusal",
        "responses_content_policy",
        "responses_refusal",
        "embedded_responses_content_policy",
        "embedded_responses_refusal",
        "guardrail_patterns",
    ),
)
def test_champion_openrouter_policy_403_does_not_retry_or_fallback_account(
    provider_status, payload,
):
    marked = []
    flagged_input = "private prompt that must not be persisted"
    payload["private_detail"] = flagged_input
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(provider_status, payload) for _ in range(4)]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context, provider, evidence))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=7,
        timeout_ms=5000,
    )

    assert result.status == 403
    assert json.loads(result.body) == {
        "error": {"code": "provider_request_refused"}
    }
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["provider_status"] == 403
    assert "champion_credential_attempts" not in result.call
    assert "provider_fallback_required" not in result.call
    assert len(transport.sent) == len(store.calls) == 1
    assert marked == []
    assert flagged_input not in repr(result.to_document())
    assert flagged_input not in repr(store.calls)


@pytest.mark.parametrize("provider_status", [401, 402, 429])
def test_champion_openrouter_zero_cost_account_failure_latches(
    provider_status,
):
    marked = []
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (
                    provider_status,
                    {"error": {"code": provider_status}},
                )
                for _ in range(4)
            ]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=10,
        timeout_ms=5000,
    )

    assert result.call["provider_fallback_required"] is True
    assert result.call["provider_fallback_marked"] is True
    assert len(transport.sent) == 4
    assert len(store.calls) == 4
    assert {
        call["kind"] for call in store.calls.values()
    } == {"settlement"}
    assert all(
        call["terminal"]["account_failure_evidence"][
            "provider_status"
        ]
        == provider_status
        for call in store.calls.values()
    )
    assert all(call["actual"] == 0 for call in store.calls.values())
    assert marked == [
        (
            CONTEXT.run_id,
            "openrouter",
            {
                "error_class": "account_credential_failure",
                "provider_status": provider_status,
                "action_sequence": 10,
                "provider_attempts": 4,
                "base_call_identity": result.call["base_call_identity"],
            },
        )
    ]


def test_champion_retry_resumes_after_settled_account_failure_replay():
    store = FakeLedgerStore()
    first, _store, first_transport = make_broker(
        store=store,
        transport=FakeTransport([(401, {"error": {"code": 401}})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
    )
    first_result = first._execute_once(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=8,
        timeout_ms=5000,
        provider_attempt=1,
        champion_credential_retry=True,
    )
    assert first_result.call["error_code"] == "miner_credentials_unavailable"
    assert len(first_transport.sent) == 1

    marked = []
    resumed, _store, resumed_transport = make_broker(
        store=store,
        transport=FakeTransport(
            [(401, {"error": {"code": 401}}) for _ in range(3)]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "marked"}
        ),
    )
    result = resumed.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=8,
        timeout_ms=5000,
    )

    assert result.call["provider_fallback_marked"] is True
    assert len(resumed_transport.sent) == 3
    assert marked[0][2]["provider_status"] == 401


def test_champion_retry_recognizes_durable_uncertain_account_evidence():
    store = FakeLedgerStore()
    broker, _store, transport = make_broker(
        store=store,
        transport=FakeTransport([(401, {"error": {"code": 401}})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
    )
    first = broker._execute_once(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "acme"}},
        action_sequence=9,
        timeout_ms=5000,
        provider_attempt=1,
        champion_credential_retry=True,
    )
    replay = broker._execute_once(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "acme"}},
        action_sequence=9,
        timeout_ms=5000,
        provider_attempt=1,
        champion_credential_retry=True,
    )

    assert first.call["error_code"] == "miner_credentials_unavailable"
    assert replay.call["error_code"] == "miner_credentials_unavailable"
    assert replay.call["provider_status"] == 401
    assert replay.call["outcome"] == "uncertain"
    assert len(transport.sent) == 1


@pytest.mark.parametrize("provider_status", [401, 402, 403, 429, 503])
@pytest.mark.parametrize("funding_source", ["miner_key", "host"])
def test_ordinary_deepline_account_failure_retains_bound_evidence_only(
    provider_status, funding_source,
):
    broker, store, transport = make_broker(
        transport=FakeTransport([(provider_status, {"error": "refused"})]),
        provider_funding_source_for=lambda _context, _provider: funding_source,
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "acme"}},
        action_sequence=9,
        timeout_ms=5000,
    )
    call = next(iter(store.calls.values()))
    assert call["kind"] == "uncertain"
    assert len(transport.sent) == 1
    evidence = call["uncertain_doc"].get("account_failure_evidence")
    if funding_source == "miner_key" and provider_status in (401, 402, 403):
        assert result.call["error_code"] == "miner_credentials_unavailable"
        assert evidence == {
            "error_class": "account_credential_failure",
            "provider_status": provider_status,
            "base_call_identity": call["call_doc"]["base_call_identity"],
            "provider_attempt": call["call_doc"]["provider_attempt"],
            "action_sequence": call["call_doc"]["action_sequence"],
        }
    else:
        assert evidence is None
    assert "account_failure_evidence" not in result.to_document()["call"]
    assert call["uncertain_doc"]["call_succeeded"] is False


def test_missing_optional_champion_credential_marks_fallback_without_dispatch():
    marked = []

    def missing(_context, provider):
        assert provider == "scrapingdog"
        raise br.BrokerError("miner_credentials_unavailable")

    broker, store, transport = make_broker(
        credential_for=missing,
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "marked"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=3,
        timeout_ms=5000,
    )

    assert result.call["provider_fallback_required"] is True
    assert result.call["champion_credential_attempts"] == 4
    assert transport.sent == []
    assert store.calls == {}
    assert marked[0][2]["provider_status"] is None
    assert marked[0][2]["base_call_identity"] is None


def test_latched_champion_fallback_aborts_old_run_without_provider_probe():
    marked = []
    broker, store, transport = make_broker(
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        provider_restart_required_for=lambda _context, _provider: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "existing"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.call["provider_fallback_required"] is True
    assert result.call["provider_fallback_marked"] is True
    assert transport.sent == []
    assert store.calls == {}
    assert marked[0][1] == "openrouter"


def test_fallback_race_at_reservation_marks_run_before_returning():
    class FallbackRaceStore(FakeLedgerStore):
        def reserve_call(self, **_kwargs):
            return {"status": "champion_restart_required"}

    marked = []
    broker, store, transport = make_broker(
        store=FallbackRaceStore(),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        provider_restart_required_for=lambda _context, _provider: False,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context.run_id, provider, dict(evidence)))
            or {"status": "existing"}
        ),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=10,
        timeout_ms=5000,
    )

    assert result.call["provider_fallback_required"] is True
    assert result.call["provider_fallback_marked"] is True
    assert result.call["outcome"] == "not_dispatched"
    assert transport.sent == []
    assert store.calls == {}
    assert marked[0][1] == "openrouter"


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


def deepline_generic_http_denial(request_id="iad1::generic-http-denied"):
    return {
        "code": "PROVIDER_AUTHORIZATION_FAILED",
        "credential_owner": "workspace",
        "credential_source": "managed",
        "error": "private upstream error",
        "error_category": "authorization",
        "failure_description": "private failure detail",
        "failure_origin": "provider",
        "message": "private provider message",
        "operation": "generic_http_request",
        "operator_hint": "private operator hint",
        "provider": "generic_http",
        "requestId": request_id,
        "request_id": request_id,
        "tool_error": {
            "category": "authorization",
            "code": "PROVIDER_AUTHORIZATION_FAILED",
            "networkKind": None,
            "networkScope": None,
            "operation": "generic_http_request",
            "origin": "provider",
            "provider": "generic_http",
            "requestId": request_id,
            "retryAfterMs": None,
            "retryable": False,
            "schemaVersion": 1,
            "statusCode": 403,
            "toolId": "generic-http",
        },
        "upstream_status": 403,
    }


def test_deepline_generic_http_403_is_a_replayable_request_refusal():
    request_id = "iad1::generic-http-denied"
    denial = deepline_generic_http_denial(request_id)
    history_entry = deepline_history_entry(
        request_id,
        "generic_http_request",
        0,
        charge_state="failed",
        provider="generic_http",
    )
    history_entry.update({"status": "error", "delta": 0})
    marked = []
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(403, denial), (200, deepline_history(history_entry))]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context, provider, evidence))
            or {"status": "marked"}
        ),
    )
    arguments = dict(
        operation_id="deepline.execute",
        parameters={
            "tool": "generic_http_request",
            "payload": {"url": "https://httpbin.org/status/403"},
        },
        action_sequence=0,
        timeout_ms=5000,
    )

    result = broker.execute(CONTEXT, **arguments)
    replay = broker.execute(CONTEXT, **arguments)

    assert result.status == 403
    assert json.loads(result.body) == {
        "error": {"code": "provider_request_refused"}
    }
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["provider_status"] == 403
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_billing_history_failed_zero"
    assert "champion_credential_attempts" not in result.call
    assert "provider_fallback_required" not in result.call
    assert marked == []
    assert replay.call["idempotent"] is True
    assert replay.call["error_code"] == "provider_request_refused"
    assert replay.call["provider_status"] == 403
    assert replay.body == result.body and replay.status == result.status
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert "account_failure_evidence" not in terminal
    for private_text in (
        denial["error"],
        denial["failure_description"],
        denial["message"],
        denial["operator_hint"],
    ):
        assert private_text not in repr(result.to_document())
        assert private_text not in repr(store.calls)


def test_deepline_generic_http_403_without_exact_cost_stays_uncertain():
    request_id = "iad1::generic-http-cost-pending"
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(403, deepline_generic_http_denial(request_id)), (200, deepline_history())]
        )
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "generic_http_request",
            "payload": {"url": "https://httpbin.org/status/403"},
        },
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert transport.sent[0]["method"] == "POST"
    assert all(sent["method"] == "GET" for sent in transport.sent[1:])


@pytest.mark.parametrize(
    ("parameters", "payload"),
    [
        (
            {"tool": "exa_search"},
            deepline_generic_http_denial(),
        ),
        (
            {"tool": "generic_http_request"},
            {**deepline_generic_http_denial(), "code": "ACCOUNT_FORBIDDEN"},
        ),
        (
            {"tool": "generic_http_request"},
            {**deepline_generic_http_denial(), "provider": "other"},
        ),
        (
            {"tool": "generic_http_request"},
            {**deepline_generic_http_denial(), "operation": "other"},
        ),
        (
            {"tool": "generic_http_request"},
            {**deepline_generic_http_denial(), "upstream_status": 401},
        ),
        (
            {"tool": "generic_http_request"},
            {
                **deepline_generic_http_denial(),
                "tool_error": {
                    **deepline_generic_http_denial()["tool_error"],
                    "statusCode": 401,
                },
            },
        ),
    ],
    ids=(
        "other_tool",
        "other_top_level_code",
        "other_top_level_provider",
        "other_top_level_operation",
        "other_upstream_status",
        "other_tool_error_status",
    ),
)
def test_deepline_request_refusal_requires_the_exact_denial_shape(
    parameters, payload
):
    response = br.ProviderResponse(
        403,
        {"content-type": "application/json"},
        json.dumps(payload).encode("utf-8"),
    )

    assert br._deepline_generic_http_request_refusal(parameters, response) is False


def test_default_http_transport_does_not_inherit_proxy_environment():
    transport = br.HttpxProviderTransport()
    try:
        client = transport._client_factory()
        assert client._trust_env is False
        asyncio.run(client.aclose())
    finally:
        transport.close()


def _async_client_factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _ReadTimeoutAfterHeaders(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"partial":'
        raise httpx.ReadTimeout("synthetic read timeout")


class _TrickleUntilCancelled(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = threading.Event()

    async def __aiter__(self):
        while True:
            await asyncio.sleep(0.01)
            yield b"x"

    async def aclose(self):
        self.closed.set()


def test_http_transport_absolute_deadline_cancels_trickle_and_retains_header():
    stream = _TrickleUntilCancelled()
    requests = []

    def respond(request):
        requests.append(request.method)
        return httpx.Response(
            200,
            headers={"X-Generation-Id": "gen-absolute-timeout"},
            stream=stream,
        )

    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(respond)
    )
    started = time.monotonic()
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(
                method="POST",
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={},
                body=b"{}",
                timeout_seconds=0.06,
            )
    finally:
        transport.close()

    assert time.monotonic() - started < 0.5
    assert requests == ["POST"]
    assert str(raised.value) == "ReadTimeout"
    assert raised.value.openrouter_generation_id == "gen-absolute-timeout"
    assert stream.closed.is_set()


def test_http_transport_absolute_deadline_cancels_before_response_headers(
    caplog,
):
    secret = "synthetic-query-secret"
    handler_cancelled = threading.Event()

    async def never_returns_headers(_request):
        try:
            logging.getLogger("httpx").debug("request URL contained %s", secret)
            await asyncio.sleep(10)
        finally:
            handler_cancelled.set()

    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(never_returns_headers)
    )
    caplog.set_level(logging.DEBUG)
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(
                method="GET",
                url="https://api.scrapingdog.com/search?api_key=" + secret,
                headers={},
                body=b"",
                timeout_seconds=0.05,
            )
    finally:
        transport.close()

    assert str(raised.value) == "ReadTimeout"
    assert raised.value.openrouter_generation_id is None
    assert raised.value.deepline_job_id is None
    assert raised.value.observed_status is None
    assert handler_cancelled.is_set()
    assert secret not in caplog.text


@pytest.mark.parametrize("status", [200, 202, 429, 503])
def test_http_transport_retains_only_bounded_status_after_headers(status):
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(
            lambda _request: httpx.Response(
                status, stream=_ReadTimeoutAfterHeaders()
            )
        )
    )
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(
                method="GET",
                url="https://api.scrapingdog.com/google",
                headers={}, body=b"", timeout_seconds=1,
            )
    finally:
        transport.close()
    assert raised.value.observed_status == status
    assert str(raised.value) == "ReadTimeout"


@pytest.mark.parametrize("invalid", [0, 99, 600, True, "200"])
def test_transport_error_rejects_untrusted_or_invalid_status(invalid):
    assert br.ProviderTransportError(
        "ReadTimeout", observed_status=invalid
    ).observed_status is None


def _scrapingdog_timeout_transport(status):
    requests = []

    def respond(request):
        requests.append(request)
        if status is None:
            raise httpx.ConnectTimeout("synthetic pre-header timeout")
        return httpx.Response(status, stream=_ReadTimeoutAfterHeaders())

    return br.HttpxProviderTransport(
        client_factory=_async_client_factory(respond)
    ), requests


@pytest.mark.parametrize("status", [200, 202])
def test_scrapingdog_post_header_success_timeout_settles_known_charge(status):
    transport, requests = _scrapingdog_timeout_transport(status)
    broker, store, _ = make_broker(transport=transport)
    try:
        result = broker.execute(
            CONTEXT, operation_id="scrapingdog.google",
            parameters={"query": "acme"}, action_sequence=0,
            timeout_ms=90_000,
        )
        replay = broker.execute(
            CONTEXT, operation_id="scrapingdog.google",
            parameters={"query": "acme"}, action_sequence=0,
            timeout_ms=90_000,
        )
    finally:
        transport.close()

    assert len(requests) == 1
    assert result.status == replay.status == 502
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result.call["outcome"] == "settled"
    assert result.call["status"] == 502
    assert result.call["provider_status"] == status
    assert result.call["actual_microusd"] == 250
    assert result.call["observed_provider_status"] == status
    assert result.call["transport_error_class"] == "ReadTimeout"
    assert store.log == ["reserve", "dispatch", "settle", "reserve"]
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "settlement" and call["actual"] == 250
    assert call["terminal"]["status"] == 502
    assert call["terminal"]["call_succeeded"] is False
    assert replay.body == base64.b64decode(call["terminal"]["body_b64"])
    assert replay.call["actual_microusd"] == 250
    assert call["terminal"]["provider_cost"] == {
        "basis": "scrapingdog_approved_operation_fallback",
        "units": "5", "unit_name": "credits",
        "operation": "scrapingdog.google",
    }


@pytest.mark.parametrize("status", [None, 429, 503])
def test_scrapingdog_without_observed_success_stays_uncertain(status):
    transport, requests = _scrapingdog_timeout_transport(status)
    broker, store, _ = make_broker(transport=transport)
    try:
        result = broker.execute(
            CONTEXT, operation_id="scrapingdog.google",
            parameters={"query": "acme"}, action_sequence=0,
            timeout_ms=90_000,
        )
    finally:
        transport.close()

    assert len(requests) == 1
    assert result.status == 502
    assert_unpriced_uncertain(result, reserved=250)
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain" and call["amount"] == 250
    assert call["uncertain_doc"] == {
        "reason": "transport_failure", "call_succeeded": False,
        "transport_error_class": (
            "ConnectTimeout" if status is None else "ReadTimeout"
        ),
        **({} if status is None else {"observed_provider_status": status}),
    }


def test_scrapingdog_known_charge_survives_settlement_failure():
    class FailingSettlementStore(FakeLedgerStore):
        def settle_call(self, **kwargs):
            self.log.append("settle")
            raise br.ArenaStoreError("synthetic settlement failure")

    transport, requests = _scrapingdog_timeout_transport(200)
    store = FailingSettlementStore()
    broker, _, _ = make_broker(store=store, transport=transport)
    try:
        result = broker.execute(
            CONTEXT, operation_id="scrapingdog.google",
            parameters={"query": "acme"}, action_sequence=0,
            timeout_ms=90_000,
        )
    finally:
        transport.close()

    assert len(requests) == 1
    assert result.status == 502
    assert_unpriced_uncertain(result, reserved=250)
    uncertain = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert uncertain["reason"] == "settle_failure"
    assert uncertain["call_succeeded"] is False
    assert uncertain["known_actual_microusd"] == 250
    assert uncertain["provider_cost"]["operation"] == "scrapingdog.google"
    assert uncertain["observed_provider_status"] == 200
    assert uncertain["transport_error_class"] == "ReadTimeout"
    assert store.log == ["reserve", "dispatch", "settle", "settle", "settle", "uncertain"]


@pytest.mark.parametrize(
    ("response_headers", "expected_generation_id"),
    (
        ({}, None),
        ({"X-Generation-Id": "invalid/generation/id"}, None),
        (
            [
                ("X-Generation-Id", "gen-conflict-one"),
                ("X-Generation-Id", "gen-conflict-two"),
            ],
            None,
        ),
        ({"X-Generation-Id": "gen-stream-timeout"}, "gen-stream-timeout"),
    ),
    ids=("missing", "invalid", "conflicting_duplicates", "valid"),
)
def test_http_transport_retains_only_one_valid_generation_header_on_read_timeout(
    response_headers, expected_generation_id
):
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(
            lambda _request: httpx.Response(
                200,
                headers=response_headers,
                stream=_ReadTimeoutAfterHeaders(),
            )
        )
    )
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(
                method="POST",
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={},
                body=b"{}",
                timeout_seconds=1,
            )
    finally:
        transport.close()

    assert raised.value.openrouter_generation_id == expected_generation_id
    assert str(raised.value) == "ReadTimeout"


@pytest.mark.parametrize(
    ("secret", "echoed_header"),
    (
        (KEY, KEY),
        (ECHO_KEY, ENCODED_ECHO_KEY),
    ),
    ids=("raw", "url_encoded"),
)
def test_stream_timeout_generation_header_credential_echo_is_not_persisted(
    caplog, secret, echoed_header
):
    def timed_out_response(request):
        assert request.headers["authorization"] == "Bearer " + secret
        return httpx.Response(
            200,
            headers={"X-Generation-Id": echoed_header},
            stream=_ReadTimeoutAfterHeaders(),
        )

    inner = br.HttpxProviderTransport(
        client_factory=_async_client_factory(timed_out_response)
    )
    exception_messages = []

    class CapturingTransport:
        def send(self, **kwargs):
            try:
                return inner.send(**kwargs)
            except br.ProviderTransportError as exc:
                exception_messages.append(str(exc))
                raise

    store = FakeLedgerStore()
    broker = br.Broker(
        store=store,
        key_for=lambda provider: (
            secret if provider == "openrouter" else HOST_KEYS[provider]
        ),
        price_table=price_table(),
        transport=CapturingTransport(),
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
    )
    caplog.set_level(logging.DEBUG)
    try:
        result = broker.execute(
            CONTEXT,
            operation_id="openrouter.chat",
            parameters=CHAT,
            action_sequence=0,
            timeout_ms=30000,
        )
    finally:
        inner.close()

    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    retained = json.dumps(store.calls)
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert diagnostic == {
        "reason": "transport_failure",
        "call_succeeded": False,
        "transport_error_class": "ReadTimeout",
        "observed_provider_status": 200,
    }
    assert exception_messages == ["ReadTimeout"]
    assert secret not in caplog.text and secret not in retained
    assert echoed_header not in caplog.text and echoed_header not in retained


CONTEXT = br.RunContext(run_id="r1", assignment_id="arena-2026-09-02:s1:1:0", icp_position=0, lease_token_hash=contracts.document_hash("lease"), miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", submission_id="s1", stage=1)
CHAT = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "find fintech companies"}], "max_tokens": 200}
LUNA_RESPONSES = {
    "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
    "input": "research regional capacity",
    "reasoning": {"effort": "xhigh"},
    "max_output_tokens": 256,
    "prompt_cache_key": "arena-luna-route-test",
}


def test_persistent_reservation_store_unavailable_propagates_after_one_readback():
    class UnavailableReservation(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            self.log.append("reserve")
            raise ArenaStoreUnavailable("synthetic RPC transport failure")

    broker, store, transport = make_broker(store=UnavailableReservation())
    with pytest.raises(ArenaStoreUnavailable):
        broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                       action_sequence=0, timeout_ms=5000)
    assert store.log == ["reserve", "reserve"]
    assert store.calls == {}
    assert transport.sent == []


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


@pytest.mark.parametrize(
    ("tool", "payload", "result_data", "basis"),
    [
        (
            "contextdev_post_web_search",
            {"query": "regional banks"},
            {"results": [{"title": "Result", "url": "https://example.com/"}]},
            "deepline_contextdev_web_search_completed_zero",
        ),
        (
            "contextdev_post_news_search",
            {
                "searchBy": {
                    "type": "entity",
                    "entity": {"type": "domain", "domain": "example.com"},
                }
            },
            [{"title": "Result", "url": "https://example.com/news"}],
            "deepline_contextdev_news_search_completed_zero",
        ),
    ],
)
def test_contextdev_search_authorization_transport_and_completed_zero_cost(
    tool, payload, result_data, basis
):
    envelope = {
        "job_id": "iad1::contextdev-search",
        "status": "completed",
        "result": {"data": result_data},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": tool, "payload": payload},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 200 and json.loads(result.body) == envelope
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == basis
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1
    sent = transport.sent[0]
    assert sent["method"] == "POST"
    assert sent["url"] == f"https://code.deepline.com/api/v2/integrations/{tool}/execute"
    assert sent["headers"]["authorization"] == "Bearer " + DL_KEY
    assert json.loads(sent["body"]) == {
        "provider": "contextdev", "operation": tool, "payload": payload,
    }


@pytest.mark.parametrize(
    ("tool", "payload"),
    [
        ("contextdev_post_web_search", {"query": "regional banks"}),
        (
            "contextdev_post_news_search",
            {
                "searchBy": {
                    "type": "entity",
                    "entity": {"type": "domain", "domain": "example.com"},
                }
            },
        ),
    ],
)
def test_contextdev_search_error_without_billing_stays_uncertain(tool, payload):
    broker, store, transport = make_broker(
        transport=FakeTransport([(502, {"error": {"code": "upstream_error"}})])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": tool, "payload": payload},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert len(transport.sent) == 1


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


def test_company_profile_uses_existing_billing_and_budget_controls():
    company = {"name": "Example", "website": "https://example.com",
               "linkedinUrl": "https://www.linkedin.com/company/example/",
               "employeeCountRange": {"start": 2, "end": 10}, "employeeCount": 6}
    envelope = {"job_id": "company-profile-1", "status": "completed",
                "result": {"data": {"status": 200, "element": company}},
                "billing": {"credits_charged": 0.03, "cost_usd": 0.003}}
    broker, store, transport = make_broker(transport=FakeTransport([(200, envelope)]))
    parameters = {"tool": "harvestapi_get_company", "payload": {"url": company["linkedinUrl"]}}
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters=parameters,
                            action_sequence=0, timeout_ms=5000)
    assert result.status == 200
    assert json.loads(result.body)["result"]["data"]["element"] == company
    assert result.call["reserved_microusd"] == 3000
    assert "reservation_basis" not in result.call
    assert result.call["actual_microusd"] == 3000
    assert result.call["outcome"] == "settled"
    assert store.openrouter_capacity == 10_000_000 - 3000
    assert json.loads(transport.sent[0]["body"])["provider"] == "harvestapi"
    repeated = broker.execute(CONTEXT, operation_id="deepline.execute", parameters=parameters,
                              action_sequence=0, timeout_ms=5000)
    assert repeated.body == result.body and len(transport.sent) == 1
    store.per_icp_quota = 1
    refused = broker.execute(CONTEXT, operation_id="deepline.execute", parameters=parameters,
                             action_sequence=1, timeout_ms=5000)
    assert refused.status == 402 and len(transport.sent) == 1


def test_company_profile_fixed_reservation_enforces_cap_before_dispatch():
    store = FakeLedgerStore(openrouter_capacity=2999)
    broker, _store, transport = make_broker(store=store)
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "harvestapi_get_company", "payload": {"universalName": "example"}},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 402
    assert result.call["outcome"] == "refused"
    assert result.call["reason"] == "provider_cost_cap"
    assert transport.sent == []


def test_company_profile_transport_failure_holds_only_fixed_reservation():
    store = FakeLedgerStore(openrouter_capacity=10_000)
    broker, _store, transport = make_broker(store=store, transport=FakeTransport(fail=True))
    parameters = {"tool": "harvestapi_get_company", "payload": {"url": "https://example.com"}}
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters=parameters,
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert_unpriced_uncertain(result, reserved=3000)
    assert store.openrouter_capacity == 7000
    sent_before_replay = len(transport.sent)
    replay = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters=parameters,
        action_sequence=0,
        timeout_ms=5000,
    )
    assert replay.status == 409
    assert json.loads(replay.body) == {"error": {"code": "call_uncertain"}}
    assert len(transport.sent) == sent_before_replay
    assert sum(sent["method"] == "POST" for sent in transport.sent) == 1


def test_company_profile_does_not_block_concurrent_dynamic_admission(monkeypatch):
    company_started = threading.Event()
    release_company = threading.Event()

    class ConcurrentTransport(FakeTransport):
        def send(self, *, method, url, headers, body, timeout_seconds, max_response_bytes=None):
            operation = json.loads(body)["operation"]
            self.sent.append({"method": method, "url": url, "headers": dict(headers),
                              "body": body, "timeout": timeout_seconds})
            if operation == "harvestapi_get_company":
                company_started.set()
                assert release_company.wait(timeout=2)
                payload = {"job_id": "company-profile-1", "status": "completed",
                           "result": {"data": {"status": 200, "element": {}}},
                           "billing": {"credits_charged": 0.03, "cost_usd": 0.003}}
            else:
                assert operation == "exa_search"
                payload = {"job_id": "exa-search-1", "status": "completed", "results": [],
                           "billing": {"credits_charged": 0.01, "cost_usd": 0.001}}
            return br.ProviderResponse(
                200, {"content-type": "application/json"}, json.dumps(payload).encode("utf-8")
            )

    monkeypatch.setattr(operations, "BUDGET_ADMISSION_MAX_SECONDS", 0)
    broker, store, transport = make_broker(transport=ConcurrentTransport())
    with ThreadPoolExecutor(max_workers=2) as pool:
        company = pool.submit(
            broker.execute,
            CONTEXT,
            operation_id="deepline.execute",
            parameters={"tool": "harvestapi_get_company", "payload": {"search": "example.com"}},
            action_sequence=0,
            timeout_ms=5000,
        )
        assert company_started.wait(timeout=2)
        dynamic = broker.execute(
            CONTEXT,
            operation_id="deepline.execute",
            parameters={"tool": "exa_search", "payload": {"query": "example"}},
            action_sequence=1,
            timeout_ms=5000,
        )
        release_company.set()
        company_result = company.result(timeout=2)

    assert company_result.status == dynamic.status == 200
    assert company_result.call["reserved_microusd"] == company_result.call["actual_microusd"] == 3000
    assert dynamic.call["reservation_basis"] == "remaining_budget_dynamic_deepline"
    assert dynamic.call["actual_microusd"] == 1000
    assert store.openrouter_capacity == 10_000_000 - 4000
    assert [json.loads(sent["body"])["operation"] for sent in transport.sent] == [
        "harvestapi_get_company", "exa_search"
    ]


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
    ("status", "chunks", "expected_provenance"),
    [
        (200, [b"abcd"], "response_too_large"),
        (302, [b"redirect"], "redirect_rejected"),
    ],
)
def test_http_transport_labels_synthetic_generic_response(status, chunks, expected_provenance):
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(
            lambda _request: httpx.Response(status, content=b"".join(chunks))
        ),
        max_response_bytes=3,
    )
    response = transport.send(
        method="POST", url="https://example.com/execute", headers={}, body=b"{}",
        timeout_seconds=1,
    )
    assert response.status == 502
    assert response.body == operations.GENERIC_UNAVAILABLE_BODY
    assert response.internal_provenance == expected_provenance


def test_http_transport_per_call_limit_accepts_exact_boundary_only():
    body = [b"abcd"]
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(
            lambda _request: httpx.Response(200, content=body[0])
        ),
        max_response_bytes=3,
    )
    accepted = transport.send(
        method="POST",
        url="https://example.com/execute",
        headers={},
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=4,
    )
    assert accepted.status == 200 and accepted.body == b"abcd"
    body[0] = b"abcde"
    refused = transport.send(
        method="POST",
        url="https://example.com/execute",
        headers={},
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=4,
    )
    assert refused.status == 502
    assert refused.internal_provenance == "response_too_large"


@pytest.mark.parametrize("credential_echo", [False, True])
def test_routed_firecrawl_settles_large_envelope_and_returns_bounded_html(credential_echo):
    requested_url = "https://wonderskin.com/"
    tail_marker = DL_KEY if credential_echo else "raw-envelope-tail-must-not-be-persisted"
    raw_html_bytes = 5_030_336
    raw_html = (
        "<html>"
        + ("x" * (raw_html_bytes - len(tail_marker) - len("<html></html>")))
        + tail_marker
        + "</html>"
    )
    assert len(raw_html.encode("utf-8")) == raw_html_bytes
    envelope_document = {
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
        "job_id": "iad1::large-firecrawl-envelope",
        "result": {"data": {
            "rawHtml": raw_html,
            "metadata": {
                "sourceURL": requested_url,
                "url": requested_url,
                "statusCode": 200,
            },
        }},
        "status": "completed",
        "provider_metadata_padding": "",
    }
    envelope = json.dumps(
        envelope_document, separators=(",", ":")
    ).encode("utf-8")
    envelope_document["provider_metadata_padding"] = "p" * (
        5_271_155 - len(envelope)
    )
    envelope = json.dumps(
        envelope_document, separators=(",", ":")
    ).encode("utf-8")
    assert len(envelope) == 5_271_155

    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(
            lambda _request: httpx.Response(200, content=envelope)
        ),
        max_response_bytes=4 * 1024 * 1024,
    )
    scoring_context = br.RunContext(**{
        **CONTEXT.__dict__,
        "kind": "score",
        "round_id": "arena-2026-09-12",
    })
    broker, store, _ = make_broker(
        transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )

    result = broker.execute(
        scoring_context,
        operation_id="scrapingdog.scrape",
        parameters={"url": requested_url},
        action_sequence=0,
        timeout_ms=60_000,
    )

    if credential_echo:
        # The credential is beyond both the old transport and visible limits.
        # Scan the full accepted envelope before adaptation can truncate it.
        assert result.status == 502 and result.call["outcome"] == "uncertain"
        assert DL_KEY not in json.dumps(result.to_document())
        assert DL_KEY not in json.dumps(store.calls)
        return

    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 2_000
    assert result.call["reserved_microusd"] == 10_000_000
    assert len(result.body) == operations.OPERATIONS[
        "scrapingdog.scrape"
    ].max_response_bytes
    assert tail_marker.encode() not in result.body
    assert store.openrouter_capacity == 9_998_000
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    persisted_body = base64.b64decode(terminal["body_b64"])
    assert persisted_body == result.body
    assert tail_marker.encode() not in persisted_body
    assert b'"billing"' not in persisted_body
    assert b'"provider_metadata_padding"' not in persisted_body
    assert terminal["provider_cost"] == {
        "basis": "deepline_billing_credits_charged_x_0.10_usd",
        "units": "0.02",
        "unit_name": "credits",
        "operation": "firecrawl_scrape",
        "request_id": "iad1::large-firecrawl-envelope",
    }


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
    assert len(transport.sent) == 1
    assert sent["headers"]["authorization"] == "Bearer " + KEY
    assert sent["headers"]["x-openrouter-metadata"] == "enabled"
    body = json.loads(sent["body"])
    assert body["provider"] == {"allow_fallbacks": False, "data_collection": "deny", "zdr": True} and body["stream"] is False
    assert store.openrouter_capacity == 10_000_000 - expected_actual


def test_native_web_search_reserves_every_bounded_search_and_settles_usage_cost_once():
    table = luna_price_table()
    # A zero catalog component must not turn an enabled paid hosted tool into
    # a zero-dollar reservation.
    assert table["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL]["web_search"] == "0"
    parameters = {
        **LUNA_RESPONSES,
        "tools": [{
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "native",
                "max_uses": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
                "max_total_results": operations.OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS,
            },
        }],
        "max_tool_calls": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
    }
    payload = {
        "id": "gen-luna-search", "object": "response", "status": "completed",
        "model": br.OPENROUTER_LUNA_RESPONSES_MODEL, "error": None,
        "output": [],
        "usage": {
            "input_tokens": 30, "output_tokens": 20, "total_tokens": 50,
            "cost": "0.03125",
            "server_tool_use": {"web_search_requests": 3},
        },
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
    )
    broker._price_table = table

    result = broker.execute(
        CONTEXT, operation_id="openrouter.responses", parameters=parameters,
        action_sequence=0, timeout_ms=300_000,
    )

    route = br._openrouter_host_route(
        kind="execute", operation_id="openrouter.responses",
        model=br.OPENROUTER_LUNA_RESPONSES_MODEL,
        pricing=table["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL],
        parameters=parameters,
    )
    priced_floor = br._max_openrouter_cost_for_pricing(
        route.reservation_pricing, parameters,
        max_output_tokens=parameters["max_output_tokens"],
    )
    plain_parameters = {
        key: value for key, value in parameters.items()
        if key not in ("tools", "max_tool_calls")
    }
    plain_reserve = br._max_openrouter_cost_for_pricing(
        route.reservation_pricing, plain_parameters,
        max_output_tokens=plain_parameters["max_output_tokens"],
    )
    expected_search_reserve = int(
        br.OPENROUTER_NATIVE_WEB_SEARCH_RESERVATION_USD_PER_CALL
        * operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS * br.MICROUSD
    )
    assert priced_floor - plain_reserve >= expected_search_reserve
    assert result.call["reserved_microusd"] == 10_000_000
    assert result.call["reservation_basis"] == "remaining_budget_native_web_search"
    assert result.call["actual_microusd"] == 31_250
    assert store.log == ["reserve", "dispatch", "settle"]
    sent = json.loads(transport.sent[0]["body"])
    assert sent["tools"] == parameters["tools"]
    assert sent["max_tool_calls"] == operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS


@pytest.mark.parametrize("funding_source", ("host", "miner_key"))
def test_luna_responses_uses_two_region_zdr_fallback_and_reserves_its_price_ceiling(
    funding_source,
):
    payload = {
        "id": "gen-luna-regional",
        "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
        "output": [],
        "usage": {"cost": "0.00025"},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        provider_funding_source_for=lambda _context, provider: (
            funding_source if provider == "openrouter" else "host"
        ),
    )
    broker._price_table = luna_price_table()

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert result.status == 200
    assert result.call["actual_microusd"] == 250
    body = json.loads(transport.sent[0]["body"])
    assert body["provider"] == {
        "allow_fallbacks": True,
        "data_collection": "deny",
        "zdr": True,
        "only": ["azure/us", "azure/eu"],
        "max_price": {"prompt": 0.275, "completion": 1.32, "request": 0},
    }
    assert "order" not in body["provider"]
    assert result.call["funding_source"] == funding_source
    assert body["store"] is False and body["stream"] is False
    base_reserve = br.max_openrouter_cost_microusd(
        luna_price_table(),
        br.OPENROUTER_LUNA_RESPONSES_MODEL,
        LUNA_RESPONSES,
        max_output_tokens=256,
    )
    assert result.call["reserved_microusd"] > base_reserve
    assert result.call["reserved_microusd"] > result.call["actual_microusd"]
    assert store.calls[result.call["call_identity"]]["actual"] == 250


@pytest.mark.parametrize(
    "kind,operation_id,model",
    [
        ("score", "openrouter.responses", br.OPENROUTER_LUNA_RESPONSES_MODEL),
        ("execute", "openrouter.chat", br.OPENROUTER_LUNA_RESPONSES_MODEL),
        ("execute", "openrouter.responses", "openai/gpt-4o-mini"),
    ],
)
def test_luna_host_policy_does_not_change_scorers_chat_or_other_models(
    kind, operation_id, model
):
    assert br._openrouter_host_route(
        kind=kind,
        operation_id=operation_id,
        model=model,
        pricing=luna_price_table()["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL],
        parameters=LUNA_RESPONSES,
    ) is None


@pytest.mark.parametrize("case", ("score", "chat", "other_model"))
def test_nonmatching_public_broker_paths_keep_strict_provider_policy(case):
    operation_id = "openrouter.responses"
    parameters = dict(LUNA_RESPONSES)
    context = CONTEXT
    if case == "score":
        context = br.RunContext(**{**CONTEXT.__dict__, "kind": "score"})
    elif case == "chat":
        operation_id = "openrouter.chat"
        parameters = {**CHAT, "model": br.OPENROUTER_LUNA_RESPONSES_MODEL}
    else:
        parameters = {
            **LUNA_RESPONSES,
            "model": "openai/gpt-4o-mini",
        }
    response = {
        "id": "gen-strict-policy",
        "object": "response",
        "created_at": 1789488000,
        "status": "completed",
        "model": parameters["model"],
        "error": None,
        "output": [],
        "usage": {"cost": "0.0001"},
    }
    store = FakeLedgerStore()
    transport = FakeTransport([(200, response)])
    broker = br.Broker(
        store=store,
        key_for=lambda provider: HOST_KEYS[provider],
        price_table=luna_price_table(),
        transport=transport,
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
        judge_models=(br.OPENROUTER_LUNA_RESPONSES_MODEL,),
    )

    result = broker.execute(
        context,
        operation_id=operation_id,
        parameters=parameters,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert result.status == 200
    assert json.loads(transport.sent[0]["body"])["provider"] == dict(
        operations.OPENROUTER_STRICT_PROVIDER_POLICY
    )


def test_luna_host_policy_reserves_published_long_context_tier():
    parameters = {
        **LUNA_RESPONSES,
        "input": [
            {"role": "user", "content": "x" * 31_000} for _ in range(9)
        ],
    }
    assert br.bounded_input_tokens(parameters) >= 272_000
    route = br._openrouter_host_route(
        kind="execute",
        operation_id="openrouter.responses",
        model=br.OPENROUTER_LUNA_RESPONSES_MODEL,
        pricing=luna_price_table()["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL],
        parameters=parameters,
    )
    assert route is not None
    assert route.provider_policy["max_price"] == {
        "prompt": 0.55,
        "completion": 1.98,
        "request": 0,
    }
    assert route.reservation_pricing["prompt"] == "0.00000055"
    assert route.reservation_pricing["completion"] == "0.00000198"


@pytest.mark.parametrize(
    "bounded_tokens,expected_max_price",
    [
        (271_999, {"prompt": 0.275, "completion": 1.32, "request": 0}),
        (272_000, {"prompt": 0.55, "completion": 1.98, "request": 0}),
    ],
)
def test_luna_host_policy_long_context_tier_boundary(
    monkeypatch, bounded_tokens, expected_max_price
):
    monkeypatch.setattr(br, "bounded_input_tokens", lambda _parameters: bounded_tokens)
    route = br._openrouter_host_route(
        kind="execute",
        operation_id="openrouter.responses",
        model=br.OPENROUTER_LUNA_RESPONSES_MODEL,
        pricing=luna_price_table()["models"][br.OPENROUTER_LUNA_RESPONSES_MODEL],
        parameters=LUNA_RESPONSES,
    )
    assert route is not None
    assert route.provider_policy["max_price"] == expected_max_price


def test_luna_regional_unproved_provider_failure_keeps_full_reservation():
    broker, store, _transport = make_broker(
        transport=FakeTransport(
            [(502, {"error": {"code": 502, "message": "provider failed"}})]
        )
    )
    broker._price_table = luna_price_table()

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_canonical_failed_responses_retain_only_billing_structure_and_stay_uncertain():
    private_text = "private source passage and provider account detail"
    payload = {
        "object": "response", "status": "failed", "error_type": "rate_limit_exceeded",
        "error": {"code": "rate_limit_exceeded", "message": private_text},
        "output": [], "usage": None,
        "openrouter_metadata": {
            "requested": br.OPENROUTER_LUNA_RESPONSES_MODEL, "is_byok": False,
            "attempt": 1, "pipeline": [], "attempts": [{"status": 429, "detail": private_text}],
        },
    }
    broker, store, transport = make_broker(transport=FakeTransport([(200, payload)]))
    broker._price_table = luna_price_table()
    result = broker.execute(CONTEXT, operation_id="openrouter.responses",
                            parameters=LUNA_RESPONSES, action_sequence=0, timeout_ms=300_000)
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert len(transport.sent) == 1
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert diagnostic["openrouter_failed_response_structure"] == {
        "schema_version": 1, "usage_kind": "null", "metadata_kind": "object",
        "output_kind": "array", "output_count": 0, "rate_limit_error": True,
        "requested_model_matches": True, "is_byok": False, "attempt": 1,
        "pipeline_present": True, "pipeline_kind": "array", "pipeline_count": 0,
        "attempts_present": True, "attempts_kind": "array", "attempts_count": 1,
        "all_attempts_failed_http": True,
    }
    assert private_text not in json.dumps(diagnostic)
    assert "openrouter_failed_response_structure" not in result.call
    assert private_text.encode() not in result.body


@pytest.mark.parametrize("invalid", [True, -1, 1_000_000_001, "private numeric field", {}, []])
def test_failed_cost_structure_bounds_untrusted_scalar_fields(invalid):
    document = {
        "status": "failed", "error": {"code": "private provider error"},
        "output": [{"text": "private lead"}],
        "usage": {"cost": None, "input_tokens": invalid, "output_tokens": invalid,
                  "total_tokens": invalid, "output_tokens_details": {"reasoning_tokens": invalid}},
        "openrouter_metadata": {"requested": "private model", "is_byok": invalid,
                                "attempt": invalid, "pipeline": "private pipeline", "attempts": [{}]},
    }
    structure = br._openrouter_failed_cost_structure(document, "expected-model")
    assert structure["output_count"] == 1 and structure["rate_limit_error"] is False
    assert all(structure[name] is None for name in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "attempt"))
    assert structure["is_byok"] is (True if invalid is True else None)
    assert structure["requested_model_matches"] is False
    assert structure["pipeline_count"] is None
    assert structure["all_attempts_failed_http"] is False
    assert "private" not in json.dumps(structure)


def test_failed_cost_structure_never_infers_missing_arrays_or_success():
    structure = br._openrouter_failed_cost_structure(
        {"status": "failed", "openrouter_metadata": {}, "usage": {"output_tokens": 0}}, "model"
    )
    assert structure["output_count"] is None
    assert structure["pipeline_present"] is False and structure["pipeline_count"] is None
    assert structure["attempts_present"] is False and structure["all_attempts_failed_http"] is False
    assert structure["output_tokens"] == 0 and structure["is_byok"] is None
    for document in (None, [], {"status": "completed"}, {"status": "private status"}):
        assert br._openrouter_failed_cost_structure(document, "model") == {}


def test_failed_cost_structure_does_not_traverse_oversized_attempts():
    structure = br._openrouter_failed_cost_structure(
        {"status": "failed", "openrouter_metadata": {"attempts": [{"status": 429}] * 129}}, "model"
    )
    assert structure["attempts_count"] is None
    assert structure["all_attempts_failed_http"] is False


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
    assert store.calls[first.call["call_identity"]]["terminal"]["call_succeeded"] is False


def test_openrouter_documented_string_rate_limit_code_is_normalized():
    payload = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502
    assert result.call["provider_status"] == 429
    assert result.call["error_code"] == "provider_unavailable"
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("provider_status", [500, 503, 599])
def test_openrouter_http_200_server_error_envelope_is_infrastructure(provider_status):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, {"error": {"code": provider_status, "message": "upstream failure"}})])
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["provider_status"] == provider_status
    assert_unpriced_uncertain(result)
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
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_error_with_known_usage_settles_the_exact_charge():
    payload = {"error": {"code": 400, "message": "request rejected"}, "usage": {"cost": "0.0000091"}}
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 400
    assert result.call["actual_microusd"] == 10
    assert store.calls[result.call["call_identity"]]["actual"] == 10
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False


def _insured_openrouter_error():
    return {
        "error": {"code": 502, "message": "Provider returned an error"},
        "openrouter_metadata": {
            "requested": "openai/gpt-4o-mini",
            "is_byok": False,
            "attempt": 1,
            "attempts": [{"provider": "OpenAI", "status": 502}],
        },
        "user_id": "harmless-documented-field",
    }


def test_openrouter_plain_non_byok_502_uses_insured_zero_and_keeps_provider_error():
    payload = _insured_openrouter_error()
    broker, store, transport = make_broker(
        transport=FakeTransport([(502, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "openrouter_zero_completion_insurance_error_20260911"
    assert store.log == ["reserve", "dispatch", "settle"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["provider_cost"] == {
        "basis": "openrouter_zero_completion_insurance_error_20260911",
        "units": "0",
        "unit_name": "usd",
        "operation": "openrouter.chat",
    }
    assert transport.sent[0]["headers"]["x-openrouter-metadata"] == "enabled"


def test_openrouter_http_200_body_502_uses_insured_zero_and_keeps_raw_provider_error():
    payload = _insured_openrouter_error()
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["provider_status"] == 502
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "openrouter_zero_completion_insurance_error_20260911"
    assert store.log == ["reserve", "dispatch", "settle"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["provider_cost"]["basis"] == (
        "openrouter_zero_completion_insurance_error_20260911"
    )
    assert transport.sent[0]["headers"]["x-openrouter-metadata"] == "enabled"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "error": {
                "code": 500,
                "message": "upstream failure",
                "metadata": {"error_code": 502},
            },
            "openrouter_metadata": {
                "requested": "openai/gpt-4o-mini",
                "is_byok": False,
                "attempt": 1,
                "attempts": [{"provider": "OpenAI", "status": 500}],
            },
        },
        {
            "metadata": {"error_code": 502},
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        },
    ],
)
def test_openrouter_http_200_ignores_arbitrary_nested_error_code(payload):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert_unpriced_uncertain(result)
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize("model", ("openai/gpt-5.6-sol", "openai/gpt-5.5"))
@pytest.mark.parametrize("unproven", (None, "request_fee", "byok", "pipeline", "missing_metadata"))
def test_pydantic_model_error_insurance_keeps_cost_uncertain_without_full_proof(model, unproven):
    # The submitted harness uses GPT-5.6 Sol, not the small model in CHAT.
    table = price_table()
    table["models"][model] = dict(
        table["models"]["openai/gpt-4o-mini"],
        prompt="0.000002",
        completion="0.00001",
        web_search="0.01",
        request="0.01" if unproven == "request_fee" else "0",
    )
    payload = {
        "error": {"code": 502, "message": "Provider returned an error"},
        "openrouter_metadata": {
            "requested": model,
            "is_byok": unproven == "byok",
            "attempt": 1,
            "attempts": [{"provider": "OpenAI", "status": 502}],
        },
    }
    if unproven == "pipeline":
        payload["openrouter_metadata"]["pipeline"] = [{"type": "web-search"}]
    if unproven == "missing_metadata":
        del payload["openrouter_metadata"]
    store = FakeLedgerStore()
    transport = FakeTransport([(502, payload)])
    broker = br.Broker(
        store=store,
        key_for=lambda provider: HOST_KEYS[provider],
        price_table=table,
        transport=transport,
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=dict(CHAT, model=model, reasoning={"enabled": True}),
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert transport.sent[0]["headers"]["x-openrouter-metadata"] == "enabled"
    assert len(transport.sent) == 1  # Never repeat the paid inference request.
    if unproven is None:
        assert result.call["actual_microusd"] == 0
        assert result.call["cost_basis"] == "openrouter_zero_completion_insurance_error_20260911"
        assert store.log == ["reserve", "dispatch", "settle"]
    else:
        assert_unpriced_uncertain(result)
        assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_insurance_accepts_valid_function_tool_history_without_assistant_content():
    payload = {
        "error": {"code": 502, "message": "Provider returned an error"},
        "openrouter_metadata": {
            "requested": "openai/gpt-4o-mini",
            "is_byok": False,
            "attempt": 1,
            "attempts": [{"provider": "OpenAI", "status": 502}],
        },
    }
    parameters = dict(
        CHAT,
        messages=[
            {"role": "user", "content": "Find the company"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup_company",
                            "arguments": '{"domain":"example.com"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"company":"Example"}',
            },
        ],
    )
    broker, store, _transport = make_broker(
        transport=FakeTransport([(502, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=parameters,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "openrouter_zero_completion_insurance_error_20260911"
    assert store.log == ["reserve", "dispatch", "settle"]


def test_openrouter_502_with_nonzero_separate_request_price_stays_uncertain():
    model = "anthropic/claude-3.5-haiku"
    payload = {
        "error": {"code": 502, "message": "Provider returned an error"},
        "openrouter_metadata": {
            "requested": model,
            "is_byok": False,
            "attempt": 1,
            "attempts": [{"provider": "Anthropic", "status": 502}],
        },
    }
    broker, store, _transport = make_broker(
        transport=FakeTransport([(502, payload)])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=dict(CHAT, model=model),
        action_sequence=0,
        timeout_ms=30000,
    )
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_exact_generation_cost_precedes_insured_error():
    generation_id = "gen-test-123"
    error = _insured_openrouter_error()
    generation = {
        "data": {
            "id": generation_id,
            "total_cost": "0.001234",
            "usage": "0.001234",
        }
    }
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [
                (502, error, {"X-Generation-Id": generation_id}),
                (200, generation),
            ]
        )
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502 and result.call["actual_microusd"] == 1234
    assert result.call["cost_basis"] == "openrouter_generation_cost"
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]
    assert transport.sent[1]["url"] == br.OPENROUTER_GENERATION_URL + generation_id
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["provider_cost"]["request_id"] == generation_id


def test_openrouter_valid_generation_header_allows_strict_insured_error_after_404s(
    monkeypatch,
):
    monkeypatch.setattr(br.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(br.time, "sleep", lambda _seconds: None)
    generation_id = "gen-insured-after-readback"
    error = _insured_openrouter_error()
    not_found = (404, {"error": {"code": 404, "message": "not found"}})
    transport = FakeTransport(
        [(502, error, {"X-Generation-Id": generation_id})]
        + [not_found] * br._OPENROUTER_BILLING_MAX_ATTEMPTS
    )
    broker, store, _transport = make_broker(transport=transport)

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == (
        "openrouter_zero_completion_insurance_error_20260911"
    )
    assert [sent["method"] for sent in transport.sent] == ["POST"] + [
        "GET"
    ] * br._OPENROUTER_BILLING_MAX_ATTEMPTS
    assert transport.responses == []
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize(
    "responses",
    [
        [
            (
                502,
                {"error": {"code": 502, "message": "Provider returned an error"}, "id": "gen-body"},
                {"X-Generation-Id": "gen-header"},
            )
        ],
        [
            (
                502,
                _insured_openrouter_error(),
                {"X-Generation-Id": "invalid/generation/id"},
            )
        ],
        [
            (
                502,
                _insured_openrouter_error(),
                {
                    "X-Generation-Id": "gen-header-one",
                    "x-generation-id": "gen-header-two",
                },
            )
        ],
        [
            (
                502,
                {"error": {"code": 502, "message": "Provider returned an error"}},
                {"X-Generation-Id": "gen-readback-fails"},
            ),
            (401, {"error": {"code": 401, "message": "not authorized"}}),
        ],
    ],
    ids=(
        "conflicting_body_and_header_ids",
        "invalid_header_id",
        "conflicting_header_ids",
        "valid_header_with_unproven_error",
    ),
)
def test_openrouter_generation_conflict_or_failed_readback_stays_uncertain(responses):
    broker, store, _transport = make_broker(transport=FakeTransport(responses))
    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    if len(responses) == 1:
        assert "openrouter_generation_id" not in diagnostic
        assert "credential_fingerprint" not in diagnostic
    else:
        assert diagnostic["openrouter_generation_id"] == "gen-readback-fails"
        assert diagnostic["credential_fingerprint"] == br._credential_fingerprint(KEY)


@pytest.mark.parametrize("exact_cost", ("0", "0.001234", "1.25"))
def test_delayed_exact_generation_settles_after_restart_without_second_post(
    monkeypatch, exact_cost
):
    generation_id = "gen-delayed-after-inline-window"
    error = {"error": {"code": 502, "message": "Provider returned an error"}}
    monkeypatch.setattr(br, "_openrouter_generation_readback", lambda **_kwargs: None)
    first_broker, store, first_transport = make_broker(
        transport=FakeTransport(
            [(502, error, {"X-Generation-Id": generation_id})]
        )
    )

    first = first_broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )
    diagnostic = store.calls[first.call["call_identity"]]["uncertain_doc"]
    candidate = {
        "uncertain_entry_id": 1,
        "round_id": "arena-2026-09-02",
        "run_id": CONTEXT.run_id,
        "submission_id": CONTEXT.submission_id,
        "miner_hotkey": CONTEXT.miner_hotkey,
        "assignment_id": CONTEXT.assignment_id,
        "stage": CONTEXT.stage,
        "icp_position": CONTEXT.icp_position,
        "attempt": CONTEXT.attempt,
        "kind": CONTEXT.kind,
        "call_identity": first.call["call_identity"],
        "generation_id": diagnostic["openrouter_generation_id"],
        "credential_fingerprint": diagnostic["credential_fingerprint"],
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-02T01:20:00Z",
    }
    second_transport = FakeTransport(
        [
            (
                200,
                {
                    "data": {
                        "id": generation_id,
                        "total_cost": exact_cost,
                        "usage": exact_cost,
                    }
                },
            )
        ]
    )
    restarted_broker, _same_store, _ = make_broker(
        store=store, transport=second_transport
    )

    reconciled = restarted_broker.reconcile_openrouter_cost(candidate)

    assert first.call["outcome"] == "uncertain"
    assert [request["method"] for request in first_transport.sent] == ["POST"]
    assert [request["method"] for request in second_transport.sent] == ["GET"]
    assert reconciled["status"] == "settled"
    assert store.calls[first.call["call_identity"]]["actual"] == (
        br.provider_costs.openrouter_generation_cost(
            {
                "data": {
                    "id": generation_id,
                    "total_cost": exact_cost,
                    "usage": exact_cost,
                }
            },
            generation_id=generation_id,
        ).microusd
    )


def test_luna_regional_failure_reconciles_exact_cost_after_restart_without_second_post(
    monkeypatch,
):
    generation_id = "gen-luna-regional-delayed"
    monkeypatch.setattr(br, "_openrouter_generation_readback", lambda **_kwargs: None)
    first_broker, store, first_transport = make_broker(
        transport=FakeTransport(
            [
                (
                    502,
                    {"error": {"code": 502, "message": "provider failed"}},
                    {"X-Generation-Id": generation_id},
                )
            ]
        )
    )
    first_broker._price_table = luna_price_table()
    parameters = {
        **LUNA_RESPONSES,
        "input": "x" * 30_000,
        "max_output_tokens": 10_000,
    }

    first = first_broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=parameters,
        action_sequence=0,
        timeout_ms=300_000,
    )
    diagnostic = store.calls[first.call["call_identity"]]["uncertain_doc"]
    candidate = {
        "uncertain_entry_id": 1,
        "round_id": "arena-2026-09-02",
        "run_id": CONTEXT.run_id,
        "submission_id": CONTEXT.submission_id,
        "miner_hotkey": CONTEXT.miner_hotkey,
        "assignment_id": CONTEXT.assignment_id,
        "stage": CONTEXT.stage,
        "icp_position": CONTEXT.icp_position,
        "attempt": CONTEXT.attempt,
        "kind": CONTEXT.kind,
        "call_identity": first.call["call_identity"],
        "generation_id": diagnostic["openrouter_generation_id"],
        "credential_fingerprint": diagnostic["credential_fingerprint"],
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-02T01:20:00Z",
    }
    second_transport = FakeTransport(
        [(200, {"data": {"id": generation_id, "total_cost": "0.0116"}})]
    )
    restarted_broker, _same_store, _ = make_broker(
        store=store, transport=second_transport
    )
    restarted_broker._price_table = luna_price_table()

    reconciled = restarted_broker.reconcile_openrouter_cost(candidate)

    assert first.call["outcome"] == "uncertain"
    assert first.call["reserved_microusd"] > 11_600
    assert [request["method"] for request in first_transport.sent] == ["POST"]
    assert [request["method"] for request in second_transport.sent] == ["GET"]
    assert reconciled["status"] == "settled"
    assert store.calls[first.call["call_identity"]]["actual"] == 11_600


def test_stream_timeout_generation_settles_exact_cost_without_second_post(caplog):
    generation_id = "gen-stream-timeout-recovery"
    methods = []
    stream = _TrickleUntilCancelled()

    def timed_out_post(request):
        methods.append(request.method)
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer " + KEY
        return httpx.Response(
            200,
            headers={"X-Generation-Id": generation_id},
            stream=stream,
        )

    first_transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(timed_out_post)
    )
    first_broker, store, _ = make_broker(transport=first_transport)
    caplog.set_level(logging.DEBUG)
    try:
        first = first_broker.execute(
            CONTEXT,
            operation_id="openrouter.chat",
            parameters=CHAT,
            action_sequence=0,
            timeout_ms=60,
        )
    finally:
        first_transport.close()

    diagnostic = store.calls[first.call["call_identity"]]["uncertain_doc"]
    assert first.status == 502 and first.call["outcome"] == "uncertain"
    assert diagnostic == {
        "reason": "missing_provider_cost",
        "call_succeeded": False,
        "provider_status": 0,
        "body_bytes": 0,
        "body_is_mapping": False,
        "usage_present": False,
        "billing_present": False,
        "openrouter_generation_id": generation_id,
        "credential_fingerprint": br._credential_fingerprint(KEY),
        "transport_failure": True,
        "transport_error_class": "ReadTimeout",
        "observed_provider_status": 200,
        }
    assert methods == ["POST"]
    assert stream.closed.is_set()
    assert KEY not in caplog.text
    assert KEY not in json.dumps(store.calls)

    replay = first_broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=60,
    )
    assert replay.status == 409 and replay.call["outcome"] == "uncertain"
    assert methods == ["POST"]

    candidate = {
        "uncertain_entry_id": 1,
        "round_id": "arena-2026-09-02",
        "run_id": CONTEXT.run_id,
        "submission_id": CONTEXT.submission_id,
        "miner_hotkey": CONTEXT.miner_hotkey,
        "assignment_id": CONTEXT.assignment_id,
        "stage": CONTEXT.stage,
        "icp_position": CONTEXT.icp_position,
        "attempt": CONTEXT.attempt,
        "kind": CONTEXT.kind,
        "call_identity": first.call["call_identity"],
        "generation_id": diagnostic["openrouter_generation_id"],
        "credential_fingerprint": diagnostic["credential_fingerprint"],
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-02T01:20:00Z",
    }

    def exact_generation_get(request):
        methods.append(request.method)
        assert request.method == "GET"
        assert str(request.url) == br.OPENROUTER_GENERATION_URL + generation_id
        assert request.headers["authorization"] == "Bearer " + KEY
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": generation_id,
                    "total_cost": "0.001234",
                    "usage": "0.001234",
                }
            },
        )

    second_transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(exact_generation_get)
    )
    restarted_broker, _same_store, _ = make_broker(
        store=store, transport=second_transport
    )
    try:
        reconciled = restarted_broker.reconcile_openrouter_cost(candidate)
    finally:
        second_transport.close()

    assert methods == ["POST", "GET"]
    assert reconciled == {
        "status": "settled",
        "idempotent": False,
        "actual_microusd": 1234,
    }
    assert store.calls[first.call["call_identity"]]["actual"] == 1234
    assert KEY not in caplog.text


def test_non_openrouter_transport_failure_ignores_generation_identity():
    class Transport:
        def send(self, **_kwargs):
            raise br.ProviderTransportError(
                "ReadTimeout",
                openrouter_generation_id="gen-must-be-ignored",
            )

    broker, store, _ = make_broker(transport=Transport())
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "fintech"}},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "transport_failure",
        "credential_fingerprint": br._credential_fingerprint(DL_KEY),
        "deepline_request_id": "ctx-tool-" + result.call["call_identity"][7:39],
        "deepline_operation": "exa_search",
        "transport_error_class": "ReadTimeout",
        "call_succeeded": False,
    }


def test_delayed_generation_requires_the_original_credential_without_a_get():
    store = FakeLedgerStore()
    transport = FakeTransport()
    broker, _store, _ = make_broker(store=store, transport=transport)
    candidate = {
        "uncertain_entry_id": 1,
        "round_id": "arena-2026-09-02",
        "run_id": CONTEXT.run_id,
        "submission_id": CONTEXT.submission_id,
        "miner_hotkey": CONTEXT.miner_hotkey,
        "assignment_id": CONTEXT.assignment_id,
        "stage": CONTEXT.stage,
        "icp_position": CONTEXT.icp_position,
        "attempt": CONTEXT.attempt,
        "kind": CONTEXT.kind,
        "call_identity": contracts.document_hash("call"),
        "generation_id": "gen-other-key",
        "credential_fingerprint": br._credential_fingerprint(
            "sk-or-v1-" + "z" * 40
        ),
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-02T01:20:00Z",
    }

    assert broker.reconcile_openrouter_cost(candidate) == {
        "status": "credential_mismatch"
    }
    assert transport.sent == []


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
    assert_unpriced_uncertain(result)
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
    {"error_type": [], "error": {"code": "invalid_prompt"}},
    {"error": {"code": 399, "message": "non-error status"}},
    {"choices": [{"finish_reason": "error", "error": {"code": 600}}]},
    {"error": {"code": 429}, "choices": [{"finish_reason": "error", "error": {"code": 502}}]},
])
def test_openrouter_http_200_malformed_error_envelope_fails_closed(payload):
    broker, store, _transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert_unpriced_uncertain(result)
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
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is True


def test_openrouter_completion_text_that_describes_an_error_is_still_successful():
    payload = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"error":{"code":502}}',
            },
            "finish_reason": "stop",
        }],
        "usage": {"cost": "0.000001"},
    }
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, payload)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30000,
    )

    assert result.status == 200
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is True


@pytest.mark.parametrize(("payload", "expected_status"), [
    ({"choices": []}, 200),
    ({"usage": {"prompt_tokens": "x", "completion_tokens": 1}}, 502),
    ({"usage": {"prompt_tokens": -1, "completion_tokens": 1}}, 502),
    ({"model": "other/model", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 502),
    ({"usage": {"prompt_tokens": 10 ** 9, "completion_tokens": 10 ** 9}}, 502),
])
def test_missing_malformed_or_wrong_model_usage_keeps_execute_result_when_valid(
    payload, expected_status
):
    broker, store, transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == expected_status
    assert_unpriced_uncertain(result)
    if expected_status == 200:
        assert json.loads(result.body) == payload
    else:
        assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "call_succeeded": isinstance(payload.get("choices"), list),
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
    broker, store, transport = make_broker(
        store=FakeLedgerStore(per_icp_quota=0),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    refused = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert refused.status == 402 and json.loads(refused.body) == {"error": {"code": "budget_refused"}}
    assert refused.call["outcome"] == "refused" and refused.call["reason"] == "per_icp_quota"
    assert transport.sent == []
    again = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert again.status == 402 and store.log == ["reserve", "reserve"]


def test_budget_refusal_preserves_only_database_proven_miner_credential_error():
    store = FakeLedgerStore(per_icp_quota=0)
    original_reserve = store.reserve_call

    def reserve_with_proof(**kwargs):
        result = original_reserve(**kwargs)
        result["prior_miner_credential_refusal"] = True
        return result

    store.reserve_call = reserve_with_proof
    broker, _store, transport = make_broker(
        store=store,
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    refused = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert refused.status == 402
    assert json.loads(refused.body) == {
        "error": {"code": "miner_credentials_unavailable"}
    }
    assert refused.call["reason"] == "per_icp_quota"
    assert transport.sent == []


@pytest.mark.parametrize(
    ("credential_proven", "expected_code"),
    [
        (False, "provider_unavailable"),
        (True, "miner_credentials_unavailable"),
    ],
)
def test_uncertain_provider_cost_refusal_is_infrastructure_unless_credentials_proven(
    credential_proven,
    expected_code,
):
    store = FakeLedgerStore(per_icp_quota=0)
    original_reserve = store.reserve_call

    def reserve_with_uncertain_cost(**kwargs):
        result = original_reserve(**kwargs)
        result["reason"] = "provider_cost_uncertain"
        result["prior_miner_credential_refusal"] = credential_proven
        return result

    store.reserve_call = reserve_with_uncertain_cost
    broker, _store, transport = make_broker(
        store=store,
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    refused = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert json.loads(refused.body) == {"error": {"code": expected_code}}
    assert refused.call["reason"] == "provider_cost_uncertain"
    assert transport.sent == []


def test_transport_failure_after_send_marks_uncertain_and_keeps_full_reservation():
    broker, store, transport = make_broker(transport=FakeTransport(fail=True))
    result = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert_unpriced_uncertain(result, reserved=10_000_000)
    assert store.openrouter_capacity == 0
    # A later identical request neither re-sends nor releases the reservation.
    late = broker.execute(CONTEXT, operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "x"}}, action_sequence=0, timeout_ms=1000)
    assert late.status == 409 and json.loads(late.body) == {"error": {"code": "call_uncertain"}} and len([request for request in transport.sent if request["method"] == "POST"]) == 1


def test_successful_deepline_reply_without_billing_returns_sanitized_execute_result():
    body = b'{"job_id":"job-empty","status":"completed","results":[]}'
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, body)])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and result.body == body
    assert_unpriced_uncertain(result, reserved=10_000_000)
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "credential_fingerprint": br._credential_fingerprint(DL_KEY),
        "deepline_request_id": "ctx-tool-" + result.call["call_identity"][7:39],
        "call_succeeded": True,
        "provider_status": 200,
        "body_bytes": len(body),
        "body_is_mapping": True,
        "usage_present": False,
        "billing_present": False,
        "top_level_job_status": "completed",
        "deepline_job_id": "job-empty",
        "deepline_operation": "exa_search",
    }


def test_zero_reservation_unknown_call_replays_without_post_but_distinct_call_runs():
    bodies = [
        b'{"job_id":"job-zero-a","status":"completed","results":[]}',
        b'{"job_id":"job-zero-b","status":"completed","results":[]}',
    ]
    class TwoUnknownReplies(FakeTransport):
        def __init__(self):
            super().__init__()
            self.post_bodies = list(bodies)

        def send(self, **kwargs):
            self.sent.append({
                "method": kwargs["method"],
                "url": kwargs["url"],
                "headers": dict(kwargs["headers"]),
                "body": kwargs["body"],
                "timeout": kwargs["timeout_seconds"],
            })
            payload = (
                self.post_bodies.pop(0)
                if kwargs["method"] == "POST"
                else b'{"data":[]}'
            )
            return br.ProviderResponse(
                200, {"content-type": "application/json"}, payload
            )

    store = ZeroReservationLedgerStore()
    broker, _store, transport = make_broker(
        store=store, transport=TwoUnknownReplies()
    )
    arguments = {
        "operation_id": "deepline.execute",
        "parameters": {"tool": "exa_search", "payload": {"query": "x"}},
        "timeout_ms": 1000,
    }

    first = broker.execute(CONTEXT, action_sequence=0, **arguments)
    replay = broker.execute(CONTEXT, action_sequence=0, **arguments)
    distinct = broker.execute(CONTEXT, action_sequence=1, **arguments)

    assert first.status == distinct.status == 200
    assert first.body == bodies[0] and distinct.body == bodies[1]
    assert_unpriced_uncertain(first, reserved=0)
    assert_unpriced_uncertain(distinct, reserved=0)
    assert replay.status == 409
    assert json.loads(replay.body) == {"error": {"code": "call_uncertain"}}
    assert first.call["call_identity"] != distinct.call["call_identity"]
    assert [request["method"] for request in transport.sent].count("POST") == 2
    assert {call["kind"] for call in store.calls.values()} == {"uncertain"}


@pytest.mark.parametrize("kind", ("execute", "score"))
def test_completed_missing_bill_requires_durable_uncertain_state(kind):
    class RejectUncertainStore(ZeroReservationLedgerStore):
        def mark_uncertain(self, **kwargs):
            self.log.append("uncertain_rejected")
            return {"status": "stale"}

    body = b'{"job_id":"job-rejected","status":"completed","results":[]}'
    store = RejectUncertainStore()
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, body)])
    )
    context = br.RunContext(**{**CONTEXT.__dict__, "kind": kind})
    result = broker.execute(
        context,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert_unpriced_uncertain(result, reserved=0)
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "dispatch" and "uncertain_doc" not in call
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]


def test_score_completed_reply_without_bill_returns_sanitized_result_once():
    body = b'{"job_id":"job-score","status":"completed","results":[]}'
    context = br.RunContext(**{**CONTEXT.__dict__, "kind": "score"})
    store = ZeroReservationLedgerStore()
    broker, store, transport = make_broker(
        store=store,
        transport=FakeTransport([(200, body)])
    )
    arguments = dict(
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    result = broker.execute(context, **arguments)
    replay = broker.execute(context, **arguments)

    assert result.status == 200 and result.body == body
    assert_unpriced_uncertain(result, reserved=0)
    assert replay.status == 409
    assert json.loads(replay.body) == {"error": {"code": "call_uncertain"}}
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain"
    assert call["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "credential_fingerprint": br._credential_fingerprint(DL_KEY),
        "deepline_request_id": "ctx-tool-" + result.call["call_identity"][7:39],
        "call_succeeded": True,
        "provider_status": 200,
        "body_bytes": len(body),
        "body_is_mapping": True,
        "usage_present": False,
        "billing_present": False,
        "top_level_job_status": "completed",
        "deepline_job_id": "job-score",
        "deepline_operation": "exa_search",
    }
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]


@pytest.mark.parametrize(
    ("body", "expected_methods"),
    (
        (b'{"status":"completed","results":[]}', ["POST"]),
        (
            b'{"job_id":"job-score-failed","status":"failed",'
            b'"error":{"code":"PROVIDER_FAILURE"}}',
            ["POST", "GET"],
        ),
    ),
    ids=("invalid_without_job_identity", "failed"),
)
def test_score_missing_bill_does_not_return_invalid_or_failed_reply(
    body, expected_methods
):
    context = br.RunContext(**{**CONTEXT.__dict__, "kind": "score"})
    store = ZeroReservationLedgerStore()
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, body)])
    )
    result = broker.execute(
        context,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert_unpriced_uncertain(result, reserved=0)
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain"
    assert call["uncertain_doc"]["call_succeeded"] is False
    assert [request["method"] for request in transport.sent] == expected_methods


def test_deepline_422_recovers_exact_failed_zero_from_billing_history():
    envelope = {
        "request_id": "iad1::bad-input",
        "requestId": "iad1::bad-input",
        "error": {"code": "UPSTREAM_BAD_INPUT"},
    }
    history = deepline_history({
        "request_id": "iad1::bad-input", "operation": "exa_search",
        "provider": "exa", "charge_state": "failed", "status": "error",
        "credits": 0, "delta": 0,
    })
    broker, store, transport = make_broker(
        transport=FakeTransport([(422, envelope), (200, history)])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 422 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    assert result.call["cost_basis"] == "deepline_billing_history_failed_zero"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]


def test_deepline_payment_refusal_releases_dynamic_reservation_as_zero():
    envelope = {
        "code": "INSUFFICIENT_CREDITS",
        "error": "Insufficient credits",
        "billing": {
            "kind": "insufficient_credits",
            "required_credits": 5,
            "balance_credits": 4.14,
            "needed_credits": 0.86,
        },
    }
    store = FakeLedgerStore(openrouter_capacity=73_321_638)
    broker, store, transport = make_broker(
        store=store, transport=FakeTransport([(402, envelope)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == 73_321_638
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_payment_required_error_zero"
    assert store.openrouter_capacity == 73_321_638
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


def test_deepline_payment_refusal_with_unknown_billing_remains_uncertain():
    envelope = {
        "error": {"code": "payment_required"},
        "billing": {"credits_charged": "unknown"},
    }
    store = FakeLedgerStore(openrouter_capacity=73_321_638)
    broker, store, _transport = make_broker(
        store=store, transport=FakeTransport([(402, envelope)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 502
    assert_unpriced_uncertain(result, reserved=73_321_638)
    assert store.openrouter_capacity == 0
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize("provider_status", [422, 502, 503])
def test_deepline_error_native_billing_settles_known_positive_charge(
    provider_status,
):
    envelope = {
        "request_id": "iad1::paid-error",
        "error": {"code": "UPSTREAM_BAD_INPUT"},
        "billing": {"credits_charged": 0.2},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(provider_status, envelope)])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == (provider_status if provider_status == 422 else 502)
    assert result.call["actual_microusd"] == 20_000
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


@pytest.mark.parametrize(
    ("provider_status", "credits", "charge_state", "expected_microusd"),
    [(502, 0, "failed", 0), (503, 0.3, "posted", 30_000)],
)
def test_deepline_5xx_recovers_exact_charge_from_billing_history(
    provider_status, credits, charge_state, expected_microusd
):
    request_id = "iad1::server-error"
    envelope = {"requestId": request_id, "error": {"code": "upstream_error"}}
    history_entry = deepline_history_entry(
        request_id, "exa_search", credits, charge_state=charge_state
    )
    if charge_state == "failed":
        history_entry.update({"status": "error", "delta": 0})
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(provider_status, envelope), (200, deepline_history(history_entry))]
        )
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["provider_status"] == provider_status
    assert result.call["actual_microusd"] == expected_microusd
    assert store.log == ["reserve", "dispatch", "settle"]
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]


@pytest.mark.parametrize(
    "envelope",
    [
        {"error": {"code": "upstream_error"}},
        {"request_id": "a", "requestId": "b", "error": {}},
        {"request_id": "expected", "error": {}},
    ],
)
def test_deepline_5xx_without_exact_final_charge_remains_uncertain(envelope):
    wrong_history = deepline_history(
        deepline_history_entry("different", "exa_search", 0.2)
    )
    broker, store, _transport = make_broker(
        transport=FakeTransport([(502, envelope), (200, wrong_history)])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize(
    ("tool_error", "top_level_id", "expected_methods"),
    [
        ({"requestId": "provider-or-deepline", "operation": "exa_search"}, None, ["POST"]),
        ({"requestId": "nested-conflict", "operation": "exa_search"}, "top-level-id", ["POST", "GET"]),
        ({"requestId": "nested-wrong-operation", "operation": "exa_contents"}, None, ["POST"]),
    ],
)
def test_deepline_tool_error_request_id_is_not_used_as_billing_identity(
    tool_error, top_level_id, expected_methods
):
    envelope = {"error": {"code": "upstream_error"}, "tool_error": tool_error}
    if top_level_id is not None:
        envelope["request_id"] = top_level_id
    broker, store, transport = make_broker(
        transport=FakeTransport([(502, envelope)])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 502
    assert_unpriced_uncertain(result)
    assert [sent["method"] for sent in transport.sent] == expected_methods
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    if top_level_id is None:
        assert "deepline_job_id" not in diagnostic
    else:
        assert diagnostic["deepline_job_id"] == top_level_id
    assert "provider-or-deepline" not in json.dumps(diagnostic)
    assert "nested-conflict" not in json.dumps(diagnostic)
    assert "nested-wrong-operation" not in json.dumps(diagnostic)


@pytest.mark.parametrize(
    "provenance", ["response_too_large", "redirect_rejected"]
)
def test_deepline_synthetic_transport_response_keeps_full_liability_and_provenance(provenance):
    class SyntheticTransport(FakeTransport):
        def send(self, **kwargs):
            self.sent.append(kwargs)
            return br.ProviderResponse(
                502, {"content-type": "application/json"},
                operations.GENERIC_UNAVAILABLE_BODY, provenance,
            )

    broker, store, transport = make_broker(transport=SyntheticTransport([]))
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 502
    assert_unpriced_uncertain(result)
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert diagnostic["response_provenance"] == provenance
    assert [sent["method"] for sent in transport.sent] == ["POST"]
    replay = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert replay.status == 409
    assert [sent["method"] for sent in transport.sent] == ["POST"]


def test_deepline_credential_echo_provenance_contains_no_secret():
    envelope = {"error": {"code": "upstream_error"}, "echo": DL_KEY}
    broker, store, transport = make_broker(
        transport=FakeTransport([(502, envelope)])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    diagnostic = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert diagnostic["response_provenance"] == "credential_echo"
    assert DL_KEY not in json.dumps(diagnostic)
    assert DL_KEY not in result.body.decode()
    assert len(transport.sent) == 1


def test_deepline_hunter_no_bill_502_without_job_id_settles_zero_and_returns_provider_error():
    broker, store, transport = make_broker(
        transport=FakeTransport(
            [(502, {"error": {"code": "upstream_error"}, "detail": {"rows": 1}})]
        )
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "hunter_discover", "payload": {"domain": "example.com"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_hunter_discover_error_zero"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert [sent["method"] for sent in transport.sent] == ["POST"]


def test_deepline_hunter_no_bill_401_settles_zero_and_stays_credential_error():
    broker, store, _transport = make_broker(
        transport=FakeTransport([(401, {"error": {"code": "auth"}})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "hunter_discover", "payload": {"domain": "example.com"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 402
    assert json.loads(result.body) == {
        "error": {"code": "miner_credentials_unavailable"}
    }
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize(
    ("tool", "body"),
    [
        ("hunter_discover", {"error": {}, "billing": None}),
        ("exa_search", {"error": {"code": "upstream_error"}}),
    ],
)
def test_deepline_hunter_malformed_billing_and_nonfree_502_stay_uncertain(tool, body):
    broker, store, _transport = make_broker(transport=FakeTransport([(502, body)]))
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": tool, "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize("provider_status", [401, 402, 403])
def test_deepline_credential_error_without_charge_stays_uncertain_and_typed(
    provider_status,
):
    envelope = {"request_id": "iad1::bad-key", "error": {"code": "auth"}}
    broker, store, _transport = make_broker(
        transport=FakeTransport(
            [(provider_status, envelope), (200, deepline_history())]
        ),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 402
    assert json.loads(result.body) == {
        "error": {"code": "miner_credentials_unavailable"}
    }
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize(
    "envelope",
    [
        {"request_id": "a", "requestId": "b", "error": {}},
        {"job_id": "a", "request_id": "b", "error": {}},
        {"request_id": "a", "error": {}},
    ],
)
def test_deepline_error_without_exact_final_charge_remains_uncertain(envelope):
    broker, store, _transport = make_broker(
        transport=FakeTransport([(422, envelope), (200, deepline_history())])
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0, timeout_ms=1000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]


@pytest.mark.parametrize("status", [{"message": "not a status"}, ["completed"], "secret-shaped-unrecognized-status"])
def test_missing_cost_diagnostics_never_store_arbitrary_job_status(status):
    body = json.dumps({"status": status}).encode()
    diagnostic = br._missing_provider_cost_call_doc(
        br.ProviderResponse(200, {}, body), {"status": status},
        call_succeeded=False,
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


def test_real_deepline_generic_http_without_billing_settles_verified_zero():
    envelope = {
        "job_id": "iad1::generic-http",
        "result": {"status": 200, "body": "example"},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "generic_http_request",
            "payload": {"url": "https://example.com/", "method": "GET"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and json.loads(result.body) == envelope
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_generic_http_request_completed_zero"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


def test_real_deepline_contextdev_scrape_without_billing_settles_verified_zero():
    envelope = {
        "job_id": "iad1::contextdev-scrape",
        "result": {"data": {
            "success": True,
            "url": "https://example.com/about",
            "markdown": "# Example",
            "contentLength": 9,
            "metadata": {
                "sourceUrl": "https://example.com/about",
                "finalUrl": "https://example.com/about",
            },
        }},
        "status": "completed",
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, json.dumps(envelope).encode("utf-8"))])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "contextdev_get_web_scrape_markdown",
            "payload": {"url": "https://example.com/about"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )
    assert result.status == 200 and json.loads(result.body) == envelope
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_contextdev_web_scrape_markdown_completed_zero"
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
    assert result.status == 200 and result.body == raw
    assert_unpriced_uncertain(result, reserved=0)
    assert store.calls[result.call["call_identity"]]["uncertain_doc"] == {
        "reason": "missing_provider_cost",
        "credential_fingerprint": br._credential_fingerprint(DL_KEY),
        "deepline_request_id": "ctx-tool-" + result.call["call_identity"][7:39],
        "call_succeeded": True,
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
    assert 0 < transport.sent[1]["timeout"] <= (
        operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    )


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


def test_deepline_large_unrelated_history_group_releases_dynamic_reservation():
    job_id = "iad1::free-history-after-large-group"
    envelope = {
        "job_id": job_id,
        "result": {"data": {"results": []}},
        "status": "completed",
    }
    unrelated = deepline_history_entry(
        "iad1::unrelated", "exa_search", 0.2
    )
    unrelated["metadata"] = {
        "chargeGroupIds": ["unrelated-%d" % index for index in range(147)]
    }
    exact_free = deepline_history_entry(job_id, "exa_search", 0)
    exact_free["charge_state"] = "free"
    openrouter_response = {
        "id": "gen-after-free",
        "model": "openai/gpt-4o-mini",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": "0.000009"},
    }
    store = FakeLedgerStore(openrouter_capacity=20_000)
    broker, store, transport = make_broker(
        store=store,
        transport=FakeTransport(
            [
                (200, json.dumps(envelope).encode()),
                (200, deepline_history(unrelated, exact_free)),
                (200, openrouter_response),
            ]
        ),
    )

    deepline_result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )
    openrouter_result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=1,
        timeout_ms=30_000,
    )

    assert deepline_result.status == 200
    assert deepline_result.call["reserved_microusd"] == 20_000
    assert deepline_result.call["actual_microusd"] == 0
    assert deepline_result.call["outcome"] == "settled"
    assert openrouter_result.status == 200
    assert openrouter_result.call["outcome"] == "settled"
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET", "POST"]
    assert store.log == ["reserve", "dispatch", "settle"] * 2


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


def test_deepline_completed_poll_can_return_inner_pending_and_still_succeed():
    envelope = {
        "job_id": "iad1::poll-request",
        "status": "completed",
        "result": {"data": {"jobId": "external-job", "status": "pending"}},
        "billing": {"credits_charged": 0.01},
    }
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, envelope)])
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={
            "tool": "harvestapi_get_job",
            "payload": {"jobId": "external-job"},
        },
        action_sequence=0,
        timeout_ms=1000,
    )

    assert result.status == 200
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is True


@pytest.mark.parametrize(
    ("provider_status", "envelope", "expected_call_succeeded"),
    [
        (
            200,
            {
                "status": "completed",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
            False,
        ),
        (
            200,
            {
                "job_id": "iad1::pending-job",
                "status": "pending",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
            False,
        ),
        (
            201,
            {
                "job_id": "iad1::created-job",
                "status": "completed",
                "result": [],
                "billing": {"credits_charged": 0.56},
            },
            True,
        ),
    ],
)
def test_deepline_native_billing_requires_http_200_completed_job(
    provider_status, envelope, expected_call_succeeded
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
    assert result.status == (provider_status if expected_call_succeeded else 502)
    assert_unpriced_uncertain(result)
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert (
        store.calls[result.call["call_identity"]]["uncertain_doc"]["call_succeeded"]
        is expected_call_succeeded
    )


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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
def test_deepline_history_refresh_keeps_reconciliation_time_bound(monkeypatch, request_seconds):
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
        operation="exa_search", reconciliation_deadline=request_seconds,
    )
    assert cost is None
    assert elapsed[0] == min(request_seconds, 30.0)
    assert 1 <= len(transport.timeouts) <= br._DEEPLINE_BILLING_MAX_ATTEMPTS
    assert all(0 < value <= min(request_seconds, 30.0) for value in transport.timeouts)


def test_deepline_reconciles_after_paid_request_uses_full_provider_window(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    job_id = "iad1::full-window-exa-contents"

    class FullWindowProvider(FakeTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            if kwargs["method"] == "POST":
                elapsed[0] += kwargs["timeout_seconds"]
            return response

    envelope = {
        "job_id": job_id,
        "status": "completed",
        "result": {"data": {"requestId": "exa-request", "results": []}},
        "billing": None,
    }
    exact_free_history = deepline_history(
        deepline_history_entry(
            job_id, "exa_contents", 0, charge_state="free", provider="exa"
        )
    )
    broker, store, transport = make_broker(
        transport=FullWindowProvider(
            [(200, envelope), (200, exact_free_history)]
        )
    )

    result = broker.execute(
        CONTEXT,
        operation_id="exa.contents",
        parameters={
            "ids": ["https://example.com/news"],
            "text": {"maxCharacters": 12000},
            "maxAgeHours": 0,
        },
        action_sequence=0,
        timeout_ms=30_000,
    )

    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_billing_history_credits_x_0.10_usd"
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]
    assert transport.sent[0]["timeout"] == pytest.approx(30.0)
    assert transport.sent[1]["timeout"] == pytest.approx(
        operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    )
    assert elapsed[0] == pytest.approx(30.0)
    assert store.log == ["reserve", "dispatch", "settle"]


def test_deepline_unresolved_reconciliation_stays_uncertain_after_bounded_extra_window(
    monkeypatch,
):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    job_id = "iad1::full-window-history-missing"

    class FullWindowProvider(FakeTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            if kwargs["method"] == "POST":
                elapsed[0] += kwargs["timeout_seconds"]
            return response

    envelope = {
        "job_id": job_id,
        "status": "completed",
        "result": {"data": {"results": []}},
        "billing": None,
    }
    broker, store, transport = make_broker(
        transport=FullWindowProvider(
            [(200, envelope)]
            + [(200, deepline_history())] * br._DEEPLINE_BILLING_MAX_ATTEMPTS
        )
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=30_000,
    )

    assert result.status == 200
    assert_unpriced_uncertain(result)
    assert elapsed[0] == pytest.approx(
        30.0 + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    )
    assert [sent["method"] for sent in transport.sent].count("POST") == 1
    assert [sent["method"] for sent in transport.sent].count("GET") > 1
    assert store.log == ["reserve", "dispatch", "uncertain"]


def test_openrouter_reconciles_after_paid_request_uses_full_provider_window(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        br.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
    )
    generation_id = "gen-full-window"

    class FullWindowProvider(FakeTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            if kwargs["method"] == "POST":
                elapsed[0] += kwargs["timeout_seconds"]
            return response

    broker, store, transport = make_broker(
        transport=FullWindowProvider(
            [
                (
                    502,
                    {"error": {"code": 502, "message": "upstream failure"}},
                    {"X-Generation-Id": generation_id},
                ),
                (
                    200,
                    {
                        "data": {
                            "id": generation_id,
                            "total_cost": "0.001234",
                            "usage": "0.001234",
                        }
                    },
                ),
            ]
        )
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=30_000,
    )

    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 1234
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]
    assert transport.sent[0]["timeout"] == pytest.approx(30.0)
    assert transport.sent[1]["timeout"] == pytest.approx(
        operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    )
    assert elapsed[0] == pytest.approx(30.0)
    assert store.log == ["reserve", "dispatch", "settle"]


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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
    assert result.status == 200
    assert_unpriced_uncertain(result)
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
    assert result.status == 502
    assert_unpriced_uncertain(result, reserved=54_321)
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


def test_dynamic_deepline_returns_proved_billing_hold_without_short_poll(monkeypatch):
    class HeldLedger(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            self.log.append("reserve")
            return {
                "status": "budget_busy",
                "reason": "provider_cost_uncertain",
                "idempotent": False,
                "call_identity": kwargs["call_identity"],
            }

    sleeps = []
    monkeypatch.setattr(br.time, "sleep", sleeps.append)
    broker, store, transport = make_broker(
        store=HeldLedger(openrouter_capacity=8_765)
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 502
    assert result.call["outcome"] == "not_dispatched"
    assert result.call["reason"] == "provider_cost_uncertain"
    assert result.call["idempotent"] is False
    assert store.log == ["reserve"] and sleeps == [] and transport.sent == []


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
    class CrashBeforeFirstDispatch(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.crash = True

        def mark_dispatched(self, **kwargs):
            if self.crash:
                self.crash = False
                raise ArenaStoreUnavailable("synthetic crash before dispatch")
            return super().mark_dispatched(**kwargs)

    broker, store, transport = make_broker(
        store=CrashBeforeFirstDispatch(),
        transport=FakeTransport([(200, {"results": []})]),
    )
    identity_args = dict(operation_id="deepline.execute", parameters={"tool": "exa_search", "payload": {"query": "crash"}}, action_sequence=5, timeout_ms=1000)
    request_hash = contracts.document_hash(operations.validate_operation_request("deepline.execute", {"tool": "exa_search", "payload": {"query": "crash"}}))
    identity = contracts.provider_call_identity(attempt=1, assignment_id=CONTEXT.assignment_id, icp_position=0, action_sequence=5, operation_id="deepline.execute", request_hash=request_hash)
    with pytest.raises(ArenaStoreUnavailable):
        broker.execute(CONTEXT, **identity_args)
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
    assert store.calls[first.call["call_identity"]]["terminal"]["call_succeeded"] is True
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


@pytest.mark.parametrize("commits_before_timeout", [False, True])
def test_reservation_transport_timeout_uses_exact_idempotent_readback_once(
    commits_before_timeout,
):
    class AmbiguousReservationStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.reserve_attempts = 0

        def reserve_call(self, **kwargs):
            self.reserve_attempts += 1
            if self.reserve_attempts == 1:
                if commits_before_timeout:
                    super().reserve_call(**kwargs)
                raise ArenaStoreUnavailable("synthetic reservation response loss")
            return super().reserve_call(**kwargs)

    store = AmbiguousReservationStore()
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, {"results": []})])
    )
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=29,
        timeout_ms=1000,
    )

    assert result.status == 200
    assert store.reserve_attempts == 2
    assert store.log.count("reserve") == (2 if commits_before_timeout else 1)
    assert store.log.count("dispatch") == 1
    assert store.log.count("settle") == 1
    assert [request["method"] for request in transport.sent].count("POST") == 1
    persisted = store.calls[result.call["call_identity"]]
    assert persisted["kind"] == "settlement"
    assert persisted["actual"] == result.call["actual_microusd"] == 0
    independent = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "independent"}},
        action_sequence=32,
        timeout_ms=1000,
    )
    assert independent.status == 200
    assert independent.call["call_identity"] != result.call["call_identity"]
    assert [request["method"] for request in transport.sent].count("POST") == 2
    assert store.log.count("settle") == 2


@pytest.mark.parametrize("kind", ("execute", "score"))
def test_zero_fixed_reservation_response_loss_matches_execute_and_score(kind):
    class CommitThenLoseStore(ZeroReservationLedgerStore):
        def __init__(self):
            super().__init__()
            self.first = True

        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            if self.first:
                self.first = False
                raise ArenaStoreUnavailable("synthetic reservation response loss")
            return result

    company = {
        "name": "Example",
        "website": "https://example.com",
        "linkedinUrl": "https://www.linkedin.com/company/example/",
        "employeeCountRange": {"start": 2, "end": 10},
        "employeeCount": 6,
    }
    response = {
        "job_id": "zero-fixed-response-loss",
        "status": "completed",
        "result": {"data": {"status": 200, "element": company}},
        "billing": {"credits_charged": 0.03, "cost_usd": 0.003},
    }
    store = CommitThenLoseStore()
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, response)])
    )
    context = br.RunContext(**{**CONTEXT.__dict__, "kind": kind})
    result = broker.execute(
        context,
        operation_id="deepline.execute",
        parameters={
            "tool": "harvestapi_get_company",
            "payload": {"url": company["linkedinUrl"]},
        },
        action_sequence=38,
        timeout_ms=1000,
    )

    assert result.status == 200
    call = next(iter(store.calls.values()))
    assert call["amount"] == 0
    assert result.call["reserved_microusd"] == 0
    assert result.call["actual_microusd"] == 3_000
    assert call["kind"] == "settlement"
    assert [request["method"] for request in transport.sent] == ["POST"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "other-run"),
        ("call_identity", "sha256:" + "0" * 64),
        ("operation_id", "scrapingdog.google"),
        ("provider", "scrapingdog"),
        ("funding_source", "miner_key"),
        ("amount_microusd", 1),
        ("entry_doc", {"request_hash": "wrong"}),
    ],
)
@pytest.mark.parametrize("kind", ("execute", "score"))
def test_zero_fixed_reservation_readback_still_requires_exact_binding(
    field, replacement, kind
):
    class CorruptCommitThenLoseStore(ZeroReservationLedgerStore):
        def __init__(self):
            super().__init__()
            self.first = True

        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            if self.first:
                self.first = False
                raise ArenaStoreUnavailable("synthetic reservation response loss")
            return result

        def list_ledger(self, **kwargs):
            rows = super().list_ledger(**kwargs)
            rows[0][field] = replacement
            return rows

    store = CorruptCommitThenLoseStore()
    broker, _store, transport = make_broker(store=store)
    context = br.RunContext(**{**CONTEXT.__dict__, "kind": kind})
    result = broker.execute(
        context,
        operation_id="deepline.execute",
        parameters={
            "tool": "harvestapi_get_company",
            "payload": {"url": "https://www.linkedin.com/company/example/"},
        },
        action_sequence=39,
        timeout_ms=1000,
    )

    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert transport.sent == []
    assert store.log.count("dispatch") == 0


def test_persistent_reservation_transport_failure_never_dispatches_provider():
    class DownStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.reserve_attempts = 0

        def reserve_call(self, **kwargs):
            self.reserve_attempts += 1
            raise ArenaStoreUnavailable("synthetic database unavailable")

    store = DownStore()
    broker, _store, transport = make_broker(store=store, transport=FakeTransport([]))
    with pytest.raises(ArenaStoreUnavailable):
        broker.execute(
            CONTEXT,
            operation_id="deepline.execute",
            parameters={"tool": "exa_search", "payload": {"query": "x"}},
            action_sequence=30,
            timeout_ms=1000,
        )
    assert store.reserve_attempts == 2
    assert store.calls == {} and store.log == [] and transport.sent == []


@pytest.mark.parametrize(
    ("reserve_result", "expected_status", "expected_code"),
    [
        ({"status": "stale"}, 409, "lease_stale"),
        (
            {"status": "refused", "reason": "per_icp_quota"},
            402,
            "budget_refused",
        ),
    ],
)
def test_reservation_response_loss_preserves_nonreservation_terminal_status(
    reserve_result, expected_status, expected_code,
):
    class TerminalAfterLossStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.reserve_attempts = 0
            self.readbacks = 0

        def reserve_call(self, **kwargs):
            self.reserve_attempts += 1
            if self.reserve_attempts == 1:
                raise ArenaStoreUnavailable("synthetic reservation response loss")
            return dict(reserve_result)

        def list_ledger(self, **kwargs):
            self.readbacks += 1
            return super().list_ledger(**kwargs)

    store = TerminalAfterLossStore()
    broker, _store, transport = make_broker(store=store)
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=35,
        timeout_ms=1000,
    )

    assert result.status == expected_status
    assert json.loads(result.body) == {"error": {"code": expected_code}}
    assert store.reserve_attempts == 2 and store.readbacks == 0
    assert transport.sent == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "other-run"),
        ("operation_id", "scrapingdog.google"),
        ("provider", "scrapingdog"),
        ("funding_source", "miner_key"),
        ("amount_microusd", -1),
        ("entry_doc", {"request_hash": "wrong"}),
    ],
)
def test_ambiguous_reservation_readback_rejects_binding_mismatch(
    field, replacement,
):
    class MismatchedReadbackStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.first = True

        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            if self.first:
                self.first = False
                raise ArenaStoreUnavailable("synthetic response loss")
            return result

        def list_ledger(self, **kwargs):
            rows = super().list_ledger(**kwargs)
            rows[0][field] = replacement
            return rows

    store = MismatchedReadbackStore()
    broker, _store, transport = make_broker(store=store, transport=FakeTransport([]))
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=33,
        timeout_ms=1000,
    )
    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert transport.sent == [] and store.log.count("dispatch") == 0


def test_luna_regional_replay_rejects_old_lower_reservation_before_dispatch(
    monkeypatch,
):
    class CrashBeforeDispatchStore(FakeLedgerStore):
        crash = True

        def mark_dispatched(self, **kwargs):
            if self.crash:
                self.log.append("dispatch")
                raise ArenaStoreUnavailable("synthetic crash before dispatch")
            return super().mark_dispatched(**kwargs)

    store = CrashBeforeDispatchStore()
    old_broker, _store, old_transport = make_broker(store=store)
    old_broker._price_table = luna_price_table()
    regional_route = br._openrouter_host_route
    monkeypatch.setattr(br, "_openrouter_host_route", lambda **_kwargs: None)
    with pytest.raises(ArenaStoreUnavailable):
        old_broker.execute(
            CONTEXT,
            operation_id="openrouter.responses",
            parameters=LUNA_RESPONSES,
            action_sequence=0,
            timeout_ms=300_000,
        )
    old_reservation = next(iter(store.calls.values()))["amount"]
    assert old_transport.sent == []

    monkeypatch.setattr(br, "_openrouter_host_route", regional_route)
    store.crash = False
    new_broker, _store, new_transport = make_broker(store=store)
    new_broker._price_table = luna_price_table()
    result = new_broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert result.call["reserved_microusd"] > old_reservation
    assert new_transport.sent == []
    assert store.log.count("dispatch") == 1


def test_luna_regional_replay_resumes_matching_reservation_once():
    class CrashBeforeDispatchStore(FakeLedgerStore):
        crash = True

        def mark_dispatched(self, **kwargs):
            if self.crash:
                self.log.append("dispatch")
                raise ArenaStoreUnavailable("synthetic crash before dispatch")
            return super().mark_dispatched(**kwargs)

    response = {
        "id": "gen-luna-replay",
        "object": "response",
        "created_at": 1789488000,
        "status": "completed",
        "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
        "error": None,
        "output": [],
        "usage": {"cost": "0.00025"},
    }
    store = CrashBeforeDispatchStore()
    first_broker, _store, first_transport = make_broker(store=store)
    first_broker._price_table = luna_price_table()
    with pytest.raises(ArenaStoreUnavailable):
        first_broker.execute(
            CONTEXT,
            operation_id="openrouter.responses",
            parameters=LUNA_RESPONSES,
            action_sequence=0,
            timeout_ms=300_000,
        )
    assert first_transport.sent == []

    store.crash = False
    second_broker, _store, second_transport = make_broker(
        store=store, transport=FakeTransport([(200, response)])
    )
    second_broker._price_table = luna_price_table()
    result = second_broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert result.status == 200 and result.call["actual_microusd"] == 250
    assert len(second_transport.sent) == 1
    assert store.log.count("dispatch") == 2


def test_luna_regional_rollout_replays_old_settlement_without_second_post(
    monkeypatch,
):
    response = {
        "id": "gen-luna-old-route-settled",
        "object": "response",
        "created_at": 1789488000,
        "status": "completed",
        "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
        "error": None,
        "output": [],
        "usage": {"cost": "0.0001"},
    }
    store = FakeLedgerStore()
    old_broker, _store, old_transport = make_broker(
        store=store, transport=FakeTransport([(200, response)])
    )
    old_broker._price_table = luna_price_table()
    regional_route = br._openrouter_host_route
    monkeypatch.setattr(br, "_openrouter_host_route", lambda **_kwargs: None)
    first = old_broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )
    assert first.status == 200 and len(old_transport.sent) == 1

    monkeypatch.setattr(br, "_openrouter_host_route", regional_route)
    new_broker, _store, new_transport = make_broker(store=store)
    new_broker._price_table = luna_price_table()
    replay = new_broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=LUNA_RESPONSES,
        action_sequence=0,
        timeout_ms=300_000,
    )

    assert replay.status == 200
    assert replay.call["outcome"] == "settled"
    assert replay.call["idempotent"] is True
    assert replay.body == first.body
    assert new_transport.sent == []


@pytest.mark.parametrize("corruption", ["non_mapping", "wrong_second_identity", "unordered"])
def test_ambiguous_reservation_readback_rejects_malformed_ledger_chain(corruption):
    class CorruptLedgerStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.first = True

        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            if self.first:
                self.first = False
                raise ArenaStoreUnavailable("synthetic response loss")
            return result

        def list_ledger(self, **kwargs):
            rows = super().list_ledger(**kwargs)
            extra = dict(rows[0], entry_id=2, entry_kind="dispatch")
            rows.append(extra)
            if corruption == "non_mapping":
                rows.append("invalid")
            elif corruption == "wrong_second_identity":
                rows[1]["call_identity"] = "sha256:" + "0" * 64
            else:
                rows[1]["entry_id"] = rows[0]["entry_id"]
            return rows

    store = CorruptLedgerStore()
    broker, _store, transport = make_broker(store=store)
    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "x"}},
        action_sequence=36,
        timeout_ms=1000,
    )

    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert transport.sent == [] and store.log.count("dispatch") == 0


def test_duplicate_same_identity_has_one_atomic_dispatch_and_one_paid_post():
    class RacingStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.reserved = threading.Barrier(2)
            self.dispatched = threading.Barrier(2)

        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            self.reserved.wait(timeout=2)
            return result

        def mark_dispatched(self, **kwargs):
            result = super().mark_dispatched(**kwargs)
            self.dispatched.wait(timeout=2)
            return result

    store = RacingStore()
    transports = [FakeTransport([(200, {"results": []})]) for _ in range(2)]
    brokers = [make_broker(store=store, transport=transport)[0]
               for transport in transports]
    results = []
    failures = []

    def execute(broker):
        try:
            results.append(broker.execute(
                CONTEXT,
                operation_id="deepline.execute",
                parameters={"tool": "exa_search", "payload": {"query": "same"}},
                action_sequence=31,
                timeout_ms=1000,
            ))
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=execute, args=(broker,)) for broker in brokers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not failures and all(not thread.is_alive() for thread in threads)
    assert sorted(result.status for result in results) == [200, 409]
    uncertain = next(result for result in results if result.status == 409)
    assert json.loads(uncertain.body) == {"error": {"code": "call_uncertain"}}
    assert sum(len(transport.sent) for transport in transports) == 1
    assert store.log.count("dispatch") == 2
    assert store.log.count("settle") == 1
    assert len(store.calls) == 1
    call = next(iter(store.calls.values()))
    assert call["kind"] == "settlement" and call["actual"] == 0


def test_cancelled_settlement_uses_the_store_authoritative_failed_delivery():
    class CancelledStore(FakeLedgerStore):
        def settle_call(self, **kwargs):
            call = self.calls[kwargs["call_identity"]]
            self.log.append("settle")
            call["kind"] = "uncertain"
            call["uncertain_doc"] = {
                "reason": "round_cancelled",
                "call_succeeded": False,
            }
            return {"status": "stale"}

    store = CancelledStore()
    broker, _store, transport = make_broker(
        store=store,
        transport=FakeTransport([(200, {"results": []})]),
    )

    result = broker.execute(
        CONTEXT,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "cancel"}},
        action_sequence=12,
        timeout_ms=1000,
    )

    persisted = store.calls[result.call["call_identity"]]
    assert result.status == 409
    assert json.loads(result.body) == {"error": {"code": "lease_stale"}}
    assert persisted["uncertain_doc"] == {
        "reason": "round_cancelled",
        "call_succeeded": False,
    }
    assert "terminal" not in persisted
    assert [sent["method"] for sent in transport.sent] == ["POST"]


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
    assert_unpriced_uncertain(result)
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
        "reason": "missing_provider_cost",
        "call_succeeded": False,
        "failure_stage": "response_adaptation",
        "error_class": "OperationResponseError",
        "provider_status": 200,
        "body_is_mapping": False,
        "billing_present": False,
        "usage_present": False,
        "body_bytes": 29,
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
    call_doc = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert call_doc["reason"] == "settle_failure"
    assert call_doc["failure_stage"] == "settlement"
    assert call_doc["error_class"] == "ArenaContractError"
    assert call_doc["call_succeeded"] is True
    assert call_doc["known_actual_microusd"] == 0
    assert call_doc["deepline_request_id"] == transport.sent[0]["headers"]["x-deepline-request-id"]
    assert call_doc["deepline_operation"] == "exa_search"
    store.settle_call = original


@pytest.mark.parametrize(
    ("fault", "expected_head", "expected_sends"),
    [
        ("none", "settlement", 1),
        ("no_commit", "reservation", 0),
        ("committed", "dispatch", 0),
        ("concurrent_settlement", "settlement", 0),
    ],
)
def test_dispatch_response_loss_never_redispatches_or_rewrites_durable_state(
    fault, expected_head, expected_sends
):
    class DispatchResponseLossStore(FakeLedgerStore):
        dispatch_attempts = 0

        def mark_dispatched(self, **kwargs):
            self.dispatch_attempts += 1
            if fault == "no_commit" and self.dispatch_attempts == 1:
                raise ArenaStoreUnavailable("synthetic response loss before commit")
            result = super().mark_dispatched(**kwargs)
            if fault in {"committed", "concurrent_settlement"} and self.dispatch_attempts == 1:
                if fault == "concurrent_settlement":
                    self.settle_call(
                        **kwargs,
                        actual_microusd=0,
                        terminal_response={
                            "status": 200,
                            "headers": {"content-type": "application/json"},
                            "body_b64": base64.b64encode(b'{"results":[]}').decode(),
                            "call_succeeded": True,
                        },
                        lease_ttl_seconds=1200,
                    )
                raise ArenaStoreUnavailable("synthetic committed response loss")
            return result

    store = DispatchResponseLossStore()
    broker, _store, transport = make_broker(
        store=store,
        transport=FakeTransport([(200, {"results": []})]),
    )
    execute = lambda: broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "dispatch loss"}},
        action_sequence=17, timeout_ms=1000,
    )
    if fault == "none":
        assert execute().status == 200
    else:
        with pytest.raises(ArenaStoreUnavailable):
            execute()
    assert len(transport.sent) == expected_sends
    assert next(iter(store.calls.values()))["kind"] == expected_head


def test_dispatch_response_loss_does_not_start_a_new_readback_or_provider_send():
    class NoReadbackStore(FakeLedgerStore):
        reserve_attempts = 0

        def reserve_call(self, **kwargs):
            self.reserve_attempts += 1
            if self.reserve_attempts > 1:
                raise AssertionError("unexpected dispatch-loss readback")
            return super().reserve_call(**kwargs)

        def mark_dispatched(self, **kwargs):
            super().mark_dispatched(**kwargs)
            raise ArenaStoreUnavailable("synthetic committed response loss")

    store = NoReadbackStore()
    broker, _store, transport = make_broker(store=store)
    with pytest.raises(ArenaStoreUnavailable, match="committed response loss"):
        broker.execute(
            CONTEXT,
            operation_id="deepline.execute",
            parameters={"tool": "exa_search", "payload": {"query": "dispatch loss"}},
            action_sequence=18,
            timeout_ms=1000,
        )
    assert transport.sent == []
    assert store.reserve_attempts == 1
    assert next(iter(store.calls.values()))["kind"] == "dispatch"


def test_duplicate_lost_idempotent_dispatch_response_does_not_rewrite_active_owner():
    mark_barrier = threading.Barrier(2)
    provider_started = threading.Event()
    release_provider = threading.Event()

    class DuplicateStore(FakeLedgerStore):
        def mark_dispatched(self, **kwargs):
            mark_barrier.wait(timeout=5)
            result = super().mark_dispatched(**kwargs)
            if result.get("idempotent") is True:
                raise ArenaStoreUnavailable(
                    "synthetic duplicate dispatch response loss"
                )
            return result

    class BlockingTransport(FakeTransport):
        def send(self, **kwargs):
            provider_started.set()
            assert release_provider.wait(timeout=5)
            return super().send(**kwargs)

    store = DuplicateStore()
    brokers = [
        make_broker(
            store=store,
            transport=BlockingTransport([(200, {"results": []})]),
        )[0]
        for _ in range(2)
    ]
    results, failures = [], []

    def execute(candidate):
        try:
            results.append(candidate.execute(
                CONTEXT,
                operation_id="deepline.execute",
                parameters={"tool": "exa_search", "payload": {"query": "duplicate"}},
                action_sequence=19,
                timeout_ms=1000,
            ))
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=execute, args=(candidate,)) for candidate in brokers]
    for thread in threads:
        thread.start()
    assert provider_started.wait(timeout=5)
    for _ in range(100):
        if failures:
            break
        time.sleep(0.01)
    assert len(failures) == 1
    assert isinstance(failures[0], ArenaStoreUnavailable)
    assert next(iter(store.calls.values()))["kind"] == "dispatch"
    assert "uncertain" not in store.log
    release_provider.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1 and results[0].status == 200
    assert next(iter(store.calls.values()))["kind"] == "settlement"


def test_store_settlement_retry_reuses_paid_reply_and_releases_the_budget():
    envelope = {
        "job_id": "settle-retry-job", "status": "completed",
        "result": {"data": {"results": []}},
        "billing": {"credits_charged": 0.02},
    }
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, envelope), (200, envelope)])
    )
    original = store.settle_call
    attempts = []

    def one_store_failure(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise br.ArenaStoreError("transient ledger RPC failure")
        return original(**kwargs)

    store.settle_call = one_store_failure
    parameters = {"tool": "exa_search", "payload": {"query": "synthetic"}}
    first = broker.execute(
        CONTEXT, operation_id="deepline.execute", parameters=parameters,
        action_sequence=0, timeout_ms=5000,
    )
    assert first.status == 200 and first.call["outcome"] == "settled"
    assert first.call["actual_microusd"] == 2_000
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len(transport.sent) == 1 and store.log == ["reserve", "dispatch", "settle"]
    assert store.openrouter_capacity == 10_000_000 - 2_000

    next_context = br.RunContext(**{
        **CONTEXT.__dict__, "run_id": "next-icp-run",
        "icp_position": CONTEXT.icp_position + 1,
    })
    second = broker.execute(
        next_context, operation_id="deepline.execute", parameters=parameters,
        action_sequence=0, timeout_ms=5000,
    )
    assert second.status == 200 and second.call["outcome"] == "settled"
    assert len(transport.sent) == 2  # exactly one paid POST per distinct ICP


@pytest.mark.parametrize(
    ("operation_id", "parameters", "reply", "actual"),
    (
        (
            "openrouter.chat", CHAT,
            {"model": "openai/gpt-4o-mini", "choices": [],
             "usage": {"cost": "0.0000123"}},
            13,
        ),
        (
            "scrapingdog.scrape", {"url": "https://example.com/about"},
            b"<html>synthetic page</html>", 250,
        ),
    ),
)
def test_other_provider_settlement_retries_only_the_store(
    operation_id, parameters, reply, actual,
):
    broker, store, transport = make_broker(transport=FakeTransport([(200, reply)]))
    original = store.settle_call
    attempts = []

    def one_store_failure(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise br.ArenaStoreError("transient settlement RPC failure")
        return original(**kwargs)

    store.settle_call = one_store_failure
    result = broker.execute(
        CONTEXT, operation_id=operation_id, parameters=parameters,
        action_sequence=0, timeout_ms=5000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == actual
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len(transport.sent) == 1
    assert store.calls[result.call["call_identity"]]["actual"] == actual


def test_store_settlement_lost_reply_accepts_only_the_same_committed_terminal():
    envelope = {
        "job_id": "lost-settle-reply", "status": "completed",
        "result": {"data": {"results": []}},
        "billing": {"credits_charged": 0.02},
    }
    broker, store, transport = make_broker(transport=FakeTransport([(200, envelope)]))
    original = store.settle_call
    attempts = []

    def lost_reply(**kwargs):
        attempts.append(kwargs)
        result = original(**kwargs)
        if len(attempts) == 1:
            raise br.ArenaStoreError("settlement reply lost after commit")
        return result

    store.settle_call = lost_reply
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "synthetic"}},
        action_sequence=0, timeout_ms=5000,
    )
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert store.log == ["reserve", "dispatch", "settle", "settle"]
    assert len(transport.sent) == 1
    assert store.openrouter_capacity == 10_000_000 - 2_000


def test_store_settlement_retry_rejects_a_different_idempotent_terminal():
    envelope = {
        "job_id": "conflicting-settle-reply", "status": "completed",
        "result": {"data": {"results": []}},
        "billing": {"credits_charged": 0.02},
    }
    broker, store, transport = make_broker(transport=FakeTransport([(200, envelope)]))
    attempts = []

    def conflicting_reply(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise br.ArenaStoreError("settlement reply lost")
        return {
            "status": "settled", "idempotent": True,
            "amount_microusd": kwargs["actual_microusd"] + 1,
            "terminal_response": kwargs["terminal_response"],
        }

    store.settle_call = conflicting_reply
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "synthetic"}},
        action_sequence=0, timeout_ms=5000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert len(attempts) == 2 and len(transport.sent) == 1
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert store.openrouter_capacity == 0


def test_persistent_store_settlement_failure_keeps_the_known_receipt_uncertain():
    envelope = {
        "job_id": "unsettled-known-receipt", "status": "completed",
        "result": {"data": {"results": []}},
        "billing": {"credits_charged": 0.02},
    }
    broker, store, transport = make_broker(transport=FakeTransport([(200, envelope)]))
    attempts = []

    def unavailable_store(**kwargs):
        attempts.append(kwargs)
        raise br.ArenaStoreError("ledger RPC unavailable")

    store.settle_call = unavailable_store
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": "synthetic"}},
        action_sequence=0, timeout_ms=5000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert len(attempts) == br._SETTLEMENT_STORE_MAX_ATTEMPTS
    assert all(attempt == attempts[0] for attempt in attempts)
    assert len(transport.sent) == 1 and store.log[-1] == "uncertain"
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain" and store.openrouter_capacity == 0
    doc = call["uncertain_doc"]
    assert doc["reason"] == "settle_failure"
    assert doc["failure_stage"] == "settlement" and doc["error_class"] == "ArenaStoreError"
    assert doc["call_succeeded"] is True and doc["provider_status"] == 200
    assert doc["known_actual_microusd"] == 2_000
    assert doc["deepline_request_id"] == transport.sent[0]["headers"]["x-deepline-request-id"]
    assert doc["deepline_operation"] == "exa_search"
    assert doc["credential_fingerprint"].startswith("sha256:")
    assert doc["deepline_job_id"] == envelope["job_id"]
    assert doc["provider_cost"] == attempts[0]["terminal_response"]["provider_cost"]
    assert "terminal_response" not in doc and DL_KEY not in json.dumps(doc)


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
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
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
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
