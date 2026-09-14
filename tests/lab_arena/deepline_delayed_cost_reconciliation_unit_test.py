"""Broker and service behavior for exact Deepline ledger reconciliation."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from lab_arena import broker as br
from lab_arena import service as svc
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT,
    DL_KEY,
    FakeLedgerStore,
    make_broker,
)


ROUND_ID = "arena-2026-09-14"
CALL_IDENTITY = "sha256:" + "b" * 64
REQUEST_ID = "ctx-tool-" + CALL_IDENTITY.removeprefix("sha256:")[:32]
SECRET = "deepline-secret"
FINGERPRINT = br._credential_fingerprint(SECRET)
RESERVATION_AT = "2026-09-14T00:43:55.521000Z"


def _candidate(**patch):
    candidate = {
        "uncertain_entry_id": 41,
        "round_id": ROUND_ID,
        "run_id": "run-1",
        "submission_id": "submission-1",
        "miner_hotkey": "miner-1",
        "assignment_id": "assignment-1",
        "stage": 1,
        "icp_position": 0,
        "attempt": 1,
        "kind": "score",
        "call_identity": CALL_IDENTITY,
        "request_id": REQUEST_ID,
        "operation": "firecrawl_scrape",
        "credential_fingerprint": FINGERPRINT,
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-14T00:50:00Z",
        "reservation_at": RESERVATION_AT,
        "uncertain_at": "2026-09-14T00:44:02.000000Z",
    }
    candidate.update(patch)
    return candidate


def _ledger_entry(*, credits=0.03):
    return {
        "id": "ledger-row-1",
        "delta": -credits,
        "reason": "charge_settle",
        "provider": "firecrawl",
        "operation": "firecrawl_scrape",
        "request_id": REQUEST_ID,
        "billing_stage": "posted",
        "charge_state": "posted",
        "billing_mode": "post_deduct",
        "pricing_model": "per_page",
        "pricing_basis": "page",
        "charge_credits": credits,
        "metadata": {
            "requestId": REQUEST_ID,
            "chargeGroupId": REQUEST_ID,
            "operation": "firecrawl_scrape",
            "provider": "firecrawl",
            "billingStage": "posted",
            "billingMode": "post_deduct",
            "pricingModel": "per_page",
            "postedCredits": credits,
        },
        "billing_audit": {
            "request_id": REQUEST_ID,
            "charge_group_id": REQUEST_ID,
            "operation": "firecrawl_scrape",
            "provider": "firecrawl",
            "billing_stage": "posted",
            "charge_state": "posted",
            "billing_mode": "post_deduct",
            "pricing_model": "per_page",
            "pricing_basis": "page",
            "charge_credits": credits,
        },
    }


def _score_context():
    return br.RunContext(
        **{
            **CONTEXT.__dict__,
            "kind": "score",
            "round_id": ROUND_ID,
        }
    )


class LostFirecrawlResponseTransport:
    def __init__(self, *, billed_credits):
        self.billed_credits = billed_credits
        self.request_id = ""
        self.sent = []

    def send(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs["method"] == "POST" and "code.deepline.com" in kwargs["url"]:
            self.request_id = kwargs["headers"]["x-deepline-request-id"]
            assert self.request_id.startswith("ctx-tool-")
            assert len(self.request_id) == len("ctx-tool-") + 32
            raise br.ProviderTransportError("ReadTimeout")
        if kwargs["method"] == "GET":
            entry = _ledger_entry(credits=self.billed_credits)
            entry["request_id"] = self.request_id
            entry["metadata"]["requestId"] = self.request_id
            entry["metadata"]["chargeGroupId"] = self.request_id
            entry["billing_audit"]["request_id"] = self.request_id
            entry["billing_audit"]["charge_group_id"] = self.request_id
            return br.ProviderResponse(
                200,
                {"content-type": "application/json"},
                json.dumps({"entries": [entry], "has_more": False}).encode(),
            )
        raise AssertionError("unexpected provider request")


def test_transport_loss_uses_pre_dispatch_caller_id_and_exact_ledger_cost():
    transport = LostFirecrawlResponseTransport(billed_credits=0.02)
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, store, _ = make_broker(
        store=store,
        transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )

    result = broker.execute(
        _score_context(),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=30_000,
    )

    assert [request["method"] for request in transport.sent] == ["POST", "GET"]
    assert result.status == 502
    assert result.call["outcome"] == "settled"
    assert result.call["reserved_microusd"] == 49_945_650
    assert result.call["actual_microusd"] == 2_000
    assert store.openrouter_capacity == 49_943_650
    assert store.log == ["reserve", "dispatch", "settle"]
    retained = store.calls[result.call["call_identity"]]
    assert retained["call_doc"]["deepline_request_id"] == transport.request_id
    assert retained["call_doc"]["credential_fingerprint"] == (
        br._credential_fingerprint(DL_KEY)
    )
    assert retained["actual"] == 2_000
    exposed = json.dumps(result.to_document())
    assert transport.request_id not in exposed
    assert retained["call_doc"]["credential_fingerprint"] not in exposed


class UnresolvedThenOpenRouterTransport:
    def __init__(self):
        self.sent = []

    def send(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs["method"] == "POST" and "code.deepline.com" in kwargs["url"]:
            raise br.ProviderTransportError("ReadTimeout")
        if kwargs["method"] == "GET":
            # Invalid evidence fails closed immediately and retains the exact
            # verified ceiling. It cannot become an assumed zero charge.
            return br.ProviderResponse(200, {}, b'{"entries":"invalid"}')
        if kwargs["method"] == "POST" and "openrouter.ai" in kwargs["url"]:
            return br.ProviderResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "id": "generation-after-deepline-loss",
                        "model": "openai/gpt-4o-mini",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "cost": "0.00001",
                        },
                    }
                ).encode(),
            )
        raise AssertionError("unexpected provider request")


def test_unresolved_dynamic_firecrawl_cost_retains_full_reservation():
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    transport = UnresolvedThenOpenRouterTransport()
    broker, _, _ = make_broker(
        store=store,
        transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )

    uncertain = broker.execute(
        _score_context(),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=30_000,
    )
    assert uncertain.status == 502
    assert uncertain.call["outcome"] == "uncertain"
    assert uncertain.call["reserved_microusd"] == 49_945_650
    assert store.openrouter_capacity == 0
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]


class ReconciliationStore:
    def __init__(self):
        self.calls = []

    def reconcile_deepline_cost(self, **kwargs):
        self.calls.append(kwargs)
        # The authenticated provider amount may exceed the reservation. The
        # broker must pass the exact amount to the append-only database RPC.
        assert kwargs["actual_microusd"] == 3_000
        assert kwargs["cost_units"] == "0.03"
        return {
            "status": "settled",
            "idempotent": False,
            "actual_microusd": kwargs["actual_microusd"],
            "released_microusd": -1_000,
            "variance_microusd": 1_000,
        }


class LedgerTransport:
    def __init__(self):
        self.sent = []

    def send(self, **kwargs):
        self.sent.append(kwargs)
        assert kwargs["method"] == "GET"
        assert kwargs["headers"]["authorization"] == "Bearer " + SECRET
        query = parse_qs(urlparse(kwargs["url"]).query)
        reservation_ms = int(
            datetime.fromisoformat(
                RESERVATION_AT.replace("Z", "+00:00")
            ).timestamp()
            * 1000
        )
        assert query["since_at"] == [str(reservation_ms - 1_000)]
        assert int(query["limit"][0]) > 0
        return br.ProviderResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(
                {"entries": [_ledger_entry()], "has_more": False}
            ).encode("utf-8"),
        )


def _broker(store, transport):
    broker = object.__new__(br.Broker)
    broker._store = store
    broker._transport = transport
    broker._credential_for = lambda _context, provider: (
        SECRET if provider == "deepline" else ""
    )
    broker._provider_funding_source_for = (
        lambda _context, provider: "host" if provider == "deepline" else ""
    )
    broker._funding_source_for = None
    broker._key_for = lambda provider: SECRET if provider == "deepline" else ""
    return broker


def test_delayed_exact_ledger_get_settles_above_reservation_without_paid_post():
    store = ReconciliationStore()
    transport = LedgerTransport()
    broker = _broker(store, transport)

    result = broker.reconcile_deepline_cost(_candidate())

    assert result == {
        "status": "settled",
        "idempotent": False,
        "actual_microusd": 3_000,
        "released_microusd": -1_000,
        "variance_microusd": 1_000,
    }
    assert [request["method"] for request in transport.sent] == ["GET"]
    assert store.calls == [
        {
            "round_id": ROUND_ID,
            "run_id": "run-1",
            "call_identity": CALL_IDENTITY,
            "uncertain_entry_id": 41,
            "request_id": REQUEST_ID,
            "operation": "firecrawl_scrape",
            "credential_fingerprint": FINGERPRINT,
            "actual_microusd": 3_000,
            "cost_units": "0.03",
        }
    ]


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({"request_id": "ctx-tool-not-hex"}, {"status": "invalid"}),
        ({"operation": "unknown_paid_tool"}, {"status": "invalid"}),
        ({"call_identity": "not-a-document-hash"}, {"status": "invalid"}),
        (
            {"credential_fingerprint": "sha256:" + "c" * 64},
            {"status": "credential_mismatch"},
        ),
    ],
)
def test_delayed_reconciliation_rejects_untrusted_identity_before_get(
    patch, expected
):
    store = ReconciliationStore()
    transport = LedgerTransport()
    broker = _broker(store, transport)

    assert broker.reconcile_deepline_cost(_candidate(**patch)) == expected
    assert transport.sent == []
    assert store.calls == []


class CursorStore:
    def __init__(self):
        self.items = [_candidate(uncertain_entry_id=10), _candidate(
            uncertain_entry_id=20,
            run_id="run-2",
            assignment_id="assignment-2",
            submission_id="submission-2",
        )]
        self.after = []

    def list_deepline_cost_reconciliations(
        self, round_id, *, run_id, after_entry_id, limit
    ):
        assert round_id == ROUND_ID and limit == 1
        self.after.append(after_entry_id)
        available = [
            item for item in self.items if not run_id or item["run_id"] == run_id
        ]
        return sorted(
            available,
            key=lambda item: (
                item["uncertain_entry_id"] <= after_entry_id,
                item["uncertain_entry_id"],
            ),
        )[:1]


class CursorBroker:
    def __init__(self, lock):
        self.lock = lock
        self.seen = []

    def reconcile_deepline_cost(self, candidate):
        assert self.lock.acquire(blocking=False)
        self.lock.release()
        self.seen.append(candidate["uncertain_entry_id"])
        return {"status": "pending" if len(self.seen) == 1 else "settled"}


def _bare_service(store, broker):
    service = object.__new__(svc.ArenaService)
    service._store = store
    service._lock = threading.RLock()
    service._brokers = {ROUND_ID: broker}
    service._deepline_reconciliation_after = {}
    return service


def test_pending_candidate_does_not_starve_the_next_deepline_candidate():
    store = CursorStore()
    lock = threading.RLock()
    broker = CursorBroker(lock)
    service = _bare_service(store, broker)
    service._lock = lock

    first = service._reconcile_deepline_cost(ROUND_ID)
    second = service._reconcile_deepline_cost(ROUND_ID)

    assert first["status"] == "pending"
    assert second["status"] == "settled"
    assert broker.seen == [10, 20]
    assert store.after == [0, 10]


def test_pending_deepline_accounting_does_not_stop_other_round_progress():
    service = object.__new__(svc.ArenaService)
    service._invalidate_hot_round = lambda: None
    service._round = lambda _round_id: {"status": "stage1_scoring"}
    service._reconcile_deepline_cost = lambda _round_id: {
        "status": "pending",
        "run_status": "leased",
    }
    service._reconcile_openrouter_cost = lambda _round_id: {"status": "none"}
    progressed = []
    service._advance_round_locked = lambda round_id: (
        progressed.append(round_id) or {"status": "advanced"}
    )

    assert service.advance_round(ROUND_ID) == {"status": "advanced"}
    assert progressed == [ROUND_ID]
