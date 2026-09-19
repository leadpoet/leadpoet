"""OpenRouter hosted-search admission through the real PostgreSQL ledger."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import broker as br, operations, submission_runtime
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.test_lab_arena_broker import LUNA_RESPONSES, luna_price_table
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_KEYS,
    Harness,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
        )
    )


@pytest.fixture(scope="module")
def database_before_314():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "312-lab-arena-temporary-hold-admission.sql",
        )
    )


@pytest.fixture(scope="module")
def leased_context(database, tmp_path_factory):
    psycopg2, dsn = database
    harness = Harness(
        lambda: psycopg2.connect(**dsn),
        tmp_path_factory.mktemp("openrouter-web-search-reservation"),
        challengers=[], runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(harness, "arena-2099-01-01-e1", slot_ceiling=10)
    lease, token, _, _ = claim(
        harness.service.store, harness.round_id, harness.runner_keys[0],
        parallelism=10, ceiling=10,
    )
    yield harness, lease, token
    harness.service.store.close()


def test_migration_replays_without_changing_the_patched_function(database):
    psycopg2, dsn = database
    migration = (
        Path(__file__).resolve().parents[2]
        / "scripts/314-lab-arena-openrouter-web-search-reservation.sql"
    ).read_text(encoding="utf-8")
    signature = (
        "public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,"
        "jsonb,integer)"
    )
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_catalog.pg_get_functiondef(%s::regprocedure)", (signature,))
        before = cursor.fetchone()[0]
        cursor.execute(migration)
        cursor.execute(migration)
        cursor.execute("SELECT pg_catalog.pg_get_functiondef(%s::regprocedure)", (signature,))
        after = cursor.fetchone()[0]
    assert after == before
    assert hashlib.sha256(after.encode()).digest() == hashlib.sha256(
        before.encode()
    ).digest()
    assert after.count("lab_arena_openrouter_web_search_reservation") == 1
    assert "lab_arena_per_icp_paid_admission" in after
    assert "lab_arena_temporary_hold_admission" in after


class BlockingOpenRouterTransport:
    """Control only the external provider boundary; retain the real broker/DB."""

    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self._lock = threading.Lock()
        self.sent: list[dict] = []

    def send(self, *, method, url, headers, body, timeout_seconds,
             max_response_bytes=None):
        request = json.loads(body)
        assert method == "POST"
        assert headers["authorization"] == "Bearer " + CANARY_KEYS["openrouter"]
        assert request["tools"][0]["type"] == "openrouter:web_search"
        with self._lock:
            index = len(self.sent)
            self.sent.append(request)
        if index == 0:
            self.first_started.set()
            assert self.release_first.wait(5)
        payload = {
            "id": "gen-hosted-search-%d" % index,
            "object": "response",
            "status": "completed",
            "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
            "error": None,
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 20,
                "cost": "0.000100",
                "server_tool_use": {"web_search_requests": 1},
            },
        }
        return br.ProviderResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(payload).encode(),
        )


def _install_openrouter_broker(harness: Harness, provider) -> None:
    store = harness.service.store
    payer = submission_runtime.SubmissionProviderKeys(
        store=store,
        credentials=harness.service.config.credential_manager,
        organizer_keys=CANARY_KEYS,
    )
    harness.service.config.broker_factory = lambda _service, _round: br.Broker(
        store=store,
        key_for=lambda name: CANARY_KEYS[name],
        credential_for=payer.credential_for,
        funding_source_for=payer.funding_source_for,
        provider_funding_source_for=payer.provider_funding_source_for,
        price_table=luna_price_table(),
        transport=provider,
        clock=harness.clock,
    )
    harness.service._brokers.clear()


def _web_search_frame(action_sequence: int) -> dict:
    return {
        "operation_id": "openrouter.responses",
        "parameters": {
            **LUNA_RESPONSES,
            "tools": [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "engine": "native",
                        "max_uses": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
                        "max_total_results": (
                            operations.OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS
                        ),
                    },
                }
            ],
            "max_tool_calls": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
        },
        "timeout_ms": 300_000,
        "action_sequence": action_sequence,
    }


def test_hosted_search_settles_waits_and_replays_without_redispatch(
    leased_context,
):
    harness, lease, token = leased_context
    store = harness.service.store

    provider = BlockingOpenRouterTransport()
    _install_openrouter_broker(harness, provider)

    def execute(sequence: int) -> dict:
        return harness.service.handle_provider(
            lease["run_id"], token, _web_search_frame(sequence)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, 0)
        assert provider.first_started.wait(5)
        second = pool.submit(execute, 1)
        time.sleep(0.3)
        assert len(provider.sent) == 1
        assert not second.done()
        provider.release_first.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert first_result["status"] == second_result["status"] == 200
    first_call = first_result["call"]
    second_call = second_result["call"]
    assert first_call["reserved_microusd"] == 4_000_000
    assert second_call["reserved_microusd"] == 3_999_900
    assert first_call["actual_microusd"] == second_call["actual_microusd"] == 100
    assert len(provider.sent) == 2

    rows = store.list_ledger(run_id=lease["run_id"])
    assert [row["entry_kind"] for row in rows] == [
        "reservation", "dispatch", "settlement",
        "reservation", "dispatch", "settlement",
    ]
    assert rows[0]["entry_doc"]["reserve_remaining_budget"] is True
    assert rows[0]["provider"] == "openrouter"
    assert rows[0]["operation_id"] == "openrouter.responses"

    replay = execute(1)
    assert replay["status"] == second_result["status"]
    assert replay["headers"] == second_result["headers"]
    assert replay["body_b64"] == second_result["body_b64"]
    assert replay["call"]["call_identity"] == second_call["call_identity"]
    assert replay["call"]["actual_microusd"] == 100
    assert len(provider.sent) == 2
    assert store.list_ledger(run_id=lease["run_id"]) == rows


def test_pre_314_current_broker_request_fails_before_provider_dispatch(
    database_before_314, tmp_path,
):
    psycopg2, dsn = database_before_314
    harness = Harness(
        lambda: psycopg2.connect(**dsn), tmp_path,
        challengers=[], runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(harness, "arena-2099-01-01-e2", slot_ceiling=10)
    lease, token, _, _ = claim(
        harness.service.store, harness.round_id, harness.runner_keys[0],
        parallelism=10, ceiling=10,
    )

    class ProviderMustNotRun:
        calls = 0

        def send(self, **_kwargs):
            self.calls += 1
            raise AssertionError("provider dispatch crossed the SQL guard")

    provider = ProviderMustNotRun()
    _install_openrouter_broker(harness, provider)
    with pytest.raises(
        ArenaStoreError, match="lab_arena_dynamic_reserve_input_invalid"
    ):
        harness.service.handle_provider(
            lease["run_id"], token, _web_search_frame(0)
        )
    assert provider.calls == 0
    assert harness.service.store.list_ledger(run_id=lease["run_id"]) == []
    harness.service.store.close()


def test_ordinary_openrouter_reservation_keeps_its_fixed_estimate(
    leased_context,
):
    harness, _prior_lease, _prior_token = leased_context
    store = harness.service.store
    lease, token, _, _ = claim(
        store, harness.round_id, harness.runner_keys[0],
        parallelism=10, ceiling=10,
    )
    identity = sha("ordinary-openrouter-fixed-estimate")
    reserved = store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id="openrouter.responses",
        provider="openrouter",
        funding_source="host",
        amount_microusd=123,
        call_doc={"request_hash": sha("ordinary-openrouter-request")},
    )
    assert reserved["status"] == "reserved"
    assert reserved["amount_microusd"] == 123
    assert store.mark_dispatched(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        actual_microusd=123,
        terminal_response={"status": 200, "call_succeeded": True},
    )["status"] == "settled"


@pytest.mark.parametrize(
    ("provider", "operation_id", "amount", "call_doc"),
    (
        ("openrouter", "openrouter.chat", 1_000, {"reserve_remaining_budget": True}),
        ("scrapingdog", "openrouter.responses", 1_000, {"reserve_remaining_budget": True}),
        (None, "openrouter.responses", 1_000, {"reserve_remaining_budget": True}),
        ("openrouter", "openrouter.responses", 0, {"reserve_remaining_budget": True}),
        ("openrouter", "openrouter.responses", 1_000, {"reserve_remaining_budget": False}),
    ),
)
def test_dynamic_reservation_rejects_every_other_shape_before_lease_or_write(
    database, leased_context, provider, operation_id, amount, call_doc,
):
    psycopg2, dsn = database
    store = ArenaStore(
        PsycopgTransport(lambda: psycopg2.connect(**dsn)),
        lease_ttl_seconds=120,
    )
    identity = sha(
        "%s:%s:%s:%s" % (provider, operation_id, amount, call_doc)
    )
    _harness, lease, _token = leased_context
    with pytest.raises(ArenaStoreError, match="lab_arena_dynamic_reserve_input_invalid"):
        store.reserve_call(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token("missing-lease"),
            call_identity=identity,
            operation_id=operation_id,
            provider=provider,
            funding_source="host",
            amount_microusd=amount,
            call_doc=call_doc,
        )
    assert store.list_ledger(call_identity=identity) == []
    store.close()


@pytest.mark.parametrize(
    ("provider", "operation_id", "amount"),
    (
        ("openrouter", "openrouter.responses", 1),
        ("openrouter", "openrouter.responses", 4_000_001),
        ("deepline", "deepline.execute", 0),
    ),
)
def test_valid_dynamic_shape_with_false_lease_is_stale_without_write(
    database, leased_context, provider, operation_id, amount,
):
    psycopg2, dsn = database
    store = ArenaStore(
        PsycopgTransport(lambda: psycopg2.connect(**dsn)),
        lease_ttl_seconds=120,
    )
    identity = sha("valid-dynamic:%s:%s:%d" % (provider, operation_id, amount))
    _harness, lease, _token = leased_context
    result = store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token("missing-lease"),
        call_identity=identity,
        operation_id=operation_id,
        provider=provider,
        funding_source="host",
        amount_microusd=amount,
        call_doc={"reserve_remaining_budget": True},
    )
    assert result == {"status": "stale"}
    assert store.list_ledger(call_identity=identity) == []
    store.close()
