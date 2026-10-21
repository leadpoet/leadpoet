"""Native issuance order is numeric and scoped on both database transports."""
import json

import httpx

from lab_arena.store import ArenaStore, PostgrestTransport, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import _local_database


def test_postgrest_uses_json_numeric_order_and_bounded_projection():
    sent = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: sent.append(request) or httpx.Response(200, json=[]))) as client:
        store = ArenaStore(PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client))
        assert store.latest_execution_model_reservation("run") is None
    query = sent[0].url.params
    assert query["run_id"] == "eq.run" and query["provider"] == "eq.openrouter"
    assert query["entry_kind"] == "eq.reservation" and query["limit"] == "1"
    assert query["order"] == "entry_doc->action_sequence.desc.nullsfirst,entry_doc->provider_attempt.desc.nullslast,entry_id.desc"
    assert "terminal_response" not in query["select"]


def test_postgres_actual_numeric_order_scope_retry_and_completion_order():
    gen = _local_database(())
    psycopg, dsn = next(gen)
    transport = PsycopgTransport(lambda: psycopg.connect(**dsn), role=None)
    store = ArenaStore(transport)
    try:
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE public.lab_arena_ledger(entry_id bigint,run_id text,call_identity text,operation_id text,provider text,funding_source text,entry_kind text,entry_doc jsonb)")
            def insert(entry_id, sequence, attempt=1, *, run="run", provider="openrouter", kind="reservation"):
                cur.execute("INSERT INTO public.lab_arena_ledger VALUES(%s,%s,%s,'openrouter.responses',%s,'miner_key',%s,%s::jsonb)",
                    (entry_id, run, str(entry_id), provider, kind, json.dumps({"action_sequence": sequence, "provider_attempt": attempt})))
            insert(1, 10)
            insert(2, 9)  # Created/completed later; issued earlier.
            insert(3, 10, 2)
            insert(4, 10, 2)
            insert(5, 100, run="other")
            insert(6, 100, provider="deepline")
            insert(7, 100, kind="settlement")
        assert store.latest_execution_model_reservation("run")["entry_id"] == 4
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger VALUES(8,'run','8','openrouter.chat','openrouter','miner_key','reservation','{}')")
        # Missing sequence must be examined and rejected by the service.
        assert store.latest_execution_model_reservation("run")["entry_id"] == 8
    finally:
        transport.close()
        gen.close()
