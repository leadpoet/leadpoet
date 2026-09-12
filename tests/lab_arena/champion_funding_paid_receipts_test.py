"""Free regression for original provider status in the paid proof's ledger query."""

import json
from types import SimpleNamespace

from tests.lab_arena.champion_funding_paid_e2e_test import _fallback_evidence
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


def test_fallback_receipt_uses_original_provider_status():
    database = database_with_lab_arena_migration(migrations=())
    try:
        psycopg2, dsn = next(database)

        def connect():
            return psycopg2.connect(**dsn)

        connection = connect()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE public.lab_arena_ledger (
                  entry_id BIGSERIAL PRIMARY KEY,
                  created_at TIMESTAMPTZ DEFAULT now(),
                  run_id TEXT, provider TEXT, entry_kind TEXT,
                  amount_microusd BIGINT, funding_source TEXT,
                  entry_doc JSONB, terminal_response JSONB
                )
            """)
            for original_status in (401, 403):
                run_id = "provider-status-%d" % original_status
                for attempt in range(1, 5):
                    account = {"provider_status": original_status}
                    call = {"provider_attempt": attempt}
                    terminal = {"status": 402}
                    if attempt % 2:
                        terminal["account_failure_evidence"] = account
                    else:
                        call["account_failure_evidence"] = account
                    cursor.execute("""
                        INSERT INTO public.lab_arena_ledger
                          (run_id, provider, entry_kind, amount_microusd,
                           funding_source, entry_doc, terminal_response)
                        VALUES (%s, 'openrouter', 'settlement', 0,
                                'miner_key', %s::jsonb, %s::jsonb)
                    """, (run_id, json.dumps({"call": call}), json.dumps(terminal)))
        connection.close()
        for original_status in (401, 403):
            run_id = "provider-status-%d" % original_status
            harness = SimpleNamespace(
                connect=connect,
                service=SimpleNamespace(store=SimpleNamespace(list_runs=lambda *a, **k: [
                    {"run_id": run_id, "champion_restart_required": True}
                ])),
            )
            proof = _fallback_evidence(harness, "test-round", "openrouter")
            assert proof["provider_statuses"] == [original_status]
            assert proof["standardized_statuses"] == [402]
            assert proof["attempts"] == [1, 2, 3, 4]
            assert proof["miner_rows"] == proof["ledger_rows"] == 4
    finally:
        database.close()
