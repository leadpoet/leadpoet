from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/241-reconcile-arena-2026-09-13-deepline-402-costs.sql"
TARGETS = (
    (315826, "sub-caf0e1ef30c9712e6385afe24a75375e", "arena-2026-09-13:sub-caf0e1ef30c9712e6385afe24a75375e:1:0:1", 1, "sha256:223335eda1ccf9fcf8b0d4d3a00f85d957cf896611b8c845fea331be83c8c14d", "deepline.execute", 79982421, 1653),
    (331371, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:6bc891aeb6faf7f07d1fe97e3f9cc12e3cac686211097473456b77613a524f2c", "deepline.execute", 56000, 1928),
    (331374, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:bbeb448fa9f273c0b58b6a82ad8ae7d7154e20a8789d39b596f0f5dcce7bbb76", "deepline.execute", 56000, 1968),
    (331383, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:070df2985bf72f35fdc6d01c93786769f6fb8f7d976294ddf1fc14fe756d3496", "deepline.execute", 56000, 1918),
    (331388, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:231f8c7f751019140a87611f800b4cc1fc6256b440aad576ba3d31899310a7da", "deepline.execute", 56000, 1928),
    (331398, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:83b62e720fd19d4bafa5bab2d2dd09345b591f28e576f4d87c7bab19198894f1", "deepline.execute", 56000, 1968),
    (331411, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1", 2, "sha256:bf2cd93f1511916c9c2c5967605d058d830c4797f0cdc055d60d29f339749fb1", "deepline.execute", 73321638, 1653),
    (336996, "sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:10:score:1", 2, "sha256:94484384d8b1dfcb5de6ec34755d9c55abb4ef2cf8fc826e624de541d5d29a93", "scrapingdog.scrape", 49049470, 1743),
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def test_exact_402_reconciliation_is_guarded_idempotent_and_releases_budget(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    hotkey = "5Hb5Sxe46cp1XjwG23FX4BcqWSmgrewZLATBEX3LwreEFfC5"
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        for uncertain_id, submission, run_id, stage, identity, operation, amount, body_bytes in TARGETS:
            for entry_id, kind in ((uncertain_id - 2, "reservation"), (uncertain_id - 1, "dispatch")):
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES (%s,%s,%s,'arena-2026-09-13',%s,%s,%s,%s,'deepline',%s,'miner_key',%s,'{}')",
                    (entry_id, kind, hotkey, submission, run_id, stage, identity, operation, amount),
                )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES (%s,'uncertain',%s,'arena-2026-09-13',%s,%s,%s,%s,'deepline',%s,'miner_key',%s,jsonb_build_object('reason','worker_reported','call',jsonb_build_object('reason','missing_provider_cost','provider_status',402,'call_succeeded',false,'body_is_mapping',true,'billing_present',true,'usage_present',false,'body_bytes',%s)))",
                (uncertain_id, hotkey, submission, run_id, stage, identity, operation, amount, body_bytes),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) VALUES (337100,'settlement',%s,'other-round','other-sub','other-run',2,%s,'deepline','deepline.execute','miner_key',7000,'{}','{\"status\":200}'),(337101,'uncertain',%s,'other-round','other-sub','other-run',2,%s,'deepline','deepline.execute','miner_key',9000,'{\"reason\":\"worker_reported\"}',NULL)",
            (hotkey, "sha256:" + "a" * 64, hotkey, "sha256:" + "b" * 64),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT setval(pg_get_serial_sequence('public.lab_arena_ledger','entry_id'),"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger))"
        )
    connection.commit()

    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,billing_present}','false') WHERE entry_id=315826")
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    with pytest.raises(Exception, match="head_mismatch:315826"):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,billing_present}','true') WHERE entry_id=315826")
        cursor.execute("SET session_replication_role=origin")
    connection.commit()

    for _ in range(2):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        connection.commit()
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*),sum(amount_microusd),sum((entry_doc->>'released_microusd')::bigint) FROM public.lab_arena_ledger WHERE entry_kind='settlement' AND entry_doc->>'deepline_402_history_reconciliation'='true'")
        count, actual, released = cursor.fetchone()
        assert (count, actual, released) == (8, 0, sum(row[6] for row in TARGETS))
        cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE entry_id=315826 AND entry_kind='uncertain'")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger WHERE entry_id IN (337100,337101) ORDER BY entry_id")
        assert cursor.fetchall() == [("settlement", 7000), ("uncertain", 9000)]
    connection.close()
