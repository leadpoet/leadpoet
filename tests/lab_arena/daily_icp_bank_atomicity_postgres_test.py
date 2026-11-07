"""The daily ICP bank publication transition is atomic and monotonic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import threading

import pytest

from tests.postgres_migration_harness import _database_with_migrations

SQL = Path(__file__).resolve().parents[2] / "scripts/667-lab-arena-atomic-icp-bank-publication.sql"
SETUP = """
CREATE ROLE lab_arena_owner NOLOGIN;
ALTER ROLE service_role BYPASSRLS;
CREATE TABLE public.qualification_private_icp_sets (
  set_id BIGINT PRIMARY KEY,
  icps JSONB NOT NULL,
  icp_set_hash TEXT,
  industry_distribution JSONB,
  active_from TIMESTAMPTZ,
  active_until TIMESTAMPTZ,
  generation_seed TEXT,
  is_active BOOLEAN NOT NULL DEFAULT FALSE
);
ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
GRANT SELECT, INSERT, UPDATE ON public.qualification_private_icp_sets TO service_role;
"""


@pytest.fixture()
def database():
    yield from _database_with_migrations((SQL.name,), setup_sql=SETUP)


def _bank(set_id: int, marker: str) -> dict:
    bank_date = date(int(str(set_id)[:4]), int(str(set_id)[4:6]), int(str(set_id)[6:8]))
    start = datetime.combine(bank_date, datetime.min.time(), timezone.utc)
    icps = [{"icp_id": marker, "prompt": marker}]
    icp_hash = hashlib.sha256(
        json.dumps(icps, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "set_id": set_id,
        "icps": icps,
        "icp_set_hash": icp_hash,
        "industry_distribution": {"Software": 1},
        "active_from": start,
        "active_until": start + timedelta(days=1),
        "generation_seed": str(set_id),
    }


def _publish(pg, dsn, bank, *, barrier=None):
    connection = pg.connect(**dsn)
    connection.autocommit = True
    try:
        if barrier is not None:
            barrier.wait(timeout=10)
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE service_role")
            cursor.execute("SET TIME ZONE 'America/Los_Angeles'")
            cursor.execute(
                "SELECT public.qualification_private_icp_set_publish_v1(%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s)",
                (
                    bank["set_id"], json.dumps(bank["icps"]), bank["icp_set_hash"],
                    json.dumps(bank["industry_distribution"]), bank["active_from"],
                    bank["active_until"], bank["generation_seed"],
                ),
            )
            return cursor.fetchone()[0]
    finally:
        connection.close()


def test_concurrent_publish_replays_and_rollover_keep_one_current_bank(database):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(SQL.read_text(encoding="utf-8"))
        cursor.execute(
            "SELECT has_function_privilege('anon', 'public.qualification_private_icp_set_publish_v1(bigint,jsonb,text,jsonb,timestamptz,timestamptz,text)', 'EXECUTE'),"
            "has_function_privilege('authenticated', 'public.qualification_private_icp_set_publish_v1(bigint,jsonb,text,jsonb,timestamptz,timestamptz,text)', 'EXECUTE'),"
            "has_function_privilege('service_role', 'public.qualification_private_icp_set_publish_v1(bigint,jsonb,text,jsonb,timestamptz,timestamptz,text)', 'EXECUTE')"
        )
        assert cursor.fetchone() == (False, False, True)

    # This date crosses the US daylight saving transition. A 24-hour UTC bank
    # window must remain valid when the session TimeZone is not UTC.
    same_date = 20260308
    same_date_banks = [_bank(same_date, "bank-a"), _bank(same_date, "bank-b")]
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda bank: _publish(pg, dsn, bank, barrier=barrier), same_date_banks,
        ))
    assert all(result["status"] == "activated" for result in results)
    assert sorted(result["inserted"] for result in results) == [False, True]
    assert results[0]["icps"] == results[1]["icps"]
    assert results[0]["icp_set_hash"] == results[1]["icp_set_hash"]
    replay = _publish(pg, dsn, same_date_banks[1])
    assert replay["status"] == "activated" and replay["inserted"] is False
    assert replay["icps"] == results[0]["icps"]

    rollover = [_bank(20260309, "older"), _bank(20260310, "newer")]
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rollover_results = list(pool.map(
            lambda bank: _publish(pg, dsn, bank, barrier=barrier), rollover,
        ))
    assert {result["status"] for result in rollover_results} <= {"activated", "stale"}
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_id, count(*) OVER () FROM public.qualification_private_icp_sets WHERE is_active"
        )
        assert cursor.fetchall() == [(20260310, 1)]

    inactive_conflict = _bank(20260311, "stored-partial")
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.qualification_private_icp_sets "
            "(set_id,icps,icp_set_hash,industry_distribution,active_from,active_until,generation_seed,is_active) "
            "VALUES (%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,FALSE)",
            (
                inactive_conflict["set_id"], json.dumps(inactive_conflict["icps"]),
                inactive_conflict["icp_set_hash"],
                json.dumps(inactive_conflict["industry_distribution"]),
                inactive_conflict["active_from"], inactive_conflict["active_until"],
                inactive_conflict["generation_seed"],
            ),
        )
    conflict = _publish(pg, dsn, _bank(20260311, "different-retry"))
    assert conflict["status"] == "conflict"
    assert conflict["icps"] == inactive_conflict["icps"]
    assert conflict["icp_set_hash"] == inactive_conflict["icp_set_hash"]
    recovered = {
        **inactive_conflict,
        "icps": conflict["icps"],
        "icp_set_hash": conflict["icp_set_hash"],
        "industry_distribution": conflict["industry_distribution"],
        "active_from": conflict["active_from"],
        "active_until": conflict["active_until"],
        "generation_seed": conflict["generation_seed"],
    }
    recovered["active_from"] = datetime.fromisoformat(
        str(recovered["active_from"]).replace("Z", "+00:00")
    )
    recovered["active_until"] = datetime.fromisoformat(
        str(recovered["active_until"]).replace("Z", "+00:00")
    )
    activated = _publish(pg, dsn, recovered)
    assert activated["status"] == "activated" and activated["inserted"] is False
    assert activated["icps"] == inactive_conflict["icps"]
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_id FROM public.qualification_private_icp_sets WHERE is_active"
        )
        assert cursor.fetchall() == [(20260311,)]


def test_failure_between_deactivate_and_activate_rolls_back(database):
    pg, dsn = database
    current = _bank(20260308, "current")
    _publish(pg, dsn, current)
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("""
          CREATE FUNCTION public.fail_daily_icp_activation()
          RETURNS trigger LANGUAGE plpgsql AS $failure$
          BEGIN
            IF NEW.is_active AND NOT OLD.is_active THEN
              RAISE EXCEPTION 'injected activation interruption';
            END IF;
            RETURN NEW;
          END;
          $failure$;
          CREATE TRIGGER fail_daily_icp_activation
          BEFORE UPDATE OF is_active ON public.qualification_private_icp_sets
          FOR EACH ROW EXECUTE FUNCTION public.fail_daily_icp_activation();
        """)

    with pytest.raises(pg.errors.RaiseException, match="injected activation interruption"):
        _publish(pg, dsn, _bank(20260309, "interrupted"))

    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_id, is_active FROM public.qualification_private_icp_sets ORDER BY set_id"
        )
        assert cursor.fetchall() == [(20260308, True)]
