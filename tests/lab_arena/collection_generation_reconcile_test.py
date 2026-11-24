import json
import time
from types import SimpleNamespace

import pytest

from lab_arena import broker, store
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.locked_collection_pipeline_test import registered, claim


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "720-locked-collection-intake.sql",
            "721-locked-collection-reward-epoch.sql",
            "722-collection-call-budget.sql",
            "723-collection-pilot-prices.sql",
            "724-collection-scenario-ranking.sql",
            "725-collection-generator-roster.sql",
            "726-collection-first-paid-job.sql",
            "727-collection-synthetic-funding.sql",
            "728-collection-pilot-attempts.sql",
            "746-reconcile-collection-generation.sql",
        )
    )


def test_terminal_receipt_can_settle_once_only_from_its_original_key_and_generation(
    database,
):
    from scripts import reconcile_collection_generation as rc

    pg, dsn = database

    def connect():
        db = pg.connect(**dsn)
        db.autocommit = True
        return db

    db = store.PsycopgTransport(connect, role="lab_arena_service")
    miner = registered(db, "M")
    job = claim(miner, "late-receipt", "P1-s1001")
    rid = job["job_id"]
    lease = store.hash_lease_token(job["lease_token"])
    request = "generate-1"
    gen = "gen-1700000000-readback1234567"
    args = dict(p_job=rid, p_lease=lease, p_request=request)
    db.rpc(
        "collection_reserve_call_v1",
        dict(args, p_provider="openrouter", p_hash="a" * 64, p_amount=40000),
    )
    db.rpc(
        "collection_record_call_v1",
        dict(args, p_hash="b" * 64, p_provider_id=gen, p_cost=None, p_status=429),
    )
    miner.complete(rid, job["lease_token"], "failed", {}, None, 7)

    class Http:
        def __init__(self, body):
            self.body = body
            self.calls = []

        def send(self, **kw):
            assert kw["method"] == "GET" and kw["url"].endswith(gen)
            self.calls.append(kw["url"])
            return broker.ProviderResponse(200, {}, json.dumps(self.body).encode())

    class Vault:
        def runtime_key(self, row, provider):
            assert row["provider"] == provider == "openrouter"
            return "not-real-test-key"

    bad = Http({"data": {"id": "gen-other", "total_cost": 0}})
    assert rc.reconcile(db, Vault(), bad, rid, request)["status"] == "unresolved"
    with pg.connect(**dsn) as c, c.cursor() as q:
        q.execute(
            "select cost_microusd from collection_call_receipts where job_id=%s", (rid,)
        )
        assert q.fetchone() == (None,)
    good = Http({"data": {"id": gen, "total_cost": 0}})
    result = rc.reconcile(db, Vault(), good, rid, request)
    assert result["status"] == "settled" and result["cost_microusd"] == 0
    assert (
        rc.reconcile(db, Vault(), good, rid, request)["status"] == "settled"
        and len(good.calls) == 1
    )
    with pg.connect(**dsn) as c, c.cursor() as q:
        q.execute(
            "select cost_microusd,http_status,response_hash,provider_id from collection_call_receipts where job_id=%s",
            (rid,),
        )
        assert q.fetchone() == (0, 429, "b" * 64, gen)
        q.execute(
            "select count(*) from collection_generation_settlements where job_id=%s",
            (rid,),
        )
        assert q.fetchone() == (1,)
        q.execute("savepoint denied")
        with pytest.raises(pg.Error, match="immutable|append-only"):
            q.execute(
                "update collection_call_receipts set cost_microusd=9 where job_id=%s",
                (rid,),
            )
        q.execute("rollback to denied")
        q.execute("set local role lab_arena_service")
        with pytest.raises(pg.Error, match="permission"):
            q.execute("select * from collection_generation_settlements")
    db.close()
