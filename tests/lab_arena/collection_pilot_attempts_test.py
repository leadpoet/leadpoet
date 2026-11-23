"""Exercise the deployed claim RPC against a failed slot before and after migration."""

from uuid import uuid4

import pytest

from lab_arena import collection_ranking, store
from tests.lab_arena.collection_calibration_test import db
from tests.lab_arena.locked_collection_pipeline_test import claim, registered


def test_failed_slot_uses_one_distinct_bounded_replacement(db):
    driver, dsn = db

    def connect():
        conn = driver.connect(**dsn)
        conn.autocommit = True
        return conn

    transport = store.PsycopgTransport(connect, role="lab_arena_service")
    try:
        miner = registered(transport, "A")
        batch, scenario = "bounded-retry", "C01-s2071"
        original = claim(miner, batch, scenario)
        job_id = original["job_id"]
        lease = store.hash_lease_token(original["lease_token"])

        def reserve(job, token, request, provider, amount):
            return transport.rpc(
                "collection_reserve_call_v1",
                dict(
                    p_job=job,
                    p_lease=token,
                    p_request=request,
                    p_provider=provider,
                    p_hash="a" * 64,
                    p_amount=amount,
                ),
            )["status"]

        def settle(job, token, request, charge):
            return transport.rpc(
                "collection_record_call_v1",
                dict(
                    p_job=job,
                    p_lease=token,
                    p_request=request,
                    p_provider_id=str(uuid4()),
                    p_status=200,
                    p_hash="b" * 64,
                    p_cost=charge,
                ),
            )

        assert reserve(job_id, lease, "model", "openrouter", 1000) == "reserved"
        settle(job_id, lease, "model", 740)
        # An unresolved owner-funded charge still consumes the original pack's budget.
        assert reserve(job_id, lease, "judge-timeout", "judge", 700) == "reserved"
        miner.complete(job_id, original["lease_token"], "failed", {}, None, 12)
        # The late bill may exceed the estimate. Keep collecting other slots until it is final.
        waiting = claim(registered(transport, "B"), batch, scenario)
        assert (waiting["slot"], waiting["attempt"]) == (1, 1)
        settle(job_id, lease, "judge-timeout", 1300)
        replacement_miner = registered(transport, "C")
        retry = claim(replacement_miner, batch, scenario)
        assert retry["slot"] == original["slot"]
        assert (
            retry["job_id"] != job_id
            and retry["miner_hotkey"] != original["miner_hotkey"]
        )
        assert (
            retry["attempt"] == 2
            and retry["model"] == original["model"]
            and retry["scenario_id"] == scenario
        )
        remaining = original["budget_microusd"] - 2040
        assert retry["budget_microusd"] == remaining and retry["price_microusd"] == 0
        lease2 = store.hash_lease_token(retry["lease_token"])
        assert (
            reserve(retry["job_id"], lease2, "overspend", "openrouter", remaining + 1)
            == "budget_exhausted"
        )
        assert (
            reserve(retry["job_id"], lease2, "new-model-call", "openrouter", 100)
            == "reserved"
        )
        settle(retry["job_id"], lease2, "new-model-call", 60)
        assert (
            reserve(retry["job_id"], lease2, "unresolved-owner-judge", "judge", 700)
            == "reserved"
        )
        # The owner judge's unknown bill cannot confiscate valid, settled miner work.
        replacement_miner.complete(
            retry["job_id"],
            retry["lease_token"],
            "accepted",
            {},
            "sha256:" + "b" * 64,
            12,
        )
        third = registered(transport, "D")
        next_job = claim(third, batch, scenario)
        assert (next_job["slot"], next_job["attempt"]) == (2, 1)
        with connect() as conn, conn.cursor() as q:
            q.execute(
                "SELECT public.collection_pilot_costs_v1(%s), public.collection_batch_jobs_v1(%s)",
                (batch, batch),
            )
            costs, jobs = q.fetchone()
            assert sorted((r["slot"], r["attempt"]) for r in jobs) == [
                (0, 1),
                (0, 2),
                (1, 1),
                (2, 1),
            ]
            measured = {r["job_id"]: r for r in costs}
            assert (
                measured[job_id]["miner_cost_microusd"] == 740
                and measured[job_id]["status"] == "failed"
            )
            assert measured[retry["job_id"]]["miner_cost_microusd"] == 60
            assert measured[retry["job_id"]]["unresolved"] == 0
            q.execute(
                "SELECT count(*) FROM public.collection_calls WHERE provider='judge' AND job_id IN (%s,%s)",
                (job_id, retry["job_id"]),
            )
            assert q.fetchone()[0] == 2
    finally:
        transport.close()


@pytest.mark.parametrize("spend_full", [False, True])
def test_exhausted_slot_releases_the_next_assignment(db, spend_full):
    pg, dsn = db
    # Do not share the job queue with the other parametrization.
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute(
            "UPDATE public.collection_registrations SET revoked_at=now() WHERE revoked_at IS NULL"
        )

    def connect():
        conn = pg.connect(**dsn)
        conn.autocommit = True
        return conn

    transport = store.PsycopgTransport(connect, role="lab_arena_service")
    try:
        batch = "recover-full" if spend_full else "recover-once"
        miners = [
            registered(transport, key) for key in ("STU" if spend_full else "GHJ")
        ]
        first = claim(miners[0], batch, "E4-s2027")
        if spend_full:
            lease = store.hash_lease_token(first["lease_token"])
            assert (
                transport.rpc(
                    "collection_reserve_call_v1",
                    dict(
                        p_job=first["job_id"],
                        p_lease=lease,
                        p_request="unknown",
                        p_provider="openrouter",
                        p_hash="e" * 64,
                        p_amount=first["budget_microusd"],
                    ),
                )["status"]
                == "reserved"
            )
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute(
                "UPDATE public.collection_jobs SET lease_expires_at=now()-interval '1 minute' WHERE job_id=%s",
                (first["job_id"],),
            )
        second = claim(miners[1], batch, "E4-s2027")
        if spend_full:
            assert (second["slot"], second["attempt"]) == (1, 1)
        else:
            assert (second["slot"], second["attempt"]) == (0, 2)
            miners[1].complete(
                second["job_id"], second["lease_token"], "failed", {}, None, 12
            )
            third = claim(miners[2], batch, "E4-s2027")
            assert (third["slot"], third["attempt"]) == (1, 1)
    finally:
        transport.close()


def test_ranking_signs_one_current_member_per_logical_slot_after_failure(db):
    from concurrent.futures import ThreadPoolExecutor

    driver, dsn = db

    def connect():
        conn = driver.connect(**dsn)
        conn.autocommit = True
        return conn

    with connect() as conn, conn.cursor() as q:
        q.execute(
            "UPDATE public.collection_registrations SET revoked_at=now() WHERE revoked_at IS NULL"
        )
    transport = store.PsycopgTransport(connect, role="lab_arena_service")
    try:
        miners = [registered(transport, k) for k in "KLMNPQR"]
        batch, scenario = "replacement-ranking", "C01-s2071"
        orig = claim(miners[0], batch, scenario)

        def submit(job):
            return transport.rpc(
                collection_ranking.RPC,
                dict(
                    p_job=job["job_id"],
                    p_lease=store.hash_lease_token(job["lease_token"]),
                    p_packet="sha256:" + "b" * 64,
                    p_assessment="sha256:" + "c" * 64,
                ),
            )

        assert submit(orig)["status"] == "waiting"
        miners[0].complete(orig["job_id"], orig["lease_token"], "failed", {}, None, 33)
        jobs = [claim(miner, batch, scenario) for miner in miners[1:]]
        assert [job["attempt"] for job in jobs] == [2, 1, 1, 1, 1, 1]
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit, jobs))
        winners = [r for r in results if r["status"] == "ready"]
        assert len(winners) == 1
        assert [r["job_id"] for r in winners[0]["members"]] == [
            job["job_id"] for job in jobs
        ]
        assert orig["job_id"] not in {r["job_id"] for r in winners[0]["members"]}
    finally:
        transport.close()
