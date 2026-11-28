"""Exercise the actual standing allocator with independent simultaneous DB clients."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from uuid import uuid4

from psycopg2.extras import Json
import pytest

from lab_arena.collection_pool import POOL_TERMS_HASH
from tests.lab_arena.collection_pool_reward_rate_pg_test import migrate, NEW_TERMS
from tests.lab_arena.collection_standing_pool_pg_test import db, pool, digest

A = "deepseek/deepseek-v4-pro-0813"
B = "nvidia/nemotron-3.5-super-vl"
ROUTES = {A: "coreweave/fp8", B: "deepinfra/fp4"}


def capacity(p, epoch=201, amount=100_000_000, *, now=None, block=None, read=False):
    now = int(time.time()) if now is None else now
    block = block or epoch * 360
    block_hash = "0x" + digest(str(block))
    valuation = dict(
        block=block,
        block_hash=block_hash,
        miner_alpha_rao=1_000_000_000,
        tao_rao_per_alpha=1_000_000_000,
        tao_price_microusd=amount,
        emission_microusd=amount,
        observed_at=now,
        price_updated_at=now,
    )
    return p.rpc(
        "epoch_capacity", epoch, block, block_hash, None if read else Json(valuation)
    )


def claim(
    p,
    batch,
    *,
    cost=100,
    models=(A, B) * 40,
    epoch=201,
    expected="*",
    job=None,
    token=None,
    routes=None,
):
    models = list(models)
    return p.rpc(
        "governed_claim",
        job or str(uuid4()),
        token or "sha256:" + digest(str(uuid4())),
        "pool-" + digest(batch),
        "live",
        p.harness,
        Json(["P9-s1001"] * len(models)),
        Json(models),
        Json(dict.fromkeys(models, 0)),
        Json(dict.fromkeys(models, cost) if type(cost) is int else cost),
        POOL_TERMS_HASH,
        expected,
        epoch,
        Json(routes or {m: ROUTES[m] for m in models}),
    )


@pytest.fixture
def gov(pool):
    migrate(pool)
    sql = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "773-governed-standing-workers.sql"
    )
    with pool.driver.connect(**pool.dsn) as conn, conn.cursor() as c:
        c.execute("SET ROLE rate_test_migrator")
        c.execute(sql.read_text())
    pool.activate()
    pool.register(0, 0, 1_000_000, alter={"terms_sha256": NEW_TERMS})
    return pool


def test_governor_keeps_ambiguous_spend_and_carries_running_leases(gov):
    p = gov
    token = "sha256:" + digest("t")
    assert claim(p, "a") is None and not capacity(p, read=True)
    assert capacity(p, amount=335) and capacity(p, read=True)
    assert p.claim("unprotected") is None
    with ThreadPoolExecutor(max_workers=8) as threads:
        picks = list(threads.map(lambda _: claim(p, "a", token=token), range(8)))
    j, k = sorted((r for r in picks if r), key=lambda r: r["slot"])
    assert (j["slot"], k["slot"]) == (0, 1) and p.rpc("epoch_liability", 201) == 300
    assert claim(p, "a", job=j["job_id"], token=token)["job_id"] == j["job_id"]
    assert claim(p, "a", expected=None) is None
    assert p.reserve(j, token, "paid", 100) == "reserved"
    p.record(j, token, "paid", 80, "gen-governed")
    p.finish(j, token, epoch=201)
    assert p.rpc("epoch_liability", 201) == 270
    assert p.reserve(k, token, "unknown", 90) == "reserved"
    p.record(k, token, "unknown", None, None, status=0)
    p.rpc("finish", k["job_id"], token, "failed", Json({}), None, 201)
    assert p.rpc("epoch_liability", 201) == 255
    assert capacity(p, amount=9999)  # no upward repricing within an epoch
    assert claim(p, "a") is None
    assert capacity(p, 202, amount=200)
    assert p.rpc("epoch_liability", 202) == 135
    assert claim(p, "stale", epoch=201) is None and claim(p, "next", epoch=202) is None
    assert claim(p, "smaller", epoch=202, cost=30, token=token)
    assert capacity(p, 203, amount=200)
    assert (
        p.rpc("epoch_liability", 203) == 180
        and claim(p, "no-rollover-spend", epoch=203, cost=1) is None
    )
    with pytest.raises(p.driver.Error):
        capacity(p, epoch=203, now=int(time.time()) - 3600)
    with pytest.raises(p.driver.Error):
        capacity(p, epoch=203, block=555)


def test_route_backoff_skips_only_that_route_and_later_fills_holes(gov):
    p = gov
    capacity(p)
    tk = "sha256:" + digest("r")
    models = (A, A, B, A)
    j = claim(p, "routes", models=models, token=tk)
    assert j["slot"] == 0
    p.reserve(j, tk, "rejected", 65)
    p.record(j, tk, "rejected", None, None, status=429)
    p.rpc("finish", j["job_id"], tk, "failed", Json({}), None, 201)
    assert claim(p, "routes", models=models, expected=None) == {"model": B}
    assert claim(p, "routes", models=models, expected=A) is None
    healthy = claim(p, "routes", models=models, expected=B, token=tk)
    assert healthy["slot"] == 2 and p.rpc("epoch_liability", 201) == 248
    assert claim(p, "routes", models=models, expected=None) is None
    changed = claim(
        p, "routes", models=models, routes={A: "deepinfra/fp8", B: ROUTES[B]}
    )
    assert changed["error"] == "batch_routes_changed"
    p.sql(
        "update public.collection_route_backoff set unavailable_until=clock_timestamp()-interval '1 second'",
        owner=True,
    )
    assert claim(p, "routes", models=models, expected=A, token=tk)["slot"] == 1
    assert claim(p, "routes", models=models, expected=A, token=tk)["slot"] == 3
    assert p.rpc("epoch_liability", 201) == 548
    assert claim(p, "routes", models=models) is None


def test_lanes_require_real_settled_overlap_at_each_prior_tier(gov):
    p = gov
    capacity(p)
    tk = "sha256:" + digest("t")
    assert p.rpc("worker_lanes") == 4
    for old, new in ((4, 16), (16, 40), (40, 66)):
        with pytest.raises(p.driver.Error):
            p.sql(
                "select public.collection_set_worker_lanes_v1(%s)", (new,), owner=True
            )
        with ThreadPoolExecutor(max_workers=min(24, old)) as threads:
            jobs = list(
                threads.map(lambda _: claim(p, str(old), token=tk), range(old + 3))
            )
        picked = [j for j in jobs if j]
        assert len(picked) == old and len({j["slot"] for j in picked}) == old
        for job in picked:
            p.reserve(job, tk, "model", 90)
            p.record(job, tk, "model", 40, "gen-" + job["job_id"])
            p.finish(job, tk, epoch=201)
        assert (
            p.sql(
                "select public.collection_set_worker_lanes_v1(%s)", (new,), owner=True
            )
            == new
        )
        assert p.rpc("worker_lanes") == new
    with pytest.raises(p.driver.Error):
        p.rpc("set_worker_lanes", 4)
    assert p.sql("select public.collection_set_worker_lanes_v1(4)", owner=True) == 4


def test_active_sibling_stops_before_spending_after_route_throttle(gov):
    p = gov
    capacity(p)
    tk = "sha256:" + digest("throttle")
    first = claim(p, "sibling", models=(A, A, B), token=tk)
    sibling = claim(p, "sibling", models=(A, A, B), token=tk)
    p.reserve(first, tk, "rate", 70)
    p.record(first, tk, "rate", None, "gen-refusal", status=429)
    assert p.reserve(first, tk, "rate", 70) == "already_reserved"
    assert p.reserve(sibling, tk, "new", 70) == "route_backoff"
    assert (
        p.sql(
            "select count(*) from public.collection_calls where job_id=%s",
            (sibling["job_id"],),
            owner=True,
        )
        == 0
    )
    other = claim(p, "sibling", models=(A, A, B), token=tk)
    assert other["model"] == B and p.reserve(other, tk, "allowed", 70) == "reserved"
    p.record(other, tk, "allowed", 15, "gen-cross-epoch")
    assert capacity(p, epoch=202)
    p.rpc("finish", first["job_id"], tk, "failed", Json({}), None, 202)
    p.rpc("finish", sibling["job_id"], tk, "failed", Json({}), None, 202)
    p.rpc("finish", other["job_id"], tk, "failed", Json({}), None, 202)
    assert p.rpc("epoch_liability", 202) == 128
    assert p.rpc("epoch_liability", 203) == 105


def test_cost_priority_skips_expensive_slots_without_discarding_them(gov):
    p = gov
    capacity(p)
    normal = p.register(1, 1, 48_100, alter={"terms_sha256": NEW_TERMS})
    # The expensive first slot can be funded by the reserve, the later cheap slot by the normal miner.
    models = (A, B, B, B)
    cost = {A: 160_000, B: 20_000}
    tk = "sha256:" + digest("cap-order")
    affordable = claim(p, "cost-priority", cost=cost, models=models, expected=None)
    assert affordable == {"model": B}
    assert claim(p, "cost-priority", cost=cost, models=models, expected=A) is None
    jobs = [
        claim(p, "cost-priority", cost=cost, models=models, token=tk) for _ in range(4)
    ]
    assert [j["slot"] for j in jobs] == [1, 2, 0, 3]
    assert [j["registration_id"] == normal[0] for j in jobs] == [
        True,
        True,
        False,
        False,
    ]
    for j in jobs:
        p.rpc("finish", j["job_id"], tk, "failed", Json({}), None, 201)

    # Neither key can pay for the first slot in a second program. Lower-cost work
    # still gets its exact frozen slot, and subsequent new funding can fill the hole.
    cost = {A: 2_000_000, B: 20_000}
    assert claim(p, "larger", cost=cost, models=(A, B), token=tk)["slot"] == 1
    assert claim(p, "larger", cost=cost, models=(A, B), expected=None) is None
    funded = p.register(2, 1, 2_000_005, alter={"terms_sha256": NEW_TERMS})
    restored = claim(p, "larger", cost=cost, models=(A, B), token=tk)
    assert (
        restored["slot"] == 0
        and restored["model"] == A
        and restored["registration_id"] == funded[0]
    )
