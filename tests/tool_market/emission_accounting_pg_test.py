"""Real reservations → settled usage → a single accepted, signed supplier target."""

from concurrent.futures import ThreadPoolExecutor
import threading

from psycopg2.extras import Json
import pytest

from lab_arena import weight_state
from lab_arena.store import PsycopgTransport
from lab_arena.tool_market.store import ToolMarketStore
from tests.lab_arena import collection_rewards_test as coll
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.tool_market.tool_market_pg_test import PROFILE, World


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "720-locked-collection-intake.sql",
            "721-locked-collection-reward-epoch.sql",
            "770-tool-market.sql",
            "771-tool-market-emission.sql",
        )
    )


def test_accepted_targets_carry_shortfall_and_late_usage_without_double_allocation(
    database,
):
    pg, dsn = database
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    store = ToolMarketStore(transport)
    store.bind_gamma_source(
        genesis=coll.GENESIS, netuid=71, service="5" * 48, registered_at=1
    )
    w = World(store)
    e = coll.EPOCH
    w.market._epoch = lambda: e - 1
    miner, _ = w.miner_key("miner-a", 10_000, miner=coll.MINER)
    _, token = w.buyer()
    assert w.market.call(token, PROFILE)["charge_microusd"] == 11_000
    assert not store.has_unallocated(epoch=e - 1)

    def project(epoch=e, cap=7_000, keys=(miner,), emission=1_000_000):
        return store.epoch_allocations(
            epoch=epoch,
            cap_microusd=cap,
            emission_microusd=emission,
            eligible_hotkeys=keys,
        )

    assert project() == {miner: 7_000}
    assert project(keys=[]) == {} and project(cap=0) == {}
    assert project(cap=7) == {}  # unrepresentable dust stays owed
    # A preview or a candidate built by any caller creates no payment.
    assert store.emission_statement(miner) == dict(
        accrued_microusd=10000,
        scheduled_microusd=0,
        unscheduled_microusd=10000,
        realized_alpha_rao=None,
    )
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        coll.insert_legacy_state(q, e - 1)
        q.execute(
            "INSERT INTO public.collection_reward_cutover VALUES ('finney',71,%s)", (e,)
        )

    def signed(epoch, amount, recipient=miner):
        start = 50 + (epoch - e) * 360
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute("SELECT public.collection_reward_quotes_v1(%s)", (epoch,))
            basis = q.fetchone()[0]
        basis["valuation"] = None
        v = coll.valuation(start)
        overlay = dict(
            schema_version="leadpoet.tool_market.supplier_allocations.v1",
            epoch=epoch,
            emission_microusd=v["emission_microusd"],
            valuation=v,
            allocations_microusd={recipient: amount},
        )
        return weight_state.build_accepted_weight_state(
            coll.SIGNER,
            network="finney",
            netuid=71,
            genesis_hash=coll.GENESIS,
            epoch=epoch,
            valid_from_block=start,
            valid_until_block=start + 359,
            reward_basis=basis,
            burn_hotkey=coll.BURN,
            issued_at="2026-11-28T00:00:00Z",
            tool_market=overlay,
        )

    def publish(doc):
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute(
                "SELECT public.lab_arena_publish_weight_state_v1('finney',71,%s,%s,%s)",
                (doc["epoch"], doc["state_hash"], Json(doc)),
            )
            return q.fetchone()[0]

    first = signed(e, 7_000)
    barrier = threading.Barrier(2)

    def same_state(_):
        barrier.wait(timeout=5)
        return publish(first)["state_hash"]

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(same_state, range(2))) == [first["state_hash"]] * 2
    assert project(epoch=e + 1) == {miner: 3_000}
    assert store.emission_statement(miner)["scheduled_microusd"] == 7000
    with pytest.raises(pg.Error, match="exceeds supplier receivable"):
        publish(signed(e + 1, 3_001))
    with pytest.raises(pg.Error, match="exceeds supplier receivable"):
        publish(signed(e + 1, 1_000, coll.SECOND))
    # Another old-epoch answer commits after issuance. History stays fixed and
    # the next epoch sees the late liability as well as the previous shortfall.
    assert w.market.call(token, PROFILE)["charge_microusd"] == 11000
    assert publish(first)["state_hash"] == first["state_hash"]
    assert project(epoch=e + 1, cap=50_000) == {miner: 13_000}
    assert publish(signed(e + 1, 13_000))["status"] == "accepted"
    assert not store.has_unallocated(epoch=e + 2)
    assert store.emission_statement(miner) == dict(
        accrued_microusd=20_000,
        scheduled_microusd=20_000,
        unscheduled_microusd=0,
        realized_alpha_rao=None,
    )
    # Two different epoch candidates must not both consume the same new debt,
    # even when they publish concurrently. No application-side locking.
    assert w.market.call(token, PROFILE)["charge_microusd"] == 11_000
    competing = [signed(e + 2, 10_000), signed(e + 3, 10_000)]
    barrier = threading.Barrier(2)

    def other_epoch(doc):
        barrier.wait(timeout=5)
        try:
            return publish(doc)["status"]
        except pg.Error as exc:
            assert "exceeds supplier receivable" in str(exc) or "epoch is stale" in str(
                exc
            )
            return "refused"

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(other_epoch, competing)) == ["accepted", "refused"]
    assert store.emission_statement(miner)["scheduled_microusd"] == 30_000
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        doc = signed(e + 4, 1000)
        with pytest.raises(pg.Error, match="requires read committed issuance"):
            q.execute(
                "SELECT public.lab_arena_publish_weight_state_v1('finney',71,%s,%s,%s)",
                (doc["epoch"], doc["state_hash"], Json(doc)),
            )
    # Operators can reconcile the accepted target to its exact identity even
    # when no validator has yet reported a reveal.
    from pathlib import Path

    report = (
        Path(__file__).parents[2] / "scripts/tool_market_epoch_targets.sql"
    ).read_text()
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute("SET TRANSACTION READ ONLY")
        q.execute("SET LOCAL ROLE lab_arena_owner")
        q.execute(report.replace(":'epoch'", "%s"), (e,))
        epoch, state_hash, block, recipient, target, emission, val, reports = (
            q.fetchone()
        )
        assert (epoch, state_hash, recipient, target, reports) == (
            e,
            first["state_hash"],
            miner,
            7000,
            0,
        )
        assert int(block) == val["block"] and int(emission) == val["emission_microusd"]
    transport.close()
