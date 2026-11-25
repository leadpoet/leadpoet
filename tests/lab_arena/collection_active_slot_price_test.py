"""Only the actual funded model needs a price, and its lease cannot race into another model."""

from pathlib import Path

import pytest

from lab_arena import collection_intake, collection_prices, store
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.locked_collection_pipeline_test import registered
from lab_arena.collection_workflow import CHAMPION_REV


def test_idle_poller_never_queries_generator_or_judge_endpoints():
    def idle(**kwargs):
        assert kwargs == {"expected_model": None}
        return None

    assert collection_prices.claim_at_current_price(
        idle, lambda _: pytest.fail("idle polled a provider")
    ) == (None, None)


def test_an_unavailable_generator_cannot_charge_a_slot_and_changed_model_cannot_be_dispatched():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    numbers = (
        720,
        721,
        722,
        723,
        724,
        725,
        726,
        727,
        728,
        729,
        732,
        733,
        738,
        740,
        741,
        742,
        743,
        744,
        745,
        746,
        747,
        748,
        749,
        750,
        751,
        752,
        754,
    )
    mig = tuple(next(scripts.glob(f"{n}-*.sql")).name for n in numbers)
    for pg, dsn in database_with_lab_arena_migration(DEFAULT_MIGRATIONS + mig):

        def connect():
            c = pg.connect(**dsn)
            c.autocommit = True
            return c

        db = store.PsycopgTransport(connect, role="lab_arena_service")
        miner = registered(db, "A")

        def claim(**kw):
            return miner.claim(
                "scoped-prices",
                ["P7-s40401"],
                CHAMPION_REV,
                dict.fromkeys(collection_intake.MODELS, 0),
                dict.fromkeys(collection_intake.MODELS, 400000),
                **kw,
            )

        first, second = collection_intake.MODELS[:2]
        table = {"test": True}

        def unavailable(model):
            assert model == first
            raise ValueError("configured endpoint unavailable")

        with pytest.raises(ValueError, match="endpoint unavailable"):
            collection_prices.claim_at_current_price(claim, unavailable)
        assert claim(expected_model=None) == {"model": first}
        assert claim(expected_model=second) is None
        assert db.rpc("collection_batch_jobs_v1", {"p_batch": "scoped-prices"}) == []
        job, actual = collection_prices.claim_at_current_price(claim, lambda m: table)
        assert job["model"] == first and actual == table
        # Another worker consumed this position. Its first candidate price cannot start the next slot.
        registered(db, "B")
        assert claim(expected_model=None) == {"model": second}
        assert claim(expected_model=first) is None
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute("select count(*) from collection_jobs")
            assert q.fetchone() == (1,)
            q.execute(
                "select has_function_privilege('anon','collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)','execute')"
            )
            assert q.fetchone() == (False,)
        db.close()


def test_selective_endpoint_pricer_excludes_unrelated_and_unavailable_reference(
    monkeypatch,
):
    from lab_arena import collection_judge as judge

    policy = judge.BatchPolicy("test", "synthetic", "a" * 64)
    pinned = {"complete": "unchanged snapshot"}

    def measured(routes, *, models):
        assert routes is pinned
        assert models == sorted(
            {
                "mistral-generator",
                policy.version(policy.roles[0]),
                policy.version(policy.roles[1]),
            }
        )
        return {"authorized": True}

    monkeypatch.setattr(collection_prices, "endpoint_prices", measured)
    assert collection_prices.prices_for_assignment(pinned, policy)(
        "mistral-generator"
    ) == {"authorized": True}
