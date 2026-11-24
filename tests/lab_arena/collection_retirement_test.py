"""A champion award ends before the first paid collection job exists."""
import pytest
from psycopg2.extras import Json

from lab_arena import collection_rewards, weight_state
from leadpoet_canonical.arena_weights import derive_arena_weights
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.collection_rewards_test import BURN, GENESIS, MINER, SIGNER, insert_legacy_state
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration


@pytest.fixture
def db():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + (
        "720-locked-collection-intake.sql", "721-locked-collection-reward-epoch.sql",
        "722-collection-call-budget.sql", "723-collection-pilot-prices.sql",
        "724-collection-scenario-ranking.sql", "725-collection-generator-roster.sql",
        "726-collection-first-paid-job.sql", "727-collection-synthetic-funding.sql",
        "728-collection-pilot-attempts.sql", "729-collection-ready-judge-recovery.sql",
        "743-retire-champion-rewards.sql",
    ))


def test_independent_retirement_and_empty_signed_epoch(db):
    pg, dsn = db
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    epoch = 26600
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        old = insert_legacy_state(q, epoch-1)
        cfg = Json(dict(mode="live", network_name="finney", netuid=71))
        q.execute("INSERT INTO public.lab_arena_rounds(round_id, configuration_doc, rewards_enabled) VALUES ('arena-2026-11-24',%s,true)", (cfg,))
        q.execute("SELECT public.collection_reward_start_v1(),public.collection_reward_schedule_v1()")
        assert q.fetchone() == ({"first_epoch": None}, {})
        insert = "INSERT INTO public.collection_champion_retirement(first_epoch,previous_state_hash) VALUES (%s,%s)"
        q.execute("SAVEPOINT before_marker")
        for wrong_epoch, wrong_hash in ((epoch-1, old["state_hash"]), (epoch+1, old["state_hash"]), (epoch, 'sha256:'+'0'*64)):
            with pytest.raises(pg.Error, match="latest signed epoch"):
                q.execute(insert, (wrong_epoch, wrong_hash))
            q.execute("ROLLBACK TO before_marker")
        q.execute(insert, (epoch, old["state_hash"]))
        q.execute("SELECT public.collection_reward_start_v1(),public.collection_reward_schedule_v1()")
        assert q.fetchone() == ({"first_epoch": epoch}, {})
        q.execute("SAVEPOINT retired")
        for sql, params in (
            ("DELETE FROM public.collection_champion_retirement", ()),
            ("TRUNCATE public.collection_champion_retirement", ()),
            ("INSERT INTO public.lab_arena_rounds(round_id, configuration_doc, rewards_enabled) VALUES ('arena-2026-11-25',%s,true)", (cfg,)),
            ("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) VALUES ('retire-upload','arena-2026-11-24',%s,'uploading')", (BURN,)),
        ):
            with pytest.raises(pg.Error, match="retired|immutable"):
                q.execute(sql, params)
            q.execute("ROLLBACK TO retired")
        with pytest.raises(pg.Error, match="quote mismatch"):
            insert_legacy_state(q, epoch)
        q.execute("ROLLBACK TO retired")
        q.execute("SELECT public.collection_reward_quotes_v1(%s)", (epoch,))
        basis = dict(q.fetchone()[0], valuation=None)
        wrong_burn = weight_state.build_accepted_weight_state(
            SIGNER, network="finney", genesis_hash=GENESIS, netuid=71, epoch=epoch,
            valid_from_block=50, valid_until_block=409, reward_basis=basis, burn_hotkey=MINER,
            issued_at="2026-11-23T00:00:00Z",
        )
        with pytest.raises(pg.Error, match="original burn hotkey"):
            q.execute("SELECT public.lab_arena_publish_weight_state_v1('finney',71,%s,%s,%s)",
                      (epoch, wrong_burn["state_hash"], Json(wrong_burn)))
        q.execute("ROLLBACK TO retired")
        q.execute("SELECT count(*) FROM public.collection_reward_cutover")
        assert q.fetchone()[0] == 0
    from types import SimpleNamespace
    chain = SimpleNamespace(accepted_weight_epoch_scope=lambda: {
        "epoch":epoch, "genesis_hash":GENESIS, "valid_from_block":50, "valid_until_block":409,
    })
    def no_valuation(_block):
        pytest.fail("an empty collection epoch must not obtain a price")
    state = collection_rewards.epoch_state(store, chain, SIGNER, epoch=epoch, burn_hotkey=BURN, valuation_source=no_valuation)
    assert state["reward_basis"]["quotes_microusd"] == {}
    assert state["reward_basis"]["job_count"] == 0 and state["reward_basis"]["valuation"] is None
    vector = derive_arena_weights(state, [MINER, BURN])
    assert vector["sparse_uids"] == [1] and vector["sparse_weights_u16"] == [65535]
    assert vector["champion_share_ppb"] == vector["collection_share_ppb"] == 0
    assert store.get_weight_state("finney",71,epoch-1)["state_doc"] == old
    assert collection_rewards.epoch_state(store, chain, SIGNER, epoch=epoch, burn_hotkey=BURN, valuation_source=no_valuation) == state


def test_qualified_paid_launch_follows_burn_without_reviving_champions(db):
    from tests.lab_arena.collection_calibration_test import exercise_ordinary_cutover

    def retired_anchor(cursor, epoch):
        prior = insert_legacy_state(cursor, epoch-1)
        cursor.execute("INSERT INTO public.collection_champion_retirement(first_epoch,previous_state_hash) VALUES (%s,%s)",
                       (epoch, prior["state_hash"]))
        cursor.execute("SELECT public.collection_reward_quotes_v1(%s)", (epoch,))
        basis = cursor.fetchone()[0]
        state = weight_state.build_accepted_weight_state(
            SIGNER, network="finney", genesis_hash=GENESIS, netuid=71,
            epoch=epoch, valid_from_block=1, valid_until_block=49,
            reward_basis=dict(basis, valuation=None), burn_hotkey=BURN,
            issued_at="2026-11-23T00:00:00Z",
        )
        cursor.execute("SELECT public.lab_arena_publish_weight_state_v1('finney',71,%s,%s,%s)",
                       (epoch, state["state_hash"], Json(state)))
        return state

    exercise_ordinary_cutover(db, retired_anchor=retired_anchor)


def test_pricing_tool_accepts_signed_unpaid_epoch_and_rejects_active_paid_epoch(monkeypatch):
    from types import SimpleNamespace
    from lab_arena import chain, signing
    from scripts.collection_batch import cutover_anchor
    from tests.lab_arena.collection_rewards_test import EPOCH, state

    empty = state({})
    monkeypatch.setenv("LAB_ARENA_CHAIN_ENDPOINT", "wss://example.invalid")
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_ID", "test")
    monkeypatch.setattr(signing, "KmsSigner", lambda *a, **kw: SIGNER)
    monkeypatch.setattr(chain, "connect_substrate", lambda cfg: None)
    monkeypatch.setattr(chain, "ArenaChain", lambda *a: SimpleNamespace(
        client=SimpleNamespace(query=lambda *a, **kw: 1000000), close=lambda: None))
    monkeypatch.setattr(chain, "load_arena_cutover", lambda: None)
    monkeypatch.setattr(chain, "finalized_epoch_snapshot", lambda client: SimpleNamespace(
        settlement_epoch_id=lambda _: EPOCH, network_genesis_hash=GENESIS,
        current_block=60, block_hash="0x"+"b"*64))
    monkeypatch.setattr(ArenaStore, "get_weight_state", lambda *a: {"state_doc": empty})
    schedule = {}
    transport = SimpleNamespace(rpc=lambda *a: schedule)
    assert cutover_anchor(transport)["state_hash"] == empty["state_hash"]
    schedule.update(first_epoch=EPOCH+1, staged=True)
    assert cutover_anchor(transport)["epoch"] == EPOCH
    del schedule["staged"]
    with pytest.raises(ValueError, match="expected an unpaid"):
        cutover_anchor(transport)
