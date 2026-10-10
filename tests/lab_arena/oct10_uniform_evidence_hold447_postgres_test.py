"""The fresh post445 owner hold permits paid judge drain and no new work."""
import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import oct10_uniform_evidence_rejudge448_postgres_test as recovery
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, hotkey

database = recovery.database
migrated = recovery.migrated


@pytest.fixture
def unheld(database, migrated):
    yield from recovery._seeded(database, apply_hold=False)


def test_hold_preserves_outputs_costs_and_allows_current_judge_drain(unheld):
    psycopg, dsn = unheld
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = recovery._hold_render(cur)
            lease, token, _, _ = claim(store,recovery.ROUND,hotkey('judge-hold447'))
            assert lease['status'] == 'leased', lease
            before = recovery._snapshot(cur)
            cur.execute(sql)
            after = recovery._snapshot(cur)
            assert {k:v for k,v in before.items() if k!='hold'} == {k:v for k,v in after.items() if k!='hold'}
            assert after['hold']['operator_paused']
            cur.execute(sql)
            assert recovery._snapshot(cur) == after
            refused, _, _, _ = claim(store,recovery.ROUND,hotkey('judge-hold447-next'))
            assert refused['status'] == 'paused', refused
            # A normal in-flight bill call extends the lease after this hold.
            identity='sha256:'+'4'*64
            assert store.reserve_call(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),
                call_identity=identity,operation_id='openrouter.chat',provider='openrouter',
                funding_source='miner_key',amount_microusd=100,call_doc={},lease_ttl_seconds=120)['status']=='reserved'
            assert store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=identity)['status']=='dispatched'
            assert store.mark_uncertain(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=identity,call_doc={})['status']=='uncertain'
            assert complete(store,lease['run_id'],hash_lease_token(token),'accepted',output_ref='arena/test/drained447.json')['status']=='accepted'
            cur.execute(sql)
            assert recovery._snapshot(cur)['hold'] == after['hold']
            with pytest.raises(psycopg.Error,match='lab_arena_round_progression_paused'):
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",(recovery.ROUND,))
            cur.execute('ROLLBACK')


@pytest.mark.parametrize('mutation',[
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,actor_ref='foreign',pause_reason='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_generation=999 WHERE singleton",
    "UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET reward_activated_at=now() WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"') WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/wrong.json' WHERE round_id='arena-2026-10-10' AND kind='execute' AND status='accepted'",
    "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r445archive' AND kind='score'",
    "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal",
])
def test_hold_rejects_wrong_proof_with_no_writes(unheld,mutation):
    psycopg,dsn=unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=recovery._hold_render(cur)
            before=recovery._snapshot(cur)
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error,match='Oct10 uniform hold447'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur)==before


@pytest.mark.parametrize('mutation',[
    "ALTER FUNCTION public.lab_arena_open_stage(text,smallint,jsonb,integer[]) SET search_path=public,pg_catalog",
    "ALTER FUNCTION public.lab_arena_open_scoring_v2(text,smallint,jsonb) SECURITY INVOKER",
])
def test_fresh_hold_snapshot_cannot_authorize_changed_canonical_function(unheld,mutation):
    psycopg,dsn=unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            before=recovery._snapshot(cur)
            cur.execute('BEGIN')
            cur.execute(mutation)
            # The independently captured445 postimage still binds canonical SQL.
            sql=recovery._hold_render(cur)
            with pytest.raises(psycopg.Error,match='Oct10 uniform hold447 terminal inventory'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur)==before
