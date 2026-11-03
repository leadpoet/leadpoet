"""The exact November 3 hold survives canonical release without outcome writes."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, hotkey
from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import GUARD, OWNER, _rpc

HOLD = Path(__file__).parents[2] / 'scripts/603-arena-2026-11-03-verifier-scoring-hold.sql'
ROUND = 'arena-2026-11-03'
CANDIDATE = '161e7443deaa66598772d0fb4a21240db9f277f0'
ACTOR = 'canonical-active-release:' + CANDIDATE
REASON = 'nov03_improved_verifier_required603'
TOKEN = hash_lease_token('hold603-fixture-token')


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + (
        '371-lab-arena-operator-hold-transition-guard.sql',
        '374-lab-arena-preserve-operator-pause-reason.sql',
    ))


def _seed(conn, psycopg, dsn):
    with conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_rounds "
                    "(round_id,status,status_generation,stage_generation,configuration_doc,"
                    "participants,evaluation_date,icp_set_date,benchmark_ref) VALUES "
                    "(%s,'stage2',5,4,'{\"mode\":\"live\"}', '[]','2026-11-03',"
                    "'2026-11-02','arena/arena-2026-11-03/benchmark.json')", (ROUND,))
        cur.execute("INSERT INTO public.lab_arena_submissions "
                    "(submission_id,round_id,miner_hotkey,status) VALUES "
                    "('hold603-sub',%s,%s,'frozen')", (ROUND, hotkey('hold603-miner')))
        cur.execute("INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,lease_generation,stage_generation,"
                    "lease_token_hash,lease_expires_at) SELECT 'hold603-'||i,'hold603-'||i,"
                    "%s,'hold603-sub',%s,2,i,1,'execute','leased',1,4,%s,"
                    "now()+interval '1 hour' FROM generate_series(0,20) i",
                    (ROUND, hotkey('hold603-miner'), TOKEN))
        cur.execute("INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
                    "per_icp_score,terminal_cause,output_ref,result_doc) VALUES "
                    "('prior-score603','prior-score603',%s,'hold603-sub',%s,1,0,1,"
                    "'score','accepted',3,'hold603-0',0.75,'accepted','arena/prior.json',"
                    "'{\"terminal_status\":\"accepted\"}')", (ROUND, hotkey('hold603-miner')))
        cur.execute("INSERT INTO public.lab_arena_ledger "
                    "(entry_kind,miner_hotkey,round_id,run_id,amount_microusd,entry_doc) "
                    "VALUES ('settlement',%s,%s,'prior-score603',1234,'{\"paid\":true}')",
                    (hotkey('hold603-miner'), ROUND))
        cur.execute('SET session_replication_role=origin')
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET guard_generation=751')
    _rpc(conn, 'lab_arena_acquire_restart_guard_v1', GUARD, OWNER, 751, 600,
         CANDIDATE, 'all', ACTOR)
    # Natural completions do not alter the captured 21-lease restart snapshot.
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    try:
        for i in range(10):
            assert complete(store, f'hold603-{i}', TOKEN, 'accepted',
                            output_ref=f'arena/hold603-{i}.json')['status'] == 'accepted'
    finally:
        store._transport.close()


def _snapshot(cur):
    return {
        table: _row(cur, f"SELECT coalesce(jsonb_agg(to_jsonb(t) ORDER BY {key}),'[]'::jsonb) "
                   f"FROM public.{table} t")
        for table, key in (
            ('lab_arena_restart_claim_control', 'singleton'),
            ('lab_arena_rounds', 'round_id'), ('lab_arena_submissions', 'submission_id'),
            ('lab_arena_runs', 'run_id'), ('lab_arena_ledger', 'entry_id'),
        )
    }


def _row(cur, sql):
    cur.execute(sql)
    return cur.fetchone()[0]


def _render(cur):
    # Substitute only the fixture's exact preimage, as in historical hold tests.
    sql = HOLD.read_text()
    substitutions = {
        'db061410be3979646bc8d8f5d50d97dec6050791a6550a16e3719fce708f14ec':
            _row(cur, "SELECT substr(guard_commitment,8) FROM public.lab_arena_restart_claim_control"),
        '7a3b581c660367b270e0ea0f331a9e3d8126fe648572dbde04ee59bd7b6455c5':
            _row(cur, "SELECT substr(owner_commitment,8) FROM public.lab_arena_restart_claim_control"),
        '242331b1dc0752323a3d475058ee556000bf506461a73f1b3a6e4ea8709abfd4':
            _row(cur, "SELECT encode(extensions.digest(captured_leases::text,'sha256'),'hex') "
                 "FROM public.lab_arena_restart_claim_control"),
        'e91499fe3037f17d0363f94ffdcfe383f49ceace8a365dc40c4a72277e21337c':
            _row(cur, "SELECT encode(extensions.digest(configuration_doc::text,'sha256'),'hex') "
                 "FROM public.lab_arena_rounds"),
        'a7b48bac1538146d8ec5ceacec4ac6b054a4537621e903ef516dd430ff5d4452':
            _row(cur, "SELECT encode(extensions.digest(participants::text,'sha256'),'hex') "
                 "FROM public.lab_arena_rounds"),
    }
    for original, fixture in substitutions.items():
        assert original in sql
        sql = sql.replace(original, fixture)
    return sql


def test_hold_only_changes_pause_preserves_drain_and_survives_release(database):
    psycopg, dsn = database
    with closing(psycopg.connect(**dsn)) as conn:
        conn.autocommit = True
        _seed(conn, psycopg, dsn)
        with conn.cursor() as cur:
            sql = _render(cur)
            before = _snapshot(cur)
            cur.execute(sql)
            after = _snapshot(cur)
            expected = before['lab_arena_restart_claim_control'][0] | {
                'operator_paused': True, 'pause_reason': REASON,
            }
            assert after == before | {'lab_arena_restart_claim_control': [expected]}
            cur.execute(sql)
            assert _snapshot(cur) == after
        # A renewal must retain the hold, even though actor and expiry change.
        _rpc(conn, 'lab_arena_acquire_restart_guard_v1', GUARD, OWNER, 752, 1200,
             CANDIDATE, 'all', 'canonical-renewal603')
        with conn.cursor() as cur:
            renewed = _snapshot(cur)
            cur.execute(sql)
            assert _snapshot(cur) == renewed
        store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
        try:
            assert claim(store, ROUND, hotkey('hold603-runner'))[0]['status'] == 'paused'
            for i in range(10,21):
                assert complete(store, f'hold603-{i}', TOKEN, 'accepted',
                                output_ref=f'arena/hold603-{i}.json')['status'] == 'accepted'
            for phase, function in (
                ('gateway_destructive','lab_arena_authorize_restart_phase_v1'),
                ('gateway_ready','lab_arena_mark_restart_ready_v1'),
                ('validator_destructive','lab_arena_authorize_restart_phase_v1'),
                ('validator_ready','lab_arena_mark_restart_ready_v1'),
            ):
                _rpc(conn, function, GUARD, OWNER, 752, phase)
            released = _rpc(conn, 'lab_arena_release_restart_guard_v1',
                            GUARD, OWNER, 752, 'canonical-release603')
            assert released['guard_present'] is False
            assert released['operator_paused'] is True and released['paused'] is True
            assert claim(store, ROUND, hotkey('hold603-runner'))[0]['status'] == 'paused'
            with conn.cursor() as cur:
                assert _row(cur, 'SELECT pause_reason FROM public.lab_arena_restart_claim_control') == REASON
                for mutation in (
                    "UPDATE public.lab_arena_rounds SET status='stage2_closed'",
                    "UPDATE public.lab_arena_runs SET per_icp_score=0.9 WHERE run_id='hold603-0'",
                    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='hold603-20'",
                ):
                    with pytest.raises(psycopg.Error, match='lab_arena_(round_progression|claims)_paused'):
                        cur.execute(mutation)
                    cur.execute('ROLLBACK')
                preserved = _snapshot(cur)
                with pytest.raises(psycopg.Error, match='Nov03 verifier hold603 restart'):
                    cur.execute(sql)
                cur.execute('ROLLBACK')
                assert _snapshot(cur) == preserved
                # An explicit later hold release must never be undone by replay.
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason=''")
                unpaused = _snapshot(cur)
                with pytest.raises(psycopg.Error, match='Nov03 verifier hold603 restart'):
                    cur.execute(sql)
                cur.execute('ROLLBACK')
                assert _snapshot(cur) == unpaused
                final = _snapshot(cur)
                assert final['lab_arena_ledger'] == before['lab_arena_ledger']
                assert final['lab_arena_rounds'] == before['lab_arena_rounds']
                assert final['lab_arena_submissions'] == before['lab_arena_submissions']
                assert next(r for r in final['lab_arena_runs'] if r['run_id']=='prior-score603') == next(
                    r for r in before['lab_arena_runs'] if r['run_id']=='prior-score603')
        finally:
            store._transport.close()


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_restart_claim_control SET owner_commitment='sha256:'||repeat('f',64)",
    "UPDATE public.lab_arena_restart_claim_control SET guard_generation=753",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('f',64)",
    "UPDATE public.lab_arena_restart_claim_control SET candidate_commit=repeat('a',40)",
    "UPDATE public.lab_arena_restart_claim_control SET restart_phase='gateway_ready'",
    "UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=now()-interval '1 second'",
    "UPDATE public.lab_arena_restart_claim_control SET captured_leases=captured_leases-'0'::int",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='foreign'",
    "UPDATE public.lab_arena_rounds SET published_at=now()",
    "UPDATE public.lab_arena_rounds SET participants='[{}]'::jsonb",
    "UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc||'{\"changed\":true}'::jsonb",
    "UPDATE public.lab_arena_runs SET kind='score' WHERE run_id='hold603-20'",
    "UPDATE public.lab_arena_runs SET kind='score',status='submitted' WHERE run_id='hold603-20'",
])
def test_changed_preimage_fails_without_writes(database, mutation):
    psycopg, dsn = database
    with closing(psycopg.connect(**dsn)) as conn:
        conn.autocommit = True
        _seed(conn, psycopg, dsn)
        with conn.cursor() as cur:
            sql = _render(cur)
            cur.execute('SET session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET session_replication_role=origin')
            before = _snapshot(cur)
            with pytest.raises(psycopg.Error, match='Nov03 verifier hold603'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert _snapshot(cur) == before
