"""Dated service fixtures install a reader, without upgrading reward writes."""
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    database_with_lab_arena_migration, prepare_correction_reward_history_reads,
)

SIGNATURE = 'public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)'


def test_exact_reader_preserves_historical_activation_and_missing_rpc_fails_closed():
    generator = database_with_lab_arena_migration()
    pg, dsn = next(generator)
    connect = lambda: pg.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    try:
        with connect() as c, c.cursor() as q:
            q.execute("SELECT pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure),nspacl::text FROM pg_namespace WHERE nspname='public'")
            before = q.fetchone()
            q.execute('SELECT to_regprocedure(%s)', (SIGNATURE,))
            assert q.fetchone()[0] is None
        with pytest.raises(ArenaStoreError, match='correction_reward_history'):
            store.published_reward_bases()
        prepare_correction_reward_history_reads(connect)
        prepare_correction_reward_history_reads(connect)
        assert store.published_reward_bases() == []
        with connect() as c, c.cursor() as q:
            q.execute("SELECT pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure),nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert q.fetchone() == before
            q.execute('SELECT p.prosrc,pg_get_userbyid(p.proowner),p.prosecdef,p.proconfig FROM pg_proc p WHERE p.oid=%s::regprocedure', (SIGNATURE,))
            body, owner, security_definer, settings = q.fetchone()
            source = (Path(__file__).parents[2] / 'scripts/696-lab-arena-correction-reward-history.sql').read_text()
            assert body == source.split('AS $history696$')[1].split('$history696$')[0]
            assert owner == 'lab_arena_owner' and security_definer
            assert settings == ['search_path=pg_catalog, public']
            q.execute('SELECT count(*) FROM lab_arena_scoring_supersessions')
            assert q.fetchone()[0] == 0
            q.execute("SELECT to_regprocedure('public.lab_arena_correct_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)')")
            assert q.fetchone()[0] is None
    finally:
        store.close()
        generator.close()
