"""Operator donor projections cannot authorize ordinary terminal rewrites."""
import pytest
from tests.lab_arena import nov04_baseline_source621_postgres_test as source

database = source.database
prepared = source.prepared
SQL = source.SCRIPTS / '630-arena-2026-11-04-baseline-donor-evidence.sql'


def test_exact_donor_guards_preserve_acl_replay_and_refuse_unreceipted_write(prepared):
    p = prepared
    source.apply(p)
    text = SQL.read_text()
    schema = text.split('DO $data630$')[0]
    restore = 'DO $restore630$' + text.split('DO $restore630$')[1].split('NOTIFY')[0]
    with p.conn.cursor() as q:
        q.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = q.fetchone()[0]
        q.execute("SELECT proname,to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%'")
        metadata = dict(q.fetchall())
        q.execute(schema + restore + 'COMMIT;')
        q.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert q.fetchone()[0] == acl
        q.execute("SELECT proname,to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%'")
        assert dict(q.fetchall()) == metadata
        q.execute(schema + restore + 'COMMIT;')
    p.conn.commit()
    with pytest.raises(p.pg.Error):
        with p.conn.cursor() as q:
            q.execute("UPDATE public.lab_arena_runs SET output_ref='unreceipted-output.json' "
                      "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                      "AND icp_position=0 AND attempt=2", (source.ROUND, source.BASELINE))
    p.conn.rollback()
