"""624 follows605's live authority bodies and preserves its narrow guards."""
import hashlib

import pytest

from tests.lab_arena.nov03_admission605_postgres_test import (
    TARGETS, apply, begin, context, database, finish, snapshot,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import SIGNATURES
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '624-lab-arena-invalid-review-reservation-budget.sql'
PREIMAGES = (
    '6c9974bae7c8b42dedcdee6e4a9145672402f5e0c74c87f8a9fff56afafeddf5',
    '947841e794d9be0b7c997d5752f892a26952d4fae81ddf01ecb60b7fac25de24',
)


def test_624_refuses_incomplete599_chain_and_preserves_all_rows(context):
    _, conn, _, _, _ = context
    before = snapshot(conn)
    with conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        definition = cur.fetchone()[0]
    with pytest.raises(Exception, match='preimage differs'):
        apply(conn, MIGRATION.read_text())
    assert snapshot(conn) == before
    with conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        assert cur.fetchone()[0] == definition


def test_624_applies_after_exact_live605_review_bodies_and_preserves_all_rows(context):
    store, conn, m605, _, _ = context
    apply(conn, m605)
    with conn.cursor() as cur:
        for signature, expected in zip(SIGNATURES, PREIMAGES):
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest() == expected
        cur.execute("SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid IN "
            "('public.lab_arena__nov03_review605_v1(public.lab_arena_rounds,public.lab_arena_submissions)'::regprocedure,"
            "'public.lab_arena_nov03_review_recovery_authorized_v1(text)'::regprocedure,"
            "'public.lab_arena_submissions_frozen_v1()'::regprocedure,"
            "'public.lab_arena_close_stage(text,smallint)'::regprocedure)")
        authority_before = cur.fetchall()
    before = snapshot(conn)
    apply(conn, MIGRATION.read_text())
    assert snapshot(conn) == before
    with conn.cursor() as cur:
        for oid, definition, metadata in authority_before:
            cur.execute("SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s", (oid,))
            assert cur.fetchone() == (definition, metadata)
        for signature in SIGNATURES[:2]:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            definition = cur.fetchone()[0]
            assert 'v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);' in definition
            assert "v_round.status <> 'open' AND NOT v_nov03_recovery" in definition
    assert store.code_review_schema(require_invalid_review_budget=True)['invalid_review_budget']['max_reservations'] == 3
    # This exact receipt-bound scoped recovery still works after ordinary budget
    # changes; it neither opens another identity nor extends the605 deadline.
    finish(store, conn, TARGETS[0], status='passed')
    row = store.get_submission(TARGETS[0])
    assert row['code_review_status'] == 'passed'
    assert begin(store, TARGETS[0], row['miner_hotkey'])['status'] == 'existing'


def test_624_does_not_open605_recovery_for_unlisted_or_changed_identity(context):
    store, conn, m605, _, _ = context
    apply(conn, m605)
    apply(conn, MIGRATION.read_text())
    target = TARGETS[0]
    miner = store.get_submission(target)['miner_hotkey']
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role='replica'")
        cur.execute('UPDATE public.lab_arena_submissions SET submission_id=%s WHERE submission_id=%s', ('unlisted624', target))
        cur.execute("SET session_replication_role='origin'")
    assert begin(store, 'unlisted624', miner)['status'] == 'round_closed'
