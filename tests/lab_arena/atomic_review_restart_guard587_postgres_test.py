"""Review reservation and restart ownership serialize in real PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena.store import hash_lease_token, new_lease_token
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.unfinished_review_recovery585_postgres_test import _begin, _verdict
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '587-lab-arena-atomic-review-restart-guard.sql'


@pytest.fixture(scope='module')
def recovery_database():
    from tests.lab_arena import restore_recorded_results538_postgres_test as prior
    databases = prior.database.__wrapped__()
    for psycopg2, dsn in databases:
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as c:
                for name in ('492-lab-arena-review-provider-observations.sql',
                             '562-lab-arena-review-context-integrity-reason.sql',
                             '563-lab-arena-review-source-digest-reason.sql',
                             '566-lab-arena-research-review-result.sql',
                             '585-lab-arena-review-deadline-recovery.sql',
                             '586-lab-arena-publication-restart-drain.sql'):
                    c.execute((SCRIPTS / name).read_text())
                c.execute("CREATE ROLE review587_migrator NOLOGIN NOSUPERUSER INHERIT")
                c.execute("GRANT lab_arena_owner TO review587_migrator")
                c.execute("ALTER SCHEMA public OWNER TO review587_migrator")
                c.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner")
                c.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                acl = c.fetchone()[0]
                c.execute("SET ROLE review587_migrator")
                c.execute(MIGRATION.read_text())
                c.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert c.fetchone()[0] == acl
        yield psycopg2, dsn


@pytest.fixture(autouse=True)
def clear_guard(recovery_superuser):
    yield
    with recovery_superuser.cursor() as c:
        c.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'::jsonb WHERE singleton")


def acquire(connection):
    with connection.cursor() as c:
        c.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        generation = c.fetchone()[0]
        c.execute("SELECT public.lab_arena_acquire_restart_guard_v1(%s,%s,%s,600,%s,'gateway','review587-test')",
                  ('lab_arena_restart_guard:' + 'a'*64, 'lab_arena_restart_owner:' + 'b'*64, generation, 'c'*40))
        return c.fetchone()[0]


def raw_begin(connection, submission, miner, token=None):
    with connection.cursor() as c:
        c.execute("SELECT public.lab_arena_begin_submission_review(%s,%s,%s,50000,'anthropic/claude-sonnet-5',2,1000)",
                  (submission, miner, hash_lease_token(token or new_lease_token())))
        return c.fetchone()[0]


def blocked(observer, waiter, blocker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with observer.cursor() as c:
            c.execute('SELECT %s=ANY(pg_blocking_pids(%s))', (blocker, waiter))
            if c.fetchone()[0]:
                return
        time.sleep(.01)
    pytest.fail('expected transaction lock was not observed')


def test_no_guard_and_operator_pause_keep_existing_admission(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f01')
    with recovery_superuser.cursor() as c:
        c.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
    assert _begin(recovery_store, submission, miner)['status'] == 'claimed'
    assert recovery_store.code_review_schema()['atomic_restart_guard'] is True


@pytest.mark.parametrize('expired', [False, True])
def test_retained_guard_prevents_any_new_claim_or_ledger_write(recovery_store, recovery_superuser, expired):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f02' if expired else 'f03')
    acquire(recovery_superuser)
    if expired:
        with recovery_superuser.cursor() as c:
            c.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=clock_timestamp()-interval '1 second' WHERE singleton")
    before = recovery_store.get_submission(submission), recovery_store.list_ledger(submission_id=submission)
    assert _begin(recovery_store, submission, miner)['status'] == 'paused'
    assert (recovery_store.get_submission(submission), recovery_store.list_ledger(submission_id=submission)) == before


def test_completed_and_exact_replay_remain_read_only_under_guard(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f04')
    token = new_lease_token()
    claim = _begin(recovery_store, submission, miner, token)
    acquire(recovery_superuser)
    before = recovery_store.list_ledger(submission_id=submission)
    replay = _begin(recovery_store, submission, miner, token)
    assert replay['status'] == 'claimed' and replay['idempotent'] is True
    assert replay['call_identity'] == claim['call_identity']
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', _verdict('pass'), 1)['status'] == 'passed'
    before = recovery_store.get_submission(submission), recovery_store.list_ledger(submission_id=submission)
    assert _begin(recovery_store, submission, miner)['status'] == 'existing'
    assert (recovery_store.get_submission(submission), recovery_store.list_ledger(submission_id=submission)) == before


def test_reservation_first_forces_guard_to_wait_until_claim_commits(recovery_store, recovery_superuser, recovery_connect, monkeypatch):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f05')
    first, second = recovery_connect(), recovery_connect()
    try:
        assert raw_begin(first, submission, miner)['status'] == 'claimed'
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(acquire, second)
            try:
                blocked(recovery_superuser, second.get_backend_pid(), first.get_backend_pid())
                assert not result.done()
            finally:
                first.commit()
            assert result.result(timeout=3)['guard_present'] is True
            second.commit()
        assert recovery_store.get_submission(submission)['code_review_claim'] is not None
        from scripts.arena_first_cutover_idle import require_idle
        from scripts import lab_arena_restart_claim_guard as guard
        state = result.result(timeout=3)
        monkeypatch.setattr(guard, '_owned', lambda _: {**state, 'restart_phase': 'gateway_destructive'})
        args = SimpleNamespace(scope='gateway', generation=state['guard_generation'])
        service = SimpleNamespace(active_rounds=lambda: [{'round_id': 'unused'}],
                                  _round=lambda _: {'round_id': 'unused', 'status': 'open'},
                                  _store=SimpleNamespace(
                                      code_review_schema=recovery_store.code_review_schema,
                                      list_submissions=lambda _, status: [recovery_store.get_submission(submission)] if status == 'accepted' else []))
        with pytest.raises(ValueError, match='claimed'):
            require_idle(service, now=datetime.now(timezone.utc), restart_guard=args)
    finally:
        first.close(); second.close()


def test_guard_first_forces_reservation_to_wait_then_pause(recovery_store, recovery_superuser, recovery_connect):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f06')
    first, second = recovery_connect(), recovery_connect()
    try:
        acquire(first)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(raw_begin, second, submission, miner)
            try:
                blocked(recovery_superuser, second.get_backend_pid(), first.get_backend_pid())
                assert not result.done()
            finally:
                first.commit()
            assert result.result(timeout=3)['status'] == 'paused'
            second.commit()
        assert recovery_store.get_submission(submission)['code_review_claim'] is None
        assert recovery_store.list_ledger(submission_id=submission) == []
    finally:
        first.close(); second.close()


def test_missing_singleton_fails_closed(recovery_store, recovery_connect):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f07')
    connection = recovery_connect()
    try:
        with connection.cursor() as c:
            c.execute('DELETE FROM public.lab_arena_restart_claim_control WHERE singleton')
        with pytest.raises(Exception, match='no rows'):
            raw_begin(connection, submission, miner)
    finally:
        connection.rollback(); connection.close()


def test_migration_reapply_preserves_definition_and_security(recovery_superuser):
    with recovery_superuser.cursor() as c:
        c.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(pg_proc)-'prosrc' FROM pg_proc WHERE oid IN ('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure,'public.lab_arena_code_review_schema_v1()'::regprocedure) ORDER BY oid")
        before = c.fetchall()
        c.execute(MIGRATION.read_text())
        c.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(pg_proc)-'prosrc' FROM pg_proc WHERE oid IN ('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure,'public.lab_arena_code_review_schema_v1()'::regprocedure) ORDER BY oid")
        assert c.fetchall() == before


def test_retained_guard_never_dispatches_runtime_provider(recovery_store, recovery_superuser):
    from tests.lab_arena.test_lab_arena_code_review_runtime import (
        ReviewTransport, _archive, _review_body, _reviewer, _source,
    )
    _, submission, _ = accepted_recovery_submission(recovery_store, 'f08')
    payload = _archive(_source())
    transport = ReviewTransport(lambda parameters: {'body': _review_body(parameters)})
    reviewer, _, _, _ = _reviewer(payload, transport, store=recovery_store)
    acquire(recovery_superuser)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'paused'
    assert transport.sent == []
    assert recovery_store.get_submission(submission)['code_review_claim'] is None
    assert recovery_store.list_ledger(submission_id=submission) == []


def test_guard_does_not_hide_malformed_rpc_inputs(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'f09')
    acquire(recovery_superuser)
    with pytest.raises(Exception, match='input_invalid'):
        with recovery_superuser.cursor() as c:
            c.execute("SELECT public.lab_arena_begin_submission_review(%s,%s,'malformed',50000,'model',2,1000)", (submission, miner))
    assert recovery_store.get_submission(submission)['code_review_claim'] is None


def test_review_direct_trigger_publish_and_guard_do_not_form_a_lock_cycle(recovery_store, recovery_superuser, recovery_connect):
    round_id, submission, miner = accepted_recovery_submission(recovery_store, 'f10')
    with recovery_superuser.cursor() as c:
        c.execute("SET session_replication_role='replica'")
        try:
            c.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s", (round_id,))
        finally:
            c.execute("SET session_replication_role='origin'")
    # Isolate the production hold trigger's lock order from unrelated final
    # baseline/result validators; migration586 separately tests full publication.
    with recovery_superuser.cursor() as c:
        c.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_complete_baseline_final_guard")
        c.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_publication_baseline_guard")
        c.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_integrity_publication_guard")
        c.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        c.execute("SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_operator_hold_transition_guard_v1()'::regprocedure),'sha256'),'hex')")
        assert c.fetchone()[0] == '754b81efbc162250f062f03b920a732d4439ce0ecc6aae590edae5c30218dffc'
    review, publisher, controller = recovery_connect(), recovery_connect(), recovery_connect()
    def publish():
        with publisher.cursor() as c:
            c.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (round_id,))
        publisher.commit()
    try:
        for connection in (review, publisher, controller):
            with connection.cursor() as c:
                c.execute("SET deadlock_timeout='200ms'; SET statement_timeout='5s'")
        with publisher.cursor() as c:
            c.execute('SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE', (round_id,))
        with controller.cursor() as c:
            c.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")
        with ThreadPoolExecutor(max_workers=3) as pool:
            reviewing = pool.submit(raw_begin, review, submission, miner)
            try:
                blocked(recovery_superuser, review.get_backend_pid(), publisher.get_backend_pid())
                publishing = pool.submit(publish)
                try:
                    blocked(recovery_superuser, publisher.get_backend_pid(), controller.get_backend_pid())
                except BaseException:
                    if publishing.done():
                        publishing.result()
                    raise
                renewing = pool.submit(acquire, controller)
                try:
                    assert renewing.result(timeout=3)['guard_present'] is True
                    controller.commit()
                    publishing.result(timeout=3)
                    assert reviewing.result(timeout=3)['status'] == 'paused'
                    review.commit()
                finally:
                    controller.rollback()
                    publisher.rollback()
                    review.rollback()
            finally:
                controller.rollback()
                publisher.rollback()
                review.rollback()
        assert recovery_store.get_submission(submission)['code_review_claim'] is None
        assert recovery_store.list_ledger(submission_id=submission) == []
        assert recovery_store.get_round(round_id)['status'] == 'published'
    finally:
        review.close(); publisher.close(); controller.close()
        with recovery_superuser.cursor() as c:
            c.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_complete_baseline_final_guard")
            c.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_publication_baseline_guard")
            c.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_integrity_publication_guard")
            c.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
