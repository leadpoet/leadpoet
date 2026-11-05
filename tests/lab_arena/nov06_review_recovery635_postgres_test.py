"""Dated recovery uses the real receipt, source, ledger and fourth-claim path.

The disposable fixture substitutes its clock and native saved error documents;
production SQL retains the eight exact source/document/generation bindings.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import patch

import pytest

from lab_arena import code_review_policy
from lab_arena.store import ArenaStore, PsycopgTransport, new_lease_token
from tests.lab_arena.invalid_review_budget624_postgres_test import recovery_database as predecessor_database, finish
from tests.lab_arena.refusal_review_fallback599_postgres_test import PRIMARY, ALTERNATE, begin, verdict
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config, source_submission_doc, encrypted_runtime_credentials
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '635-arena-2026-11-06-invalid-review-recovery.sql'
ROUND = 'arena-2026-11-06'


def apply(connection, sql):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
    except Exception:
        with connection.cursor() as cursor:
            cursor.execute('ROLLBACK')
        raise


@pytest.fixture(scope='module')
def context():
    class ReviewDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 5, 23, 55, tzinfo=timezone.utc)
    clock_patch = patch.object(code_review_policy, 'datetime', ReviewDateTime)
    clock_patch.start()
    database = predecessor_database.__wrapped__()
    store = connection = None
    try:
        psycopg2, dsn = next(database)
        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        sql = MIGRATION.read_text()
        expected_text = sql.split('$expected635$')[1]
        expected = json.loads(expected_text)
        now = datetime.now(timezone.utc)
        cutoff = (now+timedelta(minutes=30)).isoformat()
        config = round_config(ROUND, [hotkey('review635-runner')])
        config['schedule'].update(submission_open=(now-timedelta(hours=2)).isoformat(),
            submission_cutoff=cutoff, benchmark_deadline=(now+timedelta(hours=1)).isoformat())
        assert store.create_round(ROUND, config)['status'] == 'created'
        for i, item in enumerate(expected):
            sid = item['submission_id']
            miner = hotkey('review635-'+str(i))
            assert store.register_submission(ROUND, sid, miner, source_submission_doc(ROUND, sid))['status'] == 'registered'
            with connection.cursor() as cursor:
                cursor.execute('UPDATE public.lab_arena_submissions SET source_archive_sha256=%s,source_normalized_sha256=%s WHERE submission_id=%s',
                    (item['source_archive_sha256'], item['source_normalized_sha256'], sid))
            assert store.accept_submission_with_credentials(ROUND, sid, miner, encrypted_runtime_credentials(sid))['status'] == 'ok'
            for attempt in range(1, 4):
                if attempt > 1:
                    _age_review_error(connection, sid)
                token = new_lease_token()
                assert begin(store, sid, miner, PRIMARY, token)['status'] == 'claimed'
                diagnostic = {'finish_reason': 'stop', 'native_finish_reason': 'end_turn',
                    'max_output_tokens': 32768, 'context_integrity': 'not_required',
                    'generation_id': item['generation_id']}
                assert finish(store, sid, miner, token, error_reason=item['code_review_doc']['error_reason'],
                    provider_diagnostic=diagnostic)['status'] == 'error'
            _age_review_error(connection, sid)
            row = store.get_submission(sid)
            for name in ('source_size_bytes', 'source_ref', 'code_review_doc'):
                item[name] = row[name]
        sql = sql.replace(expected_text, json.dumps(expected, separators=(',', ':'), sort_keys=True))
        sql = sql.replace('2026-11-06T00:00:00Z', cutoff)
        apply(connection, (SCRIPTS/'634-lab-arena-invalid-review-alternate.sql').read_text())
        yield store, connection, sql, expected
    finally:
        clock_patch.stop()
        if store:
            store.close()
        if connection:
            connection.close()
        database.close()


def test_apply_rejects_changed_receipt_then_grants_exact_eight(context):
    store, connection, sql, expected = context
    ledger_before = {item['submission_id']: store.list_ledger(submission_id=item['submission_id']) for item in expected}
    bad = sql.replace(expected[0]['generation_id'], 'gen-1-wrong')
    with pytest.raises(Exception, match='exact source or third receipt differs'):
        apply(connection, bad)
    for item in expected:
        assert not code_review_policy.review_retry_available(store.get_submission(item['submission_id']))
    with connection.cursor() as cursor:
        cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (ROUND,))
        round_before = cursor.fetchone()[0]
    apply(connection, sql)
    apply(connection, sql)
    with connection.cursor() as cursor:
        signature = 'public.lab_arena__nov06_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        helper = cursor.fetchone()[0]
        cursor.execute(helper.replace('SELECT COALESCE(', 'SELECT /* drift */ COALESCE(', 1))
    with pytest.raises(Exception, match='helper differs'):
        apply(connection, sql)
    with connection.cursor() as cursor:
        cursor.execute(helper)
    for item in expected:
        sid = item['submission_id']
        row = store.get_submission(sid)
        assert row['code_review_attempts'] == 3
        assert row['code_review_doc']['review_invalid_attempts'] == 3
        assert code_review_policy.review_retry_available(row)
        assert code_review_policy.review_model_for(row) == ALTERNATE
        assert store.list_ledger(submission_id=sid) == ledger_before[sid]
    with connection.cursor() as cursor:
        cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (ROUND,))
        assert cursor.fetchone()[0] == round_before
        cursor.execute("SELECT recovery_doc->'originals' FROM public.lab_arena_scoring_supersessions WHERE request_id='nov06-invalid-review635-v1'")
        originals = cursor.fetchone()[0]
        assert set(originals) == {x['submission_id'] for x in expected}
        assert all(x['submission']['code_review_doc']['retryable'] is False for x in originals.values())


def test_only_one_fourth_alternate_claim_and_no_fifth(context):
    store, connection, sql, expected = context
    sid = expected[0]['submission_id']
    miner = store.get_submission(sid)['miner_hotkey']
    _age_review_error(connection, sid)
    # The receipt binds the original start time: aging it after the grant
    # invalidates authority, rather than silently granting a changed original.
    assert begin(store, sid, miner, ALTERNATE)['status'] == 'exhausted'
    with connection.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_submissions s SET code_review_started_at=(h.recovery_doc->'originals'->s.submission_id->'submission'->>'code_review_started_at')::TIMESTAMPTZ FROM public.lab_arena_scoring_supersessions h WHERE h.request_id='nov06-invalid-review635-v1' AND s.submission_id=%s", (sid,))
    assert begin(store, sid, miner, PRIMARY) == {'status': 'route_required', 'review_model': ALTERNATE}
    tokens = [new_lease_token(), new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda token: begin(store, sid, miner, ALTERNATE, token), tokens))
    assert sorted(x['status'] for x in results) == ['busy', 'claimed']
    token = tokens[next(i for i, result in enumerate(results) if result['status'] == 'claimed')]
    assert begin(store, sid, miner, ALTERNATE, token)['idempotent'] is True
    assert finish(store, sid, miner, token, model=ALTERNATE)['status'] == 'error'
    assert store.get_submission(sid)['code_review_attempts'] == 4
    assert not code_review_policy.review_retry_available(store.get_submission(sid))
    _age_review_error(connection, sid)
    before = store.list_ledger(submission_id=sid)
    assert begin(store, sid, miner, ALTERNATE)['status'] == 'exhausted'
    apply(connection, sql)
    assert begin(store, sid, miner, ALTERNATE)['status'] == 'exhausted'
    assert store.list_ledger(submission_id=sid) == before


def test_remaining_fourth_can_pass_and_cost_history_remains(context):
    store, connection, sql, expected = context
    sid = expected[1]['submission_id']
    row = store.get_submission(sid)
    before = store.list_ledger(submission_id=sid)
    token = new_lease_token()
    assert begin(store, sid, row['miner_hotkey'], ALTERNATE, token)['attempt'] == 4
    assert store.finish_submission_review(sid, row['miner_hotkey'], token, 'passed', verdict(), 4321)['status'] == 'passed'
    after = store.list_ledger(submission_id=sid)
    assert after[:len(before)] == before
    assert sum(x['amount_microusd'] for x in after if x['entry_kind'] == 'settlement') == 3*1234+4321
    assert begin(store, sid, row['miner_hotkey'], ALTERNATE)['status'] == 'existing'


def test_after_cutoff_no_recovery_claim(context):
    store, connection, _, expected = context
    sid = expected[2]['submission_id']
    row = store.get_submission(sid)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role='replica'")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,submission_cutoff}',to_jsonb((clock_timestamp()-interval '1 second')::TEXT)) WHERE round_id=%s", (ROUND,))
        cursor.execute("SET session_replication_role='origin'")
    assert begin(store, sid, row['miner_hotkey'], ALTERNATE)['status'] == 'exhausted'
