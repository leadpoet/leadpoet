"""Forward-only642 then660 on native cutoffs and retained637 finality."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import re
import time

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena import nov06_late_review635_postgres_test as late
from tests.lab_arena import nov06_ninth_review637_postgres_test as ninth
from tests.lab_arena import nov06_unused_ninth643_postgres_test as unused_ninth
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    PRIMARY, ALTERNATE, SIGNATURES, begin, failure, verdict,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.prospective_admission_primary_test import DEEPSEEK, receipt, reviewer
from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _source
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)
from tests.lab_arena.replacement_cutoff639_postgres_test import clock
from tests.lab_arena.queued_submission_replacement_postgres_test import _doc
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '660-lab-arena-prospective-admission-primary.sql'
CUTOFF = SCRIPTS / '642-lab-arena-prospective-replacement-cutoff.sql'


@pytest.fixture
def ninth_context():
    # Each finality outcome needs a true pre642 database, not a prior test's
    # additive column/trigger combined with regenerated historical receipts.
    database = ninth.database.__wrapped__()
    context = None
    try:
        context = ninth.context.__wrapped__(next(database))
        yield next(context)
    finally:
        if context is not None:
            context.close()
        database.close()


def restore_predecessor(cursor):
    text = MIGRATION.read_text()
    for i, signature in enumerate(SIGNATURES):
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        fixed = cursor.fetchone()[0]
        replacements = re.findall(r'v_old:=\$old660_'+str(i)+r'_(\d+)\$(.*?)\$old660_'+str(i)+r'_\1\$;\n  v_new:=\$new660_'+str(i)+r'_\1\$(.*?)\$new660_'+str(i)+r'_\1\$;', text, re.S)
        assert replacements
        for _, old, new in reversed(replacements):
            assert new in fixed
            fixed = fixed.replace(new, old)
        cursor.execute(fixed)


def function_snapshot(cursor):
    cursor.execute("SELECT oid::regprocedure::TEXT,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
                   "FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return {name: (definition, metadata) for name, definition, metadata in cursor.fetchall()}


def migration_state_snapshot(connection):
    with connection.cursor() as cursor:
        functions = function_snapshot(cursor)
        rows = {}
        for table in ('lab_arena_rounds', 'lab_arena_submissions',
                      'lab_arena_ledger', 'lab_arena_restart_claim_control'):
            cursor.execute(f'SELECT to_jsonb(t) FROM public.{table} t ORDER BY to_jsonb(t)::TEXT')
            rows[table] = cursor.fetchall()
        return functions, rows


def assert_primary_upgrade(before, after):
    before_functions, before_rows = before
    after_functions, after_rows = after
    assert after_rows == before_rows
    expected = {
        SIGNATURES[0].removeprefix('public.'): '03009634a9dcde9342a7428d73eaf4285bf341811241a1d51e5d54bc8b8b04b9',
        SIGNATURES[1].removeprefix('public.'): '91f4d574ab3d1d704228adac40507f6c4fb80a49fd091019ca713af1a0ff8a96',
        SIGNATURES[2].removeprefix('public.'): '55c3a3644ed53787bcff9af034e6be7b2a325474503e1bc5aa761de92a6d4d16',
    }
    changed = {name for name in before_functions if after_functions[name] != before_functions[name]}
    assert changed == set(expected)
    for name, (definition, metadata) in after_functions.items():
        assert metadata == before_functions[name][1]
        if name in expected:
            assert hashlib.sha256(definition.encode()).hexdigest() == expected[name]


def accepted_recovery_submission(store, suffix, *, cutoff_minutes=5):
    """Use a real review window for new600 rounds; change no frozen fields."""
    now = datetime.now(timezone.utc)
    round_id = 'arena-2099-03-01-' + hashlib.sha256(suffix.encode()).hexdigest()[:12]
    submission = suffix + '-recovery'
    miner = hotkey('primary660-' + suffix)
    config = round_config(round_id, [hotkey('runner660-' + suffix)])
    config['schedule'].update({
        'submission_open': (now - timedelta(hours=2)).isoformat(),
        'submission_cutoff': (now + timedelta(minutes=cutoff_minutes)).isoformat(),
        'benchmark_deadline': (now + timedelta(minutes=cutoff_minutes + 60)).isoformat(),
    })
    assert store.create_round(round_id, config)['status'] == 'created'
    assert store.register_submission(round_id, submission, miner,
                                     source_submission_doc(round_id, submission))['status'] == 'registered'
    assert store.accept_submission_with_credentials(round_id, submission, miner,
               encrypted_runtime_credentials(submission))['status'] == 'ok'
    return round_id, submission, miner


@pytest.fixture(scope='module')
def recovery_database():
    generator = late.database.__wrapped__()
    connection = None
    try:
        pg, dsn, _ = next(generator)
        connection = pg.connect(**dsn)
        connection.autocommit = True
        with connection.cursor() as cursor:
            sql = late.M635.read_text()
            # Install only committed function definitions; no dated receipt,
            # recovery membership, row changes, or additional claims are made.
            cursor.execute(sql.split('DO $recover635$')[0]
                           + 'DO $restore635$' + sql.split('DO $restore635$', 1)[1])
            ninth = (SCRIPTS / '637-arena-2026-11-06-ninth-review-recovery.sql').read_text()
            cursor.execute(ninth.split('DO $recover637$')[0]
                           + 'DO $restore637$' + ninth.split('DO $restore637$', 1)[1])
            # Retain actual old Sonnet episodes through the642 backfill. Their
            # prior native claims make NULL/legacy policy durable, not invented.
            store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
            try:
                for delta in (-1, 0, 1):
                    rid, sid, miner = accepted_recovery_submission(
                        store, 'primary660-legacyedge-' + str(delta), cutoff_minutes=180)
                    cutoff = datetime.fromisoformat(store.get_round(rid)['configuration_doc']['schedule']['submission_cutoff'])
                    with clock(connection, cutoff - timedelta(seconds=3600)):
                        token = new_lease_token()
                        assert begin(store, sid, miner, PRIMARY, token)['status'] == 'claimed'
                        assert store.finish_submission_review(sid, miner, token, 'error',
                            failure(PRIMARY, 'code_review_transport_error'), None)['status'] == 'error'
                    _age_review_error(connection, sid)
            finally:
                store.close()
            cursor.execute(CUTOFF.read_text())
            cursor.execute(MIGRATION.read_text())
        yield pg, dsn
    finally:
        if connection is not None:
            connection.close()
        generator.close()


def finish(store, submission, miner, token, model, *, error='code_review_provider_refused', actual=1234, **extra):
    return store.finish_submission_review(submission, miner, token, 'error',
        {**failure(model, error), **extra}, actual)


def test_requires_exact642_body_and_retains_native_cutoff(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        cutoff_definition = cursor.fetchone()[0]
        assert hashlib.sha256(cutoff_definition.encode()).hexdigest() == 'd101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff'
        cutoff = '- pg_catalog.make_interval(secs => COALESCE(v_round.replacement_freeze_seconds,3600))'
        old = cutoff_definition.replace(cutoff, "- INTERVAL '1 hour'")
        assert hashlib.sha256(old.encode()).hexdigest() == '44e1e4125cd373913cb1f02419ba17b7ffb1f45e8b55e7190aa256e85f5a6b1c'
        cursor.execute(old)
        before = function_snapshot(cursor)
        try:
            with pytest.raises(Exception, match='primary660 preimage differs'):
                cursor.execute(MIGRATION.read_text())
            cursor.execute('ROLLBACK')
            assert function_snapshot(cursor) == before
        finally:
            cursor.execute('ROLLBACK')
            cursor.execute(cutoff_definition)
        before = function_snapshot(cursor)
        cursor.execute(MIGRATION.read_text())
        after = function_snapshot(cursor)
        combined = after[SIGNATURES[0].removeprefix('public.')][0]
        assert hashlib.sha256(combined.encode()).hexdigest() == '03009634a9dcde9342a7428d73eaf4285bf341811241a1d51e5d54bc8b8b04b9'
        assert combined.count(cutoff) == 1
        # Independent composition check: the exact cutoff is the only delta
        # from the previously qualified660 function body.
        assert hashlib.sha256(combined.replace(cutoff, "- INTERVAL '1 hour'").encode()).hexdigest() == 'b1af0faa9fddadf702b56bd30d98d9dd4c86b1ffc4cf020e72b1b57ea431214b'
        for name in before:
            if name not in {signature.removeprefix('public.') for signature in SIGNATURES}:
                assert after[name] == before[name]


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('operation', ['register', 'credentials', 'source', 'review'])
def test_new600_native_cutoff_edges(recovery_store, recovery_superuser, delta, operation):
    rid, original, miner = accepted_recovery_submission(
        recovery_store, f'primary660-edge-{operation}-{delta}', cutoff_minutes=180)
    row = recovery_store.get_round(rid)
    assert row['replacement_freeze_seconds'] == 600
    cutoff = datetime.fromisoformat(row['configuration_doc']['schedule']['submission_cutoff'])
    replacement = f'primary660-replacement-{operation}-{delta}'
    if operation in ('credentials', 'source'):
        assert recovery_store.register_submission(rid, replacement, miner,
            _doc(rid, replacement, 124))['status'] == 'registered'
    moment = cutoff - timedelta(seconds=600) + timedelta(microseconds=delta)
    original_row = recovery_store.get_submission(original)
    with clock(recovery_superuser, moment):
        if operation == 'register':
            result = recovery_store.register_submission(rid, replacement, miner,
                _doc(rid, replacement, 124))
        elif operation == 'credentials':
            result = recovery_store.accept_submission_with_credentials(rid, replacement, miner,
                encrypted_runtime_credentials(replacement))
        elif operation == 'source':
            result = recovery_store.accept_submission_source_with_credentials(rid, replacement, miner,
                encrypted_runtime_credentials(replacement), 'sha256:' + '1' * 64, 'sha256:' + '2' * 64)
        else:
            result = begin(recovery_store, original, miner, DEEPSEEK)
        if operation == 'review':
            assert result['status'] == ('deferred' if delta < 0 else 'claimed')
        else:
            assert result['status'] == (('registered' if operation == 'register' else 'ok')
                                       if delta < 0 else 'replacement_closed')
    if operation != 'review' or delta < 0:
        assert recovery_store.list_ledger(submission_id=original) == []
    if operation == 'review' and delta < 0 or operation != 'review' and delta >= 0:
        assert recovery_store.get_submission(original) == original_row


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_retained_null_round_native3600_review_edges(recovery_store, recovery_superuser, delta):
    suffix = 'primary660-legacyedge-' + str(delta)
    sid = suffix + '-recovery'
    row = recovery_store.get_submission(sid)
    round_row = recovery_store.get_round(row['round_id'])
    assert round_row['replacement_freeze_seconds'] is None
    cutoff = datetime.fromisoformat(round_row['configuration_doc']['schedule']['submission_cutoff'])
    before = recovery_store.list_ledger(submission_id=sid)
    assert [entry['entry_doc']['review_model'] for entry in before if entry['entry_kind'] == 'reservation'] == [PRIMARY]
    with clock(recovery_superuser, cutoff - timedelta(seconds=3600) + timedelta(microseconds=delta)):
        # Even with new host preference, the actual old primary remains bound.
        if delta >= 0:
            assert begin(recovery_store, sid, row['miner_hotkey'], DEEPSEEK) == {
                'status': 'route_required', 'review_model': PRIMARY}
        result = begin(recovery_store, sid, row['miner_hotkey'], PRIMARY)
        assert result['status'] == ('deferred' if delta < 0 else 'claimed')
    after = recovery_store.list_ledger(submission_id=sid)
    assert after[:len(before)] == before
    if delta < 0:
        assert after == before
        assert recovery_store.get_submission(sid) == row
    else:
        assert result['attempt'] == 2
        assert [entry['entry_doc']['review_model'] for entry in after if entry['entry_kind'] == 'reservation'] == [PRIMARY, PRIMARY]


def test_native_readers_allow_migration_without_row_or_metadata_changes(
    recovery_connect, recovery_store, recovery_superuser,
):
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
    accepted_recovery_submission(recovery_store, 'primary660-reader')
    before = migration_state_snapshot(recovery_superuser)
    reader = recovery_connect()
    try:
        with reader.cursor() as cursor:
            for table in ('lab_arena_rounds', 'lab_arena_submissions'):
                cursor.execute(f'SELECT count(*) FROM public.{table}')
                assert cursor.fetchone()[0] > 0
        with recovery_superuser.cursor() as cursor:
            cursor.execute("SELECT mode FROM pg_locks WHERE pid=%s AND relation IN "
                           "('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass)",
                           (reader.get_backend_pid(),))
            assert cursor.fetchall() == [('AccessShareLock',), ('AccessShareLock',)]
            cursor.execute(MIGRATION.read_text())
        assert_primary_upgrade(before, migration_state_snapshot(recovery_superuser))
    finally:
        reader.rollback()
        reader.close()


def test_loaded_old_sonnet_begin_completes_before_prospective_host_activation():
    # Separate native DB: no other test's unfinished claim can stand in for
    # the release's required zero-review/in-flight drain condition.
    database = recovery_database.__wrapped__()
    controller = blocker = worker_connection = store = worker_store = executor = None
    try:
        pg, dsn = next(database)
        controller = pg.connect(**dsn)
        controller.autocommit = True
        store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
        with controller.cursor() as cursor:
            restore_predecessor(cursor)
            cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
            assert hashlib.sha256(cursor.fetchone()[0].encode()).hexdigest() == (
                'd101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff')
        _, submission, miner = accepted_recovery_submission(store, 'primary660-loaded-old')
        blocker = pg.connect(**dsn)
        with blocker.cursor() as cursor:
            cursor.execute('SELECT 1 FROM public.lab_arena_submissions WHERE submission_id=%s FOR UPDATE',
                           (submission,))
            assert cursor.fetchone() == (1,)
        worker_connection = pg.connect(**dsn)
        worker_pid = worker_connection.get_backend_pid()
        worker_store = ArenaStore(PsycopgTransport(lambda: worker_connection, pool_size=1))
        token = new_lease_token()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(begin, worker_store, submission, miner, PRIMARY, token)
        deadline = time.monotonic() + 5
        while True:
            with controller.cursor() as cursor:
                cursor.execute('SELECT wait_event_type,pg_blocking_pids(pid) FROM pg_stat_activity WHERE pid=%s',
                               (worker_pid,))
                wait = cursor.fetchone()
            if wait and wait[0] == 'Lock' and blocker.get_backend_pid() in wait[1]:
                break
            assert not future.done(), 'old native begin did not reach the row-lock wait'
            assert time.monotonic() < deadline, 'old native begin did not block in time'
            time.sleep(.01)
        with controller.cursor() as cursor:
            cursor.execute("SELECT DISTINCT mode FROM pg_locks WHERE pid=%s AND granted AND relation IN "
                           "('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass)",
                           (worker_pid,))
            assert 'RowShareLock' in {row[0] for row in cursor.fetchall()}
        # No altered/instrumented function: the actual old body is already
        # executing and waits on its native submission FOR UPDATE statement.
        before = migration_state_snapshot(controller)
        with controller.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        assert not future.done()
        assert_primary_upgrade(before, migration_state_snapshot(controller))
        blocker.rollback()
        claimed = future.result(timeout=5)
        assert claimed['status'] == 'claimed' and claimed['attempt'] == 1
        assert finish(store, submission, miner, token, PRIMARY,
                      error='code_review_transport_error')['status'] == 'error'
        row = store.get_submission(submission)
        assert row['code_review_doc']['model'] == PRIMARY
        ledger = store.list_ledger(submission_id=submission)
        assert [entry['entry_doc']['review_model'] for entry in ledger
                if entry['entry_kind'] == 'reservation'] == [PRIMARY]
        # Complete the old invocation, close its DB connection, and observe
        # zero reviewing rows before simulating prospective host activation.
        worker_store.close()
        worker_store = None
        with controller.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_submissions WHERE code_review_status='reviewing'")
            assert cursor.fetchone() == (0,)
        deadline = time.monotonic() + 2
        while True:
            with controller.cursor() as cursor:
                cursor.execute('SELECT count(*) FROM pg_stat_activity WHERE pid=%s', (worker_pid,))
                if cursor.fetchone() == (0,):
                    break
            assert time.monotonic() < deadline, 'old review backend did not drain'
            time.sleep(.01)
        _, next_submission, next_miner = accepted_recovery_submission(store, 'primary660-after-drain')
        next_token = new_lease_token()
        assert begin(store, next_submission, next_miner, DEEPSEEK, next_token)['status'] == 'claimed'
        assert finish(store, next_submission, next_miner, next_token, DEEPSEEK,
                      error='code_review_transport_error')['status'] == 'error'
        assert [entry['entry_doc']['review_model'] for entry in store.list_ledger(submission_id=next_submission)
                if entry['entry_kind'] == 'reservation'] == [DEEPSEEK]
        assert store.list_ledger(submission_id=submission) == ledger
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if executor is not None:
            executor.shutdown(wait=True)
        if worker_store is not None:
            worker_store.close()
        if store is not None:
            store.close()
        if controller is not None:
            controller.close()
        database.close()


@pytest.mark.parametrize('locked_table,key', [
    ('lab_arena_rounds', 'round_id'),
    ('lab_arena_submissions', 'submission_id'),
])
def test_busy_native_writers_refuse_migration_immediately_without_changes(
    recovery_connect, recovery_store, recovery_superuser, locked_table, key,
):
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
    round_id, submission, miner = accepted_recovery_submission(
        recovery_store, 'primary660-nowait-' + key)
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'claimed'

    def snapshot():
        return migration_state_snapshot(recovery_superuser)

    before_functions, before_rows = snapshot()
    blocker = recovery_connect()
    try:
        with blocker.cursor() as cursor:
            # UPDATE takes RowExclusive; SELECT FOR UPDATE alone is RowShare.
            target = round_id if key == 'round_id' else submission
            cursor.execute(f'UPDATE public.{locked_table} SET {key}={key} WHERE {key}=%s RETURNING {key}',
                           (target,))
            assert cursor.fetchone() == (target,)
        started = time.monotonic()
        with recovery_superuser.cursor() as cursor:
            try:
                with pytest.raises(Exception) as failure:
                    cursor.execute(MIGRATION.read_text())
                elapsed = time.monotonic() - started
                assert failure.value.pgcode == '55P03'
                assert elapsed < 1.0, f'busy migration waited {elapsed:.3f}s'
            finally:
                cursor.execute('ROLLBACK')
        assert snapshot() == (before_functions, before_rows)
    finally:
        blocker.rollback()
        blocker.close()

    with recovery_superuser.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    after_functions, after_rows = snapshot()
    assert after_rows == before_rows
    changed = {name for name in before_functions if after_functions[name] != before_functions[name]}
    assert changed == {signature.removeprefix('public.') for signature in SIGNATURES}
    for name in before_functions:
        assert after_functions[name][1] == before_functions[name][1]
    assert recovery_store.code_review_schema(require_prospective_primary=True)['prospective_primary']['model'] == DEEPSEEK


def test_fail_before_then_upgrade_preserves_rows_and_uses_actual_new_primary(recovery_store, recovery_superuser):
    text = MIGRATION.read_text()
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
        functions = function_snapshot(cursor)
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary660-before')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, DEEPSEEK)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.get_submission(submission)
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, DEEPSEEK) == {'status': 'route_required', 'review_model': PRIMARY}
    with pytest.raises(ArenaStoreError, match='primary schema mismatch'):
        recovery_store.code_review_schema(require_prospective_primary=True)
    with recovery_superuser.cursor() as cursor:
        cursor.execute(text)
        after = function_snapshot(cursor)
        assert functions.keys() == after.keys()
        for name, previous in functions.items():
            assert after[name][1] == previous[1]
            if name not in {signature.removeprefix('public.') for signature in SIGNATURES}:
                assert after[name] == previous
    assert recovery_store.get_submission(submission) == before
    assert recovery_store.list_ledger(submission_id=submission) == ledger
    assert begin(recovery_store, submission, miner, PRIMARY) == {'status': 'route_required', 'review_model': DEEPSEEK}
    assert begin(recovery_store, submission, miner, DEEPSEEK)['attempt'] == 2


def test_migration_replay_and_unknown_final_body_preserve_all_functions(recovery_store, recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        fixed = function_snapshot(cursor)
        cursor.execute(MIGRATION.read_text())
        assert function_snapshot(cursor) == fixed

        restore_predecessor(cursor)
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[-1],))
        original_schema = cursor.fetchone()[0]
        cursor.execute(original_schema.replace('SELECT', '-- unknown660\n  SELECT', 1))
        before = function_snapshot(cursor)
        with pytest.raises(Exception, match='primary660 preimage differs'):
            cursor.execute(MIGRATION.read_text())
        cursor.execute('ROLLBACK')
        assert function_snapshot(cursor) == before
        cursor.execute(original_schema)
        cursor.execute(MIGRATION.read_text())
        assert function_snapshot(cursor) == fixed


def test_unknown_parent_recovery_helper_is_rejected_without_function_changes(recovery_store, recovery_superuser):
    signature = 'public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        original = cursor.fetchone()[0]
        cursor.execute(original.replace('SELECT', '-- unknown660parent\n SELECT', 1))
        before = function_snapshot(cursor)
        with pytest.raises(Exception, match='primary660 parent637 differs'):
            cursor.execute(MIGRATION.read_text())
        cursor.execute('ROLLBACK')
        assert function_snapshot(cursor) == before
        cursor.execute(original)


@pytest.mark.parametrize('status', ['passed', 'rejected', 'error'])
def test_finalized_ninth_receipt_keeps_exact_sol_attempt_and_prior_history(ninth_context, status):
    store, connection, sql, target, *_ = ninth_context
    ninth.apply(connection, sql)
    assert ninth.finish_ninth(store, target, status)['status'] == status
    ninth.apply(connection, ninth.M638.read_text())
    before_row = store.get_submission(ninth.SID)
    before_ledger = store.list_ledger(submission_id=ninth.SID)
    before_history = store.scoring_policy_history(ninth.ROUND)
    before_runs = store.list_runs(ninth.ROUND)
    with connection.cursor() as cursor:
        functions = function_snapshot(cursor)
        cursor.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c')
        before_control = cursor.fetchall()
        cursor.execute('SELECT to_jsonb(h) FROM public.lab_arena_scoring_supersessions h ORDER BY to_jsonb(h)::TEXT')
        before_receipts = cursor.fetchall()
    ninth.apply(connection, CUTOFF.read_text())
    ninth.apply(connection, MIGRATION.read_text())
    assert store.get_submission(ninth.SID) == before_row
    assert store.list_ledger(submission_id=ninth.SID) == before_ledger
    with connection.cursor() as cursor:
        after = function_snapshot(cursor)
    for signature in ('lab_arena__nov06_late_review635_v1(lab_arena_rounds,lab_arena_submissions)',
                      'lab_arena_submissions_frozen_v1()'):
        assert after[signature] == functions[signature]
    assert before_row['code_review_attempts'] == 4
    assert before_row['code_review_doc']['model'] == ALTERNATE
    assert [entry['entry_doc']['review_model'] for entry in before_ledger if entry['entry_kind'] == 'reservation'] == [
        PRIMARY, PRIMARY, PRIMARY, ALTERNATE]
    assert store.scoring_policy_history(ninth.ROUND) == before_history
    assert store.list_runs(ninth.ROUND) == before_runs
    with connection.cursor() as cursor:
        cursor.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c')
        assert cursor.fetchall() == before_control
        cursor.execute('SELECT to_jsonb(h) FROM public.lab_arena_scoring_supersessions h ORDER BY to_jsonb(h)::TEXT')
        assert cursor.fetchall() == before_receipts
    if status == 'passed':
        assert begin(store, ninth.SID, before_row['miner_hotkey'], ALTERNATE)['status'] == 'existing'
    else:
        with pytest.raises(ArenaStoreError, match='submission_missing'):
            begin(store, ninth.SID, before_row['miner_hotkey'], ALTERNATE)
    assert store.get_submission(ninth.SID) == before_row
    assert store.list_ledger(submission_id=ninth.SID) == before_ledger


def test_unused_ninth643_then642_then660_preserves_closed_history(ninth_context):
    store, connection, *_ = ninth_context
    sql = unused_ninth.prepared(ninth_context)
    before_closure = unused_ninth.snapshot(store, connection)
    original = store.get_submission(ninth.SID)
    original_ledger = store.list_ledger(submission_id=ninth.SID)
    assert original['code_review_attempts'] == 3
    assert [entry['entry_doc']['review_model'] for entry in original_ledger
            if entry['entry_kind'] == 'reservation'] == [PRIMARY] * 3

    ninth.apply(connection, sql)
    closed = unused_ninth.snapshot(store, connection)
    proof = closed['receipts'][-1]['recovery_doc']['reviews'][ninth.SID]
    assert proof['outcome'] == 'preminer_gate_closed_without_fourth'
    assert proof['additional_reservations'] == proof['additional_cost_microusd'] == 0
    assert closed['ledger'] == before_closure['ledger']
    ninth.apply(connection, CUTOFF.read_text())
    before = unused_ninth.snapshot(store, connection)
    assert before['rows'] == closed['rows']
    assert before['ledger'] == closed['ledger']
    assert before['receipts'] == closed['receipts']
    assert before['runs'] == closed['runs']
    assert before['control'] == closed['control']
    history = store.scoring_policy_history(ninth.ROUND)
    with connection.cursor() as cursor:
        functions = function_snapshot(cursor)

    ninth.apply(connection, MIGRATION.read_text())
    assert unused_ninth.snapshot(store, connection) == before
    assert store.scoring_policy_history(ninth.ROUND) == history
    with connection.cursor() as cursor:
        after_functions = function_snapshot(cursor)
    assert set(after_functions) == set(functions)
    for name, previous in functions.items():
        assert after_functions[name][1] == previous[1]
        if name not in {signature.removeprefix('public.') for signature in SIGNATURES}:
            assert after_functions[name] == previous
    for model in (PRIMARY, ALTERNATE, DEEPSEEK):
        with pytest.raises(ArenaStoreError, match='submission_missing'):
            begin(store, ninth.SID, original['miner_hotkey'], model)
    assert store.get_submission(ninth.SID)['code_review_attempts'] == 3
    assert store.list_ledger(submission_id=ninth.SID) == original_ledger
    assert unused_ninth.snapshot(store, connection) == before


@pytest.mark.parametrize('primary', [PRIMARY, DEEPSEEK])
def test_refusal_episode_keeps_two_real_primary_and_two_alternate_claims(recovery_store, recovery_superuser, primary):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary660-'+('legacy' if primary == PRIMARY else 'new'))
    for index, model in enumerate([primary, primary, ALTERNATE, ALTERNATE], 1):
        if index > 1:
            _age_review_error(recovery_superuser, submission)
        token = new_lease_token()
        assert begin(recovery_store, submission, miner, model, token)['attempt'] == index
        assert begin(recovery_store, submission, miner, model, token)['idempotent'] is True
        finish(recovery_store, submission, miner, token, model, actual=None if index == 1 else 1234)
        row = recovery_store.get_submission(submission)
        assert row['code_review_doc']['review_primary_model'] == primary
        assert row['code_review_doc']['review_primary_attempts'] == min(index, 2)
        assert row['code_review_doc']['review_alternate_attempts'] == max(0, index-2)
        if index < 4:
            assert code_review_policy.review_model_for(row, prospective_primary_model=DEEPSEEK) == [primary, ALTERNATE, ALTERNATE][index-1]
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [row['entry_doc']['review_model'] for row in ledger if row['entry_kind'] == 'reservation'] == [primary, primary, ALTERNATE, ALTERNATE]
    assert sum(row['entry_kind'] == 'uncertain' for row in ledger) == 1
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == ledger


def test_deepseek_truncation_uses_next_existing_slot_and_preserves_unknown_cost(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary660-truncated')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, DEEPSEEK,
           error='review_response_invalid', error_reason='truncated', actual=None)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, DEEPSEEK) == {'status': 'route_required', 'review_model': ALTERNATE}
    token = new_lease_token()
    claim = begin(recovery_store, submission, miner, ALTERNATE, token)
    assert claim['attempt'] == 2 and claim['review_output_token_recommendation'] == 65536
    finish(recovery_store, submission, miner, token, ALTERNATE, error='code_review_transport_error')
    _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['attempt'] == 3
    finish(recovery_store, submission, miner, token, ALTERNATE, error='review_response_invalid')
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len([row for row in ledger if row['entry_kind'] == 'reservation']) == 3
    assert any(row['entry_kind'] == 'uncertain' for row in ledger)


@pytest.mark.parametrize('primary,other', [(PRIMARY, DEEPSEEK), (DEEPSEEK, PRIMARY)])
def test_transient_episode_cannot_change_primary_model(recovery_store, recovery_superuser, primary, other):
    _, submission, miner = accepted_recovery_submission(recovery_store,
        'primary660-transient-'+('old' if primary == PRIMARY else 'new'))
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, primary, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, primary, error='code_review_transport_error', actual=None)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, other) == {'status': 'route_required', 'review_model': primary}
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert begin(recovery_store, submission, miner, primary)['attempt'] == 2


def test_native_runtime_primary_pass_is_bound_to_real_ledger_and_final(recovery_store, recovery_superuser):
    _, submission, _ = accepted_recovery_submission(recovery_store, 'primary660-runtime')
    instance, _, _, transport = reviewer(lambda parameters: {'body': receipt(parameters)}, store=recovery_store)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(_archive(_source()))}
    assert instance.review(row)['status'] == 'passed'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len(transport.sent) == 1
    assert [item['entry_doc']['review_model'] for item in ledger if item['entry_kind'] == 'reservation'] == [DEEPSEEK]
    assert [item['amount_microusd'] for item in ledger if item['entry_kind'] == 'settlement'] == [1234]
    assert instance.review(recovery_store.get_submission(submission))['status'] == 'existing'
    assert recovery_store.list_ledger(submission_id=submission) == ledger


def test_actual_model_binding_and_db_first_schema_remain_fail_closed(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary660-binding')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    with pytest.raises(ArenaStoreError, match='input_invalid'):
        recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(PRIMARY), 1234)
    assert recovery_store.get_submission(submission)['code_review_status'] == 'reviewing'
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(DEEPSEEK), 1234)['status'] == 'passed'
    marker = recovery_store.code_review_schema(require_refusal_fallback=True, require_invalid_review_budget=True,
                                               require_prospective_primary=True)
    assert marker['refusal_fallback']['primary_model'] == PRIMARY
    assert marker['invalid_review_budget']['max_reservations'] == 3
    assert marker['prospective_primary']['model'] == DEEPSEEK
