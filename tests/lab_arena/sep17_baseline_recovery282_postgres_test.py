"""Exact cancelled recovery278 fixture; source-only same-round recovery282."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import arena_sep17_baseline_recovery as operator
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.sep17_baseline_recovery278_postgres_test import (
    _insert_rows, _restore as restore_original,
)

ROOT = Path(__file__).parents[2]
PRIVATE = Path('/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private')
SNAPSHOT = PRIVATE / 'sep17-recovery282-terminal-snapshot-20260917T080011Z.json'
SNAPSHOT_SHA = 'bd53fb0caa0696403da6f5676419da25c9c70fbc7ed103e55c03e74267005dd7'
MIGRATION = ROOT / 'scripts/282-arena-2026-09-17-baseline-recovery.sql'
ROUND = operator.ROUND
BASELINE = operator.BASELINE
ARCHIVE = 'arena-2026-09-17-rerun278archive'
ARCHIVE_BASELINE = 'baseline-2026-09-17-rerun278archive'


def test_template_matches_exact_operator_and_protected_terminal_identity():
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    captured = json.loads(SNAPSHOT.read_text())
    compact = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    replacements = {
        '__RECOVERY_SOURCE_SIZE_BYTES__': str(operator.SOURCE_SIZE),
        '__RECOVERY_SOURCE_SHA256__': operator.SOURCE_SHA256,
        '__RECOVERY_SOURCE_COMMIT__': operator.SOURCE_COMMIT,
        '__TERMINAL_NONBASELINE_LEDGER_HASH__': captured['nonbaseline_ledger']['hash'],
        '__TERMINAL_EXECUTE_COST_JSON__': compact(captured['execute_cost']),
        '__TERMINAL_SCORE_COST_JSON__': compact(captured['score_cost']),
        '__OLD_SCHEDULE_JSON__': compact(captured['round']['schedule']),
        '__FORWARD_SCHEDULE_JSON__': compact(operator.FORWARD_SCHEDULE),
        '__PRIOR278_AUTHORITY_HASH__': captured['prior278']['authority_hash'],
        '__PRIOR278_AUDIT_HASH__': captured['prior278']['audit_hash'],
    }
    rendered = Path(str(MIGRATION) + '.template').read_text()
    for token, value in replacements.items():
        assert token in rendered
        rendered = rendered.replace(token, value)
    assert rendered == MIGRATION.read_text()


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def restore_current(connection):
    assert SNAPSHOT.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    captured = json.loads(SNAPSHOT.read_text())
    with connection.cursor() as cursor:
        cursor.execute('DROP TABLE IF EXISTS public.lab_arena_sep17_baseline_recovery278_audit CASCADE')
        cursor.execute('DROP TABLE IF EXISTS public.lab_arena_sep17_baseline_recovery278_authority CASCADE')
    connection.commit()
    restore_original(connection)
    with connection.cursor() as cursor:
        # Install the historical owner schema against its original terminal
        # fixture. These inert installs do not invoke the expired278prepare RPC.
        for name in ('278-arena-2026-09-17-baseline-recovery.sql',
                     '279-sep17-reviewed-source-before-activation.sql',
                     '280-sep17-renamed-source-archive-before-activation.sql'):
            cursor.execute((ROOT / 'scripts' / name).read_text())
        cursor.execute('SET session_replication_role=replica')
        cursor.execute('TRUNCATE public.lab_arena_sep17_baseline_recovery278_audit,'
                       'public.lab_arena_sep17_baseline_recovery278_authority,'
                       'public.lab_arena_ledger,public.lab_arena_runs,'
                       'public.lab_arena_submissions,public.lab_arena_rounds '
                       'RESTART IDENTITY CASCADE')
        for table, key in (
            ('lab_arena_rounds', 'rounds'), ('lab_arena_submissions', 'submissions'),
            ('lab_arena_runs', 'runs'), ('lab_arena_ledger', 'ledger'),
            ('lab_arena_sep17_baseline_recovery278_authority', 'authority278'),
            ('lab_arena_sep17_baseline_recovery278_audit', 'audit278'),
        ):
            raw = captured['protected_rows'][key]
            documents = [json.loads(row) for row in (raw if isinstance(raw, list) else [raw])]
            for start in range(0, len(documents), 250):
                _insert_rows(cursor, table, documents[start:start + 250])
        cursor.execute("SELECT setval('public.lab_arena_ledger_entry_id_seq',"
                       '(SELECT max(entry_id) FROM public.lab_arena_ledger),true)')
        cursor.execute('SET session_replication_role=origin')
        cursor.execute('SELECT public.lab_arena_sep17_recovery278_archive_valid_v1(),'
                       'public.lab_arena_sep17_recovery278_nonbaseline_ledger_valid_v1()')
        assert cursor.fetchone() == (True, True)
    connection.commit()
    return captured


@pytest.fixture
def connection(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        # Each case gets its own database transaction state and frozen fixture.
        # Drop only this test-owned new recovery schema before restoring.
        with connection.cursor() as cursor:
            cursor.execute('DROP TABLE IF EXISTS public.lab_arena_sep17_baseline_recovery282_audit CASCADE')
            cursor.execute('DROP TABLE IF EXISTS public.lab_arena_sep17_baseline_recovery282_authority CASCADE')
        connection.commit()
        restore_current(connection)
        yield connection
    finally:
        connection.close()


def prepare(cursor, **changes):
    args = dict(size=operator.SOURCE_SIZE, sha=operator.SOURCE_SHA256,
                commit=operator.SOURCE_COMMIT, bank=operator.BANK_SHA256,
                schedule=operator.FORWARD_SCHEDULE)
    args.update(changes)
    cursor.execute('SELECT public.lab_arena_prepare_sep17_baseline_recovery282_v1('
                   '%s,%s,%s,%s,%s::jsonb)',
                   (args['size'], args['sha'], args['commit'], args['bank'],
                    json.dumps(args['schedule'])))
    return cursor.fetchone()[0]


def state_hashes(connection):
    """Only counts and digests can appear in assertion diagnostics."""
    out = {}
    with connection.cursor() as cursor:
        for table, order in (
            ('lab_arena_rounds', 'round_id'), ('lab_arena_submissions', 'submission_id'),
            ('lab_arena_runs', 'run_id'), ('lab_arena_ledger', 'entry_id'),
            ('lab_arena_sep17_baseline_recovery278_authority', 'round_id'),
            ('lab_arena_sep17_baseline_recovery278_audit', 'round_id'),
        ):
            cursor.execute("SELECT count(*),encode(extensions.digest(coalesce(string_agg("
                           "encode(extensions.digest(to_jsonb(x)::text,'sha256'),'hex'),'' ORDER BY "
                           + order + "),''),'sha256'),'hex') FROM public." + table + ' x')
            out[table] = cursor.fetchone()
    return out


def test_exact_install_archive_replay_and_progress_preserve_all_history(connection):
    before = state_hashes(connection)
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
    connection.commit()
    assert state_hashes(connection) == before
    with connection.cursor() as cursor:
        cursor.execute("SELECT has_function_privilege('lab_arena_service',"
                       "'public.lab_arena_prepare_sep17_baseline_recovery282_v1(bigint,text,text,text,jsonb)',"
                       "'EXECUTE'),has_function_privilege('service_role',"
                       "'public.lab_arena_prepare_sep17_baseline_recovery282_v1(bigint,text,text,text,jsonb)',"
                       "'EXECUTE'),has_table_privilege('lab_arena_service',"
                       "'public.lab_arena_sep17_baseline_recovery282_authority','SELECT')")
        assert cursor.fetchone() == (True, False, False)
        result = prepare(cursor)
        assert result['archived_runs'] == 35
        assert result['archived_ledger_entries'] == 6640
        assert result['preserved_nonparticipant_submissions'] == 8
        assert prepare(cursor)['status'] == 'existing'
    connection.commit()
    with connection.cursor() as cursor:
        cursor.execute('SELECT status,status_generation,stage_generation,configuration_doc '
                       'FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
        status, generation, stage_generation, config = cursor.fetchone()
        assert (status, generation, stage_generation) == ('stage1', 6, 5)
        captured = json.loads(SNAPSHOT.read_text())
        original = next(json.loads(row) for row in captured['protected_rows']['rounds']
                        if json.loads(row)['round_id'] == ROUND)['configuration_doc']
        assert {k:v for k,v in config.items() if k != 'schedule'} == {
            k:v for k,v in original.items() if k != 'schedule'}
        assert config['schedule'] == operator.FORWARD_SCHEDULE
        cursor.execute('SELECT count(*),count(DISTINCT assignment_id),max(attempt) '
                       'FROM public.lab_arena_runs WHERE round_id=%s', (ROUND,))
        assert cursor.fetchone() == (20, 20, 1)
        cursor.execute('SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s', (ARCHIVE,))
        assert cursor.fetchone() == (35,)
        cursor.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s', (ARCHIVE,))
        assert cursor.fetchone() == (6640,)
        cursor.execute('SELECT public.lab_arena_sep17_recovery282_prior_valid_v1(),'
                       'public.lab_arena_sep17_recovery282_archive_valid_v1(),'
                       'public.lab_arena_sep17_recovery282_nonbaseline_ledger_valid_v1()')
        assert cursor.fetchone() == (True, True, True)
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE round_id=%s AND icp_position=0", (ROUND,))
        cursor.execute('SET LOCAL session_replication_role=origin')
        assert prepare(cursor)['status'] == 'existing'
    connection.commit()
    after = state_hashes(connection)
    for table in ('lab_arena_sep17_baseline_recovery278_authority',
                  'lab_arena_sep17_baseline_recovery278_audit'):
        assert after[table] == before[table]


@pytest.mark.parametrize('defect', ['source', 'bank', 'schedule', 'terminal', 'miner', 'prior_audit', 'archive'])
def test_drift_refuses_transaction_without_partial_recovery(connection, defect):
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    connection.commit()
    before = state_hashes(connection)
    with pytest.raises(Exception):
        with connection.cursor() as cursor:
            kwargs = {}
            if defect == 'source': kwargs['sha'] = '0' * 64
            elif defect == 'bank': kwargs['bank'] = '0' * 64
            elif defect == 'schedule': kwargs['schedule'] = dict(operator.FORWARD_SCHEDULE, stage_1_close='2026-09-17T17:00:00Z')
            else:
                cursor.execute('SET LOCAL session_replication_role=replica')
                if defect == 'terminal':
                    cursor.execute('UPDATE public.lab_arena_runs SET terminal_cause=\'model_timeout\' '
                                   'WHERE round_id=%s AND status=\'failed\'', (ROUND,))
                elif defect == 'miner':
                    cursor.execute('UPDATE public.lab_arena_submissions SET created_at=created_at+interval \'1 second\' '
                                   'WHERE round_id=%s AND submission_id<>%s', (ROUND, BASELINE))
                elif defect == 'prior_audit':
                    cursor.execute('UPDATE public.lab_arena_sep17_baseline_recovery278_audit '
                                   'SET started_at=started_at+interval \'1 second\'')
                else:
                    cursor.execute('UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 '
                                   'WHERE round_id=\'arena-2026-09-17-archive\'')
                cursor.execute('SET LOCAL session_replication_role=origin')
            prepare(cursor, **kwargs)
    connection.rollback()
    assert state_hashes(connection) == before


def test_replay_refuses_current_and_both_archives_drift(connection):
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        assert prepare(cursor)['status'] == 'prepared'
    connection.commit()
    before = state_hashes(connection)
    defects = (
        "UPDATE public.lab_arena_submissions SET source_size_bytes=source_size_bytes+1 "
        "WHERE submission_id='baseline-2026-09-17'",
        "UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 "
        "WHERE round_id='arena-2026-09-17-rerun278archive'",
        "UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 "
        "WHERE round_id='arena-2026-09-17-archive'",
        "UPDATE public.lab_arena_sep17_baseline_recovery278_audit "
        "SET started_at=started_at+interval '1 second'",
    )
    for statement in defects:
        with pytest.raises(Exception, match='replay differs'):
            with connection.cursor() as cursor:
                cursor.execute('SET LOCAL session_replication_role=replica')
                cursor.execute(statement)
                cursor.execute('SET LOCAL session_replication_role=origin')
                prepare(cursor)
        connection.rollback()
        assert state_hashes(connection) == before


@pytest.mark.parametrize('field', ['inflight_calls', 'success_unresolved_calls'])
def test_terminal_cost_gate_fails_closed(connection, field):
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    connection.commit()
    before = state_hashes(connection)
    with pytest.raises(Exception, match='terminal seal differs'):
        with connection.cursor() as cursor:
            cursor.execute('ALTER FUNCTION public.lab_arena__successful_call_cost_state(text,text,text) '
                           'RENAME TO recovery282_test_original_cost_state')
            cursor.execute('CREATE FUNCTION public.lab_arena__successful_call_cost_state('
                           'p_submission_id text,p_kind text,p_provider text DEFAULT NULL)RETURNS jsonb '
                           'LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$ '
                           'SELECT public.recovery282_test_original_cost_state(p_submission_id,p_kind,p_provider) '
                           "|| jsonb_build_object('" + field + "',1) $$")
            prepare(cursor)
    connection.rollback()
    assert state_hashes(connection) == before


@pytest.mark.parametrize("missing_judgment", [False, True])
def test_normal_service_scores_twenty_and_publishes_positive_with_archives_intact(
    connection, database, tmp_path, monkeypatch, missing_judgment,
):
    """Only external model/judge work is a fixture; service/SQL math is real."""
    from datetime import datetime, timezone
    from lab_arena import scoring
    from tests.lab_arena.test_lab_arena_service_round import Harness
    from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
        _proof_company, _proof_breakdown, _proof_execution,
    )

    psycopg2, dsn = database
    harness = Harness(lambda: psycopg2.connect(**dsn), tmp_path,
                      challengers=[], runners=['recovery282-proof'])
    service, objects = harness.service, harness.objects
    harness.clock.now = datetime(2026, 9, 17, 12, tzinfo=timezone.utc)
    bank = json.loads((PRIVATE / 'sep17-original-bank.json').read_text())
    assert isinstance(bank, dict) and bank['round_id'] == ROUND
    from lab_arena import contracts
    assert hashlib.sha256(contracts.canonical_json(bank['icps']).encode()).hexdigest() == operator.BANK_SHA256
    objects.put(operator.BANK_REF, json.dumps(bank).encode())
    icps = bank['icps']
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        assert prepare(cursor)['status'] == 'prepared'
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute('SELECT run_id,icp_position FROM public.lab_arena_runs '
                       'WHERE round_id=%s ORDER BY icp_position', (ROUND,))
        executions = cursor.fetchall()
        assert len(executions) == 20
        for run_id, position in executions:
            _proof_execution(objects, icps[position], 0, position, run_id)
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                           ('arena/output/%s.json' % run_id, run_id))
        cursor.execute('SET LOCAL session_replication_role=origin')
    connection.commit()

    def fixture_judge_boundary(run, *, icp, companies, policy):
        document = json.loads(objects.get(run['output_ref']))
        validated = scoring.validate_scoring_output_document(document)
        return scoring.validate_breakdowns_for_item(
            validated['breakdowns'], icp=icp, companies=companies,
            max_scored_companies=int(policy['max_scored_companies']),
            integrity_policy=True, contacts_required=True,
        )

    monkeypatch.setattr(service, '_verified_breakdowns', fixture_judge_boundary)
    assert service.close_stage(ROUND, 1)['status'] == 'ok'
    for stage in (1, 2):
        assert service.open_scoring(ROUND, stage)['assignments'] == 10
        score_runs = service.store.list_runs(ROUND, stage=stage, kind='score')
        assert len(score_runs) == 10
        assert all(row['assignment_id'].endswith(':score') for row in score_runs)
        with connection.cursor() as cursor:
            cursor.execute('SET LOCAL session_replication_role=replica')
            for run in score_runs:
                position = int(run['icp_position'])
                company = _proof_company(icps[position], 0, position)
                document = scoring.build_scoring_output(
                    run['scored_run_id'], [_proof_breakdown(company, 40)],
                )
                ref = 'arena/score/%s.json' % run['run_id']
                objects.put(ref, json.dumps(document).encode())
                cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                               "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                               (ref, run['run_id']))
            if missing_judgment and stage == 1:
                cursor.execute("UPDATE public.lab_arena_runs SET status='failed',"
                               "terminal_cause='provider_error',output_ref=NULL "
                               "WHERE round_id=%s AND kind='score' AND icp_position=0", (ROUND,))
            cursor.execute('SET LOCAL session_replication_role=origin')
        connection.commit()
        closed = service.close_scoring(ROUND, stage)
        if missing_judgment and stage == 1:
            assert closed['status'] == 'cancelled'
            cancelled = service.store.get_round(ROUND)
            assert cancelled['status'] == 'cancelled'
            assert cancelled['publication_doc'] is None
            assert service.publish(ROUND)['status'] == 'stale'
            with connection.cursor() as cursor:
                cursor.execute('SELECT public.lab_arena_sep17_recovery282_archive_valid_v1()')
                assert cursor.fetchone() == (True,)
            return
        assert closed['status'] == 'closed'
        scored = service.score_stage(ROUND, stage)
        assert scored['status'] == 'ok'
        if stage == 1:
            assert service.open_stage(ROUND, 2)['status'] == 'ok'
            assert service.close_stage(ROUND, 2)['status'] == 'ok'
    assert service.publish(ROUND)['status'] == 'ok'
    row = service.store.get_round(ROUND)
    ranking = row['publication_doc']['final_ranking']
    assert row['status'] == 'published' and len(ranking) == 1
    assert ranking[0]['submission_id'] == BASELINE
    assert ranking[0]['eligible'] is True and ranking[0]['final_score'] > 0
    final_entries = service._score_entries_from_runs(row, range(20), 'final_score')
    assert len(final_entries) == 1
    runs = service.store.list_runs(ROUND, kind='execute')
    assert len(runs) == 20 and {r['icp_position'] for r in runs} == set(range(20))
    mean = sum(r['per_icp_score'] for r in runs) / 20
    assert ranking[0]['final_score'] == pytest.approx(mean)
    assert final_entries[0]['final_score'] == pytest.approx(mean)
    with connection.cursor() as cursor:
        assert prepare(cursor)['status'] == 'existing'
        cursor.execute('SELECT public.lab_arena_sep17_recovery282_prior_valid_v1(),'
                       'public.lab_arena_sep17_recovery282_archive_valid_v1(),'
                       'public.lab_arena_sep17_recovery282_nonbaseline_ledger_valid_v1()')
        assert cursor.fetchone() == (True, True, True)
        cursor.execute('SELECT count(*) FROM public.lab_arena_submissions '
                       'WHERE round_id=%s AND submission_id<>%s', (ROUND, BASELINE))
        assert cursor.fetchone() == (8,)
        cursor.execute('SELECT count(*) FROM public.lab_arena_ledger '
                       'WHERE round_id=%s AND submission_id<>%s', (ROUND, BASELINE))
        assert cursor.fetchone() == (15,)
