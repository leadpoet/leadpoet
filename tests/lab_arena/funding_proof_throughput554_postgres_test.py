"""Keep exact funding proof semantics with bounded historical receipt lookups."""
from pathlib import Path
from functools import partial
from datetime import timedelta
import hashlib
import json
import re
import time

import pytest

from tests import postgres_migration_harness as pg_harness
from tests.lab_arena import lab_arena_pg_harness as arena_pg
from tests.lab_arena import funding_proof_scalars526_postgres_test as scalar
from tests.lab_arena import proven_funding_recovery524_postgres_test as prior
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import reset
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
INDEX = ROOT / 'scripts/553-lab-arena-paid-native-request-index.sql'
SQL = ROOT / 'scripts/554-lab-arena-funding-proof-throughput.sql'
CURRENT_SQL = ROOT / 'scripts/558-lab-arena-funding-proof-keyed-lookups.sql'
CURRENT_POST = '23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6'
PRE = '46acfd38b92eb7a03613a4d55c72cc9af57339cf6602ed4611955af22b511cf6'
POST = 'a962ddae017d3e1efbbe05d296bd1b9c895523d4a5081b0a3fd2c78b9c7f3d73'
NAME = 'public.lab_arena_ledger_paid_native_request_idx'
# Source Supabase server17.6/170006 verified read-only on2026-10-29.
# Only this performance fixture uses its planner; functional controls stayPG15.
POSTGRES_IMAGE = 'postgres:17.6@sha256:00bc86618629af00d2937fdc5a5d63db3ff8450acf52f0636ec813c7f4902929'


def definition(q):
    q.execute('SELECT pg_get_functiondef(%s::regprocedure)', (prior.SIG,))
    return q.fetchone()[0]


def old_source(post):
    text = SQL.read_text()
    for label in ('duplicates554', 'outcomes554'):
        before = re.search(r'\$old_'+label+r'\$(.*?)\$old_'+label+r'\$', text, re.S)[1]
        after = re.search(r'\$new_'+label+r'\$(.*?)\$new_'+label+r'\$', text, re.S)[1]
        assert post.count(after) == 1
        post = post.replace(after, before)
    assert hashlib.sha256(post.encode()).hexdigest() == PRE
    return post


@pytest.fixture(scope='module')
def database():
    gen = scalar.database.__wrapped__()
    db = next(gen)
    try:
        c = db[0].connect(**db[1])
        c.autocommit = True
        with c.cursor() as q:
            # Exercise the committed concurrent migration as one autocommit SQL
            # statement, never inside the ordinary transaction migration helper.
            q.execute(INDEX.read_text())
            q.execute(INDEX.read_text())
            for number in (527, 551, 552):
                paths = list((ROOT/'scripts').glob(str(number)+'-*.sql'))
                assert len(paths) == 1
                q.execute(paths[0].read_text())
            assert hashlib.sha256(definition(q).encode()).hexdigest() == PRE
            q.execute(SQL.read_text())
            q.execute(SQL.read_text())
            assert hashlib.sha256(definition(q).encode()).hexdigest() == POST
        c.close()
        yield db
    finally:
        gen.close()


# Existing successful, adversarial, inherited-root and malformed JSON proofs
# run verbatim against the replacement source and actual index.
test_paid_new_dispatch_retires_stop_without_mutating_history = prior.test_paid_new_dispatch_retires_stop_without_mutating_history
test_untrusted_or_stale_paid_proof_never_retires_credit = prior.test_untrusted_or_stale_paid_proof_never_retires_credit
test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity = prior.test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity
test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof = prior.test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof
test_new_denial_consumes_old_watermark_and_stops_normal_retry_again = prior.test_new_denial_consumes_old_watermark_and_stops_normal_retry_again
test_every_exhausted_key_requires_its_own_new_paid_dispatch = prior.test_every_exhausted_key_requires_its_own_new_paid_dispatch
test_record_projection_preserves_malformed_and_exact_json_guards = scalar.test_record_projection_preserves_malformed_and_exact_json_guards


def test_replay_preserves_security_and_other_functions_and_rows(database):
    run, _ = prior.seed(database)
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        before = q.fetchall()
        rows = prior.snapshot(q)
        q.execute(old_source(definition(q)))
        q.execute('CREATE ROLE funding554_migrator NOLOGIN NOSUPERUSER INHERIT')
        q.execute('GRANT lab_arena_owner TO funding554_migrator')
        q.execute('GRANT USAGE,CREATE ON SCHEMA public TO funding554_migrator')
        q.execute('SET ROLE funding554_migrator')
        q.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert q.fetchone() == (False,)
        q.execute(SQL.read_text())
        q.execute(SQL.read_text())
        q.execute('RESET ROLE')
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        assert q.fetchall() == before
        assert prior.snapshot(q) == rows


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'security', 'volatility', 'path',
                                   'index', 'index_definition', 'index_owner', 'index_valid', 'index_ready'])
def test_source_security_and_index_guards_precede_replay(database, installed, drift):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        post = definition(q)
        if not installed:
            q.execute(old_source(post))
        q.execute('SAVEPOINT hostile')
        if drift == 'source':
            q.execute(definition(q).replace('BEGIN\n', 'BEGIN\n  NULL;\n', 1))
        elif drift == 'index':
            q.execute('DROP INDEX '+NAME)
        elif drift == 'index_definition':
            q.execute('DROP INDEX '+NAME)
            q.execute('CREATE INDEX lab_arena_ledger_paid_native_request_idx ON public.lab_arena_ledger(run_id)')
        elif drift == 'index_owner':
            # PostgreSQL index ownership follows the owning table.
            q.execute('ALTER TABLE public.lab_arena_ledger OWNER TO postgres')
        elif drift in ('index_valid', 'index_ready'):
            column = 'indisvalid' if drift == 'index_valid' else 'indisready'
            q.execute('UPDATE pg_index SET '+column+'=false WHERE indexrelid=%s::regclass', (NAME,))
        else:
            q.execute({
                'owner': 'ALTER FUNCTION '+prior.SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+prior.SIG+' TO anon',
                'security': 'ALTER FUNCTION '+prior.SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+prior.SIG+' VOLATILE',
                'path': 'ALTER FUNCTION '+prior.SIG+' SET search_path=public,pg_catalog',
            }[drift])
        body = SQL.read_text().split('DO $funding_throughput554$')[1].split('$funding_throughput554$;')[0]
        with pytest.raises(database[0].Error):
            q.execute('DO $test$'+body+'$test$')
        q.execute('ROLLBACK TO SAVEPOINT hostile')
        q.execute(post)


@pytest.mark.parametrize('fault', ['historical', 'old_floor', 'other_required_key', 'superseded'])
def test_historical_native_duplicate_keeps_exact_scope_and_latest_head(database, fault):
    run, denial = prior.seed(database)
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        funded = prior.clone(c, run)
        prior.call(c, funded, 'fresh', native_id='duplicate-native')
        other = prior.clone(c, run, 'historical', floor=run['scoring_generation_floor']-1 if fault == 'old_floor' else None)
        fp = 'sha256:'+'d'*64 if fault == 'other_required_key' else prior.FINGERPRINT
        if fault == 'other_required_key':
            prior.call(c, run['run_id'], 'second-key', outcome='credit', fingerprint=fp, at=denial)
            prior.call(c, funded, 'second-fresh', fingerprint=fp, native_id='second-unique')
        cid = prior.call(c, other, 'old-duplicate', fingerprint=fp, native_id='duplicate-native', at=denial-timedelta(minutes=5))
        if fault == 'superseded':
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,created_at) SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,0,'{\"call\":{\"call_succeeded\":false}}',clock_timestamp() FROM public.lab_arena_ledger WHERE run_id=%s AND call_identity=%s AND entry_kind='settlement'", (other, cid))
            q.execute('SET LOCAL session_replication_role=origin')
        assert prior.proof(q, run['run_id']) is (fault != 'superseded')


def _insert_copy(q, table, original, patch):
    q.execute("SELECT string_agg(quote_ident(attname),',' ORDER BY attnum) FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''", (table,))
    columns = q.fetchone()[0]
    q.execute('INSERT INTO '+table+'('+columns+') SELECT '+columns+' FROM jsonb_populate_record(NULL::'+table+',%s::jsonb)', (json.dumps({**original, **patch}),))


def _queue70(c, run, denial):
    with c.cursor() as q:
        q.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (run['scored_run_id'],))
        source = q.fetchone()[0]
        q.execute('SELECT to_jsonb(s) FROM public.lab_arena_submissions s WHERE submission_id=%s', (run['submission_id'],))
        submission = q.fetchone()[0]
        q.execute('SELECT participants FROM public.lab_arena_rounds WHERE round_id=%s', (run['round_id'],))
        participants = q.fetchone()[0]
        q.execute('SET LOCAL session_replication_role=replica')
        owners = [(run['submission_id'], run['miner_hotkey'])]
        for group in (1, 2, 3):
            sub, owner = 'throughput554-sub-'+str(group), hotkey('throughput554-'+str(group))
            _insert_copy(q, 'public.lab_arena_submissions', submission, {'submission_id': sub, 'miner_hotkey': owner})
            participants.append({'submission_id': sub, 'miner_hotkey': owner, 'is_king': False})
            owners.append((sub, owner))
        q.execute('UPDATE public.lab_arena_rounds SET participants=%s::jsonb WHERE round_id=%s', (json.dumps(participants), run['round_id']))
        ids = [run['run_id']]
        plans = {1: [], 2: []}
        for index in range(1, 70):
            group, position = divmod(index, 20)
            sub, owner = owners[group]
            stage = 1 if position < 10 else 2
            assignment = f"{run['round_id']}:{sub}:{stage}:{position}"
            source_id, score_id = assignment+':1', assignment+':score:1'
            output = 'fixture/'+source_id
            common = {'submission_id': sub, 'miner_hotkey': owner, 'stage': stage, 'icp_position': position,
                      'claim_request_id': hashlib.sha256(score_id.encode()).hexdigest()[:32]}
            _insert_copy(q, 'public.lab_arena_runs', source, {**common, 'run_id': source_id, 'assignment_id': assignment, 'attempt': 1, 'output_ref': output, 'claim_request_id': hashlib.sha256(source_id.encode()).hexdigest()[:32]})
            _insert_copy(q, 'public.lab_arena_runs', run, {**common, 'run_id': score_id, 'assignment_id': assignment+':score', 'attempt': 1, 'scored_run_id': source_id})
            plans[stage].append({'scored_run_id': source_id, 'submission_id': sub, 'icp_position': position, 'output_ref': output})
            ids.append(score_id)
        for stage, items in plans.items():
            col = 'stage'+str(stage)+'_scoring_plan_doc'
            q.execute('UPDATE public.lab_arena_rounds SET '+col+"=jsonb_set("+col+",'{work_items}',coalesce("+col+"->'work_items','[]'::jsonb)||%s::jsonb) WHERE round_id=%s", (json.dumps(items), run['round_id']))
        q.execute('SET LOCAL session_replication_role=origin')
    for score_id in ids[1:]:
        prior.call(c, score_id, 'exhausted', outcome='credit', at=denial)
    for group, (sub, owner) in enumerate(owners):
        funded = prior.clone(c, run, 'funded-'+str(group), submission=sub, owner=owner)
        prior.call(c, funded, 'new-funded-'+str(group))
    return ids


@pytest.fixture()
def plan_database():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arena_pg, '_database_with_migrations',
                      partial(pg_harness._database_with_migrations, image=POSTGRES_IMAGE))
        yield from database.__wrapped__()


def test_seventy_failed_assignments_finish_under_eight_seconds_with_wide_history(plan_database):
    database = plan_database
    pg, dsn = database
    run, denial = prior.seed(database)
    with pg.connect(**dsn) as c, c.cursor() as q:
        q.execute("SELECT current_setting('server_version'), current_setting('server_version_num')")
        print('FUNDING554_RUNTIME server_version=%s server_version_num=%s' % q.fetchone(), flush=True)
        ids = _queue70(c, run, denial)
        funded = prior.clone(c, run, 'historical-wide')
        for index in range(2500):
            prior.call(c, funded, 'history-'+str(index), at=denial-timedelta(minutes=5))
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_ledger SET terminal_response=terminal_response||jsonb_build_object('retained_body',repeat('wide-json-body-',100000)) WHERE run_id=%s AND entry_kind='settlement'", (funded,))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ANALYZE public.lab_arena_ledger')
        q.execute('ANALYZE public.lab_arena_runs')
        post = definition(q)
        q.execute(old_source(post))
        c.commit()
        q.execute("SET LOCAL statement_timeout='8s'")
        started = time.monotonic()
        with pytest.raises(pg.Error) as error:
            q.execute('SELECT bool_or(public.lab_arena_miner_credit_failure_v1(run_id)) FROM public.lab_arena_runs WHERE run_id=ANY(%s)', (ids,))
        assert error.value.pgcode == '57014'
        old_seconds = time.monotonic()-started
        c.rollback()
        # Time the ordinary locked build separately on this disposable database.
        q.execute('DROP INDEX '+NAME)
        c.commit()
        started = time.monotonic()
        q.execute(INDEX.read_text().replace('CONCURRENTLY IF NOT EXISTS ', ''))
        c.commit()
        build_seconds = time.monotonic()-started
        q.execute(SQL.read_text())
        # Performance alone follows the verified live558 function (PG17.6,
        # read-only source verification2026-10-29). Historical554 semantic,
        # preimage, security and index controls keep their original fixture.
        q.execute(CURRENT_SQL.read_text())
        assert hashlib.sha256(definition(q).encode()).hexdigest() == CURRENT_POST
        c.commit()
        q.execute("SET LOCAL statement_timeout='8s'")
        started = time.monotonic()
        q.execute('SELECT bool_or(public.lab_arena_miner_credit_failure_v1(run_id)) FROM public.lab_arena_runs WHERE run_id=ANY(%s)', (ids,))
        assert q.fetchone()[0] is False
        proof_seconds = time.monotonic()-started
        c.commit()
        q.execute("SET LOCAL statement_timeout='8s'")
        started = time.monotonic()
        q.execute('SELECT public.lab_arena_expire_leases(%s)', (run['round_id'],))
        result = q.fetchone()[0]
        sweep_seconds = time.monotonic()-started
        assert result == {'status': 'ok', 'expired': 0, 'retried': 70}
        q.execute("SELECT count(*) FROM public.lab_arena_runs WHERE run_id=ANY(%s) AND status='failed' AND terminal_cause='credential_error'", (ids,))
        assert q.fetchone() == (70,)
        print('FUNDING554_QUEUE70 old_seconds=%.3f proof_seconds=%.3f sweep_seconds=%.3f local_regular_index_build_seconds=%.3f' % (old_seconds, proof_seconds, sweep_seconds, build_seconds), flush=True)
        assert proof_seconds < 8 and sweep_seconds < 8
