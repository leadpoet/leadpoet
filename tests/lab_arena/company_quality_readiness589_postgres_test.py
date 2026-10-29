"""Current quality refs must satisfy finality without a full-group scope."""
from dataclasses import replace
from pathlib import Path
import json
from copy import deepcopy
from urllib.parse import urlsplit

from lab_arena import runtime, scoring
from qualification.scoring.competition import apply_company_judgment_context
from tests.test_arena_company_quality_policy import _positive_breakdown

import pytest

from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.test_lab_arena_service_round import _start_round

SQL = Path(__file__).resolve().parents[2] / 'scripts/589-lab-arena-company-quality-judgment-readiness.sql'


@pytest.fixture()
def database():
    # Pin the exact predecessor. Appending589 to current_service_database must
    # never silently upgrade this migration's before fixture.
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database
    generator = full_thinking_database(production_order=True)
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            for migration in (
                '443-lab-arena-competition-publication-projection.sql',
                '575-lab-arena-score-host-continuation.sql',
                '576-lab-arena-champion-score-funding.sql',
                '577-lab-arena-judge-continuation-runner-handoff.sql',
                '578-lab-arena-closed-provider-ordered-candidate.sql',
                '579-lab-arena-close-covered-credit-candidates.sql',
                '581-lab-arena-judge-generation-drain.sql',
                '583-lab-arena-promotion-rpc-privileges.sql',
                '584-lab-arena-stage-quota-run-lookup.sql',
                '585-lab-arena-champion-baseline-reasoning.sql',
                '585-lab-arena-review-deadline-recovery.sql',
                '586-lab-arena-publication-restart-drain.sql',
                '587-lab-arena-atomic-review-restart-guard.sql',
                '588-lab-arena-qualifier-training-publication.sql',
            ):
                cursor.execute((SQL.parent / migration).read_text())
            cursor.execute('TRUNCATE public.lab_arena_rounds CASCADE')
            cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef("
                "'public.lab_arena__all_judgments_ready_v1(text)'::regprocedure),'sha256'),'hex')")
            assert cursor.fetchone()[0] == '39e4ca21bb87af27fd0114b5f2d562a445ca93c2ef6ae3a515e1445dc0d6d1dc'
        yield pg, dsn
    finally:
        generator.close()


def _quality_harness(database, tmp_path, *, contacts=False, partial=False,
                     duplicate=False, empty=False, legacy=False):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness = QualityHarness(connect, tmp_path, challengers=['Clone'] if duplicate else [], runners=['alpha', 'beta'])
    if legacy:
        harness.service.config.defaults = replace(harness.service.config.defaults,
            company_quality_from=None, company_judgment_from='2000-01-01T00:00:00Z')
    if contacts:
        from tests.lab_arena.contact_round_test import _contact_icps
        source = harness.service.config.daily_icp_source
        harness.service.config.defaults = replace(harness.service.config.defaults,
                                                   contacts_from='2000-01-01T00:00:00Z')
        harness.service.config.daily_icp_source = lambda **kwargs: {
            **source(**kwargs), 'icps': _contact_icps(source(**kwargs)['icps'])}
    if partial:
        # Use the reviewed core pump, including the real SQL successor backoff.
        from tests.lab_arena.test_lab_arena_service_round import Harness
        harness.run_stage_with_runners = lambda count=2: Harness.run_stage_with_runners(harness, count)
    partial_sources = []
    original = harness.sandbox.run_icp
    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get('schema_version') != scoring.SCORING_INPUT_SCHEMA_VERSION:
            result = original(spec, **kwargs)
            rows = json.loads(result.output_bytes)['companies']
            if empty and int(document['icp']['icp_id'].rsplit('_', 1)[-1]) == 1:
                rows = []
            for company in rows:
                slug = company['company_name'].lower().replace(' ', '-')
                company.update(company_linkedin='https://linkedin.com/company/' + slug, state='CA')
                if contacts:
                    from tests.lab_arena.contact_round_test import _claim
                    company['contact'] = _claim(company, valid_role=True)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps({'companies': rows}).encode())
        def judge(companies, buyer, _reference):
            company = companies[0]
            name = company['company_name']
            if partial and not partial_sources and name.endswith('Company 1'):
                raise scoring.ScoringError('provider timeout', failure_reason='provider_error')
            raw = _positive_breakdown(name, urlsplit(company['company_website']).hostname,
                                      name.lower().replace(' ', '-'))
            if contacts:
                import asyncio
                from lab_arena.contact_evidence import source_key
                from tests.lab_arena.contact_round_test import _contact_result
                from qualification.scoring.competition import _merge_contact_breakdown
                evidence = document['contact_source_evidence'][source_key(company)]
                _merge_contact_breakdown(raw, asyncio.run(_contact_result(company, buyer, evidence)))
            return apply_company_judgment_context(companies, [raw], contacts_required=contacts)
        judge.integrity_policy = True
        judge.company_quality = not legacy
        judge.contacts_required = contacts
        completed = []
        def retain_completed(rows):
            completed[:] = rows
        try:
            full, new = scoring.score_quality_work_item(
                {'scored_run_id': document['scored_run_id']}, icp=document['icp'],
                companies=document['companies'], scorer=judge,
                cache_context=document['company_judgment_cache'],
                on_company_judgments=retain_completed)
        except scoring.ScoringError:
            assert partial and not partial_sources and completed
            partial_sources.append(document['scored_run_id'])
            return runtime.fake_result(exit_code=0, timed_out=True, output_bytes=json.dumps(
                scoring.build_scoring_failure(document['scored_run_id'], 'judge_timeout',
                                              company_judgments=completed)).encode())
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(
            scoring.build_scoring_output(document['scored_run_id'], full, company_judgments=new)).encode())
    harness.sandbox.run_icp = run_icp
    return harness, partial_sources


@pytest.mark.parametrize('contacts,empty', [(False, False), (True, False), (False, True)])
def test_current_quality_round_publishes_with_company_refs(database, tmp_path, contacts, empty):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness, _ = _quality_harness(database, tmp_path, contacts=contacts, empty=empty)
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('scored', runners=2)
    runs = harness.service.store.list_runs(harness.round_id)
    assert len([row for row in runs if row['kind'] == 'execute']) == 20
    assert len([row for row in runs if row['kind'] == 'score']) == 20
    assert all(row['status'] == 'accepted' for row in runs)
    if empty:
        accepted_empty = [row for row in runs if row['kind'] == 'score' and row['company_judgment_refs'] == []]
        assert len(accepted_empty) == 1
        assert accepted_empty[0]['claim_response']['company_judgment_cache']['hits'] == []
        assert accepted_empty[0]['claim_response']['company_judgment_cache']['misses'] == []
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (harness.round_id,))
        assert cursor.fetchone()[0] is False
        cursor.execute(SQL.read_text())
        cursor.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (harness.round_id,))
        assert cursor.fetchone()[0] is True
    _negative_refs_matrix(connect, harness)
    assert harness.service.publish(harness.round_id)['status'] == 'ok'


@pytest.mark.parametrize('legacy_company', [False, True])
def test_existing_full_group_and_legacy_company_finality_stay_valid(database, tmp_path, legacy_company):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(SQL.read_text())
    if legacy_company:
        harness, _ = _quality_harness(database, tmp_path, legacy=True)
    else:
        from tests.lab_arena.test_lab_arena_service_round import Harness
        harness = Harness(connect, tmp_path, challengers=[], runners=['alpha', 'beta'])
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('scored', runners=2)
    judges = harness.service.store.list_runs(harness.round_id, kind='score')
    assert len(judges) == 20 and all(row['status'] == 'accepted' for row in judges)
    if legacy_company:
        assert all(row['company_judgment_refs'] and row['judgment_scope_doc'] is None for row in judges)
    else:
        assert all(row['company_judgment_refs'] is None and row['judgment_scope_doc'] for row in judges)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (harness.round_id,))
        assert cursor.fetchone()[0] is True
    assert harness.service.publish(harness.round_id)['status'] == 'ok'


@pytest.mark.parametrize('contacts', [False, True])
def test_baseline_first_quality_late_scores_require_quality_authority(database, tmp_path, contacts):
    from lab_arena.store import ArenaStoreError
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness, _ = _quality_harness(database, tmp_path, contacts=contacts)
    harness.service.config.defaults = replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z', per_icp_cost_policy=True)
    from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
    original_runner = harness.runner
    def verified_runner(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner
    harness.runner = verified_runner
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    with pytest.raises(ArenaStoreError, match='lab_arena_late_judge_score_invalid'):
        harness.advance_until('published', runners=2)
    unwritten = [run for run in harness.service.store.list_runs(harness.round_id, kind='execute')
                 if run['per_icp_score'] is None]
    assert unwritten and all(run['status'] == 'accepted' for run in unwritten)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(SQL.read_text())
    harness.advance_until('published', runners=2)
    row = harness.service.store.get_round(harness.round_id)
    assert row['status'] == 'published'
    executions = harness.service.store.list_runs(harness.round_id, kind='execute')
    assert len(executions) == 20
    assert all(run['status'] == 'accepted' and run['per_icp_score'] is not None for run in executions)


def test_real_partial_producer_can_supply_an_accepted_quality_destination(database, tmp_path):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness, partial_sources = _quality_harness(database, tmp_path, partial=True, duplicate=True)
    _start_round(harness, day=29, epoch=42000)
    for participant in harness.service.store.get_round(harness.round_id)['participants']:
        harness.flavors[participant['submission_id']] = 'PublicBaseline'
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    target = None
    for _ in range(20):
        harness.run_stage_with_runners(2)
        runs = harness.service.store.list_runs(harness.round_id, kind='score')
        failed = [row for row in runs if row['status'] == 'failed']
        if failed:
            target = next((row for row in runs if row['status'] == 'accepted'
                and any(hit['evidence_doc']['source_score_run_id'] == failed[0]['run_id']
                    for hit in row['claim_response']['company_judgment_cache']['hits'])), None)
            if target:
                break
        outcome = harness.service.advance_round(harness.round_id)
        assert outcome['status'] not in ('cancelled', 'terminal', 'stale')
    assert partial_sources and target
    assert len(failed) == 1 and failed[0]['output_ref']
    assert failed[0]['result_doc']['company_judgment_completion_request_hash']
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(SQL.read_text())
    _negative_refs_matrix(connect, harness, target=target)
    # This remains a partial-cache proof. No transition was driven after589;
    # the separate continuation regression proves recovery and publication.
    assert not harness.service.store.get_round(harness.round_id).get('publication_doc')


HELPER = 'public.lab_arena__quality_judgment_refs_v1(public.lab_arena_rounds,public.lab_arena_runs,public.lab_arena_runs)'
TARGET = 'public.lab_arena__all_judgments_ready_v1(text)'
RECORD = 'public.lab_arena_record_run_scores(text,smallint,jsonb)'
LEGACY = ('public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)',
          'public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)')
CONTINUATION = 'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'
STRUCTURAL = 'public.lab_arena__quality_continuation_refs_v1(public.lab_arena_rounds,public.lab_arena_runs)'
INCOMPLETE = 'public.lab_arena__participant_incomplete_reason_v1(text,text)'


def _negative_refs_matrix(connect, harness, *, target=None):
    row = harness.service.store.get_round(harness.round_id)
    judge = target or next(run for run in harness.service.store.list_runs(harness.round_id, kind='score')
                 if run['status'] == 'accepted' and run['company_judgment_refs'] and run['claim_response']['company_judgment_cache']['misses'])
    source = harness.service.store.get_run(judge['scored_run_id'])
    def check(round_doc, judge_doc, source_doc):
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena__quality_judgment_refs_v1('
                'jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',
                tuple(json.dumps(doc) for doc in (round_doc, judge_doc, source_doc)))
            return cursor.fetchone()[0]
    assert check(row, judge, source) is True
    # Execution rows retain generation zero even when a current judge belongs
    # to a later round generation. Only destination judge authority advances.
    advanced_round, advanced_judge = deepcopy(row), deepcopy(judge)
    advanced_round['scoring_generation_floor'] += 1
    advanced_judge['scoring_generation_floor'] = advanced_round['scoring_generation_floor']
    assert source['scoring_generation_floor'] == 0
    assert check(advanced_round, advanced_judge, source) is True
    cases = ('quality_marker', 'legacy_marker', 'null_adapter', 'missing_adapter', 'unknown_adapter',
             'scorer_marker', 'contact_marker', 'missing_top_contact', 'nested_contact', 'nested_contact_null', 'round_id', 'network', 'netuid_type',
             'evaluation_date', 'scorer_image', 'missing_cache', 'input_hash',
             'index_type', 'index_order', 'ref_extra', 'scope_extra', 'full_group',
             'generation', 'source_membership', 'destination_slot', 'empty_refs')
    for change in cases:
        r, j, e = deepcopy(row), deepcopy(judge), deepcopy(source)
        ref = j['company_judgment_refs'][0]
        if change == 'quality_marker': r['configuration_doc']['company_quality_policy'] = 'foreign'
        elif change == 'legacy_marker': r['configuration_doc']['company_judgment_policy'] = 'company_judgment_v1'
        elif change == 'missing_adapter': r['configuration_doc']['scorer_policy'].pop('scoring_adapter_version')
        elif change == 'null_adapter': r['configuration_doc']['scorer_policy']['scoring_adapter_version'] = None
        elif change == 'unknown_adapter': r['configuration_doc']['scorer_policy']['scoring_adapter_version'] = 'foreign'
        elif change == 'scorer_marker': r['configuration_doc']['scorer_policy']['company_quality_policy'] = 'foreign'
        elif change == 'contact_marker': r['configuration_doc']['contact_policy'] = 'foreign'
        elif change == 'missing_top_contact':
            r['configuration_doc'].pop('contact_policy', None)
            r['configuration_doc']['scorer_policy']['scoring_adapter_version'] = 'qualification_contacts_v3'
        elif change == 'nested_contact': r['configuration_doc']['scorer_policy']['contact_policy'] = 'contacts_v1'
        elif change == 'nested_contact_null': r['configuration_doc']['scorer_policy']['contact_policy'] = None
        elif change == 'round_id': ref['scope_doc']['round_id'] = 'foreign'
        elif change == 'network': ref['scope_doc']['network_name'] = 'foreign'
        elif change == 'netuid_type': ref['scope_doc']['netuid'] = str(ref['scope_doc']['netuid'])
        elif change == 'evaluation_date': ref['scope_doc']['evaluation_date'] = '2000-01-01'
        elif change == 'scorer_image': ref['scope_doc']['scorer_image_digest'] = 'sha256:' + 'f'*64
        elif change == 'missing_cache':
            ref['cache_key'] = ref['scope_doc']['cache_key'] = 'sha256:' + 'f'*64
        elif change == 'input_hash':
            ref['company_input_hash'] = ref['scope_doc']['company_input_hash'] = 'sha256:' + 'f'*64
        elif change == 'index_type': ref['company_index'] = '0'
        elif change == 'index_order': ref['company_index'] = 1
        elif change == 'ref_extra': ref['foreign'] = True
        elif change == 'scope_extra': ref['scope_doc']['foreign'] = True
        elif change == 'full_group': j['judgment_scope_doc'] = {}
        elif change == 'generation': j['scoring_generation_floor'] += 1
        elif change == 'source_membership': e['miner_hotkey'] = 'foreign'
        elif change == 'empty_refs': j['company_judgment_refs'] = []
        elif change == 'destination_slot':
            context = j['claim_response']['company_judgment_cache']
            for item in context['hits'] + context['misses']:
                if item['company_index'] == 0: item['authority_slot'] += 1
        assert check(r, j, e) is False, change

    # Corrupt one immutable cache producer at a time inside a rolled-back
    # disposable transaction. Normal service roles cannot make these writes.
    key = judge['company_judgment_refs'][0]['cache_key']
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT source_score_run_id FROM public.lab_arena_company_judgments WHERE cache_key=%s LIMIT 1', (key,))
        producer_id = cursor.fetchone()[0]
        cursor.execute('SELECT status,terminal_cause,result_doc FROM public.lab_arena_runs WHERE run_id=%s', (producer_id,))
        producer_status, producer_cause, producer_result = cursor.fetchone()
        mutations = [
            ("UPDATE public.lab_arena_runs SET claim_request_hash=%s WHERE run_id=%s", ('sha256:'+'f'*64, producer_id)),
            ("UPDATE public.lab_arena_company_judgments SET scope_doc=jsonb_set(scope_doc,'{network_name}','\"foreign\"'::jsonb) WHERE cache_key=%s", (key,)),
            ("UPDATE public.lab_arena_company_judgments SET authority_slot=authority_slot+100,evidence_doc=jsonb_set(evidence_doc,'{authority_slot}',to_jsonb(authority_slot+100)) WHERE cache_key=%s", (key,)),
            ("UPDATE public.lab_arena_company_judgments SET evidence_doc=jsonb_set(evidence_doc,'{schema_version}','\"foreign\"'::jsonb) WHERE cache_key=%s", (key,)),
            ("UPDATE public.lab_arena_company_judgments SET evidence_doc=jsonb_set(evidence_doc,'{source_output_ref}','\"foreign\"'::jsonb) WHERE cache_key=%s", (key,)),
        ]
        if producer_status == 'failed':
            assert producer_result['company_judgment_completion_request_hash']
            mutations.extend([
                ("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{company_judgment_completion_request_hash}',%s::jsonb) WHERE run_id=%s", (json.dumps('sha256:'+'f'*64), producer_id)),
                ("UPDATE public.lab_arena_runs SET output_ref=NULL WHERE run_id=%s", (producer_id,)),
            ])
        for statement, arguments in mutations:
            cursor.execute('SAVEPOINT invalid_evidence')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(statement, arguments)
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__quality_judgment_refs_v1('
                'jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',
                tuple(json.dumps(doc) for doc in (row, judge, source)))
            assert cursor.fetchone()[0] is False, statement
            cursor.execute('ROLLBACK TO SAVEPOINT invalid_evidence')
        if producer_status == 'failed' and producer_result['terminal_status'] == 'judge_timeout':
            # Existing Python verification permits this retained timeout alias.
            cursor.execute('SAVEPOINT compatible_partial')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET terminal_cause='judge_error' WHERE run_id=%s", (producer_id,))
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__quality_judgment_refs_v1('
                'jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb),'
                'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',
                tuple(json.dumps(doc) for doc in (row, judge, source)))
            assert cursor.fetchone()[0] is True
            cursor.execute('ROLLBACK TO SAVEPOINT compatible_partial')


def _definitions(cursor, signatures):
    result = {}
    for signature in signatures:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure),proowner,proacl::text,'
                       'prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=%s::regprocedure',
                       (signature, signature))
        result[signature] = cursor.fetchone()
    return result


def _data_snapshot(cursor):
    result = {}
    for table in ('lab_arena_rounds', 'lab_arena_runs', 'lab_arena_ledger',
                  'lab_arena_company_judgments', 'lab_arena_restart_claim_control'):
        cursor.execute('SELECT jsonb_agg(to_jsonb(row)) FROM public.' + table + ' row')
        result[table] = cursor.fetchone()[0]
    return result


def test_migration_replay_security_and_atomic_drift(database):
    pg, dsn = database
    connection = pg.connect(**dsn)
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            before = _definitions(cursor, (TARGET, RECORD, CONTINUATION, INCOMPLETE, *LEGACY))
            data_before = _data_snapshot(cursor)
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            cursor.fetchone()
            cursor.execute('CREATE ROLE quality589_migrator NOLOGIN NOSUPERUSER INHERIT')
            cursor.execute('GRANT lab_arena_owner TO quality589_migrator')
            cursor.execute('GRANT USAGE,CREATE ON SCHEMA public TO quality589_migrator WITH GRANT OPTION')
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            migration_acl = cursor.fetchone()[0]
            cursor.execute('ALTER DEFAULT PRIVILEGES FOR ROLE quality589_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role,lab_arena_service')
            cursor.execute('SET ROLE quality589_migrator')
            cursor.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
            assert cursor.fetchone()[0] is False
            original = before[TARGET][0]
            cursor.execute(original.replace('SELECT NOT EXISTS', 'SELECT /* drift */ NOT EXISTS', 1))
            drift_before_install = _definitions(cursor, (TARGET, RECORD, CONTINUATION, INCOMPLETE, *LEGACY))
            with pytest.raises(pg.Error, match='preimage differs'):
                cursor.execute(SQL.read_text())
            cursor.execute('ROLLBACK')
            assert _definitions(cursor, (TARGET, RECORD, CONTINUATION, INCOMPLETE, *LEGACY)) == drift_before_install
            cursor.execute('SELECT to_regprocedure(%s)', (HELPER,))
            assert cursor.fetchone()[0] is None
            cursor.execute(original)
            incomplete_original = before[INCOMPLETE][0]
            cursor.execute(incomplete_original.replace('DECLARE r ', 'DECLARE /* pre-install drift */ r ', 1))
            incomplete_drift = _definitions(cursor, (TARGET, RECORD, CONTINUATION, INCOMPLETE, *LEGACY))
            with pytest.raises(pg.Error, match='Quality incomplete preimage/security differs'):
                cursor.execute(SQL.read_text())
            cursor.execute('ROLLBACK')
            assert _definitions(cursor, tuple(incomplete_drift)) == incomplete_drift
            assert _data_snapshot(cursor) == data_before
            cursor.execute('SELECT to_regprocedure(%s)', (HELPER,))
            assert cursor.fetchone()[0] is None
            cursor.execute(incomplete_original)
            cursor.execute(SQL.read_text())
            installed = _definitions(cursor, (TARGET, HELPER, RECORD, CONTINUATION, STRUCTURAL, INCOMPLETE, *LEGACY))
            assert installed[TARGET][1:] == before[TARGET][1:]
            assert installed[RECORD][1:] == before[RECORD][1:]
            assert installed[CONTINUATION][1:] == before[CONTINUATION][1:]
            assert installed[INCOMPLETE][1:] == before[INCOMPLETE][1:]
            assert all(installed[sig] == before[sig] for sig in LEGACY)
            cursor.execute(SQL.read_text())
            assert _definitions(cursor, tuple(installed)) == installed
            assert _data_snapshot(cursor) == data_before
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == migration_acl
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                for signature in (TARGET, HELPER, CONTINUATION, STRUCTURAL, INCOMPLETE):
                    cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                    assert cursor.fetchone()[0] is False
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, RECORD))
                assert cursor.fetchone()[0] is (role == 'lab_arena_service')
            cursor.execute('RESET ROLE')
            for signature in (TARGET, HELPER, RECORD, CONTINUATION, STRUCTURAL, INCOMPLETE):
                # A valid unknown body must not be silently replaced on replay.
                body = installed[signature][0]
                if signature in (HELPER, STRUCTURAL):
                    altered = body.replace('RETURN TRUE;', 'RETURN TRUE OR FALSE;', 1)
                elif signature == INCOMPLETE:
                    altered = body.replace('DECLARE r ', 'DECLARE /* drift */ r ', 1)
                elif signature == CONTINUATION:
                    altered = body.replace('DECLARE v_source', 'DECLARE /* drift */ v_source', 1)
                elif signature == RECORD:
                    altered = body.replace('v_late_judge BOOLEAN;', 'v_late_judge BOOLEAN; -- drift', 1)
                else:
                    altered = body.replace('SELECT NOT EXISTS', 'SELECT /* drift */ NOT EXISTS', 1)
                assert altered != body
                cursor.execute(altered)
                drift = _definitions(cursor, tuple(installed))
                with pytest.raises(pg.Error, match='(preimage|helper|preimage/security) differs'):
                    cursor.execute(SQL.read_text())
                cursor.execute('ROLLBACK')
                assert _definitions(cursor, tuple(installed)) == drift
                cursor.execute(body)
            for signature in (CONTINUATION, STRUCTURAL, INCOMPLETE):
                cursor.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO anon')
                drift = _definitions(cursor, tuple(installed))
                with pytest.raises(pg.Error, match='(preimage/security|helper) differs'):
                    cursor.execute(SQL.read_text())
                cursor.execute('ROLLBACK')
                assert _definitions(cursor, tuple(installed)) == drift
                cursor.execute('REVOKE EXECUTE ON FUNCTION ' + signature + ' FROM anon')
            assert _definitions(cursor, tuple(installed)) == installed

    finally:
        connection.close()
