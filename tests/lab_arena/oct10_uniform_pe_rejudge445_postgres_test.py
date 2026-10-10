"""Local PostgreSQL proof for the unrendered Oct10 uniform rejudge template."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import oct10_baseline_recovery440_postgres_test as old
from tests.lab_arena import settled_score_host_recovery441_postgres_test as current
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, claim, complete, hash_lease_token

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / 'scripts/445-arena-2026-10-10-uniform-pe-rejudge.sql.template'
ROUND, BASE = old.ROUND, old.BASE
ARCHIVE = ROUND + '-r445archive'
DIGEST = 'sha256:' + 'b' * 64  # Test-only candidate; never a production constant.
DEPLOYMENT = 'd' * 40  # Separate from the sourcing-model source commit.
FINAL_ACTOR = 'canonical-active-release:' + DEPLOYMENT
database = current.database
@pytest.fixture(scope='module')
def migrated(database):
    current.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        for number in (374, 442):
            paths = list((ROOT / 'scripts').glob(f'{number}-*.sql'))
            assert len(paths) == 1
            cur.execute(paths[0].read_text())
            cur.execute(paths[0].read_text())
    return True


def _hash(cur, expression):
    cur.execute("SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')")
    return cur.fetchone()[0]


def _inventory(cur):
    """Capture independent test-only preimages; never read production state."""
    result = {}
    for key, expression in {
        'round': f"SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id='{ROUND}'",
        'participants': f"SELECT participants FROM public.lab_arena_rounds WHERE round_id='{ROUND}'",
        'submissions': f"SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='{ROUND}'",
        'bank': 'SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20261009',
        'runs': f"SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id='{ROUND}'",
        'execution_full': f"SELECT jsonb_agg(to_jsonb(r)-'qualification_doc'-'per_icp_score' ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id='{ROUND}' AND kind='execute'",
        'execution': f"SELECT jsonb_agg(to_jsonb(r)-'qualification_doc'-'per_icp_score'-'updated_at' ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id='{ROUND}' AND kind='execute'",
        'ledger': f"SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE round_id='{ROUND}'",
        'events': f"SELECT jsonb_agg(to_jsonb(e) ORDER BY trajectory_id) FROM public.lab_arena_trajectory_events e WHERE round_id='{ROUND}'",
        'hold': 'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton',
    }.items():
        result[key + '_sha256'] = _hash(cur, '(' + expression + ')')
    for key, signature in (
        ('scoring', 'lab_arena_open_scoring_v2(text,smallint,jsonb)'),
        ('stage', 'lab_arena_open_stage(text,smallint,jsonb,integer[])'),
    ):
        result[key + '_function_sha256'] = _hash(cur, f"pg_get_functiondef('public.{signature}'::regprocedure)")
    result['triggers_sha256'] = _hash(cur, """(SELECT jsonb_agg(jsonb_build_array(
        c.relname,t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid),
        owner.rolname,p.proacl::text,p.prosecdef,p.proconfig) ORDER BY c.relname,t.tgname)
        FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_proc p ON p.oid=t.tgfoid
        JOIN pg_roles owner ON owner.oid=p.proowner WHERE NOT t.tgisinternal AND t.tgrelid IN
        ('public.lab_arena_rounds'::regclass,'public.lab_arena_submissions'::regclass,
        'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass,
        'public.lab_arena_trajectory_events'::regclass))""")
    result['functions_security_sha256'] = _hash(cur, """(SELECT jsonb_agg(jsonb_build_array(p.oid::regprocedure::text,owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig) ORDER BY p.oid::regprocedure::text) FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid IN('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure,'public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure))""")
    for kind in ('execute', 'score'):
        cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind=%s', (ROUND, kind))
        result[('execution' if kind == 'execute' else 'score') + '_attempts'] = cur.fetchone()[0]
    for key, table, column in (('ledger', 'lab_arena_ledger', 'entry_id'), ('events', 'lab_arena_trajectory_events', 'trajectory_id')):
        cur.execute(f'SELECT coalesce(max({column}),0) FROM public.{table} WHERE round_id=%s', (ROUND,))
        result[key + '_max'] = cur.fetchone()[0]
    cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    result['hold_generation'] = cur.fetchone()[0]
    return result


def _hold_render(cur):
    inventory = {k:v for k,v in _inventory(cur).items() if k not in ('round_sha256','runs_sha256','score_attempts','execution_full_sha256')}
    inventory['round_identity_sha256'] = _hash(cur, "(SELECT to_jsonb(r)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','stage2_scoring_plan_doc','finalists','updated_at'] FROM public.lab_arena_rounds r WHERE round_id='arena-2026-10-10')")
    cur.execute('SELECT status,stage_generation,status_generation FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
    inventory['round_status'], inventory['round_stage_generation'], inventory['round_status_generation'] = cur.fetchone()
    return (ROOT / 'scripts/444-arena-2026-10-10-uniform-pe-hold.sql.template').read_text().replace('__REVIEWED_HOLD_INVENTORY_JSON__', json.dumps(inventory).replace("'", "''"))


def _render(cur):
    return (TEMPLATE.read_text()
            .replace('__NEW_SCORER_IMAGE_DIGEST__', DIGEST)
            .replace('__NEW_SCORER_IMAGE_REFERENCE__', '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@' + DIGEST)
            .replace('__REVIEWED_SCORER_SOURCE_COMMIT__', 'c' * 40)
            .replace('__REVIEWED_FINAL_ACTOR__', FINAL_ACTOR)
            .replace('__REVIEWED_TERMINAL_INVENTORY_JSON__', json.dumps(_inventory(cur)).replace("'", "''"))
            .replace('__BASELINE_SCORING_CLOSE__', '2026-10-10T13:59:58Z')
            .replace('__STAGE2_RESUME_START__', '2026-10-10T13:59:59Z'))


def _canonical_restart(cur):
    """Exercise the current guard RPCs after the actual independent 444 hold."""
    from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import GUARD, OWNER
    def rpc(function, *values):
        cur.execute('SELECT public.' + function + '(' + ','.join(['%s'] * len(values)) + ')', values)
        return cur.fetchone()[0]
    cur.execute('SELECT operator_paused,pause_reason,guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    paused, reason, generation = cur.fetchone()
    assert paused and reason == 'oct10_uniform_pe_boundary_recovery'
    acquired = rpc('lab_arena_acquire_restart_guard_v1', GUARD, OWNER, generation, 600, DEPLOYMENT, 'gateway', FINAL_ACTOR)
    assert acquired['guard_present'] and acquired['guard_generation'] == generation + 1
    renewed = rpc('lab_arena_acquire_restart_guard_v1', GUARD, OWNER, generation + 1, 600, DEPLOYMENT, 'gateway', FINAL_ACTOR)
    assert renewed['guard_present'] and renewed['guard_generation'] == generation + 1
    rpc('lab_arena_authorize_restart_phase_v1', GUARD, OWNER, generation + 1, 'gateway_destructive')
    rpc('lab_arena_mark_restart_ready_v1', GUARD, OWNER, generation + 1, 'gateway_ready')
    released = rpc('lab_arena_release_restart_guard_v1', GUARD, OWNER, generation + 1, FINAL_ACTOR)
    assert not released['guard_present'] and released['operator_paused']
    cur.execute('SELECT actor_ref,pause_reason,guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    assert cur.fetchone() == (FINAL_ACTOR, reason, generation + 1)
    return released


def _prepare(cur, *, apply_hold=True):
    old._seed(cur)
    cur.execute(old._sql(cur))  # Install the actual existing recovery440 namespace.
    cur.execute('SET session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/test/recovered440.json' WHERE round_id=%s AND kind='execute' AND attempt=3", (ROUND,))
    cur.execute("UPDATE public.lab_arena_rounds SET status='stage2',stage_generation=9,configuration_doc=jsonb_set(configuration_doc,'{max_challengers}','69'),champion_submission_id=%s,champion_hotkey=%s,stage1_scoring_plan_doc='{}',stage2_scoring_plan_doc='{}',finalists='[]' WHERE round_id=%s", (BASE, hotkey(BASE), ROUND))
    cur.execute("UPDATE public.lab_arena_submissions SET code_review_status='passed',code_review_doc='{\"decision\":\"pass\"}',code_review_claim='sha256:'||repeat('1',64),code_review_started_at=now(),code_review_attempts=1 WHERE round_id=%s", (ROUND,))
    # Four exhausted host assignment pairs across two models, three provider
    # zeros, plus an earlier ordinary failed->accepted retry. These are distinct
    # assignments; all earlier attempts remain in the captured inventory.
    for miner in range(1, 70):
        sid = f'miner440-{miner}'
        for pos in range(10):
            host = (miner, pos) in {(1, 0), (1, 1), (2, 2), (2, 3)}
            provider = miner == 3 and pos < 3
            retry = (miner, pos) == (4, 0)
            assignment = f'{ROUND}:{sid}:2:{pos}'
            for attempt in range(1, 3 if host or provider or retry else 2):
                accepted = not host and not provider and (not retry or attempt == 2)
                cause = 'accepted' if accepted else 'lease_expired' if host else 'provider_error' if provider else 'model_error'
                cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,terminal_doc,output_ref,stage_generation,qualification_doc,per_icp_score) VALUES (%s,%s,%s,%s,%s,2,%s,%s,'execute',%s,%s,%s,%s,9,%s,%s)",
                            (assignment+f':{attempt}', assignment, ROUND, sid, hotkey(sid), pos, attempt,
                             'accepted' if accepted else 'failed', cause,
                             None,  # The normal close RPC adds the latest host marker.
                             f'arena/test/{sid}-{pos}.json' if accepted else None,
                             json.dumps({'companies': []}) if accepted else None, 42 if accepted else None))
    cur.execute('SET session_replication_role=origin')
    cur.execute('SELECT public.lab_arena_close_stage(%s,2::smallint)', (ROUND,))
    closed = cur.fetchone()[0]
    assert closed['status'] == 'closed' and closed['incomplete_assignments'] == 4
    cur.execute('SET session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=11 WHERE round_id=%s", (ROUND,))
    for index, status in enumerate(('accepted', 'failed', 'pending')):
        sid = BASE if index == 0 else 'miner440-4'
        stage = 1 if index == 0 else 2
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=%s AND status='accepted'", (ROUND, sid, index))
        execution = cur.fetchone()[0]
        score = f'old445-score-{index}'
        cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,output_ref,stage_generation,scored_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score',%s,%s,%s,11,%s)",
                    (score, f'{ROUND}:{sid}:{stage}:{index}:score:recovery440', ROUND, sid, hotkey(sid), stage, index, status,
                     'accepted' if status == 'accepted' else 'judge_error' if status == 'failed' else None,
                     'arena/test/old445-judge.json' if status == 'accepted' else None, execution))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT CASE WHEN status='pending' THEN 'uncertain' ELSE 'settlement' END,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','judge445','host','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),100 FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT 'uncertain',round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','execute445','miner_key','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),123 FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND stage=2", (ROUND,))
    cur.execute("INSERT INTO public.lab_arena_trajectory_events (run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,md5(run_id)::uuid,round_id,submission_id,miner_hotkey,miner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'baseline','runtime.result',now(),'{\"proof\":\"original\"}' FROM public.lab_arena_runs WHERE round_id=%s", (ROUND,))
    # A real delayed-cost protocol receipt on a terminal old judge lets the
    # test use the unchanged reconciliation RPC after archival.
    for entry in ('reservation', 'dispatch', 'uncertain'):
        document = {'reason': 'worker_reported', 'call': {'reason': 'missing_provider_cost',
                    'openrouter_generation_id': 'gen-test445', 'credential_fingerprint': 'sha256:' + 'a' * 64}}
        cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc) SELECT %s,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','openrouter.chat','host',%s,1000,%s FROM public.lab_arena_runs WHERE run_id='old445-score-1'", (entry, 'sha256:' + 'f' * 64, json.dumps(document)))
    cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_generation=444 WHERE singleton")
    cur.execute('SET session_replication_role=origin')
    if apply_hold:
        cur.execute(_hold_render(cur))
        _canonical_restart(cur)


def _seeded(database, *, apply_hold):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            functions = []
            for signature in ('lab_arena_open_scoring_v2(text,smallint,jsonb)', 'lab_arena_open_stage(text,smallint,jsonb,integer[])'):
                cur.execute(f"SELECT pg_get_functiondef('public.{signature}'::regprocedure)")
                functions.append(cur.fetchone()[0])
            _prepare(cur, apply_hold=apply_hold)
    yield database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        for table in ('lab_arena_trajectory_events', 'lab_arena_ledger', 'lab_arena_runs', 'lab_arena_submissions', 'lab_arena_rounds'):
            cur.execute(f'DELETE FROM public.{table} WHERE round_id IN (%s,%s,%s)', (ROUND, ARCHIVE, old.ARCHIVE))
        cur.execute('DELETE FROM public.qualification_private_icp_sets WHERE set_id=20261009')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
        cur.execute('SET session_replication_role=origin')
        for definition in functions:
            cur.execute(definition)


@pytest.fixture
def seeded(database, migrated):
    yield from _seeded(database, apply_hold=True)


def _snapshot(cur):
    result = old._snapshot(cur)
    cur.execute("SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton")
    result['hold'] = cur.fetchone()[0]
    cur.execute('SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20261009')
    result['bank'] = cur.fetchone()[0]
    cur.execute("SELECT pg_get_functiondef('public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure)")
    result['stage_function'] = cur.fetchone()[0]
    return result


def test_archive_prefixes_uniform_driver_and_no_execution_generation(seeded, monkeypatch):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = _snapshot(cur)
            sql = _render(cur)
            cur.execute(sql)
            after = _snapshot(cur)
            assert next(r for r in after['lab_arena_rounds'] if r['round_id'] == ROUND)['stage_generation'] == 12
            original_exec = [r for r in before['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'execute']
            immediate_exec = [r for r in after['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'execute']
            assert [{k:v for k,v in r.items() if k not in ('qualification_doc','per_icp_score')} for r in immediate_exec] == [{k:v for k,v in r.items() if k not in ('qualification_doc','per_icp_score')} for r in original_exec]
            cur.execute(sql)
            assert _snapshot(cur) == after
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    service = object.__new__(ArenaService)
    service._store = store
    service._lock = threading.RLock()
    service._round = lambda rid: store.get_round(rid)
    service.now = lambda: datetime(2026, 10, 10, 13, 59, 59, tzinfo=timezone.utc)
    service._require_code_review = lambda *args: None
    service.benchmark_icps = service.evaluation_icps = lambda rid: daily_icps()[:10]
    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            return json.dumps({'schema_version': contact_policy.output_schema(store.get_round(ROUND)['configuration_doc']), 'companies': []}).encode()
    service._objects = Objects()
    service._outputs_by_run = lambda rid, stage: {r['run_id']: [] for r in store.list_runs(rid, stage=stage, kind='execute') if r['status'] == 'accepted'}
    service._verified_breakdowns = lambda *args, **kwargs: []
    monkeypatch.setattr(provider_observations, 'resolve_observations', lambda *args, **kwargs: [])
    statuses = []
    for _ in range(15):
        row = store.get_round(ROUND)
        statuses.append(row['status'])
        if row['status'] == 'scored':
            break
        if row['status'] == 'stage1_scoring':
            lease, token, _, _ = claim(store, ROUND, hotkey('new-judge445'), parallelism=20, ceiling=20)
            assert lease['status'] == 'leased' and lease['kind'] == 'score'
            assert complete(store, lease['run_id'], hash_lease_token(token), 'accepted', output_ref='arena/test/rpc445-judge.json')['status'] == 'accepted'
        if row['status'].endswith('_scoring'):
            with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT 'settlement',round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','new_judge445','host','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),50 FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=public.lab_arena_runs.run_id AND l.operation_id='new_judge445')", (ROUND,))
                cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/test/new445-judge.json',result_doc='{\"terminal_status\":\"accepted\"}' WHERE round_id=%s AND kind='score' AND status='pending'", (ROUND,))
                cur.execute('SET LOCAL session_replication_role=origin')
        if row['status'] == 'stage1_scored':
            original = {r['run_id']: r for r in before['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'execute'}
            changed = {r['run_id']: {k: (original[r['run_id']].get(k), v) for k, v in r.items() if k not in ('qualification_doc','per_icp_score','updated_at') and original[r['run_id']].get(k) != v} for r in store.list_runs(ROUND,kind='execute')}
            assert not {k:v for k,v in changed.items() if v}
            with psycopg.connect(**dsn) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    original_state = _snapshot(cur)
                    participants = [{'submission_id': p['submission_id'], 'miner_hotkey': p['miner_hotkey']} for p in row['participants'] if not p['is_king']]
                    for mutation in (
                        "UPDATE public.lab_arena_runs SET output_ref='arena/tampered.json' WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
                        "UPDATE public.lab_arena_runs SET stage_generation=99 WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
                        "UPDATE public.lab_arena_submissions SET source_size_bytes=999 WHERE submission_id='miner440-4'",
                        "UPDATE public.qualification_private_icp_sets SET icps='[]' WHERE set_id=20261009",
                        "UPDATE public.lab_arena_ledger SET amount_microusd=1 WHERE round_id='arena-2026-10-10-r445archive' AND run_id='old445-score-1'",
                        "UPDATE public.lab_arena_trajectory_events SET content='{}' WHERE round_id='arena-2026-10-10-r445archive'",
                    ):
                        cur.execute('BEGIN')
                        cur.execute('SET LOCAL session_replication_role=replica')
                        cur.execute(mutation)
                        assert cur.rowcount > 0
                        cur.execute('SET LOCAL session_replication_role=origin')
                        with pytest.raises(psycopg.Error, match='preserved execution differs'):
                            cur.execute('SELECT public.lab_arena_open_stage(%s,2::smallint,%s::jsonb,%s::integer[])', (ROUND, json.dumps(participants), list(range(10))))
                        cur.execute('ROLLBACK')
                        assert _snapshot(cur) == original_state
        result = service._advance_round_locked(ROUND)
        if row['status'] == 'stage1_scored':
            assert result['resumed'] and result['stage_generation'] == row['stage_generation'] + 1
        assert result['status'] not in ('cancelled', 'retry', 'stale'), (statuses, result)
    assert statuses[-1] == 'scored', statuses
    assert statuses == ['stage1_closed', 'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored']
    row = store.get_round(ROUND)
    assert len(row['stage2_scoring_plan_doc']['incomplete_rows']) == 4
    assert len(row['stage2_scoring_plan_doc']['zero_rows']) == 3
    scores = store.list_runs(ROUND, kind='score')
    assert len(scores) == 10 + 683
    assert all(r['assignment_id'].endswith(':score:rerun445') for r in scores)
    assert all(r['judgment_scope_doc']['scorer_image_digest'] == DIGEST for r in scores)
    assert all(r['judgment_scope_doc']['scorer_image_reference'].endswith('@' + DIGEST) for r in scores)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            final = _snapshot(cur)
            original_exec = [r for r in before['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'execute']
            actual_exec = [r for r in final['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'execute']
            strip = lambda rows: [{k: v for k, v in r.items() if k not in ('qualification_doc', 'per_icp_score', 'updated_at')} for r in rows]
            assert strip(actual_exec) == strip(original_exec)
            assert len(actual_exec) == 24 + 698
            assert all(r['stage_generation'] == 9 for r in actual_exec if r['stage'] == 2)
            latest = {}
            for run in actual_exec:
                if run['stage'] == 2 and (run['assignment_id'] not in latest or run['attempt'] > latest[run['assignment_id']]['attempt']):
                    latest[run['assignment_id']] = run
            assert {r['submission_id'] for r in latest.values() if r['per_icp_score'] is None} == {'miner440-1', 'miner440-2'}
            assert all(r['per_icp_score'] == 0 for r in latest.values() if r['submission_id'] == 'miner440-3' and r['icp_position'] < 3)
            aggregates = service._score_entries_from_runs(row, range(10), 'final_score')
            assert {r['submission_id'] for r in aggregates if r['final_score'] is None} == {'miner440-1', 'miner440-2'}
            canonical = lambda r: {**r, 'round_id': ROUND, 'submission_id': BASE if r['submission_id'] == BASE+'-r445archive' else r['submission_id']}
            for table, key in (('lab_arena_ledger', 'entry_id'), ('lab_arena_trajectory_events', 'trajectory_id')):
                old_rows = [r for r in before[table] if r['round_id'] == ROUND]
                final_rows = {r[key]: canonical(r) for r in final[table] if r['round_id'] in (ROUND, ARCHIVE)}
                assert [final_rows[r[key]] for r in old_rows] == old_rows
            archived = [canonical(r) for r in final['lab_arena_runs'] if r['round_id'] == ARCHIVE]
            assert archived == [r for r in before['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'score']
            for run_id, sid in (('old445-score-0', BASE), ('old445-score-1', 'miner440-4')):
                for provider in ('openrouter', 'deepline', 'scrapingdog'):
                    cur.execute('SELECT public.lab_arena_provider_funding(%s,%s)', (run_id, provider))
                    funding = cur.fetchone()[0]
                    assert funding['funding_source'] == 'miner_key'
                    assert funding['credential_submission_id'] == sid
                    assert funding['credential_miner_hotkey'] == hotkey(sid)
            # The unchanged real RPC appends one settlement; the uncertain
            # receipt and old prefix remain immutable and replay still passes.
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s AND call_identity=%s AND entry_kind='uncertain'", (ARCHIVE, 'sha256:' + 'f' * 64))
            uncertain_id = cur.fetchone()[0]
            arguments = (ARCHIVE, 'old445-score-1', 'sha256:' + 'f' * 64, uncertain_id, 'gen-test445', 'sha256:' + 'a' * 64, 111, '0.000111')
            for idempotent in (False, True):
                cur.execute('SELECT public.lab_arena_reconcile_openrouter_cost_v1(%s,%s,%s,%s,%s,%s,%s,%s)', arguments)
                settled = cur.fetchone()[0]
                assert settled['status'] == 'settled' and settled['idempotent'] is idempotent
            cur.execute(sql)
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE round_id=%s AND operation_id='new_judge445'", (ROUND,))
            assert cur.fetchone() == (693, 693 * 50)


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND kind='execute')",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='old445-score-2'",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET updated_at=updated_at+interval '1 second' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_generation=999 WHERE singleton",
    "UPDATE public.lab_arena_submissions SET source_size_bytes=999 WHERE submission_id='miner440-1'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/wrong.json' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND kind='execute')",
    "UPDATE public.lab_arena_rounds SET stage_generation=10 WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.qualification_private_icp_sets SET icps='[]' WHERE set_id=20261009",
])
def test_exact_preimage_drift_fails_without_writes(seeded, mutation):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = _render(cur)
            original = _snapshot(cur)
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error, match='Oct10 uniform rejudge445'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            # The failed transaction restores the original reviewed state.
            assert _snapshot(cur) == original
            assert store_archive_count(cur) == 0


def store_archive_count(cur):
    cur.execute('SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s', (ARCHIVE,))
    return cur.fetchone()[0]


def test_unrendered_and_replay_payload_drift_fail_closed(seeded):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = _snapshot(cur)
            with pytest.raises(psycopg.Error):
                cur.execute(TEMPLATE.read_text())
            cur.execute('ROLLBACK')
            assert _snapshot(cur) == before
            sql = _render(cur)
            cur.execute(sql)
            after = _snapshot(cur)
            for mutation in (
                "UPDATE public.lab_arena_runs SET output_ref='arena/tampered.json' WHERE round_id='arena-2026-10-10' AND kind='execute' AND stage=2 AND icp_position=0",
                "UPDATE public.lab_arena_runs SET stage_generation=99 WHERE round_id='arena-2026-10-10' AND kind='execute' AND stage=2 AND icp_position=0",
                "UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-10-10/sources/tampered.tar.gz' WHERE submission_id='miner440-4'",
                "UPDATE public.lab_arena_rounds SET champion_hotkey=(SELECT miner_hotkey FROM public.lab_arena_submissions WHERE submission_id='miner440-4') WHERE round_id='arena-2026-10-10'",
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_challengers}','68') WHERE round_id='arena-2026-10-10'",
                "UPDATE public.lab_arena_rounds SET champion_submission_id='wrong' WHERE round_id='arena-2026-10-10-r445archive'",
                "UPDATE public.lab_arena_rounds SET rewards_enabled=true WHERE round_id='arena-2026-10-10-r445archive'",
                "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r445archive' AND run_id='old445-score-1'",
                "UPDATE public.lab_arena_ledger SET amount_microusd=1 WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger WHERE round_id='arena-2026-10-10')",
                "UPDATE public.lab_arena_trajectory_events SET content='{}' WHERE trajectory_id=(SELECT min(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id='arena-2026-10-10-r445archive')",
            ):
                cur.execute('BEGIN')
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute(mutation)
                assert cur.rowcount > 0
                cur.execute('SET LOCAL session_replication_role=origin')
                with pytest.raises(psycopg.Error, match='Oct10 uniform rejudge445'):
                    cur.execute(sql)
                cur.execute('ROLLBACK')
                assert _snapshot(cur) == after


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND kind='execute')",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='old445-score-2'",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='oct10-uniform-pe-hold444' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='canonical-active-release:'||repeat('e',40) WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET pause_reason='foreign' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.lab_arena_runs SET stage_generation=10 WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
    "UPDATE public.lab_arena_ledger SET entry_kind='dispatch' WHERE round_id='arena-2026-10-10' AND run_id='old445-score-2'",
])
def test_invalid_terminal_boundary_is_rejected_even_with_matching_snapshot(seeded, mutation):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            original = _snapshot(cur)
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
            sql = _render(cur)
            with pytest.raises(psycopg.Error, match='terminal hold or inventory invalid'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert _snapshot(cur) == original


@pytest.mark.parametrize('actor', ['__REVIEWED_FINAL_ACTOR__', 'oct10-uniform-pe-hold444', 'canonical-active-release:invalid'])
def test_final_actor_parameter_must_name_exact_canonical_deployment(seeded, actor):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            original = _snapshot(cur)
            with pytest.raises(psycopg.Error, match='reviewed parameters required'):
                cur.execute(_render(cur).replace(FINAL_ACTOR, actor))
            cur.execute('ROLLBACK')
            assert _snapshot(cur) == original
