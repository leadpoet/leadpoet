"""Local PostgreSQL proof for the unrendered Oct10 uniform rejudge template."""
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path
import threading

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena import oct10_baseline_recovery440_postgres_test as old
from tests.lab_arena import settled_score_host_recovery441_postgres_test as current
from tests.lab_arena import oct10_uniform_pe_rejudge445_postgres_test as prior
from tests.lab_arena.oct10_uniform_evidence_prior_archive import PRIOR_ARCHIVE_EXPRESSION
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, claim, complete, hash_lease_token

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / 'scripts/448-arena-2026-10-10-uniform-evidence-rejudge.sql.template'
ROUND, BASE = old.ROUND, old.BASE
ARCHIVE = ROUND + '-r448archive'
DIGEST = 'sha256:' + 'e' * 64  # Test-only candidate; never a production constant.
OLD_DIGEST = 'sha256:227d36302522165b615afa1d51c9bcd18df3ed71a461fd0143157bb9d7b82520'
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
        'ledger': f"SELECT coalesce(string_agg(encode(extensions.digest(to_jsonb(l)::text,'sha256'),'hex'),'' ORDER BY entry_id),'') FROM public.lab_arena_ledger l WHERE round_id='{ROUND}' AND run_id IN (SELECT run_id FROM public.lab_arena_runs WHERE round_id='{ROUND}' AND kind='score')",
        'events': f"SELECT coalesce(string_agg(encode(extensions.digest(to_jsonb(e)::text,'sha256'),'hex'),'' ORDER BY trajectory_id),'') FROM public.lab_arena_trajectory_events e WHERE round_id='{ROUND}' AND run_id IN (SELECT run_id FROM public.lab_arena_runs WHERE round_id='{ROUND}' AND kind='score')",
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
    cur.execute('SELECT '+PRIOR_ARCHIVE_EXPRESSION)
    result['prior445_archive_sha256'] = cur.fetchone()[0]
    for kind in ('execute', 'score'):
        cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind=%s', (ROUND, kind))
        result[('execution' if kind == 'execute' else 'score') + '_attempts'] = cur.fetchone()[0]
    for key, table, column in (('ledger', 'lab_arena_ledger', 'entry_id'), ('events', 'lab_arena_trajectory_events', 'trajectory_id')):
        cur.execute(f"SELECT coalesce(max({column}),0),count(*) FROM public.{table} WHERE round_id=%s AND run_id IN (SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='score')", (ROUND,ROUND))
        result[key + '_max'], result[key + '_rows'] = cur.fetchone()
    cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    result['hold_generation'] = cur.fetchone()[0]
    return result


def _hold_render(cur):
    inventory = {k:v for k,v in _inventory(cur).items() if k not in ('round_sha256','runs_sha256','score_attempts','execution_full_sha256','ledger_max','ledger_rows','ledger_sha256','events_max','events_rows','events_sha256')}
    inventory['round_identity_sha256'] = _hash(cur, "(SELECT to_jsonb(r)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','stage2_scoring_plan_doc','finalists','updated_at'] FROM public.lab_arena_rounds r WHERE round_id='arena-2026-10-10')")
    cur.execute('SELECT status,stage_generation,status_generation FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
    inventory['round_status'], inventory['round_stage_generation'], inventory['round_status_generation'] = cur.fetchone()
    template = (ROOT / 'scripts/447-arena-2026-10-10-uniform-evidence-hold.sql.template').read_text()
    rendered = template.replace('__REVIEWED_HOLD_INVENTORY_JSON__', json.dumps(inventory).replace("'", "''"))
    committed = ROOT / 'scripts/447-arena-2026-10-10-uniform-evidence-hold.sql'
    if committed.exists():
        # Exercise every committed SQL byte, replacing only the live capture
        # with this disposable database's independently collected preimage.
        normalized, count = re.subn(r"v_expected CONSTANT JSONB := '[^\n]*'::JSONB;",
            lambda match: "v_expected CONSTANT JSONB := '"+json.dumps(inventory).replace("'", "''")+"'::JSONB;",
            committed.read_text())
        assert count == 1
        assert normalized[normalized.index('BEGIN;'):] == rendered[rendered.index('BEGIN;'):]
        return normalized
    return rendered


def _render(cur):
    inventory = json.dumps(_inventory(cur)).replace("'", "''")
    parameters = {
        '__NEW_SCORER_IMAGE_DIGEST__': DIGEST,
        '__NEW_SCORER_IMAGE_REFERENCE__': '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'+DIGEST,
        '__REVIEWED_SCORER_SOURCE_COMMIT__': 'c'*40,
        '__REVIEWED_FINAL_ACTOR__': FINAL_ACTOR,
        '__REVIEWED_TERMINAL_INVENTORY_JSON__': inventory,
        '__BASELINE_SCORING_CLOSE__': '2026-10-10T16:00:00Z',
        '__STAGE2_RESUME_START__': '2026-10-10T16:00:01Z',
        '__STAGE2_RESUME_CLOSE__': '2026-10-10T16:00:02Z',
    }
    rendered = TEMPLATE.read_text()
    for token,value in parameters.items():
        rendered=rendered.replace(token,value)
    committed = TEMPLATE.with_suffix('')
    if not committed.exists():
        return rendered
    exact = committed.read_text()
    # Only the reviewed live parameters and captured inventory differ in this
    # disposable database. Every guard, statement, function seam and byte stays.
    replacements = []
    for variable,token in (
        ('v_final_actor','__REVIEWED_FINAL_ACTOR__'),
        ('v_new_reference','__NEW_SCORER_IMAGE_REFERENCE__'),
        ('v_new_digest','__NEW_SCORER_IMAGE_DIGEST__'),
        ('v_source_commit','__REVIEWED_SCORER_SOURCE_COMMIT__'),
    ):
        found=re.findall(variable+r" CONSTANT TEXT := '([^']+)';",exact)
        assert len(found)==1
        replacements.append((found[0],parameters[token]))
    for field,token in (
        ('stage_1_scoring_close','__BASELINE_SCORING_CLOSE__'),
        ('stage_2_start','__STAGE2_RESUME_START__'),
        ('stage_2_close','__STAGE2_RESUME_CLOSE__'),
    ):
        found=re.findall("'"+field+r"','([^']+)'",exact)
        assert len(found)==1
        replacements.append((found[0],parameters[token]))
    for original,replacement in replacements:
        exact=exact.replace(original,replacement)
    exact,count=re.subn(r"v_expected CONSTANT JSONB := '[^\n]*'::JSONB;",
        lambda match: "v_expected CONSTANT JSONB := '"+inventory+"'::JSONB;",exact)
    assert count==1
    assert exact[exact.index('BEGIN;'):]==rendered[rendered.index('BEGIN;'):]
    return exact


def _canonical_restart(cur):
    """Exercise the current guard RPCs after the actual independent 447 hold."""
    from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import GUARD, OWNER
    def rpc(function, *values):
        cur.execute('SELECT public.' + function + '(' + ','.join(['%s'] * len(values)) + ')', values)
        return cur.fetchone()[0]
    cur.execute('SELECT operator_paused,pause_reason,guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    paused, reason, generation = cur.fetchone()
    assert paused and reason == 'oct10_uniform_evidence_boundary_recovery'
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
    # Build the real post445 preimage, including its earlier score archive and
    # installed preserved-execution branch. Exactly 740 miner attempts plus 24 baseline.
    prior._prepare(cur, apply_hold=False)
    cur.execute('SET session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=CASE WHEN submission_id='miner440-6' THEN 'provider_error' ELSE 'model_error' END,output_ref=NULL,qualification_doc='{\"companies\":[]}',per_icp_score=0 WHERE round_id=%s AND kind='execute' AND stage=2 AND ((submission_id IN ('miner440-5','miner440-7') AND icp_position<5) OR (submission_id='miner440-6' AND icp_position=0))", (ROUND,))
    cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND stage=2 AND kind='execute' AND status='accepted' ORDER BY run_id LIMIT 31", (ROUND,))
    for row, in cur.fetchall():
        row.update(run_id=row['run_id']+':earlier', attempt=2, status='failed',terminal_cause='model_error',output_ref=None,qualification_doc=None,per_icp_score=None)
        # Existing accepted attempt becomes latest; preserve a realistic failed retry history.
        cur.execute('UPDATE public.lab_arena_runs SET attempt=3 WHERE assignment_id=%s', (row['assignment_id'],))
        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(row),))
    cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND stage=2 AND kind='execute' AND ((submission_id IN ('miner440-5','miner440-7') AND icp_position<5) OR (submission_id='miner440-6' AND icp_position=0))", (ROUND,))
    for row, in cur.fetchall():
        row.update(run_id=row['run_id']+':retry',attempt=2)
        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(row),))
    cur.execute('SET session_replication_role=origin')
    cur.execute(prior._hold_render(cur))
    prior._canonical_restart(cur)
    cur.execute(prior._render(cur).replace(prior.DIGEST,OLD_DIGEST))
    cur.execute('SET session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=15,stage1_scoring_plan_doc='{}',stage2_scoring_plan_doc='{}' WHERE round_id=%s", (ROUND,))
    for index, status in enumerate(('accepted', 'failed', 'pending')):
        sid = BASE if index == 0 else 'miner440-4'
        stage = 1 if index == 0 else 2
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=%s AND status='accepted'", (ROUND, sid, index))
        execution = cur.fetchone()[0]
        score = f'old448-score-{index}'
        cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,output_ref,stage_generation,scored_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score',%s,%s,%s,15,%s)",
                    (score, f'{ROUND}:{sid}:{stage}:{index}:score:rerun445', ROUND, sid, hotkey(sid), stage, index, status,
                     'accepted' if status == 'accepted' else 'judge_error' if status == 'failed' else None,
                     'arena/test/old448-judge.json' if status == 'accepted' else None, execution))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT CASE WHEN status='pending' THEN 'uncertain' ELSE 'settlement' END,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','judge448','host','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),100 FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    cur.execute("INSERT INTO public.lab_arena_trajectory_events (run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,md5(run_id)::uuid,round_id,submission_id,miner_hotkey,miner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'baseline','runtime.result',now(),'{\"proof\":\"original\"}' FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    # A real delayed-cost protocol receipt on a terminal old judge lets the
    # test use the unchanged reconciliation RPC after archival.
    for entry in ('reservation', 'dispatch', 'uncertain'):
        document = {'reason': 'worker_reported', 'call': {'reason': 'missing_provider_cost',
                    'openrouter_generation_id': 'gen-test448', 'credential_fingerprint': 'sha256:' + 'a' * 64}}
        cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc) SELECT %s,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','openrouter.chat','host',%s,1000,%s FROM public.lab_arena_runs WHERE run_id='old448-score-1'", (entry, 'sha256:' + '6' * 64, json.dumps(document)))
    cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_generation=447 WHERE singleton")
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
            cur.execute(f'DELETE FROM public.{table} WHERE round_id IN (%s,%s,%s,%s)', (ROUND, ARCHIVE, old.ARCHIVE, prior.ARCHIVE))
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
    cur.execute('SELECT '+PRIOR_ARCHIVE_EXPRESSION)
    result['prior445_archive_sha256'] = cur.fetchone()[0]
    cur.execute('SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20261009')
    result['bank'] = cur.fetchone()[0]
    cur.execute("SELECT pg_get_functiondef('public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure)")
    result['stage_function'] = cur.fetchone()[0]
    return result


def test_archive_prefixes_uniform_driver_and_no_execution_generation(seeded, monkeypatch):
    psycopg, dsn = seeded
    cost_store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    prior_cost = cost_store.submission_costs('miner440-4')
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = _snapshot(cur)
            sql = _render(cur)
            cur.execute(sql)
            after = _snapshot(cur)
            assert next(r for r in after['lab_arena_rounds'] if r['round_id'] == ROUND)['stage_generation'] == 16
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
    service.now = lambda: datetime(2026, 10, 10, 16, 0, 1, tzinfo=timezone.utc)
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
            lease, token, _, _ = claim(store, ROUND, hotkey('new-judge448'), parallelism=20, ceiling=20)
            assert lease['status'] == 'leased' and lease['kind'] == 'score'
            assert complete(store, lease['run_id'], hash_lease_token(token), 'accepted', output_ref='arena/test/rpc448-judge.json')['status'] == 'accepted'
        if row['status'].endswith('_scoring'):
            with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT 'settlement',round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','new_judge448','host','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),50 FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=public.lab_arena_runs.run_id AND l.operation_id='new_judge448')", (ROUND,))
                cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/test/new448-judge.json',result_doc='{\"terminal_status\":\"accepted\"}' WHERE round_id=%s AND kind='score' AND status='pending'", (ROUND,))
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
                        "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r445archive' AND kind='score'",
                        "UPDATE public.lab_arena_runs SET output_ref='arena/tampered.json' WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
                        "UPDATE public.lab_arena_runs SET stage_generation=99 WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
                        "UPDATE public.lab_arena_submissions SET source_size_bytes=999 WHERE submission_id='miner440-4'",
                        "UPDATE public.qualification_private_icp_sets SET icps='[]' WHERE set_id=20261009",
                        "UPDATE public.lab_arena_ledger SET amount_microusd=1 WHERE round_id='arena-2026-10-10-r448archive' AND run_id='old448-score-1'",
                        "UPDATE public.lab_arena_trajectory_events SET content='{}' WHERE round_id='arena-2026-10-10-r448archive'",
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
    assert len(row['stage2_scoring_plan_doc']['zero_rows']) == 14
    scores = store.list_runs(ROUND, kind='score')
    assert len(scores) == 10 + 672
    assert all(r['assignment_id'].endswith(':score:rerun448') for r in scores)
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
            assert len(actual_exec) == 764
            assert all(r['stage_generation'] == 9 for r in actual_exec if r['stage'] == 2)
            latest = {}
            for run in actual_exec:
                if run['stage'] == 2 and (run['assignment_id'] not in latest or run['attempt'] > latest[run['assignment_id']]['attempt']):
                    latest[run['assignment_id']] = run
            assert {r['submission_id'] for r in latest.values() if r['per_icp_score'] is None} == {'miner440-1', 'miner440-2'}
            assert all(r['per_icp_score'] == 0 for r in latest.values() if r['submission_id'] == 'miner440-3' and r['icp_position'] < 3)
            aggregates = service._score_entries_from_runs(row, range(10), 'final_score')
            assert {r['submission_id'] for r in aggregates if r['final_score'] is None} == {'miner440-1', 'miner440-2'}
            canonical = lambda r: {**r, 'round_id': ROUND, 'submission_id': BASE if r['submission_id'] == BASE+'-r448archive' else r['submission_id']}
            for table, key in (('lab_arena_ledger', 'entry_id'), ('lab_arena_trajectory_events', 'trajectory_id')):
                old_rows = [r for r in before[table] if r['round_id'] == ROUND]
                final_rows = {r[key]: canonical(r) for r in final[table] if r['round_id'] in (ROUND, ARCHIVE)}
                assert [final_rows[r[key]] for r in old_rows] == old_rows
            archived = [canonical(r) for r in final['lab_arena_runs'] if r['round_id'] == ARCHIVE]
            assert archived == [r for r in before['lab_arena_runs'] if r['round_id'] == ROUND and r['kind'] == 'score']
            for run_id, sid in (('old448-score-0', BASE), ('old448-score-1', 'miner440-4')):
                for provider in ('openrouter', 'deepline', 'scrapingdog'):
                    cur.execute('SELECT public.lab_arena_provider_funding(%s,%s)', (run_id, provider))
                    funding = cur.fetchone()[0]
                    assert funding['funding_source'] == 'miner_key'
                    assert funding['credential_submission_id'] == sid
                    assert funding['credential_miner_hotkey'] == hotkey(sid)
            # The existing public cost projection follows unchanged miner IDs
            # across both archives and includes this additional judge pass.
            current_cost = cost_store.submission_costs('miner440-4')
            prior_judge = ArenaService._cost_kind_summary(prior_cost,'score')
            current_judge = ArenaService._cost_kind_summary(current_cost,'score')
            assert current_judge['settled_microusd'] == prior_judge['settled_microusd'] + 10*50
            assert current_judge['reserved_or_uncertain_microusd'] == prior_judge['reserved_or_uncertain_microusd']
            assert ArenaService._cost_kind_summary(current_cost,'execute') == ArenaService._cost_kind_summary(prior_cost,'execute')
            # The unchanged real RPC appends one settlement; the uncertain
            # receipt and old prefix remain immutable and replay still passes.
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s AND call_identity=%s AND entry_kind='uncertain'", (ARCHIVE, 'sha256:' + '6' * 64))
            uncertain_id = cur.fetchone()[0]
            arguments = (ARCHIVE, 'old448-score-1', 'sha256:' + '6' * 64, uncertain_id, 'gen-test448', 'sha256:' + 'a' * 64, 111, '0.000111')
            for idempotent in (False, True):
                cur.execute('SELECT public.lab_arena_reconcile_openrouter_cost_v1(%s,%s,%s,%s,%s,%s,%s,%s)', arguments)
                settled = cur.fetchone()[0]
                assert settled['status'] == 'settled' and settled['idempotent'] is idempotent
            cur.execute(sql)
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE round_id=%s AND operation_id='new_judge448'", (ROUND,))
            assert cur.fetchone() == (682, 682 * 50)
    # The one-second terminal stage2 window cannot consume the final judging
    # window. The real driver already progressed at 16:00:01 with all execution
    # terminal; it now publishes through the unchanged service and SQL guards.
    assert row['configuration_doc']['schedule']['stage_2_close'] == '2026-10-10T16:00:02Z'
    assert row['configuration_doc']['schedule']['final_scoring_close'] == '2026-10-10T20:30:02Z'
    assert row['configuration_doc']['schedule']['publication_deadline'] == '2026-10-10T20:30:03Z'
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT clock_timestamp()')
        database_now = cur.fetchone()[0]
    if database_now < datetime(2026,10,10,16,0,2,tzinfo=timezone.utc):
        # Publication's real PostgreSQL clock also proves a host-incomplete
        # assignment's execution window is closed. A mocked service clock
        # cannot bypass this existing guard.
        with pytest.raises(ArenaStoreError,match='lab_arena_publication_ranking_invalid'):
            service._advance_round_locked(ROUND)
        assert store.get_round(ROUND)['status'] == 'scored'
        # Only this disposable fixture's three ordered timestamps move into
        # the past to model elapsed time. No function, ACL, row proof, final
        # judging deadline, or publication deadline changes.
        elapsed = database_now - timedelta(minutes=1)
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule}',(configuration_doc->'schedule')||%s::jsonb) WHERE round_id=%s",(json.dumps({
                'stage_1_scoring_close': (elapsed-timedelta(seconds=2)).isoformat(),
                'stage_2_start': (elapsed-timedelta(seconds=1)).isoformat(),
                'stage_2_close': elapsed.isoformat(),
            }),ROUND))
    service.now = lambda: database_now
    assert service._advance_round_locked(ROUND)['status'] == 'ok'
    published = store.get_round(ROUND)
    assert published['status'] == 'published'
    assert len(published['publication_doc']['final_ranking']) == 70
    assert {entry['submission_id'] for entry in published['publication_doc']['final_ranking'] if entry['final_score'] is None} == {'miner440-1','miner440-2'}
    assert len(store.list_runs(ROUND,kind='execute')) == 764
    assert len({run['assignment_id'] for run in store.list_runs(ROUND,stage=2,kind='execute')}) == 690


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND kind='execute')",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='old448-score-2'",
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
            with pytest.raises(psycopg.Error, match='Oct10 uniform rejudge448'):
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
                "UPDATE public.lab_arena_rounds SET champion_submission_id='wrong' WHERE round_id='arena-2026-10-10-r448archive'",
                "UPDATE public.lab_arena_rounds SET rewards_enabled=true WHERE round_id='arena-2026-10-10-r448archive'",
                "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r448archive' AND run_id='old448-score-1'",
                "UPDATE public.lab_arena_ledger SET amount_microusd=1 WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger WHERE round_id='arena-2026-10-10-r448archive')",
                "UPDATE public.lab_arena_trajectory_events SET content='{}' WHERE trajectory_id=(SELECT min(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id='arena-2026-10-10-r448archive')",
            ):
                cur.execute('BEGIN')
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute(mutation)
                assert cur.rowcount > 0
                cur.execute('SET LOCAL session_replication_role=origin')
                with pytest.raises(psycopg.Error, match='Oct10 uniform rejudge448'):
                    cur.execute(sql)
                cur.execute('ROLLBACK')
                assert _snapshot(cur) == after


@pytest.mark.parametrize('mutation', [
    "ALTER FUNCTION public.lab_arena_open_stage(text,smallint,jsonb,integer[]) SET search_path=public,pg_catalog",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND kind='execute')",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='old448-score-2'",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='oct10-uniform-evidence-hold447' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='canonical-active-release:'||repeat('e',40) WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET pause_reason='foreign' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.lab_arena_runs SET stage_generation=10 WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
    "UPDATE public.lab_arena_ledger SET entry_kind='dispatch' WHERE round_id='arena-2026-10-10' AND run_id='old448-score-2'",
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


@pytest.mark.parametrize('actor', ['__REVIEWED_FINAL_ACTOR__', 'oct10-uniform-evidence-hold447', 'canonical-active-release:invalid'])
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


def test_large_unrelated_accounting_history_remains_byte_exact(seeded):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            cur.execute("""INSERT INTO public.lab_arena_ledger
                (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,amount_microusd,entry_doc)
                SELECT 'settlement',r.round_id,r.submission_id,r.miner_hotkey,
                  CASE WHEN mod(n,2)=0 THEN NULL ELSE r.run_id END,r.stage,'openrouter','unrelated448-scale','miner_key',123,
                  jsonb_build_object('payload',repeat('0123456789abcdef',488),'ordinal',n)
                FROM (SELECT * FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' LIMIT 1) r
                CROSS JOIN generate_series(1,5000) n""", (ROUND,))
            cur.execute("""INSERT INTO public.lab_arena_trajectory_events
                (run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content)
                SELECT r.run_id,md5('unrelated448-scale-'||n)::uuid,r.round_id,r.submission_id,r.miner_hotkey,r.miner_hotkey,r.assignment_id,r.icp_position::text,r.stage,r.icp_position,r.attempt,'execute','baseline','runtime.progress',now(),
                  jsonb_build_object('payload',repeat('0123456789abcdef',128),'ordinal',n)
                FROM (SELECT * FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' LIMIT 1) r
                CROSS JOIN generate_series(1,5000) n""", (ROUND,))
            cur.execute('SET session_replication_role=origin')
            def untouched():
                result = []
                for table, alias, column in (('lab_arena_ledger','l','entry_id'),('lab_arena_trajectory_events','e','trajectory_id')):
                    cur.execute(f"""SELECT count(*),encode(extensions.digest(coalesce(string_agg(
                        encode(extensions.digest(to_jsonb({alias})::text,'sha256'),'hex'),'' ORDER BY {column}),''),'sha256'),'hex')
                        FROM public.{table} {alias} WHERE round_id=%s
                        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs s WHERE s.run_id={alias}.run_id AND s.kind='score')""", (ROUND,))
                    result.append(cur.fetchone())
                return result
            before = untouched()
            assert all(count>=5000 for count,_ in before)
            inventory = _inventory(cur)
            assert inventory['ledger_rows']==6 and inventory['events_rows']==3
            sql = _render(cur)
            cur.execute(sql)
            assert untouched()==before
            cur.execute(sql)  # Replay ignores preserved unrelated history.
            assert untouched()==before


@pytest.mark.parametrize('kind', ['reservation','dispatch'])
def test_unbound_active_provider_head_still_blocks_rejudge(seeded, kind):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("""INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd)
                SELECT %s,round_id,submission_id,miner_hotkey,NULL,stage,'openrouter','unbound448-head','miner_key','sha256:'||repeat('7',64),123
                FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' LIMIT 1""", (kind,ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error, match='terminal hold or inventory invalid'):
                cur.execute(_render(cur))
            cur.execute('ROLLBACK')


def test_orphan_or_mislabeled_score_event_is_not_archived(seeded):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_trajectory_events SET run_kind='score' WHERE trajectory_id=(SELECT min(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id=%s AND run_kind='execute')", (ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error, match='terminal hold or inventory invalid'):
                cur.execute(_render(cur))
            cur.execute('ROLLBACK')


def _billing_heads(cur, *, fenced):
    # Read every fixed-round head, so equality proves more than the Boolean gate.
    if fenced:
        source = "(SELECT call_identity,entry_kind,entry_id FROM public.lab_arena_ledger WHERE round_id=%s AND call_identity IS NOT NULL OFFSET 0) exact_round"
    else:
        source = 'public.lab_arena_ledger WHERE round_id=%s AND call_identity IS NOT NULL'
    cur.execute('SELECT DISTINCT ON(call_identity) call_identity,entry_kind,entry_id FROM ' + source + ' ORDER BY call_identity,entry_id DESC', (ROUND,))
    return cur.fetchall()


def _billing_test_entry(cur, kind, *, run_bound=False, round_id=ROUND, call='8'):
    cur.execute("""INSERT INTO public.lab_arena_ledger
        (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd)
        SELECT %s,%s,submission_id,miner_hotkey,CASE WHEN %s THEN run_id ELSE NULL END,stage,
          'openrouter','billing448-fence','miner_key','sha256:'||repeat(%s,64),123
        FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' LIMIT 1""", (kind,round_id,run_bound,call,ROUND))


@pytest.mark.parametrize('head', ['reservation','dispatch','settlement','uncertain','refusal','recovery'])
@pytest.mark.parametrize('run_bound', [False,True])
def test_fenced_billing_guard_preserves_every_head_including_unbound(seeded, head, run_bound):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            _billing_test_entry(cur,'reservation',run_bound=run_bound)
            if head != 'reservation':
                _billing_test_entry(cur,'dispatch',run_bound=run_bound)
            if head not in ('reservation','dispatch'):
                if head == 'settlement':
                    _billing_test_entry(cur,'uncertain',run_bound=run_bound)
                _billing_test_entry(cur,head,run_bound=run_bound)
            cur.execute('SET LOCAL session_replication_role=origin')
            before = _billing_heads(cur,fenced=False)
            assert _billing_heads(cur,fenced=True) == before
            assert next(row[1] for row in before if row[0]=='sha256:'+'8'*64) == head
            # Exercise the actual rendered migration for active-head refusal.
            if head in ('reservation','dispatch'):
                with pytest.raises(psycopg.Error,match='terminal hold or inventory invalid'):
                    cur.execute(_render(cur))
            cur.execute('ROLLBACK')


def test_fenced_billing_guard_excludes_foreign_later_rows_and_keeps_late_settlement(seeded):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            _billing_test_entry(cur,'reservation')
            _billing_test_entry(cur,'settlement',round_id=old.ARCHIVE)
            cur.execute('SET session_replication_role=origin')
            heads = _billing_heads(cur,fenced=False)
            assert _billing_heads(cur,fenced=True)==heads
            assert next(row[1] for row in heads if row[0]=='sha256:'+'8'*64)=='reservation'
            with pytest.raises(psycopg.Error,match='terminal hold or inventory invalid'):
                cur.execute(_render(cur))
            cur.execute('ROLLBACK')
            # A same-round later terminal receipt closes the unbound call.
            cur.execute('SET session_replication_role=replica')
            _billing_test_entry(cur,'uncertain')
            cur.execute('SET session_replication_role=origin')
            heads = _billing_heads(cur,fenced=False)
            assert _billing_heads(cur,fenced=True)==heads
            assert next(row[1] for row in heads if row[0]=='sha256:'+'8'*64)=='uncertain'
            cur.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l WHERE call_identity=%s ORDER BY entry_id",('sha256:'+'8'*64,))
            original=cur.fetchall()
            sql=_render(cur)
            cur.execute(sql)
            cur.execute(sql)
            cur.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l WHERE call_identity=%s ORDER BY entry_id",('sha256:'+'8'*64,))
            assert cur.fetchall()==original


def test_fenced_billing_guard_keeps_mutation_and_null_identity_semantics(seeded):
    psycopg, dsn = seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            _billing_test_entry(cur,'reservation')
            _billing_test_entry(cur,'dispatch',call='9')
            cur.execute("UPDATE public.lab_arena_ledger SET call_identity=NULL WHERE operation_id='billing448-fence' AND entry_kind='dispatch'")
            cur.execute('SET LOCAL session_replication_role=origin')
            assert _billing_heads(cur,fenced=True)==_billing_heads(cur,fenced=False)
            assert all(row[0]!='sha256:'+'9'*64 for row in _billing_heads(cur,fenced=True))
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_ledger SET entry_kind='refusal' WHERE operation_id='billing448-fence' AND entry_kind='reservation'")
            cur.execute('SET LOCAL session_replication_role=origin')
            heads=_billing_heads(cur,fenced=True)
            assert heads==_billing_heads(cur,fenced=False)
            assert next(row[1] for row in heads if row[0]=='sha256:'+'8'*64)=='refusal'
            cur.execute('ROLLBACK')


def test_new_image_isolates_saved_output_and_company_cache_identities():
    from lab_arena import company_judgments, judgment_cache
    from tests.lab_arena.company_judgments_test import _input
    document = _input()
    document['evaluation_date'] = '2026-10-10'
    def identity(digest):
        return dict(scoring_input=document,round_id=ROUND,network_name='finney',netuid=71,
            scorer_image_digest=digest,
            scorer_image_reference='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'+digest,
            integrity_policy='arena_integrity_v1')
    old_scope=judgment_cache.build_cache_scope(**identity(OLD_DIGEST))
    new_scope=judgment_cache.build_cache_scope(**identity(DIGEST))
    assert old_scope['scoring_input_hash']==new_scope['scoring_input_hash']
    assert old_scope['cache_key']!=new_scope['cache_key']
    old_companies=company_judgments.build_company_scopes(**identity(OLD_DIGEST),company_quality_policy='company_quality_v1')
    new_companies=company_judgments.build_company_scopes(**identity(DIGEST),company_quality_policy='company_quality_v1')
    assert old_companies and len(old_companies)==len(new_companies)
    assert {row['cache_key'] for row in old_companies}.isdisjoint(row['cache_key'] for row in new_companies)


def test_late_terminal_bill_requires_fresh_snapshot_and_preserves_prefix(seeded):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            stale_sql=_render(cur)
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s AND call_identity=%s AND entry_kind='uncertain'",(ROUND,'sha256:'+'6'*64))
            uncertain=cur.fetchone()[0]
            cur.execute('SELECT public.lab_arena_reconcile_openrouter_cost_v1(%s,%s,%s,%s,%s,%s,%s,%s)',
                (ROUND,'old448-score-1','sha256:'+'6'*64,uncertain,'gen-test448','sha256:'+'a'*64,111,'0.000111'))
            assert cur.fetchone()[0]['status']=='settled'
            cur.execute('COMMIT')  # Provider receipt is committed before the independent migration.
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='terminal preimage differs'):
                cur.execute(stale_sql)
            cur.execute('ROLLBACK')
            after=_snapshot(cur)
            assert after==before
            sql=_render(cur)
            cur.execute(sql)
            cur.execute(sql)
            cur.execute("SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id",('sha256:'+'6'*64,))
            assert cur.fetchall()==[('reservation',1000),('dispatch',1000),('uncertain',1000),('settlement',111)]


@pytest.mark.parametrize('original,replacement',[
    ('c'*40,'invalid-source'),
    ('leadpoet/sourcing-model@'+DIGEST,'leadpoet/sourcing-model@'+OLD_DIGEST),
    ('2026-10-10T16:00:02Z','2026-10-10T20:30:02Z'),
])
def test_reviewed_release_parameters_and_schedule_fail_closed(seeded,original,replacement):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            before=_snapshot(cur)
            sql=_render(cur)
            assert original in sql
            with pytest.raises(psycopg.Error,match='reviewed parameters required|terminal hold or inventory invalid'):
                cur.execute(sql.replace(original,replacement))
            cur.execute('ROLLBACK')
            assert _snapshot(cur)==before


def test_exact_release_sql_normalization_changes_only_reviewed_inputs(seeded,tmp_path,monkeypatch):
    import sys
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            expected=_render(cur)
            # These synthetic release values only exist under pytest's temp path.
            exact=(expected.replace(DIGEST,'sha256:'+'b'*64)
                .replace(FINAL_ACTOR,'canonical-active-release:'+'a'*40)
                .replace('c'*40,'a'*40)
                .replace('2026-10-10T16:00:00Z','2026-10-10T17:00:00Z')
                .replace('2026-10-10T16:00:01Z','2026-10-10T17:00:01Z')
                .replace('2026-10-10T16:00:02Z','2026-10-10T19:00:02Z'))
            template=tmp_path/TEMPLATE.name
            template.write_text(TEMPLATE.read_text())
            template.with_suffix('').write_text(exact)
            monkeypatch.setattr(sys.modules[__name__],'TEMPLATE',template)
            sql=_render(cur)
            assert sql==expected
            cur.execute(sql)
            cur.execute(sql)
