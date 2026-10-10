"""Disposable post448 fixture and independent inventory for the451 hold."""
import json
import re
from pathlib import Path

import pytest

from tests.lab_arena import oct10_baseline_recovery440_postgres_test as old
from tests.lab_arena import settled_score_host_recovery441_postgres_test as current
from tests.lab_arena import oct10_uniform_pe_rejudge445_postgres_test as prior
from tests.lab_arena import oct10_uniform_evidence_rejudge448_postgres_test as previous
from tests.lab_arena.oct10_uniform_evidence_prior_archive import PRIOR_ARCHIVE_EXPRESSION
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
ROUND, BASE = old.ROUND, old.BASE
ARCHIVE = ROUND + '-r452archive'
DIGEST = 'sha256:' + 'e' * 64  # Test-only candidate; never a production constant.
OLD_DIGEST = 'sha256:1b3d6f9005362453431ab7710cb0ba0be1288495334e2167a4425d57583db70a'
DEPLOYMENT = 'd' * 40  # Separate from the sourcing-model source commit.
FINAL_ACTOR = 'canonical-active-release:' + DEPLOYMENT
database = current.database
@pytest.fixture(scope='module')
def migrated(database):
    current.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        for number in (374, 442, 450):
            paths = list((ROOT / 'scripts').glob(f'{number}-*.sql'))
            assert len(paths) == 1
            cur.execute(paths[0].read_text())
            cur.execute(paths[0].read_text())
    return True


_hash = previous._hash


def _inventory(cur):
    result = previous._inventory(cur)
    cur.execute('SELECT '+PRIOR_ARCHIVE_EXPRESSION.replace('r445archive','r448archive'))
    result['prior448_archive_sha256'] = cur.fetchone()[0]
    return result


def _hold_render(cur):
    inventory = {k:v for k,v in _inventory(cur).items() if k not in ('round_sha256','runs_sha256','score_attempts','execution_full_sha256','ledger_max','ledger_rows','ledger_sha256','events_max','events_rows','events_sha256')}
    inventory['round_identity_sha256'] = _hash(cur, "(SELECT to_jsonb(r)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','stage2_scoring_plan_doc','finalists','updated_at'] FROM public.lab_arena_rounds r WHERE round_id='arena-2026-10-10')")
    cur.execute('SELECT status,stage_generation,status_generation FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
    inventory['round_status'], inventory['round_stage_generation'], inventory['round_status_generation'] = cur.fetchone()
    template = (ROOT / 'scripts/451-arena-2026-10-10-uniform-search-date-hold.sql.template').read_text()
    rendered = template.replace('__REVIEWED_HOLD_INVENTORY_JSON__', json.dumps(inventory).replace("'", "''"))
    committed = ROOT / 'scripts/451-arena-2026-10-10-uniform-search-date-hold.sql'
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


def _canonical_restart(cur):
    """Exercise the current guard RPCs after the actual independent 451 hold."""
    from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import GUARD, OWNER
    def rpc(function, *values):
        cur.execute('SELECT public.' + function + '(' + ','.join(['%s'] * len(values)) + ')', values)
        return cur.fetchone()[0]
    cur.execute('SELECT operator_paused,pause_reason,guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
    paused, reason, generation = cur.fetchone()
    assert paused and reason == 'oct10_uniform_search_date_recovery'
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
    # Apply both real earlier recovery migrations, preserving both archives.
    previous._prepare(cur, apply_hold=True)
    cur.execute(previous._render(cur).replace(previous.DIGEST,OLD_DIGEST))
    cur.execute('SET session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=22,stage1_scoring_plan_doc='{}',stage2_scoring_plan_doc='{}' WHERE round_id=%s", (ROUND,))
    for index, status in enumerate(('accepted', 'failed', 'pending')):
        sid = BASE if index == 0 else 'miner440-4'
        stage = 1 if index == 0 else 2
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=%s AND status='accepted'", (ROUND, sid, index))
        execution = cur.fetchone()[0]
        score = f'old452-score-{index}'
        cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,output_ref,stage_generation,scored_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score',%s,%s,%s,22,%s)",
                    (score, f'{ROUND}:{sid}:{stage}:{index}:score:rerun448', ROUND, sid, hotkey(sid), stage, index, status,
                     'accepted' if status == 'accepted' else 'judge_error' if status == 'failed' else None,
                     'arena/test/old452-judge.json' if status == 'accepted' else None, execution))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) SELECT CASE WHEN status='pending' THEN 'uncertain' ELSE 'settlement' END,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','judge452','host','sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),100 FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    cur.execute("INSERT INTO public.lab_arena_trajectory_events (run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,md5(run_id)::uuid,round_id,submission_id,miner_hotkey,miner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'baseline','runtime.result',now(),'{\"proof\":\"original\"}' FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    # A real delayed-cost protocol receipt on a terminal old judge lets the
    # test use the unchanged reconciliation RPC after archival.
    for entry in ('reservation', 'dispatch', 'uncertain'):
        document = {'reason': 'worker_reported', 'call': {'reason': 'missing_provider_cost',
                    'openrouter_generation_id': 'gen-test452', 'credential_fingerprint': 'sha256:' + 'a' * 64}}
        cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc) SELECT %s,round_id,submission_id,miner_hotkey,run_id,stage,'openrouter','openrouter.chat','host',%s,1000,%s FROM public.lab_arena_runs WHERE run_id='old452-score-1'", (entry, 'sha256:' + 'c' * 64, json.dumps(document)))
    cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_generation=451 WHERE singleton")
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
            cur.execute(f'DELETE FROM public.{table} WHERE round_id IN (%s,%s,%s,%s,%s)', (ROUND, ARCHIVE, old.ARCHIVE, prior.ARCHIVE, previous.ARCHIVE))
        cur.execute('DELETE FROM public.qualification_private_icp_sets WHERE set_id=20261009')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
        cur.execute('SET session_replication_role=origin')
        for definition in functions:
            cur.execute(definition)


@pytest.fixture
def seeded(database, migrated):
    yield from _seeded(database, apply_hold=True)


def _snapshot(cur):
    result = previous._snapshot(cur)
    cur.execute('SELECT '+PRIOR_ARCHIVE_EXPRESSION.replace('r445archive','r448archive'))
    result['prior448_archive_sha256'] = cur.fetchone()[0]
    return result
