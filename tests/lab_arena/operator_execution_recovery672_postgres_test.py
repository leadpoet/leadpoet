"""Finite counted execute recovery uses ordinary leases, billing and judging."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import re
import uuid

import pytest

from tests.lab_arena import nov06_organizer_openrouter665_postgres_test as base
from tests.lab_arena import nov03_three_executions609_postgres_test as execute
from tests.lab_arena.stage_cutoff_drain420_postgres_test import fixture_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from lab_arena import scoring

SCRIPTS = Path(__file__).parents[2] / 'scripts'
SQL = SCRIPTS / '672-lab-arena-owner-execution-infrastructure-recovery.sql'
ROUND = 'arena-2026-11-08'
TOKEN = execute.TOKEN
# The nine observed exhausted positions, including the zero-call pip ancestor.
POSITIONS = [('2dfb', 0), ('6404', 3), ('8bd0', 3), ('8bd0', 4), ('8bd0', 5),
             ('a195', 5), ('a195', 6), ('a195', 8), ('a195', 9)]


@pytest.fixture(scope='module')
def database():
    gen = base.database.__wrapped__()
    pg, dsn, _ = next(gen)
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            # Historical665's fixture carries627's claim fragments only.
            # Install627's exact committed reserve fragments as well, without
            # its dated production data repair or weakening672's preimage.
            text = (SCRIPTS / '627-lab-arena-champion-baseline-freshness.sql').read_text()
            signature = 'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
            definition = scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            for old, new in re.findall(r"\('public.lab_arena_reserve_call[^']*','[^']*','[^']*',\d+,\$old627\$(.*?)\$old627\$,\$new627\$(.*?)\$new627\$\)", text, re.S):
                assert definition.count(old) == 1
                definition = definition.replace(old, new)
            q.execute(definition)
            text = base.SQL.read_text()
            restore = 'DO $restore665$' + text.split('DO $restore665$')[1].split('NOTIFY')[0]
            q.execute(text.split('-- Exact one-time data recovery.')[0] + restore + 'COMMIT;')
            q.execute((SCRIPTS / '666-lab-arena-operator-scorer-bug-recovery.sql').read_text())
            q.execute(SQL.read_text())
        yield pg, dsn
    finally:
        gen.close()


def scalar(q, sql, args=()):
    q.execute(sql, args)
    return q.fetchone()[0]


def aid(label, pos):
    return f'{ROUND}:sub-{label}:2:{pos}'


@pytest.fixture
def case(database):
    pg, dsn = database
    c = pg.connect(**dsn)
    cfg = fixture_configuration(ROUND)
    cfg.pop('contact_policy', None)
    cfg.update(execution_sequence_policy='baseline_scored_first_v1',
               sourcing_cost_eligibility_policy='successful_calls_flat_per_icp_v1',
               infrastructure_retry_cost_policy='trusted_infrastructure_v1',
               reasoning_capture_policy='full_generated_thinking_v1',
               execution_icp_cap_microusd=6000000, cost_per_company_microusd=6000000,
               max_attempts_per_assignment=2, stage_1_icp_count=5, stage_2_icp_count=5,
               lease_ttl_seconds=4500, runner_slot_ceiling=20,
               checkpoint_deadline_policy='atomic_checkpoint_60m_v1', icp_wall_clock_seconds=3600,
               call_quotas={'openrouter': 2000, 'scrapingdog': 200, 'deepline': 0},
               deepline_catalog={'schema_version': 'leadpoet.lab_arena.deepline_catalog.v1'},
               company_judgment_policy='company_judgment_v1',
               scorer_policy=scoring.build_scorer_policy(company_judgment_cache=True,
                   scoring_adapter_version='qualification_integrity_v2'))
    now = datetime.now(timezone.utc)
    cfg['schedule'].update(stage_2_close=(now + timedelta(hours=2)).isoformat(),
                           final_scoring_close=(now + timedelta(hours=4)).isoformat(),
                           publication_deadline=(now + timedelta(hours=5)).isoformat())
    participants = [dict(submission_id='sub-' + label, miner_hotkey=hotkey('672-' + label), is_king=False)
                    for label in sorted({x[0] for x in POSITIONS})]
    with c.cursor() as q:
        q.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]',candidate_commit='',restart_scope='',restart_phase=''")
        base.insert(q, 'lab_arena_rounds', dict(round_id=ROUND, status='stage2', status_generation=9,
            stage_generation=7, configuration_doc=cfg, participants=participants,
            evaluation_date='2026-11-08', icp_set_date='2026-11-07', benchmark_ref='672-benchmark'))
        for s in participants:
            base.insert(q, 'lab_arena_submissions', dict(**s, round_id=ROUND, status='frozen',
                code_review_status='passed', code_review_doc={'decision': 'pass'},
                source_ref=f'arena/{ROUND}/sources/{s["submission_id"]}.tar.gz', source_size_bytes=123,
                source_archive_sha256='sha256:' + 'a' * 64, source_normalized_sha256='sha256:' + 'b' * 64,
                code_review_claim='sha256:' + 'c' * 64, code_review_started_at=now, code_review_attempts=1,
                consent={'public_rerun': True}))
        parents = []
        historical = []
        for index, (label, pos) in enumerate(POSITIONS):
            owner = next(s for s in participants if s['submission_id'] == 'sub-' + label)
            for n in (1, 2):
                run = dict(run_id=aid(label, pos) + ':' + str(n), assignment_id=aid(label, pos),
                    round_id=ROUND, submission_id=owner['submission_id'], miner_hotkey=owner['miner_hotkey'],
                    runner_hotkey=hotkey('672-old-yuma'), stage=2, icp_position=pos, attempt=n,
                    kind='execute', status='failed', terminal_cause='lease_expired', stage_generation=7,
                    scoring_generation_floor=0, lease_generation=n, lease_token_hash=TOKEN,
                    lease_expires_at=now - timedelta(hours=1), claim_request_id=uuid.uuid4().hex,
                    claim_request_hash='sha256:' + 'd' * 64,
                    claim_response={'status': 'leased', 'lease_generation': n,
                                    'lease_expires_at': (now - timedelta(hours=1)).isoformat()})
                if n == 1 and index < 2:
                    cause = 'model_error' if index == 0 else 'provider_error'
                    run.update(terminal_cause=cause, result_doc={'terminal_status': cause})
                base.insert(q, 'lab_arena_runs', run)
                historical.append(run)
                if n == 2:
                    parents.append(run['run_id'])
        # Preserve a realistically sized accepted cohort. It is never reset.
        for i in range(91):
            owner = participants[i % len(participants)]
            assignment = f'{ROUND}:retained:{i}'
            source = dict(run_id=assignment + ':1', assignment_id=assignment, round_id=ROUND,
                submission_id=owner['submission_id'], miner_hotkey=owner['miner_hotkey'], stage=1,
                icp_position=i, attempt=1, status='accepted', terminal_cause='accepted', kind='execute',
                stage_generation=2, scoring_generation_floor=0,
                output_ref=f'retained/{i}', result_doc={'terminal_status': 'accepted'})
            base.insert(q, 'lab_arena_runs', source)
            if i < 82:
                base.insert(q, 'lab_arena_runs', dict(source, run_id=assignment + ':score:1',
                    assignment_id=assignment + ':score', kind='score', scored_run_id=source['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
    c.commit()
    store = execute.store_for(pg, dsn)
    # These are retained paid histories, not zero-call host-credit fixtures.
    # The pip parent alone has no provider call. Native accounting creates all
    # reservation, dispatch and settlement receipts on disposable fixture rows.
    for run in historical:
        if run['run_id'] == parents[0][:-1] + '1':
            continue
        with c.cursor() as q:
            q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,result_doc=NULL,lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (run['run_id'],))
        c.commit()
        assert execute.payment(store, run['run_id'], 100)['status'] == 'settled'
        with c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,result_doc=%s::jsonb,lease_expires_at=%s WHERE run_id=%s",
                (run['terminal_cause'], json.dumps(run['result_doc']) if run.get('result_doc') is not None else None,
                 run['lease_expires_at'], run['run_id']))
            q.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal')
        c.commit()
    p = SimpleNamespace(pg=pg, dsn=dsn, c=c, cfg=cfg, parents=parents, store=store)
    try:
        yield p
    finally:
        store._transport.close()
        c.close()


def grant(p, parent=None, *, dry=False, request=None, parent_hash=None):
    parent = parent or p.parents[0]
    with p.c.cursor() as q:
        digest = parent_hash or 'sha256:' + scalar(q,
            "SELECT encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex') FROM public.lab_arena_runs a WHERE run_id=%s", (parent,))
        result = scalar(q, 'SELECT public.lab_arena_grant_execution_infrastructure_recovery_v1(%s,%s,%s,%s,%s,%s)',
            (parent, digest, request or 'recover-' + parent, '672-owner', 'verified-external-infrastructure-correction', dry))
    p.c.commit()
    return result


def snapshot(p):
    with p.c.cursor() as q:
        return {t: scalar(q, f'SELECT coalesce(jsonb_agg(to_jsonb(a)ORDER BY {key}),\'[]\') FROM public.{t} a')
                for t, key in [('lab_arena_rounds', 'round_id'), ('lab_arena_submissions', 'submission_id'),
                    ('lab_arena_runs', 'run_id'), ('lab_arena_ledger', 'entry_id'),
                    ('lab_arena_restart_claim_control', 'singleton'),
                    ('lab_arena_scoring_supersessions', 'round_id,generation_floor')]}


def claim(p):
    with p.c.cursor() as q:
        q.execute('SET ROLE lab_arena_service')
        result = scalar(q, 'SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',
            (ROUND, hotkey('672-healthy'), uuid.uuid4().hex, 'sha256:' + uuid.uuid4().hex * 2, TOKEN))
        q.execute('RESET ROLE')
    p.c.commit()
    return result


def test_nine_observed_states_dry_run_idempotence_and_preservation(case):
    p = case
    before = snapshot(p)
    assert claim(p)['status'] == 'no_pending'
    for parent in p.parents:
        assert grant(p, parent, dry=True)['status'] == 'dry_run'
        assert snapshot(p) == before
    for parent in p.parents:
        assert grant(p, parent)['status'] == 'granted'
    after = snapshot(p)
    for table in set(before) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == before[table]
    original = {r['run_id']: r for r in before['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in original} == original
    assert len(after['lab_arena_runs']) == len(original) + 9
    for parent in p.parents:
        assert grant(p, parent)['idempotent'] is True
    assert snapshot(p) == after
    with p.c.cursor() as q:
        assert scalar(q, 'SELECT array_agg(public.lab_arena_execution_attempt_count(a)) FROM public.lab_arena_runs a WHERE attempt=3') == [3] * 9


@pytest.mark.parametrize('fault', ['stale', 'not_expired', 'accepted', 'later', 'closed', 'deadline', 'published', 'source', 'guard', 'unsettled', 'cost', 'quota'])
def test_invalid_grants_change_nothing(case, fault):
    p = case
    kwargs = {}
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'stale':
            kwargs['parent_hash'] = 'sha256:' + '0' * 64
        elif fault == 'not_expired':
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (p.parents[0],))
        elif fault == 'accepted':
            q.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='accepted'WHERE run_id=%s", (p.parents[0][:-1] + '1',))
        elif fault == 'later':
            row = scalar(q, 'SELECT to_jsonb(a)FROM public.lab_arena_runs a WHERE run_id=%s', (p.parents[0],))
            row.update(run_id=row['assignment_id'] + ':3', attempt=3, claim_request_id=uuid.uuid4().hex)
            base.insert(q, 'lab_arena_runs', row)
        elif fault in ('closed', 'published'):
            q.execute('UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s', ('published' if fault == 'published' else 'stage2_judged', ROUND))
        elif fault == 'deadline':
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2000-01-01T00:00:00Z\"'::jsonb)WHERE round_id=%s", (ROUND,))
        elif fault == 'source':
            q.execute("UPDATE public.lab_arena_submissions SET status='rejected'WHERE submission_id='sub-2dfb'")
        elif fault == 'guard':
            q.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('e',64),owner_commitment='sha256:'||repeat('f',64),guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('a',40),restart_scope='gateway',restart_phase='draining'")
        elif fault == 'unsettled':
            base.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation', run_id=p.parents[0], round_id=ROUND,
                submission_id='sub-2dfb', miner_hotkey=hotkey('672-2dfb'), stage=2, provider='openrouter',
                operation_id='openrouter.chat', funding_source='miner_key', amount_microusd=1,
                call_identity='sha256:' + '7' * 64, entry_doc={}))
        elif fault == 'cost':
            # Seed retained bills through ordinary reserve/dispatch/settle.
            q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (p.parents[0],))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    if fault in ('cost', 'quota'):
        if fault == 'cost':
            assert execute.payment(p.store, p.parents[0], 5999900)['status'] == 'settled'
        else:
            with p.c.cursor() as q:
                q.execute('SET LOCAL session_replication_role=replica')
                q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas}','{\"openrouter\":1}'::jsonb)WHERE round_id=%s", (ROUND,))
            p.c.commit()
            # One paid call consumes this attempt; another retained sibling
            # consumes the second original allowance.
            first = p.parents[0][:-1] + '1'
            with p.c.cursor() as q:
                q.execute('SET LOCAL session_replication_role=replica')
                q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,result_doc=NULL,lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (first,))
            p.c.commit()
            assert execute.payment(p.store, first, 1)['status'] == 'settled'
            with p.c.cursor() as q:
                q.execute('SET LOCAL session_replication_role=replica')
                q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',lease_expires_at=now()-interval '1 hour'WHERE run_id=%s", (first,))
            p.c.commit()
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',lease_expires_at=now()-interval '1 hour'WHERE run_id=%s", (p.parents[0],))
        p.c.commit()
    before = snapshot(p)
    with pytest.raises(p.pg.Error, match='execution recovery'):
        grant(p, **kwargs)
    p.c.rollback()
    assert snapshot(p) == before


@pytest.mark.parametrize('outcome', ['accepted', 'provider_error', 'model_error', 'expired', 'host'])
def test_real_child_lifecycle_never_recurses(case, outcome):
    p = case
    child = grant(p)['child_identity']['run_id']
    lease = claim(p)
    assert lease['status'] == 'leased' and lease['run_id'] == child
    assert execute.payment(p.store, child, 100)['status'] == 'settled'
    if outcome in ('expired', 'host'):
        if outcome == 'host':
            execute.event(p.store, child, 'runtime.started', dict(status='starting', lease_generation=lease['lease_generation']))
            execute.event(p.store, child, 'runtime.error', dict(status='abandoned', failure_stage='runtime',
                error_class='RuntimeHostError', runtime_host_reason='sandbox_launcher_signaled', launch_exit_code=-2, launch_timed_out=False))
        with p.c.cursor() as q:
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second'WHERE run_id=%s", (child,))
        p.c.commit()
        p.store.expire_leases(ROUND)
    else:
        result = p.store.complete_attempt(run_id=child, lease_token_hash=TOKEN,
            result={'terminal_status': outcome}, terminal_cause=outcome,
            output_ref='fresh-output.json' if outcome == 'accepted' else None)
        assert result['status'] == ('accepted' if outcome == 'accepted' else 'failed')
    p.store.expire_leases(ROUND)
    assert claim(p)['status'] == 'no_pending'
    with p.c.cursor() as q:
        assert scalar(q, 'SELECT count(*)FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>3', (lease['assignment_id'],)) == 0
    before = snapshot(p)
    with pytest.raises(p.pg.Error, match='execution recovery'):
        grant(p, child, request='recursive-grant')
    p.c.rollback()
    assert snapshot(p) == before


def test_owner_only_and_exact_migration_replay(case):
    p = case
    with p.c.cursor() as q:
        q.execute('SET ROLE lab_arena_service')
        with pytest.raises(p.pg.Error, match='permission denied'):
            q.execute("SELECT public.lab_arena_grant_execution_infrastructure_recovery_v1('x','sha256:'||repeat('a',64),'x','x','x',true)")
    p.c.rollback()
    before = snapshot(p)
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()
    assert snapshot(p) == before


@pytest.mark.parametrize('fault', ['body', 'acl'])
def test_reapplication_refuses_tampered_helper_before_any_change(case, fault):
    p = case
    before = snapshot(p)
    with p.c.cursor() as q:
        if fault == 'acl':
            q.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs)TO lab_arena_service')
        else:
            definition = scalar(q, "SELECT pg_get_functiondef('public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs)'::regprocedure)")
            q.execute(definition.replace("p_run.kind='execute'", "p_run.kind='score'"))
        with pytest.raises(p.pg.Error, match='helper security or source differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    assert snapshot(p) == before


@pytest.mark.parametrize('after_claim', [False, True])
def test_damaged_journal_fails_closed_but_stays_a_finite_child(case, after_claim):
    p = case
    child = grant(p)['child_identity']['run_id']
    if after_claim:
        assert claim(p)['run_id'] == child
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_scoring_supersessions SET recovery_doc=recovery_doc-'original_run_hashes'")
    p.c.commit()
    before = snapshot(p)
    if after_claim:
        assert execute.payment(p.store, child, 1)['status'] == 'stale'
    else:
        assert claim(p)['status'] == 'no_pending'
    assert snapshot(p) == before


def test_real_completion_opens_only_new_judgments_and_preserves_old_bindings(case):
    p = case
    before = snapshot(p)
    for parent in p.parents:
        grant(p, parent)
    items = []
    for _ in p.parents:
        lease = claim(p)
        assert lease['status'] == 'leased'
        assert execute.payment(p.store, lease['run_id'], 100)['status'] == 'settled'
        output = 'fresh/' + lease['run_id']
        assert p.store.complete_attempt(run_id=lease['run_id'], lease_token_hash=TOKEN,
            result={'terminal_status': 'accepted'}, terminal_cause='accepted', output_ref=output)['status'] == 'accepted'
        items.append(dict(scored_run_id=lease['run_id'], submission_id=lease['submission_id'],
            icp_position=lease['icp_position'], output_ref=output, company_judgment_refs=[]))
    assert p.store.close_stage(ROUND, 2)['status'] == 'closed'
    opened = p.store.open_scoring(ROUND, 2, items, company_quality_cache=True)
    assert opened['status'] == 'ok' and opened['assignments'] == 9
    lease = claim(p)
    assert lease['status'] == 'leased' and lease['kind'] == 'score'
    assert lease['scored_run_id'] in {r['scored_run_id'] for r in items}
    with p.c.cursor() as q:
        original = {r['run_id']: r for r in before['lab_arena_runs'] if r['status'] == 'accepted'}
        current = scalar(q, 'SELECT jsonb_object_agg(run_id,to_jsonb(a))FROM public.lab_arena_runs a WHERE run_id=ANY(%s)', (list(original),))
        assert current == original


def test_counted_third_cannot_spend_a_third_provider_quota(case):
    p = case
    parent = p.parents[1]
    first = parent[:-1] + '1'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas}','{\"openrouter\":2}'::jsonb)WHERE round_id=%s", (ROUND,))
        original = scalar(q, 'SELECT to_jsonb(a)FROM public.lab_arena_runs a WHERE run_id=%s', (first,))
        q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,result_doc=NULL,lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (first,))
    p.c.commit()
    assert execute.payment(p.store, first, 1)['status'] == 'settled'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,result_doc=%s::jsonb,lease_expires_at=%s WHERE run_id=%s", (original['terminal_cause'], json.dumps(original['result_doc']), original['lease_expires_at'], first))
    p.c.commit()
    child = grant(p, parent)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    assert execute.payment(p.store, child, 1)['status'] == 'settled'
    # Before672's aggregate fence this second child call was still within its
    # per-run quota, but exceeded the two original attempts' total allowance.
    with p.c.cursor() as q:
        definition = scalar(q, "SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure)")
        q.execute('SAVEPOINT unfenced')
        q.execute(definition.replace('\n      OR public.lab_arena__operator_execute_recovery_child_v1(v_run)', ''))
        q.execute('SET ROLE lab_arena_service')
        assert scalar(q, "SELECT public.lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',1,'{}',4500)", (child, TOKEN, 'sha256:' + uuid.uuid4().hex * 2))['status'] == 'reserved'
        q.execute('RESET ROLE')
        q.execute('ROLLBACK TO SAVEPOINT unfenced')
    p.c.commit()
    denied = execute.payment(p.store, child, 1)
    assert (denied['status'], denied['reason']) == ('refused', 'per_icp_quota')


def test_late_closed_parent_cost_reconciliation_retains_child_authority(case):
    p = case
    parent = p.parents[1]
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour'WHERE run_id=%s", (parent,))
    p.c.commit()
    identity = 'sha256:' + uuid.uuid4().hex * 2
    fingerprint = 'sha256:' + '7' * 64
    args = dict(run_id=parent, lease_token_hash=TOKEN, call_identity=identity)
    assert p.store.reserve_call(**args, operation_id='openrouter.chat', provider='openrouter',
        funding_source='miner_key', amount_microusd=0, call_doc={}, lease_ttl_seconds=4500)['status'] == 'reserved'
    assert p.store.mark_dispatched(**args)['status'] == 'dispatched'
    assert p.store.mark_uncertain(**args, call_doc=dict(reason='missing_provider_cost', provider_status=503,
        openrouter_generation_id='gen-672-closed', credential_fingerprint=fingerprint,
        call_succeeded=False))['status'] == 'uncertain'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',lease_expires_at=now()-interval '1 hour'WHERE run_id=%s", (parent,))
    p.c.commit()
    child = grant(p, parent)['child_identity']['run_id']
    old_parent = p.store.get_run(parent)
    candidates = p.store.list_openrouter_cost_reconciliations(ROUND)
    candidate = next(r for r in candidates if r['call_identity'] == identity)
    assert p.store.reconcile_openrouter_cost(round_id=ROUND, run_id=parent, call_identity=identity,
        uncertain_entry_id=candidate['uncertain_entry_id'], generation_id='gen-672-closed',
        credential_fingerprint=fingerprint, actual_microusd=0, cost_units='0')['status'] == 'settled'
    assert p.store.get_run(parent) == old_parent
    assert grant(p, parent)['idempotent'] is True
    assert claim(p)['run_id'] == child


def test_existing609_counted_thirds_keep_their_original_authority_and_quota(case):
    p = case
    cfg = execute.seed(p.pg, p.dsn, quota=2, old_calls=3)
    sql, _ = execute.render(p.c, cfg)
    # The historical fixture omits609's proof helper; install that committed
    # helper and data append only. Its quota IDs remain in the current shared
    # predicate, unchanged by672.
    with p.c.cursor() as q:
        q.execute(sql[sql.index('CREATE OR REPLACE FUNCTION public.lab_arena__nov03_unfinished_execute609_v1'):sql.index('DO $helper_verify609$')])
        q.execute(sql[sql.index('DO $append609$'):sql.index('DO $restore609$')])
    p.c.commit()
    lease = execute.claim(p.c)
    assert lease['status'] == 'leased' and lease['attempt'] == 3
    assert execute.payment(p.store, lease['run_id'], 1)['status'] == 'settled'
    denied = execute.payment(p.store, lease['run_id'], 1)
    assert (denied['status'], denied['reason']) == ('refused', 'per_icp_quota')
    with p.c.cursor() as q:
        assert scalar(q, 'SELECT public.lab_arena__operator_execute_recovery_child_v1(a)FROM public.lab_arena_runs a WHERE run_id=%s', (lease['run_id'],)) is False
