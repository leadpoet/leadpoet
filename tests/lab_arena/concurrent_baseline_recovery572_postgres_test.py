"""The real recovery authority permits early mining only with a full baseline."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.lab_arena import baseline_deadline_transition564_postgres_test as current
from tests.lab_arena import evaluation_recovery536_postgres_test as recovery


scenario = current.scenario
SQL = Path(__file__).parents[2] / 'scripts/572-lab-arena-concurrent-baseline-judging.sql'


def authority_snapshot(cur, rid):
    cur.execute('SELECT configuration_doc,scoring_generation_floor FROM '
                'public.lab_arena_rounds WHERE round_id=%s', (rid,))
    round_authority = cur.fetchone()
    cur.execute('SELECT to_jsonb(s) FROM public.lab_arena_scoring_supersessions s '
                'WHERE round_id=%s ORDER BY generation_floor', (rid,))
    return round_authority, cur.fetchall()


@pytest.mark.parametrize('missing_source', [False, True])
def test_real_supersession_allows_mining_while_baseline_judges_remain_pending(scenario, missing_source):
    db = scenario[0]
    store, row, cfg, items = recovery.seed(db)
    rid = row['round_id']
    try:
        # The small historical seed has one saved baseline position. Complete
        # its two-position frozen test bank before the protected recovery RPC.
        # These are saved test inputs; no provider or production data is used.
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            old_cfg = deepcopy(row['configuration_doc'])
            old_cfg['stage_1_icp_count'] = old_cfg['stage_2_icp_count'] = 1
            cfg['stage_1_icp_count'] = cfg['stage_2_icp_count'] = 1
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb '
                        'WHERE round_id=%s', (json.dumps(old_cfg), rid))
            cur.execute("UPDATE public.lab_arena_runs SET result_doc=%s::jsonb "
                        "WHERE round_id=%s AND kind='execute' AND status='accepted'",
                        (json.dumps({'terminal_status': 'accepted'}), rid))
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',
                        (items[0]['scored_run_id'],))
            source = cur.fetchone()[0]
            source.update(icp_position=1,
                          assignment_id=source['assignment_id'].rsplit(':', 1)[0] + ':1')
            source['run_id'] = source['assignment_id'] + ':' + str(source['attempt'])
            cur.execute('INSERT INTO public.lab_arena_runs SELECT '
                        '(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',
                        (json.dumps(source),))
            extra = deepcopy(items[0])
            extra.update(scored_run_id=source['run_id'], icp_position=1)
            items.append(extra)
            cur.execute('UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc='
                        'jsonb_set(stage1_scoring_plan_doc,\'{work_items}\', '
                        "(stage1_scoring_plan_doc->'work_items') || %s::jsonb) WHERE round_id=%s",
                        (json.dumps([extra]), rid))
            cur.execute('SET LOCAL session_replication_role=origin')
        row = store.get_round(rid)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            result = recovery.invoke(cur, row, cfg, items)
            assert result['queued_executions'] == 1 and result['queued_judges'] == 2
            cur.execute('UPDATE public.lab_arena_restart_claim_control '
                        'SET operator_paused=false WHERE singleton')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '30 minutes' "
                        "WHERE round_id=%s AND status='pending'", (rid,))
            if missing_source:
                cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (source['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            baseline = next(p for p in row['participants'] if p['is_king'])
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                        "AND stage=1 AND kind='score' AND status='pending' "
                        'AND scoring_generation_floor=%s', (rid, result['generation_floor']))
            assert cur.fetchone() == (2,)
            before = authority_snapshot(cur, rid)
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',
                        ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',))
            installed = 'lab_arena_concurrent_baseline_priority_572' in cur.fetchone()[0]
        runner = recovery.base.company.original.hotkey('overlap572-recovery')
        # Exclusion makes this an execution-only claim without changing any
        # queue authority. Before 572 the genuine 536 barrier rejects it.
        if not installed:
            lease, *_ = recovery.base.company.original.claim(
                store, rid, runner, excluded=[baseline['miner_hotkey']])
            assert lease['status'] == 'no_pending', lease
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            assert authority_snapshot(cur, rid) == before
            cur.execute('SELECT public.lab_arena__baseline_execution_ready_v1(%s)', (rid,))
            assert cur.fetchone() == (not missing_source,)
        lease, *_ = recovery.base.company.original.claim(
            store, rid, runner, excluded=[baseline['miner_hotkey']])
        if missing_source:
            assert lease['status'] == 'no_pending', lease
        else:
            assert lease['status'] == 'leased' and lease['kind'] == 'execute'
            assert lease['stage'] == 2 and lease['attempt'] == 2
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert authority_snapshot(cur, rid) == before
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                        "AND stage=1 AND kind='score' AND status='pending' "
                        'AND scoring_generation_floor=%s', (rid, result['generation_floor']))
            assert cur.fetchone() == (2,)
            cur.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)', (rid,))
            assert cur.fetchone() == (False,)
    finally:
        store.close()
