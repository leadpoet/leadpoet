"""Opening a full scoring cohort validates one frozen configuration efficiently."""
import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

from tests.lab_arena import company_judgment_validation_projection548_postgres_test as prior
from tests.lab_arena import incomplete_participant_publication540_postgres_test as base

SQL = Path(__file__).parents[2] / 'scripts/549-lab-arena-open-scoring-frozen-config-projection.sql'
SIG = 'public.lab_arena_open_scoring_v3(text,smallint,jsonb)'


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    try:
        yield next(gen)
    finally:
        gen.close()


@pytest.fixture(scope='module')
def completed(database, tmp_path_factory):
    gen = base.completed.__wrapped__(database, tmp_path_factory)
    try:
        yield next(gen)
    finally:
        gen.close()


def _function_state(cur):
    cur.execute("SELECT encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex'),"
                "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIG,))
    return cur.fetchone()


def _seed(database, completed):
    pg, dsn = database
    _, round_row, runs = completed
    cfg, refs = prior.sample()
    cfg = {**round_row['configuration_doc'], **cfg}
    cfg['stage_1_icp_count'] = cfg['stage_2_icp_count'] = 10
    cfg['scorer_policy'] = {**round_row['configuration_doc']['scorer_policy'],
                            **cfg['scorer_policy']}
    for ref in refs:
        scope = ref['scope_doc']
        scope.update(round_id=round_row['round_id'],
                     evaluation_date=round_row['evaluation_date'],
                     network_name=round_row['arena_network_name'],
                     netuid=round_row['arena_netuid'])
    source = next(r for r in runs if r['kind'] == 'execute' and r['stage'] == 2
                  and r['status'] == 'accepted')
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
                    "WHERE round_id=%s AND status IN ('accepted','frozen') LIMIT 1",
                    (round_row['round_id'],))
        submission = cur.fetchone()[0]
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    participants = [p for p in round_row['participants'] if p['is_king']]
    items = []
    for i in range(239):
        n, pos = divmod(i, 10)
        sid = f'profile-sub-{n:02d}'
        rid = f"{round_row['round_id']}:{sid}:2:{pos}:1"
        if pos == 0:
            participant = dict(round_row['participants'][0])
            participant['submission_id'] = sid
            participant['miner_hotkey'] = '5' + '1' * 45 + alphabet[n // 58] + alphabet[n % 58]
            participants.append(participant)
        items.append(dict(scored_run_id=rid, submission_id=sid,
                          icp_position=pos, output_ref=f'arena/{rid}.json',
                          company_judgment_refs=refs))
    miners = {p['submission_id']: p['miner_hotkey'] for p in participants}
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND stage=2",
                    (round_row['round_id'],))
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed',"
                    "configuration_doc=%s::jsonb,participants=%s::jsonb,stage_generation=5 "
                    "WHERE round_id=%s", (json.dumps(cfg), json.dumps(participants), round_row['round_id']))
        for participant in participants:
            if participant['is_king']:
                continue
            value = dict(submission, submission_id=participant['submission_id'],
                         miner_hotkey=participant['miner_hotkey'])
            cur.execute('INSERT INTO public.lab_arena_submissions SELECT '
                        '(jsonb_populate_record(NULL::public.lab_arena_submissions,%s::jsonb)).*',
                        (json.dumps(value),))
        for item in items:
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                        "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
                        "stage_generation,scoring_generation_floor,kind,output_ref,terminal_cause,"
                        "result_doc) VALUES(%s,%s,%s,%s,%s,2,%s,1,'accepted',5,0,'execute',"
                        "%s,'accepted',%s::jsonb)",
                        (item['scored_run_id'], item['scored_run_id'].rsplit(':', 1)[0],
                         round_row['round_id'], item['submission_id'],
                         miners[item['submission_id']], item['icp_position'],
                         item['output_ref'], json.dumps(source['result_doc'])))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('SELECT pg_column_size(configuration_doc) FROM public.lab_arena_rounds '
                    'WHERE round_id=%s', (round_row['round_id'],))
        assert 400_000 < cur.fetchone()[0] < 1_000_000
    return round_row, items, refs


def _invoke(cur, round_id, stage, items):
    started = time.monotonic()
    cur.execute('SELECT public.lab_arena_open_scoring_v3(%s,%s::smallint,%s::jsonb)',
                (round_id, stage, json.dumps(items)))
    outcome = cur.fetchone()[0]
    elapsed = time.monotonic() - started
    cur.execute("SELECT scored_run_id,submission_id,icp_position,company_judgment_refs "
                "FROM public.lab_arena_runs WHERE round_id=%s AND stage=%s "
                "AND kind='score' ORDER BY scored_run_id", (round_id, stage))
    bindings = cur.fetchall()
    return outcome, elapsed, bindings


def _source_state(cur, round_id):
    cur.execute("SELECT md5(COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.run_id)::text,'null')) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute'", (round_id,))
    return cur.fetchone()[0]


def test_full_cohort_and_guards(database, completed):
    pg, dsn = database
    round_row, items, refs = _seed(database, completed)
    rid = round_row['round_id']
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        before_hash, before_metadata = _function_state(cur)
        assert before_hash == '51743119194087238e0b30e1f7803b1b5699950573d3215064b40c1844c9c42b'
        source_before = _source_state(cur, rid)
        original, old_seconds, old_bindings = _invoke(cur, rid, 2, items)
        assert original['status'] == 'ok' and original['assignments'] == 239
        assert len(old_bindings) == 239 and _source_state(cur, rid) == source_before
        conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        after_hash, after_metadata = _function_state(cur)
        assert after_hash == '0d2c9cd4aadea7d8de4f455dfc9d3d352527940b3833a9ad9af5ca967fce18e3'
        assert after_metadata == before_metadata
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        updated, new_seconds, new_bindings = _invoke(cur, rid, 2, items)
        assert updated == original and new_bindings == old_bindings
        assert len(new_bindings) == 239
        assert {x[0] for x in new_bindings} == {x['scored_run_id'] for x in items}
        assert _source_state(cur, rid) == source_before
        repeated, _, repeated_bindings = _invoke(cur, rid, 2, items)
        assert repeated['status'] == 'existing'
        assert repeated_bindings == new_bindings
        conn.rollback()
    assert new_seconds < old_seconds / 3, (old_seconds, new_seconds)

    # A missing round and invalid references, scorer binding, source identity,
    # output, and accepted status still fail without creating a score row.
    faults = []
    invalid_ref = copy.deepcopy(items[0]); invalid_ref['company_judgment_refs'][0]['cache_key'] = 'bad'
    faults.append((rid, [invalid_ref], '22023', None))
    invalid_last = copy.deepcopy(items[-1]); invalid_last['company_judgment_refs'][0]['cache_key'] = 'bad'
    faults.append((rid, items[:-1] + [invalid_last], '22023', None))
    invalid_scorer = copy.deepcopy(items[0])
    invalid_scorer['company_judgment_refs'][0]['scope_doc']['scorer_image_digest'] = 'sha256:' + 'b' * 64
    faults.append((rid, [invalid_scorer], '22023', None))
    invalid_run = dict(items[0], scored_run_id='missing-run')
    faults.append((rid, [invalid_run], '22023', None))
    invalid_output = dict(items[0], output_ref='wrong-output')
    faults.append((rid, [invalid_output], '22023', None))
    faults.append(('missing-round', [items[0]], 'P0002', None))
    faults.append((rid, [items[0]], '22023', 'failed'))
    for round_id, work, state, source_status in faults:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            if source_status:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('UPDATE public.lab_arena_runs SET status=%s WHERE run_id=%s',
                            (source_status, items[0]['scored_run_id']))
                cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(pg.Error) as error:
                _invoke(cur, round_id, 2, work)
            assert error.value.pgcode == state
            conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND stage=2 AND kind='score'", (rid,))
        assert cur.fetchone() == (0,)
        assert _source_state(cur, rid) == source_before

    # The baseline stage uses the same function, and an operator hold still
    # prevents score creation at the native trigger boundary.
    baseline = next(r for r in completed[2] if r['kind'] == 'execute' and r['stage'] == 1
                    and r['status'] == 'accepted')
    baseline_item = dict(scored_run_id=baseline['run_id'],
                         submission_id=baseline['submission_id'],
                         icp_position=baseline['icp_position'],
                         output_ref=baseline['output_ref'], company_judgment_refs=refs)
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 AND kind='score'", (rid,))
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_closed' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        result, _, bindings = _invoke(cur, rid, 1, [baseline_item])
        assert result['status'] == 'ok' and result['assignments'] == 1
        assert len(bindings) == 1 and bindings[0][0] == baseline['run_id']
        conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        cur.execute('SET LOCAL session_replication_role=origin')
        with pytest.raises(pg.Error) as error:
            _invoke(cur, rid, 2, [items[0]])
        assert error.value.pgcode == '55000'
        conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND stage=2", (rid,))
        assert cur.fetchone() == (0,)

    # Fail closed on a changed function body or execute privilege. A failed
    # migration rolls back both the injected drift and any partial DDL.
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
        post_definition = cur.fetchone()[0]
        cur.execute(post_definition.replace('lab_arena_round_missing',
                                            'unreviewed_round_missing'))
        with pytest.raises(pg.Error) as error:
            cur.execute(SQL.read_text())
        assert error.value.pgcode == '55000'
        conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert _function_state(cur) == (after_hash, after_metadata)
        cur.execute('GRANT EXECUTE ON FUNCTION ' + SIG + ' TO anon')
        with pytest.raises(pg.Error) as error:
            cur.execute(SQL.read_text())
        assert error.value.pgcode == '42501'
        conn.rollback()
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        assert _function_state(cur) == (after_hash, after_metadata)
