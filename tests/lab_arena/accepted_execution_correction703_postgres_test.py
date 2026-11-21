"""Accepted infrastructure correction retains evidence and uses finite native work."""
from copy import deepcopy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring, verify
from lab_arena.service import ArenaService
from scripts import prepare_accepted_execution_correction as preparation
from tests.lab_arena import operator_execution_recovery672_postgres_test as old
from tests.lab_arena.company_judgments_test import _company, _input
from tests.test_arena_score_integrity import _positive_breakdown
from qualification.scoring.competition import apply_company_judgment_context, raw_company_judgment

SCRIPTS = Path(__file__).parents[2] / 'scripts'
SQL = SCRIPTS / '703-lab-arena-accepted-execution-infrastructure-correction.sql'
ROUND = old.ROUND
HEALTHY = old.hotkey('672-healthy')
WRONG = old.hotkey('703-wrong')
TOKEN = old.TOKEN
GRANT = 'public.lab_arena_correct_accepted_execution_v1(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
PLAN = 'public.lab_arena_prepare_corrected_execution_judgment_v1(%s::jsonb,%s,%s,%s)'


@pytest.fixture(scope='module')
def database():
    gen = old.database.__wrapped__()
    pg, dsn = next(gen)
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            # The existing672 fixture has production's execute/claim/close
            # lineage. Install699's exact terminal guard and proof reader,
            # without its unrelated scorer-correction function patches.
            text = (SCRIPTS / '699-lab-arena-targeted-accepted-judgment-correction.sql').read_text()
            before, after = re.search(r'\$old699_0\$(.*?)\$old699_0\$,\$new699_0\$(.*?)\$new699_0\$', text, re.S).groups()
            signature = 'public.lab_arena_runs_terminal_v1()'
            d = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            assert d.count(before) == 1
            q.execute(d.replace(before, after))
            reader = text[text.index('CREATE OR REPLACE FUNCTION public.lab_arena_retired_accepted_judgment_v1'):text.index('DO $restore699$')]
            q.execute(reader)
            q.execute(SQL.read_text())
        yield pg, dsn
    finally:
        gen.close()


def row_hash(p, run_id):
    with p.c.cursor() as q:
        return old.scalar(q, "SELECT 'sha256:'||encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (run_id,))


@pytest.fixture(params=[0, 1], ids=['empty', 'nonempty'])
def case(database, request):
    gen = old.case.__wrapped__(database)
    p = next(gen)
    try:
        p.parent = p.parents[0]
        p.cfg.pop('company_quality_policy', None)
        p.cfg.pop('intent_details_policy', None)
        p.judge = p.parent.rsplit(':', 1)[0] + ':score:1'
        p.companies = [_company('Retained')] if request.param else []
        p.objects = {p.parent + '-output': contracts.canonical_json({
            'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, 'companies': p.companies}).encode()}
        with p.c.cursor() as q:
            q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted',result_doc='{\"terminal_status\":\"accepted\"}',output_ref=run_id||'-output',per_icp_score=17,qualification_doc='{\"companies\":[]}' WHERE run_id=%s", (p.parent,))
            source = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s', (p.parent,))
            judge = dict(source, run_id=p.judge, assignment_id=source['assignment_id'] + ':score',
                         kind='score', scored_run_id=p.parent, attempt=1, output_ref=p.judge + '-output',
                         per_icp_score=None, qualification_doc=None, runner_hotkey=HEALTHY, claim_request_id=uuid.uuid4().hex,
                         company_judgment_refs=[])
            old.base.insert(q, 'lab_arena_runs', judge)
            sibling = dict(source, run_id=source['assignment_id'].rsplit(':', 1)[0] + ':1:1',
                           assignment_id=source['assignment_id'].rsplit(':', 1)[0] + ':1',
                           icp_position=1, attempt=1, runner_hotkey=HEALTHY, claim_request_id=uuid.uuid4().hex)
            p.healthy = sibling['run_id']
            old.base.insert(q, 'lab_arena_runs', sibling)
            sibling_judge = dict(judge, run_id=sibling['assignment_id'] + ':score:1',
                                 assignment_id=sibling['assignment_id'] + ':score', scored_run_id=p.healthy,
                                 icp_position=1, claim_request_id=uuid.uuid4().hex)
            old.base.insert(q, 'lab_arena_runs', sibling_judge)
            plan = {'round_id': ROUND, 'stage': 2, 'work_items': [dict(scored_run_id=r['run_id'],
                submission_id=r['submission_id'], icp_position=r['icp_position'], output_ref=r['output_ref'])
                for r in (source, sibling)], 'zero_rows': []}
            q.execute("UPDATE lab_arena_rounds SET status='stage2_scoring',stage2_scoring_plan_doc=%s::jsonb,configuration_doc=%s::jsonb WHERE round_id=%s", (json.dumps(plan), json.dumps(p.cfg), ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
            q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        p.c.commit()
        p.parent_hash = row_hash(p, p.parent)
        p.judge_hash = row_hash(p, p.judge)
        p.service = object.__new__(ArenaService)
        p.service._config = SimpleNamespace(mode=p.cfg['mode'])
        p.service._store = p.store
        p.service._objects = SimpleNamespace(get_bounded=lambda ref, limit: p.objects[ref], get=lambda ref: p.objects[ref])
        p.service.evaluation_icps = lambda _: [_input()['icp']] * 10
        yield p
    finally:
        gen.close()


def arguments(p, **changes):
    args = dict(run=p.parent, run_hash=p.parent_hash, judge=p.judge, judge_hash=p.judge_hash,
                runner=HEALTHY, evidence='private:retained-provider-response', evidence_hash='sha256:' + 'e' * 64,
                request='accepted-execution-fixture', actor='fixture-owner', dry=False)
    args.update(changes)
    return tuple(args.values())


def grant(p, **changes):
    with p.c.cursor() as q:
        result = old.scalar(q, 'SELECT ' + GRANT, arguments(p, **changes))
    p.c.commit()
    return result


def snapshot(p):
    before = old.snapshot(p)
    with p.c.cursor() as q:
        for table, key in [('lab_arena_trajectory_events', 'event_id'),
                           ('lab_arena_company_judgments', 'cache_key,authority_slot'),
                           ('lab_arena_signal_judgments', 'cache_key,authority_slot')]:
            before[table] = old.scalar(q, 'SELECT coalesce(jsonb_agg(to_jsonb(r) ORDER BY ' + key + "),'[]') FROM " + table + ' r')
    return before


def claim(p, runner=HEALTHY):
    with p.c.cursor() as q:
        q.execute('SET ROLE lab_arena_service')
        result = old.scalar(q, 'SELECT lab_arena_claim_assignment(%s,%s,20,20,%s::text[],%s,%s,%s,4500)',
                           (ROUND, runner, [runner], uuid.uuid4().hex, 'sha256:' + uuid.uuid4().hex * 2, TOKEN))
        q.execute('RESET ROLE')
    p.c.commit()
    return result


def age(p, run_id):
    with p.c.cursor() as q:
        q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s", (run_id,))
    p.c.commit()


def finalize(p, child, *, dry=False, payload=None):
    payload = payload or preparation.prepare(p.service, child)
    with p.c.cursor() as q:
        result = old.scalar(q, 'SELECT ' + PLAN, (json.dumps(payload), 'accepted-execution-plan', 'fixture-owner', dry))
    p.c.commit()
    return result, payload


def complete_company_judgment(p, lease):
    """Use frozen cache refs and the ordinary evidence/completion contract."""
    cache = company_judgments.validate_lease_context(lease['company_judgment_cache'])
    run = p.store.get_run(lease['run_id'])
    refs = {r['company_index']: r for r in run['company_judgment_refs']}
    new = [{**miss, 'raw_judgment': {**raw_company_judgment(scoring.project_company_judgments(
        [p.companies[miss['company_index']]], [_positive_breakdown('Retained', 'retained.example', 'retained')],
        company_quality=False, scored_indexes=[0])[0]),
        'company_judgment_policy': company_judgments.POLICY}} for miss in cache['misses']]
    raw = {row['company_index']: row['evidence_doc']['raw_judgment'] for row in cache['hits']}
    raw.update({row['company_index']: row['raw_judgment'] for row in new})
    consumed = set()
    breakdowns = apply_company_judgment_context(p.companies,
        [raw[i] for i in range(len(p.companies))], company_quality=False,
        consumed_indexes=consumed)
    output = scoring.build_scoring_output(lease['scored_run_id'], breakdowns, company_judgments=new)
    output_ref = lease['run_id'] + '-output'
    p.objects[output_ref] = contracts.canonical_json(output).encode()
    completion_hash = contracts.document_hash({'run_id': lease['run_id'], 'output': output})
    evidence = [company_judgments.build_evidence_snapshot(new_judgment=row,
        company_ref=refs[row['company_index']], source_score_run_id=lease['run_id'],
        source_scored_run_id=lease['scored_run_id'], source_output_ref=output_ref,
        source_output_hash=contracts.document_hash(output), source_runner_hotkey=run['runner_hotkey'],
        source_claim_request_id=run['claim_request_id'], source_claim_request_hash=run['claim_request_hash'],
        source_lease_generation=run['lease_generation'], source_completion_request_hash=completion_hash,
        runner_authority_exclusions=lease['runner_authority_exclusions']) for row in new]
    result = p.store.complete_attempt(run_id=lease['run_id'], lease_token_hash=TOKEN,
        result={'terminal_status': 'accepted', 'company_judgment_consumed_indexes': sorted(consumed)},
        terminal_cause='accepted', output_ref=output_ref, output_hash=contracts.document_hash(output),
        company_judgment_evidence=[{k: row[k] for k in ('cache_key', 'company_input_hash', 'authority_slot')}
            | {'evidence_doc': row, 'evidence_hash': contracts.document_hash(row)} for row in evidence],
        completion_request_hash=completion_hash)
    assert result['status'] == 'accepted'
    return breakdowns


@pytest.fixture
def cached_case(case):
    p = case
    if p.companies:
        # Historical fixture setup only. Establish the original authority with
        # real claim/completion RPCs before taking the recovery preimage.
        round_row = p.store.get_round(ROUND)
        plan = round_row['stage2_scoring_plan_doc']
        item = p.service._scoring_work_items(round_row, 2, {'work_items': plan['work_items'][:1]})[0]
        plan['work_items'][0] = item
        with p.c.cursor() as q:
            q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE lab_arena_runs SET status='pending',terminal_cause=NULL,result_doc=NULL,output_ref=NULL,company_judgment_refs=%s::jsonb,claim_request_id=NULL,claim_request_hash=NULL,claim_response=NULL,runner_hotkey=NULL,lease_token_hash=NULL,lease_expires_at=NULL,created_at=now()-interval '1 hour' WHERE run_id=%s",
                      (json.dumps(item['company_judgment_refs']), p.judge))
            q.execute('UPDATE lab_arena_rounds SET stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s',
                      (json.dumps(plan), ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
            q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        p.c.commit()
        lease = claim(p)
        assert lease['run_id'] == p.judge
        assert len(lease['company_judgment_cache']['misses']) == 1
        complete_company_judgment(p, lease)
        p.parent_hash = row_hash(p, p.parent)
        p.judge_hash = row_hash(p, p.judge)
    return p


def test_dry_run_retirement_and_same_miner_control(case):
    p = case
    before = snapshot(p)
    preview = grant(p, dry=True)
    assert snapshot(p) == before
    result = grant(p)
    assert result['child_identity'] == preview['child_identity']
    after = snapshot(p)
    original = {r['run_id']: r for r in before['lab_arena_runs']}
    actual = {r['run_id']: r for r in after['lab_arena_runs']}
    assert len(actual) == len(original) + 1
    for run_id, row in original.items():
        omitted = {'status', 'terminal_cause', 'terminal_doc', 'updated_at'} if run_id in (p.parent, p.judge) else set()
        assert {k: v for k, v in row.items() if k not in omitted} == {k: v for k, v in actual[run_id].items() if k not in omitted}
    for table in set(before) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert before[table] == after[table]
    assert grant(p)['idempotent'] is True
    assert snapshot(p) == after
    assert claim(p, WRONG)['status'] == 'no_pending'
    child = result['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT lab_arena_retired_accepted_judgment_v1(%s)', (p.judge,)) is True
    assert p.store.close_scoring(ROUND, 2)['status'] == 'draining'


def test_native_execution_payment_judgment_and_plan(cached_case):
    p = cached_case
    before = snapshot(p)
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    assert old.execute.payment(p.store, child, 100)['status'] == 'settled'
    output = child + '-output'
    p.objects[output] = contracts.canonical_json({'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                                                'companies': p.companies}).encode()
    assert p.store.complete_attempt(run_id=child, lease_token_hash=TOKEN,
        result={'terminal_status': 'accepted'}, terminal_cause='accepted', output_ref=output)['status'] == 'accepted'
    assert p.store.close_scoring(ROUND, 2)['status'] == 'draining'
    unchanged = snapshot(p)
    preview, payload = finalize(p, child, dry=True)
    assert snapshot(p) == unchanged
    result, _ = finalize(p, child, payload=payload)
    assert result['child_identity'] == preview['child_identity']
    assert finalize(p, child, payload=payload)[0]['idempotent'] is True
    current = p.store.get_round(ROUND)['stage2_scoring_plan_doc']
    original = before['lab_arena_rounds'][0]['stage2_scoring_plan_doc']
    assert sum(a != b for a, b in zip(current['work_items'], original['work_items'])) == 1
    assert current['zero_rows'] == original['zero_rows']
    judge = result['child_identity']['run_id']
    assert p.store.close_scoring(ROUND, 2)['status'] == 'draining'
    age(p, judge)
    assert claim(p, WRONG)['status'] == 'no_pending'
    lease = claim(p)
    assert lease['run_id'] == judge and lease['scored_run_id'] == child
    cache = lease['company_judgment_cache']
    assert cache['misses'] == []
    assert len(cache['hits']) == len(p.companies)
    if p.companies:
        assert cache['hits'][0]['evidence_doc']['source_score_run_id'] == p.judge
        assert cache['hits'][0]['evidence_doc']['source_scored_run_id'] == p.parent
    complete_company_judgment(p, lease)
    accepted_judge = p.store.get_run(judge)
    breakdowns = p.service._verified_breakdowns(accepted_judge, icp=_input()['icp'],
        companies=p.companies, policy=p.cfg['scorer_policy'])
    assert len(breakdowns) == len(p.companies)
    assert p.store.close_scoring(ROUND, 2)['status'] == 'closed'
    executed = p.store.get_run(child)
    score = verify.scored_row(executed['submission_id'], 0, child, _input()['icp'],
                             p.companies, breakdowns, p.cfg['scorer_policy'])
    records = scoring.run_scores_for_store({'rows': [score], 'integrity_policy': 'arena_integrity_v1'}, [executed])
    assert p.service._record_current_run_scores(p.store.get_round(ROUND), 2, records,
        judgments={child: accepted_judge})['recorded'] == 1
    assert p.store.get_run(child)['per_icp_score'] == score['per_icp_score']
    assert p.store.get_run(child)['qualification_doc'] == records[0]['qualification_doc']
    # This focused fixture has no complete baseline/stage bundle. Recording one
    # repaired score cannot publish the round or activate reward authority.
    assert p.service.publish(ROUND) == {'status': 'stale', 'round_status': 'stage2_judged'}
    after = snapshot(p)
    retained = {r['run_id']: r for r in before['lab_arena_runs']}
    assert next(r for r in after['lab_arena_runs'] if r['run_id'] == p.healthy) == retained[p.healthy]
    old_ledger = before['lab_arena_ledger']
    assert after['lab_arena_ledger'][:len(old_ledger)] == old_ledger
    for table in ('lab_arena_company_judgments', 'lab_arena_signal_judgments'):
        assert after[table] == before[table]
    for field in ('published_at', 'reward_activated_at'):
        assert after['lab_arena_rounds'][0][field] == before['lab_arena_rounds'][0][field] is None
    for run_id in (p.parent, p.judge):
        actual = next(r for r in after['lab_arena_runs'] if r['run_id'] == run_id)
        for field in ('result_doc', 'output_ref', 'per_icp_score', 'qualification_doc', 'claim_response'):
            assert actual[field] == retained[run_id][field]


@pytest.mark.parametrize('fault', ['source_hash', 'judge_hash', 'runner', 'evidence_hash', 'published', 'reward', 'guard', 'deadline', 'generation', 'source_result', 'plan'])
def test_invalid_grant_is_atomic(case, fault):
    p = case
    changes = {}
    if fault in ('source_hash', 'judge_hash', 'evidence_hash'):
        changes[{'source_hash': 'run_hash', 'judge_hash': 'judge_hash', 'evidence_hash': 'evidence_hash'}[fault]] = 'sha256:' + '0' * 64
        if fault == 'evidence_hash': changes['evidence_hash'] = 'bad'
    elif fault == 'runner':
        changes['runner'] = old.hotkey('672-old-yuma')
    else:
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            if fault == 'published': q.execute("UPDATE lab_arena_rounds SET published_at=now() WHERE round_id=%s", (ROUND,))
            elif fault == 'reward': q.execute("UPDATE lab_arena_rounds SET reward_basis_doc='{}' WHERE round_id=%s", (ROUND,))
            elif fault == 'guard': q.execute("UPDATE lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton")
            elif fault == 'deadline': q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"') WHERE round_id=%s", (ROUND,))
            elif fault == 'source_result':
                q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
                q.execute("UPDATE lab_arena_runs SET result_doc='{\"terminal_status\":\"model_error\"}' WHERE run_id=%s", (p.parent,))
                q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
            elif fault == 'plan': q.execute("UPDATE lab_arena_rounds SET stage2_scoring_plan_doc='{}' WHERE round_id=%s", (ROUND,))
            elif fault == 'generation': q.execute('UPDATE lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s', (ROUND,))
        p.c.commit()
    before = snapshot(p)
    with pytest.raises(p.pg.Error): grant(p, **changes)
    p.c.rollback()
    assert snapshot(p) == before


@pytest.mark.parametrize('outcome', ['model_error', 'provider_error', 'expired'])
def test_one_execution_attempt_never_recurses(case, outcome):
    p = case
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s", (child,))
        p.c.commit()
    else:
        assert p.store.complete_attempt(run_id=child, lease_token_hash=TOKEN,
            result={'terminal_status': outcome}, terminal_cause=outcome, output_ref=None)['status'] == 'failed'
    p.store.expire_leases(ROUND)
    assert claim(p)['status'] == 'no_pending'
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT max(attempt) FROM lab_arena_runs WHERE assignment_id=%s', (child.rsplit(':', 1)[0],)) == int(child.rsplit(':', 1)[1])


def test_owner_only_and_exact_replay(case):
    p = case
    before = snapshot(p)
    for role in ('lab_arena_service', 'anon', 'authenticated', 'service_role'):
        with p.c.cursor() as q:
            q.execute('SET ROLE ' + role)
            with pytest.raises(p.pg.Error): old.scalar(q, 'SELECT ' + GRANT, arguments(p))
        p.c.rollback()
    with p.c.cursor() as q:
        definitions = old.scalar(q, "SELECT jsonb_object_agg(p.oid::text,to_jsonb(p)) FROM pg_proc p WHERE pronamespace='public'::regnamespace")
        q.execute(SQL.read_text())
        assert old.scalar(q, "SELECT jsonb_object_agg(p.oid::text,to_jsonb(p)) FROM pg_proc p WHERE pronamespace='public'::regnamespace") == definitions
    p.c.commit()
    assert snapshot(p) == before


def test_wrong_runner_cannot_reserve_even_with_current_token(case):
    p = case
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s', (WRONG, child))
    p.c.commit()
    before = snapshot(p)
    assert old.execute.payment(p.store, child, 100)['status'] == 'stale'
    assert snapshot(p) == before


def test_concurrent_grants_share_one_child(case):
    p = case
    before = snapshot(p)
    args = arguments(p)
    def invoke():
        with p.pg.connect(**p.dsn) as c, c.cursor() as q:
            return old.scalar(q, 'SELECT ' + GRANT, args)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert {x['child_identity']['run_id'] for x in results} == {results[0]['child_identity']['run_id']}
    assert sorted(x['idempotent'] for x in results) == [False, True]
    assert len(snapshot(p)['lab_arena_runs']) == len(before['lab_arena_runs']) + 1
    with pytest.raises(p.pg.Error): grant(p, request='second-request')
    p.c.rollback()


@pytest.mark.parametrize('outcome', ['judge_error', 'expired'])
def test_replacement_judgment_is_also_finite(case, outcome):
    p = case
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    p.objects[child + '-output'] = contracts.canonical_json({'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                                                          'companies': p.companies}).encode()
    assert p.store.complete_attempt(run_id=child, lease_token_hash=TOKEN, result={'terminal_status': 'accepted'},
                                   terminal_cause='accepted', output_ref=child + '-output')['status'] == 'accepted'
    judge = finalize(p, child)[0]['child_identity']['run_id']
    assert p.store.close_scoring(ROUND, 2)['status'] == 'draining'
    age(p, judge)
    assert claim(p)['run_id'] == judge
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s", (judge,))
        p.c.commit()
    else:
        assert p.store.complete_attempt(run_id=judge, lease_token_hash=TOKEN,
            result={'terminal_status': 'judge_error'}, terminal_cause='judge_error', output_ref=None)['status'] == 'failed'
    p.store.expire_leases(ROUND)
    assert claim(p)['status'] == 'no_pending'
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT max(attempt) FROM lab_arena_runs WHERE assignment_id=%s', (judge.rsplit(':', 1)[0],)) == int(judge.rsplit(':', 1)[1])


def test_tampered_receipt_cannot_claim_or_gain_another_attempt(case):
    p = case
    child = grant(p)['child_identity']['run_id']
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_scoring_supersessions DISABLE TRIGGER lab_arena_scoring_supersessions_append_only')
        q.execute("UPDATE lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,'{parent_sha256}',to_jsonb('sha256:'||repeat('0',64))) WHERE request_id='accepted-execution-fixture'")
        q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||encode(extensions.digest(recovery_doc::text,'sha256'),'hex') WHERE request_id='accepted-execution-fixture'")
        q.execute('ALTER TABLE lab_arena_scoring_supersessions ENABLE ALWAYS TRIGGER lab_arena_scoring_supersessions_append_only')
    p.c.commit()
    assert claim(p)['status'] == 'no_pending'
    with pytest.raises(p.pg.Error): grant(p)
    p.c.rollback()
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT lab_arena_retired_accepted_judgment_v1(%s)', (p.judge,)) is False
        assert old.scalar(q, 'SELECT lab_arena__operator_execute_recovery_child_v1(r) FROM lab_arena_runs r WHERE run_id=%s', (child,)) is True


def test_preparation_hash_and_work_item_use_one_object_read(case):
    p = case
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    raw = contracts.canonical_json({'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                                   'companies': p.companies}).encode()
    p.objects[child + '-output'] = raw
    assert p.store.complete_attempt(run_id=child, lease_token_hash=TOKEN, result={'terminal_status': 'accepted'},
        terminal_cause='accepted', output_ref=child + '-output')['status'] == 'accepted'
    reads = []
    def read(ref, limit):
        reads.append(ref)
        assert len(reads) == 1
        return raw
    p.service._objects = SimpleNamespace(get_bounded=read)
    payload = preparation.prepare(p.service, child)
    assert reads == [child + '-output']
    assert payload['output_sha256'] == contracts.hash_bytes(raw)
    before = snapshot(p)
    bad = deepcopy(payload)
    bad['work_item']['scored_run_id'] = p.healthy
    with pytest.raises(p.pg.Error): finalize(p, child, payload=bad)
    p.c.rollback()
    assert snapshot(p) == before


def test_original_expired_execution_recovery_stays_unchanged(database):
    gen = old.case.__wrapped__(database)
    p = next(gen)
    try:
        old.test_nine_observed_states_dry_run_idempotence_and_preservation(p)
    finally:
        gen.close()


def test_original_spending_cap_is_not_reset(case):
    p = case
    preview = grant(p, dry=True)
    prior_cost = int(preview['original_cost']['settled_microusd'])
    child = grant(p)['child_identity']['run_id']
    assert claim(p)['run_id'] == child
    assert old.execute.payment(p.store, child, 6000000)['status'] == 'settled'
    refused = old.execute.payment(p.store, child, 1)
    assert (refused['status'], refused['reason']) == ('refused', 'money_cap')
    with p.c.cursor() as q:
        cost = old.scalar(q, 'SELECT lab_arena__successful_icp_cost_state(%s,%s,%s)',
                          (ROUND, 'sub-2dfb', 0))
    assert cost['settled_microusd'] == prior_cost + 6000000
    assert cost['competition_settled_microusd'] == prior_cost + 6000000


@pytest.mark.parametrize('fault', ['helper', 'acl'])
def test_migration_reapplication_refuses_drift(case, fault):
    p = case
    signature = 'public.lab_arena__accepted_execution_receipt_v1(public.lab_arena_rounds,public.lab_arena_runs)'
    with p.c.cursor() as q:
        original = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        if fault == 'helper': q.execute(original.replace(' RETURN TRUE;', ' RETURN FALSE;'))
        else: q.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO lab_arena_service')
    p.c.commit()
    before = snapshot(p)
    try:
        with p.c.cursor() as q:
            with pytest.raises(p.pg.Error, match='installed helper differs'): q.execute(SQL.read_text())
        p.c.rollback()
        assert snapshot(p) == before
    finally:
        with p.c.cursor() as q:
            q.execute(original)
            q.execute('REVOKE ALL ON FUNCTION ' + signature + ' FROM lab_arena_service')
        p.c.commit()
