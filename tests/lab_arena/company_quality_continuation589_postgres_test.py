"""Restore the original partial-quality retry through real SQL and backoff."""
from copy import deepcopy
import json
import time

import pytest

from tests.lab_arena.company_quality_readiness589_postgres_test import (
    SQL, database, _quality_harness,
)
from tests.lab_arena.test_lab_arena_service_round import _start_round


CONTINUATION = 'public.lab_arena__judge_continuation_v1'
STRUCTURAL = 'public.lab_arena__quality_continuation_refs_v1'


def _continuation(cursor, round_row, run):
    cursor.execute('SELECT ' + CONTINUATION + '('
        'jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),'
        'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',
        (json.dumps(round_row), json.dumps(run)))
    return cursor.fetchone()[0]


def _structural(cursor, round_row, run):
    cursor.execute('SELECT ' + STRUCTURAL + '('
        'jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb),'
        'jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',
        (json.dumps(round_row), json.dumps(run)))
    return cursor.fetchone()[0]


def _negative_continuation_matrix(cursor, row, failed, pending):
    assert _continuation(cursor, row, failed) is True
    assert _continuation(cursor, row, pending) is True
    for change in ('refs_null', 'empty_refs', 'ref_hash', 'ref_order', 'ref_schema',
                   'scope_schema', 'scope_extra', 'scope_round', 'quality_policy',
                   'nested_quality', 'adapter_null', 'adapter_unknown', 'contact',
                   'nested_contact', 'legacy', 'source', 'participant', 'floor',
                   'assignment', 'parent_refs', 'parent_hash', 'claim_inventory'):
        r, j = deepcopy(row), deepcopy(pending if change.startswith('parent_') else failed)
        refs = j['company_judgment_refs']
        if change == 'refs_null': j['company_judgment_refs'] = None
        elif change == 'empty_refs': j['company_judgment_refs'] = []
        elif change == 'ref_hash': refs[0]['company_input_hash'] = 'sha256:' + '0' * 64
        elif change == 'ref_order': refs[0]['company_index'] = 1
        elif change == 'ref_schema': refs[0]['schema_version'] = 'foreign'
        elif change == 'scope_schema': refs[0]['scope_doc']['schema_version'] = 'foreign'
        elif change == 'scope_extra': refs[0]['scope_doc']['extra'] = True
        elif change == 'scope_round': refs[0]['scope_doc']['round_id'] = 'foreign'
        elif change == 'quality_policy': r['configuration_doc']['company_quality_policy'] = 'foreign'
        elif change == 'nested_quality': r['configuration_doc']['scorer_policy']['company_quality_policy'] = None
        elif change == 'adapter_null': r['configuration_doc']['scorer_policy']['scoring_adapter_version'] = None
        elif change == 'adapter_unknown': r['configuration_doc']['scorer_policy']['scoring_adapter_version'] = 'foreign'
        elif change == 'contact': r['configuration_doc']['contact_policy'] = 'foreign'
        elif change == 'nested_contact': r['configuration_doc']['scorer_policy']['contact_policy'] = None
        elif change == 'legacy': r['configuration_doc']['company_judgment_policy'] = 'company_judgment_v1'
        elif change == 'source': j['scored_run_id'] = 'foreign'
        elif change == 'participant': r['participants'] = []
        elif change == 'floor': j['scoring_generation_floor'] += 1
        elif change == 'assignment': j['assignment_id'] += ':foreign'
        elif change == 'parent_refs': j['company_judgment_refs'] = []
        elif change == 'parent_hash':
            refs[0]['company_input_hash'] = 'sha256:' + '0' * 64
            refs[0]['scope_doc']['company_input_hash'] = refs[0]['company_input_hash']
        elif change == 'claim_inventory': j['claim_response']['company_judgment_cache']['misses'] = []
        assert _continuation(cursor, r, j) is False, change
    # Genuine empty inventories remain structurally valid. Unclaimed retries
    # are bound to their actual parent's exact refs by the outer predicate.
    empty = deepcopy(failed)
    empty['company_judgment_refs'] = []
    empty['claim_response']['company_judgment_cache'].update(hits=[], misses=[])
    empty.update(status='accepted', terminal_cause='accepted')
    assert _structural(cursor, row, empty) is True
    empty.update(status='pending', terminal_cause=None, claim_response=None)
    assert _structural(cursor, row, empty) is True
    assert _continuation(cursor, row, {**pending, 'company_judgment_refs': []}) is False


@pytest.mark.parametrize('contacts', [False, True])
def test_partial_quality_retry_restores_original_contract(database, tmp_path, contacts):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness, partial_sources = _quality_harness(database, tmp_path, contacts=contacts, partial=True)
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('stage1_scoring', runners=2)
    harness.run_stage_with_runners(2)
    scores = harness.service.store.list_runs(harness.round_id, kind='score')
    failed, = [run for run in scores if run['status'] == 'failed']
    assert partial_sources == [failed['scored_run_id']]
    assert failed['attempt'] == 1 and failed['terminal_cause'] == 'judge_timeout'
    assert failed['result_doc']['company_judgment_completion_request_hash']
    assert len(failed['company_judgment_refs']) == 5
    assert not any(run['status'] == 'pending' for run in scores)
    assert harness.service.close_scoring(harness.round_id, 1)['status'] == 'closed'
    assert harness.service.score_stage(harness.round_id, 1) == {
        'status': 'retry', 'round_status': 'stage1_judged', 'reason': 'judge_incomplete', 'remaining': 1,
    }
    row = harness.service.store.get_round(harness.round_id)
    with connect() as connection, connection.cursor() as cursor:
        assert _continuation(cursor, row, failed) is False
        cursor.execute('SELECT public.lab_arena__judge_calls_clear_v1(%s)', (failed['assignment_id'],))
        assert cursor.fetchone()[0] is True
        cursor.execute(SQL.read_text())
        assert _continuation(cursor, row, failed) is True
    # An unresolved provider reservation must still prevent retry creation.
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,"
            "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
            "VALUES('reservation',%s,%s,%s,%s,1,%s,'openrouter','quality589-unresolved','host',1,'{}')",
            (failed['miner_hotkey'], harness.round_id, failed['submission_id'], failed['run_id'], 'sha256:' + 'a' * 64))
        cursor.execute('SET LOCAL session_replication_role=origin')
        cursor.execute('SELECT public.lab_arena__judge_calls_clear_v1(%s)', (failed['assignment_id'],))
        assert cursor.fetchone()[0] is False
        cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (harness.round_id,))
        assert cursor.fetchone()[0]['retried'] == 0
        cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>1", (failed['assignment_id'],))
        assert cursor.fetchone()[0] == 0
        connection.rollback()
    assert harness.service.store.expire_leases(harness.round_id)['retried'] == 1
    pending, = [run for run in harness.service.store.list_runs(harness.round_id, kind='score') if run['status'] == 'pending']
    assert pending['attempt'] == 2 and pending['claim_response'] is None
    with connect() as connection, connection.cursor() as cursor:
        _negative_continuation_matrix(cursor, row, failed, pending)
    # A claim before the real first30s backoff cannot mutate or lease the retry.
    before = harness.service.store.get_run(pending['run_id'])
    early_runners = [harness.runner(index) for index in range(2)]
    try:
        assert [runner.run_once() for runner in early_runners] == [0, 0]
    finally:
        for runner in early_runners:
            runner.close()
    assert harness.service.store.get_run(pending['run_id']) == before
    started = time.monotonic()
    harness.run_stage_with_runners(2)
    assert time.monotonic() - started < 60
    accepted = harness.service.store.get_run(pending['run_id'])
    assert accepted['status'] == 'accepted'
    context = accepted['claim_response']['company_judgment_cache']
    assert len(context['hits']) == 4 and len(context['misses']) == 1
    assert all(hit['evidence_doc']['source_score_run_id'] == failed['run_id'] for hit in context['hits'])
    assert harness.service.store.get_run(failed['run_id']) == failed
    harness.advance_until('scored', runners=2)
    assert harness.service.publish(harness.round_id)['status'] == 'ok'
    assert harness.service.store.get_run(failed['run_id']) == failed
