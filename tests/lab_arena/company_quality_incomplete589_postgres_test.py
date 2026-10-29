"""Mixed quality publication validates all retained accepted company evidence."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import hashlib

import pytest

from lab_arena import runtime, scoring
from lab_arena.store import ArenaStoreError
from tests.lab_arena.company_quality_readiness589_postgres_test import (
    SQL, database, _quality_harness, _negative_refs_matrix, _data_snapshot,
)
from tests.lab_arena.test_lab_arena_service_round import _start_round

REASON = 'public.lab_arena__participant_incomplete_reason_v1(text,text)'
PRE = 'f0947c0d204d0883d926781cfe03ba22ad2b940ea16cf7618de90a9533cd44c4'
POST = '4a2ada51252a041add7be1aa4595d8aedd18b68ae85c64e2ca3163dae28f870f'


def _reason(cursor, harness, miner):
    cursor.execute('SELECT ' + REASON.split('(')[0] + '(%s,%s)', (harness.round_id, miner))
    return cursor.fetchone()[0]


@pytest.mark.parametrize('contacts', [False, True])
def test_real_quality_judge_outage_publishes_incomplete_challenger_at_deadline(database, tmp_path, contacts):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (REASON,))
        predecessor = cursor.fetchone()[0]
        assert hashlib.sha256(predecessor.encode()).hexdigest() == PRE
        cursor.execute(SQL.read_text())
        # Model the exact reviewed three-gate predecessor: the new helper and
        # three previous finality/continuation fixes are present, but this
        # classifier retains its actual547 preimage until the before proof.
        cursor.execute(predecessor)
    harness, _ = _quality_harness(database, tmp_path, contacts=contacts, duplicate=True)
    normal = harness.sandbox.run_icp
    failures = []

    def fail_last_challenger(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get('schema_version') == scoring.SCORING_INPUT_SCHEMA_VERSION:
            source = harness.service.store.get_run(document['scored_run_id'])
            participant = next(p for p in harness.service.store.get_round(harness.round_id)['participants']
                               if p['submission_id'] == source['submission_id'])
            if not participant['is_king'] and source['icp_position'] == 19:
                failures.append((source['run_id'], datetime.now(timezone.utc)))
                return runtime.fake_result(exit_code=0, output_bytes=json.dumps(
                    scoring.build_scoring_failure(document['scored_run_id'], 'judge_error')).encode())
        return normal(spec, **kwargs)

    harness.sandbox.run_icp = fail_last_challenger
    harness.service.config.defaults = replace(harness.service.config.defaults, per_icp_cost_policy=True)
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('stage2_scoring', runners=2)
    harness.run_stage_with_runners(2)  # Real first30-second SQL backoff.
    row = harness.service.store.get_round(harness.round_id)
    miner = next(p['submission_id'] for p in row['participants'] if not p['is_king'])
    sources = harness.service.store.list_runs(harness.round_id, kind='execute')
    judges = harness.service.store.list_runs(harness.round_id, kind='score')
    accepted_before = [r for r in judges if r['status'] == 'accepted']
    assert len(accepted_before) == 39
    affected = [r for r in judges if r['submission_id'] == miner and r['icp_position'] == 19]
    failed = sorted((r for r in affected if r['status'] == 'failed'), key=lambda r: r['attempt'])
    assert [r['attempt'] for r in failed] == [1, 2]
    assert all(r['terminal_cause'] == 'judge_error' for r in failed)
    assert len(failures) == 2 and failures[0][0] == failures[1][0]
    pending, = [r for r in affected if r['status'] == 'pending']
    assert pending['attempt'] == 3
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT created_at FROM public.lab_arena_runs WHERE run_id=%s', (failed[1]['run_id'],))
        second_created = cursor.fetchone()[0]
        second_started = failures[1][1]
        assert (second_started - second_created).total_seconds() >= 30
    # Seed only the deadline boundary, as540's reset fixture does. All accepted
    # sources, judgments, cache evidence, receipts and real failure rows remain
    # unaltered. The existing close/score/SQL publication paths run normally.
    deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
            "configuration_doc,'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s",
            (deadline.isoformat(), harness.round_id))
        cursor.execute('SET LOCAL session_replication_role=origin')
    harness.clock.now = deadline
    assert harness.service.close_scoring(harness.round_id, 2)['incomplete_assignments'] == 1
    assert harness.service.score_stage(harness.round_id, 2)['status'] == 'ok'
    target = next(r for r in accepted_before if r['submission_id'] == miner)
    with connect() as connection, connection.cursor() as cursor:
        assert _reason(cursor, harness, miner) is None
    with pytest.raises(ArenaStoreError, match='lab_arena_incomplete_participant_publication_invalid'):
        harness.service.publish(harness.round_id)
    with connect() as connection, connection.cursor() as cursor:
        before = _data_snapshot(cursor)
        cursor.execute(SQL.read_text())
        assert _data_snapshot(cursor) == before
        # The real deadline close marks the latest pending successor stage_closed;
        # preserve that existing reason rather than rewriting the earlier errors.
        assert _reason(cursor, harness, miner) == 'stage_closed'
        cursor.execute('SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),\'sha256\'),\'hex\')', (REASON,))
        assert cursor.fetchone()[0] == POST
        cursor.execute('BEGIN')
        # Corrupt an accepted member of the incomplete participant: a valid
        # missing judgment cannot hide malformed refs, source or score authority.
        for mutation in (
            "company_judgment_refs='[]'::jsonb",
            "company_judgment_refs=jsonb_set(company_judgment_refs,'{0,company_input_hash}',to_jsonb('sha256:'||repeat('0',64)))",
            "scoring_generation_floor=scoring_generation_floor+1",
            "scored_run_id='foreign-source'",
        ):
            cursor.execute('SAVEPOINT invalid_accepted')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute('UPDATE public.lab_arena_runs SET ' + mutation + ' WHERE run_id=%s', (target['run_id'],))
            cursor.execute('SET LOCAL session_replication_role=origin')
            assert _reason(cursor, harness, miner) is None, mutation
            cursor.execute('ROLLBACK TO SAVEPOINT invalid_accepted')
        cursor.execute('SAVEPOINT invalid_score')
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL WHERE run_id=%s', (target['scored_run_id'],))
        cursor.execute('SET LOCAL session_replication_role=origin')
        assert _reason(cursor, harness, miner) is None
        cursor.execute('ROLLBACK TO SAVEPOINT invalid_score')
    _negative_refs_matrix(connect, harness, target=target)
    assert harness.service.publish(harness.round_id)['status'] == 'ok'
    final = harness.service.store.get_round(harness.round_id)
    entry = next(r for r in final['publication_doc']['final_ranking'] if r['submission_id'] == miner)
    assert entry['final_score'] is None and entry['cost_summary'] is None
    assert entry['eligible'] is False and entry['eligibility_reason'] == 'execution_incomplete'
    assert entry['evaluation'] == 'not_evaluated' and entry['evaluation_reason'] == 'stage_closed'
    assert harness.service._baseline_fully_succeeded(final)
    assert final['reward_activated_at'] is None and final['reward_basis_doc'] is None
    accepted_after = [r for r in harness.service.store.list_runs(harness.round_id, kind='score') if r['status'] == 'accepted']
    assert accepted_after == accepted_before
    assert all(harness.service.store.get_run(r['run_id']) == r for r in failed)


@pytest.mark.parametrize('legacy_company', [False, True])
def test_existing_incomplete_full_group_and_legacy_company_authority_is_unchanged(database, tmp_path, legacy_company):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    if legacy_company:
        harness, _ = _quality_harness(database, tmp_path, legacy=True, duplicate=True)
    else:
        from tests.lab_arena.test_lab_arena_service_round import Harness
        harness = Harness(connect, tmp_path, challengers=['Clone'], runners=['alpha', 'beta'])
    harness.service.config.defaults = replace(harness.service.config.defaults, per_icp_cost_policy=True)
    _start_round(harness, day=29, epoch=42000)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('scored', runners=2)
    row = harness.service.store.get_round(harness.round_id)
    miner = next(p['submission_id'] for p in row['participants'] if not p['is_king'])
    source = max(harness.service.store.list_runs(harness.round_id, kind='execute', submission_id=miner),
                 key=lambda r: r['icp_position'])
    # Existing540 gap-fixture boundary; this compatibility control changes no
    # policy and compares the exact old classifier with the replacement.
    deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',"
            "output_ref=NULL,result_doc='{\"terminal_status\":\"judge_error\"}' WHERE scored_run_id=%s AND kind='score'",
            (source['run_id'],))
        cursor.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s', (source['run_id'],))
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
            "configuration_doc,'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s",
            (deadline.isoformat(), harness.round_id))
        cursor.execute('SET LOCAL session_replication_role=origin')
        assert _reason(cursor, harness, miner) == 'judge_error'
        before = _data_snapshot(cursor)
        cursor.execute(SQL.read_text())
        assert _data_snapshot(cursor) == before
        assert _reason(cursor, harness, miner) == 'judge_error'
    harness.clock.now = deadline
    assert harness.service.publish(harness.round_id)['status'] == 'ok'
