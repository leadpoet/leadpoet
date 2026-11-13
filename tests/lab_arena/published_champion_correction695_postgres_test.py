"""Published correction uses finite native children and preserves old authority."""
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.lab_arena import finite_child_scorer_correction679_postgres_test as child
from tests.lab_arena import postpublication_judgments678_postgres_test as postpub

SQL = Path(__file__).parents[2] / 'scripts/695-lab-arena-published-champion-correction.sql'
SIGNATURE = 'public.lab_arena_correct_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)'
CALL = 'lab_arena_correct_published_judgments_v1(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)'
old, base, prior = child.old, child.base, child.prior

# Reuse the native finite-budget lineage. Its local ``before`` fixture is the
# unpublished compatible fixture; publication is constructed only AFTER the
# failed budget child, not by using the historical678 published-round grant.
database = child.database
prepared = child.prepared
prepared_company = child.prepared_company
case = child.case
base_extended = child.base_extended
extended = child.extended
provider_case = child.provider_case
plain_finite = child.plain_finite
before = child.before
post = child.post
prior_bug = child.prior_bug
budget_parent = child.budget_parent
corrected = child.corrected


@pytest.fixture
def published(corrected):
    p = postpub.before.__wrapped__(corrected)
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE lab_arena_rounds SET promotion_doc=NULL,baseline_promoted_at=NULL WHERE round_id=%s', (old.ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
        p.round_hash = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex') FROM lab_arena_rounds r WHERE round_id=%s", (old.ROUND,))
        q.execute('SELECT to_regprocedure(%s)', (SIGNATURE,))
        existed = q.fetchone()[0]
        original = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,)) if existed else None
        q.execute(SQL.read_text())
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute(original if original else 'DROP FUNCTION ' + SIGNATURE)
        p.c.commit()


def args(p, **changes):
    values = postpub.args(p, runtime=child.RUNTIME, request='champion-correction',
                          approval='approval:reviewed-seven-corrections')
    values['requests'][0]['request_id'] = 'champion-correction-child'
    values.update(changes)
    return values


def recover(p, values):
    wire = copy.deepcopy(values)
    wire['requests'] = json.dumps(wire['requests'])
    with p.c.cursor() as q:
        result = old.scalar(q, 'SELECT public.' + CALL, tuple(wire.values()))
    p.c.commit()
    return result


def test_capability_is_absent_before_migration(corrected):
    with corrected.c.cursor() as q:
        assert old.scalar(q, 'SELECT to_regprocedure(%s)', (SIGNATURE,)) is None
    corrected.c.commit()


def test_atomic_preview_replay_archives_reward_and_preserves_every_old_row(published):
    p = published
    values = args(p)
    before_rows = base.snapshot(p)
    preview = recover(p, {**values, 'dry': True})
    assert preview['status'] == 'dry_run'
    assert base.snapshot(p) == before_rows
    result = recover(p, values)
    assert result['status'] == 'recovered' and len(result['grants']) == 1
    after = base.snapshot(p)
    original = before_rows['lab_arena_rounds'][0]
    round_row = after['lab_arena_rounds'][0]
    assert round_row['status'] == 'stage2_judged'
    for field in ['published_at', 'publication_doc', 'reward_basis_doc', 'reward_basis_hash',
                  'signing_key_doc', 'reward_activated_at', 'effective_reward_epoch',
                  'king_start_epoch', 'king_outcome', 'king_hotkey', 'promotion_doc', 'baseline_promoted_at']:
        assert round_row[field] is None
    for field in ['stage_generation', 'scoring_generation_floor', 'promotion_required', 'champion_hotkey',
                  'champion_submission_id', 'benchmark_ref']:
        assert round_row[field] == original[field]
    records = {r['run_id']: r for r in before_rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in records} == records
    assert len(after['lab_arena_runs']) == len(records) + 1
    for table in set(before_rows) - {'lab_arena_rounds', 'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == before_rows[table]
    journal = next(r for r in after['lab_arena_scoring_supersessions'] if r['request_id'] == values['request'])
    assert journal['old_round_doc'] == {k: v for k, v in original.items() if k != 'updated_at'}
    assert journal['recovery_doc']['recovery_kind'] == 'operator_postpublication_champion_correction_v1'
    assert journal['recovery_doc']['requests'] == values['requests']
    assert not any(r['recovery_doc'].get('recovery_kind') == 'operator_postpublication_judge_recovery_v1'
                   for r in after['lab_arena_scoring_supersessions'])
    assert recover(p, values)['idempotent'] is True
    assert base.snapshot(p) == after
    run_id = result['grants'][0]['child_identity']['run_id']
    assert old.funding(p, run_id)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, run_id, 'deepline')['funding_source'] == 'miner_key'
    prior.claim_child(p, run_id)
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT reviewer_runtime_doc->>'runtime_image_digest' FROM lab_arena_runs WHERE run_id=%s", (run_id,)) == child.RUNTIME
    p.c.commit()


@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_native_terminal_children_have_no_automatic_successor(published, outcome):
    p = published
    values = args(p)
    run_id = recover(p, values)['grants'][0]['child_identity']['run_id']
    prior.claim_child(p, run_id)
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (run_id,))
        p.c.commit()
    else:
        old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            run_id, old.TOKEN, json.dumps({'terminal_status': outcome, 'failure_diagnostic': {
                'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'malformed_response'}}),
            outcome, run_id + '-output' if outcome == 'accepted' else None))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    rows = base.snapshot(p)
    assert recover(p, values)['idempotent'] is True and base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (p.parent['assignment_id'], p.parent['attempt'] + 1)) == 0
        assert old.scalar(q, 'SELECT status FROM lab_arena_runs WHERE run_id=%s', (run_id,)) == ('accepted' if outcome == 'accepted' else 'failed')
    p.c.commit()


@pytest.mark.parametrize('fault', ['round_hash', 'runtime', 'parent_hash', 'checkpoints', 'signals',
                                  'deadline', 'duplicate', 'active', 'guard', 'winner', 'not_activated',
                                  'newer_frozen', 'promotion_done', 'frozen_source'])
def test_invalid_authority_is_atomic_without_reopening(published, fault):
    p = published
    values = args(p)
    if fault == 'round_hash':
        values['sha'] = 'sha256:' + '0' * 64
    elif fault == 'runtime':
        values['runtime'] = 'sha256:' + 'f' * 64
    elif fault == 'parent_hash':
        values['requests'][0]['parent_sha256'] = 'sha256:' + '0' * 64
    elif fault == 'checkpoints':
        values['requests'][0]['checkpoints'] = {}
    elif fault == 'signals':
        values['requests'][0]['signal_checkpoints'] = {'wrong': 'value'}
    elif fault == 'deadline':
        values['publication'] = values['final']
    elif fault == 'duplicate':
        values['requests'] *= 2
    else:
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            if fault == 'active':
                q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
                q.execute("UPDATE lab_arena_runs SET status='pending' WHERE run_id=%s", (p.original_parent['run_id'],))
                q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
            elif fault == 'guard':
                q.execute('UPDATE lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
            elif fault == 'winner':
                q.execute("UPDATE lab_arena_rounds SET king_outcome='crowned' WHERE round_id=%s", (old.ROUND,))
            elif fault == 'not_activated':
                q.execute('UPDATE lab_arena_rounds SET reward_activated_at=NULL WHERE round_id=%s', (old.ROUND,))
            elif fault == 'promotion_done':
                q.execute("UPDATE lab_arena_rounds SET promotion_doc='{}'::jsonb WHERE round_id=%s", (old.ROUND,))
            elif fault == 'newer_frozen':
                row = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (old.ROUND,))
                day = datetime.fromisoformat(row['evaluation_date']) + timedelta(days=1)
                cfg = copy.deepcopy(row['configuration_doc'])
                cfg['round_id'] = rid = 'arena-' + day.strftime('%Y-%m-%d') + '-frozen'
                old.insert(q, 'lab_arena_rounds', dict(round_id=rid, status='committed', configuration_doc=cfg,
                           rewards_enabled=True, evaluation_date=day.strftime('%Y-%m-%d'), participants=[]))
            elif fault == 'frozen_source':
                q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
                q.execute("UPDATE lab_arena_runs SET judgment_input_hash='sha256:'||repeat('0',64) WHERE run_id=%s", (p.parent['run_id'],))
                q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
                values['requests'][0]['parent_sha256'] = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (p.parent['run_id'],))
            q.execute('SET LOCAL session_replication_role=origin')
            values['sha'] = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex') FROM lab_arena_rounds r WHERE round_id=%s", (old.ROUND,))
        p.c.commit()
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        recover(p, values)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_service_role_cannot_open_correction_and_migration_replays(published):
    p = published
    values = args(p)
    wire = copy.deepcopy(values)
    wire['requests'] = json.dumps(wire['requests'])
    with pytest.raises(p.pg.errors.InsufficientPrivilege):
        old.rpc(p, CALL, tuple(wire.values()))
    rows = base.snapshot(p)
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('field', ['approval', 'runtime', 'requests'])
def test_replay_cannot_change_approved_authority(published, field):
    p = published
    values = args(p)
    recover(p, values)
    rows = base.snapshot(p)
    changed = copy.deepcopy(values)
    if field == 'requests':
        changed[field][0]['checkpoints'] = {}
    else:
        changed[field] = 'changed' if field == 'approval' else 'sha256:' + '0' * 64
    with pytest.raises(p.pg.Error, match='replay differs'):
        recover(p, changed)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_new_request_cannot_open_a_second_published_correction(published):
    p = published
    recover(p, args(p))
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        recover(p, args(p, request='second-owner-correction'))
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_batch_bad_second_parent_rolls_back_every_row(published):
    p = published
    values = args(p)
    second = copy.deepcopy(values['requests'][0])
    second.update(parent_run_id='missing-parent', request_id='second-correction-child')
    values['requests'].append(second)
    rows = base.snapshot(p)
    for dry in [True, False]:
        with pytest.raises(p.pg.Error):
            recover(p, {**values, 'dry': dry})
        p.c.rollback()
        assert base.snapshot(p) == rows


def test_correction_has_no_old_reward_seal_or_baseline_freshness_exemption(published):
    p = published
    result = recover(p, args(p))
    run_id = result['grants'][0]['child_identity']['run_id']
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT public.lab_arena__postpublication_receipt_v1(h) FROM lab_arena_scoring_supersessions h WHERE request_id='champion-correction'") is False
        assert old.scalar(q, 'SELECT public.lab_arena__postpublication_judge_v1(r,a) FROM lab_arena_rounds r JOIN lab_arena_runs a USING(round_id) WHERE a.run_id=%s', (run_id,)) is False
        row = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (old.ROUND,))
        day = datetime.fromisoformat(row['evaluation_date']) + timedelta(days=1)
        cfg = copy.deepcopy(row['configuration_doc'])
        cfg['round_id'] = rid = 'arena-' + day.strftime('%Y-%m-%d') + '-pending'
        cfg['reasoning_capture_policy'] = 'full_generated_thinking_v1'
        q.execute('SET LOCAL session_replication_role=replica')
        old.insert(q, 'lab_arena_rounds', dict(round_id=rid, status='committed', configuration_doc=cfg,
                   rewards_enabled=True, evaluation_date=day.strftime('%Y-%m-%d'), participants=[]))
        q.execute('SET LOCAL session_replication_role=origin')
        assert old.scalar(q, 'SELECT public.lab_arena__baseline_source_current_v1(r) FROM lab_arena_rounds r WHERE round_id=%s', (rid,)) is False
    p.c.commit()


def test_unfrozen_next_round_does_not_block_and_is_unchanged(published):
    p = published
    with p.c.cursor() as q:
        row = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (old.ROUND,))
        day = datetime.fromisoformat(row['evaluation_date']) + timedelta(days=1)
        cfg = copy.deepcopy(row['configuration_doc'])
        cfg['round_id'] = rid = 'arena-' + day.strftime('%Y-%m-%d') + '-open'
        q.execute('SET LOCAL session_replication_role=replica')
        old.insert(q, 'lab_arena_rounds', dict(round_id=rid, status='open', configuration_doc=cfg,
                   rewards_enabled=True, evaluation_date=day.strftime('%Y-%m-%d'), participants=[]))
        q.execute('SET LOCAL session_replication_role=origin')
        original = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (rid,))
    p.c.commit()
    recover(p, args(p))
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (rid,)) == original
    p.c.commit()


@pytest.mark.parametrize('prior_bug', [True], indirect=True)
def test_earlier_scorer_bug_cannot_receive_another_after_budget(published):
    p = published
    rows = base.snapshot(p)
    for dry in [True, False]:
        with pytest.raises(p.pg.Error):
            recover(p, args(p, dry=dry))
        p.c.rollback()
        assert base.snapshot(p) == rows
