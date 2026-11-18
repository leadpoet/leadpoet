"""A published round can finish an owner-approved, unchanged-scorer judgment."""
import copy
import json
from pathlib import Path

import pytest

from tests.lab_arena import published_champion_correction695_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/700-lab-arena-published-same-policy-recovery.sql'
database = prior.database
prepared = prior.prepared
prepared_company = prior.prepared_company
case = prior.case
base_extended = prior.base_extended
extended = prior.extended
provider_case = prior.provider_case
plain_finite = prior.plain_finite
same_policy = prior.child.f.compatible_plain
published = prior.published


@pytest.fixture
def corrected(same_policy):
    prior.old.release_guard(same_policy)
    return same_policy


def arguments(p):
    with p.c.cursor() as q:
        runtime = prior.old.scalar(q, "SELECT runtime_image_digest FROM lab_arena_reviewer_runtimes WHERE grading_image_digest=(SELECT configuration_doc->>'scorer_image_digest' FROM lab_arena_rounds WHERE round_id=%s)", (prior.old.ROUND,))
    p.c.commit()
    return prior.args(p, runtime=runtime, approval='approval:one-provider-recovery-attempt')


@pytest.fixture
def updated(published):
    p = published
    with p.c.cursor() as q:
        original = prior.old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (prior.SIGNATURE,))
        q.execute(SQL.read_text())
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute(original)
        p.c.commit()


def test_existing_entrypoint_refuses_unchanged_scorer(published):
    p = published
    before = prior.base.snapshot(p)
    with pytest.raises(p.pg.Error, match='exact parent differs'):
        prior.recover(p, arguments(p))
    p.c.rollback()
    assert prior.base.snapshot(p) == before


def test_unchanged_scorer_preserves_results_and_reassesses_champion(updated):
    p = updated
    values = arguments(p)
    before = prior.base.snapshot(p)
    preview = prior.recover(p, {**values, 'dry': True})
    assert preview['status'] == 'dry_run'
    assert prior.base.snapshot(p) == before
    result = prior.recover(p, values)
    assert result['status'] == 'recovered'
    after = prior.base.snapshot(p)
    old_runs = {row['run_id']: row for row in before['lab_arena_runs']}
    assert {row['run_id']: row for row in after['lab_arena_runs'] if row['run_id'] in old_runs} == old_runs
    assert len(after['lab_arena_runs']) == len(old_runs) + 1
    for table in set(before) - {'lab_arena_rounds', 'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == before[table]
    round_row = after['lab_arena_rounds'][0]
    assert round_row['status'] == 'stage2_judged'
    assert round_row['reward_basis_doc'] is None
    assert round_row['king_outcome'] is None
    archive = next(row for row in after['lab_arena_scoring_supersessions'] if row['request_id'] == values['request'])
    assert archive['old_round_doc']['reward_basis_doc'] == before['lab_arena_rounds'][0]['reward_basis_doc']
    grant = result['grants'][0]
    grant_proof = next(row['recovery_doc'] for row in after['lab_arena_scoring_supersessions']
                       if row['request_id'] == values['requests'][0]['request_id'])
    assert grant_proof['recovery_kind'] == 'operator_judge_budget_exception_v1'
    assert grant_proof['approval_ref'] == values['approval']
    assert grant['checkpoints'] == p.checkpoints
    assert grant['signal_checkpoints'] == p.signal_checkpoints
    assert prior.recover(p, values)['idempotent'] is True
    assert prior.base.snapshot(p) == after
    run_id = grant['child_identity']['run_id']
    assert prior.old.funding(p, run_id)['funding_source'] == ('host' if p.organizer else 'miner_key')
    prior.prior.claim_child(p, run_id)
    with p.c.cursor() as q:
        assert prior.old.scalar(q, "SELECT reviewer_runtime_doc->>'runtime_image_digest' FROM lab_arena_runs WHERE run_id=%s", (run_id,)) == values['runtime']
    p.c.commit()


@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_same_policy_child_remains_bounded_and_replay_does_not_charge_again(updated, outcome):
    p = updated
    values = arguments(p)
    run_id = prior.recover(p, values)['grants'][0]['child_identity']['run_id']
    prior.prior.claim_child(p, run_id)
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (run_id,))
        p.c.commit()
    else:
        prior.old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            run_id, prior.old.TOKEN, json.dumps({'terminal_status': outcome, 'failure_diagnostic': {
                'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'provider_error'}}),
            outcome, run_id + '-output' if outcome == 'accepted' else None))
    prior.old.rpc(p, 'lab_arena_expire_leases(%s)', (prior.old.ROUND,))
    rows = prior.base.snapshot(p)
    assert prior.recover(p, values)['idempotent'] is True
    assert prior.base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert prior.old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                                (p.parent['assignment_id'], p.parent['attempt'] + 1)) == 0
    p.c.commit()
    prior.prior.saved_parent(p, run_id)
    with pytest.raises(p.pg.Error):
        prior.child.f.old.grant(p, runtime=values['runtime'], request_id='no-second-budget')
    p.c.rollback()
    assert prior.base.snapshot(p) == rows


@pytest.mark.parametrize('fault', ['parent_hash', 'checkpoints', 'runtime', 'approval', 'guard', 'pending_call', 'replay_request'])
def test_same_policy_recovery_keeps_exact_safety_checks(updated, fault):
    p = updated
    values = arguments(p)
    if fault == 'parent_hash':
        values['requests'][0]['parent_sha256'] = 'sha256:' + '0' * 64
    elif fault == 'checkpoints':
        values['requests'][0]['checkpoints'] = {}
    elif fault == 'runtime':
        values['runtime'] = 'sha256:' + '0' * 64
    elif fault == 'approval':
        values['approval'] = ''
    elif fault == 'guard':
        with p.c.cursor() as q:
            q.execute('UPDATE lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
        p.c.commit()
    elif fault == 'pending_call':
        with p.c.cursor() as q:
            prior.old.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation',
                run_id=p.parent['run_id'], round_id=prior.old.ROUND,
                submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'],
                stage=2, call_identity='sha256:' + '7' * 64, provider='openrouter',
                operation_id='openrouter.chat', funding_source='host' if p.organizer else 'miner_key',
                amount_microusd=500, entry_doc={}))
        p.c.commit()
    else:
        prior.recover(p, values)
        values['requests'][0]['request_id'] = 'different-child-request'
    rows = prior.base.snapshot(p)
    with pytest.raises(p.pg.Error):
        prior.recover(p, values)
    p.c.rollback()
    assert prior.base.snapshot(p) == rows


def test_service_role_and_unexpected_function_state_stay_refused(updated):
    p = updated
    values = arguments(p)
    wire = copy.deepcopy(values)
    wire['requests'] = json.dumps(wire['requests'])
    with pytest.raises(p.pg.errors.InsufficientPrivilege):
        prior.old.rpc(p, prior.CALL, tuple(wire.values()))
    rows = prior.base.snapshot(p)
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()
    assert prior.base.snapshot(p) == rows
    with p.c.cursor() as q:
        definition = prior.old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (prior.SIGNATURE,))
        source = prior.old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (prior.SIGNATURE,))
        q.execute(definition.replace(source, source + '\n-- unknown source'))
        with pytest.raises(p.pg.Error, match='preimage differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    assert prior.base.snapshot(p) == rows
