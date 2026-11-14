"""Explicit follow-up is transaction-local; native claim/finality stay bounded."""
import copy
import json
from datetime import datetime, timezone, timedelta

import pytest

from scripts import arena_approved_judge_followup as operator
from tests.lab_arena import native_timeout_recovery694_postgres_test as native
from tests.lab_arena import published_champion_correction695_postgres_test as published

old, base, prior = native.old, native.base, native.prior
database = published.database
prepared = published.prepared
prepared_company = published.prepared_company
case = published.case
base_extended = published.base_extended
extended = published.extended
provider_case = published.provider_case
plain_finite = published.plain_finite
before = published.before
post = published.post
prior_bug = published.prior_bug
budget_parent = published.budget_parent
corrected = published.corrected
first_published = published.published


@pytest.fixture
def republished(first_published):
    p = first_published
    first = published.args(p)
    child_id = published.recover(p, first)['grants'][0]['child_identity']['run_id']
    prior.claim_child(p, child_id)
    prior.complete_failed(p, child_id)
    prior.saved_parent(p, child_id)
    # Recreate an actual signed second publication in disposable fixtures only.
    second = published.published.__wrapped__(p)
    try:
        p = next(second)
        with p.c.cursor() as q:
            timestamp = old.scalar(q, 'SELECT published_at FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
            signer = published.signing.LocalSigner.generate()
            basis = published.signing.sign_document(signer, published.rewards.reward_basis_document(
                round_id=old.ROUND, published_at=timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'), finalized_epoch=1,
                king_outcome='no_king', king_hotkey=''), hash_field='reward_basis_hash')
            key = published.signing.signing_key_document(signer.public_key_der)
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute('UPDATE lab_arena_rounds SET effective_reward_epoch=2,reward_basis_doc=%s::jsonb,reward_basis_hash=%s,signing_key_doc=%s::jsonb WHERE round_id=%s',
                      (json.dumps(basis), basis['reward_basis_hash'], json.dumps(key), old.ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
            p.round_hash = old.scalar(q, "SELECT 'sha256:'||encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex') FROM lab_arena_rounds r WHERE round_id=%s", (old.ROUND,))
            for filename in ('685-lab-arena-owner-provider-judge-budget-proof.sql',
                             '686-lab-arena-owner-natural-provider-expiry.sql',
                             '689-lab-arena-interrupted-judge-response.sql',
                             '690-lab-arena-retained-proof-call-lookup.sql',
                             '691-lab-arena-provider-terminal-scorer-correction.sql',
                             '694-lab-arena-native-judge-timeout-recovery.sql'):
                q.execute(published.SQL.with_name(filename).read_text())
        p.c.commit()
        prior.mapping(p, 'sha256:' + 'e' * 64)
        yield p
    finally:
        p.c.rollback()
        second.close()


def approval(p):
    values = published.args(p, runtime='sha256:' + 'e' * 64,
        request='explicit-followup', approval='user:one-further-attempt')
    values['requests'][0]['request_id'] = 'explicit-followup-child'
    values.pop('dry')
    return values


def run(p, values, *, rollback=False):
    with p.c.cursor() as q:
        operator.install(q, values)
        result = operator.correct(q, values, dry_run=False)
    p.c.rollback() if rollback else p.c.commit()
    return result


def definitions(p):
    with p.c.cursor() as q:
        q.execute("SELECT oid::regprocedure::text,to_jsonb(p) FROM pg_proc p WHERE oid=ANY(%s::regprocedure[])",
                  ([operator.GRANT_SIGNATURE, operator.ROUND_SIGNATURE],))
        return q.fetchall()


def test_existing_controls_refuse_a_second_reopening(republished):
    p = republished
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='published authority'):
        published.recover(p, {**approval(p), 'dry': False})
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('terminal', ['accepted', 'judge_error', 'expired'])
def test_approved_followup_preserves_history_and_native_finite_lifecycle(republished, terminal):
    p = republished
    values = approval(p)
    rows, source = base.snapshot(p), definitions(p)
    if p.organizer:
        # Legacy organizer grants cannot silently turn into miner-funded spend.
        with pytest.raises(ValueError, match='preserve existing payer'):
            run(p, values)
        p.c.rollback()
        assert base.snapshot(p) == rows and definitions(p) == source
        return
    preview = run(p, values, rollback=True)
    assert base.snapshot(p) == rows and definitions(p) == source
    result = run(p, values)
    assert result == preview
    after = base.snapshot(p)
    old_rows = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in old_rows} == old_rows
    for table in set(rows) - {'lab_arena_rounds', 'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    assert len(after['lab_arena_runs']) == len(old_rows) + 1
    assert definitions(p) == source
    assert run(p, values)['idempotent'] is True
    assert base.snapshot(p) == after
    run_id = result['grants'][0]['child_identity']['run_id']
    assert old.funding(p, run_id)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, run_id, 'deepline')['funding_source'] == 'miner_key'
    lease = prior.claim_child(p, run_id)
    assert lease['company_judgment_cache']['misses'] == []
    if terminal == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (run_id,))
        p.c.commit()
    else:
        old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            run_id, old.TOKEN, json.dumps({'terminal_status': terminal,
              'failure_diagnostic': {'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'malformed_response'}}),
            terminal, run_id + '-output' if terminal == 'accepted' else None))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (p.parent['assignment_id'], p.parent['attempt'] + 1)) == 0
        history = old.scalar(q, "SELECT lab_arena_correction_reward_history_v1('live','finney',71,200,0,false)")
        assert len(history) >= 1
    p.c.commit()
    assert definitions(p) == source


@pytest.mark.parametrize('field', ['round', 'sha', 'runtime', 'final', 'publication', 'request', 'actor', 'approval', 'requests'])
def test_changed_authorization_is_refused_atomically(republished, field):
    p = republished
    values = approval(p)
    rows = base.snapshot(p)
    changed = copy.deepcopy(values)
    if field == 'requests':
        changed[field][0]['request_id'] += '-different'
    elif field in ('final', 'publication'):
        changed[field] = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    else:
        changed[field] += 'x'
    with pytest.raises(p.pg.Error, match='exact owner followup'):
        with p.c.cursor() as q:
            operator.install(q, values)
            operator.correct(q, changed, dry_run=False)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('fault', ['accepted', 'calls_pending', 'same_runtime', 'checkpoints', 'deadline', 'guard'])
def test_existing_validation_still_refuses_invalid_recovery(republished, fault):
    p = republished
    values = approval(p)
    if fault == 'checkpoints':
        values['requests'][0]['checkpoints'] = {}
    elif fault == 'deadline':
        values['final'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    elif fault == 'same_runtime':
        values['runtime'] = p.parent['reviewer_runtime_doc']['runtime_image_digest']
        prior.mapping(p, values['runtime'])
    else:
        with p.c.cursor() as q:
            if fault == 'guard':
                q.execute("UPDATE lab_arena_restart_claim_control SET operator_paused=true WHERE singleton")
            elif fault == 'calls_pending':
                base.old.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation', run_id=p.parent['run_id'],
                    round_id=old.ROUND, submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'],
                    stage=2, call_identity='sha256:'+'7'*64, provider='openrouter', operation_id='openrouter.chat',
                    funding_source='host' if p.organizer else 'miner_key', amount_microusd=500, entry_doc={}))
            else:
                q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
                q.execute('SET LOCAL session_replication_role=replica')
                q.execute("UPDATE lab_arena_runs SET status='accepted' WHERE run_id=%s", (p.parent['run_id'],))
                q.execute('SET LOCAL session_replication_role=origin')
                q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        p.c.commit()
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        run(p, values)
    p.c.rollback()
    assert base.snapshot(p) == rows

@pytest.mark.parametrize('state', ['committed', 'published'])
def test_newer_round_is_never_modified(republished, state):
    p = republished
    if p.organizer:
        return  # The independent payer-change control covers this legacy path.
    with p.c.cursor() as q:
        row = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (old.ROUND,))
        day = datetime.fromisoformat(row['evaluation_date']) + timedelta(days=1)
        cfg = copy.deepcopy(row['configuration_doc'])
        cfg['round_id'] = newer = 'arena-' + day.strftime('%Y-%m-%d') + '-active'
        q.execute('SET LOCAL session_replication_role=replica')
        old.insert(q, 'lab_arena_rounds', dict(round_id=newer, status=state, configuration_doc=cfg,
                   rewards_enabled=True, evaluation_date=day.strftime('%Y-%m-%d'), participants=[],
                   benchmark_ref='preserved-frozen-source'))
        q.execute('SET LOCAL session_replication_role=origin')
        before_newer = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (newer,))
    p.c.commit()
    try:
        values = approval(p)
        if state == 'published':
            with pytest.raises(p.pg.Error, match='published authority'):
                run(p, values)
            p.c.rollback()
        else:
            run(p, values)
        with p.c.cursor() as q:
            assert old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s', (newer,)) == before_newer
        p.c.commit()
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute('DELETE FROM lab_arena_rounds WHERE round_id=%s', (newer,))
        p.c.commit()


def test_unapproved_parent_and_duplicate_request_are_refused(republished):
    p = republished
    values = approval(p)
    rows = base.snapshot(p)
    values['requests'].append(copy.deepcopy(values['requests'][0]))
    with pytest.raises(p.pg.Error, match='parent/request list'):
        run(p, values)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_temp_helpers_do_not_grant_service_role_authority(republished):
    p = republished
    values = approval(p)
    with p.c.cursor() as q:
        operator.install(q, values)
        q.execute("SELECT has_function_privilege('lab_arena_service','pg_temp.approved_round_followup(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)','EXECUTE')")
        assert q.fetchone() == (False,)
        q.execute('SET LOCAL ROLE lab_arena_service')
        with pytest.raises(p.pg.errors.InsufficientPrivilege):
            operator.correct(q, values, dry_run=False)
    p.c.rollback()
