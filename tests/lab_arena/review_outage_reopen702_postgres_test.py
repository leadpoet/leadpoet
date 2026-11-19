"""Native PostgreSQL proof for the owner-run review-outage reopen (702).

Runs on the review lineage whose begin/finish, Nov 6 hook and frozen trigger
match production (660/669 verified hashes), so 702 applies with its real pins.
"""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import code_review, rewards, signing
from lab_arena.store import ArenaStore, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission669_postgres_test as alternate
from tests.lab_arena.refusal_review_fallback599_postgres_test import failure, verdict
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '702-lab-arena-review-outage-reopen.sql'
DS = code_review.PROSPECTIVE_REVIEW_MODEL
KIMI = alternate.KIMI
REOPEN = ('public.lab_arena_reopen_review_outage_v1(text,text,jsonb,timestamptz,timestamptz,'
          'timestamptz,timestamptz,text,text,text,boolean)')
ADMIT = 'public.lab_arena_admit_review_outage_v1(text,text,text,text,boolean)'
HOOK = 'public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'
FROZEN = 'public.lab_arena_submissions_frozen_v1()'
READER = 'public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)'
REOPEN_CALL = 'SELECT public.lab_arena_reopen_review_outage_v1(%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)'
ADMIT_CALL = 'SELECT public.lab_arena_admit_review_outage_v1(%s,%s,%s,%s,%s)'


def definitions(cursor):
    out = {}
    for signature in (HOOK, FROZEN, READER, REOPEN, ADMIT):
        cursor.execute('SELECT to_regprocedure(%s)', (signature,))
        oid = cursor.fetchone()[0]
        if oid is None:
            out[signature] = None
            continue
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure), proacl::text FROM pg_proc WHERE oid=%s::regprocedure',
                       (signature, signature))
        out[signature] = cursor.fetchone()
    return out


@pytest.fixture(scope='module')
def recovery_database(request):
    generator = alternate.recovery_database.__wrapped__()
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                before = definitions(cursor)
                assert before[REOPEN] is None and before[ADMIT] is None
                hook_before = before[HOOK][0]
                cursor.execute(MIGRATION.read_text())
                after = definitions(cursor)
                # The migration is replayable and changes nothing the second time.
                cursor.execute(MIGRATION.read_text())
                assert definitions(cursor) == after
                request.config.cache.set('review702/hook_before', hook_before)
        yield pg, dsn
    finally:
        generator.close()


@pytest.fixture()
def db(recovery_database):
    pg, dsn = recovery_database
    connection = pg.connect(**dsn)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    store = ArenaStore(transport)
    try:
        yield pg, connection, store
    finally:
        transport.close()
        connection.close()


def scalar(connection, sql, params=()):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchone()[0]


def iso(moment):
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def make_round(store, connection, tag, *, count=2, error='code_review_provider_unavailable', network=None,
               cutoff_minutes=5):
    """A published no-king round whose challengers all failed review."""
    now = datetime.now(timezone.utc)
    network = network or 'review702-' + tag
    round_id = 'arena-2099-04-01-' + hashlib.sha256(tag.encode()).hexdigest()[:12]
    config = round_config(round_id, [hotkey('runner702-' + tag)], rewards_enabled=True)
    config.update(network_name=network, netuid=71, execution_sequence_policy='baseline_scored_first_v1',
                  max_challengers=15)
    config['schedule'].update({
        'submission_open': iso(now - timedelta(hours=2)),
        'submission_cutoff': iso(now + timedelta(minutes=cutoff_minutes)),
        'benchmark_deadline': iso(now + timedelta(minutes=cutoff_minutes + 60)),
        'stage_2_start': iso(now - timedelta(hours=1)), 'stage_2_close': iso(now - timedelta(minutes=50)),
        'final_scoring_close': iso(now - timedelta(minutes=40)),
        'publication_deadline': iso(now - timedelta(minutes=39)),
    })
    assert store.create_round(round_id, config)['status'] == 'created'
    sids = []
    for index in range(count):
        sid = '%s-entrant-%d' % (tag, index)
        miner = hotkey('miner702-%s-%d' % (tag, index))
        assert store.register_submission(round_id, sid, miner, source_submission_doc(round_id, sid))['status'] == 'registered'
        assert store.accept_submission_with_credentials(round_id, sid, miner,
                                                        encrypted_runtime_credentials(sid))['status'] == 'ok'
        for _ in range(2):
            token = new_lease_token()
            claim = alternate.begin(store, sid, miner, DS, token)
            assert claim['status'] == 'claimed', claim
            store.finish_submission_review(sid, miner, token, 'error', failure(DS, error, retryable=False), 1234)
            _age_review_error(connection, sid)
        sids.append((sid, miner))
    published = now.replace(microsecond=0) - timedelta(minutes=30)
    signer = signing.LocalSigner.generate()
    basis = signing.sign_document(signer, rewards.reward_basis_document(
        round_id=round_id, published_at=iso(published), finalized_epoch=26482,
        king_outcome='no_king', king_hotkey=''), hash_field='reward_basis_hash')
    key = signing.signing_key_document(signer.public_key_der)
    baseline_id = 'baseline-' + round_id.removeprefix('arena-')
    baseline = {'submission_id': baseline_id, 'miner_hotkey': config['baseline_hotkey'], 'is_king': True,
                'source_ref': 'baseline', 'source_size_bytes': 1}
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute(
            "INSERT INTO lab_arena_submissions SELECT (jsonb_populate_record(NULL::lab_arena_submissions,"
            " to_jsonb(s)||jsonb_build_object('submission_id',%s,'miner_hotkey',%s,'is_king',true,'status','frozen',"
            " 'frozen_at',clock_timestamp(),'code_review_status','pending','code_review_doc',NULL,'code_review_attempts',0,"
            " 'code_review_claim',NULL,'code_review_started_at',NULL,'code_review_expires_at',NULL))).* FROM lab_arena_submissions s WHERE submission_id=%s",
            (baseline_id, config['baseline_hotkey'], sids[0][0]))
        cursor.execute('SELECT source_ref, source_size_bytes FROM lab_arena_submissions WHERE submission_id=%s', (baseline_id,))
        baseline['source_ref'], baseline['source_size_bytes'] = cursor.fetchone()
        cursor.execute("UPDATE lab_arena_submissions SET status='rejected',rejection_rule='code_review_incomplete'"
                       " WHERE round_id=%s AND NOT is_king", (round_id,))
        cursor.execute(
            "UPDATE lab_arena_rounds SET status='published',status_generation=status_generation+9,evaluation_date=%s,"
            " participants=%s::jsonb,publication_doc=%s::jsonb,king_outcome='no_king',published_at=%s,"
            " reward_basis_doc=%s::jsonb,reward_basis_hash=%s,signing_key_doc=%s::jsonb,reward_activated_at=%s,"
            " effective_reward_epoch=%s,king_start_epoch=%s,stage1_scoring_plan_doc='{\"work_items\":[]}'::jsonb,"
            " stage2_scoring_plan_doc='{\"work_items\":[],\"zero_rows\":[]}'::jsonb,finalists='[]'::jsonb WHERE round_id=%s",
            ('2099-04-01', json.dumps([baseline]),
             json.dumps({'king_decision': {'outcome': 'no_king', 'king_hotkey': '', 'winner_submission_id': ''}}),
             published, json.dumps(basis), basis['reward_basis_hash'], json.dumps(key), published,
             basis['effective_reward_epoch'], basis['effective_reward_epoch'], round_id))
        cursor.execute('COMMIT')
    return round_id, sids, basis


def round_hash(connection, round_id):
    return 'sha256:' + scalar(connection, "SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex')"
                              " FROM lab_arena_rounds r WHERE round_id=%s", (round_id,))


def row_hashes(connection, round_id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT jsonb_agg(jsonb_build_object('submission_id',submission_id,'row_sha256','sha256:'||"
                       "encode(extensions.digest((to_jsonb(s)-'updated_at')::text,'sha256'),'hex')) ORDER BY submission_id)"
                       " FROM lab_arena_submissions s WHERE round_id=%s AND status='rejected'", (round_id,))
        return cursor.fetchone()[0]


def deadlines(review_seconds=1800):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    review = now + timedelta(seconds=review_seconds)
    return review, review + timedelta(hours=3), review + timedelta(hours=4), review + timedelta(hours=4, seconds=60)


def reopen(connection, round_id, *, hashes='auto', digest='auto', times=None, request='reopen-1',
           actor='operator:review702', approval='approval:review702', dry=False):
    times = times or deadlines()
    hashes = row_hashes(connection, round_id) if hashes == 'auto' else hashes
    digest = round_hash(connection, round_id) if digest == 'auto' else digest
    return scalar(connection, REOPEN_CALL, (round_id, digest, None if hashes is None else json.dumps(hashes),
                                            *times, request, actor, approval, dry))


def snapshot(connection, round_id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_jsonb(r)-'updated_at' FROM lab_arena_rounds r WHERE round_id=%s", (round_id,))
        rnd = cursor.fetchone()[0]
        cursor.execute("SELECT jsonb_agg(to_jsonb(s)-'updated_at' ORDER BY submission_id) FROM lab_arena_submissions s"
                       " WHERE round_id=%s", (round_id,))
        subs = cursor.fetchone()[0]
        cursor.execute('SELECT count(*) FROM lab_arena_scoring_supersessions WHERE round_id=%s', (round_id,))
        return rnd, subs, cursor.fetchone()[0]


def test_hook_keeps_nov06_body_and_adds_only_the_receipt_fallback(db, request):
    _, connection, _ = db
    before = request.config.cache.get('review702/hook_before', None)
    with connection.cursor() as cursor:
        after = definitions(cursor)[HOOK][0]
    body = before.split('AS $function$', 1)[1].rsplit('$function$', 1)[0].rstrip(' \n')[:-1]
    assert body in after
    assert 'lab_arena__review_outage_review_deadline_v1(p_round,p_submission)' in after
    assert after.count('SELECT coalesce(') == 1


def test_dry_run_without_hashes_reports_exact_changes_and_changes_nothing(db):
    _, connection, store = db
    round_id, sids, _ = make_round(store, connection, 'dry')
    before = snapshot(connection, round_id)
    result = reopen(connection, round_id, hashes=None, digest=None, dry=True)
    assert snapshot(connection, round_id) == before
    assert result['status'] == 'dry_run'
    assert result['round_sha256'] == round_hash(connection, round_id)
    assert result['submissions'] == row_hashes(connection, round_id)
    assert {item['submission_id'] for item in result['submission_changes']} == {sid for sid, _ in sids}
    changes = result['round_changes']
    assert changes['status'] == {'old': 'published', 'new': 'stage1_scored'}
    assert set(changes) == {'status', 'status_generation', 'configuration_doc', 'published_at', 'publication_doc',
                            'reward_basis_doc', 'reward_basis_hash', 'signing_key_doc', 'reward_activated_at',
                            'effective_reward_epoch', 'king_start_epoch', 'king_outcome',
                            'stage2_scoring_plan_doc', 'finalists'}
    assert 'participants' not in changes and 'stage1_scoring_plan_doc' not in changes


def test_reopen_review_admit_runs_normal_competition_and_keeps_archived_authority(db):
    _, connection, store = db
    round_id, sids, basis = make_round(store, connection, 'full', count=3)
    times = deadlines()
    result = reopen(connection, round_id, times=times)
    assert result['status'] == 'reopened' and result['idempotent'] is False
    row = store.get_round(round_id)
    assert row['status'] == 'stage1_scored' and row['published_at'] is None and row['reward_basis_hash'] is None
    assert row['participants'] == [p for p in row['participants'] if p['is_king']]
    assert row['configuration_doc']['schedule']['stage_2_start'] == iso(times[0])
    assert row['configuration_doc']['schedule']['publication_deadline'] == iso(times[3])
    # The archived signed no-king basis stays the reward authority during the rerun.
    history = scalar(connection, 'SELECT public.lab_arena_correction_reward_history_v1(%s,%s,%s,200,0,false)',
                     ('live', 'review702-full', 71))
    assert [(h['round_id'], h['reward_basis_hash'], h['effective_reward_epoch']) for h in history] == [
        (round_id, basis['reward_basis_hash'], basis['effective_reward_epoch'])]
    # Exact replay is idempotent; a changed replay is refused.
    replay = reopen(connection, round_id, hashes=result['submissions'],
                    digest=scalar(connection, "SELECT recovery_doc->>'round_sha256' FROM lab_arena_scoring_supersessions"
                                  " WHERE round_id=%s AND request_id='reopen-1'", (round_id,)), times=times)
    assert replay['idempotent'] is True
    with pytest.raises(Exception, match='replay differs'):
        reopen(connection, round_id, hashes=result['submissions'], digest=replay and 'sha256:' + '0' * 64, times=times)
    # Driver cannot open stage 2 before admission.
    with pytest.raises(Exception, match='lab_arena_review_outage_admission_pending'):
        with connection.cursor() as cursor:
            cursor.execute("UPDATE lab_arena_rounds SET status='stage2' WHERE round_id=%s", (round_id,))
    with pytest.raises(Exception, match='review still pending'):
        scalar(connection, ADMIT_CALL, (round_id, 'reopen-1', 'admit-1', 'operator:review702', True))
    # Reviews run again on the reopened round through the unchanged begin/finish.
    outcomes = ['passed', 'rejected', 'passed']
    for (sid, miner), outcome in zip(sids, outcomes):
        token = new_lease_token()
        claim = alternate.begin(store, sid, miner, DS, token)
        assert claim['status'] == 'claimed', claim
        document = verdict(DS) if outcome == 'passed' else {**verdict(DS), 'passed': False, 'verdict': 'reject',
            'categories': ['fabricated_evidence']}
        assert store.finish_submission_review(sid, miner, token, outcome, document, 1234)['status'] == outcome
    preview = scalar(connection, ADMIT_CALL, (round_id, 'reopen-1', 'admit-1', 'operator:review702', True))
    assert preview['status'] == 'dry_run' and store.get_round(round_id)['participants'] == row['participants']
    admitted = scalar(connection, ADMIT_CALL, (round_id, 'reopen-1', 'admit-1', 'operator:review702', False))
    assert admitted['admitted'] == [sids[0][0], sids[2][0]]
    assert admitted['excluded'] == {sids[1][0]: 'code_review_rejected'}
    participants = store.get_round(round_id)['participants']
    assert [p['submission_id'] for p in participants[1:]] == [sids[0][0], sids[2][0]]
    assert all(set(p) == {'submission_id', 'miner_hotkey', 'source_ref', 'source_size_bytes', 'is_king'}
               for p in participants)
    statuses = {sid: store.get_submission(sid)['status'] for sid, _ in sids}
    assert statuses == {sids[0][0]: 'frozen', sids[1][0]: 'rejected', sids[2][0]: 'frozen'}
    assert scalar(connection, ADMIT_CALL, (round_id, 'reopen-1', 'admit-1', 'operator:review702', False))['idempotent']
    # The hold is released; the hook no longer opens a review window.
    assert scalar(connection, 'SELECT (public.lab_arena__review_outage_open_receipt_v1(%s)).round_id', (round_id,)) is None
    assert alternate.begin(store, sids[0][0], sids[0][1], DS)['status'] == 'existing'
    with pytest.raises(Exception, match='submission_missing'):
        alternate.begin(store, sids[1][0], sids[1][1], DS)


def test_hold_expires_at_review_deadline_and_cancel_is_never_held(db):
    _, connection, store = db
    round_id, _, _ = make_round(store, connection, 'expiry', count=1)
    review, close, final, publication = deadlines(3)
    reopen(connection, round_id, times=(review, close, final, publication))
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        cursor.execute("UPDATE lab_arena_rounds SET status='cancelled',cancel_reason='operator' WHERE round_id=%s", (round_id,))
        cursor.execute('ROLLBACK')
    time.sleep(4)
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        try:
            cursor.execute("UPDATE lab_arena_rounds SET status='stage2' WHERE round_id=%s", (round_id,))
        except Exception as exc:  # Other existing guards may refuse; the hold must not.
            assert 'admission_pending' not in str(exc)
        finally:
            cursor.execute('ROLLBACK')
    admitted = scalar(connection, ADMIT_CALL, (round_id, 'reopen-1', 'admit-1', 'operator:review702', False))
    assert admitted['admitted'] == [] and list(admitted['excluded'].values()) == ['code_review_incomplete']


def test_unrelated_rounds_are_not_held_or_given_a_review_window(db):
    _, connection, store = db
    round_id, sids, _ = make_round(store, connection, 'unrelated', count=1)
    assert scalar(connection, 'SELECT (public.lab_arena__review_outage_open_receipt_v1(%s)).round_id', (round_id,)) is None
    assert scalar(connection, 'SELECT public.lab_arena__nov06_late_review635_v1(r,s) FROM lab_arena_rounds r,'
                  ' lab_arena_submissions s WHERE r.round_id=%s AND s.submission_id=%s', (round_id, sids[0][0])) is None
    with pytest.raises(Exception, match='lab_arena_submissions_frozen|frozen|rejected'):
        with connection.cursor() as cursor:
            cursor.execute("UPDATE lab_arena_submissions SET status='accepted',rejection_rule=NULL WHERE submission_id=%s",
                           (sids[0][0],))


def damage(connection, round_id, sql, params=()):
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute(sql, params)
        cursor.execute('COMMIT')


FAULTS = {
    'round_hash': (None, 'round hash differs'),
    'crowned': ("UPDATE lab_arena_rounds SET publication_doc=jsonb_set(publication_doc,'{king_decision,outcome}','\"crowned\"'),"
                "king_outcome='crowned' WHERE round_id=%s", 'published no-king authority'),
    'not_published': ("UPDATE lab_arena_rounds SET status='scored' WHERE round_id=%s", 'published no-king authority'),
    'parallel': ("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||'{\"parallel_twenty_icp_execution\":true}'"
                 " WHERE round_id=%s", 'published no-king authority'),
    'extra_participant': ("UPDATE lab_arena_rounds SET participants=participants||'[{\"submission_id\":\"x\",\"is_king\":false}]'"
                          " WHERE round_id=%s", 'baseline-only'),
    'stage2_run': ("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,"
                   "status,kind,attempt,scoring_generation_floor) SELECT 'x702-'||round_id,'x702-'||round_id,round_id,"
                   "participants->0->>'submission_id',participants->0->>'miner_hotkey',2,0,'accepted','execute',1,0"
                   " FROM lab_arena_rounds WHERE round_id=%s", 'baseline-only'),
    'paused': ("UPDATE lab_arena_restart_claim_control SET operator_paused=true WHERE singleton AND %s IS NOT NULL", 'paused'),
}


@pytest.mark.parametrize('fault', sorted(FAULTS))
def test_refusals_change_nothing(db, fault):
    _, connection, store = db
    round_id, _, _ = make_round(store, connection, 'fault-' + fault, count=1)
    sql, message = FAULTS[fault]
    digest = 'sha256:' + '1' * 64 if fault == 'round_hash' else 'auto'
    if sql:
        damage(connection, round_id, sql, (round_id,))
    before = snapshot(connection, round_id)
    try:
        for dry in (True, False):
            with pytest.raises(Exception, match=message):
                reopen(connection, round_id, digest=digest, dry=dry)
            assert snapshot(connection, round_id) == before
    finally:
        if fault == 'paused':
            damage(connection, round_id, 'UPDATE lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')


def test_list_must_be_every_provider_outage_rejection_with_unchanged_rows(db):
    _, connection, store = db
    round_id, _, _ = make_round(store, connection, 'partial', count=2)
    hashes = row_hashes(connection, round_id)
    before = snapshot(connection, round_id)
    with pytest.raises(Exception, match='listed submissions differ'):
        reopen(connection, round_id, hashes=hashes[:1])
    changed = [dict(hashes[0], row_sha256='sha256:' + '2' * 64)] + hashes[1:]
    with pytest.raises(Exception, match='listed submissions differ'):
        reopen(connection, round_id, hashes=changed)
    assert snapshot(connection, round_id) == before


def test_other_rejection_codes_are_not_reopened(db):
    _, connection, store = db
    round_id, _, _ = make_round(store, connection, 'othercode', count=1, error='code_review_provider_refused')
    with pytest.raises(Exception, match='absent or above capacity'):
        reopen(connection, round_id, hashes=None, digest=None, dry=True)


def test_next_round_must_not_be_frozen_and_bounds_publication(db):
    _, connection, store = db
    round_id, _, _ = make_round(store, connection, 'next', count=1)
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    config = round_config('arena-2099-04-02-next702', [hotkey('runner702-next2')], rewards_enabled=True)
    config.update(network_name='review702-next', netuid=71)
    config['schedule']['submission_cutoff'] = iso(later)
    assert store.create_round('arena-2099-04-02-next702', config)['status'] == 'created'
    damage(connection, round_id, "UPDATE lab_arena_rounds SET evaluation_date='2099-04-02' WHERE round_id=%s",
           ('arena-2099-04-02-next702',))
    with pytest.raises(Exception, match='deadline invalid'):
        reopen(connection, round_id, dry=True)  # publication 4h later is after the next cutoff
    now = datetime.now(timezone.utc).replace(microsecond=0)
    times = (now + timedelta(minutes=10), now + timedelta(minutes=40), now + timedelta(minutes=60),
             now + timedelta(minutes=61))
    assert reopen(connection, round_id, times=times, dry=True)['next_round_id'] == 'arena-2099-04-02-next702'
    damage(connection, round_id, 'UPDATE lab_arena_rounds SET champion_funding_frozen=true WHERE round_id=%s',
           ('arena-2099-04-02-next702',))
    with pytest.raises(Exception, match='later round has already frozen work'):
        reopen(connection, round_id, times=times, dry=True)


def test_service_roles_cannot_call_owner_functions(db):
    _, connection, _ = db
    for role in ('service_role', 'lab_arena_service'):
        for signature in (REOPEN, ADMIT):
            assert scalar(connection, 'SELECT has_function_privilege(%s,%s,%s)', (role, signature, 'EXECUTE')) is False
    assert scalar(connection, 'SELECT has_function_privilege(%s,%s,%s)', ('lab_arena_service', READER, 'EXECUTE')) is True


def test_admission_constants_match_python_review_policy():
    text = MIGRATION.read_text()
    for value in (code_review.RESEARCH_POLICY_ID, code_review.RESEARCH_POLICY_HASH, code_review.RESEARCH_REVIEWER_VERSION):
        assert "'%s'" % value in text
