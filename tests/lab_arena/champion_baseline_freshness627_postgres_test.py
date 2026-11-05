"""Late prior promotions stop only stale baseline work and new final authority."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import re
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from lab_arena import service as arena_service
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov04_baseline_source621_postgres_test as prior
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL = prior.SCRIPTS / '627-lab-arena-champion-baseline-freshness.sql'
ROUND = 'arena-2026-11-05'
BASELINE = 'baseline-2026-11-05'
COMMIT = 'b' * 40
URL = 'https://github.com/leadpoet/leadpoet-sales-agent/archive/' + COMMIT + '.tar.gz'
scalar = prior.scalar


@pytest.fixture(scope='module')
def database():
    fixture = prior.database.__wrapped__()
    pg, dsn = next(fixture)
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute(prior.SQL.read_text().split('-- Bind the correction to an exact terminal snapshot.')[0] + 'COMMIT;')
            cur.execute((prior.SCRIPTS / '619-lab-arena-release-undispatched-call.sql').read_text())
            # Match production's committed 610/612/613/614 claim hooks. The
            # historical 621 fixture intentionally predates their data repairs.
            for tag in ('610', '612', '613', '614'):
                source = next(prior.SCRIPTS.glob(tag + '-*.sql')).read_text()
                helper = re.search(r'CREATE OR REPLACE FUNCTION public.lab_arena__nov03_.*?\n\$[a-z0-9]+\$;', source, re.S)[0]
                cur.execute(helper)
                patch = re.search(r"\('public.lab_arena_claim_assignment[^']*','([a-f0-9]+)','([a-f0-9]+)'.*?\$old" + tag + r"\$(.*?)\$old" + tag + r"\$,\$new" + tag + r"\$(.*?)\$new" + tag + r"\$", source, re.S)
                definition = scalar(cur, "SELECT pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)")
                assert hashlib.sha256(definition.encode()).hexdigest() == patch[1]
                cur.execute(definition.replace(patch[3], patch[4]))
                assert scalar(cur, "SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure),'sha256'),'hex')") == patch[2]
        yield pg, dsn
    finally:
        fixture.close()


@pytest.fixture
def prepared(database):
    pg, dsn = database
    fixture = prior.prepared.__wrapped__(database)
    p = next(fixture)
    try:
        with p.conn.cursor() as cur:
            cur.execute(SQL.read_text())
        p.conn.commit()
        config = deepcopy(p.row['configuration_doc'])
        config['round_id'] = ROUND
        config['baseline_source_url'] = URL
        config['reasoning_baseline_source_sha256'] = 'sha256:' + 'b' * 64
        config['schedule'] = {key: '2099-11-05T23:00:00Z' for key in config['schedule']}
        baseline = dict(submission_id=BASELINE, miner_hotkey=config['baseline_hotkey'], is_king=True,
                        source_ref='arena/' + ROUND + '/sources/' + BASELINE + '.tar.gz', source_size_bytes=100)
        miner = dict(submission_id='freshness627-miner', miner_hotkey=hotkey('freshness627-miner'), is_king=False,
                     source_ref='arena/' + ROUND + '/sources/freshness627-miner.tar.gz', source_size_bytes=100)
        with p.conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb WHERE round_id=%s", (json.dumps({'commit': COMMIT}), prior.ROUND))
            cur.execute("UPDATE public.lab_arena_rounds SET rewards_enabled=true WHERE round_id='arena-2026-11-03'")
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage1',stage_generation=1,configuration_doc=%s::jsonb,participants=%s::jsonb,champion_funding_frozen=true,champion_submission_id='source621-miner-0',champion_hotkey=%s,benchmark_ref='benchmark.json' WHERE round_id=%s", (json.dumps(config), json.dumps([baseline, miner]), p.row['king_hotkey'], ROUND))
            for participant in (baseline, miner):
                cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,is_king,status,source_ref,source_size_bytes,code_review_status,code_review_attempts,code_review_doc,code_review_claim,code_review_started_at) VALUES (%s,%s,%s,%s,'frozen',%s,100,'passed',1,'{}','sha256:'||repeat('b',64),now())", (participant['submission_id'], ROUND, participant['miner_hotkey'], participant['is_king'], participant['source_ref']))
            cur.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit()
        p.target_config = config
        p.target_baseline = baseline
        p.target_miner = miner
        yield p
    finally:
        fixture.close()


def change(p, statement, arguments=()):
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute(statement, arguments)
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()


def stale(p):
    change(p, "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{baseline_source_url}',%s::jsonb) WHERE round_id=%s", (json.dumps(URL.replace(COMMIT, 'a' * 40)), ROUND))


def current(p):
    with p.conn.cursor() as cur:
        return scalar(cur, 'SELECT public.lab_arena__baseline_source_current_v1(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (ROUND,))


def freeze(p):
    row = p.store.get_round(ROUND)
    config = row['configuration_doc']
    return p.store.freeze_full_thinking_baseline(ROUND, expected_promotion_round_id=None,
        expected_winner_submission_id=None, expected_commit=None,
        expected_source_url=config['baseline_source_url'], expected_source_sha256=config['reasoning_baseline_source_sha256'],
        source_url=config['baseline_source_url'], source_sha256=config['reasoning_baseline_source_sha256'])


def run(p, baseline=True):
    participant = p.target_baseline if baseline else p.target_miner
    assignment = ROUND + ':' + participant['submission_id'] + ':1:0'
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,1,0,1,'execute','pending',1,0)", (assignment + ':1', assignment, ROUND, participant['submission_id'], participant['miner_hotkey']))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    return assignment + ':1'


def claim(p, request=None):
    return p.store.claim_assignment(round_id=ROUND, runner_hotkey=hotkey('freshness627-runner'),
        declared_parallelism=20, slot_ceiling=20, excluded_miner_hotkeys=[], request_id=request or uuid.uuid4().hex,
        request_hash='sha256:' + 'b' * 64, lease_token_hash=hash_lease_token('freshness627'), lease_ttl_seconds=1500)


def reserve(p, run_id, identity):
    return p.store.reserve_call(run_id=run_id, lease_token_hash=hash_lease_token('freshness627'),
        call_identity=identity, operation_id='openrouter.responses', provider='openrouter', funding_source=p.store.provider_funding(run_id, 'openrouter')['funding_source'],
        amount_microusd=100, call_doc={'fixture': 'freshness627'}, lease_ttl_seconds=1500)


def test_frozen_late_promotion_blocks_without_rewriting_history(prepared):
    p = prepared
    assert current(p) is True
    before = p.store.get_round(ROUND)
    stale(p)
    original = p.store.get_round(ROUND)
    assert freeze(p) == {'status': 'promotion_pending'}
    assert p.store.get_round(ROUND) == original
    assert before['champion_submission_id'] == original['champion_submission_id']
    assert original['configuration_doc']['baseline_source_url'] != URL


def test_baseline_claim_and_replay_stop_but_miner_claim_remains(prepared):
    p = prepared
    baseline = run(p)
    request = uuid.uuid4().hex
    leased = claim(p, request)
    assert leased['status'] == 'leased' and leased['run_id'] == baseline
    stale(p)
    assert claim(p, request) == {'status': 'promotion_pending'}
    # Preserve the existing lease; it can still deliver completion and bills.
    assert p.store.get_run(baseline)['status'] == 'leased'
    miner = run(p, baseline=False)
    leased_miner = claim(p)
    assert leased_miner['status'] == 'leased' and leased_miner['run_id'] == miner


def test_pending_baseline_is_not_leased_and_publication_is_held(prepared):
    p = prepared
    baseline = run(p)
    stale(p)
    assert claim(p) == {'status': 'promotion_pending'}
    assert p.store.get_run(baseline)['status'] == 'pending'
    change(p, "UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s", (ROUND,))
    before = p.store.get_round(ROUND)
    assert p.store.transition_round(ROUND, 'scored', 'published', {}) == {'status': 'promotion_pending'}
    assert p.store.get_round(ROUND) == before


def test_bootstrap_own_future_and_historical_authority(prepared):
    p = prepared
    # Own and future promotions must not invalidate an earlier round's source.
    change(p, "UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb,baseline_promoted_at=now() WHERE round_id=%s", (json.dumps({'commit': 'f' * 40}), ROUND))
    assert current(p) is True
    change(p, "UPDATE public.lab_arena_rounds SET evaluation_date='2026-11-06' WHERE round_id=%s", (prior.ROUND,))
    # The actual latest prior champion is now November 3, not the future row.
    assert current(p) is False
    change(p, "UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id='arena-2026-11-03'")
    change(p, "UPDATE public.lab_arena_rounds SET champion_submission_id=NULL,champion_hotkey=NULL WHERE round_id=%s", (ROUND,))
    assert current(p) is True
    assert freeze(p) == {'status': 'existing'}
    change(p, "UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (ROUND,))
    assert current(p) is True


def test_invalid_known_champion_cannot_silently_bootstrap(prepared):
    p = prepared
    change(p, "UPDATE public.lab_arena_rounds SET champion_submission_id=NULL,champion_hotkey=NULL WHERE round_id=%s", (ROUND,))
    change(p, "UPDATE public.lab_arena_submissions SET status='rejected' WHERE submission_id='source621-miner-0'")
    assert current(p) is False
    assert freeze(p) == {'status': 'promotion_pending'}


@pytest.mark.parametrize('entrypoint', ['freeze', 'claim', 'publish', 'reserve', 'dispatch'])
def test_late_promotion_is_read_after_waiting_for_existing_lock(prepared, entrypoint):
    p = prepared
    baseline = run(p)
    if entrypoint in ('reserve', 'dispatch'):
        assert claim(p)['status'] == 'leased'
    identity = 'sha256:' + 'e' * 64
    if entrypoint == 'dispatch':
        assert reserve(p, baseline, identity)['status'] == 'reserved'
    if entrypoint == 'publish':
        change(p, "UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s", (ROUND,))
    with p.conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0))")
    funding = p.store.provider_funding(baseline, 'openrouter')['funding_source'] if entrypoint in ('reserve', 'dispatch') else None
    entered = threading.Event()
    state = {}
    def attempt():
        with p.pg.connect(**p.dsn) as conn, conn.cursor() as cur:
            cur.execute("SET statement_timeout='10s'")
            state['pid'] = scalar(cur, 'SELECT pg_backend_pid()')
            cur.execute('SET ROLE lab_arena_service')
            entered.set()
            if entrypoint == 'freeze':
                config = p.target_config
                cur.execute("SELECT public.lab_arena_freeze_full_thinking_baseline_v1(%s,NULL,NULL,NULL,%s,%s,%s,%s)",
                    (ROUND, config['baseline_source_url'], config['reasoning_baseline_source_sha256'], config['baseline_source_url'], config['reasoning_baseline_source_sha256']))
            elif entrypoint == 'claim':
                cur.execute("SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)",
                    (ROUND, hotkey('freshness627-runner'), uuid.uuid4().hex, 'sha256:' + 'b' * 64, hash_lease_token('freshness627')))
            elif entrypoint == 'publish':
                cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published','{}')", (ROUND,))
            elif entrypoint == 'reserve':
                cur.execute("SELECT public.lab_arena_reserve_call(%s,%s,%s,'openrouter.responses','openrouter',%s,100,'{\"fixture\":\"freshness627\"}',1500)",
                    (baseline, hash_lease_token('freshness627'), identity, funding))
            else:
                cur.execute("SELECT public.lab_arena_mark_dispatched(%s,%s,%s)", (baseline, hash_lease_token('freshness627'), identity))
            return cur.fetchone()[0]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(attempt)
        assert entered.wait(3)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with p.conn.cursor() as cur:
                cur.execute('SELECT pg_stat_clear_snapshot()')
                waiting = scalar(cur, "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s", (state['pid'],))
            if waiting: break
            entered.wait(.02)
        else: pytest.fail('entrypoint did not reach the promotion-lock boundary')
        # Simulates the exact last promotion write while owning its real lock.
        with p.conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb WHERE round_id=%s", (json.dumps({'commit': 'c' * 40}), prior.ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit()
        assert pending.result(timeout=10) == {'status': 'promotion_pending'}


def test_stale_dispatch_releases_only_undispatched_reservation(prepared):
    p = prepared
    baseline = run(p)
    assert claim(p)['status'] == 'leased'
    identity = 'sha256:' + 'd' * 64
    assert reserve(p, baseline, identity)['status'] == 'reserved'
    stale(p)
    assert reserve(p, baseline, 'sha256:' + 'e' * 64) == {'status': 'promotion_pending'}
    assert p.store.mark_dispatched(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity) == {'status': 'promotion_pending'}
    replay = p.store.mark_dispatched(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity)
    assert replay['status'] == 'recovered'
    with p.conn.cursor() as cur:
        rows = scalar(cur, 'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE call_identity=%s', (identity,))
        assert [r['entry_kind'] for r in rows] == ['reservation', 'recovery']
        assert rows[-1]['amount_microusd'] == 0
        assert scalar(cur, "SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s", ('sha256:' + 'e' * 64,)) == 0


def test_already_dispatched_call_and_miner_provider_calls_survive(prepared):
    p = prepared
    baseline = run(p)
    assert claim(p)['status'] == 'leased'
    identity = 'sha256:' + 'a' * 64
    assert reserve(p, baseline, identity)['status'] == 'reserved'
    dispatched = p.store.mark_dispatched(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity)
    assert dispatched['status'] == 'dispatched'
    stale(p)
    assert reserve(p, baseline, identity)['status'] == 'dispatched'
    assert p.store.mark_dispatched(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity)['status'] == 'dispatched'
    settled = p.store.settle_call(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'),
        call_identity=identity, actual_microusd=123, terminal_response={'status': 200})
    assert settled['status'] == 'settled'
    assert reserve(p, baseline, identity)['status'] == 'settled'
    assert p.store.mark_dispatched(run_id=baseline, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity)['status'] == 'settled'
    miner = run(p, baseline=False)
    assert claim(p)['run_id'] == miner
    miner_identity = 'sha256:' + 'f' * 64
    assert reserve(p, miner, miner_identity)['status'] == 'reserved'
    assert p.store.mark_dispatched(run_id=miner, lease_token_hash=hash_lease_token('freshness627'), call_identity=miner_identity)['status'] == 'dispatched'


def test_native_null_date_creation_holds_correction_then_freezes_and_commits(prepared, tmp_path):
    """Exercise the real create→freeze→commit order, including 621's NULL gap."""
    from datetime import datetime, timezone
    from tests.lab_arena.full_thinking_round_test import opted_in
    from tests.lab_arena.test_lab_arena_promotion_git import source
    p = prepared
    service = opted_in()
    service._store = p.store
    service._objects = arena_service.LocalObjectStore(tmp_path / 'objects')
    payload = source('native627-champion')
    service._config.baseline_source_fetcher = lambda *_args: payload
    with p.conn.cursor() as cur:
        old = scalar(cur, 'SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (prior.ROUND,))
        new = dict(old, status='scored', published_at=None, publication_doc=None)
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) VALUES (%s,1,'native627-correction','sha256:'||repeat('c',64),'fixture',txid_current(),pg_backend_pid(),%s::jsonb,%s::jsonb,'{}','{}')", (prior.ROUND, json.dumps(old), json.dumps(new)))
    p.conn.commit()
    change(p, "UPDATE public.lab_arena_rounds SET status='scored',published_at=NULL,publication_doc=NULL WHERE round_id=%s", (prior.ROUND,))
    config = service.create_round(datetime(2026, 11, 6, tzinfo=timezone.utc))
    rid = config['round_id']
    created = p.store.get_round(rid)
    assert created['evaluation_date'] is None
    assert (created['arena_network_name'], created['arena_netuid']) == ('finney', 71)
    # Restore only the exact old freeze entrypoints in this disposable DB:
    # the same real NULL-date row was admitted by 621 despite the receipt.
    with p.conn.cursor() as cur:
        for name, signature in (
            ('lab_arena_freeze_champion_funding', 'text'),
            ('lab_arena_freeze_full_thinking_baseline_v1', 'text,text,text,text,text,text,text,text'),
        ):
            definition = scalar(cur, "SELECT pg_get_functiondef(%s::regprocedure)", ('public.' + name + '(' + signature + ')',))
            rows = re.findall(r"\('public\." + name + r"\([^']*','[a-f0-9]+','[a-f0-9]+',\d+,\$old627\$(.*?)\$old627\$,\$new627\$(.*?)\$new627\$\)", SQL.read_text(), re.S)
            for before, after in reversed(rows):
                assert definition.count(after) == 1
                definition = definition.replace(after, before)
            cur.execute(definition)
    p.conn.commit()
    assert service._freeze_full_thinking_baseline(rid, created) == {'status': 'frozen'}
    old_frozen = p.store.get_round(rid)
    assert old_frozen['evaluation_date'] is None
    with p.conn.cursor() as cur:
        cur.execute(SQL.read_text())
    p.conn.commit()
    with pytest.raises(arena_service.ServiceError, match='baseline_promotion_pending'):
        service.freeze_participants(rid)
    assert p.store.get_round(rid) == old_frozen
    # Resolve the prior correction in fixture state. A fresh normal round can
    # adopt its exact latest source; the older frozen comparison stays blocked.
    change(p, "UPDATE public.lab_arena_rounds SET status='published',published_at=%s,publication_doc=%s::jsonb WHERE round_id=%s", (old['published_at'], json.dumps(old['publication_doc']), prior.ROUND))
    with pytest.raises(arena_service.ServiceError, match='baseline_promotion_pending'):
        service.freeze_participants(rid)
    next_config = service.create_round(datetime(2026, 11, 7, tzinfo=timezone.utc))
    next_id = next_config['round_id']
    participants = service.freeze_participants(next_id)
    frozen = p.store.get_round(next_id)
    assert frozen['evaluation_date'] is None
    assert frozen['configuration_doc']['baseline_source_url'] == URL
    assert frozen['champion_submission_id'] == 'source621-miner-0'
    assert service._objects.get(participants[0]['source_ref']) == payload
    committed = p.store.commit_round_v2(next_id, participants=participants,
        benchmark_ref='arena/' + next_id + '/benchmark.json', evaluation_date='2026-11-07',
        icp_set_date='2026-11-06', scorer_image_digest=next_config['scorer_image_digest'],
        scorer_image_reference=next_config['scorer_image_reference'])
    assert committed['status'] == 'ok'
    stored = p.store.get_round(next_id)
    assert stored['evaluation_date'] == '2026-11-07' and stored['status'] == 'committed'


def test_future_champion_is_excluded_before_source_download(prepared, tmp_path):
    from datetime import datetime, timezone
    from tests.lab_arena.full_thinking_round_test import opted_in
    from tests.lab_arena.test_lab_arena_promotion_git import source
    p = prepared
    change(p, "UPDATE public.lab_arena_rounds SET evaluation_date='2026-11-08',baseline_promoted_at=now()+interval '1 day' WHERE round_id='arena-2026-11-03'")
    assert p.store.latest_promoted_champion(network_name='finney', netuid=71)['round_id'] == 'arena-2026-11-03'
    assert p.store.latest_promoted_champion(network_name='finney', netuid=71,
        before_evaluation_date='2026-11-06')['round_id'] == prior.ROUND
    service = opted_in()
    service._store = p.store
    service._objects = arena_service.LocalObjectStore(tmp_path / 'objects')
    fetched = []
    payload = source('bounded627')
    service._config.baseline_source_fetcher = lambda url, _limit: fetched.append(url) or payload
    configuration = service.create_round(datetime(2026, 11, 6, tzinfo=timezone.utc))
    service.freeze_participants(configuration['round_id'])
    assert fetched and all(url == URL for url in fetched)


def test_replay_preserves_security_and_rejects_changed_helper(prepared):
    p = prepared
    with p.conn.cursor() as cur:
        acl = scalar(cur, "SELECT nspacl::TEXT FROM pg_namespace WHERE nspname='public'")
        before = scalar(cur, "SELECT jsonb_agg(to_jsonb(proc) ORDER BY oid) FROM pg_proc proc WHERE proname IN('lab_arena__baseline_source_current_v1','lab_arena_latest_promoted_champion_v1','lab_arena_freeze_full_thinking_baseline_v1','lab_arena_freeze_champion_funding','lab_arena_claim_assignment','lab_arena_transition_round','lab_arena_reserve_call','lab_arena_mark_dispatched')")
        cur.execute(SQL.read_text())
        assert scalar(cur, "SELECT nspacl::TEXT FROM pg_namespace WHERE nspname='public'") == acl
        assert scalar(cur, "SELECT jsonb_agg(to_jsonb(proc) ORDER BY oid) FROM pg_proc proc WHERE proname IN('lab_arena__baseline_source_current_v1','lab_arena_latest_promoted_champion_v1','lab_arena_freeze_full_thinking_baseline_v1','lab_arena_freeze_champion_funding','lab_arena_claim_assignment','lab_arena_transition_round','lab_arena_reserve_call','lab_arena_mark_dispatched')") == before
        assert scalar(cur, "SELECT has_function_privilege('lab_arena_service','public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)','EXECUTE')") is False
        assert scalar(cur, "SELECT has_function_privilege('lab_arena_service','public.lab_arena_latest_promoted_champion_v1(text,integer,text)','EXECUTE')") is True
        assert scalar(cur, "SELECT has_function_privilege('anon','public.lab_arena_latest_promoted_champion_v1(text,integer,text)','EXECUTE')") is False
        cur.execute('BEGIN')
        cur.execute('ALTER FUNCTION public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds) STRICT')
        with pytest.raises(p.pg.Error, match='helper differs'):
            cur.execute(SQL.read_text())
    p.conn.rollback()


def score_run(p, baseline=True):
    source = run(p, baseline=baseline)
    participant = p.target_baseline if baseline else p.target_miner
    change(p, "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',result_doc='{\"terminal_status\":\"accepted\"}',output_ref='baseline-output.json' WHERE run_id=%s", (source,))
    assignment = ROUND + ':' + participant['submission_id'] + ':1:0:score'
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,scored_run_id) VALUES (%s,%s,%s,%s,%s,1,0,1,'score','pending',1,0,%s)", (assignment + ':1', assignment, ROUND, participant['submission_id'], participant['miner_hotkey'], source))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    return assignment + ':1'


def test_baseline_judge_claim_and_new_paid_dispatch_wait_for_prior_correction(prepared):
    p = prepared
    # Match the live November 5 incident: its saved source still agrees with
    # November 3, but November 4 has an unfinished published correction.
    change(p, "UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s", (prior.ROUND,))
    change(p, "UPDATE public.lab_arena_rounds r SET status='stage1_scoring',champion_submission_id=prior.publication_doc#>>'{king_decision,winner_submission_id}',champion_hotkey=prior.publication_doc#>>'{king_decision,king_hotkey}',configuration_doc=jsonb_set(r.configuration_doc,'{baseline_source_url}',to_jsonb('https://github.com/leadpoet/leadpoet-sales-agent/archive/'||(prior.promotion_doc->>'commit')||'.tar.gz')) FROM public.lab_arena_rounds prior WHERE r.round_id=%s AND prior.round_id='arena-2026-11-03'", (ROUND,))
    assert current(p) is True
    score = score_run(p)
    request = uuid.uuid4().hex
    lease = claim(p, request)
    assert lease['status'] == 'leased' and lease['run_id'] == score
    identity = 'sha256:' + 'c' * 64
    assert reserve(p, score, identity)['status'] == 'reserved'
    with p.conn.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,1,'judge627-correction','sha256:'||repeat('c',64),'fixture',txid_current(),pg_backend_pid(),jsonb_set(to_jsonb(r),'{status}','\"published\"'),to_jsonb(r),'{}','{}' FROM public.lab_arena_rounds r WHERE round_id=%s", (prior.ROUND,))
    p.conn.commit()
    assert current(p) is False
    assert claim(p, request) == {'status': 'promotion_pending'}
    assert reserve(p, score, 'sha256:' + '9' * 64) == {'status': 'promotion_pending'}
    assert p.store.mark_dispatched(run_id=score, lease_token_hash=hash_lease_token('freshness627'), call_identity=identity) == {'status': 'promotion_pending'}
    assert p.store.get_run(score)['status'] == 'leased'
    miner_score = score_run(p, baseline=False)
    assert claim(p)['run_id'] == miner_score
    miner_identity = 'sha256:' + '8' * 64
    assert reserve(p, miner_score, miner_identity)['status'] == 'reserved'
    assert p.store.mark_dispatched(run_id=miner_score, lease_token_hash=hash_lease_token('freshness627'), call_identity=miner_identity)['status'] == 'dispatched'


@pytest.mark.parametrize('mode', ['live', 'shadow'])
def test_legacy_funding_keeps_existing_timestamp_order_and_future_scope(prepared, mode):
    p = prepared
    # Legacy funding intentionally keeps its old timestamp selector, including
    # future rows. Only modern full-thinking freezes use the prior-day reader.
    change(p, "UPDATE public.lab_arena_rounds SET status='open',champion_funding_frozen=false,champion_submission_id=NULL,champion_hotkey=NULL,configuration_doc=(configuration_doc-'reasoning_capture_policy')||jsonb_build_object('mode',%s) WHERE round_id=%s", (mode, ROUND))
    change(p, "UPDATE public.lab_arena_rounds SET evaluation_date='2026-11-08',baseline_promoted_at=now()+interval '1 day',configuration_doc=jsonb_set(configuration_doc,'{mode}',to_jsonb(%s::text)) WHERE round_id='arena-2026-11-03'", (mode,))
    assert p.store.freeze_champion_funding(ROUND) == {'status': 'frozen'}
    assert p.store.get_round(ROUND)['champion_submission_id'] == 'sub-28594d902080fb5213f4344c09ffb531'
