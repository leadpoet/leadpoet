"""A proved infrastructure loss retains its bill without spending retry budget."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import host_interruption_retry474_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/478-lab-arena-infrastructure-retry-cost.sql'
POLICY = 'trusted_infrastructure_v1'


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for name in ('476-lab-arena-closed-failed-execute-host-drain.sql',
                         '477-lab-arena-settled-score-call-continuation.sql'):
                cur.execute((ROOT / 'scripts' / name).read_text())
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield db
    finally:
        generator.close()


def seed(conn, *, enabled=True):
    cfg = prior.seed(conn)
    if enabled:
        cfg['infrastructure_retry_cost_policy'] = POLICY
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',
                        (json.dumps(cfg), prior.ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    return cfg


def state(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__successful_icp_cost_state(%s,%s,8)',
                    (prior.ROUND, prior.cutoff.SUB))
        return cur.fetchone()[0]


def quota(conn, lease, token='b'):
    return prior.rpc(conn, 'lab_arena_run_quota_snapshot_v2(%s,%s)',
                     (lease['run_id'], 'sha256:' + token * 64))['sourcing_cost']


def eligible(conn, qualified=1):
    return prior.rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,8,%s)',
                     (prior.ROUND, prior.cutoff.SUB, qualified))


def assert_actual(conn, actual, successful=None):
    s = state(conn)
    assert s['settled_microusd'] == actual
    assert s['successful_microusd'] == (actual if successful is None else successful)
    with conn.cursor() as cur:
        cur.execute("SELECT sum(amount_microusd) FROM public.lab_arena_ledger WHERE entry_kind='settlement'")
        assert cur.fetchone()[0] == actual
    return s


# Exact settled charges from all six captured October 14 host-loss attempts.
CAPTURED_HOST_CHARGES = [620628, 633019, 701236, 778569, 585221, 655128]


@pytest.mark.parametrize('amount', CAPTURED_HOST_CHARGES + [778569, 4000000])
def test_host_loss_retains_actual_bill_and_releases_all_competition_cost(database, amount):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.host_loss(conn, amount=amount)
        second = prior.claim(conn, 'b')
        s = assert_actual(conn, amount)
        assert s['competition_settled_microusd'] == 0
        assert s['competition_successful_microusd'] == 0
        assert s['excluded_infrastructure_settled_microusd'] == amount
        assert eligible(conn)['competition_sourcing_microusd'] == 0
        q = quota(conn, second)
        assert q['settled_microusd'] == q['successful_microusd'] == 0
        assert len(q) == 8  # Preserve the frozen external checkpoint protocol.
        assert prior.pay(conn, second, token='b', call='f', amount=750000)['status'] == 'settled'
        s = assert_actual(conn, amount + 750000)
        assert s['competition_successful_microusd'] == 750000
        assert eligible(conn)['eligible'] is True
        assert prior.finish(conn, second, 'accepted')['status'] == 'accepted'
        with conn.cursor() as cur:
            cur.execute('SELECT status,result_doc FROM public.lab_arena_runs WHERE run_id=%s', (first['run_id'],))
            assert cur.fetchone() == ('failed', None)


@pytest.mark.parametrize('cause', ['model_error', 'invalid_output', 'credential_error', 'budget_exhausted', 'model_timeout', 'provider_error'])
def test_labels_and_genuine_miner_failures_do_not_create_credit(database, cause):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=900000)
        prior.finish(conn, first, cause, token='a')
        s = assert_actual(conn, 900000)
        assert s['competition_settled_microusd'] == 900000
        assert s['excluded_infrastructure_settled_microusd'] == 0
        assert eligible(conn)['eligibility_reason'] == 'cost_per_company_exceeded'


def provider_finish(conn, lease, *, unresolved=1):
    return prior.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,NULL)',
                     (lease['run_id'], 'sha256:' + 'a' * 64,
                      json.dumps({'terminal_status': 'provider_error', 'resource_summary': {
                          'provider_unresolved_failure_count': unresolved}}), 'provider_error'))


def failed_call(conn, lease, *, call='f', status=502, amount=25):
    assert prior.reserve(conn, lease, call=call)['status'] == 'reserved'
    prior.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
              (lease['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + call * 64))
    return prior.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,4500)',
                     (lease['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + call * 64,
                      amount, json.dumps({'status': status, 'call_succeeded': False})))


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_latest_transient_broker_failure_excludes_failed_attempt(database, status):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=700000)
        assert failed_call(conn, first, status=status)['status'] == 'settled'
        provider_finish(conn, first)
        s = assert_actual(conn, 700025, successful=700000)
        assert s['competition_settled_microusd'] == s['competition_successful_microusd'] == 0
        assert s['excluded_infrastructure_settled_microusd'] == 700025


@pytest.mark.parametrize('status', [400, 401, 402, 403, 404, 422])
def test_request_and_credential_errors_do_not_create_credit(database, status):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=700000)
        failed_call(conn, first, status=status)
        provider_finish(conn, first)
        assert assert_actual(conn, 700025, 700000)['competition_settled_microusd'] == 700025


def test_recovered_transient_call_followed_by_model_failure_has_no_credit(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=700000)
        failed_call(conn, first)
        prior.pay(conn, first, call='d', amount=100000)
        provider_finish(conn, first, unresolved=0)
        assert assert_actual(conn, 800025, 800000)['competition_successful_microusd'] == 800000


def test_host_cost_credit_never_discards_unresolved_success(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        prior.host_loss(conn, unsettled=True)
        s = state(conn)
        assert s['success_unresolved_calls'] == 1
        assert s['uncertain_calls'] == 1
        assert eligible(conn)['eligibility_reason'] == 'provider_cost_uncertain'


def test_absent_opt_in_preserves_old_published_cost_policy(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn, enabled=False)
        first = prior.host_loss(conn, amount=778569)
        second = prior.claim(conn, 'b')
        assert 'competition_settled_microusd' not in assert_actual(conn, 778569)
        assert quota(conn, second)['settled_microusd'] == 778569
        assert eligible(conn)['competition_sourcing_microusd'] == 778569
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (prior.ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert eligible(conn)['competition_sourcing_microusd'] == 778569


def test_late_failure_fee_keeps_original_transient_proof_and_exact_cost(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=700000)
        assert prior.reserve(conn, first, call='f')['status'] == 'reserved'
        prior.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                  (first['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + 'f' * 64))
        prior.rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                  (first['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + 'f' * 64,
                   json.dumps({'reason': 'transport_failure', 'call_succeeded': False, 'provider_status': 502})))
        provider_finish(conn, first)
        assert state(conn)['competition_settled_microusd'] == 0
        assert state(conn)['uncertain_calls'] == 1
        # Model the immutable, provider-bound settlement produced by delayed
        # reconciliation. Its final price receipt need not repeat HTTP 502.
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
               provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
              SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
                provider,operation_id,funding_source,25,
                jsonb_build_object('deepline_delayed_reconciliation',true,
                  'reconciled_uncertainty_entry_id',entry_id),
                '{"status":200,"call_succeeded":false}'::jsonb
              FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='uncertain'""", (first['run_id'],))
        conn.commit()
        s = assert_actual(conn, 700025, 700000)
        assert s['competition_settled_microusd'] == 0
        assert s['uncertain_calls'] == s['success_unresolved_calls'] == 0
        assert eligible(conn)['eligible'] is True
        second = prior.claim(conn, 'b')
        assert quota(conn, second)['settled_microusd'] == 0
        prior.pay(conn, second, token='b', call='d', amount=750000)
        replay = prior.reserve(conn, second, token='b', call='d')
        assert replay['status'] == 'settled' and replay['idempotent'] is True
        s = assert_actual(conn, 1450025, 1450000)
        assert s['competition_successful_microusd'] == 750000


def test_schema_and_private_helpers_keep_service_only_authority(database):
    from lab_arena.store import ArenaStore, PsycopgTransport
    transport = PsycopgTransport(lambda: database[0].connect(**database[1]))
    try:
        assert ArenaStore(transport).infrastructure_retry_cost_schema() == {
            'schema_version': 'leadpoet.lab_arena.infrastructure_retry_cost_schema.v1',
            'version': 478, 'policy': POLICY,
        }
    finally:
        transport.close()
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT public.lab_arena_infrastructure_retry_cost_schema_v1()")
        assert cur.fetchone()[0]['policy'] == POLICY
        for name in ('lab_arena__execute_host_interruption_v1', 'lab_arena__execute_infrastructure_cost_exempt_v1'):
            cur.execute("SELECT has_function_privilege('anon',%s,'EXECUTE'),has_function_privilege('lab_arena_service',%s,'EXECUTE')",
                        ('public.' + name + '(public.lab_arena_runs)',) * 2)
            assert cur.fetchone() == (False, False)


def test_round_cost_policy_configuration_round_trips_and_rejects_bad_scope():
    from lab_arena import contracts, integrity, scoring
    from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
    cfg = base_round_configuration()
    assert 'infrastructure_retry_cost_policy' not in contracts.validate_round_configuration(cfg)
    cfg.update(integrity_policy=integrity.POLICY,
               scorer_policy=scoring.build_scorer_policy(scoring_adapter_version=integrity.SCORING_ADAPTER),
               sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
               cost_per_company_microusd=800000, execution_icp_cap_microusd=4000000,
               infrastructure_retry_cost_policy=POLICY)
    validated = contracts.validate_round_configuration(cfg)
    assert contracts.validate_round_configuration(json.loads(json.dumps(validated))) == validated
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(dict(cfg, infrastructure_retry_cost_policy='untrusted'))
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(dict(cfg, sourcing_cost_eligibility_policy=contracts.SUCCESSFUL_CALLS_COST_POLICY))


def test_missing_cost_capability_fails_closed_at_startup():
    from lab_arena.service import ServiceError
    from lab_arena.store import ArenaStoreError
    from tests.lab_arena.cost_eligibility_test import _startup_service
    svc, _ = _startup_service(ArenaStoreError('lab_arena_submission_missing'))
    svc._config.defaults.per_icp_cost_policy = True
    svc._store.per_icp_cost_schema = lambda: {}
    def missing():
        raise ArenaStoreError('missing infrastructure retry cost schema')
    svc._store.infrastructure_retry_cost_schema = missing
    with pytest.raises(ServiceError):
        svc.startup_checks()


def open_legacy_round(conn, *, started=False):
    cfg = seed(conn, enabled=False)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if not started:
            cur.execute('DELETE FROM public.lab_arena_runs WHERE round_id=%s', (prior.ROUND,))
        cur.execute("UPDATE public.lab_arena_rounds SET status='open',benchmark_ref=NULL,icp_set_date=NULL WHERE round_id=%s", (prior.ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return cfg


def test_atomic_freeze_guard_accepts_only_fixed_unstarted_policy_addition(database):
    with database[0].connect(**database[1]) as conn:
        cfg = open_legacy_round(conn)
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_rounds SET status='committed',configuration_doc=%s::jsonb WHERE round_id=%s",
                        (json.dumps(dict(cfg, infrastructure_retry_cost_policy=POLICY)), prior.ROUND))
        conn.commit()
        with conn.cursor() as cur:
            cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (prior.ROUND,))
            assert cur.fetchone()[0] == dict(cfg, infrastructure_retry_cost_policy=POLICY)


@pytest.mark.parametrize('case', ['null', 'unknown', 'started', 'other_field', 'remove_frozen', 'published'])
def test_policy_freeze_guard_rejects_null_unknown_started_and_other_mutations(database, case):
    with database[0].connect(**database[1]) as conn:
        cfg = open_legacy_round(conn, started=case == 'started')
        new = dict(cfg, infrastructure_retry_cost_policy=POLICY)
        if case == 'null': new['infrastructure_retry_cost_policy'] = None
        if case == 'unknown': new['infrastructure_retry_cost_policy'] = 'untrusted'
        if case == 'other_field': new['execution_icp_cap_microusd'] += 1
        if case in ('published', 'remove_frozen'):
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('UPDATE public.lab_arena_rounds SET status=%s,configuration_doc=%s::jsonb WHERE round_id=%s',
                            ('published' if case == 'published' else 'committed',
                             json.dumps(new if case == 'remove_frozen' else cfg), prior.ROUND))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
        if case == 'remove_frozen': new.pop('infrastructure_retry_cost_policy')
        with conn.cursor() as cur, pytest.raises(database[0].Error):
            cur.execute("UPDATE public.lab_arena_rounds SET status='committed',configuration_doc=%s::jsonb WHERE round_id=%s",
                        (json.dumps(new), prior.ROUND))
        conn.rollback()


def test_active_retry_still_stops_at_its_own_fixed_money_cap(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        prior.host_loss(conn, amount=4000000)
        second = prior.claim(conn, 'b')
        prior.pay(conn, second, token='b', call='f', amount=4000000)
        denial = prior.reserve(conn, second, token='b', call='d')
        assert denial['status'] == 'refused' and denial['reason'] == 'money_cap'
        s = assert_actual(conn, 8000000)
        assert s['competition_settled_microusd'] == 4000000


def test_cost_credit_does_not_extend_cumulative_provider_call_cap(database):
    with database[0].connect(**database[1]) as conn:
        cfg = seed(conn)
        cfg['call_quotas'] = {p: 1 for p in ('openrouter', 'scrapingdog', 'deepline')}
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',
                        (json.dumps(cfg), prior.ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        prior.host_loss(conn)
        second = prior.claim(conn, 'b')
        prior.pay(conn, second, token='b', call='f')
        assert prior.finish(conn, second)['confirmation_attempt'] == 3
        third = prior.claim(conn, 'c')
        denial = prior.reserve(conn, third, token='c', call='d')
        assert denial['status'] == 'refused' and denial['reason'] == 'per_icp_quota'


@pytest.mark.parametrize('amount,passes', [(800000, True), (800001, False)])
def test_retry_cost_per_qualified_company_boundary_is_unchanged(database, amount, passes):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        prior.host_loss(conn, amount=778569)
        second = prior.claim(conn, 'b')
        prior.pay(conn, second, token='b', call='f', amount=amount)
        verdict = eligible(conn)
        assert verdict['eligible'] is passes
        assert verdict['competition_sourcing_microusd'] == amount


@pytest.mark.parametrize('catalog_enabled', [False, True])
def test_existing_unstarted_round_enrolls_only_when_commit_rpc_freezes_it(database, catalog_enabled):
    from datetime import datetime, timedelta, timezone
    from lab_arena.store import ArenaStore, PsycopgTransport
    from tests.lab_arena.test_lab_arena_migration_postgres import round_config, frozen_participants, hotkey
    connect = lambda: database[0].connect(**database[1])
    store = ArenaStore(PsycopgTransport(connect))
    rid = 'arena-2099-08-09-infra' + str(catalog_enabled).lower()
    day = datetime.now(timezone.utc).date()
    bank = day - timedelta(days=1)
    cfg = round_config(rid, [hotkey('freeze-runner')], cost_per_company_microusd=800000)
    cfg.update(sourcing_cost_eligibility_policy='successful_calls_per_icp_v1', execution_icp_cap_microusd=4000000)
    cfg['schedule'] = {'submission_open': bank.isoformat() + 'T00:00:00Z',
                       'submission_cutoff': (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()}
    assert store.create_round(rid, cfg)['status'] == 'created'
    participants = frozen_participants(store, rid, 1, prefix='infra-freeze-' + str(catalog_enabled), king_index=0)
    catalog = {'schema_version': 'leadpoet.lab_arena.deepline_catalog.v1',
               'policy_version': 'leadpoet.lab_arena.deepline_company_research.v1',
               'allow_people': False, 'tools': [{'tool_id': 'exa_search'}], 'catalog_hash': 'a' * 64}
    args = dict(participants=participants, benchmark_ref='arena/' + rid + '/benchmark.json',
                evaluation_date=day.isoformat(), icp_set_date=bank.isoformat(),
                scorer_image_digest=cfg['scorer_image_digest'], scorer_image_reference=cfg['scorer_image_reference'])
    if catalog_enabled: args['deepline_catalog'] = catalog
    before = store.get_round(rid)['configuration_doc']
    assert 'infrastructure_retry_cost_policy' not in before
    assert store.commit_round_v2(rid, **args)['status'] == 'ok'
    after = store.get_round(rid)['configuration_doc']
    assert after['infrastructure_retry_cost_policy'] == POLICY
    if catalog_enabled:
        assert after['deepline_catalog'] == catalog
        assert after['call_quotas']['deepline'] == after['scoring_call_quotas']['deepline'] == 0
    else:
        assert {k: v for k, v in after.items() if k != 'infrastructure_retry_cost_policy'} == before
    assert store.commit_round_v2(rid, **args)['status'] == 'stale'
    assert store.get_round(rid)['configuration_doc'] == after
