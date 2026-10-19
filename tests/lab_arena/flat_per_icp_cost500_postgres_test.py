"""The frozen flat allowance is independent of qualification and keeps billing."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import infrastructure_retry_cost478_postgres_test as infra

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/500-lab-arena-flat-per-icp-cost.sql'
FLAT = 'successful_calls_flat_per_icp_v1'
LEGACY = 'successful_calls_per_icp_v1'
prior = infra.prior


def rows(cur):
    result = []
    for table, order in (('lab_arena_rounds', 'round_id'),
                         ('lab_arena_runs', 'run_id'), ('lab_arena_ledger', 'entry_id')):
        cur.execute(f"SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY {order})::text,'null')) "
                    f'FROM public.{table} t')
        result.append(cur.fetchone()[0])
    return result


def catalog(cur):
    cur.execute("SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),"
                "pg_get_userbyid(p.proowner),p.proacl::text,p.prosecdef,p.provolatile,p.proconfig "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "WHERE n.nspname='public' AND p.prokind='f' AND p.proname LIKE 'lab_arena_%' "
                "ORDER BY p.oid")
    return cur.fetchall()


def database_with_flat_cost_migration(*, seed_existing=False):
    """Exact latest release for SQL authority and full round lifecycle tests."""
    from tests.lab_arena import progression_lock_order490_postgres_test as progression
    gen = infra.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # The ordinary fixture omits historical guarded production branches.
            # Apply their existing source-only capture exactly as490's own test.
            names = ['460-lab-arena-publication-round-witness.sql']
            names += [p.name for n in (479, 480, 481, 482, 484)
                      for p in sorted((ROOT / 'scripts').glob(f'{n}-*.sql'))]
            names += ['478-lab-arena-baseline-closed-settlement-recovery.sql',
                      '485-lab-arena-baseline-infrastructure-continuation.sql',
                      '486-lab-arena-restart-baseline-continuation-phase.sql',
                      '487-lab-arena-baseline-drain-lineage.sql',
                      '490-lab-arena-progression-lock-order.sql',
                      '261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql',
                      '453-lab-arena-deepline-adaptation-cost-reconciliation.sql',
                      '456-lab-arena-score-deepline-cost-priority.sql',
                      '457-lab-arena-closed-adaptation-score-billing.sql',
                      '483-lab-arena-deepline-candidate-head-scope.sql']
            names += [p.name for n in range(492, 500)
                      for p in sorted((ROOT / 'scripts').glob(f'{n}-*.sql'))]
            for name in names:
                if name.startswith('490-'):
                    # The captured scorer contains a guarded325 call. Install
                    # its two committed read-only helpers without running that
                    # historical migration's round or ledger writes.
                    source325 = (ROOT / 'scripts' / (
                        '325-arena-2026-09-19-terminal324-saved-progress-baseline-rerun.sql'
                    )).read_text()
                    start = source325.index('CREATE OR REPLACE FUNCTION '
                        'public.lab_arena_sep19_rerun325_archive_valid_v1()')
                    end = source325.index('-- Replace the exact rerun324 guard')
                    helpers325 = source325[start:end]
                    assert helpers325.count('CREATE OR REPLACE FUNCTION') == 2
                    cur.execute(helpers325)
                    cur.execute(progression.HISTORICAL.read_text())
                cur.execute((ROOT / 'scripts' / name).read_text())
            # Preserve the exact deployed ACL of this invoker trigger. The
            # historical superuser fixture has different default grants.
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_integrity_publication_guard_v1() '
                        'TO anon,authenticated,service_role')
            if seed_existing:
                infra.seed(conn)
            old = catalog(cur)
            before_rows = rows(cur)
            # Hosted migrations run through an owner-member role with hostile
            # default grants. Replacement must preserve the current ACLs.
            cur.execute('CREATE ROLE flat500_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO flat500_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO flat500_migrator')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE flat500_migrator IN SCHEMA public '
                        'GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            conn.commit()
            # Both a changed source and an invalid computed target must roll
            # the whole transaction back before any function or row changes.
            import re
            first_values = re.search(r"\('public\.[^']+','([0-9a-f]{64})','([0-9a-f]{64})'", SQL.read_text())
            assert first_values
            for digest, message in zip(first_values.groups(), ('exact preimage differs', 'target differs')):
                with pytest.raises(db[0].Error, match=message):
                    cur.execute(SQL.read_text().replace(digest, '0' * 64, 1))
                conn.rollback()
                assert catalog(cur) == old and rows(cur) == before_rows
            cur.execute('SET ROLE flat500_migrator')
            cur.execute(SQL.read_text())
            first = catalog(cur)
            assert rows(cur) == before_rows
            assert len(old) == len(first)
            changed = 0
            for before, after in zip(old, first):
                assert before[0] == after[0] and before[2:] == after[2:]
                changed += before[1] != after[1]
            assert changed == 25
            cur.execute(SQL.read_text())
            assert catalog(cur) == first and rows(cur) == before_rows
            cur.execute('RESET ROLE')
            # Keep the schema capability and embedded historical round JSON.
            for signature, definition, *_ in first:
                if LEGACY in definition and not signature.startswith((
                    'lab_arena_per_icp_cost_schema_v1(', 'lab_arena_open_scoring_v2('
                )):
                    assert FLAT in definition, signature
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def database():
    yield from database_with_flat_cost_migration(seed_existing=True)


def seed(conn, *, flat=True):
    cfg = infra.seed(conn)
    cfg['sourcing_cost_eligibility_policy'] = FLAT if flat else LEGACY
    cfg['cost_per_company_microusd'] = 4_000_000 if flat else 800_000
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',
                    (json.dumps(cfg), prior.ROUND))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return cfg


@pytest.mark.parametrize('qualified', range(6))
@pytest.mark.parametrize('amount,allowed', [(4_000_000, True), (4_000_001, False)])
def test_flat_successful_cost_boundary_for_every_qualified_count(database, qualified, amount, allowed):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=amount)['status'] == 'settled'
        result = infra.eligible(conn, qualified)
        assert result['eligible'] is allowed
        assert result['eligibility_cap_microusd'] == 4_000_000
        assert result['competition_sourcing_microusd'] == amount
        assert result['eligibility_reason'] == ('eligible' if allowed else 'execution_cap_exceeded')
        quota = infra.quota(conn, lease, token='a')
        assert len(quota) == 8
        assert quota['admission_cap_microusd'] == 4_000_000
        assert quota['per_qualified_pair_cap_microusd'] == 4_000_000


@pytest.mark.parametrize('amount,allowed', [(800_000, True), (800_001, False)])
def test_legacy_round_still_scales_cost_by_qualified_count(database, amount, allowed):
    with database[0].connect(**database[1]) as conn:
        seed(conn, flat=False)
        lease = prior.claim(conn)
        prior.pay(conn, lease, amount=amount)
        result = infra.eligible(conn, 1)
        assert result['eligible'] is allowed
        assert result['eligibility_cap_microusd'] == 800_000
        assert result['eligibility_reason'] == ('eligible' if allowed else 'cost_per_company_exceeded')


def test_flat_admission_aggregates_providers_and_attempts_and_retains_completion(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        prior.pay(conn, first, amount=3_000_000)
        prior.finish(conn, first, 'model_error', token='a')
        retry = prior.claim(conn, 'b')
        assert retry['attempt'] == 2
        # The final call starts below the cap and may report a larger true bill.
        paid = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',1,'{}'::jsonb,4500)",
                         (retry['run_id'], 'sha256:'+'b'*64, 'sha256:'+'f'*64))
        assert paid['status'] == 'reserved'
        assert infra.eligible(conn)['eligibility_reason'] == 'provider_calls_inflight'
        prior.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                  (retry['run_id'], 'sha256:'+'b'*64, 'sha256:'+'f'*64))
        settled = prior.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,1000000,%s::jsonb,4500)',
                            (retry['run_id'], 'sha256:'+'b'*64, 'sha256:'+'f'*64,
                             json.dumps({'status': 200, 'call_succeeded': True})))
        assert settled['status'] == 'settled'
        for provider, operation in (('deepline', 'deepline.execute'),
                                    ('scrapingdog', 'scrapingdog.scrape'),
                                    ('openrouter', 'openrouter.chat')):
            refused = prior.rpc(conn, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,1,%s::jsonb,4500)',
                                (retry['run_id'], 'sha256:'+'b'*64,
                                 'sha256:'+{'deepline':'1','scrapingdog':'2','openrouter':'3'}[provider]*64,
                                 operation, provider, 'miner_key', json.dumps({'tool':'exa_search'})))
            assert (refused['status'], refused['reason']) == ('refused', 'money_cap')
        assert infra.eligible(conn, 0)['eligible'] is True
        assert infra.state(conn)['successful_microusd'] == 4_000_000
        assert prior.finish(conn, retry, 'accepted')['status'] == 'accepted'


def test_flat_failed_billing_still_blocks_admission_without_changing_successful_cost(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert infra.failed_call(conn, lease, amount=4_000_000)['status'] == 'settled'
        assert infra.eligible(conn, 0)['competition_sourcing_microusd'] == 0
        assert infra.eligible(conn, 0)['eligible'] is True
        refused = prior.reserve(conn, lease, call='2')
        assert (refused['status'], refused['reason']) == ('refused', 'money_cap')


def test_flat_unknown_cost_fails_closed(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        prior.pay(conn, lease, unsettled=True)
        assert infra.eligible(conn, 0)['eligibility_reason'] == 'provider_cost_uncertain'


def test_flat_infrastructure_exemption_keeps_actual_ledger_cost(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        prior.host_loss(conn, amount=4_000_000)
        retry = prior.claim(conn, 'b')
        assert retry['status'] == 'leased'
        state = infra.state(conn)
        assert state['settled_microusd'] == 4_000_000
        assert state['competition_settled_microusd'] == 0
        assert infra.eligible(conn, 0)['eligible'] is True
        assert prior.pay(conn, retry, token='b', call='f', amount=4_000_000)['status'] == 'settled'
        assert infra.eligible(conn, 0)['competition_sourcing_microusd'] == 4_000_000


def test_flat_config_must_match_its_frozen_cap(database):
    with database[0].connect(**database[1]) as conn:
        cfg = seed(conn)
        cfg['cost_per_company_microusd'] = 800_000
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',
                        (json.dumps(cfg), prior.ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with pytest.raises(database[0].Error, match='lab_arena_per_icp_cost_policy_invalid'):
            infra.eligible(conn)
        conn.rollback()


@pytest.mark.parametrize('provider,operation,amount,doc', [
    ('deepline', 'deepline.execute', 0,
     {'reserve_remaining_budget': True, 'tool': 'exa_search'}),
    ('openrouter', 'openrouter.responses', 1,
     {'reserve_remaining_budget': True}),
])
def test_flat_dynamic_paid_calls_are_refused_at_exact_cap(database, provider, operation, amount, doc):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        prior.pay(conn, lease, amount=4_000_000)
        result = prior.rpc(conn, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)',
                           (lease['run_id'], 'sha256:'+'a'*64, 'sha256:'+'2'*64,
                            operation, provider, 'miner_key', amount, json.dumps(doc)))
        assert (result['status'], result['reason']) == ('refused', 'money_cap')
        assert infra.state(conn)['settled_microusd'] == 4_000_000
        assert infra.state(conn)['inflight_calls'] == 0


def test_openrouter_dynamic_zero_cannot_claim_to_be_free(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        prior.pay(conn, lease, amount=4_000_000)
        with pytest.raises(database[0].Error, match='lab_arena_dynamic_reserve_input_invalid'):
            prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.responses',"
                      "'openrouter','miner_key',0,%s::jsonb,4500)",
                      (lease['run_id'], 'sha256:'+'a'*64, 'sha256:'+'2'*64,
                       json.dumps({'reserve_remaining_budget': True})))
        conn.rollback()
        assert infra.state(conn)['settled_microusd'] == 4_000_000
        assert infra.state(conn)['inflight_calls'] == 0


def _capped_broker(database, *, transport=None):
    from lab_arena import broker as br
    from lab_arena.store import ArenaStore, PsycopgTransport
    from tests.lab_arena.test_lab_arena_broker import FakeTransport, HOST_KEYS, gpt6_luna_price_table
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        prior.pay(conn, lease, amount=4_000_000)
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    recorded = []
    reserve = store.reserve_call
    def recording_reserve(**kwargs):
        recorded.append(kwargs)
        return reserve(**kwargs)
    store.reserve_call = recording_reserve
    transport = transport or FakeTransport()
    broker = br.Broker(store=store, key_for=lambda provider: HOST_KEYS[provider],
                       price_table=gpt6_luna_price_table(), transport=transport,
                       lease_ttl_seconds=4500,
                       provider_funding_source_for=lambda _context, _provider: 'miner_key')
    context = br.RunContext(run_id=lease['run_id'], assignment_id=lease['assignment_id'],
                            icp_position=lease['icp_position'], lease_token_hash='sha256:'+'a'*64,
                            miner_hotkey=lease['miner_hotkey'], submission_id=lease['submission_id'],
                            stage=lease['stage'], attempt=lease['attempt'], round_id=prior.ROUND)
    return broker, context, store, transport, recorded


@pytest.mark.parametrize('kind', ['deepline_dynamic', 'openrouter_chat', 'openrouter_web'])
def test_broker_owns_paid_cost_classification_at_cap(database, kind):
    from tests.lab_arena.test_lab_arena_broker import CHAT, GPT6_LUNA_RESPONSES
    broker, context, store, transport, recorded = _capped_broker(database)
    try:
        if kind == 'deepline_dynamic':
            # These opaque provider fields are caller text. They cannot
            # replace the broker's own admission amount or dynamic marker.
            operation = 'deepline.execute'
            parameters = {'tool': 'exa_search', 'payload': {
                'query': 'example', 'amount_microusd': 0,
                'reserve_remaining_budget': False, 'free': True}}
        elif kind == 'openrouter_web':
            operation = 'openrouter.responses'
            parameters = {**GPT6_LUNA_RESPONSES, 'max_tool_calls': 1,
                          'tools': [{'type': 'openrouter:web_search', 'parameters': {
                              'engine': 'native', 'max_uses': 1, 'max_total_results': 5}}]}
        else:
            operation = 'openrouter.chat'
            parameters = CHAT
        result = broker.execute(context, operation_id=operation, parameters=parameters,
                                action_sequence=3, timeout_ms=5000)
        assert result.status == 402, result.call
        assert result.call['error_code'] == 'budget_refused'
        assert len(recorded) == 1 and transport.sent == []
        if kind == 'deepline_dynamic':
            assert recorded[0]['amount_microusd'] == 0
            assert recorded[0]['call_doc']['reserve_remaining_budget'] is True
        else:
            assert recorded[0]['amount_microusd'] > 0
            assert (recorded[0]['call_doc'].get('reserve_remaining_budget') is True) == (kind == 'openrouter_web')
    finally:
        store.close()


def test_broker_rejects_caller_openrouter_pricing_fields_at_cap(database):
    from tests.lab_arena.test_lab_arena_broker import CHAT
    broker, context, store, transport, recorded = _capped_broker(database)
    try:
        result = broker.execute(context, operation_id='openrouter.chat',
                                parameters={**CHAT, 'amount_microusd': 0,
                                            'reserve_remaining_budget': False, 'free': True},
                                action_sequence=3, timeout_ms=5000)
        assert result.status == 400
        assert result.call['error_code'] == 'invalid_request'
        assert recorded == [] and transport.sent == []
    finally:
        store.close()


def test_broker_trusted_free_tool_stays_free_after_cap(database):
    broker, context, store, transport, recorded = _capped_broker(database)
    try:
        result = broker.execute(context, operation_id='deepline.execute',
                                parameters={'tool': 'free_simple_company_search',
                                            'payload': {'query': 'example'}},
                                action_sequence=3, timeout_ms=5000)
        assert result.status == 200, result.call
        assert len(recorded) == len(transport.sent) == 1
        assert recorded[0]['amount_microusd'] == 0
        assert 'reserve_remaining_budget' not in recorded[0]['call_doc']
        assert result.call['actual_microusd'] == 0
        with database[0].connect(**database[1]) as conn:
            assert infra.state(conn)['settled_microusd'] == 4_000_000
            assert infra.eligible(conn, 0)['eligible'] is True
    finally:
        store.close()
