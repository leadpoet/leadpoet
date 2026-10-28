"""Real lease and credit gates for durable legacy full-group review checkpoints."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scorer_entrypoint, scoring
from lab_arena.service import ArenaService, ServiceConfig, S3ObjectStore, ServiceError
from lab_arena.store import ArenaStore, PsycopgTransport
from qualification.scoring.competition import apply_company_judgment_context
from qualification.scoring import lead_scorer
from tests.lab_arena import zero_call_setup_recovery467_postgres_test as deployed
from tests.lab_arena.company_judgments_postgres_test import _frozen_participants
from tests.lab_arena.partial_company_judgments_postgres_test import _sign
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment
from tests.lab_arena.test_lab_arena_migration_postgres import _commit_plan, _execute_everything, claim, commit_round, round_config
from tests.lab_arena.test_lab_arena_object_store import _ConditionalS3
from tests.lab_arena.test_lab_arena_service_round import keypair, wallet_verify, FakeChain
from tests.test_arena_score_integrity import _icp, _positive_breakdown, _public_company

SQL = Path(__file__).parents[2] / 'scripts/468-lab-arena-legacy-review-checkpoint.sql'


@pytest.fixture(scope='module')
def database():
    generator = deployed.database.__wrapped__()
    value = next(generator)
    try:
        with value[0].connect(**value[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        with value[0].connect(**value[1]) as conn:
            deployed._reset_control(conn)
        yield value
    finally:
        generator.close()


@pytest.fixture()
def store(database):
    psycopg, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))


def _setup(store, marker):
    round_id = 'arena-2026-10-12-l' + marker.replace('_', '')
    runner_key = keypair('legacy-checkpoint-' + marker)
    config = round_config(round_id, [runner_key.ss58_address], cost_per_company_microusd=1000)
    config['runner_slot_ceiling'] = 8
    config['lease_ttl_seconds'] = contracts.LEASE_TTL_SECONDS
    config.update(integrity_policy='arena_integrity_v1', scorer_policy=scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2'))
    config['schedule']['stage_1_scoring_close'] = '2030-01-01T23:00:00Z'
    assert store.create_round(round_id, config)['status'] == 'created'
    participants = _frozen_participants(store, round_id, 1, prefix=marker)
    commit_round(store, round_id, participants)
    assert store.open_stage(round_id, 1, participants, list(contracts.stage_positions(1)))['status'] == 'ok'
    executed = _execute_everything(store, round_id, runner_key.ss58_address)
    assert store.close_stage(round_id, 1)['status'] == 'closed'
    _commit_plan(store, round_id, 1)
    execution = store.get_run(executed[(participants[0]['submission_id'], 0)])
    companies = [_public_company(name, domain=name.lower()+'.example') for name in ('A', 'B')]
    objects = S3ObjectStore('test-arena', client=_ConditionalS3())
    objects.put(execution['output_ref'], json.dumps({'schema_version': 'leadpoet.lab_arena.output.v1', 'companies': companies}).encode())
    scoring_input = scoring.build_scoring_input(scored_run_id=execution['run_id'], icp=_icp(), companies=companies, policy=config['scorer_policy'], evaluation_date='2026-09-02')
    scope = judgment_cache.build_cache_scope(scoring_input=scoring_input, round_id=round_id, network_name='finney', netuid=71, scorer_image_digest=config['scorer_image_digest'], scorer_image_reference=config['scorer_image_reference'], integrity_policy='arena_integrity_v1')
    item = {'scored_run_id': execution['run_id'], 'submission_id': execution['submission_id'], 'icp_position': 0, 'output_ref': execution['output_ref'],
            'judgment_cache_key': scope['cache_key'], 'judgment_scope_doc': scope, 'judgment_input_hash': scope['scoring_input_hash'], 'judgment_group_leader': True, 'judgment_group_miner_hotkeys': [execution['miner_hotkey']]}
    assert store.open_scoring(round_id, 1, [item], integrity_cache=True)['status'] == 'ok'
    lease, token, _, _ = claim(store, round_id, runner_key.ss58_address, excluded=[runner_key.ss58_address])
    assert lease['status'] == 'leased'
    service = ArenaService(ServiceConfig(mode='live', store=store, object_store=objects, signer=None, chain=FakeChain([runner_key.ss58_address]), verify_signature=wallet_verify, daily_icp_source=lambda **_: {}, banned_hotkeys_source=lambda: (), broker_factory=lambda *_: None))
    service.evaluation_icps = service.benchmark_icps = lambda *_: [_icp()]*20
    rows = apply_company_judgment_context(companies, [_positive_breakdown('A', 'a.example'), _positive_breakdown('B', 'b.example')])
    return service, round_id, runner_key, lease, token, rows, companies


def _completion(round_id, key, lease, token, output, terminal='judge_error'):
    result = {'schema_version': contracts.RUN_RESULT_SCHEMA_VERSION, 'resource_summary': {'wall_seconds': 1.0, 'cpu_seconds': 0.0, 'max_rss_bytes': 1024, 'stdout_bytes': 0, 'stderr_bytes': 0, 'provider_call_count': 1},
              'started_at': '2026-10-12T00:00:00Z', 'finished_at': '2026-10-12T00:00:01Z', 'terminal_status': terminal}
    return _sign(round_id, key, {'run_id': lease['run_id'], 'lease_token': token, 'result': result, 'output': output})


def test_signed_failure_persists_then_retry_scores_only_unresolved_and_final_cache_accepts(store, monkeypatch):
    # The hermetic scorer uses sandbox placeholders, independent of host keys.
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    service, round_id, key, lease, token, rows, companies = _setup(store, 'life')
    output = scoring.build_scoring_failure(lease['scored_run_id'], 'judge_error', terminal_breakdowns=rows[:1])
    request = _completion(round_id, key, lease, token, output)
    assert service.handle_complete(request)['status'] == 'failed'
    failed = store.get_run(lease['run_id'])
    assert failed['per_icp_score'] is None
    assert store.get_judgment_cache(failed['judgment_cache_key']) is None
    assert service.handle_complete(request)['idempotent'] is True
    claim_request = contracts.build_signed_request(scope=contracts.SCOPE_CLAIM, round_id=round_id, hotkey=key.ss58_address, body={'declared_parallelism': 1}, timestamp=int(datetime.now(timezone.utc).timestamp()), sign_message=lambda message: key.sign(message.encode()).hex())
    retry = service.handle_claim(claim_request)
    retry_token = retry['lease_token']
    assert retry['status'] == 'leased' and retry['attempt'] == 2
    prior = retry['prior_terminal_breakdowns']
    document, _ = service._legacy_partial_input(store.get_run(retry['run_id']), store.get_round(round_id))
    document['prior_terminal_breakdowns'] = prior
    attempted = []
    async def judge(**kwargs):
        name = kwargs['company'].company_name
        attempted.append(name)
        return _positive_breakdown(name, name.lower()+'.example')
    monkeypatch.setattr(lead_scorer, 'score_company_competition_intent', judge)
    final = scorer_entrypoint.score_input(document)
    assert attempted == ['B']
    assert [row['company_index'] for row in final['breakdowns']] == [0, 1]
    assert service.handle_complete(_completion(round_id, key, retry, retry_token, final, 'accepted'))['status'] == 'accepted'
    accepted = store.get_run(retry['run_id'])
    assert store.get_judgment_cache(accepted['judgment_cache_key'])['source_score_run_id'] == accepted['run_id']
    verified = service._verified_breakdowns(accepted, icp=_icp(), companies=companies, policy=document['scorer_policy'])
    assert verified == final['breakdowns']
    assert store.get_run(lease['run_id'])['result_doc'] == failed['result_doc']


@pytest.mark.parametrize('change', ['wrong_run', 'retryable', 'expired', 'claim', 'open_accounting'])
def test_uncommitted_or_invalid_partial_artifact_cannot_resume(store, database, change):
    service, round_id, key, lease, token, rows, _ = _setup(store, change)
    output = scoring.build_scoring_failure(lease['scored_run_id'], 'judge_error', terminal_breakdowns=rows[:1])
    psycopg, dsn = database
    if change == 'wrong_run':
        output['scored_run_id'] = 'foreign-execution'
    elif change == 'retryable':
        output['terminal_breakdowns'][0].update(final_score=0, failure_reason='provider timeout')
    elif change == 'claim':
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET claim_request_hash=%s WHERE run_id=%s", ('sha256:'+'f'*64, lease['run_id']))
        # A valid receipt binds the current claim, so the negative is a direct
        # privileged-RPC caller's fabricated receipt rather than worker fields.
        run = store.get_run(lease['run_id'])
        document, execution_hash = service._legacy_partial_input(run, store.get_round(round_id))
        receipt = service._legacy_partial_receipt(run, store.get_round(round_id), document, execution_hash, output, 'sha256:'+'e'*64)
        receipt['source_claim_request_hash'] = 'sha256:'+'a'*64
        from lab_arena.store import ArenaStoreError, hash_lease_token
        with pytest.raises(ArenaStoreError, match='complete_input_invalid'):
            store.complete_attempt(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token), result={'terminal_status': 'judge_error', 'legacy_partial_checkpoint': receipt}, terminal_cause='judge_error', output_ref=receipt['source_output_ref'])
        assert store.get_run(lease['run_id'])['status'] == 'leased'
        return
    elif change == 'expired':
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
    else:
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) VALUES ('reservation',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','miner_key',0)", (lease['miner_hotkey'], round_id, lease['submission_id'], lease['run_id'], contracts.document_hash('open')))
    request = _completion(round_id, key, lease, token, output)
    if change in ('wrong_run', 'retryable'):
        with pytest.raises(ServiceError):
            service.handle_complete(request)
    else:
        assert service.handle_complete(request)['status'] in ('stale', 'accounting_open')
    run = store.get_run(lease['run_id'])
    assert run['status'] == 'leased' and not (run.get('result_doc') or {}).get('legacy_partial_checkpoint')


@pytest.mark.parametrize('change', ['valid', 'wrong_owner', 'unproved_credit', 'charged_credit', 'receipt', 'artifact', 'scope', 'deadline'])
def test_owner_credit_retry_requires_bound_checkpoint_and_original_credit_gates(store, database, change):
    from tests.lab_arena.partial_company_credit_retry_postgres_test import _retry_request
    service, round_id, key, lease, token, rows, _ = _setup(store, 'c'+change)
    output = scoring.build_scoring_failure(lease['scored_run_id'], 'judge_error', terminal_breakdowns=rows[:1])
    assert service.handle_complete(_completion(round_id, key, lease, token, output, 'credential_error'))['status'] == 'failed'
    source = store.get_run(lease['run_id'])
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        response = {'status': 402, 'call_succeeded': False}
        charge = 7 if change == 'charged_credit' else 0
        if change != 'unproved_credit':
            response['credit_failure_proof'] = {'schema_version': 'leadpoet.lab_arena.credit_failure_proof.v1', 'provider': 'openrouter', 'reason': 'out_of_credit', 'provider_status': 402, 'actual_microusd': charge}
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,terminal_response) VALUES ('settlement',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','miner_key',%s,%s::jsonb)", (lease['miner_hotkey'], round_id, lease['submission_id'], lease['run_id'], contracts.document_hash('credit-'+change), charge, json.dumps(response)))
        if change in ('receipt', 'scope', 'deadline'):
            cur.execute('SET session_replication_role=replica')
            if change == 'receipt':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc=result_doc-'legacy_partial_checkpoint' WHERE run_id=%s", (lease['run_id'],))
            elif change == 'scope':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{legacy_partial_checkpoint,scoring_input_hash}',%s::jsonb) WHERE run_id=%s", (json.dumps('sha256:'+'a'*64), lease['run_id']))
            else:
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}','\"2020-01-01T23:00:00Z\"'::jsonb) WHERE round_id=%s", (round_id,))
            cur.execute('SET session_replication_role=origin')
    if change == 'artifact':
        service._objects._client.objects[source['output_ref']] = b'{"tampered":true}'
    owner = keypair('c'+change+'-miner-0')
    assert owner.ss58_address == lease['miner_hotkey']
    if change == 'wrong_owner':
        owner = keypair('wrong-legacy-credit-owner')
    service._invalidate_hot_round()
    request = _retry_request(round_id, owner, lease['submission_id'])
    if change == 'wrong_owner':
        with pytest.raises(ServiceError, match='credit_retry_owner_required'):
            service.handle_submission_credit_retry(lease['submission_id'], request)
    elif change in ('artifact', 'scope'):
        with pytest.raises(ServiceError, match='credit_retry_source_unavailable'):
            service.handle_submission_credit_retry(lease['submission_id'], request)
    elif change != 'valid':
        assert service.handle_submission_credit_retry(lease['submission_id'], request)['status'] == 'no_eligible'
    else:
        before_ledger = store.list_ledger(run_id=lease['run_id'])
        assert service.handle_submission_credit_retry(lease['submission_id'], request) == {'status': 'queued', 'requeued_count': 1}
        assert service.handle_submission_credit_retry(lease['submission_id'], request) == {'status': 'replayed', 'requeued_count': 1}
        retry, _, _, _ = claim(store, round_id, key.ss58_address, excluded=[key.ss58_address])
        assert retry['status'] == 'leased' and retry['attempt'] == 2
        assert service._prior_legacy_terminal_breakdowns(store.get_run(retry['run_id']), store.get_round(round_id)) == rows[:1]
        assert store.list_ledger(run_id=lease['run_id']) == before_ledger
        assert store.get_run(lease['run_id'])['result_doc'] == source['result_doc']


@pytest.fixture()
def round_database():
    # Separate complete rounds preserve their own baseline packages and reward
    # state. Reuse the exact deployed schema fixture, including migration468.
    generator = database.__wrapped__()
    value = next(generator)
    try:
        yield value
    finally:
        generator.close()


@pytest.mark.parametrize('failure', ['judge_error', 'judge_timeout'])
def test_legacy_partial_round_publishes_original_scores_costs_and_rewards(round_database, tmp_path, monkeypatch, failure):
    from dataclasses import replace
    from datetime import timedelta
    from threading import Lock
    from urllib.parse import urlsplit
    from lab_arena import runtime, weight_state
    from lab_arena.promotion import GitPromoter
    from leadpoet_canonical import arena_weights
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena.historical_service_harness import HistoricalStoreHarnessMixin, freeze_historical_intake_dates
    class FrozenHarness(HistoricalStoreHarnessMixin, IntegrityHarness):
        def build_service(self):
            service = super().build_service()
            commit = service.commit_benchmark
            def commit_frozen(round_id):
                freeze_historical_intake_dates(self, "2026-10-12")
                return commit(round_id)
            service.commit_benchmark = commit_frozen
            return service
    from tests.lab_arena import test_lab_arena_service_round as fixtures

    psycopg, dsn = round_database
    connect = lambda: psycopg.connect(**dsn)
    harness = FrozenHarness(connect, tmp_path, challengers=['Recovered'], runners=['alpha', 'beta'])
    harness.chain.epoch = 25080
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.config.defaults = replace(harness.service.config.defaults, rewards_enabled=True)
    original_runtime = harness.sandbox.run_icp
    state = {'target': None, 'failed': False, 'document': None}
    attempted = []
    leases = []
    scoring_lock = Lock()

    async def judge(**kwargs):
        company = kwargs['company']
        name = company.company_name
        document = state['document']
        attempted.append((document['scored_run_id'], name))
        if (document['scored_run_id'] == state['target'] and not state['failed']
            and name == 'Recovered Company 1'):
            raise RuntimeError('provider timeout')
        score = 40.0 if name.startswith('PublicBaseline') else 80.0
        row = _positive_breakdown(name, urlsplit(company.company_website).hostname)
        row.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
        row['intent_signals_detail'][0].update(raw=score, after_decay=score)
        return row

    monkeypatch.setattr(lead_scorer, 'score_company_competition_intent', judge)

    def run_icp(spec, **kwargs):
        with scoring_lock:
            document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
            base = original_runtime(spec, **kwargs)
            if document.get('schema_version') != scoring.SCORING_INPUT_SCHEMA_VERSION:
                return base
            assert 'company_quality_policy' not in document['scorer_policy']
            assert 'company_judgment_cache' not in document
            state['document'] = document
            if state['target'] is None and document['companies'][0]['company_name'].startswith('Recovered'):
                state['target'] = document['scored_run_id']
            leases.append(copy.deepcopy(document))
            output = scorer_entrypoint.score_input(document)
            timed_out = False
            if 'failure' in output:
                assert document['scored_run_id'] == state['target'] and not state['failed']
                assert [row['company_index'] for row in output['terminal_breakdowns']] == [0]
                state['failed'] = True
                timed_out = failure == 'judge_timeout'
            return runtime.fake_result(exit_code=0, timed_out=timed_out, output_bytes=json.dumps(output).encode())

    harness.sandbox.run_icp = run_icp
    round_id = 'arena-2026-12-02-legacy' + ('err' if failure == 'judge_error' else 'time')
    configuration = harness.service.create_round(harness.clock.now + timedelta(minutes=30), round_id=round_id)
    harness.round_id = round_id
    assert 'company_quality_policy' not in configuration
    # Keep code review eligible without closing the submission window.
    harness.clock.now = datetime.fromisoformat(configuration['schedule']['submission_cutoff'].replace('Z', '+00:00'))-timedelta(minutes=15)
    challenger = harness.submit('Recovered', round_id)
    harness.clock.advance_to(harness.schedule()['submission_cutoff'])
    harness.advance_until('scored', runners=2, max_steps=120)
    failed = next(run for run in harness.service.store.list_runs(round_id, kind='score') if run['status'] == 'failed')
    assert failed['scored_run_id'] == state['target']
    assert failed['terminal_cause'] == failure and failed['per_icp_score'] is None
    assert failed['result_doc']['legacy_partial_checkpoint']['terminal_count'] == 1
    retried = [document for document in leases if document['scored_run_id'] == state['target']]
    assert len(retried) == 2
    assert [row['company_index'] for row in retried[1]['prior_terminal_breakdowns']] == [0]
    target_names = [name for run_id, name in attempted if run_id == state['target']]
    assert target_names.count('Recovered Company 0') == 1
    assert target_names.count('Recovered Company 1') == scoring.MAX_JUDGE_RETRIES+1
    assert all(target_names.count('Recovered Company '+str(index)) == 1 for index in (2, 3, 4))
    source_before = copy.deepcopy(failed)
    costs_before = harness.service.store.submission_costs(challenger)
    harness.service = harness.build_service()
    published = harness.advance_until('published', runners=2, max_steps=120)
    assert published['king_outcome'] == 'crowned'
    ranking = {row['submission_id']: row for row in published['publication_doc']['final_ranking']}
    assert ranking[challenger]['final_score'] == 80.0
    assert ranking[challenger]['cost_summary']['qualified_company_count'] == 100
    assert published['publication_doc']['final_ranking'][0]['submission_id'] == challenger
    results = harness.service.public_results(round_id, challenger)
    assert len(results['outputs']) == contracts.BENCHMARK_ICP_COUNT
    assert all(float(row['per_icp_score']) == 80.0 for stage in results['scores'].values() for row in stage)
    executed = [run for run in harness.service.store.list_runs(round_id, kind='execute') if run['submission_id'] == challenger]
    assert len(executed) == 20 and all(run['status'] == 'accepted' and run['per_icp_score'] == 80.0 for run in executed)
    assert harness.service.store.submission_costs(challenger) == costs_before
    assert ranking[challenger]['cost_summary']['execution'] == harness.service._cost_kind_summary(costs_before, 'execute', include_successful=True)
    assert ranking[challenger]['cost_summary']['judge'] == harness.service._cost_kind_summary(costs_before, 'score', include_successful=True)
    assert harness.service.store.get_run(failed['run_id'])['result_doc'] == source_before['result_doc']
    fixtures.assert_canary_absent(harness, connect)

    repository_root = tmp_path / 'promotion-repository'
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / 'promotion-objects')
    harness.clock.now = datetime.fromisoformat(str(published['published_at']).replace('Z', '+00:00'))
    assert harness.service.promote_pending_baselines() == {'status': 'ok', 'promoted': 1}
    assert harness.service.activate_reward(round_id)['status'] == 'activated'
    basis = harness.service.store.get_round(round_id)['reward_basis_doc']
    burn = fixtures.keypair('legacy-partial-burn').ss58_address
    accepted = weight_state.build_accepted_weight_state(harness.signer, network='finney', genesis_hash='1'*64, netuid=71, epoch=int(basis['effective_reward_epoch']), valid_from_block=1, valid_until_block=360, reward_basis=basis, burn_hotkey=burn, issued_at=basis['published_at'])
    arena_weights.verify_accepted_weight_state_signature(accepted, public_key_der=harness.signer.public_key_der, expected_public_key_hash=harness.signer.public_key_hash)
    vector = arena_weights.derive_arena_weights(accepted, [basis['king_hotkey'], burn])
    assert vector['champion_share_ppb'] > 0
