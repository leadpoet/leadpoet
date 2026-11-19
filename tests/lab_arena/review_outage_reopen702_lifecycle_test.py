"""A review-outage recovery uses real SQL claims, service scoring and publication.

The provider and sandbox are local deterministic fixtures. Intake, review spend,
admission, execution leases, accepted judgments, publication and reward activation
use the production service/store paths with both 701 and 702 installed.
Full-thinking capture and champion Git promotion have separate fixtures; this
scenario keeps the measured baseline ahead and exercises a no-king republication.
"""
import copy
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from lab_arena import contracts, rewards, runtime, scoring
from lab_arena.store import new_lease_token
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena import review_outage_reopen702_postgres_test as recovery
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.test_arena_company_quality_policy import _positive_breakdown

recovery_database = recovery.recovery_database
db = recovery.db


def submit_unreviewed(harness, flavor):
    """Use the same signed intake as Harness.submit without its passing review."""
    miner = fixtures.keypair('svc-miner-' + flavor)
    harness.chain.registered.append(miner.ss58_address)
    payload = fixtures.flavor_source_archive(flavor)
    facts = fixtures.source_bundle.validate_source_archive(payload)

    def request(scope, body):
        return contracts.build_signed_request(
            scope=scope, round_id=harness.round_id, hotkey=miner.ss58_address,
            body=body, timestamp=int(harness.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex())

    target = harness.service.handle_submission_presign(request(
        contracts.SCOPE_SUBMISSION_PRESIGN,
        {'source_size_bytes': facts['source_size_bytes'], 'consent': {'public_rerun': True}}))
    sid = target['submission_id']
    harness.flavors[sid] = flavor
    harness.objects.put(target['source_ref'], payload)
    result = harness.service.handle_submission_finalize(sid, request(
        contracts.SCOPE_SUBMISSION_FINALIZE, {
            'submission_id': sid, 'source_ref': target['source_ref'],
            'source_size_bytes': facts['source_size_bytes'],
            'credentials': {
                'openrouter_api_key': fixtures.CANARY_OPENROUTER_KEY,
                'openrouter_management_key': fixtures.CANARY_OPENROUTER_MANAGEMENT_KEY,
                'deepline_api_key': fixtures.CANARY_DEEPLINE_KEY,
            },
        }))
    assert result['status'] == 'accepted'
    assert harness.service.store.get_submission(sid)['code_review_status'] == 'pending'
    return sid, miner.ss58_address


def install_deterministic_company_judges(harness):
    original = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get('schema_version') != scoring.SCORING_INPUT_SCHEMA_VERSION:
            result = original(spec, **kwargs)
            rows = json.loads(result.output_bytes)['companies']
            for row in rows:
                slug = row['company_name'].lower().replace(' ', '-')
                row.update(company_linkedin='https://linkedin.com/company/' + slug, state='CA')
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps({'companies': rows}).encode())

        def judge(companies, buyer, reference):
            rows = []
            for company in companies:
                name = company['company_name']
                score = 60.0 if name.startswith('PublicBaseline') else 10.0
                raw = _positive_breakdown(name, urlsplit(company['company_website']).hostname,
                                          name.lower().replace(' ', '-'))
                raw.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
                raw['intent_signals_detail'][0].update(raw=score, after_decay=score)
                rows.append(raw)
            return apply_company_judgment_context(companies, rows, contacts_required=False)

        judge.company_quality = judge.integrity_policy = True
        judge.contacts_required = False
        full, new = scoring.score_quality_work_item(
            {'scored_run_id': document['scored_run_id']}, icp=document['icp'],
            companies=document['companies'], scorer=judge,
            cache_context=document['company_judgment_cache'])
        output = scoring.build_scoring_output(document['scored_run_id'], full, company_judgments=new)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())

    harness.sandbox.run_icp = run_icp


def test_combined_recovery_executes_judges_republishes_and_preserves_next_round(db, tmp_path):
    pg, connection, _ = db
    # The review fixture deliberately omits unrelated reward-write upgrades.
    # Install their exact deployed function patches for this lifecycle proof.
    with connection.cursor() as cursor:
        for filename, block in (
            ('592-arena-2026-10-29-authorized-champion.sql', 'oct29_snapshot592'),
            ('696-lab-arena-correction-reward-history.sql', 'activation696'),
        ):
            source = (recovery.SCRIPTS / filename).read_text()
            patch = re.search(r'DO \$' + block + r'\$.*?\$' + block + r'\$;', source, re.S)
            assert patch is not None
            cursor.execute(patch.group())
        cursor.execute((recovery.SCRIPTS / '681-lab-arena-total-emissions-reward-slots.sql').read_text())
    harness = QualityHarness(lambda: pg.connect(**connection.get_dsn_parameters()), tmp_path,
                             challengers=['ReviewRecovered'], runners=['alpha', 'beta'])
    store = harness.service.store
    harness.service.config.defaults = replace(
        harness.service.config.defaults, benchmark_icp_count=10, rewards_enabled=True,
        runner_slot_ceiling=10, execution_sequence_from='2026-01-01T00:00:00Z',
        benchmark_disclosure_from='2026-01-01T00:00:00Z')
    bank = copy.deepcopy(daily_icps()[:10])
    harness.service.config.daily_icp_source = lambda **kwargs: {
        'status': 'ready', 'set_id': int(kwargs['set_id']), 'icps': copy.deepcopy(bank)}
    install_deterministic_company_judges(harness)
    original_runner = harness.runner

    def runner(index, parallel=4):
        result = original_runner(index, parallel)
        result._config.proxy_worker_pool = _verified_test_pool(parallel)
        return result

    harness.runner = runner
    harness.chain.epoch = 40_000
    harness.clock.now = datetime.now(timezone.utc)
    cutoff = harness.clock.now + timedelta(seconds=5)
    rid = 'arena-' + cutoff.strftime('%Y-%m-%d') + '-r702lifecycle'
    harness.round_id = rid
    configuration = harness.service.create_round(cutoff, round_id=rid)
    recovery.scalar(connection, 'SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)', (
        configuration['scorer_image_digest'], configuration['scorer_image_reference'],
        configuration['scorer_image_digest'], configuration['scorer_image_reference']))
    sid, miner = submit_unreviewed(harness, 'ReviewRecovered')
    for attempt in range(1, 7):
        token = new_lease_token()
        claimed = store.begin_submission_review(sid, miner, token, 700_000,
            recovery.DS, 2, 1000, recovery.KIMI)
        assert claimed['status'] == 'claimed' and claimed['attempt'] == attempt
        diagnostic = {**recovery.failure(recovery.DS, 'code_review_provider_unavailable'),
                      'provider_http_status': 404}
        assert store.finish_submission_review(sid, miner, token, 'error', diagnostic, None)['status'] == 'error'
        recovery._age_review_error(connection, sid)
    old_review_ledger = store.list_ledger(submission_id=sid)
    assert sum(row['entry_kind'] == 'uncertain' for row in old_review_ledger) == 6
    assert sum(row['amount_microusd'] for row in old_review_ledger
               if row['entry_kind'] == 'uncertain') == 4_200_000
    assert store.get_submission(sid)['code_review_attempts'] == 6
    # Reproduce the historical baseline-only result through an explicit release
    # of 701's hold. No rejected row or published authority is fabricated.
    time.sleep(max(0, (cutoff - datetime.now(timezone.utc)).total_seconds()) + 0.05)
    harness.clock.advance_to(configuration['schedule']['benchmark_deadline'])
    assert store.hold_round_review(rid)['status'] == 'held'
    assert store.decide_review_hold(rid, 'release')['status'] == 'released'
    historical = harness.advance_until('published', runners=2, max_steps=40)
    assert historical['publication_doc']['king_decision']['outcome'] == 'no_king'
    assert store.get_submission(sid)['status'] == 'rejected'
    assert store.get_submission(sid)['rejection_rule'] == 'code_review_incomplete'
    assert harness.service.activate_reward(rid)['status'] == 'activated'
    historical = store.get_round(rid)
    baseline_runs = store.list_runs(rid)
    assert baseline_runs and all(row['stage'] == 1 for row in baseline_runs)
    assert all(row['status'] == 'accepted' for row in baseline_runs)
    baseline = historical['participants'][0]['submission_id']
    next_id = 'arena-' + (cutoff + timedelta(days=1)).strftime('%Y-%m-%d') + '-r702next'
    harness.service.create_round(cutoff + timedelta(days=1), round_id=next_id)
    next_before = store.get_round(next_id)
    times = recovery.deadlines(review_seconds=3)
    assert recovery.reopen(connection, rid, times=times)['status'] == 'reopened'
    assert store.list_runs(rid) == baseline_runs
    assert store.list_ledger(submission_id=sid) == old_review_ledger
    assert store.get_round(next_id) == next_before
    old_basis = historical['reward_basis_doc']
    assert harness.service.public_reward_basis(historical['effective_reward_epoch']) == old_basis
    token = new_lease_token()
    claim = recovery.alternate.begin(store, sid, miner, recovery.DS, token)
    assert claim['status'] == 'claimed' and claim['attempt'] == 7
    assert store.finish_submission_review(sid, miner, token, 'passed',
        recovery.verdict(recovery.DS), 1234)['status'] == 'passed'
    assert store.advance_review_outage(rid)['admitted'] == [sid]
    harness.service._invalidate_hot_round()
    harness.clock.now = max(datetime.now(timezone.utc), times[0])
    assert harness.service.advance_round(rid)['status'] == 'ok'
    assert harness.status() == 'stage2'
    pending = store.list_runs(rid, stage=2, kind='execute')
    assert len(pending) == 10 and {row['submission_id'] for row in pending} == {sid}
    assert all(row['status'] == 'pending' for row in pending)
    # Actual signed runner claims, sandbox outputs and accepted judgments.
    harness.run_stage_with_runners(2)
    executions = store.list_runs(rid, stage=2, kind='execute')
    assert len(executions) == 10 and all(row['status'] == 'accepted' for row in executions)
    assert all(row['claim_request_id'] and row['output_ref'] for row in executions)
    republished = harness.advance_until('published', runners=2, max_steps=40)
    assert republished['publication_doc']['king_decision']['outcome'] == 'no_king'
    ranking = republished['publication_doc']['final_ranking']
    assert {row['submission_id'] for row in ranking} == {baseline, sid}
    assert all(row['final_score'] is not None for row in ranking)
    judges = store.list_runs(rid, stage=2, kind='score')
    assert judges and all(row['status'] == 'accepted' for row in judges)
    assert {row['scored_run_id'] for row in judges} == {row['run_id'] for row in executions}
    assert [row for row in store.list_runs(rid) if row['stage'] == 1] == baseline_runs
    harness.chain.epoch += 10
    assert harness.service.activate_reward(rid)['status'] == 'activated'
    activated = store.get_round(rid)
    assert activated['effective_reward_epoch'] > historical['effective_reward_epoch']
    assert activated['reward_basis_hash'] != historical['reward_basis_hash']
    bases = store.published_reward_bases(mode='live', network_name=configuration['network_name'],
                                       netuid=configuration['netuid'])
    docs = [row['reward_basis_doc'] for row in bases]
    assert rewards.governing_reward_basis(docs, activated['effective_reward_epoch'] - 1) == old_basis
    assert rewards.governing_reward_basis(docs, activated['effective_reward_epoch']) == activated['reward_basis_doc']
    assert store.get_round(next_id) == next_before
    assert store.freeze_champion_funding(next_id)['status'] == 'frozen'
    final_ledger = {row['entry_id']: row for row in store.list_ledger(submission_id=sid)}
    assert all(final_ledger[row['entry_id']] == row for row in old_review_ledger)
    assert store.get_submission(sid)['code_review_attempts'] == 7
    for key in ('scoring_generation_floor', 'stage1_scoring_plan_doc', 'benchmark_ref'):
        assert activated[key] == historical[key]
    before_policy, after_policy = copy.deepcopy(configuration), copy.deepcopy(activated['configuration_doc'])
    before_policy.pop('schedule')
    after_policy.pop('schedule')
    assert before_policy == after_policy
    fixtures.assert_canary_absent(harness, harness.connect)
    store.close()
