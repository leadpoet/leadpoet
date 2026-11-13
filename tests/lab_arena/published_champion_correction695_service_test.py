"""Seven corrected judgments follow normal publication and reward activation."""
import copy
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from lab_arena import contracts, intent_details_policy, runtime, scoring
from lab_arena.promotion import GitPromoter
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena import published_champion_correction695_postgres_test as native
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.test_arena_company_quality_policy import _positive_breakdown

database = native.database
prepared = native.prepared
prepared_company = native.prepared_company
case = native.case
base_extended = native.base_extended
extended = native.extended
provider_case = native.provider_case
plain_finite = native.plain_finite
before = native.before
post = native.post
prior_bug = native.prior_bug
budget_parent = native.budget_parent
corrected = native.corrected
published = native.published
RUNTIME = 'sha256:' + '6' * 64


def parent_request(p, parent, request_id):
    with p.c.cursor() as q:
        row = native.old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s', (parent['run_id'],))
        digest = 'sha256:' + native.old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (parent['run_id'],))
        checkpoints = native.old.scalar(q, "SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::text,encode(extensions.digest(to_jsonb(j)::text,'sha256'),'hex')),'{}') FROM lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(%s::jsonb) ref WHERE ref->>'cache_key'=j.cache_key)", (json.dumps(row['company_judgment_refs']),))
        signals = native.old.scalar(q, "SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::text,encode(extensions.digest(to_jsonb(j)::text,'sha256'),'hex')),'{}') FROM lab_arena_signal_judgments j WHERE source_scored_run_id=%s", (row['scored_run_id'],))
    p.c.commit()
    return dict(parent_run_id=row['run_id'], parent_sha256=digest, request_id=request_id,
                checkpoints=checkpoints, signal_checkpoints=signals)


@pytest.mark.parametrize('winner', [True, False], ids=['new-champion', 'no-winner'])
def test_seven_corrections_republish_and_activate_prospectively(published, tmp_path, winner):
    p = published
    if p.organizer or p.parent_category != 'ordinary':
        pytest.skip('payer/provider lineages are covered by the native correction controls')
    with p.c.cursor() as q:
        q.execute('TRUNCATE lab_arena_rounds,lab_arena_ledger,lab_arena_trajectory_events CASCADE')
    p.c.commit()
    connect = lambda: p.pg.connect(**p.dsn)
    flavors = ['Miner' + letter for letter in 'ABCDEFG']
    harness = QualityHarness(connect, tmp_path, challengers=flavors, runners=['alpha', 'beta'])
    harness.service.config.defaults = replace(
        harness.service.config.defaults, benchmark_icp_count=10, promotion_margin=5.0,
        runner_slot_ceiling=10, pool_percent=30, rewards_enabled=True,
        per_icp_cost_policy=True, normalize_intent_scale=False, contacts_from=None,
        company_quality_from='2026-01-01T00:00:00Z', intent_details_from='2026-01-01T00:00:00Z',
        execution_sequence_from='2026-01-01T00:00:00Z', benchmark_disclosure_from='2026-01-01T00:00:00Z')
    bank = copy.deepcopy(daily_icps()[:10])
    for index, row in enumerate(bank):
        row.update(icp_id=f'icp_20260923_{index + 1:03d}', prompt=f'Public fixture ICP {index + 1}.')
    harness.service.config.daily_icp_source = lambda **kwargs: {
        'status': 'ready', 'set_id': int(kwargs['set_id']), 'icps': copy.deepcopy(bank)}
    original_runner = harness.runner

    def verified_runner(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = verified_runner
    original_run = harness.sandbox.run_icp
    recovering = False
    corrected_inputs = []

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get('schema_version') == scoring.SCORING_INPUT_SCHEMA_VERSION:
            is_miner = ':sub-' in document['scored_run_id']
            position = int(document['scored_run_id'].rsplit(':', 2)[-2])
            must_fail = is_miner and position == 6 and not recovering
            if recovering and is_miner and position == 6:
                corrected_inputs.append(document['scored_run_id'])

            def judge(companies, buyer, reference):
                rows = []
                for company in companies:
                    name = company['company_name']
                    if must_fail and name.endswith(' Company 1'):
                        raise scoring.ScoringError('fixture invalid citation', failure_reason='malformed_response')
                    baseline = name.startswith('PublicBaseline')
                    score = (1.0 if winner else 60.0) if baseline else (
                        100.0 if recovering and name.startswith('MinerA ') else 10.0)
                    slug = name.casefold().replace(' ', '-')
                    raw = _positive_breakdown(name, str(urlsplit(company['company_website']).hostname), slug)
                    raw.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
                    raw['intent_signals_detail'][0].update(raw=score, after_decay=score)
                    raw['verifier_gate_receipts'].append({'gate': 'intent_details', 'decision': 'match'})
                    rows.append(raw)
                return apply_company_judgment_context(companies, rows, contacts_required=False)

            judge.company_quality = True
            judge.integrity_policy = True
            judge.contacts_required = False
            completed = []
            try:
                full, new = scoring.score_quality_work_item(
                    {'scored_run_id': document['scored_run_id']}, icp=document['icp'],
                    companies=document['companies'], scorer=judge, cache_context=document['company_judgment_cache'],
                    on_company_judgments=lambda rows: completed.__setitem__(slice(None), rows))
            except scoring.ScoringError:
                output = scoring.build_scoring_failure(document['scored_run_id'], 'judge_error',
                    reason='malformed_response', company_judgments=completed)
            else:
                output = scoring.build_scoring_output(document['scored_run_id'], full, company_judgments=new)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())
        result = original_run(spec, **kwargs)
        if not result.output_bytes:
            return result
        rows = json.loads(result.output_bytes)['companies']
        for row in rows:
            slug = row['company_name'].casefold().replace(' ', '-')
            row.update(company_linkedin=f'https://linkedin.com/company/{slug}', state='CA',
                       intent_details=f"{row['company_name']} announced a relevant product launch.",
                       company_stage_evidence=[])
            row.pop('fit_summary', None)
            row.pop('fit_evidence_urls', None)
            for signal in row['intent_signals']:
                signal.pop('why_now', None)
                signal.pop('snippet', None)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps({
            'schema_version': intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA, 'companies': rows}).encode())

    harness.sandbox.run_icp = run_icp
    harness.chain.epoch = 40_000
    harness.clock.now = datetime.now(timezone.utc)
    round_id = 'arena-2026-11-13-correction'
    configuration = harness.service.create_round(harness.clock.now + timedelta(seconds=5), round_id=round_id)
    harness.round_id = round_id
    with p.c.cursor() as q:
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)', (
            configuration['scorer_image_digest'], configuration['scorer_image_reference'],
            configuration['scorer_image_digest'], configuration['scorer_image_reference']))
    p.c.commit()
    miners = [harness.submit(flavor, round_id) for flavor in flavors]

    def age_pending():
        with harness.connect() as c, c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE round_id=%s AND kind='score' AND status='pending' AND attempt>1", (round_id,))

    harness.clock.advance_to(harness.schedule()['submission_cutoff'])
    harness.advance_until('stage2_scoring', runners=2, max_steps=120)
    for _ in range(8):
        waiting = harness.service.store.list_runs(round_id, kind='score')
        if any(r['status'] == 'pending' and r['attempt'] > 1 for r in waiting):
            age_pending()
        harness.run_stage_with_runners(2)
        if not any(r['status'] == 'pending' for r in harness.service.store.list_runs(round_id, kind='score')):
            break
    parents = [r for r in harness.service.store.list_runs(round_id, kind='score')
               if r['status'] == 'failed' and r['terminal_cause'] == 'judge_error' and int(r['attempt']) == 5]
    assert len(parents) == 7 and {r['submission_id'] for r in parents} == set(miners)
    for index, parent in enumerate(parents):
        item = parent_request(p, parent, 'native-final-budget-' + str(index))
        with p.c.cursor() as q:
            native.old.scalar(q, 'SELECT public.lab_arena_grant_judge_budget_exception_v1(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,FALSE)', (
                item['parent_run_id'], item['parent_sha256'], configuration['scorer_image_digest'],
                item['request_id'], 'native-owner', 'native-earlier-approval',
                json.dumps(item['checkpoints']), json.dumps(item['signal_checkpoints'])))
        p.c.commit()
    age_pending()
    harness.run_stage_with_runners(2)
    parents = [r for r in harness.service.store.list_runs(round_id, kind='score')
               if r['status'] == 'failed' and r['terminal_cause'] == 'judge_error' and int(r['attempt']) == 6]
    assert len(parents) == 7
    # Only the disposable test clock closes the old publication window.
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb(to_char((clock_timestamp()-interval '2 seconds') AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'))),'{schedule,publication_deadline}',to_jsonb(to_char((clock_timestamp()-interval '1 second') AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'))) WHERE round_id=%s", (round_id,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    harness.clock.now = max(harness.clock.now, datetime.now(timezone.utc))
    harness.advance_until('published', runners=2, max_steps=10)
    assert harness.service.store.get_round(round_id)['king_outcome'] == 'no_king'
    assert harness.service.activate_reward(round_id)['status'] == 'activated'
    historical = harness.service.store.get_round(round_id)
    assert historical['promotion_required'] is True
    frozen_configuration = copy.deepcopy(historical['configuration_doc'])
    requests = [parent_request(p, parent, 'native-corrected-' + str(index)) for index, parent in enumerate(parents)]
    with p.c.cursor() as q:
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)', (
            configuration['scorer_image_digest'], configuration['scorer_image_reference'], RUNTIME,
            'registry.example/scorer@' + RUNTIME))
        round_sha = 'sha256:' + native.old.scalar(q, "SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::text,'sha256'),'hex') FROM lab_arena_rounds r WHERE round_id=%s", (round_id,))
    p.c.commit()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    close, publication = now + timedelta(seconds=8), now + timedelta(seconds=12)
    values = dict(round=round_id, sha=round_sha, runtime=RUNTIME, final=close.isoformat(),
                  publication=publication.isoformat(), requests=requests, request='native-seven-corrections',
                  actor='native-owner', approval='native-reviewed-fix', dry=False)
    original_runs = harness.service.store.list_runs(round_id)
    original_executions = {r['run_id']: r for r in original_runs if r['kind'] == 'execute'}
    response = native.recover(p, values)
    assert len(response['grants']) == 7
    recovering = True
    age_pending()
    harness.run_stage_with_runners(2)
    assert set(corrected_inputs) == {parent['scored_run_id'] for parent in parents}
    for grant in response['grants']:
        result = harness.service.store.get_run(grant['child_identity']['run_id'])
        assert result['status'] == 'accepted' and result['reviewer_runtime_doc']['runtime_image_digest'] == RUNTIME
    time.sleep(max(0, (publication - datetime.now(timezone.utc)).total_seconds()) + 0.1)
    harness.clock.now = datetime.now(timezone.utc)
    amended = harness.advance_until('published', runners=2, max_steps=10)
    assert amended['king_outcome'] == ('crowned' if winner else 'no_king')
    assert amended['reward_basis_doc'] is None and amended['reward_activated_at'] is None
    assert amended['promotion_required'] is True
    if winner:
        assert amended['king_hotkey'] == fixtures.keypair('svc-miner-MinerA').ss58_address
        ranking = amended['publication_doc']['final_ranking']
        winner_row = next(r for r in ranking if r['submission_id'] == miners[0])
        baseline_id = next(r['submission_id'] for r in amended['participants'] if r['is_king'])
        baseline_row = next(r for r in ranking if r['submission_id'] == baseline_id)
        assert winner_row['final_score'] >= baseline_row['final_score'] + 5.0
        repository = tmp_path / 'promotion-repository'
        repository.mkdir()
        remote = fixtures.promotion_repository(repository)
        harness.service._config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / 'promotion-objects')
        assert harness.service.promote_pending_baselines() == {'status': 'ok', 'promoted': 1}
    harness.chain.epoch += 10
    assert harness.service.activate_reward(round_id)['status'] == 'activated'
    activated = harness.service.store.get_round(round_id)
    assert activated['effective_reward_epoch'] > historical['effective_reward_epoch']
    assert activated['reward_basis_hash'] != historical['reward_basis_hash']
    with p.c.cursor() as q:
        archive = native.old.scalar(q, 'SELECT public.lab_arena_correction_reward_history_v1(NULL,NULL,NULL,200,0,FALSE)')
    p.c.commit()
    assert any(row['round_id'] == round_id and row['reward_basis_hash'] == historical['reward_basis_hash']
               and row['effective_reward_epoch'] == historical['effective_reward_epoch'] for row in archive)
    latest = {r['run_id']: r for r in harness.service.store.list_runs(round_id)}
    for rid, original in original_executions.items():
        # A newly completed judgment fills these derived execution projections.
        # The accepted source, output, lease and terminal work remain exact.
        projected = {'per_icp_score', 'qualification_doc', 'updated_at'}
        assert {k: v for k, v in latest[rid].items() if k not in projected} == {
            k: v for k, v in original.items() if k not in projected}
        if original['qualification_doc'] is not None:
            assert latest[rid]['qualification_doc'] == original['qualification_doc']
        elif latest[rid]['qualification_doc'] is not None:
            assert rid in {parent['scored_run_id'] for parent in parents}
    for row in original_runs:
        assert latest[row['run_id']]['status'] == row['status']
        if row['per_icp_score'] is not None:
            assert latest[row['run_id']]['per_icp_score'] == row['per_icp_score']
    assert len([r for r in latest.values() if r['kind'] == 'execute']) == len(original_executions) == 80
    assert harness.service.config.daily_icp_source(set_id=1)['icps'] == bank
    cfg = copy.deepcopy(activated['configuration_doc'])
    for snapshot in (cfg, frozen_configuration):
        snapshot['schedule'].pop('final_scoring_close')
        snapshot['schedule'].pop('publication_deadline')
    assert cfg == frozen_configuration  # Includes every source, capture and training gate.
    fixtures.assert_canary_absent(harness, connect)
