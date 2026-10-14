"""Real broker, queue, scoring, final eligibility and publication share cost credit."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from urllib.parse import urlsplit

from lab_arena import runtime, scoring, shim
from lab_arena.contact_evidence import source_key
from lab_arena.runtime_host import RuntimeHostError
from qualification.scoring.competition import apply_company_judgment_context, _merge_contact_breakdown
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.contact_round_test import _claim, _contact_result
from tests.lab_arena.infrastructure_retry_cost478_postgres_test import database
from tests.lab_arena.per_icp_cost_round_test import PerIcpHarness
from tests.test_arena_company_quality_policy import _positive_breakdown


def test_both_roles_keep_cost_credit_through_real_judge_and_publication(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    h = PerIcpHarness(connect, tmp_path, challengers=['Healthy'], runners=['alpha'])
    h.service.config.defaults = replace(h.service.config.defaults, benchmark_icp_count=2, rewards_enabled=False,
        execution_sequence_from='2026-01-01T00:00:00Z', company_quality_from=None)
    original_source = h.service.config.daily_icp_source
    h.service.config.daily_icp_source = lambda **kw: {**original_source(**kw), 'icps': original_source(**kw)['icps'][:2]}
    class CurrentApi(fixtures.InProcessApi):
        def trajectory(self, run_id, lease_token, events):
            return self.service.handle_trajectory(run_id, lease_token, {'events': list(events)})
        def quota_usage(self, run_id, lease_token, *, include_sourcing_cost=False):
            return self.service.handle_quota_snapshot(run_id, lease_token,
                include_sourcing_cost=include_sourcing_cost)
    h.api_factory = lambda: CurrentApi(h.service)
    original = h.sandbox.run_icp
    lost = set()
    snapshots = []

    def run(spec, **kwargs):
        doc = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if doc.get('schema_version') == scoring.SCORING_INPUT_SCHEMA_VERSION:
            def judge(companies, buyer, _reference):
                rows = []
                for company in companies:
                    name = company['company_name']
                    raw = _positive_breakdown(name, urlsplit(company['company_website']).hostname,
                                              name.lower().replace(' ', '-'))
                    score = 30.0 if name.startswith('PublicBaseline') else 60.0
                    raw.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
                    raw['intent_signals_detail'][0].update(raw=score, after_decay=score)
                    evidence = doc['contact_source_evidence'][source_key(company)]
                    _merge_contact_breakdown(raw, asyncio.run(_contact_result(company, buyer, evidence)))
                    rows.append(raw)
                return apply_company_judgment_context(companies, rows, contacts_required=True)
            judge.integrity_policy = judge.contacts_required = True
            judge.company_quality = False
            full = scoring.score_work_item({'scored_run_id': doc['scored_run_id']},
                icp=doc['icp'], companies=doc['companies'], scorer=judge)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(
                scoring.build_scoring_output(doc['scored_run_id'], full)).encode())
        # The fixed model-output fixture performs unrelated free discovery.
        # Control that boundary; the charged OpenRouter calls below use the
        # complete broker and immutable SQL accounting path.
        from unittest.mock import patch
        with patch.object(shim, 'dispatch', return_value=(200, {}, b'{}')):
            base = original(spec, **kwargs)
        rows = json.loads(base.output_bytes)['companies']
        position = int(doc['icp']['icp_id'].rsplit('_', 1)[-1]) - 1
        for company in rows:
            company.update(company_linkedin='https://linkedin.com/company/' + company['company_name'].lower().replace(' ', '-'), state='CA')
            company['contact'] = _claim(company, valid_role=True)
        if position == 0:
            rows = rows[:1]
            role = rows[0]['company_name'].split(' Company ', 1)[0]
            first = role not in lost
            with h.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    if not first:
                        from lab_arena.lab_arena_checkpoint import quota_usage
                        snapshots.append(quota_usage(include_sourcing_cost=True)['sourcing_cost'])
                    cost = '0.778569' if first else '0.75'
                    status, _, _ = shim.dispatch('openrouter.chat', {'model': 'openai/gpt-4o-mini',
                        'messages': [{'role': 'user', 'content': cost + '|ok'}], 'max_tokens': 100}, 5000)
                    assert status == 200
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            if first:
                lost.add(role)
                raise RuntimeHostError(reason='sandbox_launcher_signaled', launch_exit_code=-2, launch_timed_out=False)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps({'companies': rows}).encode())

    h.sandbox.run_icp = run
    observed_host_errors = []
    def workers(count=1):
        scheduled = h.clock.now
        h.clock.now = datetime.now(timezone.utc)
        runner = h.runner(0, parallel=1)
        from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
        runner._config.proxy_worker_pool = _verified_test_pool(2)
        try:
            while runner.run_once():
                h.service.store.expire_leases(h.round_id)
            errors = [item for item in runner.completed if item.get('error')]
            assert all('RuntimeHostError' in str(item) for item in errors), errors
            observed_host_errors.extend(errors)
        finally:
            runner.close()
            h.clock.now = scheduled
    h.run_stage_with_runners = workers
    # Simulate an existing intake round created before the flag shipped.
    original_create = h.service.store.create_round
    intake_configurations = []
    def legacy_create(round_id, configuration):
        legacy = dict(configuration)
        legacy.pop('infrastructure_retry_cost_policy')
        intake_configurations.append(legacy)
        return original_create(round_id, legacy)
    h.service.store.create_round = legacy_create
    fixtures._start_round(h, day=19, epoch=31819)
    cfg = h.service.store.get_round(h.round_id)['configuration_doc']
    assert cfg['infrastructure_retry_cost_policy'] == 'trusted_infrastructure_v1'
    h.advance_until('published', runners=1)
    saved = h.service.store.get_round(h.round_id)
    assert len(lost) == len(observed_host_errors) == len(snapshots) == 2
    assert all(s['settled_microusd'] == s['successful_microusd'] == 0 for s in snapshots)
    assert saved['configuration_doc'] == cfg
    for rank in saved['publication_doc']['final_ranking']:
        first = rank['cost_summary']['per_icp'][0]
        assert first['qualified_company_count'] == 1
        assert first['eligible'] is True
        assert first['competition_sourcing_microusd'] == 750000
        assert first['execution']['settled_microusd'] == 1528569
        assert first['execution']['excluded_infrastructure_settled_microusd'] == 778569
        with connect() as conn, conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)',
                        (h.round_id, json.dumps(rank)))
            assert cur.fetchone()[0] is True
    assert saved['reward_activated_at'] is None
