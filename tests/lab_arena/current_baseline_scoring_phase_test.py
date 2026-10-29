"""Current baseline-first rounds persist late judgments in the authorized phase."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena import dynamic_benchmark_lifecycle_test as lifecycle
from tests.lab_arena.lab_arena_pg_harness import current_service_database


@pytest.fixture()
def database():
    yield from current_service_database()


@pytest.mark.parametrize('per_icp_cost_policy,score_outage', [
    (False, False), (True, False), (True, True),
])
def test_current_baseline_first_records_unscored_judgments_before_publication(
    database, tmp_path, per_icp_cost_policy, score_outage,
):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    connect.object_store_namespace = 'baseline-phase-' + tmp_path.name
    harness = lifecycle._harness(connect, tmp_path, count=10, margin=0.5,
                                 challengers=['PhaseMiner'], baseline_first=True)
    harness.service.config.defaults = replace(harness.service.config.defaults,
        per_icp_cost_policy=per_icp_cost_policy)
    _, participants = lifecycle._start(harness,
        label='baselinephase' + str(int(per_icp_cost_policy)), epoch=37000)
    baseline = next(row['submission_id'] for row in participants if row['is_king'])
    store = harness.service.store
    phases = []
    blocked = [score_outage]
    record_scores = store.record_run_scores

    def observe(round_id, stage, records, **kwargs):
        sources = [store.get_run(record['run_id']) for record in records]
        if stage == 1 and any(source['submission_id'] == baseline for source in sources):
            phases.append(store.get_round(round_id)['status'])
            if blocked[0]:
                raise ArenaStoreError('controlled baseline score write outage')
        return record_scores(round_id, stage, records, **kwargs)

    store.record_run_scores = observe
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    if score_outage:
        with pytest.raises(ArenaStoreError, match='controlled baseline score write outage'):
            harness.advance_until('published', runners=2)
        interrupted = store.get_round(harness.round_id)
        assert interrupted['status'] == 'scored'
        assert interrupted['publication_doc'] is None
        source_before = store.list_runs(harness.round_id, kind='execute')
        assert all(source['per_icp_score'] is None for source in source_before
                   if source['submission_id'] == baseline)

        def concurrent_publish():
            with pytest.raises(ArenaStoreError, match='controlled baseline score write outage'):
                harness.service.publish(harness.round_id)

        with ThreadPoolExecutor(max_workers=2) as publishers:
            list(publishers.map(lambda _: concurrent_publish(), range(2)))
        assert store.list_runs(harness.round_id, kind='execute') == source_before
        still_held = store.get_round(harness.round_id)
        assert still_held['status'] == 'scored'
        assert still_held['publication_doc'] is None and still_held['reward_activated_at'] is None
        blocked[0] = False
    harness.advance_until('published', runners=2)
    row = store.get_round(harness.round_id)
    assert row['status'] == 'published' and row['cancel_reason'] is None
    assert phases and set(phases) == {'scored'}
    assert len(phases) == (4 if score_outage else 1)
    sources = store.list_runs(harness.round_id, stage=1,
                             submission_id=baseline, kind='execute')
    assert len(sources) == 10
    assert all(source['status'] == 'accepted' and source['per_icp_score'] is not None
               for source in sources)
    ranking = row['publication_doc']['final_ranking']
    assert len(ranking) == len(participants)
    assert all(entry['final_score'] is not None for entry in ranking)
