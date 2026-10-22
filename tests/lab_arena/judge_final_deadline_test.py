"""Incomplete judging drains at the frozen deadline without fabricated scores."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from lab_arena import service
from tests.lab_arena import complete_baseline_final_authority_test as baseline
from tests.lab_arena import partial_close_scoring_test as stages

DEADLINE = datetime(2026, 10, 21, 20, 30, 3, tzinfo=timezone.utc)


@pytest.mark.parametrize('offset,expected', [(-1, 'retry'), (0, 'cancelled'), (1, 'cancelled')])
@pytest.mark.parametrize('reason', ['judge_incomplete', 'baseline_incomplete', 'execution_infrastructure_incomplete'])
def test_generic_hold_respects_exact_deadline_and_keeps_history(offset, expected, reason):
    obj, row, executions, judges, writes, clock, costs = baseline.fixture()
    row['configuration_doc']['schedule']['publication_deadline'] = service._iso(DEADLINE)
    clock[0] = DEADLINE + timedelta(microseconds=offset)
    before = deepcopy((row, executions, judges, costs))
    result = obj._hold_incomplete_evaluation(row, reason=reason)
    assert result['status'] == expected
    if expected == 'retry':
        assert result['reason'] == reason and writes == []
    else:
        assert writes == [(row['round_id'], 'scoring_incomplete')]
    assert (row, executions, judges, costs) == before


@pytest.mark.parametrize('active', ['leased', 'submitted', 'provider'])
def test_deadline_still_drains_live_work(active):
    obj, row, executions, _, writes, clock, costs = baseline.fixture()
    clock[0] = DEADLINE
    if active == 'provider':
        costs['providers'][0]['inflight_calls'] = 1
    else:
        executions[0]['status'] = active
    assert obj._hold_incomplete_evaluation(row, reason='judge_incomplete')['status'] == 'retry'
    assert writes == []


def test_atomic_cancel_race_retry_is_not_mistaken_for_terminal_failure():
    obj, row, _, _, _, clock, _ = baseline.fixture()
    clock[0] = DEADLINE
    obj._store.cancel_round = lambda *a: {'status': 'retry', 'reason': 'evaluation_draining'}
    assert obj._hold_incomplete_evaluation(row, reason='judge_incomplete') == {
        'status': 'retry', 'reason': 'evaluation_draining'}


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('offset,status', [(-1, 'retry'), (0, 'cancelled'), (1, 'cancelled')])
def test_final_stage_unmeasured_judge_reaches_same_deadline_hold(monkeypatch, offset, status, stage):
    obj, clock, events = stages._score_stage_stub(
        monkeypatch, incomplete_submission=None, judge_incomplete=True,
        judge_cause='judge_error', stage=stage)
    original_outputs = obj._scoring_outputs
    obj._scoring_outputs = lambda *a, **k: original_outputs(*a)
    row = obj._round('ignored')
    row['configuration_doc']['schedule']['publication_deadline'] = service._iso(DEADLINE)
    clock[0] = DEADLINE + timedelta(microseconds=offset)
    obj._store.submission_costs = lambda _: {'providers': []}
    obj._store.cancel_round = lambda *a: events.append('cancel') or {'status': 'cancelled'}
    if stage == 1 and offset < 0:
        # Baseline-first stage1 retains its earlier partial progression; only
        # final-deadline behavior is changed by this control.
        obj._store.record_run_scores = lambda *a, **k: {'status': 'ok'}
        result = obj.score_stage(row['round_id'], stage)
        assert result['status'] == 'ok' and events == ['transition']
        return
    result = obj.score_stage(row['round_id'], stage)
    assert result['status'] == status
    assert events == ([] if status == 'retry' else ['cancel'])


@pytest.mark.parametrize('offset,status', [(-1, 'retry'), (0, 'publication'), (1, 'publication')])
def test_publish_unfinished_judge_cannot_bypass_final_deadline(offset, status):
    obj, row, _, _, writes, clock, _ = baseline.fixture()
    row['evaluation_date'] = '2026-10-21'
    row['configuration_doc']['schedule']['publication_deadline'] = service._iso(DEADLINE)
    clock[0] = DEADLINE + timedelta(microseconds=offset)
    recorded = []
    obj._record_recovered_judgment_scores = lambda r: recorded.append(r['round_id'])
    obj._unfinished_judgments = lambda r: True
    obj._baseline_fully_succeeded = lambda r: True
    obj._execution_infrastructure_failures = lambda _: {}
    obj._participant_incomplete_reasons = lambda *a: {'miner': 'judge_error'}
    class PublicationReached(Exception):
        pass
    def boundary(*a, **kw):
        raise PublicationReached
    obj._score_entries_from_runs = boundary
    if status == 'publication':
        with pytest.raises(PublicationReached): obj.publish(row['round_id'])
    else:
        assert obj.publish(row['round_id'])['status'] == status
    assert recorded == [row['round_id']] and writes == []


def test_complete_late_judgments_continue_publication_instead_of_cancel():
    obj, row, _, _, writes, clock, _ = baseline.fixture()
    row['evaluation_date'] = '2026-10-21'
    clock[0] = DEADLINE + timedelta(hours=1)
    obj._record_recovered_judgment_scores = lambda _: None
    obj._unfinished_judgments = lambda _: False
    obj._baseline_fully_succeeded = lambda _: True
    # Crossing this existing publication boundary proves the incomplete hold
    # is conditional, rather than an unconditional clock-based cancellation.
    class PublicationReached(Exception):
        pass
    def boundary(*a, **k):
        raise PublicationReached
    obj._score_entries_from_runs = boundary
    obj._store.list_runs = lambda *a, **k: []
    with pytest.raises(PublicationReached):
        obj.publish(row['round_id'])
    assert writes == []
