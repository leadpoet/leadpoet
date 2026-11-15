"""Final scoring removes polling waits without bypassing service guards."""
from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace

import pytest

from lab_arena.driver import drive_once
from lab_arena.service import ArenaService


class FinalScoringService(ArenaService):
    def __init__(self, stop=None):
        self.moment = datetime(2026, 11, 10, tzinfo=timezone.utc)
        self._clock = lambda: self.moment
        self._lock = Lock()
        self.row = {"status": "stage2_scoring", "evaluation_date": "2026-11-10",
                    "configuration_doc": {"schedule": {
                        "submission_cutoff": (self.moment-timedelta(hours=1)).isoformat(),
                        "final_scoring_close": self.moment.isoformat()}}}
        self.calls = []
        self.stop = stop
        self._round = lambda _: dict(self.row)
        self._invalidate_hot_round = lambda: None
        self._reconcile_active_deepline_cost = lambda _: {"status": "none"}
        self._reconcile_active_openrouter_cost = self.billing
        self._store = SimpleNamespace(operator_hold_active=self.held,
                                      expire_leases=lambda _: self.calls.append("expire"))
        self._recover_baseline = lambda _: None
        self._scoring_runs_for_stage = lambda *_: [{"status": "accepted"}]
        self._frozen_unfinished_judgment = lambda _: self.stop == "frozen"

    def billing(self, _):
        self.calls.append("billing:" + self.row['status'])
        if self.row['status'] == 'stage2_judged' and self.stop == 'billing':
            return {"status": "pending", "run_status": "leased",
                    "lease_expires_at": (self.moment+timedelta(hours=1)).isoformat()}
        return {"status": "none"}

    def held(self):
        self.calls.append('hold-check')
        return self.row['status'] == 'stage2_judged' and self.stop == 'hold'

    def active_rounds(self):
        return [{"round_id": "round", "status": self.row['status']}]

    def ensure_daily_round(self):
        return {"status": "existing"}

    def promote_pending_baselines(self):
        self.calls.append('promote')
        return {"promoted": 0}

    def close_scoring(self, *_):
        self.calls.append('close')
        self.row['status'] = 'stage2_judged'
        return {"status": "closed", "round_status": "stage2_judged"}

    def score_stage(self, *_):
        self.calls.append('score')
        if self.stop == 'exception':
            raise RuntimeError('synthetic')
        if self.stop in ('retry', 'stale', 'waiting', 'existing'):
            return {"status": self.stop, "round_status": "scored"}
        self.row['status'] = 'scored'
        return {"status": "ok", "round_status": "scored"}

    def publish(self, _):
        self.calls.append('publish')
        self.row['status'] = 'published'
        return {"status": "ok", "king_outcome": "no_king"}

    def activate_pending_rewards(self):
        self.calls.append('reward:' + self.row['status'])
        return {"activated": int(self.row['status'] == 'published')}

    def reconcile_closed_provider_costs(self):
        self.calls.append('closed-billing')
        return {"status": "none"}


def test_completed_final_scoring_publishes_and_activates_without_two_tick_waits():
    service = FinalScoringService()
    assert drive_once(service) == 'advanced round; activated rewards 1'
    assert service.row['status'] == 'published'
    assert service.calls == ['promote', 'billing:stage2_scoring', 'hold-check', 'expire',
                             'close', 'billing:stage2_judged', 'hold-check', 'expire',
                             'score', 'billing:scored', 'hold-check', 'expire', 'publish',
                             'reward:published', 'closed-billing']
    # Original one-step scheduling needs three ticks; 60-second intervals add
    # 120 seconds after closure. The same transitions now need no sleep.
    original = FinalScoringService()
    for step in range(3):
        original.advance_round('round')
        if step < 2:
            original.moment += timedelta(seconds=60)
    assert original.row == service.row
    assert original.moment-service.moment == timedelta(seconds=120)


@pytest.mark.parametrize('stop', ['hold', 'billing', 'frozen', 'exception',
                                  'retry', 'stale', 'waiting', 'existing'])
def test_followup_rechecks_guards_and_stops_without_retry(stop):
    service = FinalScoringService(stop)
    result = drive_once(service)
    assert service.calls.count('close') == 1
    assert service.calls.count('score') <= 1
    assert 'publish' not in service.calls
    assert service.calls[-1] == 'closed-billing'
    assert ('failed advance_round' in result) == (stop == 'exception')


def test_each_pass_serves_other_rounds_and_is_bounded_for_bad_repeated_progress():
    service = FinalScoringService()
    service.active_rounds = lambda: [{'round_id': 'a'}, {'round_id': 'b'}]
    calls = []
    def advance(round_id):
        calls.append(round_id)
        return {'status': 'closed', 'round_status': 'stage2_judged'}
    service.advance_round = advance
    drive_once(service)
    assert calls == ['a', 'b', 'a', 'b', 'a', 'b']


@pytest.mark.parametrize('result', [None, {}, {'status': 'ok'},
    {'status': 'closed', 'round_status': 'stage1_judged'},
    {'status': 'ok', 'round_status': 'stage2'},
    {'status': 'existing', 'round_status': 'stage2_judged'}])
def test_only_exact_final_scoring_progress_gets_followup(result):
    service = FinalScoringService()
    calls = []
    service.advance_round = lambda _: calls.append(1) or result
    drive_once(service)
    assert calls == [1]


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('cas_status', ['ok', 'stale', 'retry', 'existing'])
def test_real_score_stage_reports_final_progress_only_after_successful_cas(
    monkeypatch, stage, cas_status,
):
    from tests.lab_arena.partial_close_scoring_test import _score_stage_stub

    service, _, _ = _score_stage_stub(
        monkeypatch, incomplete_submission=None, stage=stage,
    )
    transitions = []

    def transition(*args):
        transitions.append(args)
        return {'status': cas_status, 'round_status': 'concurrent-state'}

    service._store.transition_round = transition
    round_id = service._round('ignored')['round_id']
    result = service.score_stage(round_id, stage)

    assert result['status'] == cas_status
    assert transitions[0][:3] == (
        round_id, 'stage%d_judged' % stage,
        'scored' if stage == 2 else 'stage1_scored',
    )
    if stage == 2 and cas_status == 'ok':
        assert result['round_status'] == 'scored'
    else:
        assert 'round_status' not in result
