"""Deadline isolation preserves real scores without inventing incomplete totals."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from lab_arena import contracts, scoring, service, public_dashboard, verify
from tests.lab_arena.test_lab_arena_scoring import ROUND, _ICPS, runs_for, company, breakdown


def partial_plan(cause="stage_closed", marked=True):
    runs = runs_for(["king", "good", "partial"])
    last = next(r for r in runs if r['submission_id'] == 'partial' and r['icp_position'] == 3)
    last.update(status='failed', terminal_cause=cause, output_ref=None,
                terminal_doc={'infrastructure_incomplete': marked})
    return runs, scoring.build_scoring_plan(round_id=ROUND, stage=1, runs=runs)


def test_partial_baseline_transition_wait_uses_only_relevant_proof_deadlines():
    round_row = {"configuration_doc": {
        "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
        "integrity_policy": "arena_integrity_v1",
        "sourcing_cost_eligibility_policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        "schedule": {
            "stage_1_close": "2026-10-10T03:00:00Z",
            "stage_1_scoring_close": "2026-10-10T04:00:00Z",
        },
    }}
    deadline = service._partial_baseline_transition_deadline
    kwargs = {"stage": 1, "execution_incomplete": False, "judge_incomplete": False}
    assert deadline(round_row, **kwargs) is None  # complete baseline or genuine zero
    assert deadline(round_row, **{**kwargs, "execution_incomplete": True}) == datetime(2026, 10, 10, 3, tzinfo=timezone.utc)
    assert deadline(round_row, **{**kwargs, "judge_incomplete": True}) == datetime(2026, 10, 10, 4, tzinfo=timezone.utc)
    assert deadline(round_row, **{**kwargs, "execution_incomplete": True, "judge_incomplete": True}) == datetime(2026, 10, 10, 4, tzinfo=timezone.utc)
    assert deadline(round_row, **{**kwargs, "stage": 2, "execution_incomplete": True}) is None
    parallel = {"configuration_doc": {**round_row["configuration_doc"], "execution_sequence_policy": "parallel_twenty_icp_v1"}}
    assert deadline(parallel, **{**kwargs, "execution_incomplete": True}) is None


def _score_stage_stub(monkeypatch, *, incomplete_submission="baseline", zero=False,
                      judge_incomplete=False, judge_cause="stage_closed", stage=1):
    monkeypatch.setattr(scoring, "build_stage_scores", lambda **kwargs: {"rows": []})
    monkeypatch.setattr(scoring, "run_scores_for_store", lambda *args: [{"run_id": "execution-1", "per_icp_score": 42.0}])
    clock = [datetime(2026, 10, 10, 2, tzinfo=timezone.utc)]
    events = []

    class Store:
        def list_runs(self, *args, **kwargs):
            return [{"run_id": "execution-1", "submission_id": "baseline", "icp_position": 0, "status": "accepted"}]

        def record_run_scores(self, *args):
            events.append("record")
            return {"status": "ok"}

        def transition_round(self, *args):
            events.append("transition")
            return {"status": "ok"}

    plan = {
        "work_items": [{"scored_run_id": "execution-1", "submission_id": "baseline", "icp_position": 0}],
        "incomplete_rows": ([{"submission_id": incomplete_submission, "icp_position": 1, "cause": "stage_closed"}]
                            if incomplete_submission else []),
        "zero_rows": ([{"submission_id": "baseline", "icp_position": 1, "cause": "model_error"}] if zero else []),
    }
    round_row = {
        "status": "stage%d_judged" % stage,
        "participants": [{"submission_id": "baseline", "is_king": True},
                         {"submission_id": "candidate", "is_king": False}],
        "configuration_doc": {
            "scorer_policy": scoring.build_scorer_policy(),
            "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
            "integrity_policy": "arena_integrity_v1",
            "sourcing_cost_eligibility_policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
            "schedule": {"stage_1_close": "2026-10-10T03:00:00Z",
                         "stage_1_scoring_close": "2026-10-10T04:00:00Z"},
        },
    }
    fake = SimpleNamespace(
        _round=lambda _rid: round_row,
        _load_scoring_plan=lambda _row, _stage: plan,
        evaluation_icps=lambda _rid: [{}],
        _outputs_by_run=lambda _rid, _stage: {"execution-1": []},
        _scoring_outputs=lambda _rid, _stage: {"execution-1": {
            "status": "failed" if judge_incomplete else "accepted",
            "terminal_cause": judge_cause if judge_incomplete else "accepted",
        }},
        _verified_breakdowns=lambda *args, **kwargs: [],
        _store=Store(), now=lambda: clock[0],
    )
    return fake, clock, events


def test_score_stage_keeps_receipts_and_retries_until_baseline_proof_deadline(monkeypatch):
    fake, clock, events = _score_stage_stub(monkeypatch)
    pending = service.ArenaService.score_stage(fake, ROUND, 1)
    assert pending == {"status": "retry", "round_status": "stage1_judged",
                       "reason": "partial_baseline_deadline_pending", "judge_executions": 1}
    assert events == ["record"]
    clock[0] = datetime(2026, 10, 10, 3, tzinfo=timezone.utc)
    assert service.ArenaService.score_stage(fake, ROUND, 1)["status"] == "ok"
    assert events == ["record", "record", "transition"]


def test_score_stage_waits_for_incomplete_judge_proof_deadline(monkeypatch):
    fake, clock, events = _score_stage_stub(
        monkeypatch, incomplete_submission=None, judge_incomplete=True,
    )
    clock[0] = datetime(2026, 10, 10, 3, tzinfo=timezone.utc)
    assert service.ArenaService.score_stage(fake, ROUND, 1)["status"] == "retry"
    assert events == ["record"]
    clock[0] = datetime(2026, 10, 10, 4, tzinfo=timezone.utc)
    assert service.ArenaService.score_stage(fake, ROUND, 1)["status"] == "ok"
    assert events == ["record", "record", "transition"]


def test_score_stage_does_not_mask_unproven_judge_failure(monkeypatch):
    fake, _, events = _score_stage_stub(
        monkeypatch, incomplete_submission=None,
        judge_incomplete=True, judge_cause="judge_error",
    )
    assert service.ArenaService.score_stage(fake, ROUND, 1)["status"] == "ok"
    assert events == ["record", "transition"]


@pytest.mark.parametrize("incomplete_submission,zero,stage", [
    (None, False, 1), (None, True, 1), ("candidate", False, 1),
    ("baseline", False, 2),
])
def test_score_stage_does_not_delay_other_outcomes(monkeypatch, incomplete_submission, zero, stage):
    fake, _, events = _score_stage_stub(
        monkeypatch, incomplete_submission=incomplete_submission, zero=zero, stage=stage,
    )
    assert service.ArenaService.score_stage(fake, ROUND, stage)["status"] == "ok"
    assert events == ["record", "transition"]


@pytest.mark.parametrize('cause', ['stage_closed', 'provider_error', 'worker_lost'])
def test_partial_work_persists_and_full_results_keep_same_scores(cause):
    runs, plan = partial_plan(cause)
    assert len(plan['work_items']) == 29
    assert plan['zero_rows'] == []
    assert plan['incomplete_rows'] == [{'submission_id':'partial', 'icp_position':3, 'cause':cause}]
    assert contracts.validate_scoring_plan(json.loads(contracts.canonical_json(plan))) == plan
    outputs = {i['scored_run_id']:[company(0)] for i in plan['work_items']}
    breakdowns = {i['scored_run_id']:[breakdown(60.0)] for i in plan['work_items']}
    scores = scoring.build_stage_scores(plan=plan, policy=scoring.build_scorer_policy(),
        icps_by_position=_ICPS, outputs_by_run=outputs, breakdowns_by_item=breakdowns)
    assert len(scores['rows']) == 29
    assert set(scores['submission_scores']) == {'king', 'good'}
    assert scores['submission_scores']['king'] == scores['submission_scores']['good']
    stored = scoring.run_scores_for_store(scores, runs)
    assert len(stored) == 29
    assert not any(r['run_id'] == next(x['run_id'] for x in runs if x['submission_id']=='partial' and x['icp_position']==3) for r in stored)


def test_unmarked_infrastructure_failure_is_not_silently_dropped():
    with pytest.raises(contracts.ArenaContractError):
        partial_plan(marked=False)


def test_partial_marker_cannot_mask_a_model_failure():
    _, plan = partial_plan('model_timeout')
    assert 'incomplete_rows' not in plan
    assert plan['zero_rows'][0]['cause'] == 'model_timeout'


def test_duplicate_or_out_of_stage_incomplete_row_rejected():
    _, plan = partial_plan()
    plan['incomplete_rows'][0]['icp_position'] = 0
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_scoring_plan(plan)


def test_incomplete_final_result_is_visible_but_cannot_win():
    runs = [dict(submission_id='partial', icp_position=0, attempt=1,
                 per_icp_score=100.0, terminal_cause='accepted')]
    fake = SimpleNamespace(_store=SimpleNamespace(list_runs=lambda *a,**k:runs))
    rr = {'round_id':ROUND,'participants':[{'submission_id':'partial','miner_hotkey':'miner','is_king':False}]}
    entries = service.ArenaService._score_entries_from_runs(fake,rr,[0,1],'final_score')
    assert entries[0]['final_score'] is None and entries[0]['execution_incomplete']
    assert verify.king_decision(entries, {'submission_id':'king','hotkey':'baseline','final_score':0.0})['outcome']=='no_king'
    assert verify.king_decision([{'submission_id':'complete','hotkey':'miner','final_score':100}],
        {'submission_id':'king','hotkey':'baseline','final_score':None})['outcome']=='no_king'
    cost = {'eligible':False,'eligibility_reason':'execution_incomplete','cost_summary':None}
    assert public_dashboard._cost_projection(cost) == cost


def test_incomplete_only_submission_still_requires_every_stage_position():
    _, plan = partial_plan()
    plan['work_items'] = []
    kwargs = dict(plan=plan, policy=scoring.build_scorer_policy(),
                  icps_by_position=_ICPS, outputs_by_run={}, breakdowns_by_item={})
    with pytest.raises(scoring.ScoringError, match="every stage"):
        scoring.build_stage_scores(**kwargs)
    plan['incomplete_rows'] = [dict(submission_id='partial', icp_position=p, cause='stage_closed') for p in range(10)]
    result = scoring.build_stage_scores(**kwargs)
    assert result['rows'] == [] and result['submission_scores'] == {}
