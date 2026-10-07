"""Deadline isolation preserves real scores without inventing incomplete totals."""
import json
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
