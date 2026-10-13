"""Recovery read and completion controls preserve real scores and old evidence."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lab_arena import contracts, scoring
from lab_arena.service import ArenaService


def fixture():
    obj=object.__new__(ArenaService)
    rid='arena-2099-10-13'; sid='baseline-2099-10-13'
    cfg=dict(execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        integrity_policy='arena_integrity_v1',max_attempts_per_assignment=2,
        stage_1_icp_count=1,stage_2_icp_count=1)
    row=dict(round_id=rid,status='stage2_judged',configuration_doc=cfg,
        participants=[dict(submission_id=sid,miner_hotkey='baseline-owner',is_king=True)])
    executions=[]; judges=[]
    for position,attempt,score in [(0,3,60),(1,1,80)]:
        assignment=f'{rid}:{sid}:1:{position}'
        execution=dict(run_id=assignment+f':{attempt}',assignment_id=assignment,round_id=rid,
            submission_id=sid,miner_hotkey='baseline-owner',kind='execute',stage=1,icp_position=position,
            attempt=attempt,status='accepted',terminal_cause='accepted',per_icp_score=score,output_ref='output',result_doc={})
        executions.append(execution)
        judges.append(dict(execution,run_id=assignment+':score:1',assignment_id=assignment+':score',kind='score',
            attempt=1,scored_run_id=execution['run_id'],status='accepted'))
    for attempt in (1,2):
        executions.append(dict(executions[0],run_id=executions[0]['assignment_id']+f':{attempt}',attempt=attempt,
            status='failed',terminal_cause='provider_error',per_icp_score=0,output_ref=None))
    row['stage1_scoring_plan_doc']=dict(schema_version=contracts.SCORING_PLAN_SCHEMA_VERSION,round_id=rid,stage=1,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        work_items=[dict(scored_run_id=executions[1]['run_id'],submission_id=sid,icp_position=1,output_ref='output')],
        zero_rows=[dict(submission_id=sid,icp_position=0,cause='provider_error')])
    def list_runs(_rid,kind=None,stage=None,status=None):
        values=executions if kind=='execute' else judges if kind=='score' else executions+judges
        return [r for r in values if (stage is None or r['stage']==stage) and (status is None or r['status']==status)]
    obj._store=SimpleNamespace(list_runs=list_runs,record_run_scores=Mock(return_value={'status':'ok'}))
    obj._round=lambda _rid:row
    obj._submission_cost_eligibility=lambda *a,**k:dict(eligible=True,eligibility_reason='eligible',
        cost_summary=dict(per_icp=[dict(icp_position=i,eligible=True) for i in range(2)]))
    return obj,row,executions,judges


def test_new_judge_pending_hides_baseline_then_accepted_uses_real_score():
    obj,row,executions,judges=fixture(); before=deepcopy(row['stage1_scoring_plan_doc'])
    judges[0]['status']='pending'; judges[0]['terminal_cause']=None
    assert obj._read_completed_submission_scores(row)=={}
    assert obj._baseline_recovery_plan(row,executions,judges) is None
    judges[0].update(status='accepted',terminal_cause='accepted')
    plan=obj._baseline_recovery_plan(row,executions,judges)
    assert not plan['zero_rows'] and len(plan['work_items'])==2
    assert obj._read_completed_submission_scores(row)[executions[0]['submission_id']]['final_score']==70
    assert row['stage1_scoring_plan_doc']==before
    assert executions[-1]['per_icp_score']==0 and executions[-1]['status']=='failed'


@pytest.mark.parametrize('fault',['failed_source','failed_judge','missing','alias','wrong_round','wrong_stage','outside'])
def test_incomplete_or_alias_recovery_never_writes_baseline(fault):
    obj,row,executions,judges=fixture()
    if fault=='failed_source':executions[0].update(status='failed',terminal_cause='provider_error')
    elif fault=='failed_judge':judges[0].update(status='failed',terminal_cause='judge_error')
    elif fault=='missing':judges.pop()
    elif fault=='alias':executions[0]['assignment_id']+=':alias'
    elif fault=='wrong_round':judges[0]['round_id']='archived'
    elif fault=='wrong_stage':judges[0]['stage']=2
    elif fault=='outside':executions[0]['icp_position']=9
    assert obj._baseline_recovery_plan(row,executions,judges) is None
    obj._score_recovered_baseline(row)
    obj._store.record_run_scores.assert_not_called()


def test_no_recovery_complete_label_for_old_exhausted_provider_zero():
    obj,row,executions,judges=fixture(); executions.pop(0); judges.pop(0)
    assert obj._read_completed_submission_scores(row)=={}


def test_original_pending_execution_and_ordinary_recovery_judge_hold_completion():
    obj,row,executions,judges=fixture()
    executions[0]['status']='pending'
    assert not obj.stage_is_complete(row['round_id'],2)
    executions[0]['status']='accepted'; judges[0]['status']='submitted'
    assert not obj.scoring_is_complete(row['round_id'],2)
    judges[0]['status']='accepted'
    assert obj.scoring_is_complete(row['round_id'],2)
