"""Completion refresh reuses bounded proof reads only within one model."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from lab_arena import contracts, scoring
from lab_arena.service import _PublicResultEvidenceReads
from lab_arena.store import ArenaStore
from tests.lab_arena.public_result_authority_batch_test import Transport
from tests.lab_arena.public_result_company_evidence_reads_test import _public_company_result
from tests.lab_arena.scoring_validator_attribution_test import ROUND_ID, SUBMISSION_ID, VALIDATOR_A


def _completed_company_case():
    service, ref, lease, cached, source, objects, counts = _public_company_result()
    row = service._round(ROUND_ID)
    row['status'] = 'stage2_scoring'
    row['participants'] = row['publication_doc']['participants']
    row['configuration_doc'].update(stage_1_icp_count=1, stage_2_icp_count=1,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        execution_icp_cap_microusd=4000000, cost_per_company_microusd=800000)
    for run in service._store.rows.values():
        run['stage'] = 2
    current = service._store.rows['score-current']
    execution = service._store.rows['execute-current']
    execution['terminal_cause'] = 'accepted'
    execution2, current2 = deepcopy(execution), deepcopy(current)
    execution2.update(run_id='execute-current-2', icp_position=1, output_ref='execution-output-2')
    current2.update(run_id='score-current-2', scored_run_id=execution2['run_id'], icp_position=1,
                    output_ref='current-output-2')
    service._store.rows.update({item['run_id']:item for item in (execution2,current2)})
    objects['execution-output-2'] = deepcopy(objects['execution-output'])
    objects['current-output-2'] = {**deepcopy(objects['current-output']), 'scored_run_id':execution2['run_id']}
    row['stage2_scoring_plan_doc'] = scoring.build_scoring_plan(
        round_id=ROUND_ID, stage=2, runs=[execution,execution2],
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        configuration=row['configuration_doc'],
    )
    icp = service._benchmark_icps_from_row(ROUND_ID,row)[0]
    service.evaluation_icps = lambda _: [deepcopy(icp),deepcopy(icp)]
    transport = Transport({'lab_arena_company_judgments':[{**cached,'authority_slot':0}]})
    service._store.get_company_judgment_batch = ArenaStore(transport).get_company_judgment_batch
    def cost(**kwargs):
        counts['eligibility_queries'] += 1
        return {'icp_position':kwargs['icp_position'],'eligible':True,'eligibility_reason':'eligible',
                'competition_sourcing_microusd':0}
    def submission_costs(_):
        counts['submission_cost_queries'] += 1
        return {'providers':[]}
    service._store.icp_cost_eligibility = cost
    service._store.submission_costs = submission_costs
    original_round = service._round
    def round_read(rid):
        counts['round_rows'] += 1
        return original_round(rid)
    service._round = round_read
    counts.clear()
    return service,row,cached,source,objects,counts,transport


def test_completion_reuses_exact_proof_and_outputs_but_rechecks_every_cost():
    service,row,_,_,_,counts,transport = _completed_company_case()
    actual = service._read_completed_submission_scores(row)
    assert SUBMISSION_ID in actual
    assert counts['eligibility_queries'] == 2 and counts['submission_cost_queries'] == 1
    assert counts['round_rows'] == counts['run_rows'] == counts['company_rows'] == 0
    assert counts['source-output'] == 1
    assert counts['execution-output'] == counts['execution-output-2'] == 1
    assert len(transport.calls) == 1
    assert actual[SUBMISSION_ID]['eligible'] is True


@pytest.mark.parametrize('stored', [True,False])
def test_completed_map_is_exactly_equal_to_individual_proof_reads(stored):
    service,row,_,_,_,counts,_ = _completed_company_case()
    if not stored:
        for name in ['execute-current','execute-current-2']:
            service._store.rows[name]['per_icp_score'] = None
    cost = service._submission_cost_eligibility
    def individual(*args,**kwargs):
        kwargs.pop('evidence_reads',None)
        return cost(*args,**kwargs)
    with patch.object(service,'_prefetch_public_result_evidence',lambda *a,**k:None), \
         patch.object(service,'_submission_cost_eligibility',individual):
        expected = service._read_completed_submission_scores(row)
    counts.clear()
    actual = service._read_completed_submission_scores(row)
    assert SUBMISSION_ID in expected
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert counts['eligibility_queries'] == 2 and counts['submission_cost_queries'] == 1


@pytest.mark.parametrize('fault', ['missing','hash','foreign','excluded','consumer_output','source_output'])
def test_completion_rejects_invalid_authority_after_prefetch(fault):
    service,row,cached,source,objects,counts,transport = _completed_company_case()
    if fault=='missing':transport.tables['lab_arena_company_judgments']=[]
    elif fault=='hash':transport.tables['lab_arena_company_judgments'][0]['evidence_hash']='sha256:'+'f'*64
    elif fault=='foreign':source['claim_request_hash']='sha256:'+'f'*64
    elif fault=='excluded':service._store.rows['score-current']['miner_hotkey']=VALIDATOR_A
    elif fault=='consumer_output':objects['current-output']['breakdowns'][0]['final_score']=1
    else:objects['source-output']['company_judgments'][0]['raw_judgment']['final_score']=1
    assert service._read_completed_submission_scores(row) == {}
    assert counts['eligibility_queries']==counts['submission_cost_queries']==0


@pytest.mark.parametrize('bound', ['MAX_IDENTITIES','MAX_BYTES'])
def test_cache_bound_keeps_original_validation_and_reads_available(bound):
    service,row,_,_,_,counts,transport = _completed_company_case()
    contexts=[]
    original=service._prefetch_public_result_evidence
    def observed(reads,**kwargs):
        contexts.append(reads)
        return original(reads,**kwargs)
    service._prefetch_public_result_evidence=observed
    with patch.object(_PublicResultEvidenceReads,bound,0):
        result=service._read_completed_submission_scores(row)
    assert SUBMISSION_ID in result
    assert counts['company_rows']>=1 and counts['run_rows']>=1
    assert counts['source-output']>=1
    assert counts['execution-output']==counts['execution-output-2']==2
    assert contexts[0].runs==contexts[0].companies==contexts[0].objects=={}
    assert contexts[0]._bytes==0
    assert counts['eligibility_queries']==2 and counts['submission_cost_queries']==1


def test_store_without_optional_company_batch_retains_individual_validation():
    service,row,_,_,_,counts,_ = _completed_company_case()
    del service._store.get_company_judgment_batch
    assert SUBMISSION_ID in service._read_completed_submission_scores(row)
    assert counts['company_rows']==1
    assert counts['eligibility_queries']==2 and counts['submission_cost_queries']==1


@pytest.mark.parametrize('failure', ['unavailable','foreign_row'])
def test_batch_errors_do_not_fall_back_to_single_authority_reads(failure):
    from lab_arena.store import ArenaStoreError,ArenaStoreUnavailable
    service,row,_,_,_,counts,_ = _completed_company_case()
    def failed(_):raise (ArenaStoreUnavailable if failure=='unavailable' else ArenaStoreError)('test failure')
    service._store.get_company_judgment_batch=failed
    assert service._read_completed_submission_scores(row)=={}
    assert counts['company_rows']==0
    assert counts['eligibility_queries']==0


@pytest.mark.parametrize('change', ['authority','source','cost','exclusion'])
def test_later_refresh_rechecks_authority_source_cost_and_dynamic_exclusion(change):
    from lab_arena import signal_judgments
    service,row,_,source,_,counts,transport = _completed_company_case()
    row['configuration_doc']['scorer_policy']['signal_judgment_policy']=signal_judgments.POLICY
    service._store.signal_authority_exclusions=lambda _: [VALIDATOR_A]
    assert SUBMISSION_ID in service._read_completed_submission_scores(row)
    if change=='authority':transport.tables['lab_arena_company_judgments']=[]
    elif change=='source':source['claim_request_hash']='sha256:'+'f'*64
    elif change=='exclusion':service._store.signal_authority_exclusions=lambda _: []
    else:
        def in_flight(**kwargs):
            counts['eligibility_queries']+=1
            return dict(icp_position=kwargs['icp_position'],eligible=False,
                        eligibility_reason='provider_calls_inflight',competition_sourcing_microusd=0)
        service._store.icp_cost_eligibility=in_flight
    assert service._read_completed_submission_scores(row)=={}
    assert len(transport.calls)==2


def test_pending_peers_are_not_seeded_and_each_completed_model_has_fresh_context():
    service,row,_,_,objects,_,_ = _completed_company_case()
    other='second-model'
    row['participants'].append({'submission_id':other,'miner_hotkey':'other','is_king':False})
    plan=row['stage2_scoring_plan_doc']
    for name in ['execute-current','execute-current-2','score-current','score-current-2']:
        run=deepcopy(service._store.rows[name]);run['run_id']='other-'+name;run['submission_id']=other
        if run['kind']=='score':
            run['scored_run_id']='other-'+run['scored_run_id']
            original_ref=run['output_ref']
            run['output_ref']='other-'+original_ref
            objects[run['output_ref']]={**deepcopy(objects[original_ref]),'scored_run_id':run['scored_run_id']}
        service._store.rows[run['run_id']]=run
    plan['work_items'] += [{**item,'submission_id':other,'scored_run_id':'other-'+item['scored_run_id']}
                           for item in deepcopy(plan['work_items'])]
    service._store.rows['pending-peer']={**service._store.rows['execute-current'],
        'run_id':'pending-peer','submission_id':'pending-model','status':'pending'}
    contexts=[]
    original=service._prefetch_public_result_evidence
    def observed(reads,**kwargs):
        original(reads,**kwargs)
        assert kwargs['include_signal_attribution'] is False
        assert 'pending-peer' not in reads.runs
        contexts.append(reads)
    service._prefetch_public_result_evidence=observed
    completed=service._read_completed_submission_scores(row)
    assert set(completed)=={SUBMISSION_ID,other}
    assert len(contexts)==2 and contexts[0] is not contexts[1]

    # A later model in the same refresh must re-read its raw authority. A
    # successful first model cannot lend its proof cache to that consumer.
    def changed_after_first(submission_id,result):
        if submission_id==SUBMISSION_ID and result is not None:
            authority=deepcopy(next(iter(contexts[0].companies.values())))
            authority['evidence_hash']='sha256:'+'f'*64
            identity=(authority['cache_key'],authority['authority_slot'])
            service._store.get_company_judgment_batch=lambda _pairs:{identity:deepcopy(authority)}
    completed=service._read_completed_submission_scores(row,on_completed=changed_after_first)
    assert set(completed)=={SUBMISSION_ID}
