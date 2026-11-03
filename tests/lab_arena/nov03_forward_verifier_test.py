"""Completed November 3 judgments need exact database preservation authority."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab_arena import company_judgments, contracts, scoring
from tests.lab_arena.company_judgments_test import _input
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.company_quality_judgment_authority589_postgres_test import canonical


def context():
    row, refs = canonical()
    row['round_id']='arena-2026-11-03'
    row['evaluation_date']='2026-11-03'
    for ref in refs:
        ref['scope_doc']['round_id']=row['round_id']
    old=deepcopy(row['configuration_doc'])
    old.pop('company_quality_policy')
    old['company_judgment_policy']='company_judgment_v1'
    old['scorer_policy']=scoring.build_scorer_policy(
        scoring_adapter_version='qualification_integrity_v2',company_judgment_cache=True)
    document=_input()
    document['scorer_policy']=old['scorer_policy']
    document['evaluation_date']='2026-11-03'
    refs=company_judgments.build_company_scopes(scoring_input=document,round_id=row['round_id'],
        network_name='finney',netuid=71,scorer_image_digest=old['scorer_image_digest'],
        scorer_image_reference=old['scorer_image_reference'],integrity_policy='arena_integrity_v1',
        company_judgment_policy='company_judgment_v1')
    row['configuration_doc']={**old,'scorer_image_digest':'sha256:'+'d'*64,
                              'scorer_image_reference':'registry/scorer@sha256:'+'d'*64}
    source=dict(run_id='source',round_id=row['round_id'],submission_id='miner',
                miner_hotkey='miner',stage=1,icp_position=0,kind='execute',
                status='accepted',terminal_cause='accepted',output_ref='execute.json')
    judge=dict(source,run_id='judge',kind='score',scored_run_id='source',
               company_judgment_refs=refs,output_ref='score.json')
    proof={key:old[key] for key in ('scorer_policy','scorer_image_digest','scorer_image_reference')}
    proof.update(operation='nov03_forward_only_verifier_v1',scoring_generation_floor=0,
                 **{'current_'+key:row['configuration_doc'][key]
                    for key in ('scorer_image_digest','scorer_image_reference')})
    service=ArenaService.__new__(ArenaService)
    service._store=SimpleNamespace(preserved_judgment_policy=Mock(return_value=proof))
    service._round_from_context=lambda *args:row
    return service,row,source,judge,proof


def test_old_image_needs_exact_readonly_authority_and_preserves_rows():
    service,row,source,judge,proof=context()
    before=deepcopy((row,source,judge))
    service._store.preserved_judgment_policy.return_value=None
    assert not service._judgment_matches_execution(row,judge,source)
    service._store.preserved_judgment_policy.return_value=proof
    assert service._judgment_matches_execution(row,judge,source)
    assert (row,source,judge)==before
    service._store.preserved_judgment_policy.assert_called_with(
        row['round_id'],'judge','source',judge['company_judgment_refs'])


def test_old_breakdowns_use_original_policy_only_after_exact_authority():
    service,row,source,judge,proof=context()
    service._company_quality_breakdowns=Mock(return_value=[])
    current=deepcopy(proof['scorer_policy'])
    current['judge_models']['industry_evidence_review']='different/model'
    assert service._verified_breakdowns(judge,icp={},companies=[],policy=current,round_context=row)==[]
    assert service._company_quality_breakdowns.call_args.kwargs['policy']==proof['scorer_policy']
    service._store.preserved_judgment_policy.return_value=None
    with pytest.raises(scoring.ScoringError,match='preserved judgment authority is missing'):
        service._verified_breakdowns(judge,icp={},companies=[],policy=current,round_context=row)


@pytest.mark.parametrize('field,value',[('operation','uniform_regrade'),('scoring_generation_floor',1),
    ('current_scorer_image_digest','sha256:'+'f'*64)])
def test_changed_receipt_binding_fails_closed(field,value):
    service,row,source,judge,proof=context()
    proof[field]=value
    with pytest.raises(ServiceError,match='preserved_judgment_policy_invalid'):
        service._judgment_matches_execution(row,judge,source)


def test_other_round_and_current_image_keep_existing_authority_path():
    service,row,source,judge,proof=context()
    row['configuration_doc'].update({key:proof[key] for key in ('scorer_image_digest','scorer_image_reference')})
    assert service._judgment_matches_execution(row,judge,source)
    service._store.preserved_judgment_policy.assert_not_called()
    row['round_id']=judge['round_id']=source['round_id']='arena-other'
    for ref in judge['company_judgment_refs']:
        ref['scope_doc']['round_id']=row['round_id']
        ref['scope_doc']['scorer_image_digest']=proof['scorer_image_digest']
    assert not service._judgment_matches_execution(row,judge,source)
    service._store.preserved_judgment_policy.assert_not_called()
