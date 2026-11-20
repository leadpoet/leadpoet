"""Retiring one judgment must not discard its unrelated immutable checkpoints."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena import scoring
from lab_arena.service import ArenaService
from tests.lab_arena.public_result_company_evidence_reads_test import _five_company_source_proofs


@pytest.mark.parametrize('operation', ['operator_accepted_judge_correction_v1',
                                       'operator_accepted_execution_correction_v1'])
def test_retired_accepted_source_preserves_all_independently_bound_checkpoints(operation):
    service, refs, _, source, _, _ = _five_company_source_proofs(accepted_source=True)
    before = [service._company_judgment_evidence(cache_key=r['cache_key'],
        company_input_hash=r['company_input_hash'], authority_slot=0) for r in refs]
    source.update(status='failed', terminal_cause='stage_closed',
                  terminal_doc={'operation': operation})
    proofs = []
    service._store.retired_accepted_judgment = lambda run_id: proofs.append(run_id) or True
    after = [service._company_judgment_evidence(cache_key=r['cache_key'],
        company_input_hash=r['company_input_hash'], authority_slot=0) for r in refs]
    assert after == before
    assert proofs == ['score-source'] * 5


@pytest.mark.parametrize('fault', ['no_proof', 'wrong_cause', 'wrong_operation', 'source_hash',
                                  'runner', 'input', 'partial_output'])
@pytest.mark.parametrize('operation', ['operator_accepted_judge_correction_v1',
                                       'operator_accepted_execution_correction_v1'])
def test_retired_source_does_not_bypass_substantive_evidence_validation(fault, operation):
    service, refs, caches, source, objects, _ = _five_company_source_proofs(
        accepted_source=fault != 'partial_output')
    source.update(status='failed', terminal_cause='stage_closed',
                  terminal_doc={'operation': operation})
    service._store.retired_accepted_judgment = lambda _: fault != 'no_proof'
    if fault == 'wrong_cause': source['terminal_cause'] = 'judge_error'
    if fault == 'wrong_operation': source['terminal_doc']['operation'] = 'not-authorized'
    if fault == 'source_hash': objects['source-output']['scored_run_id'] = 'other'
    if fault == 'runner': source['runner_hotkey'] = 'other'
    if fault == 'input': caches[(refs[0]['cache_key'],0)]['company_input_hash'] = 'other'
    with pytest.raises(scoring.ScoringError):
        service._company_judgment_evidence(cache_key=refs[0]['cache_key'],
            company_input_hash=refs[0]['company_input_hash'], authority_slot=0)


def test_score_projection_uses_the_original_chosen_judgment_without_requery():
    service = object.__new__(ArenaService)
    recorded = []
    service._store = SimpleNamespace(record_run_scores=lambda *args, **kwargs: recorded.append((args,kwargs)))
    scores = [{'run_id':'exec-1','per_icp_score':23,'qualification_doc':{'companies':[]}}]
    original = deepcopy(scores)
    service._record_current_run_scores({'round_id':'round-1','scoring_generation_floor':2},2,scores,
                                      judgments={'exec-1':{'run_id':'accepted-judge-2'}})
    assert recorded == [(('round-1',2,[{**scores[0],'score_run_id':'accepted-judge-2'}]),
                        {'scoring_generation_floor':2})]
    assert scores == original


@pytest.mark.parametrize('proof', [True, False])
def test_retired_execution_keeps_only_proved_company_attribution(proof):
    service, refs, caches, source, _, _ = _five_company_source_proofs(accepted_source=True)
    execution = service._store.rows['execute-source']
    execution['miner_hotkey'] = source['miner_hotkey']
    terminal = {'operation': 'operator_accepted_execution_correction_v1',
                'request_id': 'correction', 'replacement_run_id': 'replacement'}
    for row in (source, execution):
        row.update(status='failed', terminal_cause='stage_closed', terminal_doc=terminal)
    service._store.retired_accepted_judgment = lambda _: proof
    evidence = caches[(refs[0]['cache_key'], 0)]['evidence_doc']
    kwargs = dict(source_score_run_id=source['run_id'], source_scored_run_id=execution['run_id'],
                  source_runner_hotkey=source['runner_hotkey'], round_id=source['round_id'],
                  runs_by_id=service._store.rows, company_evidence=evidence)
    if proof:
        assert service._validated_attribution_source(**kwargs) == source['runner_hotkey']
    else:
        with pytest.raises(scoring.ScoringError):
            service._validated_attribution_source(**kwargs)
