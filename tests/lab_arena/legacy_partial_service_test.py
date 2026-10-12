"""Legacy checkpoint receipts retain original group context and authority."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring
from lab_arena.service import ArenaService, ServiceError
from qualification.scoring.competition import apply_company_judgment_context
from tests.test_arena_score_integrity import _icp, _positive_breakdown, _public_company


def _bound_source():
    companies = [_public_company(name, domain=name.lower()+'.example') for name in ('A', 'B')]
    policy = scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2')
    document = scoring.build_scoring_input(scored_run_id='execute-1', icp=_icp(), companies=companies, policy=policy, evaluation_date='2026-10-12')
    row = apply_company_judgment_context(companies, [_positive_breakdown('A', 'a.example'), _positive_breakdown('B', 'b.example')])[0]
    output = scoring.build_scoring_failure('execute-1', 'judge_error', terminal_breakdowns=[row])
    round_row = {'round_id': 'arena-2026-10-12', 'evaluation_date': '2026-10-12', 'configuration_doc': {
        'integrity_policy': 'arena_integrity_v1', 'scorer_policy': policy,
        'scorer_image_digest': 'sha256:'+'a'*64, 'scorer_image_reference': 'test@sha256:'+'a'*64}}
    source = {'run_id': 'score-1', 'assignment_id': 'score', 'attempt': 1, 'kind': 'score', 'status': 'failed',
              'terminal_cause': 'judge_error', 'scored_run_id': 'execute-1', 'round_id': round_row['round_id'],
              'submission_id': 'sub-1', 'miner_hotkey': 'miner-1', 'stage': 1, 'icp_position': 0,
              'runner_hotkey': 'validator-1', 'claim_request_id': 'claim-1', 'claim_request_hash': 'sha256:'+'b'*64,
              'lease_generation': 1, 'claim_response': {'runner_authority_exclusions': ['validator-1']},
              'result_doc': {'terminal_status': 'judge_error'}}
    service = ArenaService.__new__(ArenaService)
    service._require_validator_authority = lambda *_: None
    execution_hash = contracts.document_hash({'companies': companies})
    service._legacy_partial_input = lambda *_: (document, execution_hash)
    receipt = service._legacy_partial_receipt(source, round_row, document, execution_hash, output, 'sha256:'+'c'*64)
    source['output_ref'] = receipt['source_output_ref']
    source['result_doc']['legacy_partial_checkpoint'] = receipt
    service._objects = SimpleNamespace(get_bounded=lambda *_: json.dumps(output).encode())
    service._store = SimpleNamespace(get_submission=lambda *_: {'is_king': False}, list_runs=lambda *_a, **_k: [source])
    return service, source, round_row, output, document


def test_committed_failed_source_returns_original_terminal_row_only():
    service, source, round_row, output, _ = _bound_source()
    assert service._validated_legacy_partial(source, round_row) == output['terminal_breakdowns']
    retry = {**source, 'run_id': 'score-2', 'attempt': 2}
    assert service._prior_legacy_terminal_breakdowns(retry, round_row) == output['terminal_breakdowns']
    assert service._prior_legacy_terminal_breakdowns({**retry, 'scored_run_id': 'another-execute'}, round_row) == []
    assert service._prior_legacy_terminal_breakdowns({**retry, 'assignment_id': 'another-assignment'}, round_row) == []


@pytest.mark.parametrize('change', ['not_committed', 'no_receipt', 'completion', 'artifact', 'accepted', 'input', 'image', 'date', 'claim', 'lease', 'authority', 'unregistered', 'retryable'])
def test_unbound_checkpoint_never_returns_resume_rows(change):
    service, source, round_row, output, document = _bound_source()
    if change == 'not_committed':
        source['status'] = 'leased'
    elif change == 'no_receipt':
        source['result_doc'].pop('legacy_partial_checkpoint')
    elif change == 'completion':
        source['result_doc']['legacy_partial_checkpoint']['source_completion_request_hash'] = ''
    elif change == 'artifact':
        output['terminal_breakdowns'][0]['final_score'] = 99
    elif change == 'accepted':
        output.pop('failure')
        output['breakdowns'] = output.pop('terminal_breakdowns')
    elif change == 'input':
        document['companies'].reverse()
    elif change == 'image':
        round_row['configuration_doc']['scorer_image_digest'] = 'sha256:'+'f'*64
    elif change == 'date':
        round_row['evaluation_date'] = '2026-10-13'
    elif change == 'claim':
        source['claim_request_hash'] = 'sha256:'+'f'*64
    elif change == 'lease':
        source['lease_generation'] += 1
    elif change == 'authority':
        source['claim_response']['runner_authority_exclusions'].append('miner-1')
    elif change == 'unregistered':
        def deny(*_):
            raise ServiceError('runner_hotkey_unregistered', 403)
        service._require_validator_authority = deny
    else:
        output['terminal_breakdowns'][0].update(final_score=0, failure_reason='provider timeout')
    with pytest.raises((scoring.ScoringError, ServiceError)):
        service._validated_legacy_partial(source, round_row)
