"""A public response reuses only its already validated exact round context."""
from collections import Counter
from copy import deepcopy

import pytest

from lab_arena import contracts, signal_judgments
from lab_arena.service import ServiceError
from tests.lab_arena.public_result_company_evidence_reads_test import (
    _make_current_score_the_authority, _public_company_result,
)
from tests.lab_arena.scoring_validator_attribution_test import (
    ROUND_ID, SUBMISSION_ID, VALIDATOR_A,
)


def _round_reads(service):
    reads = Counter()
    original = service._round

    def read(round_id):
        reads[round_id] += 1
        return original(round_id)

    service._round = read
    # Production stores expose this method even for rounds without signal reuse.
    service._store.get_signal_judgment_uses = lambda _run_id: []
    return reads


@pytest.mark.parametrize('reused_failed_source', [False, True])
@pytest.mark.parametrize('published', [False, True])
def test_public_round_context_preserves_response_and_reads_round_once(
    reused_failed_source, published,
):
    service, ref, _, cached, _, objects, _ = _public_company_result()
    if not reused_failed_source:
        _make_current_score_the_authority(service, ref, cached, objects)
    if not published:
        row = service._round(ROUND_ID)
        row['status'] = 'stage2'
        row['participants'] = row['publication_doc']['participants']
        service.completed_submission_scores = lambda _row: {
            SUBMISSION_ID: {'execution_runs': [service._store.rows['execute-current']],
                            'final_score': 1.0},
        }
    reads = _round_reads(service)
    original = service._round_from_context
    service._round_from_context = lambda round_id, _context: service._round(round_id)
    expected = service.public_results(ROUND_ID, SUBMISSION_ID)
    before = reads[ROUND_ID]
    reads.clear()
    service._round_from_context = original
    actual = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert actual['scoring_attribution']['unattributed_icp_count'] == 0
    assert before >= 3
    assert reads == {ROUND_ID: 1}


def test_later_request_reads_current_round_and_dynamic_signal_authority():
    service, _, _, _, _, _, _ = _public_company_result()
    row = service._round(ROUND_ID)
    reads = _round_reads(service)
    assert service.public_results(ROUND_ID, SUBMISSION_ID)['scoring_attribution']['unattributed_icp_count'] == 0
    assert reads == {ROUND_ID: 1}
    # A new request must not retain the prior round's signal policy or proof.
    row['configuration_doc']['scorer_policy']['signal_judgment_policy'] = signal_judgments.POLICY
    exclusions = []
    service._store.signal_authority_exclusions = lambda run_id: exclusions.append(run_id) or []
    with pytest.raises(ServiceError, match='public_company_verification_unavailable'):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert reads == {ROUND_ID: 2}
    assert exclusions and set(exclusions) == {'score-source'}


def test_signal_policy_still_checks_live_authority_with_reused_round():
    service, ref, _, _, source, _, _ = _public_company_result()
    source['company_judgment_refs'] = [ref]
    row = service._round(ROUND_ID)
    row['configuration_doc']['scorer_policy']['signal_judgment_policy'] = signal_judgments.POLICY
    reads = _round_reads(service)
    exclusions = []
    service._store.signal_authority_exclusions = lambda run_id: exclusions.append(run_id) or [VALIDATOR_A]
    actual = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert actual['scoring_attribution']['unattributed_icp_count'] == 0
    assert reads == {ROUND_ID: 1}
    assert exclusions == ['score-source']


@pytest.mark.parametrize('context', [None, {}, {'round_id': 'arena-foreign'}])
def test_absent_or_foreign_context_keeps_validated_round_read(context):
    service, _, _, _, _, _, _ = _public_company_result()
    expected = service._round(ROUND_ID)
    reads = _round_reads(service)
    assert service._round_from_context(ROUND_ID, context) == expected
    assert reads == {ROUND_ID: 1}


def test_foreign_source_round_cannot_use_request_context_or_skip_ownership():
    service, ref, _, _, source, _, _ = _public_company_result()
    context = deepcopy(service._round(ROUND_ID))
    source['round_id'] = 'arena-foreign'
    reads = []

    def blocked(round_id):
        reads.append(round_id)
        raise ServiceError('round_not_owned', 404)

    service._round = blocked
    memo = {}
    with pytest.raises(ServiceError, match='round_not_owned'):
        service._company_judgment_evidence(
            cache_key=ref['cache_key'], company_input_hash=ref['company_input_hash'],
            authority_slot=0, evidence_reads=memo, round_context=context,
        )
    assert reads == ['arena-foreign']
    assert memo == {}
