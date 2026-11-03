"""Publication may reuse immutable proofs only inside one qualification call."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lab_arena import scoring
from lab_arena import contracts
from lab_arena import service as service_module
from lab_arena.service import _PublicationReads, _PublicResultEvidenceReads
from tests.lab_arena.completed_score_evidence_reads_test import _completed_company_case
from tests.lab_arena.scoring_validator_attribution_test import SUBMISSION_ID, VALIDATOR_A


def case():
    service, row, cached, source, objects, counts, _ = _completed_company_case()
    runs = [run for run in service._store.rows.values()
            if run['kind'] == 'execute' and run['submission_id'] == SUBMISSION_ID]
    return service, row, runs, cached, source, objects, counts


def qualify(service, row, runs, **kwargs):
    return service._qualified_company_counts(
        row, SUBMISSION_ID, runs, positions=(0, 1),
        publication_reads=_PublicationReads(service, row, runs), **kwargs,
    )


def test_publication_reuses_proofs_and_keeps_exact_qualified_counts():
    service, row, runs, _, _, _, counts = case()
    expected = service._qualified_company_counts(row, SUBMISSION_ID, runs, positions=(0, 1))
    assert counts['company_rows'] == counts['run_rows'] == counts['source-output'] == 4
    counts.clear()
    assert qualify(service, row, runs) == expected == {0: 1, 1: 1}
    assert counts['company_rows'] == counts['run_rows'] == counts['source-output'] == 1
    assert counts['round_rows'] == 0
    assert counts['current-output'] == counts['current-output-2'] == 1
    assert counts['execution-output'] == counts['execution-output-2'] == 1


@pytest.mark.parametrize('fault', ['source_output', 'cache_hash', 'source_claim', 'excluded_consumer'])
def test_later_publication_call_rechecks_changed_authority(fault):
    service, row, runs, cached, source, objects, counts = case()
    assert qualify(service, row, runs) == {0: 1, 1: 1}
    if fault == 'source_output':
        objects['source-output']['company_judgments'][0]['raw_judgment']['final_score'] = 1
    elif fault == 'cache_hash':
        cached['evidence_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'source_claim':
        source['claim_request_hash'] = 'sha256:' + 'f' * 64
    else:
        service._store.rows['score-current']['miner_hotkey'] = VALIDATOR_A
    with pytest.raises(scoring.ScoringError):
        qualify(service, row, runs)
    assert counts['company_rows'] >= 2


def test_supplied_empty_cache_is_honored():
    service, row, runs, _, _, _, counts = case()
    supplied = _PublicResultEvidenceReads()
    assert not supplied
    assert qualify(service, row, runs, evidence_reads=supplied) == {0: 1, 1: 1}
    assert len(supplied) == 1
    assert counts['company_rows'] == counts['source-output'] == 1


@pytest.mark.parametrize('slot', [True, 0.0])
def test_invalid_slot_cannot_alias_a_warm_integer_proof(slot):
    service, row, runs, _, _, _, _ = case()
    memo = _PublicResultEvidenceReads()
    assert qualify(service, row, runs, evidence_reads=memo) == {0: 1, 1: 1}
    service._store.rows['score-current']['claim_response']['company_judgment_cache']['hits'][0]['authority_slot'] = slot
    with pytest.raises(scoring.ScoringError, match='lease context is invalid'):
        qualify(service, row, runs, evidence_reads=memo)


@pytest.mark.parametrize('bound', ['MAX_IDENTITIES', 'MAX_BYTES'])
def test_raw_cache_overflow_keeps_validation_and_fallback(monkeypatch, bound):
    service, row, runs, _, _, _, _ = case()
    contexts = []

    class ObservedReads(_PublicResultEvidenceReads):
        def __init__(self):
            super().__init__()
            contexts.append(self)

    monkeypatch.setattr(ObservedReads, bound, 0)
    monkeypatch.setattr(service_module, '_PublicResultEvidenceReads', ObservedReads)
    assert qualify(service, row, runs) == {0: 1, 1: 1}
    assert len(contexts) == 1
    memo = contexts[0]
    assert memo.runs == memo.companies == memo.objects == memo.submissions == {}
    assert memo._bytes == 0
    # The existing validated-proof map is separate from the raw-byte limit.
    assert len(memo) == 1


def test_nonpublication_call_keeps_original_read_path(monkeypatch):
    service, row, runs, _, _, _, counts = case()
    contexts = []

    class ObservedReads(_PublicResultEvidenceReads):
        def __init__(self):
            super().__init__()
            contexts.append(self)

    monkeypatch.setattr(service_module, '_PublicResultEvidenceReads', ObservedReads)
    assert service._qualified_company_counts(row, SUBMISSION_ID, runs, positions=(0, 1)) == {0: 1, 1: 1}
    assert contexts == []
    assert counts['company_rows'] == counts['source-output'] == 4


@pytest.mark.parametrize(('margin', 'outcome'), [(7.99, 'no_king'), (8.0, 'crowned')])
def test_full_company_publication_equals_uncached_document(monkeypatch, margin, outcome):
    service, row, runs, cached, _, objects, counts = case()
    baseline = row['participants'][0]['submission_id']
    baseline_hotkey = runs[0]['miner_hotkey'] = '5' + 'Q' * 47
    challenger_hotkey = '5' + 'R' * 47
    row.update(status='scored', evaluation_date='2026-09-02', finalists=['challenger'])
    row['participants'] = [
        {'submission_id': baseline, 'miner_hotkey': baseline_hotkey, 'is_king': True},
        {'submission_id': 'challenger', 'miner_hotkey': challenger_hotkey, 'is_king': False},
    ]
    row['configuration_doc'].update(
        promotion_margin=8.0,
        scorer_image_digest=cached['scope_doc']['scorer_image_digest'],
        scorer_image_reference=cached['scope_doc']['scorer_image_reference'],
    )
    row['configuration_doc']['schedule'] = {'publication_deadline': '2026-09-03T00:00:00Z'}
    for execution in runs:
        judge = next(r for r in service._store.rows.values()
                     if r.get('scored_run_id') == execution['run_id'])
        for r in (execution, judge):
            r.update(stage=1, miner_hotkey=baseline_hotkey, terminal_cause='accepted')
            r.setdefault('result_doc', {})
            if r['result_doc'] is None:
                r['result_doc'] = {}
            r['result_doc']['terminal_status'] = 'accepted'
        execution['per_icp_score'] = 50.0
        clone_execution, clone_judge = deepcopy(execution), deepcopy(judge)
        clone_execution.update(run_id=execution['run_id'] + '-challenger',
                               submission_id='challenger', miner_hotkey=challenger_hotkey,
                               output_ref=execution['output_ref'] + '-challenger', per_icp_score=50.0 + margin)
        clone_judge.update(run_id=judge['run_id'] + '-challenger',
                           scored_run_id=clone_execution['run_id'], submission_id='challenger',
                           miner_hotkey=challenger_hotkey, output_ref=judge['output_ref'] + '-challenger')
        objects[clone_execution['output_ref']] = deepcopy(objects[execution['output_ref']])
        objects[clone_judge['output_ref']] = {
            **deepcopy(objects[judge['output_ref']]), 'scored_run_id': clone_execution['run_id'],
        }
        service._store.rows.update({r['run_id']: r for r in (clone_execution, clone_judge)})
    service._store.get_submission = lambda submission: {'is_king': submission == baseline}
    service._clock = lambda: datetime(2026, 9, 2, 23, 0, tzinfo=timezone.utc)
    captured = []

    def capture(_round, old, new, patch):
        assert (old, new) == ('scored', 'published')
        captured.append(deepcopy(patch))
        return {'status': 'ok'}

    service._store.transition_round = capture
    original = service._qualified_company_counts

    def uncached(*args, **kwargs):
        kwargs.pop('publication_reads', None)
        return original(*args, **kwargs)

    with monkeypatch.context() as reference:
        reference.setattr(service, '_qualified_company_counts', uncached)
        assert service.publish(row['round_id'])['status'] == 'ok'
    reference_reads = counts.copy()
    counts.clear()
    assert service.publish(row['round_id'])['status'] == 'ok'
    assert contracts.canonical_json(captured[0]) == contracts.canonical_json(captured[1])
    document = captured[1]['publication_doc']
    assert document['king_decision']['outcome'] == outcome
    assert len(document['final_ranking']) == 2
    assert all(entry['cost_summary']['eligible_icp_count'] == 2 for entry in document['final_ranking'])
    assert counts['eligibility_queries'] == reference_reads['eligibility_queries'] == 4
    assert counts['submission_cost_queries'] == reference_reads['submission_cost_queries'] == 2
    assert counts['source-output'] < reference_reads['source-output']
