"""Archived judge inputs remain evidence, never replacement label authority."""
import copy

import pytest

from lab_arena import contracts, training_dataset, training_input
from lab_arena.training_reader import export_run
from tests.lab_arena.training_export_forward_only_test import preserved_fixture, PreservedStore
from tests.lab_arena.training_export_test import runtime_binding, save_input


def archived_fixture(monkeypatch):
    values, authority = preserved_fixture(monkeypatch, current=True)
    old, _ = preserved_fixture(monkeypatch, current=False)
    archived = copy.deepcopy(old['runs'][1])
    document = training_input.read_input(old['objects'], archived['run_id'], 'score',
        archived['lease_generation'], archived['claim_request_hash'])['input']
    values['objects'].put('archived-judge-output', old['objects'].values[archived['output_ref']])
    archived.update(run_id='judge:archived', output_ref='archived-judge-output', status='failed',
        terminal_cause='stage_closed', terminal_doc={'operation': 'test_superseded_judgment',
                                                   'reason': 'superseded_old_verifier_judgment'})
    values['runs'][1]['attempt'] = archived['attempt'] + 1
    save_input(values['objects'], archived['run_id'], 'score', document,
        generation=archived['lease_generation'], claim_hash=archived['claim_request_hash'])
    values['runs'].append(archived)
    return values, authority, document


def export(values, authority):
    store = PreservedStore(values, authority)
    return export_run(store, values['objects'], values['run_id']), store


def exclusion(example):
    return training_dataset.example_exclusion(example, training_dataset.Rules(production_rounds_only=False))


def test_archived_failed_old_policy_does_not_poison_selected_current_labels(monkeypatch):
    values, authority, old_document = archived_fixture(monkeypatch)
    before = copy.deepcopy(values['runs'])
    result, store = export(values, authority)
    assert result['score_run_id'] == values['runs'][1]['run_id']
    assert exclusion(result) is None
    assert result['completeness']['complete'] is False  # Historical warning is retained.
    assert result['completeness']['dimensions']['source_identity'] is True
    assert result['completeness']['dimensions']['exact_inputs'] is True
    assert result['completeness']['dimensions']['input_output_label'] is True
    prior_codes = {'prior_attempt_judge_input_context_mismatch',
                   'prior_attempt_reviewer_runtime_identity_unverified'}
    assert prior_codes <= set(result['completeness']['reasons_by_dimension']['research_process'])
    assert prior_codes <= set(result['related_run_gap_attribution']['judge:archived']['research_process_gaps'])
    assert result['input_snapshots']['judge:archived']['input'] == old_document
    archived = next(row for row in result['attempts'] if row['run_id'] == 'judge:archived')
    assert archived['status'] == 'failed' and 'runtime_identity' not in archived
    assert result['judgment']['runtime_identity']['grading_image_digest'] == authority['current_scorer_image_digest']
    assert 'preserved_judgment_policy' not in result['judgment']
    assert store.reads == []  # Failed rows cannot obtain accepted-only604 authority.
    records = training_dataset.records_from_example(result)
    assert records and all(row['meta']['scorer_image_digest'] == authority['current_scorer_image_digest'] for row in records)
    assert values['runs'] == before
    assert result['example_hash'] == contracts.document_hash({k:v for k,v in result.items() if k != 'example_hash'})


@pytest.mark.parametrize('mutation,hard_code', [
    ('judge_policy', 'judge_input_context_mismatch'),
    ('judge_companies', 'judge_input_context_mismatch'),
    ('judge_runtime', 'reviewer_runtime_identity_unverified'),
    ('judge_label_scope', 'judge_evidence_unverified'),
    ('source_context', 'runtime_input_context_mismatch'),
])
def test_selected_faults_remain_hard_with_archived_attempt(monkeypatch, mutation, hard_code):
    values, authority, old_document = archived_fixture(monkeypatch)
    selected = values['runs'][0] if mutation == 'source_context' else values['runs'][1]
    if mutation == 'judge_runtime':
        runtime_binding(values, selected)['grading_image_digest'] = 'sha256:' + 'e' * 64
    elif mutation == 'judge_label_scope':
        selected['company_judgment_refs'][0]['scope_doc']['scorer_image_digest'] = 'sha256:' + 'e' * 64
    else:
        document = training_input.read_input(values['objects'], selected['run_id'], selected['kind'],
            selected['lease_generation'], selected['claim_request_hash'])['input']
        if mutation == 'judge_policy': document['scorer_policy'] = old_document['scorer_policy']
        elif mutation == 'judge_companies': document['companies'] = []
        else: document['icp']['industry'] = 'Different saved source context'
        save_input(values['objects'], selected['run_id'], selected['kind'], document,
            generation=selected['lease_generation'], claim_hash=selected['claim_request_hash'])
    result, _ = export(values, authority)
    assert hard_code in result['completeness']['incomplete_reasons']
    assert exclusion(result) in {'label_unverified', 'label_authority_unverified', 'input_conflict'}
    # An archived warning never overrides the selected hard reason.
    assert 'prior_attempt_judge_input_context_mismatch' in result['completeness']['incomplete_reasons']


@pytest.mark.parametrize('status', ['failed', 'pending'])
def test_archived_judge_cannot_supply_labels_when_replacement_not_accepted(monkeypatch, status):
    values, authority, _ = archived_fixture(monkeypatch)
    values['runs'][1].update(status=status, terminal_cause='lease_expired' if status == 'failed' else None)
    result, store = export(values, authority)
    assert 'final_judge_unavailable' in result['completeness']['incomplete_reasons']
    assert 'score_run_id' not in result and result['companies'] == []
    assert exclusion(result) == 'label_unverified'
    assert store.reads == []


@pytest.mark.parametrize('fault,code', [
    ('snapshot', 'runtime_input_snapshot_invalid'),
    ('privacy', 'runtime_input_privacy_nonexact'),
    ('privacy_shape', 'runtime_input_snapshot_invalid'),
    ('context', 'runtime_input_context_mismatch'),
])
@pytest.mark.parametrize('selected', [False, True])
def test_snapshot_faults_are_scoped_without_granting_prior_authority(monkeypatch, fault, code, selected):
    values, authority, _ = archived_fixture(monkeypatch)
    target = values['runs'][1]['run_id'] if selected else 'judge:archived'
    original = training_input.read_input
    def read(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        if args[1] == target:
            if fault == 'snapshot': snapshot['input_sha256'] = 'sha256:' + '0' * 64
            elif fault == 'privacy': snapshot['privacy'] = {'redacted': True, 'redactions': ['credential']}
            elif fault == 'privacy_shape': snapshot['privacy'] = {'redacted': False, 'redactions': ['credential']}
            else:
                snapshot['input']['icp']['industry'] = 'Different retained context'
                snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
        return snapshot
    monkeypatch.setattr(training_input, 'read_input', read)
    result, store = export(values, authority)
    reasons = result['completeness']['incomplete_reasons']
    if selected:
        assert code in result['completeness']['reasons_by_dimension']['exact_inputs']
        if code in {'runtime_input_snapshot_invalid', 'runtime_input_context_mismatch'}:
            assert exclusion(result) in {'label_unverified', 'input_conflict'}
    else:
        assert code not in reasons
        assert 'prior_attempt_' + code in result['completeness']['reasons_by_dimension']['research_process']
        assert 'prior_attempt_' + code in result['related_run_gap_attribution'][target]['research_process_gaps']
        assert exclusion(result) is None
    assert result['completeness']['complete'] is False
    assert store.reads == []
