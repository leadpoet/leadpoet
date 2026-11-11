"""Saved failed-attempt feedback stays separate from accepted labels."""
from copy import deepcopy
import json

import pytest

from lab_arena import contact_policy, contracts, scoring, training_export, training_thinking
from lab_arena.service import _score_item_output_ref
from tests.lab_arena.training_export_signal_test import _company_cache_fixture
from tests.lab_arena.training_export_test import save_input
from lab_arena.training_input import read_input


def values_with_partial_feedback(*, accepted=False):
    values = _company_cache_fixture(signal_policy=False)
    source, judge = values['runs'][:2]
    output = json.loads(values['objects'].values[judge['output_ref']])
    if accepted:
        failed = deepcopy(judge)
        failed.update(run_id='judge:failed', attempt=0)
        document = read_input(values['objects'], judge['run_id'], 'score',
                              judge['lease_generation'], judge['claim_request_hash'])['input']
        save_input(values['objects'], failed['run_id'], 'score', document)
        values['runs'].append(failed)
    else:
        failed = judge
    failed.update(status='failed', terminal_cause='judge_error')
    failed['result_doc']['terminal_status'] = 'judge_error'
    failed['result_doc']['started_at'] = '2026-10-16T00:00:00Z'
    partial = scoring.build_scoring_failure(source['run_id'], 'judge_error',
        reason='provider_error', company_judgments=output['company_judgments'])
    failed['output_ref'] = _score_item_output_ref(failed['round_id'], failed['run_id'],
                                                contracts.document_hash(partial))
    values['objects'].put(failed['output_ref'], contracts.canonical_json(partial).encode())
    # Real November10 rows have no recorded output_hash column.
    failed.pop('output_hash', None)
    return values, failed, partial


def feedback(example, identity):
    return next(attempt for attempt in example['attempts'] if attempt['run_id'] == identity)['partial_judge_feedback']


def test_real_shaped_failed_judge_output_is_available_in_raw_and_readable_without_labels():
    values, failed, partial = values_with_partial_feedback()
    example = training_export.assemble_example(**values)
    saved = feedback(example, failed['run_id'])
    assert saved['status'] == 'available'
    assert saved['output'] == partial
    assert saved['source_ref'] == failed['output_ref']
    assert saved['output_sha256'] == contracts.document_hash(partial)
    assert saved['recorded_hash_present'] is False
    assert saved['ground_truth'] == 'not_established'
    assert saved['final_label'] is False
    assert saved['scope'] == 'failed_attempt_partial_feedback'
    assert 'judgment' not in example and 'score_run_id' not in example and example['companies'] == []
    assert example['label_authority']['origin'] == 'unavailable'
    assert example['completeness']['dimensions']['input_output_label'] is False
    report = training_thinking.readable_trace(example)
    attempts = next(item['value'] for item in report['document_context'] if item['field'] == 'attempts')
    assert feedback({'attempts': attempts}, failed['run_id']) == saved
    assert example['example_hash'] == contracts.document_hash({k:v for k,v in example.items() if k != 'example_hash'})


@pytest.mark.parametrize('damage', [
    'scored_run', 'reference', 'ref_digest', 'schema', 'recorded_hash', 'claim',
    'input_hash', 'input_privacy', 'input_company', 'input_policy', 'input_lease',
    'company_key', 'company_input', 'company_slot', 'raw_mode',
])
def test_invalid_feedback_is_reported_without_removing_the_export(damage):
    values, failed, partial = values_with_partial_feedback()
    original_ref = failed['output_ref']
    document = read_input(values['objects'], failed['run_id'], 'score',
                          failed['lease_generation'], failed['claim_request_hash'])['input']
    if damage == 'scored_run': partial['scored_run_id'] = 'foreign-execution'
    elif damage == 'reference': failed['output_ref'] = original_ref.replace(failed['run_id'], 'foreign-judge')
    elif damage == 'ref_digest': failed['output_ref'] = original_ref[:-69] + '0' * 64 + '.json'
    elif damage == 'schema': partial['schema_version'] = 'wrong'
    elif damage == 'recorded_hash': failed['output_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'claim': failed['claim_request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage in {'input_hash', 'input_privacy'}:
        from lab_arena import training_input
        snapshot = read_input(values['objects'], failed['run_id'], 'score',
                              failed['lease_generation'], failed['claim_request_hash'])
        if damage == 'input_hash': snapshot['input_sha256'] = contracts.hash_bytes(b'wrong')
        else: snapshot['privacy'] = {'redacted': True, 'redactions': ['credential']}
        values['input_snapshots'] = {failed['run_id']: snapshot}
    elif damage == 'input_company': document['companies'][0]['company_name'] = 'Changed saved company'
    elif damage == 'input_policy': document['scorer_policy'] = {}
    elif damage == 'input_lease': document['company_judgment_cache']['misses'][0]['authority_slot'] = 1
    elif damage == 'company_key': partial['company_judgments'][0]['cache_key'] = contracts.hash_bytes(b'wrong')
    elif damage == 'company_input': partial['company_judgments'][0]['company_input_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'company_slot': partial['company_judgments'][0]['authority_slot'] = 1
    elif damage == 'raw_mode': partial['company_judgments'][0]['raw_judgment'].pop('company_judgment_policy')
    if damage in {'input_company', 'input_policy', 'input_lease'}:
        save_input(values['objects'], failed['run_id'], 'score', document)
    if damage not in {'reference', 'ref_digest'}:
        # Keep native content addressing consistent so deeper validators run.
        failed['output_ref'] = _score_item_output_ref(failed['round_id'], failed['run_id'],
                                                    contracts.document_hash(partial))
    values['objects'].put(failed['output_ref'], contracts.canonical_json(partial).encode())
    example = training_export.assemble_example(**values)
    artifact = feedback(example, failed['run_id'])
    assert artifact['status'] == 'invalid', artifact
    assert 'output' not in artifact
    assert artifact.get('reason')
    assert 'task-private-sensitive-value' not in json.dumps(example)
    assert 'judgment' not in example and example['label_authority']['origin'] == 'unavailable'


@pytest.mark.parametrize('missing', ['ref', 'object', 'input'])
def test_absent_feedback_is_explicit_and_does_not_fail_export(missing):
    values, failed, _ = values_with_partial_feedback()
    if missing == 'ref': failed['output_ref'] = None
    elif missing == 'object': del values['objects'].values[failed['output_ref']]
    else:
        from lab_arena.training_input import object_ref
        del values['objects'].values[object_ref(failed['run_id'], failed['lease_generation'])]
    result = training_export.assemble_example(**values)
    artifact = feedback(result, failed['run_id'])
    assert artifact['status'] == 'unavailable'
    assert 'output' not in artifact
    assert result['run_id'] == values['run_id']


@pytest.mark.parametrize('private', [False, True])
def test_partial_feedback_does_not_change_successful_labels_eligibility_or_capture_gaps(private):
    from lab_arena import training_eligibility
    values, failed, partial = values_with_partial_feedback(accepted=True)
    if private:
        partial['detail'] = 'Authorization: Bearer task-private-sensitive-value'
        failed['output_ref'] = _score_item_output_ref(failed['round_id'], failed['run_id'],
                                                    contracts.document_hash(partial))
        values['objects'].put(failed['output_ref'], contracts.canonical_json(partial).encode())
    absent_values = deepcopy(values)
    next(row for row in absent_values['runs'] if row['run_id'] == failed['run_id'])['output_ref'] = None
    before = training_export.assemble_example(**absent_values)
    after = training_export.assemble_example(**values)
    artifact = feedback(after, failed['run_id'])
    assert artifact['status'] == ('privacy_filtered' if private else 'available')
    if not private:
        assert artifact['output'] == partial
    else:
        assert 'task-private-sensitive-value' not in json.dumps(after)
        assert artifact['privacy']['redacted'] is True
    for key in ('judgment', 'companies', 'output', 'label_authority', 'privacy', 'completeness',
                'provider_calls', 'source', 'input_snapshots', 'competition_cost_eligibility'):
        assert before.get(key) == after.get(key), key
    for example in (before, after):
        assert next(a for a in example['attempts'] if a['run_id'] == example['score_run_id']).get('partial_judge_feedback') is None
    def eligibility(example):
        return training_eligibility.assess_capture(values['submission_row'], {values['run_id']: example},
            [values['run_id']], enabled=True, expected_run_count=1, minimum_coverage_percent=80,
            maximum_missing_reasoning_percent=1)
    assert eligibility(before) == eligibility(after)


def test_available_feedback_owns_its_document_and_roundtrips_json_without_mutating_rows():
    values, failed, _ = values_with_partial_feedback()
    original = deepcopy(values['runs'])
    example = training_export.assemble_example(**values)
    saved = feedback(example, failed['run_id'])
    assert json.loads(contracts.canonical_json(saved)) == saved
    saved['output']['company_judgments'][0]['raw_judgment']['failure_reason'] = 'changed exported copy'
    assert values['runs'] == original
    assert 'changed exported copy' not in values['objects'].values[failed['output_ref']].decode()


@pytest.mark.parametrize('private_field', ['credential', 'opaque_reasoning'])
def test_useful_sanitized_feedback_survives_with_nonexact_privacy_provenance(private_field):
    values, failed, partial = values_with_partial_feedback()
    if private_field == 'credential':
        partial['detail'] = 'Authorization: Bearer task-private-sensitive-value'
    else:
        partial['company_judgments'][0]['raw_judgment']['encrypted_content'] = 'task-private-sensitive-value'
    failed['output_ref'] = _score_item_output_ref(failed['round_id'], failed['run_id'],
                                                contracts.document_hash(partial))
    values['objects'].put(failed['output_ref'], contracts.canonical_json(partial).encode())
    example = training_export.assemble_example(**values)
    artifact = feedback(example, failed['run_id'])
    assert artifact['status'] == 'privacy_filtered'
    assert artifact['verified_for_training'] is False and artifact['final_label'] is False
    assert artifact['privacy']['redacted'] is True
    assert ('credential' if private_field == 'credential' else 'hidden_reasoning') in artifact['privacy']['redactions']
    assert len(artifact['output']['company_judgments']) == 1
    assert artifact['output_sha256'] == contracts.document_hash(partial)
    assert artifact['exported_output_sha256'] == contracts.document_hash(artifact['output'])
    assert artifact['exported_output_sha256'] != artifact['output_sha256']
    assert 'task-private-sensitive-value' not in json.dumps(example)
    assert example['label_authority']['origin'] == 'unavailable'
    readable = training_thinking.readable_trace(example)
    assert 'task-private-sensitive-value' not in json.dumps(readable)


def test_invalid_reference_metadata_is_sanitized_even_without_a_payload():
    values, failed, _ = values_with_partial_feedback()
    failed['output_ref'] = 'Authorization: Bearer task-private-sensitive-value'
    result = training_export.assemble_example(**values)
    artifact = feedback(result, failed['run_id'])
    assert artifact['status'] == 'invalid' and artifact['reason'] == 'output_reference_mismatch'
    assert artifact['privacy'] == {'redacted': True, 'redactions': ['credential']}
    assert 'task-private-sensitive-value' not in json.dumps(result)


def alternate_policy_values(*, quality):
    from lab_arena.output import validate_output_document
    from tests.lab_arena.training_export_test import fixture
    if quality:
        from tests.lab_arena.company_judgments_test import _input, _new
    else:
        from tests.lab_arena.legacy_partial_runtime_test import _input, _row
    values = fixture()
    source, judge = values['runs']
    document = _input()
    configuration = values['round_row']['configuration_doc']
    configuration['scorer_policy'] = document['scorer_policy']
    if quality:
        configuration['company_quality_policy'] = document['scorer_policy']['company_quality_policy']
    benchmark = json.loads(values['objects'].values['benchmark'])
    benchmark['icps'] = [document['icp'] for _ in benchmark['icps']]
    values['objects'].put('benchmark', contracts.canonical_json(benchmark).encode())
    schema = contact_policy.output_schema(configuration)
    output = validate_output_document({'schema_version': schema, 'companies': document['companies']},
                                     expected_schema_version=schema, require_intent_dates=False)
    values['objects'].put(source['output_ref'], contracts.canonical_json(output).encode())
    save_input(values['objects'], source['run_id'], 'execute', {
        'icp': document['icp'], 'evaluation_date': values['round_row']['evaluation_date']})
    document.update(scored_run_id=source['run_id'], evaluation_date=values['round_row']['evaluation_date'],
                    companies=output['companies'])
    if quality:
        from lab_arena import company_judgments as cj
        refs = cj.build_company_scopes(scoring_input=document, round_id=judge['round_id'],
            network_name='finney', netuid=71, scorer_image_digest=configuration['scorer_image_digest'],
            scorer_image_reference=configuration['scorer_image_reference'], integrity_policy='arena_integrity_v1',
            company_quality_policy=document['scorer_policy']['company_quality_policy'])
        lease = {'schema_version': cj.LEASE_SCHEMA_VERSION, 'hits': [], 'misses': [
            {**{key: ref[key] for key in ('company_index', 'cache_key', 'company_input_hash')}, 'authority_slot': 0}
            for ref in refs]}
        document['company_judgment_cache'] = lease
        judge['claim_response'] = {'company_judgment_cache': lease}
        judge['company_judgment_refs'] = refs
        partial = scoring.build_scoring_failure(source['run_id'], 'judge_error', company_judgments=[_new(refs[0])])
    else:
        partial = scoring.build_scoring_failure(source['run_id'], 'judge_error', terminal_breakdowns=[_row()])
    save_input(values['objects'], judge['run_id'], 'score', document)
    judge.update(status='failed', terminal_cause='judge_error')
    judge['result_doc']['terminal_status'] = 'judge_error'
    judge['output_ref'] = _score_item_output_ref(judge['round_id'], judge['run_id'], contracts.document_hash(partial))
    values['objects'].put(judge['output_ref'], contracts.canonical_json(partial).encode())
    return values, judge, partial


@pytest.mark.parametrize('quality', [False, True])
def test_partial_feedback_uses_existing_legacy_and_company_quality_validators(quality):
    values, judge, partial = alternate_policy_values(quality=quality)
    result = training_export.assemble_example(**values)
    artifact = feedback(result, judge['run_id'])
    assert artifact['status'] == 'available', artifact
    assert artifact['output'] == partial
    assert artifact['verified_for_training'] is False and artifact['final_label'] is False
    assert 'judgment' not in result and result['companies'] == []


@pytest.mark.parametrize('damage', [None, 'receipt_hash', 'receipt_reference', 'terminal_index', 'terminal_score'])
def test_legacy_run_only_output_reference_retains_computed_and_recorded_hash_provenance(damage):
    values, judge, partial = alternate_policy_values(quality=False)
    judge['output_ref'] = 'arena/%s/scores/items/%s.json' % (judge['round_id'], judge['run_id'])
    if damage == 'terminal_index': partial['terminal_breakdowns'][0]['company_index'] = 5
    elif damage == 'terminal_score': partial['terminal_breakdowns'][0]['final_score'] = 101
    judge['result_doc']['legacy_partial_checkpoint'] = {
        'source_output_ref': judge['output_ref'], 'source_output_hash': contracts.document_hash(partial)}
    if damage == 'receipt_hash': judge['result_doc']['legacy_partial_checkpoint']['source_output_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'receipt_reference': judge['result_doc']['legacy_partial_checkpoint']['source_output_ref'] = 'foreign'
    values['objects'].put(judge['output_ref'], contracts.canonical_json(partial).encode())
    result = training_export.assemble_example(**values)
    artifact = feedback(result, judge['run_id'])
    assert artifact['status'] == ('available' if damage is None else 'invalid')
    if damage is None:
        assert artifact['output'] == partial and artifact['recorded_hash_present'] is True
    else:
        assert 'output' not in artifact


def test_successful_only_export_keeps_the_existing_attempt_shape():
    values = _company_cache_fixture(signal_policy=False)
    result = training_export.assemble_example(**values)
    assert result['score_run_id'] == values['runs'][1]['run_id']
    assert all('partial_judge_feedback' not in attempt for attempt in result['attempts'])


def test_native_credential_error_remapping_preserves_judge_error_feedback():
    values, judge, partial = values_with_partial_feedback()
    judge['terminal_cause'] = judge['result_doc']['terminal_status'] = 'credential_error'
    result = training_export.assemble_example(**values)
    assert feedback(result, judge['run_id'])['status'] == 'available'
    assert feedback(result, judge['run_id'])['output'] == partial
    assert result['label_authority']['origin'] == 'unavailable'


@pytest.mark.parametrize('raw', [b'not JSON', b'[]', b'null', b'x' * (scoring.MAX_SCORING_OUTPUT_BYTES + 1)])
def test_malformed_or_oversize_optional_output_does_not_erase_the_source(raw):
    values, judge, _ = values_with_partial_feedback()
    values['objects'].put(judge['output_ref'], raw)
    result = training_export.assemble_example(**values)
    artifact = feedback(result, judge['run_id'])
    assert artifact['status'] == 'invalid'
    assert 'output' not in artifact
    assert result['run_id'] == values['run_id']
