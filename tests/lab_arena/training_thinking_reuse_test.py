"""Per-build reuse keeps strict full-thinking outputs and failure counts exact."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, refresh_response_links, submission
from tests.lab_arena.training_hidden_retry_test import recovered_example
from tests.lab_arena.training_steps_test import seal


RULES = training_steps.Rules(production_rounds_only=False)


def example_for(case):
    if case == 'diagnostic_gaps':
        from tests.lab_arena.training_host_result_test import local_runtime_diagnostics
        return local_runtime_diagnostics()
    if case in {'failed_retry', 'cap_retry'}:
        return recovered_example(truncated=case == 'cap_retry')
    example = captured_example()
    if case != 'legacy':
        example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    if case in {'missing_at_limit', 'missing_over_limit', 'combined_failure'}:
        missing = 1 if case == 'missing_at_limit' else 2
        for index, tokens in enumerate((missing, 40, 100 - missing - 40)):
            example['provider_calls'][index]['response']['usage']['completion_tokens_details']['reasoning_tokens'] = tokens
            refresh_response_links(example, index)
    if case == 'combined_failure':
        example['provider_calls'][2]['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'bad')
    elif case == 'tampered_route':
        example['provider_calls'][0]['reasoning_capture']['provider_route'] = 'tampered'
    elif case == 'missing_usage':
        example['provider_calls'][0]['response']['usage'].pop('completion_tokens_details')
        refresh_response_links(example, 0)
    elif case == 'privacy':
        example['provider_calls'][0]['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif case == 'invalid_policy':
        example['configuration']['training_maximum_missing_reasoning_percent'] = False
    return seal(example)


# Digests bind the complete result after shared source grouping. Counts make
# the policy outcomes readable.
BASELINES = {'diagnostic_gaps': ('sha256:d47052b6a53fdcf6c44b9717664e6f748d6962b642faf838e60a91d971f061e1',
                                    2, {'requested_empty': 1}, 2),'legacy': ('sha256:8b956f89410e040da8ca935fb924ba776ac5d4adab95a5fc4a2ca4fa11198fc7',
            2,
            {'requested_empty': 1},
            2),
 'new_policy': ('sha256:7582907f59288f994599dfb47a826af95a67477dd612c89710e75501b146448c',
                2,
                {'requested_empty': 1},
                2),
 'missing_at_limit': ('sha256:487b590a05407687ffab74f47343b845ba81fc853967f2ee8f5feabea19d97a1',
                      2,
                      {'missing_thinking': 1},
                      2),
 'missing_over_limit': ('sha256:a111618f9d3f4edc3264f214b4ec0879d18c740c226446c7093c8a875e635cc9',
                        2,
                        {'missing_thinking': 1},
                        1),
 'combined_failure': ('sha256:5953da43b2decebd9d54cf7562ddb6c7f01f2d32dfbaed57f1cc64f979187da5',
                      1,
                      {'missing_thinking': 1},
                      1),
 'tampered_route': ('sha256:fbd2869e10e1293486a6d3747e97e8a6bc5154b2b81dfe620c4302730f6716d3',
                    2,
                    {'requested_empty': 1},
                    1),
 'missing_usage': ('sha256:bc50ceb37b375ee156f1d93a060948e5349d5a3de00d59120464dde924a30707',
                   2,
                   {'requested_empty': 1},
                   1),
 'privacy': ('sha256:f64f664153edef7e2557776999dcaf2e2ef458ea5d9137c58d6a389c5251002b',
             2,
             {},
             1),
 'invalid_policy': ('sha256:fde2839486f7ada8ca61d87003ad32300c877ba7fba005b5d1485a27b04796d4',
                    0,
                    {'trajectory_evidence_incomplete': 3},
                    1),
 'failed_retry': ('sha256:295c113687a0b27e3ca52c1098cf90e6de230bb6bc6233dface0d72e1a78b525',
                  3,
                  {},
                  2),
 'cap_retry': ('sha256:43b5ac16d9bcdfbedcbd2b142ad4cc2db1a632b7de2c486dceb91e04beb7c5ba', 2, {}, 2)}


@pytest.mark.parametrize('case', BASELINES)
def test_reuse_preserves_complete_result_and_counts_once_per_source(case):
    expected_hash, expected_rows, expected_reasons, expected_validations = BASELINES[case]
    example = example_for(case)
    before = deepcopy(example)
    with patch.object(training_thinking, 'selected_research_evidence',
                       wraps=training_thinking.selected_research_evidence) as selected:
        result = training_thinking.build([example], RULES)
    splits, manifest = result
    assert contracts.document_hash(result) == expected_hash
    assert len(splits['train']) + len(splits['eval']) == expected_rows
    assert manifest['counts']['exclusion_reasons'] == expected_reasons
    assert selected.call_count == expected_validations
    assert example == before


def test_reuse_is_local_to_one_build_and_rechecks_changed_saved_evidence():
    example = example_for('new_policy')
    with patch.object(training_thinking, 'selected_research_evidence',
                       wraps=training_thinking.selected_research_evidence) as selected:
        first = training_thinking.build([example], RULES)
        # An independent build must revalidate even an unchanged hash.
        assert training_thinking.build([deepcopy(example)], RULES) == first
        assert selected.call_count == 4
        example['input_snapshots']['score-run']['input']['scored_run_id'] = 'tampered'
        splits, manifest = training_thinking.build([seal(example)], RULES)
        assert selected.call_count == 6
    assert not splits['train'] and not splits['eval']
    assert manifest['counts']['exclusion_reasons'] == {'trajectory_evidence_incomplete': 3}


@pytest.mark.parametrize('case', BASELINES)
def test_request_control_diagnostics_preserve_targets_policy_and_eligibility(case):
    example = example_for(case)
    before = deepcopy(example)
    current = training_thinking.build([example], RULES)
    attempt_coverage = training_thinking.attempt_coverage
    diagnostic_fields = ('outbound_reasoning_controls', 'outbound_reasoning_controls_status',
                         'probe_reasoning_mode')

    def previous_attempt_shape(*args, **kwargs):
        row = attempt_coverage(*args, **kwargs)
        for field in diagnostic_fields:
            row.pop(field)
        return row

    def eligibility():
        run_id = example['run_id']
        return training_eligibility.assess_capture(
            submission(), {run_id: example}, [run_id], enabled=True,
            minimum_coverage_percent=80, maximum_missing_reasoning_percent=(
                1 if 'training_maximum_missing_reasoning_percent' in example['configuration'] else 0))

    current_eligibility = eligibility()
    with patch.object(training_thinking, 'attempt_coverage', side_effect=previous_attempt_shape):
        previous = training_thinking.build([example], RULES)
        assert eligibility() == current_eligibility
    # Entire records bind targets, controls, source/provenance and record hashes.
    assert current[0] == previous[0]
    projected_manifest = deepcopy(current[1])
    for row in projected_manifest['coverage']['attempts']:
        for field in diagnostic_fields:
            row.pop(field)
    # Only these diagnostics and their derived manifest digest may differ.
    projected_manifest.pop('dataset_version')
    projected_manifest['dataset_version'] = contracts.document_hash(projected_manifest)
    assert projected_manifest == previous[1]
    assert example == before


def test_distinct_sources_sharing_candidates_each_get_their_own_proof():
    example = example_for('new_policy')
    second = deepcopy(example)
    second['submission_id'] = 'other-submission'
    seal(second)
    with patch.object(training_thinking, 'selected_research_evidence',
                       wraps=training_thinking.selected_research_evidence) as selected:
        with patch.object(training_thinking, 'provider_visible_view', wraps=training_thinking.provider_visible_view) as view:
            splits, _ = training_thinking.build([example, second], RULES)
    # Selected evidence is checked once by coverage and once by the builder,
    # then reused for the source's other candidates.
    assert selected.call_count == 4
    assert view.call_count == 6
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all({ref['example_hash'] for ref in record['source_refs']} ==
               {example['example_hash'], second['example_hash']} for record in records)


@pytest.mark.parametrize('damage', ['missing_over_limit', 'bad_route'])
def test_bad_other_turn_does_not_veto_independent_valid_targets(damage):
    example = example_for('new_policy')
    other = example['provider_calls'][2]
    if damage == 'missing_over_limit':
        message = other['response']['choices'][0]['message']
        for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
            message.pop(field, None)
        other['response']['usage']['completion_tokens_details']['reasoning_tokens'] = 900
        refresh_response_links(example, 2)
    else:
        other['reasoning_capture']['provider_route'] = 'unreviewed-route'
        seal(example)
    splits, manifest = training_thinking.build([example], RULES)
    records = splits['train'] + splits['eval']
    assert {record['source_refs'][0]['call_identity'] for record in records} == {'probe-call-1'}
    assert all(record['generated_thinking'] for record in records)
    if damage == 'missing_over_limit':
        assert manifest['counts']['exclusion_reasons'] == {
            'missing_thinking': 1, 'requested_empty': 1}


def test_malformed_selected_current_pair_is_rejected_without_erasing_other_target():
    example = example_for('new_policy')
    bad = example['provider_calls'][1]
    bad['request']['request_hash'] = contracts.hash_bytes(b'wrong selected request')
    seal(example)
    splits, manifest = training_thinking.build([example], RULES)
    records = splits['train'] + splits['eval']
    assert {record['source_refs'][0]['call_identity'] for record in records} == {'probe-call-2'}
    assert manifest['counts']['exclusion_reasons'] == {'requested_empty': 1}


def test_saved_score_binding_changes_the_sealed_export_digest():
    example = example_for('new_policy')
    original = training_thinking.build([example], RULES)
    changed = deepcopy(example)
    score_snapshot = changed['input_snapshots']['score-run']
    score_snapshot['input']['companies'] = [{'company': 'different accepted label'}]
    score_snapshot['input_sha256'] = contracts.document_hash(score_snapshot['input'])
    changed['output']['companies'] = [{'company': 'different accepted label'}]
    changed['output_identity']['document_hash'] = contracts.document_hash(changed['output'])
    changed['judgment']['per_icp_score'] = 1
    changed = seal(changed)
    updated = training_thinking.build([changed], RULES)
    assert contracts.document_hash(updated) != contracts.document_hash(original)
    assert all(ref['outcome']['per_icp_score'] == 1
               for record in updated[0]['train'] + updated[0]['eval']
               for ref in record['source_refs'])


def test_delivery_failure_drops_candidates_after_source_evidence_reuse():
    example = example_for('new_policy')
    splits, historical = training_steps._build([example], RULES)
    for split in ('train', 'eval'):
        for record in splits[split]:
            for ref in record['source_refs']:
                ref['delivered_response_sha256'] = contracts.hash_bytes(b'unretained')
    def deliver_candidates(examples, rules, *, full_thinking_candidate):
        for source in examples:
            for split in ('train', 'eval'):
                for record in splits[split]:
                    full_thinking_candidate(source, record)
        return splits, historical

    with patch.object(training_steps, '_build', side_effect=deliver_candidates):
        with patch.object(training_thinking, 'selected_research_evidence',
                           wraps=training_thinking.selected_research_evidence) as selected:
            output, manifest = training_thinking.build([example], RULES)
    assert not output['train'] and not output['eval']
    assert selected.call_count == 2
    assert manifest['counts']['exclusion_reasons'] == {'delivered_response_unavailable': 3}
