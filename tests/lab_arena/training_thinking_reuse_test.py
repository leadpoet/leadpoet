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
BASELINES = {'diagnostic_gaps': ('sha256:7d811bfcf5f8828551e9dc64ffc7c3b27b88036c7ace6017761a745f800a243f',
                                    2, {'requested_empty': 1}, 1),'legacy': ('sha256:c87cb732f2b95898ff134fdc1865a297c6b780734e0c2af3124295d18408d92e',
            2,
            {'requested_empty': 1},
            1),
 'new_policy': ('sha256:cbcdbf6b37705e93fb9cf6e369984396c16610c928b6e0ac7909ea955f541939',
                2,
                {'requested_empty': 1},
                1),
 'missing_at_limit': ('sha256:32e370ea000f6ec435a07e8dc840f6a29331f00022884a04a181208d6467648d',
                      2,
                      {'missing_thinking': 1},
                      1),
 'missing_over_limit': ('sha256:b4b4e682ada6ef2660b83e3aba9d3badab68e1c4d692100bb839d054d64e20e8',
                        0,
                        {'missing_reasoning_share_exceeded': 3},
                        1),
 'combined_failure': ('sha256:fccf782635abab7ea8ab75ce2e8593a5710bc653431ffd7501c88e2f6dfe00f0',
                      0,
                      {'missing_reasoning_share_exceeded': 2},
                      1),
 'tampered_route': ('sha256:33e20b97d3c18e74dc51052806789217c7c0bdae7a9beeae7fff94fa10165bd2',
                    0,
                    {'trajectory_reasoning_ineligible': 3},
                    1),
 'missing_usage': ('sha256:c91bef69bc360ecb181335d400ae1b81f0a6ed17c8e1cf240c29fdfedf598bbd',
                   0,
                   {'trajectory_reasoning_ineligible': 3},
                   1),
 'privacy': ('sha256:553cbc7e7bd0eec845e8cca9e3e2943f5174fac5ceac16f3ecf023b9402e65e7',
             0,
             {'trajectory_evidence_incomplete': 2},
             0),
 'invalid_policy': ('sha256:dd2e0b7f3940d7eb9350f6e819c007cdc6db8d695938d22a8da306178db80fab',
                    0,
                    {'frozen_reasoning_policy_invalid': 3},
                    1),
 'failed_retry': ('sha256:4e396526a6f081311c2db42fb0823131fff78304634c93653956366481accdb7',
                  3,
                  {},
                  1),
 'cap_retry': ('sha256:20aa2d26f3db0874b7cc8b8f812c708a23f0cceb66d89383c19ec370a40c155e', 2, {}, 1)}


@pytest.mark.parametrize('case', BASELINES)
def test_reuse_preserves_complete_result_and_counts_once_per_source(case):
    expected_hash, expected_rows, expected_reasons, expected_validations = BASELINES[case]
    example = example_for(case)
    before = deepcopy(example)
    with patch.object(training_thinking, '_trajectory_reason', wraps=training_thinking._trajectory_reason) as validate:
        result = training_thinking.build([example], RULES)
    splits, manifest = result
    assert contracts.document_hash(result) == expected_hash
    assert len(splits['train']) + len(splits['eval']) == expected_rows
    assert manifest['counts']['exclusion_reasons'] == expected_reasons
    assert validate.call_count == expected_validations
    assert example == before


def test_reuse_is_local_to_one_build_and_rechecks_changed_saved_evidence():
    example = example_for('new_policy')
    with patch.object(training_thinking, '_trajectory_reason', wraps=training_thinking._trajectory_reason) as validate:
        first = training_thinking.build([example], RULES)
        # An independent build must revalidate even an unchanged hash.
        assert training_thinking.build([deepcopy(example)], RULES) == first
        assert validate.call_count == 2
        example['provider_calls'][0]['reasoning_capture']['provider_route'] = 'tampered'
        splits, manifest = training_thinking.build([seal(example)], RULES)
        assert validate.call_count == 3
    assert not splits['train'] and not splits['eval']
    assert manifest['counts']['exclusion_reasons'] == {'trajectory_reasoning_ineligible': 3}


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
    with patch.object(training_thinking, '_trajectory_reason', wraps=training_thinking._trajectory_reason) as validate:
        with patch.object(training_thinking, 'provider_visible_view', wraps=training_thinking.provider_visible_view) as view:
            splits, _ = training_thinking.build([example, second], RULES)
    assert validate.call_count == 2
    # Selected evidence is checked by both the build and coverage report.
    # Only one additional diagnostic view is needed per retained source.
    assert view.call_count == 6
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all({ref['example_hash'] for ref in record['source_refs']} ==
               {example['example_hash'], second['example_hash']} for record in records)


def test_delivery_failure_precedes_trajectory_validation():
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
        with patch.object(training_thinking, '_trajectory_reason', wraps=training_thinking._trajectory_reason) as validate:
            output, manifest = training_thinking.build([example], RULES)
    assert not output['train'] and not output['eval']
    assert validate.call_count == 0
    assert manifest['counts']['exclusion_reasons'] == {'delivered_response_unavailable': 3}
