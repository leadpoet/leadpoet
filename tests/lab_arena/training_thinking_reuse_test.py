"""Per-build reuse keeps strict full-thinking outputs and failure counts exact."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, refresh_response_links
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
BASELINES = {'diagnostic_gaps': ('sha256:1f652461e4945d14b33b1574e74d29e08103c91564cd1c24dd65bf03d7122e1e',
                                    2, {'requested_empty': 1}, 1),'legacy': ('sha256:03e6d0528b576472b25d028e607b7b82c7aec1ebcb8b68f0e936adf032fb30df',
            2,
            {'requested_empty': 1},
            1),
 'new_policy': ('sha256:3f6070eca4b5885546253ae8735ed090411be60305ce1dcaee9917c7d3d0cd6c',
                2,
                {'requested_empty': 1},
                1),
 'missing_at_limit': ('sha256:1e0ac8dca4fba16ea238b75c4b4cf3da293107a1046a3e69b18f440bf841907a',
                      2,
                      {'missing_thinking': 1},
                      1),
 'missing_over_limit': ('sha256:b8312102f6bff139355adda832db71897de9ffe670beed2ab55c46cf9fcff94a',
                        0,
                        {'missing_reasoning_share_exceeded': 3},
                        1),
 'combined_failure': ('sha256:403d69678e8bd0ec369ae07a484eddeb844bcb864de41b51e77037f29f64be3d',
                      0,
                      {'missing_reasoning_share_exceeded': 2},
                      1),
 'tampered_route': ('sha256:f28d15d7c34efd3ce5a49729d6706fc2a43346eaa62f4a2c3d43e30b4d77155b',
                    0,
                    {'trajectory_reasoning_ineligible': 3},
                    1),
 'missing_usage': ('sha256:2125b81ed65743254bbf88fa0832007f856e2fc83001635a4ca1215814a85c25',
                   0,
                   {'trajectory_reasoning_ineligible': 3},
                   1),
 'privacy': ('sha256:3190e152711788bf7888956f399be70e83e5364628eb7203bc890318cde1cd24',
             0,
             {'trajectory_evidence_incomplete': 2},
             0),
 'invalid_policy': ('sha256:fba0c7383440db855937480f5bd6905b79cd219267f83773d92a233507e85200',
                    0,
                    {'frozen_reasoning_policy_invalid': 3},
                    1),
 'failed_retry': ('sha256:00e1cdeef48081f62b7025c0b3eba3c64925e27ef05fcc2dabcb48a9aa9fe8db',
                  3,
                  {},
                  1),
 'cap_retry': ('sha256:d3a48e8320c61955512adf1d8c41fd6baafaf955bdce92bffc33f799225e0708', 2, {}, 1)}


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
