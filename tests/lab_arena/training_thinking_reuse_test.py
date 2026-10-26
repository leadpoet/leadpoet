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


# Digests bind the complete (all splits, full manifest) result produced before
# reuse was added. Counts below also make the relevant policy outcomes readable.
BASELINES = {'diagnostic_gaps': ('sha256:4fee893a37406c479c7af96f7fe94d56ad990911c3d6f0cedf03b3598421d98c',
                                    2, {'requested_empty': 1}, 1),'legacy': ('sha256:2bf0e6b7e1940731a55580b55a65f403aac4adef53892dbb4b66432dbac14ec5',
            2,
            {'requested_empty': 1},
            1),
 'new_policy': ('sha256:88b2674cf6b745cccafc60fa6a47c1d40edeb9a6904b28438c09bc9c52b89a02',
                2,
                {'requested_empty': 1},
                1),
 'missing_at_limit': ('sha256:5a4731a0987e3467de73e616d081cd4337f6a8e965bbbd25f94aac7edb55c75a',
                      2,
                      {'missing_thinking': 1},
                      1),
 'missing_over_limit': ('sha256:7a48e084e3cc3b662727e8aa75f7286ecb9b986cd17deea46a109e2612d64c51',
                        0,
                        {'missing_reasoning_share_exceeded': 3},
                        1),
 'combined_failure': ('sha256:b06b0a95ee497607990fdee8643f1195bd677a58bd6ae2db9834186e3c5f4498',
                      0,
                      {'missing_reasoning_share_exceeded': 2},
                      1),
 'tampered_route': ('sha256:93007b0baee9bf58a8da04107db28e860f3b7fe6af91ef31e2d5e4ce8e12952f',
                    0,
                    {'trajectory_reasoning_ineligible': 3},
                    1),
 'missing_usage': ('sha256:c1b5698bb40507217bb493460c8166f4f408c76eca89235f84139821f683e7ef',
                   0,
                   {'trajectory_reasoning_ineligible': 3},
                   1),
 'privacy': ('sha256:ac5e3dcbbc97df59821f83e8eabf63d2cdc5b09632dae6a33fbf1d19ec8fe773',
             0,
             {'trajectory_evidence_incomplete': 2},
             0),
 'invalid_policy': ('sha256:2a942863e1c53f90e2e981328b7a81ca9a893fdc812fcf94bbe88e0aac7a9310',
                    0,
                    {'frozen_reasoning_policy_invalid': 3},
                    1),
 'failed_retry': ('sha256:57d8f010703a05d9095972bc08b806513fd481d10a02c0a460272eef4847e51d',
                  3,
                  {},
                  1),
 'cap_retry': ('sha256:f94d1ec7295cad78c58242626e5caf732db1a3701e1c91fb90ca4c372c5a6229', 2, {}, 1)}


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
    splits, historical = training_steps.build([example], RULES)
    for split in ('train', 'eval'):
        for record in splits[split]:
            for ref in record['source_refs']:
                ref['delivered_response_sha256'] = contracts.hash_bytes(b'unretained')
    with patch.object(training_steps, 'build', return_value=(splits, historical)):
        with patch.object(training_thinking, '_trajectory_reason', wraps=training_thinking._trajectory_reason) as validate:
            output, manifest = training_thinking.build([example], RULES)
    assert not output['train'] and not output['eval']
    assert validate.call_count == 0
    assert manifest['counts']['exclusion_reasons'] == {'delivered_response_unavailable': 3}
