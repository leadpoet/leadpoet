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
# the generic host-observation path was added. These values include the earlier
# reasoning-default annotation and valid Responses-fixture corrections. Counts
# below also make the relevant policy outcomes readable.
BASELINES = {'diagnostic_gaps': ('sha256:0163c0c82f00d9928ff843f15a42edae7d45bdbaff446c8acd96ade11b9986d3',
                                    2, {'requested_empty': 1}, 1),'legacy': ('sha256:490e3ec9b82cb73edfd7dd7ee7464a57becf6783c4bc3554408823aa7675aa02',
            2,
            {'requested_empty': 1},
            1),
 'new_policy': ('sha256:2ccae9fa8506087650221b58eab4a7cd240fe229e7b255b514874bca86a579b2',
                2,
                {'requested_empty': 1},
                1),
 'missing_at_limit': ('sha256:91cceea8981cf526e07023c1ee336707cb79141702b04d1308884f9ebb4771a4',
                      2,
                      {'missing_thinking': 1},
                      1),
 'missing_over_limit': ('sha256:ddb38fdf58c70e8e4d1d3457d54bd83f29e32742f5e4f48654f1b4cf6bd5c324',
                        0,
                        {'missing_reasoning_share_exceeded': 3},
                        1),
 'combined_failure': ('sha256:3536d2c51d438cc1811a19446622f5abda1e3e11f7238acdec48d88c263e0b4a',
                      0,
                      {'missing_reasoning_share_exceeded': 2},
                      1),
 'tampered_route': ('sha256:6c4dbfc5a74e3203b67fc3f8d7264720c9b5b706d86d675439382eb0fffda2d9',
                    0,
                    {'trajectory_reasoning_ineligible': 3},
                    1),
 'missing_usage': ('sha256:b24ca105f0e6af19f425a85cafd4d0357b4e68ed5bcc8d029076daba4fb52f2b',
                   0,
                   {'trajectory_reasoning_ineligible': 3},
                   1),
 'privacy': ('sha256:8aa73bbc308295dab624d2e028921aaea1c767d419987627e1b82816607f2261',
             0,
             {'trajectory_evidence_incomplete': 2},
             0),
 'invalid_policy': ('sha256:6637f07f755f7a92f14b8dc34d6120f3ca6e4aaa18e4f33c44a43c8f090bcd31',
                    0,
                    {'frozen_reasoning_policy_invalid': 3},
                    1),
 'failed_retry': ('sha256:c7e077ffb63c1251821f470c9f4e0f78c56d4f46c87c570d0747fabaeb327d89',
                  3,
                  {},
                  1),
 'cap_retry': ('sha256:6639aa9c30140f4a09625398dbf388f1c1718760781f273fae92504f386b7435', 2, {}, 1)}


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
