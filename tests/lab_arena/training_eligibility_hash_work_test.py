"""Eligibility verifies each immutable export without hashing its whole graph twice."""
import copy
from collections.abc import Mapping
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_eligibility, training_thinking
from tests.lab_arena.training_eligibility_test import capture, captured_example


def test_eligibility_hashes_whole_export_once():
    example = captured_example()
    original = copy.deepcopy(example)
    full_hashes = []
    real_hash = contracts.document_hash

    def measured(value):
        if isinstance(value, Mapping) and set(value) == set(example) - {'example_hash'}:
            full_hashes.append(value.get('run_id'))
        return real_hash(value)

    with patch.object(contracts, 'document_hash', measured):
        result = capture(example)
    assert result['eligible'], result
    assert full_hashes == ['run']
    assert example == original


@pytest.mark.parametrize('field', ['output', 'configuration', 'source', 'provider_calls',
                                  'worker_observation_artifacts', 'judgment'])
def test_changed_export_is_not_admitted_or_read_as_verified(field):
    example = captured_example()
    example[field] = {'changed_without_resealing': True}
    assert not capture(example)['eligible']
    with pytest.raises(ValueError, match='example_hash_mismatch'):
        training_thinking.readable_trace(example)


def test_reassessment_never_reuses_a_stale_export_hash():
    example = captured_example()
    assert capture(example)['eligible']
    example['unrelated_metadata'] = 'changed after assessment'
    assert not capture(example)['eligible']


def test_valid_trace_is_required_before_accepting_any_thinking():
    example = captured_example()
    with patch.object(training_thinking, 'readable_trace', side_effect=ValueError('example_hash_mismatch')):
        assert not capture(example)['eligible']
