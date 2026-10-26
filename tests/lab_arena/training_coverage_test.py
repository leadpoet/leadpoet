"""One frozen coverage calculation for champion and strict E2E gates."""

import pytest

from lab_arena import training_coverage


@pytest.mark.parametrize("bank,required", [
    (1, 1), (2, 2), (3, 3), (9, 8), (10, 8), (11, 9), (20, 16),
])
def test_eighty_percent_uses_integer_ceiling(bank, required):
    assert training_coverage.required_count(bank, 80) == required


def test_legacy_missing_setting_requires_every_icp():
    assert training_coverage.percent_from_configuration({}) == 100
    assert training_coverage.required_count(10, 100) == 10


@pytest.mark.parametrize("value", [None, True, False, 0, 101, -1, 90.0, "90"])
def test_explicit_invalid_coverage_rejected(value):
    with pytest.raises(ValueError):
        training_coverage.percent_from_configuration({
            "training_minimum_icp_coverage_percent": value,
        })
