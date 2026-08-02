from __future__ import annotations

import pytest

from tests.restart_rehearsal.fixture_contract import (
    validate_rehearsal_finalized_authority_epochs,
)


def _rows(*, legacy_epoch: int) -> dict[str, list[dict[str, object]]]:
    return {
        "research_lab_finalized_allocation_epochs_v2": [
            {"netuid": 71, "epoch_id": 24218},
            {"netuid": 71, "epoch_id": 24219},
        ],
        "research_lab_legacy_finalized_allocation_migrations_v2": [
            {"netuid": 71, "epoch_id": legacy_epoch},
        ],
    }


def test_legacy_authority_must_predate_native_authorities() -> None:
    validate_rehearsal_finalized_authority_epochs(
        _rows(legacy_epoch=24217)
    )


@pytest.mark.parametrize("legacy_epoch", [24218, 24219, 24220])
def test_overlapping_or_newer_legacy_authority_is_rejected(
    legacy_epoch: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="legacy allocation authority does not predate",
    ):
        validate_rehearsal_finalized_authority_epochs(
            _rows(legacy_epoch=legacy_epoch)
        )
