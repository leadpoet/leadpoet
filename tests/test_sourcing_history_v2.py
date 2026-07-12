from __future__ import annotations

import copy

import pytest

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.sourcing_history_v2 import (
    SourcingHistoryV2Error,
    build_sourcing_decision_v2,
    build_sourcing_epoch_v2,
    effective_rep_score_v2,
    rolling_sourcing_history_v2,
    validate_sourcing_decision_v2,
    validate_sourcing_epoch_v2,
)


@pytest.mark.parametrize(
    "rep_score,adjustment,decision,expected",
    [
        (40, 1.0, "approve", 40.0),
        (40, 1.5, "approve", 60.0),
        (40, 5.0, "approve", 200.0),
        (40, 20, "approve", 60),
        (10, -15, "approve", 0),
        (40, 0.0, "approve", 40),
        (40, 20, "deny", 0),
    ],
)
def test_effective_score_matches_existing_multiplier_and_adjustment_rules(
    rep_score, adjustment, decision, expected
):
    value = effective_rep_score_v2(
        rep_score=rep_score,
        is_icp_multiplier=adjustment,
        decision=decision,
    )
    assert value == expected
    assert type(value) is type(expected)


def _decision(sequence, hotkey, decision, score, adjustment=0.0, epoch=10):
    return build_sourcing_decision_v2(
        epoch_id=epoch,
        sequence=sequence,
        lead_id_hash=sha256_json({"lead": sequence, "epoch": epoch}),
        miner_hotkey=hotkey,
        decision=decision,
        rep_score=score,
        is_icp_multiplier=adjustment,
    )


def test_epoch_and_rolling_aggregation_match_mutable_history_semantics():
    epoch_10 = build_sourcing_epoch_v2(
        epoch_id=10,
        decisions=[
            _decision(2, "miner-b", "approve", 10, -15),
            _decision(0, "miner-a", "approve", 40, 20),
            _decision(1, "miner-a", "approve", 20, 1.5),
            _decision(3, "miner-c", "deny", 48, 5.0),
        ],
    )
    assert epoch_10["miner_scores"] == [
        {"hotkey": "miner-a", "score": 90.0},
        {"hotkey": "miner-b", "score": 0},
    ]
    assert epoch_10["approved_lead_count"] == 3
    assert epoch_10["decision_count"] == 4
    validate_sourcing_epoch_v2(epoch_10)

    epoch_11 = build_sourcing_epoch_v2(
        epoch_id=11,
        decisions=[_decision(0, "miner-b", "approve", 12, 0, epoch=11)],
    )
    scores, count = rolling_sourcing_history_v2(
        current_epoch=12,
        epochs=[epoch_11, epoch_10],
        window=30,
    )
    assert scores == {"miner-a": 90.0, "miner-b": 12}
    assert count == 4


def test_epoch_preserves_first_decision_order_like_legacy_dict():
    epoch = build_sourcing_epoch_v2(
        epoch_id=10,
        decisions=[
            _decision(0, "miner-z", "approve", 1),
            _decision(1, "miner-a", "approve", 2),
            _decision(2, "miner-z", "approve", 3),
        ],
    )
    assert epoch["miner_scores"] == [
        {"hotkey": "miner-z", "score": 4},
        {"hotkey": "miner-a", "score": 2},
    ]


def test_sourcing_records_fail_on_tampering_or_duplicate_sequence():
    decision = _decision(0, "miner-a", "approve", 40, 20)
    validate_sourcing_decision_v2(decision)
    tampered = copy.deepcopy(decision)
    tampered["effective_rep_score"] = 999
    with pytest.raises(SourcingHistoryV2Error, match="not canonical"):
        validate_sourcing_decision_v2(tampered)
    with pytest.raises(SourcingHistoryV2Error, match="duplicated"):
        build_sourcing_epoch_v2(epoch_id=10, decisions=[decision, decision])


def test_rolling_window_excludes_current_and_old_epochs_exactly():
    epochs = [
        build_sourcing_epoch_v2(
            epoch_id=epoch,
            decisions=[_decision(0, "miner", "approve", epoch, 0, epoch=epoch)],
        )
        for epoch in (69, 70, 99, 100)
    ]
    scores, count = rolling_sourcing_history_v2(
        current_epoch=100, epochs=epochs, window=30
    )
    assert scores == {"miner": 169}
    assert count == 2
