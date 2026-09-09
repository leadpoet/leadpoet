"""The frozen Day 0 bank becomes public on Day 1, never during intake."""

from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.icp_disclosure import baseline_disclosure, disclosure_metadata
from lab_arena.service import ArenaService, ServiceError


DAY_0 = datetime(2026, 9, 9, tzinfo=timezone.utc)
DAY_1 = datetime(2026, 9, 10, tzinfo=timezone.utc)


def round_row(*, status="committed", explicit=True):
    row = {
        "round_id": "arena-2026-09-10",
        "status": status,
        "evaluation_date": "2026-09-10",
        "configuration_doc": {
            "schedule": {
                "submission_open": "2026-09-09T00:00:00Z",
                "submission_cutoff": "2026-09-10T00:00:00Z",
            }
        },
        "participants": [
            {"submission_id": "base", "is_king": True},
            {"submission_id": "miner", "is_king": False},
        ],
    }
    if explicit:
        row["icp_set_date"] = "2026-09-09"
    return row


def baseline_runs(count=20):
    return [
        {
            "submission_id": "base",
            "kind": "execute",
            "icp_position": position,
            "attempt": 1,
            "per_icp_score": float(position),
            "terminal_cause": "accepted",
        }
        for position in range(count)
    ]


def test_open_round_has_planned_dates_but_never_discloses():
    row = round_row(status="open", explicit=False)
    row["evaluation_date"] = None
    metadata = disclosure_metadata(row)
    assert metadata == {
        "icp_set_date": "2026-09-09",
        "public_at": "2026-09-10T00:00:00Z",
        "disclosure_policy": "all_20_next_day",
        "evaluation_date": "2026-09-10",
        "planned": True,
    }
    assert baseline_disclosure(row, [], DAY_1) is None


def test_day_zero_hidden_and_exact_day_one_cutoff_reveals_all_twenty():
    row = round_row()
    assert baseline_disclosure(row, [], DAY_1 - timedelta(microseconds=1)) is None
    result = baseline_disclosure(row, [], DAY_1)
    assert result["public_positions"] == list(range(20))
    assert result["private_positions"] == []
    assert result["baseline_scores"] == {}
    assert result["public_at"] == "2026-09-10T00:00:00Z"


def test_current_day_bank_remains_hidden():
    row = round_row()
    row["icp_set_date"] = "2026-09-10"
    row["evaluation_date"] = "2026-09-11"
    row["configuration_doc"]["schedule"] = {
        "submission_open": "2026-09-10T00:00:00Z",
        "submission_cutoff": "2026-09-11T00:00:00Z",
    }
    assert baseline_disclosure(row, baseline_runs(), DAY_1) is None


def test_late_day_one_cutoff_does_not_reveal_at_midnight():
    row = round_row()
    row["configuration_doc"]["schedule"]["submission_cutoff"] = (
        "2026-09-10T06:00:00Z"
    )
    assert baseline_disclosure(
        row, [], datetime(2026, 9, 10, 5, 59, 59, tzinfo=timezone.utc)
    ) is None
    assert baseline_disclosure(
        row, [], datetime(2026, 9, 10, 6, tzinfo=timezone.utc)
    )["public_at"] == "2026-09-10T06:00:00Z"


def test_scores_are_nullable_until_publication_and_then_include_completed_baseline():
    row = round_row(status="stage1_scored")
    assert baseline_disclosure(row, baseline_runs(), DAY_1)["baseline_scores"] == {}
    row["status"] = "published"
    partial = baseline_disclosure(row, baseline_runs(3), DAY_1)
    assert partial["baseline_scores"] == {0: 0.0, 1: 1.0, 2: 2.0}
    complete = baseline_disclosure(row, baseline_runs(), DAY_1)
    assert complete["baseline_scores"] == {
        position: float(position) for position in range(20)
    }


def test_legacy_round_uses_evaluation_date_and_waits_one_more_utc_day():
    row = round_row(explicit=False)
    assert baseline_disclosure(row, [], datetime(2026, 9, 10, 23, 59, 59, tzinfo=timezone.utc)) is None
    ready = baseline_disclosure(row, [], datetime(2026, 9, 11, tzinfo=timezone.utc))
    assert ready["icp_set_date"] == "2026-09-10"
    assert ready["public_at"] == "2026-09-11T00:00:00Z"
    assert ready["public_positions"] == list(range(20))


def test_malformed_or_naive_time_fails_closed():
    row = round_row()
    assert baseline_disclosure(row, [], datetime(2026, 9, 10)) is None
    row["configuration_doc"]["schedule"]["submission_cutoff"] = "2026-09-09T23:00:00Z"
    assert disclosure_metadata(row) is None
    row = round_row()
    row["evaluation_date"] = "2026-09-11"
    assert disclosure_metadata(row) is None


def test_public_benchmark_returns_all_twenty_and_nullable_baseline_scores():
    row = round_row()
    disclosure = baseline_disclosure(row, [], DAY_1)
    service = ArenaService.__new__(ArenaService)
    service._round = lambda _: row
    service._public_icp_disclosure = lambda _: disclosure
    service.benchmark_icps = lambda _: [
        {"icp_id": f"icp-{position}", "prompt": f"PROMPT-{position:02d}!"}
        for position in range(20)
    ]
    result = service.public_benchmark(row["round_id"])
    assert len(result["icps"]) == 20
    assert [item["icp_position"] for item in result["icps"]] == list(range(20))
    assert all(item["baseline_score"] is None for item in result["icps"])
    assert result["icp_set_date"] == "2026-09-09"
    assert result["public_at"] == "2026-09-10T00:00:00Z"
    assert result["public_icp_count"] == 20
    assert result["private_icp_count"] == 0
    assert result["disclosure_policy"] == "all_20_next_day"
    service._public_icp_disclosure = lambda _: None
    with pytest.raises(ServiceError, match="benchmark_not_public"):
        service.public_benchmark(row["round_id"])
