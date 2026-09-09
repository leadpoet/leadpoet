from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import public_dashboard, source_disclosure
from lab_arena.service import ArenaService, ServiceError


BASELINE_HOTKEY = "5" + "A" * 47
MINER_HOTKEY = "5" + "B" * 47


def _now():
    return datetime(2026, 9, 11, 2, 0, tzinfo=timezone.utc)


def _configuration(*, mode="live", network="finney", netuid=71):
    return {
        "mode": mode,
        "network_name": network,
        "netuid": netuid,
        "schedule": {
            "submission_open": "2026-09-08T00:00:00Z",
            "submission_cutoff": "2026-09-09T00:00:00Z",
        },
        "runner_hotkeys": ["must-not-leak"],
    }


def _published_round(*, outcome="no_king", promoted=False):
    decision = {
        "outcome": outcome,
        "king_submission_id": "sub-miner" if outcome == "crowned" else None,
        "king_hotkey": MINER_HOTKEY if outcome == "crowned" else "",
        "winner_submission_id": "sub-miner" if outcome == "crowned" else None,
    }
    return {
        "round_id": "arena-2026-09-09",
        "status": "published",
        "created_at": "2026-09-08T00:00:00+00:00",
        "published_at": "2026-09-09T14:00:00Z",
        "configuration_doc": _configuration(),
        "participants": [{"source_ref": "private-source-must-not-leak"}],
        "publication_doc": {
            "participants": [
                {
                    "submission_id": "baseline-1",
                    "miner_hotkey": BASELINE_HOTKEY,
                    "is_baseline": True,
                },
                {
                    "submission_id": "sub-miner",
                    "miner_hotkey": MINER_HOTKEY,
                    "is_baseline": False,
                },
            ],
            "stage1_ranking": [
                {"submission_id": "sub-miner", "stage1_score": 4.5}
            ],
            "final_ranking": [
                {
                    "rank": 1,
                    "submission_id": "baseline-1",
                    "final_score": 8.25,
                    "is_baseline": True,
                },
                {
                    "rank": 2,
                    "submission_id": "sub-miner",
                    "final_score": 4.0,
                    "is_baseline": False,
                },
            ],
            "king_decision": decision,
        },
        "promotion_required": True,
        "baseline_promoted_at": "2026-09-09T15:00:00Z" if promoted else None,
        "promotion_doc": {"private": "must-not-leak"},
    }


def test_round_summary_does_not_mistake_ranked_baseline_for_champion():
    summary = public_dashboard.round_summary(_published_round())

    assert summary["baseline"] == {
        "submission_id": "baseline-1",
        "miner_hotkey": BASELINE_HOTKEY,
        "final_score": 8.25,
    }
    assert summary["champion"] is None
    assert summary["promotion_status"] == "not_required"
    serialized = json.dumps(summary, sort_keys=True)
    assert "private-source" not in serialized
    assert "promotion_doc" not in serialized
    assert "runner_hotkeys" not in serialized


def test_round_summary_projects_only_the_crowned_miner_and_promotion_state():
    pending = public_dashboard.round_summary(_published_round(outcome="crowned"))
    promoted = public_dashboard.round_summary(
        _published_round(outcome="crowned", promoted=True)
    )

    assert pending["champion"] == {
        "submission_id": "sub-miner",
        "miner_hotkey": MINER_HOTKEY,
        "final_score": 4.0,
    }
    assert pending["promotion_status"] == "pending"
    assert promoted["promotion_status"] == "promoted"

    no_promotion = _published_round(outcome="crowned")
    no_promotion["promotion_required"] = False
    assert public_dashboard.round_summary(no_promotion)["promotion_status"] == "not_required"


def test_competition_snapshot_is_scoped_and_separates_open_running_and_completed():
    published = _published_round()
    cancelled = {
        "round_id": "arena-2026-09-10-old",
        "status": "cancelled",
        "created_at": "2026-09-10T00:00:00Z",
        "cancel_reason": "scoring_incomplete",
        "configuration_doc": _configuration(),
        "participants": [],
    }
    opened = {
        "round_id": "arena-2026-09-11",
        "status": "open",
        "created_at": "2026-09-10T12:00:00Z",
        "configuration_doc": _configuration(),
        "participants": None,
    }
    running = {
        "round_id": "arena-2026-09-10-running",
        "status": "stage2_scoring",
        "created_at": "2026-09-10T06:00:00Z",
        "configuration_doc": _configuration(),
        "participants": [],
    }

    class Store:
        def list_rounds(self, **kwargs):
            self.query = kwargs
            return [opened, running, cancelled, published]

    store = Store()
    service = SimpleNamespace(
        _config=SimpleNamespace(mode="live"),
        _store=store,
        _chain_scope=lambda: ("finney", 71),
    )

    result = public_dashboard.competition_snapshot(service)

    assert store.query["mode"] == "live"
    assert store.query["network_name"] == "finney"
    assert store.query["netuid"] == 71
    assert "source_ref" not in store.query["columns"]
    assert result["open_round"]["round_id"] == "arena-2026-09-11"
    assert result["latest_round"]["round_id"] == "arena-2026-09-10-running"
    assert result["latest_round"]["status"] == "stage2_scoring"
    assert result["latest_completed_round"]["round_id"] == published["round_id"]
    assert result["repo_url"] == "https://github.com/leadpoet/pydantic-harness/tree/lab"


def test_competition_snapshot_fetches_the_latest_published_outside_recent_window():
    opened = {
        "round_id": "arena-2026-09-11",
        "status": "open",
        "created_at": "2026-09-11T00:00:00Z",
        "configuration_doc": _configuration(),
    }
    cancelled = {
        "round_id": "arena-2026-09-10",
        "status": "cancelled",
        "created_at": "2026-09-10T00:00:00Z",
        "configuration_doc": _configuration(),
        "participants": [],
    }
    published = _published_round()

    class Store:
        def __init__(self):
            self.queries = []

        def list_rounds(self, **kwargs):
            self.queries.append(kwargs)
            return [published] if kwargs.get("status") == "published" else [opened, cancelled]

    store = Store()
    service = SimpleNamespace(
        _config=SimpleNamespace(mode="live"),
        _store=store,
        _chain_scope=lambda: ("finney", 71),
    )

    result = public_dashboard.competition_snapshot(service)

    assert result["latest_round"]["round_id"] == cancelled["round_id"]
    assert result["latest_completed_round"]["round_id"] == published["round_id"]
    assert store.queries[1]["status"] == "published"
    assert store.queries[1]["network_name"] == "finney"
    assert store.queries[1]["netuid"] == 71


def test_open_submissions_disclose_only_accepted_intake_as_queued():
    round_row = {
        "round_id": "arena-2026-09-11",
        "status": "open",
        "configuration_doc": _configuration(),
        "participants": None,
    }
    records = [
        {
            "submission_id": "sub-queued",
            "round_id": round_row["round_id"],
            "miner_hotkey": MINER_HOTKEY,
            "status": "accepted",
            "is_king": False,
            "created_at": "2026-09-10T01:00:00Z",
            "accepted_at": "2026-09-10T01:05:00Z",
            "consent": {"public_rerun": True},
            "source_ref": "private-queued-source",
        },
        {
            "submission_id": "sub-uploading",
            "miner_hotkey": "5" + "C" * 47,
            "status": "uploading",
            "source_ref": "private-upload",
        },
        {
            "submission_id": "sub-rejected",
            "miner_hotkey": "5" + "D" * 47,
            "status": "rejected",
            "rejection_rule": "private-rule",
        },
    ]

    class Store:
        @staticmethod
        def list_submissions(round_id, **kwargs):
            assert round_id == round_row["round_id"]
            assert "encrypted_credentials" not in kwargs["columns"]
            return records

    service = SimpleNamespace(
        _round=lambda _round_id: round_row,
        _store=Store(),
        now=_now,
    )

    result = public_dashboard.submissions_snapshot(service, round_row["round_id"])

    assert result["submissions"] == [
        {
            "submission_id": "sub-queued",
            "miner_hotkey": MINER_HOTKEY,
            "is_baseline": False,
            "status": "queued",
            "submitted_at": "2026-09-10T01:05:00Z",
            "stage1_score": None,
            "final_score": None,
            "is_champion": False,
            "code": {
                "available": True,
                "available_at": "2026-09-11T01:05:00Z",
                "url": "/arena/v1/submissions/sub-queued/code",
            },
        }
    ]
    serialized = json.dumps(result, sort_keys=True)
    for private in (
        "private-queued-source",
        "private-upload",
        "private-rule",
        "sub-uploading",
        "sub-rejected",
    ):
        assert private not in serialized


def test_published_submissions_expose_judged_stage1_and_final_scores():
    round_row = _published_round(outcome="crowned")
    records = [
        {
            "submission_id": "baseline-1",
            "miner_hotkey": BASELINE_HOTKEY,
            "status": "frozen",
            "is_king": True,
            "created_at": "2026-09-08T00:00:00Z",
            "accepted_at": "2026-09-08T00:00:01Z",
        },
        {
            "submission_id": "sub-miner",
            "miner_hotkey": MINER_HOTKEY,
            "status": "frozen",
            "is_king": False,
            "created_at": "2026-09-08T01:00:00Z",
            "accepted_at": "2026-09-08T01:05:00Z",
        },
    ]
    runs = [
        {
            "submission_id": submission_id,
            "icp_position": position,
            "attempt": 1,
            "per_icp_score": score,
        }
        for submission_id, score in (("baseline-1", 8.0), ("sub-miner", 4.5))
        for position in range(10)
    ]

    class Store:
        @staticmethod
        def list_submissions(_round_id, **_kwargs):
            return records

        @staticmethod
        def list_runs(_round_id, **filters):
            assert filters == {"stage": 1, "kind": "execute"}
            return runs

    service = SimpleNamespace(
        _round=lambda _round_id: round_row,
        _store=Store(),
        now=_now,
    )

    result = public_dashboard.submissions_snapshot(service, round_row["round_id"])
    by_id = {row["submission_id"]: row for row in result["submissions"]}

    assert by_id["baseline-1"]["stage1_score"] == 8.0
    assert by_id["baseline-1"]["final_score"] == 8.25
    assert by_id["baseline-1"]["is_champion"] is False
    assert by_id["sub-miner"]["stage1_score"] == 4.5
    assert by_id["sub-miner"]["final_score"] == 4.0
    assert by_id["sub-miner"]["status"] == "champion"
    assert by_id["sub-miner"]["is_champion"] is True


def test_cancelled_round_never_fabricates_aggregate_scores():
    row = _published_round(outcome="crowned")
    row["status"] = "cancelled"
    row["cancel_reason"] = "scoring_incomplete"
    row["participants"] = [
        {
            "submission_id": "sub-miner",
            "miner_hotkey": MINER_HOTKEY,
            "is_king": False,
        }
    ]
    records = [
        {
            "submission_id": "sub-miner",
            "miner_hotkey": MINER_HOTKEY,
            "status": "frozen",
            "is_king": False,
            "created_at": "2026-09-08T01:00:00Z",
            "accepted_at": "2026-09-08T01:05:00Z",
        }
    ]
    service = SimpleNamespace(
        _round=lambda _round_id: row,
        _store=SimpleNamespace(list_submissions=lambda *_args, **_kwargs: records),
        now=_now,
    )

    result = public_dashboard.submissions_snapshot(service, row["round_id"])

    assert result["submissions"][0]["status"] == "cancelled"
    assert result["submissions"][0]["stage1_score"] is None
    assert result["submissions"][0]["final_score"] is None
    assert result["submissions"][0]["is_champion"] is False

    summary = public_dashboard.round_summary(row)
    assert summary["baseline"] is None
    assert summary["champion"] is None


def test_public_code_service_hook_checks_round_scope_and_translates_denial(monkeypatch):
    submission = {
        "submission_id": "sub-miner",
        "round_id": "arena-2026-09-09",
    }
    checked = []
    service = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _submission_id: submission),
        _round=lambda round_id: checked.append(round_id),
        _objects=object(),
        now=_now,
    )

    monkeypatch.setattr(
        source_disclosure,
        "public_source_code",
        lambda _objects, row, _now_value: {"submission_id": row["submission_id"]},
    )
    assert ArenaService.public_submission_code(service, "sub-miner") == {
        "submission_id": "sub-miner"
    }
    assert checked == ["arena-2026-09-09"]

    def deny(_objects, _row, _now_value):
        raise source_disclosure.SourceDisclosureError("source_not_public", 403)

    monkeypatch.setattr(source_disclosure, "public_source_code", deny)
    with pytest.raises(ServiceError) as error:
        ArenaService.public_submission_code(service, "sub-miner")
    assert error.value.code == "source_not_public"
    assert error.value.status == 403


def test_submission_status_uses_full_round_scope_check():
    checked = []
    service = SimpleNamespace(
        _store=SimpleNamespace(
            get_submission=lambda _submission_id: {
                "submission_id": "sub-miner",
                "round_id": "arena-testnet",
                "status": "accepted",
                "rejection_rule": None,
            }
        ),
        _round=lambda round_id: checked.append(round_id),
    )

    assert ArenaService.submission_status(service, "sub-miner")["status"] == "accepted"
    assert checked == ["arena-testnet"]
