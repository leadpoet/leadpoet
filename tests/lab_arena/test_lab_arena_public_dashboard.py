from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, public_dashboard, source_disclosure
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


def test_round_summary_names_submission_bank_and_evaluation_days():
    row = _published_round()
    row.update(icp_set_date="2026-09-08", evaluation_date="2026-09-09")
    summary = public_dashboard.round_summary(row)
    assert summary["icp_set_date"] == "2026-09-08"
    assert summary["evaluation_date"] == "2026-09-09"
    assert summary["public_at"] == "2026-09-09T00:00:00Z"


def test_open_round_dates_are_planned_but_no_scores_are_disclosed():
    row = _published_round()
    row.update(status="open", participants=None, publication_doc=None)
    summary = public_dashboard.round_summary(row)
    assert summary["icp_set_date"] == "2026-09-08"
    assert summary["evaluation_date"] == "2026-09-09"
    assert summary["baseline"] is None
    assert summary["champion"] is None


@pytest.mark.parametrize("status", ["stage1_scored", "stage2", "scored", "cancelled"])
def test_intermediate_scores_are_not_read_or_released(status):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unpublished score rows must not be read")
    service = SimpleNamespace(_store=SimpleNamespace(list_runs=forbidden))
    assert public_dashboard._stage1_scores(service, {"status": status}) == {}


def test_missing_final_score_is_not_labelled_scored_but_real_zero_is():
    common = {"raw_status": "frozen", "round_status": "published", "is_champion": False}
    assert public_dashboard._submission_lifecycle(**common, final_score=None) == "scoring_failed"
    assert public_dashboard._submission_lifecycle(**common, final_score=0.0) == "scored"


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
    assert result["repo_url"] == "https://github.com/leadpoet/leadpoet-sales-agent/tree/lab"


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


def test_competition_snapshot_pages_past_server_authored_archives_with_scope():
    # These exact reasons are authored by SQL256/265 and SQL269 respectively.
    prior_archive = {
        "round_id": "arena-2026-09-15-archive",
        "status": "cancelled",
        "created_at": "2026-09-16T02:00:00Z",
        "cancel_reason": "authorized_sep15_baseline_evidence_archive",
        "configuration_doc": _configuration(mode="shadow"),
        "participants": [],
    }
    recovery_archive = {
        "round_id": "arena-2026-09-16-rerun265archive",
        "status": "cancelled",
        "created_at": "2026-09-16T03:00:00Z",
        "cancel_reason": "authorized_sep16_failed_native_rerun_archive",
        "configuration_doc": _configuration(),
        "participants": [],
    }
    running = {
        "round_id": "arena-2026-09-16",
        "status": "stage1",
        "created_at": "2026-09-15T00:00:00Z",
        "configuration_doc": _configuration(),
        "participants": [],
    }
    published = _published_round()

    class Store:
        def __init__(self):
            self.queries = []

        def list_rounds(self, **kwargs):
            self.queries.append(kwargs)
            assert kwargs["mode"] == "live"
            assert kwargs["network_name"] == "finney"
            assert kwargs["netuid"] == 71
            assert "source_ref" not in kwargs["columns"]
            pages = {
                0: [recovery_archive, prior_archive],
                2: [running, published],
            }
            return pages.get(kwargs.get("offset", 0), [])

    store = Store()
    service = SimpleNamespace(
        _config=SimpleNamespace(mode="live"),
        _store=store,
        _chain_scope=lambda: ("finney", 71),
    )

    result = public_dashboard.competition_snapshot(service, limit=2)

    assert [query["offset"] for query in store.queries] == [0, 2]
    assert [row["round_id"] for row in result["rounds"]] == [
        running["round_id"],
        published["round_id"],
    ]
    assert result["latest_round"]["round_id"] == running["round_id"]
    assert result["latest_completed_round"]["round_id"] == published["round_id"]


def test_competition_snapshot_keeps_a_real_cancelled_round():
    cancelled = {
        "round_id": "arena-2026-09-17",
        "status": "cancelled",
        "created_at": "2026-09-17T00:00:00Z",
        "cancel_reason": "capacity:stage1:20",
        "configuration_doc": _configuration(),
        "participants": [],
    }
    published = _published_round()

    class Store:
        @staticmethod
        def list_rounds(**_kwargs):
            return [cancelled, published]

    service = SimpleNamespace(
        _config=SimpleNamespace(mode="live"),
        _store=Store(),
        _chain_scope=lambda: ("finney", 71),
    )

    result = public_dashboard.competition_snapshot(service, limit=2)

    assert result["latest_round"]["round_id"] == cancelled["round_id"]
    assert [row["round_id"] for row in result["rounds"]] == [
        cancelled["round_id"],
        published["round_id"],
    ]


def test_competition_snapshot_honors_an_explicitly_pinned_archive():
    archive = {
        "round_id": "arena-2026-09-16-rerun265archive",
        "status": "cancelled",
        "created_at": "2026-09-16T03:00:00Z",
        "cancel_reason": "authorized_sep16_failed_native_rerun_archive",
        "configuration_doc": _configuration(),
        "participants": [],
    }

    class Store:
        @staticmethod
        def list_rounds(**_kwargs):
            raise AssertionError("pinned public reads must not enumerate rounds")

    service = SimpleNamespace(
        _config=SimpleNamespace(
            mode="live", pinned_round_id=archive["round_id"]
        ),
        _store=Store(),
        _chain_scope=lambda: ("finney", 71),
        _round=lambda round_id: archive if round_id == archive["round_id"] else None,
    )

    result = public_dashboard.competition_snapshot(service)

    assert result["latest_round"]["round_id"] == archive["round_id"]
    assert result["rounds"] == [public_dashboard.round_summary(archive)]


def test_competition_snapshot_stops_on_a_repeated_archive_page():
    archive = {
        "round_id": "arena-archive",
        "status": "cancelled",
        "cancel_reason": "authorized_test_evidence_archive",
        "configuration_doc": _configuration(),
        "participants": [],
    }

    class Store:
        def __init__(self):
            self.calls = 0

        def list_rounds(self, **kwargs):
            self.calls += 1
            return [] if kwargs.get("status") == "published" else [archive]

    store = Store()
    service = SimpleNamespace(
        _config=SimpleNamespace(mode="live"),
        _store=store,
        _chain_scope=lambda: ("finney", 71),
    )

    result = public_dashboard.competition_snapshot(service, limit=1)

    assert result["rounds"] == []
    assert result["latest_round"] is None
    assert store.calls == 3


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
                "available": False,
                "available_at": None,
                "url": None,
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


def _successful_cost_bucket():
    counters = {
        "settled_microusd": 100,
        "reserved_or_uncertain_microusd": 0,
        "conservative_microusd": 100,
        "inflight_calls": 0,
        "uncertain_calls": 0,
        "refused_calls": 0,
        "call_count": 1,
        "successful_microusd": 100,
        "successful_calls": 1,
        "success_unresolved_microusd": 0,
        "success_unresolved_calls": 0,
    }
    return {**counters, "providers": []}


def _per_icp_cost_summary(*, eligible_positions):
    return {
        "returned_company_count": 20,
        "qualified_company_count": len(eligible_positions),
        "eligible_icp_count": len(eligible_positions),
        "competition_sourcing_microusd": 2_000,
        "execution_icp_cap_microusd": 4_000_000,
        "cost_per_company_cap_microusd": 800_000,
        "per_icp": [
            {
                "icp_position": position,
                "returned_company_count": 1,
                "qualified_company_count": int(position in eligible_positions),
                "competition_sourcing_microusd": 100,
                "eligibility_cap_microusd": (
                    800_000 if position in eligible_positions else 0
                ),
                "eligible": position in eligible_positions,
                "eligibility_reason": (
                    "eligible"
                    if position in eligible_positions
                    else "cost_per_company_exceeded"
                ),
            }
            for position in range(contracts.BENCHMARK_ICP_COUNT)
        ],
        "execution": _successful_cost_bucket(),
        "judge": _successful_cost_bucket(),
    }


def _per_icp_published_round():
    row = _published_round(outcome="no_king")
    row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    row["publication_doc"]["stage1_ranking"] = [
        {"rank": 1, "submission_id": "sub-miner", "stage1_score": 2.1}
    ]
    for ranking in row["publication_doc"]["final_ranking"]:
        ranking["final_score"] = (
            1.62 if ranking["submission_id"] == "baseline-1" else 1.59
        )
        ranking.update(
            {
                "eligible": True,
                "eligibility_reason": "eligible",
                "cost_summary": _per_icp_cost_summary(eligible_positions={8}),
            }
        )
    return row


def _stage1_runs(*, miner_position=8):
    baseline_scores = [10.2, 0.0, 10.8, 0.0, 0.0, 0.0, 21.6, 0.0, 32.4, 0.0]
    return [
        {
            "submission_id": submission_id,
            "icp_position": position,
            "attempt": 1,
            "per_icp_score": score,
        }
        for submission_id, scores in (
            ("baseline-1", baseline_scores),
            (
                "sub-miner",
                [21.0 if position == miner_position else 0.0 for position in range(10)],
            ),
        )
        for position, score in enumerate(scores)
    ]


def _published_submission_records():
    return [
        {
            "submission_id": "baseline-1",
            "miner_hotkey": BASELINE_HOTKEY,
            "status": "frozen",
            "is_king": True,
            "accepted_at": "2026-09-08T00:00:01Z",
        },
        {
            "submission_id": "sub-miner",
            "miner_hotkey": MINER_HOTKEY,
            "status": "frozen",
            "is_king": False,
            "accepted_at": "2026-09-08T01:05:00Z",
        },
    ]


def test_per_icp_stage1_uses_frozen_cost_adjusted_baseline_and_miner_scores():
    row = _per_icp_published_round()
    service = SimpleNamespace(
        _store=SimpleNamespace(list_runs=lambda *_args, **_kwargs: _stage1_runs())
    )

    scores = public_dashboard._stage1_scores(service, row)

    assert scores == {"baseline-1": 3.24, "sub-miner": 2.1}
    assert sum(
        run["per_icp_score"]
        for run in _stage1_runs()
        if run["submission_id"] == "baseline-1"
    ) / 10 == 7.5


def test_per_icp_submissions_snapshot_changes_only_stage1_projection():
    row = _per_icp_published_round()

    class Store:
        @staticmethod
        def list_runs(*_args, **_kwargs):
            return _stage1_runs()

        @staticmethod
        def list_submissions(*_args, **_kwargs):
            return _published_submission_records()

    service = SimpleNamespace(
        _round=lambda _round_id: row,
        _store=Store(),
        now=_now,
    )

    result = public_dashboard.submissions_snapshot(service, row["round_id"])
    by_id = {item["submission_id"]: item for item in result["submissions"]}

    assert by_id["baseline-1"]["stage1_score"] == 3.24
    assert by_id["baseline-1"]["final_score"] == 1.62
    assert by_id["baseline-1"]["status"] == "scored"
    assert by_id["baseline-1"]["is_champion"] is False
    assert by_id["sub-miner"]["stage1_score"] == 2.1
    assert by_id["sub-miner"]["final_score"] == 1.59
    assert by_id["sub-miner"]["status"] == "scored"
    assert by_id["sub-miner"]["is_champion"] is False


def test_per_icp_stage1_does_not_restore_raw_cost_ineligible_miner_score():
    row = _per_icp_published_round()
    row["publication_doc"]["stage1_ranking"][0]["stage1_score"] = 0.0
    service = SimpleNamespace(
        _store=SimpleNamespace(
            list_runs=lambda *_args, **_kwargs: _stage1_runs(miner_position=0)
        )
    )

    scores = public_dashboard._stage1_scores(service, row)

    assert scores["sub-miner"] == 0.0


@pytest.mark.parametrize("published_score", [None, "invalid"])
def test_per_icp_stage1_does_not_fall_back_to_raw_miner_score(published_score):
    row = _per_icp_published_round()
    ranking = row["publication_doc"]["stage1_ranking"][0]
    if published_score is None:
        del ranking["stage1_score"]
    else:
        ranking["stage1_score"] = published_score
    service = SimpleNamespace(
        _store=SimpleNamespace(list_runs=lambda *_args, **_kwargs: _stage1_runs())
    )

    scores = public_dashboard._stage1_scores(service, row)

    assert "sub-miner" not in scores


@pytest.mark.parametrize("invalid_cost_basis", [None, "duplicate_position"])
def test_per_icp_stage1_hides_baseline_when_frozen_cost_basis_is_invalid(
    invalid_cost_basis,
):
    row = _per_icp_published_round()
    baseline = row["publication_doc"]["final_ranking"][0]
    if invalid_cost_basis is None:
        baseline["cost_summary"] = None
    else:
        baseline["cost_summary"]["per_icp"][1]["icp_position"] = 0
    service = SimpleNamespace(
        _store=SimpleNamespace(list_runs=lambda *_args, **_kwargs: _stage1_runs())
    )

    scores = public_dashboard._stage1_scores(service, row)

    assert "baseline-1" not in scores
    assert scores["sub-miner"] == 2.1


def test_successful_calls_policy_round_keeps_historical_raw_stage1_rule():
    row = _per_icp_published_round()
    row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        contracts.SUCCESSFUL_CALLS_COST_POLICY
    )
    service = SimpleNamespace(
        _store=SimpleNamespace(list_runs=lambda *_args, **_kwargs: _stage1_runs())
    )

    scores = public_dashboard._stage1_scores(service, row)

    assert scores == {"baseline-1": 7.5, "sub-miner": 2.1}


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
        lambda _objects, row, _now_value, **_kwargs: {"submission_id": row["submission_id"]},
    )
    assert ArenaService.public_submission_code(service, "sub-miner") == {
        "submission_id": "sub-miner"
    }
    assert checked == ["arena-2026-09-09"]

    def deny(_objects, _row, _now_value, **_kwargs):
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
        _public_code_review=ArenaService._public_code_review,
        now=_now,
    )

    assert ArenaService.submission_status(service, "sub-miner")["status"] == "accepted"
    assert checked == ["arena-testnet"]


@pytest.mark.parametrize("round_status", ["open", "stage1", "published"])
@pytest.mark.parametrize("rule,status", [
    ("code_review_incomplete", "review_failed"),
    ("code_review_rejected", "review_rejected"),
])
def test_previously_admitted_review_exclusion_remains_visible_without_scores(round_status, rule, status):
    row = _published_round()
    row["status"] = round_status
    submission = {
        "submission_id": "sub-3c67802803dcabc7844267cba4e84d76",
        "round_id": row["round_id"], "miner_hotkey": MINER_HOTKEY,
        "status": "rejected", "is_king": False,
        "accepted_at": "2026-09-08T17:50:08Z",
        "rejection_rule": rule, "replaced_by_submission_id": None,
        "source_ref": "private-source-do-not-publish", "consent": {"public_rerun": True},
        "code_review_status": "error" if status == "review_failed" else "rejected",
        "code_review_attempts": 3,
        "code_review_doc": {
            "error_code": "code_review_provider_unavailable", "cost_status": "uncertain",
            "review_cost_microusd": 4976874,
            "findings": [{"evidence": "private-finding-do-not-publish"}],
            "provider_response": "private-provider-do-not-publish",
        },
    }
    service = SimpleNamespace(_round=lambda _: row, now=_now,
        _store=SimpleNamespace(list_submissions=lambda *_a, **_k: [submission],
                             list_runs=lambda *_a, **_k: []))
    result = public_dashboard.submissions_snapshot(service, row["round_id"])
    assert len(result["submissions"]) == 1
    view = result["submissions"][0]
    assert view["status"] == status
    assert view["stage1_score"] is None and view["final_score"] is None
    assert view["is_champion"] is False and view["code"]["available"] is False
    assert "cost_summary" not in view and "eligible" not in view
    assert view["code_review"]["cost_status"] == "uncertain"
    assert view["code_review"]["attempts"] == 3
    serialized = json.dumps(result)
    for private in ["private-source-do-not-publish", "private-finding-do-not-publish", "private-provider-do-not-publish"]:
        assert private not in serialized


@pytest.mark.parametrize("change", [
    {"accepted_at": None}, {"accepted_at": "invalid"},
    {"rejection_rule": "source_replaced"}, {"rejection_rule": "source_upload_incomplete"},
    {"replaced_by_submission_id": "sub-new"}, {"is_king": True},
])
def test_review_visibility_does_not_disclose_unadmitted_or_superseded_sources(change):
    row = _published_round()
    submission = {"submission_id": "private-id", "miner_hotkey": MINER_HOTKEY,
        "status": "rejected", "rejection_rule": "code_review_incomplete",
        "accepted_at": "2026-09-08T17:50:08Z", **change}
    service = SimpleNamespace(_round=lambda _: row, now=_now,
        _store=SimpleNamespace(list_submissions=lambda *_a, **_k: [submission],
                             list_runs=lambda *_a, **_k: []))
    assert public_dashboard.submissions_snapshot(service, row["round_id"])["submissions"] == []


def test_public_review_diagnostics_project_only_typed_status_and_retry_flag():
    value = public_dashboard.code_review_summary({"code_review_status": "error",
        "code_review_attempts": 2, "code_review_doc": {"retryable": True,
            "provider_http_status": 503, "exception": "private-secret", "body": "private-source"}})
    assert value["retryable"] is True and value["provider_http_status"] == 503
    assert value["attempts"] == 2
    assert "private" not in json.dumps(value)
    invalid = public_dashboard.code_review_summary({"code_review_doc": {
        "retryable": "true", "provider_http_status": True}})
    assert "retryable" not in invalid and "provider_http_status" not in invalid


@pytest.mark.parametrize('attempts,round_status,offset,expected', [
    (3, 'open', -1, True), (6, 'open', -1, False),
    (3, 'open', 0, False), (3, 'open', 1, False),
    (3, 'published', -1, False),
])
def test_public_review_retry_flag_stops_at_cap_and_deadline(attempts, round_status, offset, expected):
    from datetime import timedelta
    deadline = _now()
    view = public_dashboard.code_review_summary({
        'status': 'accepted', 'code_review_status': 'error',
        'code_review_attempts': attempts,
        'code_review_doc': {'error_code': 'code_review_provider_unavailable',
                            'provider_http_status': 503, 'retryable': True},
    }, round_row={'status': round_status, 'configuration_doc': {
        'schedule': {'benchmark_deadline': deadline.isoformat()}}},
        now=deadline + timedelta(seconds=offset))
    assert view['retryable'] is expected


def test_public_review_does_not_promise_retry_after_deadline():
    view = public_dashboard.code_review_summary({
        'status': 'accepted', 'code_review_status': 'error',
        'code_review_attempts': 5, 'code_review_started_at': _now().isoformat(),
        'code_review_doc': {'retryable': True},
    }, round_row={'status': 'open', 'configuration_doc': {'schedule': {
        'benchmark_deadline': '2026-09-11T02:01:00Z'}}}, now=_now())
    assert view['retryable'] is False
