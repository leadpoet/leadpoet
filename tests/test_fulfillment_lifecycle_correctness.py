from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from fastapi import HTTPException

from gateway.fulfillment import api, lifecycle, rewards
from gateway.fulfillment.score_coverage import (
    FulfillmentScoreCoverage,
    summarize_score_coverage,
)


class _MemoryQuery:
    def __init__(self, client: "_MemorySupabase", table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.range_bounds: tuple[int, int] | None = None
        self.limit_count: int | None = None
        self.order_field: str | None = None
        self.order_desc = False
        self.update_payload: dict | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, field: str, value):
        self.filters.append(("eq", field, value))
        return self

    def in_(self, field: str, values):
        self.filters.append(("in", field, set(values)))
        return self

    def gte(self, field: str, value):
        self.filters.append(("gte", field, value))
        return self

    def or_(self, _expression: str):
        # Tests use scoring rows, which satisfy the production OR predicate.
        return self

    def range(self, start: int, end: int):
        self.range_bounds = (start, end)
        return self

    def order(self, field: str, desc: bool = False):
        self.order_field = field
        self.order_desc = desc
        return self

    def limit(self, count: int):
        self.limit_count = count
        return self

    def update(self, payload: dict):
        self.update_payload = deepcopy(payload)
        return self

    def _matches(self, row: dict) -> bool:
        for operation, field, expected in self.filters:
            actual = row.get(field)
            if operation == "eq" and actual != expected:
                return False
            if operation == "in" and actual not in expected:
                return False
            if operation == "gte" and actual < expected:
                return False
        return True

    def execute(self):
        source = self.client.tables.setdefault(self.table_name, [])
        matched = [row for row in source if self._matches(row)]

        if self.update_payload is not None:
            for row in matched:
                row.update(deepcopy(self.update_payload))
            return SimpleNamespace(data=deepcopy(matched), count=len(matched))

        if self.order_field:
            matched.sort(
                key=lambda row: row.get(self.order_field),
                reverse=self.order_desc,
            )
        if self.range_bounds:
            start, end = self.range_bounds
            matched = matched[start : end + 1]
        if self.limit_count is not None:
            matched = matched[: self.limit_count]
        return SimpleNamespace(data=deepcopy(matched), count=len(matched))


class _MemorySupabase:
    def __init__(
        self,
        tables: dict[str, list[dict]],
        *,
        rpc_results: dict[str, object] | None = None,
    ) -> None:
        self.tables = deepcopy(tables)
        self.rpc_results = rpc_results or {}
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, table_name: str) -> _MemoryQuery:
        return _MemoryQuery(self, table_name)

    def rpc(self, function_name: str, payload: dict):
        self.rpc_calls.append((function_name, deepcopy(payload)))
        result = self.rpc_results.get(function_name)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=deepcopy(result)))


def _lead(lead_id: str, business: str = "Example") -> dict:
    return {
        "lead_id": lead_id,
        "data": {"business": business},
    }


def _score(
    submission_id: str,
    lead_id: str,
    validator_hotkey: str = "validator-a",
) -> dict:
    return {
        "request_id": "request-1",
        "submission_id": submission_id,
        "lead_id": lead_id,
        "validator_hotkey": validator_hotkey,
    }


def test_score_coverage_requires_every_revealed_lead() -> None:
    submissions = [
        {
            "submission_id": "submission-1",
            "lead_data": [_lead("lead-1"), _lead("lead-2"), _lead("lead-3")],
        },
        {
            "submission_id": "submission-2",
            "lead_data": [_lead("lead-4"), _lead("lead-5")],
        },
    ]
    partial = summarize_score_coverage(
        submissions,
        [
            _score("submission-1", "lead-1"),
            _score("submission-1", "lead-2"),
            _score("submission-2", "lead-4"),
        ],
        required_validators=1,
    )

    assert partial.expected_leads == 5
    assert partial.covered_leads == 3
    assert partial.incomplete_submissions == 2
    assert partial.missing_score_slots == 2
    assert not partial.complete

    complete = summarize_score_coverage(
        submissions,
        [
            _score("submission-1", "lead-1"),
            _score("submission-1", "lead-2"),
            _score("submission-1", "lead-3"),
            _score("submission-2", "lead-4"),
            _score("submission-2", "lead-5"),
        ],
        required_validators=1,
    )
    assert complete.complete


def test_score_coverage_requires_distinct_validator_quorum_per_lead() -> None:
    submissions = [
        {
            "submission_id": "submission-1",
            "lead_data": [_lead("lead-1")],
        }
    ]
    duplicate_validator_rows = [
        _score("submission-1", "lead-1", "validator-a"),
        _score("submission-1", "lead-1", "validator-a"),
    ]

    one_of_two = summarize_score_coverage(
        submissions,
        duplicate_validator_rows,
        required_validators=2,
    )
    assert not one_of_two.complete
    assert one_of_two.missing_score_slots == 1

    two_of_two = summarize_score_coverage(
        submissions,
        [
            *duplicate_validator_rows,
            _score("submission-1", "lead-1", "validator-b"),
        ],
        required_validators=2,
    )
    assert two_of_two.complete


def test_score_coverage_records_exact_finalization_watermark() -> None:
    coverage = summarize_score_coverage(
        [
            {
                "submission_id": "submission-1",
                "lead_data": [_lead("lead-1")],
            }
        ],
        [
            {
                **_score("submission-1", "lead-1"),
                "scored_at": "2026-07-23T21:40:00+00:00",
            },
            {
                # A stray row still belongs in the database watermark even
                # though it cannot prove revealed-lead coverage.
                **_score("unknown-submission", "unknown-lead"),
                # Use Z notation to ensure timestamp order is chronological,
                # not a lexical comparison of differently formatted strings.
                "scored_at": "2026-07-23T21:41:00Z",
            },
        ],
        required_validators=1,
    )

    assert coverage.complete
    assert coverage.score_row_count == 2
    assert coverage.latest_scored_at == "2026-07-23T21:41:00Z"


@pytest.mark.parametrize(
    ("coverage", "timed_out", "expected"),
    [
        (
            FulfillmentScoreCoverage(
                expected_leads=3,
                covered_leads=2,
                incomplete_submissions=1,
                missing_score_slots=1,
                malformed_revealed_leads=0,
                validator_hotkeys=frozenset({"validator-a"}),
            ),
            True,
            "wait_score_coverage",
        ),
        (
            FulfillmentScoreCoverage(
                expected_leads=3,
                covered_leads=0,
                incomplete_submissions=1,
                missing_score_slots=3,
                malformed_revealed_leads=0,
                validator_hotkeys=frozenset(),
            ),
            False,
            "wait_validators",
        ),
        (
            FulfillmentScoreCoverage(
                expected_leads=3,
                covered_leads=0,
                incomplete_submissions=1,
                missing_score_slots=3,
                malformed_revealed_leads=0,
                validator_hotkeys=frozenset(),
            ),
            True,
            "no_validators_timeout",
        ),
        (
            FulfillmentScoreCoverage(
                expected_leads=3,
                covered_leads=3,
                incomplete_submissions=0,
                missing_score_slots=0,
                malformed_revealed_leads=0,
                validator_hotkeys=frozenset({"validator-a"}),
            ),
            False,
            "ready",
        ),
    ],
)
def test_consensus_readiness_never_forces_partial_coverage(
    coverage: FulfillmentScoreCoverage,
    timed_out: bool,
    expected: str,
) -> None:
    assert (
        lifecycle._consensus_readiness(
            coverage,
            timed_out=timed_out,
        )
        == expected
    )


def test_scoring_feed_reoffers_only_missing_leads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = _MemorySupabase(
        {
            "fulfillment_requests": [
                {
                    "request_id": "request-1",
                    "status": "scoring",
                    "reveal_window_end": "2099-01-01T00:00:00+00:00",
                    "icp_details": {"num_leads": 4},
                    "required_attributes": None,
                }
            ],
            "fulfillment_submissions": [
                {
                    "request_id": "request-1",
                    "submission_id": "submission-1",
                    "miner_hotkey": "miner-1",
                    "revealed": True,
                    "lead_data": [
                        _lead("lead-1"),
                        _lead("lead-2"),
                        _lead("lead-3"),
                    ],
                },
                {
                    "request_id": "request-1",
                    "submission_id": "submission-2",
                    "miner_hotkey": "miner-2",
                    "revealed": True,
                    "lead_data": [_lead("lead-4")],
                },
            ],
            "fulfillment_scores": [
                _score("submission-1", "lead-1"),
                _score("submission-1", "lead-2"),
                _score("submission-2", "lead-4"),
            ],
        }
    )
    monkeypatch.setattr(api, "_get_supabase", lambda: supabase)

    payload = api._collect_scoring_requests_sync("validator-a")

    assert len(payload["requests"]) == 1
    submissions = payload["requests"][0]["submissions"]
    assert submissions == [
        {
            "submission_id": "submission-1",
            "miner_hotkey": "miner-1",
            "leads": [{"business": "Example"}],
            "lead_ids": ["lead-3"],
        }
    ]

    other_validator_payload = api._collect_scoring_requests_sync("validator-b")
    other_submissions = other_validator_payload["requests"][0]["submissions"]
    assert [row["lead_ids"] for row in other_submissions] == [
        ["lead-1", "lead-2", "lead-3"],
        ["lead-4"],
    ]


def test_load_score_coverage_paginates_past_postgrest_default_limit() -> None:
    leads = [_lead(f"lead-{index}") for index in range(1001)]
    scores = [_score("submission-1", f"lead-{index}") for index in range(1001)]
    supabase = _MemorySupabase(
        {
            "fulfillment_submissions": [
                {
                    "request_id": "request-1",
                    "submission_id": "submission-1",
                    "revealed": True,
                    "lead_data": leads,
                }
            ],
            "fulfillment_scores": scores,
        }
    )

    coverage = lifecycle._load_score_coverage(supabase, "request-1")

    assert coverage.expected_leads == 1001
    assert coverage.covered_leads == 1001
    assert coverage.complete


def test_claim_finalization_passes_exact_score_watermark() -> None:
    supabase = _MemorySupabase(
        {},
        rpc_results={"fulfillment_claim_finalization": True},
    )
    coverage = FulfillmentScoreCoverage(
        expected_leads=2,
        covered_leads=2,
        incomplete_submissions=0,
        missing_score_slots=0,
        malformed_revealed_leads=0,
        validator_hotkeys=frozenset({"validator-a"}),
        score_row_count=2,
        latest_scored_at="2026-07-23T21:40:00+00:00",
    )

    assert lifecycle._claim_finalization(supabase, "request-1", coverage)
    assert supabase.rpc_calls == [
        (
            "fulfillment_claim_finalization",
            {
                "p_request_id": "request-1",
                "p_expected_score_count": 2,
                "p_expected_latest_scored_at": "2026-07-23T21:40:00+00:00",
            },
        )
    ]


def test_terminal_request_rejects_late_score_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = _MemorySupabase(
        {
            "fulfillment_requests": [
                {
                    "request_id": "request-1",
                    "status": "fulfilled",
                }
            ]
        }
    )
    monkeypatch.setattr(api, "_get_supabase", lambda: supabase)

    with pytest.raises(HTTPException) as exc_info:
        api._submit_scores_impl(
            "request-1",
            "validator-a",
            [_score("submission-1", "lead-1")],
        )

    assert exc_info.value.status_code == 409
    assert not supabase.rpc_calls


def test_score_submission_rejects_cross_request_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = _MemorySupabase({})
    monkeypatch.setattr(api, "_get_supabase", lambda: supabase)

    with pytest.raises(HTTPException) as exc_info:
        api._submit_scores_impl(
            "request-1",
            "validator-a",
            [
                {
                    **_score("submission-1", "lead-1"),
                    "request_id": "request-2",
                }
            ],
        )

    assert exc_info.value.status_code == 422
    assert not supabase.rpc_calls


def test_reward_write_failure_prevents_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = _MemorySupabase(
        {
            # No matching row: the update returns an empty representation.
            "fulfillment_score_consensus": [],
        }
    )
    monkeypatch.setattr(rewards, "_get_supabase", lambda: supabase)

    with pytest.raises(RuntimeError, match="FULFILLMENT_REWARD_WRITE_INCOMPLETE"):
        rewards.calculate_lead_rewards(
            "request-1",
            [
                {
                    "request_id": "request-1",
                    "submission_id": "submission-1",
                    "lead_id": "lead-1",
                    "tie_count": 1,
                }
            ],
            0.5,
            100,
            10,
        )


def test_active_successor_quota_and_icp_are_reconciled_together() -> None:
    supabase = _MemorySupabase(
        {
            "fulfillment_requests": [
                {
                    "request_id": "successor-1",
                    "status": "continued_open",
                    "num_leads": 28,
                    "icp_details": {
                        "num_leads": 28,
                        "excluded_companies": ["Original exclusion"],
                    },
                }
            ],
        }
    )

    changed = lifecycle._reconcile_active_successor(
        supabase,
        "successor-1",
        remaining_leads=18,
        held_companies=["New held company", "Original exclusion"],
    )

    assert changed
    row = supabase.tables["fulfillment_requests"][0]
    assert row["num_leads"] == 18
    assert row["icp_details"]["num_leads"] == 18
    assert row["icp_details"]["excluded_companies"] == [
        "Original exclusion",
        "New held company",
    ]


def test_scoring_successor_is_not_mutated_mid_cycle() -> None:
    original = {
        "request_id": "successor-1",
        "status": "scoring",
        "num_leads": 28,
        "icp_details": {"num_leads": 28},
    }
    supabase = _MemorySupabase(
        {
            "fulfillment_requests": [original],
        }
    )

    changed = lifecycle._reconcile_active_successor(
        supabase,
        "successor-1",
        remaining_leads=18,
        held_companies=["New held company"],
    )

    assert not changed
    assert supabase.tables["fulfillment_requests"][0] == original


def test_initial_successor_payload_updates_nested_num_leads() -> None:
    payload = lifecycle._successor_quota_payload(
        {
            "num_leads": 30,
            "excluded_companies": ["Existing"],
        },
        remaining_leads=18,
        held_companies=["Held"],
    )

    assert payload == {
        "num_leads": 18,
        "icp_details": {
            "num_leads": 18,
            "excluded_companies": ["Existing", "Held"],
        },
    }
