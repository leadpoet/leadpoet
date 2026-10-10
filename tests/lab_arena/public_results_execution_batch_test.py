"""One fresh execution read serves both disclosure and published results."""

from copy import deepcopy
from datetime import datetime, timezone

import httpx
import pytest

from lab_arena import icp_disclosure
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport, PsycopgTransport
from tests.lab_arena.scoring_validator_attribution_test import (
    MINER, ROUND_ID, SUBMISSION_ID, VALIDATOR_A, _execution, _score, _service,
)


BASELINE = "submission-baseline"


def _round():
    return {
        "round_id": ROUND_ID, "status": "published",
        "icp_set_date": "2026-09-19", "evaluation_date": "2026-09-20",
        "benchmark_ref": "arena/benchmark.json",
        "participants": [
            {"submission_id": BASELINE, "miner_hotkey": "5" + "B" * 47,
             "is_king": True},
            {"submission_id": SUBMISSION_ID, "miner_hotkey": MINER,
             "is_baseline": False},
        ],
        "configuration_doc": {
            "mode": "live", "network_name": "finney", "netuid": 71,
            "benchmark_disclosure_policy": icp_disclosure.CUTOFF_PUBLIC_POLICY,
            "schedule": {"submission_open": "2026-09-19T00:00:00Z",
                         "submission_cutoff": "2026-09-20T00:00:00Z"},
            "stage_1_icp_count": 10, "stage_2_icp_count": 10,
        },
        "publication_doc": {
            "participants": [
                {"submission_id": BASELINE, "miner_hotkey": "5" + "B" * 47,
                 "is_king": True},
                {"submission_id": SUBMISSION_ID, "miner_hotkey": MINER,
                 "is_baseline": False},
            ],
            "stage1_ranking": [], "final_ranking": [],
        },
    }


def _scenario(*, optimized, target, now, baseline_count=1):
    rows = []
    for position in range(20):
        for submission_id, prefix in ((BASELINE, "a"), (SUBMISSION_ID, "b")):
            run = _execution("execute-%s-%02d" % (prefix, position), position,
                             submission_id=submission_id)
            run["stage"] = 1 if position < 10 else 2
            rows.append(run)
            if position == 0:
                rows.append(_score("score-%s" % prefix, run, VALIDATOR_A))
    rows.sort(key=lambda run: run["run_id"])
    service = _service(rows, public_positions=None)
    row = _round()
    if baseline_count == 0:
        row["participants"][0]["is_king"] = False
    elif baseline_count == 2:
        row["participants"][1]["is_king"] = True
    service._round = lambda _round_id: deepcopy(row)
    service._clock = lambda: now
    service._public_icp_disclosure = ArenaService._public_icp_disclosure.__get__(service)
    calls = []
    original_list_runs = service._store.list_runs

    def list_runs(round_id, **filters):
        calls.append(("list", dict(filters)))
        return original_list_runs(round_id, **filters)

    service._store.list_runs = list_runs
    if optimized:
        def combined(round_id, baseline_id, submission_id):
            calls.append(("combined", (baseline_id, submission_id)))
            assert round_id == ROUND_ID
            return [run for run in rows if run["kind"] == "execute"
                    and run["submission_id"] in {baseline_id, submission_id}]

        service._store.list_public_result_execution_runs = combined
    if optimized == "new":
        def combined_all(round_id, baseline_id, submission_id):
            calls.append(("combined_all", (baseline_id, submission_id)))
            assert round_id == ROUND_ID
            return [run for run in rows
                    if (run["submission_id"] == baseline_id and run["kind"] == "execute")
                    or (run["submission_id"] == submission_id and run["kind"] in {"execute", "score"})]

        service._store.list_public_result_runs = combined_all
    return service.public_results(ROUND_ID, target), calls


@pytest.mark.parametrize("target", [BASELINE, SUBMISSION_ID])
@pytest.mark.parametrize("now", [
    datetime(2026, 9, 19, 23, 59, tzinfo=timezone.utc),
    datetime(2026, 9, 20, 0, 1, tzinfo=timezone.utc),
])
def test_combined_read_keeps_disclosure_results_and_order(target, now):
    prior, prior_calls = _scenario(optimized=False, target=target, now=now)
    current, current_calls = _scenario(optimized=True, target=target, now=now)
    assert current == prior
    assert len([call for call in prior_calls if call[0] == "list"]) == 3
    assert current_calls == [
        ("combined", (BASELINE, target)),
        ("list", {"kind": "score", "submission_id": target}),
    ]
    assert current["public_icp_status"] == (
        "ready" if now.day == 20 else "pending"
    )


def test_combined_store_read_filters_only_the_two_execute_submissions():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=[])
    )) as http:
        store = ArenaStore(PostgrestTransport(
            "https://example.test", service_key="sb_secret_test", http_client=http,
        ))
        assert store.list_public_result_execution_runs(
            ROUND_ID, BASELINE, SUBMISSION_ID) == []
        assert store.list_public_result_execution_runs(
            ROUND_ID, BASELINE, BASELINE) == []
    first, second = (request.url.params for request in requests)
    assert first["kind"] == "eq.execute"
    assert first["round_id"] == "eq." + ROUND_ID
    assert first["submission_id"] == "in.(%s,%s)" % (BASELINE, SUBMISSION_ID)
    assert first["order"] == "run_id.asc"
    assert second["submission_id"] == "in.(%s)" % BASELINE


@pytest.mark.parametrize("target", [BASELINE, SUBMISSION_ID])
def test_single_run_read_keeps_results_and_target_scores(target):
    now = datetime(2026, 9, 20, 0, 1, tzinfo=timezone.utc)
    prior, _ = _scenario(optimized=True, target=target, now=now)
    candidate, calls = _scenario(optimized="new", target=target, now=now)
    assert candidate == prior
    assert calls == [("combined_all", (BASELINE, target))]


def test_single_run_read_preserves_500_row_cursor_and_excludes_baseline_score():
    rows = [{"run_id": "run-%03d" % index} for index in range(501)]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=rows[:500] if len(requests) == 1 else rows[500:])

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        store = ArenaStore(PostgrestTransport(
            "https://example.test", service_key="sb_secret_test", http_client=http,
        ))
        assert store.list_public_result_runs(ROUND_ID, BASELINE, SUBMISSION_ID) == rows
    assert len(requests) == 2
    for request in requests:
        params = request.url.params
        assert params["round_id"] == "eq." + ROUND_ID
        assert params["order"] == "run_id.asc"
        assert params["limit"] == "500"
        assert params["or"] == (
            "(and(submission_id.eq.%s,kind.eq.execute),"
            "and(submission_id.eq.%s,kind.in.(execute,score)))"
        ) % (BASELINE, SUBMISSION_ID)
    assert "run_id" not in requests[0].url.params
    assert requests[1].url.params["run_id"] == "gt.run-499"


@pytest.mark.parametrize("transport_kind", ["postgrest", "psycopg"])
@pytest.mark.parametrize("overrides", [
    {"public_result_pair": ("bad,filter", SUBMISSION_ID)},
    {"public_result_pair": (BASELINE, SUBMISSION_ID), "filters": {"kind": "execute"}},
    {"public_result_pair": (BASELINE, SUBMISSION_ID), "submission_ids": [SUBMISSION_ID]},
])
def test_single_run_filter_rejects_unsafe_or_mixed_predicates_before_io(transport_kind, overrides):
    if transport_kind == "postgrest":
        client = httpx.Client(transport=httpx.MockTransport(
            lambda _request: pytest.fail("invalid filter reached HTTP")))
        transport = PostgrestTransport(
            "https://example.test", service_key="sb_secret_test", http_client=client,
        )
    else:
        transport = PsycopgTransport(
            lambda: pytest.fail("invalid filter opened a database connection"), role=None,
        )
    try:
        with pytest.raises(ArenaStoreError):
            transport.select("lab_arena_runs", order="run_id", limit=500, **overrides)
    finally:
        transport.close()


@pytest.mark.parametrize("baseline_count", [0, 2])
def test_single_run_read_does_not_override_missing_or_disputed_baseline(baseline_count):
    now = datetime(2026, 9, 20, 0, 1, tzinfo=timezone.utc)
    prior, _ = _scenario(optimized=True, target=SUBMISSION_ID, now=now,
                         baseline_count=baseline_count)
    candidate, calls = _scenario(optimized="new", target=SUBMISSION_ID, now=now,
                                 baseline_count=baseline_count)
    assert candidate == prior
    assert not any(name == "combined_all" for name, _ in calls)
