"""Published result reads preserve disclosure while omitting frozen catalogs."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import icp_disclosure
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import (
    ArenaStore, ArenaStoreError, PUBLIC_RESULTS_CONFIGURATION_FIELDS, PostgrestTransport,
    PsycopgTransport,
)
from tests.lab_arena.competition_summary_projection_test import _project
from tests.lab_arena.scoring_validator_attribution_test import (
    MINER, ROUND_ID, SUBMISSION_ID, _execution, _service,
)


def _round(status="published", **configuration):
    baseline = {"submission_id": "baseline", "miner_hotkey": "5" + "B" * 47,
                "is_king": True}
    miner = {"submission_id": SUBMISSION_ID, "miner_hotkey": MINER,
             "is_baseline": False}
    return {
        "round_id": ROUND_ID, "status": status,
        "evaluation_date": "2026-09-20", "icp_set_date": "2026-09-19",
        "benchmark_ref": "arena/benchmark.json", "participants": [baseline, miner],
        "configuration_doc": {
            "mode": "live", "network_name": "finney", "netuid": 71,
            "schedule": {"submission_open": "2026-09-19T00:00:00Z",
                         "submission_cutoff": "2026-09-20T00:00:00Z"},
            "benchmark_disclosure_policy": icp_disclosure.CUTOFF_PUBLIC_POLICY,
            "stage_1_icp_count": 5, "stage_2_icp_count": 5,
            "deepline_catalog": {"private-catalog": "x" * 100_000},
            **configuration,
        },
        "publication_doc": {"participants": [miner],
                            "stage1_ranking": [{"submission_id": SUBMISSION_ID,
                                                "stage1_score": 12.5}],
                            "final_ranking": [{"submission_id": SUBMISSION_ID,
                                               "final_score": 15.0,
                                               "cost_summary": {"unused": "x" * 20_000}}]},
    }


class RoundTransport:
    def __init__(self, row):
        self.row, self.calls, self.returned = row, [], []

    def select(self, table, **query):
        assert table == "lab_arena_published_results_v1"
        self.calls.append(query)
        if self.row is None or any(self.row.get(key) != value for key, value
                                   in query["filters"].items()):
            return []
        result = _project(self.row, query.get("columns", "*"))
        publication = result.get("publication_doc")
        if isinstance(publication, dict) and isinstance(publication.get("final_ranking"), list):
            publication["final_ranking"] = [
                {key: value for key, value in entry.items() if key != "cost_summary"}
                if isinstance(entry, dict) else entry
                for entry in publication["final_ranking"]
            ]
        self.returned.append(result)
        return [result]


def _configured_service(row):
    service = _service([_execution("execute-0", 0)], public_positions={0})
    service._round = lambda _round_id: deepcopy(row)
    service._config = SimpleNamespace(mode="live", pinned_round_id=None,
                                      network_name="finney", netuid=71)
    service._chain_scope = lambda: ("finney", 71)
    transport = RoundTransport(row)
    read = ArenaStore(transport)
    service._store.get_published_results_round = read.get_published_results_round
    service._store.get_round = lambda _round_id: deepcopy(row) if row else None
    return service, transport


def test_published_results_match_full_row_and_omit_catalog():
    row = _round()
    full = _service([_execution("execute-0", 0)], public_positions={0})
    full._round = lambda _round_id: deepcopy(row)
    compact, transport = _configured_service(row)
    compact._round = lambda _round_id: pytest.fail("published result fetched full round")
    expected = full.public_results(ROUND_ID, SUBMISSION_ID)
    assert compact.public_results(ROUND_ID, SUBMISSION_ID) == expected
    assert transport.calls[0]["filters"] == {"round_id": ROUND_ID, "status": "published"}
    assert "configuration_doc" not in transport.returned[0]
    assert "cost_summary" not in json.dumps(transport.returned[0])
    assert "private-catalog" not in json.dumps(transport.returned)
    assert len(json.dumps(transport.returned)) < len(json.dumps(row)) / 10


@pytest.mark.parametrize("day,hour,expected_status", [
    (19, 23, "pending"), (20, 1, "ready"),
])
def test_real_disclosure_boundary_matches_full_read(day, hour, expected_status):
    row = _round()
    baseline = _execution("baseline-0", 0, submission_id="baseline")
    for compact in (False, True):
        service, _ = _configured_service(row)
        service._store.rows[baseline["run_id"]] = baseline
        service._clock = lambda: datetime(2026, 9, day, hour, 0, tzinfo=timezone.utc)
        service._public_icp_disclosure = ArenaService._public_icp_disclosure.__get__(service)
        if not compact:
            del service._store.get_published_results_round
        result = service.public_results(ROUND_ID, SUBMISSION_ID)
        assert result["public_icp_status"] == expected_status
        assert result["public_icp_count"] == (10 if day == 20 else 0)
        if compact:
            projected = result
        else:
            full = result
    assert projected == full


@pytest.mark.parametrize("status", ["open", "stage2", "cancelled"])
def test_unpublished_results_use_full_round(status):
    row = _round(status)
    service, transport = _configured_service(row)
    with pytest.raises(ServiceError, match="results_not_public"):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert len(transport.calls) == 1
    assert transport.returned == []


def test_missing_published_round_keeps_missing_error():
    service, transport = _configured_service(None)
    service._round = lambda _round_id: (_ for _ in ()).throw(ServiceError("round_missing", 404))
    with pytest.raises(ServiceError, match="round_missing"):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert len(transport.calls) == 1


def test_projected_read_error_does_not_fall_back_to_full_round():
    service, _ = _configured_service(_round())
    service._store.get_published_results_round = lambda _round_id: (_ for _ in ()).throw(
        ArenaStoreError("read failed"))
    service._round = lambda _round_id: pytest.fail("full round must not hide read error")
    with pytest.raises(ArenaStoreError, match="read failed"):
        service.public_results(ROUND_ID, SUBMISSION_ID)


@pytest.mark.parametrize("key,value", [
    ("mode", "shadow"), ("network_name", "test"),
    ("integrity_policy", "invalid"),
    ("benchmark_disclosure_policy", "invalid"),
])
def test_projected_round_keeps_ownership_and_policy_checks(key, value):
    service, _ = _configured_service(_round(**{key: value}))
    with pytest.raises(ServiceError):
        service.public_results(ROUND_ID, SUBMISSION_ID)


@pytest.mark.parametrize("key,value", [
    ("integrity_policy", None), ("contact_policy", False),
    ("stage_1_icp_count", 0), ("mode", "null"),
    ("schedule", []), ("scorer_policy", {"nested": [1, None]}),
])
def test_projected_configuration_preserves_json_types(key, value):
    row = _round(**{key: value})
    transport = RoundTransport(row)
    projected = ArenaStore(transport).get_published_results_round(ROUND_ID)
    service, _ = _configured_service(row)
    service._require_round_mode = lambda row: row
    rebuilt = service._public_results_round(ROUND_ID)
    assert rebuilt["configuration_doc"][key] == value
    assert projected["cfg_" + key] == json.dumps(value)


def test_absent_configuration_key_stays_absent():
    row = _round()
    row["configuration_doc"].pop("benchmark_disclosure_policy")
    service, _ = _configured_service(row)
    rebuilt = service._public_results_round(ROUND_ID)
    assert "benchmark_disclosure_policy" not in rebuilt["configuration_doc"]


def test_result_projection_uses_fixed_json_aliases_in_both_transports():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=[])
    )) as http:
        ArenaStore(PostgrestTransport("https://example.test", service_key="sb_secret_test",
                                     http_client=http)).get_published_results_round(ROUND_ID)
    params = requests[0].url.params
    assert requests[0].url.path.endswith("/lab_arena_published_results_v1")
    assert params["status"] == "eq.published"
    assert "configuration_doc" not in params["select"].split(",")
    assert "stage2_scoring_plan_doc" not in params["select"]
    assert {column for column in params["select"].split(",") if ":configuration_doc" in column} == {
        "cfg_%s:configuration_doc->%s::text" % (key, key)
        for key in PUBLIC_RESULTS_CONFIGURATION_FIELDS
    }

    queries = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []

    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    ArenaStore(direct).get_published_results_round(ROUND_ID)
    sql, values = queries[0]
    assert "FROM public.lab_arena_published_results_v1" in sql
    assert "status = %s" in sql and values == [ROUND_ID, "published"]
    assert ":configuration_doc" not in sql
    for key in PUBLIC_RESULTS_CONFIGURATION_FIELDS:
        assert "(configuration_doc -> '%s')::text AS cfg_%s" % (key, key) in sql
