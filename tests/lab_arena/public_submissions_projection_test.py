"""Submission polling keeps fresh authorities without reading frozen catalogs."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contact_policy, contracts, icp_disclosure, integrity
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import (
    ArenaStore, ArenaStoreUnavailable, PUBLIC_CONFIGURATION_COLUMN,
    PostgrestTransport, PsycopgTransport,
)
from tests.lab_arena.competition_summary_projection_test import _project
from tests.lab_arena.public_dashboard_cost_policy_test import _bucket


CUTOFF = datetime(2026, 10, 13, tzinfo=timezone.utc)


def _row(status="stage2"):
    participants = [
        {"submission_id": sid, "miner_hotkey": "5" + letter * 47,
         "is_baseline": index == 0}
        for index, (sid, letter) in enumerate(
            (("baseline", "A"), ("champion", "B"), ("other-miner", "C"))
        )
    ]
    summary = {
        "returned_company_count": 10, "qualified_company_count": 10,
        "eligible_icp_count": 10, "competition_sourcing_microusd": 1_000,
        "execution_icp_cap_microusd": 4_000_000,
        "cost_per_company_cap_microusd": 800_000,
        "per_icp": [{
            "icp_position": position, "returned_company_count": 1,
            "qualified_company_count": 1, "competition_sourcing_microusd": 100,
            "eligibility_cap_microusd": 800_000,
            "eligible": True, "eligibility_reason": "eligible",
        } for position in range(10)],
        "execution": _bucket(successful=True), "judge": _bucket(successful=True),
    }
    return {
        "round_id": "arena-projection", "status": status, "status_generation": 7,
        "participants": participants, "benchmark_ref": "retained-bank",
        "icp_set_date": "2026-10-12", "evaluation_date": "2026-10-13",
        "published_at": "2026-10-13T12:00:00Z", "cancel_reason": None,
        "configuration_doc": {
            "mode": "live", "network_name": "finney", "netuid": 71,
            "schedule": {"submission_open": "2026-10-12T00:00:00Z",
                         "submission_cutoff": CUTOFF.isoformat()},
            "benchmark_disclosure_policy": icp_disclosure.CUTOFF_PUBLIC_POLICY,
            "integrity_policy": integrity.POLICY,
            "scorer_policy": {"scoring_adapter_version": integrity.SCORING_ADAPTER},
            "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
            "sourcing_cost_eligibility_policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
            "stage_1_icp_count": 5, "stage_2_icp_count": 5,
            "deepline_catalog": {"private-frozen-catalog": "x" * 1_800_000},
        },
        "publication_doc": {
            "participants": participants,
            "stage1_ranking": [{"submission_id": p["submission_id"], "stage1_score": 20 + i}
                               for i, p in enumerate(participants)],
            "final_ranking": [{"submission_id": p["submission_id"], "final_score": 50 + i,
                               "eligible": True, "eligibility_reason": "eligible",
                               "cost_summary": deepcopy(summary)}
                              for i, p in enumerate(participants)],
            "king_decision": {"outcome": "crowned", "winner_submission_id": "champion"},
        },
    }


class Transport:
    def __init__(self, row):
        self.row, self.round_reads, self.returned = row, [], []
        self.submissions = [{
            **participant, "round_id": row["round_id"], "status": "frozen",
            "source_ref": "source-" + participant["submission_id"],
            "consent": {"public_rerun": True},
        } for participant in row["participants"]]

    def select(self, table, **query):
        if table == "lab_arena_rounds":
            self.round_reads.append(query)
            if self.row is None:
                return []
            result = _project(self.row, query.get("columns", "*"))
            self.returned.append(result)
            return [result]
        if table == "lab_arena_submissions":
            return deepcopy(self.submissions)
        if table == "lab_arena_runs":
            return []
        raise AssertionError(table)


def _service(row, *, full=False, now=CUTOFF + timedelta(hours=13)):
    service = object.__new__(ArenaService)
    transport = Transport(row)
    service._store = ArenaStore(transport)
    service._config = SimpleNamespace(mode="live", pinned_round_id=None,
                                      network_name="finney", netuid=71)
    service._clock = lambda: now
    service._completed_scores_lock = threading.Lock()
    service._completed_scores_cache = {}
    # This route test holds refresh admission closed. Worker generation fences
    # and lazy full reads are covered in completed_model_scores_test.py.
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    service.completed_submission_scores(row)
    if row["round_id"] in service._completed_scores_cache:
        entry = service._completed_scores_cache[row["round_id"]]
        entry["completed"] = {"champion": row["publication_doc"]["final_ranking"][1]}
        entry["refreshed_at"] = time.monotonic()
    if full:
        service._public_submissions_round = service._round
    return service, transport


def _get(service):
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),
                                     base_url="http://audit.test") as client:
            return await client.get("/arena/v1/rounds/arena-projection/submissions")
    return asyncio.run(request())


def test_warm_exact_route_never_fetches_full_round_and_matches_cached_score():
    row = _row()
    full, _ = _service(row, full=True)
    compact, transport = _service(row)
    compact._round = lambda _: pytest.fail("warm submissions fetched the full round")
    expected = _get(full).json()
    for _ in range(2):
        response = _get(compact)
        assert response.status_code == 200 and response.json() == expected
    assert next(p for p in expected["submissions"] if p["submission_id"] == "champion")["final_score"] == 51
    assert len(transport.round_reads) == 2
    for query in transport.round_reads:
        assert query["filters"] == {"round_id": row["round_id"]}
        assert PUBLIC_CONFIGURATION_COLUMN in query["columns"].split(",")
        assert "publication_doc" in query["columns"].split(",")
    assert "private-frozen-catalog" not in json.dumps(transport.returned)
    assert len(json.dumps(transport.returned[0])) < len(json.dumps(row)) / 100


def test_published_route_preserves_all_miner_rankings_and_frozen_cost_basis():
    row = _row("published")
    compact, transport = _service(row)
    full, _ = _service(row, full=True)
    response = _get(compact)
    assert response.status_code == 200 and response.json() == _get(full).json()
    entries = {p["submission_id"]: p for p in response.json()["submissions"]}
    assert entries["other-miner"]["final_score"] == 52
    assert entries["other-miner"]["stage1_score"] == 22
    assert entries["other-miner"]["cost_summary"]["competition_sourcing_microusd"] == 1_000
    assert len(transport.returned[0]["publication_doc"]["final_ranking"]) == 3


@pytest.mark.parametrize("policy,status,offset,available", [
    (icp_disclosure.CUTOFF_PUBLIC_POLICY, "stage2", -1, False),
    (icp_disclosure.CUTOFF_PUBLIC_POLICY, "stage2", 0, True),
    (icp_disclosure.DELAYED_DISCLOSURE_POLICY, "stage2", 24 * 60, False),
    (icp_disclosure.DELAYED_DISCLOSURE_POLICY, "published", 24 * 60 - 1, False),
    (icp_disclosure.DELAYED_DISCLOSURE_POLICY, "published", 24 * 60, True),
    (None, "published", 13 * 60, True),
])
def test_source_disclosure_boundary_matches_full_route(policy, status, offset, available):
    row = _row(status)
    if policy == icp_disclosure.DELAYED_DISCLOSURE_POLICY:
        row["published_at"] = (CUTOFF + timedelta(days=1)).isoformat()
    if policy is None:
        row["configuration_doc"].pop("benchmark_disclosure_policy")
    else:
        row["configuration_doc"]["benchmark_disclosure_policy"] = policy
    now = CUTOFF + timedelta(minutes=offset)
    compact, _ = _service(row, now=now)
    full, _ = _service(row, full=True, now=now)
    response = _get(compact)
    assert response.status_code == 200 and response.json() == _get(full).json()
    assert all(p["code"]["available"] is available for p in response.json()["submissions"])


@pytest.mark.parametrize("missing", ["benchmark_ref", "consent"])
def test_incomplete_source_authority_stays_private(missing):
    row = _row()
    if missing == "benchmark_ref":
        row.pop("benchmark_ref")
    compact, ct = _service(row)
    full, ft = _service(row, full=True)
    if missing == "consent":
        for transport in (ct, ft):
            for submission in transport.submissions:
                submission["consent"] = {"public_rerun": False}
    response = _get(compact)
    assert response.json() == _get(full).json()
    assert all(p["code"]["available"] is False for p in response.json()["submissions"])


def test_fresh_status_and_generation_remove_old_cached_score():
    row = _row()
    service, transport = _service(row)
    assert _get(service).json()["submissions"][1]["final_score"] == 51
    row["status_generation"] += 1
    assert _get(service).json()["submissions"][1]["final_score"] is None
    assert service._completed_scores_cache[row["round_id"]]["generation"] == 8
    row["status"] = "open"
    for submission in transport.submissions:
        submission["status"] = "accepted"
    response = _get(service).json()
    assert all(p["status"] == "queued" and not p["code"]["available"] for p in response["submissions"])
    assert len(transport.round_reads) == 3


@pytest.mark.parametrize("patch,code", [
    ({"mode": "sandbox"}, "round_mode_mismatch"),
    ({"network_name": "other"}, "round_network_mismatch"),
    ({"netuid": 72}, "round_network_mismatch"),
    ({"integrity_policy": "invalid"}, "unsupported_integrity_policy"),
    ({"contact_policy": "invalid"}, "unsupported_integrity_policy"),
    ({"company_quality_policy": "invalid"}, "unsupported_integrity_policy"),
    ({"intent_details_policy": "invalid"}, "unsupported_integrity_policy"),
    ({"benchmark_disclosure_policy": "invalid"}, "benchmark_disclosure_policy_invalid"),
    ({"scorer_policy": {}}, "integrity_scorer_policy_mismatch"),
])
def test_mode_network_and_policy_rejections_match_full_route(patch, code):
    row = _row()
    row["configuration_doc"].update(patch)
    compact, _ = _service(row)
    full, _ = _service(row, full=True)
    response = _get(compact)
    assert response.status_code == _get(full).status_code
    assert response.json() == _get(full).json() == {"status": "rejected", "code": code}


def test_contact_adapter_remains_valid_and_wrong_pinned_round_is_refused():
    row = _row()
    row["configuration_doc"].update(contact_policy=contact_policy.POLICY,
                                     scorer_policy={"scoring_adapter_version": contact_policy.SCORING_ADAPTER})
    compact, transport = _service(row)
    full, _ = _service(row, full=True)
    assert _get(compact).json() == _get(full).json()
    transport.round_reads.clear()
    compact._config.pinned_round_id = "other"
    assert _get(compact).json()["code"] == "round_scope_mismatch"
    assert transport.round_reads == []


def test_scorer_policy_absent_and_explicit_null_keep_full_read_failures():
    for explicit_null in (False, True):
        row = _row()
        if explicit_null:
            row["configuration_doc"]["scorer_policy"] = None
        else:
            row["configuration_doc"].pop("scorer_policy")
        compact, _ = _service(row)
        full, _ = _service(row, full=True)
        for service in (full, compact):
            if explicit_null:
                with pytest.raises(AttributeError):
                    service._public_submissions_round(row["round_id"])
            else:
                with pytest.raises(ServiceError, match="integrity_scorer_policy_mismatch"):
                    service._public_submissions_round(row["round_id"])


def test_missing_round_and_read_failure_do_not_fall_back_to_full_read():
    service, transport = _service(_row())
    service._round = lambda _: pytest.fail("projected read fell back to full round")
    transport.row = None
    assert _get(service).status_code == 404
    def unavailable(*args, **kwargs):
        raise ArenaStoreUnavailable("test unavailable")
    transport.select = unavailable
    response = _get(service)
    assert response.status_code == 503 and response.json()["code"] == "arena_store_unavailable"


def test_store_projection_uses_existing_postgrest_and_psycopg_fields():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[]))) as client:
        store = ArenaStore(PostgrestTransport("https://example.test", service_key="sb_secret_test", http_client=client))
        assert store.get_public_submissions_round("arena-projection") is None
    fields = requests[0].url.params["select"].split(",")
    assert PUBLIC_CONFIGURATION_COLUMN in fields
    assert "cfg_scorer_policy:configuration_doc->scorer_policy::text" in fields
    assert "configuration_doc" not in fields
    assert "publication_doc" in fields
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _: None
    assert ArenaStore(direct).get_public_submissions_round("arena-projection") is None
    sql, values = queries[0]
    assert "public.lab_arena_public_configuration_v1(lab_arena_rounds) AS configuration_doc" in sql
    assert "(configuration_doc -> 'scorer_policy')::text AS cfg_scorer_policy" in sql
    assert "arena-projection" in values
