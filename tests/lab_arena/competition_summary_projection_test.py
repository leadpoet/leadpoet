"""Competition reads omit frozen catalogs without changing public summaries."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from lab_arena import contracts, icp_disclosure, public_dashboard as dashboard
from lab_arena.service import ArenaService
from lab_arena.store import (
    ArenaStore, COMPETITION_CONFIGURATION_FIELDS, PostgrestTransport, PsycopgTransport,
)


def _project(row, columns):
    if columns == "*":
        return deepcopy(row)
    projected = {}
    for column in columns.split(","):
        if ":configuration_doc->" in column:
            alias, path = column.split(":", 1)
            key = path.removeprefix("configuration_doc->").removesuffix("::text")
            configuration = row.get("configuration_doc") or {}
            projected[alias] = json.dumps(configuration[key]) if key in configuration else None
        elif column == "publication_doc:lab_arena_competition_publication_v1":
            # Unit fixtures are canonical. PostgreSQL tests exercise the actual
            # function, including conservative fallback for legacy documents.
            publication = deepcopy(row.get("publication_doc"))
            if row.get("status") == "published" and isinstance(publication, dict):
                selected = {
                    item["submission_id"] for item in publication["participants"]
                    if item.get("is_baseline", item.get("is_king", False))
                }
                selected.add(dashboard._champion_submission_id(row))
                publication = {
                    "participants": publication["participants"],
                    "king_decision": publication["king_decision"],
                    "final_ranking": [item for item in publication["final_ranking"]
                                      if item["submission_id"] in selected],
                }
            projected["publication_doc"] = publication
        elif column in row:
            projected[column] = deepcopy(row[column])
    return projected


class RoundTransport:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.returned = []

    def select(self, table, **query):
        assert table == "lab_arena_rounds"
        query.setdefault("columns", "*")
        self.calls.append(query)
        rows = [row for row in self.rows if all(
            (row["configuration_doc"].get("mode") if key == "configuration_doc->>mode"
             else row.get(key)) == value for key, value in (query.get("filters") or {}).items()
        )]
        if query.get("order"):
            rows.sort(key=lambda row: row[query["order"]], reverse=query.get("descending", False))
        start = query.get("offset") or 0
        result = [_project(row, query["columns"]) for row in rows[start:start + query["limit"]]]
        self.returned.extend(result)
        return result


def _row(day="2026-10-09", **configuration):
    participants = [
        {"submission_id": "model-%d" % index, "miner_hotkey": "5" + str(index),
         "is_baseline": index == 0, "is_king": index == 0}
        for index in range(145)
    ]
    return {
        "round_id": "arena-" + day, "status": "published", "created_at": day + "T00:00:00Z",
        "evaluation_date": day, "icp_set_date": None, "published_at": day + "T15:00:00Z",
        "arena_network_name": "finney", "arena_netuid": 71,
        "promotion_required": True, "baseline_promoted_at": None, "cancel_reason": None,
        "configuration_doc": {
            "mode": "live", "network_name": "finney", "netuid": 71,
            "schedule": {"submission_open": "2026-10-08T00:00:00Z",
                         "submission_cutoff": "2026-10-09T00:00:00Z"},
            "deepline_catalog": {"private_frozen_catalog": "x" * 100_000},
            "runner_hotkeys": ["private-runner"], **configuration,
        },
        "participants": participants,
        "publication_doc": {
            "participants": participants,
            "final_ranking": [
                {"submission_id": participant["submission_id"], "final_score": index % 100,
                 "eligible": True, "eligibility_reason": "historical_round", "cost_summary": None}
                for index, participant in enumerate(participants)
            ],
            "king_decision": {"outcome": "crowned", "winner_submission_id": "model-1"},
        },
    }


def _service(rows, *, mode="live", pinned=None):
    service = object.__new__(ArenaService)
    transport = RoundTransport(rows)
    service._store = ArenaStore(transport)
    service._config = SimpleNamespace(mode=mode, pinned_round_id=pinned)
    service._chain_scope = lambda: ("finney", 71)
    service._round = lambda round_id: next(row for row in rows if row["round_id"] == round_id)
    return service, transport


@pytest.mark.parametrize("configuration", [
    {},
    {"stage_1_icp_count": 5, "stage_2_icp_count": 5, "promotion_margin": 1.5,
     "benchmark_disclosure_policy": icp_disclosure.CUTOFF_PUBLIC_POLICY,
     "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
     "sourcing_cost_eligibility_policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY},
    {"sourcing_cost_eligibility_policy": contracts.SUCCESSFUL_CALLS_COST_POLICY},
])
def test_145_model_competition_response_matches_full_configuration(monkeypatch, configuration):
    rows = [_row(**configuration), _row("2026-10-08", **configuration)]
    before = deepcopy(rows)
    candidate_columns = dashboard._COMPETITION_ROUND_COLUMNS
    monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", dashboard._ROUND_COLUMNS)
    original, _ = _service(rows)
    expected = original.public_competition()
    monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", candidate_columns)
    candidate, transport = _service(rows)
    assert candidate.public_competition() == expected
    assert rows == before
    assert expected["rounds"][0]["participant_count"] == 145
    assert expected["rounds"][0]["baseline"]["final_score"] == 0
    assert len(transport.returned[0]["publication_doc"]["final_ranking"]) == 2
    assert expected["rounds"][1]["promotion_status"] == "superseded"
    assert "configuration_doc" not in transport.returned[0]
    assert "private_frozen_catalog" not in json.dumps(transport.returned)
    assert len(json.dumps(transport.returned)) < len(json.dumps(rows)) / 2
    assert transport.calls[-1]["columns"] == "round_id,evaluation_date"


@pytest.mark.parametrize("key", COMPETITION_CONFIGURATION_FIELDS)
@pytest.mark.parametrize("value", [None, False, 0, 1.5, "null", [], {"nested": [1, None]}])
def test_configuration_projection_preserves_json_types_and_missing_keys(key, value):
    row = {"configuration_doc": {key: value}}
    projected = _project(row, dashboard._COMPETITION_ROUND_COLUMNS)
    assert dashboard._competition_round(projected)["configuration_doc"] == {key: value}
    assert dashboard._competition_round(_project({"configuration_doc": {}}, dashboard._COMPETITION_ROUND_COLUMNS))["configuration_doc"] == {}


@pytest.mark.parametrize("configuration", [
    {"promotion_margin": None}, {"stage_1_icp_count": None, "stage_2_icp_count": None},
    {"stage_1_icp_count": 5}, {"benchmark_disclosure_policy": None},
])
def test_explicit_invalid_configuration_keeps_original_failure(configuration):
    row = _row(**configuration)
    projected = dashboard._competition_round(_project(row, dashboard._COMPETITION_ROUND_COLUMNS))
    with pytest.raises(ValueError) as original:
        dashboard.round_summary(row)
    with pytest.raises(type(original.value), match=str(original.value)):
        dashboard.round_summary(projected)


def test_competition_projection_keeps_archive_pagination_scope_fallback_and_full_reads():
    archive = _row("2026-10-11")
    archive.update(status="cancelled", cancel_reason="authorized_oct09_evidence_archive441")
    opened = _row("2026-10-10")
    opened.update(status="open", publication_doc=None)
    published = _row()
    shadow = _row("2026-10-12", mode="shadow")
    other_chain = _row("2026-10-13")
    other_chain["arena_netuid"] = 72
    service, transport = _service([archive, opened, published, shadow, other_chain])
    result = dashboard.competition_snapshot(service, limit=1)
    assert result["open_round"]["round_id"] == opened["round_id"]
    assert result["latest_completed_round"]["round_id"] == published["round_id"]
    assert [call.get("offset") for call in transport.calls[:3]] == [0, 1, None]
    assert all(call["filters"]["configuration_doc->>mode"] == "live" for call in transport.calls)
    assert all(call["filters"]["arena_netuid"] == 71 for call in transport.calls)
    assert all(call["columns"] == dashboard._COMPETITION_ROUND_COLUMNS for call in transport.calls[:3])
    assert "private" not in json.dumps(result)
    assert service._store.get_round(published["round_id"]) == published
    assert transport.calls[-1]["columns"] == "*"
    assert "configuration_doc" in dashboard._ROUND_COLUMNS.split(",")
    pinned, pinned_transport = _service([archive], pinned=archive["round_id"])
    assert pinned.public_competition()["latest_round"]["round_id"] == archive["round_id"]
    assert not pinned_transport.calls
    shadow_service, _ = _service([shadow, published], mode="shadow")
    assert shadow_service.public_competition()["latest_completed_round"]["mode"] == "shadow"


def test_completed_score_selection_still_loads_full_round():
    row = _row(execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
               sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY)
    row.update(status="stage2_scoring", stage1_scoring_plan_doc={"private": True})
    service = SimpleNamespace(_round=Mock(return_value=row), completed_submission_scores=Mock(return_value={"model-0": {"final_score": 0}}))
    projected = dashboard._competition_round(_project(row, dashboard._COMPETITION_ROUND_COLUMNS))
    assert dashboard._completed_scores(service, projected) == {"model-0": {"final_score": 0}}
    service._round.assert_called_once_with(row["round_id"])
    service.completed_submission_scores.assert_called_once_with(row)
    row["configuration_doc"].pop("execution_sequence_policy")
    projected = dashboard._competition_round(_project(row, dashboard._COMPETITION_ROUND_COLUMNS))
    assert dashboard._completed_scores(service, projected) == {}
    assert service._round.call_count == 1


def test_per_icp_cost_projection_keeps_frozen_stage_counts():
    row = _row(stage_1_icp_count=1, stage_2_icp_count=1,
               sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY)
    bucket = {key: 0 for key in dashboard._COST_COUNTER_KEYS + dashboard._SUCCESSFUL_CALL_COST_COUNTER_KEYS}
    bucket["providers"] = []
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligibility_reason="eligible", cost_summary={
        "returned_company_count": 0, "qualified_company_count": 0, "eligible_icp_count": 2,
        "competition_sourcing_microusd": 0, "execution_icp_cap_microusd": 100,
        "cost_per_company_cap_microusd": 100, "execution": bucket, "judge": bucket,
        "per_icp": [{"icp_position": position, "returned_company_count": 0,
                     "qualified_company_count": 0, "competition_sourcing_microusd": 0,
                     "eligibility_cap_microusd": 0, "eligible": True,
                     "eligibility_reason": "eligible"} for position in range(2)],
    })
    projected = dashboard._competition_round(_project(row, dashboard._COMPETITION_ROUND_COLUMNS))
    expected = dashboard.round_summary(row)
    assert dashboard.round_summary(projected) == expected
    assert len(expected["baseline"]["cost_summary"]["per_icp"]) == 2


def test_exact_postgrest_select_and_fixed_psycopg_projection():
    requests = []
    row = _row()
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[_project(row, dashboard._COMPETITION_ROUND_COLUMNS)]))) as http:
        store = ArenaStore(PostgrestTransport("https://example.test", service_key="sb_secret_test", http_client=http))
        returned = store.list_rounds(mode="live", network_name="finney", netuid=71, limit=30, offset=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
    params = requests[0].url.params
    assert params["select"] == dashboard._COMPETITION_ROUND_COLUMNS
    assert params["configuration_doc->>mode"] == "eq.live"
    assert params["arena_network_name"] == "eq.finney"
    assert params["arena_netuid"] == "eq.71"
    assert params["limit"] == "30" and params["offset"] == "30"
    assert params["order"] == "created_at.desc"
    assert set(returned[0]) == {column.split(":")[0] for column in dashboard._COMPETITION_ROUND_COLUMNS.split(",")}
    queries = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []

    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    store = ArenaStore(direct)
    store.list_rounds(mode="live", network_name="finney", netuid=71, limit=30, offset=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
    sql, values = queries[-1]
    for key in COMPETITION_CONFIGURATION_FIELDS:
        assert "(configuration_doc -> '%s')::text AS cfg_%s" % (key, key) in sql
    assert "cfg_" in sql and ":configuration_doc" not in sql
    assert "public.lab_arena_competition_publication_v1(lab_arena_rounds) AS publication_doc" in sql
    assert "ORDER BY created_at DESC LIMIT 30 OFFSET 30" in sql
    assert values == ["live", "finney", 71]
    store.latest_published_day(network_name="finney", netuid=71)
    sql, values = queries[-1]
    assert "SELECT round_id,evaluation_date FROM" in sql
    assert "ORDER BY evaluation_date DESC LIMIT 1" in sql
    assert values == ["published", "live", "finney", 71]


def test_pending_promotion_uses_narrow_latest_day_without_changing_decision():
    older, newer = _row("2026-10-08"), _row("2026-10-09")
    service, transport = _service([older, newer])
    service._store.pending_promotions = lambda **_kwargs: [{"round_id": older["round_id"]}]
    assert service._pending_promotion_blocks() is False
    assert transport.calls[0]["columns"] == "round_id,evaluation_date"
    service._store.pending_promotions = lambda **_kwargs: [{"round_id": newer["round_id"]}]
    assert service._pending_promotion_blocks() is True
