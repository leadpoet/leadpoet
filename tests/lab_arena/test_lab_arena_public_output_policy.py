"""Public round output policy follows the configuration frozen for that round."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError


POLICY_FIELDS = (
    "output_schema_version",
    "integrity_policy",
    "contact_policy",
    "company_quality_policy",
    "intent_details_policy",
)


def _round(round_id: str, status: str, **policies: str) -> dict:
    return {
        "round_id": round_id,
        "status": status,
        "configuration_doc": {
            "mode": "shadow",
            "schedule": {"submission_cutoff": "2026-09-15T12:00:00Z"},
            "private_icps": ["PRIVATE ICP MUST NOT LEAK"],
            "runner_hotkeys": ["PRIVATE RUNNER MUST NOT LEAK"],
            **policies,
        },
        "participants": [],
        "publication_doc": None,
    }


def _service(rows: list[dict]) -> ArenaService:
    service = ArenaService.__new__(ArenaService)
    by_id = {row["round_id"]: row for row in rows}

    class Store:
        def list_rounds(self, **query):
            if query.get("statuses"):
                return [row for row in rows if row["status"] != "published"]
            return []

    service._store = Store()
    service._config = SimpleNamespace(
        mode="shadow",
        network_name="finney",
        netuid=71,
        pinned_round_id=None,
        chain=SimpleNamespace(current_settlement_epoch=lambda: 100),
    )
    service._round = lambda round_id: by_id[round_id] if round_id in by_id else _missing_round()
    service.latest_published_round = lambda: None
    return service


def _missing_round():
    raise ServiceError("round_missing", 404)


@pytest.mark.parametrize(
    ("policies", "schema"),
    [
        ({}, "leadpoet.lab_arena.output.v1"),
        ({"integrity_policy": "arena_integrity_v1", "contact_policy": "contacts_v1"}, "leadpoet.lab_arena.output.v2"),
        ({"integrity_policy": "arena_integrity_v1", "company_quality_policy": "company_quality_v1"}, "leadpoet.lab_arena.output.v3"),
        ({"integrity_policy": "arena_integrity_v1", "contact_policy": "contacts_v1", "company_quality_policy": "company_quality_v1"}, "leadpoet.lab_arena.output.v4"),
        ({"integrity_policy": "arena_integrity_v1", "intent_details_policy": "intent_details_v1"}, "leadpoet.lab_arena.output.v6"),
        ({"integrity_policy": "arena_integrity_v1", "contact_policy": "contacts_v1", "company_quality_policy": "company_quality_v1", "intent_details_policy": "intent_details_v1"}, "leadpoet.lab_arena.output.v5"),
    ],
)
def test_round_endpoint_reports_frozen_output_contract_without_private_configuration(policies, schema):
    row = _round("arena-2026-09-14", "published", **policies)
    service = _service([row])

    with TestClient(create_app(service)) as http:
        response = http.get("/arena/v1/rounds/arena-2026-09-14")

    assert response.status_code == 200
    view = response.json()
    assert {key: view[key] for key in POLICY_FIELDS if key in view} == {
        "output_schema_version": schema,
        **policies,
    }
    serialized = json.dumps(view)
    assert "PRIVATE ICP" not in serialized
    assert "PRIVATE RUNNER" not in serialized


def test_current_endpoint_discloses_open_and_running_round_policies_before_intake():
    open_row = _round(
        "arena-2026-09-15", "open",
        integrity_policy="arena_integrity_v1",
        contact_policy="contacts_v1",
        company_quality_policy="company_quality_v1",
        intent_details_policy="intent_details_v1",
    )
    running_row = _round("arena-2026-09-14", "stage_1", integrity_policy="arena_integrity_v1")
    service = _service([open_row, running_row])

    with TestClient(create_app(service)) as http:
        current = http.get("/arena/v1/current")
        opened = http.get("/arena/v1/rounds/arena-2026-09-15")

    assert current.status_code == opened.status_code == 200
    view = current.json()
    open_policy = {key: view["open_round"][key] for key in POLICY_FIELDS if key in view["open_round"]}
    round_policy = {key: opened.json()[key] for key in POLICY_FIELDS if key in opened.json()}
    assert view["round"]["round_id"] == open_row["round_id"]
    assert open_policy == round_policy
    assert view["running_rounds"][0]["output_schema_version"] == "leadpoet.lab_arena.output.v1"
    assert view["running_rounds"][0]["integrity_policy"] == "arena_integrity_v1"
    serialized = json.dumps(view)
    assert "PRIVATE ICP" not in serialized
    assert "PRIVATE RUNNER" not in serialized
