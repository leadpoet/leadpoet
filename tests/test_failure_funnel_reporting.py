"""Private Research Lab failure-funnel contract tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab import failure_funnel


INDEX_MIGRATION = Path(
    "scripts/150-research-lab-failure-funnel-indexes.concurrent.sql"
)
MIGRATION = Path("scripts/151-research-lab-failure-funnel-reporting.sql")


def _valid_telemetry(status: str = "complete", **overrides: int) -> dict[str, Any]:
    telemetry = {
        field: 0 for field in failure_funnel._TELEMETRY_COUNT_FIELDS
    }
    telemetry.update(overrides)
    return {"status": status, **telemetry}


def test_failure_funnel_migration_is_service_only_and_read_only():
    index_sql = INDEX_MIGRATION.read_text(encoding="utf-8")
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CREATE OR REPLACE FUNCTION public.get_research_lab_failure_funnel" in sql
    assert "SECURITY INVOKER" in sql
    assert (
        "REVOKE ALL ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)"
        in sql
    )
    assert "FROM PUBLIC, anon, authenticated" in sql
    assert (
        "GRANT EXECUTE ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)"
        in sql
    )
    assert "TO service_role" in sql
    assert "CREATE TABLE" not in sql
    assert "CREATE INDEX" not in sql
    assert index_sql.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 3
    assert "\nBEGIN;\n" not in index_sql
    assert "\nCOMMIT;\n" not in index_sql
    assert "indisvalid" in index_sql
    assert "indisready" in index_sql
    assert "indislive" in index_sql
    assert "RESET lock_timeout" in index_sql
    assert "NOTIFY pgrst, 'reload schema'" in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE" not in sql
    assert "raw provider" not in sql.lower()


async def test_failure_funnel_loader_returns_rpc_report(monkeypatch):
    expected = {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": "ticket-1",
        "candidate_id": None,
        "stages": [
            {
                "stage": "sourcing",
                "unit": "icp_attempts",
                "reviewed": 2,
                "passed": 1,
                "rejected": 1,
            }
        ],
        "rejections": [],
        "model_revisions": [],
        "telemetry": _valid_telemetry(),
    }
    captured = {}

    async def fake_call_rpc(name, params):
        captured.update({"name": name, "params": params})
        return expected

    monkeypatch.setattr(failure_funnel, "call_rpc", fake_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report == expected
    assert captured == {
        "name": "get_research_lab_failure_funnel",
        "params": {"p_ticket_id": "ticket-1", "p_candidate_id": None},
    }


async def test_failure_funnel_loader_degrades_to_explicit_missing(monkeypatch):
    async def failed_call_rpc(*_args, **_kwargs):
        raise RuntimeError("migration not applied")

    monkeypatch.setattr(failure_funnel, "call_rpc", failed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1", "candidate-1")

    assert report["stages"] == []
    assert report["rejections"] == []
    assert report["telemetry"] == {
        "status": "missing",
        "report_available": False,
    }


async def test_failure_funnel_loader_rejects_malformed_rpc_result(monkeypatch):
    async def malformed_call_rpc(*_args, **_kwargs):
        return {"telemetry": None}

    monkeypatch.setattr(failure_funnel, "call_rpc", malformed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report["telemetry"] == {
        "status": "missing",
        "report_available": True,
    }


@pytest.mark.parametrize(
    "invalid_field,invalid_value",
    [
        ("schema_version", "research_lab_failure_funnel.v2"),
        ("ticket_id", "other-ticket"),
        ("candidate_id", "other-candidate"),
        ("telemetry", {"status": "complete"}),
    ],
)
async def test_failure_funnel_loader_validates_full_rpc_envelope(
    monkeypatch, invalid_field, invalid_value
):
    raw_report = {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": "ticket-1",
        "candidate_id": "candidate-1",
        "stages": [],
        "rejections": [],
        "model_revisions": [],
        "telemetry": _valid_telemetry(),
    }
    raw_report[invalid_field] = invalid_value

    async def malformed_call_rpc(*_args, **_kwargs):
        return raw_report

    monkeypatch.setattr(failure_funnel, "call_rpc", malformed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel(
        "ticket-1", "candidate-1"
    )

    assert report["telemetry"] == {
        "status": "missing",
        "report_available": True,
    }


@pytest.mark.parametrize(
    "stages,rejections,revisions",
    [
        (
            [
                {
                    "stage": "sourcing",
                    "unit": "icp_attempts",
                    "reviewed": 1,
                    "passed": 1,
                    "rejected": 1,
                }
            ],
            [],
            [],
        ),
        (
            [],
            [
                {
                    "stage": "scoring",
                    "reason_code": "failure",
                    "unit": "companies",
                    "count": -1,
                }
            ],
            [],
        ),
        ([], [], ["not-a-model-revision"]),
    ],
)
async def test_failure_funnel_loader_validates_nested_counts_and_revisions(
    monkeypatch, stages, rejections, revisions
):
    async def malformed_call_rpc(*_args, **_kwargs):
        return {
            "schema_version": "research_lab_failure_funnel.v1",
            "ticket_id": "ticket-1",
            "candidate_id": None,
            "stages": stages,
            "rejections": rejections,
            "model_revisions": revisions,
            "telemetry": _valid_telemetry(),
        }

    monkeypatch.setattr(failure_funnel, "call_rpc", malformed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report["telemetry"] == {
        "status": "missing",
        "report_available": True,
    }


async def test_failure_funnel_loader_contains_mapping_conversion_failure(monkeypatch):
    class BrokenMapping(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            if key == "telemetry":
                return {"status": "complete"}
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("decode failed")

        def __len__(self) -> int:
            return 1

    async def malformed_call_rpc(*_args, **_kwargs):
        return BrokenMapping()

    monkeypatch.setattr(failure_funnel, "call_rpc", malformed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report["telemetry"] == {
        "status": "missing",
        "report_available": True,
    }


def test_miner_failure_funnel_is_counts_only_provider_neutral_and_has_no_revision():
    projected = failure_funnel.miner_failure_funnel_projection(
        {
            "ticket_id": "ticket-1",
            "candidate_id": "candidate-1",
            "stages": [
                {
                    "stage": "sourcing",
                    "unit": "icp_attempts",
                    "reviewed": 2,
                    "passed": 1,
                    "rejected": 1,
                }
            ],
            "rejections": [
                {
                    "stage": "provider_transport",
                    "reason_code": "provider_timeout",
                    "unit": "icp_attempts",
                    "count": 1,
                },
                {
                    "stage": "infrastructure",
                    "reason_code": "supabase_read_timeout",
                    "unit": "icp_attempts",
                    "count": 2,
                },
                {
                    "stage": "openrouter",
                    "reason_code": "vendor_fault",
                    "unit": "provider_calls",
                    "count": 4,
                },
            ],
            "model_revisions": ["sha256:" + "a" * 64],
            "telemetry": {"status": "partial", "infrastructure_failure_count": 3},
        }
    )

    assert "model_revisions" not in projected
    assert projected["rejections"] == [
        {
            "stage": "infrastructure",
            "reason_code": "external_service_failure",
            "unit": "icp_attempts",
            "count": 3,
        },
        {
            "stage": "unclassified",
            "reason_code": "unclassified_failure",
            "unit": "units",
            "count": 4,
        },
    ]
    assert "provider" not in str(projected).lower()
    assert "supabase" not in str(projected).lower()
    assert "openrouter" not in str(projected).lower()


def test_miner_projection_preserves_common_scorer_stage_codes():
    projected = failure_funnel.miner_failure_funnel_projection(
        {
            "ticket_id": "ticket-1",
            "candidate_id": "candidate-1",
            "stages": [],
            "rejections": [
                {
                    "stage": "company_fit",
                    "reason_code": "employee_count_mismatch",
                    "unit": "companies",
                    "count": 2,
                },
                {
                    "stage": "attribute",
                    "reason_code": "other",
                    "unit": "companies",
                    "count": 1,
                },
            ],
            "telemetry": {"status": "complete"},
        }
    )

    assert projected["rejections"] == [
        {
            "stage": "attribute",
            "reason_code": "other",
            "unit": "companies",
            "count": 1,
        },
        {
            "stage": "company_fit",
            "reason_code": "employee_count_mismatch",
            "unit": "companies",
            "count": 2,
        },
    ]


def test_miner_projection_preserves_current_company_fit_reason_codes():
    projected = failure_funnel.miner_failure_funnel_projection(
        {
            "ticket_id": "ticket-1",
            "candidate_id": "candidate-1",
            "stages": [],
            "rejections": [
                {
                    "stage": "company_fit",
                    "reason_code": "company_fit_not_proven",
                    "unit": "companies",
                    "count": 2,
                },
                {
                    "stage": "attribute",
                    "reason_code": "required_attribute_not_proven",
                    "unit": "companies",
                    "count": 1,
                },
            ],
            "telemetry": {"status": "complete"},
        }
    )

    assert {
        (row["stage"], row["reason_code"], row["count"])
        for row in projected["rejections"]
    } == {
        ("company_fit", "company_fit_not_proven", 2),
        ("attribute", "required_attribute_not_proven", 1),
    }


def test_miner_projection_does_not_excuse_model_invalid_output_as_infrastructure():
    projected = failure_funnel.miner_failure_funnel_projection(
        {
            "ticket_id": "ticket-1",
            "candidate_id": "candidate-1",
            "stages": [],
            "rejections": [
                {
                    "stage": "scoring",
                    "reason_code": "candidate_model_runtime_invalid_json",
                    "unit": "icp_attempts",
                    "count": 1,
                }
            ],
            "telemetry": {"status": "complete"},
        }
    )

    assert projected["rejections"] == [
        {
            "stage": "scoring",
            "reason_code": "model_invalid_output",
            "unit": "icp_attempts",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    ("raw_reason", "expected_stage", "expected_reason"),
    [
        (
            "candidate_model_runtime_provider_error",
            "infrastructure",
            "external_service_failure",
        ),
        (
            "candidate_model_runtime_timeout",
            "infrastructure",
            "external_service_failure",
        ),
        (
            "provider_cost_cap_blocked",
            "infrastructure",
            "external_service_failure",
        ),
        (
            "candidate_model_runtime_skipped_after_timeout",
            "sourcing",
            "model_runtime_skipped",
        ),
        (
            "candidate_model_zero_companies",
            "sourcing",
            "no_companies_qualified",
        ),
        (
            "candidate_model_zero_scoreable_companies",
            "scoring",
            "no_scoreable_companies",
        ),
    ],
)
def test_miner_projection_coarsens_only_external_receipt_details(
    raw_reason, expected_stage, expected_reason
):
    projected = failure_funnel.miner_failure_funnel_projection(
        {
            "ticket_id": "ticket-1",
            "candidate_id": "candidate-1",
            "stages": [],
            "rejections": [
                {
                    "stage": "scoring",
                    "reason_code": raw_reason,
                    "unit": "icp_attempts",
                    "count": 1,
                }
            ],
            "telemetry": {"status": "partial"},
        }
    )

    assert projected["rejections"] == [
        {
            "stage": expected_stage,
            "reason_code": expected_reason,
            "unit": "icp_attempts",
            "count": 1,
        }
    ]
