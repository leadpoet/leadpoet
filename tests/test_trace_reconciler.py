"""Tests for gateway/research_lab/trace_reconciler.py (P6/P7)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from gateway.research_lab import trace_reconciler as tr
from research_lab.canonical import sha256_json


class FakeStore:
    def __init__(self, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.tables = {name: [dict(row) for row in rows] for name, rows in tables.items()}

    async def select_all(self, table, *, columns="*", filters=(), order_by=(), max_rows=1000):
        return [dict(row) for row in self.tables.get(table, [])][:max_rows]


class FakeS3:
    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)

    def head_object(self, Bucket, Key):
        if f"s3://{Bucket}/{Key}" not in self.objects:
            raise RuntimeError("404 Not Found: NoSuchKey")
        return {}

    def get_object(self, Bucket, Key):
        body = self.objects[f"s3://{Bucket}/{Key}"]

        class _Body:
            def read(self_inner):
                return body

        return {"Body": _Body()}


def _tables() -> dict[str, list[dict[str, Any]]]:
    return {
        tr.LOOP_EVENTS_TABLE: [
            {
                "run_id": "run-1",
                "provider_usage": [
                    {
                        "provider": "openrouter",
                        "raw_trace_ref": {
                            "s3_ref": "s3://bucket/traces/0001.json.enc",
                            "sha256": "sha256:aaaa",
                        },
                    }
                ],
                "event_doc": {
                    "diagnostic_artifact_uri": "s3://bucket/candidates/run-1/diagnostics/002.json",
                    "diagnostic_artifact_hash": "sha256:dddd",
                },
                "created_at": "2026-07-01T00:00:00Z",
            }
        ],
        tr.EVALUATION_EVENTS_TABLE: [
            {
                "run_id": "run-1",
                "candidate_id": "cand-1",
                "event_type": "scored",
                "event_doc": {
                    "scorer_trace_refs": {
                        "icp-1": {
                            "s3_ref": "s3://bucket/scorer-traces/cand-1/icp-1.json",
                            "sha256": "sha256:ssss",
                        }
                    }
                },
                "created_at": "2026-07-01T00:01:00Z",
            }
        ],
        tr.SCORE_BUNDLES_TABLE: [
            {
                "run_id": "run-1",
                "score_bundle_id": "sb-1",
                "score_bundle_doc": {
                    "aggregates": {
                        "per_icp_results": [
                            {
                                "icp_ref": "icp-1",
                                "incontainer_trace_ref": "s3://bucket/incontainer/cand-1/icp-1.json",
                                "incontainer_trace_sha256": "sha256:iiii",
                                "incontainer_trace_call_count": 3,
                                "scorer_trace_ref": "s3://bucket/scorer-traces/cand-1/icp-1.json",
                                "scorer_trace_sha256": "sha256:ssss",
                            },
                            {
                                "icp_ref": "icp-2",
                                "incontainer_trace_ref": "",
                                "incontainer_trace_dropped": True,
                                "incontainer_trace_dropped_call_count": 2,
                            },
                        ]
                    }
                },
                "created_at": "2026-07-01T00:02:00Z",
            }
        ],
        tr.EXECUTION_TRACES_TABLE: [
            {
                "run_id": "trace-uuid-1",
                "status": "crash",
                "calls": [
                    {
                        "call_kind": "engine_raw_trace",
                        "s3_ref": "s3://bucket/traces/0001.json.enc",
                        "sha256": "sha256:aaaa",
                    },
                    {
                        "call_kind": "build_diagnostic_artifact",
                        "s3_ref": "s3://bucket/candidates/run-1/diagnostics/002.json",
                        "sha256": "sha256:dddd",
                    },
                    {
                        "call_kind": "incontainer_trace",
                        "s3_ref": "s3://bucket/incontainer/cand-1/icp-1.json",
                        "sha256": "sha256:iiii",
                    },
                ],
                "judge_verdicts": [
                    {
                        "verdict_kind": "scorer_judgment_trace",
                        "icp_ref": "icp-1",
                        "s3_ref": "s3://bucket/scorer-traces/cand-1/icp-1.json",
                        "sha256": "sha256:ssss",
                    }
                ],
                "trace_doc": {},
                "created_at": "2026-07-01T00:03:00Z",
            }
        ],
        tr.EVIDENCE_BUNDLES_TABLE: [
            {
                "bundle_id": "eb-1",
                "snapshots": [
                    {
                        "snapshot_kind": "per_icp_score_evidence",
                        "incontainer_trace_ref": "s3://bucket/incontainer/cand-1/icp-1.json",
                        "incontainer_trace_sha256": "sha256:iiii",
                    }
                ],
                "bundle_doc": {},
                "created_at": "2026-07-01T00:04:00Z",
            }
        ],
    }


async def test_collect_trace_pointers_finds_all_channels():
    pointers = await tr.collect_trace_pointers(store=FakeStore(_tables()))
    refs = {pointer["s3_ref"] for pointer in pointers}
    assert refs == {
        "s3://bucket/traces/0001.json.enc",
        "s3://bucket/candidates/run-1/diagnostics/002.json",
        "s3://bucket/scorer-traces/cand-1/icp-1.json",
        "s3://bucket/incontainer/cand-1/icp-1.json",
    }


async def test_reconcile_classifies_verified_and_missing():
    s3 = FakeS3(
        {
            "s3://bucket/traces/0001.json.enc": b"x",
            "s3://bucket/scorer-traces/cand-1/icp-1.json": b"y",
            # diagnostics + incontainer objects deliberately missing
        }
    )
    report = await tr.reconcile_trace_pointers(store=FakeStore(_tables()), s3_client=s3)
    assert report["status"] == "completed"
    assert report["counts"]["verified"] == 2
    assert report["counts"]["missing"] == 2
    missing_refs = {problem["s3_ref"] for problem in report["problems"]}
    assert "s3://bucket/incontainer/cand-1/icp-1.json" in missing_refs


async def test_reconcile_hash_mismatch_detected():
    body = b'{"doc": 1}'
    good_hash = "sha256:" + __import__("hashlib").sha256(body).hexdigest()
    tables = {
        tr.LOOP_EVENTS_TABLE: [
            {
                "run_id": "run-1",
                "provider_usage": [
                    {
                        "raw_trace_ref": {
                            "s3_ref": "s3://bucket/a.json",
                            "sha256": good_hash,
                        }
                    },
                    {
                        "raw_trace_ref": {
                            "s3_ref": "s3://bucket/b.json",
                            "sha256": "sha256:not-the-real-hash",
                        }
                    },
                ],
                "event_doc": {},
                "created_at": "t",
            }
        ],
    }
    s3 = FakeS3({"s3://bucket/a.json": body, "s3://bucket/b.json": body})
    report = await tr.reconcile_trace_pointers(
        store=FakeStore(tables), s3_client=s3, verify_hash=True
    )
    assert report["counts"]["verified"] == 1
    assert report["counts"]["hash_mismatch"] == 1


async def test_capture_coverage_counts_all_channels():
    report = await tr.summarize_capture_coverage(store=FakeStore(_tables()))
    captured = report["captured"]
    assert captured["engine_raw_traces"] == 1
    assert captured["scorer_traces"] == 1
    assert captured["incontainer_traces"] == 1
    assert captured["incontainer_dropped"] == 1
    assert captured["build_diagnostics"] == 1
    projected = report["projected"]
    assert projected["engine_raw_calls"] == 1
    assert projected["incontainer_calls"] == 1
    assert projected["build_diagnostic_calls"] == 1
    assert projected["judge_verdicts"] == 1
    assert projected["crash_rows"] == 1
    gaps = report["projection_gaps"]
    assert gaps["scorer_traces_unprojected"] == 0
    assert gaps["build_diagnostics_unprojected"] == 0


async def test_capture_coverage_reports_projection_gap():
    tables = _tables()
    tables[tr.EXECUTION_TRACES_TABLE] = []  # nothing projected yet
    report = await tr.summarize_capture_coverage(store=FakeStore(tables))
    assert report["projection_gaps"]["scorer_traces_unprojected"] == 1
    assert report["projection_gaps"]["build_diagnostics_unprojected"] == 1


# ---------------------------------------------------------------------------
# P18: shadow window import + trajectory_id join key
# ---------------------------------------------------------------------------


class InsertingFakeStore(FakeStore):
    def __init__(self, tables):
        super().__init__(tables)
        self.inserted: dict[str, list[dict[str, Any]]] = {}

    async def select_many(self, table, *, columns="*", filters=(), order_by=(), limit=100):
        rows = self.tables.get(table, [])
        out = []
        for row in rows:
            keep = True
            for spec in filters:
                if len(spec) == 2 and str(row.get(spec[0])) != str(spec[1]):
                    keep = False
            if keep:
                out.append(dict(row))
        return out[:limit]

    async def insert_row(self, table, row):
        stored = self.inserted.setdefault(table, [])
        if any(r.get("window_id") == row.get("window_id") for r in stored):
            raise RuntimeError("duplicate key value violates unique constraint")
        stored.append(dict(row))
        return dict(row)


def _promotion_tables():
    return {
        tr.PROMOTION_EVENTS_TABLE: [
            {
                "promotion_event_id": "pe-1",
                "candidate_id": "cand-1",
                "private_model_version_id": "ver-1",
                "event_type": "active_version_created",
                "event_doc": {"new_model_manifest_uri": "s3://bucket/lab/current.json"},
                "created_at": "2026-07-01T00:00:00Z",
            },
            {
                "promotion_event_id": "pe-2",
                "candidate_id": "cand-2",
                "private_model_version_id": "ver-2",
                "event_type": "active_version_created",
                "event_doc": {"new_model_manifest_uri": "s3://bucket/lab/current.json"},
                "created_at": "2026-07-02T00:00:00Z",
            },
        ]
    }


async def test_import_shadow_windows_reports_and_not_monitored(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_SHADOW_REPORT_URI_PREFIX", raising=False)
    docs = {
        # ver-1 has a completed window report with one alert.
        "s3://bucket/lab/shadow-windows/ver-1/window-report.json": {
            "active_version_id": "ver-1",
            "comparable_day_count": 7,
            "cumulative_shadow_live_diff_points": -2.4,
            "mean_shadow_live_diff_points": -0.34,
            "alerts": [{"alert_type": "cumulative_regression"}],
        },
        # ver-2 has nothing: never monitored.
    }
    store = InsertingFakeStore(_promotion_tables())
    result = await tr.import_shadow_windows(store=store, get_json=docs.get)
    assert result["merges_seen"] == 2
    assert result["imported"] == 2
    assert result["not_monitored"] == 1

    rows = {row["active_version_id"]: row for row in store.inserted[tr.SHADOW_WINDOWS_TABLE]}
    monitored = rows["ver-1"]
    assert monitored["window_status"] == "alerted"
    assert monitored["comparable_day_count"] == 7
    assert monitored["alert_count"] == 1
    assert monitored["promotion_event_ref"] == "promotion_event:pe-1"
    unmonitored = rows["ver-2"]
    assert unmonitored["window_status"] == "not_monitored"
    assert unmonitored["window_doc"]["reason"] == "window_docs_not_found"


async def test_import_shadow_windows_idempotent(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_SHADOW_REPORT_URI_PREFIX", raising=False)
    store = InsertingFakeStore(_promotion_tables())
    await tr.import_shadow_windows(store=store, get_json=lambda uri: None)
    second = await tr.import_shadow_windows(store=store, get_json=lambda uri: None)
    assert second["imported"] == 0
    assert second["skipped_existing"] == 2
