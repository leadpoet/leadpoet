"""Production-shaped read-only tests for the hidden dev-set exporter."""

from __future__ import annotations

import json
from types import SimpleNamespace

from gateway.research_lab.config import DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG
from gateway.research_lab.icp_window import (
    WINDOW_MODE_LEGACY_ROLLING,
    select_rolling_icp_window_from_sets,
)
from scripts import export_research_lab_dev_icp_inputs as exporter


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)
        self.start = 0
        self.end = 999

    def select(self, _columns):
        return self

    def order(self, _field, *, desc):
        self.rows.sort(key=lambda row: row.get(_field), reverse=desc)
        return self

    def range(self, start, end):
        self.start, self.end = start, end
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows[self.start : self.end + 1])


class _Client:
    def __init__(self, rows_by_table):
        self.rows_by_table = rows_by_table

    def table(self, name):
        return _Query(self.rows_by_table[name])


def _icp(index: int, *, signal: str | None = None):
    return {
        "icp_id": f"icp-{index}",
        "industry": ["Software Development", "Financial Services", "Manufacturing"][
            index % 3
        ],
        "sub_industry": f"segment-{index % 5}",
        "product_service": f"product-{index}",
        "country": ["United States", "Canada", "Germany"][index % 3],
        "geography": ["United States", "Canada", "Germany"][index % 3],
        "employee_count": ["11-50", "51-200", "201-500"][index % 3],
        "intent_signals": [signal or f"intent-{index}"],
    }


def test_rows_paginates_past_postgrest_default_cap():
    rows = [{"set_id": index} for index in range(1005)]
    loaded = exporter._rows(
        _Client({"sets": rows}),
        "sets",
        "set_id",
        order="set_id",
        desc=False,
    )
    assert len(loaded) == 1005
    assert loaded[0]["set_id"] == 0
    assert loaded[-1]["set_id"] == 1004


def test_exporter_is_deterministic_and_uses_completed_current_day_bank(
    monkeypatch,
    tmp_path,
):
    benchmark_date = "2026-07-13"
    model_manifest_hash = "sha256:" + "f" * 64
    source_set = {
        "set_id": 4,
        "icps": [_icp(index) for index in range(8)],
        "icp_set_hash": "sha256:" + "4" * 64,
        "is_active": True,
    }
    window = select_rolling_icp_window_from_sets(
        [source_set],
        days=1,
        icps_per_day=8,
        window_mode=WINDOW_MODE_LEGACY_ROLLING,
    )
    categories = ("public", "private", "conditional")
    per_icp_summaries = [
        {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": float(index * 10),
        }
        for index, item in enumerate(window.benchmark_items, start=1)
    ]
    category_items = [
        {
            "icp_ref": item["icp_ref"],
            "category": categories[index % len(categories)],
        }
        for index, item in enumerate(window.benchmark_items)
    ]
    rows_by_table = {
        "research_lab_private_model_benchmark_current": [
            {
                "benchmark_bundle_id": "benchmark-bundle-current",
                "benchmark_bundle_hash": "sha256:" + "b" * 64,
                "benchmark_date": benchmark_date,
                "private_model_manifest_hash": model_manifest_hash,
                "rolling_window_hash": window.window_hash,
                "evaluation_epoch": 24000,
                "benchmark_quality": "passed",
                "current_benchmark_status": "completed",
                "score_summary_doc": {
                    "per_icp_summaries": per_icp_summaries,
                    "category_assignment": {"items": category_items},
                },
                "created_at": "2026-07-13T23:00:00Z",
            }
        ],
        "research_lab_rolling_icp_windows": [
            {
                "rolling_window_hash": window.window_hash,
                "window_doc": window.public_doc,
                "created_at": "2026-07-13T22:00:00Z",
            }
        ],
        "qualification_private_icp_sets": [source_set],
    }
    client = _Client(rows_by_table)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-only")
    monkeypatch.setitem(
        __import__("sys").modules,
        "supabase",
        SimpleNamespace(create_client=lambda _url, _key: client),
    )

    outputs = []
    for suffix in ("first", "second"):
        out = tmp_path / suffix
        monkeypatch.setattr(
            __import__("sys"),
            "argv",
            [
                "export",
                "--out-dir",
                str(out),
                "--benchmark-date",
                benchmark_date,
                "--expected-private-model-manifest-hash",
                model_manifest_hash,
            ],
        )
        assert exporter.main() == 0
        outputs.append(
            json.loads((out / "source_icps.json").read_text(encoding="utf-8"))
        )

    assert outputs[0] == outputs[1]
    assert outputs[0]["schema_version"] == "research_lab.dev_icp_export.v2"
    assert outputs[0]["configured_dev_icp_count"] == (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    )
    assert len(outputs[0]["items"]) == len(window.benchmark_items)
    assert {
        row["benchmark_category"] for row in outputs[0]["items"]
    } == {"public", "private", "conditional"}
    manifest = outputs[0]["daily_bank_manifest"]
    assert manifest["benchmark_date"] == benchmark_date
    assert manifest["rolling_window_hash"] == window.window_hash
    assert manifest["private_model_manifest_hash"] == model_manifest_hash
    assert manifest["bank_size"] == len(window.benchmark_items)
    assert not (tmp_path / "first" / "holdout_window_hashes.json").exists()


def test_exporter_fails_without_service_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["export", "--out-dir", str(tmp_path)],
    )
    assert exporter.main() == 1
