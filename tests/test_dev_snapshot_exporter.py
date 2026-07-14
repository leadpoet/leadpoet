"""Production-shaped read-only tests for the hidden dev-set exporter."""

from __future__ import annotations

import json
from types import SimpleNamespace

from research_lab.canonical import sha256_json
from research_lab.eval.dev_eval import intent_signal_signature
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


def test_exporter_is_deterministic_and_excludes_full_current_horizon(
    monkeypatch,
    tmp_path,
):
    horizon_a = _icp(100, signal="private-current-signal")
    horizon_b = _icp(101)
    retired = [_icp(index) for index in range(20)]
    retired.append(_icp(999, signal="private-current-signal"))
    rows_by_table = {
        "research_lab_rolling_icp_windows": [
            {
                "rolling_window_hash": "sha256:" + "a" * 64,
                "required_days": 2,
                "window_doc": {
                    "fresh_set_id": 4,
                    "required_days": 2,
                    "sets": [{"set_id": 4}, {"set_id": 3}],
                },
                "created_at": "2026-07-13T00:00:00Z",
            }
        ],
        "qualification_private_icp_sets": [
            {"set_id": 4, "icps": [horizon_a], "is_active": True},
            {"set_id": 3, "icps": [horizon_b], "is_active": True},
            {"set_id": 2, "icps": retired[:11], "is_active": False},
            {"set_id": 1, "icps": retired[11:], "is_active": False},
        ],
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
            ["export", "--out-dir", str(out), "--seed", "fixed-seed"],
        )
        assert exporter.main() == 0
        outputs.append(
            json.loads((out / "source_icps.json").read_text(encoding="utf-8"))
        )

    assert outputs[0] == outputs[1]
    assert len(outputs[0]["items"]) == 8
    selected_signatures = {
        intent_signal_signature(row["icp"]) for row in outputs[0]["items"]
    }
    assert intent_signal_signature(horizon_a) not in selected_signatures
    selected_hashes = {row["icp_hash"] for row in outputs[0]["items"]}
    assert sha256_json({"icp": horizon_a}) not in selected_hashes
    exclusions = json.loads(
        (tmp_path / "first" / "holdout_window_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    assert intent_signal_signature(horizon_a) in exclusions["hashes"]
    assert "qualification_private_icp_sets:4:icp-100" in exclusions["hashes"]


def test_exporter_fails_without_service_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["export", "--out-dir", str(tmp_path)],
    )
    assert exporter.main() == 1
