"""Aggregate selected native miner readback reports without re-reading production."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)

def _valid_count(value: Any) -> tuple[bool, str]:
    """Return whether value is a non-negative JSON integer, excluding booleans."""
    if value is None:
        return False, "null"
    if type(value) is bool:
        return False, "bool"
    if type(value) is not int:
        return False, "bad_type"
    if value < 0:
        return False, "negative"
    return True, "measured"


def _capture_count(capture: dict[str, Any], field: str, alias: str) -> dict[str, Any]:
    """Read one count without treating missing or malformed data as zero."""
    has_field = field in capture
    has_alias = alias in capture
    if not has_field and not has_alias:
        return {"value": None, "source_field": None, "status": "absent"}

    source_field = field if has_field else alias
    value = capture[source_field]
    valid, status = _valid_count(value)
    if not valid:
        return {"value": None, "source_field": source_field, "status": status}

    if has_field and has_alias:
        alias_value = capture[alias]
        alias_valid, _ = _valid_count(alias_value)
        if alias_valid and alias_value != value:
            return {
                "value": None,
                "source_field": field,
                "status": "conflicting_values",
                "conflicting_fields": [field, alias],
            }
    return {"value": value, "source_field": source_field, "status": "measured"}


def _report_capture_count(report: dict[str, Any], field: str, alias: str) -> dict[str, Any]:
    measured, invalid = [], []
    for name in ("capture_proof", "model_observations"):
        container = report.get(name)
        if container is None:
            continue
        if not isinstance(container, dict):
            invalid.append({"value": None, "source_field": name, "status": "bad_container"})
            continue
        item = _capture_count(container, field, alias)
        if item["source_field"] is not None:
            item["source_field"] = name + "." + item["source_field"]
        if item["status"] == "measured":
            measured.append(item)
        elif item["status"] != "absent":
            invalid.append(item)
    if invalid:
        return invalid[0]
    if not measured:
        return {"value": None, "source_field": None, "status": "absent"}
    sources = [item["source_field"] for item in measured]
    if len({item["value"] for item in measured}) != 1:
        return {"value": None, "source_field": "|".join(sources),
                "status": "conflicting_values", "conflicting_fields": sources}
    return {"value": measured[0]["value"], "source_field": "|".join(sources), "status": "measured"}


def aggregate_index(index_path: Path, data_dir: Path | None = None) -> dict[str, Any]:
    """Aggregate one saved coverage index and its saved reports."""
    data_dir = data_dir or index_path.parent
    idx = json.loads(index_path.read_text())
    _require(idx["accepted_count"] == idx["native_verified_count"] and idx["missing_count"] == 0,
             "incomplete_index")
    rows = idx["runs"]
    _require(len(rows) == idx["native_verified_count"], "index_count_mismatch")
    _require(len({row["run_id"] for row in rows}) == len(rows), "duplicate_run")

    classes = collections.Counter()
    routes = collections.Counter()
    settings = collections.Counter()
    reasons = collections.Counter()
    source_versions = collections.Counter()
    accounting = collections.Counter()
    summary = {
        "selected_provider_calls": 0,
        "selected_model_calls": 0,
        "reasoning_tokens_reported": 0,
        "positive_tokens_without_full_text_known": 0,
        "positive_tokens_without_full_text_unknown_runs": 0,
        "full_text_characters_known": 0,
        "full_text_characters_legacy_fallback_runs": 0,
        "full_text_characters_unknown_runs": 0,
        "summary_model_calls_known": 0,
        "summary_model_calls_unknown_runs": 0,
        "summary_model_calls_total": None,
        "opaque_model_calls_known": 0,
        "opaque_model_calls_unknown_runs": 0,
        "opaque_model_calls_total": None,
        "readable_projection_truncated_runs": 0,
        "raw_hash_verified_runs": 0,
        "readable_path_verified_runs": 0,
        "score_linked_at_report_time": 0,
        "judgment_present_at_report_time": 0,
    }
    positive = []
    missing_fields = []
    report_coverage = {key: {"known_runs": 0, "unknown_runs": 0}
                       for key in ("model_routes", "model_settings", "accounting")}
    call_coverage = {
        "summary_model_calls": {"known_runs": 0, "unknown_runs": 0, "source_fields": collections.Counter(), "unknown_reasons": collections.Counter()},
        "opaque_model_calls": {"known_runs": 0, "unknown_runs": 0, "source_fields": collections.Counter(), "unknown_reasons": collections.Counter()},
    }

    for row in rows:
        _require(row["status"] == "native_export_verified", "unverified_run")
        source_versions[row["serving_sha"]] += 1
        summary["selected_provider_calls"] += row["selected_provider_calls"]
        summary["selected_model_calls"] += row["selected_model_calls"]
        summary["reasoning_tokens_reported"] += row["reasoning_tokens_reported"]
        positive_value = row.get("positive_tokens_without_full_text")
        if type(positive_value) is int and positive_value >= 0:
            summary["positive_tokens_without_full_text_known"] += positive_value
        else:
            summary["positive_tokens_without_full_text_unknown_runs"] += 1
            missing_fields.append({"run_id": row["run_id"], "field": "positive_tokens_without_full_text"})
        summary["readable_projection_truncated_runs"] += row["readable_projection_truncated"] is True
        summary["raw_hash_verified_runs"] += bool(row["raw_example_hash"])
        summary["score_linked_at_report_time"] += row.get("score_run_id") is not None
        summary["judgment_present_at_report_time"] += row.get("judgment_present") is True
        classes.update(row.get("model_classes") or {})
        reasons.update(set(row.get("raw_incomplete_reasons") or []))
        if type(positive_value) is int and positive_value > 0:
            positive.append({"run_id": row["run_id"], "tokens": positive_value, "truncated_model_calls": (row.get("model_classes") or {}).get("truncated", 0)})

        doc = json.loads((data_dir / row["report_file"]).read_text())
        matches = [report for report in doc.get("reports", [doc]) if report.get("run_id") == row["run_id"]]
        _require(len(matches) == 1, "ambiguous_report")
        report = matches[0]
        if type(report.get("full_text_characters")) is int and report["full_text_characters"] >= 0:
            summary["full_text_characters_known"] += report["full_text_characters"]
        elif type((report.get("model_observations") or {}).get("full_text_characters")) is int and (report["model_observations"]["full_text_characters"] >= 0):
            summary["full_text_characters_known"] += report["model_observations"]["full_text_characters"]
            summary["full_text_characters_legacy_fallback_runs"] += 1
        else:
            summary["full_text_characters_unknown_runs"] += 1
            missing_fields.append({"run_id": row["run_id"], "field": "full_text_characters"})

        for metric, field, alias in (
            ("summary_model_calls", "summary_model_calls", "summary_calls"),
            ("opaque_model_calls", "opaque_model_calls", "opaque_calls"),
        ):
            measured = _report_capture_count(report, field, alias)
            coverage = call_coverage[metric]
            if measured["status"] == "measured":
                summary[metric + "_known"] += measured["value"]
                coverage["known_runs"] += 1
                coverage["source_fields"][measured["source_field"]] += 1
            else:
                summary[metric + "_unknown_runs"] += 1
                coverage["unknown_runs"] += 1
                coverage["unknown_reasons"][measured["status"]] += 1
                detail = {
                    "run_id": row["run_id"],
                    "report_file": row["report_file"],
                    "field": field,
                    "source_field": measured["source_field"],
                    "status": measured["status"],
                }
                if measured.get("conflicting_fields"):
                    detail["conflicting_fields"] = measured["conflicting_fields"]
                missing_fields.append(detail)

        if (report.get("readable_path_proof") or {}).get("verified") is True:
            summary["readable_path_verified_runs"] += 1
        if isinstance(report.get("model_routes"), dict):
            report_coverage["model_routes"]["known_runs"] += 1
            routes.update(report["model_routes"])
        else:
            report_coverage["model_routes"]["unknown_runs"] += 1
            missing_fields.append({"run_id": row["run_id"], "field": "model_routes"})
        if isinstance(report.get("model_settings"), dict):
            report_coverage["model_settings"]["known_runs"] += 1
            settings.update(report["model_settings"])
        else:
            report_coverage["model_settings"]["unknown_runs"] += 1
            missing_fields.append({"run_id": row["run_id"], "field": "model_settings"})
        if isinstance(report.get("accounting"), dict):
            report_coverage["accounting"]["known_runs"] += 1
            accounting.update(report["accounting"])
        else:
            report_coverage["accounting"]["unknown_runs"] += 1
            missing_fields.append({"run_id": row["run_id"], "field": "accounting"})

    for metric, coverage in call_coverage.items():
        known_key = metric + "_known"
        summary[metric + "_total"] = summary[known_key] if coverage["unknown_runs"] == 0 else None
        coverage["source_fields"] = dict(sorted(coverage["source_fields"].items()))
        coverage["unknown_reasons"] = dict(sorted(coverage["unknown_reasons"].items()))
        coverage["known_subtotal"] = summary[known_key]
        coverage["total"] = summary[metric + "_total"]

    _require(sum(classes.values()) == summary["selected_model_calls"], "model_class_count_mismatch")
    _require(summary["readable_projection_truncated_runs"] == 0, "truncated_projection")
    return {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_index": index_path.name,
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "summary": summary,
        "capture_call_source_coverage": call_coverage,
        "report_source_coverage": report_coverage,
        "legacy_field_note": "The *_known_129_reports keys are compatibility names; report_source_coverage gives actual measured run counts.",
        "model_classes": dict(sorted(classes.items())),
        "accounting_known_129_reports": dict(sorted(accounting.items())),
        "route_counts_known_129_reports": dict(sorted(routes.items())),
        "settings_counts_known_129_reports": dict(sorted(settings.items())),
        "raw_incomplete_reason_run_counts": dict(sorted(reasons.items())),
        "source_versions": dict(sorted(source_versions.items())),
        "positive_tokens_without_full_text_runs": positive,
        "missing_fields": missing_fields,
        "scope": "Latest per-run native readbacks, exact selected execution calls; source versions mixed and reports pre-final-score. Missing, null, invalid, negative, boolean, or conflicting report counts remain unknown. This does not establish 80/1 eligibility.",
    }


def write_aggregate(result: dict[str, Any], output_dir: Path) -> Path:
    """Write a new timestamped aggregate without replacing saved evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "miner_native_aggregate_"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    fd = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as output_file:
        json.dump(result, output_file, sort_keys=True, indent=2)
        output_file.write("\n")
    _require(json.loads(output_path.read_text()) == result, "output_readback_mismatch")
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="saved coverage index JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for a new aggregate JSON")
    args = parser.parse_args(argv)
    try:
        index_path = args.index.resolve()
        result = aggregate_index(index_path, index_path.parent)
        output_path = write_aggregate(result, args.output_dir)
    except Exception as exc:
        # Do not print report content or exception values from private artifacts.
        parser.exit(1, f"aggregate failed: {type(exc).__name__}\n")
    print(json.dumps({
        "path": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "summary": result["summary"],
        "model_classes": result["model_classes"],
        "positive_runs": len(result["positive_tokens_without_full_text_runs"]),
        "source_versions": result["source_versions"],
        "missing_fields_count": len(result["missing_fields"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
