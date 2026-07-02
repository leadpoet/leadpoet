"""Tests for the post-merge shadow monitor (audit §9.2 / follow-up 4.2).

All side-effect channels are faked (DB reader, S3 report store, docker runner,
scorer), so these cover:
  * discovery of candidate-derived merges without a completed shadow window;
  * daily window math (live-minus-shadow on identical ICPs, symmetric
    exclusion of shadow runtime errors and hash-drifted ICPs);
  * alert thresholds, including sd-aware cases around §5.1's noise floor;
  * the read-only guarantee (zero lab-state writes; S3 reports only) and the
    env-hygiene hard-fail when a live-mutation flag is set;
  * window state round-trip through the (fake) S3 doc across monitor restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Any, Mapping

import pytest

import gateway.research_lab.shadow_monitor as shadow_monitor
from gateway.research_lab.shadow_monitor import (
    ShadowMonitorDeps,
    ShadowMonitorEnvHygieneError,
    ShadowMonitorSettings,
    ShadowWindowSetupError,
    discover_unshadowed_merges,
    evaluate_window_alerts,
    ensure_shadow_process_read_only,
    new_window_state,
    resolve_shadow_artifact,
    run_shadow_day,
    run_window,
    shadow_window_prefix,
    stamped_read_only_doc,
    watch_once,
    window_report_uri,
    window_state_uri,
)
from research_lab.canonical import sha256_json
from research_lab.eval import PrivateModelArtifactManifest, PrivateModelRuntimeError


NEW_HASH = "sha256:" + "1" * 64
OLD_HASH = "sha256:" + "2" * 64
NEW_VERSION_ID = "private_model_version:sha256:" + "a" * 64
OLD_VERSION_ID = "private_model_version:sha256:" + "b" * 64
NEW_MANIFEST_URI = "s3://models/research-lab/sourcing-model/versions/new.json"
OLD_MANIFEST_URI = "s3://models/research-lab/sourcing-model/current.json"

CLEAN_ENV: dict[str, str] = {}


def _manifest_doc(model_artifact_hash: str, manifest_uri: str) -> dict[str, Any]:
    doc = {
        "model_artifact_hash": model_artifact_hash,
        "git_commit_sha": "c" * 40,
        "image_digest": (
            "123.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "d" * 64
        ),
        "config_hash": "sha256:" + "e" * 64,
        "component_registry_version": "1.0",
        "scoring_adapter_version": "1.0",
        "manifest_uri": manifest_uri,
        "signature_ref": "kms:sig",
        "build_id": "",
    }
    doc["manifest_hash"] = sha256_json(doc)
    return doc


def _icp(icp_id: str) -> dict[str, Any]:
    return {"icp_id": icp_id, "industry": "software", "intent_signals": ["hiring sdrs"]}


def _icp_ref(set_id: int, icp_id: str) -> str:
    return f"qualification_private_icp_sets:{set_id}:{icp_id}"


def _summary(set_id: int, icp_id: str, score: float, *, icp_hash: str | None = None) -> dict[str, Any]:
    return {
        "icp_ref": _icp_ref(set_id, icp_id),
        "icp_hash": icp_hash if icp_hash is not None else sha256_json({"icp": _icp(icp_id)}),
        "score": score,
        "company_count": 3,
    }


def _benchmark_row(
    benchmark_date: str,
    *,
    bundle_id: str,
    artifact_hash: str = NEW_HASH,
    summaries: list[dict[str, Any]],
    quality: str = "passed",
    epoch: int = 12,
    created_at: str | None = None,
) -> dict[str, Any]:
    scores = [item["score"] for item in summaries]
    return {
        "benchmark_bundle_id": bundle_id,
        "benchmark_date": benchmark_date,
        "private_model_artifact_hash": artifact_hash,
        "private_model_manifest_hash": "sha256:" + "f" * 64,
        "rolling_window_hash": "sha256:" + "9" * 64,
        "benchmark_quality": quality,
        "evaluation_epoch": epoch,
        "current_benchmark_status": "completed",
        "aggregate_score": sum(scores) / len(scores) if scores else 0.0,
        "score_summary_doc": {"per_icp_summaries": summaries},
        "created_at": created_at or f"{benchmark_date}T20:00:00+00:00",
    }


def _merge_event(created_at: str = "2026-07-01T12:00:00+00:00") -> dict[str, Any]:
    return {
        "promotion_event_id": "pe-merge-1",
        "candidate_id": "cand-1",
        "event_type": "active_version_created",
        "promotion_status": "merged",
        "private_model_version_id": NEW_VERSION_ID,
        "active_parent_artifact_hash": OLD_HASH,
        "created_at": created_at,
        "event_doc": {"candidate_kind": "image_build"},
    }


def _version_rows() -> list[dict[str, Any]]:
    return [
        {
            "private_model_version_id": NEW_VERSION_ID,
            "model_artifact_hash": NEW_HASH,
            "private_model_manifest_hash": "sha256:" + "f" * 64,
            "private_model_manifest_uri": NEW_MANIFEST_URI,
            "current_version_status": "active",
            "current_status_at": "2026-07-01T12:00:00+00:00",
        },
        {
            "private_model_version_id": OLD_VERSION_ID,
            "model_artifact_hash": OLD_HASH,
            "private_model_manifest_hash": "sha256:" + "0" * 64,
            "private_model_manifest_uri": OLD_MANIFEST_URI,
            "current_version_status": "superseded",
            "current_status_at": "2026-07-01T12:00:00+00:00",
        },
    ]


class FakeDB:
    """Read-only table fake honoring the store filter grammar the monitor uses."""

    def __init__(self, tables: Mapping[str, list[dict[str, Any]]]):
        self.tables = {name: [dict(row) for row in rows] for name, rows in tables.items()}
        self.calls: list[tuple[str, tuple]] = []

    async def select_many(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: tuple = (),
        order_by: tuple = (),
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.calls.append((table, tuple(filters)))
        rows = [dict(row) for row in self.tables.get(table, [])]
        for spec in filters:
            if len(spec) == 2:
                name, value = spec
                rows = [row for row in rows if str(row.get(name)) == str(value)]
            else:
                name, operator, value = spec
                if operator == "in":
                    rows = [row for row in rows if row.get(name) in value]
                elif operator == "gte":
                    rows = [row for row in rows if str(row.get(name)) >= str(value)]
                else:
                    raise AssertionError(f"unsupported fake filter operator: {operator}")
        for name, desc in reversed(tuple(order_by)):
            rows.sort(key=lambda row: str(row.get(name) or ""), reverse=bool(desc))
        return rows[:limit]


class FakeReportStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.puts: list[str] = []

    def get_json(self, uri: str) -> dict[str, Any] | None:
        doc = self.docs.get(uri)
        return json.loads(json.dumps(doc)) if doc is not None else None

    def put_json(self, uri: str, doc: Mapping[str, Any]) -> None:
        # JSON round-trip: everything persisted must be serializable.
        self.docs[uri] = json.loads(json.dumps(doc, default=str))
        self.puts.append(uri)


@dataclass
class FakeRunner:
    scores_by_icp_id: Mapping[str, float]
    error_icp_ids: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def __call__(self, icp: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
        icp_id = str(icp.get("icp_id"))
        self.calls.append(icp_id)
        assert context.get("mode") == "post_merge_shadow"
        if icp_id in self.error_icp_ids:
            raise PrivateModelRuntimeError(f"provider 502 for {icp_id}")
        return [{"planted_score": float(self.scores_by_icp_id[icp_id])}]


class FakeScorer:
    async def score_with_breakdowns(
        self,
        companies: list[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        assert is_reference_model is True
        return [{"final_score": company["planted_score"]} for company in companies]


@dataclass
class Clock:
    current: datetime

    def now(self) -> datetime:
        return self.current


def _settings(**overrides: Any) -> ShadowMonitorSettings:
    values: dict[str, Any] = {
        "enabled": True,
        "window_days": 2,
        "concurrency": 2,
        "grace_days": 3,
        "report_uri_prefix": "s3://reports/research-lab",
    }
    values.update(overrides)
    return ShadowMonitorSettings(**values)


def _deps(
    tables: Mapping[str, list[dict[str, Any]]],
    *,
    store: FakeReportStore | None = None,
    runner: FakeRunner | None = None,
    clock: Clock | None = None,
) -> tuple[ShadowMonitorDeps, FakeDB, FakeReportStore, FakeRunner, Clock]:
    db = FakeDB(tables)
    store = store or FakeReportStore()
    runner = runner if runner is not None else FakeRunner(scores_by_icp_id={})
    clock = clock or Clock(datetime(2026, 7, 2, 6, 0, tzinfo=timezone.utc))
    manifests = {
        NEW_MANIFEST_URI: _manifest_doc(NEW_HASH, NEW_MANIFEST_URI),
        OLD_MANIFEST_URI: _manifest_doc(OLD_HASH, OLD_MANIFEST_URI),
    }
    built_artifacts: list[str] = []

    def _runner_factory(artifact: PrivateModelArtifactManifest):
        built_artifacts.append(artifact.model_artifact_hash)
        return runner

    deps = ShadowMonitorDeps(
        select_many=db.select_many,
        load_manifest=lambda uri: manifests[uri],
        runner_factory=_runner_factory,
        scorer_factory=FakeScorer,
        report_store=store,
        now=clock.now,
    )
    deps.runner_factory.built_artifacts = built_artifacts  # type: ignore[attr-defined]
    return deps, db, store, runner, clock


def _base_tables(*, benchmarks: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    return {
        "research_lab_candidate_promotion_events": [
            _merge_event(),
            # A non-merge event: discovery must never pick it up.
            {
                "promotion_event_id": "pe-passed-1",
                "candidate_id": "cand-1",
                "event_type": "promotion_passed",
                "promotion_status": "passed",
                "private_model_version_id": None,
                "active_parent_artifact_hash": OLD_HASH,
                "created_at": "2026-07-01T11:59:00+00:00",
                "event_doc": {},
            },
        ],
        "research_lab_private_model_version_current": _version_rows(),
        "research_lab_private_model_benchmark_current": list(benchmarks or []),
        "qualification_private_icp_sets": [
            {"set_id": 401, "icps": [_icp("icp_a"), _icp("icp_b")], "icp_set_hash": "sha256:" + "3" * 64},
            {"set_id": 402, "icps": [_icp("icp_c"), _icp("icp_d")], "icp_set_hash": "sha256:" + "4" * 64},
        ],
    }


def _state_with_diffs(diffs: list[float | None], *, window_days: int = 5) -> dict[str, Any]:
    days = [
        {
            "benchmark_date": f"2026-07-{index + 1:02d}",
            "benchmark_bundle_id": f"private_benchmark:{index}",
            "shadow_live_diff_points": diff,
            "compared_icp_count": 0 if diff is None else 2,
        }
        for index, diff in enumerate(diffs)
    ]
    return {
        "active_version_id": NEW_VERSION_ID,
        "window_days": window_days,
        "days": days,
        "alerts": [],
    }


# ---------------------------------------------------------------------------
# Settings + env hygiene
# ---------------------------------------------------------------------------


def test_settings_default_off_and_defaults() -> None:
    settings = ShadowMonitorSettings.from_env({})
    assert settings.enabled is False
    assert settings.window_days == 5
    assert settings.alert_threshold_points == 2.0
    assert settings.early_alert_day_points == 1.0
    assert settings.early_alert_consecutive_days == 2

    tuned = ShadowMonitorSettings.from_env(
        {
            "RESEARCH_LAB_SHADOW_MONITOR_ENABLED": "true",
            "RESEARCH_LAB_SHADOW_WINDOW_DAYS": "7",
            "RESEARCH_LAB_SHADOW_ALERT_THRESHOLD_POINTS": "3.5",
        }
    )
    assert tuned.enabled is True
    assert tuned.window_days == 7
    assert tuned.alert_threshold_points == 3.5


def test_env_hygiene_clean_env_passes() -> None:
    flags = ensure_shadow_process_read_only(CLEAN_ENV)
    assert all(value is False for value in flags.values())


def test_env_hygiene_live_mutation_flag_hard_fails() -> None:
    with pytest.raises(ShadowMonitorEnvHygieneError) as excinfo:
        ensure_shadow_process_read_only({"RESEARCH_LAB_PAID_LOOPS_ENABLED": "true"})
    assert "RESEARCH_LAB_PAID_LOOPS_ENABLED" in str(excinfo.value)


def test_stamped_doc_honors_read_only_guard() -> None:
    doc = stamped_read_only_doc({"artifact_type": "x"}, CLEAN_ENV)
    assert doc["shadow_only"] is True
    assert doc["read_only"] is True
    assert doc["submission_allowed"] is False
    assert doc["read_only_evidence"]["passed"] is True

    with pytest.raises(ShadowMonitorEnvHygieneError):
        stamped_read_only_doc({"artifact_type": "x"}, {"RESEARCH_LAB_PAID_LOOPS_ENABLED": "1"})


# ---------------------------------------------------------------------------
# Discovery + previous-champion resolution
# ---------------------------------------------------------------------------


async def test_discovery_finds_unshadowed_merges() -> None:
    deps, db, store, _runner, _clock = _deps(_base_tables())
    pending = await discover_unshadowed_merges(deps, _settings())
    assert len(pending) == 1
    entry = pending[0]
    assert entry["event"]["promotion_event_id"] == "pe-merge-1"
    assert entry["state"] is None
    assert entry["prefix"].startswith("s3://reports/research-lab/shadow-windows/")

    # A completed window state suppresses re-discovery.
    store.docs[window_state_uri(entry["prefix"])] = {"status": "completed"}
    assert await discover_unshadowed_merges(deps, _settings()) == []


async def test_discovery_skips_missing_version_row_and_versionless_events() -> None:
    tables = _base_tables()
    tables["research_lab_private_model_version_current"] = []
    deps, _db, _store, _runner, _clock = _deps(tables)
    assert await discover_unshadowed_merges(deps, _settings()) == []


def test_shadow_window_prefix_derived_from_manifest_uri() -> None:
    prefix = shadow_window_prefix(
        active_version_id=NEW_VERSION_ID,
        live_manifest_uri=NEW_MANIFEST_URI,
        settings=_settings(report_uri_prefix=""),
    )
    assert prefix == (
        "s3://models/research-lab/sourcing-model/versions/shadow-windows/" + "a" * 16
    )
    with pytest.raises(ShadowWindowSetupError):
        shadow_window_prefix(
            active_version_id=NEW_VERSION_ID,
            live_manifest_uri="/local/path.json",
            settings=_settings(report_uri_prefix=""),
        )


async def test_resolve_shadow_artifact_verifies_lineage_hash() -> None:
    deps, _db, _store, _runner, _clock = _deps(_base_tables())
    artifact = await resolve_shadow_artifact(deps, previous_artifact_hash=OLD_HASH)
    assert artifact.model_artifact_hash == OLD_HASH

    # A drifted manifest (mutable-manifest hazard) must refuse to shadow.
    tables = _base_tables()
    tables["research_lab_private_model_version_current"][1]["model_artifact_hash"] = (
        "sha256:" + "7" * 64
    )
    drifted, _db2, _store2, _runner2, _clock2 = _deps(tables)
    with pytest.raises(ShadowWindowSetupError, match="no longer matches"):
        await resolve_shadow_artifact(drifted, previous_artifact_hash="sha256:" + "7" * 64)


# ---------------------------------------------------------------------------
# Daily window math
# ---------------------------------------------------------------------------


async def test_run_shadow_day_diff_is_live_minus_shadow() -> None:
    benchmark = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[_summary(401, "icp_a", 10.0), _summary(401, "icp_b", 12.0)],
    )
    runner = FakeRunner(scores_by_icp_id={"icp_a": 11.5, "icp_b": 13.5})
    deps, _db, _store, _runner, _clock = _deps(_base_tables(), runner=runner)
    state = new_window_state(
        event=_merge_event(), version_row=_version_rows()[0], settings=_settings(), env=CLEAN_ENV
    )
    artifact = await resolve_shadow_artifact(deps, previous_artifact_hash=OLD_HASH)

    entry, report = await run_shadow_day(
        deps, _settings(), state=state, benchmark_row=benchmark, shadow_artifact=artifact, env=CLEAN_ENV
    )
    assert entry["shadow_live_diff_points"] == pytest.approx(-1.5)
    assert entry["compared_icp_count"] == 2
    assert report["live_aggregate_score"] == pytest.approx(11.0)
    assert report["shadow_aggregate_score"] == pytest.approx(12.5)
    assert report["read_only_evidence"]["passed"] is True
    # Reused per-key diff API: shadow-minus-live milli-points over indexed ICPs.
    per_icp = report["per_icp_diff_millipoints"]
    assert per_icp["uid_count"] == 2
    assert per_icp["changed_uid_count"] == 2
    assert per_icp["max_abs_delta_u16"] == 1500
    legend = report["icp_index_legend"]
    assert set(legend.values()) == {_icp_ref(401, "icp_a"), _icp_ref(401, "icp_b")}
    assert sorted(runner.calls) == ["icp_a", "icp_b"]


async def test_run_shadow_day_excludes_runtime_errors_symmetrically() -> None:
    benchmark = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[
            _summary(401, "icp_a", 10.0),
            _summary(401, "icp_b", 12.0),
            _summary(402, "icp_c", 14.0),
        ],
    )
    runner = FakeRunner(
        scores_by_icp_id={"icp_a": 11.0, "icp_c": 16.0},
        error_icp_ids={"icp_b"},
    )
    deps, _db, _store, _runner, _clock = _deps(_base_tables(), runner=runner)
    state = new_window_state(
        event=_merge_event(), version_row=_version_rows()[0], settings=_settings(), env=CLEAN_ENV
    )
    artifact = await resolve_shadow_artifact(deps, previous_artifact_hash=OLD_HASH)

    entry, report = await run_shadow_day(
        deps, _settings(), state=state, benchmark_row=benchmark, shadow_artifact=artifact, env=CLEAN_ENV
    )
    # icp_b dropped from BOTH sides: live (10+14)/2=12, shadow (11+16)/2=13.5.
    assert entry["compared_icp_count"] == 2
    assert entry["shadow_live_diff_points"] == pytest.approx(-1.5)
    excluded = {item["icp_ref"]: item["reason"] for item in report["excluded_icps"]}
    assert excluded == {_icp_ref(401, "icp_b"): "shadow_runtime_error"}


async def test_run_shadow_day_excludes_hash_drifted_icps() -> None:
    benchmark = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[
            _summary(401, "icp_a", 10.0),
            _summary(401, "icp_b", 12.0, icp_hash="sha256:" + "8" * 64),
        ],
    )
    runner = FakeRunner(scores_by_icp_id={"icp_a": 10.5, "icp_b": 99.0})
    deps, _db, _store, _runner, _clock = _deps(_base_tables(), runner=runner)
    state = new_window_state(
        event=_merge_event(), version_row=_version_rows()[0], settings=_settings(), env=CLEAN_ENV
    )
    artifact = await resolve_shadow_artifact(deps, previous_artifact_hash=OLD_HASH)

    entry, report = await run_shadow_day(
        deps, _settings(), state=state, benchmark_row=benchmark, shadow_artifact=artifact, env=CLEAN_ENV
    )
    assert entry["compared_icp_count"] == 1
    assert entry["shadow_live_diff_points"] == pytest.approx(-0.5)
    reasons = {item["reason"] for item in report["excluded_icps"]}
    assert reasons == {"icp_hash_mismatch"}
    assert runner.calls == ["icp_a"]  # the drifted ICP is never executed


# ---------------------------------------------------------------------------
# Alert semantics (sd-aware, §5.1)
# ---------------------------------------------------------------------------


def test_no_alert_within_same_model_noise() -> None:
    # Cumulative -1.9 == 1 same-model sd: inside the default 2.0 bar.
    state = _state_with_diffs([-0.9, -0.4, -0.6])
    assert evaluate_window_alerts(state, _settings(window_days=5)) == []


def test_cumulative_regression_alert_beyond_threshold() -> None:
    state = _state_with_diffs([-0.5, -0.5, -0.5, -0.5, -0.5])
    alerts = evaluate_window_alerts(state, _settings(window_days=5))
    assert [alert["alert_type"] for alert in alerts] == ["cumulative_regression"]
    alert = alerts[0]
    assert alert["cumulative_shadow_live_diff_points"] == pytest.approx(-2.5)
    assert alert["threshold_points"] == 2.0
    assert alert["noise_context"]["same_model_benchmark_sd_points"] == 1.9
    assert alert["noise_context"]["day_diff_se_points"] == pytest.approx(2.687, abs=1e-3)
    assert alert["cumulative_diff_over_se"] > 0


def test_cumulative_threshold_is_strictly_more_negative() -> None:
    settings = _settings(window_days=5)
    crossed = evaluate_window_alerts(_state_with_diffs([-1.5, 0.4, -0.95]), settings)
    assert [alert["alert_type"] for alert in crossed] == ["cumulative_regression"]
    # Exactly -2.0 is NOT "more negative than" the default threshold; a single
    # -2.0 day also cannot trip the 2-consecutive-day early alert.
    assert evaluate_window_alerts(_state_with_diffs([-2.0, 0.0]), settings) == []


def test_early_alert_on_two_consecutive_negative_days() -> None:
    # Cumulative stays inside a raised bar; consecutive day dips still alert.
    settings = _settings(window_days=5, alert_threshold_points=5.0)
    alerts = evaluate_window_alerts(_state_with_diffs([-1.2, -1.1]), settings)
    assert [alert["alert_type"] for alert in alerts] == ["consecutive_negative_days"]
    assert alerts[0]["consecutive_dates"] == ["2026-07-01", "2026-07-02"]
    assert alerts[0]["day_points_threshold"] == 1.0

    # Non-consecutive dips do not.
    assert evaluate_window_alerts(_state_with_diffs([-1.2, 0.3, -1.1]), settings) == []
    # -1.0 exactly is not beyond the 1.0 day bar (strict).
    assert evaluate_window_alerts(_state_with_diffs([-1.5, -1.0]), settings) == []


def test_days_without_comparable_icps_are_ignored_by_alerts() -> None:
    state = _state_with_diffs([None, None])
    assert evaluate_window_alerts(state, _settings(window_days=5)) == []


def test_alert_logged_once_with_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    state = _state_with_diffs([-1.5, -1.5])
    with caplog.at_level(logging.WARNING, logger=shadow_monitor.__name__):
        first = shadow_monitor._log_new_alerts(state, _settings(window_days=5))
        second = shadow_monitor._log_new_alerts(state, _settings(window_days=5))
    assert {alert["alert_type"] for alert in first} == {
        "cumulative_regression",
        "consecutive_negative_days",
    }
    assert second == []
    lines = [record.message for record in caplog.records]
    assert len(lines) == 2
    assert all(line.startswith("research_lab_shadow_regression_alert:") for line in lines)
    assert any("cumulative_points=-3.0000" in line for line in lines)
    assert any("day_diff_se_points=2.687" in line for line in lines)


# ---------------------------------------------------------------------------
# Window orchestration + state round-trip
# ---------------------------------------------------------------------------


async def test_full_window_flow_with_state_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings()  # window_days=2
    day1 = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[_summary(401, "icp_a", 10.0), _summary(401, "icp_b", 12.0)],
    )
    # Same-day benchmark of the OLD champion (recorded pre-merge): must be ignored.
    old_champion_row = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:old",
        artifact_hash=OLD_HASH,
        summaries=[_summary(401, "icp_a", 9.0)],
    )
    runner = FakeRunner(
        scores_by_icp_id={"icp_a": 11.5, "icp_b": 13.5, "icp_c": 11.2, "icp_d": 11.8}
    )
    store = FakeReportStore()
    tables = _base_tables(benchmarks=[day1, old_champion_row])
    deps, db, store, runner, clock = _deps(tables, store=store, runner=runner)

    summary = await watch_once(deps, settings, CLEAN_ENV)
    assert summary["discovered_windows"] == 1
    assert summary["windows"][0]["status"] == "open"
    assert summary["windows"][0]["evaluated_days"] == 1
    prefix = shadow_window_prefix(
        active_version_id=NEW_VERSION_ID, live_manifest_uri=NEW_MANIFEST_URI, settings=settings
    )
    state = store.get_json(window_state_uri(prefix))
    assert state["status"] == "open"
    assert [day["benchmark_date"] for day in state["days"]] == ["2026-07-01"]
    assert state["days"][0]["shadow_live_diff_points"] == pytest.approx(-1.5)
    assert deps.runner_factory.built_artifacts == [OLD_HASH]  # type: ignore[attr-defined]

    # Day 2's champion benchmark lands; a FRESH monitor process resumes from S3.
    day2 = _benchmark_row(
        "2026-07-02",
        bundle_id="private_benchmark:day2",
        summaries=[_summary(402, "icp_c", 10.0), _summary(402, "icp_d", 11.0)],
    )
    tables2 = _base_tables(benchmarks=[day1, old_champion_row, day2])
    deps2, db2, store, runner, clock2 = _deps(tables2, store=store, runner=runner)
    with caplog.at_level(logging.WARNING, logger=shadow_monitor.__name__):
        summary2 = await watch_once(deps2, settings, CLEAN_ENV)

    assert summary2["windows"][0]["status"] == "completed"
    # Day 1 was NOT re-evaluated: 2 ICPs on tick 1, 2 more on tick 2.
    assert sorted(runner.calls) == ["icp_a", "icp_b", "icp_c", "icp_d"]

    final_state = store.get_json(window_state_uri(prefix))
    assert final_state["status"] == "completed"
    diffs = [day["shadow_live_diff_points"] for day in final_state["days"]]
    assert diffs == [pytest.approx(-1.5), pytest.approx(-1.0)]

    report = store.get_json(window_report_uri(prefix))
    assert report["cumulative_shadow_live_diff_points"] == pytest.approx(-2.5)
    assert report["observed_day_count"] == 2
    assert report["comparable_day_count"] == 2
    assert report["read_only_evidence"]["passed"] is True
    assert [alert["alert_type"] for alert in report["alerts"]] == ["cumulative_regression"]
    assert any(
        record.message.startswith("research_lab_shadow_regression_alert:")
        for record in caplog.records
    )
    # Completed windows drop out of discovery.
    assert await discover_unshadowed_merges(deps2, settings) == []

    # Day reports were written for both days under the window prefix.
    day_uris = [uri for uri in store.docs if "/day-" in uri]
    assert sorted(day_uris) == [
        f"{prefix}/day-2026-07-01.json",
        f"{prefix}/day-2026-07-02.json",
    ]


async def test_window_closes_at_deadline_without_benchmarks() -> None:
    settings = _settings()
    deps, _db, store, _runner, clock = _deps(_base_tables(benchmarks=[]))
    # merge 07-01, window 2 + grace 3 -> deadline 07-06; poll on 07-07.
    clock.current = datetime(2026, 7, 7, 6, 0, tzinfo=timezone.utc)
    summary = await watch_once(deps, settings, CLEAN_ENV)
    assert summary["windows"][0]["status"] == "completed"
    assert summary["windows"][0]["observed_day_count"] == 0
    prefix = shadow_window_prefix(
        active_version_id=NEW_VERSION_ID, live_manifest_uri=NEW_MANIFEST_URI, settings=settings
    )
    report = store.get_json(window_report_uri(prefix))
    assert report["cumulative_shadow_live_diff_points"] is None
    assert "observed_fewer_benchmark_days_than_window" in report["report_blockers"]


async def test_invalid_benchmark_rows_are_not_shadowed() -> None:
    bad_quality = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:bad",
        summaries=[_summary(401, "icp_a", 10.0)],
        quality="failed",
        epoch=0,
    )
    empty_summaries = _benchmark_row(
        "2026-07-02",
        bundle_id="private_benchmark:empty",
        summaries=[],
    )
    deps, _db, _store, runner, _clock = _deps(
        _base_tables(benchmarks=[bad_quality, empty_summaries])
    )
    summary = await watch_once(deps, _settings(), CLEAN_ENV)
    assert summary["windows"][0]["status"] == "open"
    assert summary["windows"][0]["evaluated_days"] == 0
    assert runner.calls == []


async def test_run_window_unknown_version_errors() -> None:
    deps, _db, _store, _runner, _clock = _deps(_base_tables())
    result = await run_window(
        deps, _settings(), active_version_id="private_model_version:sha256:" + "9" * 64, env=CLEAN_ENV
    )
    assert result["status"] == "error"
    assert result["error"] == "active_version_created_event_not_found"


async def test_run_window_processes_named_version() -> None:
    day1 = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[_summary(401, "icp_a", 10.0), _summary(401, "icp_b", 12.0)],
    )
    runner = FakeRunner(scores_by_icp_id={"icp_a": 9.0, "icp_b": 11.0})
    deps, _db, store, runner, _clock = _deps(_base_tables(benchmarks=[day1]), runner=runner)
    result = await run_window(deps, _settings(), active_version_id=NEW_VERSION_ID, env=CLEAN_ENV)
    assert result["status"] == "open"
    assert result["evaluated_days"] == 1
    prefix = shadow_window_prefix(
        active_version_id=NEW_VERSION_ID, live_manifest_uri=NEW_MANIFEST_URI, settings=_settings()
    )
    state = store.get_json(window_state_uri(prefix))
    # Positive diff (live better than shadow) recorded, no alerts.
    assert state["days"][0]["shadow_live_diff_points"] == pytest.approx(1.0)
    assert state["alerts"] == []


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


async def test_monitor_reads_only_known_tables_and_writes_only_s3(
    caplog: pytest.LogCaptureFixture,
) -> None:
    day1 = _benchmark_row(
        "2026-07-01",
        bundle_id="private_benchmark:day1",
        summaries=[_summary(401, "icp_a", 10.0), _summary(401, "icp_b", 12.0)],
    )
    runner = FakeRunner(scores_by_icp_id={"icp_a": 11.5, "icp_b": 13.5})
    deps, db, store, runner, _clock = _deps(_base_tables(benchmarks=[day1]), runner=runner)
    settings = _settings()

    await watch_once(deps, settings, CLEAN_ENV)

    read_tables = {table for table, _filters in db.calls}
    assert read_tables <= {
        "research_lab_candidate_promotion_events",
        "research_lab_private_model_version_current",
        "research_lab_private_model_benchmark_current",
        "qualification_private_icp_sets",
    }
    # Every write went to the injected S3 store, under the window prefix.
    prefix = shadow_window_prefix(
        active_version_id=NEW_VERSION_ID, live_manifest_uri=NEW_MANIFEST_URI, settings=settings
    )
    assert store.puts, "expected S3 state/report writes"
    assert all(uri.startswith(prefix + "/") for uri in store.puts)
    # Every persisted doc carries the read-only markers.
    for doc in store.docs.values():
        assert doc["shadow_only"] is True
        assert doc["read_only"] is True
        assert doc["submission_allowed"] is False
    # The module never imports lab-state writers (events/rows/bundles).
    for writer in (
        "insert_row",
        "append_event_with_seq",
        "create_candidate_promotion_event",
        "create_private_model_version",
        "create_private_model_version_event",
        "create_private_model_benchmark_bundle",
        "create_scoring_dispatch_event",
        "create_rolling_icp_window",
    ):
        assert not hasattr(shadow_monitor, writer)


async def test_watch_once_refuses_live_mutation_env() -> None:
    deps, _db, store, _runner, _clock = _deps(_base_tables())
    with pytest.raises(ShadowMonitorEnvHygieneError):
        await watch_once(deps, _settings(), {"RESEARCH_LAB_PAID_LOOPS_ENABLED": "true"})
    assert store.puts == []


# ---------------------------------------------------------------------------
# CLI gating
# ---------------------------------------------------------------------------


def test_main_disabled_by_default_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_SHADOW_MONITOR_ENABLED", raising=False)
    assert shadow_monitor.main(["--once"]) == 2


def test_main_env_hygiene_gate_returns_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_SHADOW_MONITOR_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_LAB_PAID_LOOPS_ENABLED", "true")
    assert shadow_monitor.main(["--watch"]) == 3
