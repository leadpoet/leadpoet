"""Tests for score-only promotion, the legacy confirmation-hold drain, §0-N6
scorer isolation, and the baseline any-worker lease confirm.

Score-only promotion (2026-07-02 design decision, commit 3aaee73c): automatic
promotion merges on the stored final score vs the stored daily baseline alone.
The §5.2-2 confirmation re-run is permanently disabled as a gate
(``promotion_confirmation_rerun_enabled()`` returns False unconditionally) and
provider/runtime health never vetoes champions.

Covers, with fake store rows (no live Supabase):
  * score-only promotion gate — first-pass winners merge on one measurement;
    recorded confirmation events (passing OR failing) and exhausted attempt
    budgets no longer gate the decision; already-promoted re-drive triggers
    reward/source finalization instead of a re-merge.
  * legacy worker runner — the confirmation runner machinery still drains
    HISTORICAL holds written before the score-only switch: hold discovery,
    started-claim leasing, measurement recording (side artifact, never the
    day's benchmark), symmetric ICP exclusion + unhealthy-measurement re-hold,
    closed-marker settling.
  * §0-N6 — benchmark scorer key isolation scope (env + module attrs restored),
    scorer-error non-fatality (marked unresolved + retryable), inline retry,
    and the scorer concurrency semaphore.
  * Baseline lease — post-write confirm: earliest unexpired open lease wins,
    closed/expired leases ignored, read failures confirm optimistically.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import gateway.research_lab.promotion as promotion
import gateway.research_lab.scoring_worker as sw
from gateway.research_lab.promotion import (
    ActivePrivateModel,
    CONFIRMATION_ATTEMPT_FAILED_REASON,
    CONFIRMATION_CLOSED_REASON,
    CONFIRMATION_HOLD_REASON,
    CONFIRMATION_REJECTED_REASON,
    CONFIRMATION_RESULT_REASON,
    CONFIRMATION_STARTED_REASON,
    ResearchLabPromotionController,
    confirmation_min_delta,
    load_confirmation_state,
    promotion_confirmation_rerun_enabled,
)


PARENT_HASH = "sha256:" + "a" * 64
CANDIDATE_HASH = "sha256:" + "c" * 64
WINDOW_HASH = "sha256:" + "3" * 64


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class FakeArtifact:
    def __init__(self, model_artifact_hash: str = PARENT_HASH):
        self.model_artifact_hash = model_artifact_hash
        self.manifest_hash = "sha256:" + "b" * 64
        self.manifest_uri = "s3://bucket/manifest.json"
        self.git_commit_sha = "d" * 40
        self.component_registry_version = "1.0"
        self.scoring_adapter_version = "1.0"
        self.image_digest = "sha256:" + "e" * 64

    def to_dict(self) -> dict[str, Any]:
        return {"model_artifact_hash": self.model_artifact_hash}


class ConfirmationFakeStore:
    """Promotion-event store with real filter semantics (incl. the
    event_doc->>reason JSON-path filter the hold discovery uses), so the
    event-sourced confirmation state machine is exercised end to end."""

    def __init__(self) -> None:
        self.promotion_events: list[dict[str, Any]] = []
        self.select_many_results: dict[str, Any] = {}
        self.select_one_results: dict[str, Any] = {}
        self.json_filter_supported = True
        self._seq = 0
        # Fresh, strictly-increasing timestamps: the claim-confirm and lease
        # logic age-filter events, so seeded rows must read as current.
        self._base = datetime.now(timezone.utc)

    def seed_promotion_event(self, **kwargs: Any) -> dict[str, Any]:
        self._seq += 1
        row = {
            "promotion_event_id": f"pe-{self._seq}",
            "created_at": (self._base + timedelta(milliseconds=self._seq)).isoformat(),
            "event_doc": {},
            **kwargs,
        }
        self.promotion_events.append(row)
        return row

    async def create_candidate_promotion_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.seed_promotion_event(**kwargs)

    def promotion_writes(self, reason: str | None = None) -> list[dict[str, Any]]:
        rows = self.promotion_events
        if reason is None:
            return list(rows)
        return [r for r in rows if (r.get("event_doc") or {}).get("reason") == reason]

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [r for r in self.promotion_events if r.get("event_type") == event_type]

    async def select_many(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        filters = tuple(kwargs.get("filters") or ())
        if table != "research_lab_candidate_promotion_events":
            result = self.select_many_results.get(table, [])
            if isinstance(result, Exception):
                raise result
            if callable(result):
                result = result(kwargs)
            return list(result)
        rows = list(self.promotion_events)
        for spec in filters:
            if len(spec) == 2:
                field, value = spec
                if field == "event_doc->>reason":
                    if not self.json_filter_supported:
                        raise RuntimeError("json path filters unsupported")
                    rows = [
                        r
                        for r in rows
                        if str((r.get("event_doc") or {}).get("reason") or "") == str(value)
                    ]
                else:
                    rows = [r for r in rows if str(r.get(field) or "") == str(value)]
            else:
                field, operator, value = spec
                if operator == "in":
                    allowed = {str(item) for item in value}
                    rows = [r for r in rows if str(r.get(field) or "") in allowed]
        # Newest-first mirrors order_by=(("created_at", True),).
        rows = sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return [dict(r) for r in rows[: int(kwargs.get("limit") or 100)]]

    async def select_one(self, table: str, **kwargs: Any) -> dict[str, Any] | None:
        result = self.select_one_results.get(table)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return result


def _controller_config(**overrides: Any) -> Any:
    values = {
        "auto_promotion_enabled": True,
        "auto_commit_enabled": False,
        "improvement_threshold_points": 1.0,
        "private_model_manifest_uri": "s3://bucket/bootstrap-manifest.json",
        "scoring_worker_max_claim_requeues": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _candidate() -> dict[str, Any]:
    return {
        "candidate_id": "cand-1",
        "parent_artifact_hash": PARENT_HASH,
        "candidate_kind": "image_build",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
        "candidate_model_manifest_doc": {"model_artifact_hash": CANDIDATE_HASH},
    }


def _approved_gate(**overrides: Any) -> dict[str, Any]:
    gate = {
        "decision": "private_holdout_approved",
        "private_holdout_evaluated": True,
        "baseline_aggregate_score": 10.0,
        "candidate_total_score": 12.5,
        "candidate_delta_vs_daily_baseline": 2.5,
        "baseline_benchmark_bundle_id": "bb-1",
    }
    gate.update(overrides)
    return gate


def _score_bundle(gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "private_holdout_gate": dict(gate or _approved_gate()),
        "aggregates": {},
        "icp_set_hash": WINDOW_HASH,
    }


def _healthy_benchmark_row() -> dict[str, Any]:
    return {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": {
            "per_icp_summaries": [{"icp_ref": "icp:a", "score": 10.0}],
            "aggregate_score": 10.0,
            "baseline_health": {"unresolved_provider_errors": 0, "gate_passed": True},
        },
    }


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> ConfirmationFakeStore:
    fake = ConfirmationFakeStore()
    for module in (promotion, sw):
        monkeypatch.setattr(module, "select_many", fake.select_many)
        monkeypatch.setattr(module, "select_one", fake.select_one)
        monkeypatch.setattr(module, "create_candidate_promotion_event", fake.create_candidate_promotion_event)
    fake.select_one_results["research_lab_private_model_benchmark_current"] = _healthy_benchmark_row()
    monkeypatch.delenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, raising=False)
    monkeypatch.delenv(promotion.CONFIRMATION_MIN_DELTA_ENV, raising=False)
    return fake


@pytest.fixture
def controller(store: ConfirmationFakeStore, monkeypatch: pytest.MonkeyPatch) -> ResearchLabPromotionController:
    artifact = FakeArtifact()

    async def _fake_load_active(config: Any, *, register_bootstrap: bool = False) -> ActivePrivateModel:
        return ActivePrivateModel(artifact=artifact, version_row={"private_model_version_id": "v-1"})

    monkeypatch.setattr(promotion, "load_active_private_model", _fake_load_active)

    merges: list[dict[str, Any]] = []

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        merges.append(kwargs)
        # Faithful to the real path: a merge records active_version_created.
        await promotion.create_candidate_promotion_event(
            candidate_id=str(kwargs["candidate"]["candidate_id"]),
            source_score_bundle_id=str(kwargs["score_bundle_row"]["score_bundle_id"]),
            event_type="active_version_created",
            promotion_status="merged",
            worker_ref="test-worker",
            event_doc={"candidate_kind": "image_build"},
        )
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)
    instance = ResearchLabPromotionController(_controller_config(), worker_ref="test-worker")
    instance._test_merges = merges  # type: ignore[attr-defined]
    return instance


async def _process(controller: ResearchLabPromotionController, **kwargs: Any) -> dict[str, Any]:
    return await controller.process_scored_candidate(
        candidate=_candidate(),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Score-only promotion gate (confirmation re-run permanently disabled)
# ---------------------------------------------------------------------------


async def test_first_pass_winner_merges_score_only(controller, store):
    """Score-only: a candidate whose stored score beats the stored daily
    baseline by the threshold merges on that single measurement — no hold."""
    result = await _process(controller)
    assert result["status"] == "merged"
    assert store.promotion_writes(CONFIRMATION_HOLD_REASON) == []
    assert len(store.events_of_type("promotion_passed")) == 1
    assert len(store.events_of_type("active_version_created")) == 1
    assert len(controller._test_merges) == 1


async def test_historical_hold_does_not_block_score_only_merge(controller, store):
    """A hold written before the score-only switch is ignored by the gate: the
    re-driven decision merges and no new hold is written."""
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={"reason": CONFIRMATION_HOLD_REASON, "first_pass_improvement_points": 2.5},
    )
    result = await _process(controller)
    assert result["status"] == "merged"
    assert len(store.promotion_writes(CONFIRMATION_HOLD_REASON)) == 1  # only the seed


async def test_env_flag_cannot_reenable_confirmation_gate(controller, store, monkeypatch):
    """The rerun env flag is dead: even set truthy, promotion stays score-only."""
    monkeypatch.setenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, "true")
    result = await _process(controller)
    assert result["status"] == "merged"
    assert store.promotion_writes(CONFIRMATION_HOLD_REASON) == []
    assert len(store.events_of_type("promotion_passed")) == 1


async def test_bypass_gates_param_is_inert(controller, store):
    """bypass_gates is accepted for replay-command compatibility but there is
    no confirmation gate left to bypass — nothing is reported bypassed."""
    result = await _process(controller, bypass_gates=frozenset({"confirmation_rerun"}))
    assert result["status"] == "merged"
    assert result.get("bypassed_gates", []) == []
    assert store.promotion_writes(CONFIRMATION_HOLD_REASON) == []


async def test_recorded_passing_confirmation_is_not_consulted(controller, store):
    """Historical confirmation results don't alter the score-only decision and
    no confirmation summary is attached to the merge."""
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={
            "reason": CONFIRMATION_RESULT_REASON,
            "confirmation": {
                "confirmation_delta": 1.4,
                "rolling_window_hash": WINDOW_HASH,
                "window_match": True,
            },
        },
    )
    result = await _process(controller)
    assert result["status"] == "merged"
    assert "confirmation" not in result
    passed = store.events_of_type("promotion_passed")
    assert len(passed) == 1
    assert "confirmation" not in passed[0]["event_doc"]


async def test_recorded_failing_confirmation_no_longer_blocks_merge(controller, store):
    """Score-only design decision: a historical FAILING confirmation
    measurement does not reject the candidate — the stored score alone
    decides. (Noise control now rests entirely on the daily baseline.)"""
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={
            "reason": CONFIRMATION_RESULT_REASON,
            "confirmation": {"confirmation_delta": 0.3, "window_match": True},
        },
    )
    result = await _process(controller)
    assert result["status"] == "merged"
    assert store.promotion_writes(CONFIRMATION_REJECTED_REASON) == []
    assert len(controller._test_merges) == 1


def test_confirmation_min_delta_falls_back_to_threshold(monkeypatch):
    # Still used by the legacy worker runner when draining historical holds.
    monkeypatch.delenv(promotion.CONFIRMATION_MIN_DELTA_ENV, raising=False)
    assert confirmation_min_delta(1.0) == pytest.approx(1.0)
    monkeypatch.setenv(promotion.CONFIRMATION_MIN_DELTA_ENV, "not-a-number")
    assert confirmation_min_delta(1.5) == pytest.approx(1.5)
    monkeypatch.setenv(promotion.CONFIRMATION_MIN_DELTA_ENV, "2.0")
    assert confirmation_min_delta(1.0) == pytest.approx(2.0)


def test_confirmation_rerun_disabled_unconditionally(monkeypatch):
    """The gate is retired by design, not by config: no env value re-arms it."""
    monkeypatch.delenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, raising=False)
    assert promotion_confirmation_rerun_enabled() is False
    monkeypatch.setenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, "true")
    assert promotion_confirmation_rerun_enabled() is False
    monkeypatch.setenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, "false")
    assert promotion_confirmation_rerun_enabled() is False


async def test_exhausted_confirmation_attempts_do_not_gate(controller, store):
    """Historical attempt-budget bookkeeping no longer rejects: score-only
    merges regardless of prior confirmation attempts."""
    for attempt in (1, 2, 3):
        store.seed_promotion_event(
            candidate_id="cand-1",
            source_score_bundle_id="sb-1",
            event_type="promotion_checked",
            promotion_status="checked",
            event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": attempt},
        )
    result = await _process(controller)
    assert result["status"] == "merged"
    assert store.promotion_writes(CONFIRMATION_REJECTED_REASON) == []


async def test_already_promoted_redrive_finalizes_side_effects(controller, store):
    """A re-drive of a merged candidate never re-merges; it re-attempts the
    merge side effects (private source push, champion reward) that may have
    been dropped — with auto-commit off and no resolvable UID that surfaces as
    a pending-uid reward event, not a second merge."""
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="active_version_created",
        promotion_status="merged",
    )
    before = len(store.promotion_events)
    result = await _process(controller)
    assert result["status"] == "already_promoted"
    assert result["private_source_status"]["status"] == "skipped_auto_commit_disabled"
    assert result["champion_reward_status"] == "uid_resolution_pending"
    assert controller._test_merges == []
    new_events = store.promotion_events[before:]
    assert [e["event_type"] for e in new_events] == ["champion_reward_pending_uid"]


async def test_legacy_bundle_without_holdout_gate_merges_unconfirmed(controller, store):
    result = await controller.process_scored_candidate(
        candidate=_candidate(),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle={"aggregates": {"mean_delta": 1.75}, "icp_set_hash": WINDOW_HASH},
    )
    # Legacy paired-metric bundles have no stored-baseline machinery to
    # confirm against; their path is unchanged.
    assert result["status"] == "merged"
    assert store.promotion_writes(CONFIRMATION_HOLD_REASON) == []


async def test_load_confirmation_state_derivation(store):
    def seed(reason: str, **extra: Any) -> dict[str, Any]:
        return store.seed_promotion_event(
            candidate_id="cand-1",
            source_score_bundle_id="sb-1",
            event_type="promotion_checked",
            promotion_status="checked",
            event_doc={"reason": reason, **extra},
        )

    seed(CONFIRMATION_HOLD_REASON)
    seed(CONFIRMATION_STARTED_REASON, attempt=1)
    seed(CONFIRMATION_ATTEMPT_FAILED_REASON, attempt=1)
    seed(CONFIRMATION_STARTED_REASON, attempt=2)
    recorded = seed(CONFIRMATION_RESULT_REASON, confirmation={"confirmation_delta": 1.2})
    # Unrelated checked event (no confirmation reason) must be ignored.
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={"reason": "held_baseline_health_gate_failed"},
    )
    # A different bundle's events must not bleed in.
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-OTHER",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": 9},
    )
    state = await load_confirmation_state(candidate_id="cand-1", score_bundle_id="sb-1")
    assert state["latest_reason"] == CONFIRMATION_RESULT_REASON
    assert state["result_event"]["promotion_event_id"] == recorded["promotion_event_id"]
    assert state["attempts"] == 2
    assert len(state["attempt_failed_events"]) == 1
    assert state["held_event"] is not None


# ---------------------------------------------------------------------------
# §5.2-2 worker runner
# ---------------------------------------------------------------------------


def _worker(config: Any | None = None) -> Any:
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.config = config or SimpleNamespace(
        improvement_threshold_points=1.0,
        auto_commit_enabled=False,
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        scoring_worker_max_claim_requeues=3,
        lab_champion_eval_days=1,
        lab_champion_icps_per_day=2,
        scoring_worker_allow_partial_icp_window=False,
        scoring_worker_model_timeout_seconds=900,
        private_baseline_concurrency=2,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=1,
        benchmark_exa_api_key="",
        benchmark_exa_max_rps=0.0,
        private_model_docker_global_proxy_enabled=False,
        auto_promotion_enabled=True,
    )
    worker.worker_ref = "worker-test"
    worker.proxy_ref_hash = None
    return worker


def _seed_hold(store: ConfirmationFakeStore, **extra_doc: Any) -> dict[str, Any]:
    return store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        rolling_window_hash=WINDOW_HASH,
        improvement_points=2.5,
        threshold_points=1.0,
        event_doc={
            "reason": CONFIRMATION_HOLD_REASON,
            "first_pass_improvement_points": 2.5,
            "baseline_benchmark_bundle_id": "bb-1",
            **extra_doc,
        },
    )


def _seed_candidate_rows(store: ConfirmationFakeStore) -> None:
    store.select_one_results["research_lab_candidate_evaluation_current"] = _candidate()
    store.select_one_results["research_evaluation_score_bundle_current"] = {
        "score_bundle_id": "sb-1",
        "score_bundle_doc": _score_bundle(),
    }


def _fake_window(window_hash: str = WINDOW_HASH) -> Any:
    items = tuple(
        {
            "icp": {"industry": "saas"},
            "icp_ref": f"icp:{tag}",
            "icp_hash": f"hash-{tag}",
            "set_id": 1,
            "day_index": 1,
            "day_rank": rank,
        }
        for rank, tag in enumerate(("a", "b"), start=1)
    )
    return SimpleNamespace(
        window_hash=window_hash,
        benchmark_id=f"research_lab:rolling_icp_window:{window_hash}",
        split_ref="split",
        public_doc={"rolling_window_hash": window_hash, "required_days": 1, "icps_per_day": 2, "sets": []},
        benchmark_items=items,
        item_refs=tuple(item["icp_ref"] for item in items),
        set_ids=(1,),
    )


def _side_summaries(scores: Mapping[str, float], *, unresolved: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    return [
        {
            "icp_ref": ref,
            "score": score,
            "company_count": 5,
            "_nonempty": True,
            "_runtime_error": "provider blew up" if ref in unresolved else "",
        }
        for ref, score in scores.items()
    ]


@pytest.fixture
def confirmation_env(store: ConfirmationFakeStore, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Worker wired for confirmation runs with fake window/artifact plumbing."""
    worker = _worker()
    window = _fake_window()
    artifact = FakeArtifact()

    async def _fake_fetch(**kwargs: Any) -> Any:
        return window

    async def _fake_create_window(w: Any) -> dict[str, Any]:
        return {"rolling_window_hash": w.window_hash}

    async def _fake_load_active(config: Any, *, register_bootstrap: bool = False) -> Any:
        return SimpleNamespace(artifact=artifact, version_row=None)

    monkeypatch.setattr(sw, "fetch_rolling_icp_window", _fake_fetch)
    monkeypatch.setattr(sw, "create_rolling_icp_window", _fake_create_window)
    monkeypatch.setattr(sw, "load_active_private_model", _fake_load_active)
    monkeypatch.setattr(
        sw,
        "PrivateModelArtifactManifest",
        SimpleNamespace(from_mapping=lambda doc: FakeArtifact(str(doc.get("model_artifact_hash") or CANDIDATE_HASH))),
    )

    sides: dict[str, list[dict[str, Any]]] = {
        # Champion (baseline) side then candidate side, keyed by mode_label.
        "confirmation_baseline": _side_summaries({"icp:a": 10.0, "icp:b": 12.0}),
        "confirmation_candidate": _side_summaries({"icp:a": 12.0, "icp:b": 14.0}),
    }
    side_calls: list[str] = []

    async def _fake_side(*, artifact: Any, window: Any, mode_label: str, run_start: float) -> Any:
        side_calls.append(mode_label)
        return list(sides[mode_label]), {"retried": 0, "recovered": 0, "unresolved": 0}

    monkeypatch.setattr(worker, "_run_confirmation_side", _fake_side)

    redrive_results: list[dict[str, Any]] = []

    async def _fake_promote(*, candidate: Any, score_bundle_row: Any, score_bundle: Any) -> dict[str, Any]:
        result = {"status": "merged"}
        redrive_results.append(result)
        return result

    monkeypatch.setattr(worker, "_maybe_promote_scored_candidate", _fake_promote)
    _seed_candidate_rows(store)
    return {
        "worker": worker,
        "window": window,
        "sides": sides,
        "side_calls": side_calls,
        "redrive_results": redrive_results,
        "store": store,
    }


async def test_worker_measures_records_and_redrives(confirmation_env, store):
    _seed_hold(store)
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome is not None
    assert outcome["processed"] is True
    assert outcome["status"] == "confirmation_recorded"
    # candidate mean 13.0 - champion mean 11.0 = +2.0
    assert outcome["confirmation_delta"] == pytest.approx(2.0)
    assert outcome["promotion_status"] == "merged"
    # Both sides measured, champion first.
    assert confirmation_env["side_calls"] == ["confirmation_baseline", "confirmation_candidate"]
    started = store.promotion_writes(CONFIRMATION_STARTED_REASON)
    assert len(started) == 1
    assert started[0]["event_doc"]["attempt"] == 1
    recorded = store.promotion_writes(CONFIRMATION_RESULT_REASON)
    assert len(recorded) == 1
    doc = recorded[0]["event_doc"]["confirmation"]
    assert doc["confirmation_delta"] == pytest.approx(2.0)
    assert doc["window_match"] is True
    assert doc["per_icp_candidate_scores"]["icp:a"] == pytest.approx(12.0)
    assert doc["per_icp_baseline_scores"]["icp:b"] == pytest.approx(12.0)
    assert doc["measurement_type"] == "confirmation_rerun_side_measurement"
    # The decision was re-driven and, having settled (merged), closed.
    assert confirmation_env["redrive_results"] == [{"status": "merged"}]
    assert len(store.promotion_writes(CONFIRMATION_CLOSED_REASON)) == 1


async def test_worker_symmetric_exclusion(confirmation_env, store):
    _seed_hold(store)
    # icp:b unresolved on the CANDIDATE side only: it must drop from BOTH
    # sides' means: candidate 12.0 vs champion 10.0 -> +2.0 (not (12+0)/2 ...).
    confirmation_env["sides"]["confirmation_candidate"] = _side_summaries(
        {"icp:a": 12.0, "icp:b": 0.0}, unresolved=frozenset({"icp:b"})
    )
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_recorded"
    doc = store.promotion_writes(CONFIRMATION_RESULT_REASON)[0]["event_doc"]["confirmation"]
    assert doc["provider_excluded_icp_refs"] == ["icp:b"]
    assert doc["included_icp_count"] == 1
    assert doc["baseline_aggregate_score"] == pytest.approx(10.0)
    assert doc["candidate_total_score"] == pytest.approx(12.0)
    assert doc["confirmation_delta"] == pytest.approx(2.0)


async def test_worker_unhealthy_measurement_reholds(confirmation_env, store, monkeypatch):
    _seed_hold(store)
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS", "0")
    confirmation_env["sides"]["confirmation_candidate"] = _side_summaries(
        {"icp:a": 12.0, "icp:b": 0.0}, unresolved=frozenset({"icp:b"})
    )
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_attempt_failed"
    failed = store.promotion_writes(CONFIRMATION_ATTEMPT_FAILED_REASON)
    assert len(failed) == 1
    assert failed[0]["event_doc"]["unhealthy_measurement"] is True
    assert failed[0]["event_doc"]["retryable"] is True
    # No measurement recorded, nothing re-driven: the candidate stays held.
    assert store.promotion_writes(CONFIRMATION_RESULT_REASON) == []
    assert confirmation_env["redrive_results"] == []


async def test_worker_infra_failure_records_attempt_failed(confirmation_env, store, monkeypatch):
    _seed_hold(store)
    worker = confirmation_env["worker"]

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("docker daemon unavailable")

    monkeypatch.setattr(worker, "_run_confirmation_side", _boom)
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_attempt_failed"
    failed = store.promotion_writes(CONFIRMATION_ATTEMPT_FAILED_REASON)
    assert len(failed) == 1
    doc = failed[0]["event_doc"]
    assert doc["attempt"] == 1
    assert doc["retryable"] is True
    # Error text never lands raw in event docs (bug #36 discipline).
    assert "error_diagnostics" in doc
    assert "docker daemon" not in str(doc)


async def test_worker_skips_fresh_started_lease(confirmation_env, store):
    _seed_hold(store)
    row = store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        worker_ref="another-worker",
        event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": 1},
    )
    row["created_at"] = "2099-01-01T00:00:00+00:00"  # fresh forever
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome is None
    assert confirmation_env["side_calls"] == []


async def test_worker_reclaims_expired_started_lease(confirmation_env, store):
    _seed_hold(store)
    row = store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        worker_ref="dead-worker",
        event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": 1},
    )
    row["created_at"] = "2020-01-01T00:00:00+00:00"  # long expired
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_recorded"
    # The dead attempt still counted toward the budget.
    assert store.promotion_writes(CONFIRMATION_STARTED_REASON)[0]["event_doc"]["attempt"] == 1
    new_started = [
        e
        for e in store.promotion_writes(CONFIRMATION_STARTED_REASON)
        if e.get("worker_ref") == "worker-test"
    ]
    assert new_started[0]["event_doc"]["attempt"] == 2


async def test_worker_redrives_recorded_without_remeasuring(confirmation_env, store):
    _seed_hold(store)
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={
            "reason": CONFIRMATION_RESULT_REASON,
            "confirmation": {"confirmation_delta": 1.7},
        },
    )
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_decision_redriven"
    assert outcome["promotion_status"] == "merged"
    assert confirmation_env["side_calls"] == []  # no re-measurement
    assert store.promotion_writes(CONFIRMATION_STARTED_REASON) == []
    assert len(store.promotion_writes(CONFIRMATION_CLOSED_REASON)) == 1


async def test_worker_skips_closed_confirmations(confirmation_env, store):
    _seed_hold(store)
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={"reason": CONFIRMATION_CLOSED_REASON, "decision_status": "merged"},
    )
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome is None
    assert confirmation_env["side_calls"] == []
    assert confirmation_env["redrive_results"] == []


async def test_worker_skips_already_promoted_candidate(confirmation_env, store):
    _seed_hold(store)
    store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="active_version_created",
        promotion_status="merged",
    )
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome is None
    assert confirmation_env["side_calls"] == []


async def test_worker_attempts_exhausted_redrives_for_terminal_rejection(confirmation_env, store):
    _seed_hold(store)
    for attempt in (1, 2, 3):
        row = store.seed_promotion_event(
            candidate_id="cand-1",
            source_score_bundle_id="sb-1",
            event_type="promotion_checked",
            promotion_status="checked",
            event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": attempt},
        )
        row["created_at"] = f"2020-01-01T00:00:0{attempt}+00:00"  # all expired
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_decision_redriven"
    # No fourth measurement was started.
    assert len(store.promotion_writes(CONFIRMATION_STARTED_REASON)) == 3
    assert confirmation_env["side_calls"] == []


async def test_worker_stale_parent_skips_measurement_and_redrives(confirmation_env, store, monkeypatch):
    _seed_hold(store)

    async def _rolled_champion(config: Any, *, register_bootstrap: bool = False) -> Any:
        return SimpleNamespace(artifact=FakeArtifact("sha256:" + "9" * 64), version_row=None)

    monkeypatch.setattr(sw, "load_active_private_model", _rolled_champion)
    worker = confirmation_env["worker"]
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_skipped_stale_parent"
    assert confirmation_env["side_calls"] == []  # never measured either side
    assert store.promotion_writes(CONFIRMATION_RESULT_REASON) == []
    assert confirmation_env["redrive_results"] == [{"status": "merged"}]


async def test_redrive_stale_parent_queues_rebase_once(confirmation_env, store, monkeypatch):
    """A re-drive that lands on stale_parent_needs_rescore must queue the
    rebase (mirroring _score_candidate) instead of stranding the candidate —
    and must not double-queue on a repeat re-drive."""
    worker = confirmation_env["worker"]

    async def _stale_promote(**kwargs: Any) -> dict[str, Any]:
        return {"status": "stale_parent_needs_rescore"}

    monkeypatch.setattr(worker, "_maybe_promote_scored_candidate", _stale_promote)

    async def _epoch() -> int:
        return 7

    monkeypatch.setattr(worker, "_resolve_evaluation_epoch", _epoch)
    rebase_calls: list[str] = []

    async def _fake_rebase(candidate: Any, **kwargs: Any) -> dict[str, Any]:
        rebase_calls.append(str(candidate["candidate_id"]))
        await store.create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            event_type="rebase_queued",
            promotion_status="rebenchmarking",
            event_doc={"stage": kwargs.get("stage")},
        )
        return {"status": "stale_parent_rebased_to_current", "derived_candidate_id": "cand-2"}

    monkeypatch.setattr(worker, "_queue_stale_parent_rebase", _fake_rebase)
    result = await worker._redrive_confirmation_decision(
        candidate_id="cand-1", score_bundle_id="sb-1", context="test"
    )
    assert result["status"] == "stale_parent_rebased_to_current"
    assert rebase_calls == ["cand-1"]
    # The decision settled: the confirmation is closed.
    assert len(store.promotion_writes(CONFIRMATION_CLOSED_REASON)) == 1
    # A repeat re-drive dedupes on the existing rebase_queued event.
    result = await worker._redrive_confirmation_decision(
        candidate_id="cand-1", score_bundle_id="sb-1", context="test"
    )
    assert result["status"] == "stale_parent_rebase_already_queued"
    assert rebase_calls == ["cand-1"]


def _racing_create(store: ConfirmationFakeStore, *, racer_older: bool):
    """Wrap the store's create so a racing worker's started claim lands right
    around ours — older or newer — after both passed the pre-check."""
    original_create = store.create_candidate_promotion_event

    async def _racy_create(**kwargs: Any) -> dict[str, Any]:
        row = await original_create(**kwargs)
        doc = kwargs.get("event_doc") or {}
        if doc.get("reason") == CONFIRMATION_STARTED_REASON and kwargs.get("worker_ref") == "worker-test":
            racing = await original_create(
                candidate_id=kwargs["candidate_id"],
                source_score_bundle_id=kwargs["source_score_bundle_id"],
                event_type="promotion_checked",
                promotion_status="checked",
                worker_ref="racing-worker",
                event_doc={"reason": CONFIRMATION_STARTED_REASON, "attempt": 1},
            )
            if racer_older:
                ours = datetime.fromisoformat(str(row["created_at"]))
                racing["created_at"] = (ours - timedelta(microseconds=1)).isoformat()
        return row

    return _racy_create


async def test_worker_claim_confirm_loses_to_older_racer(confirmation_env, store, monkeypatch):
    """Two workers race past the pre-check; the OLDEST unexpired claim wins —
    ours landed later, so we back off before measuring."""
    _seed_hold(store)
    worker = confirmation_env["worker"]
    monkeypatch.setattr(sw, "create_candidate_promotion_event", _racing_create(store, racer_older=True))
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_claim_lost"
    assert outcome["processed"] is False
    assert confirmation_env["side_calls"] == []  # backed off before measuring
    assert store.promotion_writes(CONFIRMATION_RESULT_REASON) == []


async def test_worker_claim_confirm_wins_against_newer_racer(confirmation_env, store, monkeypatch):
    """Symmetric side of the race: our claim is the oldest, so the racer's
    newer claim loses and exactly one measurement (ours) proceeds."""
    _seed_hold(store)
    worker = confirmation_env["worker"]
    monkeypatch.setattr(sw, "create_candidate_promotion_event", _racing_create(store, racer_older=False))
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome["status"] == "confirmation_recorded"
    assert confirmation_env["side_calls"] == ["confirmation_baseline", "confirmation_candidate"]


async def test_worker_hold_discovery_json_filter_fallback(confirmation_env, store):
    _seed_hold(store)
    store.json_filter_supported = False
    worker = confirmation_env["worker"]
    holds = await worker._find_pending_confirmation_holds()
    assert len(holds) == 1
    assert holds[0]["candidate_id"] == "cand-1"


async def test_end_to_end_score_only_merge_and_legacy_hold_drain(confirmation_env, store, monkeypatch):
    """End to end against the real gate under score-only promotion: a fresh
    scored candidate merges on pass 1; re-drives never double-merge; and the
    worker runner drains a HISTORICAL hold for a second candidate by measuring,
    recording, and re-driving to a merge exactly once."""
    worker = confirmation_env["worker"]
    artifact = FakeArtifact()

    async def _fake_load_active(config: Any, *, register_bootstrap: bool = False) -> ActivePrivateModel:
        return ActivePrivateModel(artifact=artifact, version_row={"private_model_version_id": "v-1"})

    monkeypatch.setattr(promotion, "load_active_private_model", _fake_load_active)

    merges: list[str] = []

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        merges.append(str(kwargs["candidate"]["candidate_id"]))
        await promotion.create_candidate_promotion_event(
            candidate_id=str(kwargs["candidate"]["candidate_id"]),
            source_score_bundle_id=str(kwargs["score_bundle_row"]["score_bundle_id"]),
            event_type="active_version_created",
            promotion_status="merged",
            worker_ref="worker-test",
            event_doc={"candidate_kind": "image_build"},
        )
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)

    # Use the real promotion path from the worker (undo the fixture stub).
    monkeypatch.setattr(
        worker,
        "_maybe_promote_scored_candidate",
        sw.ResearchLabGatewayScoringWorker._maybe_promote_scored_candidate.__get__(worker),
    )
    controller = ResearchLabPromotionController(_controller_config(), worker_ref="worker-test")

    # Pass 1: score-only — the scored candidate merges on one measurement.
    first = await controller.process_scored_candidate(
        candidate=_candidate(),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(),
    )
    assert first["status"] == "merged"
    assert merges == ["cand-1"]
    assert store.promotion_writes(CONFIRMATION_HOLD_REASON) == []

    # Pass 2: a re-drive of the same candidate is idempotent — no double merge.
    again = await controller.process_scored_candidate(
        candidate=_candidate(),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(),
    )
    assert again["status"] == "already_promoted"
    assert merges == ["cand-1"]
    assert len(store.events_of_type("active_version_created")) == 1

    # Pass 3: legacy drain — cand-1 is promoted so the worker skips its
    # historical holds; nothing is measured and nothing merges twice.
    _seed_hold(store)
    outcome = await worker._maybe_run_pending_confirmation()
    assert outcome is None
    assert confirmation_env["side_calls"] == []
    assert merges == ["cand-1"]


# ---------------------------------------------------------------------------
# §0-N6 — scorer-side burst isolation
# ---------------------------------------------------------------------------


def _install_fake_scorer_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    helpers = types.ModuleType("gateway.qualification.utils.helpers")
    helpers.OPENROUTER_API_KEY = "prod-openrouter"
    verification = types.ModuleType("qualification.scoring.verification_helpers")
    verification.OPENROUTER_API_KEY = "prod-openrouter"
    verification.SCRAPINGDOG_API_KEY = "prod-scrapingdog"
    monkeypatch.setitem(sys.modules, "gateway.qualification.utils.helpers", helpers)
    monkeypatch.setitem(sys.modules, "qualification.scoring.verification_helpers", verification)
    return {"helpers": helpers, "verification": verification}


def test_scorer_isolation_overrides_and_restores(monkeypatch):
    modules = _install_fake_scorer_modules(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY", "bench-sd")
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY", "bench-or")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "prod-sd")
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setenv("QUALIFICATION_OPENROUTER_API_KEY", "prod-or")
    import os

    with sw._benchmark_scorer_isolation():
        assert os.environ["SCRAPINGDOG_API_KEY"] == "bench-sd"
        assert os.environ["QUALIFICATION_SCRAPINGDOG_API_KEY"] == "bench-sd"
        assert os.environ["QUALIFICATION_OPENROUTER_API_KEY"] == "bench-or"
        assert modules["helpers"].OPENROUTER_API_KEY == "bench-or"
        assert modules["verification"].OPENROUTER_API_KEY == "bench-or"
        assert modules["verification"].SCRAPINGDOG_API_KEY == "bench-sd"
    # Restored, including the previously-absent variable being removed again.
    assert os.environ["SCRAPINGDOG_API_KEY"] == "prod-sd"
    assert "QUALIFICATION_SCRAPINGDOG_API_KEY" not in os.environ
    assert os.environ["QUALIFICATION_OPENROUTER_API_KEY"] == "prod-or"
    assert modules["helpers"].OPENROUTER_API_KEY == "prod-openrouter"
    assert modules["verification"].SCRAPINGDOG_API_KEY == "prod-scrapingdog"


def test_scorer_isolation_noop_when_unset(monkeypatch):
    modules = _install_fake_scorer_modules(monkeypatch)
    monkeypatch.delenv("RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "prod-sd")
    import os

    with sw._benchmark_scorer_isolation():
        # Falls back to prod values untouched.
        assert os.environ["SCRAPINGDOG_API_KEY"] == "prod-sd"
        assert modules["verification"].SCRAPINGDOG_API_KEY == "prod-scrapingdog"


def test_benchmark_scorer_max_concurrency_env(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY", raising=False)
    assert sw._benchmark_scorer_max_concurrency() == 0  # unlimited
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY", "3")
    assert sw._benchmark_scorer_max_concurrency() == 3
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY", "junk")
    assert sw._benchmark_scorer_max_concurrency() == 0


async def test_score_baseline_outputs_retries_transient_once():
    worker = _worker()
    calls = {"n": 0}

    class FlakyScorer:
        async def score_with_breakdowns(self, outputs: Any, icp: Any, is_reference: bool) -> list[dict[str, Any]]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP Error 429: Too Many Requests")
            return [{"final_score": 12.0}]

    result = await worker._score_baseline_outputs(
        scorer=FlakyScorer(), outputs=[{"company_name": "acme"}], icp={"industry": "saas"}
    )
    assert calls["n"] == 2
    assert result == [{"final_score": 12.0}]


async def test_score_baseline_outputs_no_retry_for_permanent_errors():
    worker = _worker()
    calls = {"n": 0}

    class AuthFailScorer:
        async def score_with_breakdowns(self, outputs: Any, icp: Any, is_reference: bool) -> list[dict[str, Any]]:
            calls["n"] += 1
            raise RuntimeError("openrouter HTTP Error 401: Unauthorized")

    with pytest.raises(RuntimeError):
        await worker._score_baseline_outputs(
            scorer=AuthFailScorer(), outputs=[{"company_name": "acme"}], icp={}
        )
    assert calls["n"] == 1


async def test_score_baseline_outputs_semaphore_bounds_concurrency():
    worker = _worker()
    active = {"now": 0, "max": 0}

    class SlowScorer:
        async def score_with_breakdowns(self, outputs: Any, icp: Any, is_reference: bool) -> list[dict[str, Any]]:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return [{"final_score": 1.0}]

    semaphore = asyncio.Semaphore(1)
    scorer = SlowScorer()
    await asyncio.gather(
        *(
            worker._score_baseline_outputs(
                scorer=scorer,
                outputs=[{"company_name": "acme"}],
                icp={},
                scorer_semaphore=semaphore,
            )
            for _ in range(4)
        )
    )
    assert active["max"] == 1


async def test_run_baseline_icp_scorer_error_marks_unresolved_not_fatal():
    """§0-N6: a scorer exception no longer kills the batch — the ICP is marked
    unresolved (retryable per the runner classifier) for the retry rounds and
    the baseline health gate."""
    worker = _worker()
    import concurrent.futures

    class RateLimitedScorer:
        async def score_with_breakdowns(self, outputs: Any, icp: Any, is_reference: bool) -> list[dict[str, Any]]:
            raise RuntimeError("scrapingdog HTTP Error 429: Too Many Requests")

    item = {
        "icp": {"industry": "saas"},
        "icp_ref": "icp:a",
        "icp_hash": "hash-a",
        "set_id": 1,
        "day_index": 1,
        "day_rank": 1,
    }
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        summary = await worker._run_baseline_icp(
            runner=lambda icp, ctx: [{"company_name": "acme", "employee_count": "11-50"}],
            scorer=RateLimitedScorer(),
            item=item,
            item_index=1,
            total_icps=1,
            run_start=0.0,
            executor=executor,
        )
    finally:
        executor.shutdown(wait=False)
    assert summary["_runtime_error"]  # unresolved
    assert summary["_retryable"] is True  # classified like a runner 429
    assert summary["_nonempty"] is True  # the model DID produce companies
    assert summary["score"] == pytest.approx(0.0)
    diagnostics = summary["diagnostics"]
    assert "scorer_provider_error" in diagnostics["failure_categories"]
    assert diagnostics["runtime_error"]["status"] == 429
    # Health-gate integration: the unresolved scorer error is counted.
    health = sw._build_baseline_health(
        per_icp_summaries=[summary], retried=0, recovered=0, max_unresolved_icps=0
    )
    assert health["unresolved_provider_errors"] == 1
    assert health["gate_passed"] is False


async def test_run_baseline_icp_mode_label_reaches_runner():
    worker = _worker()
    import concurrent.futures

    seen: list[dict[str, Any]] = []

    def runner(icp: Any, ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        seen.append(dict(ctx))
        return []

    class NullScorer:
        async def score_with_breakdowns(self, *args: Any) -> list[dict[str, Any]]:
            return []

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        await worker._run_baseline_icp(
            runner=runner,
            scorer=NullScorer(),
            item={"icp": {}, "icp_ref": "icp:a", "icp_hash": "h", "set_id": 1, "day_index": 1, "day_rank": 1},
            item_index=1,
            total_icps=1,
            run_start=0.0,
            executor=executor,
            mode_label="confirmation_candidate",
        )
    finally:
        executor.shutdown(wait=False)
    assert seen == [{"mode": "confirmation_candidate"}]


# ---------------------------------------------------------------------------
# Baseline any-worker lease confirm
# ---------------------------------------------------------------------------


def _lease_row(
    *,
    event_id: str,
    worker_ref: str,
    status: str = "assigned",
    created_at: str = "2099-01-01T00:00:00+00:00",
    date: str = "2026-07-02",
) -> dict[str, Any]:
    return {
        "dispatch_event_id": event_id,
        "dispatch_status": status,
        "worker_ref": worker_ref,
        "event_doc": {"benchmark_date": date},
        "created_at": created_at,
    }


@pytest.fixture
def lease_worker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    worker = _worker()
    rows: list[dict[str, Any]] = []

    async def _select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_lab_scoring_dispatch_events"
        return sorted(rows, key=lambda r: str(r.get("created_at")), reverse=True)

    monkeypatch.setattr(sw, "select_many", _select_many)
    return {"worker": worker, "rows": rows}


async def test_lease_confirm_earliest_open_lease_wins(lease_worker):
    worker = lease_worker["worker"]
    now = "2099-01-01T00:00:00"
    lease_worker["rows"].extend(
        [
            _lease_row(event_id="d-1", worker_ref="other-worker", created_at=f"{now}.000001+00:00"),
            _lease_row(event_id="d-2", worker_ref="worker-test", created_at=f"{now}.000002+00:00"),
        ]
    )
    # Our lease is newer: the other worker wins, we back off.
    assert await worker._confirm_baseline_lease(
        today="2026-07-02", lease_event={"dispatch_event_id": "d-2"}
    ) is False
    # Reverse the timestamps: we win.
    lease_worker["rows"][0]["created_at"] = f"{now}.000003+00:00"
    assert await worker._confirm_baseline_lease(
        today="2026-07-02", lease_event={"dispatch_event_id": "d-2"}
    ) is True


async def test_lease_confirm_ignores_closed_and_expired_leases(lease_worker):
    worker = lease_worker["worker"]
    lease_worker["rows"].extend(
        [
            # Another worker's run already failed (latest event failed): closed.
            _lease_row(event_id="d-1", worker_ref="failed-worker", created_at="2099-01-01T00:00:00.000001+00:00"),
            _lease_row(
                event_id="d-2",
                worker_ref="failed-worker",
                status="failed",
                created_at="2099-01-01T00:00:00.000002+00:00",
            ),
            # A long-dead worker's lease: expired by age.
            _lease_row(event_id="d-3", worker_ref="dead-worker", created_at="2020-01-01T00:00:00+00:00"),
            # Ours.
            _lease_row(event_id="d-4", worker_ref="worker-test", created_at="2099-01-01T00:00:00.000003+00:00"),
        ]
    )
    assert await worker._confirm_baseline_lease(
        today="2026-07-02", lease_event={"dispatch_event_id": "d-4"}
    ) is True


async def test_lease_confirm_ignores_other_days(lease_worker):
    worker = lease_worker["worker"]
    lease_worker["rows"].extend(
        [
            _lease_row(
                event_id="d-1",
                worker_ref="other-worker",
                created_at="2099-01-01T00:00:00.000001+00:00",
                date="2026-07-01",
            ),
            _lease_row(event_id="d-2", worker_ref="worker-test", created_at="2099-01-01T00:00:00.000002+00:00"),
        ]
    )
    assert await worker._confirm_baseline_lease(
        today="2026-07-02", lease_event={"dispatch_event_id": "d-2"}
    ) is True


async def test_lease_confirm_read_failure_proceeds(monkeypatch):
    worker = _worker()

    async def _boom(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("postgrest reset")

    monkeypatch.setattr(sw, "select_many", _boom)
    assert await worker._confirm_baseline_lease(
        today="2026-07-02", lease_event={"dispatch_event_id": "d-1"}
    ) is True
