"""Tests for the Research Lab promotion fail-closed fixes and the score-only
merge path (bugs 2/3/4/24, N3; score-only design decision 2026-07-02).

Score-only promotion (commit 3aaee73c): a candidate promotes purely on its
stored final score vs the stored daily baseline aggregate. Provider/runtime
health, quarantine bookkeeping, and provider-exclusion audit fields are
recorded for observability but never veto the merge.

Covers, with fake store rows (no live Supabase):
  * bug #2  — lineage fail-closed: read error vs genuinely-empty lineage vs
    flag-gated bootstrap registration; manifest hash mismatch raises.
  * bug #3  — reconcile re-activates the newest superseded version.
  * N3      — unavailable basis is an explicit rejection, not 0.0-below-threshold.
  * score-only merge path — health/quarantine/baseline-doc state cannot hold
    or block the decision; provider exclusions never adjust the basis.
  * champion reward windows start at the live epoch at creation time, never
    the bundle's scoring epoch (the 2026-07-02 backdating incident).
  * bug #24 — pending champion reward reconciler happy/retry paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import gateway.research_lab.promotion as promotion
from gateway.research_lab.promotion import (
    ActiveManifestHashMismatchError,
    ActivePrivateModel,
    NoActivePrivateModelVersionError,
    PrivateModelLineageUnavailableError,
    ResearchLabPromotionController,
    load_active_private_model,
    promotion_improvement_metric,
    reconcile_active_private_model_lineage,
    reconcile_pending_champion_rewards,
)


@dataclass
class FakeArtifact:
    model_artifact_hash: str = "sha256:" + "a" * 64
    manifest_hash: str = "sha256:" + "b" * 64
    manifest_uri: str = "s3://bucket/manifest.json"
    git_commit_sha: str = "c" * 40
    component_registry_version: str = "1.0"
    scoring_adapter_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_artifact_hash": self.model_artifact_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_uri": self.manifest_uri,
            "git_commit_sha": self.git_commit_sha,
            "config_hash": "sha256:" + "d" * 64,
            "component_registry_version": self.component_registry_version,
            "scoring_adapter_version": self.scoring_adapter_version,
            "signature_ref": "sig",
            "build_id": None,
        }


@dataclass
class FakeStore:
    """Table-name-dispatched fakes for the store functions promotion.py uses."""

    select_many_results: dict[str, Any] = field(default_factory=dict)
    select_one_results: dict[str, Any] = field(default_factory=dict)
    select_many_calls: list[tuple[str, tuple]] = field(default_factory=list)
    version_writes: list[dict[str, Any]] = field(default_factory=list)
    version_event_writes: list[dict[str, Any]] = field(default_factory=list)
    promotion_event_writes: list[dict[str, Any]] = field(default_factory=list)
    reward_obligation_writes: list[dict[str, Any]] = field(default_factory=list)

    async def select_many(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.select_many_calls.append((table, tuple(kwargs.get("filters") or ())))
        result = self.select_many_results.get(self._select_many_key(table, kwargs))
        if result is None:
            result = self.select_many_results.get(table, [])
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return list(result)

    def _select_many_key(self, table: str, kwargs: Mapping[str, Any]) -> str:
        filters = tuple(kwargs.get("filters") or ())
        for spec in filters:
            if len(spec) == 2 and spec[0] in ("current_version_status", "event_type"):
                return f"{table}:{spec[1]}"
        if not filters:
            return f"{table}:unfiltered"
        return table

    async def select_one(self, table: str, **kwargs: Any) -> dict[str, Any] | None:
        result = self.select_one_results.get(table)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return result

    async def create_private_model_version(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.version_writes.append(kwargs)
        return (
            {"private_model_version_id": "private_model_version:sha256:" + "e" * 64, **kwargs},
            {"event_type": kwargs.get("version_status")},
        )

    async def create_private_model_version_event(self, **kwargs: Any) -> dict[str, Any]:
        self.version_event_writes.append(kwargs)
        return {"event_id": f"evt-{len(self.version_event_writes)}", **kwargs}

    async def create_candidate_promotion_event(self, **kwargs: Any) -> dict[str, Any]:
        self.promotion_event_writes.append(kwargs)
        return {"promotion_event_id": f"pe-{len(self.promotion_event_writes)}", **kwargs}

    async def create_champion_reward_obligation(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reward_obligation_writes.append(kwargs)
        return {"champion_reward_id": "cr-1"}, {"event_type": "active"}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(promotion, "select_many", fake.select_many)
    monkeypatch.setattr(promotion, "select_one", fake.select_one)
    monkeypatch.setattr(promotion, "create_private_model_version", fake.create_private_model_version)
    monkeypatch.setattr(promotion, "create_private_model_version_event", fake.create_private_model_version_event)
    monkeypatch.setattr(promotion, "create_candidate_promotion_event", fake.create_candidate_promotion_event)
    monkeypatch.setattr(promotion, "create_champion_reward_obligation", fake.create_champion_reward_obligation)
    monkeypatch.delenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, raising=False)
    monkeypatch.delenv(promotion.AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV, raising=False)
    return fake


@pytest.fixture
def bootstrap_artifact(monkeypatch: pytest.MonkeyPatch) -> FakeArtifact:
    artifact = FakeArtifact()
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: artifact)
    return artifact


def _config() -> Any:
    return SimpleNamespace(private_model_manifest_uri="s3://bucket/bootstrap-manifest.json")


def _active_row(artifact: FakeArtifact) -> dict[str, Any]:
    return {
        "private_model_version_id": "private_model_version:sha256:" + "f" * 64,
        "private_model_manifest_uri": artifact.manifest_uri,
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "current_version_status": "active",
        "current_status_at": "2026-07-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Bug #2 — lineage fail-closed
# ---------------------------------------------------------------------------


async def test_lineage_read_error_raises_retryable_and_never_bootstraps(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = RuntimeError("supabase blip")
    with pytest.raises(PrivateModelLineageUnavailableError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_lineage_empty_without_flag_returns_unregistered_bootstrap(store, bootstrap_artifact):
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.artifact is bootstrap_artifact
    assert result.version_row is None
    assert store.version_writes == []


async def test_lineage_empty_with_flag_registers_bootstrap(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.version_row is not None
    assert len(store.version_writes) == 1
    assert store.version_writes[0]["version_status"] == "active"
    assert store.version_writes[0]["reason"] == "bootstrap_private_model_manifest_uri"


async def test_lineage_empty_register_bootstrap_false_never_writes(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=False)
    assert result.version_row is None
    assert store.version_writes == []


async def test_zero_active_but_nonempty_lineage_raises_instead_of_bootstrap(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = [
        {"private_model_version_id": "v1", "current_version_status": "superseded"}
    ]
    with pytest.raises(NoActivePrivateModelVersionError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []


async def test_manifest_hash_mismatch_raises_explicit_operator_error(store, monkeypatch):
    row_artifact = FakeArtifact()
    row = _active_row(row_artifact)
    # The manifest URI now yields different hashes than the lineage row recorded.
    loaded = FakeArtifact(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
        manifest_uri=row_artifact.manifest_uri,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: loaded)
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    with pytest.raises(ActiveManifestHashMismatchError) as excinfo:
        await load_active_private_model(_config(), register_bootstrap=True)
    assert "reregister-active-manifest" in str(excinfo.value)
    assert excinfo.value.detail["row_model_artifact_hash"] == row_artifact.model_artifact_hash
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_active_manifest_load_failure_raises_retryable(store, monkeypatch):
    row = _active_row(FakeArtifact())

    def _boom(uri: str) -> FakeArtifact:
        raise RuntimeError("s3 timeout")

    monkeypatch.setattr(promotion, "_load_valid_artifact", _boom)
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    with pytest.raises(PrivateModelLineageUnavailableError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []


async def test_matching_active_row_returned(store, monkeypatch):
    artifact = FakeArtifact()
    row = _active_row(artifact)
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: artifact)
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.version_row == row
    assert result.artifact is artifact
    assert store.version_writes == []


# ---------------------------------------------------------------------------
# Bug #3 — reconcile re-activates the newest superseded version
# ---------------------------------------------------------------------------


async def test_reconcile_noop_when_active_present(store):
    store.select_many_results["research_lab_private_model_version_current:active"] = [
        {"private_model_version_id": "v-active", "current_version_status": "active"}
    ]
    result = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=False)
    assert result["status"] == "active_version_present"
    assert store.version_event_writes == []


async def test_reconcile_picks_newest_superseded(store):
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = [
        # Ordered newest-first by current_status_at, as the query requests.
        {"private_model_version_id": "v-tomb", "current_version_status": "tombstoned", "current_status_at": "2026-07-03"},
        {"private_model_version_id": "v-new", "current_version_status": "superseded", "current_status_at": "2026-07-02", "model_artifact_hash": "sha256:" + "9" * 64},
        {"private_model_version_id": "v-old", "current_version_status": "superseded", "current_status_at": "2026-07-01"},
    ]
    dry = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=True)
    assert dry["status"] == "would_reactivate_newest_superseded"
    assert dry["planned"]["private_model_version_id"] == "v-new"
    assert store.version_event_writes == []

    applied = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=False)
    assert applied["status"] == "reactivated_newest_superseded"
    assert len(store.version_event_writes) == 1
    write = store.version_event_writes[0]
    assert write["private_model_version_id"] == "v-new"
    assert write["event_type"] == "active"
    assert write["version_status"] == "active"


async def test_reconcile_lineage_empty(store):
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=False)
    assert result["status"] == "lineage_empty"
    assert store.version_event_writes == []


# ---------------------------------------------------------------------------
# N3 — unavailable basis is an explicit rejection
# ---------------------------------------------------------------------------


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


def test_metric_approved_gate_has_no_rejection() -> None:
    metric = promotion_improvement_metric({"private_holdout_gate": _approved_gate(), "aggregates": {}})
    assert metric.rejection_status is None
    assert metric.daily_baseline_available is True
    assert metric.improvement_points == pytest.approx(2.5)


def test_metric_missing_basis_is_explicit_rejection_not_zero_pass() -> None:
    gate = _approved_gate(
        baseline_aggregate_score=None,
        candidate_total_score=None,
        candidate_delta_vs_daily_baseline=None,
    )
    metric = promotion_improvement_metric({"private_holdout_gate": gate, "aggregates": {}})
    assert metric.rejection_status == "rejected_basis_unavailable"
    assert metric.daily_baseline_available is False
    assert metric.improvement_points == 0.0
    # A future improvement_threshold_points=0 must never promote this bundle:
    # the rejection is carried explicitly, not implied by 0.0 < threshold.
    assert metric.event_doc()["rejection_status"] == "rejected_basis_unavailable"


def test_metric_unapproved_holdout_is_explicit_rejection() -> None:
    gate = _approved_gate(decision="rejected_before_private_holdout")
    metric = promotion_improvement_metric({"private_holdout_gate": gate, "aggregates": {}})
    assert metric.rejection_status == "rejected_basis_unavailable"
    assert "rejected_before_private_holdout" in metric.basis


def test_metric_legacy_bundle_keeps_paired_mean_delta_path() -> None:
    metric = promotion_improvement_metric({"aggregates": {"mean_delta": 1.75}})
    assert metric.rejection_status is None
    assert metric.basis == "legacy_paired_mean_delta_no_holdout_gate"
    assert metric.improvement_points == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Score-only basis — provider exclusions are audit metadata, never arithmetic
# ---------------------------------------------------------------------------


def _baseline_doc(scores: Mapping[str, float], **extra: Any) -> dict[str, Any]:
    return {
        "per_icp_summaries": [
            {"icp_ref": ref, "score": value} for ref, value in scores.items()
        ],
        "aggregate_score": sum(scores.values()) / len(scores),
        **extra,
    }


def test_metric_exclusions_never_adjust_the_basis() -> None:
    """Score-only: provider-excluded ICPs are carried through as audit fields
    but the delta stays exactly the gate's stored candidate-vs-baseline delta —
    no per-ICP baseline re-aggregation."""
    doc = _baseline_doc({"icp:a": 10.0, "icp:b": 0.0, "icp:c": 20.0})
    gate = _approved_gate(
        baseline_aggregate_score=10.0,
        candidate_total_score=12.0,
        candidate_delta_vs_daily_baseline=2.0,
    )
    bundle = {
        "private_holdout_gate": gate,
        "aggregates": {"provider_excluded_icp_ids": ["icp:b"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.baseline_basis_adjusted is False
    assert metric.baseline_aggregate_score == pytest.approx(10.0)
    assert metric.unadjusted_baseline_aggregate_score is None
    assert metric.improvement_points == pytest.approx(2.0)
    assert metric.provider_excluded_icp_ids == ("icp:b",)
    assert metric.basis == "stored_daily_baseline_total_delta"


def test_metric_exclusions_without_baseline_doc_still_compute_stored_delta() -> None:
    """The stored gate delta is self-sufficient: no baseline per-ICP doc is
    needed (or consulted), so its absence cannot reject the candidate."""
    bundle = {
        "private_holdout_gate": _approved_gate(),
        "aggregates": {"provider_excluded_icp_ids": ["icp:b"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=None)
    assert metric.rejection_status is None
    assert metric.improvement_points == pytest.approx(2.5)
    assert metric.basis == "stored_daily_baseline_total_delta"


def test_metric_unknown_exclusion_ids_do_not_reject() -> None:
    doc = _baseline_doc({"icp:a": 10.0, "icp:c": 20.0})
    bundle = {
        "private_holdout_gate": _approved_gate(),
        "aggregates": {"provider_excluded_icp_ids": ["icp:zzz"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.provider_excluded_icp_ids == ("icp:zzz",)


def test_metric_tolerates_absent_exclusion_list() -> None:
    doc = _baseline_doc({"icp:a": 10.0, "icp:c": 20.0})
    bundle = {"private_holdout_gate": _approved_gate(), "aggregates": {}}
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.baseline_basis_adjusted is False
    assert metric.improvement_points == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Score-only merge path — health/quarantine state cannot hold or block (N3
# basis rejection is the only non-score gate left besides the threshold)
# ---------------------------------------------------------------------------


def _controller_config(**overrides: Any) -> Any:
    values = {
        "auto_promotion_enabled": True,
        "auto_commit_enabled": False,
        "improvement_threshold_points": 1.0,
        "private_model_manifest_uri": "s3://bucket/bootstrap-manifest.json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _candidate(artifact: FakeArtifact) -> dict[str, Any]:
    return {
        "candidate_id": "cand-1",
        "parent_artifact_hash": artifact.model_artifact_hash,
        "candidate_kind": "image_build",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }


def _score_bundle(gate: Mapping[str, Any], aggregates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "private_holdout_gate": dict(gate),
        "aggregates": dict(aggregates or {}),
        "icp_set_hash": "sha256:" + "3" * 64,
    }


@pytest.fixture
def controller_env(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    artifact = FakeArtifact()

    async def _fake_load_active(config: Any, *, register_bootstrap: bool = False) -> ActivePrivateModel:
        return ActivePrivateModel(artifact=artifact, version_row=_active_row(artifact))

    monkeypatch.setattr(promotion, "load_active_private_model", _fake_load_active)
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = []
    controller = ResearchLabPromotionController(_controller_config(), worker_ref="test-worker")
    return {"artifact": artifact, "controller": controller, "store": store}


async def test_baseline_health_gate_failed_does_not_hold(controller_env, monkeypatch):
    """Score-only: an unhealthy daily baseline (gate_passed=False) is audit
    metadata — the merge proceeds on the stored score alone."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc(
            {"icp:a": 10.0},
            baseline_health={"unresolved_provider_errors": 7, "gate_passed": False},
        ),
    }

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(_approved_gate()),
    )
    assert result["status"] == "merged"
    held = [
        event
        for event in store.promotion_event_writes
        if str((event.get("event_doc") or {}).get("reason") or "").startswith("held_")
    ]
    assert held == []


async def test_baseline_health_absent_is_tolerated(controller_env):
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),  # legacy: no baseline_health
    }
    # Below-threshold delta proves the flow passed the health gate.
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_baseline_doc_not_consulted_on_merge_path(controller_env):
    """Score-only: the merge path never fetches the baseline per-ICP doc, so a
    store error on that table cannot hold the decision. The below-threshold
    rejection proves the flow ran to its normal conclusion."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = RuntimeError("postgrest reset")
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_quarantine_events_do_not_block_score_only_merge(controller_env, monkeypatch):
    """Score-only: historical scoring_health_quarantined bookkeeping does not
    veto the merge — the stored score decides."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),
    }
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = [
        {
            "promotion_event_id": "pe-q1",
            "event_type": "scoring_health_quarantined",
            "promotion_status": "rejected",
            "source_score_bundle_id": "sb-1",
        }
    ]

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(_approved_gate()),
    )
    assert result["status"] == "merged"
    blocked = [e for e in store.promotion_event_writes if e["event_type"] == "scoring_health_quarantined"]
    assert blocked == []


async def test_bypass_gates_param_cannot_waive_the_threshold(controller_env):
    """bypass_gates is accepted for replay-command compatibility, but with the
    health/quarantine gates retired the score threshold is the decision — and
    it can never be bypassed."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc(
            {"icp:a": 10.0},
            baseline_health={"unresolved_provider_errors": 7, "gate_passed": False},
        ),
    }
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = [
        {"promotion_event_id": "pe-q1", "source_score_bundle_id": "sb-1"}
    ]
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
        bypass_gates=frozenset({"scoring_health_quarantine", "baseline_health"}),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_basis_unavailable_rejected_on_merge_path(controller_env):
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),
    }
    gate = _approved_gate(
        baseline_aggregate_score=None,
        candidate_total_score=None,
        candidate_delta_vs_daily_baseline=None,
    )
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_basis_unavailable"
    rejected = [
        e
        for e in store.promotion_event_writes
        if (e.get("event_doc") or {}).get("reason") == "rejected_basis_unavailable"
    ]
    assert len(rejected) == 1
    assert rejected[0]["event_type"] == "below_threshold"
    assert rejected[0]["promotion_status"] == "rejected"


# ---------------------------------------------------------------------------
# Champion reward start_epoch — windows start at creation time (2026-07-02
# backdating incident: a reward scored at epoch N but merged at N+15 paid
# ~2.5h of a ~24h window)
# ---------------------------------------------------------------------------


def _capture_obligation(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def _build(obligation_input: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
        captured.append(dict(obligation_input))
        return {"status": "active", "champion_reward_id": "cr-1", **obligation_input}

    monkeypatch.setattr(promotion, "build_champion_reward_obligation", _build)
    return captured


def _epoch_config() -> Any:
    return SimpleNamespace(
        auto_promotion_enabled=True,
        auto_commit_enabled=False,
        improvement_threshold_points=1.0,
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        reimbursement_policy_doc=lambda enabled: {"policy_id": "policy-1"},
        lab_reward_epochs=20,
        evaluation_epoch=0,  # no operator override: the live epoch is resolved
    )


async def test_reward_start_epoch_uses_live_epoch_not_bundle_epoch(store, monkeypatch):
    captured = _capture_obligation(monkeypatch)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 150, None, "test"

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")
    result = await controller._maybe_create_champion_reward(
        candidate={
            "candidate_id": "cand-1",
            "miner_hotkey": "hk-1",
            "ticket_id": "ticket-1",
            "run_id": "run-1",
            "island": "generalist",
        },
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle={"evaluation_epoch": 100, "aggregates": {"per_icp_results": []}},
        improvement_points=2.5,
        threshold=1.0,
    )
    assert result["champion_reward_status"] == "created"
    assert len(captured) == 1
    # Scoring provenance is kept, but the window starts NOW: live 150 -> 151,
    # never bundle-epoch 100 -> 101 (which would expire 50 epochs pre-paid).
    assert captured[0]["evaluation_epoch"] == 100
    assert captured[0]["start_epoch"] == 151


async def test_reward_start_epoch_falls_back_to_bundle_epoch_when_chain_unreachable(store, monkeypatch):
    captured = _capture_obligation(monkeypatch)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    async def _broken_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        raise RuntimeError("subtensor unreachable")

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _broken_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")
    result = await controller._maybe_create_champion_reward(
        candidate={
            "candidate_id": "cand-1",
            "miner_hotkey": "hk-1",
            "ticket_id": "ticket-1",
            "run_id": "run-1",
        },
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle={"evaluation_epoch": 100, "aggregates": {"per_icp_results": []}},
        improvement_points=2.5,
        threshold=1.0,
    )
    # Degraded but never blocked: the legacy bundle-epoch derivation applies.
    assert result["champion_reward_status"] == "created"
    assert captured[0]["start_epoch"] == 101


# ---------------------------------------------------------------------------
# Bug #24 — champion reward reconciler
# ---------------------------------------------------------------------------


def _reward_config() -> Any:
    return SimpleNamespace(
        auto_promotion_enabled=True,
        auto_commit_enabled=False,
        improvement_threshold_points=1.0,
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        reimbursement_policy_doc=lambda enabled: {"policy_id": "policy-1"},
        lab_reward_epochs=3,
        evaluation_epoch=7,
    )


def _pending_reward_rows(store: FakeStore) -> None:
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_pending_uid"] = [
        {
            "promotion_event_id": "pe-pending",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "improvement_points": 2.5,
            "threshold_points": 1.0,
            "created_at": "2026-07-01T00:00:00+00:00",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_created"] = []
    store.select_one_results["research_lab_candidate_evaluation_current"] = {
        "candidate_id": "cand-1",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
        "island": "generalist",
        "current_score_bundle_id": "sb-1",
    }
    store.select_one_results["research_evaluation_score_bundle_current"] = {
        "score_bundle_id": "sb-1",
        "score_bundle_doc": {
            "evaluation_epoch": 7,
            "score_bundle_hash": "sha256:" + "4" * 64,
            "aggregates": {"per_icp_results": []},
        },
    }


async def test_reward_reconciler_happy_path_creates_reward(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(
        promotion,
        "build_champion_reward_obligation",
        lambda obligation_input, policy: {
            "status": "active",
            "champion_reward_id": "cr-1",
            "candidate_id": obligation_input["candidate_id"],
            "score_bundle_id": obligation_input["score_bundle_id"],
            "run_id": obligation_input["run_id"],
            "miner_hotkey": obligation_input["miner_hotkey"],
            "uid": obligation_input["uid"],
            "island": obligation_input["island"],
            "evaluation_epoch": obligation_input["evaluation_epoch"],
            "start_epoch": obligation_input["start_epoch"],
            "epoch_count": 3,
            "improvement_points": obligation_input["improvement_points"],
            "threshold_points": obligation_input["threshold_points"],
            "desired_alpha_percent": 1.0,
            "input_hash": "sha256:" + "5" * 64,
            "anchored_hash": "sha256:" + "6" * 64,
        },
    )
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["ok"] is True
    assert result["found_pending"] == 1
    entry = result["results"][0]
    assert entry["status"] == "created"
    assert entry["champion_reward_id"] == "cr-1"
    assert entry["resolved_uid"] == 5
    assert len(store.reward_obligation_writes) == 1
    created_events = [
        e for e in store.promotion_event_writes if e["event_type"] == "champion_reward_created"
    ]
    assert len(created_events) == 1


async def test_reward_reconciler_uid_still_unresolved_retries_later_without_event_spam(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return None

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["results"][0]["status"] == "uid_still_unresolved"
    assert store.promotion_event_writes == []
    assert store.reward_obligation_writes == []


async def test_reward_reconciler_dry_run_plans_without_writes(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=True,
    )
    assert result["results"][0]["status"] == "would_create_champion_reward"
    assert store.promotion_event_writes == []
    assert store.reward_obligation_writes == []


async def test_reward_reconciler_skips_already_created(store, monkeypatch):
    _pending_reward_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_created"] = [
        {"promotion_event_id": "pe-created", "event_type": "champion_reward_created"}
    ]
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["results"][0]["status"] == "already_created"
    assert store.promotion_event_writes == []
