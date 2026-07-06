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
from datetime import datetime, timezone
import json
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
    reconcile_failed_private_source_pushes,
    reconcile_pending_champion_rewards,
    sync_active_model_to_repo_head,
)
from research_lab.canonical import sha256_json


@dataclass
class FakeArtifact:
    model_artifact_hash: str = "sha256:" + "a" * 64
    manifest_hash: str = "sha256:" + "b" * 64
    manifest_uri: str = "s3://bucket/manifest.json"
    git_commit_sha: str = "c" * 40
    image_digest: str = "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:" + "a" * 64
    component_registry_version: str = "1.0"
    scoring_adapter_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_artifact_hash": self.model_artifact_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_uri": self.manifest_uri,
            "git_commit_sha": self.git_commit_sha,
            "image_digest": self.image_digest,
            "config_hash": "sha256:" + "d" * 64,
            "component_registry_version": self.component_registry_version,
            "scoring_adapter_version": self.scoring_adapter_version,
            "signature_ref": "sig",
            "build_id": "",
        }


def _valid_fake_artifact(**overrides: Any) -> FakeArtifact:
    artifact = FakeArtifact(**overrides)
    payload = artifact.to_dict()
    payload.pop("manifest_hash", None)
    artifact.manifest_hash = sha256_json(payload)
    return artifact


@dataclass
class FakeStore:
    """Table-name-dispatched fakes for the store functions promotion.py uses."""

    select_many_results: dict[str, Any] = field(default_factory=dict)
    select_one_results: dict[str, Any] = field(default_factory=dict)
    select_many_calls: list[tuple[str, tuple]] = field(default_factory=list)
    version_writes: list[dict[str, Any]] = field(default_factory=list)
    version_event_writes: list[dict[str, Any]] = field(default_factory=list)
    promotion_event_writes: list[dict[str, Any]] = field(default_factory=list)
    candidate_evaluation_event_writes: list[dict[str, Any]] = field(default_factory=list)
    scoring_dispatch_event_writes: list[dict[str, Any]] = field(default_factory=list)
    reward_obligation_writes: list[dict[str, Any]] = field(default_factory=list)
    private_benchmark_writes: list[dict[str, Any]] = field(default_factory=list)
    public_report_writes: list[dict[str, Any]] = field(default_factory=list)

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

    async def create_candidate_evaluation_event(self, **kwargs: Any) -> dict[str, Any]:
        self.candidate_evaluation_event_writes.append(kwargs)
        return {"event_id": f"ce-{len(self.candidate_evaluation_event_writes)}", **kwargs}

    async def create_scoring_dispatch_event(self, **kwargs: Any) -> dict[str, Any]:
        self.scoring_dispatch_event_writes.append(kwargs)
        return {"dispatch_event_id": f"sd-{len(self.scoring_dispatch_event_writes)}", **kwargs}

    async def create_champion_reward_obligation(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reward_obligation_writes.append(kwargs)
        return {"champion_reward_id": "cr-1"}, {"event_type": "active"}

    async def create_private_model_benchmark_bundle(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.private_benchmark_writes.append(kwargs)
        return (
            {
                "benchmark_bundle_id": "private_benchmark:" + "8" * 64,
                "current_benchmark_status": "completed",
                **kwargs,
            },
            {"event_type": "completed"},
        )

    async def create_public_benchmark_report(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.public_report_writes.append(kwargs)
        return (
            {
                "report_id": "public_benchmark:sha256:" + "9" * 64,
                "current_report_status": "published",
                **kwargs,
            },
            {"event_type": "published"},
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(promotion, "select_many", fake.select_many)
    monkeypatch.setattr(promotion, "select_one", fake.select_one)
    monkeypatch.setattr(promotion, "create_private_model_version", fake.create_private_model_version)
    monkeypatch.setattr(promotion, "create_private_model_version_event", fake.create_private_model_version_event)
    monkeypatch.setattr(promotion, "create_candidate_promotion_event", fake.create_candidate_promotion_event)
    monkeypatch.setattr(promotion, "create_candidate_evaluation_event", fake.create_candidate_evaluation_event)
    monkeypatch.setattr(promotion, "create_scoring_dispatch_event", fake.create_scoring_dispatch_event)
    monkeypatch.setattr(promotion, "create_champion_reward_obligation", fake.create_champion_reward_obligation)
    monkeypatch.setattr(promotion, "create_private_model_benchmark_bundle", fake.create_private_model_benchmark_bundle)
    monkeypatch.setattr(promotion, "create_public_benchmark_report", fake.create_public_benchmark_report)
    monkeypatch.setattr(promotion, "sign_digest_with_kms", lambda **kwargs: "kms-signature:test")
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


def _assert_db_doc_safe(doc: Mapping[str, Any]) -> None:
    encoded = json.dumps(doc, sort_keys=True, default=str)
    assert not promotion._DB_DOC_FORBIDDEN_RE.search(encoded)


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
        "score_bundle_kms_key_id": "arn:aws:kms:us-east-1:123456789012:alias/test",
        "score_bundle_signature_uri_prefix": "s3://bucket/signatures",
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


async def test_repo_head_sync_registers_current_json_with_db_safe_doc(store, monkeypatch):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "9" * 64
        ),
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    previous_row = {
        **_active_row(previous_artifact),
        "git_commit_sha": previous_artifact.git_commit_sha,
    }
    store.select_many_results["research_lab_private_model_version_current:active"] = [previous_row]

    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda *, repo_url, branch_name: current_artifact.git_commit_sha,
    )

    async def _fake_current_manifest(config: Any, **kwargs: Any) -> tuple[FakeArtifact, dict[str, Any]]:
        return current_artifact, {
            "status": "manifest_ready",
            "current_json_git_sha": current_artifact.git_commit_sha,
            "current_json_manifest_hash": current_artifact.manifest_hash,
            "current_json_model_artifact_hash": current_artifact.model_artifact_hash,
            "current_json_image_digest": current_artifact.image_digest,
        }

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _fake_current_manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@github.com:tasnimuldatascience/Sourcing_model.git",
            private_repo_branch="main",
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert result["status"] == "synced_active_model_to_repo_head"
    assert len(store.version_writes) == 1
    version_doc = store.version_writes[0]["redacted_version_doc"]
    _assert_db_doc_safe(version_doc)
    assert version_doc["image_ref_hash"].startswith("sha256:")
    assert "image_digest" not in json.dumps(version_doc, sort_keys=True, default=str)


def _bridge_baseline_row(window_hash: str, baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "benchmark_bundle_id": baseline_bundle_id,
        "benchmark_date": "2026-07-02",
        "rolling_window_hash": window_hash,
        "evaluation_epoch": 23697,
        "benchmark_attempt": 0,
        "benchmark_quality": "passed",
        "aggregate_score": 16.353333,
        "current_benchmark_status": "completed",
        "score_summary_doc": {
            "schema_version": "1.0",
            "aggregate_score": 16.353333,
            "per_icp_summaries": [
                {
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "score": 10.0,
                    "company_count": 1,
                    "industry": "Software",
                    "sub_industry": "Sales Software",
                    "country": "United States",
                    "company_size_bucket": "51-200",
                    "intent_category_bucket": "vendor_replacement",
                    "diagnostics": {"failure_categories": []},
                },
                {
                    "icp_ref": "icp:b",
                    "icp_hash": "sha256:" + "b" * 64,
                    "score": 22.706666,
                    "company_count": 1,
                    "industry": "Healthcare",
                    "sub_industry": "Clinics",
                    "country": "United States",
                    "company_size_bucket": "201-500",
                    "intent_category_bucket": "growth",
                    "diagnostics": {"failure_categories": []},
                },
            ],
            "visibility_split": {
                "schema_version": "1.0",
                "split_policy": "test_split",
                "rolling_window_hash": window_hash,
                "public_count": 1,
                "private_count": 1,
                "public_strength_counts": {"weak": 1},
                "private_strength_counts": {"strong": 1},
                "items": [
                    {
                        "item_rank": 1,
                        "icp_ref": "icp:a",
                        "icp_hash": "sha256:" + "a" * 64,
                        "set_id": 1,
                        "day_index": 1,
                        "day_rank": 1,
                        "score": 10.0,
                        "visibility": "public",
                        "strength_label": "weak",
                    },
                    {
                        "item_rank": 2,
                        "icp_ref": "icp:b",
                        "icp_hash": "sha256:" + "b" * 64,
                        "set_id": 1,
                        "day_index": 1,
                        "day_rank": 2,
                        "score": 22.706666,
                        "visibility": "private",
                        "strength_label": "strong",
                    },
                ],
            },
        },
    }


def _bridge_public_report_row(baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "report_id": "public_benchmark:sha256:" + "6" * 64,
        "benchmark_bundle_id": baseline_bundle_id,
        "current_report_status": "published",
        "report_doc": {
            "public_icps": [
                {
                    "item_rank": 1,
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "set_id": 1,
                    "day_index": 1,
                    "day_rank": 1,
                    "score": 10.0,
                    "company_count": 1,
                    "strength_label": "weak",
                    "icp": {"industry": "Software"},
                    "diagnostics": {"failure_categories": []},
                }
            ],
        },
    }


def _bridge_score_bundle(candidate_artifact: FakeArtifact, window_hash: str, baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
        "parent_artifact_hash": "sha256:" + "a" * 64,
        "private_model_manifest_hash": "sha256:" + "b" * 64,
        "icp_set_hash": window_hash,
        "evaluation_epoch": 23697,
        "score_bundle_hash": "sha256:" + "5" * 64,
        "aggregates": {
            "per_icp_results": [
                {
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "candidate_company_scores": [30.0, 20.0],
                    "failure_reason": "",
                },
                {
                    "icp_ref": "icp:b",
                    "icp_hash": "sha256:" + "b" * 64,
                    "candidate_company_scores": [35.945454, 28.945454],
                    "failure_reason": "",
                },
            ],
        },
        "private_holdout_gate": _approved_gate(
            baseline_benchmark_bundle_id=baseline_bundle_id,
            baseline_aggregate_score=16.353333,
            candidate_total_score=28.472727,
            candidate_delta_vs_daily_baseline=12.119394,
            reference_evaluation_mode="stored_daily_baseline",
        ),
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
# Promoted candidate benchmark bridge
# ---------------------------------------------------------------------------


async def test_promoted_candidate_writes_derived_benchmark_and_links_active_version(store, monkeypatch):
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    activation_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "e" * 64,
        git_commit_sha="e" * 40,
        manifest_uri="s3://bucket/research-lab/sourcing-model/current.json",
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "e" * 64
        ),
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    def _public_report_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("benchmark_bundle_id") == baseline_bundle_id:
            return [_bridge_public_report_row(baseline_bundle_id)]
        return []

    store.select_many_results["research_lab_public_benchmark_report_current"] = _public_report_rows

    async def _fake_push(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "private_source_pushed", "git_commit_sha": "e" * 40}

    async def _fake_wait(
        config: Any,
        *,
        expected_git_sha: str,
        timeout_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> tuple[FakeArtifact, dict[str, Any]]:
        assert expected_git_sha == "e" * 40
        return activation_artifact, {
            "status": "manifest_ready",
            "expected_git_sha": expected_git_sha,
            "current_json_git_sha": activation_artifact.git_commit_sha,
            "current_json_manifest_hash": activation_artifact.manifest_hash,
            "current_json_model_artifact_hash": activation_artifact.model_artifact_hash,
            "current_json_image_digest": activation_artifact.image_digest,
            "manifest_uri": activation_artifact.manifest_uri,
        }

    async def _fake_reward(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"champion_reward_status": "created", "champion_reward_id": "cr-1"}

    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_push_private_repo_candidate", _fake_push)
    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_create_champion_reward", _fake_reward)
    monkeypatch.setattr(promotion, "wait_for_current_manifest_git_sha", _fake_wait)

    controller = ResearchLabPromotionController(_controller_config(auto_commit_enabled=True), worker_ref="test-worker")
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        "candidate_source_diff_hash": "sha256:" + "2" * 64,
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }
    score_bundle = _bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id)
    result = await controller._promote_built_image_candidate(
        candidate=candidate,
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=score_bundle,
        active=ActivePrivateModel(artifact=parent, version_row=_active_row(parent)),
        active_parent=parent.model_artifact_hash,
        candidate_parent=parent.model_artifact_hash,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert result["status"] == "merged"
    assert len(store.private_benchmark_writes) == 1
    benchmark_write = store.private_benchmark_writes[0]
    assert benchmark_write["private_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert benchmark_write["private_model_manifest_hash"] == activation_artifact.manifest_hash
    assert benchmark_write["aggregate_score"] == pytest.approx(28.472727)
    assert benchmark_write["benchmark_quality"] == "passed"
    summary_doc = benchmark_write["score_summary_doc"]
    assert summary_doc["source"] == "promoted_candidate_score_bundle"
    assert summary_doc["derived_from_candidate_score"] is True
    assert summary_doc["source_score_bundle_id"] == "score_bundle:" + "7" * 64
    assert summary_doc["source_candidate_artifact_hash"] == candidate_artifact.model_artifact_hash
    assert summary_doc["activation_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert summary_doc["activation_manifest_hash"] == activation_artifact.manifest_hash
    assert summary_doc["activation_git_commit_sha"] == "e" * 40
    assert summary_doc["activation_artifact_differs_from_scored_candidate"] is True
    assert len(store.public_report_writes) == 1
    report_doc = store.public_report_writes[0]["report_doc"]
    assert report_doc["aggregate_score"] == pytest.approx(28.472727)
    assert report_doc["source"] == "promoted_candidate_score_bundle"
    assert report_doc["source_candidate_artifact_hash"] == candidate_artifact.model_artifact_hash
    assert report_doc["activation_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert report_doc["activation_artifact_differs_from_scored_candidate"] is True
    assert report_doc["public_icps"][0]["score"] == pytest.approx(25.0)
    assert len(store.version_writes) == 1
    assert store.version_writes[0]["source_benchmark_bundle_id"] == "private_benchmark:" + "8" * 64
    assert store.version_writes[0]["manifest_uri"] == activation_artifact.manifest_uri
    assert store.version_writes[0]["artifact_manifest"]["git_commit_sha"] == "e" * 40
    assert store.version_writes[0]["artifact_manifest"]["model_artifact_hash"] == (
        activation_artifact.model_artifact_hash
    )
    version_doc = store.version_writes[0]["redacted_version_doc"]
    _assert_db_doc_safe(version_doc)
    assert version_doc["image_ref_hash"].startswith("sha256:")
    assert version_doc["manifest_wait_status"]["current_json_image_ref_hash"].startswith("sha256:")
    active_events = [event for event in store.promotion_event_writes if event["event_type"] == "active_version_created"]
    _assert_db_doc_safe(active_events[0]["event_doc"])
    assert active_events[0]["event_doc"]["derived_benchmark_bundle_id"] == "private_benchmark:" + "8" * 64
    assert active_events[0]["event_doc"]["scored_candidate_model_artifact_hash"] == (
        candidate_artifact.model_artifact_hash
    )
    assert active_events[0]["event_doc"]["new_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert active_events[0]["event_doc"]["new_image_ref_hash"].startswith("sha256:")


async def test_promoted_candidate_source_push_pending_leaves_previous_active_model_active(store, monkeypatch):
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    async def _fake_push(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "private_source_pushed", "git_commit_sha": "e" * 40}

    async def _fake_wait(
        config: Any,
        *,
        expected_git_sha: str,
        timeout_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> tuple[None, dict[str, Any]]:
        assert expected_git_sha == "e" * 40
        return None, {
            "status": "source_pushed_manifest_pending",
            "expected_git_sha": expected_git_sha,
            "current_json_git_sha": "d" * 40,
        }

    async def _fake_reward(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("champion reward must not be created while source manifest is pending")

    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_push_private_repo_candidate", _fake_push)
    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_create_champion_reward", _fake_reward)
    monkeypatch.setattr(promotion, "wait_for_current_manifest_git_sha", _fake_wait)

    controller = ResearchLabPromotionController(_controller_config(auto_commit_enabled=True), worker_ref="test-worker")
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        "candidate_source_diff_hash": "sha256:" + "2" * 64,
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }
    result = await controller._promote_built_image_candidate(
        candidate=candidate,
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id),
        active=ActivePrivateModel(artifact=parent, version_row=_active_row(parent)),
        active_parent=parent.model_artifact_hash,
        candidate_parent=parent.model_artifact_hash,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert result["status"] == "source_pushed_manifest_pending"
    assert store.private_benchmark_writes == []
    assert store.public_report_writes == []
    assert store.version_writes == []
    pending_events = [
        event
        for event in store.promotion_event_writes
        if (event.get("event_doc") or {}).get("reason") == "source_pushed_manifest_pending"
    ]
    assert len(pending_events) == 1
    assert pending_events[0]["event_doc"]["action"] == (
        "leave_previous_active_model_active_until_current_json_matches_pushed_commit"
    )


async def test_promoted_candidate_bridge_reuses_existing_rows_without_duplicate_writes(store):
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    existing_benchmark_id = "private_benchmark:" + "8" * 64
    existing_report_id = "public_benchmark:sha256:" + "9" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    def _private_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("private_model_manifest_hash") == candidate_artifact.manifest_hash:
            return [
                {
                    "benchmark_bundle_id": existing_benchmark_id,
                    "current_benchmark_status": "completed",
                    "aggregate_score": 28.472727,
                }
            ]
        return []

    def _public_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("private_model_manifest_hash") == candidate_artifact.manifest_hash:
            return [
                {
                    "report_id": existing_report_id,
                    "benchmark_bundle_id": existing_benchmark_id,
                    "current_report_status": "published",
                    "aggregate_score": 28.472727,
                }
            ]
        if filters.get("benchmark_bundle_id") == baseline_bundle_id:
            return [_bridge_public_report_row(baseline_bundle_id)]
        return []

    store.select_many_results["research_lab_private_model_benchmark_current"] = _private_rows
    store.select_many_results["research_lab_public_benchmark_report_current"] = _public_rows
    controller = ResearchLabPromotionController(_controller_config(), worker_ref="test-worker")
    bridge = await controller._create_promoted_candidate_benchmark_bridge(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id),
        new_artifact=candidate_artifact,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert bridge["status"] == "already_exists"
    assert bridge["benchmark_bundle_id"] == existing_benchmark_id
    assert bridge["public_report_id"] == existing_report_id
    assert store.private_benchmark_writes == []
    assert store.public_report_writes == []


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
# Failed private-source push reconciler
# ---------------------------------------------------------------------------


def _failed_private_source_push_rows(store: FakeStore, *, created_at: str = "2026-07-01T00:00:00+00:00") -> None:
    store.select_many_results["research_lab_candidate_promotion_events:promotion_failed"] = [
        {
            "promotion_event_id": "pe-failed",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "promotion_failed",
            "promotion_status": "failed",
            "event_doc": {"reason": "private_source_push_failed", "source_push_attempt": 1},
            "created_at": created_at,
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []
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
            "candidate_artifact_hash": "sha256:" + "c" * 64,
            "parent_artifact_hash": "sha256:" + "a" * 64,
            "icp_set_hash": "sha256:" + "3" * 64,
            "evaluation_epoch": 23770,
            "score_bundle_hash": "sha256:" + "5" * 64,
            "aggregates": {},
        },
    }


async def test_private_source_push_reconciler_dry_run_plans_retry(store):
    _failed_private_source_push_rows(store)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=True,
        retry_after_seconds=0,
    )

    assert result["ok"] is True
    assert result["found_failed"] == 1
    assert result["results"][0]["status"] == "would_retry_private_source_push"
    assert store.promotion_event_writes == []
    assert store.version_writes == []
    assert store.reward_obligation_writes == []


async def test_private_source_push_reconciler_candidate_filter_queries_candidate_directly(store):
    _failed_private_source_push_rows(store)

    await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        candidate_ids=["cand-1"],
        dry_run=True,
        retry_after_seconds=0,
    )

    assert (
        "research_lab_candidate_promotion_events",
        (("event_type", "promotion_failed"), ("candidate_id", "cand-1")),
    ) in store.select_many_calls


async def test_private_source_push_reconciler_applies_retry_through_promotion_path(store, monkeypatch):
    _failed_private_source_push_rows(store)
    calls: list[dict[str, Any]] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "merged",
            "private_model_version_id": "pmv-1",
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        }

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["retried"] == 1
    assert result["finalized"] == 1
    assert result["results"][0]["status"] == "merged"
    assert calls[0]["candidate"]["candidate_id"] == "cand-1"
    assert calls[0]["score_bundle_row"]["score_bundle_id"] == "sb-1"


async def test_private_source_push_reconciler_marks_fresh_stale_parent_for_rebase(store, monkeypatch):
    _failed_private_source_push_rows(store)

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "stale_parent_needs_rescore"}

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["retried"] == 1
    entry = result["results"][0]
    assert entry["status"] == "stale_parent_needs_rescore"
    assert entry["stale_parent_rebase_eligible"] is True
    assert entry["stale_parent_recovery_event_status"] == "marked"
    assert store.candidate_evaluation_event_writes == [
        {
            "candidate_id": "cand-1",
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "event_type": "rejected",
            "candidate_status": "rejected",
            "evaluator_ref": "test-reconciler",
            "reason": "stale_parent_needs_rescore",
            "score_bundle_id": "sb-1",
            "event_doc": store.candidate_evaluation_event_writes[0]["event_doc"],
        }
    ]
    assert store.candidate_evaluation_event_writes[0]["event_doc"]["reason"] == (
        "private_source_push_failed_retry_stale_parent"
    )
    assert store.scoring_dispatch_event_writes[0]["dispatch_status"] == "rejected"
    assert store.scoring_dispatch_event_writes[0]["event_doc"]["dispatch_context"] == (
        "private_source_push_reconcile"
    )


async def test_private_source_push_reconciler_marks_existing_stale_parent_event_for_rebase(store, monkeypatch):
    _failed_private_source_push_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": "pe-stale",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "stale_parent_detected",
            "promotion_status": "rebase_required",
            "event_doc": {"reason": "stale_parent_needs_rescore"},
            "created_at": "2026-07-01T00:02:00+00:00",
        }
    ]

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("existing stale-parent promotion event should not be retried")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    entry = result["results"][0]
    assert entry["status"] == "stale_parent_needs_rescore"
    assert entry["latest_promotion_event_id"] == "pe-stale"
    assert entry["stale_parent_rebase_eligible"] is True
    assert store.candidate_evaluation_event_writes[0]["reason"] == "stale_parent_needs_rescore"
    assert store.candidate_evaluation_event_writes[0]["event_doc"]["latest_promotion_event_id"] == "pe-stale"


async def test_private_source_push_reconciler_respects_event_backoff(store, monkeypatch):
    _failed_private_source_push_rows(store, created_at=datetime.now(timezone.utc).isoformat())

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("retry should be held by backoff")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=300,
    )

    assert result["results"][0]["status"] == "retry_backoff"
    assert result["retried"] == 0


async def test_private_source_push_reconciler_skips_already_rewarded(store, monkeypatch):
    _failed_private_source_push_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": "pe-created",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "champion_reward_created",
            "promotion_status": "reward_created",
            "event_doc": {},
            "created_at": "2026-07-01T00:01:00+00:00",
        }
    ]

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("already rewarded candidates should not be retried")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["results"][0]["status"] == "already_rewarded"
    assert result["retried"] == 0


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
