"""Fail-closed scoring: health-gate quarantine, baseline publication gates,
and the day-jump default.

Covers the priority-#5 hardening:
- the scoring health gate honors ``scoring_health_gate_enabled`` and returns a
  ``quarantine`` decision on threshold violations instead of observe-only,
- a quarantined candidate records a ``scoring_health_quarantined`` promotion
  event and is never handed to the promotion controller,
- a baseline whose own health gate failed refuses publication, and the
  day-over-day jump limit is enforced by default.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.models import _score_bundle_hash


_SCORE_BUNDLE_DOCS_BY_HASH: dict[str, dict] = {}


def _worker(**config_overrides) -> sw.ResearchLabGatewayScoringWorker:
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    defaults = dict(
        scoring_health_gate_enabled=True,
        scoring_health_max_reference_runtime_failure_rate=0.25,
        scoring_health_max_candidate_runtime_failure_rate=0.25,
        scoring_health_max_reference_zero_company_rate=1.0,
        scoring_health_max_candidate_zero_company_rate=1.0,
        scoring_health_max_provider_error_rate=0.10,
        scoring_health_max_timeout_rate=0.10,
        improvement_threshold_points=1.0,
    )
    defaults.update(config_overrides)
    worker.config = SimpleNamespace(**defaults)
    worker.worker_ref = "test-scorer-0"
    return worker


def _bundle(provider_error_rate: float = 0.0, **health) -> dict:
    doc = {"provider_error_rate": provider_error_rate, **health}
    return {"scoring_health": doc, "icp_set_hash": "sha256:w", "aggregates": {}}


def _candidate_identity() -> dict:
    build_doc = {
        "build_doc_hash": "sha256:build",
        "loop_node_id": "loop-node-1",
    }
    patch_payload = {
        "candidate_kind": "image_build",
        "patch_type": "IMAGE_BUILD",
        "parent_artifact_hash": "sha256:parent",
        "candidate_artifact_hash": "sha256:a",
        "candidate_model_manifest_hash": "sha256:candidate-manifest",
        "patch_payload_hash": "sha256:source-diff",
        "candidate_source_diff_hash": "sha256:source-diff",
        "candidate_build_doc_hash": "sha256:build",
    }
    patch = {**patch_payload, "manifest_hash": sw.sha256_json(patch_payload)}
    return {
        "candidate_id": "candidate:a",
        "run_id": "run-1",
        "ticket_id": "11111111-1111-4111-8111-111111111111",
        "receipt_id": "22222222-2222-4222-8222-222222222222",
        "miner_hotkey": "miner-1",
        "island": "generalist",
        "parent_artifact_hash": "sha256:parent",
        "candidate_artifact_hash": "sha256:a",
        "private_model_manifest_hash": "sha256:parent-manifest",
        "private_model_manifest_doc": {
            "manifest_hash": "sha256:parent-manifest",
        },
        "candidate_patch_hash": sw.sha256_json(patch),
        "candidate_patch_manifest": patch,
        "candidate_kind": "image_build",
        "candidate_model_manifest_hash": "sha256:candidate-manifest",
        "candidate_model_manifest_doc": {
            "manifest_hash": "sha256:candidate-manifest",
        },
        "candidate_source_diff_hash": "sha256:source-diff",
        "candidate_build_doc": build_doc,
    }


def _parent_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        model_artifact_hash="sha256:parent",
        manifest_hash="sha256:parent-manifest",
    )


def _candidate_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        model_artifact_hash="sha256:a",
        manifest_hash="sha256:candidate-manifest",
    )


def _find_reusable(worker, *, evaluation_epoch: int):
    return worker._find_reusable_scored_bundle(
        candidate=_candidate_identity(),
        artifact=_parent_artifact(),
        candidate_artifact=_candidate_artifact(),
        evaluation_epoch=evaluation_epoch,
    )


def _signed_score_bundle_doc(
    *,
    evaluation_epoch: int,
    provider_error_rate: float = 0.0,
    identity: str = "default",
) -> dict:
    candidate = _candidate_identity()
    doc = {
        "bundle_type": "research_lab_evaluation_score_bundle",
        "schema_version": "1.1",
        "execution_trace_ref": (
            "execution_trace:"
            + sw.execution_trace_id_for_node("run-1", "loop-node-1")
        ),
        "scoring_health": {"provider_error_rate": provider_error_rate},
        "evaluation_epoch": evaluation_epoch,
        "run_id": "run-1",
        "ticket_id": "11111111-1111-4111-8111-111111111111",
        "miner_hotkey": "miner-1",
        "island": "generalist",
        "parent_artifact_hash": "sha256:parent",
        "candidate_artifact_hash": "sha256:a",
        "private_model_manifest_hash": "sha256:parent-manifest",
        "candidate_patch_hash": candidate["candidate_patch_hash"],
        "candidate_model_manifest_hash": "sha256:candidate-manifest",
        "candidate_source_diff_hash": "sha256:source-diff",
        "candidate_build_ref": "sha256:build",
        "icp_set_hash": "sha256:window",
        "identity": identity,
        "serving_model_version": {
            "candidate_id": "candidate:a",
            "run_id": "run-1",
            "ticket_id": "11111111-1111-4111-8111-111111111111",
            "candidate_patch_hash": candidate["candidate_patch_hash"],
            "candidate_source_diff_hash": "sha256:source-diff",
            "candidate_build_ref": "sha256:build",
            "parent_model": {
                "model_artifact_hash": "sha256:parent",
                "manifest_hash": "sha256:parent-manifest",
            },
            "candidate_model": {
                "model_artifact_hash": "sha256:a",
                "manifest_hash": "sha256:candidate-manifest",
            },
        },
        "reward_path": {
            "eligible_for_crown": False,
            "eligible_for_improvement_grant": False,
        },
    }
    score_bundle_hash = _score_bundle_hash(doc)
    signed = {
        **doc,
        "score_bundle_hash": score_bundle_hash,
        "anchored_hash": score_bundle_hash,
        "signature_ref": "s3://sig",
    }
    _SCORE_BUNDLE_DOCS_BY_HASH[score_bundle_hash] = signed
    return signed


def _durable_score_bundle_row(
    doc: dict,
    *,
    current_event_status: str | None,
) -> dict:
    score_bundle_hash = str(doc["score_bundle_hash"])
    _SCORE_BUNDLE_DOCS_BY_HASH[score_bundle_hash] = doc
    return {
        "score_bundle_id": "score_bundle:" + score_bundle_hash.split(":", 1)[1],
        "score_bundle_hash": score_bundle_hash,
        "anchored_hash": score_bundle_hash,
        "score_bundle_doc": doc,
        "signature_ref": "s3://sig",
        "receipt_id": "22222222-2222-4222-8222-222222222222",
        "ticket_id": str(doc["ticket_id"]),
        "miner_hotkey": str(doc["miner_hotkey"]),
        "island": str(doc["island"]),
        "run_id": str(doc["run_id"]),
        "parent_artifact_hash": str(doc["parent_artifact_hash"]),
        "candidate_artifact_hash": str(doc["candidate_artifact_hash"]),
        "private_model_manifest_hash": str(doc["private_model_manifest_hash"]),
        "candidate_patch_hash": str(doc["candidate_patch_hash"]),
        "icp_set_hash": str(doc["icp_set_hash"]),
        "evaluation_epoch": int(doc["evaluation_epoch"]),
        "bundle_status": "scored",
        "current_event_status": current_event_status,
    }


def _current_reader(*rows: dict, status: str | None = None):
    async def read_current(*_args, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if str(filters.get("candidate_id") or ""):
            return _durable_candidate_row()
        score_bundle_id = str(filters.get("score_bundle_id") or "")
        row = next(
            item for item in rows if str(item["score_bundle_id"]) == score_bundle_id
        )
        return {
            "score_bundle_id": row["score_bundle_id"],
            "score_bundle_hash": row["score_bundle_hash"],
            "anchored_hash": row["anchored_hash"],
            "current_event_status": (
                status if status is not None else row.get("current_event_status")
            ),
        }

    return read_current


def _durable_candidate_row() -> dict:
    return dict(_candidate_identity())


def _valid_lineage_root(doc: dict) -> dict:
    return {
        "role": "gateway_scoring",
        "purpose": "research_lab.candidate_score.v2",
        "status": "succeeded",
        "epoch_id": int(doc["evaluation_epoch"]),
        "sequence": 0,
        "output_root": sw.sha256_json({"score_bundle": dict(doc)}),
    }


@pytest.fixture(autouse=True)
def _valid_recovery_authority(monkeypatch):
    async def default_select_one(table, **kwargs):
        if table == "research_lab_candidate_artifacts":
            return _durable_candidate_row()
        raise AssertionError(f"unexpected unmocked select_one: {table}")

    async def resolve_lineage(*, artifact_kind, artifact_ref, artifact_hash):
        assert artifact_kind == "score_bundle"
        assert artifact_ref == "score_bundle:" + artifact_hash.split(":", 1)[1]
        doc = _SCORE_BUNDLE_DOCS_BY_HASH[artifact_hash]
        root = _valid_lineage_root(doc)
        return root, [root]

    monkeypatch.setattr(sw, "select_one", default_select_one)
    monkeypatch.setattr(sw, "resolve_attested_artifact_lineage", resolve_lineage)


# ---------------------------------------------------------------------------
# _scoring_health_gate_result decisions
# ---------------------------------------------------------------------------


def test_gate_quarantines_on_violation_when_enabled():
    result = _worker()._scoring_health_gate_result(_bundle(provider_error_rate=0.5))
    assert result["enabled"] is True
    assert result["decision"] == "quarantine"
    assert result["would_quarantine"] is True
    assert any(v["metric"] == "provider_error_rate" for v in result["violations"])


def test_gate_observes_only_when_disabled():
    result = _worker(scoring_health_gate_enabled=False)._scoring_health_gate_result(
        _bundle(provider_error_rate=0.5)
    )
    assert result["enabled"] is False
    assert result["decision"] == "observe_only"
    assert result["would_quarantine"] is True  # still recorded for audit


def test_gate_observes_only_without_violations():
    result = _worker()._scoring_health_gate_result(_bundle(provider_error_rate=0.0))
    assert result["decision"] == "observe_only"
    assert result["violations"] == []


def test_gate_zero_company_rate_not_a_violation_at_default_threshold():
    # Legitimate zero-company outcomes must not quarantine at the defaults.
    result = _worker()._scoring_health_gate_result(
        _bundle(candidate_zero_company_rate=0.9, reference_zero_company_rate=0.9)
    )
    assert result["decision"] == "observe_only"


def test_gate_timeout_violation_quarantines():
    result = _worker()._scoring_health_gate_result(_bundle(timeout_rate=0.5))
    assert result["decision"] == "quarantine"


# ---------------------------------------------------------------------------
# quarantine recorder: event written, promotion withheld
# ---------------------------------------------------------------------------


def test_quarantine_recorder_writes_promotion_event_and_skips_promotion():
    worker = _worker()
    candidate = {
        "candidate_id": "cand-1",
        "parent_artifact_hash": "sha256:parent",
        "candidate_kind": "auto_research",
    }
    gate = {"decision": "quarantine", "violations": [{"metric": "provider_error_rate"}]}
    recorded = {}

    async def fake_promotion_event(**kwargs):
        recorded.update(kwargs)
        return {"event_id": "e1"}

    with mock.patch.object(sw, "create_candidate_promotion_event", fake_promotion_event), \
            mock.patch.object(
                sw.ResearchLabGatewayScoringWorker,
                "_maybe_promote_scored_candidate",
                side_effect=AssertionError("promotion must not run for quarantined candidates"),
            ):
        result = asyncio.run(
            worker._record_scoring_health_quarantined(
                candidate=candidate,
                score_bundle_row={"score_bundle_id": "sb-1"},
                score_bundle=_bundle(),
                scoring_health_gate=gate,
            )
        )
    assert result == {"status": "scoring_health_quarantined"}
    assert recorded["event_type"] == "scoring_health_quarantined"
    assert recorded["promotion_status"] == "rejected"
    assert recorded["candidate_id"] == "cand-1"
    assert recorded["event_doc"]["scoring_health_gate"] == gate


# ---------------------------------------------------------------------------
# baseline publication gates
# ---------------------------------------------------------------------------


def _health(gate_passed: bool, unresolved: int = 0) -> dict:
    return {
        "gate_passed": gate_passed,
        "unresolved_provider_errors": unresolved,
        "max_unresolved_icps": 2,
        "decision": "observe_only",
    }


def test_baseline_gate_blocks_degraded_publication():
    with pytest.raises(sw.BaselineHealthGateFailure) as excinfo:
        sw._enforce_baseline_publication_gates(
            baseline_health=_health(False, unresolved=5),
            aggregate_score=24.46,
            day_jump_points=None,
            health_gate_enforced=True,
            max_day_jump=15.0,
        )
    assert "unresolved_provider_errors_gate_failed" in str(excinfo.value)
    assert excinfo.value.baseline_health["unresolved_provider_errors"] == 5


def test_baseline_gate_allows_healthy_publication():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(True),
        aggregate_score=34.35,
        day_jump_points=2.0,
        health_gate_enforced=True,
        max_day_jump=15.0,
    )


def test_baseline_gate_observe_only_when_disabled():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(False, unresolved=5),
        aggregate_score=24.46,
        day_jump_points=None,
        health_gate_enforced=False,
        max_day_jump=None,
    )


def test_day_jump_gate_blocks_large_swing():
    with pytest.raises(sw.BaselineHealthGateFailure) as excinfo:
        sw._enforce_baseline_publication_gates(
            baseline_health=_health(True),
            aggregate_score=10.0,
            day_jump_points=-24.0,
            health_gate_enforced=True,
            max_day_jump=15.0,
        )
    assert "day_over_day_jump_gate_failed" in str(excinfo.value)
    assert excinfo.value.baseline_health["gate_passed"] is False


def test_day_jump_gate_allows_swing_within_limit():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(True),
        aggregate_score=34.35,
        day_jump_points=9.9,
        health_gate_enforced=True,
        max_day_jump=15.0,
    )


# ---------------------------------------------------------------------------
# day-jump threshold default
# ---------------------------------------------------------------------------


def test_day_jump_threshold_defaults_enforced(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", raising=False)
    assert sw._baseline_max_day_jump_points() == sw.DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS


@pytest.mark.parametrize("raw", ["0", "0.0", "off", "none", "disabled", "OFF"])
def test_day_jump_threshold_explicit_disable(monkeypatch, raw):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", raw)
    assert sw._baseline_max_day_jump_points() is None


def test_day_jump_threshold_custom_value(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "-25.5")
    assert sw._baseline_max_day_jump_points() == 25.5


def test_day_jump_threshold_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "not-a-number")
    assert sw._baseline_max_day_jump_points() == sw.DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS


# ---------------------------------------------------------------------------
# config defaults
# ---------------------------------------------------------------------------


def test_config_defaults_fail_closed(monkeypatch):
    from gateway.research_lab.config import ResearchLabGatewayConfig

    for name in (
        "RESEARCH_LAB_SCORING_HEALTH_GATE_ENABLED",
        "RESEARCH_LAB_BASELINE_HEALTH_GATE_ENFORCED",
        "RESEARCH_LAB_SCORING_HEALTH_MAX_REFERENCE_ZERO_COMPANY_RATE",
        "RESEARCH_LAB_SCORING_HEALTH_MAX_CANDIDATE_ZERO_COMPANY_RATE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = ResearchLabGatewayConfig.from_env()
    assert config.scoring_health_gate_enabled is True
    assert config.baseline_health_gate_enforced is True
    assert config.scoring_health_max_reference_zero_company_rate == 1.0
    assert config.scoring_health_max_candidate_zero_company_rate == 1.0


# ---------------------------------------------------------------------------
# provider-recovery rescore (quarantine requeue)
# ---------------------------------------------------------------------------


def _recovery_worker(monkeypatch, **overrides):
    worker = _worker(**overrides)
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_INTERVAL_SECONDS", "0")
    return worker


def test_recovery_requeues_quarantined_scored_candidate(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if table == "research_lab_candidate_promotion_events":
            if filters.get("event_type") == "scoring_health_quarantined":
                return [{"candidate_id": "cand-q", "created_at": "2026-07-10T10:00:00"}]
            # latest promotion for the candidate is still the quarantine
            return [{"event_type": "scoring_health_quarantined", "created_at": "2026-07-10T10:00:00"}]
        if table == "research_lab_candidate_evaluation_events":
            return []  # no prior recovery attempts
        raise AssertionError(f"unexpected table {table}")

    async def fake_select_one(table, **kwargs):
        assert table == "research_lab_candidate_evaluation_current"
        return {
            "candidate_id": "cand-q",
            "current_candidate_status": "scored",
            "run_id": "run-1",
            "ticket_id": "tick-1",
        }

    async def fake_create_event(**kwargs):
        writes.append(kwargs)
        return {"event_id": "e1"}

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 1
    assert writes[0]["event_type"] == "queued"
    assert writes[0]["candidate_status"] == "queued"
    assert writes[0]["reason"] == "provider_recovery_rescore"
    assert writes[0]["candidate_id"] == "cand-q"


def test_recovery_skips_candidate_already_requeued(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "queued"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_skips_when_quarantine_superseded(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        if table == "research_lab_candidate_promotion_events":
            return [{"event_type": "merged", "created_at": "t2"}]  # superseded
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "scored"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_respects_attempt_cap(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_MAX_ATTEMPTS", "2")
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        if table == "research_lab_candidate_promotion_events":
            return [{"event_type": "scoring_health_quarantined", "created_at": "t"}]
        if table == "research_lab_candidate_evaluation_events":
            return [{"event_id": "r1"}, {"event_id": "r2"}]  # cap reached
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "scored"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_disabled_with_gate(monkeypatch):
    worker = _recovery_worker(monkeypatch, scoring_health_gate_enabled=False)

    async def boom(*a, **k):
        raise AssertionError("must not query when gate disabled")

    with mock.patch.object(sw, "select_many", boom):
        assert asyncio.run(worker._requeue_quarantined_candidates()) == 0


def test_recovery_interval_throttles(monkeypatch):
    worker = _worker()
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_INTERVAL_SECONDS", "3600")
    calls = {"n": 0}

    async def fake_select_many(table, **kwargs):
        calls["n"] += 1
        return []

    with mock.patch.object(sw, "select_many", fake_select_many):
        asyncio.run(worker._requeue_quarantined_candidates())
        asyncio.run(worker._requeue_quarantined_candidates())
    assert calls["n"] == 1  # second call throttled by the interval


def test_reusable_bundle_skips_quarantine_worthy_bundle():
    worker = _worker()
    degraded_doc = _signed_score_bundle_doc(
        evaluation_epoch=1,
        provider_error_rate=0.9,
    )

    durable_row = _durable_score_bundle_row(
        degraded_doc, current_event_status="scored"
    )

    async def fake_select_many(table, **kwargs):
        return [durable_row]

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))
    assert row is None  # degraded bundle must not be reused


def test_reusable_bundle_returns_healthy_bundle():
    worker = _worker()
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=1)

    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="scored"
    )

    async def fake_select_many(table, **kwargs):
        return [durable_row]

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))
    assert row is not None
    assert row["score_bundle_id"].startswith("score_bundle:")


def test_reusable_bundle_rejects_malformed_matching_document():
    worker = _worker()
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="scored"
    )
    durable_row["score_bundle_doc"] = {}

    with mock.patch.object(
        sw, "select_many", mock.AsyncMock(return_value=[durable_row])
    ):
        with pytest.raises(RuntimeError, match="malformed durable score bundle"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reusable_bundle_rejects_bounded_lookup_overflow():
    worker = _worker()
    rows = [
        _durable_score_bundle_row(
            _signed_score_bundle_doc(evaluation_epoch=index, identity=str(index)),
            current_event_status="scored",
        )
        for index in range(6)
    ]
    with mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=rows)):
        with pytest.raises(RuntimeError, match="exceeded its bounded scan"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=6))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("private_model_manifest_hash", "sha256:different-parent-manifest"),
        ("candidate_model_manifest_hash", "sha256:different-candidate-manifest"),
        ("candidate_artifact_hash", "sha256:different-candidate"),
        ("candidate_source_diff_hash", "sha256:different-source"),
        ("candidate_build_doc", {"build_doc_hash": "sha256:different-build"}),
    ],
)
def test_reusable_bundle_rejects_different_immutable_candidate_row(
    field,
    replacement,
):
    worker = _worker()
    durable_candidate = _durable_candidate_row()
    durable_candidate[field] = replacement

    async def select_candidate(table, **kwargs):
        assert table == "research_lab_candidate_artifacts"
        return durable_candidate

    with mock.patch.object(sw, "select_one", select_candidate):
        with pytest.raises(RuntimeError, match="immutable candidate row"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("role", "gateway_coordinator"),
        ("purpose", "research_lab.baseline_score.v2"),
        ("status", "failed"),
        ("epoch_id", 2),
        ("sequence", 1),
        ("output_root", "sha256:" + "f" * 64),
    ],
)
def test_reusable_bundle_requires_exact_v2_score_authority(field, replacement):
    worker = _worker()
    doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(doc, current_event_status="scored")

    async def wrong_lineage(**_kwargs):
        root = {**_valid_lineage_root(doc), field: replacement}
        return root, [root]

    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=[durable_row])),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
        mock.patch.object(sw, "resolve_attested_artifact_lineage", wrong_lineage),
    ):
        with pytest.raises(RuntimeError, match="lacks exact V2 score authority"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reusable_bundle_legacy_history_uses_legacy_trace_binding():
    worker = _worker()
    candidate = _candidate_identity()
    candidate["candidate_build_doc"] = {"build_doc_hash": "sha256:build"}
    doc = _signed_score_bundle_doc(evaluation_epoch=1)
    legacy_trace = "gateway_qualification_worker:old-worker:candidate:a"
    changed = {**doc, "execution_trace_ref": legacy_trace}
    changed_hash = _score_bundle_hash(changed)
    changed = {
        **changed,
        "score_bundle_hash": changed_hash,
        "anchored_hash": changed_hash,
    }
    durable_row = _durable_score_bundle_row(changed, current_event_status="scored")
    durable_candidate = _durable_candidate_row()
    durable_candidate["candidate_build_doc"] = candidate["candidate_build_doc"]

    async def read(table, **kwargs):
        if table == "research_lab_candidate_artifacts":
            return durable_candidate
        return await _current_reader(durable_row)(table, **kwargs)

    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=[durable_row])),
        mock.patch.object(sw, "select_one", read),
    ):
        row = asyncio.run(
            worker._find_reusable_scored_bundle(
                candidate=candidate,
                artifact=_parent_artifact(),
                candidate_artifact=_candidate_artifact(),
                evaluation_epoch=1,
            )
        )
    assert row is not None


def test_reusable_bundle_rejects_arbitrary_git_tree_execution_trace():
    worker = _worker()
    doc = _signed_score_bundle_doc(evaluation_epoch=1)
    changed = {**doc, "execution_trace_ref": "execution_trace:arbitrary"}
    changed_hash = _score_bundle_hash(changed)
    changed = {
        **changed,
        "score_bundle_hash": changed_hash,
        "anchored_hash": changed_hash,
    }
    durable_row = _durable_score_bundle_row(changed, current_event_status="scored")
    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=[durable_row])),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        with pytest.raises(RuntimeError, match="candidate lineage"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reusable_bundle_rereads_terminal_state_even_when_snapshot_was_scored():
    worker = _worker()
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="scored"
    )

    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=[durable_row])),
        mock.patch.object(
            sw,
            "select_one",
            _current_reader(durable_row, status="tombstoned"),
        ),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))

    assert row is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("receipt_id", "33333333-3333-4333-8333-333333333333"),
        ("ticket_id", "33333333-3333-4333-8333-333333333333"),
        ("miner_hotkey", "different-miner"),
        ("island", "specialist"),
        ("parent_artifact_hash", "sha256:different-parent"),
        ("private_model_manifest_hash", "sha256:different-parent-manifest"),
        ("candidate_patch_hash", "sha256:different-patch"),
    ],
)
def test_reusable_bundle_rejects_different_durable_candidate_lineage(
    field,
    replacement,
):
    worker = _worker()
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="scored"
    )
    durable_row[field] = replacement

    with mock.patch.object(
        sw,
        "select_many",
        mock.AsyncMock(return_value=[durable_row]),
    ):
        with pytest.raises(RuntimeError, match="durable identity"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("candidate_id",), "candidate:different"),
        (("run_id",), "different-run"),
        (("ticket_id",), "33333333-3333-4333-8333-333333333333"),
        (("candidate_patch_hash",), "sha256:different-patch"),
        (("candidate_source_diff_hash",), "sha256:different-source"),
        (("candidate_build_ref",), "sha256:different-build"),
        (("parent_model", "model_artifact_hash"), "sha256:different-parent"),
        (("parent_model", "manifest_hash"), "sha256:different-parent-manifest"),
        (("candidate_model", "model_artifact_hash"), "sha256:different-candidate"),
        (("candidate_model", "manifest_hash"), "sha256:different-candidate-manifest"),
    ],
)
def test_reusable_bundle_rejects_different_signed_serving_lineage(path, replacement):
    worker = _worker()
    original = _signed_score_bundle_doc(evaluation_epoch=1)
    serving = {
        **original["serving_model_version"],
        "parent_model": dict(original["serving_model_version"]["parent_model"]),
        "candidate_model": dict(original["serving_model_version"]["candidate_model"]),
    }
    target = serving
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement
    changed = {**original, "serving_model_version": serving}
    changed_hash = _score_bundle_hash(changed)
    changed = {
        **changed,
        "score_bundle_hash": changed_hash,
        "anchored_hash": changed_hash,
    }
    durable_row = _durable_score_bundle_row(
        changed, current_event_status="scored"
    )

    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=[durable_row])),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        with pytest.raises(RuntimeError, match="candidate lineage"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reused_bundle_terminal_readback_blocks_all_recovery_side_effects():
    worker = _worker()
    worker.proxy_ref_hash = None
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="scored"
    )
    side_effect = mock.AsyncMock(
        side_effect=AssertionError("terminal bundle must not produce derivatives")
    )

    with (
        mock.patch.object(
            sw,
            "select_one",
            _current_reader(durable_row, status="tombstoned"),
        ),
        mock.patch.object(sw, "_persist_conditional_finalization_events", side_effect),
        mock.patch.object(sw, "_persist_candidate_category_results", side_effect),
        mock.patch.object(worker, "_create_scored_evaluation_event", side_effect),
    ):
        with pytest.raises(RuntimeError, match="no longer current and promotable"):
            asyncio.run(
                worker._complete_candidate_from_reused_bundle(
                    _candidate_identity(),
                    candidate_id=_candidate_identity()["candidate_id"],
                    bundle_row=durable_row,
                    evaluation_epoch=1,
                    start=0.0,
                )
            )

    side_effect.assert_not_awaited()


def test_reusable_bundle_survives_evaluation_epoch_rollover():
    worker = _worker()
    captured = {}
    healthy_doc = _signed_score_bundle_doc(evaluation_epoch=7)

    durable_row = _durable_score_bundle_row(
        healthy_doc, current_event_status="verified"
    )

    async def fake_select_many(table, **kwargs):
        captured.update(kwargs)
        return [durable_row]

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=8))

    assert row is not None
    assert row["current_event_status"] == "verified"
    assert ("evaluation_epoch", 8) not in captured["filters"]


def test_multiple_reusable_signed_bundles_fail_closed():
    worker = _worker()

    def row(bundle_id, epoch):
        doc = _signed_score_bundle_doc(
            evaluation_epoch=epoch,
            identity=bundle_id,
        )
        return _durable_score_bundle_row(doc, current_event_status="scored")

    rows = [row("first", 7), row("second", 8)]
    with (
        mock.patch.object(sw, "select_many", mock.AsyncMock(return_value=rows)),
        mock.patch.object(sw, "select_one", _current_reader(*rows)),
    ):
        with pytest.raises(RuntimeError, match="multiple healthy signed"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=8))


def test_reusable_bundle_lookup_failure_is_not_treated_as_not_found():
    worker = _worker()

    async def unavailable(*_args, **_kwargs):
        raise TimeoutError("score bundle read timed out")

    with mock.patch.object(sw, "select_many", unavailable):
        with pytest.raises(TimeoutError, match="score bundle read timed out"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


@pytest.mark.parametrize("current_status", ["failed", "rejected", "tombstoned"])
def test_reusable_bundle_does_not_revive_terminal_lifecycle_state(current_status):
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)

    async def fake_select_many(*_args, **_kwargs):
        return [
            _durable_score_bundle_row(
                score_bundle_doc,
                current_event_status=current_status,
            )
        ]

    with mock.patch.object(sw, "select_many", fake_select_many):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))
    assert row is None


def test_reusable_bundle_repairs_missing_opening_event_without_rescoring():
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        score_bundle_doc,
        current_event_status=None,
    )
    requests = []

    async def fake_select_many(*_args, **_kwargs):
        return [durable_row]

    async def fake_create_score_bundle(request):
        requests.append(request)
        assert request.score_bundle == score_bundle_doc
        return durable_row, {"event_status": "scored"}

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(
            sw,
            "select_one",
            _current_reader(durable_row, status="scored"),
        ),
        mock.patch.object(sw, "create_score_bundle", fake_create_score_bundle),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))

    assert row is not None
    assert row["current_event_status"] == "scored"
    assert len(requests) == 1


def test_reusable_bundle_missing_event_recovery_respects_concurrent_tombstone():
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        score_bundle_doc,
        current_event_status=None,
    )

    async def fake_select_many(*_args, **_kwargs):
        return [durable_row]

    async def fake_create_score_bundle(_request):
        return durable_row, {"event_status": "scored"}

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(
            sw,
            "select_one",
            _current_reader(durable_row, status="tombstoned"),
        ),
        mock.patch.object(sw, "create_score_bundle", fake_create_score_bundle),
    ):
        row = asyncio.run(_find_reusable(worker, evaluation_epoch=1))
    assert row is None


def test_reusable_bundle_rejects_altered_body_with_stale_stored_hash():
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        score_bundle_doc,
        current_event_status="scored",
    )
    durable_row["score_bundle_doc"] = {
        **score_bundle_doc,
        "identity": "altered-after-hashing",
    }

    async def fake_select_many(*_args, **_kwargs):
        return [durable_row]

    with mock.patch.object(sw, "select_many", fake_select_many):
        with pytest.raises(ValueError, match="score bundle hash mismatch"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reusable_bundle_rejects_unknown_current_lifecycle_state():
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    durable_row = _durable_score_bundle_row(
        score_bundle_doc,
        current_event_status="future_state",
    )

    async def fake_select_many(*_args, **_kwargs):
        return [durable_row]

    with (
        mock.patch.object(sw, "select_many", fake_select_many),
        mock.patch.object(sw, "select_one", _current_reader(durable_row)),
    ):
        with pytest.raises(RuntimeError, match="unknown current lifecycle state"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))


def test_reusable_bundle_rejects_valid_doc_under_different_projected_candidate():
    worker = _worker()
    score_bundle_doc = _signed_score_bundle_doc(evaluation_epoch=1)
    score_bundle_doc = {
        **score_bundle_doc,
        "candidate_artifact_hash": "sha256:b",
    }
    replacement_hash = _score_bundle_hash(score_bundle_doc)
    score_bundle_doc = {
        **score_bundle_doc,
        "score_bundle_hash": replacement_hash,
        "anchored_hash": replacement_hash,
    }
    durable_row = _durable_score_bundle_row(
        score_bundle_doc,
        current_event_status="scored",
    )
    durable_row["candidate_artifact_hash"] = "sha256:a"

    async def fake_select_many(*_args, **_kwargs):
        return [durable_row]

    with mock.patch.object(sw, "select_many", fake_select_many):
        with pytest.raises(RuntimeError, match="durable identity"):
            asyncio.run(_find_reusable(worker, evaluation_epoch=1))
