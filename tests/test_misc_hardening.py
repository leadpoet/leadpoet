"""Tests for the deferred smaller hardening items (fablefollowup.md).

Covers:
- Parent-image pre-pull before the private build command (§5.3): skip when the
  image is already cached, explicit pull with its own generous timeout when
  absent (RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS, default 900), failures
  always classified as infra (bug #30 machinery — never charged to the
  candidate), RESEARCH_LAB_PARENT_IMAGE_PREPULL kill switch, and the pre-pull
  running before the build command inside CodeEditCandidateBuilder.build.
- fetch_public_loop_summary honors the same RESEARCH_LAB_PUBLIC_LOOP_LIST_MAX_CARDS
  cap as the list path instead of a hardcoded 1000.
- Public label mapping for `paused/requeue_capacity_conflict_parked` (bug 28's
  recoverable park): surfaces as the allowlisted `queued` outcome with a truthful
  parked detail, instead of falling through the generic paused paths as
  `failed`/`running`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import gateway.research_lab.code_build as code_build
import gateway.research_lab.public_activity as public_activity
import gateway.research_lab.worker as worker
from gateway.research_lab.public_activity import derive_public_loop_outcome

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_BUILD_SOURCE = REPO_ROOT / "gateway" / "research_lab" / "code_build.py"
CODE_LOOP_ENGINE_SOURCE = REPO_ROOT / "gateway" / "research_lab" / "code_loop_engine.py"

IMAGE_REF = "public.ecr.aws/example/sourcing-model@sha256:" + "a" * 64

T1 = "2026-07-01T00:00:00+00:00"
T2 = "2026-07-01T01:00:00+00:00"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fake_run_recorder(calls, *, inspect_fails=False, pull_fails=False):
    """Build a fake code_build._run that records (cmd, timeout_seconds)."""

    def fake_run(cmd, *, cwd, timeout_seconds):
        calls.append((list(cmd), timeout_seconds))
        if inspect_fails and cmd[:3] == ["docker", "image", "inspect"]:
            raise code_build.CodeEditBuildError("No such image: " + cmd[-1])
        if pull_fails and cmd[:2] == ["docker", "pull"]:
            raise code_build.CodeEditBuildError(
                "docker pull failed: dial tcp: i/o timeout",
                stderr="dial tcp: i/o timeout",
                exit_code=1,
            )
        return ""

    return fake_run


def _pull_calls(calls):
    return [call for call in calls if call[0][:2] == ["docker", "pull"]]


def _ticket(status: str = "running") -> dict:
    return {"current_ticket_status": status, "created_at": T1}


def _queue(status: str, reason: str = "", at: str = T2) -> dict:
    return {
        "run_id": "run-1",
        "current_queue_status": status,
        "current_reason": reason,
        "current_status_at": at,
    }


def _candidate(status: str, reason: str = "", at: str = T2, artifact_hash: str = "") -> dict:
    return {
        "candidate_id": f"cand-{status}-{reason or 'none'}",
        "run_id": "run-1",
        "receipt_id": "receipt-1",
        "current_candidate_status": status,
        "current_reason": reason,
        "current_status_at": at,
        "candidate_artifact_hash": artifact_hash,
        "redacted_public_summary": "improved intent scoring",
    }


def _bundle(delta: float, artifact_hash: str = "hash-1", at: str = T2) -> dict:
    return {
        "score_bundle_id": "bundle-1",
        "candidate_artifact_hash": artifact_hash,
        "current_status_at": at,
        "score_bundle_doc": {
            "private_holdout_gate": {
                "decision": "private_holdout_approved",
                "private_holdout_evaluated": True,
                "baseline_aggregate_score": 20.0,
                "candidate_total_score": 20.0 + delta,
                "candidate_delta_vs_daily_baseline": delta,
            }
        },
    }


def _derive(
    *,
    ticket: dict | None = None,
    queue_rows: list | None = None,
    receipt_rows: list | None = None,
    candidate_rows: list | None = None,
    score_bundle_rows: list | None = None,
    promotion_event_rows: list | None = None,
):
    return derive_public_loop_outcome(
        ticket=ticket or _ticket(),
        queue_rows=queue_rows or [],
        receipt_rows=receipt_rows or [],
        candidate_rows=candidate_rows or [],
        score_bundle_rows=score_bundle_rows or [],
        promotion_event_rows=promotion_event_rows or [],
        improvement_threshold_points=1.0,
    )


# --------------------------------------------------------------------------- #
# §5.3 — parent-image pre-pull outside the build budget
# --------------------------------------------------------------------------- #
class TestParentImagePrepull:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_PARENT_IMAGE_PREPULL", raising=False)
        monkeypatch.delenv("RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", raising=False)
        monkeypatch.setattr(code_build, "_infra_retry_backoff_seconds", lambda: 0.0)

    def test_flag_default_on(self, monkeypatch):
        assert code_build._parent_image_prepull_enabled() is True
        monkeypatch.setenv("RESEARCH_LAB_PARENT_IMAGE_PREPULL", "false")
        assert code_build._parent_image_prepull_enabled() is False

    def test_pull_timeout_default_and_env(self, monkeypatch):
        assert code_build._parent_image_pull_timeout_seconds() == 900
        monkeypatch.setenv("RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS", "1800")
        assert code_build._parent_image_pull_timeout_seconds() == 1800
        monkeypatch.setenv("RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS", "junk")
        assert code_build._parent_image_pull_timeout_seconds() == 900

    def test_skips_pull_when_image_cached(self, monkeypatch):
        calls = []
        monkeypatch.setattr(code_build, "_run", _fake_run_recorder(calls))
        code_build._prepull_parent_image_for_build(IMAGE_REF)
        assert calls == [(["docker", "image", "inspect", IMAGE_REF], 120)]
        assert not _pull_calls(calls)

    def test_pulls_when_image_absent_with_dedicated_timeout(self, monkeypatch):
        calls = []
        monkeypatch.setattr(code_build, "_run", _fake_run_recorder(calls, inspect_fails=True))
        code_build._prepull_parent_image_for_build(IMAGE_REF)
        pulls = _pull_calls(calls)
        assert len(pulls) == 1
        cmd, timeout = pulls[0]
        assert cmd == ["docker", "pull", "--platform", code_build._docker_platform(), IMAGE_REF]
        # The dedicated generous pull budget, NOT code_edit_build_timeout_seconds.
        assert timeout == 900

    def test_pull_timeout_env_honored(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS", "1234")
        calls = []
        monkeypatch.setattr(code_build, "_run", _fake_run_recorder(calls, inspect_fails=True))
        code_build._prepull_parent_image_for_build(IMAGE_REF)
        assert [timeout for _, timeout in _pull_calls(calls)] == [1234]

    def test_disabled_flag_skips_everything(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_PARENT_IMAGE_PREPULL", "false")
        calls = []
        monkeypatch.setattr(code_build, "_run", _fake_run_recorder(calls))
        code_build._prepull_parent_image_for_build(IMAGE_REF)
        assert calls == []

    def test_blank_image_ref_is_a_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(code_build, "_run", _fake_run_recorder(calls))
        code_build._prepull_parent_image_for_build("   ")
        assert calls == []

    def test_pull_failure_is_infra_classified_after_retry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            code_build, "_run", _fake_run_recorder(calls, inspect_fails=True, pull_fails=True)
        )
        with pytest.raises(code_build.CodeEditInfraFailureError) as exc_info:
            code_build._prepull_parent_image_for_build(IMAGE_REF)
        # Retried once via the bug-30 machinery, then surfaced as infra — the
        # failure_stage requeues the candidate instead of charging it.
        assert len(_pull_calls(calls)) == 2
        assert exc_info.value.failure_stage == "candidate_build_infra_failed"
        assert exc_info.value.retryable is True

    def test_pull_failure_is_infra_even_when_retry_disabled(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", "false")
        calls = []
        monkeypatch.setattr(
            code_build, "_run", _fake_run_recorder(calls, inspect_fails=True, pull_fails=True)
        )
        with pytest.raises(code_build.CodeEditInfraFailureError):
            code_build._prepull_parent_image_for_build(IMAGE_REF)
        # No retry, but the classification stays infra: a registry/network pull
        # failure can never be the candidate's fault.
        assert len(_pull_calls(calls)) == 1

    def test_retry_recovers_from_transient_pull_failure(self, monkeypatch):
        calls = []
        state = {"pull_attempts": 0}

        def fake_run(cmd, *, cwd, timeout_seconds):
            calls.append((list(cmd), timeout_seconds))
            if cmd[:3] == ["docker", "image", "inspect"]:
                raise code_build.CodeEditBuildError("No such image")
            if cmd[:2] == ["docker", "pull"]:
                state["pull_attempts"] += 1
                if state["pull_attempts"] == 1:
                    raise code_build.CodeEditBuildError("received unexpected http status: 503")
            return ""

        monkeypatch.setattr(code_build, "_run", fake_run)
        code_build._prepull_parent_image_for_build(IMAGE_REF)
        assert state["pull_attempts"] == 2

    def test_build_prepulls_before_the_build_command(self):
        """The pre-pull call sits inside build() BEFORE the private build cmd."""
        tree = ast.parse(CODE_BUILD_SOURCE.read_text())
        build_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_under_deadline"
        )
        call_linenos: dict[str, int] = {}
        for node in ast.walk(build_fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                call_linenos.setdefault(node.func.id, node.lineno)
        assert "_prepull_parent_image_for_build" in call_linenos
        assert "_run_private_build_cmd_with_infra_retry" in call_linenos
        assert (
            call_linenos["_prepull_parent_image_for_build"]
            < call_linenos["_run_private_build_cmd_with_infra_retry"]
        )


# --------------------------------------------------------------------------- #
# ECR auth/push failures are infra, not candidate failures
# --------------------------------------------------------------------------- #
class TestEcrPushInfraClassification:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", raising=False)
        monkeypatch.setattr(code_build, "_infra_retry_backoff_seconds", lambda: 0.0)

    def test_ecr_initiate_layer_upload_denial_is_infra_marker(self):
        message = (
            "User: arn:aws:sts::493765492819:assumed-role/"
            "leadpoet-gateway-s3-cloudwatch-role/i-123 is not authorized to perform: "
            "ecr:InitiateLayerUpload on resource: "
            "arn:aws:ecr:us-east-1:493765492819:repository/leadpoet/sourcing-model"
        )
        assert code_build._is_infra_failure_text(message)

    def test_private_build_ecr_denial_raises_retryable_infra_after_retry(self, monkeypatch, tmp_path):
        attempts = {"count": 0}

        def fake_run_shell(cmd, *, cwd, env, timeout_seconds):
            attempts["count"] += 1
            raise code_build.CodeEditBuildError(
                "docker push failed",
                stderr="denied: User is not authorized to perform: ecr:InitiateLayerUpload",
                exit_code=1,
            )

        monkeypatch.setattr(code_build, "_run_shell", fake_run_shell)
        with pytest.raises(code_build.CodeEditInfraFailureError) as exc_info:
            code_build._run_private_build_cmd_with_infra_retry(
                cmd="docker push example",
                cwd=tmp_path,
                env={},
                timeout_seconds=30,
            )
        assert attempts["count"] == 2
        assert exc_info.value.failure_stage == "candidate_build_infra_failed"
        assert exc_info.value.retryable is True

    def test_loop_engine_propagates_infra_before_candidate_rejection(self):
        source = CODE_LOOP_ENGINE_SOURCE.read_text()
        assert "except CodeEditInfraFailureError" in source
        assert (
            source.index("except CodeEditInfraFailureError")
            < source.index("except (CodeEditPrivateTestError, CodeEditImageBuildError, CodeEditPatchApplyError)")
        )

    def test_worker_requeues_typed_infra_even_when_message_contains_auth_denial(self):
        exc = code_build.CodeEditInfraFailureError(
            "not authorized to perform: ecr:InitiateLayerUpload"
        )
        assert worker._is_retryable_worker_exception(exc) is True


# --------------------------------------------------------------------------- #
# Summary route honors the shared list-cap env
# --------------------------------------------------------------------------- #
class TestPublicLoopSummaryCap:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv(public_activity.PUBLIC_LOOP_LIST_MAX_CARDS_ENV, raising=False)

    def _capture_select_many(self, monkeypatch):
        captured: list[dict] = []

        async def fake_select_many(table, **kwargs):
            captured.append({"table": table, "limit": kwargs.get("limit")})
            return []

        monkeypatch.setattr(public_activity, "select_many", fake_select_many)
        return captured

    async def test_summary_uses_default_cap(self, monkeypatch):
        captured = self._capture_select_many(monkeypatch)
        summary = await public_activity.fetch_public_loop_summary()
        assert captured[0]["table"] == "research_lab_public_loop_card_current"
        assert captured[0]["limit"] == public_activity.DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS
        assert summary["total_visible"] == 0

    async def test_summary_honors_cap_env(self, monkeypatch):
        monkeypatch.setenv(public_activity.PUBLIC_LOOP_LIST_MAX_CARDS_ENV, "77")
        captured = self._capture_select_many(monkeypatch)
        await public_activity.fetch_public_loop_summary()
        assert captured[0]["limit"] == 77

    async def test_summary_and_list_share_the_same_cap(self, monkeypatch):
        monkeypatch.setenv(public_activity.PUBLIC_LOOP_LIST_MAX_CARDS_ENV, "42")
        captured = self._capture_select_many(monkeypatch)
        await public_activity.fetch_public_loop_summary()
        await public_activity.fetch_public_loop_rows()
        card_limits = [
            entry["limit"]
            for entry in captured
            if entry["table"] == "research_lab_public_loop_card_current"
        ]
        assert card_limits == [42, 42]


# --------------------------------------------------------------------------- #
# Public label for paused/requeue_capacity_conflict_parked (bug 28 park)
# --------------------------------------------------------------------------- #
class TestRequeueCapacityParkedLabel:
    def test_parked_run_maps_to_queued_with_truthful_detail(self):
        outcome = _derive(queue_rows=[_queue("paused", "requeue_capacity_conflict_parked")])
        assert outcome.event_type == "queued"
        assert outcome.outcome_label == "queued"
        assert outcome.outcome_band == "pending"
        assert outcome.event_doc["queue_status"] == "paused"
        assert outcome.event_doc["queue_reason"] == "requeue_capacity_conflict_parked"
        assert outcome.event_doc["queue_parked"] is True
        assert "parked" in outcome.event_doc["queue_parked_detail"]

    def test_parked_run_with_terminal_candidates_is_not_failed(self):
        # Pre-fix this fell through the all-candidates-terminal path as `failed`,
        # even though the reaper will requeue the parked run.
        outcome = _derive(
            queue_rows=[_queue("paused", "requeue_capacity_conflict_parked")],
            candidate_rows=[_candidate("failed"), _candidate("rejected")],
        )
        assert outcome.outcome_label == "queued"
        assert outcome.outcome_band == "pending"

    def test_parked_run_without_candidates_is_not_running(self):
        # Pre-fix a candidate-less parked run surfaced as `running` via the
        # ticket status fallthrough.
        outcome = _derive(
            ticket=_ticket("running"),
            queue_rows=[_queue("paused", "requeue_capacity_conflict_parked")],
        )
        assert outcome.outcome_label == "queued"

    def test_parked_does_not_mask_scored_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("paused", "requeue_capacity_conflict_parked")],
            candidate_rows=[_candidate("scored", artifact_hash="hash-1")],
            score_bundle_rows=[_bundle(2.0)],
        )
        assert outcome.outcome_label == "scored_promising"

    def test_parked_does_not_mask_actively_scoring_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("paused", "requeue_capacity_conflict_parked")],
            candidate_rows=[_candidate("evaluating")],
        )
        assert outcome.outcome_label == "scoring"

    def test_generic_paused_reason_keeps_existing_behavior(self):
        # Only the capacity park is remapped; other paused reasons still flow
        # through the generic paths (here: ticket running -> `running`).
        outcome = _derive(
            ticket=_ticket("running"),
            queue_rows=[_queue("paused", "maintenance_paused")],
        )
        assert outcome.outcome_label == "running"
        assert "queue_parked" not in outcome.event_doc

    def test_credit_block_still_outranks_park_mapping(self):
        outcome = _derive(queue_rows=[_queue("paused", "blocked_for_credit_insufficient_credit")])
        assert outcome.outcome_label == "blocked_for_credit"
