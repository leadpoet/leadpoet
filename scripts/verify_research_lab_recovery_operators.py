"""Functional verification for the Research Lab lifecycle recovery operators.

Mirrors the existing verify-script harness (monkeypatch the module-under-test's
imported names with fakes, then drive each operator with asyncio.run). No DB.

Run: python3 scripts/verify_research_lab_recovery_operators.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.recovery as recovery  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes / harness
# --------------------------------------------------------------------------- #
class FakeConfig:
    @staticmethod
    def from_env():
        return SimpleNamespace(
            lab_champion_eval_days=10,
            lab_champion_icps_per_day=6,
            scoring_worker_allow_partial_icp_window=True,
        )


class FakeArtifact:
    @classmethod
    def from_mapping(cls, _doc):
        return SimpleNamespace(manifest_hash="sha256:manifest", model_artifact_hash="sha256:artifact")


def _capacity_doc(_config=None):
    return {
        "autoresearch_capacity_policy": "test:v1",
        "autoresearch_capacity": 5,
        "active_loop_stale_after_seconds": 300,
    }


@contextlib.contextmanager
def patched(**overrides):
    """Temporarily set recovery.<name> overrides, restoring originals after."""
    sentinel = object()
    originals = {name: getattr(recovery, name, sentinel) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(recovery, name, value)
        yield
    finally:
        for name, original in originals.items():
            if original is sentinel:
                delattr(recovery, name)
            else:
                setattr(recovery, name, original)


def _base_overrides(**extra):
    base = {
        "ResearchLabGatewayConfig": FakeConfig,
        "default_actor_ref": lambda: "operator@test",
        "autoresearch_queue_capacity_doc": _capacity_doc,
        "PrivateModelArtifactManifest": FakeArtifact,
    }
    base.update(extra)
    return base


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


# --------------------------------------------------------------------------- #
# Operator 1 — resume_failed_runs_from_checkpoint
# --------------------------------------------------------------------------- #
def test_resume_failed_runs(errors):
    failed_row = {
        "run_id": "run-1",
        "ticket_id": "ticket-1",
        "current_queue_status": "failed",
        "queue_priority": 0,
        "current_event_hash": "sha256:prev",
        "current_status_at": "2026-06-30T00:00:00+00:00",
    }
    created = []

    async def fake_select_all(table, **kwargs):
        return [dict(failed_row)] if table == "research_loop_run_queue_current" else []

    async def fake_create_queue_event(**kwargs):
        created.append(kwargs)
        return {"event_id": "evt-1", "seq": 5, "anchored_hash": "sha256:evt"}

    # --- (8) checkpoint resume requeue: checkpoint present, non-credit failure, apply
    async def ck_present(_run_id):
        return {"checkpoint_hash": "sha256:ckpt"}

    async def err_generic(_run_id):
        return "adapter timed out"

    with patched(**_base_overrides(
        select_all=fake_select_all,
        latest_auto_research_checkpoint=ck_present,
        _latest_queue_event_error=err_generic,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=False))
    check(errors, res["found"] == 1 and res["resumed"] == 1, f"[resume/apply] expected 1 resumed, got {res}")
    check(errors, len(created) == 1, "[resume/apply] expected exactly one queue event written")
    if created:
        doc = created[0]["event_doc"]
        check(errors, created[0]["reason"] == "operator_resume_from_checkpoint", "[resume/apply] wrong reason")
        check(errors, doc.get("resume_source") == "resume_failed_runs_from_checkpoint", "[resume/apply] missing resume_source")
        check(errors, doc.get("checkpoint_hash") == "sha256:ckpt", "[resume/apply] missing checkpoint_hash")
        check(errors, doc.get("previous_queue_status") == "failed", "[resume/apply] missing previous_queue_status")
        check(errors, doc.get("actor_ref") == "operator@test", "[resume/apply] missing actor_ref")

    # --- (1) dry-run writes nothing
    created.clear()
    with patched(**_base_overrides(
        select_all=fake_select_all,
        latest_auto_research_checkpoint=ck_present,
        _latest_queue_event_error=err_generic,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=True))
    check(errors, res["dry_run"] is True and res["resumed"] == 0, "[resume/dry] resumed should be 0")
    check(errors, len(created) == 0, "[resume/dry] must not write")
    check(errors, res["runs"] and res["runs"][0].get("dry_run") is True, "[resume/dry] run plan should be marked dry_run")

    # --- (3) missing checkpoint -> not_resumable_no_checkpoint, no write
    created.clear()

    async def ck_absent(_run_id):
        return None

    with patched(**_base_overrides(
        select_all=fake_select_all,
        latest_auto_research_checkpoint=ck_absent,
        _latest_queue_event_error=err_generic,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=False))
    check(errors, res["not_resumable_no_checkpoint"] == ["run-1"], "[resume/no-ckpt] expected not_resumable_no_checkpoint")
    check(errors, res["resumed"] == 0 and len(created) == 0, "[resume/no-ckpt] must not write")

    # --- credit-blocked: 402 failure + not funded -> blocked_for_credit, no write
    created.clear()

    async def err_402(_run_id):
        return 'OpenRouter HTTP 402: {"message":"requires more credits","code":402}'

    async def credit_not_ready(_config, *, ticket_id):
        return (False, "limit_remaining=0")

    with patched(**_base_overrides(
        select_all=fake_select_all,
        latest_auto_research_checkpoint=ck_present,
        _latest_queue_event_error=err_402,
        _openrouter_credit_ready=credit_not_ready,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=False))
    check(errors, res["blocked_for_credit"] == ["run-1"], f"[resume/credit] expected blocked_for_credit, got {res}")
    check(errors, len(created) == 0, "[resume/credit] must not write when credit-blocked")

    # --- credit failure but now funded -> resumes
    created.clear()

    async def credit_ready(_config, *, ticket_id):
        return (True, "limit_remaining=10")

    with patched(**_base_overrides(
        select_all=fake_select_all,
        latest_auto_research_checkpoint=ck_present,
        _latest_queue_event_error=err_402,
        _openrouter_credit_ready=credit_ready,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=False))
    check(errors, res["resumed"] == 1 and len(created) == 1, "[resume/credit-topup] expected resume after top-up")

    # --- (2) idempotency: run already 'queued' (not failed) -> nothing happens
    created.clear()

    async def select_all_queued(table, **kwargs):
        # the failed filter returns nothing once the run is already queued
        return []

    with patched(**_base_overrides(
        select_all=select_all_queued,
        latest_auto_research_checkpoint=ck_present,
        _latest_queue_event_error=err_generic,
        create_queue_event=fake_create_queue_event,
    )):
        res = asyncio.run(recovery.resume_failed_runs_from_checkpoint(dry_run=False))
    check(errors, res["found"] == 0 and res["resumed"] == 0 and len(created) == 0,
          "[resume/idempotent] re-run after success must be a no-op")


# --------------------------------------------------------------------------- #
# Operator 2 — requeue_baseline_not_ready_candidates
# --------------------------------------------------------------------------- #
def test_requeue_baseline(errors):
    cand_row = {
        "candidate_id": "candidate:base1",
        "run_id": "run-2",
        "ticket_id": "ticket-2",
        "current_candidate_status": "queued",
        "current_reason": "baseline_not_ready",
        "private_model_manifest_doc": {"manifest_hash": "sha256:manifest"},
    }
    created = []

    async def fake_select_all(table, **kwargs):
        return [dict(cand_row)] if table == "research_lab_candidate_evaluation_current" else []

    async def fake_window(**kwargs):
        return SimpleNamespace(window_hash="sha256:window")

    async def fake_create_eval(**kwargs):
        created.append(kwargs)
        return {"event_id": "evt-2", "seq": 3, "anchored_hash": "sha256:evt2"}

    class WorkerReady:
        def __init__(self, config, worker_ref=None):
            pass

        async def _candidate_private_holdout_gate(self, *, artifact, window_hash):
            return {"baseline_benchmark_bundle_id": "bundle-x"}

    class WorkerNotReady:
        def __init__(self, config, worker_ref=None):
            pass

        async def _candidate_private_holdout_gate(self, *, artifact, window_hash):
            raise recovery.CandidateBaselineNotReady("matching_completed_private_baseline_required...")

    # --- (5) baseline ready -> requeue once
    with patched(**_base_overrides(
        select_all=fake_select_all,
        fetch_rolling_icp_window=fake_window,
        ResearchLabGatewayScoringWorker=WorkerReady,
        create_candidate_evaluation_event=fake_create_eval,
    )):
        res = asyncio.run(recovery.requeue_baseline_not_ready_candidates(dry_run=False))
    check(errors, res["found"] == 1 and res["requeued"] == 1, f"[baseline/ready] expected 1 requeued, got {res}")
    check(errors, len(created) == 1, "[baseline/ready] expected exactly one eval event")
    if created:
        check(errors, created[0]["reason"] == "baseline_ready_requeue", "[baseline/ready] wrong reason")
        check(errors, created[0]["candidate_status"] == "queued", "[baseline/ready] status should be queued")
        check(errors, created[0]["event_doc"].get("resume_source") == "requeue_baseline_not_ready_candidates",
              "[baseline/ready] missing resume_source")

    # --- (1) dry-run writes nothing
    created.clear()
    with patched(**_base_overrides(
        select_all=fake_select_all,
        fetch_rolling_icp_window=fake_window,
        ResearchLabGatewayScoringWorker=WorkerReady,
        create_candidate_evaluation_event=fake_create_eval,
    )):
        res = asyncio.run(recovery.requeue_baseline_not_ready_candidates(dry_run=True))
    check(errors, res["dry_run"] and res["requeued"] == 0 and len(created) == 0, "[baseline/dry] must not write")

    # --- (4) baseline still not ready -> no mutation
    created.clear()
    with patched(**_base_overrides(
        select_all=fake_select_all,
        fetch_rolling_icp_window=fake_window,
        ResearchLabGatewayScoringWorker=WorkerNotReady,
        create_candidate_evaluation_event=fake_create_eval,
    )):
        res = asyncio.run(recovery.requeue_baseline_not_ready_candidates(dry_run=False))
    check(errors, res["still_waiting_for_baseline"] == ["candidate:base1"],
          f"[baseline/not-ready] expected still_waiting, got {res}")
    check(errors, res["requeued"] == 0 and len(created) == 0, "[baseline/not-ready] must not write")

    # --- (2) idempotency: after requeue, reason becomes baseline_ready_requeue -> no longer selected
    created.clear()

    async def select_all_empty(table, **kwargs):
        return []

    with patched(**_base_overrides(
        select_all=select_all_empty,
        fetch_rolling_icp_window=fake_window,
        ResearchLabGatewayScoringWorker=WorkerReady,
        create_candidate_evaluation_event=fake_create_eval,
    )):
        res = asyncio.run(recovery.requeue_baseline_not_ready_candidates(dry_run=False))
    check(errors, res["found"] == 0 and len(created) == 0, "[baseline/idempotent] re-run must be a no-op")


# --------------------------------------------------------------------------- #
# Operator 3 — rebase_stale_parent_candidates
# --------------------------------------------------------------------------- #
def test_rebase_stale_parent(errors):
    cand_row = {
        "candidate_id": "candidate:stale1",
        "run_id": "run-3",
        "ticket_id": "ticket-3",
        "candidate_kind": "image_build",
        "current_candidate_status": "rejected",
        "current_reason": "stale_parent_needs_rescore",
        "parent_artifact_hash": "sha256:oldparent",
    }
    rebase_calls = []

    async def fake_select_all(table, **kwargs):
        return [dict(cand_row)] if table == "research_lab_candidate_evaluation_current" else []

    async def fake_load_active(config, register_bootstrap=False):
        return SimpleNamespace(artifact=SimpleNamespace(model_artifact_hash="sha256:active"))

    async def promo_none(table, **kwargs):
        return []

    async def promo_existing(table, **kwargs):
        return [{"promotion_event_id": "p1", "event_type": "rebase_queued", "derived_candidate_id": "candidate:derived0"}]

    class WorkerRebase:
        def __init__(self, config, worker_ref=None):
            pass

        async def _resolve_evaluation_epoch(self):
            return 7

        async def _maybe_rebase_stale_candidate_before_scoring(self, candidate, *, evaluation_epoch, elapsed_seconds):
            rebase_calls.append((candidate["candidate_id"], evaluation_epoch, elapsed_seconds()))
            return {"status": "stale_parent_rebased_to_current",
                    "derived_candidate_id": "candidate:derived1", "repair_used": False}

    # --- (7) successful rebase
    with patched(**_base_overrides(
        select_all=fake_select_all,
        select_many=promo_none,
        load_active_private_model=fake_load_active,
        ResearchLabGatewayScoringWorker=WorkerRebase,
    )):
        res = asyncio.run(recovery.rebase_stale_parent_candidates(dry_run=False))
    check(errors, res["found"] == 1 and res["rebased"] == 1, f"[rebase/apply] expected 1 rebased, got {res}")
    check(errors, len(rebase_calls) == 1 and rebase_calls[0][1] == 7, "[rebase/apply] rebase not invoked with epoch")
    check(errors, res["rebases"] and res["rebases"][0].get("derived_candidate_id") == "candidate:derived1",
          "[rebase/apply] missing derived_candidate_id")

    # --- (1) dry-run does NOT call the rebase build path
    rebase_calls.clear()
    with patched(**_base_overrides(
        select_all=fake_select_all,
        select_many=promo_none,
        load_active_private_model=fake_load_active,
        ResearchLabGatewayScoringWorker=WorkerRebase,
    )):
        res = asyncio.run(recovery.rebase_stale_parent_candidates(dry_run=True))
    check(errors, res["dry_run"] and res["rebased"] == 0, "[rebase/dry] rebased should be 0")
    check(errors, len(rebase_calls) == 0, "[rebase/dry] must NOT call the build path")
    check(errors, res["rebases"] and res["rebases"][0].get("dry_run") is True, "[rebase/dry] should report rebase_eligible plan")

    # --- (6) already rebased -> skipped, no build call
    rebase_calls.clear()
    with patched(**_base_overrides(
        select_all=fake_select_all,
        select_many=promo_existing,
        load_active_private_model=fake_load_active,
        ResearchLabGatewayScoringWorker=WorkerRebase,
    )):
        res = asyncio.run(recovery.rebase_stale_parent_candidates(dry_run=False))
    check(errors, len(res["already_rebased"]) == 1 and res["rebased"] == 0,
          f"[rebase/already] expected already_rebased, got {res}")
    check(errors, len(rebase_calls) == 0, "[rebase/already] must not rebase again")

    # --- (2) idempotency: candidate already on current parent -> skipped
    rebase_calls.clear()
    on_current = dict(cand_row, parent_artifact_hash="sha256:active")

    async def select_all_current(table, **kwargs):
        return [on_current] if table == "research_lab_candidate_evaluation_current" else []

    with patched(**_base_overrides(
        select_all=select_all_current,
        select_many=promo_none,
        load_active_private_model=fake_load_active,
        ResearchLabGatewayScoringWorker=WorkerRebase,
    )):
        res = asyncio.run(recovery.rebase_stale_parent_candidates(dry_run=False))
    check(errors, res["already_on_current_parent"] == ["candidate:stale1"] and len(rebase_calls) == 0,
          "[rebase/on-current] candidate already on current parent must be skipped")


# --------------------------------------------------------------------------- #
# Wiring assertions
# --------------------------------------------------------------------------- #
def test_wiring(errors):
    for fn in ("resume_failed_runs_from_checkpoint", "requeue_baseline_not_ready_candidates",
               "rebase_stale_parent_candidates"):
        check(errors, hasattr(recovery, fn), f"[wiring] recovery missing {fn}")
    import gateway.research_lab.admin as admin
    parser = admin.build_parser()
    sub = {a.dest: a for a in parser._subparsers._group_actions} if parser._subparsers else {}
    # subcommand presence via help text
    help_text = parser.format_help()
    for cmd in ("resume-failed-runs", "requeue-baseline-not-ready", "rebase-stale-parents"):
        check(errors, cmd in help_text, f"[wiring] admin CLI missing subcommand {cmd}")


def main() -> int:
    errors: list[str] = []
    for test in (test_resume_failed_runs, test_requeue_baseline, test_rebase_stale_parent, test_wiring):
        try:
            test(errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{test.__name__}] raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — recovery operator verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — recovery operator verification (resume / baseline / rebase: dry-run, idempotency, "
          "missing-checkpoint, baseline-not-ready, baseline-ready, already-rebased, successful-rebase, "
          "checkpoint-resume, credit-block, wiring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
