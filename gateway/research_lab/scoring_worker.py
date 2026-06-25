"""Gateway-owned Research Lab private scoring worker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Mapping

from gateway.research_lab.bundles import build_research_lab_audit_bundle
from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    fetch_rolling_icp_window,
)
from gateway.research_lab.logging_utils import compact_ref, format_worker_block
from gateway.research_lab.models import ResearchLabScoreBundleCreateRequest
from gateway.research_lab.promotion import (
    ResearchLabPromotionController,
    _rebased_candidate_request,
    load_active_private_model,
)
from gateway.research_lab.public_activity import safe_project_public_loop_activity
from gateway.research_lab.public_benchmarks import (
    build_benchmark_visibility_split,
    build_public_benchmark_report,
    sanitize_benchmark_item_summary,
)
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_artifact,
    create_candidate_evaluation_event,
    create_candidate_promotion_event,
    create_private_model_benchmark_bundle,
    create_private_model_benchmark_event,
    create_public_benchmark_report,
    create_receipt_event,
    create_rolling_icp_window,
    create_score_bundle,
    create_scoring_dispatch_event,
    create_signed_audit_bundle,
    create_ticket_event,
    select_many,
    select_one,
)
from research_lab.auto_research_prompt import coerce_component_registry
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    ensure_private_model_outputs,
    sign_digest_with_kms,
)
from research_lab.eval.evaluator import QualificationStyleCompanyScorer


logger = logging.getLogger(__name__)
PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER = 6
PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS = 90.0


def _idle_log_seconds() -> float:
    try:
        return max(10.0, float(os.getenv("RESEARCH_LAB_WORKER_IDLE_LOG_SECONDS", "60")))
    except ValueError:
        return 60.0


def _error_backoff_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("RESEARCH_LAB_WORKER_ERROR_BACKOFF_SECONDS", "60")))
    except ValueError:
        return 60.0


def _short_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


class ResearchLabGatewayScoringWorker:
    """Scores Research Lab candidates inside the gateway trust boundary."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str | None = None):
        config.validate_public_benchmark_split()
        self.config = config
        self.worker_ref = worker_ref or config.scoring_worker_id or "research-lab-scoring-worker"
        self.proxy_url = config.scoring_worker_proxy_url or os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", "")
        self.proxy_ref_hash = canonical_hash({"proxy_ref": self.proxy_url}) if self.proxy_url else None
        self._baseline_skip_logged = False
        self._baseline_already_logged_date: str | None = None
        self._resolved_epoch_cache: tuple[int, float] | None = None

    async def run_forever(self) -> None:
        last_idle_log = 0.0
        last_error_log = 0.0
        idle_log_seconds = _idle_log_seconds()
        error_backoff_seconds = _error_backoff_seconds()
        while True:
            try:
                outcome = await self.run_once()
            except Exception as exc:
                now = time.monotonic()
                if now - last_error_log >= idle_log_seconds:
                    logger.error(
                        format_worker_block(
                            "RESEARCH LAB SCORING WORKER PASS FAILED",
                            (
                                ("Worker", self.worker_ref),
                                ("Error", _short_error(exc)),
                            ),
                        )
                    )
                    last_error_log = now
                await asyncio.sleep(max(self.config.scoring_worker_poll_seconds, error_backoff_seconds))
                continue
            if outcome.get("processed") or outcome.get("status") != "idle":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB SCORING WORKER PASS",
                        (
                            ("Worker", self.worker_ref),
                            ("Status", outcome.get("status")),
                            ("Candidates", len(outcome.get("candidate_ids") or [])),
                            (
                                "Baseline status",
                                (outcome.get("baseline") or {}).get("status")
                                if isinstance(outcome.get("baseline"), Mapping)
                                else None,
                            ),
                        ),
                    )
                )
            elif time.monotonic() - last_idle_log >= idle_log_seconds:
                logger.info(
                    "Research Lab scoring worker idle: worker_ref=%s poll_seconds=%s",
                    self.worker_ref,
                    self.config.scoring_worker_poll_seconds,
                )
                last_idle_log = time.monotonic()
            await asyncio.sleep(max(1, self.config.scoring_worker_poll_seconds))

    async def run_once(self) -> dict[str, Any]:
        if not self.config.scoring_worker_enabled:
            return {"processed": False, "status": "disabled"}
        if not self.config.production_writes_enabled or not self.config.evaluation_bundles_enabled:
            return {"processed": False, "status": "writes_or_eval_disabled"}
        if self.config.scoring_worker_require_proxy and not self.proxy_url:
            return {"processed": False, "status": "scoring_worker_proxy_required"}

        baseline_result = None
        if self.config.private_baseline_rebenchmark_enabled and self._is_private_baseline_owner():
            baseline_result = await self._maybe_run_private_baseline()
        elif self.config.private_baseline_rebenchmark_enabled and not self._baseline_skip_logged:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE SKIPPED",
                    (
                        ("Worker", self.worker_ref),
                        ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ("Owner worker index", 1),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            self._baseline_skip_logged = True

        processed: list[str] = []
        for _ in range(max(1, self.config.scoring_worker_max_candidates)):
            candidate = await self._claim_next_candidate()
            if not candidate:
                break
            await self._score_candidate(candidate)
            processed.append(str(candidate["candidate_id"]))

        baseline_completed = (
            isinstance(baseline_result, Mapping)
            and str(baseline_result.get("status") or "") == "completed"
        )
        return {
            "processed": bool(processed or baseline_completed),
            "status": "processed" if processed else ("baseline_completed" if baseline_completed else "idle"),
            "candidate_ids": processed,
            "baseline": baseline_result,
        }

    async def _claim_next_candidate(self) -> dict[str, Any] | None:
        from gateway.db.client import get_write_client

        def _fetch() -> Any:
            return (
                get_write_client()
                .table("research_lab_candidate_evaluation_current")
                .select("*")
                .eq("current_candidate_status", "queued")
                .order("current_status_at", desc=False)
                .limit(1)
                .execute()
            )

        response = await asyncio.to_thread(_fetch)
        rows = getattr(response, "data", None) or []
        if not rows:
            return None
        candidate = dict(rows[0])
        candidate_id = str(candidate.get("candidate_id") or "")
        fresh = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status",
            filters=(("candidate_id", candidate_id),),
        )
        if not fresh or fresh.get("current_candidate_status") != "queued":
            return None
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="assigned",
            candidate_status="assigned",
            evaluator_ref=self.worker_ref,
            reason="assigned_to_gateway_qualification_worker",
            event_doc={
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
            },
        )
        await safe_project_public_loop_activity(
            str(candidate["ticket_id"]),
            source_ref=f"candidate_assigned:{candidate_id}",
            reason="assigned_to_gateway_qualification_worker",
            config=self.config,
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB CANDIDATE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Ticket", compact_ref(candidate.get("ticket_id"))),
                    ("Proxy ref", self.proxy_ref_hash),
                ),
            )
        )
        return candidate

    async def _score_candidate(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = str(candidate["candidate_id"])
        start = time.time()
        try:
            evaluation_epoch = await self._resolve_evaluation_epoch()
            stale_result = await self._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=lambda: round(time.time() - start, 3),
            )
            if stale_result.get("status") in {"stale_parent_rebased_candidate_queued", "stale_parent_rebase_failed"}:
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_rebased:{candidate_id}",
                    reason=str(stale_result["status"]),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after stale-parent rebase")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE STALE PARENT HANDLED",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
                            ("Derived", compact_ref(stale_result.get("derived_candidate_id"))),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
            )
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING STARTED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Ticket", compact_ref(candidate.get("ticket_id"))),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Model timeout", f"{self.config.scoring_worker_model_timeout_seconds}s"),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="evaluating",
                candidate_status="evaluating",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_started",
                event_doc={"worker_ref": self.worker_ref, "proxy_ref_hash": self.proxy_ref_hash},
            )
            window = await fetch_rolling_icp_window(
                days=self.config.lab_champion_eval_days,
                icps_per_day=self.config.lab_champion_icps_per_day,
                allow_partial=self.config.scoring_worker_allow_partial_icp_window,
            )
            await create_rolling_icp_window(window)

            artifact = PrivateModelArtifactManifest.from_mapping(candidate["private_model_manifest_doc"])
            patch = candidate["candidate_patch_manifest"]
            benchmark = SealedBenchmarkSet(
                benchmark_id=window.benchmark_id,
                icp_set_hash=window.window_hash,
                split_ref=window.split_ref,
                item_refs=window.item_refs,
                scoring_version="qualification-company-scorer:v1",
                hidden_plaintext_available=True,
            )
            runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    extra_env=self._private_scoring_env(),
                )
            )
            run_context = self._candidate_run_context(
                candidate,
                window_hash=window.window_hash,
                evaluation_epoch=evaluation_epoch,
            )
            score_bundle = await evaluate_private_model_pair(
                artifact_manifest=artifact,
                benchmark=benchmark,
                patch_manifest=patch,
                benchmark_items=window.benchmark_items,
                base_runner=runner,
                candidate_runner=runner,
                run_context={**run_context, "signature_ref": "pending"},
                policy=self._evaluation_policy(),
            )
            unsigned_hash = str(score_bundle["score_bundle_hash"])
            signature_ref = await asyncio.to_thread(
                sign_digest_with_kms,
                key_id=self.config.score_bundle_kms_key_id,
                digest_hash=unsigned_hash,
                signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
            )
            score_bundle = {**score_bundle, "signature_ref": signature_ref}
            score_bundle_request = ResearchLabScoreBundleCreateRequest(
                bundle_status="scored",
                receipt_id=candidate.get("receipt_id") or None,
                score_bundle=score_bundle,
            )
            bundle, _bundle_event = await create_score_bundle(score_bundle_request)
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="scored",
                candidate_status="scored",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_scored_candidate",
                score_bundle_id=str(bundle["score_bundle_id"]),
                event_doc={
                    "score_bundle_hash": score_bundle["score_bundle_hash"],
                    "rolling_window_hash": window.window_hash,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="scored",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=window.window_hash,
                score_bundle_id=str(bundle["score_bundle_id"]),
                event_doc={"elapsed_seconds": round(time.time() - start, 3)},
            )
            promotion_result = await self._maybe_promote_scored_candidate(
                candidate=candidate,
                score_bundle_row=bundle,
                score_bundle=score_bundle,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored:{candidate_id}:{bundle['score_bundle_id']}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(int(run_context["evaluation_epoch"]))
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Score bundle", compact_ref(bundle["score_bundle_id"])),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Promotion", promotion_result.get("status")),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="failed",
                candidate_status="failed",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_failed",
                event_doc={
                    "error": str(exc)[:500],
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={"error": str(exc)[:500]},
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_failed:{candidate_id}",
                reason="gateway_qualification_worker_failed",
                config=self.config,
            )
            try:
                await self._write_audit_bundle(await self._resolve_evaluation_epoch())
            except Exception:
                logger.exception("Research Lab audit bundle write failed after candidate failure")
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Error", str(exc)[:300]),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )

    async def _maybe_promote_scored_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.auto_promotion_enabled:
            return {"status": "disabled"}
        active_registry = None
        try:
            active = await load_active_private_model(self.config, register_bootstrap=True)
            if str(candidate.get("parent_artifact_hash") or "") != active.artifact.model_artifact_hash:
                active_runner = DockerPrivateModelRunner(
                    DockerPrivateModelSpec(
                        image_digest=active.artifact.image_digest,
                        timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                        extra_env=self._private_scoring_env(),
                    )
                )
                active_registry = coerce_component_registry(active_runner.metadata())
        except Exception as exc:
            logger.warning("Research Lab active model registry unavailable for promotion: %s", str(exc)[:200])
        return await ResearchLabPromotionController(
            self.config,
            worker_ref=self.worker_ref,
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            active_component_registry=active_registry,
        )

    async def _maybe_rebase_stale_candidate_before_scoring(
        self,
        candidate: Mapping[str, Any],
        *,
        evaluation_epoch: int,
        elapsed_seconds: Any,
    ) -> dict[str, Any]:
        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        if candidate_parent == active_parent:
            return {"status": "current_parent"}

        candidate_id = str(candidate["candidate_id"])
        base_event_doc = {
            "action": "rebase_queued_candidate_against_active_model_before_scoring",
            "active_parent_artifact_hash": active_parent,
            "candidate_parent_artifact_hash": candidate_parent,
            "evaluation_epoch": int(evaluation_epoch),
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
        }
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            event_type="stale_parent_detected",
            promotion_status="rebase_required",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc={**base_event_doc, "stage": "before_scoring"},
        )
        try:
            active_runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=active.artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    extra_env=self._private_scoring_env(),
                )
            )
            active_registry = coerce_component_registry(active_runner.metadata())
            request = _rebased_candidate_request(candidate, active.artifact, active_registry)
            derived_row, _event = await create_candidate_artifact(request)
        except Exception as exc:
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="failed",
                candidate_status="failed",
                evaluator_ref=self.worker_ref,
                reason="stale_parent_rebase_failed_before_scoring",
                event_doc={**base_event_doc, "error": str(exc)[:500], "elapsed_seconds": elapsed_seconds()},
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={**base_event_doc, "error": str(exc)[:500]},
            )
            logger.exception("Research Lab stale-parent rebase failed before scoring candidate %s", candidate_id)
            return {"status": "stale_parent_rebase_failed", "error": str(exc)[:200]}

        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            derived_candidate_id=str(derived_row["candidate_id"]),
            event_type="rebase_queued",
            promotion_status="rebenchmarking",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc={
                **base_event_doc,
                "derived_candidate_id": str(derived_row["candidate_id"]),
                "derived_candidate_artifact_hash": str(derived_row["candidate_artifact_hash"]),
                "derived_parent_artifact_hash": active_parent,
            },
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason="stale_parent_rebased_before_scoring",
            event_doc={
                **base_event_doc,
                "derived_candidate_id": str(derived_row["candidate_id"]),
                "elapsed_seconds": elapsed_seconds(),
            },
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={
                **base_event_doc,
                "derived_candidate_id": str(derived_row["candidate_id"]),
                "reason": "stale_parent_rebased_before_scoring",
            },
        )
        return {
            "status": "stale_parent_rebased_candidate_queued",
            "derived_candidate_id": str(derived_row["candidate_id"]),
        }

    async def _maybe_run_private_baseline(self) -> dict[str, Any] | None:
        today = datetime.now(timezone.utc).date().isoformat()
        start = time.time()
        evaluation_epoch = await self._resolve_evaluation_epoch()
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Benchmark date", today),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Eval days", self.config.lab_champion_eval_days),
                    ("ICPs per day", self.config.lab_champion_icps_per_day),
                    ("Expected ICPs", self.config.lab_champion_eval_days * self.config.lab_champion_icps_per_day),
                ),
            )
        )
        window = await fetch_rolling_icp_window(
            days=self.config.lab_champion_eval_days,
            icps_per_day=self.config.lab_champion_icps_per_day,
            allow_partial=self.config.scoring_worker_allow_partial_icp_window,
        )
        active = await load_active_private_model(self.config, register_bootstrap=True)
        artifact = active.artifact
        existing = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="*",
            filters=(
                ("benchmark_date", today),
                ("rolling_window_hash", window.window_hash),
                ("private_model_manifest_hash", artifact.manifest_hash),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        valid_existing = [row for row in existing if _private_benchmark_row_is_valid(row)]
        if valid_existing:
            already_key = f"{today}:{window.window_hash}:{artifact.manifest_hash}"
            if self._baseline_already_logged_date != already_key:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ALREADY BENCHMARKED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Selected ICPs", len(window.item_refs)),
                            ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ),
                    )
                )
                self._baseline_already_logged_date = already_key
            return {
                "status": "already_benchmarked",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        benchmark_attempt = _next_benchmark_attempt(existing)
        await create_rolling_icp_window(window)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected sets", len(window.set_ids)),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Private model", compact_ref(artifact.model_artifact_hash)),
                ),
            )
        )
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                extra_env=self._private_scoring_env(),
            )
        )
        scorer = QualificationStyleCompanyScorer()
        per_icp_summaries: list[dict[str, Any]] = []
        nonempty_output_count = 0
        try:
            await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                },
            )
            total_icps = len(window.benchmark_items)
            for item_index, item in enumerate(window.benchmark_items, start=1):
                item_start = time.time()
                label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
                outputs = ensure_private_model_outputs(
                    await asyncio.to_thread(runner, item["icp"], {"mode": "private_baseline"}),
                    context_label=f"private baseline for {label}",
                    require_non_empty=False,
                )
                item_elapsed = time.time() - item_start
                if outputs:
                    nonempty_output_count += 1
                score_breakdowns = await scorer.score_with_breakdowns(outputs, item["icp"], True)
                scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
                icp_score = _average(scores)
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP SCORED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                            ("Score", f"{icp_score:.4f}"),
                            ("Companies", len(scores)),
                            ("Non-empty output", bool(outputs)),
                            ("ICP runtime", f"{item_elapsed:.1f}s"),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                if (
                    item_index >= PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER
                    and nonempty_output_count <= 0
                    and time.time() - start < PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS
                ):
                    raise PrivateModelRuntimeError(
                        "private baseline fast-empty guard tripped: "
                        f"first {item_index} ICPs returned zero companies in {time.time() - start:.1f}s. "
                        "The private model is not executing the full provider-backed sourcing path; "
                        "check Docker env passthrough, provider keys, proxy connectivity, and ICP canonicalization."
                    )
                per_icp_summaries.append(
                    sanitize_benchmark_item_summary(
                        item=item,
                        score=icp_score,
                        company_count=len(scores),
                        score_breakdowns=score_breakdowns,
                    )
                )
            if nonempty_output_count <= 0:
                raise PrivateModelRuntimeError(
                    f"private baseline returned zero companies across all {len(window.benchmark_items)} ICPs"
                )
        except Exception as exc:
            await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                    "error": str(exc)[:500],
                    "elapsed_seconds": round(time.time() - start, 3),
                },
            )
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Attempt", benchmark_attempt),
                        ("Error", str(exc)[:300]),
                    ),
                )
            )
            return {
                "status": "failed",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "error": str(exc)[:300],
            }
        aggregate_score = _average([summary["score"] for summary in per_icp_summaries])
        visibility_split = build_benchmark_visibility_split(
            rolling_window_hash=window.window_hash,
            benchmark_items=window.benchmark_items,
            per_icp_summaries=per_icp_summaries,
            public_icps_per_day=self.config.public_benchmark_public_icps_per_day,
            public_weak_per_day=self.config.public_benchmark_public_weak_per_day,
        )
        score_summary_doc = {
            "schema_version": "1.0",
            "benchmark_quality": "passed",
            "benchmark_attempt": benchmark_attempt,
            "rolling_window_hash": window.window_hash,
            "per_icp_summaries": per_icp_summaries,
            "visibility_split": visibility_split,
            "aggregate_score": aggregate_score,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        bundle_hash = canonical_hash(score_summary_doc)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=bundle_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        bundle, _event = await create_private_model_benchmark_bundle(
            benchmark_date=today,
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            evaluation_epoch=evaluation_epoch,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            aggregate_score=aggregate_score,
            scoring_worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            signature_ref=signature_ref,
            score_summary_doc=score_summary_doc,
        )
        await create_scoring_dispatch_event(
            dispatch_type="private_baseline_rebenchmark",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            rolling_window_hash=window.window_hash,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            event_doc={
                "benchmark_date": today,
                "elapsed_seconds": round(time.time() - start, 3),
                "selected_icp_count": len(window.item_refs),
                "public_icp_count": int(visibility_split.get("public_count") or 0),
                "private_holdout_icp_count": int(visibility_split.get("private_count") or 0),
            },
        )
        public_report_doc = build_public_benchmark_report(
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            per_icp_summaries=per_icp_summaries,
            benchmark_items=window.benchmark_items,
            public_icps_per_day=self.config.public_benchmark_public_icps_per_day,
            public_weak_per_day=self.config.public_benchmark_public_weak_per_day,
        )
        public_report, _report_event = await create_public_benchmark_report(
            benchmark_date=today,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            report_doc=public_report_doc,
        )
        await self._write_audit_bundle(evaluation_epoch)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE COMPLETED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Benchmark bundle", compact_ref(bundle["benchmark_bundle_id"])),
                    ("Public report", compact_ref(public_report["report_id"])),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Attempt", benchmark_attempt),
                    ("Public ICPs", visibility_split.get("public_count")),
                    ("Private holdout ICPs", visibility_split.get("private_count")),
                    ("Public strength", visibility_split.get("public_strength_counts")),
                    ("Private strength", visibility_split.get("private_strength_counts")),
                    ("Aggregate score", f"{aggregate_score:.4f}"),
                    ("Elapsed", f"{time.time() - start:.1f}s"),
                ),
            )
        )
        return {
            "status": "completed",
            "benchmark_date": today,
            "benchmark_bundle_id": str(bundle["benchmark_bundle_id"]),
            "public_report_id": str(public_report["report_id"]),
            "rolling_window_hash": window.window_hash,
        }

    def _is_private_baseline_owner(self) -> bool:
        return self.config.scoring_worker_index == 0

    async def _resolve_evaluation_epoch(self) -> int:
        now = time.monotonic()
        if self._resolved_epoch_cache is not None:
            cached_epoch, cached_at = self._resolved_epoch_cache
            if now - cached_at <= 60.0:
                return cached_epoch

        epoch, block, source = await resolve_research_lab_evaluation_epoch(self.config.evaluation_epoch)

        if epoch <= 0:
            raise RuntimeError(
                "Research Lab evaluation epoch resolved to 0; refusing to write epoch-0 score/audit bundles"
            )
        self._resolved_epoch_cache = (epoch, now)
        logger.info(
            "Research Lab scoring worker resolved evaluation epoch: epoch=%s block=%s source=%s",
            epoch,
            block,
            source,
        )
        return epoch

    async def _write_audit_bundle(self, epoch: int) -> None:
        ticket_rows = await select_many("research_loop_ticket_current", filters=(), limit=1000)
        queue_rows = await select_many("research_loop_run_queue_current", filters=(), limit=1000)
        receipt_rows = await select_many("research_loop_receipt_current", filters=(), limit=1000)
        candidate_rows = await select_many("research_lab_candidate_evaluation_current", filters=(), limit=1000)
        candidate_event_rows = await select_many("research_lab_candidate_evaluation_events", filters=(), limit=1000)
        loop_event_rows = await select_many("research_lab_auto_research_loop_events", filters=(), limit=1000)
        dispatch_event_rows = await select_many("research_lab_scoring_dispatch_events", filters=(), limit=1000)
        rolling_window_rows = await select_many("research_lab_rolling_icp_windows", filters=(), limit=1000)
        benchmark_rows = await select_many("research_lab_private_model_benchmark_current", filters=(), limit=1000)
        private_model_version_rows = await select_many("research_lab_private_model_version_current", filters=(), limit=1000)
        promotion_event_rows = await select_many("research_lab_candidate_promotion_events", filters=(), limit=1000)
        private_repo_commit_event_rows = await select_many("research_lab_private_repo_commit_events", filters=(), limit=1000)
        public_benchmark_report_rows = await select_many("research_lab_public_benchmark_report_current", filters=(), limit=1000)
        score_bundle_rows = await select_many(
            "research_evaluation_score_bundle_current",
            filters=(("evaluation_epoch", epoch),),
            limit=1000,
        )
        bundle_doc = build_research_lab_audit_bundle(
            epoch=epoch,
            ticket_rows=ticket_rows,
            queue_rows=queue_rows,
            receipt_rows=receipt_rows,
            candidate_rows=candidate_rows,
            candidate_event_rows=candidate_event_rows,
            loop_event_rows=loop_event_rows,
            dispatch_event_rows=dispatch_event_rows,
            rolling_window_rows=rolling_window_rows,
            benchmark_rows=benchmark_rows,
            private_model_version_rows=private_model_version_rows,
            promotion_event_rows=promotion_event_rows,
            private_repo_commit_event_rows=private_repo_commit_event_rows,
            public_benchmark_report_rows=public_benchmark_report_rows,
            score_bundle_rows=score_bundle_rows,
        )
        audit_hash = canonical_hash(bundle_doc)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=audit_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        bundle, _event = await create_signed_audit_bundle(
            epoch=epoch,
            bundle_doc=bundle_doc,
            signature_ref=signature_ref,
        )
        await create_scoring_dispatch_event(
            dispatch_type="audit_bundle_build",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            event_doc={
                "audit_bundle_id": str(bundle["audit_bundle_id"]),
                "audit_bundle_hash": str(bundle["audit_bundle_hash"]),
            },
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB AUDIT BUNDLE WRITTEN",
                (
                    ("Worker", self.worker_ref),
                    ("Epoch", epoch),
                    ("Audit bundle", compact_ref(bundle["audit_bundle_id"])),
                    ("Audit hash", compact_ref(bundle["audit_bundle_hash"])),
                ),
            )
        )

    async def _maybe_finalize_candidate_receipt(self, candidate: Mapping[str, Any]) -> bool:
        receipt_id = candidate.get("receipt_id")
        if not receipt_id:
            return False
        candidates = await select_many(
            "research_lab_candidate_evaluation_current",
            filters=(("run_id", str(candidate["run_id"])),),
            limit=1000,
        )
        if not candidates:
            return False
        terminal_statuses = {"scored", "failed", "rejected", "tombstoned"}
        status_counts: dict[str, int] = {}
        score_bundle_ids: list[str] = []
        for row in candidates:
            status = str(row.get("current_candidate_status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in terminal_statuses:
                return False
            score_bundle_id = row.get("current_score_bundle_id")
            if score_bundle_id:
                score_bundle_ids.append(str(score_bundle_id))
        receipt = await select_one(
            "research_loop_receipt_current",
            filters=(("receipt_id", str(receipt_id)),),
        )
        if not receipt or receipt.get("current_receipt_status") != "queued":
            return False
        has_scored_candidate = status_counts.get("scored", 0) > 0
        event_doc = {
            "run_id": str(candidate["run_id"]),
            "candidate_status_counts": status_counts,
            "score_bundle_ids": score_bundle_ids,
            "finalization_source": "gateway_qualification_worker_results",
        }
        await create_receipt_event(
            receipt_id=str(receipt_id),
            ticket_id=str(candidate["ticket_id"]),
            event_type="completed" if has_scored_candidate else "failed",
            receipt_status="completed" if has_scored_candidate else "failed",
            event_doc=event_doc,
        )
        await create_ticket_event(
            ticket_id=str(candidate["ticket_id"]),
            event_type="completed" if has_scored_candidate else "cancelled",
            actor_hotkey=None,
            reason=(
                "gateway_research_lab_candidate_evaluation_completed"
                if has_scored_candidate
                else "gateway_research_lab_candidate_evaluation_failed"
            ),
            event_doc=event_doc,
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB RECEIPT FINALIZED",
                (
                    ("Worker", self.worker_ref),
                    ("Receipt", compact_ref(receipt_id)),
                    ("Run", compact_ref(candidate["run_id"])),
                    ("Status", "completed" if has_scored_candidate else "failed"),
                    ("Candidates scored", status_counts.get("scored", 0)),
                    ("Candidates failed", status_counts.get("failed", 0)),
                    ("Score bundles", len(score_bundle_ids)),
                ),
            )
        )
        return True

    def _candidate_run_context(
        self,
        candidate: Mapping[str, Any],
        *,
        window_hash: str,
        evaluation_epoch: int,
    ) -> dict[str, Any]:
        return {
            "run_id": str(candidate["run_id"]),
            "ticket_id": str(candidate["ticket_id"]),
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "evaluation_epoch": int(evaluation_epoch),
            "evaluator_version": "leadpoet-gateway-qualification-worker:research-lab:v1",
            "evidence_bundle_refs": [f"research_lab_rolling_icp_window:{window_hash}"],
            "execution_trace_ref": f"gateway_qualification_worker:{self.worker_ref}:{candidate['candidate_id']}",
            "cost_ledger_ref": "cost_ledger:" + canonical_hash(
                {
                    "candidate_id": candidate["candidate_id"],
                    "worker_ref": self.worker_ref,
                    "rolling_window_hash": window_hash,
                }
            ).split(":", 1)[1],
        }

    def _evaluation_policy(self) -> dict[str, Any]:
        return {
            "min_delta": float(
                os.environ.get(
                    "RESEARCH_LAB_MIN_DELTA",
                    str(self.config.improvement_threshold_points),
                )
            ),
            "min_delta_lcb": float(
                os.environ.get(
                    "RESEARCH_LAB_MIN_DELTA_LCB",
                    str(self.config.improvement_min_delta_lcb),
                )
            ),
            "min_successful_icps": int(
                os.environ.get(
                    "RESEARCH_LAB_MIN_SUCCESSFUL_ICPS",
                    str(self.config.lab_champion_eval_days * self.config.lab_champion_icps_per_day),
                )
            ),
            "max_hard_failures": int(os.environ.get("RESEARCH_LAB_MAX_HARD_FAILURES", "0")),
            "min_candidate_score": float(os.environ.get("RESEARCH_LAB_MIN_CANDIDATE_SCORE", "0")),
            "observed_cost_usd": 0.0,
        }

    def _private_scoring_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in (
            "EXA_API_KEY",
            "SCRAPINGDOG_API_KEY",
            "QUALIFICATION_SCRAPINGDOG_API_KEY",
            "OPENROUTER_API_KEY",
            "QUALIFICATION_OPENROUTER_API_KEY",
            "OPENROUTER_KEY",
        ):
            value = os.getenv(name)
            if value:
                env[name] = value
        if self.proxy_url:
            env.update(
                {
                    "HTTP_PROXY": self.proxy_url,
                    "HTTPS_PROXY": self.proxy_url,
                    "http_proxy": self.proxy_url,
                    "https_proxy": self.proxy_url,
                }
            )
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        if no_proxy:
            env["NO_PROXY"] = no_proxy
            env["no_proxy"] = no_proxy
        return env


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _next_benchmark_attempt(rows: list[Mapping[str, Any]]) -> int:
    attempts: list[int] = []
    for row in rows:
        try:
            attempts.append(int(row.get("benchmark_attempt") or 0))
        except (TypeError, ValueError):
            attempts.append(0)
    return (max(attempts) + 1) if attempts else 0


def _private_benchmark_row_is_valid(row: Mapping[str, Any]) -> bool:
    status = str(row.get("current_benchmark_status") or row.get("benchmark_status") or "")
    if status and status != "completed":
        return False
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    summaries = doc.get("per_icp_summaries") if isinstance(doc, Mapping) else None
    if not isinstance(summaries, list) or not summaries:
        return False
    if not any(_benchmark_summary_has_companies(item) for item in summaries):
        return False
    if str(row.get("benchmark_quality") or "") == "passed":
        return True
    try:
        return int(row.get("evaluation_epoch") or 0) > 0
    except (TypeError, ValueError):
        return False


def _benchmark_summary_has_companies(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    try:
        return int(item.get("company_count") or 0) > 0
    except (TypeError, ValueError):
        return False
