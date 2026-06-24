"""Gateway-owned Research Lab private scoring worker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Mapping

from gateway.research_lab.bundles import build_research_lab_audit_bundle
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    fetch_rolling_icp_window,
)
from gateway.research_lab.models import ResearchLabScoreBundleCreateRequest
from gateway.research_lab.promotion import ResearchLabPromotionController, load_active_private_model
from gateway.research_lab.public_benchmarks import build_public_benchmark_report, sanitize_benchmark_item_summary
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_evaluation_event,
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
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    sign_digest_with_kms,
)
from research_lab.eval.evaluator import QualificationStyleCompanyScorer


logger = logging.getLogger(__name__)


class ResearchLabGatewayScoringWorker:
    """Scores Research Lab candidates inside the gateway trust boundary."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str | None = None):
        self.config = config
        self.worker_ref = worker_ref or config.scoring_worker_id or "research-lab-scoring-worker"
        self.proxy_url = config.scoring_worker_proxy_url or os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", "")
        self.proxy_ref_hash = canonical_hash({"proxy_ref": self.proxy_url}) if self.proxy_url else None

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(max(1, self.config.scoring_worker_poll_seconds))

    async def run_once(self) -> dict[str, Any]:
        if not self.config.scoring_worker_enabled:
            return {"processed": False, "status": "disabled"}
        if not self.config.production_writes_enabled or not self.config.evaluation_bundles_enabled:
            return {"processed": False, "status": "writes_or_eval_disabled"}
        if self.config.scoring_worker_require_proxy and not self.proxy_url:
            return {"processed": False, "status": "scoring_worker_proxy_required"}

        baseline_result = None
        if self.config.private_baseline_rebenchmark_enabled:
            baseline_result = await self._maybe_run_private_baseline()

        processed: list[str] = []
        for _ in range(max(1, self.config.scoring_worker_max_candidates)):
            candidate = await self._claim_next_candidate()
            if not candidate:
                break
            await self._score_candidate(candidate)
            processed.append(str(candidate["candidate_id"]))

        return {
            "processed": bool(processed or baseline_result),
            "status": "processed" if processed else "idle",
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
        return candidate

    async def _score_candidate(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = str(candidate["candidate_id"])
        start = time.time()
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="assigned",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
        )
        try:
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
            run_context = self._candidate_run_context(candidate, window_hash=window.window_hash)
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
            if promotion_result.get("status") not in {"disabled", ""}:
                logger.info(
                    "Research Lab candidate promotion result: candidate_id=%s status=%s",
                    candidate_id,
                    promotion_result.get("status"),
                )
            await self._maybe_finalize_candidate_receipt(candidate)
            await self._write_audit_bundle(int(run_context["evaluation_epoch"]))
            logger.info("Research Lab candidate scored by gateway worker: %s", candidate_id)
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
            try:
                await self._write_audit_bundle(int(self.config.evaluation_epoch or 0))
            except Exception:
                logger.exception("Research Lab audit bundle write failed after candidate failure")
            logger.exception("Research Lab candidate scoring failed: %s", candidate_id)

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

    async def _maybe_run_private_baseline(self) -> dict[str, Any] | None:
        today = datetime.now(timezone.utc).date().isoformat()
        existing = await select_many(
            "research_lab_private_model_benchmark_bundles",
            columns="benchmark_bundle_id",
            filters=(("benchmark_date", today),),
            limit=1,
        )
        if existing:
            return {"status": "already_benchmarked", "benchmark_date": today}

        start = time.time()
        window = await fetch_rolling_icp_window(
            days=self.config.lab_champion_eval_days,
            icps_per_day=self.config.lab_champion_icps_per_day,
            allow_partial=self.config.scoring_worker_allow_partial_icp_window,
        )
        await create_rolling_icp_window(window)
        active = await load_active_private_model(self.config, register_bootstrap=True)
        artifact = active.artifact
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                extra_env=self._private_scoring_env(),
            )
        )
        scorer = QualificationStyleCompanyScorer()
        per_icp_summaries: list[dict[str, Any]] = []
        for item in window.benchmark_items:
            outputs = await asyncio.to_thread(runner, item["icp"], {"mode": "private_baseline"})
            score_breakdowns = await scorer.score_with_breakdowns(outputs, item["icp"], True)
            scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
            per_icp_summaries.append(
                sanitize_benchmark_item_summary(
                    item=item,
                    score=_average(scores),
                    company_count=len(scores),
                    score_breakdowns=score_breakdowns,
                )
            )
        aggregate_score = _average([summary["score"] for summary in per_icp_summaries])
        score_summary_doc = {
            "schema_version": "1.0",
            "rolling_window_hash": window.window_hash,
            "per_icp_summaries": per_icp_summaries,
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
            evaluation_epoch=int(self.config.evaluation_epoch or 0),
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
            event_doc={"benchmark_date": today, "elapsed_seconds": round(time.time() - start, 3)},
        )
        public_report_doc = build_public_benchmark_report(
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            per_icp_summaries=per_icp_summaries,
        )
        public_report, _report_event = await create_public_benchmark_report(
            benchmark_date=today,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            report_doc=public_report_doc,
        )
        await self._write_audit_bundle(int(self.config.evaluation_epoch or 0))
        return {
            "status": "completed",
            "benchmark_date": today,
            "benchmark_bundle_id": str(bundle["benchmark_bundle_id"]),
            "public_report_id": str(public_report["report_id"]),
            "rolling_window_hash": window.window_hash,
        }

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
        return True

    def _candidate_run_context(self, candidate: Mapping[str, Any], *, window_hash: str) -> dict[str, Any]:
        return {
            "run_id": str(candidate["run_id"]),
            "ticket_id": str(candidate["ticket_id"]),
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "evaluation_epoch": int(self.config.evaluation_epoch or 0),
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
