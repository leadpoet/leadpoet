"""Gateway-owned Research Lab private scoring worker."""

from __future__ import annotations

import asyncio
import contextlib
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
    select_all,
    select_many,
    select_one,
)
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    ensure_private_model_outputs,
    private_model_env_passthrough,
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


def _runtime_error_diagnostics(error_text: str) -> dict[str, Any]:
    """Return DB-safe runtime diagnostics without provider URLs or request text."""

    lowered = error_text.lower()
    provider = "unknown"
    if "scrapingdog" in lowered:
        provider = "scrapingdog"
    elif "exa" in lowered:
        provider = "exa"
    elif "openrouter" in lowered:
        provider = "openrouter"

    status = 0
    for code in (400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504):
        if f"http error {code}" in lowered or f"status={code}" in lowered or f'"status":{code}' in lowered:
            status = code
            break

    if status >= 500:
        category = "provider_http_5xx"
    elif status >= 400:
        category = "provider_http_4xx"
    else:
        category = "runtime_provider_error"

    return {
        "error_class": "PrivateModelRuntimeError" if "privatemodelruntimeerror" in lowered else "RuntimeError",
        "provider": provider,
        "status": status,
        "category": category,
    }


def _status_age_seconds(raw_status_at: object) -> float | None:
    if not raw_status_at:
        return None
    try:
        status_at = datetime.fromisoformat(str(raw_status_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - status_at.astimezone(timezone.utc)).total_seconds()


def _status_is_stale(raw_status_at: object, stale_after_seconds: int) -> bool:
    age_seconds = _status_age_seconds(raw_status_at)
    return age_seconds is not None and age_seconds > max(60, int(stale_after_seconds))


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
        self._private_scoring_env_not_ready_logged = False
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

        missing_private_env = self._missing_private_scoring_env()
        if missing_private_env:
            if not self._private_scoring_env_not_ready_logged:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB SCORING WORKER PRIVATE MODEL ENV NOT READY",
                        (
                            ("Worker", self.worker_ref),
                            ("Missing", ", ".join(missing_private_env)),
                            ("Action", "leaving queued candidates untouched"),
                        ),
                    )
                )
                self._private_scoring_env_not_ready_logged = True
            return {
                "processed": False,
                "status": "idle",
                "private_model_env_ready": False,
                "missing_private_model_env": list(missing_private_env),
            }
        if self._private_scoring_env_not_ready_logged:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB SCORING WORKER PRIVATE MODEL ENV READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Action", "candidate scoring enabled"),
                    ),
                )
            )
            self._private_scoring_env_not_ready_logged = False

        await self._recover_stale_candidate_claims()

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
        try:
            assigned_event = await create_candidate_evaluation_event(
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
        except Exception as exc:
            if _is_candidate_claim_race_error(exc):
                logger.info(
                    "research_lab_candidate_claim_race candidate_id=%s worker_ref=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                )
                return None
            raise
        assigned_current = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status,current_evaluator_ref,current_event_hash",
            filters=(("candidate_id", candidate_id),),
        )
        if (
            not assigned_current
            or assigned_current.get("current_candidate_status") != "assigned"
            or assigned_current.get("current_evaluator_ref") != self.worker_ref
            or assigned_current.get("current_event_hash") != assigned_event.get("anchored_hash")
        ):
            logger.info(
                "research_lab_candidate_claim_lost candidate_id=%s worker_ref=%s",
                compact_ref(candidate_id),
                self.worker_ref,
            )
            return None
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

    async def _recover_stale_candidate_claims(self) -> int:
        stale_after_seconds = max(120, int(self.config.scoring_worker_model_timeout_seconds or 900) + 60)
        rows: list[dict[str, Any]] = []
        for status in ("assigned", "evaluating"):
            rows.extend(
                await select_many(
                    "research_lab_candidate_evaluation_current",
                    columns=(
                        "candidate_id,run_id,ticket_id,current_candidate_status,current_status_at,"
                        "current_evaluator_ref,current_event_hash"
                    ),
                    filters=(("current_candidate_status", status),),
                    order_by=(("current_status_at", True),),
                    limit=50,
                )
            )
        recovered = 0
        for row in rows:
            if not _status_is_stale(row.get("current_status_at"), stale_after_seconds):
                continue
            candidate_id = str(row.get("candidate_id") or "")
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not candidate_id or not run_id or not ticket_id:
                continue
            claim_attempts = await self._candidate_claim_attempt_count(candidate_id)
            max_attempts = int(self.config.scoring_worker_max_claim_requeues)
            if claim_attempts >= max_attempts:
                try:
                    await create_candidate_evaluation_event(
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_type="failed",
                        candidate_status="failed",
                        evaluator_ref=self.worker_ref,
                        reason="stale_gateway_scoring_retry_limit_exceeded",
                        event_doc={
                            "recovering_worker_ref": self.worker_ref,
                            "previous_evaluator_ref": row.get("current_evaluator_ref"),
                            "previous_candidate_status": row.get("current_candidate_status"),
                            "previous_event_hash": row.get("current_event_hash"),
                            "previous_status_at": row.get("current_status_at"),
                            "stale_after_seconds": stale_after_seconds,
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    await create_scoring_dispatch_event(
                        dispatch_type="candidate_scoring_recovery",
                        dispatch_status="failed",
                        worker_ref=self.worker_ref,
                        proxy_ref_hash=self.proxy_ref_hash,
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_doc={
                            "reason": "stale_gateway_scoring_retry_limit_exceeded",
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    recovered += 1
                except Exception as exc:
                    logger.warning(
                        "research_lab_stale_candidate_fail_limit_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:240],
                    )
                continue
            try:
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="stale_gateway_scoring_requeued",
                    event_doc={
                        "recovering_worker_ref": self.worker_ref,
                        "previous_evaluator_ref": row.get("current_evaluator_ref"),
                        "previous_candidate_status": row.get("current_candidate_status"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                    },
                )
                recovered += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_candidate_requeue_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:240],
                )
        if recovered:
            logger.info(
                "research_lab_stale_candidates_requeued worker_ref=%s count=%s stale_after_seconds=%s",
                self.worker_ref,
                recovered,
                stale_after_seconds,
            )
        return recovered

    async def _candidate_claim_attempt_count(self, candidate_id: str) -> int:
        rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,candidate_status,reason",
            filters=(("candidate_id", candidate_id),),
            order_by=(("seq", True),),
            limit=100,
        )
        return sum(
            1
            for row in rows
            if str(row.get("event_type") or "") == "assigned"
            or str(row.get("reason") or "") == "stale_gateway_scoring_requeued"
        )

    async def _candidate_scoring_heartbeat(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        started_at: float,
    ) -> None:
        try:
            interval = max(
                60.0,
                float(os.environ.get("RESEARCH_LAB_SCORING_HEARTBEAT_SECONDS", "120")),
            )
        except ValueError:
            interval = 120.0
        while True:
            await asyncio.sleep(interval)
            try:
                current = await select_one(
                    "research_lab_candidate_evaluation_current",
                    columns="candidate_id,current_candidate_status,current_evaluator_ref",
                    filters=(("candidate_id", candidate_id),),
                )
                if (
                    not current
                    or current.get("current_candidate_status") != "evaluating"
                    or current.get("current_evaluator_ref") != self.worker_ref
                ):
                    logger.warning(
                        "research_lab_candidate_heartbeat_claim_lost candidate_id=%s worker_ref=%s",
                        compact_ref(candidate_id),
                        self.worker_ref,
                    )
                    return
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="gateway_qualification_worker_heartbeat",
                    event_doc={
                        "worker_ref": self.worker_ref,
                        "proxy_ref_hash": self.proxy_ref_hash,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "research_lab_candidate_heartbeat_failed candidate_id=%s worker_ref=%s error=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                    str(exc)[:240],
                )

    async def _score_candidate(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = str(candidate["candidate_id"])
        start = time.time()
        scored_event_written = False
        scored_score_bundle_id = ""
        try:
            evaluation_epoch = await self._resolve_evaluation_epoch()
            stale_result = await self._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=lambda: round(time.time() - start, 3),
            )
            if stale_result.get("status") in {
                "legacy_patch_candidate_unsupported",
                "stale_parent_needs_rescore",
            }:
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_or_legacy_rejected:{candidate_id}",
                    reason=str(stale_result["status"]),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after candidate rejection")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE PRE-SCORING REJECTED",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
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
            candidate_kind = str(candidate.get("candidate_kind") or "")
            if candidate_kind != "image_build":
                raise RuntimeError("candidate scoring requires image_build candidate_kind")
            candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
            if not isinstance(candidate_manifest_doc, Mapping):
                raise RuntimeError("image_build candidate is missing candidate_model_manifest_doc")
            candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)
            benchmark = SealedBenchmarkSet(
                benchmark_id=window.benchmark_id,
                icp_set_hash=window.window_hash,
                split_ref=window.split_ref,
                item_refs=window.item_refs,
                scoring_version="qualification-company-scorer:v1",
                hidden_plaintext_available=True,
            )
            private_holdout_gate = await self._candidate_private_holdout_gate(
                artifact=artifact,
                window_hash=window.window_hash,
            )
            runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    env_passthrough=self._private_model_env_passthrough(),
                    extra_env=self._private_scoring_env(),
                )
            )
            candidate_runner = DockerPrivateModelRunner(
                DockerPrivateModelSpec(
                    image_digest=candidate_artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    env_passthrough=self._private_model_env_passthrough(),
                    extra_env=self._private_scoring_env(),
                )
            )
            run_context = self._candidate_run_context(
                candidate,
                window_hash=window.window_hash,
                evaluation_epoch=evaluation_epoch,
            )
            heartbeat_task = asyncio.create_task(
                self._candidate_scoring_heartbeat(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    started_at=start,
                )
            )
            try:
                score_bundle = await evaluate_private_model_pair(
                    artifact_manifest=artifact,
                    benchmark=benchmark,
                    patch_manifest=patch,
                    candidate_artifact_manifest=candidate_artifact.to_dict(),
                    benchmark_items=window.benchmark_items,
                    base_runner=runner,
                    candidate_runner=candidate_runner,
                    run_context={**run_context, "signature_ref": "pending"},
                    policy=self._evaluation_policy(),
                    private_holdout_gate=private_holdout_gate,
                )
            finally:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            gate_result = score_bundle.get("private_holdout_gate")
            private_holdout_rejected = (
                isinstance(gate_result, Mapping)
                and str(gate_result.get("decision") or "") == "rejected_before_private_holdout"
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
            scored_score_bundle_id = str(bundle["score_bundle_id"])
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="scored",
                candidate_status="scored",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_scored_candidate",
                score_bundle_id=scored_score_bundle_id,
                event_doc={
                    "score_bundle_hash": score_bundle["score_bundle_hash"],
                    "rolling_window_hash": window.window_hash,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                },
            )
            scored_event_written = True
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
                event_doc={
                    "elapsed_seconds": round(time.time() - start, 3),
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                },
            )
            promotion_result = (
                {"status": "rejected_public_holdout_gate"}
                if private_holdout_rejected
                else await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                )
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
                        ("Private holdout gate", (gate_result or {}).get("decision") if isinstance(gate_result, Mapping) else "-"),
                        ("Promotion", promotion_result.get("status")),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            if scored_event_written:
                await self._record_scored_candidate_side_effect_failure(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    score_bundle_id=scored_score_bundle_id,
                    error=exc,
                    elapsed_seconds=round(time.time() - start, 3),
                )
                return
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

    async def _record_scored_candidate_side_effect_failure(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        score_bundle_id: str,
        error: BaseException,
        elapsed_seconds: float,
    ) -> None:
        event_doc = {
            "error": str(error)[:500],
            "elapsed_seconds": elapsed_seconds,
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
            "score_bundle_id": score_bundle_id,
            "candidate_status_preserved": "scored",
        }
        try:
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring_side_effect",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                score_bundle_id=score_bundle_id or None,
                event_doc=event_doc,
            )
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_dispatch_failed")
        logger.exception(
            format_worker_block(
                "RESEARCH LAB CANDIDATE POST-SCORE SIDE EFFECT FAILED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Error", str(error)[:300]),
                    ("Candidate state", "scored"),
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
        return await ResearchLabPromotionController(
            self.config,
            worker_ref=self.worker_ref,
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
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
        candidate_id = str(candidate["candidate_id"])
        candidate_kind = str(candidate.get("candidate_kind") or "patch")
        if candidate_kind != "image_build":
            base_event_doc = {
                "action": "legacy_patch_candidate_rejected_before_scoring",
                "candidate_kind": candidate_kind,
                "active_parent_artifact_hash": active_parent,
                "candidate_parent_artifact_hash": candidate_parent,
                "evaluation_epoch": int(evaluation_epoch),
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
            }
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                event_type="unsupported_candidate_kind",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                worker_ref=self.worker_ref,
                event_doc={**base_event_doc, "stage": "before_scoring"},
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="rejected",
                candidate_status="rejected",
                evaluator_ref=self.worker_ref,
                reason="legacy_patch_candidate_unsupported",
                event_doc={**base_event_doc, "elapsed_seconds": elapsed_seconds()},
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="rejected",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={**base_event_doc, "reason": "legacy_patch_candidate_unsupported"},
            )
            return {"status": "legacy_patch_candidate_unsupported"}

        if candidate_parent == active_parent:
            return {"status": "current_parent"}

        base_event_doc = {
            "action": "image_build_candidate_parent_changed_before_scoring",
            "active_parent_artifact_hash": active_parent,
            "candidate_parent_artifact_hash": candidate_parent,
            "evaluation_epoch": int(evaluation_epoch),
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
        }
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            event_type="stale_parent_detected",
            promotion_status="stale_parent_needs_rescore",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc={**base_event_doc, "stage": "before_scoring"},
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason="stale_parent_needs_rescore",
            event_doc={**base_event_doc, "elapsed_seconds": elapsed_seconds()},
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={**base_event_doc, "reason": "stale_parent_needs_rescore"},
        )
        return {"status": "stale_parent_needs_rescore"}

    async def _candidate_private_holdout_gate(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        window_hash: str,
    ) -> dict[str, Any]:
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,private_model_manifest_hash,rolling_window_hash,"
                "benchmark_quality,evaluation_epoch,score_summary_doc,current_benchmark_status,created_at"
            ),
            filters=(
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("rolling_window_hash", window_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("created_at", True),),
            limit=10,
        )
        for row in rows:
            if not _private_benchmark_row_is_valid(row):
                continue
            gate = _private_holdout_gate_from_baseline_row(row)
            if gate:
                return gate
        raise RuntimeError(
            "matching_completed_private_baseline_required_before_candidate_private_holdout: "
            f"manifest={compact_ref(artifact.manifest_hash)} window={compact_ref(window_hash)}"
        )

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
                env_passthrough=self._private_model_env_passthrough(),
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
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP STARTED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                        ),
                    )
                )
                runtime_error = ""
                try:
                    outputs = ensure_private_model_outputs(
                        await asyncio.to_thread(runner, item["icp"], {"mode": "private_baseline"}),
                        context_label=f"private baseline for {label}",
                        require_non_empty=False,
                    )
                except PrivateModelRuntimeError as exc:
                    outputs = []
                    runtime_error = _short_error(exc)
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB PRIVATE BASELINE ICP RUNTIME ERROR",
                            (
                                ("Worker", self.worker_ref),
                                ("ICP", f"{item_index}/{total_icps}"),
                                ("ICP ref", compact_ref(label)),
                                ("ICP hash", compact_ref(item.get("icp_hash"))),
                                ("Error", runtime_error),
                            ),
                        )
                    )
                item_elapsed = time.time() - item_start
                if outputs:
                    nonempty_output_count += 1
                score_breakdowns = (
                    await scorer.score_with_breakdowns(outputs, item["icp"], True)
                    if outputs
                    else []
                )
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
                            ("Runtime error", runtime_error or "-"),
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
                item_summary = sanitize_benchmark_item_summary(
                    item=item,
                    score=icp_score,
                    company_count=len(scores),
                    score_breakdowns=score_breakdowns,
                    # Model output count BEFORE the scorer's employee-bucket
                    # pre-filter, so the funnel's first stage is the true
                    # "companies discovered" number.
                    sourced_count=len(outputs),
                )
                if runtime_error:
                    diagnostics = dict(item_summary.get("diagnostics") or {})
                    runtime_diagnostics = _runtime_error_diagnostics(runtime_error)
                    categories = set(diagnostics.get("failure_categories") or [])
                    categories.add("runtime_provider_error")
                    categories.add(str(runtime_diagnostics["category"]))
                    diagnostics["failure_categories"] = sorted(categories)
                    diagnostics["runtime_error"] = runtime_diagnostics
                    item_summary["diagnostics"] = diagnostics
                per_icp_summaries.append(item_summary)
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
            public_total_icps=self.config.public_benchmark_public_total_icps,
            public_weak_total=self.config.public_benchmark_public_weak_total,
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
            public_total_icps=self.config.public_benchmark_public_total_icps,
            public_weak_total=self.config.public_benchmark_public_weak_total,
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
        ticket_rows = await self._audit_select_all("research_loop_ticket_current", current_view=True)
        queue_rows = await self._audit_select_all("research_loop_run_queue_current", current_view=True)
        receipt_rows = await self._audit_select_all("research_loop_receipt_current", current_view=True)
        candidate_rows = await self._audit_select_all("research_lab_candidate_evaluation_current", current_view=True)
        candidate_event_rows = await self._audit_select_all("research_lab_candidate_evaluation_events")
        loop_event_rows = await self._audit_select_all("research_lab_auto_research_loop_events")
        dispatch_event_rows = await self._audit_select_all("research_lab_scoring_dispatch_events")
        rolling_window_rows = await self._audit_select_all("research_lab_rolling_icp_windows")
        benchmark_rows = await self._audit_select_all("research_lab_private_model_benchmark_current", current_view=True)
        private_model_version_rows = await self._audit_select_all("research_lab_private_model_version_current", current_view=True)
        promotion_event_rows = await self._audit_select_all("research_lab_candidate_promotion_events")
        private_repo_commit_event_rows = await self._audit_select_all("research_lab_private_repo_commit_events")
        public_benchmark_report_rows = await self._audit_select_all(
            "research_lab_public_benchmark_report_current",
            current_view=True,
        )
        score_bundle_rows = await self._audit_select_all(
            "research_evaluation_score_bundle_current",
            filters=(("evaluation_epoch", epoch),),
            current_view=True,
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

    async def _audit_select_all(
        self,
        table: str,
        *,
        filters: tuple[tuple[Any, ...], ...] = (),
        current_view: bool = False,
    ) -> list[dict[str, Any]]:
        primary_order = (("current_status_at", True),) if current_view else (("created_at", True),)
        try:
            return await select_all(table, filters=filters, order_by=primary_order, max_rows=50000)
        except Exception as exc:
            logger.warning(
                "research_lab_audit_select_order_fallback table=%s order=%s error=%s",
                table,
                primary_order,
                str(exc)[:200],
            )
            return await select_all(table, filters=filters, max_rows=50000)

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
        try:
            await create_receipt_event(
                receipt_id=str(receipt_id),
                ticket_id=str(candidate["ticket_id"]),
                event_type="completed" if has_scored_candidate else "failed",
                receipt_status="completed" if has_scored_candidate else "failed",
                event_doc=event_doc,
            )
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            latest_receipt = await select_one(
                "research_loop_receipt_current",
                filters=(("receipt_id", str(receipt_id)),),
            )
            if latest_receipt and latest_receipt.get("current_receipt_status") != "queued":
                logger.info(
                    "research_lab_receipt_finalization_race_lost receipt_id=%s status=%s",
                    compact_ref(receipt_id),
                    latest_receipt.get("current_receipt_status"),
                )
                return False
            raise
        try:
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
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            logger.warning(
                "research_lab_ticket_finalization_race_lost ticket_id=%s receipt_id=%s error=%s",
                compact_ref(candidate["ticket_id"]),
                compact_ref(receipt_id),
                str(exc)[:240],
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
        context = {
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
        if str(candidate.get("candidate_kind") or "") == "image_build":
            if candidate.get("candidate_source_diff_hash"):
                context["candidate_source_diff_hash"] = str(candidate["candidate_source_diff_hash"])
            build_doc = candidate.get("candidate_build_doc")
            if isinstance(build_doc, Mapping):
                context["candidate_build_ref"] = str(
                    build_doc.get("build_doc_hash")
                    or canonical_hash(build_doc)
                )
        return context

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
        if self.proxy_url and self.config.private_model_docker_global_proxy_enabled:
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

    def _missing_private_scoring_env(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not os.getenv("EXA_API_KEY"):
            missing.append("EXA_API_KEY")
        if not (os.getenv("SCRAPINGDOG_API_KEY") or os.getenv("QUALIFICATION_SCRAPINGDOG_API_KEY")):
            missing.append("SCRAPINGDOG_API_KEY or QUALIFICATION_SCRAPINGDOG_API_KEY")
        if not (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("QUALIFICATION_OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_KEY")
        ):
            missing.append("OPENROUTER_API_KEY or QUALIFICATION_OPENROUTER_API_KEY or OPENROUTER_KEY")
        return tuple(missing)

    def _private_model_env_passthrough(self) -> tuple[str, ...]:
        return private_model_env_passthrough(
            include_proxy=self.config.private_model_docker_global_proxy_enabled
        )


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


def _private_holdout_gate_from_baseline_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    split = doc.get("visibility_split") if isinstance(doc.get("visibility_split"), Mapping) else {}
    items = split.get("items") if isinstance(split.get("items"), list) else []
    public_items = [
        item for item in items
        if isinstance(item, Mapping) and str(item.get("visibility") or "") == "public"
    ]
    private_count = _safe_int(split.get("private_count"), default=0)
    if private_count <= 0:
        private_count = sum(
            1
            for item in items
            if isinstance(item, Mapping) and str(item.get("visibility") or "") == "private"
        )
    public_refs = [
        str(item.get("icp_ref") or "")
        for item in public_items
        if str(item.get("icp_ref") or "").strip()
    ]
    if not public_refs or private_count <= 0:
        return None
    public_scores = [_safe_float(item.get("score"), default=0.0) for item in public_items]
    return {
        "schema_version": "1.0",
        "gate_type": "public_score_before_private_holdout",
        "baseline_benchmark_bundle_id": str(row.get("benchmark_bundle_id") or ""),
        "baseline_public_score": _average(public_scores),
        "baseline_public_icp_count": len(public_refs),
        "baseline_private_holdout_icp_count": private_count,
        "rolling_window_hash": str(row.get("rolling_window_hash") or ""),
        "private_model_manifest_hash": str(row.get("private_model_manifest_hash") or ""),
        "public_icp_refs": public_refs,
    }


def _candidate_gate_event_doc(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "gate_type": str(value.get("gate_type") or ""),
        "decision": str(value.get("decision") or ""),
        "baseline_benchmark_bundle_id": str(value.get("baseline_benchmark_bundle_id") or ""),
        "baseline_public_score": _safe_float(value.get("baseline_public_score"), default=0.0),
        "candidate_public_score": _safe_float(value.get("candidate_public_score"), default=0.0),
        "paired_base_public_score": _safe_float(value.get("paired_base_public_score"), default=0.0),
        "public_icp_count": _safe_int(value.get("public_icp_count"), default=0),
        "private_holdout_icp_count": _safe_int(value.get("private_holdout_icp_count"), default=0),
        "private_holdout_evaluated": bool(value.get("private_holdout_evaluated")),
    }


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_candidate_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_candidate_claim_conflict" in message
        or "research_lab_candidate_eval_events_candidate_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _is_event_sequence_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )
