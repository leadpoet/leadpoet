"""Code-edit image candidate generation loop for hosted Research Lab runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields, replace
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from gateway.research_lab.code_build import (
    CodeEditBuildError,
    CodeEditImageBuildError,
    CodeEditPatchApplyError,
    CodeEditPrivateTestError,
    CodeEditBuildResult,
    CodeEditCandidateBuilder,
    resolve_source_inspection_requests,
)
from gateway.research_lab.loop_engine import (
    AutoResearchLoopEvent,
    AutoResearchLoopResult,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
    _budget_limit_microusd,
    _coerce_call_result,
    _estimated_call_microusd,
    _running_cost_ledger,
    _safe_budget_doc,
    _settings_doc,
    _would_exceed_budget,
)
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    FORBIDDEN_CODE_EDIT_TERMS,
    CodeEditDraft,
    build_code_edit_auto_research_messages,
    build_code_edit_repair_messages,
    build_code_edit_source_inspection_messages,
    build_loop_direction_planner_messages,
    build_plan_alignment_judge_messages,
    code_edit_no_viable_patch_reason,
    code_edit_plan_alignment_errors,
    loop_direction_plan_from_mapping,
    parse_loop_direction_plan_response,
    parse_plan_alignment_judge_response,
    parse_code_edit_repair_response,
    parse_code_edit_response,
    parse_code_edit_source_inspection_response,
)
from research_lab.eval import PrivateModelArtifactManifest


CodeEditOpenRouterCaller = Callable[
    [Sequence[Mapping[str, str]], int, int, str],
    Awaitable[Union[str, OpenRouterCallResult]],
]

logger = logging.getLogger(__name__)

_ENGINE_FLAG_TRUTHY = {"1", "true", "yes", "on"}

# Exception class names (checked by name to avoid a circular import with
# worker.py) that must always escape bug-17 stage containment: they change the
# run's funding or ownership, not just the current stage.
_STAGE_PROPAGATE_ERROR_CLASS_NAMES = frozenset(
    {
        "CreditBlockedHostedRunError",
        "HostedResearchLabClaimLost",
    }
)


def _engine_env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _ENGINE_FLAG_TRUTHY


def _resume_restore_selected_enabled() -> bool:
    """Bug 5 kill switch: restore already-built candidates from the checkpoint on resume."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_RESUME_RESTORE_SELECTED", "true")


def _planner_parse_retry_enabled() -> bool:
    """Bug 16 kill switch: retry the planner once on parse failure, then fall back to plan-less mode."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_PLANNER_PARSE_RETRY", "true")


def _stage_error_containment_enabled() -> bool:
    """Bug 17 kill switch: a failed stage LLM call skips that stage/iteration instead of failing the run."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "true")


def _stop_at_candidate_cap_enabled() -> bool:
    """Bug 20 kill switch: stop iterating/building once max_candidates is reached (no build-and-discard)."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_STOP_AT_CANDIDATE_CAP", "true")


def _judge_parse_soft_skip_enabled() -> bool:
    """Bug 21 kill switch: retry the alignment judge once on parse failure, then accept neutrally
    instead of recording a confident rejection."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_JUDGE_PARSE_SOFT_SKIP", "true")


def _within_run_memory_enabled() -> bool:
    """Feed this run's prior rejections into later iterations and dedupe rejected diff hashes."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_WITHIN_RUN_MEMORY", "true")


def _min_runtime_skip_when_selected_enabled() -> bool:
    """Skip the post-loop minimum-runtime sleep when candidates are already selected."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_MIN_RUNTIME_SKIP_WHEN_SELECTED", "true")


def _build_heartbeat_enabled() -> bool:
    """Run the docker build off the event loop and emit heartbeat loop events during it."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_BUILD_HEARTBEAT", "true")


@dataclass(frozen=True)
class BuiltCodeEditCandidate:
    draft: CodeEditDraft
    build: CodeEditBuildResult
    node_id: str
    iteration: int
    rehydration_artifact_uri: str = ""
    rehydration_artifact_hash: str = ""


@dataclass(frozen=True)
class CodeEditLoopResult:
    selected_candidates: tuple[BuiltCodeEditCandidate, ...]
    iterations_completed: int
    stop_reason: str
    elapsed_seconds: float
    estimated_cost_usd: float
    actual_openrouter_cost_usd: float
    actual_openrouter_cost_microusd: int
    openrouter_call_count: int
    provider_usage: tuple[dict[str, Any], ...] = ()
    status: str = "completed"
    checkpoint_doc: dict[str, Any] | None = None

    def cost_ledger(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "status": self.status if self.status in {"paused", "completed", "failed"} else "completed",
            "total_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_microusd": int(self.actual_openrouter_cost_microusd),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "openrouter_call_count": self.openrouter_call_count,
            "iterations_completed": self.iterations_completed,
            "stop_reason": self.stop_reason,
        }


@dataclass
class CodeEditLoopEngine:
    settings: AutoResearchLoopSettings
    call_openrouter: CodeEditOpenRouterCaller
    event_sink: Any
    builder: CodeEditCandidateBuilder

    async def _restore_selected_from_resume(
        self,
        *,
        resume: Mapping[str, Any],
        run_id: str,
        artifact: PrivateModelArtifactManifest,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> list[BuiltCodeEditCandidate]:
        """Rehydrate already-built candidates from checkpoint artifacts (bug #5).

        Each checkpoint summary carries a ``rehydration_artifact_uri``/``_hash``
        pointing at the full S3 rehydration doc. Any per-candidate failure is
        logged and skipped; the caller degrades to the legacy empty ``selected``
        on total failure. The ``loop_resumed`` event reports the restored count.
        """
        del elapsed, openrouter_calls, estimated_cost, actual_cost_microusd  # event-free restore
        summaries = resume.get("selected_candidates")
        if not isinstance(summaries, Sequence):
            return []
        restored: list[BuiltCodeEditCandidate] = []
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            uri = str(summary.get("rehydration_artifact_uri") or "")
            expected_hash = str(summary.get("rehydration_artifact_hash") or "")
            if not uri.startswith("s3://"):
                continue
            try:
                bucket, object_key = _parse_s3_uri(uri)

                def _get(bucket: str = bucket, object_key: str = object_key) -> dict[str, Any]:
                    import boto3  # type: ignore

                    body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
                    return json.loads(body.decode("utf-8"))

                payload = await asyncio.to_thread(_get)
                stored_hash = str(payload.get("loop_candidate_artifact_hash") or "")
                if expected_hash and stored_hash and stored_hash != expected_hash:
                    raise ValueError("rehydration_artifact_hash_mismatch")
                candidate = _rehydrated_candidate_from_artifact_payload(payload)
                restored.append(
                    replace(
                        candidate,
                        rehydration_artifact_uri=uri,
                        rehydration_artifact_hash=expected_hash or stored_hash,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_loop_candidate_restore_failed run_id=%s node_id=%s error=%s",
                    run_id,
                    str(summary.get("node_id") or "")[:80],
                    str(exc)[:200],
                )
                continue
        if restored:
            logger.info(
                "research_lab_loop_candidates_restored run_id=%s count=%s parent=%s",
                run_id,
                len(restored),
                artifact.model_artifact_hash[:24],
            )
        return restored

    async def _call_stage_contained(
        self,
        messages: Sequence[Mapping[str, str]],
        timeout_seconds: int,
        max_tokens: int,
        stage: str,
    ) -> tuple[OpenRouterCallResult | None, str | None]:
        """Run one stage LLM call with bug-17 error containment.

        Returns ``(result, None)`` on success and ``(None, error_text)`` on a
        contained failure — the caller skips the stage/iteration and the run
        keeps whatever it already built. Credit blocks and claim losses always
        propagate (they change run ownership/funding, not just this stage), as
        does everything when containment is disabled.
        """
        try:
            raw = await self.call_openrouter(messages, timeout_seconds, max_tokens, stage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if any(
                base.__name__ in _STAGE_PROPAGATE_ERROR_CLASS_NAMES
                for base in type(exc).__mro__
            ):
                raise
            if not _stage_error_containment_enabled():
                raise
            lost_cost_microusd = max(0, int(getattr(exc, "cost_microusd", 0) or 0))
            logger.warning(
                "research_lab_loop_stage_call_contained stage=%s lost_cost_microusd=%s error=%s",
                stage,
                lost_cost_microusd,
                str(exc)[:200],
            )
            return None, _diagnostic_text(f"{type(exc).__name__}: {exc}", limit=300)
        return _coerce_call_result(raw), None

    async def _build_candidate_with_heartbeat(
        self,
        *,
        draft: CodeEditDraft,
        artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
        source_context: Any,
        node_id: str,
        iteration: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> CodeEditBuildResult:
        """Run the docker build off the event loop with liveness heartbeats.

        The synchronous build used to block the event loop for up to the full
        build timeout with no loop events — exactly the no-loop-event window
        the stale-claim guard requeues (Chain E). Heartbeat events keep the run
        visibly alive during long builds.
        """
        if not _build_heartbeat_enabled():
            return self.builder.build(
                draft=draft,
                parent_artifact=artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
            )
        source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="candidate_build_started",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                cost_ledger=_running_cost_ledger(
                    openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"
                ),
                event_doc={
                    "iteration": iteration,
                    "candidate_index": candidate_index,
                    "status": "started",
                    "source_diff_hash": source_diff_hash,
                },
            )
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                self.builder.build,
                draft=draft,
                parent_artifact=artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
            )
        )
        heartbeat_index = 0
        try:
            while not task.done():
                done, _pending = await asyncio.wait({task}, timeout=30.0)
                if done:
                    break
                heartbeat_index += 1
                try:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_build_heartbeat",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "candidate_index": candidate_index,
                                "status": "heartbeat",
                                "heartbeat_index": heartbeat_index,
                                "source_diff_hash": source_diff_hash,
                            },
                        )
                    )
                except Exception:
                    # A failed heartbeat write must never fail the build itself.
                    pass
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _prepare_parent_source_context_with_heartbeat(
        self,
        *,
        run_id: str,
        artifact: PrivateModelArtifactManifest,
        workspace_dir: Path,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> Any:
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="source_inspection_requested",
                loop_status="running",
                elapsed_seconds=elapsed(),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "parent_image_source_prepare_started",
                ),
                event_doc={
                    "operation": "parent_image_source_prepare",
                    "status": "started",
                    "run_id": run_id,
                    "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                },
            )
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                self.builder.prepare_parent_source_context,
                parent_artifact=artifact,
                workspace_dir=workspace_dir,
            )
        )
        heartbeat_index = 0
        try:
            while not task.done():
                done, _pending = await asyncio.wait({task}, timeout=30.0)
                if done:
                    break
                heartbeat_index += 1
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "parent_image_source_prepare_heartbeat",
                        ),
                        event_doc={
                            "operation": "parent_image_source_prepare",
                            "status": "heartbeat",
                            "heartbeat_index": heartbeat_index,
                            "run_id": run_id,
                            "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                        },
                    )
                )
            source_context = await task
        except Exception as exc:
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="source_inspection_failed",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "parent_image_source_prepare_failed",
                    ),
                    event_doc={
                        "operation": "parent_image_source_prepare",
                        "status": "failed",
                        "run_id": run_id,
                        "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                        "error": str(exc)[:500],
                        "error_hash": sha256_json({"error": str(exc)}),
                    },
                )
            )
            raise
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="source_inspection_resolved",
                loop_status="running",
                elapsed_seconds=elapsed(),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "parent_image_source_prepare_completed",
                ),
                event_doc={
                    "operation": "parent_image_source_prepare",
                    "status": "completed",
                    "run_id": run_id,
                    "source_mode": source_context.source_mode,
                    "source_tree_hash": source_context.source_tree_hash,
                    "parent_image_digest_hash": source_context.parent_image_digest_hash,
                    "extracted_top_level_paths": list(source_context.top_level_paths),
                    "editable_file_count": len(source_context.editable_files),
                },
            )
        )
        return source_context

    async def run(
        self,
        *,
        run_id: str,
        ticket: Mapping[str, Any],
        artifact: PrivateModelArtifactManifest,
        component_registry: Mapping[str, Any],
        benchmark_public_summary: Mapping[str, Any],
        model_id: str,
        budget_context: Mapping[str, Any],
        requested_loop_count: int,
        resume_state: Mapping[str, Any] | None = None,
        should_pause: Any | None = None,
    ) -> CodeEditLoopResult:
        start = time.monotonic()
        settings = self.settings.normalized()
        selected: list[BuiltCodeEditCandidate] = []
        resume = dict(resume_state or {})
        iteration = max(0, int(resume.get("iterations_completed") or 0))
        openrouter_calls = max(0, int(resume.get("openrouter_call_count") or 0))
        estimated_cost = max(0.0, float(resume.get("estimated_cost_usd") or 0.0))
        actual_cost_microusd = max(0, int(resume.get("actual_openrouter_cost_microusd") or 0))
        provider_usage: list[dict[str, Any]] = [
            dict(item) for item in resume.get("provider_usage", []) if isinstance(item, Mapping)
        ]
        elapsed_offset = max(0.0, float(resume.get("elapsed_seconds") or 0.0))
        elapsed = lambda: elapsed_offset + (time.monotonic() - start)
        budget_limit_microusd = _budget_limit_microusd(budget_context)
        built_candidate_total = max(0, int(resume.get("built_candidate_count") or 0))
        within_run_memory_active = _within_run_memory_enabled()
        rejected_diff_hashes: set[str] = set()
        within_run_rejections: list[dict[str, Any]] = []

        def _record_within_run_rejection(
            *,
            stage: str,
            reason: str,
            iteration_index: int,
            draft: CodeEditDraft | None = None,
            diff_hash: str | None = None,
        ) -> None:
            if not within_run_memory_active:
                return
            resolved_hash = diff_hash or (
                sha256_json({"unified_diff": draft.unified_diff}) if draft is not None else ""
            )
            if resolved_hash:
                rejected_diff_hashes.add(resolved_hash)
            within_run_rejections.append(
                {
                    "iteration": int(iteration_index),
                    "stage": str(stage)[:80],
                    "reason": _memory_safe_text(reason),
                    "lane": str(draft.lane if draft is not None else "")[:80],
                    "plan_path_id": str(draft.plan_path_id if draft is not None else "")[:120],
                    "target_files": list(draft.target_files)[:10] if draft is not None else [],
                    "unified_diff_hash": resolved_hash,
                    "status": "rejected",
                }
            )
            del within_run_rejections[:-25]

        def _within_run_memory_doc() -> dict[str, Any] | None:
            if not within_run_memory_active or not within_run_rejections:
                return None
            return {
                "schema_version": "1.0",
                "note": (
                    "Rejections recorded earlier in this run. Do not propose a diff identical to a "
                    "rejected one; address the recorded rejection reason instead."
                ),
                "rejected_attempt_count": len(within_run_rejections),
                "rejected_diff_hashes": sorted(rejected_diff_hashes)[:50],
                "recent_rejections": [dict(item) for item in within_run_rejections[-10:]],
            }

        def _memory_budget_context(base: dict[str, Any]) -> dict[str, Any]:
            memory_doc = _within_run_memory_doc()
            if memory_doc is None:
                return base
            return {**base, "within_run_memory": memory_doc}

        source_tmp = tempfile.TemporaryDirectory(prefix="research-lab-parent-image-source-")

        def _cleanup_source_tmp() -> None:
            nonlocal source_tmp
            if source_tmp is None:
                return
            source_tmp.cleanup()
            source_tmp = None

        try:
            source_context = await self._prepare_parent_source_context_with_heartbeat(
                run_id=run_id,
                artifact=artifact,
                workspace_dir=Path(source_tmp.name),
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
            )
        except Exception:
            _cleanup_source_tmp()
            raise

        # Bug 5: resume used to discard already-built candidates, so a paused/requeued run
        # that had built+pushed an image resumed empty-handed and failed with "no finalists".
        # Restoration is strictly best-effort: any failure degrades to the previous behavior.
        restored_candidate_count = 0
        if resume and _resume_restore_selected_enabled():
            try:
                selected = await self._restore_selected_from_resume(
                    resume=resume,
                    run_id=run_id,
                    artifact=artifact,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                )
            except Exception:
                selected = []
            restored_candidate_count = len(selected)
            built_candidate_total = max(built_candidate_total, restored_candidate_count)

        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_resumed" if resume else "loop_started",
                loop_status="running",
                elapsed_seconds=elapsed_offset,
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "code_edit_loop_resumed" if resume else "code_edit_loop_started",
                ),
                event_doc={
                    "run_id": run_id,
                    "candidate_kind": "image_build",
                    "requested_loop_count": int(requested_loop_count),
                    "settings": _settings_doc(settings),
                    "budget_context": _safe_budget_doc(budget_context),
                    "resumed_from_checkpoint": bool(resume),
                    "restored_selected_candidate_count": restored_candidate_count,
                    "checkpoint_hash": resume.get("checkpoint_hash"),
                    "source_mode": source_context.source_mode,
                    "source_tree_hash": source_context.source_tree_hash,
                    "parent_image_digest_hash": source_context.parent_image_digest_hash,
                    "extracted_top_level_paths": list(source_context.top_level_paths),
                    "editable_file_count": len(source_context.editable_files),
                    "editable_file_sample": list(source_context.editable_files[:25]),
                    "file_preview_count": len(source_context.file_previews),
                },
            )
        )

        prior_attempts = _prior_attempts_from_budget_context(budget_context)
        loop_direction_plan_doc: dict[str, Any] | None = None
        if isinstance(resume.get("loop_direction_plan"), Mapping):
            try:
                loop_direction_plan_doc = loop_direction_plan_from_mapping(
                    resume["loop_direction_plan"]
                ).to_dict()
            except Exception as exc:
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "resume_loop_direction_plan_parse_failed",
                        ),
                        event_doc={
                            "stage": "resume_loop_direction_plan_parse",
                            "error": str(exc)[:500],
                            "checkpoint_hash": resume.get("checkpoint_hash"),
                        },
                    )
                )
                loop_direction_plan_doc = None
        planner_terminal_without_candidate = False
        last_checkpoint: dict[str, Any] | None = None
        stop_reason = "max_iterations"
        if (
            self.builder.config.loop_planner_enabled
            and loop_direction_plan_doc is None
            # Bug 5/20: skip the planner call when restored candidates already fill the cap;
            # the loop will finalize immediately, so a plan would be paid for and unused.
            and not (selected and len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled())
        ):
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                stop_reason = "compute_budget_exhausted_before_loop_planner"
                planner_terminal_without_candidate = True
            else:
                # Bug 16: the planner used to be single-shot — one malformed planner response
                # terminally failed the paid run. Retry the call once on failure, then fall
                # back to the existing plan-less mode instead of a terminal failure.
                planner_attempt_limit = 2 if _planner_parse_retry_enabled() else 1
                for planner_attempt in range(1, planner_attempt_limit + 1):
                    if planner_attempt > 1 and _would_exceed_budget(
                        actual_cost_microusd,
                        _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                        budget_limit_microusd,
                    ):
                        break
                    remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                    raw_plan = ""
                    planner_result, planner_call_error = await self._call_stage_contained(
                        build_loop_direction_planner_messages(
                            ticket={
                                "ticket_id": str(ticket.get("ticket_id") or ""),
                                "run_id": run_id,
                                "miner_hotkey": ticket.get("miner_hotkey"),
                                "island": ticket.get("island"),
                                "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                                "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                                "requested_loop_count": requested_loop_count,
                                "focus_signature_hash": _focus_signature_hash(ticket),
                            },
                            artifact_manifest=artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_index=source_context.inspection_index(),
                            budget_context={
                                **dict(budget_context),
                                "candidate_kind": "image_build",
                                "focus_signature_hash": _focus_signature_hash(ticket),
                            },
                            prior_attempts=prior_attempts,
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        self.builder.config.loop_planner_max_tokens,
                        "loop_planner",
                    )
                    if planner_result is None:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "loop_direction_planner_call_failed",
                                ),
                                event_doc={
                                    "stage": "loop_direction_planner",
                                    "planner_attempt": planner_attempt,
                                    "error": planner_call_error or "loop_direction_planner_call_failed",
                                    "fallback": (
                                        "plan_less_mode" if planner_attempt >= planner_attempt_limit else "retry"
                                    ),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                        continue
                    raw_plan = planner_result.content
                    openrouter_calls += 1
                    estimated_cost += settings.estimated_iteration_cost_usd
                    actual_cost_microusd += max(0, int(planner_result.cost_microusd))
                    if planner_result.provider_usage:
                        provider_usage.append({**planner_result.provider_usage, "call_stage": "loop_planner"})
                    try:
                        loop_plan = parse_loop_direction_plan_response(raw_plan)
                        loop_direction_plan_doc = loop_plan.to_dict()
                    except Exception as exc:
                        if planner_attempt >= planner_attempt_limit and not _planner_parse_retry_enabled():
                            stop_reason = "loop_direction_plan_parse_failed"
                            planner_terminal_without_candidate = True
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "loop_direction_plan_parse_failed",
                                ),
                                event_doc={
                                    "stage": "loop_direction_planner",
                                    "planner_attempt": planner_attempt,
                                    "error": str(exc)[:500],
                                    "raw_response_hash": sha256_json({"raw_response": raw_plan}),
                                    "fallback": (
                                        "terminal"
                                        if planner_terminal_without_candidate
                                        else (
                                            "plan_less_mode"
                                            if planner_attempt >= planner_attempt_limit
                                            else "retry"
                                        )
                                    ),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                        continue
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="loop_direction_planned",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "loop_direction_planned",
                            ),
                            event_doc={
                                "focus_signature_hash": _focus_signature_hash(ticket),
                                "loop_direction_plan": loop_direction_plan_doc,
                                "prior_attempt_count": len(prior_attempts),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    if loop_plan.no_new_safe_path:
                        stop_reason = "loop_direction_no_new_safe_path"
                        planner_terminal_without_candidate = True
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="no_viable_patch",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "no_viable_patch",
                                ),
                                event_doc={
                                    "reason": loop_plan.reason or "planner returned no_new_safe_path",
                                    "loop_direction_plan_hash": loop_direction_plan_doc.get("plan_hash"),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                    break
        elif not self.builder.config.loop_planner_enabled:
            loop_direction_plan_doc = None
        while iteration < settings.max_iterations:
            if planner_terminal_without_candidate:
                break
            if elapsed() >= settings.max_seconds:
                stop_reason = "max_seconds"
                break
            # Bug 20: iterations past the candidate cap used to build docker images and then
            # discard them (selected is truncated to max_candidates every iteration). Once the
            # cap is filled, stop iterating; minimum runtime is enforced by the post-loop wait.
            if len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled():
                stop_reason = "candidate_limit_reached"
                break
            if (
                iteration >= settings.min_iterations
                and elapsed() >= settings.min_seconds
                and len(selected) >= settings.max_candidates
            ):
                stop_reason = "candidate_limit_reached_after_minimum_runtime"
                break
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                stop_reason = "compute_budget_exhausted_before_next_code_edit"
                break
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="pause_before_next_code_edit",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                )
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )

            iteration += 1
            remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
            source_inspection_context: dict[str, Any] = {
                "schema_version": "1.0",
                "source_tree_hash": source_context.source_tree_hash,
                "read_files": [],
                "results": [],
                "bytes_returned": 0,
            }
            read_paths: set[str] = set()
            source_bytes_returned = 0
            budget_exhausted_after_source_inspection = False
            for inspection_round in range(1, max(1, int(self.builder.config.code_edit_source_inspection_rounds)) + 1):
                if elapsed() >= settings.max_seconds:
                    stop_reason = "max_seconds"
                    break
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    stop_reason = "compute_budget_exhausted_before_source_inspection"
                    budget_exhausted_after_source_inspection = True
                    break
                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                inspection_result, inspection_call_error = await self._call_stage_contained(
                    build_code_edit_source_inspection_messages(
                        ticket={
                            "ticket_id": str(ticket.get("ticket_id") or ""),
                            "run_id": run_id,
                            "miner_hotkey": ticket.get("miner_hotkey"),
                            "island": ticket.get("island"),
                            "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                            "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                            "requested_loop_count": requested_loop_count,
                            "loop_iteration": iteration,
                            "inspection_round": inspection_round,
                        },
                        artifact_manifest=artifact.to_dict(),
                        component_registry=dict(component_registry),
                        benchmark_public_summary=benchmark_public_summary,
                        runtime_source_index=source_context.inspection_index(),
                        source_inspection_context=source_inspection_context,
                        loop_direction_plan=loop_direction_plan_doc,
                        budget_context=_memory_budget_context({
                            **dict(budget_context),
                            "loop_iteration": iteration,
                            "inspection_round": inspection_round,
                            "candidate_kind": "image_build",
                            "loop_direction_plan_hash": (
                                (loop_direction_plan_doc or {}).get("plan_hash")
                                if isinstance(loop_direction_plan_doc, Mapping)
                                else None
                            ),
                        }),
                        max_requests=4,
                    ),
                    min(settings.draft_timeout_seconds, remaining_call_seconds),
                    3000,
                    "source_inspection",
                )
                if inspection_result is None:
                    # Bug 17: a non-retryable LLM error mid-inspection used to abort the run.
                    # Skip the remaining inspection rounds and continue with what was gathered.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_call_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "stage": "source_inspection_call_failed",
                                "error": inspection_call_error or "source_inspection_call_failed",
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                raw_inspection = inspection_result.content
                openrouter_calls += 1
                estimated_cost += settings.estimated_iteration_cost_usd
                actual_cost_microusd += max(0, int(inspection_result.cost_microusd))
                if inspection_result.provider_usage:
                    provider_usage.append(
                        {
                            **inspection_result.provider_usage,
                            "loop_iteration": iteration,
                            "inspection_round": inspection_round,
                            "call_stage": "source_inspection",
                        }
                    )
                try:
                    requests = parse_code_edit_source_inspection_response(raw_inspection, max_requests=4)
                except Exception as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_parse_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "error": str(exc)[:500],
                                "raw_response_hash": sha256_json({"raw_response": raw_inspection}),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "source_inspection_requested",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "inspection_round": inspection_round,
                            "source_tree_hash": source_context.source_tree_hash,
                            "requests": [request.to_event_doc() for request in requests],
                            "request_hash": sha256_json({"requests": [request.to_event_doc() for request in requests]}),
                        },
                    )
                )
                if any(request.operation == "finish" for request in requests):
                    break
                try:
                    batch = resolve_source_inspection_requests(
                        source_context,
                        requests,
                        already_read_paths=tuple(sorted(read_paths)),
                        max_files=self.builder.config.code_edit_source_inspection_max_files,
                        max_file_bytes=self.builder.config.code_edit_source_inspection_file_bytes,
                        max_total_bytes=max(
                            0,
                            self.builder.config.code_edit_source_inspection_total_bytes - source_bytes_returned,
                        ),
                        max_search_matches=self.builder.config.code_edit_source_inspection_search_matches,
                    )
                except CodeEditBuildError as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_resolution_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "error": str(exc)[:500],
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                source_bytes_returned += batch.bytes_returned
                read_paths = set(batch.read_paths)
                source_inspection_context = _merge_source_inspection_context(
                    source_inspection_context,
                    batch.model_context,
                    total_bytes=source_bytes_returned,
                    read_paths=read_paths,
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_resolved",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "source_inspection_resolved",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "inspection_round": inspection_round,
                            **batch.event_doc,
                        },
                    )
                )
                if source_bytes_returned >= self.builder.config.code_edit_source_inspection_total_bytes:
                    break
                if actual_cost_microusd >= budget_limit_microusd > 0:
                    stop_reason = "compute_budget_exhausted_after_source_inspection"
                    budget_exhausted_after_source_inspection = True
                    break
            if budget_exhausted_after_source_inspection:
                break
            if not read_paths:
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "code_edit_no_source_files_read",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "error": "code_edit_no_source_files_read",
                            "source_tree_hash": source_context.source_tree_hash,
                        },
                    )
                )
                drafts = []
                budget_exhausted_after_call = actual_cost_microusd >= budget_limit_microusd > 0
                raw = ""
            else:
                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    stop_reason = "compute_budget_exhausted_before_code_edit"
                    break
                draft_result, draft_call_error = await self._call_stage_contained(
                    build_code_edit_auto_research_messages(
                        ticket={
                            "ticket_id": str(ticket.get("ticket_id") or ""),
                            "run_id": run_id,
                            "miner_hotkey": ticket.get("miner_hotkey"),
                            "island": ticket.get("island"),
                            "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                            "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                            "requested_loop_count": requested_loop_count,
                            "loop_iteration": iteration,
                        },
                        artifact_manifest=artifact.to_dict(),
                        component_registry=dict(component_registry),
                        benchmark_public_summary=benchmark_public_summary,
                        runtime_source_context=source_context.prompt_context(),
                        source_inspection_context=source_inspection_context,
                        loop_direction_plan=loop_direction_plan_doc,
                        budget_context=_memory_budget_context({
                            **dict(budget_context),
                            "loop_iteration": iteration,
                            "candidate_kind": "image_build",
                            "loop_direction_plan_hash": (
                                (loop_direction_plan_doc or {}).get("plan_hash")
                                if isinstance(loop_direction_plan_doc, Mapping)
                                else None
                            ),
                        }),
                        max_candidates=settings.max_candidates,
                    ),
                    min(settings.draft_timeout_seconds, remaining_call_seconds),
                    3000,
                    "code_edit_draft",
                )
                if draft_result is None:
                    # Bug 17: a non-retryable LLM error on the draft call used to abort the
                    # run and discard prior iterations/candidates. Skip this iteration instead.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_draft_call_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "stage": "code_edit_draft_call_failed",
                                "error": draft_call_error or "code_edit_draft_call_failed",
                            },
                        )
                    )
                    drafts = []
                    raw = ""
                    budget_exhausted_after_call = actual_cost_microusd >= budget_limit_microusd > 0
                else:
                    raw = draft_result.content
                    openrouter_calls += 1
                    estimated_cost += settings.estimated_iteration_cost_usd
                    actual_cost_microusd += max(0, int(draft_result.cost_microusd))
                    if draft_result.provider_usage:
                        provider_usage.append({**draft_result.provider_usage, "loop_iteration": iteration, "call_stage": "code_edit_draft"})
                    budget_exhausted_after_call = (
                        budget_limit_microusd > 0 and actual_cost_microusd >= budget_limit_microusd
                    )
                    try:
                        drafts = parse_code_edit_response(raw, max_candidates=settings.max_candidates)
                    except Exception as exc:
                        no_viable_reason = code_edit_no_viable_patch_reason(raw)
                        if no_viable_reason:
                            await self.event_sink(
                                AutoResearchLoopEvent(
                                    event_type="no_viable_patch",
                                    loop_status="running",
                                    elapsed_seconds=elapsed(),
                                    cost_ledger=_running_cost_ledger(
                                        openrouter_calls,
                                        estimated_cost,
                                        actual_cost_microusd,
                                        "no_viable_patch",
                                    ),
                                    event_doc={
                                        "iteration": iteration,
                                        "reason": no_viable_reason,
                                        "raw_response_hash": sha256_json({"raw_response": raw}),
                                        "loop_direction_plan_hash": (
                                            (loop_direction_plan_doc or {}).get("plan_hash")
                                            if isinstance(loop_direction_plan_doc, Mapping)
                                            else None
                                        ),
                                    },
                                )
                            )
                        else:
                            await self.event_sink(
                                AutoResearchLoopEvent(
                                    event_type="code_edit_validation_failed",
                                    loop_status="running",
                                    elapsed_seconds=elapsed(),
                                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "code_edit_parse_failed"),
                                    event_doc={
                                        "iteration": iteration,
                                        "error": str(exc)[:500],
                                        "raw_response_hash": sha256_json({"raw_response": raw}),
                                    },
                                )
                            )
                        drafts = []
            for draft_index, draft in enumerate(drafts):
                if len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled():
                    # Bug 20: never start a docker build for a candidate that would be
                    # truncated away by the max_candidates cap.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_cap_reached",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "stage": "candidate_cap_reached",
                                "error": "candidate_cap_reached_draft_skipped",
                                "selected_candidate_count": len(selected),
                            },
                        )
                    )
                    break
                draft_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
                if within_run_memory_active and draft_diff_hash in rejected_diff_hashes:
                    # Within-run memory: skip a draft identical to a diff already rejected
                    # earlier in this run instead of paying to judge/build it again.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "within_run_duplicate_rejected_diff",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "stage": "within_run_duplicate_rejected_diff",
                                "error": "duplicate_rejected_diff_skipped",
                                "unified_diff_hash": draft_diff_hash,
                            },
                        )
                    )
                    continue
                node_id = _node_id(run_id, iteration, draft_index, draft)
                source_errors = self.builder.validate_draft_against_source_context(
                    draft,
                    source_context,
                    read_paths=tuple(sorted(read_paths)),
                    require_read=True,
                )
                if source_errors:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_source_context_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(draft.target_files),
                                "error": "; ".join(source_errors)[:500],
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    _record_within_run_rejection(
                        stage="source_context_validation",
                        reason="; ".join(source_errors),
                        iteration_index=iteration,
                        draft=draft,
                        diff_hash=draft_diff_hash,
                    )
                    continue
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "code_edit_drafted"),
                        event_doc={
                            "iteration": iteration,
                            "lane": draft.lane,
                            "plan_path_id": draft.plan_path_id,
                            "loop_direction_plan_hash": (
                                (loop_direction_plan_doc or {}).get("plan_hash")
                                if isinstance(loop_direction_plan_doc, Mapping)
                                else None
                            ),
                            "target_files": list(draft.target_files),
                            "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                            "hypothesis": {
                                "failure_mode": draft.failure_mode,
                                "mechanism": draft.mechanism,
                                "expected_improvement": draft.expected_improvement,
                                "risk": draft.risk,
                                "predicted_delta": draft.predicted_delta,
                            },
                        },
                    )
                )
                (
                    candidate_draft,
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    patch_budget_exhausted,
                ) = await self._ensure_patch_applies_or_repair(
                    draft=draft,
                    run_id=run_id,
                    node_id=node_id,
                    iteration=iteration,
                    settings=settings,
                    artifact=artifact,
                    source_context=source_context,
                    source_inspection_context=source_inspection_context,
                    read_paths=tuple(sorted(read_paths)),
                    budget_context=budget_context,
                    budget_limit_microusd=budget_limit_microusd,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                    within_run_memory=_within_run_memory_doc(),
                )
                if patch_budget_exhausted:
                    budget_exhausted_after_call = True
                    continue
                if candidate_draft is None:
                    _record_within_run_rejection(
                        stage="patch_apply_repair_exhausted",
                        reason="patch did not apply after repair attempts",
                        iteration_index=iteration,
                        draft=draft,
                        diff_hash=draft_diff_hash,
                    )
                    continue
                (
                    alignment_ok,
                    candidate_draft,
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    alignment_budget_exhausted,
                ) = await self._judge_plan_alignment(
                    draft=candidate_draft,
                    loop_direction_plan=loop_direction_plan_doc,
                    prior_attempts=prior_attempts,
                    node_id=node_id,
                    iteration=iteration,
                    settings=settings,
                    budget_limit_microusd=budget_limit_microusd,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                )
                if alignment_budget_exhausted:
                    budget_exhausted_after_call = True
                    continue
                if not alignment_ok:
                    alignment_doc = dict(candidate_draft.plan_alignment or {})
                    _record_within_run_rejection(
                        stage="plan_alignment_rejected",
                        reason=str(alignment_doc.get("blocking_issue") or alignment_doc.get("reason") or "plan alignment rejected"),
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    continue
                build_completed = False
                try:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"),
                            event_doc={
                                "iteration": iteration,
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "loop_direction_plan_hash": (
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                            },
                        )
                    )
                    # Bug 20: candidate_index is a monotonically increasing per-run build counter
                    # (previously len(selected), which repeats after the post-cap truncation and
                    # overwrote the persisted S3 source-diff artifact key each iteration).
                    build = await self._build_candidate_with_heartbeat(
                        draft=candidate_draft,
                        artifact=artifact,
                        run_id=run_id,
                        candidate_index=built_candidate_total,
                        source_context=source_context,
                        node_id=node_id,
                        iteration=iteration,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    built_candidate_total += 1
                    build_completed = True
                    # Bug 5: persist a full rehydration doc so a paused/requeued run can restore
                    # this candidate on resume. Best-effort: failure only loses restorability.
                    rehydration_doc = await _write_private_loop_candidate_artifact(
                        artifact=artifact,
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        build=build,
                    )
                    selected.append(
                        BuiltCodeEditCandidate(
                            draft=candidate_draft,
                            build=build,
                            node_id=node_id,
                            iteration=iteration,
                            rehydration_artifact_uri=str(rehydration_doc.get("loop_candidate_artifact_uri") or ""),
                            rehydration_artifact_hash=str(rehydration_doc.get("loop_candidate_artifact_hash") or ""),
                        )
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_passed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            candidate_artifact_hash=build.candidate_model_manifest.model_artifact_hash,
                            candidate_patch_hash=sha256_json(build.code_edit_manifest),
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_passed"),
                            event_doc={
                                "iteration": iteration,
                                "candidate_kind": "image_build",
                                "candidate_model_manifest_hash": build.candidate_model_manifest.manifest_hash,
                                "candidate_source_diff_hash": build.source_diff_hash,
                                "build_doc_hash": build.build_doc.get("build_doc_hash"),
                                "loop_direction_plan_hash": (
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                                **{
                                    key: value
                                    for key, value in rehydration_doc.items()
                                    if key.startswith("loop_candidate_artifact")
                                },
                            },
                        )
                    )
                except (CodeEditPrivateTestError, CodeEditImageBuildError, CodeEditPatchApplyError) as exc:
                    event_type = str(getattr(exc, "failure_stage", "") or "candidate_build_failed")
                    _record_within_run_rejection(
                        stage=event_type,
                        reason=str(exc),
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    diagnostic_doc = await _write_private_code_edit_diagnostic(
                        artifact=artifact,
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        stage=event_type,
                        draft=candidate_draft,
                        error=exc,
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type=event_type,
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, event_type),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": str(exc)[:500],
                                "error_hash": sha256_json({"error": str(exc)}),
                                **diagnostic_doc,
                            },
                        )
                    )
                except CodeEditBuildError as exc:
                    _record_within_run_rejection(
                        stage="candidate_build_failed",
                        reason=str(exc),
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_failed"),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": str(exc)[:500],
                                "error_hash": sha256_json({"error": str(exc)}),
                            },
                        )
                    )
                except Exception as exc:
                    # Bug 17: an unexpected infra error during build/event emission used to
                    # abort the run. Contain it to this candidate; the run keeps whatever it
                    # has already built.
                    if not _stage_error_containment_enabled():
                        raise
                    if not build_completed:
                        _record_within_run_rejection(
                            stage="candidate_build_unexpected_error",
                            reason=str(exc),
                            iteration_index=iteration,
                            draft=candidate_draft,
                        )
                    try:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_build_failed" if not build_completed else "code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                node_id=node_id,
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_build_unexpected_error",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "stage": (
                                        "candidate_build_unexpected_error"
                                        if not build_completed
                                        else "post_build_event_emit_failed"
                                    ),
                                    "target_files": list(candidate_draft.target_files),
                                    "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                    "error": str(exc)[:500],
                                    "error_hash": sha256_json({"error": str(exc)}),
                                },
                            )
                        )
                    except Exception:
                        pass
            selected = selected[: settings.max_candidates]
            try:
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="code_edit_iteration_completed",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                )
            except Exception:
                # Bug 17: a transient checkpoint-write failure must not fail a run that may
                # already hold built candidates; the previous checkpoint remains usable.
                if not _stage_error_containment_enabled():
                    raise
            if should_pause and await should_pause():
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )
            if budget_exhausted_after_call:
                stop_reason = "compute_budget_exhausted_after_code_edit"
                break

        if selected:
            remaining_minimum = settings.min_seconds - elapsed()
            remaining_maximum = settings.max_seconds - elapsed()
            if _min_runtime_skip_when_selected_enabled():
                # Candidates are already selected; parking the worker slot until min_seconds
                # elapses is pure waste. Proceed straight to finalization.
                remaining_minimum = 0.0
            if remaining_minimum > 0 and remaining_maximum > 0:
                import asyncio

                sleep_remaining = min(remaining_minimum, remaining_maximum)
                while sleep_remaining > 0:
                    await asyncio.sleep(min(5.0, sleep_remaining))
                    sleep_remaining = min(settings.min_seconds - elapsed(), settings.max_seconds - elapsed())
                    if should_pause and await should_pause():
                        break
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="pause_after_code_edit_minimum_runtime",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                )
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )
        if not selected:
            stop_reason = "no_valid_image_build_candidates"

        for index, candidate in enumerate(selected):
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="candidate_selected",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=candidate.node_id,
                    candidate_artifact_hash=candidate.build.candidate_model_manifest.model_artifact_hash,
                    candidate_patch_hash=sha256_json(candidate.build.code_edit_manifest),
                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_selected"),
                    event_doc={
                        "candidate_index": index,
                        "iteration": candidate.iteration,
                        "candidate_kind": "image_build",
                        "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                        "candidate_source_diff_hash": candidate.build.source_diff_hash,
                        "redacted_summary": candidate.draft.redacted_summary,
                        "loop_direction_plan_hash": (
                            (loop_direction_plan_doc or {}).get("plan_hash")
                            if isinstance(loop_direction_plan_doc, Mapping)
                            else None
                        ),
                        "selected_path_id": (
                            (loop_direction_plan_doc or {}).get("selected_path_id")
                            if isinstance(loop_direction_plan_doc, Mapping)
                            else candidate.draft.plan_path_id
                        ),
                        "plan_alignment": dict(candidate.draft.plan_alignment or {}),
                    },
                )
            )

        result = self._result(
            selected=selected,
            status="completed" if selected else "failed",
            stop_reason=stop_reason,
            iterations_completed=iteration,
            elapsed_seconds=elapsed(),
            estimated_cost=estimated_cost,
            actual_cost_microusd=actual_cost_microusd,
            openrouter_calls=openrouter_calls,
            provider_usage=provider_usage,
            checkpoint=last_checkpoint,
        )
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_completed" if selected else "loop_failed",
                loop_status="completed" if selected else "failed",
                elapsed_seconds=result.elapsed_seconds,
                provider_usage=list(result.provider_usage),
                cost_ledger=result.cost_ledger(),
                event_doc={
                    "candidate_kind": "image_build",
                    "iterations_completed": result.iterations_completed,
                    "selected_candidate_count": len(selected),
                    "stop_reason": result.stop_reason,
                    "loop_direction_plan_hash": (
                        (loop_direction_plan_doc or {}).get("plan_hash")
                        if isinstance(loop_direction_plan_doc, Mapping)
                        else None
                    ),
                    "selected_path_id": (
                        (loop_direction_plan_doc or {}).get("selected_path_id")
                        if isinstance(loop_direction_plan_doc, Mapping)
                        else None
                    ),
                    "gateway_scoring_queue_visible_after_this_event": bool(selected),
                },
            )
        )
        _cleanup_source_tmp()
        return result

    async def _emit_checkpoint(
        self,
        *,
        run_id: str,
        settings: AutoResearchLoopSettings,
        artifact: PrivateModelArtifactManifest,
        model_id: str,
        budget_context: Mapping[str, Any],
        iterations_completed: int,
        elapsed_seconds: float,
        selected: Sequence[BuiltCodeEditCandidate],
        provider_usage: Sequence[Mapping[str, Any]],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        stage: str,
        loop_direction_plan: Mapping[str, Any] | None = None,
        built_candidate_count: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "stage": stage,
            "candidate_kind": "image_build",
            "model_id": model_id,
            "artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "settings": _settings_doc(settings),
            "budget_context": _safe_budget_doc(budget_context),
            "iterations_completed": int(iterations_completed),
            "next_iteration": int(iterations_completed) + 1,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "built_candidate_count": max(0, int(built_candidate_count)),
            "selected_candidates": [
                {
                    "node_id": candidate.node_id,
                    "iteration": candidate.iteration,
                    "candidate_artifact_hash": candidate.build.candidate_model_manifest.model_artifact_hash,
                    "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                    "candidate_source_diff_hash": candidate.build.source_diff_hash,
                    "build_doc_hash": candidate.build.build_doc.get("build_doc_hash"),
                    "draft": _redacted_draft_doc(candidate.draft),
                    # Bug 5: private artifact refs (hashes/URIs only — never raw diffs or
                    # manifests) that let resume rehydrate this already-built candidate.
                    "rehydration_artifact_uri": candidate.rehydration_artifact_uri or None,
                    "rehydration_artifact_hash": candidate.rehydration_artifact_hash or None,
                    "source_diff_artifact_uri": candidate.build.build_doc.get("source_diff_artifact_uri"),
                    "source_diff_artifact_hash": candidate.build.build_doc.get("source_diff_artifact_hash"),
                }
                for candidate in selected
            ],
            "loop_direction_plan": dict(loop_direction_plan or {}),
            "loop_direction_plan_hash": (
                (loop_direction_plan or {}).get("plan_hash")
                if isinstance(loop_direction_plan, Mapping)
                else None
            ),
            "openrouter_call_count": int(openrouter_calls),
            "estimated_cost_usd": round(float(estimated_cost), 6),
            "actual_openrouter_cost_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
            "actual_openrouter_cost_microusd": int(actual_cost_microusd),
            "provider_usage": [dict(item) for item in provider_usage if isinstance(item, Mapping)],
        }
        checkpoint = {**payload, "checkpoint_hash": sha256_json(payload)}
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="checkpoint_saved",
                loop_status="running",
                elapsed_seconds=elapsed_seconds,
                provider_usage=[dict(item) for item in provider_usage if isinstance(item, Mapping)],
                cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "checkpoint_saved"),
                event_doc={"checkpoint": checkpoint},
            )
        )
        return checkpoint

    def _result(
        self,
        *,
        selected: Sequence[BuiltCodeEditCandidate],
        status: str,
        stop_reason: str,
        iterations_completed: int,
        elapsed_seconds: float,
        estimated_cost: float,
        actual_cost_microusd: int,
        openrouter_calls: int,
        provider_usage: Sequence[Mapping[str, Any]],
        checkpoint: dict[str, Any] | None,
    ) -> CodeEditLoopResult:
        return CodeEditLoopResult(
            selected_candidates=tuple(selected[: self.settings.normalized().max_candidates]),
            iterations_completed=int(iterations_completed),
            stop_reason=stop_reason,
            elapsed_seconds=round(float(elapsed_seconds), 3),
            estimated_cost_usd=round(float(estimated_cost), 6),
            actual_openrouter_cost_usd=round(int(actual_cost_microusd) / 1_000_000, 6),
            actual_openrouter_cost_microusd=int(actual_cost_microusd),
            openrouter_call_count=int(openrouter_calls),
            provider_usage=tuple(dict(item) for item in provider_usage if isinstance(item, Mapping)),
            status=status,
            checkpoint_doc=checkpoint,
        )

    async def _judge_plan_alignment(
        self,
        *,
        draft: CodeEditDraft,
        loop_direction_plan: Mapping[str, Any] | None,
        prior_attempts: Sequence[Mapping[str, Any]],
        node_id: str,
        iteration: int,
        settings: AutoResearchLoopSettings,
        budget_limit_microusd: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: list[dict[str, Any]],
    ) -> tuple[bool, CodeEditDraft, int, float, int, bool]:
        if not loop_direction_plan:
            return True, draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        heuristic_errors = code_edit_plan_alignment_errors(
            draft,
            loop_direction_plan=loop_direction_plan,
            prior_attempts=prior_attempts,
            strict=bool(self.builder.config.loop_novelty_strict),
        )
        if heuristic_errors:
            verdict_doc = {
                "schema_version": "1.0",
                "verdict": "fail",
                "source": "local_heuristic",
                "reason": "; ".join(heuristic_errors)[:700],
                "detected_lane": draft.lane,
                "detected_mechanism": draft.mechanism,
                "novel": not any("duplicate" in error for error in heuristic_errors),
                "blocking_issue": heuristic_errors[0],
                "confidence": 1.0,
                "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                "selected_path_id": loop_direction_plan.get("selected_path_id"),
            }
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="plan_alignment_judged",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=node_id,
                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "plan_alignment_judged",
                    ),
                    event_doc={
                        "iteration": iteration,
                        "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                        "selected_path_id": loop_direction_plan.get("selected_path_id"),
                        "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                        "verdict": verdict_doc,
                    },
                )
            )
            await self._emit_alignment_rejection(
                draft=draft,
                node_id=node_id,
                iteration=iteration,
                verdict_doc=verdict_doc,
                loop_direction_plan=loop_direction_plan,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
            )
            return False, replace(draft, plan_alignment=verdict_doc), openrouter_calls, estimated_cost, actual_cost_microusd, False

        if not self.builder.config.loop_alignment_judge_enabled:
            verdict_doc = {
                "schema_version": "1.0",
                "verdict": "pass",
                "source": "local_heuristic",
                "reason": "alignment judge disabled; local heuristic passed",
                "detected_lane": draft.lane,
                "detected_mechanism": draft.mechanism,
                "novel": True,
                "blocking_issue": "",
                "confidence": 0.5,
                "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                "selected_path_id": loop_direction_plan.get("selected_path_id"),
            }
            return True, replace(draft, plan_alignment=verdict_doc), openrouter_calls, estimated_cost, actual_cost_microusd, False

        if elapsed() >= settings.max_seconds:
            return False, draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        if _would_exceed_budget(
            actual_cost_microusd,
            _estimated_call_microusd(settings.estimated_iteration_cost_usd),
            budget_limit_microusd,
        ):
            await self._emit_alignment_rejection(
                draft=draft,
                node_id=node_id,
                iteration=iteration,
                verdict_doc={
                    "schema_version": "1.0",
                    "verdict": "fail",
                    "source": "budget_guard",
                    "reason": "compute budget exhausted before plan alignment judge",
                    "blocking_issue": "compute_budget_exhausted_before_plan_alignment_judge",
                    "novel": True,
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                },
                loop_direction_plan=loop_direction_plan,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
            )
            return False, draft, openrouter_calls, estimated_cost, actual_cost_microusd, True

        # Bug 21: a judge parse failure used to be recorded as a hard rejection at
        # confidence 1.0. Retry the judge once on a failed/unparseable call; if it is still
        # unusable, accept neutrally on the already-passed local heuristics instead of
        # recording a confident rejection.
        judge_attempt_limit = 2 if _judge_parse_soft_skip_enabled() else 1
        verdict_doc: dict[str, Any] | None = None
        raw_judge = ""
        judge_failure_reason = ""
        for judge_attempt in range(1, judge_attempt_limit + 1):
            if judge_attempt > 1 and (
                elapsed() >= settings.max_seconds
                or _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                )
            ):
                break
            remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
            judge_result, judge_call_error = await self._call_stage_contained(
                build_plan_alignment_judge_messages(
                    loop_direction_plan=loop_direction_plan,
                    draft=draft,
                    prior_attempts=prior_attempts,
                ),
                min(settings.draft_timeout_seconds, remaining_call_seconds),
                self.builder.config.loop_alignment_judge_max_tokens,
                "plan_alignment_judge",
            )
            if judge_result is None:
                judge_failure_reason = judge_call_error or "plan_alignment_judge_call_failed"
                continue
            raw_judge = judge_result.content
            openrouter_calls += 1
            estimated_cost += settings.estimated_iteration_cost_usd
            actual_cost_microusd += max(0, int(judge_result.cost_microusd))
            if judge_result.provider_usage:
                provider_usage.append({**judge_result.provider_usage, "loop_iteration": iteration, "call_stage": "plan_alignment_judge"})
            try:
                verdict = parse_plan_alignment_judge_response(raw_judge)
                verdict_doc = {**verdict.to_dict(), "source": "model_judge"}
                break
            except Exception as exc:
                judge_failure_reason = str(exc)[:700]
        if verdict_doc is None:
            if _judge_parse_soft_skip_enabled():
                verdict_doc = {
                    "schema_version": "1.0",
                    "verdict": "pass",
                    "source": "model_judge_unavailable",
                    "reason": (
                        "plan alignment judge unavailable or unparseable after retry; "
                        "accepted on local heuristics: " + judge_failure_reason
                    )[:700],
                    "detected_lane": draft.lane,
                    "detected_mechanism": draft.mechanism,
                    "novel": True,
                    "blocking_issue": "",
                    "confidence": 0.0,
                    "raw_response_hash": sha256_json({"raw_response": raw_judge}),
                }
            else:
                verdict_doc = {
                    "schema_version": "1.0",
                    "verdict": "fail",
                    "source": "model_judge_parse_failed",
                    "reason": judge_failure_reason,
                    "detected_lane": "",
                    "detected_mechanism": "",
                    "novel": False,
                    "blocking_issue": "plan_alignment_judge_parse_failed",
                    "confidence": 1.0,
                    "raw_response_hash": sha256_json({"raw_response": raw_judge}),
                }
        verdict_doc["loop_direction_plan_hash"] = loop_direction_plan.get("plan_hash")
        verdict_doc["selected_path_id"] = loop_direction_plan.get("selected_path_id")
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="plan_alignment_judged",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                provider_usage=([provider_usage[-1]] if provider_usage else []),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "plan_alignment_judged",
                ),
                event_doc={
                    "iteration": iteration,
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                    "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                    "verdict": verdict_doc,
                },
            )
        )
        accepted = verdict_doc.get("verdict") == "pass" and verdict_doc.get("novel") is not False
        judged_draft = replace(draft, plan_alignment=verdict_doc)
        if accepted:
            return True, judged_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        await self._emit_alignment_rejection(
            draft=judged_draft,
            node_id=node_id,
            iteration=iteration,
            verdict_doc=verdict_doc,
            loop_direction_plan=loop_direction_plan,
            elapsed=elapsed,
            openrouter_calls=openrouter_calls,
            estimated_cost=estimated_cost,
            actual_cost_microusd=actual_cost_microusd,
            provider_usage=provider_usage,
        )
        return False, judged_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False

    async def _emit_alignment_rejection(
        self,
        *,
        draft: CodeEditDraft,
        node_id: str,
        iteration: int,
        verdict_doc: Mapping[str, Any],
        loop_direction_plan: Mapping[str, Any],
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: Sequence[Mapping[str, Any]],
    ) -> None:
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="code_edit_alignment_rejected",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                provider_usage=([dict(provider_usage[-1])] if provider_usage else []),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "code_edit_alignment_rejected",
                ),
                event_doc={
                    "iteration": iteration,
                    "lane": draft.lane,
                    "plan_path_id": draft.plan_path_id,
                    "target_files": list(draft.target_files),
                    "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                    "verdict": dict(verdict_doc),
                },
            )
        )

    async def _ensure_patch_applies_or_repair(
        self,
        *,
        draft: CodeEditDraft,
        run_id: str,
        node_id: str,
        iteration: int,
        settings: AutoResearchLoopSettings,
        artifact: PrivateModelArtifactManifest,
        source_context: Any,
        source_inspection_context: Mapping[str, Any],
        read_paths: Sequence[str],
        budget_context: Mapping[str, Any],
        budget_limit_microusd: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: list[dict[str, Any]],
        within_run_memory: Mapping[str, Any] | None = None,
    ) -> tuple[CodeEditDraft | None, int, float, int, bool]:
        candidate_draft = draft.with_unified_diff(draft.unified_diff)
        max_repairs = max(0, int(self.builder.config.code_edit_patch_repair_attempts))
        for repair_attempt in range(0, max_repairs + 1):
            try:
                self.builder.check_patch_applies(
                    draft=candidate_draft,
                    parent_artifact=artifact,
                    source_context=source_context,
                )
                return candidate_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
            except CodeEditPatchApplyError as exc:
                diagnostic_doc = await _write_private_code_edit_diagnostic(
                    artifact=artifact,
                    run_id=run_id,
                    node_id=node_id,
                    iteration=iteration,
                    stage="candidate_patch_apply_failed",
                    draft=candidate_draft,
                    error=exc,
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="candidate_patch_apply_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "candidate_patch_apply_failed",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "error": str(exc)[:500],
                            "error_hash": sha256_json({"error": str(exc)}),
                            "stderr_hash": sha256_json({"stderr": getattr(exc, "stderr", "")}),
                            **diagnostic_doc,
                        },
                    )
                )
                if repair_attempt >= max_repairs:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempts": max_repairs,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": str(exc)[:500],
                                "error_hash": sha256_json({"error": str(exc)}),
                            },
                        )
                    )
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, False

                if elapsed() >= settings.max_seconds:
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_budget_exhausted",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "error": "compute_budget_exhausted_before_code_edit_repair",
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            },
                        )
                    )
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, True

                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_repair_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "code_edit_repair_requested",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "apply_error_hash": sha256_json({"error": str(exc)}),
                        },
                    )
                )

                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                repair_result = _coerce_call_result(
                    await self.call_openrouter(
                        build_code_edit_repair_messages(
                            draft=candidate_draft,
                            apply_error=str(exc),
                            source_inspection_context=source_inspection_context,
                            runtime_source_context=source_context.prompt_context(),
                            budget_context={
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "candidate_kind": "image_build",
                            },
                            repair_attempt=repair_attempt + 1,
                            max_candidates=1,
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        3000,
                        "code_edit_repair",
                    )
                )
                openrouter_calls += 1
                estimated_cost += settings.estimated_iteration_cost_usd
                actual_cost_microusd += max(0, int(repair_result.cost_microusd))
                if repair_result.provider_usage:
                    provider_usage.append(
                        {
                            **repair_result.provider_usage,
                            "loop_iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "call_stage": "code_edit_repair",
                        }
                    )
                try:
                    repaired_drafts = parse_code_edit_repair_response(
                        repair_result.content,
                        original_draft=draft,
                    )
                except Exception as parse_exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_parse_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "error": str(parse_exc)[:500],
                                "raw_response_hash": sha256_json({"raw_response": repair_result.content}),
                            },
                        )
                    )
                    if repair_attempt + 1 >= max_repairs:
                        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                    continue
                repaired = repaired_drafts[0].with_unified_diff(repaired_drafts[0].unified_diff)
                source_errors = self.builder.validate_draft_against_source_context(
                    repaired,
                    source_context,
                    read_paths=read_paths,
                    require_read=True,
                )
                if source_errors:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_source_context_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "target_files": list(repaired.target_files),
                                "error": "; ".join(source_errors)[:500],
                                "source_diff_hash": sha256_json({"unified_diff": repaired.unified_diff}),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    if repair_attempt + 1 >= max_repairs:
                        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                    continue
                candidate_draft = repaired
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_repair_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "code_edit_repair_drafted",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "original_source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                        },
                    )
                )
        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False


async def _write_private_code_edit_diagnostic(
    *,
    artifact: PrivateModelArtifactManifest,
    run_id: str,
    node_id: str,
    iteration: int,
    stage: str,
    draft: CodeEditDraft,
    error: BaseException,
) -> dict[str, Any]:
    manifest_uri = str(getattr(artifact, "manifest_uri", "") or "")
    if not manifest_uri.startswith("s3://"):
        return {"diagnostic_artifact_skipped": "manifest_uri_not_s3"}
    try:
        bucket, key = _parse_s3_uri(manifest_uri)
    except ValueError as exc:
        return {"diagnostic_artifact_error": str(exc)[:200]}
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_node = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(node_id or "node"))[:80]
    safe_stage = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(stage or "failure"))[:80]
    object_key = f"{base_prefix}/candidates/{run_id}/diagnostics/{int(iteration):03d}-{safe_node}-{safe_stage}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_failure_diagnostic",
        "run_id": str(run_id),
        "node_id": str(node_id),
        "iteration": int(iteration),
        "stage": str(stage),
        "target_files": list(draft.target_files),
        "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
        "unified_diff": draft.unified_diff,
        "error": _diagnostic_text(str(error), limit=12000),
        "stderr": _diagnostic_text(str(getattr(error, "stderr", "") or ""), limit=12000),
        "stdout": _diagnostic_text(str(getattr(error, "stdout", "") or ""), limit=12000),
        "exit_code": getattr(error, "exit_code", None),
    }
    payload_hash = sha256_json(payload)

    def _put() -> None:
        import boto3  # type: ignore

        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=object_key,
            Body=json.dumps({**payload, "diagnostic_hash": payload_hash}, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

    try:
        await asyncio.to_thread(_put)
    except Exception as exc:
        return {
            "diagnostic_artifact_hash": payload_hash,
            "diagnostic_artifact_error": str(exc)[:300],
        }
    return {
        "diagnostic_artifact_uri": f"s3://{bucket}/{object_key}",
        "diagnostic_artifact_hash": payload_hash,
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri or "")
    if not raw.startswith("s3://"):
        raise ValueError("s3_uri_required")
    rest = raw[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        raise ValueError("invalid_s3_uri")
    return bucket, key


def _diagnostic_text(value: str, *, limit: int) -> str:
    text = str(value or "")
    text = re.sub(r"sk-or-[A-Za-z0-9._:-]+", "[redacted-openrouter-key]", text)
    text = re.sub(r"sb_secret_[A-Za-z0-9._:-]+", "[redacted-supabase-service-role-key]", text)
    text = re.sub(r"sb_publishable_[A-Za-z0-9._:-]+", "[redacted-supabase-anon-key]", text)
    text = re.sub(r"AKIA[A-Z0-9]{16}", "[redacted-aws-access-key-id]", text)
    text = re.sub(r"https?://[^@\s]+@([^\s/]+)", r"[redacted-proxy-url]@\1", text)
    text = re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[redacted]", text)
    replacements = (
        "service_role",
        "openrouter_api_key",
        "raw_openrouter_key",
        "raw_secret",
        "aws_secret_access_key",
        "password",
        "proxy",
        "webshare",
    )
    lowered = text.lower()
    if any(marker in lowered for marker in replacements):
        return "[redacted secret-like diagnostic text]"
    return text[: max(1, int(limit))]


def _memory_safe_text(value: str) -> str:
    """Sanitized short text for within-run memory records (§6.2-5): rejection
    reasons re-enter later prompts and event docs, so secret-shaped content is
    redacted the same way diagnostics are."""
    return _diagnostic_text(str(value or ""), limit=280)


async def _write_private_loop_candidate_artifact(
    *,
    artifact: PrivateModelArtifactManifest,
    run_id: str,
    node_id: str,
    iteration: int,
    draft: CodeEditDraft,
    build: CodeEditBuildResult,
) -> dict[str, Any]:
    """Persist a full rehydration doc for a built candidate (bug #5).

    The checkpoint keeps only URI + hash; this S3 artifact carries everything
    needed to reconstruct the ``BuiltCodeEditCandidate`` on resume. Best-effort:
    a failed write only loses restorability for this candidate.
    """
    manifest_uri = str(getattr(artifact, "manifest_uri", "") or "")
    if not manifest_uri.startswith("s3://"):
        return {"loop_candidate_artifact_skipped": "manifest_uri_not_s3"}
    try:
        bucket, key = _parse_s3_uri(manifest_uri)
    except ValueError as exc:
        return {"loop_candidate_artifact_error": str(exc)[:200]}
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_node = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(node_id or "node"))[:80]
    object_key = f"{base_prefix}/candidates/{run_id}/loop-candidates/{int(iteration):03d}-{safe_node}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_loop_candidate_rehydration",
        "run_id": str(run_id),
        "node_id": str(node_id),
        "iteration": int(iteration),
        "draft": draft.to_dict(),
        "candidate_model_manifest": build.candidate_model_manifest.to_dict(),
        "code_edit_manifest": dict(build.code_edit_manifest),
        "source_diff_hash": str(build.source_diff_hash),
        "build_doc": dict(build.build_doc),
    }
    payload_hash = sha256_json(payload)

    def _put() -> None:
        import boto3  # type: ignore

        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=object_key,
            Body=json.dumps({**payload, "loop_candidate_artifact_hash": payload_hash}, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

    try:
        await asyncio.to_thread(_put)
    except Exception as exc:
        return {
            "loop_candidate_artifact_hash": payload_hash,
            "loop_candidate_artifact_error": str(exc)[:300],
        }
    return {
        "loop_candidate_artifact_uri": f"s3://{bucket}/{object_key}",
        "loop_candidate_artifact_hash": payload_hash,
    }


def _rehydrated_candidate_from_artifact_payload(
    payload: Mapping[str, Any],
) -> BuiltCodeEditCandidate:
    """Reconstruct a ``BuiltCodeEditCandidate`` from a rehydration artifact.

    Raises on any shape mismatch — the caller treats a failed candidate as
    unrestorable and degrades to the legacy empty-``selected`` behavior.
    """
    stored = dict(payload)
    expected_hash = str(stored.pop("loop_candidate_artifact_hash", "") or "")
    if expected_hash and sha256_json(stored) != expected_hash:
        raise ValueError("loop_candidate_artifact_hash_mismatch")
    draft_doc = dict(stored.get("draft") or {})
    draft_fields = {f.name for f in fields(CodeEditDraft)}
    draft_kwargs = {name: value for name, value in draft_doc.items() if name in draft_fields}
    draft_kwargs["target_files"] = tuple(draft_kwargs.get("target_files") or ())
    draft_kwargs["plan_alignment"] = dict(draft_kwargs.get("plan_alignment") or {})
    draft = CodeEditDraft(**draft_kwargs)
    build = CodeEditBuildResult(
        candidate_model_manifest=PrivateModelArtifactManifest.from_mapping(
            stored["candidate_model_manifest"]
        ),
        code_edit_manifest=dict(stored.get("code_edit_manifest") or {}),
        source_diff_hash=str(stored.get("source_diff_hash") or ""),
        build_doc=dict(stored.get("build_doc") or {}),
    )
    return BuiltCodeEditCandidate(
        draft=draft,
        build=build,
        node_id=str(stored.get("node_id") or ""),
        iteration=int(stored.get("iteration") or 0),
    )


def _node_id(run_id: str, iteration: int, candidate_index: int, draft: CodeEditDraft) -> str:
    digest = sha256_json(
        {
            "run_id": run_id,
            "iteration": iteration,
            "candidate_index": candidate_index,
            "draft": _redacted_draft_doc(draft),
        }
    ).split(":", 1)[1]
    return f"node:code-edit:{digest[:16]}"


def _redacted_draft_doc(draft: CodeEditDraft) -> dict[str, Any]:
    return {
        "failure_mode": draft.failure_mode,
        "mechanism": draft.mechanism,
        "expected_improvement": draft.expected_improvement,
        "risk": draft.risk,
        "lane": draft.lane,
        "plan_path_id": draft.plan_path_id,
        "plan_alignment": dict(draft.plan_alignment or {}),
        "target_files": list(draft.target_files),
        "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
        "redacted_summary": draft.redacted_summary,
        "test_plan": draft.test_plan,
        "rollback_plan": draft.rollback_plan,
        "predicted_delta": draft.predicted_delta,
    }


def _focus_signature_hash(ticket: Mapping[str, Any]) -> str:
    focus = _ticket_doc_value(ticket, "brief_public_summary")
    normalized = re.sub(r"\s+", " ", str(focus or "").strip().lower())[:2000]
    return sha256_json({"focus": normalized})


def _prior_attempts_from_budget_context(budget_context: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    memory = budget_context.get("active_parent_outcome_memory")
    if not isinstance(memory, Mapping):
        return ()
    attempts = memory.get("recent_attempts")
    if not isinstance(attempts, list):
        return ()
    cleaned: list[dict[str, Any]] = []
    for item in attempts[:100]:
        if not isinstance(item, Mapping):
            continue
        cleaned.append(
            {
                "candidate_id": str(item.get("candidate_id") or "")[:120],
                "run_id": str(item.get("run_id") or "")[:120],
                "lane": str(item.get("lane") or "")[:120],
                "plan_path_id": str(item.get("plan_path_id") or "")[:120],
                "target_files": [
                    str(path)[:240]
                    for path in (item.get("target_files") or [])
                    if isinstance(path, str)
                ][:20],
                "unified_diff_hash": str(item.get("unified_diff_hash") or "")[:120],
                "candidate_source_diff_hash": str(item.get("candidate_source_diff_hash") or "")[:120],
                "semantic_edit_summary": str(item.get("semantic_edit_summary") or "")[:500],
                "status": str(item.get("status") or "")[:120],
                "reason": str(item.get("reason") or "")[:240],
            }
        )
    return tuple(cleaned)


def _merge_source_inspection_context(
    existing: Mapping[str, Any],
    update: Mapping[str, Any],
    *,
    total_bytes: int,
    read_paths: set[str],
) -> dict[str, Any]:
    existing_results = existing.get("results") if isinstance(existing, Mapping) else []
    update_results = update.get("results") if isinstance(update, Mapping) else []
    results: list[dict[str, Any]] = []
    for item in list(existing_results or []) + list(update_results or []):
        if isinstance(item, Mapping):
            results.append(dict(item))
    return {
        "schema_version": "1.0",
        "source_tree_hash": str(update.get("source_tree_hash") or existing.get("source_tree_hash") or ""),
        "read_files": sorted(read_paths),
        "results": results,
        "bytes_returned": int(total_bytes),
        "context_hash": sha256_json(
            {
                "read_files": sorted(read_paths),
                "result_hashes": [sha256_json(item) for item in results],
                "bytes_returned": int(total_bytes),
            }
        ),
    }


def _ticket_doc_value(ticket: Mapping[str, Any], key: str) -> Any:
    if key in ticket:
        return ticket.get(key)
    doc = ticket.get("ticket_doc")
    if isinstance(doc, Mapping):
        return doc.get(key)
    return None
