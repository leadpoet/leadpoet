"""Code-edit image candidate generation loop for hosted Research Lab runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

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
    OpenRouterCaller,
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
    CodeEditDraft,
    build_code_edit_auto_research_messages,
    build_code_edit_repair_messages,
    build_code_edit_source_inspection_messages,
    parse_code_edit_repair_response,
    parse_code_edit_response,
    parse_code_edit_source_inspection_response,
)
from research_lab.eval import PrivateModelArtifactManifest


@dataclass(frozen=True)
class BuiltCodeEditCandidate:
    draft: CodeEditDraft
    build: CodeEditBuildResult
    node_id: str
    iteration: int


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
    call_openrouter: OpenRouterCaller
    event_sink: Any
    builder: CodeEditCandidateBuilder

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

        last_checkpoint: dict[str, Any] | None = None
        stop_reason = "max_iterations"
        while iteration < settings.max_iterations:
            if elapsed() >= settings.max_seconds:
                stop_reason = "max_seconds"
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
                inspection_result = _coerce_call_result(
                    await self.call_openrouter(
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
                            budget_context={
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "inspection_round": inspection_round,
                                "candidate_kind": "image_build",
                            },
                            max_requests=4,
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        3000,
                    )
                )
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
                draft_result = _coerce_call_result(
                    await self.call_openrouter(
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
                            budget_context={
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "candidate_kind": "image_build",
                            },
                            max_candidates=settings.max_candidates,
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        3000,
                    )
                )
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
                )
                if patch_budget_exhausted:
                    budget_exhausted_after_call = True
                    continue
                if candidate_draft is None:
                    continue
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
                            },
                        )
                    )
                    build = self.builder.build(
                        draft=candidate_draft,
                        parent_artifact=artifact,
                        run_id=run_id,
                        candidate_index=len(selected),
                        source_context=source_context,
                    )
                    selected.append(BuiltCodeEditCandidate(draft=candidate_draft, build=build, node_id=node_id, iteration=iteration))
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
                            },
                        )
                    )
                except (CodeEditPrivateTestError, CodeEditImageBuildError, CodeEditPatchApplyError) as exc:
                    event_type = str(getattr(exc, "failure_stage", "") or "candidate_build_failed")
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
            selected = selected[: settings.max_candidates]
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
            )
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
            "selected_candidates": [
                {
                    "node_id": candidate.node_id,
                    "iteration": candidate.iteration,
                    "candidate_artifact_hash": candidate.build.candidate_model_manifest.model_artifact_hash,
                    "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                    "candidate_source_diff_hash": candidate.build.source_diff_hash,
                    "build_doc_hash": candidate.build.build_doc.get("build_doc_hash"),
                    "draft": _redacted_draft_doc(candidate.draft),
                }
                for candidate in selected
            ],
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
        "target_files": list(draft.target_files),
        "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
        "redacted_summary": draft.redacted_summary,
        "test_plan": draft.test_plan,
        "rollback_plan": draft.rollback_plan,
        "predicted_delta": draft.predicted_delta,
    }


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
