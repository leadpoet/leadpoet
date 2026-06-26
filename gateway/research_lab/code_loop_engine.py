"""Code-edit image candidate generation loop for hosted Research Lab runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

from gateway.research_lab.code_build import CodeEditBuildError, CodeEditBuildResult, CodeEditCandidateBuilder
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
    parse_code_edit_response,
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
                try:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"),
                            event_doc={"iteration": iteration, "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff})},
                        )
                    )
                    build = self.builder.build(
                        draft=draft,
                        parent_artifact=artifact,
                        run_id=run_id,
                        candidate_index=len(selected),
                    )
                    selected.append(BuiltCodeEditCandidate(draft=draft, build=build, node_id=node_id, iteration=iteration))
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
                except CodeEditBuildError as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_failed"),
                            event_doc={"iteration": iteration, "error": str(exc)[:500]},
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


def _ticket_doc_value(ticket: Mapping[str, Any], key: str) -> Any:
    if key in ticket:
        return ticket.get(key)
    doc = ticket.get("ticket_doc")
    if isinstance(doc, Mapping):
        return doc.get(key)
    return None
