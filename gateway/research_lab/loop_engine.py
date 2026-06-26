"""Iterative hosted auto-research loop for Research Lab candidate generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from research_lab.auto_research_prompt import (
    AutoResearchCandidateDraft,
    build_default_auto_research_messages,
    build_validated_candidate_manifest,
)
from research_lab.canonical import sha256_json
from research_lab.engine_v1 import ComponentRegistry, HypothesisRecord, PatchRecord
from research_lab.eval import CandidatePatchManifest, PrivateModelArtifactManifest


@dataclass(frozen=True)
class OpenRouterCallResult:
    content: str
    provider_usage: dict[str, Any] = field(default_factory=dict)
    cost_microusd: int = 0


LoopEventSink = Callable[["AutoResearchLoopEvent"], Awaitable[None]]
CheckpointSink = Callable[[dict[str, Any]], Awaitable[None]]
PauseChecker = Callable[[], Awaitable[bool]]
OpenRouterCaller = Callable[
    [Sequence[Mapping[str, str]], int, int],
    Awaitable[Union[str, OpenRouterCallResult]],
]


@dataclass(frozen=True)
class AutoResearchLoopSettings:
    min_seconds: int
    max_seconds: int
    min_iterations: int
    max_iterations: int
    draft_timeout_seconds: int
    reflection_timeout_seconds: int
    estimated_iteration_cost_usd: float
    max_candidates: int

    def normalized(self) -> "AutoResearchLoopSettings":
        min_seconds = max(0, int(self.min_seconds))
        max_seconds = max(1, int(self.max_seconds))
        if max_seconds < min_seconds:
            max_seconds = min_seconds
        min_iterations = max(1, int(self.min_iterations))
        max_iterations = max(min_iterations, int(self.max_iterations))
        return AutoResearchLoopSettings(
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            draft_timeout_seconds=max(10, int(self.draft_timeout_seconds)),
            reflection_timeout_seconds=max(10, int(self.reflection_timeout_seconds)),
            estimated_iteration_cost_usd=max(0.01, float(self.estimated_iteration_cost_usd)),
            max_candidates=max(1, int(self.max_candidates)),
        )


@dataclass(frozen=True)
class AutoResearchLoopEvent:
    event_type: str
    loop_status: str
    elapsed_seconds: float
    node_id: str | None = None
    candidate_artifact_hash: str | None = None
    candidate_patch_hash: str | None = None
    provider_usage: list[dict[str, Any]] = field(default_factory=list)
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    event_doc: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoResearchSelectedCandidate:
    draft: AutoResearchCandidateDraft
    patch_manifest: CandidatePatchManifest
    hypothesis: HypothesisRecord
    patch: PatchRecord
    node_id: str
    iteration: int


@dataclass(frozen=True)
class AutoResearchLoopResult:
    selected_candidates: tuple[AutoResearchSelectedCandidate, ...]
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
            "status": self.status if self.status in {"paused", "completed", "failed"} else (
                "completed" if self.selected_candidates else "failed"
            ),
            "total_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_microusd": int(self.actual_openrouter_cost_microusd),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "openrouter_call_count": self.openrouter_call_count,
            "iterations_completed": self.iterations_completed,
            "stop_reason": self.stop_reason,
        }


class AutoResearchLoopEngine:
    """Run iterative private-model patch research before gateway scoring."""

    def __init__(
        self,
        *,
        settings: AutoResearchLoopSettings,
        call_openrouter: OpenRouterCaller,
        event_sink: LoopEventSink,
        checkpoint_sink: CheckpointSink | None = None,
    ):
        self.settings = settings.normalized()
        self.call_openrouter = call_openrouter
        self.event_sink = event_sink
        self.checkpoint_sink = checkpoint_sink

    async def run(
        self,
        *,
        run_id: str,
        ticket: Mapping[str, Any],
        artifact: PrivateModelArtifactManifest,
        component_registry: ComponentRegistry,
        benchmark_public_summary: Mapping[str, Any],
        model_id: str,
        budget_context: Mapping[str, Any],
        requested_loop_count: int,
        miner_brief_ref: str,
        resume_state: Mapping[str, Any] | None = None,
        should_pause: PauseChecker | None = None,
    ) -> AutoResearchLoopResult:
        start = time.monotonic()
        settings = self._settings_for_budget(requested_loop_count, budget_context)
        resume = dict(resume_state or {})
        selected = _selected_candidates_from_checkpoint(
            run_id=run_id,
            checkpoint=resume,
            artifact=artifact,
            component_registry=component_registry,
            miner_brief_ref=miner_brief_ref,
        )
        seen_artifacts: set[str] = {
            candidate.patch_manifest.candidate_artifact_hash
            for candidate in selected
        }
        seen_artifacts.update(str(item) for item in resume.get("seen_artifacts", []) if item)
        reflections: list[dict[str, Any]] = [
            dict(item) for item in resume.get("reflections", []) if isinstance(item, Mapping)
        ]
        openrouter_calls = max(0, int(resume.get("openrouter_call_count") or 0))
        estimated_cost = max(0.0, float(resume.get("estimated_cost_usd") or 0.0))
        actual_cost_microusd = max(0, int(resume.get("actual_openrouter_cost_microusd") or 0))
        provider_usage: list[dict[str, Any]] = [
            dict(item) for item in resume.get("provider_usage", []) if isinstance(item, Mapping)
        ]
        elapsed_offset = max(0.0, float(resume.get("elapsed_seconds") or 0.0))
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
                    "loop_resumed" if resume else "loop_started",
                ),
                event_doc={
                    "run_id": run_id,
                    "requested_loop_count": int(requested_loop_count),
                    "settings": _settings_doc(settings),
                    "budget_context": _safe_budget_doc(budget_context),
                    "scoring_owner": "gateway_qualification_workers",
                    "candidate_queueing": "after_loop_completion_only",
                    "resumed_from_checkpoint": bool(resume),
                    "checkpoint_hash": resume.get("checkpoint_hash"),
                },
            )
        )

        iteration = max(0, int(resume.get("iterations_completed") or 0))
        stop_reason = "max_iterations"
        last_checkpoint: dict[str, Any] | None = None
        elapsed = lambda: elapsed_offset + (time.monotonic() - start)
        while iteration < settings.max_iterations:
            current_elapsed = elapsed()
            if current_elapsed >= settings.max_seconds:
                stop_reason = "max_seconds"
                break
            if (
                iteration >= settings.min_iterations
                and current_elapsed >= settings.min_seconds
                and len(selected) >= settings.max_candidates
            ):
                stop_reason = "candidate_limit_reached_after_minimum_runtime"
                break
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                stop_reason = "compute_budget_exhausted_before_next_draft"
                break
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    selected=selected,
                    seen_artifacts=seen_artifacts,
                    reflections=reflections,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    stage="pause_before_next_iteration",
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                )
                return await self._paused_result(
                    checkpoint=last_checkpoint,
                    selected=selected,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                )

            iteration += 1
            draft_context = {
                **dict(budget_context),
                "loop_iteration": iteration,
                "prior_reflections": reflections[-3:],
                "selected_candidate_count": len(selected),
                "required_output": "typed_patch_candidates_only",
            }
            draft_result = _coerce_call_result(
                await self.call_openrouter(
                    build_default_auto_research_messages(
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
                        component_registry=component_registry.to_dict(),
                        benchmark_public_summary=benchmark_public_summary,
                        budget_context=draft_context,
                        max_candidates=settings.max_candidates,
                    ),
                    settings.draft_timeout_seconds,
                    1800,
                )
            )
            raw = draft_result.content
            openrouter_calls += 1
            estimated_cost += settings.estimated_iteration_cost_usd
            actual_cost_microusd += max(0, int(draft_result.cost_microusd))
            if draft_result.provider_usage:
                provider_usage.append({**draft_result.provider_usage, "loop_iteration": iteration, "call_stage": "draft"})
            attempt_summaries: list[dict[str, Any]] = []
            try:
                from research_lab.auto_research_prompt import parse_auto_research_response

                drafts = parse_auto_research_response(raw, max_candidates=settings.max_candidates)
            except Exception as exc:
                draft_parse_node_id = f"node:draft-parse-failed:{iteration}"
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="patch_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=draft_parse_node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "draft_parse_failed",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "validation_result": "failed",
                            "error": str(exc)[:500],
                            "raw_response_hash": sha256_json({"raw_response": raw}),
                        },
                    )
                )
                attempt_summaries.append(
                    {
                        "node_id": draft_parse_node_id,
                        "status": "dropped",
                        "error": str(exc)[:240],
                    }
                )
                drafts = []
            for draft_index, draft in enumerate(drafts):
                node_id = _node_id(run_id, iteration, draft_index, draft)
                running_ledger = _running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "hypothesis_drafted",
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="hypothesis_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=running_ledger,
                        event_doc={
                            "iteration": iteration,
                            "hypothesis": draft.to_dict()["hypothesis"],
                            "patch_type": draft.patch_type,
                            "target_component_id": draft.target_component_id,
                        },
                    )
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="patch_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "patch_drafted"),
                        event_doc={
                            "iteration": iteration,
                            "patch": draft.to_dict()["patch"],
                        },
                    )
                )
                try:
                    patch_manifest, hypothesis, patch = build_validated_candidate_manifest(
                        draft=draft,
                        artifact_manifest=artifact,
                        component_registry=component_registry,
                        run_id=run_id,
                        sequence=(iteration * 1000) + draft_index,
                        miner_brief_ref=miner_brief_ref,
                    )
                    candidate_patch_hash = patch_manifest.manifest_hash()
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="patch_validation_passed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            candidate_artifact_hash=patch_manifest.candidate_artifact_hash,
                            candidate_patch_hash=candidate_patch_hash,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "patch_validation_passed"),
                            event_doc={
                                "iteration": iteration,
                                "patch_type": patch_manifest.patch_type,
                                "target_component_id": patch_manifest.target_component_id,
                                "validation_result": patch_manifest.validation_result,
                            },
                        )
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="dev_check_passed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            candidate_artifact_hash=patch_manifest.candidate_artifact_hash,
                            candidate_patch_hash=candidate_patch_hash,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "dev_check_passed"),
                            event_doc={
                                "iteration": iteration,
                                "check": "typed_patch_contract_validation",
                                "official_scoring": False,
                                "sealed_icp_used": False,
                            },
                        )
                    )
                    attempt_summaries.append(
                        {
                            "node_id": node_id,
                            "status": "kept",
                            "patch_type": patch_manifest.patch_type,
                            "target_component_id": patch_manifest.target_component_id,
                            "predicted_delta": draft.predicted_delta,
                        }
                    )
                    if patch_manifest.candidate_artifact_hash not in seen_artifacts:
                        selected.append(
                            AutoResearchSelectedCandidate(
                                draft=draft,
                                patch_manifest=patch_manifest,
                                hypothesis=hypothesis,
                                patch=patch,
                                node_id=node_id,
                                iteration=iteration,
                            )
                        )
                        seen_artifacts.add(patch_manifest.candidate_artifact_hash)
                except Exception as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="patch_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "patch_validation_failed"),
                            event_doc={
                                "iteration": iteration,
                                "patch_type": draft.patch_type,
                                "target_component_id": draft.target_component_id,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="dev_check_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "dev_check_failed"),
                            event_doc={
                                "iteration": iteration,
                                "check": "typed_patch_contract_validation",
                                "official_scoring": False,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                    attempt_summaries.append(
                        {
                            "node_id": node_id,
                            "status": "dropped",
                            "patch_type": draft.patch_type,
                            "target_component_id": draft.target_component_id,
                            "error": str(exc)[:240],
                        }
                    )

            reflection_estimate_microusd = _estimated_call_microusd(
                max(0.01, settings.estimated_iteration_cost_usd * 0.25)
            )
            if _would_exceed_budget(actual_cost_microusd, reflection_estimate_microusd, budget_limit_microusd):
                reflections.append(
                    {
                        "worked": "draft call completed",
                        "failed": "reflection skipped",
                        "why": "compute budget would be exceeded by next estimated call",
                        "next_question": "stop or continue with additional budget",
                        "decision": "stop",
                    }
                )
                stop_reason = "compute_budget_exhausted_before_reflection"
                break
            reflection, reflection_usage = await self._reflect(
                run_id=run_id,
                iteration=iteration,
                model_id=model_id,
                attempt_summaries=attempt_summaries,
                selected_count=len(selected),
                timeout_seconds=settings.reflection_timeout_seconds,
            )
            openrouter_calls += 1
            estimated_cost += max(0.01, settings.estimated_iteration_cost_usd * 0.25)
            actual_cost_microusd += max(0, int(reflection_usage.cost_microusd))
            if reflection_usage.provider_usage:
                provider_usage.append({**reflection_usage.provider_usage, "loop_iteration": iteration, "call_stage": "reflection"})
            reflections.append(reflection)
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="reflection_recorded",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "reflection_recorded"),
                    event_doc={
                        "iteration": iteration,
                        "reflection": reflection,
                    },
                )
            )

            selected = _rank_candidates(selected)[: settings.max_candidates]
            last_checkpoint = await self._emit_checkpoint(
                run_id=run_id,
                settings=settings,
                selected=selected,
                seen_artifacts=seen_artifacts,
                reflections=reflections,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                stage="iteration_completed",
                artifact=artifact,
                model_id=model_id,
                budget_context=budget_context,
            )
            if should_pause and await should_pause():
                return await self._paused_result(
                    checkpoint=last_checkpoint,
                    selected=selected,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                )
            if not selected and iteration >= settings.min_iterations and elapsed() >= settings.min_seconds:
                stop_reason = "no_valid_candidates_after_minimum_runtime"
                break

        if not selected:
            stop_reason = "no_valid_candidates"
        elif iteration >= settings.max_iterations:
            stop_reason = "max_iterations"

        if should_pause and await should_pause():
            last_checkpoint = await self._emit_checkpoint(
                run_id=run_id,
                settings=settings,
                selected=selected,
                seen_artifacts=seen_artifacts,
                reflections=reflections,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                stage="pause_before_finalization",
                artifact=artifact,
                model_id=model_id,
                budget_context=budget_context,
            )
            return await self._paused_result(
                checkpoint=last_checkpoint,
                selected=selected,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                openrouter_calls=openrouter_calls,
                provider_usage=provider_usage,
            )

        if selected:
            remaining_minimum = settings.min_seconds - elapsed()
            remaining_maximum = settings.max_seconds - elapsed()
            if remaining_minimum > 0 and remaining_maximum > 0:
                await asyncio.sleep(min(remaining_minimum, remaining_maximum))
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    selected=selected,
                    seen_artifacts=seen_artifacts,
                    reflections=reflections,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    stage="pause_after_minimum_runtime",
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                )
                return await self._paused_result(
                    checkpoint=last_checkpoint,
                    selected=selected,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                )

        ranked = tuple(_rank_candidates(selected)[: settings.max_candidates])
        for index, candidate in enumerate(ranked):
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="candidate_selected",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=candidate.node_id,
                    candidate_artifact_hash=candidate.patch_manifest.candidate_artifact_hash,
                    candidate_patch_hash=candidate.patch_manifest.manifest_hash(),
                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_selected"),
                    event_doc={
                        "candidate_index": index,
                        "iteration": candidate.iteration,
                        "patch_type": candidate.patch_manifest.patch_type,
                        "target_component_id": candidate.patch_manifest.target_component_id,
                        "redacted_summary": candidate.patch_manifest.redacted_summary,
                    },
                )
            )

        result = AutoResearchLoopResult(
            selected_candidates=ranked,
            iterations_completed=iteration,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed(),
            estimated_cost_usd=estimated_cost,
            actual_openrouter_cost_usd=round(actual_cost_microusd / 1_000_000, 6),
            actual_openrouter_cost_microusd=actual_cost_microusd,
            openrouter_call_count=openrouter_calls,
            provider_usage=tuple(provider_usage),
            status="completed" if ranked else "failed",
            checkpoint_doc=last_checkpoint,
        )
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_completed" if ranked else "loop_failed",
                loop_status="completed" if ranked else "failed",
                elapsed_seconds=result.elapsed_seconds,
                provider_usage=list(result.provider_usage),
                cost_ledger=result.cost_ledger(),
                event_doc={
                    "iterations_completed": result.iterations_completed,
                    "selected_candidate_count": len(ranked),
                    "stop_reason": result.stop_reason,
                    "validator_queue_visible_after_this_event": bool(ranked),
                },
            )
        )
        return result

    async def _emit_checkpoint(
        self,
        *,
        run_id: str,
        settings: AutoResearchLoopSettings,
        selected: Sequence[AutoResearchSelectedCandidate],
        seen_artifacts: set[str],
        reflections: Sequence[Mapping[str, Any]],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: Sequence[Mapping[str, Any]],
        iterations_completed: int,
        elapsed_seconds: float,
        stage: str,
        artifact: PrivateModelArtifactManifest,
        model_id: str,
        budget_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "stage": stage,
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
                    "draft": candidate.draft.to_dict(),
                    "candidate_artifact_hash": candidate.patch_manifest.candidate_artifact_hash,
                    "candidate_patch_hash": candidate.patch_manifest.manifest_hash(),
                }
                for candidate in _rank_candidates(selected)
            ],
            "seen_artifacts": sorted(str(item) for item in seen_artifacts if item),
            "reflections": [dict(item) for item in reflections[-10:] if isinstance(item, Mapping)],
            "openrouter_call_count": int(openrouter_calls),
            "estimated_cost_usd": round(float(estimated_cost), 6),
            "actual_openrouter_cost_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
            "actual_openrouter_cost_microusd": int(actual_cost_microusd),
            "provider_usage": [dict(item) for item in provider_usage if isinstance(item, Mapping)],
        }
        checkpoint = {**payload, "checkpoint_hash": sha256_json(payload)}
        if self.checkpoint_sink:
            await self.checkpoint_sink(checkpoint)
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="checkpoint_saved",
                loop_status="running",
                elapsed_seconds=elapsed_seconds,
                provider_usage=[dict(item) for item in provider_usage if isinstance(item, Mapping)],
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "checkpoint_saved",
                ),
                event_doc={"checkpoint": checkpoint},
            )
        )
        return checkpoint

    async def _paused_result(
        self,
        *,
        checkpoint: dict[str, Any],
        selected: Sequence[AutoResearchSelectedCandidate],
        iterations_completed: int,
        elapsed_seconds: float,
        estimated_cost: float,
        actual_cost_microusd: int,
        openrouter_calls: int,
        provider_usage: Sequence[Mapping[str, Any]],
    ) -> AutoResearchLoopResult:
        ranked = tuple(_rank_candidates(selected)[: self.settings.max_candidates])
        result = AutoResearchLoopResult(
            selected_candidates=ranked,
            iterations_completed=int(iterations_completed),
            stop_reason="maintenance_pause_requested",
            elapsed_seconds=round(float(elapsed_seconds), 3),
            estimated_cost_usd=round(float(estimated_cost), 6),
            actual_openrouter_cost_usd=round(int(actual_cost_microusd) / 1_000_000, 6),
            actual_openrouter_cost_microusd=int(actual_cost_microusd),
            openrouter_call_count=int(openrouter_calls),
            provider_usage=tuple(dict(item) for item in provider_usage if isinstance(item, Mapping)),
            status="paused",
            checkpoint_doc=checkpoint,
        )
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_paused",
                loop_status="paused",
                elapsed_seconds=result.elapsed_seconds,
                provider_usage=list(result.provider_usage),
                cost_ledger=result.cost_ledger(),
                event_doc={
                    "checkpoint_hash": checkpoint.get("checkpoint_hash"),
                    "iterations_completed": result.iterations_completed,
                    "selected_candidate_count": len(result.selected_candidates),
                    "stop_reason": result.stop_reason,
                    "validator_queue_visible_after_this_event": False,
                },
            )
        )
        return result

    def _settings_for_budget(self, requested_loop_count: int, budget_context: Mapping[str, Any]) -> AutoResearchLoopSettings:
        settings = self.settings
        budget = float(budget_context.get("requested_compute_budget_usd") or 0.0)
        estimated_iteration_cost = max(0.01, float(settings.estimated_iteration_cost_usd))
        budget_iterations = max(1, int(budget // estimated_iteration_cost)) if budget > 0 else settings.max_iterations
        requested_iterations = max(1, int(requested_loop_count or 1))
        max_iterations = max(1, min(settings.max_iterations, requested_iterations, budget_iterations))
        return AutoResearchLoopSettings(
            min_seconds=settings.min_seconds,
            max_seconds=settings.max_seconds,
            min_iterations=min(settings.min_iterations, max_iterations),
            max_iterations=max_iterations,
            draft_timeout_seconds=settings.draft_timeout_seconds,
            reflection_timeout_seconds=settings.reflection_timeout_seconds,
            estimated_iteration_cost_usd=settings.estimated_iteration_cost_usd,
            max_candidates=settings.max_candidates,
        ).normalized()

    async def _reflect(
        self,
        *,
        run_id: str,
        iteration: int,
        model_id: str,
        attempt_summaries: Sequence[Mapping[str, Any]],
        selected_count: int,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], OpenRouterCallResult]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Leadpoet Research Lab reflection engine. Return strict JSON only. "
                    "Do not include private code, raw keys, judge prompts, hidden ICP plaintext, or secrets."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "reflect_on_auto_research_iteration",
                        "run_id": run_id,
                        "iteration": iteration,
                        "model_id": model_id,
                        "attempt_summaries": list(attempt_summaries),
                        "selected_count": selected_count,
                        "expected_shape": {
                            "worked": "short text",
                            "failed": "short text",
                            "why": "short text",
                            "next_question": "short text",
                            "decision": "continue|stop",
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            call_result = _coerce_call_result(await self.call_openrouter(messages, timeout_seconds, 700))
            decoded = json.loads(_extract_json_object(call_result.content))
            if not isinstance(decoded, Mapping):
                raise ValueError("reflection response must be an object")
            return (
                {
                    "worked": str(decoded.get("worked") or "")[:500],
                    "failed": str(decoded.get("failed") or "")[:500],
                    "why": str(decoded.get("why") or "")[:500],
                    "next_question": str(decoded.get("next_question") or "")[:500],
                    "decision": str(decoded.get("decision") or "continue")[:40],
                },
                call_result,
            )
        except Exception as exc:
            return (
                {
                    "worked": "valid candidates were kept" if selected_count else "no valid candidates yet",
                    "failed": "reflection model failed",
                    "why": str(exc)[:240],
                    "next_question": "try a narrower typed patch",
                    "decision": "continue",
                },
                OpenRouterCallResult(content="", provider_usage={}, cost_microusd=0),
            )


def _rank_candidates(candidates: Sequence[AutoResearchSelectedCandidate]) -> list[AutoResearchSelectedCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.draft.predicted_delta),
            candidate.patch_manifest.patch_type,
            candidate.patch_manifest.target_component_id,
            candidate.patch_manifest.candidate_artifact_hash,
        ),
        reverse=True,
    )


def _selected_candidates_from_checkpoint(
    *,
    run_id: str,
    checkpoint: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
    component_registry: ComponentRegistry,
    miner_brief_ref: str,
) -> list[AutoResearchSelectedCandidate]:
    raw_candidates = checkpoint.get("selected_candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return []
    rebuilt: list[AutoResearchSelectedCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            continue
        try:
            draft = _draft_from_checkpoint_item(raw_candidate.get("draft"))
            iteration = max(1, int(raw_candidate.get("iteration") or 1))
            patch_manifest, hypothesis, patch = build_validated_candidate_manifest(
                draft=draft,
                artifact_manifest=artifact,
                component_registry=component_registry,
                run_id=run_id,
                sequence=(iteration * 1000) + index,
                miner_brief_ref=miner_brief_ref,
            )
        except Exception:
            continue
        rebuilt.append(
            AutoResearchSelectedCandidate(
                draft=draft,
                patch_manifest=patch_manifest,
                hypothesis=hypothesis,
                patch=patch,
                node_id=str(raw_candidate.get("node_id") or _node_id(run_id, iteration, index, draft)),
                iteration=iteration,
            )
        )
    return _rank_candidates(rebuilt)


def _draft_from_checkpoint_item(value: Any) -> AutoResearchCandidateDraft:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint selected candidate is missing draft")
    hypothesis = value.get("hypothesis")
    patch = value.get("patch")
    if not isinstance(hypothesis, Mapping) or not isinstance(patch, Mapping):
        raise ValueError("checkpoint draft requires hypothesis and patch")
    patch_doc = patch.get("patch_doc")
    if not isinstance(patch_doc, Mapping):
        raise ValueError("checkpoint draft patch_doc must be an object")
    return AutoResearchCandidateDraft(
        failure_mode=str(hypothesis.get("failure_mode") or "")[:600],
        mechanism=str(hypothesis.get("mechanism") or "")[:900],
        expected_improvement=str(hypothesis.get("expected_improvement") or "")[:900],
        risk=str(hypothesis.get("risk") or "")[:600],
        focus_alignment=str(hypothesis.get("focus_alignment") or "")[:500],
        predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
        falsifier=str(hypothesis.get("falsifier") or "proxy_score"),
        patch_type=str(patch.get("patch_type") or ""),
        target_component_id=str(patch.get("target_component_id") or ""),
        patch_doc=dict(patch_doc),
        redacted_summary=str(patch.get("redacted_summary") or "")[:900],
    )


def _node_id(run_id: str, iteration: int, draft_index: int, draft: AutoResearchCandidateDraft) -> str:
    digest = sha256_json(
        {
            "run_id": run_id,
            "iteration": iteration,
            "draft_index": draft_index,
            "draft": draft.to_dict(),
        }
    ).split(":", 1)[1]
    return f"node:{digest[:32]}"


def _settings_doc(settings: AutoResearchLoopSettings) -> dict[str, Any]:
    return {
        "min_seconds": settings.min_seconds,
        "max_seconds": settings.max_seconds,
        "min_iterations": settings.min_iterations,
        "max_iterations": settings.max_iterations,
        "draft_timeout_seconds": settings.draft_timeout_seconds,
        "reflection_timeout_seconds": settings.reflection_timeout_seconds,
        "estimated_iteration_cost_usd": settings.estimated_iteration_cost_usd,
        "max_candidates": settings.max_candidates,
    }


def _running_cost_ledger(
    openrouter_call_count: int,
    estimated_cost_usd: float,
    actual_cost_microusd: int,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "stage": stage,
        "total_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
        "actual_openrouter_cost_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
        "actual_openrouter_cost_microusd": int(actual_cost_microusd),
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
        "openrouter_call_count": int(openrouter_call_count),
        "official_scoring": False,
    }


def _safe_budget_doc(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "research_model_tier",
        "requested_compute_budget_usd",
        "max_compute_budget_usd",
        "payment_kind",
        "budget_policy_version",
        "additional_compute_budget_usd",
        "continue_from_run_id",
        "continuation_context",
        "topup_reason",
    }
    return {key: value[key] for key in allowed if key in value}


def _budget_limit_microusd(budget_context: Mapping[str, Any]) -> int:
    try:
        budget_usd = float(budget_context.get("requested_compute_budget_usd") or 0.0)
    except (TypeError, ValueError):
        budget_usd = 0.0
    return max(0, int(round(budget_usd * 1_000_000)))


def _estimated_call_microusd(estimated_cost_usd: float) -> int:
    try:
        estimate = float(estimated_cost_usd)
    except (TypeError, ValueError):
        estimate = 0.0
    return max(1, int(round(max(0.0, estimate) * 1_000_000)))


def _would_exceed_budget(actual_cost_microusd: int, estimated_next_call_microusd: int, budget_limit_microusd: int) -> bool:
    return budget_limit_microusd > 0 and max(0, actual_cost_microusd) + max(0, estimated_next_call_microusd) > budget_limit_microusd


def _coerce_call_result(value: str | OpenRouterCallResult) -> OpenRouterCallResult:
    if isinstance(value, OpenRouterCallResult):
        return value
    return OpenRouterCallResult(content=str(value or ""), provider_usage={}, cost_microusd=0)


def _ticket_doc_value(ticket: Mapping[str, Any], key: str) -> Any:
    ticket_doc = ticket.get("ticket_doc")
    if isinstance(ticket_doc, Mapping):
        return ticket_doc.get(key)
    return None


def _extract_json_object(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response did not contain a JSON object")
    return text[start : end + 1]
