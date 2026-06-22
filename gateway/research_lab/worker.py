"""Hosted Research Lab worker for paid private-model auto-research runs."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.key_vault import OpenRouterKeyVaultError, decrypt_openrouter_key
from gateway.research_lab.models import ResearchLabReceiptCreateRequest, ResearchLabScoreBundleCreateRequest
from gateway.research_lab.store import (
    canonical_hash,
    create_queue_event,
    create_receipt,
    create_receipt_event,
    create_score_bundle,
    create_ticket_event,
    select_many,
    select_one,
)
from research_lab.auto_research_prompt import (
    build_default_auto_research_messages,
    build_validated_candidate_manifest,
    coerce_component_registry,
    parse_auto_research_response,
)
from research_lab.canonical import sha256_json
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    load_private_artifact_manifest,
    sign_digest_with_kms,
    validate_private_model_artifact_manifest,
    validate_sealed_benchmark_set,
)
from research_lab.eval.benchmark import compute_public_benchmark_hash


logger = logging.getLogger(__name__)


class HostedResearchLabWorkerError(RuntimeError):
    """Raised when a hosted Research Lab run cannot complete safely."""


class HostedResearchLabClaimLost(HostedResearchLabWorkerError):
    """Raised when another worker safely claimed the queued run first."""


@dataclass(frozen=True)
class HostedWorkerOutcome:
    processed: bool
    dry_run: bool
    run_id: str | None = None
    ticket_id: str | None = None
    status: str = "idle"
    receipt_id: str | None = None
    score_bundle_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "dry_run": self.dry_run,
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "status": self.status,
            "receipt_id": self.receipt_id,
            "score_bundle_ids": list(self.score_bundle_ids),
            "error": self.error,
        }


@dataclass
class HostedRunContext:
    queue_row: Mapping[str, Any]
    ticket: Mapping[str, Any]
    payment: Mapping[str, Any] | None
    ticket_events: tuple[Mapping[str, Any], ...] = ()
    queue_events: tuple[Mapping[str, Any], ...] = ()
    receipt_id: str | None = None
    provider_env: dict[str, str] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.queue_row["run_id"])

    @property
    def ticket_id(self) -> str:
        return str(self.queue_row["ticket_id"])


class OpenRouterKeyResolver:
    """Resolve miner key refs to process-local env values without persisting raw keys."""

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config

    async def resolve(self, key_ref: str, *, miner_hotkey: str) -> dict[str, str]:
        value = await self._resolve_key_value(key_ref, miner_hotkey=miner_hotkey)
        return {
            "OPENROUTER_API_KEY": value,
            "QUALIFICATION_OPENROUTER_API_KEY": value,
            "OPENROUTER_KEY": value,
        }

    async def _resolve_key_value(self, key_ref: str, *, miner_hotkey: str) -> str:
        if str(key_ref).startswith("encrypted_ref:openrouter:"):
            row = await select_one(
                "research_lab_openrouter_key_refs",
                filters=(("key_ref", key_ref), ("miner_hotkey", miner_hotkey)),
            )
            if not row:
                raise HostedResearchLabWorkerError("encrypted OpenRouter key ref was not found for miner")
            try:
                return await asyncio.to_thread(
                    decrypt_openrouter_key,
                    ciphertext_b64=str(row["encrypted_key_ciphertext"]),
                    miner_hotkey=miner_hotkey,
                    key_ref=key_ref,
                )
            except OpenRouterKeyVaultError as exc:
                raise HostedResearchLabWorkerError(str(exc)) from exc

        env_name = self._env_name_for_ref(key_ref)
        if not env_name:
            raise HostedResearchLabWorkerError("no OpenRouter key env var configured for miner key ref")
        value = os.getenv(env_name)
        if not value:
            raise HostedResearchLabWorkerError(f"configured OpenRouter key env var is empty: {env_name}")
        return value

    def _env_name_for_ref(self, key_ref: str) -> str:
        if self.config.miner_openrouter_key_ref_env_map_json:
            try:
                mapping = json.loads(self.config.miner_openrouter_key_ref_env_map_json)
            except json.JSONDecodeError as exc:
                raise HostedResearchLabWorkerError("invalid OpenRouter key-ref env map JSON") from exc
            if not isinstance(mapping, Mapping):
                raise HostedResearchLabWorkerError("OpenRouter key-ref env map must be an object")
            mapped = mapping.get(str(key_ref))
            if mapped:
                return str(mapped)
        return self.config.miner_openrouter_key_env_var


class ResearchLabHostedWorker:
    """Poll and execute one hosted Research Lab queue run at a time."""

    def __init__(self, config: ResearchLabGatewayConfig | None = None, *, worker_ref: str | None = None):
        self.config = config or ResearchLabGatewayConfig.from_env()
        self.worker_ref = (
            worker_ref
            or self.config.hosted_worker_id
            or f"research-lab-hosted-worker:{os.uname().nodename}:{os.getpid()}"
        )
        self.key_resolver = OpenRouterKeyResolver(self.config)

    async def run_forever(self) -> None:
        processed = 0
        while True:
            outcome = await self.run_once()
            logger.info("Research Lab hosted worker outcome: %s", outcome.to_dict())
            if outcome.processed:
                processed += 1
            if self.config.hosted_worker_max_runs and processed >= self.config.hosted_worker_max_runs:
                return
            await asyncio.sleep(max(1, self.config.hosted_worker_poll_seconds))

    async def run_once(self) -> HostedWorkerOutcome:
        self._require_enabled()
        queued = await self._next_queued_run()
        if not queued:
            return HostedWorkerOutcome(processed=False, dry_run=self.config.hosted_worker_dry_run)
        run_id = str(queued["run_id"])
        ticket_id = str(queued["ticket_id"])
        if self.config.hosted_worker_dry_run:
            return HostedWorkerOutcome(
                processed=False,
                dry_run=True,
                run_id=run_id,
                ticket_id=ticket_id,
                status="dry_run_queued_run_found",
            )
        try:
            self._require_worker_proxy_for_execution()
        except HostedResearchLabWorkerError as exc:
            logger.error(
                "Research Lab worker proxy is required before claiming run: run_id=%s ticket_id=%s worker_ref=%s",
                run_id,
                ticket_id,
                self.worker_ref,
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                run_id=run_id,
                ticket_id=ticket_id,
                status="worker_proxy_required",
                error=str(exc)[:500],
            )
        context = await self._load_run_context(queued)
        try:
            return await self._process_run(context)
        except HostedResearchLabClaimLost as exc:
            logger.info(
                "Research Lab queued run was claimed elsewhere: run_id=%s ticket_id=%s worker_ref=%s",
                run_id,
                ticket_id,
                self.worker_ref,
            )
            return HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                run_id=run_id,
                ticket_id=ticket_id,
                status="claim_lost",
                error=str(exc)[:500],
            )
        except Exception as exc:
            logger.exception("Research Lab hosted run failed: run_id=%s ticket_id=%s", run_id, ticket_id)
            return await self._mark_failed(context, str(exc))

    def _require_enabled(self) -> None:
        if not self.config.hosted_worker_enabled:
            raise HostedResearchLabWorkerError("Research Lab hosted worker is disabled")
        if not self.config.production_writes_enabled:
            raise HostedResearchLabWorkerError("Research Lab production writes are disabled")
        if not self.config.hosted_runs_enabled:
            raise HostedResearchLabWorkerError("Research Lab hosted runs are disabled")
        if not self.config.receipts_enabled:
            raise HostedResearchLabWorkerError("Research Lab receipt writes are disabled")
        if not self.config.evaluation_bundles_enabled:
            raise HostedResearchLabWorkerError("Research Lab evaluation bundle writes are disabled")
        if not self.config.private_model_manifest_uri:
            raise HostedResearchLabWorkerError("private model manifest URI is not configured")
        if not self.config.private_benchmark_path:
            raise HostedResearchLabWorkerError("private benchmark path is not configured")
        if not self.config.approved_auto_research_models():
            raise HostedResearchLabWorkerError("auto-research OpenRouter model is not configured")

    def _require_worker_proxy_for_execution(self) -> None:
        if not self.config.hosted_worker_require_proxy:
            return
        if not _worker_proxy_url(self.config):
            raise HostedResearchLabWorkerError(
                "worker proxy is required for hosted auto-research execution; "
                "set RESEARCH_LAB_HOSTED_WORKER_PROXY or per-worker RESEARCH_LAB_WORKER_PROXY_N"
            )

    async def _next_queued_run(self) -> Mapping[str, Any] | None:
        rows = await select_many(
            "research_loop_run_queue_current",
            filters=(("current_queue_status", "queued"),),
            order_by=(("queue_priority", False), ("current_status_at", False)),
            limit=self.config.hosted_worker_queue_fetch_limit,
        )
        return self._select_preferred_queued_row(rows)

    def _select_preferred_queued_row(self, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        if not rows:
            return None
        total_workers = max(1, int(self.config.hosted_worker_total_workers or 1))
        worker_index = int(self.config.hosted_worker_index or 0) % total_workers
        if total_workers <= 1:
            return rows[0]
        for row in rows:
            if _row_partition(row, total_workers) == worker_index:
                return row
        # Avoid starvation if this worker's preferred shard is temporarily empty.
        # Claim conflicts are handled as no-ops, not failures.
        return rows[0]

    async def _load_run_context(self, queue_row: Mapping[str, Any]) -> HostedRunContext:
        ticket = await select_one(
            "research_loop_ticket_current",
            filters=(("ticket_id", str(queue_row["ticket_id"])),),
        )
        if not ticket:
            raise HostedResearchLabWorkerError("queued Research Lab run is missing ticket")
        ticket_events = await select_many(
            "research_loop_ticket_events",
            filters=(("ticket_id", str(queue_row["ticket_id"])),),
            order_by=(("seq", True),),
            limit=20,
        )
        queue_events = await select_many(
            "research_loop_run_queue_events",
            filters=(("run_id", str(queue_row["run_id"])),),
            order_by=(("seq", True),),
            limit=20,
        )
        payment = None
        payment_id = _payment_id_from_queue_events(queue_events)
        if payment_id:
            payment = await select_one("research_loop_start_payments", filters=(("payment_id", payment_id),))
        if not payment:
            payment_rows = await select_many(
                "research_loop_start_payments",
                filters=(("ticket_id", str(queue_row["ticket_id"])),),
                order_by=(("verified_at", True),),
                limit=1,
            )
            payment = payment_rows[0] if payment_rows else None
        return HostedRunContext(
            queue_row=queue_row,
            ticket=ticket,
            payment=payment,
            ticket_events=tuple(ticket_events),
            queue_events=tuple(queue_events),
        )

    async def _process_run(self, context: HostedRunContext) -> HostedWorkerOutcome:
        await self._append_started_events(context)
        resolved_openrouter_env = await self.key_resolver.resolve(
            _miner_openrouter_key_ref(context),
            miner_hotkey=str(context.ticket["miner_hotkey"]),
        )
        provider_env = {
            **resolved_openrouter_env,
            **_worker_proxy_env(self.config),
        }
        context.provider_env = provider_env
        receipt, _event = await create_receipt(self._queued_receipt_request(context))
        context.receipt_id = str(receipt["receipt_id"])
        budget_context = self._run_budget_context(context)
        _tier, model_id, model_doc = self.config.resolve_auto_research_model(
            str(budget_context.get("research_model_tier") or "")
        )
        max_candidates = self._max_candidates_for_run(budget_context, model_doc)

        artifact = self._load_private_artifact()
        benchmark, benchmark_items = self._load_private_benchmark()
        runner = DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                extra_env=provider_env,
                timeout_seconds=900,
            )
        )
        with _temporary_env(provider_env):
            metadata = runner.metadata()
            registry = coerce_component_registry(metadata)
            benchmark_summary = {
                "benchmark_id": benchmark.benchmark_id,
                "icp_set_hash": benchmark.icp_set_hash,
                "split_ref": benchmark.split_ref,
                "item_count": len(benchmark_items),
                "scoring_version": benchmark.scoring_version,
            }
            drafts = await self._generate_candidate_drafts(
                context=context,
                artifact=artifact,
                component_registry=registry.to_dict(),
                benchmark_public_summary=benchmark_summary,
                model_id=model_id,
                budget_context=budget_context,
                max_candidates=max_candidates,
            )
            if not drafts:
                raise HostedResearchLabWorkerError("auto-research generated no valid candidate drafts")

        bundle_ids: list[str] = []
        candidate_summaries: list[dict[str, Any]] = []
        for index, draft in enumerate(drafts):
            patch_manifest, hypothesis, patch = build_validated_candidate_manifest(
                draft=draft,
                artifact_manifest=artifact,
                component_registry=registry,
                run_id=context.run_id,
                sequence=index,
                miner_brief_ref=str(context.ticket.get("brief_sanitized_ref") or ""),
            )
            run_context = self._evaluation_run_context(
                context=context,
                candidate_index=index,
                patch_hash=patch_manifest.manifest_hash(),
            )
            with _temporary_env(provider_env):
                score_bundle = await evaluate_private_model_pair(
                    artifact_manifest=artifact,
                    benchmark=benchmark,
                    patch_manifest=patch_manifest,
                    benchmark_items=benchmark_items,
                    base_runner=runner,
                    candidate_runner=runner,
                    run_context=run_context,
                    policy=self._evaluation_policy(len(benchmark_items)),
                )
            signature_ref = await asyncio.to_thread(
                sign_digest_with_kms,
                key_id=self.config.score_bundle_kms_key_id,
                digest_hash=score_bundle["score_bundle_hash"],
                signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
            )
            score_bundle = {**score_bundle, "signature_ref": signature_ref}
            request = ResearchLabScoreBundleCreateRequest(
                receipt_id=context.receipt_id,
                bundle_status="scored",
                score_bundle=score_bundle,
            )
            bundle_row, _bundle_event = await create_score_bundle(request)
            bundle_ids.append(str(bundle_row["score_bundle_id"]))
            candidate_summaries.append(
                {
                    "candidate_index": index,
                    "score_bundle_id": str(bundle_row["score_bundle_id"]),
                    "score_bundle_hash": str(bundle_row["score_bundle_hash"]),
                    "candidate_artifact_hash": patch_manifest.candidate_artifact_hash,
                    "candidate_patch_hash": patch_manifest.manifest_hash(),
                    "mean_delta": score_bundle.get("aggregates", {}).get("mean_delta"),
                    "delta_lcb": score_bundle.get("aggregates", {}).get("delta_lcb"),
                    "eligible_for_probation": score_bundle.get("improvement_gate", {}).get("eligible_for_probation"),
                    "hypothesis": hypothesis.to_dict(),
                    "patch": patch.to_dict(),
                }
            )

        best = _best_candidate_summary(candidate_summaries)
        await create_receipt_event(
            receipt_id=context.receipt_id,
            ticket_id=context.ticket_id,
            event_type="completed",
            receipt_status="completed",
            event_doc={
                "score_bundle_ids": bundle_ids,
                "best_candidate": best,
                "candidate_count": len(candidate_summaries),
                "budget_context": _redacted_budget_context(budget_context),
                "followup_status": _followup_status(best, self.config.topup_promising_delta_threshold),
            },
        )
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="completed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="hosted_research_lab_run_completed",
            event_doc={
                "receipt_id": context.receipt_id,
                "score_bundle_ids": bundle_ids,
                "followup_status": _followup_status(best, self.config.topup_promising_delta_threshold),
            },
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="completed",
            actor_hotkey=None,
            reason="hosted_research_lab_run_completed",
            event_doc={
                "run_id": context.run_id,
                "receipt_id": context.receipt_id,
                "score_bundle_ids": bundle_ids,
                "followup_status": _followup_status(best, self.config.topup_promising_delta_threshold),
            },
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="completed",
            receipt_id=context.receipt_id,
            score_bundle_ids=tuple(bundle_ids),
        )

    async def _append_started_events(self, context: HostedRunContext) -> None:
        current = await select_one(
            "research_loop_run_queue_current",
            filters=(("run_id", context.run_id),),
        )
        if not current or current.get("current_queue_status") != "queued":
            raise HostedResearchLabClaimLost("queued Research Lab run is no longer queued")
        try:
            await create_queue_event(
                run_id=context.run_id,
                ticket_id=context.ticket_id,
                event_type="started",
                queue_priority=int(context.queue_row.get("queue_priority") or 0),
                worker_ref=self.worker_ref,
                reason="hosted_worker_started",
                event_doc={"worker_ref": self.worker_ref},
            )
        except Exception as exc:
            if _is_claim_race_error(exc):
                raise HostedResearchLabClaimLost("queued Research Lab run was claimed by another worker") from exc
            raise
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="running",
            actor_hotkey=None,
            reason="hosted_worker_started",
            event_doc={"run_id": context.run_id, "worker_ref": self.worker_ref},
        )

    async def _mark_failed(self, context: HostedRunContext, error: str) -> HostedWorkerOutcome:
        event_doc = {"run_id": context.run_id, "worker_ref": self.worker_ref, "error": error[:500]}
        receipt_id = context.receipt_id
        if receipt_id:
            await create_receipt_event(
                receipt_id=receipt_id,
                ticket_id=context.ticket_id,
                event_type="failed",
                receipt_status="failed",
                event_doc=event_doc,
            )
        else:
            receipt, _event = await create_receipt(self._failed_receipt_request(context, error))
            receipt_id = str(receipt["receipt_id"])
        await create_queue_event(
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            event_type="failed",
            queue_priority=int(context.queue_row.get("queue_priority") or 0),
            worker_ref=self.worker_ref,
            reason="hosted_research_lab_run_failed",
            event_doc={**event_doc, "receipt_id": receipt_id},
        )
        await create_ticket_event(
            ticket_id=context.ticket_id,
            event_type="cancelled",
            actor_hotkey=None,
            reason="hosted_research_lab_run_failed",
            event_doc={**event_doc, "receipt_id": receipt_id},
        )
        return HostedWorkerOutcome(
            processed=True,
            dry_run=False,
            run_id=context.run_id,
            ticket_id=context.ticket_id,
            status="failed",
            receipt_id=receipt_id,
            error=error[:500],
        )

    def _queued_receipt_request(self, context: HostedRunContext) -> ResearchLabReceiptCreateRequest:
        budget_context = self._run_budget_context(context)
        return ResearchLabReceiptCreateRequest(
            internal_run_ref=f"research_lab_hosted_worker:{context.run_id}",
            ticket_id=context.ticket_id,
            run_id=context.run_id,
            loop_start_payment_id=context.payment.get("payment_id") if context.payment else None,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
            island=str(context.ticket["island"]),
            receipt_status="queued",
            loop_count=int(context.ticket.get("requested_loop_count") or 1),
            miner_openrouter_key_ref=_miner_openrouter_key_ref(context),
            provider_usage=self._provider_usage(context),
            cost_ledger={
                "schema_version": "1.0",
                "status": "queued",
                "total_usd": 0.0,
                "budget_context": _redacted_budget_context(budget_context),
            },
            receipt_doc={
                "schema_version": "1.0",
                "run_id": context.run_id,
                "worker_ref": self.worker_ref,
                "private_model_manifest_uri": self.config.private_model_manifest_uri,
                "budget_context": _redacted_budget_context(budget_context),
            },
        )

    def _failed_receipt_request(self, context: HostedRunContext, error: str) -> ResearchLabReceiptCreateRequest:
        budget_context = self._run_budget_context(context)
        return ResearchLabReceiptCreateRequest(
            internal_run_ref=f"research_lab_hosted_worker:{context.run_id}",
            ticket_id=context.ticket_id,
            run_id=context.run_id,
            loop_start_payment_id=context.payment.get("payment_id") if context.payment else None,
            miner_hotkey=str(context.ticket["miner_hotkey"]),
            island=str(context.ticket["island"]),
            receipt_status="failed",
            loop_count=int(context.ticket.get("requested_loop_count") or 1),
            miner_openrouter_key_ref=_miner_openrouter_key_ref(context),
            provider_usage=self._provider_usage(context),
            cost_ledger={
                "schema_version": "1.0",
                "status": "failed",
                "total_usd": 0.0,
                "budget_context": _redacted_budget_context(budget_context),
            },
            receipt_doc={
                "schema_version": "1.0",
                "run_id": context.run_id,
                "worker_ref": self.worker_ref,
                "failure_reason": error[:500],
                "budget_context": _redacted_budget_context(budget_context),
            },
        )

    def _provider_usage(self, context: HostedRunContext) -> list[dict[str, Any]]:
        budget_context = self._run_budget_context(context)
        return [
            {
                "provider": "openrouter",
                "key_source": "miner_key_ref",
                "key_ref": _miner_openrouter_key_ref(context),
                "research_model_tier": budget_context.get("research_model_tier"),
            },
            {"provider": "exa", "key_source": "leadpoet_server_side"},
            {"provider": "scrapingdog", "key_source": "leadpoet_server_side"},
        ]

    def _load_private_artifact(self) -> PrivateModelArtifactManifest:
        payload = load_private_artifact_manifest(self.config.private_model_manifest_uri)
        artifact = PrivateModelArtifactManifest.from_mapping(payload)
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise HostedResearchLabWorkerError("private artifact manifest failed validation: " + "; ".join(errors))
        return artifact

    def _load_private_benchmark(self) -> tuple[SealedBenchmarkSet, list[dict[str, Any]]]:
        path = Path(self.config.private_benchmark_path).expanduser()
        if not path.exists():
            raise HostedResearchLabWorkerError(f"private benchmark path does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise HostedResearchLabWorkerError("private benchmark file must be a JSON object")
        items = payload.get("items") or payload.get("benchmark_items")
        if not isinstance(items, list) or not items:
            raise HostedResearchLabWorkerError("private benchmark requires non-empty items")
        normalized_items = [_normalize_benchmark_item(item) for item in items]
        benchmark_payload = payload.get("benchmark") or payload.get("sealed_benchmark")
        if not isinstance(benchmark_payload, Mapping):
            benchmark_payload = {
                "benchmark_id": str(payload.get("benchmark_id") or "sealed-benchmark:research-lab:private:v1"),
                "icp_set_hash": compute_public_benchmark_hash(normalized_items),
                "split_ref": str(payload.get("split_ref") or "sealed_benchmark:research-lab:private:v1"),
                "item_refs": [item["icp_ref"] for item in normalized_items],
                "scoring_version": str(payload.get("scoring_version") or "qualification-company-scorer:v1"),
                "hidden_plaintext_available": True,
            }
        benchmark = SealedBenchmarkSet.from_mapping(benchmark_payload)
        errors = validate_sealed_benchmark_set(benchmark)
        if errors:
            raise HostedResearchLabWorkerError("private benchmark failed validation: " + "; ".join(errors))
        return benchmark, normalized_items

    async def _generate_candidate_drafts(
        self,
        *,
        context: HostedRunContext,
        artifact: PrivateModelArtifactManifest,
        component_registry: Mapping[str, Any],
        benchmark_public_summary: Mapping[str, Any],
        model_id: str,
        budget_context: Mapping[str, Any],
        max_candidates: int,
    ):
        messages = build_default_auto_research_messages(
            ticket={
                "ticket_id": context.ticket_id,
                "run_id": context.run_id,
                "miner_hotkey": context.ticket.get("miner_hotkey"),
                "island": context.ticket.get("island"),
                "brief_sanitized_ref": context.ticket.get("brief_sanitized_ref"),
                "brief_public_summary": _ticket_doc_value(context, "brief_public_summary"),
                "requested_loop_count": context.ticket.get("requested_loop_count"),
            },
            artifact_manifest=artifact.to_dict(),
            component_registry=component_registry,
            benchmark_public_summary=benchmark_public_summary,
            budget_context=budget_context,
            max_candidates=max_candidates,
        )
        raw = await self._call_openrouter(
            messages=messages,
            api_key=context.provider_env["OPENROUTER_API_KEY"],
            model_id=model_id,
        )
        return parse_auto_research_response(raw, max_candidates=max_candidates)

    async def _call_openrouter(self, *, messages: Sequence[Mapping[str, str]], api_key: str, model_id: str) -> str:
        if not api_key:
            raise HostedResearchLabWorkerError("OpenRouter key is required for hosted auto-research")
        if not model_id:
            raise HostedResearchLabWorkerError("OpenRouter auto-research model is required")
        body = {
            "model": model_id,
            "messages": list(messages),
            "temperature": 0.2,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        }

        def _call() -> str:
            req = urlrequest.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=90) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")[:500]
                raise HostedResearchLabWorkerError(f"OpenRouter candidate generation failed: HTTP {exc.code}: {message}") from exc
            except URLError as exc:
                raise HostedResearchLabWorkerError(f"OpenRouter candidate generation failed: {exc}") from exc
            choices = decoded.get("choices") if isinstance(decoded, Mapping) else None
            if not choices:
                raise HostedResearchLabWorkerError("OpenRouter returned no candidate-generation choices")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise HostedResearchLabWorkerError("OpenRouter returned empty candidate-generation content")
            return str(content)

        return await asyncio.to_thread(_call)

    def _evaluation_run_context(
        self,
        *,
        context: HostedRunContext,
        candidate_index: int,
        patch_hash: str,
    ) -> dict[str, Any]:
        cost_ledger = {
            "run_id": context.run_id,
            "candidate_index": candidate_index,
            "patch_hash": patch_hash,
            "observed_cost_usd": 0.0,
        }
        return {
            "run_id": context.run_id,
            "ticket_id": context.ticket_id,
            "miner_hotkey": str(context.ticket["miner_hotkey"]),
            "island": str(context.ticket["island"]),
            "evaluation_epoch": int(self.config.evaluation_epoch),
            "evaluator_version": "research-lab-private-evaluator:v1",
            "evidence_bundle_refs": ["evidence_bundle:" + sha256_json({"run_id": context.run_id, "candidate_index": candidate_index})],
            "execution_trace_ref": f"execution_trace:{context.run_id}:{candidate_index}",
            "cost_ledger_ref": "cost_ledger:" + canonical_hash(cost_ledger),
            "signature_ref": "",
        }

    def _evaluation_policy(self, benchmark_item_count: int) -> dict[str, Any]:
        return {
            "min_delta": 2.0,
            "min_delta_lcb": 2.0,
            "min_successful_icps": max(1, min(benchmark_item_count, 3)),
            "max_hard_failures": 0,
            "min_candidate_score": 15.0,
            "observed_cost_usd": 0.0,
        }

    def _run_budget_context(self, context: HostedRunContext) -> dict[str, Any]:
        ticket_doc = context.ticket.get("ticket_doc") if isinstance(context.ticket.get("ticket_doc"), Mapping) else {}
        queue_doc = _latest_event_doc(context.queue_events)
        payment_doc = (
            context.payment.get("verification_doc")
            if context.payment and isinstance(context.payment.get("verification_doc"), Mapping)
            else {}
        )
        tier = (
            queue_doc.get("research_model_tier")
            or payment_doc.get("research_model_tier")
            or ticket_doc.get("research_model_tier")
            or self.config.default_auto_research_model_tier
        )
        requested_budget = (
            queue_doc.get("requested_compute_budget_usd")
            or payment_doc.get("compute_budget_usd")
            or payment_doc.get("requested_compute_budget_usd")
            or ticket_doc.get("requested_compute_budget_usd")
            or self.config.default_compute_budget_usd
        )
        max_budget = (
            queue_doc.get("max_compute_budget_usd")
            or payment_doc.get("max_compute_budget_usd")
            or ticket_doc.get("max_compute_budget_usd")
            or self.config.max_compute_budget_usd
        )
        payment_kind = queue_doc.get("payment_kind") or payment_doc.get("payment_kind") or "loop_start"
        additional_budget = queue_doc.get("additional_compute_budget_usd") or payment_doc.get("additional_compute_budget_usd")
        context_doc: dict[str, Any] = {
            "schema_version": "1.0",
            "research_model_tier": str(tier),
            "requested_compute_budget_usd": self.config.clamp_compute_budget_usd(requested_budget),
            "max_compute_budget_usd": self.config.clamp_compute_budget_usd(max_budget),
            "payment_kind": str(payment_kind),
            "budget_policy_version": "research-lab-budget:v1",
        }
        if additional_budget is not None:
            context_doc["additional_compute_budget_usd"] = self.config.clamp_compute_budget_usd(additional_budget)
        if queue_doc.get("continue_from_run_id"):
            context_doc["continue_from_run_id"] = str(queue_doc["continue_from_run_id"])
        if queue_doc.get("topup_reason"):
            context_doc["topup_reason"] = str(queue_doc["topup_reason"])
        return context_doc

    def _max_candidates_for_run(self, budget_context: Mapping[str, Any], model_doc: Mapping[str, Any]) -> int:
        configured = int(model_doc.get("max_candidates") or self.config.hosted_worker_max_candidates)
        budget = float(budget_context.get("requested_compute_budget_usd") or self.config.default_compute_budget_usd)
        budget_limited = max(1, min(configured, int(max(1.0, budget // max(1.0, self.config.min_compute_budget_usd)))))
        return max(1, min(self.config.hosted_worker_max_candidates, budget_limited))


def _normalize_benchmark_item(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise HostedResearchLabWorkerError("private benchmark item must be an object")
    icp = item.get("icp")
    if not isinstance(icp, Mapping):
        raise HostedResearchLabWorkerError("private benchmark item is missing private ICP payload")
    icp_hash = str(item.get("icp_hash") or sha256_json({"icp": icp}))
    icp_ref = str(item.get("icp_ref") or icp_hash)
    return {**dict(item), "icp": dict(icp), "icp_hash": icp_hash, "icp_ref": icp_ref}


def _miner_openrouter_key_ref(context: HostedRunContext) -> str:
    direct = str(context.ticket.get("miner_openrouter_key_ref") or "")
    if direct:
        return direct
    for event in context.ticket_events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping) and event_doc.get("miner_openrouter_key_ref"):
            return str(event_doc["miner_openrouter_key_ref"])
    raise HostedResearchLabWorkerError("Research Lab run is missing miner OpenRouter key ref")


def _ticket_doc_value(context: HostedRunContext, key: str) -> Any:
    ticket_doc = context.ticket.get("ticket_doc")
    if isinstance(ticket_doc, Mapping):
        return ticket_doc.get(key)
    return None


def _payment_id_from_queue_events(events: Sequence[Mapping[str, Any]]) -> str:
    for event in events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping) and event_doc.get("payment_id"):
            return str(event_doc["payment_id"])
    return ""


def _latest_event_doc(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for event in events:
        event_doc = event.get("event_doc")
        if isinstance(event_doc, Mapping):
            return dict(event_doc)
    return {}


def _redacted_budget_context(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "research_model_tier",
        "requested_compute_budget_usd",
        "max_compute_budget_usd",
        "payment_kind",
        "budget_policy_version",
        "additional_compute_budget_usd",
        "continue_from_run_id",
        "topup_reason",
    }
    return {key: value[key] for key in allowed if key in value}


def _followup_status(best_candidate: Mapping[str, Any] | None, promising_delta_threshold: float) -> str:
    if not best_candidate:
        return "completed_no_candidates"
    if best_candidate.get("eligible_for_probation"):
        return "improvement_candidate"
    mean_delta = float(best_candidate.get("mean_delta") or 0.0)
    delta_lcb = float(best_candidate.get("delta_lcb") or 0.0)
    if max(mean_delta, delta_lcb) >= float(promising_delta_threshold):
        return "promising_needs_topup"
    return "completed_no_upgrade"


def _worker_proxy_url(config: ResearchLabGatewayConfig) -> str:
    return str(config.hosted_worker_proxy_url or "").strip()


def _worker_proxy_env(config: ResearchLabGatewayConfig) -> dict[str, str]:
    proxy = _worker_proxy_url(config)
    if not proxy:
        return {}
    env = {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
    }
    no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    return env


def _row_partition(row: Mapping[str, Any], total_workers: int) -> int:
    total = max(1, int(total_workers))
    digest = canonical_hash({"run_id": str(row.get("run_id", ""))}).split(":", 1)[1]
    return int(digest[:12], 16) % total


def _is_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_loop_run_queue_events_run_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _best_candidate_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return dict(
        sorted(
            candidates,
            key=lambda row: (
                float(row.get("mean_delta") or 0.0),
                float(row.get("delta_lcb") or 0.0),
                bool(row.get("eligible_for_probation")),
            ),
            reverse=True,
        )[0]
    )


@contextmanager
def _temporary_env(values: Mapping[str, str]):
    original: dict[str, str | None] = {}
    try:
        for key, value in values.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
