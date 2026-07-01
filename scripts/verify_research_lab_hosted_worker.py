#!/usr/bin/env python3
"""Verify hosted Research Lab worker contracts without network, Docker, or Supabase."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.research_evaluation import build_research_evaluation_score_bundle  # noqa: E402
from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.worker_autostart import build_research_lab_worker_autostart_plan  # noqa: E402
from gateway.research_lab.code_build import (  # noqa: E402
    CodeEditBuildError,
    CodeEditCandidateBuilder,
    resolve_source_inspection_requests,
)
from gateway.research_lab.code_loop_engine import CodeEditLoopEngine  # noqa: E402
from gateway.research_lab.loop_engine import (  # noqa: E402
    AutoResearchLoopEngine,
    AutoResearchLoopEvent,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
)
from gateway.research_lab.store import canonical_hash, scoring_dispatch_event_anchor_payload  # noqa: E402
from gateway.research_lab.scoring_worker import _private_benchmark_row_is_valid  # noqa: E402
import gateway.research_lab.worker as worker_module  # noqa: E402
from gateway.research_lab.worker import (  # noqa: E402
    HostedResearchLabBuilderNotReady,
    RetryableHostedResearchLabWorkerError,
    HostedResearchLabWorkerError,
    ResearchLabHostedWorker,
    _build_openrouter_provider_usage,
    _is_claim_race_error,
    _is_retryable_worker_exception,
    _row_partition,
    _worker_proxy_env,
)
from research_lab.auto_research_prompt import (  # noqa: E402
    build_default_auto_research_messages,
    build_validated_candidate_manifest,
    coerce_component_registry,
    parse_auto_research_response,
)
from research_lab.canonical import sha256_json  # noqa: E402
from research_lab.code_editing import (
    CodeEditDraft,
    CodeEditSourceInspectionRequest,
    validate_code_edit_draft,
)  # noqa: E402
from research_lab.eval.private_runtime import DEFAULT_ENV_PASSTHROUGH  # noqa: E402
from research_lab.eval.artifacts import PrivateModelArtifactManifest  # noqa: E402
from research_lab.validator_integration import verify_research_lab_evaluation_bundle_page  # noqa: E402


def main() -> int:
    errors: list[str] = []

    async def _verify_builder_not_ready_skips_queue() -> None:
        class _NoClaimWorker(ResearchLabHostedWorker):
            def __init__(self, config: ResearchLabGatewayConfig):
                super().__init__(config, worker_ref="test-worker-builder-not-ready")
                self.queue_was_read = False

            async def _next_queued_run(self):  # type: ignore[override]
                self.queue_was_read = True
                return {
                    "run_id": "11111111-1111-4111-8111-111111111111",
                    "ticket_id": "22222222-2222-4222-8222-222222222222",
                    "queue_priority": 0,
                }

        cfg = ResearchLabGatewayConfig(
            hosted_worker_enabled=True,
            hosted_worker_dry_run=True,
            production_writes_enabled=True,
            hosted_runs_enabled=True,
            receipts_enabled=True,
            private_model_manifest_uri=(
                "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json"
            ),
            auto_research_model="test/model",
            private_build_cmd="",
        )
        worker = _NoClaimWorker(cfg)
        original_pause_check = worker_module.is_autoresearch_maintenance_paused

        async def _not_paused() -> bool:
            return False

        worker_module.is_autoresearch_maintenance_paused = _not_paused
        try:
            outcome = await worker.run_once()
        finally:
            worker_module.is_autoresearch_maintenance_paused = original_pause_check
        if worker.queue_was_read:
            errors.append("builder-not-ready worker read the queue before readiness passed")
        if outcome.status != "code_edit_builder_not_ready":
            errors.append(f"builder-not-ready worker returned unexpected status: {outcome.status}")

    async def _verify_openrouter_no_choices_is_retryable() -> None:
        cfg = ResearchLabGatewayConfig(auto_research_model="test/model")
        hosted_worker = ResearchLabHostedWorker(cfg, worker_ref="test-worker-openrouter")
        original_urlopen = worker_module.urlrequest.urlopen
        fake_key = "sk-or-" + "v1-test-key"

        def _fake_no_choices(_request, timeout: int):
            if timeout != 7:
                raise AssertionError("OpenRouter call timeout was not forwarded")
            return _FakeOpenRouterResponse({"id": "gen-no-choices", "model": "test/model", "choices": []})

        worker_module.urlrequest.urlopen = _fake_no_choices
        try:
            try:
                await hosted_worker._call_openrouter(
                    messages=[{"role": "user", "content": '{"task":"test"}'}],
                    api_key=fake_key,
                    model_id="test/model",
                    timeout_seconds=7,
                    max_tokens=16,
                )
                errors.append("OpenRouter no-choices response did not raise")
            except RetryableHostedResearchLabWorkerError as exc:
                message = str(exc)
                if "no candidate-generation choices" not in message:
                    errors.append("OpenRouter no-choices retryable error lost failure reason")
                if "gen-no-choices" not in message:
                    errors.append("OpenRouter no-choices retryable error lost response id")
                if fake_key in message:
                    errors.append("OpenRouter no-choices retryable error leaked API key")
            except HostedResearchLabWorkerError as exc:
                errors.append(f"OpenRouter no-choices response was terminal instead of retryable: {exc}")
        finally:
            worker_module.urlrequest.urlopen = original_urlopen

        flaky_calls = {"count": 0}
        original_attempts = os.environ.get("RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS")

        def _fake_flaky_then_success(_request, timeout: int):
            url = str(getattr(_request, "full_url", "") or "")
            if "/api/v1/generation" in url:
                return _FakeOpenRouterResponse({"data": {"id": "gen-flaky-success", "total_cost": 0.000001}})
            flaky_calls["count"] += 1
            if flaky_calls["count"] == 1:
                return _FakeOpenRouterResponse({"id": "gen-flaky-empty", "model": "test/model", "choices": []})
            return _FakeOpenRouterResponse(
                {
                    "id": "gen-flaky-success",
                    "model": "test/model",
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"candidates":[]}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        os.environ["RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS"] = "2"
        worker_module.urlrequest.urlopen = _fake_flaky_then_success
        try:
            result = await hosted_worker._call_openrouter(
                messages=[{"role": "user", "content": '{"task":"test"}'}],
                api_key=fake_key,
                model_id="test/model",
                timeout_seconds=7,
                max_tokens=16,
            )
            if flaky_calls["count"] != 2:
                errors.append("OpenRouter retry did not retry transient no-choices response exactly once")
            if result.content != '{"candidates":[]}':
                errors.append("OpenRouter retry did not return successful retry content")
        except Exception as exc:
            errors.append(f"OpenRouter transient retry did not recover: {exc}")
        finally:
            worker_module.urlrequest.urlopen = original_urlopen
            if original_attempts is None:
                os.environ.pop("RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS", None)
            else:
                os.environ["RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS"] = original_attempts

        captured_reasoning_body: dict[str, object] = {}

        def _fake_reasoning_request(_request, timeout: int):
            captured_reasoning_body.update(json.loads((_request.data or b"{}").decode("utf-8")))
            return _FakeOpenRouterResponse(
                {
                    "model": "z-ai/glm-5.2",
                    "choices": [{"message": {"content": '{"candidates":[]}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        worker_module.urlrequest.urlopen = _fake_reasoning_request
        try:
            await hosted_worker._call_openrouter(
                messages=[{"role": "user", "content": '{"task":"test"}'}],
                api_key=fake_key,
                model_id="z-ai/glm-5.2",
                reasoning_effort="xhigh",
                timeout_seconds=7,
                max_tokens=16,
            )
            if captured_reasoning_body.get("reasoning_effort") != "xhigh":
                errors.append("OpenRouter reasoning_effort was not forwarded")
            provider_doc = captured_reasoning_body.get("provider")
            if not isinstance(provider_doc, dict) or provider_doc.get("data_collection") != "deny" or provider_doc.get("zdr") is not True:
                errors.append("OpenRouter reasoning request lost provider privacy settings")
        finally:
            worker_module.urlrequest.urlopen = original_urlopen

        def _fake_permanent_error(_request, timeout: int):
            return _FakeOpenRouterResponse(
                {"error": {"code": 401, "message": "invalid api key " + fake_key}}
            )

        worker_module.urlrequest.urlopen = _fake_permanent_error
        try:
            try:
                await hosted_worker._call_openrouter(
                    messages=[{"role": "user", "content": '{"task":"test"}'}],
                    api_key=fake_key,
                    model_id="test/model",
                    timeout_seconds=7,
                    max_tokens=16,
                )
                errors.append("OpenRouter permanent error did not raise")
            except RetryableHostedResearchLabWorkerError as exc:
                errors.append(f"OpenRouter permanent error was incorrectly retryable: {exc}")
            except HostedResearchLabWorkerError as exc:
                if fake_key in str(exc):
                    errors.append("OpenRouter permanent error leaked API key")
        finally:
            worker_module.urlrequest.urlopen = original_urlopen

    async def _verify_worker_retry_classification() -> None:
        retryable_messages = (
            "AWS KMS ThrottlingException: rate exceeded",
            "docker daemon unavailable during image build",
            "Docker exited with exit status 137",
            "No space left on device while extracting parent image",
        )
        for message in retryable_messages:
            if not _is_retryable_worker_exception(RuntimeError(message)):
                errors.append(f"worker did not classify transient infra error as retryable: {message}")
        permanent_messages = (
            "AccessDeniedException: not authorized to perform kms:Decrypt",
            "permission denied for table research_loop_ticket_current",
        )
        for message in permanent_messages:
            if _is_retryable_worker_exception(RuntimeError(message)):
                errors.append(f"worker incorrectly classified permanent auth error as retryable: {message}")

    async def _noop_call(*_args, **_kwargs):
        return OpenRouterCallResult(content='{"candidates":[]}')

    async def _noop_event(_event: AutoResearchLoopEvent):
        return None

    budget_engine = AutoResearchLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=60,
            min_iterations=2,
            max_iterations=12,
            draft_timeout_seconds=30,
            reflection_timeout_seconds=30,
            estimated_iteration_cost_usd=0.5,
            max_candidates=3,
        ),
        call_openrouter=_noop_call,
        event_sink=_noop_event,
    )
    budget_settings = budget_engine._settings_for_budget(
        12,
        {"requested_compute_budget_usd": 1.0},
    )
    if budget_settings.max_iterations != 2:
        errors.append("auto-research budget did not cap requested loop iterations")
    glm_config = ResearchLabGatewayConfig(
        auto_research_model="z-ai/glm-5.2",
        auto_research_reasoning_effort="xhigh",
    )
    _tier, glm_model, glm_doc = glm_config.resolve_auto_research_model(None)
    if glm_model != "z-ai/glm-5.2" or glm_doc.get("reasoning_effort") != "xhigh":
        errors.append("default auto-research model config did not preserve GLM xhigh reasoning effort")
    tier_reasoning_config = ResearchLabGatewayConfig(
        auto_research_model="fallback/model",
        auto_research_reasoning_effort="medium",
        approved_auto_research_models_json='{"glm":{"model":"z-ai/glm-5.2","reasoning_effort":"xhigh"}}',
        default_auto_research_model_tier="glm",
    )
    _tier, tier_model, tier_doc = tier_reasoning_config.resolve_auto_research_model(None)
    if tier_model != "z-ai/glm-5.2" or tier_doc.get("reasoning_effort") != "xhigh":
        errors.append("approved auto-research model tier did not preserve reasoning_effort")
    provider_usage, reconciled_cost = _build_openrouter_provider_usage(
        decoded={"id": "gen-test-1", "model": "test/model", "choices": [{"finish_reason": "stop"}]},
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.000111},
        model_id="fallback/model",
        api_key="test-openrouter-api-key",
        generation_stats_opener=_fake_generation_stats_opener(
            {
                "data": {
                    "id": "gen-test-1",
                    "model": "test/model",
                    "provider_name": "test-provider",
                    "total_cost": 0.001234,
                    "tokens_prompt": 10,
                    "tokens_completion": 20,
                }
            }
        ),
    )
    if reconciled_cost != 1234:
        errors.append("OpenRouter generation stats reconciliation did not override usage cost")
    if provider_usage.get("cost_source") != "openrouter_generation_stats":
        errors.append("OpenRouter provider usage did not record authoritative cost source")
    if provider_usage.get("cost_reconciliation_status") != "confirmed":
        errors.append("OpenRouter provider usage did not mark reconciled cost as confirmed")
    if provider_usage.get("usage_generation_cost_delta_microusd") != 1123:
        errors.append("OpenRouter provider usage did not preserve usage-vs-generation cost delta")
    if not isinstance(provider_usage.get("generation_stats"), dict) or provider_usage["generation_stats"].get("provider_name") != "test-provider":
        errors.append("OpenRouter provider usage did not store sanitized generation stats")
    if provider_usage.get("finish_reason") != "stop":
        errors.append("OpenRouter provider usage did not preserve finish_reason")

    artifact = _artifact()
    registry = coerce_component_registry(_metadata())
    raw_response = _candidate_response()
    drafts = parse_auto_research_response(raw_response, max_candidates=3)
    if len(drafts) != 3:
        errors.append("candidate parser did not return three drafts")
    for idx, draft in enumerate(drafts):
        try:
            manifest, hypothesis, patch = build_validated_candidate_manifest(
                draft=draft,
                artifact_manifest=artifact,
                component_registry=registry,
                run_id="11111111-1111-4111-8111-111111111111",
                sequence=idx,
                miner_brief_ref="brief_sanitized:sha256:abc123",
            )
            if manifest.validation_result != "passed":
                errors.append("candidate patch manifest did not pass validation")
            if hypothesis.component != patch.component:
                errors.append("hypothesis and patch component diverged")
            if draft.target_component_id == "discovery_query_builder" and "append_instruction" not in manifest.patch_doc:
                errors.append("prompt candidate was not normalized to runtime append_instruction")
            if draft.target_component_id == "source_router" and "strategy_option" not in manifest.patch_doc:
                errors.append("source_router candidate was not normalized to runtime strategy_option")
            if draft.target_component_id == "output_budget" and "params" not in manifest.patch_doc:
                errors.append("param candidate was not normalized to runtime params")
        except Exception as exc:
            errors.append(f"valid candidate failed hosted-worker validation: {exc}")

    dispatch_payload = {
        "dispatch_type": "private_baseline_rebenchmark",
        "dispatch_status": "failed",
        "candidate_id": None,
        "run_id": None,
        "ticket_id": None,
        "rolling_window_hash": "sha256:" + "d" * 64,
        "score_bundle_id": None,
        "benchmark_bundle_id": None,
        "worker_ref": "research-lab-scorer-1",
        "proxy_ref_hash": "sha256:" + "e" * 64,
        "event_doc": {"error": "same failure"},
    }
    first_hash = canonical_hash(scoring_dispatch_event_anchor_payload(dispatch_payload, "11111111-1111-4111-8111-111111111111"))
    second_hash = canonical_hash(scoring_dispatch_event_anchor_payload(dispatch_payload, "22222222-2222-4222-8222-222222222222"))
    if first_hash == second_hash:
        errors.append("scoring dispatch event anchored_hash still collides for repeated events")
    if _private_benchmark_row_is_valid(
        {
            "current_benchmark_status": "completed",
            "benchmark_quality": "passed",
            "score_summary_doc": {"per_icp_summaries": [{"company_count": 0}, {"company_count": 0}]},
        }
    ):
        errors.append("all-empty private baseline benchmark row was treated as valid")
    if not _private_benchmark_row_is_valid(
        {
            "current_benchmark_status": "completed",
            "benchmark_quality": "passed",
            "score_summary_doc": {"per_icp_summaries": [{"company_count": 0}, {"company_count": 1}]},
        }
    ):
        errors.append("nonempty private baseline benchmark row was treated as invalid")

    try:
        parse_auto_research_response('{"candidates":[{"hypothesis":{},"patch":{"patch_type":"CODE_EDIT","patch_doc":{}}}]}')
        errors.append("parser accepted CODE_EDIT candidate")
    except ValueError:
        pass

    prompt_messages = build_default_auto_research_messages(
        ticket={"ticket_id": "ticket-1", "run_id": "run-1", "miner_hotkey": "5FminerHotkey111"},
        artifact_manifest=artifact.to_dict(),
        component_registry=registry.to_dict(),
        benchmark_public_summary={"item_count": 3},
        budget_context={
            "research_model_tier": "default",
            "requested_compute_budget_usd": 5.0,
            "payment_kind": "top_up",
            "continue_from_run_id": "run-0",
        },
        max_candidates=2,
    )
    if "budget_context" not in prompt_messages[1]["content"] or "top_up" not in prompt_messages[1]["content"]:
        errors.append("auto-research prompt did not include budget/top-up context")
    if "Do not overfit to one supplied market segment" not in prompt_messages[1]["content"]:
        errors.append("auto-research prompt did not guard against client-specific overfitting")
    if "HTTPS_PROXY" not in DEFAULT_ENV_PASSTHROUGH or "HTTP_PROXY" not in DEFAULT_ENV_PASSTHROUGH:
        errors.append("private Docker runner does not pass through proxy env vars")
    worker_text = (ROOT / "gateway" / "research_lab" / "worker.py").read_text(encoding="utf-8")
    scoring_worker_text = (ROOT / "gateway" / "research_lab" / "scoring_worker.py").read_text(encoding="utf-8")
    local_proxy_text = (ROOT / "qualification" / "validator" / "local_proxy.py").read_text(encoding="utf-8")
    if '"data_collection": "deny"' not in worker_text or '"zdr": True' not in worker_text:
        errors.append("hosted worker OpenRouter calls must enforce data_collection=deny and ZDR")
    if '"data_collection": "deny"' not in local_proxy_text or '"zdr": True' not in local_proxy_text:
        errors.append("local proxy must inject OpenRouter data_collection=deny and ZDR")
    if "_maybe_rebase_stale_candidate_before_scoring" not in scoring_worker_text:
        errors.append("scoring worker must rebase stale-parent candidates before evaluation")
    if "private_model_manifest_hash\", artifact.manifest_hash" not in scoring_worker_text:
        errors.append("private baseline lookup must filter by active private model manifest hash")
    errors.extend(_verify_image_extracted_code_builder(artifact))
    errors.extend(asyncio.run(_verify_code_edit_loop_uses_extracted_source_context(artifact)))

    eval_bundle = _score_bundle()
    page = {
        "schema_version": "1.0",
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": 301,
        "score_bundles": [
            {"bundle_status": "rejected", "current_event_status": "rejected", "score_bundle_doc": eval_bundle},
            {"bundle_status": "scored", "current_event_status": "scored", "score_bundle_doc": eval_bundle},
        ],
        "on_chain_submission_allowed": False,
    }
    verification = verify_research_lab_evaluation_bundle_page(
        page,
        flags={"RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
    )
    if not verification["passed"]:
        errors.append("validator rejected valid scored bundle page: " + "; ".join(verification["errors"]))
    if verification["verified_bundle_count"] != 1 or verification["ignored_bundle_count"] != 1:
        errors.append("validator did not ignore non-scored bundle rows")

    rows = [{"run_id": f"run-{idx}"} for idx in range(64)]
    partitions = {_row_partition(row, 4) for row in rows}
    if partitions != {0, 1, 2, 3}:
        errors.append("worker partitioning does not distribute queued runs across all workers")
    preferred_cfg = ResearchLabGatewayConfig(
        hosted_worker_enabled=True,
        production_writes_enabled=True,
        hosted_runs_enabled=True,
        receipts_enabled=True,
        private_model_manifest_uri="s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json",
        auto_research_model="test/model",
        hosted_worker_index=2,
        hosted_worker_total_workers=4,
        hosted_worker_queue_fetch_limit=64,
    )
    preferred_worker = ResearchLabHostedWorker(preferred_cfg, worker_ref="test-worker-3")
    try:
        preferred_worker._require_enabled()
    except HostedResearchLabWorkerError as exc:
        errors.append(f"worker config should not require private benchmark path: {exc}")
    selected = preferred_worker._select_preferred_queued_row(rows)
    if not selected or _row_partition(selected, 4) != 2:
        errors.append("worker did not prefer its assigned queue partition")
    auto_worker_env = {
        "RESEARCH_LAB_AUTO_START_WORKERS": "true",
        "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS": "true",
        "RESEARCH_LAB_AUTO_START_SCORING_WORKERS": "true",
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "2",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3",
        "RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS": "2",
        "RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS": "3",
        "RESEARCH_LAB_HOSTED_WORKER_INDEX": "5",
        "RESEARCH_LAB_SCORING_WORKER_INDEX": "7",
    }
    for idx in range(1, 5):
        auto_worker_env[f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{idx}"] = f"http://auto-proxy-{idx}"
    for idx in range(1, 7):
        auto_worker_env[f"RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_{idx}"] = f"http://score-proxy-{idx}"
    auto_plan = build_research_lab_worker_autostart_plan(auto_worker_env)
    if auto_plan.hosted.worker_count != 4 or len(auto_plan.hosted.proxy_refs) != 4:
        errors.append("hosted autostart worker count did not follow auto-research proxy count")
    if auto_plan.scoring.worker_count != 6 or len(auto_plan.scoring.proxy_refs) != 6:
        errors.append("scoring autostart worker count did not follow qualification proxy count")
    old_environ = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(auto_worker_env)
        auto_cfg = ResearchLabGatewayConfig.from_env()
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
    if auto_cfg.hosted_worker_total_workers != 4 or auto_cfg.hosted_worker_index != 1:
        errors.append("hosted config did not derive total/index from auto-research proxy count")
    if auto_cfg.scoring_worker_total_workers != 6 or auto_cfg.scoring_worker_index != 1:
        errors.append("scoring config did not derive total/index from qualification proxy count")
    if not _is_claim_race_error(Exception("duplicate key violates research_loop_run_queue_events_run_seq_key")):
        errors.append("worker claim-race detector missed queue event duplicate-key errors")
    ready_worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(), worker_ref="test-worker-ready")
    if ready_worker._code_edit_builder_unavailable_reason():
        errors.append("default code-edit image builder should be ready from code-level defaults")
    disabled_builder_worker = ResearchLabHostedWorker(
        ResearchLabGatewayConfig(code_edit_candidates_enabled=False),
        worker_ref="test-worker-builder-disabled",
    )
    disabled_reason = disabled_builder_worker._code_edit_builder_unavailable_reason() or ""
    if "RESEARCH_LAB_CODE_EDIT_CANDIDATES_ENABLED" not in disabled_reason:
        errors.append("disabled code-edit builder did not report a readiness reason")
    missing_builder_worker = ResearchLabHostedWorker(
        ResearchLabGatewayConfig(private_build_cmd=""),
        worker_ref="test-worker-builder-missing",
    )
    missing_reason = missing_builder_worker._code_edit_builder_unavailable_reason() or ""
    if "RESEARCH_LAB_PRIVATE_BUILD_CMD" not in missing_reason:
        errors.append("missing private build command did not report a readiness reason")
    if not _is_retryable_worker_exception(HostedResearchLabBuilderNotReady("builder not ready")):
        errors.append("builder-not-ready errors must be retryable/requeue-classified")
    asyncio.run(_verify_builder_not_ready_skips_queue())
    asyncio.run(_verify_openrouter_no_choices_is_retryable())
    asyncio.run(_verify_worker_retry_classification())
    proxy_cfg = ResearchLabGatewayConfig(hosted_worker_require_proxy=True, hosted_worker_proxy_url="http://proxy.example:8080")
    proxy_env = _worker_proxy_env(proxy_cfg)
    if proxy_env.get("HTTPS_PROXY") != "http://proxy.example:8080" or proxy_env.get("HTTP_PROXY") != "http://proxy.example:8080":
        errors.append("worker proxy env did not map configured proxy to HTTP/HTTPS proxy variables")
    missing_proxy_cfg = ResearchLabGatewayConfig(hosted_worker_require_proxy=True)
    try:
        ResearchLabHostedWorker(missing_proxy_cfg, worker_ref="test-worker-no-proxy")._require_worker_proxy_for_execution()
        errors.append("worker accepted missing proxy while proxy enforcement was enabled")
    except HostedResearchLabWorkerError:
        pass
    errors.extend(asyncio.run(_verify_auto_research_loop_engine(artifact, registry)))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab hosted worker contracts verified: candidate parser, Engine v1 validation, "
        "image-extracted code builder, validator scored-bundle filtering, worker partitioning, "
        "claim-race detection, no gateway benchmark path."
    )
    return 0


class _FakeOpenRouterResponse:
    def __init__(self, body: dict[str, object]):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


def _fake_generation_stats_opener(body: dict[str, object]):
    def _open(_request, timeout: int):
        if timeout != 5:
            raise AssertionError("generation stats timeout changed unexpectedly")
        return _FakeOpenRouterResponse(body)

    return _open


async def _verify_auto_research_loop_engine(artifact: PrivateModelArtifactManifest, registry) -> list[str]:
    errors: list[str] = []
    events: list[AutoResearchLoopEvent] = []
    calls: list[dict[str, object]] = []

    async def _call_model(messages, timeout_seconds: int, max_tokens: int) -> OpenRouterCallResult:
        calls.append({"timeout_seconds": timeout_seconds, "max_tokens": max_tokens, "message_count": len(messages)})
        if max_tokens <= 700:
            return OpenRouterCallResult(
                content='{"worked":"kept valid typed patches","failed":"none","why":"contract validation passed","next_question":"try one narrower variant","decision":"continue"}',
                provider_usage={"provider": "openrouter", "response_id": f"reflection-{len(calls)}", "cost_microusd": 100000},
                cost_microusd=100000,
            )
        draft_call_count = sum(1 for call in calls if int(call["max_tokens"]) > 700)
        if draft_call_count == 2:
            return OpenRouterCallResult(
                content='{"candidates":[{"hypothesis":{"failure_mode":"bad draft"},"patch":{"patch_type":"PROMPT_EDIT","patch_doc":"not-an-object"}}]}',
                provider_usage={"provider": "openrouter", "response_id": f"draft-{len(calls)}", "cost_microusd": 400000},
                cost_microusd=400000,
            )
        return OpenRouterCallResult(
            content=_candidate_response(),
            provider_usage={"provider": "openrouter", "response_id": f"draft-{len(calls)}", "cost_microusd": 400000},
            cost_microusd=400000,
        )

    async def _event_sink(event: AutoResearchLoopEvent) -> None:
        events.append(event)

    result = await AutoResearchLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=5,
            min_iterations=2,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.5,
            max_candidates=2,
        ),
        call_openrouter=_call_model,
        event_sink=_event_sink,
    ).run(
        run_id="11111111-1111-4111-8111-111111111111",
        ticket={
            "ticket_id": "22222222-2222-4222-8222-222222222222",
            "miner_hotkey": "5FminerHotkey111",
            "island": "generalist",
            "brief_sanitized_ref": "brief_sanitized:sha256:abc123",
            "ticket_doc": {"brief_public_summary": "benchmark-wide improvement"},
            "requested_loop_count": 2,
        },
        artifact=artifact,
        component_registry=registry,
        benchmark_public_summary={"item_count": "validator_resolved"},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=2,
        miner_brief_ref="brief_sanitized:sha256:abc123",
    )

    event_types = [event.event_type for event in events]
    if result.iterations_completed != 2:
        errors.append("auto-research loop engine did not honor requested multi-iteration count")
    if len(calls) < 4:
        errors.append("auto-research loop engine did not make draft and reflection calls per iteration")
    if not result.selected_candidates:
        errors.append("auto-research loop engine did not select candidate finalists")
    if result.actual_openrouter_cost_microusd != 1000000:
        errors.append("auto-research loop engine did not aggregate actual OpenRouter spend")
    if len(result.provider_usage) != 4:
        errors.append("auto-research loop engine did not retain provider usage for all model calls")
    if "patch_validation_failed" not in event_types:
        errors.append("auto-research loop engine did not record malformed draft failures")
    for expected in ("loop_started", "hypothesis_drafted", "patch_drafted", "reflection_recorded", "candidate_selected", "loop_completed"):
        if expected not in event_types:
            errors.append(f"auto-research loop engine missing event: {expected}")
    if "candidate_selected" in event_types and "reflection_recorded" in event_types:
        if event_types.index("candidate_selected") < event_types.index("reflection_recorded"):
            errors.append("auto-research loop selected candidates before recording any reflection")

    budget_guard_events: list[AutoResearchLoopEvent] = []
    budget_guard_calls: list[dict[str, object]] = []

    async def _budget_guard_call(messages, timeout_seconds: int, max_tokens: int) -> OpenRouterCallResult:
        budget_guard_calls.append({"timeout_seconds": timeout_seconds, "max_tokens": max_tokens, "message_count": len(messages)})
        if max_tokens <= 700:
            return OpenRouterCallResult(
                content='{"worked":"reflection should not run","failed":"budget guard failed","why":"unexpected reflection","next_question":"stop","decision":"stop"}',
                provider_usage={"provider": "openrouter", "response_id": "unexpected-reflection", "cost_microusd": 100000},
                cost_microusd=100000,
            )
        return OpenRouterCallResult(
            content=_candidate_response(),
            provider_usage={"provider": "openrouter", "response_id": "budget-draft", "cost_microusd": 1000000},
            cost_microusd=1000000,
        )

    async def _budget_guard_event_sink(event: AutoResearchLoopEvent) -> None:
        budget_guard_events.append(event)

    budget_guard_result = await AutoResearchLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=5,
            min_iterations=1,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.5,
            max_candidates=1,
        ),
        call_openrouter=_budget_guard_call,
        event_sink=_budget_guard_event_sink,
    ).run(
        run_id="33333333-3333-4333-8333-333333333333",
        ticket={
            "ticket_id": "44444444-4444-4444-8444-444444444444",
            "miner_hotkey": "5FminerHotkey222",
            "island": "generalist",
            "brief_sanitized_ref": "brief_sanitized:sha256:def456",
            "ticket_doc": {"brief_public_summary": "benchmark-wide improvement"},
            "requested_loop_count": 2,
        },
        artifact=artifact,
        component_registry=registry,
        benchmark_public_summary={"item_count": "validator_resolved"},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 1.1, "research_model_tier": "default"},
        requested_loop_count=2,
        miner_brief_ref="brief_sanitized:sha256:def456",
    )
    budget_guard_event_types = [event.event_type for event in budget_guard_events]
    if len(budget_guard_calls) != 1:
        errors.append("auto-research hard budget guard did not stop before reflection")
    if budget_guard_result.stop_reason != "compute_budget_exhausted_before_reflection":
        errors.append("auto-research hard budget guard did not report reflection budget exhaustion")
    if budget_guard_result.actual_openrouter_cost_microusd != 1000000:
        errors.append("auto-research hard budget guard did not preserve actual draft spend only")
    if "reflection_recorded" in budget_guard_event_types:
        errors.append("auto-research hard budget guard still recorded a reflection")

    pause_events: list[AutoResearchLoopEvent] = []
    pause_calls: list[dict[str, object]] = []

    async def _pause_call(messages, timeout_seconds: int, max_tokens: int) -> OpenRouterCallResult:
        pause_calls.append({"timeout_seconds": timeout_seconds, "max_tokens": max_tokens, "message_count": len(messages)})
        if max_tokens <= 700:
            return OpenRouterCallResult(
                content='{"worked":"kept valid typed patches","failed":"none","why":"checkpoint boundary reached","next_question":"resume later","decision":"continue"}',
                provider_usage={"provider": "openrouter", "response_id": f"pause-reflection-{len(pause_calls)}", "cost_microusd": 100000},
                cost_microusd=100000,
            )
        return OpenRouterCallResult(
            content=_candidate_response(),
            provider_usage={"provider": "openrouter", "response_id": f"pause-draft-{len(pause_calls)}", "cost_microusd": 400000},
            cost_microusd=400000,
        )

    async def _pause_event_sink(event: AutoResearchLoopEvent) -> None:
        pause_events.append(event)

    async def _pause_after_first_checkpoint() -> bool:
        return any(event.event_type == "checkpoint_saved" for event in pause_events)

    paused_result = await AutoResearchLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=5,
            min_iterations=1,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.5,
            max_candidates=2,
        ),
        call_openrouter=_pause_call,
        event_sink=_pause_event_sink,
    ).run(
        run_id="55555555-5555-4555-8555-555555555555",
        ticket={
            "ticket_id": "66666666-6666-4666-8666-666666666666",
            "miner_hotkey": "5FminerHotkey333",
            "island": "generalist",
            "brief_sanitized_ref": "brief_sanitized:sha256:ghi789",
            "ticket_doc": {"brief_public_summary": "benchmark-wide improvement"},
            "requested_loop_count": 2,
        },
        artifact=artifact,
        component_registry=registry,
        benchmark_public_summary={"item_count": "validator_resolved"},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=2,
        miner_brief_ref="brief_sanitized:sha256:ghi789",
        should_pause=_pause_after_first_checkpoint,
    )
    pause_event_types = [event.event_type for event in pause_events]
    if paused_result.status != "paused" or paused_result.stop_reason != "maintenance_pause_requested":
        errors.append("auto-research maintenance pause did not return paused result")
    if "checkpoint_saved" not in pause_event_types or "loop_paused" not in pause_event_types:
        errors.append("auto-research maintenance pause did not emit checkpoint and paused events")
    if not paused_result.checkpoint_doc or not paused_result.checkpoint_doc.get("checkpoint_hash"):
        errors.append("auto-research maintenance pause did not return checkpoint doc")

    resume_events: list[AutoResearchLoopEvent] = []

    async def _resume_event_sink(event: AutoResearchLoopEvent) -> None:
        resume_events.append(event)

    resumed_result = await AutoResearchLoopEngine(
        settings=AutoResearchLoopSettings(
            min_seconds=0,
            max_seconds=5,
            min_iterations=1,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.5,
            max_candidates=2,
        ),
        call_openrouter=_pause_call,
        event_sink=_resume_event_sink,
    ).run(
        run_id="55555555-5555-4555-8555-555555555555",
        ticket={
            "ticket_id": "66666666-6666-4666-8666-666666666666",
            "miner_hotkey": "5FminerHotkey333",
            "island": "generalist",
            "brief_sanitized_ref": "brief_sanitized:sha256:ghi789",
            "ticket_doc": {"brief_public_summary": "benchmark-wide improvement"},
            "requested_loop_count": 2,
        },
        artifact=artifact,
        component_registry=registry,
        benchmark_public_summary={"item_count": "validator_resolved"},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=2,
        miner_brief_ref="brief_sanitized:sha256:ghi789",
        resume_state=paused_result.checkpoint_doc,
    )
    resume_event_types = [event.event_type for event in resume_events]
    if "loop_resumed" not in resume_event_types:
        errors.append("auto-research resume did not emit loop_resumed event")
    if resumed_result.status != "completed" or not resumed_result.selected_candidates:
        errors.append("auto-research resume did not complete with selected candidates")
    if resumed_result.openrouter_call_count < paused_result.openrouter_call_count:
        errors.append("auto-research resume reset OpenRouter call accounting")
    return errors


def _verify_image_extracted_code_builder(artifact: PrivateModelArtifactManifest) -> list[str]:
    errors: list[str] = []
    try:
        validate_code_edit_draft(_new_top_level_folder_draft())
        errors.append("code-edit validation accepted a new top-level folder")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory(prefix="research-lab-builder-verify-") as tmp:
        tmp_dir = Path(tmp)
        fake_app = tmp_dir / "fake_parent_app"
        _write_fake_parent_app(fake_app)
        fake_docker = _write_fake_docker(tmp_dir)
        manifest_writer = _write_fake_manifest_writer(tmp_dir)
        docker_log = tmp_dir / "docker.log"
        old_env = {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_PARENT_APP": os.environ.get("FAKE_PARENT_APP"),
            "FAKE_DOCKER_LOG": os.environ.get("FAKE_DOCKER_LOG"),
        }
        os.environ["PATH"] = str(fake_docker.parent) + os.pathsep + old_env["PATH"]
        os.environ["FAKE_PARENT_APP"] = str(fake_app)
        os.environ["FAKE_DOCKER_LOG"] = str(docker_log)
        try:
            config = ResearchLabGatewayConfig(
                private_test_cmd="python3 -m py_compile research_lab_adapter.py sourcing_model/__init__.py",
                private_build_cmd=f"python3 {manifest_writer}",
                private_artifact_manifest_output=".research_lab/candidate_manifest.json",
                code_edit_build_timeout_seconds=30,
            )
            builder = CodeEditCandidateBuilder(config)
            source_context = builder.prepare_parent_source_context(
                parent_artifact=artifact,
                workspace_dir=tmp_dir / "source_context",
            )
            if "requirements.txt" in source_context.editable_files:
                errors.append("source context exposed requirements.txt as editable")
            if any(path.endswith(".env") or "/.env" in path for path in source_context.editable_files):
                errors.append("source context exposed env files as editable")
            unread_errors = builder.validate_draft_against_source_context(
                _allowed_runtime_patch_draft(),
                source_context,
                read_paths=(),
                require_read=True,
            )
            if "code_edit_unread_source_file:sourcing_model/__init__.py" not in unread_errors:
                errors.append("source context validation did not reject unread target file")
            batch = resolve_source_inspection_requests(
                source_context,
                [
                    CodeEditSourceInspectionRequest(
                        operation="search",
                        query="qualify",
                        rationale="locate model entry point",
                    ),
                    CodeEditSourceInspectionRequest(
                        operation="read_file",
                        path="sourcing_model/__init__.py",
                        rationale="read exact target file",
                    ),
                    CodeEditSourceInspectionRequest(
                        operation="read_file",
                        path="sourcing_model/secret_like.py",
                        rationale="verify redaction",
                    ),
                ],
                already_read_paths=(),
                max_files=8,
                max_file_bytes=24_000,
                max_total_bytes=120_000,
                max_search_matches=30,
            )
            if "sourcing_model/__init__.py" not in batch.read_paths:
                errors.append("source inspection read did not record requested file")
            if batch.model_context.get("bytes_returned", 0) <= 0:
                errors.append("source inspection read returned no source bytes")
            serialized_context = json.dumps(batch.model_context)
            if "password=not-a-real-secret" in serialized_context:
                errors.append("source inspection leaked secret-like source content")
            if "[redacted secret-like source line]" not in serialized_context:
                errors.append("source inspection did not redact secret-like source line")
            result = CodeEditCandidateBuilder(config).build(
                draft=_allowed_runtime_patch_draft(),
                parent_artifact=artifact,
                run_id="77777777-7777-4777-8777-777777777777",
                candidate_index=0,
            )
            build_doc = result.build_doc
            if build_doc.get("source_mode") != "parent_image_extract":
                errors.append("code builder did not record parent image extraction source mode")
            if not build_doc.get("generated_build_scaffold"):
                errors.append("code builder did not record generated build scaffold")
            if not build_doc.get("extracted_source_tree_hash_before_patch", "").startswith("sha256:"):
                errors.append("code builder did not record extracted source tree hash")
            if "sourcing_model/__init__.py" not in build_doc.get("changed_files", []):
                errors.append("code builder did not apply the allowed runtime patch")
            if result.candidate_model_manifest.model_artifact_hash == artifact.model_artifact_hash:
                errors.append("code builder accepted candidate manifest with unchanged artifact hash")
            docker_calls = docker_log.read_text(encoding="utf-8").splitlines()
            for expected in ("image inspect", "create", "cp", "rm -f"):
                if not any(expected in call for call in docker_calls):
                    errors.append(f"fake docker did not observe expected call: {expected}")
            if any(" pull " in f" {call} " for call in docker_calls):
                errors.append("code builder pulled parent image even though image inspect succeeded")

            result_recount = CodeEditCandidateBuilder(config).build(
                draft=_allowed_runtime_patch_draft_with_bad_hunk_counts(),
                parent_artifact=artifact,
                run_id="77777777-7777-4777-8777-777777777778",
                candidate_index=1,
            )
            if "sourcing_model/__init__.py" not in result_recount.build_doc.get("changed_files", []):
                errors.append("code builder did not tolerate generated diff hunk count drift")
        except Exception as exc:
            errors.append(f"image-extracted code builder failed valid fake build: {exc}")
        finally:
            os.environ["PATH"] = old_env["PATH"]
            for key in ("FAKE_PARENT_APP", "FAKE_DOCKER_LOG"):
                old_value = old_env[key]
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    with tempfile.TemporaryDirectory(prefix="research-lab-builder-missing-") as tmp:
        tmp_dir = Path(tmp)
        fake_app = tmp_dir / "fake_parent_app"
        _write_fake_parent_app(fake_app, omit=("validator_models",))
        fake_docker = _write_fake_docker(tmp_dir)
        manifest_writer = _write_fake_manifest_writer(tmp_dir)
        docker_log = tmp_dir / "docker.log"
        old_env = {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_PARENT_APP": os.environ.get("FAKE_PARENT_APP"),
            "FAKE_DOCKER_LOG": os.environ.get("FAKE_DOCKER_LOG"),
        }
        os.environ["PATH"] = str(fake_docker.parent) + os.pathsep + old_env["PATH"]
        os.environ["FAKE_PARENT_APP"] = str(fake_app)
        os.environ["FAKE_DOCKER_LOG"] = str(docker_log)
        try:
            config = ResearchLabGatewayConfig(
                private_test_cmd="python3 -m py_compile research_lab_adapter.py sourcing_model/__init__.py",
                private_build_cmd=f"python3 {manifest_writer}",
                private_artifact_manifest_output=".research_lab/candidate_manifest.json",
                code_edit_build_timeout_seconds=30,
            )
            CodeEditCandidateBuilder(config).build(
                draft=_allowed_runtime_patch_draft(),
                parent_artifact=artifact,
                run_id="88888888-8888-4888-8888-888888888888",
                candidate_index=0,
            )
            errors.append("code builder accepted parent image /app missing required runtime path")
        except CodeEditBuildError as exc:
            if "missing required runtime paths" not in str(exc):
                errors.append(f"missing runtime path error was not clear: {exc}")
        except Exception as exc:
            errors.append(f"missing runtime path check raised wrong exception: {exc}")
        finally:
            os.environ["PATH"] = old_env["PATH"]
            for key in ("FAKE_PARENT_APP", "FAKE_DOCKER_LOG"):
                old_value = old_env[key]
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
    return errors


async def _verify_code_edit_loop_uses_extracted_source_context(
    artifact: PrivateModelArtifactManifest,
) -> list[str]:
    errors: list[str] = []
    events: list[AutoResearchLoopEvent] = []
    calls: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="research-lab-code-loop-verify-") as tmp:
        tmp_dir = Path(tmp)
        fake_app = tmp_dir / "fake_parent_app"
        _write_fake_parent_app(fake_app)
        fake_docker = _write_fake_docker(tmp_dir)
        manifest_writer = _write_fake_manifest_writer(tmp_dir)
        docker_log = tmp_dir / "docker.log"
        old_env = {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_PARENT_APP": os.environ.get("FAKE_PARENT_APP"),
            "FAKE_DOCKER_LOG": os.environ.get("FAKE_DOCKER_LOG"),
        }
        os.environ["PATH"] = str(fake_docker.parent) + os.pathsep + old_env["PATH"]
        os.environ["FAKE_PARENT_APP"] = str(fake_app)
        os.environ["FAKE_DOCKER_LOG"] = str(docker_log)
        inspection_call_count = 0
        repair_call_count = 0

        async def _call_model(messages, timeout_seconds: int, max_tokens: int) -> OpenRouterCallResult:
            nonlocal inspection_call_count, repair_call_count
            content = "\n".join(str(message.get("content") or "") for message in messages)
            is_source_inspection = "runtime_source_index" in content
            is_repair = "failed_patch" in content and "git_apply_error" in content
            if is_source_inspection:
                inspection_call_count += 1
            if is_repair:
                repair_call_count += 1
            calls.append(
                {
                    "stage": (
                        "source_inspection"
                        if is_source_inspection
                        else "code_edit_repair"
                        if is_repair
                        else "code_edit_draft"
                    ),
                    "timeout_seconds": timeout_seconds,
                    "max_tokens": max_tokens,
                    "has_runtime_source_context": "runtime_source_context" in content,
                    "has_runtime_source_index": "runtime_source_index" in content,
                    "has_source_inspection_context": "source_inspection_context" in content,
                    "has_real_file": "sourcing_model/__init__.py" in content,
                    "has_source_excerpt": "VALUE" in content and "def qualify" in content,
                    "has_bad_example": "sourcing_model/example.py" in content,
                }
            )
            if is_source_inspection:
                if inspection_call_count == 1:
                    return OpenRouterCallResult(
                        content=json.dumps(
                            {
                                "requests": [
                                    {
                                        "operation": "read_file",
                                        "path": "sourcing_model/__init__.py",
                                        "rationale": "read exact model entry source before editing",
                                    }
                                ]
                            },
                            sort_keys=True,
                        ),
                        provider_usage={
                            "provider": "openrouter",
                            "response_id": f"code-edit-source-inspection-{inspection_call_count}",
                            "cost_microusd": 1000,
                        },
                        cost_microusd=1000,
                    )
                return OpenRouterCallResult(
                    content=json.dumps(
                        {"requests": [{"operation": "finish", "rationale": "enough source has been read"}]},
                        sort_keys=True,
                    ),
                    provider_usage={
                        "provider": "openrouter",
                        "response_id": f"code-edit-source-inspection-{inspection_call_count}",
                        "cost_microusd": 1000,
                    },
                        cost_microusd=1000,
                    )
            if is_repair:
                if repair_call_count == 1:
                    return OpenRouterCallResult(
                        content=json.dumps(
                            {"candidates": [{"code_edit": {"target_files": ["sourcing_model/__init__.py"]}}]},
                            sort_keys=True,
                        ),
                        provider_usage={
                            "provider": "openrouter",
                            "response_id": "code-edit-repair-bad-schema",
                            "cost_microusd": 1000,
                        },
                        cost_microusd=1000,
                    )
                return OpenRouterCallResult(
                    content=json.dumps(
                        {
                            "code_edit": {
                                "target_files": ["sourcing_model/__init__.py"],
                                "unified_diff": _allowed_runtime_patch_draft().unified_diff,
                                "redacted_summary": "repair existing extracted file edit",
                                "test_plan": "py_compile changed file",
                                "rollback_plan": "revert patch",
                            },
                        },
                        sort_keys=True,
                    ),
                    provider_usage={"provider": "openrouter", "response_id": "code-edit-repair-draft", "cost_microusd": 1000},
                    cost_microusd=1000,
                )
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            {
                                "lane": "query_construction",
                                "hypothesis": {
                                    "failure_mode": "stale query constant",
                                    "mechanism": "small source edit against extracted file",
                                    "expected_improvement": "better source-context grounded edits",
                                    "risk": "low",
                                    "predicted_delta": 1.0,
                                },
                                "code_edit": {
                                    "target_files": ["sourcing_model/__init__.py"],
                                    "unified_diff": (
                                        "Here is the patch:\n```diff\n"
                                        "diff --git a/sourcing_model/__init__.py b/sourcing_model/__init__.py\n"
                                        "--- a/sourcing_model/__init__.py\n"
                                        "+++ b/sourcing_model/__init__.py\n"
                                        "@@ -1,5 +1,5 @@\n"
                                        "this is not a valid unified diff hunk\n"
                                        "```\n"
                                    ),
                                    "redacted_summary": "edit existing extracted file",
                                    "test_plan": "py_compile changed file",
                                    "rollback_plan": "revert patch",
                                },
                            }
                        ]
                    },
                    sort_keys=True,
                ),
                provider_usage={"provider": "openrouter", "response_id": "code-edit-source-context-draft", "cost_microusd": 1000},
                cost_microusd=1000,
            )

        async def _event_sink(event: AutoResearchLoopEvent) -> None:
            events.append(event)

        try:
            config = ResearchLabGatewayConfig(
                private_test_cmd="python3 -m py_compile research_lab_adapter.py sourcing_model/__init__.py",
                private_build_cmd=f"python3 {manifest_writer}",
                private_artifact_manifest_output=".research_lab/candidate_manifest.json",
                code_edit_build_timeout_seconds=30,
            )
            result = await CodeEditLoopEngine(
                settings=AutoResearchLoopSettings(
                    min_seconds=0,
                    max_seconds=30,
                    min_iterations=1,
                    max_iterations=1,
                    draft_timeout_seconds=10,
                    reflection_timeout_seconds=10,
                    estimated_iteration_cost_usd=0.01,
                    max_candidates=1,
                ),
                call_openrouter=_call_model,
                event_sink=_event_sink,
                builder=CodeEditCandidateBuilder(config),
            ).run(
                run_id="99999999-9999-4999-8999-999999999999",
                ticket={
                    "ticket_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "miner_hotkey": "5FminerHotkeyCodeEdit",
                    "island": "generalist",
                    "brief_sanitized_ref": "brief_sanitized:sha256:sourcecontext",
                    "ticket_doc": {"brief_public_summary": "source-context grounded code edit"},
                    "requested_loop_count": 1,
                },
                artifact=artifact,
                component_registry={},
                benchmark_public_summary={"item_count": 10},
                model_id="test/code-edit-model",
                budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
                requested_loop_count=1,
            )
            if not result.selected_candidates:
                errors.append("code-edit loop did not build a candidate from extracted source context")
            if not calls or calls[0].get("stage") != "source_inspection":
                errors.append("code-edit loop did not inspect source before drafting")
            if not calls or not calls[0].get("has_runtime_source_index"):
                errors.append("source inspection prompt did not include runtime_source_index")
            if calls and calls[0].get("has_source_excerpt"):
                errors.append("source inspection prompt included raw source before read_file")
            draft_calls = [call for call in calls if call.get("stage") == "code_edit_draft"]
            if not draft_calls or not draft_calls[0].get("has_runtime_source_context"):
                errors.append("code-edit draft prompt did not include runtime_source_context")
            if not draft_calls or not draft_calls[0].get("has_source_inspection_context"):
                errors.append("code-edit draft prompt did not include source_inspection_context")
            if not draft_calls or not draft_calls[0].get("has_real_file"):
                errors.append("code-edit draft prompt did not include real extracted editable file paths")
            if not draft_calls or not draft_calls[0].get("has_source_excerpt"):
                errors.append("code-edit draft prompt did not include inspected source content")
            if calls and any(call.get("has_bad_example") for call in calls):
                errors.append("code-edit draft prompt still suggests nonexistent sourcing_model/example.py")
            event_types = [event.event_type for event in events]
            if "source_inspection_requested" not in event_types:
                errors.append("code-edit loop did not record source_inspection_requested")
            if "source_inspection_resolved" not in event_types:
                errors.append("code-edit loop did not record source_inspection_resolved")
            if "code_edit_drafted" not in event_types:
                errors.append("code-edit loop did not record code_edit_drafted")
            elif (
                "source_inspection_resolved" in event_types
                and event_types.index("source_inspection_resolved") > event_types.index("code_edit_drafted")
            ):
                errors.append("code-edit loop drafted before resolving source inspection")
            if "candidate_patch_apply_failed" not in event_types:
                errors.append("code-edit loop did not classify the malformed draft as candidate_patch_apply_failed")
            if "code_edit_repair_requested" not in event_types:
                errors.append("code-edit loop did not request patch repair after apply failure")
            if "code_edit_repair_drafted" not in event_types:
                errors.append("code-edit loop did not record repaired patch draft")
            if "candidate_build_passed" not in event_types:
                errors.append("code-edit loop did not record candidate_build_passed")
            elif (
                "code_edit_repair_drafted" in event_types
                and event_types.index("code_edit_repair_drafted") > event_types.index("candidate_build_passed")
            ):
                errors.append("candidate build passed before repaired patch was drafted")
            start_events = [event for event in events if event.event_type in {"loop_started", "loop_resumed"}]
            first_doc = start_events[0].event_doc if start_events else {}
            if first_doc.get("source_mode") != "parent_image_extract":
                errors.append("code-edit loop start event did not record parent image extraction")
            source_prepare_events = [
                event for event in events if event.event_doc.get("operation") == "parent_image_source_prepare"
            ]
            if not source_prepare_events:
                errors.append("code-edit loop did not record parent image source preparation events")
            elif source_prepare_events[0].event_type != "source_inspection_requested":
                errors.append("code-edit source preparation did not start with source_inspection_requested")
            docker_calls = docker_log.read_text(encoding="utf-8").splitlines()
            if sum(1 for call in docker_calls if call.startswith("cp ")) != 1:
                errors.append("code-edit loop should extract parent image once before drafting")
        except Exception as exc:
            errors.append(f"code-edit loop source-context verification failed: {exc}")
        finally:
            os.environ["PATH"] = old_env["PATH"]
            for key in ("FAKE_PARENT_APP", "FAKE_DOCKER_LOG"):
                old_value = old_env[key]
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    return errors


def _write_fake_parent_app(root: Path, *, omit: tuple[str, ...] = ()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel in ("gateway", "qualification", "sourcing_model", "validator_models"):
        if rel in omit:
            continue
        (root / rel).mkdir(parents=True, exist_ok=True)
        (root / rel / "__init__.py").write_text("", encoding="utf-8")
    (root / "sourcing_model" / "__init__.py").write_text(
        'VALUE = "old"\n\n\ndef qualify():\n    return VALUE\n',
        encoding="utf-8",
    )
    (root / "sourcing_model" / "secret_like.py").write_text(
        'CONFIG_VALUE = "password=not-a-real-secret"\n',
        encoding="utf-8",
    )
    (root / "gateway" / ".env").write_text("SHOULD_NOT_BE_INDEXED=1\n", encoding="utf-8")
    if "research_lab_adapter.py" not in omit:
        (root / "research_lab_adapter.py").write_text(
            'def adapter_metadata():\n    return {"adapter_version": "fake-adapter:v1"}\n',
            encoding="utf-8",
        )
    if "requirements.txt" not in omit:
        (root / "requirements.txt").write_text("", encoding="utf-8")


def _write_fake_docker(root: Path) -> Path:
    docker_path = root / "docker"
    docker_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import shutil
            import sys
            from pathlib import Path

            log_path = Path(os.environ["FAKE_DOCKER_LOG"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.open("a", encoding="utf-8").write(" ".join(sys.argv[1:]) + "\\n")

            args = sys.argv[1:]
            if args[:2] == ["image", "inspect"]:
                sys.exit(0)
            if args and args[0] == "pull":
                sys.exit(0)
            if args and args[0] == "create":
                print("fake-container")
                sys.exit(0)
            if args and args[0] == "cp":
                src = Path(os.environ["FAKE_PARENT_APP"])
                dest = Path(args[-1])
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
                sys.exit(0)
            if args[:2] == ["rm", "-f"]:
                sys.exit(0)
            sys.stderr.write("unexpected fake docker command: " + " ".join(args))
            sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    docker_path.chmod(0o700)
    return docker_path


def _write_fake_manifest_writer(root: Path) -> Path:
    writer = root / "write_candidate_manifest.py"
    writer.write_text(
        textwrap.dedent(
            """\
            import hashlib
            import json
            import os
            from pathlib import Path

            def sha256_json(value):
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                return "sha256:" + hashlib.sha256(encoded).hexdigest()

            output = Path(os.environ["RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT"])
            git_commit_sha = os.environ["RESEARCH_LAB_PRIVATE_COMMIT_SHA"]
            dockerfile = Path("Dockerfile.research-lab").read_text(encoding="utf-8")
            if "python:3.11-slim" in dockerfile:
                raise SystemExit("candidate Dockerfile should not use Docker Hub python base image")
            if not dockerfile.startswith("FROM 123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:"):
                raise SystemExit("candidate Dockerfile should inherit from parent ECR image")
            payload = {
                "model_artifact_hash": "sha256:" + "8" * 64,
                "git_commit_sha": git_commit_sha,
                "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "9" * 64,
                "config_hash": "sha256:" + "a" * 64,
                "component_registry_version": "sourcing-model-components:v1",
                "scoring_adapter_version": "qualification-company-scorer:v1",
                "manifest_uri": "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/candidates/test.json",
                "signature_ref": "kms-signature:test",
                "build_id": "fake-build",
            }
            manifest = {**payload, "manifest_hash": sha256_json(payload)}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    return writer


def _allowed_runtime_patch_draft() -> CodeEditDraft:
    return CodeEditDraft(
        failure_mode="Runtime query term is stale.",
        mechanism="Small source edit inside extracted runtime.",
        expected_improvement="Better precision on future ICPs.",
        risk="Low.",
        lane="query_construction",
        target_files=("sourcing_model/__init__.py",),
        unified_diff=(
            "diff --git a/sourcing_model/__init__.py b/sourcing_model/__init__.py\n"
            "--- a/sourcing_model/__init__.py\n"
            "+++ b/sourcing_model/__init__.py\n"
            "@@ -1,5 +1,5 @@\n"
            '-VALUE = "old"\n'
            '+VALUE = "new"\n'
            " \n"
            " \n"
            " def qualify():\n"
            "     return VALUE\n"
        ),
        redacted_summary="Change one runtime source constant.",
        test_plan="py_compile changed file.",
        rollback_plan="Revert this diff.",
    )


def _allowed_runtime_patch_draft_with_bad_hunk_counts() -> CodeEditDraft:
    draft = _allowed_runtime_patch_draft()
    return CodeEditDraft(
        failure_mode=draft.failure_mode,
        mechanism=draft.mechanism,
        expected_improvement=draft.expected_improvement,
        risk=draft.risk,
        lane=draft.lane,
        target_files=draft.target_files,
        unified_diff=draft.unified_diff.replace("@@ -1,5 +1,5 @@", "@@ -1,99 +1,99 @@"),
        redacted_summary=draft.redacted_summary,
        test_plan=draft.test_plan,
        rollback_plan=draft.rollback_plan,
        predicted_delta=draft.predicted_delta,
    )


def _new_top_level_folder_draft() -> CodeEditDraft:
    return CodeEditDraft(
        failure_mode="Bad scope.",
        mechanism="Attempts to create a new top-level folder.",
        expected_improvement="None.",
        risk="High.",
        lane="query_construction",
        target_files=("new_folder/file.py",),
        unified_diff=textwrap.dedent(
            """\
            diff --git a/new_folder/file.py b/new_folder/file.py
            new file mode 100644
            --- /dev/null
            +++ b/new_folder/file.py
            @@ -0,0 +1 @@
            +VALUE = 1
            """
        ),
        redacted_summary="Invalid top-level folder edit.",
        test_plan="N/A",
        rollback_plan="N/A",
    )


def _metadata() -> dict[str, object]:
    return {
        "component_registry": {
            "manifest_version": "sourcing-model-components:v1",
            "champion_base": "sourcing-model-research-lab-adapter:v1",
            "eval_version": "research-lab-private-evaluator:v1",
            "source_receipt_refs": ["receipt:sha256:" + "a" * 64],
            "entries": [
                {
                    "name": "discovery_query_builder",
                    "purpose": "Build precise search prompts for high-intent lead discovery.",
                    "input_contract": "ICP and observable intent signals",
                    "output_contract": "Search query template",
                    "ablation_leverage": 1.0,
                    "allowed_patch_types": ["PROMPT_EDIT"],
                    "token_budget": 1200,
                    "cost_budget_cents": 10,
                    "prompt_required_placeholders": ["icp", "intent_signals"],
                    "source_evidence_refs": ["component_registry:query-builder:v1"],
                },
                {
                    "name": "source_router",
                    "purpose": "Route discovery toward the best evidence source.",
                    "input_contract": "ICP and signal type",
                    "output_contract": "Source strategy",
                    "ablation_leverage": 1.0,
                    "allowed_patch_types": ["STRATEGY_SWAP"],
                    "token_budget": 800,
                    "cost_budget_cents": 8,
                    "strategy_options": ["news", "job_listing", "social", "company_site"],
                    "source_evidence_refs": ["component_registry:source-router:v1"],
                },
                {
                    "name": "output_budget",
                    "purpose": "Control number of returned leads per ICP.",
                    "input_contract": "Budget and lead cap",
                    "output_contract": "max_leads parameter",
                    "ablation_leverage": 0.5,
                    "allowed_patch_types": ["PARAM_EDIT"],
                    "token_budget": 400,
                    "cost_budget_cents": 4,
                    "param_bounds": {"max_leads": {"min": 1, "max": 5}},
                    "source_evidence_refs": ["component_registry:output-budget:v1"],
                },
            ],
        }
    }


def _candidate_response() -> str:
    return """
{
  "candidates": [
    {
      "hypothesis": {
        "failure_mode": "Queries are too broad for observable buying intent.",
        "mechanism": "Tightening the query template around fresh budget, hiring, and vendor replacement signals should improve precision across benchmark ICPs.",
        "expected_improvement": "Higher candidate score on ICPs where intent evidence is fresh and specific.",
        "risk": "Could reduce coverage on sparse markets.",
        "predicted_delta": 3.0,
        "falsifier": "proxy_score"
      },
      "patch": {
        "patch_type": "PROMPT_EDIT",
        "target_component_id": "discovery_query_builder",
        "patch_doc": {
          "template_name": "general_intent_query",
          "new_template": "Find companies matching {icp} with recent budget, hiring, expansion, vendor replacement, or compliance signals tied to {intent_signals}."
        },
        "redacted_summary": "Tighten discovery query toward fresh buying-intent evidence."
      }
    },
    {
      "hypothesis": {
        "failure_mode": "The model underuses hiring evidence for fast-moving teams.",
        "mechanism": "Routing one candidate through job listings should capture active buildout intent.",
        "expected_improvement": "Better recall for companies hiring around the target workflow.",
        "risk": "Hiring can be noisy if roles are generic.",
        "predicted_delta": 2.0,
        "falsifier": "coverage"
      },
      "patch": {
        "patch_type": "STRATEGY_SWAP",
        "target_component_id": "source_router",
        "patch_doc": {"strategy_name": "job_listing"},
        "redacted_summary": "Route a candidate through job-listing evidence."
      }
    },
    {
      "hypothesis": {
        "failure_mode": "Too many returned leads dilutes precision.",
        "mechanism": "Reducing max leads prioritizes strongest evidence.",
        "expected_improvement": "Higher average company quality.",
        "risk": "Potential coverage loss.",
        "predicted_delta": 1.5,
        "falsifier": "evidence_defect_rate"
      },
      "patch": {
        "patch_type": "PARAM_EDIT",
        "target_component_id": "output_budget",
        "patch_doc": {"param_name": "max_leads", "new_value": 3},
        "redacted_summary": "Reduce output budget to improve precision."
      }
    }
  ]
}
"""


def _artifact() -> PrivateModelArtifactManifest:
    payload = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "git_commit_sha": "abcdef1234567890",
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:" + "2" * 64,
        "config_hash": "sha256:" + "3" * 64,
        "component_registry_version": "sourcing-model-components:v1",
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "manifest_uri": "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json",
        "signature_ref": "kms-signature:research-lab-artifact-signing:test",
        "build_id": "test-build",
    }
    return PrivateModelArtifactManifest.from_mapping({**payload, "manifest_hash": sha256_json(payload)})


def _score_bundle() -> dict[str, object]:
    return build_research_evaluation_score_bundle(
        run_id="11111111-1111-4111-8111-111111111111",
        ticket_id="22222222-2222-4222-8222-222222222222",
        miner_hotkey="5FevalMiner111111111111111111111111111111111",
        island="generalist",
        evaluation_epoch=301,
        parent_artifact_hash="sha256:" + "1" * 64,
        candidate_artifact_hash="sha256:" + "2" * 64,
        private_model_manifest_hash="sha256:" + "3" * 64,
        candidate_patch_hash="sha256:" + "4" * 64,
        icp_set_hash="sha256:" + "5" * 64,
        scoring_version="qualification-company-scorer:v1",
        evaluator_version="research-lab-private-evaluator:v1",
        per_icp_results=[
            {
                "icp_ref": "icp:a",
                "icp_hash": "sha256:" + "a" * 64,
                "base_company_scores": [80, 60],
                "candidate_company_scores": [90, 70],
            }
        ],
        evidence_bundle_refs=["evidence_bundle:sha256:" + "6" * 64],
        execution_trace_ref="execution_trace:11111111-1111-4111-8111-111111111111",
        cost_ledger_ref="cost_ledger:sha256:" + "7" * 64,
        benchmark_split_ref="sealed_benchmark:qualification:intent:v1",
        policy={
            "min_delta": 2.0,
            "min_delta_lcb": 2.0,
            "min_successful_icps": 1,
            "min_candidate_score": 15.0,
            "observed_cost_usd": 1.25,
        },
        signature_ref="kms-signature:research-lab-eval:test",
    )


if __name__ == "__main__":
    raise SystemExit(main())
