#!/usr/bin/env python3
"""Verify hosted Research Lab worker contracts without network, Docker, or Supabase."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.research_evaluation import build_research_evaluation_score_bundle  # noqa: E402
from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.loop_engine import (  # noqa: E402
    AutoResearchLoopEngine,
    AutoResearchLoopEvent,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
)
from gateway.research_lab.store import canonical_hash, scoring_dispatch_event_anchor_payload  # noqa: E402
from gateway.research_lab.scoring_worker import _private_benchmark_row_is_valid  # noqa: E402
from gateway.research_lab.worker import (  # noqa: E402
    HostedResearchLabWorkerError,
    ResearchLabHostedWorker,
    _is_claim_race_error,
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
from research_lab.eval.private_runtime import DEFAULT_ENV_PASSTHROUGH  # noqa: E402
from research_lab.eval.artifacts import PrivateModelArtifactManifest  # noqa: E402
from research_lab.validator_integration import verify_research_lab_evaluation_bundle_page  # noqa: E402


def main() -> int:
    errors: list[str] = []
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
            if draft.target_component_id == "source_router":
                errors.append("runtime-incompatible source_router strategy candidate passed validation")
            if manifest.validation_result != "passed":
                errors.append("candidate patch manifest did not pass validation")
            if hypothesis.component != patch.component:
                errors.append("hypothesis and patch component diverged")
        except Exception as exc:
            if draft.target_component_id == "source_router":
                continue
            errors.append(f"valid candidate failed hosted-worker validation: {exc}")
    if "source_router" in registry.by_name():
        errors.append("runtime-incompatible source_router strategy component was exposed to auto-research")

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
    if not _is_claim_race_error(Exception("duplicate key violates research_loop_run_queue_events_run_seq_key")):
        errors.append("worker claim-race detector missed queue event duplicate-key errors")
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
        "validator scored-bundle filtering, worker partitioning, claim-race detection, no gateway benchmark path."
    )
    return 0


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
    return errors


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
