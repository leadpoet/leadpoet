#!/usr/bin/env python3
"""Verify hosted Research Lab worker contracts without network, Docker, or Supabase."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.research_evaluation import build_research_evaluation_score_bundle  # noqa: E402
from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
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
        except Exception as exc:
            errors.append(f"valid candidate failed hosted-worker validation: {exc}")

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
        evaluation_bundles_enabled=True,
        private_benchmark_path="/sealed/benchmark.json",
        auto_research_model="test/model",
        hosted_worker_index=2,
        hosted_worker_total_workers=4,
        hosted_worker_queue_fetch_limit=64,
    )
    preferred_worker = ResearchLabHostedWorker(preferred_cfg, worker_ref="test-worker-3")
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

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab hosted worker contracts verified: candidate parser, Engine v1 validation, "
        "validator scored-bundle filtering, worker partitioning, claim-race detection."
    )
    return 0


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
