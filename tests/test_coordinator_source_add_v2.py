from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from gateway.research_lab.source_add_provenance import (
    PRECHECK_MANUAL,
    PRECHECK_PASSED,
)
from gateway.tee.coordinator_executor_v2 import CoordinatorExecutorV2
from gateway.tee.coordinator_source_add_v2 import (
    CoordinatorSourceAddProvenanceV2,
    SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.reward_executor_v2 import OP_RESEARCH_LAB_REWARD_DECISION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _payload():
    return {
        "schema_version": SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION,
        "submission_id": "source_add_submission:1234567890abcdef",
        "source_name": "Example Data",
        "source_kind": "firmographic",
        "declared_base_domains": ["api.example.com"],
        "source_metadata": {
            "api_base_url": "https://api.example.com/v1",
            "documentation_url": "https://docs.example.com/api",
            "auth_type": "bearer",
            "endpoint_examples": [{"method": "GET", "path": "/companies"}],
            "rate_limit_notes": "100 requests per minute",
            "third_party_refs": ["https://www.g2.com/products/example"],
        },
        "timeout_seconds": 45,
    }


def _context():
    attempts = []
    artifacts = []
    return SimpleNamespace(
        job_id="source-add-job",
        purpose="research_lab.source_add_provenance.v2",
        record_transport=lambda attempt: attempts.append(dict(attempt)),
        record_artifact=lambda digest: artifacts.append(str(digest)),
        attempts=attempts,
        artifacts=artifacts,
    )


def _authenticated_provider():
    calls = []

    def execute(request):
        calls.append(dict(request))
        if "/scrape?" in request["url"]:
            body = (
                "<html><title>Example API Reference</title>"
                "Quickstart endpoint authentication rate limit curl status code"
                "</html>"
            ).encode()
            content_type = "text/html"
        else:
            body = json.dumps(
                {
                    "markdown": (
                        "VERDICT: CREDIBLE. Example operates the official API and "
                        "maintains official documentation."
                    ),
                    "references": [
                        {"url": "https://example.com/developers", "title": "Official"},
                        {"url": "https://www.g2.com/products/example", "title": "G2"},
                    ],
                }
            ).encode()
            content_type = "application/json"
        response_hash = sha256_bytes(body)
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": content_type},
            "body_b64": base64.b64encode(body).decode(),
            "transport_attempt": {
                "terminal_status": "authenticated_response",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": response_hash,
                "response_hash": response_hash,
            },
        }

    return execute, calls


def test_source_add_provenance_uses_two_authenticated_provider_terminals():
    execute, calls = _authenticated_provider()
    context = _context()
    resolver = CoordinatorSourceAddProvenanceV2(
        execute_provider=execute,
        retry_policy_hash=HASH_A,
    )

    result = resolver.resolve(payload=_payload(), context=context)

    assert result["schema_version"] == SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION
    assert result["precheck_status"] == PRECHECK_PASSED
    assert len(calls) == 2
    assert all(call["provider_id"] == "scrapingdog" for call in calls)
    assert all("api_key" not in call["url"] for call in calls)
    assert len(context.attempts) == 2
    assert len(context.artifacts) == 5


def test_source_add_transport_failure_is_visible_manual_review():
    def execute(_request):
        return {
            "terminal_status": "transport_failure",
            "failure_code": "tls_error",
            "transport_attempt": {
                "terminal_status": "transport_failure",
                "failure_code": "tls_error",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": None,
            },
        }

    context = _context()
    resolver = CoordinatorSourceAddProvenanceV2(
        execute_provider=execute,
        retry_policy_hash=HASH_A,
    )
    result = resolver.resolve(payload=_payload(), context=context)

    assert result["precheck_status"] == PRECHECK_MANUAL
    assert "documentation_provider_error" in result["reasons"]
    assert len(context.attempts) == 2
    assert all(item["terminal_status"] == "transport_failure" for item in context.attempts)


@pytest.mark.asyncio
async def test_leg1_reward_requires_parent_output_and_exact_purpose():
    provenance = {
        "schema_version": SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
        "submission_id": "source_add_submission:1234567890abcdef",
        "precheck_status": PRECHECK_PASSED,
        "reasons": ["provenance_reference_backed"],
        "precheck_doc": {
            "precheck_status": PRECHECK_PASSED,
            "reasons": ["provenance_reference_backed"],
        },
    }
    root_hash = HASH_A
    graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "purpose": "research_lab.source_add_provenance.v2",
                "output_root": sha256_json(provenance),
            }
        ],
    }
    context = ExecutionContextV2(
        job_id="reward-job",
        purpose="research_lab.reward_decision.v2",
        epoch_id=10,
        parent_receipt_hashes=(root_hash,),
        external_receipt_graphs=[graph],
    )
    payload = {
        "decision_kind": "source_add_leg1",
        "decision_payload": {
            "adapter_id": "adapter:example",
            "miner_ref": "miner-hotkey",
            "start_epoch": 11,
            "existing_rewards": [],
            "alpha_percent": 1.0,
            "reward_epochs": 20,
            "provenance_result": provenance,
        },
    }
    executor = CoordinatorExecutorV2(
        reward_source_resolver=lambda measured_payload, _context: measured_payload
    )

    outcome = await executor(
        OP_RESEARCH_LAB_REWARD_DECISION,
        payload,
        context,
    )
    assert outcome.output["reward"]["leg"] == 1

    context.external_receipt_graphs[0]["receipts"][0]["output_root"] = HASH_B
    with pytest.raises(ValueError, match="parent output"):
        await executor(OP_RESEARCH_LAB_REWARD_DECISION, payload, context)


def test_champion_reward_requires_exact_promotion_output():
    promotion_decision = {
        "status": "promotion_passed",
        "improvement_points": 2.5,
        "threshold_points": 1.0,
        "candidate_kind": "image_build",
        "auto_promotion_enabled": True,
        "active_parent_matches": True,
        "metric_rejection_status": None,
    }
    root_hash = HASH_A
    graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "purpose": "research_lab.promotion_decision.v2",
                "output_root": sha256_json({"decision": promotion_decision}),
            }
        ],
    }
    context = ExecutionContextV2(
        job_id="champion-reward-job",
        purpose="research_lab.reward_decision.v2",
        epoch_id=10,
        parent_receipt_hashes=(root_hash,),
        external_receipt_graphs=[graph],
    )
    payload = {
        "decision_kind": "champion",
        "decision_payload": {
            "promotion_decision": promotion_decision,
        },
    }

    CoordinatorExecutorV2._validate_reward_ancestry(payload, context)
    context.external_receipt_graphs[0]["receipts"][0]["output_root"] = HASH_B
    with pytest.raises(ValueError, match="parent output"):
        CoordinatorExecutorV2._validate_reward_ancestry(payload, context)


def test_reimbursement_requires_exact_autoresearch_output():
    autoresearch_result = {
        "status": "completed",
        "actual_openrouter_cost_microusd": 1_250_000,
    }
    root_hash = HASH_A
    context = ExecutionContextV2(
        job_id="reimbursement-reward-job",
        purpose="research_lab.reward_decision.v2",
        epoch_id=10,
        parent_receipt_hashes=(root_hash,),
        external_receipt_graphs=[
            {
                "root_receipt_hash": root_hash,
                "receipts": [
                    {
                        "receipt_hash": root_hash,
                        "purpose": "research_lab.candidate_decision.v2",
                        "output_root": sha256_json(autoresearch_result),
                    }
                ],
            }
        ],
    )
    payload = {
        "decision_kind": "reimbursement",
        "decision_payload": {"autoresearch_result": autoresearch_result},
    }
    CoordinatorExecutorV2._validate_reward_ancestry(payload, context)
    payload["decision_payload"]["autoresearch_result"][
        "actual_openrouter_cost_microusd"
    ] = 9_999_999
    with pytest.raises(ValueError, match="parent output"):
        CoordinatorExecutorV2._validate_reward_ancestry(payload, context)
