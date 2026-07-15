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
    CoordinatorSourceAddFunctionalProbeV2,
    CoordinatorSourceAddProvenanceV2,
    SOURCE_ADD_FUNCTIONAL_PROBE_EVALUATOR_VERSION,
    SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION,
    SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.reward_executor_v2 import OP_RESEARCH_LAB_REWARD_DECISION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
CONFIG_REF = "source_add_probe_config:1234567890abcdef"


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


class _FunctionalReader:
    def read(self, *, policy_id, **_kwargs):
        if policy_id == "source_add_submission_by_id":
            return [
                {
                    "submission_id": "source_add_submission:1234567890abcdef",
                    "adapter_id": "adapter:example",
                    "miner_hotkey": "miner-hotkey",
                }
            ]
        if policy_id == "source_add_probe_config_by_submission":
            return [
                {
                    "submission_id": "source_add_submission:1234567890abcdef",
                    "adapter_id": "adapter:example",
                    "config_ref": CONFIG_REF,
                    "config_status": "active",
                    "probe_doc": {
                        "schema_version": "leadpoet.source_add_probe_config.v2",
                        "provider_id": "sourceadd_1234567890abcdef",
                        "base_url": "https://api.example.com/v1",
                        "auth_kind": "none",
                        "auth_name": "",
                        "request_headers": {},
                        "probes": [
                            {
                                "method": "GET",
                                "path": "/records",
                                "query": {"limit": 1},
                                "body_json": None,
                            }
                        ],
                    },
                    "credential_envelope": {},
                }
            ]
        raise AssertionError(policy_id)


def _functional_context():
    context = _context()
    context.purpose = "research_lab.source_add_functional_probe.v2"
    return context


def _functional_payload():
    return {
        "schema_version": "leadpoet.source_add_functional_probe_request.v2",
        "submission_id": "source_add_submission:1234567890abcdef",
        "config_ref": CONFIG_REF,
        "evaluation_mode": "functional_probe",
        "timeout_seconds": 30,
    }


def _authenticated_provider():
    calls = []

    def execute(request):
        calls.append(dict(request))
        if request["provider_id"] == "wayback":
            body = json.dumps({"archived_snapshots": {}}).encode()
            content_type = "application/json"
        elif "/scrape?" in request["url"]:
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


def test_source_add_provenance_uses_three_authenticated_provider_terminals():
    execute, calls = _authenticated_provider()
    context = _context()
    resolver = CoordinatorSourceAddProvenanceV2(
        execute_provider=execute,
        retry_policy_hash=HASH_A,
        wayback_retry_policy_hash=HASH_B,
    )

    result = resolver.resolve(payload=_payload(), context=context)

    assert result["schema_version"] == SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION
    assert result["precheck_status"] == PRECHECK_PASSED
    assert len(calls) == 3
    assert [call["provider_id"] for call in calls] == [
        "scrapingdog",
        "wayback",
        "scrapingdog",
    ]
    assert all("api_key" not in call["url"] for call in calls)
    assert len(context.attempts) == 3
    assert len(context.artifacts) == 7


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
        wayback_retry_policy_hash=HASH_B,
    )
    result = resolver.resolve(payload=_payload(), context=context)

    assert result["precheck_status"] == PRECHECK_MANUAL
    assert "documentation_provider_error" in result["reasons"]
    assert len(context.attempts) == 3
    assert all(item["terminal_status"] == "transport_failure" for item in context.attempts)


def test_functional_probe_uses_exact_dynamic_route_and_persists_hash_summary_only():
    calls = []
    body = b'{"records":[{"id":"one"}]}'

    def execute(request):
        calls.append(dict(request))
        response_hash = sha256_bytes(body)
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": {
                "terminal_status": "authenticated_response",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": HASH_B,
                "response_hash": response_hash,
            },
        }

    context = _functional_context()
    resolver = CoordinatorSourceAddFunctionalProbeV2(
        reader=_FunctionalReader(),
        execute_provider=execute,
    )
    result = resolver.resolve(payload=_functional_payload(), context=context)

    assert result["result_status"] == "passed"
    assert result["response_hash"] == sha256_bytes(body)
    assert result["byte_count"] == len(body)
    assert "records" not in str(result)
    assert calls[0]["url"] == "https://api.example.com/v1/records?limit=1"
    assert calls[0]["max_response_bytes"] == 1024 * 1024
    assert calls[0]["artifact_mode"] == "hash_only"
    assert calls[0]["dynamic_route"]["allowed_routes"] == [
        {"method": "GET", "path": "/v1/records"}
    ]
    assert len(context.attempts) == 1
    assert len(context.artifacts) == 3


@pytest.mark.parametrize(
    ("status", "content_type", "body", "expected_status", "reason"),
    [
        (302, "text/html", b"redirect", "failed", "redirect_forbidden"),
        (401, "application/json", b'{"error":"denied"}', "awaiting_operator", "operator_credential_or_headers_required"),
        (404, "application/json", b'{"error":"missing"}', "failed", "endpoint_not_found"),
        (408, "application/json", b'{"error":"timeout"}', "retryable", "http_408"),
        (429, "application/json", b'{"error":"slow"}', "retryable", "http_429"),
        (200, "text/html", b"<html>login</html>", "failed", "non_json_content_type"),
        (200, "application/json", b'{"error":"bad key"}', "failed", "json_error_envelope"),
    ],
)
def test_functional_probe_failure_policy(status, content_type, body, expected_status, reason):
    def execute(_request):
        response_hash = sha256_bytes(body)
        return {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "headers": {"content-type": content_type, "retry-after": "999999"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": {
                "terminal_status": "authenticated_response",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": HASH_B,
                "response_hash": response_hash,
            },
        }

    result = CoordinatorSourceAddFunctionalProbeV2(
        reader=_FunctionalReader(),
        execute_provider=execute,
    ).resolve(payload=_functional_payload(), context=_functional_context())

    assert result["result_status"] == expected_status
    assert reason in result["reason_codes"]
    assert result["retry_after_seconds"] <= 21_600


def test_functional_probe_rejects_deep_or_empty_json():
    deep = {}
    cursor = deep
    for _ in range(14):
        cursor["next"] = {}
        cursor = cursor["next"]

    assert CoordinatorSourceAddFunctionalProbeV2._validate_json_response(
        response_body=b"{}",
        content_type="application/json",
    ) == ("failed", ["empty_json_payload"])
    status, reasons = CoordinatorSourceAddFunctionalProbeV2._validate_json_response(
        response_body=json.dumps(deep).encode("utf-8"),
        content_type="application/json",
    )
    assert status == "failed"
    assert reasons == ["json_depth_exceeded"]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b'{"message":"Unauthorized"}', "json_auth_or_login_envelope"),
        (b'{"status":"failed"}', "json_error_status"),
        (b'{"status":"ok","version":"1.2"}', "json_non_data_envelope"),
        (b'{"records":[],"total":0}', "empty_json_data"),
        (b'[{"message":"Unauthorized"}]', "json_error_envelope"),
        (b'["Unauthorized"]', "json_auth_or_login_envelope"),
        (b'[{}]', "empty_json_data"),
    ],
)
def test_functional_probe_rejects_json_without_usable_data(body, reason):
    status, reasons = CoordinatorSourceAddFunctionalProbeV2._validate_json_response(
        response_body=body,
        content_type="application/json",
    )

    assert status == "failed"
    assert reasons == [reason]


def test_functional_probe_health_endpoint_cannot_be_sole_pass():
    class _HealthReader(_FunctionalReader):
        def read(self, *, policy_id, **kwargs):
            rows = super().read(policy_id=policy_id, **kwargs)
            if policy_id == "source_add_probe_config_by_submission":
                rows[0]["probe_doc"]["probes"] = [
                    {
                        "method": "GET",
                        "path": "/health",
                        "query": {},
                        "body_json": None,
                    }
                ]
            return rows

    body = b'{"service":"example","records":[{"status":"ready"}]}'

    def execute(_request):
        response_hash = sha256_bytes(body)
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": {
                "terminal_status": "authenticated_response",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": HASH_B,
                "response_hash": response_hash,
            },
        }

    result = CoordinatorSourceAddFunctionalProbeV2(
        reader=_HealthReader(),
        execute_provider=execute,
    ).resolve(payload=_functional_payload(), context=_functional_context())

    assert result["result_status"] == "manual_review"
    assert result["reason_codes"] == ["non_data_probe_endpoint"]


def test_functional_probe_can_skip_health_and_pass_later_data_endpoint():
    class _HealthThenDataReader(_FunctionalReader):
        def read(self, *, policy_id, **kwargs):
            rows = super().read(policy_id=policy_id, **kwargs)
            if policy_id == "source_add_probe_config_by_submission":
                rows[0]["probe_doc"]["probes"] = [
                    {
                        "method": "GET",
                        "path": "/health",
                        "query": {},
                        "body_json": None,
                    },
                    {
                        "method": "GET",
                        "path": "/records",
                        "query": {},
                        "body_json": None,
                    },
                ]
            return rows

    body = b'{"records":[{"id":"one"}]}'

    def execute(_request):
        response_hash = sha256_bytes(body)
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": {
                "terminal_status": "authenticated_response",
                "request_artifact_hash": HASH_A,
                "response_artifact_hash": HASH_B,
                "response_hash": response_hash,
            },
        }

    result = CoordinatorSourceAddFunctionalProbeV2(
        reader=_HealthThenDataReader(),
        execute_provider=execute,
    ).resolve(payload=_functional_payload(), context=_functional_context())

    assert result["result_status"] == "passed"
    assert result["selected_probe_index"] == 1
    assert result["probe_summaries"][0]["result_status"] == "manual_review"


@pytest.mark.asyncio
async def test_leg1_reward_requires_parent_output_and_exact_purpose():
    functional = {
        "schema_version": SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION,
        "evaluator_version": SOURCE_ADD_FUNCTIONAL_PROBE_EVALUATOR_VERSION,
        "submission_id": "source_add_submission:1234567890abcdef",
        "adapter_id": "adapter:example",
        "config_ref": "source_add_probe_config:1234567890abcdef",
        "evaluation_mode": "functional_probe",
        "result_status": "passed",
        "route_hash": HASH_B,
        "selected_probe_index": 0,
        "response_hash": HASH_A,
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 20,
        "duration_ms": 10,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "probe_summaries": [],
    }
    root_hash = HASH_A
    graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "purpose": "research_lab.source_add_functional_probe.v2",
                "output_root": sha256_json(functional),
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
            "functional_probe_result": functional,
            "trigger_evidence": {
                "functional_probe_passed": True,
                "functional_probe_result_hash": sha256_json(functional),
            },
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
