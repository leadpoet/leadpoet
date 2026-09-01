from __future__ import annotations

import base64
from dataclasses import asdict
import httpx
import pytest
import time

from gateway.research_lab.config import DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG
from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    PARENT_RECEIPT_GRAPHS_FIELD,
    ExecutionContextV2,
    ExecutionJobManagerV2,
)
from gateway.tee.scoring_executor import (
    OP_BENCHMARK_ICP_SCORE,
    OP_QUALIFICATION_COMPANY_SCORES,
    execute_scoring_operation,
)
from gateway.tee.scoring_executor_v2 import (
    DEV_HYBRID_REQUEST_SCHEMA_VERSION,
    DEV_REPLAY_REQUEST_SCHEMA_VERSION,
    MODEL_COMPATIBILITY_PURPOSE_V2,
    OP_DEV_HYBRID_V2,
    OP_DEV_REPLAY_V2,
    OP_PROVIDER_PREFLIGHT_V2,
    OP_RUN_MODEL_SANDBOX_V2,
    OP_SOURCE_ADD_LEG2_JUDGE_V2,
    OP_ATTEST_ROUTING_EXPERIMENT_V2,
    OP_ATTEST_ROUTING_PROVIDER_CALL_V2,
    OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2,
    OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2,
    PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
    ScoringExecutorV2,
)
from gateway.research_lab.routing_execution_authorization import (
    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    build_routing_budget_reservation_v3,
)
from research_lab.routing_experiments import ProviderBindingIdentity
from gateway.research_lab.routing_experiment_attestation import (
    ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
    build_routing_experiment_attestation_input_v2,
    routing_experiment_attestation_receipt_output_v2,
)
from gateway.tee.model_sandbox_v2 import provider_evidence_tape_input_root
from gateway.tee.provider_client_v2 import ProviderClientV2Error
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_receipt_graph,
    build_transport_attempt,
    canonical_json,
    create_boot_identity,
    sha256_bytes,
    sha256_json,
    transport_root,
)
from research_lab.eval import build_local_private_artifact_manifest
from research_lab.eval.dev_eval import compute_dev_set_hash, evaluate_dev
from research_lab.eval.private_runtime import canonicalize_private_model_icp
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    icp_evidence_cache_key,
)
from research_lab.eval.snapshot_store import MODE_RECORD, MODE_REPLAY, ProviderSnapshotStore
from tests.v2_epoch_test_utils import epoch_test_environment
from tests.test_sourcing_model_semantic_compatibility_v1 import (
    _install_future_tree,
)
from tests.test_routing_provider_terminal_protected import _call_fixture
from tests.routing_experiment_authority_fixture import authority_fixture


HASH = "sha256:" + "a" * 64


def _transport_failure_result(request, *, request_id: str = "1" * 32):
    attempt = build_transport_attempt(
        request_id=request_id,
        logical_operation_id=request["logical_operation_id"],
        job_id=request["job_id"],
        purpose=request["purpose"],
        provider_id=request["provider_id"],
        attempt_number=request["attempt_number"],
        method=request["method"],
        destination_host="example.com",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH,
        credential_ref_hash=HASH,
        egress_proxy_ref_hash=HASH,
        retry_policy_hash=HASH,
        timeout_ms=request["timeout_ms"],
        started_at="2026-07-10T20:00:00Z",
        terminal_status="transport_failure",
        http_status=None,
        response_hash=None,
        request_artifact_hash=HASH,
        response_artifact_hash=None,
        tls_peer_chain_hash=None,
        tls_protocol=None,
        failure_code="connection_reset",
        completed_at="2026-07-10T20:00:01Z",
    )
    return {
        "terminal_status": "transport_failure",
        "http_status": None,
        "headers": {},
        "body_b64": "",
        "failure_code": "connection_reset",
        "encrypted_request_artifact_id": HASH,
        "transport_attempt": attempt,
    }


@pytest.fixture(autouse=True)
def _official_epoch_authority(monkeypatch):
    for name, value in epoch_test_environment().items():
        monkeypatch.setenv(name, value)


def _model_catalog_evidence():
    runtime_catalog = build_source_add_runtime_catalog_v2([])
    result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "provisioned_sources_hash": sha256_json([]),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": runtime_catalog,
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    root_hash = "sha256:" + "c" * 64
    evidence = {"result": result, "root_receipt_hash": root_hash}
    graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.source_add_catalog_snapshot.v2",
                "status": "succeeded",
                "output_root": sha256_json(result),
            }
        ],
    }
    return runtime_catalog, evidence, graph


def _seal_artifact(*, plaintext, job_id, purpose, artifact_kind):
    identity = sha256_json(
        {
            "job_id": job_id,
            "purpose": purpose,
            "artifact_kind": artifact_kind,
            "plaintext_hash": sha256_bytes(plaintext),
        }
    )
    return {
        "status": "sealed",
        "artifact_id": identity,
        "plaintext_hash": sha256_bytes(plaintext),
        "ciphertext_hash": sha256_json({"ciphertext": identity}),
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "purpose": purpose,
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T00:00:00Z",
        "encryption_context_hash": sha256_json({"aad": identity}),
        "persisted": False,
    }


def test_scoring_executor_installs_openrouter_broker_sentinels(monkeypatch):
    import os
    from qualification.scoring.intent_verification_three_stage import (
        _get_openrouter_key,
    )

    names = (
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("no request expected"),
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        for name in names:
            assert os.environ[name] == "leadpoet-v2-brokered-credential"
        assert _get_openrouter_key() == "leadpoet-v2-brokered-credential"
    finally:
        executor.close()


@pytest.mark.asyncio
async def test_company_scoring_preserves_handled_signed_transport_failure(
    monkeypatch,
):
    from research_lab.eval import evaluator

    class HandledEvidenceFailureScorer:
        def __init__(self, **_kwargs):
            pass

        async def score_with_breakdowns(
            self,
            _companies,
            _icp,
            _is_reference_model,
        ):
            try:
                async with httpx.AsyncClient() as client:
                    await client.get("https://example.com/evidence")
            except httpx.TransportError:
                return [
                    {
                        "final_score": 0.0,
                        "failure_reason": (
                            "Company verification failed: website unreachable: "
                            "connection reset"
                        ),
                    }
                ]
            raise AssertionError("transport failure was not delivered to the scorer")

    monkeypatch.setattr(
        evaluator,
        "QualificationStyleCompanyScorer",
        HandledEvidenceFailureScorer,
    )
    context = ExecutionContextV2(
        job_id="company-score-handled-transport",
        purpose="research_lab.company_score.v2",
        epoch_id=1,
    )
    executor = ScoringExecutorV2(
        provider_execute=_transport_failure_result,
        retry_policy_hashes={"public_web": HASH},
    )
    try:
        result = await executor(
            OP_QUALIFICATION_COMPANY_SCORES,
            {
                "companies": [{"company_name": "Example"}],
                "icp": {"industry": "Software"},
                "is_reference_model": True,
                "scoring_adapter_version": "qualification-company-scorer:v1",
                "provider_execution_mode": "live_enclave",
            },
            context,
        )
    finally:
        executor.close()

    assert result.output["scores"] == [0.0]
    assert result.output["breakdowns"][0]["failure_reason"] == (
        "Company verification failed: website unreachable: connection reset"
    )
    assert evaluator.scorer_breakdown_has_retryable_infrastructure_failure(
        result.output["breakdowns"][0]
    )
    assert len(context.transport_attempts) == 1
    assert context.transport_attempts[0]["terminal_status"] == (
        "transport_failure"
    )


@pytest.mark.asyncio
async def test_non_company_scoring_still_rejects_handled_transport_failure(
    monkeypatch,
):
    from gateway.tee import scoring_executor_v2

    async def handled_transport_failure(_operation, _payload):
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://example.com/evidence")
        except httpx.TransportError:
            return {"score": 0.0}
        raise AssertionError("transport failure was not delivered to the operation")

    monkeypatch.setattr(
        scoring_executor_v2,
        "execute_scoring_operation",
        handled_transport_failure,
    )
    context = ExecutionContextV2(
        job_id="benchmark-score-handled-transport",
        purpose="research_lab.benchmark.v2",
        epoch_id=1,
    )
    executor = ScoringExecutorV2(
        provider_execute=_transport_failure_result,
        retry_policy_hashes={"public_web": HASH},
    )
    try:
        with pytest.raises(
            ProviderClientV2Error,
            match="provider transport did not authenticate",
        ):
            await executor(
                OP_BENCHMARK_ICP_SCORE,
                {"scores": [0.0]},
                context,
            )
    finally:
        executor.close()

    assert len(context.transport_attempts) == 1
    assert context.transport_attempts[0]["terminal_status"] == (
        "transport_failure"
    )


@pytest.mark.asyncio
async def test_v2_preflight_reuses_existing_cache_and_failure_streak_logic(
    monkeypatch,
):
    from gateway.research_lab import provider_preflight

    calls = {"exa": 0, "scrapingdog": 0}

    def _healthy(provider):
        def probe(_timeout=None):
            calls[provider] += 1
            return provider_preflight.ProviderVerdict(
                provider=provider,
                healthy=True,
                status="healthy",
            )

        return probe

    monkeypatch.setitem(provider_preflight._PROBES, "exa", _healthy("exa"))
    monkeypatch.setitem(
        provider_preflight._PROBES,
        "scrapingdog",
        _healthy("scrapingdog"),
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("healthy stub must not call network"),
        retry_policy_hashes={"exa": HASH, "scrapingdog": HASH},
    )
    payload = {
        "_v2_provider_credential_profile": "provider_preflight",
        "_v2_provider_credential_ref_hashes": {
            "exa": HASH,
            "scrapingdog": HASH,
        },
        "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "measurement_id": "1" * 32,
        "scope_key": "scoring:worker-1",
        "force": False,
        "settings": {
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
    }
    try:
        for job_id in ("preflight-job-1", "preflight-job-2"):
            result = await executor(
                OP_PROVIDER_PREFLIGHT_V2,
                payload,
                ExecutionContextV2(
                    job_id=job_id,
                    purpose="research_lab.provider_preflight.v2",
                    epoch_id=0,
                    provider_credential_profile="provider_preflight",
                    provider_credential_ref_hashes={
                        "exa": HASH,
                        "scrapingdog": HASH,
                    },
                ),
            )
            assert result.output["healthy"] is True
    finally:
        executor.close()
    assert calls == {"exa": 1, "scrapingdog": 1}


@pytest.mark.asyncio
async def test_v2_preflight_authenticates_both_providers_and_worker_proxy():
    provider_refs = {
        "exa": "sha256:" + "b" * 64,
        "scrapingdog": "sha256:" + "c" * 64,
        "egress_proxy": "sha256:" + "d" * 64,
    }
    observed = []

    def _provider_execute(request):
        observed.append(dict(request))
        provider_id = request["provider_id"]
        host = {
            "exa": "api.exa.ai",
            "scrapingdog": "api.scrapingdog.com",
        }[provider_id]
        body = b"{}"
        attempt = build_transport_attempt(
            request_id=("%032x" % (len(observed))),
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=provider_id,
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host=host,
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=provider_refs[provider_id],
            egress_proxy_ref_hash=provider_refs["egress_proxy"],
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        outcome_attempt = build_transport_attempt(
            request_id=("%032x" % (100 + len(observed))),
            logical_operation_id=(
                "%s:provider-outcome:%d:append"
                % (request["job_id"], len(observed))
            ),
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="supabase",
            attempt_number=0,
            method="POST",
            destination_host="qplwoislplkcegvdmbim.supabase.co",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash="sha256:" + "e" * 64,
            egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
            "additional_transport_attempts": [outcome_attempt],
        }

    executor = ScoringExecutorV2(
        provider_execute=_provider_execute,
        retry_policy_hashes={"exa": HASH, "scrapingdog": HASH},
    )
    payload = {
        "_v2_provider_credential_profile": "provider_preflight",
        "_v2_provider_credential_ref_hashes": provider_refs,
        "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "measurement_id": "2" * 32,
        "scope_key": "scoring:worker-2",
        "force": True,
        "settings": {
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
    }
    context = ExecutionContextV2(
        job_id="provider-preflight-job",
        purpose="research_lab.provider_preflight.v2",
        epoch_id=0,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes=provider_refs,
    )
    try:
        result = await executor(
            OP_PROVIDER_PREFLIGHT_V2,
            payload,
            context,
        )
    finally:
        executor.close()

    assert result.output["healthy"] is True
    assert {
        item["provider"] for item in result.output["verdicts"]
    } == {"exa", "scrapingdog"}
    assert {item["provider_id"] for item in context.transport_attempts} == {
        "exa",
        "scrapingdog",
        "supabase",
    }
    assert len(context.transport_attempts) == 4
    assert all(
        item["egress_proxy_ref_hash"] == provider_refs["egress_proxy"]
        for item in context.transport_attempts
        if item["provider_id"] != "supabase"
    )
    assert all(
        item["egress_proxy_ref_hash"] == DIRECT_EGRESS_REF_HASH
        for item in context.transport_attempts
        if item["provider_id"] == "supabase"
    )


@pytest.mark.asyncio
async def test_v2_preflight_transport_scope_failure_does_not_poison_cache():
    provider_refs = {
        "exa": "sha256:" + "b" * 64,
        "scrapingdog": "sha256:" + "c" * 64,
        "egress_proxy": "sha256:" + "d" * 64,
    }
    state = {"fail": True, "calls": 0}

    def provider_execute(request):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("coordinator relay failed before terminal")
        provider_id = request["provider_id"]
        body = b"{}"
        attempt = build_transport_attempt(
            request_id="%032x" % state["calls"],
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=provider_id,
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host={
                "exa": "api.exa.ai",
                "scrapingdog": "api.scrapingdog.com",
            }[provider_id],
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=provider_refs[provider_id],
            egress_proxy_ref_hash=provider_refs["egress_proxy"],
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
        }

    executor = ScoringExecutorV2(
        provider_execute=provider_execute,
        retry_policy_hashes={"exa": HASH, "scrapingdog": HASH},
    )
    payload = {
        "_v2_provider_credential_profile": "provider_preflight",
        "_v2_provider_credential_ref_hashes": provider_refs,
        "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "measurement_id": "3" * 32,
        "scope_key": "scoring:worker-3",
        "force": False,
        "settings": {
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
    }
    try:
        with pytest.raises(ProviderClientV2Error, match="signed terminal"):
            await executor(
                OP_PROVIDER_PREFLIGHT_V2,
                payload,
                ExecutionContextV2(
                    job_id="provider-preflight-failed-scope",
                    purpose="research_lab.provider_preflight.v2",
                    epoch_id=0,
                    provider_credential_profile="provider_preflight",
                    provider_credential_ref_hashes=provider_refs,
                ),
            )
        failed_call_count = state["calls"]
        state["fail"] = False
        result = await executor(
            OP_PROVIDER_PREFLIGHT_V2,
            {**payload, "measurement_id": "4" * 32},
            ExecutionContextV2(
                job_id="provider-preflight-recovered-scope",
                purpose="research_lab.provider_preflight.v2",
                epoch_id=0,
                provider_credential_profile="provider_preflight",
                provider_credential_ref_hashes=provider_refs,
            ),
        )
    finally:
        executor.close()

    assert result.output["healthy"] is True
    assert state["calls"] == failed_call_count + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("measurement_id", ("", "1" * 31, "g" * 32))
async def test_v2_preflight_rejects_invalid_measurement_identity(
    measurement_id,
):
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "invalid preflight request must not call a provider"
        ),
        retry_policy_hashes={"exa": HASH, "scrapingdog": HASH},
    )
    payload = {
        "_v2_provider_credential_profile": "provider_preflight",
        "_v2_provider_credential_ref_hashes": {
            "exa": HASH,
            "scrapingdog": HASH,
        },
        "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "measurement_id": measurement_id,
        "scope_key": "scoring:worker-1",
        "force": False,
        "settings": {
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
    }
    try:
        with pytest.raises(
            ValueError,
            match="measurement identity",
        ):
            await executor(
                OP_PROVIDER_PREFLIGHT_V2,
                payload,
                ExecutionContextV2(
                    job_id="invalid-preflight-job",
                    purpose="research_lab.provider_preflight.v2",
                    epoch_id=0,
                    provider_credential_profile="provider_preflight",
                    provider_credential_ref_hashes={
                        "exa": HASH,
                        "scrapingdog": HASH,
                    },
                ),
            )
    finally:
        executor.close()


@pytest.mark.asyncio
async def test_v2_adapter_calls_exact_existing_pure_scoring_function():
    payload = {"scores": [100.0, 80.0, 60.0, 40.0, 20.0, 1.0]}
    expected = await execute_scoring_operation(OP_BENCHMARK_ICP_SCORE, payload)
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"public_web": HASH},
    )
    try:
        result = await executor(
            OP_BENCHMARK_ICP_SCORE,
            payload,
            ExecutionContextV2(
                job_id="score-job-1",
                purpose="research_lab.benchmark.v2",
                epoch_id=24000,
            ),
        )
    finally:
        executor.close()
    assert dict(result.output) == dict(expected)


@pytest.mark.asyncio
async def test_v2_adapter_routes_provider_call_and_collects_terminal(monkeypatch):
    observed = []

    def _provider_execute(request):
        observed.append(request)
        body = b'{"ok":true}'
        attempt = build_transport_attempt(
            request_id="1" * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
        }

    async def _existing_scoring(_operation, _payload):
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "model-1"},
            )
        return {"provider_result": response.json()}

    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.execute_scoring_operation",
        _existing_scoring,
    )
    context = ExecutionContextV2(
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24000,
    )
    executor = ScoringExecutorV2(
        provider_execute=_provider_execute,
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        result = await executor(
            "qualification_company_scores",
            {},
            context,
        )
    finally:
        executor.close()
    assert result.output == {"provider_result": {"ok": True}}
    assert len(observed) == 1
    assert len(context.transport_attempts) == 1


@pytest.mark.asyncio
async def test_source_add_judge_runs_existing_logic_inside_measured_transport(monkeypatch):
    from gateway.research_lab.source_add_llm_judge import SourceAddJudgeVerdict

    observed = []

    def _provider_execute(request):
        observed.append(dict(request))
        body = b'{"choices":[{"message":{"content":"{}"}}]}'
        attempt = build_transport_attempt(
            request_id="2" * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="openrouter",
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
        }

    async def _judge(**kwargs):
        assert kwargs["api_key"] == "leadpoet-v2-brokered-credential"
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "openai/gpt-5.6-sol"},
            )
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test",
            registry_provider_id="test",
            evidence_summary="matched",
            reason_codes=("matched_api_usage",),
            model_id="openai/gpt-5.6-sol",
            provider_usage={"cost_usd": 0.01},
            raw_doc={"verdict": "helped"},
        )

    monkeypatch.setattr(
        "gateway.research_lab.source_add_llm_judge.judge_source_add_implementation",
        _judge,
    )
    provisioned_sources = [{"adapter_id": "adapter:test"}]
    catalog_output = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": provisioned_sources,
        "provisioned_sources_hash": sha256_json(provisioned_sources),
    }
    catalog_receipt_hash = "sha256:" + "b" * 64
    context = ExecutionContextV2(
        job_id="source-add-judge-1",
        purpose="research_lab.source_add_judge.v2",
        epoch_id=24000,
        parent_receipt_hashes=(catalog_receipt_hash,),
        provider_credential_profile="source_add_judge",
        provider_credential_ref_hashes={"openrouter": HASH},
        external_receipt_graphs=[
            {
                "root_receipt_hash": catalog_receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": catalog_receipt_hash,
                        "purpose": "research_lab.source_add_catalog_snapshot.v2",
                        "output_root": sha256_json(catalog_output),
                    }
                ],
            }
        ],
    )
    executor = ScoringExecutorV2(
        provider_execute=_provider_execute,
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        result = await executor(
            OP_SOURCE_ADD_LEG2_JUDGE_V2,
            {
                "_v2_provider_credential_profile": "source_add_judge",
                "_v2_provider_credential_ref_hashes": {"openrouter": HASH},
                "schema_version": SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
                "candidate": {"candidate_id": "candidate:1"},
                "score_bundle": {"score_bundle_hash": HASH},
                "provisioned_sources": provisioned_sources,
                "timeout_seconds": 180,
            },
            context,
        )
    finally:
        executor.close()

    assert result.output["verdict"]["verdict"] == "helped"
    assert result.output["verdict"]["judge_doc_hash"].startswith("sha256:")
    assert len(observed) == 1
    assert observed[0]["purpose"] == "research_lab.source_add_judge.v2"
    assert len(context.transport_attempts) == 1


@pytest.mark.asyncio
async def test_v2_adapter_routes_model_jobs_through_measured_sandbox():
    observed = {}
    runtime_catalog, catalog_evidence, catalog_graph = _model_catalog_evidence()
    commitments = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "model_manifest_hash": "sha256:" + "2" * 64,
        "source_bundle_hash": "sha256:" + "3" * 64,
        "compatibility_policy_hash": "sha256:" + "a" * 64,
        "compatibility_admission_hash": "sha256:" + "b" * 64,
        "runtime_config_hash": "sha256:" + "4" * 64,
        "input_hash": "sha256:" + "5" * 64,
        "provider_evidence_cache_hash": "sha256:" + "6" * 64,
        "provider_snapshot_archive_hash": sha256_json({}),
        "provider_snapshot_tree_hash": sha256_json({}),
        "provider_snapshot_manifest_hash": sha256_json({}),
        "provider_runtime_catalog_hash": runtime_catalog["catalog_hash"],
        "generated_provider_evidence_cache_hash": "sha256:" + "7" * 64,
        "trace_entries_hash": "sha256:" + "8" * 64,
        "output_hash": "sha256:" + "9" * 64,
    }

    class _Sandbox:
        def execute(self, payload, **kwargs):
            observed["payload"] = dict(payload)
            observed["kwargs"] = dict(kwargs)
            return {"output": {"companies": []}, **commitments}

    context = ExecutionContextV2(
        job_id="model-job-1",
        purpose="research_lab.private_model_run.v2",
        epoch_id=24000,
        external_ancestry_proofs=[
            {"disclosed_receipts": list(catalog_graph["receipts"])}
        ],
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=_Sandbox(),
        artifact_seal=_seal_artifact,
    )
    try:
        result = await executor(
            OP_RUN_MODEL_SANDBOX_V2,
            {
                "model_kind": "private",
                "environment": {},
                "provider_evidence_cache": {},
                "provider_evidence_cache_ref": "",
                "provider_evidence_mode": "live",
                "provider_snapshot_bundle": {},
                "provider_snapshot_tree_hash": "",
                "provider_snapshot_manifest_hash": "",
                "provider_cost_scope": HASH,
                "provider_cost_cap_microusd": 0,
                "provider_call_cap": 0,
                "provider_runtime_catalog": runtime_catalog,
                "provider_catalog_evidence": catalog_evidence,
            },
            context,
        )
    finally:
        executor.close()

    assert observed["payload"] == {
        "model_kind": "private",
        "environment": {},
        "provider_evidence_cache": {},
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "live",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": HASH,
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": runtime_catalog,
        "provider_catalog_evidence": catalog_evidence,
    }
    assert observed["kwargs"]["job_id"] == "model-job-1"
    assert observed["kwargs"]["purpose"] == "research_lab.private_model_run.v2"
    assert observed["kwargs"]["retry_policy_hashes"] == {"openrouter": HASH}
    assert observed["kwargs"]["terminal_sink"] == context.record_transport
    assert result.output["output"] == {"companies": []}
    assert result.artifact_hashes[: len(commitments)] == tuple(commitments.values())
    assert [item["artifact_kind"] for item in result.output["sealed_artifacts"]] == [
        "model_output",
        "model_trace",
    ]


def _metadata_compatibility_payload() -> dict:
    return {
        "model_kind": "private",
        "operation": "metadata",
        "input": {},
        "environment": {},
        "provider_evidence_cache": {},
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": "",
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": {},
        "provider_catalog_evidence": {},
    }


@pytest.mark.asyncio
async def test_model_metadata_uses_dedicated_zero_credential_authority(
    monkeypatch,
):
    observed = {}
    commitments = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "model_manifest_hash": "sha256:" + "2" * 64,
        "source_bundle_hash": "sha256:" + "3" * 64,
        "compatibility_policy_hash": "sha256:" + "4" * 64,
        "compatibility_admission_hash": "sha256:" + "5" * 64,
        "runtime_config_hash": "sha256:" + "6" * 64,
        "input_hash": sha256_json({}),
        "provider_evidence_cache_hash": sha256_json({}),
        "provider_snapshot_archive_hash": sha256_json({}),
        "provider_snapshot_tree_hash": sha256_json({}),
        "provider_snapshot_manifest_hash": sha256_json({}),
        "provider_runtime_catalog_hash": sha256_json({}),
        "generated_provider_evidence_cache_hash": sha256_json({}),
        "trace_entries_hash": sha256_json([]),
        "output_hash": sha256_json({"version": "measured"}),
    }

    class _Sandbox:
        def execute(self, payload, **kwargs):
            observed["payload"] = dict(payload)
            observed["kwargs"] = dict(kwargs)
            return {
                **commitments,
                "output": {"version": "measured"},
                "trace_entries": [],
                "generated_provider_evidence_cache": {},
            }

    context = ExecutionContextV2(
        job_id="model-metadata-compatibility",
        purpose=MODEL_COMPATIBILITY_PURPOSE_V2,
        epoch_id=24000,
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "metadata compatibility must not call a provider"
        ),
        retry_policy_hashes={},
        model_sandbox=_Sandbox(),
        artifact_seal=lambda **_kwargs: pytest.fail(
            "metadata compatibility must not seal model artifacts"
        ),
    )
    monkeypatch.setattr(
        executor,
        "_validate_model_provider_catalog_ancestry",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata compatibility must not validate provider ancestry"
        ),
    )
    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.validate_model_sandbox_environment",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata compatibility must not derive provider environment"
        ),
    )
    try:
        result = await executor(
            OP_RUN_MODEL_SANDBOX_V2,
            _metadata_compatibility_payload(),
            context,
        )
    finally:
        executor.close()

    assert observed["payload"] == _metadata_compatibility_payload()
    assert observed["kwargs"]["purpose"] == MODEL_COMPATIBILITY_PURPOSE_V2
    assert context.provider_credential_ref_hashes == {}
    assert result.output["trace_entries"] == []
    assert result.output["sealed_artifacts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operation", "run_icp"),
        ("input", {"unexpected": True}),
        ("environment", {"OPENROUTER_API_KEY": "must-not-cross"}),
        ("provider_evidence_mode", "live"),
        ("provider_cost_scope", HASH),
        ("provider_runtime_catalog", {"catalog_hash": HASH}),
        ("provider_catalog_evidence", {"result": {}}),
    ),
)
async def test_model_metadata_authority_rejects_provider_state(field, value):
    calls = []

    class _Sandbox:
        def execute(self, payload, **kwargs):
            calls.append((dict(payload), dict(kwargs)))
            return {}

    context = ExecutionContextV2(
        job_id="model-metadata-leak",
        purpose=MODEL_COMPATIBILITY_PURPOSE_V2,
        epoch_id=24000,
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={},
        model_sandbox=_Sandbox(),
        artifact_seal=_seal_artifact,
    )
    payload = _metadata_compatibility_payload()
    payload[field] = value
    try:
        with pytest.raises(
            ValueError,
            match="metadata authority is not isolated",
        ):
            await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
    finally:
        executor.close()
    assert calls == []


@pytest.mark.asyncio
async def test_model_metadata_cannot_use_ordinary_model_purpose():
    context = ExecutionContextV2(
        job_id="model-metadata-wrong-purpose",
        purpose="research_lab.private_model_run.v2",
        epoch_id=24000,
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={},
        model_sandbox=object(),
        artifact_seal=_seal_artifact,
    )
    try:
        with pytest.raises(
            ValueError,
            match="metadata requires compatibility authority",
        ):
            await executor(
                OP_RUN_MODEL_SANDBOX_V2,
                _metadata_compatibility_payload(),
                context,
            )
    finally:
        executor.close()


@pytest.mark.asyncio
async def test_v2_model_cache_requires_exact_tape_ancestry():
    runtime_catalog, catalog_evidence, catalog_graph = _model_catalog_evidence()
    cache_ref = "a" * 64
    cache_doc = {
        "schema_version": "1.1",
        "rolling_window_hash": "",
        "icp_ref": cache_ref,
        "utc_day": "2026-07-10",
        "entries": {},
    }
    cache_hash = sha256_json(cache_doc)
    calls = []

    class _Sandbox:
        def execute(self, payload, **kwargs):
            calls.append((dict(payload), dict(kwargs)))
            return {
                "model_artifact_hash": HASH,
                "model_manifest_hash": HASH,
                    "source_bundle_hash": HASH,
                    "compatibility_policy_hash": "sha256:" + "a" * 64,
                    "compatibility_admission_hash": "sha256:" + "b" * 64,
                "runtime_config_hash": HASH,
                "input_hash": HASH,
                "provider_evidence_cache_hash": cache_hash,
                "provider_snapshot_archive_hash": sha256_json({}),
                "provider_snapshot_tree_hash": sha256_json({}),
                "provider_snapshot_manifest_hash": sha256_json({}),
                "provider_runtime_catalog_hash": runtime_catalog["catalog_hash"],
                "generated_provider_evidence_cache_hash": sha256_json({}),
                "trace_entries_hash": HASH,
                "output_hash": HASH,
                "output": [],
                "generated_provider_evidence_cache": {},
            }

    context = ExecutionContextV2(
        job_id="model-job-cache",
        purpose="research_lab.candidate_model_run.v2",
        epoch_id=24000,
        external_receipt_graphs=[catalog_graph],
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=_Sandbox(),
        artifact_seal=_seal_artifact,
    )
    payload = {
        "model_kind": "candidate",
        "environment": {},
        "provider_evidence_cache": cache_doc,
        "provider_evidence_cache_ref": cache_ref,
        "provider_evidence_mode": "cache_live",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": HASH,
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": runtime_catalog,
        "provider_catalog_evidence": catalog_evidence,
    }
    try:
        with pytest.raises(ValueError, match="measured tape ancestry"):
            await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
        tape_receipt = {
            "receipt_hash": "sha256:" + "d" * 64,
            "role": "gateway_scoring",
            "purpose": "research_lab.provider_evidence_tape.v2",
            "status": "succeeded",
            "input_root": provider_evidence_tape_input_root(
                cache_ref,
                cache_hash,
            ),
            "output_root": cache_hash,
        }
        context.external_receipt_graphs = []
        context.external_ancestry_proofs = [
            {"disclosed_receipts": [*catalog_graph["receipts"], tape_receipt]},
            # Overlapping compact authorities may disclose the same signed
            # receipt. It remains one authority, not an ambiguity.
            {"disclosed_receipts": [tape_receipt]},
        ]
        result = await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
        assert result.output["output"] == []
        context.external_ancestry_proofs.append(
            {
                "disclosed_receipts": [
                    {**tape_receipt, "receipt_hash": "sha256:" + "e" * 64}
                ]
            }
        )
        with pytest.raises(ValueError, match="measured tape ancestry"):
            await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
        context.external_ancestry_proofs.pop()
        changed = {**cache_doc, "entries": {"f" * 64: {"status": 500}}}
        with pytest.raises(ValueError, match="measured tape ancestry"):
            await executor(
                OP_RUN_MODEL_SANDBOX_V2,
                {**payload, "provider_evidence_cache": changed},
                context,
            )
    finally:
        executor.close()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_v2_dev_replay_preserves_score_and_adds_tree_commitments(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "candidate-source"
    _install_future_tree(source, monkeypatch)
    artifact = build_local_private_artifact_manifest(
        source_path=source,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/candidate@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/candidates/manifest.json",
        signature_ref="kms:signature",
        component_registry_version="1",
        scoring_adapter_version="1",
    )
    dev_items = [
        {
            "icp": {
                "icp_id": f"dev-{index}",
                "industry": "Software Development",
                "sub_industry": f"DevOps Tooling {index}",
                "product_service": "CI/CD platform",
                "geography": "United States",
                "country": "United States",
                "employee_count": "51-200",
                "intent_signals": [f"Hiring platform engineer {index}"],
            },
            "icp_ref": f"dev_set:{index}",
            "icp_hash": "sha256:" + str(index) * 64,
        }
        for index in range(
            1,
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node + 1,
        )
    ]
    snapshot_root = tmp_path / "snapshot-set"
    record_store = ProviderSnapshotStore(str(snapshot_root), mode=MODE_RECORD)
    record_store.write_dev_icp_items(dev_items)
    manifest = record_store.build_manifest(
        icp_set_hash=compute_dev_set_hash(dev_items),
        dev_set_manifest={"manifest_type": "research_lab_dev_icp_set"},
        recorded_at="2026-07-10T00:00:00Z",
    )
    record_store.write_manifest(manifest)
    snapshot_bundle = build_source_bundle_v2(snapshot_root)
    source_bundle = build_source_bundle_v2(source)

    def companies(icp):
        return [
            {
                "company_name": "Acme " + str(icp["icp_id"]),
                "company_website": "https://acme.test",
                "industry": "Software Development",
                "sub_industry": "DevOps Tooling",
                "employee_count": "51-200",
                "country": "United States",
                "description": "CI/CD platform for DevOps teams",
                "intent_signals": [
                    {
                        "source": "job_board",
                        "description": "Hiring a DevOps engineer",
                        "url": "https://acme.test/jobs/1",
                        "date": "2026-05-01",
                    }
                ],
            }
        ]

    class _Sandbox:
        def __init__(self):
            self.calls = []
            self.hybrid_calls = []

        def execute_dev_replay(self, **kwargs):
            self.calls.append(dict(kwargs))
            return companies(kwargs["icp"])

        def execute_dev_provider_replay(self, **kwargs):
            self.hybrid_calls.append(dict(kwargs))
            return companies(kwargs["icp"])

    replay_store = ProviderSnapshotStore(str(snapshot_root), mode=MODE_REPLAY)

    async def direct_runner(icp, _context):
        return companies(icp)

    expected = await evaluate_dev(
        candidate_runner=direct_runner,
        dev_items=dev_items,
        snapshot_store=replay_store,
        run_label="candidate-node-1",
        install_replay_seams=False,
        require_manifest=True,
        expected_icp_count=(
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
        ),
    )
    sandbox = _Sandbox()
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "frozen dev replay must not call a provider"
        ),
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=sandbox,
        execution_config=build_research_lab_execution_config(
            environment=epoch_test_environment(
                RESEARCH_LAB_LOOP_DEV_EVAL_ICP_TIMEOUT_SECONDS="30",
                RESEARCH_LAB_LOOP_DEV_EVAL_TIMEOUT_SECONDS="60",
            )
        ),
    )
    cohort_hash = "sha256:" + "d" * 64
    payload = {
        "schema_version": DEV_REPLAY_REQUEST_SCHEMA_VERSION,
        "artifact": artifact,
        "source_bundle": source_bundle,
        "snapshot_bundle": snapshot_bundle,
        "snapshot_tree_hash": snapshot_bundle["source_tree_hash"],
        "snapshot_manifest_hash": manifest["manifest_hash"],
        "dev_selection_request": {
            "selection_seed": "candidate-tree-1",
            "miner_direction": "improve DevOps hiring intent",
            "selection_manifest_hash": sha256_json(
                {
                    "schema_version": "research_lab.exact_snapshot_selection.v1",
                    "selection_policy": "exact_snapshot_compat_v1",
                    "requested_size": len(dev_items),
                    "dev_set_hash": compute_dev_set_hash(dev_items),
                    "selection_seed_hash": sha256_json(
                        {"selection_seed": "candidate-tree-1"}
                    ),
                    "miner_direction_hash": sha256_json(
                        {"miner_direction": "improve DevOps hiring intent"}
                    ),
                    "selected_items": [
                        {
                            "icp_ref": str(item["icp_ref"]),
                            "icp_hash": str(item["icp_hash"]),
                        }
                        for item in sorted(
                            dev_items,
                            key=lambda row: str(row["icp_hash"]),
                        )
                    ],
                }
            ),
        },
        "module_name": "research_lab_adapter",
        "callable_name": "run_icp",
        "environment": {},
        "credential_env_names": [],
        "run_label": "candidate-node-1",
        "cohort_hash": cohort_hash,
        "miss_policy": "strict",
        "per_icp_timeout_seconds": 30,
        "total_timeout_seconds": 60,
    }
    caches = {}
    cache_graphs = []
    for item in dev_items:
        canonical_icp = canonicalize_private_model_icp(item["icp"])
        cache_ref = icp_evidence_cache_key(canonical_icp)
        cache = {
            "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
            "icp_ref": cache_ref,
            "entries": {},
        }
        cache_hash = sha256_json(cache)
        receipt_hash = sha256_json({"cache_ref": cache_ref})
        caches[cache_ref] = cache
        cache_graphs.append(
            {
                "root_receipt_hash": receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": receipt_hash,
                        "role": "gateway_scoring",
                        "purpose": "research_lab.provider_evidence_tape.v2",
                        "status": "succeeded",
                        "input_root": provider_evidence_tape_input_root(
                            cache_ref, cache_hash
                        ),
                        "output_root": cache_hash,
                    }
                ],
            }
        )
    overlay_hash = sha256_json(caches)
    hybrid_cohort_hash = "sha256:" + "e" * 64
    hybrid_payload = {
        **payload,
        "schema_version": DEV_HYBRID_REQUEST_SCHEMA_VERSION,
        "cohort_hash": hybrid_cohort_hash,
        "provider_evidence_caches": caches,
        "overlay_hash": overlay_hash,
    }
    try:
        measured = await executor(
            OP_DEV_REPLAY_V2,
            payload,
            ExecutionContextV2(
                job_id="dev-replay-job-1",
                purpose="research_lab.candidate_test.v2",
                epoch_id=24000,
            ),
        )
        hybrid_measured = await executor(
            OP_DEV_HYBRID_V2,
            hybrid_payload,
            ExecutionContextV2(
                job_id="dev-hybrid-job-1",
                purpose="research_lab.candidate_hybrid_test.v2",
                epoch_id=24000,
                external_receipt_graphs=cache_graphs,
            ),
        )
        compact_hybrid_measured = await executor(
            OP_DEV_HYBRID_V2,
            hybrid_payload,
            ExecutionContextV2(
                job_id="dev-hybrid-job-compact-1",
                purpose="research_lab.candidate_hybrid_test.v2",
                epoch_id=24000,
                external_ancestry_proofs=[
                    {"disclosed_receipts": graph["receipts"]}
                    for graph in cache_graphs
                ],
            ),
        )
    finally:
        executor.close()

    expected_output = {
        **expected.to_dict(),
        "evaluation_mode": "replay",
        "overlay_hash": sha256_json({}),
        "cohort_hash": cohort_hash,
    }
    expected_output["score_commitment"] = sha256_json(
        {
            "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
            "dev_score_version": expected.dev_score_version,
            "dev_set_hash": expected.dev_set_hash,
            "snapshot_manifest_hash": expected.snapshot_manifest_hash,
            "miss_policy": expected.miss_policy,
            "evaluation_mode": "replay",
            "overlay_hash": sha256_json({}),
            "cohort_hash": cohort_hash,
        }
    )
    assert canonical_json(measured.output) == canonical_json(expected_output)
    assert measured.output["aggregate_dev_score"] == expected.aggregate_dev_score
    assert cohort_hash in measured.artifact_hashes
    assert len(sandbox.calls) == len(dev_items)
    assert all(call["timeout_seconds"] == 30 for call in sandbox.calls)
    assert all(call["miss_policy"] == "strict" for call in sandbox.calls)
    assert hybrid_measured.output["aggregate_dev_score"] == (
        expected.aggregate_dev_score
    )
    assert hybrid_measured.output["evaluation_mode"] == "hybrid"
    assert hybrid_measured.output["overlay_hash"] == overlay_hash
    assert hybrid_measured.output["cohort_hash"] == hybrid_cohort_hash
    assert compact_hybrid_measured.output == hybrid_measured.output
    assert hybrid_measured.output["score_commitment"] == sha256_json(
        {
            "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
            "dev_score_version": expected.dev_score_version,
            "dev_set_hash": expected.dev_set_hash,
            "snapshot_manifest_hash": expected.snapshot_manifest_hash,
            "miss_policy": expected.miss_policy,
            "evaluation_mode": "hybrid",
            "overlay_hash": overlay_hash,
            "cohort_hash": hybrid_cohort_hash,
        }
    )
    assert len(sandbox.hybrid_calls) == 2 * len(dev_items)
    assert hybrid_cohort_hash in hybrid_measured.artifact_hashes
    assert overlay_hash in hybrid_measured.artifact_hashes


@pytest.mark.asyncio
async def test_routing_experiment_attestation_operation_is_isolated_from_provider_execution(monkeypatch):
    fixture = authority_fixture()
    payload = build_routing_experiment_attestation_input_v2(
        spec_doc=fixture["spec"].to_dict(),
        evaluation_doc=fixture["evaluation"].to_dict(),
        gold_label_authority=fixture["labels"],
        artifact_lineage=fixture["lineage"],
        execution_envelope=fixture["execution_envelope"],
        decision_receipts=fixture["decisions"],
        provider_attempts=fixture["attempts"],
        budget_events=fixture["budgets"],
    )
    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr("gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network)
    provider_calls = []
    executor = ScoringExecutorV2(
        provider_execute=lambda request: provider_calls.append(request) or {},
        retry_policy_hashes={},
    )
    try:
        result = await executor(
            OP_ATTEST_ROUTING_EXPERIMENT_V2,
            payload,
            ExecutionContextV2(
                job_id="routing-attestation-1",
                purpose=ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
                epoch_id=24000,
            ),
        )
    finally:
        executor.close()
    assert result.output == routing_experiment_attestation_receipt_output_v2(payload)
    assert result.artifact_hashes == ()
    assert provider_calls == []


@pytest.mark.asyncio
async def test_routing_experiment_attestation_operation_rejects_provider_credentials(monkeypatch):
    fixture = authority_fixture()
    payload = build_routing_experiment_attestation_input_v2(
        spec_doc=fixture["spec"].to_dict(),
        evaluation_doc=fixture["evaluation"].to_dict(),
        gold_label_authority=fixture["labels"],
        artifact_lineage=fixture["lineage"],
        execution_envelope=fixture["execution_envelope"],
        decision_receipts=fixture["decisions"],
        provider_attempts=fixture["attempts"],
        budget_events=fixture["budgets"],
    )
    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr("gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network)
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {}, retry_policy_hashes={}
    )
    try:
        with pytest.raises(ValueError, match="must not use provider credentials"):
            await executor(
                OP_ATTEST_ROUTING_EXPERIMENT_V2,
                {
                    **payload,
                    "_v2_provider_credential_ref_hashes": {"deepline": HASH},
                },
                ExecutionContextV2(
                    job_id="routing-attestation-2",
                    purpose=ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
                    epoch_id=24000,
                    provider_credential_ref_hashes={"deepline": HASH},
                ),
            )
    finally:
        executor.close()


@pytest.mark.asyncio
async def test_routing_provider_call_authorization_is_exact_and_has_no_provider_access(monkeypatch):
    from gateway.research_lab.routing_execution_authorization import (
        build_routing_provider_authorization_request_v2,
    )
    from tests.test_routing_provider_authorization_context import _context

    authority = _context()
    payload = build_routing_provider_authorization_request_v2(
        authorization=authority["grant"],
        artifact_lineage=authority["lineage"],
        model_binding_observation=authority["observation"],
        execution_envelope=authority["envelope"],
        admission_bundle=authority["admission"],
        prepared_call=authority["prepared"],
        protected_release_receipt=authority["protected_receipt"],
    )

    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr("gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network)
    provider_calls = []
    executor = ScoringExecutorV2(
        provider_execute=lambda request: provider_calls.append(request) or {},
        retry_policy_hashes={},
        routing_artifact_lineage=authority["lineage"],
        routing_binding_catalog=authority["catalog"],
        routing_unit_dataset=authority["unit_dataset"],
    )
    parent_receipts = (
        authority["observation"].signed_receipt,
        authority["protected_receipt"],
    )
    try:
        result = await executor(
            OP_ATTEST_ROUTING_PROVIDER_CALL_V2,
            payload,
            ExecutionContextV2(
                job_id="routing-authorization-job",
                purpose=ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                epoch_id=24000,
                parent_receipt_hashes=tuple(
                    receipt["receipt_hash"] for receipt in parent_receipts
                ),
                external_receipt_graphs=[
                    {"receipts": [dict(receipt) for receipt in parent_receipts]}
                ],
            ),
        )
    finally:
        executor.close()
    assert result.output == execute_routing_provider_call_authorization_v2(
        authority["grant"].to_dict(),
        authorization_job_id="routing-authorization-job",
    )
    assert result.artifact_hashes == ()
    assert provider_calls == []


@pytest.mark.asyncio
async def test_protected_routing_terminal_executor_uses_reviewed_catalog_and_receipt_output():
    from gateway.research_lab.routing_provider_terminal_protected import (
        ROUTING_PROVIDER_TERMINAL_RESULT_SCHEMA_V2,
    )

    compiler, prepared, request, proof, result, record, boot, body, _key, auth_boot = _call_fixture(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    prepared_payload = asdict(prepared)
    prepared_payload["binding"] = prepared.binding.to_dict()
    payload = {
        "schema_version": "leadpoet.routing_provider_terminal_request.v2",
        "authorization_proof": proof,
        "prepared_call": prepared_payload,
        "broker_request": request,
        "broker_result": result,
        "provider_record": record,
        "raw_response_body_b64": base64.b64encode(body).decode(),
    }
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider transport must not run"),
        retry_policy_hashes={"deepline": prepared.retry_policy_hash},
        routing_binding_catalog=compiler.binding_catalog,
        routing_unit_dataset=compiler.unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: boot,
    )
    context = ExecutionContextV2(
        job_id=proof["authorization_result"]["authorization_job_id"],
        purpose="research_lab.routing_provider_evidence.v2",
        epoch_id=1,
        parent_receipt_hashes=(proof["authorization_receipt"]["receipt_hash"],),
    )
    try:
        output = await executor(OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2, payload, context)
    finally:
        executor.close()
    assert output.output["schema_version"] == ROUTING_PROVIDER_TERMINAL_RESULT_SCHEMA_V2
    assert output.receipt_output == output.output
    assert output.output["projection"]["outcome"] == "source_miss"


def _protected_dispatch_fixture():
    """Return a complete dispatch payload with the exact V3 budget document."""

    compiler, prepared, request, proof, result, record, boot, body, _key, auth_boot = _call_fixture(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    prepared_payload = asdict(prepared)
    prepared_payload["binding"] = prepared.binding.to_dict()
    authorization = RoutingProviderCallAuthorizationV2.from_mapping(
        proof["authorization"]
    )
    reservation = build_routing_budget_reservation_v3(
        authorization=authorization,
        prepared_call=prepared,
        lease_seconds=5,
    )
    payload = {
        "schema_version": "leadpoet.routing_provider_dispatch_request.v2",
        "authorization_proof": proof,
        "prepared_call": prepared_payload,
        "broker_request": request,
        "budget_reservation": reservation,
    }
    context = ExecutionContextV2(
        job_id=request["job_id"],
        purpose="research_lab.routing_provider_evidence.v2",
        epoch_id=1,
        parent_receipt_hashes=(proof["authorization_receipt"]["receipt_hash"],),
        external_receipt_graphs=[{"receipts": [proof["authorization_receipt"]]}],
    )
    return {
        "compiler": compiler,
        "prepared": prepared,
        "request": request,
        "proof": proof,
        "result": result,
        "record": record,
        "boot": boot,
        "auth_boot": auth_boot,
        "payload": payload,
        "context": context,
        "reservation": reservation,
    }


def _protected_budget_transport_result(
    request, reservation, *, document=None, http_status=200
):
    if document is None:
        document = {
            "schema_version": "leadpoet.research_lab.routing_budget_reservation_result.v3",
            "reserved": True,
            "idempotent": False,
            "reservation_id": reservation["reservation_id"],
            "event_key": reservation["event_key"],
            "experiment_hash": reservation["experiment_hash"],
            "binding_id": reservation["binding_id"],
            "claim_key": reservation["claim_key"],
            "claim_generation": reservation["claim_generation"],
            "credit_microunits": reservation["credit_microunits"],
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }
    response_body = canonical_json(document).encode()
    request_body = base64.b64decode(request["body_b64"], validate=True)
    attempt = build_transport_attempt(
        request_id="b" * 32,
        logical_operation_id=request["logical_operation_id"],
        job_id=request["job_id"],
        purpose=request["purpose"],
        provider_id=request["provider_id"],
        attempt_number=request["attempt_number"],
        method=request["method"],
        destination_host="supabase.example.com",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=sha256_bytes(request_body),
        credential_ref_hash=HASH,
        retry_policy_hash=request["retry_policy_hash"],
        timeout_ms=request["timeout_ms"],
        started_at="2026-08-19T12:00:00Z",
        terminal_status="authenticated_response",
        http_status=http_status,
        response_hash=sha256_bytes(response_body),
        request_artifact_hash=HASH,
        response_artifact_hash=HASH,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-08-19T12:00:01Z",
    )
    return {
        "terminal_status": "authenticated_response",
        "http_status": http_status,
        "headers": {},
        "body_b64": base64.b64encode(response_body).decode(),
        "transport_attempt": attempt,
    }


@pytest.mark.asyncio
async def test_protected_routing_dispatch_calls_coordinator_only_after_authority(
    monkeypatch,
):
    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network
    )
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = _call_fixture(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    prepared_payload = asdict(prepared)
    prepared_payload["binding"] = prepared.binding.to_dict()
    authorization = RoutingProviderCallAuthorizationV2.from_mapping(
        proof["authorization"]
    )
    budget_reservation = build_routing_budget_reservation_v3(
        authorization=authorization,
        prepared_call=prepared,
        lease_seconds=5,
    )
    payload = {
        "schema_version": "leadpoet.routing_provider_dispatch_request.v2",
        "authorization_proof": proof,
        "prepared_call": prepared_payload,
        "broker_request": request,
        "budget_reservation": budget_reservation,
    }
    provider_calls = []

    def budget_result(compiled_request):
        response_document = {
            "schema_version": "leadpoet.research_lab.routing_budget_reservation_result.v3",
            "reserved": True,
            "idempotent": False,
            "reservation_id": budget_reservation["reservation_id"],
            "event_key": budget_reservation["event_key"],
            "experiment_hash": budget_reservation["experiment_hash"],
            "binding_id": budget_reservation["binding_id"],
            "claim_key": budget_reservation["claim_key"],
            "claim_generation": budget_reservation["claim_generation"],
            "credit_microunits": budget_reservation["credit_microunits"],
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }
        response_body = canonical_json(response_document).encode()
        attempt = build_transport_attempt(
            request_id="b" * 32,
            logical_operation_id=compiled_request["logical_operation_id"],
            job_id=compiled_request["job_id"],
            purpose=compiled_request["purpose"],
            provider_id=compiled_request["provider_id"],
            attempt_number=compiled_request["attempt_number"],
            method=compiled_request["method"],
            destination_host="supabase.example.com",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=sha256_bytes(
                base64.b64decode(compiled_request["body_b64"], validate=True)
            ),
            credential_ref_hash=HASH,
            retry_policy_hash=compiled_request["retry_policy_hash"],
            timeout_ms=compiled_request["timeout_ms"],
            started_at="2026-08-19T12:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=sha256_bytes(response_body),
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-08-19T12:00:01Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {},
            "body_b64": base64.b64encode(response_body).decode(),
            "transport_attempt": attempt,
        }

    def provider_execute(compiled_request):
        provider_calls.append(dict(compiled_request))
        if compiled_request["provider_id"] == "supabase":
            return budget_result(compiled_request)
        return {**result, "routing_provider_record": record}

    executor = ScoringExecutorV2(
        provider_execute=provider_execute,
        retry_policy_hashes={"deepline": prepared.retry_policy_hash, "supabase": HASH},
        routing_binding_catalog=compiler.binding_catalog,
        routing_unit_dataset=compiler.unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: boot,
    )
    context = ExecutionContextV2(
        job_id=request["job_id"],
        purpose="research_lab.routing_provider_evidence.v2",
        epoch_id=1,
        parent_receipt_hashes=(proof["authorization_receipt"]["receipt_hash"],),
        external_receipt_graphs=[{"receipts": [proof["authorization_receipt"]]}],
    )
    try:
        output = await executor(
            OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2, payload, context
        )
        forged = dict(payload)
        forged["broker_request"] = {**request, "url": request["url"] + "?forged=1"}
        with pytest.raises(ValueError, match="authorization"):
            await executor(
                OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2, forged, context
            )
    finally:
        executor.close()
    assert len(provider_calls) == 2
    assert [call["provider_id"] for call in provider_calls] == [
        "supabase",
        "deepline",
    ]
    assert output.output["projection"]["outcome"] == "source_miss"
    assert output.receipt_output == output.output
    assert "routing_provider_record" not in output.output
    assert "body_b64" not in output.output
    assert len(output.transport_attempts) == 2
    assert output.transport_attempts[0]["provider_id"] == "supabase"
    assert output.transport_attempts[1]["provider_id"] == "deepline"
    budget_proof = output.output["budget_reservation"]
    assert set(budget_proof) == {
        "schema_version",
        "reservation_id",
        "event_key",
        "experiment_hash",
        "binding_id",
        "claim_key",
        "claim_generation",
        "credit_microunits",
        "lease_expires_at",
        "response_hash",
        "transport_attempt_hash",
    }
    assert "body_b64" not in budget_proof
    assert "credential_ref" not in budget_proof
    assert "supabase.example.com" not in repr(budget_proof)


@pytest.mark.asyncio
async def test_protected_routing_dispatch_rejects_missing_or_tampered_budget_before_provider(
    monkeypatch,
):
    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network
    )
    fixture = _protected_dispatch_fixture()
    provider_calls = []
    executor = ScoringExecutorV2(
        provider_execute=lambda request: provider_calls.append(request),
        retry_policy_hashes={"deepline": fixture["prepared"].retry_policy_hash, "supabase": HASH},
        routing_binding_catalog=fixture["compiler"].binding_catalog,
        routing_unit_dataset=fixture["compiler"].unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: fixture["boot"],
    )
    try:
        missing = dict(fixture["payload"])
        del missing["budget_reservation"]
        with pytest.raises(ValueError, match="payload is invalid"):
            await executor(
                OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2,
                missing,
                fixture["context"],
            )

        tampered = dict(fixture["payload"])
        tampered["budget_reservation"] = {
            **fixture["reservation"],
            "credit_microunits": fixture["reservation"]["credit_microunits"] + 1,
        }
        with pytest.raises(ValueError, match="authorization"):
            await executor(
                OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2,
                tampered,
                fixture["context"],
            )
    finally:
        executor.close()
    assert provider_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "malformed", "identity"])
async def test_protected_routing_dispatch_reservation_failure_never_calls_paid_provider(
    monkeypatch, failure
):
    class _Network:
        def install(self):
            return None

        def restore(self):
            return None

    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.SecureQualificationNetworkV2", _Network
    )
    fixture = _protected_dispatch_fixture()
    provider_calls = []

    def provider_execute(request):
        provider_calls.append(dict(request))
        if request["provider_id"] != "supabase":
            pytest.fail("paid provider was called after failed reservation")
        if failure == "transport":
            return _transport_failure_result(request, request_id="c" * 32)
        if failure == "malformed":
            return _protected_budget_transport_result(
                request, fixture["reservation"], document={}
            )
        document = {
            "schema_version": "leadpoet.research_lab.routing_budget_reservation_result.v3",
            "reserved": True,
            "idempotent": False,
            "reservation_id": "routing-reservation:forged",
            "event_key": fixture["reservation"]["event_key"],
            "experiment_hash": fixture["reservation"]["experiment_hash"],
            "binding_id": fixture["reservation"]["binding_id"],
            "claim_key": fixture["reservation"]["claim_key"],
            "claim_generation": fixture["reservation"]["claim_generation"],
            "credit_microunits": fixture["reservation"]["credit_microunits"],
            "lease_expires_at": "2099-01-01T00:00:00+00:00",
        }
        return _protected_budget_transport_result(
            request, fixture["reservation"], document=document
        )

    executor = ScoringExecutorV2(
        provider_execute=provider_execute,
        retry_policy_hashes={"deepline": fixture["prepared"].retry_policy_hash, "supabase": HASH},
        routing_binding_catalog=fixture["compiler"].binding_catalog,
        routing_unit_dataset=fixture["compiler"].unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: fixture["boot"],
    )
    try:
        with pytest.raises(ValueError, match="budget reservation"):
            await executor(
                OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2,
                fixture["payload"],
                fixture["context"],
            )
    finally:
        executor.close()
    assert [request["provider_id"] for request in provider_calls] == ["supabase"]


def test_protected_routing_dispatch_job_manager_commits_budget_and_paid_attempts():
    """The dispatch receipt keeps the auth parent separate and commits both calls."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    fixture = _protected_dispatch_fixture()
    prepared = fixture["prepared"]
    proof = fixture["proof"]
    payload_document = {
        **fixture["payload"],
        PARENT_RECEIPT_GRAPHS_FIELD: [
            build_receipt_graph(
                root_receipt_hash=proof["authorization_receipt"]["receipt_hash"],
                boot_identities=[fixture["auth_boot"]],
                receipts=[proof["authorization_receipt"]],
                transport_attempts=[],
            )
        ],
    }
    payload = canonical_json(payload_document).encode()
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    manager_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH,
            config_hash=HASH,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH,
            attestation_user_data_hash=HASH,
            issued_at="2026-08-19T12:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"manager-attestation").decode(),
    )
    calls = []

    def provider_execute(request):
        calls.append(dict(request))
        if request["provider_id"] == "supabase":
            return _protected_budget_transport_result(
                request, fixture["reservation"]
            )
        return {
            **fixture["result"],
            "routing_provider_record": fixture["record"],
        }

    executor = ScoringExecutorV2(
        provider_execute=provider_execute,
        retry_policy_hashes={"deepline": prepared.retry_policy_hash, "supabase": HASH},
        routing_binding_catalog=fixture["compiler"].binding_catalog,
        routing_unit_dataset=fixture["compiler"].unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: fixture["boot"],
    )
    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: manager_boot,
        sign_digest=key.sign,
        operations={
            OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2: {
                "research_lab.routing_provider_evidence.v2"
            }
        },
        executor=executor,
        worker_count=1,
    )
    job_id = routing_provider_dispatch_job_id_v2(proof)
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "operation": OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2,
        "purpose": "research_lab.routing_provider_evidence.v2",
        "epoch_id": 1,
        "sequence": 1,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "parent_receipt_hashes": [proof["authorization_receipt"]["receipt_hash"]],
        "input_artifact_hashes": [],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    try:
        manager.submit(manifest)
        manager.put_chunk(
            job_id=job_id,
            offset=0,
            data_b64=base64.b64encode(payload).decode(),
            chunk_sha256=sha256_bytes(payload),
        )
        manager.seal(job_id)
        deadline = time.time() + 2
        while time.time() < deadline:
            status = manager.status(job_id)
            if status["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert status["state"] == "succeeded"
        receipt = manager.receipt(job_id)
        attempts = manager.transport_attempts(job_id)
        assert [item["provider_id"] for item in calls] == ["supabase", "deepline"]
        assert [item["provider_id"] for item in attempts] == ["supabase", "deepline"]
        assert all(item["job_id"] == job_id for item in attempts)
        assert all(
            item["purpose"] == "research_lab.routing_provider_evidence.v2"
            or item["purpose"] == "research_lab.routing_budget_reservation.v3"
            for item in attempts
        )
        assert receipt["job_id"] == job_id
        assert receipt["parent_receipt_hashes"] == [
            proof["authorization_receipt"]["receipt_hash"]
        ]
        assert receipt["transport_root"] == transport_root(attempts)
        assert receipt["transport_root"] != EMPTY_TRANSPORT_ROOT
    finally:
        executor.close()


@pytest.mark.asyncio
async def test_protected_routing_terminal_executor_fails_closed_without_reviewed_catalog():
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = _call_fixture(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    prepared_payload = asdict(prepared)
    prepared_payload["binding"] = prepared.binding.to_dict()
    payload = {
        "schema_version": "leadpoet.routing_provider_terminal_request.v2",
        "authorization_proof": proof,
        "prepared_call": prepared_payload,
        "broker_request": request,
        "broker_result": result,
        "provider_record": record,
        "raw_response_body_b64": base64.b64encode(body).decode(),
    }
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider transport must not run"),
        retry_policy_hashes={"deepline": prepared.retry_policy_hash},
    )
    try:
        with pytest.raises(ValueError, match="authorities are unavailable"):
            await executor(
                OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2,
                payload,
                ExecutionContextV2(
                    job_id=proof["authorization_result"]["authorization_job_id"],
                    purpose="research_lab.routing_provider_evidence.v2",
                    epoch_id=1,
                    parent_receipt_hashes=(proof["authorization_receipt"]["receipt_hash"],),
                ),
            )
    finally:
        executor.close()


def test_protected_routing_terminal_job_manager_uses_standard_receipt_roots():
    """The manager, not the terminal normalizer, signs the final receipt."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    compiler, prepared, request, proof, result, record, boot, body, _key, auth_boot = _call_fixture(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    prepared_payload = asdict(prepared)
    prepared_payload["binding"] = prepared.binding.to_dict()
    payload_document = {
        "schema_version": "leadpoet.routing_provider_terminal_request.v2",
        "authorization_proof": proof,
        "prepared_call": prepared_payload,
        "broker_request": request,
        "broker_result": result,
        "provider_record": record,
        "raw_response_body_b64": base64.b64encode(body).decode(),
        PARENT_RECEIPT_GRAPHS_FIELD: [
            build_receipt_graph(
                root_receipt_hash=proof["authorization_receipt"]["receipt_hash"],
                boot_identities=[auth_boot],
                receipts=[proof["authorization_receipt"]],
                transport_attempts=[],
            )
        ],
    }
    payload = canonical_json(payload_document).encode()
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    manager_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH,
            config_hash=HASH,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH,
            attestation_user_data_hash=HASH,
            issued_at="2026-08-19T12:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"manager-attestation").decode(),
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("provider transport must not run"),
        retry_policy_hashes={"deepline": prepared.retry_policy_hash},
        routing_binding_catalog=compiler.binding_catalog,
        routing_unit_dataset=compiler.unit_dataset,
        routing_coordinator_boot_identity_supplier=lambda: boot,
    )
    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: manager_boot,
        sign_digest=key.sign,
        operations={
            OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2: {
                "research_lab.routing_provider_evidence.v2"
            }
        },
        executor=executor,
        worker_count=1,
    )
    job_id = proof["authorization_result"]["authorization_job_id"]
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "operation": OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2,
        "purpose": "research_lab.routing_provider_evidence.v2",
        "epoch_id": 1,
        "sequence": 1,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "parent_receipt_hashes": [proof["authorization_receipt"]["receipt_hash"]],
        "input_artifact_hashes": [],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    try:
        manager.submit(manifest)
        manager.put_chunk(
            job_id=job_id,
            offset=0,
            data_b64=base64.b64encode(payload).decode(),
            chunk_sha256=sha256_bytes(payload),
        )
        manager.seal(job_id)
        deadline = time.time() + 2
        while time.time() < deadline:
            status = manager.status(job_id)
            if status["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert status["state"] == "succeeded"
        receipt = manager.receipt(job_id)
        assert receipt["input_root"] == sha256_bytes(payload)
        output_bytes = base64.b64decode(
            manager.result_chunk(job_id=job_id)["data_b64"]
        )
        assert receipt["output_root"] == sha256_bytes(output_bytes)
        assert receipt["parent_receipt_hashes"] == [
            proof["authorization_receipt"]["receipt_hash"]
        ]
    finally:
        executor.close()
