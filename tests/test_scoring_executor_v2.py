from __future__ import annotations

import base64
import httpx
import pytest

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.scoring_executor_v2 import (
    OP_PROVIDER_PREFLIGHT_V2,
    OP_SOURCE_ADD_LEG2_JUDGE_V2,
    PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
    ScoringExecutorV2,
)
from gateway.tee.provider_client_v2 import ProviderClientV2Error
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    build_transport_attempt,
    sha256_json,
)
from tests.v2_epoch_test_utils import epoch_test_environment


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
async def test_scoring_executor_rejects_retired_company_score_operation():
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("no request expected"),
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        with pytest.raises(ValueError, match="unsupported V2 scoring operation"):
            await executor(
                "qualification_company_scores",
                {},
                ExecutionContextV2(
                    job_id="retired-company-score",
                    purpose="research_lab.company_score.v2",
                    epoch_id=1,
                ),
            )
    finally:
        executor.close()


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
