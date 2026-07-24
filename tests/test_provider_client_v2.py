from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import aiohttp
import httpx
import pytest
import requests

from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    ProviderBrokerV2,
    credential_reference_hash,
)
from gateway.tee.provider_client_v2 import (
    BrokeredProviderTransportV2,
    ProviderClientV2Error,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
)
from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_verifier.semantic_gates import (
    EvidenceSource,
    SemanticGateEvaluator,
)


HASH = "sha256:" + "a" * 64


class Transport:
    def __init__(self, *, status=200, error=None):
        self.status = status
        self.error = error
        self.calls = []

    def __call__(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return {
            "http_status": self.status,
            "headers": {"content-type": "application/json", "retry-after": "2"},
            "body": json.dumps({"ok": self.status < 400}).encode("utf-8"),
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }


def _router(transport):
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(value)
            for slot, value in credentials.items()
        },
        retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: "2026-07-10T20:00:00Z",
    )
    broker.provision_credentials(credentials)
    observed = []

    def _execute(request):
        result = broker.execute(request)
        observed.append(result)
        return result

    return BrokeredProviderTransportV2(_execute), observed


def _scope(router):
    return router.scope(
        job_id="job-1",
        purpose="research_lab.provider_evidence.v2",
        logical_operation_id="score-icp-1",
        retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
    )


def test_httpx_request_uses_coordinator_and_preserves_response_shape():
    transport = Transport(status=200)
    router, observed = _router(transport)
    try:
        with _scope(router):
            with httpx.Client(trust_env=False) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": "Bearer runner-placeholder"},
                    json={"model": "model-1"},
                )
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert observed[0]["transport_attempt"]["terminal_status"] == "authenticated_response"
        assert transport.calls[0]["headers"]["Authorization"] == "Bearer openrouter-secret"
        assert "runner-placeholder" not in str(observed)
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_semantic_gate_openrouter_call_uses_attested_provider_transport():
    class SemanticTransport(Transport):
        def __call__(self, **request):
            self.calls.append(request)
            judgment = {
                "decision": "match",
                "confidence": 0.99,
                "relationship": "exact",
                "entity_match": True,
                "evidence_ids": ["source_1"],
                "reason": "The cited source directly supports the criterion.",
            }
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(judgment)}}
                        ],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 8,
                        },
                    }
                ).encode("utf-8"),
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }

    transport = SemanticTransport()
    router, observed = _router(transport)
    evaluator = SemanticGateEvaluator(
        api_key="leadpoet-v2-brokered-credential",
        models=("openai/gpt-4.1-mini",),
    )
    source = EvidenceSource(
        source_id="source_1",
        url="https://example.com/evidence",
        source_type="official_company",
        entity_match=True,
        content="A" * 300,
        content_sha256="a" * 64,
        fetch_stage="test",
    )
    try:
        with _scope(router):
            judgment, model, prompt_tokens, completion_tokens = (
                await evaluator._call_model(
                    "industry",
                    {"requested_criterion": "Legal technology"},
                    [source],
                )
            )

        assert judgment.decision == "match"
        assert model == "openai/gpt-4.1-mini"
        assert prompt_tokens == 12
        assert completion_tokens == 8
        assert len(observed) == 1
        assert observed[0]["transport_attempt"]["provider_id"] == "openrouter"
        assert transport.calls[0]["headers"]["Authorization"] == (
            "Bearer openrouter-secret"
        )
        body = json.loads(transport.calls[0]["body"])
        assert body["provider"] == {"data_collection": "deny", "zdr": True}
        assert body["response_format"]["type"] == "json_schema"
        assert "leadpoet-v2-brokered-credential" not in str(observed)
    finally:
        router.restore()


def test_truelist_placeholder_is_removed_and_kms_credential_is_injected():
    transport = Transport(status=200)
    router, observed = _router(transport)
    try:
        with _scope(router):
            with httpx.Client(trust_env=False) as client:
                response = client.post(
                    "https://api.truelist.io/api/v1/batches",
                    headers={"Authorization": "Bearer leadpoet-v2-brokered-credential"},
                    content=b"batch",
                )
        assert response.status_code == 200
        assert transport.calls[0]["headers"]["Authorization"] == (
            "Bearer truelist-secret"
        )
        assert "leadpoet-v2-brokered-credential" not in str(observed)
    finally:
        router.restore()


def test_requests_call_and_executor_thread_are_bound_to_the_active_job():
    transport = Transport(status=200)
    router, observed = _router(transport)
    try:
        async def _call():
            with _scope(router):
                return await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        "https://api.scrapingdog.com/google",
                        params={"query": "leadpoet"},
                        timeout=7,
                    ),
                )

        response = asyncio.run(_call())
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert len(observed) == 1
        assert observed[0]["transport_attempt"]["provider_id"] == "scrapingdog"
        assert "api_key=scrapingdog-secret" in transport.calls[0]["url"]
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_aiohttp_preserves_params_json_and_authenticated_error_shape():
    transport = Transport(status=503)
    router, observed = _router(transport)
    try:
        with _scope(router):
            async with aiohttp.ClientSession(
                headers={"x-title": "Leadpoet"}
            ) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    params={"trace": "1"},
                    json={"model": "model-1"},
                    timeout=aiohttp.ClientTimeout(total=4),
                ) as response:
                    assert response.status == 503
                    assert await response.json() == {"ok": False}
                    with pytest.raises(aiohttp.ClientResponseError) as error:
                        response.raise_for_status()
        assert error.value.status == 503
        assert len(observed) == 1
        assert transport.calls[0]["url"].endswith("?trace=1")
        assert json.loads(transport.calls[0]["body"]) == {"model": "model-1"}
        assert transport.calls[0]["headers"]["Authorization"] == (
            "Bearer openrouter-secret"
        )
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_aiohttp_follows_only_authenticated_https_redirects():
    calls = []

    def transport(**request):
        calls.append(request)
        if len(calls) == 1:
            return {
                "http_status": 302,
                "headers": {"location": "https://example.org/final"},
                "body": b"",
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }
        return {
            "http_status": 200,
            "headers": {"content-type": "text/plain"},
            "body": b"done",
            "tls_peer_chain_hash": "sha256:" + "c" * 64,
            "tls_protocol": "TLSv1.3",
        }

    router, observed = _router(transport)
    try:
        with _scope(router):
            async with aiohttp.ClientSession() as session:
                async with session.get("https://example.com/start") as response:
                    assert await response.text() == "done"
                    assert len(response.history) == 1
        assert [item["url"] for item in calls] == [
            "https://example.com/start",
            "https://example.org/final",
        ]
        assert len(observed) == 2
    finally:
        router.restore()


def test_authenticated_http_error_remains_a_provider_response():
    router, observed = _router(Transport(status=503))
    try:
        with _scope(router):
            request = urllib.request.Request(
                "https://api.exa.ai/search",
                data=b"{}",
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 503
        assert json.loads(error.value.read()) == {"ok": False}
        assert observed[0]["transport_attempt"]["http_status"] == 503
    finally:
        router.restore()


def test_transport_failure_never_becomes_provider_502():
    router, observed = _router(Transport(error=RuntimeError("proxy generated 502")))
    try:
        with pytest.raises(ProviderClientV2Error, match="did not authenticate"):
            with _scope(router):
                with httpx.Client(trust_env=False) as client:
                    with pytest.raises(httpx.TransportError, match="proxy_failure"):
                        client.get("https://openrouter.ai/api/v1/models")
        assert observed[0]["terminal_status"] == "transport_failure"
        assert observed[0]["transport_attempt"]["http_status"] is None
    finally:
        router.restore()


def test_preflight_scope_can_return_a_signed_transport_failure_terminal():
    router, observed = _router(Transport(error=TimeoutError("timed out")))
    try:
        with router.scope(
            job_id="preflight-job-1",
            purpose="research_lab.provider_preflight.v2",
            logical_operation_id="preflight-worker-1",
            retry_policy_hashes={
                provider: HASH for provider in BUILTIN_PROVIDER_ROUTES
            },
            allow_transport_failures=True,
        ):
            with httpx.Client(trust_env=False) as client:
                with pytest.raises(httpx.TransportError, match="timeout"):
                    client.get("https://api.exa.ai/search")
        assert observed[0]["transport_attempt"]["terminal_status"] == (
            "transport_failure"
        )
        assert observed[0]["transport_attempt"]["failure_code"] == "timeout"
    finally:
        router.restore()


def test_caught_broker_rejection_without_terminal_cannot_authorize_result():
    router = BrokeredProviderTransportV2(
        lambda _request: (_ for _ in ()).throw(RuntimeError("broker rejected"))
    )
    try:
        with pytest.raises(ProviderClientV2Error, match="missing a signed terminal"):
            with router.scope(
                job_id="job-1",
                purpose="research_lab.provider_evidence.v2",
                logical_operation_id="score-1",
                retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
            ):
                try:
                    urllib.request.urlopen("https://api.exa.ai/search")
                except RuntimeError:
                    pass
    finally:
        router.restore()


def test_retry_that_ends_in_authenticated_response_can_authorize_result():
    class RetryTransport(Transport):
        def __call__(self, **request):
            self.calls.append(request)
            if len(self.calls) == 1:
                raise TimeoutError("timed out")
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": b'{"ok":true}',
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }

    transport = RetryTransport()
    router, observed = _router(transport)
    try:
        with _scope(router):
            with httpx.Client(trust_env=False) as client:
                with pytest.raises(httpx.TransportError, match="timeout"):
                    client.get("https://openrouter.ai/api/v1/models")
                response = client.get("https://openrouter.ai/api/v1/models")
                assert response.status_code == 200
        assert [row["transport_attempt"]["terminal_status"] for row in observed] == [
            "transport_failure",
            "authenticated_response",
        ]
    finally:
        router.restore()


def test_runner_has_no_external_network_fallback_outside_scope():
    router, _ = _router(Transport())
    router.install()
    try:
        with httpx.Client(trust_env=False) as client:
            with pytest.raises(ProviderClientV2Error, match="outside"):
                client.get("https://openrouter.ai/api/v1/models")
        with pytest.raises(ProviderClientV2Error, match="outside"):
            urllib.request.urlopen("https://api.exa.ai/search")
        with pytest.raises(ProviderClientV2Error, match="outside"):
            requests.get("https://api.exa.ai/search")
    finally:
        router.restore()


def test_custom_urllib_proxy_opener_cannot_bypass_attested_transport():
    transport = Transport(status=200)
    router, observed = _router(transport)
    try:
        with _scope(router):
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler(
                    {"https": "http://untrusted-parent-proxy.invalid:8080"}
                )
            )
            response = opener.open(
                urllib.request.Request(
                    "https://openrouter.ai/api/v1/key",
                    headers={"Authorization": "Bearer runner-placeholder"},
                )
            )
            assert response.status == 200
        assert len(observed) == 1
        assert transport.calls[0]["headers"]["Authorization"] == (
            "Bearer openrouter-secret"
        )
        assert "untrusted-parent-proxy" not in str(transport.calls)
    finally:
        router.restore()


def test_dynamic_source_add_route_is_selected_from_measured_job_catalog():
    transport = Transport(status=200)
    router, observed = _router(transport)
    row = {
        "adapter_id": "adapter:public-source",
        "miner_hotkey": "miner-one",
        "provision_status": "provisioned_autoresearch_eligible",
        "registry_provider_id": "public_source",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "public_source",
                "base_url": "https://api.public-source.example",
                "auth_kind": "none",
                "auth_name": "",
                "credential_ref": [],
                "per_day_quota": 5,
                "cost_model": {"est_cost_microusd_per_call": 0},
                "capability_policy": {
                    "routes": [{"method": "GET", "path": "/status"}]
                },
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "public_source.status",
                    "provider_id": "public_source",
                    "method": "GET",
                    "path": "/status",
                    "params": [],
                }
            ],
        },
    }
    catalog = build_source_add_runtime_catalog_v2([row])
    route = catalog["routes"][0]
    try:
        with router.scope(
            job_id="source-add-job",
            purpose="research_lab.provider_evidence.v2",
            logical_operation_id="source-add-operation",
            retry_policy_hashes={
                "public_source": source_add_dynamic_retry_policy_hash(route)
            },
            dynamic_provider_catalog=catalog,
        ):
            with httpx.Client(trust_env=False) as client:
                response = client.get(
                    "https://api.public-source.example/status?verbose=1"
                )
        assert response.status_code == 200
        assert observed[0]["transport_attempt"]["provider_id"] == "public_source"
        assert transport.calls[0]["url"].endswith("/status?verbose=1")
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_aiohttp_has_no_external_network_fallback_outside_scope():
    router, _ = _router(Transport())
    router.install()
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ProviderClientV2Error, match="outside"):
                await session.get("https://api.exa.ai/search")
    finally:
        router.restore()
