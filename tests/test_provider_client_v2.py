from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request

import aiohttp
import httpx
import pytest
import requests

from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    HTTPXProviderTransport,
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
    _close_client_transports,
    credential_reference_hash,
)
from gateway.tee.provider_client_v2 import (
    BrokeredProviderTransportV2,
    ProviderClientV2Error,
    _ExecutionScope,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
)
from leadpoet_canonical.attested_v2 import DIRECT_EGRESS_REF_HASH, sha256_bytes
from leadpoet_verifier.semantic_gates import (
    EvidenceSource,
    SemanticGateEvaluator,
)
from qualification.scoring.company_verification import verify_company_exists
HASH = "sha256:" + "a" * 64


class _AuthenticatedNetworkStream:
    class _TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    def get_extra_info(self, name):
        assert name == "ssl_object"
        return self._TLS()

    def close(self):
        return None


def _raw_httpx_response(request, body=b'{"ok":true}'):
    return httpx.Response(
        200,
        headers={
            "content-length": str(len(body)),
            "content-type": "application/json",
        },
        content=body,
        extensions={"network_stream": _AuthenticatedNetworkStream()},
        request=request,
    )


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


def test_execution_scope_keeps_request_intents_and_terminal_observation():
    scope = _ExecutionScope(
        job_id="job-observation",
        purpose="research_lab.provider_evidence.v2",
        logical_operation_id="score-observation",
        retry_policy_hashes={},
        default_timeout_ms=1000,
        terminal_sink=None,
    )
    scope.record_intent("operation-1", 0)
    scope.record_terminal(
        "operation-1",
        0,
        "authenticated_response",
        200,
        "sha256:" + "b" * 64,
    )
    scope.record_intent("operation-2", 0)
    scope.record_terminal(
        "operation-2",
        0,
        "transport_failure",
        None,
        "sha256:" + "c" * 64,
    )

    assert scope.completion_observation() == {
        "schema_version": "leadpoet.provider-terminal-observation.v1",
        "request_intent_count": 2,
        "terminal_count": 2,
        "latest_operation_count": 2,
        "accepted_latest_terminal_count": 1,
        "successful_latest_terminal_count": 1,
        "failed_latest_terminal_count": 1,
        "unresolved_latest_terminal_count": 1,
        "latest_terminal_attempt_hashes": [
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
        ],
        "successful_latest_terminal_attempt_hashes": [
            "sha256:" + "b" * 64
        ],
    }


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
                    timeout=httpx.Timeout(180, connect=7),
                )
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert observed[0]["transport_attempt"]["terminal_status"] == "authenticated_response"
        assert transport.calls[0]["headers"]["Authorization"] == "Bearer openrouter-secret"
        assert transport.calls[0]["timeout_ms"] == 180_000
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


@pytest.mark.parametrize(
    ("failure_code", "expected_type", "retryable"),
    [
        ("timeout", httpx.ReadTimeout, True),
        ("connection_refused", httpx.ConnectError, True),
        ("dns_failure", httpx.ConnectError, True),
        ("proxy_failure", httpx.ConnectError, True),
        ("tls_failure", httpx.ConnectError, True),
        ("connection_reset", httpx.ReadError, True),
        ("host_dropped", httpx.ReadError, True),
        ("malformed_reply", httpx.ReadError, True),
        ("unexpected_eof", httpx.ReadError, True),
        ("cancelled", httpx.TransportError, False),
        ("certificate_invalid", httpx.TransportError, False),
        ("plaintext_forbidden", httpx.TransportError, False),
        ("policy_denied", httpx.TransportError, False),
        ("response_too_large", httpx.TransportError, False),
    ],
)
def test_httpx_failure_preserves_retry_semantics(
    failure_code, expected_type, retryable
):
    router, _ = _router(Transport())
    router._execute_request = lambda **_kwargs: {
        "terminal_status": "transport_failure",
        "failure_code": failure_code,
    }
    try:
        request = httpx.Request("GET", "https://openrouter.ai/api/v1/models")
        with pytest.raises(expected_type) as error:
            router._httpx_response(request)
        assert isinstance(
            error.value,
            (httpx.TimeoutException, httpx.NetworkError),
        ) is retryable
        assert str(error.value) == "attested transport failure: %s" % failure_code
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


@pytest.mark.asyncio
async def test_source_style_httpx_retry_advances_signed_attempt_number():
    class RetryTransport(Transport):
        def __call__(self, **request):
            self.calls.append(request)
            if len(self.calls) == 1:
                raise RuntimeError("malformed reply")
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": b'{"ok":true}',
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }

    transport = RetryTransport()
    router, observed = _router(transport)
    terminal_attempts = []
    try:
        with router.scope(
            job_id="job-1",
            purpose="research_lab.provider_evidence.v2",
            logical_operation_id="score-icp-1",
            retry_policy_hashes={
                provider: HASH for provider in BUILTIN_PROVIDER_ROUTES
            },
            terminal_sink=terminal_attempts.append,
        ):
            async with httpx.AsyncClient(trust_env=False) as client:
                for attempt in range(3):
                    try:
                        response = await client.get(
                            "https://openrouter.ai/api/v1/models"
                        )
                        break
                    except (httpx.TimeoutException, httpx.NetworkError):
                        if attempt == 2:  # pragma: no cover - regression guard
                            raise

        assert response.status_code == 200
        assert [row["attempt_number"] for row in terminal_attempts] == [0, 1]
        assert [row["terminal_status"] for row in terminal_attempts] == [
            "transport_failure",
            "authenticated_response",
        ]
        assert [row["failure_code"] for row in terminal_attempts] == [
            "malformed_reply",
            None,
        ]
        assert [row["transport_attempt"]["attempt_number"] for row in observed] == [
            0,
            1,
        ]
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_company_verification_retries_signed_transient_homepage_failure():
    class TransientHomepageTransport(Transport):
        def __call__(self, **request):
            self.calls.append(request)
            if len(self.calls) == 1:
                raise RuntimeError("malformed reply")
            return {
                "http_status": 200,
                "headers": {"content-type": "text/html"},
                "body": (
                    b"<html><title>Example Company</title>"
                    b'<a href="https://www.linkedin.com/company/example-company">'
                    b"LinkedIn</a></html>"
                ),
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }

    transport = TransientHomepageTransport()
    router, observed = _router(transport)
    terminal_attempts = []
    try:
        with router.scope(
            job_id="company-score-job",
            purpose="research_lab.company_score.v2",
            logical_operation_id="company-score-operation",
            retry_policy_hashes={
                provider: HASH for provider in BUILTIN_PROVIDER_ROUTES
            },
            terminal_sink=terminal_attempts.append,
        ):
            result = await verify_company_exists(
                "Example Company",
                "https://example.com/",
                company_linkedin=(
                    "https://www.linkedin.com/company/example-company"
                ),
                require_https_transport=True,
            )

        assert result.passed is True
        assert (result.reason or "").startswith("verified:")
        assert [row["attempt_number"] for row in terminal_attempts] == [0, 1]
        assert [row["terminal_status"] for row in terminal_attempts] == [
            "transport_failure",
            "authenticated_response",
        ]
        assert [row["transport_attempt"]["attempt_number"] for row in observed] == [
            0,
            1,
        ]
    finally:
        router.restore()


@pytest.mark.asyncio
async def test_company_verification_exhaustion_remains_fail_closed():
    transport = Transport(error=RuntimeError("unexpected eof"))
    router, observed = _router(transport)
    try:
        with router.scope(
            job_id="company-score-job",
            purpose="research_lab.company_score.v2",
            logical_operation_id="company-score-operation",
            retry_policy_hashes={
                provider: HASH for provider in BUILTIN_PROVIDER_ROUTES
            },
            allow_transport_failures=True,
        ):
            result = await verify_company_exists(
                "Example Company",
                "https://example.com/",
                company_linkedin=(
                    "https://www.linkedin.com/company/example-company"
                ),
                require_https_transport=True,
            )

        assert result.decision == "unavailable"
        assert result.passed is False
        assert len(observed) == 2
        assert [
            row["transport_attempt"]["attempt_number"] for row in observed
        ] == [0, 1]
        assert all(
            row["transport_attempt"]["terminal_status"] == "transport_failure"
            for row in observed
        )
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


def test_raw_broker_httpx_uses_captured_send_outside_scope_and_keeps_direct_receipt(
    monkeypatch,
):
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    raw_requests = []

    def raw_send(_client, request, *args, **kwargs):
        raw_requests.append(request)
        return _raw_httpx_response(request, b'[{"sequence":17738}]')

    monkeypatch.setattr(httpx.Client, "send", raw_send)
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-secret",
        "truelist": "truelist-secret",
    }
    physical_transport = HTTPXProviderTransport()
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(value)
            for slot, value in credentials.items()
        },
        retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
        transport=physical_transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: "2026-07-10T20:00:00Z",
    )
    broker.provision_credentials(credentials)
    router = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("raw broker request was recursively intercepted")
    )
    router.install()
    try:
        result = broker.execute(
            {
                "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                "logical_operation_id": "supabase-preflight-outcome",
                "job_id": "preflight-job-1",
                "purpose": "research_lab.provider_preflight.v2",
                "provider_id": "supabase",
                "attempt_number": 0,
                "method": "GET",
                "url": (
                    "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                    "research_lab_rebenchmark_controls?select=sequence"
                ),
                "headers": {},
                "body_b64": base64.b64encode(b"").decode("ascii"),
                "timeout_ms": 30000,
                "retry_policy_hash": HASH,
            }
        )
    finally:
        router.restore()
        physical_transport.close()

    assert len(raw_requests) == 1
    assert result["terminal_status"] == "authenticated_response"
    assert (
        result["transport_attempt"]["egress_proxy_ref_hash"]
        == DIRECT_EGRESS_REF_HASH
    )


@pytest.mark.parametrize("spoof_explicit_transport_marker", (False, True))
def test_unowned_httpx_client_cannot_bypass_hooks_even_with_spoofed_marker(
    spoof_explicit_transport_marker,
    monkeypatch,
):
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    underlying_requests = []

    def underlying(request):
        underlying_requests.append(request)
        return _raw_httpx_response(request)

    router, observed = _router(Transport())
    client = httpx.Client(transport=httpx.MockTransport(underlying))
    if spoof_explicit_transport_marker:
        setattr(
            client,
            "_leadpoet_explicit_http_transport",
            client._transport,
        )
    router.install()
    try:
        with pytest.raises(ProviderClientV2Error, match="outside"):
            client.get("https://openrouter.ai/api/v1/models")
        with _scope(router):
            response = client.get("https://openrouter.ai/api/v1/models")
    finally:
        client.close()
        router.restore()

    assert response.status_code == 200
    assert len(observed) == 1
    assert underlying_requests == []


@pytest.mark.parametrize(
    "tamper",
    ("none", "transport", "mount", "role"),
)
def test_registered_httpx_client_is_fail_closed_without_exact_send_grant_and_shape(
    monkeypatch,
    tamper,
):
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    raw_requests = []

    def raw_send(_client, request, *args, **kwargs):
        raw_requests.append(request)
        return _raw_httpx_response(request)

    monkeypatch.setattr(httpx.Client, "send", raw_send)
    physical_transport = HTTPXProviderTransport()
    client = physical_transport._new_client()
    if tamper == "transport":
        client._transport = httpx.MockTransport(raw_send)
    elif tamper == "mount":
        client._mounts = {object(): httpx.MockTransport(raw_send)}
    elif tamper == "role":
        monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_scoring")
    router = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("out-of-scope HTTPX reached the broker")
    )
    router.install()
    try:
        with pytest.raises(ProviderClientV2Error, match="outside"):
            client.get("https://openrouter.ai/api/v1/models")
    finally:
        router.restore()
        _close_client_transports(client)
        physical_transport.close()

    assert raw_requests == []


@pytest.mark.asyncio
async def test_async_httpx_client_cannot_spoof_broker_owned_marker(monkeypatch):
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_coordinator")
    underlying_requests = []

    async def underlying(request):
        underlying_requests.append(request)
        return _raw_httpx_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(underlying))
    setattr(
        client,
        "_leadpoet_explicit_http_transport",
        client._transport,
    )
    router = BrokeredProviderTransportV2(
        lambda _request: pytest.fail("out-of-scope HTTPX reached the broker")
    )
    router.install()
    try:
        with pytest.raises(ProviderClientV2Error, match="outside"):
            await client.get("https://openrouter.ai/api/v1/models")
    finally:
        await client.aclose()
        router.restore()

    assert underlying_requests == []


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
