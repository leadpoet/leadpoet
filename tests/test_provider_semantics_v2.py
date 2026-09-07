from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
from types import SimpleNamespace
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.provider_evidence_proxy import (
    REPLAY_ONLY_HEADER,
)
from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    PROVIDER_BROKER_SCHEMA_VERSION,
    PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION,
    ProviderBrokerV2,
    ProviderBrokerV2Error,
    ProviderTransportCleanupError,
    _provider_rpc_response_body_limit,
    credential_reference_hash,
    credential_value_hash,
    expected_provider_credential_slots,
    validate_provider_transport_failure_diagnostic,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.inter_enclave_tls import (
    MAX_FRAME_BYTES,
    MAX_RPC_DELIVERY_ATTEMPTS,
    REPLAY_WAIT_SECONDS,
    AttestedTLSRPCClient,
    _RetryableInterEnclaveTransportError,
)
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionJobV2Error,
)
from gateway.tee.provider_semantics_v2 import (
    ProviderSemanticsAuthorityV2,
    ProviderSemanticsV2Error,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
)
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
)
from research_lab.eval.provider_costs import decode_cost_event_header
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint


def _hash(character: str) -> str:
    return "sha256:" + character * 64


class _Artifacts:
    def __init__(self) -> None:
        self.values = []

    def seal(self, plaintext, *, job_id, purpose, artifact_kind):
        body = bytes(plaintext)
        index = len(self.values) + 1
        value = {
            "status": "sealed",
            "artifact_id": sha256_json(
                {
                    "index": index,
                    "job_id": job_id,
                    "purpose": purpose,
                    "artifact_kind": artifact_kind,
                }
            ),
            "plaintext_hash": sha256_bytes(body),
            "ciphertext_hash": sha256_json({"ciphertext": index}),
            "encryption_context_hash": sha256_json({"context": index}),
        }
        self.values.append((body, value))
        return value


class _CacheStore:
    def __init__(self) -> None:
        self.payloads = {}
        self.load_count = 0
        self.persist_count = 0

    def load(
        self,
        *,
        utc_day,
        request_fingerprint,
        job_id,
        purpose,
        attempt_number=0,
    ):
        del attempt_number
        self.load_count += 1
        payload = self.payloads.get((utc_day, request_fingerprint))
        return {
            "found": payload is not None,
            "payload": dict(payload or {}),
            "transport_attempts": [],
            "evidence_artifact_hashes": [],
        }

    def persist_recorded(self, terminal, *, utc_day, job_id, purpose):
        self.persist_count += 1
        fingerprint = terminal["record"]["request_fingerprint"]
        self.payloads[(utc_day, fingerprint)] = {
            "schema_version": "leadpoet.provider_evidence_cache_payload.v2",
            "utc_day": utc_day,
            "request_fingerprint": fingerprint,
            "status": terminal["status"],
            "body_b64": terminal["body_b64"],
            "source_record": dict(terminal["record"]),
            "source_boot_identity": dict(terminal["coordinator_boot_identity"]),
            "source_transport_attempt": dict(terminal["transport_attempts"][0]),
            "source_evidence_artifact_hashes": list(
                terminal["evidence_artifact_hashes"]
            ),
        }
        return {
            "transport_attempts": [],
            "evidence_artifact_hashes": [],
        }








class _Broker:
    def __init__(self) -> None:
        self.calls = []
        self.terminal_transaction_count = 0
        self.queued = {}
        self.available_credentials = {
            "openrouter",
            "exa",
            "scrapingdog",
            "deepline",
        }
        self.retry_policy_hashes = {
            provider: sha256_json({"retry": provider})
            for provider in (
                "openrouter",
                "exa",
                "scrapingdog",
                "deepline",
            )
        }
        self.local_credential_ref_hash = _hash("3")
        self.local_egress_proxy_ref_hash = DIRECT_EGRESS_REF_HASH

    @contextmanager
    def transient_terminal_transaction(self):
        self.terminal_transaction_count += 1
        yield

    def health(self):
        return {"status": "ready", "registry_hash": _hash("a")}

    def credential_available(self, *, job_id, slot):
        return slot in self.available_credentials

    def transport_reference_hashes(self, request):
        del request
        return {
            "credential_ref_hash": self.local_credential_ref_hash,
            "egress_proxy_ref_hash": self.local_egress_proxy_ref_hash,
        }

    def execute(self, request):
        request = dict(request)
        self.calls.append(request)
        provider = request["provider_id"]
        queued = self.queued.get(provider) or []
        if queued:
            status, body, terminal_status = queued.pop(0)
        elif provider == "exa":
            status, body, terminal_status = (
                200,
                b'{"costDollars":0.005,"results":[]}',
                "authenticated_response",
            )
        else:
            status, body, terminal_status = (
                200,
                b'{"ok":true}',
                "authenticated_response",
            )
        return self._result(
            request,
            status=status,
            body=body,
            terminal_status=terminal_status,
        )

    @staticmethod
    def _result(request, *, status, body, terminal_status):
        request_hash = sha256_json(
            {
                "provider": request["provider_id"],
                "attempt": request["attempt_number"],
                "kind": "request",
            }
        )
        if terminal_status == "authenticated_response":
            response_hash = sha256_bytes(body)
            attempt = build_transport_attempt(
                request_id=("%032x" % (request["attempt_number"] + 1))[-32:],
                logical_operation_id=request["logical_operation_id"],
                job_id=request["job_id"],
                purpose=request["purpose"],
                provider_id=request["provider_id"],
                attempt_number=request["attempt_number"],
                method=request["method"],
                destination_host="openrouter.ai"
                if request["provider_id"].startswith("openrouter")
                else "api.exa.ai",
                destination_port=443,
                path_hash=_hash("1"),
                nonsecret_headers_hash=_hash("2"),
                body_hash=sha256_bytes(
                    base64.b64decode(request["body_b64"], validate=True)
                ),
                credential_ref_hash=_hash("3"),
                egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
                retry_policy_hash=request["retry_policy_hash"],
                timeout_ms=request["timeout_ms"],
                started_at="2026-07-10T00:00:00Z",
                terminal_status="authenticated_response",
                http_status=status,
                response_hash=response_hash,
                request_artifact_hash=request_hash,
                response_artifact_hash=response_hash,
                tls_peer_chain_hash=_hash("4"),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at="2026-07-10T00:00:01Z",
            )
            return {
                "terminal_status": terminal_status,
                "http_status": status,
                "headers": {},
                "body_b64": base64.b64encode(body).decode("ascii"),
                "encrypted_request_artifact_id": request_hash,
                "encrypted_artifact_id": response_hash,
                "transport_attempt": attempt,
                "evidence_artifact_hashes": [request_hash, response_hash],
            }
        attempt = build_transport_attempt(
            request_id=("%032x" % (request["attempt_number"] + 1))[-32:],
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=_hash("1"),
            nonsecret_headers_hash=_hash("2"),
            body_hash=sha256_bytes(
                base64.b64decode(request["body_b64"], validate=True)
            ),
            credential_ref_hash=_hash("3"),
            egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T00:00:00Z",
            terminal_status="transport_failure",
            http_status=None,
            response_hash=None,
            request_artifact_hash=request_hash,
            response_artifact_hash=None,
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code="timeout",
            completed_at="2026-07-10T00:00:01Z",
        )
        return {
            "terminal_status": "transport_failure",
            "failure_code": "timeout",
            "encrypted_request_artifact_id": request_hash,
            "transport_attempt": attempt,
            "evidence_artifact_hashes": [request_hash],
        }


def _authority(
    *,
    broker=None,
    cache=None,
    artifacts=None,
    artifact_transaction=None,
    clock=None,
    sleeper=None,
    signing_key=None,
    boot_identity=None,
):
    broker = broker or _Broker()
    cache = cache or _CacheStore()
    artifacts = artifacts or _Artifacts()
    key = signing_key or Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = dict(boot_identity or {
        "boot_identity_hash": _hash("b"),
        "signing_pubkey": pubkey,
    })
    authority = ProviderSemanticsAuthorityV2(
        broker=broker,
        cache_store=cache,
        artifact_sink=artifacts.seal,
        boot_identity_supplier=lambda: boot,
        sign_digest=key.sign,
        artifact_transaction=artifact_transaction,
        clock=clock or (lambda: "2026-07-10T00:00:00Z"),
        sleeper=sleeper or (lambda _seconds: None),
    )
    return authority, broker, cache, artifacts


def _request(
    *,
    provider="exa",
    url="https://api.exa.ai/search",
    body=b'{"query":"example"}',
    headers=None,
    logical_operation_id="provider-operation",
    dynamic_route=None,
    job_id="job-provider-semantics",
    purpose="research_lab.company_score.v2",
):
    retry = (
        source_add_dynamic_retry_policy_hash(dynamic_route)
        if dynamic_route is not None
        else sha256_json({"retry": provider})
    )
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": logical_operation_id,
        "job_id": job_id,
        "purpose": purpose,
        "provider_id": provider,
        "attempt_number": 0,
        "method": "POST",
        "url": url,
        "headers": dict(headers or {"X-Research-Lab-Cost-Scope": "icp-1"}),
        "body_b64": base64.b64encode(body).decode("ascii"),
        "timeout_ms": 30000,
        "retry_policy_hash": retry,
    }
    if dynamic_route is not None:
        request["dynamic_route"] = dict(dynamic_route)
    return request


def _dynamic_public_route(*, per_day_quota=1):
    row = {
        "adapter_id": "adapter:public-source",
        "miner_hotkey": "miner-one",
        "provision_status": "provisioned",
        "registry_provider_id": "public_source",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "public_source",
                "base_url": "https://api.public-source.example",
                "auth_kind": "none",
                "auth_name": "",
                "credential_ref": [],
                "per_day_quota": per_day_quota,
                "cost_model": {"est_cost_microusd_per_call": 500},
                "capability_policy": {
                    "routes": [{"method": "POST", "path": "/search"}]
                },
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "public_source.search",
                    "provider_id": "public_source",
                    "method": "POST",
                    "path": "/search",
                    "params": [],
                }
            ],
        },
    }
    return build_source_add_runtime_catalog_v2([row])["routes"][0]


def test_live_record_then_cache_hit_preserves_existing_fingerprint_and_costs():
    authority, broker, cache, _artifacts = _authority()
    request = _request()
    live = authority.execute(request)
    replay = authority.execute(
        {**request, "logical_operation_id": "provider-operation-replay"}
    )

    assert live["terminal_status"] == "authenticated_response"
    assert live["evidence"] == "recorded"
    assert replay["terminal_status"] == "attested_local_response"
    assert replay["evidence"] == "hit"
    assert base64.b64decode(replay["body_b64"]) == base64.b64decode(
        live["body_b64"]
    )
    assert len(broker.calls) == 1
    assert cache.persist_count == 1
    assert broker.calls[0]["headers"] == {"Accept-Encoding": "identity"}
    assert canonical_request_fingerprint("POST", request["url"], b'{"query":"example"}') in {
        key[1] for key in cache.payloads
    }
    live_cost = decode_cost_event_header(
        live["headers"]["X-Research-Lab-Provider-Cost-Event"]
    )
    replay_cost = decode_cost_event_header(
        replay["headers"]["X-Research-Lab-Provider-Cost-Event"]
    )
    assert live_cost["cost_usd"] == 0.005
    assert live_cost["cost_source"] == "exa_cost_dollars"
    assert replay_cost["cost_usd"] == 0.0
    assert replay_cost["cost_source"] == "cache_hit_zero_cost"
    assert live_cost["cost_usd"] == 0.005


def test_infrastructure_routes_bypass_paid_provider_cache_and_outcomes():
    authority, broker, cache, _artifacts = _authority()
    request = {
        **_request(
            provider="supabase",
            url="https://fixture.supabase.co/rest/v1/allocation_inputs",
            body=b"",
            headers={"Accept": "application/json"},
            purpose="research_lab.allocation.v2",
        ),
        "method": "GET",
    }

    first = authority.execute(request)
    second = authority.execute(
        {**request, "logical_operation_id": "infrastructure-read-retry"}
    )

    assert first["terminal_status"] == "authenticated_response"
    assert second["terminal_status"] == "authenticated_response"
    assert first["evidence"] == second["evidence"] == "live_unrecorded"
    assert first["additional_transport_attempts"] == []
    assert second["additional_transport_attempts"] == []
    assert len(broker.calls) == 2
    assert cache.load_count == 0
    assert cache.persist_count == 0
    assert authority.health()["cost_scope_count"] == 0


def test_assigned_provider_keeps_nested_supabase_cache_direct():
    proxy_url = "https://worker:test-secret@proxy.example.com:443"
    job_id = "nested-provider-job"
    purpose = "research_lab.company_score.v2"

    class RecordingTransport:
        def __init__(self, *, fail_supabase_call=0):
            self.calls = []
            self.supabase_calls = 0
            self.fail_supabase_call = fail_supabase_call

        def __call__(self, **request):
            self.calls.append(dict(request))
            is_supabase = ".supabase.co/" in str(request["url"])
            if is_supabase:
                self.supabase_calls += 1
                if self.supabase_calls == self.fail_supabase_call:
                    raise ConnectionRefusedError(111, "test-only direct outage")
                body = b'{"ok":true}'
            else:
                body = b'{"costDollars":0.005,"results":[]}'
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": body,
                "tls_peer_chain_hash": _hash("4"),
                "tls_protocol": "TLSv1.3",
            }

    def make_authority(*, fail_supabase_call=0):
        transport = RecordingTransport(
            fail_supabase_call=fail_supabase_call,
        )
        credentials = {
            "openrouter": "openrouter-secret",
            "exa": "exa-secret",
            "scrapingdog": "scrapingdog-secret",
            "deepline": "deepline-secret",
            "supabase_service_role": "supabase-secret",
            "truelist": "truelist-secret",
        }
        retry_hashes = {
            provider: sha256_json({"retry": provider})
            for provider in BUILTIN_PROVIDER_ROUTES
        }
        broker = ProviderBrokerV2(
            credential_ref_hashes={
                name: credential_reference_hash(value)
                for name, value in credentials.items()
            },
            retry_policy_hashes=retry_hashes,
            transport=transport,
            artifact_sink=lambda body, **_: {
                "artifact_id": sha256_bytes(b"nested:" + body),
                "plaintext_hash": sha256_bytes(body),
            },
            clock=lambda: "2026-07-10T00:00:00Z",
        )
        broker.provision_credentials(credentials)
        broker.provision_job_credential(
            job_id=job_id,
            slot="egress_proxy",
            credential=proxy_url,
            credential_value_hash_expected=credential_value_hash(proxy_url),
        )
        operation_index = 0

        def direct_supabase(*, stage, operation_job_id, operation_purpose):
            nonlocal operation_index
            operation_index += 1
            result = broker.execute(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": (
                        f"{operation_job_id}:nested-supabase:{stage}:"
                        f"{operation_index}"
                    ),
                    "job_id": operation_job_id,
                    "purpose": operation_purpose,
                    "provider_id": "supabase",
                    "attempt_number": 0,
                    "method": "POST",
                    "url": (
                        "https://qplwoislplkcegvdmbim.supabase.co/"
                        "rest/v1/rpc/nested_provider_test"
                    ),
                    "headers": {"content-type": "application/json"},
                    "body_b64": base64.b64encode(b"{}").decode("ascii"),
                    "timeout_ms": 30_000,
                    "retry_policy_hash": retry_hashes["supabase"],
                }
            )
            if result.get("terminal_status") != "authenticated_response":
                raise RuntimeError("nested Supabase operation failed closed")

        class NestedCache(_CacheStore):
            def load(self, *, job_id, purpose, **kwargs):
                direct_supabase(
                    stage="cache-lookup",
                    operation_job_id=job_id,
                    operation_purpose=purpose,
                )
                return super().load(job_id=job_id, purpose=purpose, **kwargs)

            def persist_recorded(self, terminal, *, job_id, purpose, **kwargs):
                direct_supabase(
                    stage="cache-write",
                    operation_job_id=job_id,
                    operation_purpose=purpose,
                )
                return super().persist_recorded(
                    terminal,
                    job_id=job_id,
                    purpose=purpose,
                    **kwargs,
                )

        authority, _, _, _ = _authority(
            broker=broker,
            cache=NestedCache(),
        )
        return authority, broker, transport

    authority, broker, transport = make_authority()
    result = authority.execute(
        _request(job_id=job_id, purpose=purpose)
    )

    assert result["terminal_status"] == "authenticated_response"
    assert [".supabase.co/" in call["url"] for call in transport.calls] == [
        True,
        False,
        True,
    ]
    assert transport.calls[1]["upstream_proxy_url"] == proxy_url
    assert transport.calls[1]["connection_scope"].startswith("sha256:")
    for call in (transport.calls[0], transport.calls[2]):
        assert "upstream_proxy_url" not in call
        assert "connection_scope" not in call
    health = authority.health()
    assert health["stage_counters"]["provider_transport"] == {
        "started": 1,
        "succeeded": 1,
        "failed": 0,
    }
    assert health["stage_counters"]["provider_cache_lookup"]["succeeded"] == 1
    assert health["stage_counters"]["provider_cache_write"]["succeeded"] == 1
    assert broker.release_job_credentials(job_id)["released_slot_count"] == 1



def test_cross_worker_cache_hit_uses_current_job_transport_profile():
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2

    authority, broker, _cache, _artifacts = _authority()
    live = authority.execute(
        _request(job_id="rebenchmark-worker-a", logical_operation_id="source-a")
    )
    worker_b_credential_ref = _hash("8")
    worker_b_proxy_ref = _hash("9")
    broker.local_credential_ref_hash = worker_b_credential_ref
    broker.local_egress_proxy_ref_hash = worker_b_proxy_ref
    replay = authority.execute(
        _request(job_id="rebenchmark-worker-b", logical_operation_id="replay-b")
    )

    context = ExecutionContextV2(
        job_id="rebenchmark-worker-b",
        purpose="research_lab.company_score.v2",
        epoch_id=1,
        provider_credential_ref_hashes={
            "exa": worker_b_credential_ref,
            "egress_proxy": worker_b_proxy_ref,
        },
    )
    context.record_transport(replay["transport_attempt"])

    assert (
        replay["transport_attempt"]["credential_ref_hash"]
        == worker_b_credential_ref
    )
    assert replay["transport_attempt"]["egress_proxy_ref_hash"] == worker_b_proxy_ref
    assert (
        replay["source_record"]["transport_attempt_hash"]
        == live["transport_attempt"]["attempt_hash"]
    )


def test_concurrent_identical_requests_single_flight_one_paid_call(monkeypatch):
    from gateway.tee import provider_semantics_v2 as semantics_module

    broker = _Broker()
    broker_entered = threading.Event()
    release_broker = threading.Event()
    cache_persisted = threading.Event()
    release_cache = threading.Event()
    original_execute = broker.execute

    def blocking_execute(request):
        broker_entered.set()
        assert release_broker.wait(timeout=2.0)
        return original_execute(request)

    broker.execute = blocking_execute
    waiter_entered = threading.Event()
    observed_wait_timeouts = []

    class BlockingCache(_CacheStore):
        def persist_recorded(self, *args, **kwargs):
            cache_persisted.set()
            assert release_cache.wait(timeout=2.0)
            return super().persist_recorded(*args, **kwargs)

    class TrackingEvent(threading.Event):
        def wait(self, timeout=None):
            waiter_entered.set()
            observed_wait_timeouts.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(
        semantics_module,
        "threading",
        SimpleNamespace(
            RLock=threading.RLock,
            Condition=threading.Condition,
            Event=TrackingEvent,
            local=threading.local,
        ),
    )
    cache = BlockingCache()
    authority, _broker, cache, _artifacts = _authority(broker=broker, cache=cache)
    request = _request()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(authority.execute, request)
        assert broker_entered.wait(timeout=2.0)
        second = executor.submit(
            authority.execute,
            {**request, "logical_operation_id": "provider-operation-sibling"},
        )
        assert waiter_entered.wait(timeout=2.0)
        release_broker.set()
        assert cache_persisted.wait(timeout=2.0)
        assert not first.done()
        assert not second.done()
        release_cache.set()
        results = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert len(broker.calls) == 1
    assert cache.persist_count == 1
    cost_events = [
        decode_cost_event_header(
            result["headers"]["X-Research-Lab-Provider-Cost-Event"]
        )
        for result in results
    ]
    assert sum(bool(event["billable"]) for event in cost_events) == 1
    assert sum(float(event["cost_usd"]) for event in cost_events) == 0.005
    assert {result["evidence"] for result in results} == {"recorded", "hit"}
    assert observed_wait_timeouts == [REPLAY_WAIT_SECONDS]




def test_concurrent_preflights_measure_each_worker_profile_without_cache_replay(
    monkeypatch,
):
    monkeypatch.setenv(
        "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
        "0.001",
    )
    broker = _Broker()
    both_live_calls_entered = threading.Barrier(2)
    original_execute = broker.execute
    block_preflight = False

    def blocking_execute(request):
        if block_preflight:
            both_live_calls_entered.wait(timeout=2.0)
        return original_execute(request)

    broker.execute = blocking_execute
    authority, _broker, cache, _artifacts = _authority(broker=broker)
    seeded = authority.execute(
        _request(headers={"Content-Type": "application/json"})
    )
    assert seeded["evidence"] == "recorded"
    broker.calls.clear()
    load_count_before_preflight = cache.load_count
    persist_count_before_preflight = cache.persist_count
    block_preflight = True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                authority.execute,
                _request(
                    logical_operation_id=f"provider-preflight-{index}",
                    job_id=f"job-provider-preflight-{index}",
                    purpose="research_lab.provider_preflight.v2",
                    headers={"Content-Type": "application/json"},
                ),
            )
            for index in range(2)
        ]
        results = [future.result(timeout=3.0) for future in futures]

    assert len(broker.calls) == 2
    assert cache.load_count == load_count_before_preflight
    assert cache.persist_count == persist_count_before_preflight
    assert {result["terminal_status"] for result in results} == {
        "authenticated_response"
    }
    assert {result["evidence"] for result in results} == {"live_unrecorded"}


def test_persistent_cache_survives_authority_restart_without_live_call():
    first, broker, cache, artifacts = _authority()
    request = _request()
    first.execute(request)
    restarted, _broker, _cache, _artifacts = _authority(
        broker=broker,
        cache=cache,
        artifacts=artifacts,
    )
    replay = restarted.execute(
        {**request, "logical_operation_id": "provider-operation-after-restart"}
    )
    assert replay["terminal_status"] == "attested_local_response"
    assert replay["evidence"] == "hit"
    assert len(broker.calls) == 1














@pytest.mark.parametrize(
    ("failure_stage", "cleanup_resource_kind"),
    (
        ("response_stream_cleanup", "network_stream"),
        ("client_transport_cleanup", "client_transport"),
    ),
)
def test_transport_cleanup_failure_seals_only_strict_diagnostic_commitments(
    failure_stage,
    cleanup_resource_kind,
) -> None:
    artifacts = _Artifacts()
    primary_error = ValueError(
        "secret-primary https://user:password@example.invalid/private"
    )
    cleanup_error = OSError(errno.ENOBUFS, "secret-cleanup-token")

    def transport(**_request):
        raise ProviderTransportCleanupError(
            stage=failure_stage,
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )

    credentials = {
        slot: "%s-secret" % slot for slot in expected_provider_credential_slots()
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(secret)
            for slot, secret in credentials.items()
        },
        retry_policy_hashes={
            provider_id: sha256_json({"retry": provider_id})
            for provider_id in BUILTIN_PROVIDER_ROUTES
        },
        transport=transport,
        artifact_sink=artifacts.seal,
        clock=lambda: "2026-07-10T00:00:00Z",
    )
    broker.provision_credentials(credentials)

    result = broker.execute(_request())

    assert result["terminal_status"] == "transport_failure"
    validate_transport_attempt(result["transport_attempt"])
    assert set(result["transport_attempt"]) == {
        "schema_version",
        "request_id",
        "request_hash",
        "logical_operation_id",
        "job_id",
        "purpose",
        "provider_id",
        "attempt_number",
        "method",
        "destination_host",
        "destination_port",
        "path_hash",
        "nonsecret_headers_hash",
        "body_hash",
        "credential_ref_hash",
        "egress_proxy_ref_hash",
        "retry_policy_hash",
        "timeout_ms",
        "started_at",
        "terminal_status",
        "http_status",
        "response_hash",
        "request_artifact_hash",
        "response_artifact_hash",
        "tls_peer_chain_hash",
        "tls_protocol",
        "failure_code",
        "completed_at",
        "attempt_hash",
    }
    assert len(artifacts.values) == 2
    diagnostic_bytes, diagnostic_descriptor = artifacts.values[1]
    diagnostic = json.loads(diagnostic_bytes)
    assert diagnostic == {
        "schema_version": (
            PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "provider": "exa",
        "request_hash": result["transport_attempt"]["request_hash"],
        "attempt_number": 0,
        "failure_stage": failure_stage,
        "outer_error_type": "ProviderTransportCleanupError",
        "primary_error_type": "ValueError",
        "cleanup_error_type": "OSError",
        "cleanup_errno": errno.ENOBUFS,
        "cleanup_resource_kind": cleanup_resource_kind,
    }
    assert validate_provider_transport_failure_diagnostic(diagnostic) == diagnostic
    assert {
        diagnostic_descriptor["artifact_id"],
        diagnostic_descriptor["plaintext_hash"],
        diagnostic_descriptor["ciphertext_hash"],
        diagnostic_descriptor["encryption_context_hash"],
    }.issubset(result["evidence_artifact_hashes"])
    assert "diagnostic" not in result
    assert "secret-primary" not in str(result)
    assert "secret-cleanup-token" not in str(result)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda doc: {**doc, "unknown": "field"},
        lambda doc: {**doc, "failure_stage": "unknown"},
        lambda doc: {**doc, "cleanup_resource_kind": "client_transport"},
        lambda doc: {**doc, "cleanup_errno": 65536},
        lambda doc: {**doc, "outer_error_type": "unsafe.type"},
        lambda doc: {**doc, "cleanup_error_type": ""},
    ),
)
def test_transport_failure_diagnostic_rejects_malformed_projection(mutation):
    document = {
        "schema_version": (
            PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "provider": "exa",
        "request_hash": _hash("a"),
        "attempt_number": 0,
        "failure_stage": "response_stream_cleanup",
        "outer_error_type": "ProviderTransportCleanupError",
        "primary_error_type": "RuntimeError",
        "cleanup_error_type": "OSError",
        "cleanup_errno": errno.ENOBUFS,
        "cleanup_resource_kind": "network_stream",
    }

    with pytest.raises(ProviderBrokerV2Error):
        validate_provider_transport_failure_diagnostic(mutation(document))


def test_model_retry_after_semantics_failure_has_one_terminal_per_intent():
    cache = _CacheStore()
    artifact_commits = 0

    @contextmanager
    def fail_first_artifact_commit():
        nonlocal artifact_commits
        artifact_commits += 1
        yield
        if artifact_commits == 1:
            raise RuntimeError("artifact commit failed")

    authority, broker, _cache, _artifacts = _authority(
        cache=cache,
        artifact_transaction=fail_first_artifact_commit,
    )
    terminals = []
    router = BrokeredProviderTransportV2(
        authority.execute,
        terminal_sink=lambda attempt: terminals.append(dict(attempt)),
    )
    try:
        with router.scope(
            job_id="job-provider-semantics",
            purpose="research_lab.company_score.v2",
            logical_operation_id="model-operation",
            retry_policy_hashes={
                provider: sha256_json({"retry": provider})
                for provider in BUILTIN_PROVIDER_ROUTES
            },
        ):
            request = urllib.request.Request(
                "https://api.exa.ai/search",
                data=b'{"query":"example"}',
                method="POST",
            )
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen(request, timeout=30)
            response = urllib.request.urlopen(request, timeout=30)
            assert response.status == 200
    finally:
        router.restore()

    assert [item["attempt_number"] for item in terminals] == [0, 1]
    assert [item["terminal_status"] for item in terminals] == [
        "transport_failure",
        "attested_local_response",
    ]
    assert len({item["logical_operation_id"] for item in terminals}) == 1
    assert sum(call["provider_id"] == "exa" for call in broker.calls) == 1
    assert cache.persist_count == 1


def test_inter_enclave_replay_preserves_one_provider_cache_record_across_restart():
    authority, broker, cache, artifacts = _authority()
    request = _request(logical_operation_id="provider-operation-replayed-terminal")
    client = object.__new__(AttestedTLSRPCClient)
    delivery = {"attempts": 0, "result": None}

    def _call_once(**_kwargs):
        delivery["attempts"] += 1
        if delivery["result"] is None:
            delivery["result"] = authority.execute(request)
        if delivery["attempts"] < MAX_RPC_DELIVERY_ATTEMPTS:
            raise _RetryableInterEnclaveTransportError(
                "simulated terminal response loss"
            )
        return delivery["result"]

    client._call_once = _call_once
    result = client.call(
        target_physical_role="gateway_coordinator",
        method="provider_execute",
        params=request,
        channel_id="9" * 32,
    )

    assert result["terminal_status"] == "authenticated_response"
    assert delivery["attempts"] == MAX_RPC_DELIVERY_ATTEMPTS
    assert len(broker.calls) == 1
    assert cache.persist_count == 1

    restarted, _broker, _cache, _artifacts = _authority(
        broker=broker,
        cache=cache,
        artifacts=artifacts,
    )
    replay = restarted.execute(
        {**request, "logical_operation_id": "provider-operation-after-restart"}
    )
    assert replay["terminal_status"] == "attested_local_response"
    assert replay["evidence"] == "hit"
    assert len(broker.calls) == 1


def test_oversized_provider_response_is_signed_and_fails_closed(monkeypatch) -> None:
    from gateway.tee import provider_broker_v2

    frame_bytes = 64 * 1024
    reserve_bytes = 8 * 1024
    response_limit = _provider_rpc_response_body_limit(
        frame_bytes=frame_bytes,
        reserve_bytes=reserve_bytes,
    )
    monkeypatch.setattr(
        provider_broker_v2,
        "MAX_RESPONSE_BODY_BYTES",
        response_limit,
    )

    class OversizedTransport:
        def __call__(self, **_request):
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": b"x" * (response_limit + 1),
                "tls_peer_chain_hash": _hash("4"),
                "tls_protocol": "TLSv1.3",
            }

    credentials = {
        slot: "%s-secret" % slot for slot in expected_provider_credential_slots()
    }
    broker_artifacts = _Artifacts()
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={
            name: sha256_json({"retry": name}) for name in BUILTIN_PROVIDER_ROUTES
        },
        transport=OversizedTransport(),
        artifact_sink=broker_artifacts.seal,
        clock=lambda: "2026-07-10T00:00:00Z",
    )
    broker.provision_credentials(credentials)
    authority, _broker, cache, _artifacts = _authority(broker=broker)

    result = authority.execute(_request(job_id="job-oversized-response"))

    assert result["terminal_status"] == "transport_failure"
    assert result["failure_code"] == "response_too_large"
    assert result["transport_attempt"]["failure_code"] == "response_too_large"
    assert broker.health()["terminal_count"] == 1
    assert cache.persist_count == 0
    assert len(broker_artifacts.values) == 2
    diagnostic = json.loads(broker_artifacts.values[1][0])
    assert diagnostic["schema_version"] == (
        PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert diagnostic["failure_stage"] == "provider_request"
    assert len(canonical_json({"result": result, "channel_id": "f" * 32}).encode("utf-8")) <= frame_bytes


def test_semantics_rollback_reseals_cleanup_diagnostic_with_outer_cause():
    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=_hash("b"),
        retention_days=30,
        clock=lambda: datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    sealed = []

    def seal(plaintext, **kwargs):
        descriptor = vault.seal(plaintext, **kwargs)
        sealed.append((bytes(plaintext), dict(kwargs), dict(descriptor)))
        return descriptor

    primary_error = ValueError(
        "secret-primary https://user:password@example.invalid/private"
    )
    cleanup_error = OSError(errno.ENOBUFS, "secret-cleanup-token")

    def transport(**_request):
        raise ProviderTransportCleanupError(
            stage="client_transport_cleanup",
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )

    credentials = {
        slot: "%s-secret" % slot for slot in expected_provider_credential_slots()
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(secret)
            for slot, secret in credentials.items()
        },
        retry_policy_hashes={
            provider_id: sha256_json({"retry": provider_id})
            for provider_id in BUILTIN_PROVIDER_ROUTES
        },
        transport=transport,
        artifact_sink=seal,
        clock=lambda: "2026-07-10T00:00:00Z",
    )
    broker.provision_credentials(credentials)
    prior_broker_results = []
    broker_execute = broker.execute

    def recording_broker_execute(request):
        broker_result = broker_execute(request)
        prior_broker_results.append(dict(broker_result))
        return broker_result

    broker.execute = recording_broker_execute

    transaction_calls = 0

    @contextmanager
    def fail_first_artifact_commit():
        nonlocal transaction_calls
        transaction_calls += 1
        with vault.transient_artifact_transaction():
            yield
            if transaction_calls == 1:
                raise RuntimeError("cache persistence failed")

    authority, _broker, _cache, _artifacts = _authority(
        broker=broker,
        artifacts=SimpleNamespace(seal=seal),
        artifact_transaction=fail_first_artifact_commit,
    )

    result = authority.execute(_request())

    assert result["terminal_status"] == "transport_failure"
    assert result["failure_stage"] == "provider_semantics"
    assert result["failure_error_type"] == "RuntimeError"
    validate_transport_attempt(result["transport_attempt"])
    assert broker.health()["terminal_count"] == 0
    retained = vault.job_artifacts(
        job_id="job-provider-semantics",
        purpose="research_lab.company_score.v2",
    )
    assert sorted(item["artifact_kind"] for item in retained) == [
        "provider_request",
        "provider_transport_failure_diagnostic",
    ]
    retained_by_kind = {item["artifact_kind"]: item for item in retained}
    diagnostic_descriptor = retained_by_kind[
        "provider_transport_failure_diagnostic"
    ]
    diagnostic_plaintext = vault.decrypt_storage_document(
        vault.export_ciphertext(diagnostic_descriptor["artifact_id"])[
            "storage_document"
        ]
    )
    diagnostic = json.loads(diagnostic_plaintext)
    assert diagnostic["outer_error_type"] == "RuntimeError"
    assert diagnostic["primary_error_type"] == "ValueError"
    assert diagnostic["cleanup_error_type"] == "OSError"
    assert diagnostic["cleanup_errno"] == errno.ENOBUFS
    assert diagnostic["cleanup_resource_kind"] == "client_transport"
    validate_provider_transport_failure_diagnostic(diagnostic)
    combined = "%s%s" % (
        canonical_json(result),
        diagnostic_plaintext.decode("utf-8"),
    )
    assert "secret-primary" not in combined
    assert "secret-cleanup-token" not in combined
    assert "password" not in combined
    assert len(prior_broker_results) == 1
    with pytest.raises(RuntimeError, match="second outer commit failed"):
        with vault.transient_artifact_transaction():
            broker.reseal_transport_failure_diagnostic(
                prior_result=prior_broker_results[0],
                outer_error=OSError("second persistence failure"),
            )
            raise RuntimeError("second outer commit failed")
    with vault.transient_artifact_transaction():
        recovered_descriptor = broker.reseal_transport_failure_diagnostic(
            prior_result=prior_broker_results[0],
            outer_error=TimeoutError("third persistence failure"),
        )
    assert recovered_descriptor is not None
    recovered_plaintext = vault.decrypt_storage_document(
        vault.export_ciphertext(recovered_descriptor["artifact_id"])[
            "storage_document"
        ]
    )
    recovered = json.loads(recovered_plaintext)
    assert recovered["outer_error_type"] == "TimeoutError"
    assert recovered["primary_error_type"] == "ValueError"
    assert recovered["cleanup_error_type"] == "OSError"
















def test_preflight_retry_recreates_terminal_record_and_encrypted_artifacts():
    class CountingTransport:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, **request):
            self.calls.append(dict(request))
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": b'{"costDollars":0.005,"results":[]}',
                "tls_peer_chain_hash": _hash("c"),
                "tls_protocol": "TLSv1.3",
            }

    class FailOnceCache(_CacheStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed_once = False

        def persist_recorded(self, *args, **kwargs):
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("cache persistence failed")
            return super().persist_recorded(*args, **kwargs)

    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=_hash("b"),
        retention_days=30,
        clock=lambda: datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    credentials = {
        slot: "%s-secret" % slot for slot in expected_provider_credential_slots()
    }
    transport = CountingTransport()
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(secret)
            for slot, secret in credentials.items()
        },
        retry_policy_hashes={
            provider_id: sha256_json({"retry": provider_id})
            for provider_id in BUILTIN_PROVIDER_ROUTES
        },
        transport=transport,
        artifact_sink=vault.seal,
        clock=lambda: "2026-07-10T00:00:00Z",
    )
    broker.provision_credentials(credentials)
    cache = FailOnceCache()
    authority, _broker, _cache, _artifacts = _authority(
        broker=broker,
        cache=cache,
        artifacts=SimpleNamespace(seal=vault.seal),
        artifact_transaction=vault.transient_artifact_transaction,
    )
    request = _request(
        job_id="job-preflight-artifact-retry",
        purpose="research_lab.company_score.v2",
    )

    failed = authority.execute(request)

    assert failed["terminal_status"] == "transport_failure"
    assert broker.health()["terminal_count"] == 0
    first_artifacts = vault.job_artifacts(
        job_id=request["job_id"],
        purpose=request["purpose"],
    )
    assert [item["artifact_id"] for item in first_artifacts] == [
        failed["encrypted_request_artifact_id"]
    ]
    assert [item["artifact_kind"] for item in first_artifacts] == [
        "provider_request"
    ]

    result = authority.execute(request)

    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 2
    assert broker.health()["terminal_count"] == 1
    assert len(
        vault.job_artifacts(
            job_id=request["job_id"],
            purpose=request["purpose"],
        )
    ) == 3


def test_terminal_record_commits_only_after_artifact_transaction_exit():
    events = []

    class OrderedBroker(_Broker):
        @contextmanager
        def transient_terminal_transaction(self):
            events.append("terminal_enter")
            try:
                yield
            except BaseException:
                events.append("terminal_rollback")
                raise
            else:
                events.append("terminal_commit")

    @contextmanager
    def failing_artifact_transaction():
        events.append("artifact_enter")
        yield
        events.append("artifact_commit_failed")
        raise RuntimeError("artifact commit failed")

    authority, _broker, _cache, _artifacts = _authority(
        broker=OrderedBroker(),
        artifact_transaction=failing_artifact_transaction,
    )

    with pytest.raises(RuntimeError, match="artifact commit failed"):
        authority.execute(_request())

    assert events == [
        "terminal_enter",
        "artifact_enter",
        "artifact_commit_failed",
        "terminal_rollback",
        "terminal_enter",
        "artifact_enter",
        "artifact_commit_failed",
        "terminal_rollback",
    ]


@pytest.mark.parametrize("legacy_header", [False, True])
def test_replay_miss_and_budget_cap_do_not_call_provider_or_cache(legacy_header):
    authority, broker, cache, _artifacts = _authority()
    replay_miss = authority.execute(
        _request(
            headers={
                "X-Research-Lab-Cost-Scope": "replay-only",
                REPLAY_ONLY_HEADER: "1",
            }
        )
    )
    blocked = authority.execute(
        _request(
            logical_operation_id="former-soft-stop",
            body=b'{"query":"blocked"}',
            headers={
                "X-Research-Lab-Cost-Scope": "former-soft-stop",
                "X-Research-Lab-Cost-Cap-Usd": "0",
                **({"X-Research-Lab-Budget-Soft-Stop": "1"} if legacy_header else {}),
            },
        )
    )
    assert replay_miss["http_status"] == 409
    assert replay_miss["evidence"] == "replay_miss"
    assert blocked["http_status"] == 402
    assert blocked["evidence"] == "blocked"
    assert json.loads(base64.b64decode(blocked["body_b64"])) == {
        "endpoint": "/search",
        "error": "research_lab_provider_cost_cap_exceeded",
        "provider": "exa",
    }
    assert "X-Research-Lab-Budget-Soft-Stopped" not in blocked["headers"]
    assert broker.calls == []
    assert cache.persist_count == 0
    assert cache.payloads == {}


def test_dynamic_source_add_keeps_daily_cache_and_enforces_measured_quota():
    authority, broker, _cache, _artifacts = _authority()
    route = _dynamic_public_route(per_day_quota=1)
    first = authority.execute(
        _request(
            provider="public_source",
            url="https://api.public-source.example/search",
            dynamic_route=route,
        )
    )
    cached = authority.execute(
        _request(
            provider="public_source",
            url="https://api.public-source.example/search",
            logical_operation_id="public-source-cache-hit",
            dynamic_route=route,
        )
    )
    quota = authority.execute(
        _request(
            provider="public_source",
            url="https://api.public-source.example/search",
            body=b'{"query":"different"}',
            logical_operation_id="public-source-quota",
            dynamic_route=route,
        )
    )

    assert first["terminal_status"] == "authenticated_response"
    assert cached["terminal_status"] == "attested_local_response"
    assert cached["evidence"] == "hit"
    assert quota["terminal_status"] == "attested_local_response"
    assert quota["http_status"] == 429
    assert quota["evidence"] == "quota_exhausted"
    assert len(broker.calls) == 1
    assert broker.calls[0]["dynamic_route"]["route_hash"] == route["route_hash"]


def test_openrouter_reconciliation_uses_organizer_key_and_exact_cost():
    authority, broker, _cache, _artifacts = _authority()
    broker.queued["openrouter"] = [
        (
            200,
            b'{"id":"gen-cost-1","choices":[],"usage":{"prompt_tokens":13,"completion_tokens":4}}',
            "authenticated_response",
        ),
        (
            200,
            b'{"data":{"cost":0.0027,"tokens_prompt":13,"tokens_completion":4}}',
            "authenticated_response",
        ),
    ]
    request_body = json.dumps(
        {
            "model": "example/model",
            "messages": [{"role": "user", "content": "redacted"}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    result = authority.execute(
        _request(
            provider="openrouter",
            url="https://openrouter.ai/api/v1/chat/completions",
            body=request_body,
            headers={
                "Content-Type": "application/json",
                "X-Research-Lab-Cost-Scope": "openrouter-cost",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
        )
    )
    assert [item["provider_id"] for item in broker.calls] == [
        "openrouter",
        "openrouter",
    ]
    upstream_doc = json.loads(base64.b64decode(broker.calls[0]["body_b64"]))
    assert upstream_doc["usage"]["include"] is True
    assert len(result["additional_transport_attempts"]) == 1
    event = decode_cost_event_header(
        result["headers"]["X-Research-Lab-Provider-Cost-Event"]
    )
    assert event["cost_usd"] == 0.0027
    assert event["cost_source"] == "openrouter_generation_reconciliation"
    assert event["tracking_failed"] is False
    assert event["generation_id"] == "gen-cost-1"
    assert event["prompt_tokens"] == 13
    assert event["completion_tokens"] == 4


def test_openrouter_usage_metadata_rewrite_drops_stale_framing_headers():
    authority, broker, _cache, _artifacts = _authority()
    request_body = json.dumps(
        {
            "model": "example/model",
            "messages": [{"role": "user", "content": "redacted"}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    authority.execute(
        _request(
            provider="openrouter",
            url="https://openrouter.ai/api/v1/chat/completions",
            body=request_body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(request_body)),
                "Transfer-Encoding": "chunked",
                "X-Research-Lab-Cost-Scope": "openrouter-framing",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
        )
    )

    upstream = broker.calls[0]
    upstream_headers = {
        str(name).lower(): str(value)
        for name, value in upstream["headers"].items()
    }
    upstream_body = base64.b64decode(upstream["body_b64"], validate=True)
    assert upstream_body != request_body
    assert json.loads(upstream_body)["usage"]["include"] is True
    assert upstream_headers["content-type"] == "application/json"
    assert "content-length" not in upstream_headers
    assert "transfer-encoding" not in upstream_headers


def test_transport_failure_is_committed_as_error_not_provider_response():
    authority, broker, _cache, _artifacts = _authority()
    broker.queued["exa"] = [(0, b"", "transport_failure")]
    result = authority.execute(_request())
    assert result["terminal_status"] == "transport_failure"
