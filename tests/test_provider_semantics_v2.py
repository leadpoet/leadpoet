from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.provider_evidence_proxy import (
    BUDGET_SOFT_STOP_HEADER,
    BUDGET_SOFT_STOP_RESPONSE_HEADER,
    REPLAY_ONLY_HEADER,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from gateway.tee.provider_semantics_v2 import ProviderSemanticsAuthorityV2
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
)
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
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
        self.persist_count = 0

    def load(self, *, utc_day, request_fingerprint, job_id, purpose):
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


class _OutcomeStore:
    def __init__(self) -> None:
        self.document = None
        self.checkpoint_hash = ""
        self.persist_count = 0
        self.fail_persist = False

    def load_latest(self, *, utc_day, job_id, purpose):
        return {
            "found": self.document is not None,
            "state_document": dict(self.document or {}),
            "checkpoint_hash": self.checkpoint_hash,
            "transport_attempts": [],
            "evidence_artifact_hashes": [],
        }

    def persist(self, document, *, previous_checkpoint_hash, job_id, purpose):
        if self.fail_persist:
            raise RuntimeError("outcome persistence failed")
        if previous_checkpoint_hash != self.checkpoint_hash:
            raise RuntimeError("outcome checkpoint ancestry differs")
        self.persist_count += 1
        self.document = dict(document)
        self.checkpoint_hash = sha256_json(
            {
                "sequence": document["sequence"],
                "state": document["document_hash"],
                "previous": previous_checkpoint_hash,
            }
        )
        return {
            "checkpoint_hash": self.checkpoint_hash,
            "state_document_hash": document["document_hash"],
            "transport_attempts": [],
            "evidence_artifact_hashes": [self.checkpoint_hash],
        }


class _Broker:
    def __init__(self) -> None:
        self.calls = []
        self.queued = {}
        self.available_credentials = {
            "openrouter",
            "openrouter_management",
            "exa",
            "scrapingdog",
            "deepline",
        }
        self.retry_policy_hashes = {
            provider: sha256_json({"retry": provider})
            for provider in (
                "openrouter",
                "openrouter_management",
                "exa",
                "scrapingdog",
                "deepline",
            )
        }

    def health(self):
        return {"status": "ready", "registry_hash": _hash("a")}

    def credential_available(self, *, job_id, slot):
        return slot in self.available_credentials

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


def _authority(*, broker=None, cache=None, artifacts=None, outcome_store=None):
    broker = broker or _Broker()
    cache = cache or _CacheStore()
    artifacts = artifacts or _Artifacts()
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = {
        "boot_identity_hash": _hash("b"),
        "signing_pubkey": pubkey,
    }
    authority = ProviderSemanticsAuthorityV2(
        broker=broker,
        cache_store=cache,
        artifact_sink=artifacts.seal,
        boot_identity_supplier=lambda: boot,
        sign_digest=key.sign,
        clock=lambda: "2026-07-10T00:00:00Z",
        sleeper=lambda _seconds: None,
        outcome_store=outcome_store,
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
):
    retry = (
        source_add_dynamic_retry_policy_hash(dynamic_route)
        if dynamic_route is not None
        else sha256_json({"retry": provider})
    )
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": logical_operation_id,
        "job_id": "job-provider-semantics",
        "purpose": "research_lab.company_score.v2",
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
    digest = authority.provider_outcome_snapshot()["provider_outcome_digest"]
    assert digest["providers"]["exa"]["call_count"] == 2
    assert digest["providers"]["exa"]["live_call_count"] == 1
    assert digest["providers"]["exa"]["cache_hit_count"] == 1
    assert digest["providers"]["exa"]["measured_spend_microusd"] == 5000


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


def test_provider_outcome_state_survives_restart_and_persistence_is_fail_closed():
    outcome_store = _OutcomeStore()
    first, broker, cache, artifacts = _authority(outcome_store=outcome_store)
    first.execute(_request())
    restarted, _broker, _cache, _artifacts = _authority(
        broker=broker,
        cache=cache,
        artifacts=artifacts,
        outcome_store=outcome_store,
    )
    restarted.execute(
        _request(logical_operation_id="provider-operation-after-restart")
    )

    digest = restarted.provider_outcome_snapshot()["provider_outcome_digest"]
    assert digest["providers"]["exa"]["call_count"] == 2
    assert digest["providers"]["exa"]["live_call_count"] == 1
    assert digest["providers"]["exa"]["cache_hit_count"] == 1
    assert outcome_store.persist_count == 2

    outcome_store.fail_persist = True
    with pytest.raises(RuntimeError, match="persistence failed"):
        restarted.execute(
            _request(
                logical_operation_id="provider-operation-persistence-failure",
                body=b'{"query":"new"}',
            )
        )


def test_replay_only_miss_and_budget_modes_do_not_call_provider():
    authority, broker, _cache, _artifacts = _authority()
    replay_miss = authority.execute(
        _request(
            headers={
                "X-Research-Lab-Cost-Scope": "replay-only",
                REPLAY_ONLY_HEADER: "1",
            }
        )
    )
    soft = authority.execute(
        _request(
            logical_operation_id="soft-stop",
            body=b'{"query":"soft"}',
            headers={
                "X-Research-Lab-Cost-Scope": "soft-stop",
                "X-Research-Lab-Cost-Cap-Usd": "0",
                BUDGET_SOFT_STOP_HEADER: "1",
            },
        )
    )
    hard = authority.execute(
        _request(
            logical_operation_id="hard-stop",
            body=b'{"query":"hard"}',
            headers={
                "X-Research-Lab-Cost-Scope": "hard-stop",
                "X-Research-Lab-Cost-Cap-Usd": "0",
            },
        )
    )
    assert replay_miss["http_status"] == 409
    assert replay_miss["evidence"] == "replay_miss"
    assert soft["http_status"] == 200
    assert soft["evidence"] == "budget_soft_stop"
    assert soft["headers"][BUDGET_SOFT_STOP_RESPONSE_HEADER] == "1"
    assert hard["http_status"] == 402
    assert hard["evidence"] == "blocked"
    assert broker.calls == []


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


def test_openrouter_reconciliation_uses_management_then_runtime_and_exact_cost():
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
    broker.queued["openrouter_management"] = [
        (403, b'{"error":"forbidden"}', "authenticated_response")
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
        "openrouter_management",
        "openrouter",
    ]
    upstream_doc = json.loads(base64.b64decode(broker.calls[0]["body_b64"]))
    assert upstream_doc["usage"]["include"] is True
    assert len(result["additional_transport_attempts"]) == 2
    event = decode_cost_event_header(
        result["headers"]["X-Research-Lab-Provider-Cost-Event"]
    )
    assert event["cost_usd"] == 0.0027
    assert event["cost_source"] == "openrouter_generation_reconciliation"
    assert event["tracking_failed"] is False
    assert event["generation_id"] == "gen-cost-1"
    assert event["prompt_tokens"] == 13
    assert event["completion_tokens"] == 4


def test_transport_failure_is_committed_as_error_not_provider_response():
    authority, broker, _cache, _artifacts = _authority()
    broker.queued["exa"] = [(0, b"", "transport_failure")]
    result = authority.execute(_request())
    assert result["terminal_status"] == "transport_failure"
    digest = authority.provider_outcome_snapshot()["provider_outcome_digest"]
    exa = digest["providers"]["exa"]
    assert exa["call_count"] == 1
    assert exa["live_call_count"] == 1
    assert exa["error_count"] == 1
    assert exa["status_histogram"] == {"502": 1}
    assert exa["measured_spend_microusd"] == 0
