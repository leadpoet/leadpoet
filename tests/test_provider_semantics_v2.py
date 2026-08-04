from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import threading
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.provider_evidence_proxy import (
    BUDGET_SOFT_STOP_HEADER,
    BUDGET_SOFT_STOP_RESPONSE_HEADER,
    REPLAY_ONLY_HEADER,
)
from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
    credential_reference_hash,
    expected_provider_credential_slots,
)
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_outcome_store_v2 import ProviderOutcomeStoreV2
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


class _OutcomeStore:
    def __init__(self) -> None:
        self.document = None
        self.checkpoint_hash = ""
        self.persist_count = 0
        self.fail_persist = False

    def load_latest(
        self,
        *,
        utc_day,
        job_id,
        purpose,
        operation_suffix="restore",
    ):
        return {
            "found": self.document is not None,
            "state_document": dict(self.document or {}),
            "checkpoint_hash": self.checkpoint_hash,
            "transport_attempts": [],
            "evidence_artifact_hashes": [],
        }

    def persist(
        self,
        document,
        *,
        previous_checkpoint_hash,
        job_id,
        purpose,
        attempt_number=0,
    ):
        del attempt_number
        if self.fail_persist:
            raise RuntimeError("outcome persistence failed")
        expected_sequence = (
            int(self.document["sequence"]) + 1
            if self.document is not None
            else 1
        )
        if (
            previous_checkpoint_hash != self.checkpoint_hash
            or int(document["sequence"]) != expected_sequence
        ):
            return {
                "status": "conflict",
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }
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
            "status": "persisted",
            "checkpoint_hash": self.checkpoint_hash,
            "state_document_hash": document["document_hash"],
            "transport_attempts": [],
            "evidence_artifact_hashes": [self.checkpoint_hash],
        }


class _Broker:
    def __init__(self) -> None:
        self.calls = []
        self.terminal_transaction_count = 0
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

    @contextmanager
    def transient_terminal_transaction(self):
        self.terminal_transaction_count += 1
        yield

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


def _authority(
    *,
    broker=None,
    cache=None,
    artifacts=None,
    outcome_store=None,
    artifact_transaction=None,
    clock=None,
    sleeper=None,
):
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
        artifact_transaction=artifact_transaction,
        clock=clock or (lambda: "2026-07-10T00:00:00Z"),
        sleeper=sleeper or (lambda _seconds: None),
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


def test_concurrent_identical_requests_single_flight_one_paid_call(monkeypatch):
    from gateway.tee import provider_semantics_v2 as semantics_module

    broker = _Broker()
    broker_entered = threading.Event()
    release_broker = threading.Event()
    original_execute = broker.execute

    def blocking_execute(request):
        broker_entered.set()
        assert release_broker.wait(timeout=2.0)
        return original_execute(request)

    broker.execute = blocking_execute
    waiter_entered = threading.Event()

    class TrackingEvent(threading.Event):
        def wait(self, timeout=None):
            waiter_entered.set()
            return super().wait(timeout)

    monkeypatch.setattr(
        semantics_module,
        "threading",
        SimpleNamespace(RLock=threading.RLock, Event=TrackingEvent),
    )
    authority, _broker, cache, _artifacts = _authority(broker=broker)
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

    before_failure = restarted.provider_outcome_snapshot()
    outcome_store.fail_persist = True
    with pytest.raises(RuntimeError, match="persistence failed"):
        restarted.execute(
            _request(
                logical_operation_id="provider-operation-persistence-failure",
                body=b'{"query":"new"}',
            )
        )
    assert restarted.provider_outcome_snapshot() == before_failure


def test_provider_outcome_conflict_rebases_onto_durable_head() -> None:
    outcome_store = _OutcomeStore()
    first, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store
    )
    stale, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store
    )

    first.execute(_request(job_id="job-first"))
    stale.execute(
        _request(
            job_id="job-stale",
            logical_operation_id="provider-operation-stale",
            body=b'{"query":"second"}',
        )
    )

    digest = stale.provider_outcome_snapshot()["provider_outcome_digest"]
    assert outcome_store.document["sequence"] == 2
    assert outcome_store.persist_count == 2
    assert digest["providers"]["exa"]["call_count"] == 2
    assert digest["providers"]["exa"]["live_call_count"] == 2


def test_provider_outcome_busy_empty_head_backs_off_and_retries() -> None:
    class BusyOnceOutcomeStore(_OutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self.busy = True
            self.persist_attempts = 0
            self.attempt_numbers = []

        def persist(
            self,
            document,
            *,
            previous_checkpoint_hash,
            job_id,
            purpose,
            attempt_number=0,
        ):
            self.persist_attempts += 1
            self.attempt_numbers.append(attempt_number)
            if self.busy:
                self.busy = False
                return {
                    "status": "busy",
                    "transport_attempts": [],
                    "evidence_artifact_hashes": [],
                }
            return super().persist(
                document,
                previous_checkpoint_hash=previous_checkpoint_hash,
                job_id=job_id,
                purpose=purpose,
                attempt_number=attempt_number,
            )

    sleeps = []
    outcome_store = BusyOnceOutcomeStore()
    authority, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store,
        sleeper=sleeps.append,
    )

    authority.execute(_request(job_id="job-busy-empty-head"))

    assert outcome_store.persist_attempts == 2
    assert outcome_store.persist_count == 1
    assert outcome_store.document["sequence"] == 1
    assert outcome_store.attempt_numbers == [0, 1]
    assert len(sleeps) == 1
    assert 0.01 <= sleeps[0] <= 0.02


def test_provider_outcome_persistent_contention_is_bounded_and_fails_closed() -> None:
    class AlwaysBusyOutcomeStore(_OutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self.persist_attempts = 0
            self.load_attempts = 0

        def persist(
            self,
            document,
            *,
            previous_checkpoint_hash,
            job_id,
            purpose,
            attempt_number=0,
        ):
            del document, previous_checkpoint_hash, job_id, purpose, attempt_number
            self.persist_attempts += 1
            return {
                "status": "busy",
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }

        def load_latest(
            self,
            *,
            utc_day,
            job_id,
            purpose,
            operation_suffix="restore",
        ):
            del utc_day, job_id, purpose, operation_suffix
            self.load_attempts += 1
            return {
                "found": False,
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }

    sleeps = []
    outcome_store = AlwaysBusyOutcomeStore()
    authority, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store,
        sleeper=sleeps.append,
    )

    with pytest.raises(
        ProviderSemanticsV2Error,
        match="provider outcome checkpoint contention did not converge",
    ):
        authority.execute(_request(job_id="job-persistent-contention"))

    assert outcome_store.persist_attempts == 64
    assert outcome_store.load_attempts == 1  # startup restore only
    assert len(sleeps) == 63
    assert (
        authority.provider_outcome_snapshot()["provider_outcome_digest"].get(
            "providers", {}
        )
        == {}
    )


def test_provider_outcome_contention_converges_across_25_writers() -> None:
    writer_count = 25

    class RoundContentionOutcomeStore(_OutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self._condition = threading.Condition()
            self._arrivals = {}
            self._round_results = {}

        def load_latest(
            self,
            *,
            utc_day,
            job_id,
            purpose,
            operation_suffix="restore",
        ):
            with self._condition:
                return super().load_latest(
                    utc_day=utc_day,
                    job_id=job_id,
                    purpose=purpose,
                    operation_suffix=operation_suffix,
                )

        def persist(
            self,
            document,
            *,
            previous_checkpoint_hash,
            job_id,
            purpose,
            attempt_number=0,
        ):
            del attempt_number
            sequence = int(document["sequence"])
            with self._condition:
                expected_sequence = (
                    int(self.document["sequence"]) + 1
                    if self.document is not None
                    else 1
                )
                if (
                    sequence != expected_sequence
                    or previous_checkpoint_hash != self.checkpoint_hash
                ):
                    return {
                        "status": "conflict",
                        "transport_attempts": [],
                        "evidence_artifact_hashes": [],
                    }
                arrivals = self._arrivals.setdefault(sequence, {})
                arrivals[job_id] = dict(document)
                expected_arrivals = writer_count - (sequence - 1)
                if len(arrivals) == expected_arrivals:
                    winner = sorted(arrivals)[0]
                    self.persist_count += 1
                    self.document = dict(arrivals[winner])
                    self.checkpoint_hash = sha256_json(
                        {
                            "sequence": sequence,
                            "state": self.document["document_hash"],
                            "previous": previous_checkpoint_hash,
                        }
                    )
                    self._round_results[sequence] = (
                        winner,
                        self.checkpoint_hash,
                    )
                    self._condition.notify_all()
                assert self._condition.wait_for(
                    lambda: sequence in self._round_results,
                    timeout=5.0,
                )
                winner, checkpoint_hash = self._round_results[sequence]
                if job_id != winner:
                    return {
                        "status": "conflict",
                        "transport_attempts": [],
                        "evidence_artifact_hashes": [],
                    }
                return {
                    "status": "persisted",
                    "checkpoint_hash": checkpoint_hash,
                    "state_document_hash": document["document_hash"],
                    "transport_attempts": [],
                    "evidence_artifact_hashes": [checkpoint_hash],
                }

    outcome_store = RoundContentionOutcomeStore()
    authorities = [
        _authority(outcome_store=outcome_store)[0]
        for _index in range(writer_count)
    ]
    start = threading.Barrier(writer_count)

    def execute(index):
        start.wait(timeout=5.0)
        return authorities[index].execute(
            _request(
                job_id="job-contention-%02d" % index,
                logical_operation_id="provider-contention-%02d" % index,
                body=('{"query":"contention-%02d"}' % index).encode(),
            )
        )

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        results = list(executor.map(execute, range(writer_count)))

    assert len(results) == writer_count
    assert outcome_store.persist_count == writer_count
    assert outcome_store.document["sequence"] == writer_count
    assert outcome_store.document["totals"]["call_count"] == writer_count


def test_provider_outcome_structured_conflict_rebases_without_head_read() -> None:
    class EmbeddedHeadOutcomeStore(_OutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self.load_attempts = 0

        def load_latest(
            self,
            *,
            utc_day,
            job_id,
            purpose,
            operation_suffix="restore",
        ):
            self.load_attempts += 1
            return super().load_latest(
                utc_day=utc_day,
                job_id=job_id,
                purpose=purpose,
                operation_suffix=operation_suffix,
            )

        def persist(
            self,
            document,
            *,
            previous_checkpoint_hash,
            job_id,
            purpose,
            attempt_number=0,
        ):
            result = super().persist(
                document,
                previous_checkpoint_hash=previous_checkpoint_hash,
                job_id=job_id,
                purpose=purpose,
                attempt_number=attempt_number,
            )
            if result["status"] == "conflict":
                return {
                    **result,
                    "head_checkpoint_hash": self.checkpoint_hash,
                    "head_state_document": dict(self.document),
                }
            return result

    outcome_store = EmbeddedHeadOutcomeStore()
    first, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store
    )
    stale, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store
    )
    assert outcome_store.load_attempts == 2

    first.execute(_request(job_id="job-embedded-first"))
    stale.execute(
        _request(
            job_id="job-embedded-stale",
            logical_operation_id="provider-operation-embedded-stale",
            body=b'{"query":"embedded-second"}',
        )
    )

    assert outcome_store.load_attempts == 2
    assert outcome_store.document["sequence"] == 2
    assert (
        stale.provider_outcome_snapshot()["provider_outcome_digest"]["providers"][
            "exa"
        ]["call_count"]
        == 2
    )


def test_real_outcome_store_busy_retry_uses_distinct_broker_attempts() -> None:
    class LocalProviderTransport:
        def __init__(self) -> None:
            self.calls = []
            self.rows = {}
            self.append_count = 0

        def __call__(self, **request):
            self.calls.append(dict(request))
            parsed = urlsplit(request["url"])
            if parsed.hostname == "api.exa.ai":
                body = b'{"costDollars":0.005,"results":[]}'
            elif parsed.hostname == "qplwoislplkcegvdmbim.supabase.co":
                if request["method"] == "POST":
                    self.append_count += 1
                    row = json.loads(request["body"].decode("utf-8"))[
                        "checkpoint_row"
                    ]
                    if self.append_count == 1:
                        body = json.dumps(
                            {
                                "status": "busy",
                                "checkpoint_hash": row["checkpoint_hash"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    else:
                        self.rows[
                            (
                                row["artifact_master_key_ref_hash"],
                                row["utc_day"],
                                int(row["sequence"]),
                            )
                        ] = row
                        body = json.dumps(
                            {
                                "status": "inserted",
                                "checkpoint_hash": row["checkpoint_hash"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                else:
                    query = parse_qs(parsed.query)
                    day = query["utc_day"][0].split("eq.", 1)[1]
                    key_hash = query["artifact_master_key_ref_hash"][0].split(
                        "eq.", 1
                    )[1]
                    rows = [
                        row
                        for (row_key_hash, row_day, _sequence), row in self.rows.items()
                        if row_key_hash == key_hash and row_day == day
                    ]
                    if "sequence" in query:
                        sequence = int(query["sequence"][0].split("eq.", 1)[1])
                        rows = [
                            row
                            for row in rows
                            if int(row["sequence"]) == sequence
                        ]
                    if query.get("order") == ["sequence.desc"]:
                        rows.sort(
                            key=lambda row: int(row["sequence"]),
                            reverse=True,
                        )
                    rows = rows[: int(query.get("limit", ["2"])[0])]
                    body = json.dumps(
                        rows,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
            else:
                raise AssertionError("unexpected local provider destination")
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": body,
                "tls_peer_chain_hash": _hash("c"),
                "tls_protocol": "TLSv1.3",
            }

    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=_hash("b"),
        retention_days=30,
        clock=lambda: datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    credentials = {
        slot: "%s-secret" % slot for slot in expected_provider_credential_slots()
    }
    transport = LocalProviderTransport()
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
    outcome_store = ProviderOutcomeStoreV2(broker=broker, vault=vault)
    authority, _broker, _cache, _artifacts = _authority(
        broker=broker,
        artifacts=SimpleNamespace(seal=vault.seal),
        outcome_store=outcome_store,
        artifact_transaction=vault.transient_artifact_transaction,
    )

    result = authority.execute(_request(job_id="job-real-busy-retry"))

    append_attempts = [
        attempt
        for attempt in result["additional_transport_attempts"]
        if str(attempt["logical_operation_id"]).endswith(":append")
    ]
    assert [attempt["attempt_number"] for attempt in append_attempts] == [0, 1]
    assert transport.append_count == 2
    assert len(transport.rows) == 1
    assert authority.provider_outcome_snapshot()["provider_outcome_digest"][
        "sidecar_sequence"
    ] == 1


def test_provider_outcome_rollback_preserves_original_error_across_midnight() -> None:
    timestamps = ["2026-07-10T23:59:59Z"]
    outcome_store = _OutcomeStore()
    authority, _broker, _cache, _artifacts = _authority(
        outcome_store=outcome_store,
        clock=lambda: timestamps[0],
    )
    authority.execute(_request())

    timestamps[0] = "2026-07-11T00:00:01Z"
    outcome_store.fail_persist = True
    with pytest.raises(RuntimeError, match="outcome persistence failed"):
        authority.execute(
            _request(
                logical_operation_id="provider-operation-next-day",
                body=b'{"query":"next-day"}',
            )
        )

    restored = authority._outcome_ledger.state_document()
    assert restored["utc_day"] == "2026-07-10"
    assert restored["sequence"] == 1


def test_failed_outcome_persistence_removes_uncommitted_job_artifacts():
    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=_hash("b"),
        retention_days=30,
        clock=lambda: datetime(2026, 7, 10, tzinfo=timezone.utc),
    )

    class FailingOutcomeStore(_OutcomeStore):
        def persist(
            self,
            document,
            *,
            previous_checkpoint_hash,
            job_id,
            purpose,
            attempt_number=0,
        ):
            del document, previous_checkpoint_hash, attempt_number
            vault.seal(
                b"uncommitted provider outcome checkpoint",
                job_id=job_id,
                purpose=purpose,
                artifact_kind="provider_outcome_checkpoint",
            )
            raise RuntimeError("outcome persistence failed")

    authority, _broker, _cache, _artifacts = _authority(
        artifacts=SimpleNamespace(seal=vault.seal),
        outcome_store=FailingOutcomeStore(),
        artifact_transaction=vault.transient_artifact_transaction,
    )
    with pytest.raises(RuntimeError, match="outcome persistence failed"):
        authority.execute(_request())

    assert vault.job_artifacts(
        job_id="job-provider-semantics",
        purpose="research_lab.company_score.v2",
    ) == ()


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

    class FailOnceOutcomeStore(_OutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed_once = False

        def persist(self, *args, **kwargs):
            if not self.failed_once:
                self.failed_once = True
                self.fail_persist = True
                try:
                    return super().persist(*args, **kwargs)
                finally:
                    self.fail_persist = False
            return super().persist(*args, **kwargs)

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
    outcome_store = FailOnceOutcomeStore()
    authority, _broker, _cache, _artifacts = _authority(
        broker=broker,
        artifacts=SimpleNamespace(seal=vault.seal),
        outcome_store=outcome_store,
        artifact_transaction=vault.transient_artifact_transaction,
    )
    request = _request(
        job_id="job-preflight-artifact-retry",
        purpose="research_lab.provider_preflight.v2",
    )

    with pytest.raises(RuntimeError, match="outcome persistence failed"):
        authority.execute(request)

    assert broker.health()["terminal_count"] == 0
    assert vault.job_artifacts(
        job_id=request["job_id"],
        purpose=request["purpose"],
    ) == ()

    result = authority.execute(request)

    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 2
    assert broker.health()["terminal_count"] == 1
    assert len(
        vault.job_artifacts(
            job_id=request["job_id"],
            purpose=request["purpose"],
        )
    ) == 2


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
    ]


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
