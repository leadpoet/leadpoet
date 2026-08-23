from __future__ import annotations

from dataclasses import dataclass

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentRuntimeConfig,
    RoutingExperimentRuntimeError,
    RoutingExperimentService,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExecutionRequestLease,
    RoutingExperimentExecutionClaim,
    RoutingExperimentStoreError,
    routing_claim_fence_hash_v3,
)


def _hash(char: str) -> str:
    return "sha256:" + char * 64


@dataclass(frozen=True)
class _Spec:
    value: str = _hash("a")

    def experiment_hash(self) -> str:
        return self.value


def _lease(*, experiment_hash: str = _hash("a"), worker_ref: str = "worker-1"):
    return RoutingExecutionRequestLease(
        request_hash=_hash("b"),
        experiment_hash=experiment_hash,
        lease_hash=_hash("c"),
        worker_ref=worker_ref,
        lease_generation=2,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )


class _Store:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def claim_execution(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


class _RpcResponse:
    def __init__(self, data):
        self.data = data


class _RpcClient:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, name, params):
        self.calls.append((name, dict(params)))
        result = (
            {
                "event_hash": params["p_event_hash"],
                "idempotent": False,
            }
            if name == "research_lab_routing_append_fenced_event_v3"
            else self.result
        )
        return type("_Call", (), {"execute": lambda _self: _RpcResponse(result)})()


def test_claim_is_deterministic_and_contains_only_queue_lease_fence_fields():
    lease = _lease()
    store = _Store(
        {
            "claimed": True,
            "claim_generation": 1,
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "lease_expires_at": lease.lease_expires_at,
        }
    )
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )

    first = service.claim_execution(spec=_Spec(), worker_ref="worker-1", lease=lease)
    second = service.claim_execution(spec=_Spec(), worker_ref="worker-1", lease=lease)

    assert first.claim_key == second.claim_key
    assert first.claim_fence_hash == routing_claim_fence_hash_v3(
        experiment_hash=_hash("a"), claim_key=first.claim_key, claim_generation=1
    )
    assert first.request_hash == lease.request_hash
    assert first.lease_hash == lease.lease_hash
    assert first.lease_generation == lease.lease_generation
    assert first.worker_ref == lease.worker_ref
    assert all(
        "token" not in key and "capability" not in key and "nonce" not in key
        for call in store.calls
        for key in call
    )
    assert store.calls[0]["request_hash"] == lease.request_hash
    assert store.calls[0]["lease_hash"] == lease.lease_hash
    assert store.calls[0]["lease_generation"] == lease.lease_generation


def test_claim_rejects_queue_lease_for_another_experiment_before_store_call():
    store = _Store({"claimed": True, "claim_generation": 1})
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="another experiment"):
        service.claim_execution(
            spec=_Spec(),
            worker_ref="worker-1",
            lease=_lease(experiment_hash=_hash("d")),
        )
    assert store.calls == []


def test_claim_rejects_queue_lease_for_another_worker_before_store_call():
    store = _Store({"claimed": True, "claim_generation": 1})
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="another worker"):
        service.claim_execution(
            spec=_Spec(),
            worker_ref="worker-1",
            lease=_lease(worker_ref="worker-2"),
        )
    assert store.calls == []


def test_claim_identity_rejects_authority_response_for_another_queue_lease():
    lease = _lease()
    store = _Store(
        {
            "claimed": True,
            "claim_generation": 1,
            "request_hash": _hash("d"),
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
        }
    )
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="queue identity"):
        service.claim_execution(spec=_Spec(), worker_ref="worker-1", lease=lease)


@pytest.mark.parametrize("lease_expires_at", [None, "later", "2099-01-01T00:00:00"])
def test_claim_rejects_untrusted_authority_expiry_before_execution(
    lease_expires_at,
):
    lease = _lease()
    store = _Store(
        {
            "claimed": True,
            "claim_generation": 1,
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "lease_expires_at": lease_expires_at,
        }
    )
    service = RoutingExperimentService(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=store,
    )
    with pytest.raises(RoutingExperimentStoreError, match="claim lease expiry"):
        service.claim_execution(spec=_Spec(), worker_ref="worker-1", lease=lease)


def test_store_uses_bearer_free_v3_rpc_for_claim_and_fenced_event():
    lease = _lease()
    claim_key = _hash("d")
    claim = RoutingExperimentExecutionClaim(
        _hash("a"),
        claim_key,
        1,
        routing_claim_fence_hash_v3(
            experiment_hash=_hash("a"),
            claim_key=claim_key,
            claim_generation=1,
        ),
        lease.request_hash,
        lease.lease_hash,
        lease.lease_generation,
        lease.worker_ref,
        lease.lease_expires_at,
    )
    client = _RpcClient(
        {
            "claimed": True,
            "claim_key": claim_key,
            "claim_generation": 1,
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "lease_expires_at": lease.lease_expires_at,
        }
    )
    from gateway.research_lab.routing_experiment_store import SupabaseRoutingExperimentStore

    store = SupabaseRoutingExperimentStore(client)
    store.claim_execution(
        experiment_hash=_hash("a"),
        request_hash=lease.request_hash,
        lease_hash=lease.lease_hash,
        lease_generation=lease.lease_generation,
        claim_key=claim_key,
        worker_ref=lease.worker_ref,
        lease_seconds=30,
        claim_doc={
            "schema_version": "leadpoet.research_lab.routing_claim.v3",
            "request_hash": lease.request_hash,
            "lease_hash": lease.lease_hash,
            "lease_generation": lease.lease_generation,
            "worker_ref": lease.worker_ref,
        },
    )
    store.append_event(
        experiment_hash=_hash("a"),
        event_type="run_started",
        event_doc={"worker_ref": lease.worker_ref},
        claim=claim,
    )
    assert [name for name, _params in client.calls] == [
        "research_lab_routing_claim_execution_v3",
        "research_lab_routing_append_fenced_event_v3",
    ]
    assert all(
        "token" not in key and "capability" not in key and "nonce" not in key
        for _name, params in client.calls
        for key in params
    )
