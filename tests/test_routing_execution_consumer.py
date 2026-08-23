from __future__ import annotations

import threading
import time
from types import MappingProxyType, SimpleNamespace

import pytest

import gateway.research_lab.routing_execution_consumer as consumer_module

from gateway.research_lab.routing_execution_consumer import (
    REVIEWED_ROUTING_FACTORY_NAME,
    RoutingExecutionConsumerConfig,
    RoutingExecutionConsumerError,
    RoutingExecutionRequestConsumer,
    build_reviewed_routing_execution_consumer,
)
from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentDeferredRecoveryError,
    RoutingExperimentRuntimeConfig,
    RoutingExperimentTerminalRecoveryError,
)
from gateway.research_lab.routing_experiment_store import RoutingExecutionRequestLease
from gateway.research_lab.routing_experiment_worker import (
    RoutingExperimentCoordinator,
    RoutingExperimentWorkerError,
)


def _hash(char: str) -> str:
    return "sha256:" + char * 64


AUTHORITY_ENV = {
    "RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED": "true",
    "RESEARCH_LAB_ROUTING_EXPERIMENT_AUTHORITY": "attested",
    "RESEARCH_LAB_ROUTING_EXPERIMENT_CLAIM_AUTHORITY": "supabase_v3",
    "RESEARCH_LAB_ROUTING_EXPERIMENT_ATTESTATION_AUTHORITY": "tee_v2",
    "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED": "false",
    "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED": "true",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "service-role-test-only",
}


def _lease(char: str = "a", worker: str = "routing-execution-consumer") -> RoutingExecutionRequestLease:
    return RoutingExecutionRequestLease(
        request_hash=_hash(char),
        experiment_hash=_hash("b"),
        lease_hash=_hash("c"),
        worker_ref=worker,
        lease_generation=1,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )


class _Store:
    def __init__(self, leases=()):
        self.leases = list(leases)
        self.claim_calls = 0
        self.renew_calls = 0
        self.closed: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def claim_pending_execution_requests(self, *, worker_ref, batch_size, lease_seconds):
        del batch_size, lease_seconds
        with self._lock:
            self.claim_calls += 1
            selected = [item for item in self.leases if item.worker_ref == worker_ref]
            self.leases = []
            return tuple(selected)

    def renew_execution_request_lease(self, *, lease, lease_seconds):
        del lease, lease_seconds
        self.renew_calls += 1
        return {"renewed": True, "request_hash": _hash("a"), "lease_generation": 1, "lease_expires_at": "later"}

    def close_execution_request_lease(self, *, lease, close_reason):
        self.closed.append((lease.request_hash, close_reason))
        return {
            "closed": True,
            "stale": False,
            "request_hash": lease.request_hash,
            "lease_generation": lease.lease_generation,
            "close_reason": close_reason,
        }


class _Coordinator(RoutingExperimentCoordinator):
    def __init__(self, action, *, factory=None):
        self.action = action
        self._factories = {
            REVIEWED_ROUTING_FACTORY_NAME: factory
            or SimpleNamespace(
                name=REVIEWED_ROUTING_FACTORY_NAME,
                validate_readiness=lambda: None,
            )
        }

    def resume(self, *, experiment_hash, factory_name, lease=None):
        assert factory_name == REVIEWED_ROUTING_FACTORY_NAME
        assert isinstance(lease, RoutingExecutionRequestLease)
        return self.action(experiment_hash)


def _consumer(
    store,
    action=lambda _experiment_hash: object(),
    *,
    env=None,
    factory=None,
):
    return RoutingExecutionRequestConsumer(
        config=RoutingExecutionConsumerConfig(enabled=True, poll_seconds=0.1),
        runtime_config=RoutingExperimentRuntimeConfig(
            enabled=True,
            worker_lease_seconds=30,
            attested_authority_mode="attested",
        ),
        store=store,
        coordinator=_Coordinator(action, factory=factory),
        environment=env or AUTHORITY_ENV,
    )


def test_disabled_consumer_fails_before_queue_claim_or_provider_factory():
    store = _Store([_lease()])
    with pytest.raises(RoutingExecutionConsumerError, match="disabled"):
        RoutingExecutionRequestConsumer(
            config=RoutingExecutionConsumerConfig(enabled=False),
            runtime_config=RoutingExperimentRuntimeConfig(enabled=True),
            store=store,
            coordinator=_Coordinator(lambda _hash: pytest.fail("provider path reached")),
            environment=AUTHORITY_ENV,
        )
    assert store.claim_calls == 0


def test_missing_reviewed_factory_fails_before_claim():
    store = _Store([_lease()])
    coordinator = object.__new__(RoutingExperimentCoordinator)
    coordinator._factories = {}
    with pytest.raises(RoutingExecutionConsumerError, match="factory registry"):
        RoutingExecutionRequestConsumer(
            config=RoutingExecutionConsumerConfig(enabled=True),
            runtime_config=RoutingExperimentRuntimeConfig(
                enabled=True, attested_authority_mode="attested"
            ),
            store=store,
            coordinator=coordinator,
            environment=AUTHORITY_ENV,
        )
    assert store.claim_calls == 0


def test_unready_reviewed_factory_fails_before_queue_claim_or_provider_factory():
    store = _Store([_lease()])
    provider_called = False

    def _must_not_run(_experiment_hash):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider path reached")

    def _unready():
        raise RoutingExperimentWorkerError("model adapter release is mismatched")

    with pytest.raises(RoutingExecutionConsumerError, match="readiness"):
        _consumer(
            store,
            action=_must_not_run,
            factory=SimpleNamespace(
                name=REVIEWED_ROUTING_FACTORY_NAME,
                validate_readiness=_unready,
            ),
        )
    assert store.claim_calls == 0
    assert provider_called is False


def test_two_consumers_do_not_duplicate_a_queue_lease():
    store = _Store([_lease()])
    first = _consumer(store)
    second = _consumer(store)
    assert first.run_once() == 1
    assert second.run_once() == 0
    assert store.closed == [(_hash("a"), "completed")]


def test_terminal_recovery_is_closed_as_recovered_and_not_retried():
    store = _Store([_lease()])
    consumer = _consumer(
        store,
        action=lambda _hash: (_ for _ in ()).throw(
            RoutingExperimentTerminalRecoveryError(
                "routing experiment claim recovered; submit a new immutable experiment"
            )
        ),
    )
    assert consumer.run_once() == 1
    assert store.closed == [(_hash("a"), "recovered")]
    assert consumer.run_once() == 0


def test_deferred_recovery_leaves_queue_lease_open_for_sql_expiry():
    store = _Store([_lease()])
    consumer = _consumer(
        store,
        action=lambda _hash: (_ for _ in ()).throw(
            RoutingExperimentDeferredRecoveryError(
                "routing provider budget recovery could not be confirmed"
            )
        ),
    )
    with pytest.raises(RoutingExperimentDeferredRecoveryError):
        consumer.run_once()
    assert store.closed == []


def test_provider_error_worded_like_terminal_is_not_marked_recovered():
    store = _Store([_lease()])
    consumer = _consumer(
        store,
        action=lambda _hash: (_ for _ in ()).throw(
            RuntimeError("provider returned terminal response; retry is safe")
        ),
    )
    assert consumer.run_once() == 1
    assert store.closed == [(_hash("a"), "failed")]


def test_forged_terminal_recovery_message_is_not_marked_recovered():
    store = _Store([_lease()])
    consumer = _consumer(
        store,
        action=lambda _hash: (_ for _ in ()).throw(
            RuntimeError(
                "routing experiment claim recovered; submit a new immutable experiment"
            )
        ),
    )
    assert consumer.run_once() == 1
    assert store.closed == [(_hash("a"), "failed")]


def test_stale_generation_is_not_reported_as_processed():
    store = _Store([_lease()])
    store.close_execution_request_lease = lambda **_kwargs: {
        "closed": False,
        "stale": True,
        "request_hash": _hash("a"),
        "lease_generation": 1,
    }
    assert _consumer(store).run_once() == 0


def test_heartbeat_is_started_and_stopped_during_long_work(monkeypatch):
    store = _Store([_lease()])
    started = threading.Event()
    stopped = threading.Event()

    class _Heartbeat:
        lost = False

        def __init__(self, **_kwargs):
            pass

        def start(self):
            started.set()

        def stop(self):
            stopped.set()

    monkeypatch.setattr(
        "gateway.research_lab.routing_execution_consumer._LeaseHeartbeat", _Heartbeat
    )
    assert _consumer(store).run_once() == 1
    assert started.is_set() and stopped.is_set()


def test_batch_keeps_all_claimed_leases_heartbeating_until_processed(monkeypatch):
    leases = [_lease("a"), _lease("d")]
    store = _Store(leases)
    started: list[str] = []
    stopped: list[str] = []

    class _Heartbeat:
        lost = False

        def __init__(self, *, lease, **_kwargs):
            self.request_hash = lease.request_hash

        def start(self):
            started.append(self.request_hash)

        def stop(self):
            stopped.append(self.request_hash)

    monkeypatch.setattr(
        "gateway.research_lab.routing_execution_consumer._LeaseHeartbeat", _Heartbeat
    )
    consumer = RoutingExecutionRequestConsumer(
        config=RoutingExecutionConsumerConfig(enabled=True, batch_size=2),
        runtime_config=RoutingExperimentRuntimeConfig(
            enabled=True, worker_lease_seconds=30, attested_authority_mode="attested"
        ),
        store=store,
        coordinator=_Coordinator(lambda _experiment_hash: object()),
        environment=AUTHORITY_ENV,
    )

    assert consumer.run_once() == 2
    assert started == [_hash("a"), _hash("d")]
    assert stopped.count(_hash("a")) == 1
    assert stopped.count(_hash("d")) == 1
    assert store.closed == [(_hash("a"), "completed"), (_hash("d"), "completed")]


def test_lost_batch_lease_fails_closed_before_coordinator(monkeypatch):
    store = _Store([_lease()])

    class _Heartbeat:
        lost = True

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(
        "gateway.research_lab.routing_execution_consumer._LeaseHeartbeat", _Heartbeat
    )
    called = False

    def _must_not_run(_experiment_hash):
        nonlocal called
        called = True
        raise AssertionError("provider coordinator reached after lease loss")

    assert _consumer(store, action=_must_not_run).run_once() == 1
    assert called is False
    assert store.closed == [(_hash("a"), "failed")]


def test_graceful_stop_exits_poll_loop_without_claiming_again():
    store = _Store([])
    consumer = _consumer(store)
    thread = threading.Thread(target=consumer.run_forever)
    thread.start()
    time.sleep(0.02)
    consumer.stop()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_static_entrypoint_is_disabled_or_missing_registry_without_provider_call(monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_AUTHORITY", "attested")
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_CLAIM_AUTHORITY", "supabase_v3")
    monkeypatch.setenv("RESEARCH_LAB_ROUTING_EXPERIMENT_ATTESTATION_AUTHORITY", "tee_v2")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test-only")
    monkeypatch.setattr(
        "gateway.research_lab.routing_execution_consumer.REVIEWED_ROUTING_FACTORY_REGISTRY",
        {},
    )
    assert build_reviewed_routing_execution_consumer  # static symbol is present
    with pytest.raises(RoutingExecutionConsumerError, match="factory registry"):
        build_reviewed_routing_execution_consumer()


def test_reviewed_factory_registry_is_immutable_after_install(monkeypatch):
    factory = SimpleNamespace(
        name=REVIEWED_ROUTING_FACTORY_NAME,
        validate_readiness=lambda: None,
    )
    monkeypatch.setattr(
        consumer_module,
        "REVIEWED_ROUTING_FACTORY_REGISTRY",
        MappingProxyType({}),
    )
    consumer_module.install_reviewed_routing_factory_registry(
        {REVIEWED_ROUTING_FACTORY_NAME: factory}
    )

    with pytest.raises(TypeError):
        consumer_module.REVIEWED_ROUTING_FACTORY_REGISTRY["unreviewed"] = factory


def test_consumer_builder_requires_release_owned_store_factory(monkeypatch):
    factory = SimpleNamespace(
        name=REVIEWED_ROUTING_FACTORY_NAME,
        validate_readiness=lambda: None,
    )
    monkeypatch.setattr(
        consumer_module,
        "REVIEWED_ROUTING_FACTORY_REGISTRY",
        MappingProxyType({REVIEWED_ROUTING_FACTORY_NAME: factory}),
    )
    with pytest.raises(RoutingExecutionConsumerError, match="store factory"):
        build_reviewed_routing_execution_consumer(environment=AUTHORITY_ENV)


def test_consumer_builder_uses_release_owned_store_factory(monkeypatch):
    factory = SimpleNamespace(
        name=REVIEWED_ROUTING_FACTORY_NAME,
        validate_readiness=lambda: None,
    )
    monkeypatch.setattr(
        consumer_module,
        "REVIEWED_ROUTING_FACTORY_REGISTRY",
        MappingProxyType({REVIEWED_ROUTING_FACTORY_NAME: factory}),
    )
    store = _Store()
    consumer = build_reviewed_routing_execution_consumer(
        environment=AUTHORITY_ENV,
        store_factory=lambda: store,
    )
    assert consumer.store is store


def test_consumer_builder_preflights_factory_before_store_creation(monkeypatch):
    store_factory_calls = 0

    def _store_factory():
        nonlocal store_factory_calls
        store_factory_calls += 1
        raise AssertionError("store must not be constructed")

    def _unready():
        raise RoutingExperimentWorkerError("binding catalog mismatch")

    factory = SimpleNamespace(
        name=REVIEWED_ROUTING_FACTORY_NAME,
        validate_readiness=_unready,
    )
    monkeypatch.setattr(
        consumer_module,
        "REVIEWED_ROUTING_FACTORY_REGISTRY",
        MappingProxyType({REVIEWED_ROUTING_FACTORY_NAME: factory}),
    )
    with pytest.raises(RoutingExecutionConsumerError, match="readiness"):
        build_reviewed_routing_execution_consumer(
            environment=AUTHORITY_ENV,
            store_factory=_store_factory,
        )
    assert store_factory_calls == 0
