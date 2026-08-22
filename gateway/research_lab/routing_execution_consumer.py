"""Disabled-by-default consumer for Research Lab routing execution requests.

The consumer is a queue coordinator only.  It does not accept an import path,
provider endpoint, credential, or caller-supplied factory.  A deployment must
register the reviewed attested factory in this module before the process can
claim a queue item.  The execution claim is bound to the database-owned queue
lease; a claim failure closes the queue lease and
cannot reach a provider.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from gateway.research_lab.routing_experiment_runtime import (
    RoutingExperimentRuntimeConfig,
    RoutingExperimentDeferredRecoveryError,
    RoutingExperimentRuntimeError,
    RoutingExperimentService,
    RoutingExperimentTerminalRecoveryError,
)
from gateway.research_lab.provider_evidence_proxy import PROXY_URL_ENV
from gateway.research_lab.routing_experiment_store import (
    RoutingExecutionRequestLease,
    RoutingExperimentStoreError,
)
from gateway.research_lab.routing_experiment_worker import (
    REVIEWED_ROUTING_FACTORY_NAME,
    RoutingExperimentCoordinator,
    RoutingExperimentRunFactory,
    RoutingExperimentWorker,
    RoutingExperimentWorkerError,
    assert_reviewed_routing_factory_ready,
    assert_reviewed_routing_runtime_registered,
)


ROUTING_CONSUMER_ENABLED_ENV = "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_ENABLED"
ROUTING_CONSUMER_BATCH_ENV = "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_BATCH_SIZE"
ROUTING_CONSUMER_POLL_ENV = "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_POLL_SECONDS"
ROUTING_CONSUMER_WORKER_REF_ENV = "RESEARCH_LAB_ROUTING_EXECUTION_CONSUMER_WORKER_REF"
WORKER_READY_FD_ENV = "RESEARCH_LAB_WORKER_READY_FD"


class RoutingExecutionConsumerError(RuntimeError):
    """The queue consumer cannot safely start or process a lease."""


class RoutingExecutionQueueStore(Protocol):
    def claim_pending_execution_requests(
        self, *, worker_ref: str, batch_size: int, lease_seconds: int
    ) -> tuple[RoutingExecutionRequestLease, ...]: ...

    def renew_execution_request_lease(
        self, *, lease: RoutingExecutionRequestLease, lease_seconds: int
    ) -> Mapping[str, Any]: ...

    def close_execution_request_lease(
        self, *, lease: RoutingExecutionRequestLease, close_reason: str
    ) -> Mapping[str, Any]: ...


# A reviewed bootstrap may install exactly one named factory.  There is no
# request, CLI, or import-path registration surface.
REVIEWED_ROUTING_FACTORY_REGISTRY: Mapping[str, RoutingExperimentRunFactory] = (
    MappingProxyType({})
)


def install_reviewed_routing_factory_registry(
    registry: Mapping[str, RoutingExperimentRunFactory] | None,
) -> None:
    """Install the static composition for this separate consumer process."""

    global REVIEWED_ROUTING_FACTORY_REGISTRY
    candidate = dict(registry or {})
    if set(candidate) != {REVIEWED_ROUTING_FACTORY_NAME}:
        raise RoutingExecutionConsumerError(
            "reviewed routing factory registry must contain exactly one named factory"
        )
    factory = candidate[REVIEWED_ROUTING_FACTORY_NAME]
    if getattr(factory, "name", None) != REVIEWED_ROUTING_FACTORY_NAME:
        raise RoutingExecutionConsumerError(
            "reviewed routing factory name is inconsistent"
        )
    existing = REVIEWED_ROUTING_FACTORY_REGISTRY
    if existing:
        if (
            set(existing) != {REVIEWED_ROUTING_FACTORY_NAME}
            or existing[REVIEWED_ROUTING_FACTORY_NAME] is not factory
        ):
            raise RoutingExecutionConsumerError(
                "reviewed routing factory registry is already frozen"
            )
        # Exact re-installation is idempotent.  Do not replace the mapping or
        # its factory object after the consumer process has started.
        return
    REVIEWED_ROUTING_FACTORY_REGISTRY = MappingProxyType(candidate)


def _bool_env(environment: Mapping[str, str], name: str) -> bool:
    return str(environment.get(name, "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _bounded_int(
    environment: Mapping[str, str], name: str, *, default: int, minimum: int, maximum: int
) -> int:
    raw = str(environment.get(name, str(default))).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RoutingExecutionConsumerError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise RoutingExecutionConsumerError(f"{name} must be {minimum}..{maximum}")
    return value


def _runtime_config_from_environment(
    environment: Mapping[str, str] | None,
) -> RoutingExperimentRuntimeConfig:
    """Resolve runtime gates without mutating process environment in tests."""

    source = os.environ if environment is None else environment
    raw_lease = str(source.get("RESEARCH_LAB_ROUTING_EXPERIMENT_LEASE_SECONDS", "300"))
    try:
        lease = int(raw_lease)
    except ValueError as exc:
        raise RoutingExecutionConsumerError(
            "RESEARCH_LAB_ROUTING_EXPERIMENT_LEASE_SECONDS must be an integer"
        ) from exc
    if lease < 30 or lease > 3600:
        raise RoutingExecutionConsumerError(
            "RESEARCH_LAB_ROUTING_EXPERIMENT_LEASE_SECONDS must be 30..3600"
        )
    return RoutingExperimentRuntimeConfig(
        enabled=_bool_env(source, "RESEARCH_LAB_ROUTING_EXPERIMENT_ENABLED"),
        live_execution_enabled=_bool_env(
            source, "RESEARCH_LAB_ROUTING_EXPERIMENT_LIVE_ENABLED"
        ),
        worker_lease_seconds=lease,
        evidence_proxy_url=str(source.get(PROXY_URL_ENV, "") or "").strip(),
        attested_authority_mode=str(
            source.get("RESEARCH_LAB_ROUTING_EXPERIMENT_AUTHORITY", "") or ""
        ).strip(),
    )


@dataclass(frozen=True)
class RoutingExecutionConsumerConfig:
    enabled: bool = False
    batch_size: int = 1
    poll_seconds: float = 5.0
    worker_ref: str = "routing-execution-consumer"

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None
    ) -> "RoutingExecutionConsumerConfig":
        source = dict(os.environ if environment is None else environment)
        poll_raw = str(source.get(ROUTING_CONSUMER_POLL_ENV, "5")).strip()
        try:
            poll_seconds = float(poll_raw)
        except ValueError as exc:
            raise RoutingExecutionConsumerError(
                f"{ROUTING_CONSUMER_POLL_ENV} must be a number"
            ) from exc
        if poll_seconds < 0.1 or poll_seconds > 300:
            raise RoutingExecutionConsumerError(
                f"{ROUTING_CONSUMER_POLL_ENV} must be 0.1..300"
            )
        worker_ref = str(
            source.get(ROUTING_CONSUMER_WORKER_REF_ENV, cls.worker_ref)
        ).strip()
        if not worker_ref:
            raise RoutingExecutionConsumerError("routing execution consumer worker ref is required")
        return cls(
            enabled=_bool_env(source, ROUTING_CONSUMER_ENABLED_ENV),
            batch_size=_bounded_int(
                source,
                ROUTING_CONSUMER_BATCH_ENV,
                default=1,
                minimum=1,
                maximum=100,
            ),
            poll_seconds=poll_seconds,
            worker_ref=worker_ref,
        )

    def assert_enabled(self) -> None:
        if not self.enabled:
            raise RoutingExecutionConsumerError(
                "routing execution request consumer is disabled"
            )


class _LeaseHeartbeat:
    def __init__(
        self,
        *,
        store: RoutingExecutionQueueStore,
        lease: RoutingExecutionRequestLease,
        lease_seconds: int,
    ) -> None:
        self._store = store
        self._lease = lease
        self._lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="routing-execution-lease-heartbeat",
            daemon=True,
        )

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._lease_seconds / 2))

    def _run(self) -> None:
        interval = max(1.0, self._lease_seconds / 3)
        while not self._stop.wait(interval):
            try:
                result = self._store.renew_execution_request_lease(
                    lease=self._lease,
                    lease_seconds=self._lease_seconds,
                )
            except Exception:
                self._lost.set()
                return
            if result.get("renewed") is not True:
                self._lost.set()
                return


def _is_terminal_recovery_error(exc: BaseException) -> bool:
    # The runtime raises this type only after a durable recovery fact. Never
    # infer terminal state from an exception string: provider and adapter
    # output is untrusted and can forge either recovery message.
    return isinstance(exc, RoutingExperimentTerminalRecoveryError)


class RoutingExecutionRequestConsumer:
    """Poll, fence, execute, heartbeat, and terminally close queue leases."""

    def __init__(
        self,
        *,
        config: RoutingExecutionConsumerConfig,
        runtime_config: RoutingExperimentRuntimeConfig,
        store: RoutingExecutionQueueStore,
        coordinator: RoutingExperimentCoordinator,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        config.assert_enabled()
        assert_reviewed_routing_runtime_registered(runtime_config, environment=environment)
        for name in (
            "claim_pending_execution_requests",
            "renew_execution_request_lease",
            "close_execution_request_lease",
        ):
            if not callable(getattr(store, name, None)):
                raise RoutingExecutionConsumerError(
                    f"routing execution queue store is missing {name}"
                )
        if not isinstance(coordinator, RoutingExperimentCoordinator):
            raise RoutingExecutionConsumerError(
                "reviewed routing factory registry is unavailable"
            )
        factories = getattr(coordinator, "_factories", {})
        if set(factories) != {REVIEWED_ROUTING_FACTORY_NAME}:
            raise RoutingExecutionConsumerError(
                "reviewed routing factory registry is unavailable"
            )
        try:
            assert_reviewed_routing_factory_ready(
                factories[REVIEWED_ROUTING_FACTORY_NAME]
            )
        except RoutingExperimentWorkerError as exc:
            raise RoutingExecutionConsumerError(
                "reviewed routing factory readiness is unavailable"
            ) from exc
        self.config = config
        self.runtime_config = runtime_config
        self.store = store
        self.coordinator = coordinator
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> int:
        if self._stop.is_set():
            return 0
        leases = self.store.claim_pending_execution_requests(
            worker_ref=self.config.worker_ref,
            batch_size=self.config.batch_size,
            lease_seconds=self.runtime_config.worker_lease_seconds,
        )
        # The queue RPC can return more than one lease.  Keep every claimed
        # lease alive while this consumer processes the batch sequentially;
        # otherwise a slow first experiment can let a later lease expire and
        # be claimed by another consumer.  That would allow duplicate provider
        # work even though the database lease generation is correct.
        heartbeats = {
            lease.request_hash: _LeaseHeartbeat(
                store=self.store,
                lease=lease,
                lease_seconds=self.runtime_config.worker_lease_seconds,
            )
            for lease in leases
        }
        processed = 0
        try:
            for heartbeat in heartbeats.values():
                heartbeat.start()
            for lease in leases:
                if self._stop.is_set():
                    break
                processed += 1
                heartbeat = heartbeats[lease.request_hash]
                close_reason = "completed"
                deferred_recovery = False
                try:
                    if heartbeat.lost:
                        raise RoutingExecutionConsumerError(
                            "routing execution queue lease was lost"
                        )
                    # The named coordinator reloads the immutable spec and only
                    # then invokes the reviewed factory. It never receives a
                    # provider endpoint or credential from the queue.
                    self.coordinator.resume(
                        experiment_hash=lease.experiment_hash,
                        factory_name=REVIEWED_ROUTING_FACTORY_NAME,
                        lease=lease,
                    )
                    if heartbeat.lost:
                        raise RoutingExecutionConsumerError(
                            "routing execution queue lease was lost"
                        )
                except RoutingExperimentDeferredRecoveryError:
                    # The provider budget or claim cleanup could not be
                    # confirmed.  Do not close the queue lease: stopping its
                    # heartbeat lets the SQL authority recover it after
                    # expiry, without permitting a duplicate provider call.
                    deferred_recovery = True
                    raise
                except Exception as exc:  # noqa: BLE001 - close with redacted state
                    close_reason = "recovered" if _is_terminal_recovery_error(exc) else "failed"
                finally:
                    heartbeat.stop()
                    heartbeats.pop(lease.request_hash, None)
                    if not deferred_recovery:
                        result = self.store.close_execution_request_lease(
                            lease=lease,
                            close_reason=close_reason,
                        )
                        if result.get("closed") is not True:
                            # A stale generation is terminal for this queue
                            # attempt. It must not be retried by this consumer.
                            processed -= 1
        finally:
            # If shutdown interrupts a batch, unprocessed leases are allowed to
            # expire and be reclaimed with a new generation.  Stop their
            # heartbeat threads so the stopped consumer cannot renew them.
            for heartbeat in heartbeats.values():
                heartbeat.stop()
        return processed

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except (
                RoutingExperimentDeferredRecoveryError,
                RoutingExperimentStoreError,
                RoutingExecutionConsumerError,
            ):
                # Configuration and authority failures fail closed.  The
                # process remains stoppable and does not spin provider calls.
                self._stop.wait(self.config.poll_seconds)
                continue
            self._stop.wait(self.config.poll_seconds)


def build_reviewed_routing_execution_consumer(
    *,
    environment: Mapping[str, str] | None = None,
    store_factory: Callable[[], RoutingExecutionQueueStore] | None = None,
) -> RoutingExecutionRequestConsumer:
    """Build only from static reviewed registrations and store authority."""

    config = RoutingExecutionConsumerConfig.from_env(environment)
    config.assert_enabled()
    runtime_config = _runtime_config_from_environment(environment)
    assert_reviewed_routing_runtime_registered(runtime_config, environment=environment)
    if set(REVIEWED_ROUTING_FACTORY_REGISTRY) != {REVIEWED_ROUTING_FACTORY_NAME}:
        raise RoutingExecutionConsumerError(
            "reviewed routing factory registry is unavailable"
        )
    try:
        assert_reviewed_routing_factory_ready(
            REVIEWED_ROUTING_FACTORY_REGISTRY[REVIEWED_ROUTING_FACTORY_NAME]
        )
    except RoutingExperimentWorkerError as exc:
        raise RoutingExecutionConsumerError(
            "reviewed routing factory readiness is unavailable"
        ) from exc
    if not callable(store_factory):
        raise RoutingExecutionConsumerError(
            "reviewed routing store factory is unavailable"
        )
    store = store_factory()
    for name in (
        "claim_pending_execution_requests",
        "renew_execution_request_lease",
        "close_execution_request_lease",
    ):
        if not callable(getattr(store, name, None)):
            raise RoutingExecutionConsumerError(
                f"reviewed routing store is missing {name}"
            )
    worker = RoutingExperimentWorker(
        service=RoutingExperimentService(
            config=runtime_config,
            store=store,
        ),
        worker_ref=config.worker_ref,
    )
    coordinator = RoutingExperimentCoordinator(
        worker=worker,
        factories=REVIEWED_ROUTING_FACTORY_REGISTRY,
    )
    return RoutingExecutionRequestConsumer(
        config=config,
        runtime_config=runtime_config,
        store=worker.service.store,
        coordinator=coordinator,
        environment=environment,
    )


def routing_execution_consumer_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        # The child has no access to the gateway app object.  It must pass the
        # same static release bootstrap before it can construct a store or
        # claim a queue lease.  The current release intentionally provides no
        # dependency bundle, so this remains fail closed.
        from gateway.research_lab.routing_product_bootstrap import (
            bootstrap_reviewed_routing_consumer_process,
        )

        composition = bootstrap_reviewed_routing_consumer_process()
        install_reviewed_routing_factory_registry(composition.factory_registry)
        consumer = build_reviewed_routing_execution_consumer(
            store_factory=composition.api_service.store_factory,
        )
        ready_fd = str(os.environ.pop(WORKER_READY_FD_ENV, "")).strip()
        if ready_fd:
            fd = int(ready_fd)
            try:
                os.write(fd, b"ready\n")
            finally:
                os.close(fd)
        if args.once:
            consumer.run_once()
        else:
            consumer.run_forever()
        return 0
    except (
        RoutingExecutionConsumerError,
        RoutingExperimentDeferredRecoveryError,
        RoutingExperimentRuntimeError,
        RoutingExperimentWorkerError,
        RoutingExperimentStoreError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2


__all__ = [
    "ROUTING_CONSUMER_ENABLED_ENV",
    "ROUTING_CONSUMER_BATCH_ENV",
    "ROUTING_CONSUMER_POLL_ENV",
    "ROUTING_CONSUMER_WORKER_REF_ENV",
    "REVIEWED_ROUTING_FACTORY_REGISTRY",
    "install_reviewed_routing_factory_registry",
    "RoutingExecutionConsumerError",
    "RoutingExecutionConsumerConfig",
    "RoutingExecutionRequestConsumer",
    "build_reviewed_routing_execution_consumer",
    "routing_execution_consumer_main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(routing_execution_consumer_main())
