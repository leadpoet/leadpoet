"""Fail-closed proxy inventory, verification, and exclusive worker leases."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
import ipaddress
import json
import os
import re
import ssl
import threading
import time
from typing import Callable, Mapping
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

from leadpoet_canonical.proxy_transport import (
    ProxyTransportError,
    validate_http_connect_proxy_url,
    verify_tls_proxy_connect,
)


PROXY_ENVIRONMENT_PREFIXES = (
    "LAB_ARENA_WEBSHARE_PROXY",
    "QUALIFICATION_WEBSHARE_PROXY",
    "WEBSHARE_PROXY",
    "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY",
)
DEFAULT_MAX_PROXIES = 250
DEFAULT_PREFLIGHT_MAX_WORKERS = 8
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 5.0
DEFAULT_CONNECT_DESTINATION = ("openrouter.ai", 443)
DEFAULT_EXIT_IP_URL = "https://api.ipify.org?format=json"
_MAX_EXIT_IP_RESPONSE_BYTES = 1024
_PROCESS_QUARANTINE_LOCK = threading.Lock()
_PROCESS_QUARANTINED_EXIT_FINGERPRINTS: set[str] = set()


class ProxyWorkerConfigurationError(ValueError):
    """The validator proxy environment does not define a safe inventory."""


class ProxyWorkerPreflightError(RuntimeError):
    """The configured inventory did not pass startup network verification."""


class ProxyWorkerPoolError(RuntimeError):
    """An exclusive proxy worker slot cannot be leased safely."""


def _proxy_fingerprint(proxy_url: str) -> str:
    return sha256(proxy_url.encode("utf-8")).hexdigest()[:16]


def _exit_ip_fingerprint(exit_ip: str) -> str:
    return sha256(exit_ip.encode("ascii")).hexdigest()[:16]


@dataclass(frozen=True)
class ProxyWorker:
    """One configured worker. Its secret URL is deliberately absent from repr."""

    slot_index: int
    source_name: str
    proxy_url: str = field(repr=False)
    proxy_fingerprint: str


@dataclass(frozen=True)
class ProxyWorkerInventory:
    workers: tuple[ProxyWorker, ...]
    native_coordinator_count: int = 1

    @property
    def webshare_worker_count(self) -> int:
        return len(self.workers)

    @property
    def total_process_capacity(self) -> int:
        return self.webshare_worker_count + self.native_coordinator_count


@dataclass(frozen=True)
class VerifiedProxyWorker:
    """A configured worker whose transport and distinct exit were verified."""

    slot_index: int
    source_name: str
    proxy_url: str | None = field(repr=False)
    proxy_fingerprint: str
    exit_ip: str = field(repr=False)
    exit_ip_fingerprint: str


@dataclass(frozen=True)
class VerifiedProxyWorkerInventory:
    workers: tuple[VerifiedProxyWorker, ...]
    native_exit_ip: str = field(repr=False)
    native_exit_ip_fingerprint: str
    native_coordinator_count: int = 1

    @property
    def verified_workers(self) -> tuple[VerifiedProxyWorker, ...]:
        native_worker = VerifiedProxyWorker(
            slot_index=0,
            source_name="native_coordinator",
            proxy_url=None,
            proxy_fingerprint="native-coordinator",
            exit_ip=self.native_exit_ip,
            exit_ip_fingerprint=self.native_exit_ip_fingerprint,
        )
        return (native_worker, *self.workers)

    @property
    def webshare_worker_count(self) -> int:
        return len(self.workers)

    @property
    def total_process_capacity(self) -> int:
        return self.webshare_worker_count + self.native_coordinator_count


def proxy_workers_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    max_proxies: int = DEFAULT_MAX_PROXIES,
) -> ProxyWorkerInventory:
    """Read every positive indexed proxy alias without exposing secret values.

    A slot can be present through more than one alias only when every populated
    alias has exactly the same value. Gaps in the positive indices are allowed;
    returned worker slots are compact and stable in source-index order.
    """

    env = os.environ if environment is None else environment
    try:
        normalized_max = int(max_proxies)
    except (TypeError, ValueError) as exc:
        raise ProxyWorkerConfigurationError("proxy inventory bound is invalid") from exc
    if not 1 <= normalized_max <= DEFAULT_MAX_PROXIES:
        raise ProxyWorkerConfigurationError("proxy inventory bound is invalid")

    prefix_rank = {
        prefix: rank for rank, prefix in enumerate(PROXY_ENVIRONMENT_PREFIXES)
    }
    matched: dict[int, list[tuple[int, str, str]]] = {}
    for raw_name, raw_value in env.items():
        name = str(raw_name)
        prefix = next(
            (
                candidate
                for candidate in PROXY_ENVIRONMENT_PREFIXES
                if name.startswith(candidate + "_")
            ),
            None,
        )
        if prefix is None:
            continue
        suffix = name[len(prefix) + 1 :]
        value = str(raw_value or "").strip()
        if not value:
            continue
        if re.fullmatch(r"[0-9]+", suffix) is None:
            raise ProxyWorkerConfigurationError(
                "%s must use a positive numeric proxy index" % name
            )
        index = int(suffix)
        if index < 1:
            raise ProxyWorkerConfigurationError(
                "%s must use a positive numeric proxy index" % name
            )
        matched.setdefault(index, []).append(
            (prefix_rank[prefix], name, value)
        )
        if len(matched) > normalized_max:
            raise ProxyWorkerConfigurationError("proxy inventory exceeds its bound")

    if not matched:
        raise ProxyWorkerConfigurationError(
            "validator Webshare proxy inventory is unavailable"
        )
    workers = []
    seen_values: dict[str, int] = {}
    for index in sorted(matched):
        candidates = sorted(matched[index])
        candidate_values = {candidate[2] for candidate in candidates}
        if len(candidate_values) != 1:
            raise ProxyWorkerConfigurationError(
                "proxy index %d has conflicting environment aliases" % index
            )
        _rank, source_name, proxy_url = candidates[0]
        try:
            normalized_url = validate_http_connect_proxy_url(proxy_url)
        except (ProxyTransportError, ValueError) as exc:
            raise ProxyWorkerConfigurationError(
                "proxy index %d is not a valid HTTP CONNECT or HTTPS proxy" % index
            ) from exc
        previous_index = seen_values.get(normalized_url)
        if previous_index is not None:
            raise ProxyWorkerConfigurationError(
                "proxy indices %d and %d contain the same endpoint"
                % (previous_index, index)
            )
        seen_values[normalized_url] = index
        workers.append(
            ProxyWorker(
                slot_index=len(workers) + 1,
                source_name=source_name,
                proxy_url=normalized_url,
                proxy_fingerprint=_proxy_fingerprint(normalized_url),
            )
        )

    return ProxyWorkerInventory(workers=tuple(workers))


def _validated_exit_ip(value: object) -> str:
    try:
        parsed = ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise ProxyWorkerPreflightError(
            "exit IP probe returned an invalid address"
        ) from exc
    if not parsed.is_global:
        raise ProxyWorkerPreflightError(
            "exit IP probe returned a non-public address"
        )
    return parsed.compressed


def probe_https_exit_ip(
    proxy_url: str | None,
    *,
    timeout_seconds: float = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    endpoint: str = DEFAULT_EXIT_IP_URL,
) -> str:
    """Return the public IP observed by a TLS-verified HTTPS endpoint."""

    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ProxyWorkerPreflightError("exit IP probe timeout is invalid") from exc
    if not 0 < timeout <= 30:
        raise ProxyWorkerPreflightError("exit IP probe timeout is invalid")
    try:
        parsed_endpoint = urlsplit(str(endpoint or ""))
    except ValueError as exc:
        raise ProxyWorkerPreflightError(
            "exit IP probe endpoint is invalid"
        ) from exc
    if (
        parsed_endpoint.scheme.lower() != "https"
        or not parsed_endpoint.hostname
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
    ):
        raise ProxyWorkerPreflightError("exit IP probe endpoint is invalid")

    proxy_configuration = (
        {} if proxy_url is None else {"http": proxy_url, "https": proxy_url}
    )
    opener = build_opener(
        ProxyHandler(proxy_configuration),
        HTTPSHandler(context=ssl.create_default_context()),
    )
    request = Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "leadpoet-proxy-preflight/1",
        },
        method="GET",
    )
    probe_failed = False
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(_MAX_EXIT_IP_RESPONSE_BYTES + 1)
        if len(payload) > _MAX_EXIT_IP_RESPONSE_BYTES:
            raise ProxyWorkerPreflightError("exit IP probe response is too large")
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise ProxyWorkerPreflightError("exit IP probe response is invalid")
        return _validated_exit_ip(decoded.get("ip"))
    except ProxyWorkerPreflightError:
        raise
    except Exception:
        # A urllib exception can include proxy credentials. Raise after leaving
        # the handler so it is not retained as an implicit exception context.
        probe_failed = True
    if probe_failed:
        raise ProxyWorkerPreflightError("HTTPS exit IP probe failed")


def _probe_native_exit_ip(*, timeout_seconds: float) -> str:
    return probe_https_exit_ip(None, timeout_seconds=timeout_seconds)


def preflight_proxy_workers(
    inventory: ProxyWorkerInventory,
    *,
    transport_probe: Callable[..., None] = verify_tls_proxy_connect,
    exit_ip_probe: Callable[..., str] = probe_https_exit_ip,
    native_ip_probe: Callable[..., str] = _probe_native_exit_ip,
    destination_host: str = DEFAULT_CONNECT_DESTINATION[0],
    destination_port: int = DEFAULT_CONNECT_DESTINATION[1],
    timeout_seconds: float = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    max_workers: int = DEFAULT_PREFLIGHT_MAX_WORKERS,
) -> VerifiedProxyWorkerInventory:
    """Verify every configured transport and require N+1 distinct public exits."""

    if not isinstance(inventory, ProxyWorkerInventory) or not inventory.workers:
        raise ProxyWorkerPreflightError("proxy worker inventory is unavailable")
    try:
        timeout = float(timeout_seconds)
        concurrency = int(max_workers)
    except (TypeError, ValueError) as exc:
        raise ProxyWorkerPreflightError("proxy preflight bounds are invalid") from exc
    if not 0 < timeout <= 30 or not 1 <= concurrency <= 32:
        raise ProxyWorkerPreflightError("proxy preflight bounds are invalid")

    try:
        native_exit_ip = _validated_exit_ip(
            native_ip_probe(timeout_seconds=timeout)
        )
    except ProxyWorkerPreflightError:
        raise
    except Exception:
        raise ProxyWorkerPreflightError("native HTTPS exit IP probe failed") from None

    def verify(worker: ProxyWorker) -> VerifiedProxyWorker:
        verification_failed = False
        try:
            transport_probe(
                worker.proxy_url,
                destination_host=destination_host,
                destination_port=destination_port,
                timeout_seconds=timeout,
            )
            exit_ip = _validated_exit_ip(
                exit_ip_probe(worker.proxy_url, timeout_seconds=timeout)
            )
        except Exception:
            # The injected transport can retain its proxy URL in the exception.
            # Raise outside the handler so no implicit context holds that URL.
            verification_failed = True
        if verification_failed:
            raise ProxyWorkerPreflightError(
                "proxy worker %d failed startup verification" % worker.slot_index
            )
        return VerifiedProxyWorker(
            slot_index=worker.slot_index,
            source_name=worker.source_name,
            proxy_url=worker.proxy_url,
            proxy_fingerprint=worker.proxy_fingerprint,
            exit_ip=exit_ip,
            exit_ip_fingerprint=_exit_ip_fingerprint(exit_ip),
        )

    verified_by_slot: dict[int, VerifiedProxyWorker] = {}
    worker_count = min(concurrency, len(inventory.workers))
    failed_slots = []
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="arena-proxy-preflight",
    ) as executor:
        futures = {
            executor.submit(verify, worker): worker.slot_index
            for worker in inventory.workers
        }
        for future in as_completed(futures):
            slot_index = futures[future]
            try:
                verified_by_slot[slot_index] = future.result()
            except ProxyWorkerPreflightError:
                failed_slots.append(slot_index)
    if failed_slots:
        raise ProxyWorkerPreflightError(
            "proxy worker %d failed startup verification" % min(failed_slots)
        )

    verified = tuple(
        verified_by_slot[index]
        for index in range(1, len(inventory.workers) + 1)
    )
    exits: dict[str, int] = {native_exit_ip: -1}
    for worker in verified:
        previous = exits.get(worker.exit_ip)
        if previous is not None:
            if previous == -1:
                detail = "matches the native coordinator"
            else:
                detail = "matches proxy worker %d" % previous
            raise ProxyWorkerPreflightError(
                "proxy worker %d exit IP %s" % (worker.slot_index, detail)
            )
        exits[worker.exit_ip] = worker.slot_index

    return VerifiedProxyWorkerInventory(
        workers=verified,
        native_exit_ip=native_exit_ip,
        native_exit_ip_fingerprint=_exit_ip_fingerprint(native_exit_ip),
    )


class ProxyWorkerLease:
    """One stable proxy assignment held exclusively until explicit release."""

    def __init__(
        self,
        pool: "ProxyWorkerPool",
        worker: VerifiedProxyWorker,
    ) -> None:
        self._pool = pool
        self._worker = worker
        self._released = False
        self._quarantined = False
        self._state_lock = threading.Lock()

    @property
    def slot_index(self) -> int:
        return self._worker.slot_index

    @property
    def proxy_url(self) -> str | None:
        return self._worker.proxy_url

    @property
    def proxy_fingerprint(self) -> str:
        return self._worker.proxy_fingerprint

    @property
    def exit_ip(self) -> str:
        return self._worker.exit_ip

    @property
    def exit_ip_fingerprint(self) -> str:
        return self._worker.exit_ip_fingerprint

    def release(self) -> None:
        with self._state_lock:
            if self._released or self._quarantined:
                return
            self._pool._release(self._worker)
            self._released = True

    def quarantine(self) -> None:
        """Permanently remove this slot after unproven transport cleanup."""

        with self._state_lock:
            if self._quarantined:
                return
            if self._released:
                raise ProxyWorkerPoolError(
                    "released proxy worker lease cannot be quarantined"
                )
            self._pool._quarantine(self._worker)
            self._quarantined = True

    def __enter__(self) -> "ProxyWorkerLease":
        if self._released or self._quarantined:
            raise ProxyWorkerPoolError("proxy worker lease is already released")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            "ProxyWorkerLease(slot_index=%d, proxy_fingerprint=%r, "
            "exit_ip_fingerprint=%r, released=%r, quarantined=%r)"
            % (
                self.slot_index,
                self.proxy_fingerprint,
                self.exit_ip_fingerprint,
                self._released,
                self._quarantined,
            )
        )


class ProxyWorkerPool:
    """Thread-safe exclusive leases for all verified execution slots.

    Keep the lease across retries for one live attempt. Release it and acquire a
    new lease only when the caller has classified an independent retry.
    """

    def __init__(self, inventory: VerifiedProxyWorkerInventory) -> None:
        if (
            not isinstance(inventory, VerifiedProxyWorkerInventory)
            or not inventory.workers
        ):
            raise ProxyWorkerPoolError("verified proxy worker inventory is unavailable")
        self._inventory = inventory
        with _PROCESS_QUARANTINE_LOCK:
            process_quarantined = set(_PROCESS_QUARANTINED_EXIT_FINGERPRINTS)
        self._available = deque(
            worker
            for worker in inventory.verified_workers
            if worker.exit_ip_fingerprint not in process_quarantined
        )
        self._leased: set[int] = set()
        self._quarantined: set[int] = {
            worker.slot_index
            for worker in inventory.verified_workers
            if worker.exit_ip_fingerprint in process_quarantined
        }
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._inventory.total_process_capacity

    @property
    def webshare_worker_count(self) -> int:
        return self._inventory.webshare_worker_count

    @property
    def total_process_capacity(self) -> int:
        return self._inventory.total_process_capacity

    @property
    def available(self) -> int:
        with self._condition:
            return len(self._available)

    @property
    def quarantined(self) -> int:
        with self._condition:
            return len(self._quarantined)

    def acquire(self, timeout: float | None = None) -> ProxyWorkerLease:
        if timeout is not None:
            try:
                normalized_timeout = float(timeout)
            except (TypeError, ValueError) as exc:
                raise ProxyWorkerPoolError(
                    "proxy worker lease timeout is invalid"
                ) from exc
            if normalized_timeout < 0:
                raise ProxyWorkerPoolError("proxy worker lease timeout is invalid")
        else:
            normalized_timeout = None
        deadline = (
            None
            if normalized_timeout is None
            else time.monotonic() + normalized_timeout
        )
        with self._condition:
            while not self._available:
                if len(self._quarantined) == self.capacity:
                    raise ProxyWorkerPoolError(
                        "all proxy worker slots are quarantined"
                    )
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProxyWorkerPoolError("no proxy worker slot is available")
                self._condition.wait(remaining)
            worker = self._available.popleft()
            if worker.slot_index in self._leased:
                raise ProxyWorkerPoolError(
                    "proxy worker slot ownership is inconsistent"
                )
            self._leased.add(worker.slot_index)
            return ProxyWorkerLease(self, worker)

    def _release(self, worker: VerifiedProxyWorker) -> None:
        with self._condition:
            if worker.slot_index not in self._leased:
                raise ProxyWorkerPoolError("proxy worker slot is not leased")
            self._leased.remove(worker.slot_index)
            self._available.append(worker)
            self._condition.notify()

    def _quarantine(self, worker: VerifiedProxyWorker) -> None:
        with _PROCESS_QUARANTINE_LOCK:
            with self._condition:
                if worker.slot_index not in self._leased:
                    raise ProxyWorkerPoolError("proxy worker slot is not leased")
                _PROCESS_QUARANTINED_EXIT_FINGERPRINTS.add(
                    worker.exit_ip_fingerprint
                )
                self._leased.remove(worker.slot_index)
                self._quarantined.add(worker.slot_index)
                self._condition.notify_all()
