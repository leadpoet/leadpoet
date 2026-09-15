from __future__ import annotations

import threading

import pytest

from lab_arena import proxy_workers as module
from lab_arena.proxy_workers import (
    ProxyWorkerConfigurationError,
    ProxyWorkerPool,
    ProxyWorkerPoolError,
    ProxyWorkerPreflightError,
    preflight_proxy_workers,
    probe_https_exit_ip,
    proxy_workers_from_environment,
)
from leadpoet_canonical import proxy_transport


@pytest.fixture(autouse=True)
def _reset_process_proxy_quarantine():
    with module._PROCESS_QUARANTINE_LOCK:
        module._PROCESS_QUARANTINED_EXIT_FINGERPRINTS.clear()
    yield
    with module._PROCESS_QUARANTINE_LOCK:
        module._PROCESS_QUARANTINED_EXIT_FINGERPRINTS.clear()


def _proxy(index: int) -> str:
    return "https://worker-%d:secret-%d@proxy-%d.example.com:443" % (
        index,
        index,
        index,
    )


def _environment(count: int, *, prefix: str = "LAB_ARENA_WEBSHARE_PROXY"):
    return {"%s_%d" % (prefix, index): _proxy(index) for index in range(1, count + 1)}


def _verified_inventory(count: int = 2):
    inventory = proxy_workers_from_environment(_environment(count))
    exit_ips = {
        worker.proxy_url: "8.8.4.%d" % (worker.slot_index + 1)
        for worker in inventory.workers
    }
    return preflight_proxy_workers(
        inventory,
        transport_probe=lambda *_args, **_kwargs: None,
        exit_ip_probe=lambda proxy_url, **_kwargs: exit_ips[proxy_url],
        native_ip_probe=lambda **_kwargs: "1.1.1.1",
    )


def test_inventory_accepts_all_aliases_and_compacts_gapped_indices():
    environment = {
        "QUALIFICATION_WEBSHARE_PROXY_2": _proxy(2),
        "WEBSHARE_PROXY_7": _proxy(7),
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_10001": _proxy(11),
        "LAB_ARENA_WEBSHARE_PROXY_10001": _proxy(11),
        "UNRELATED_SECRET": "ignore-me",
    }

    inventory = proxy_workers_from_environment(environment)

    assert [worker.slot_index for worker in inventory.workers] == [1, 2, 3]
    assert [worker.source_name for worker in inventory.workers] == [
        "QUALIFICATION_WEBSHARE_PROXY_2",
        "WEBSHARE_PROXY_7",
        "LAB_ARENA_WEBSHARE_PROXY_10001",
    ]
    assert [worker.proxy_url for worker in inventory.workers] == [
        _proxy(2),
        _proxy(7),
        _proxy(11),
    ]
    assert inventory.webshare_worker_count == 3
    assert inventory.total_process_capacity == 4
    assert "secret" not in repr(inventory)
    assert "proxy-" not in repr(inventory)


@pytest.mark.parametrize(
    ("proxy_count", "expected_capacity"),
    ((9, 10), (19, 20)),
)
def test_total_process_capacity_adds_one_native_coordinator(
    proxy_count,
    expected_capacity,
):
    inventory = proxy_workers_from_environment(_environment(proxy_count))

    assert inventory.webshare_worker_count == proxy_count
    assert inventory.total_process_capacity == expected_capacity


def test_existing_validator_qualification_alias_is_supported():
    inventory = proxy_workers_from_environment(
        _environment(10, prefix="QUALIFICATION_WEBSHARE_PROXY")
    )

    assert inventory.webshare_worker_count == 10
    assert inventory.total_process_capacity == 11


def test_conflicting_aliases_for_one_index_fail_without_secret_values():
    first = _proxy(1)
    second = _proxy(2)

    with pytest.raises(
        ProxyWorkerConfigurationError,
        match="proxy index 1 has conflicting environment aliases",
    ) as raised:
        proxy_workers_from_environment(
            {
                "LAB_ARENA_WEBSHARE_PROXY_1": first,
                "QUALIFICATION_WEBSHARE_PROXY_1": second,
            }
        )

    assert first not in str(raised.value)
    assert second not in str(raised.value)


def test_duplicate_endpoint_indices_fail_instead_of_reducing_capacity():
    value = _proxy(1)

    with pytest.raises(
        ProxyWorkerConfigurationError,
        match="proxy indices 1 and 9 contain the same endpoint",
    ):
        proxy_workers_from_environment(
            {
                "LAB_ARENA_WEBSHARE_PROXY_1": value,
                "LAB_ARENA_WEBSHARE_PROXY_9": value,
            }
        )


@pytest.mark.parametrize(
    "environment",
    (
        {},
        {"LAB_ARENA_WEBSHARE_PROXY_0": _proxy(1)},
        {"LAB_ARENA_WEBSHARE_PROXY_BAD": _proxy(1)},
    ),
)
def test_missing_or_out_of_bound_inventory_fails_closed(environment):
    with pytest.raises(ProxyWorkerConfigurationError):
        proxy_workers_from_environment(environment)


def test_inventory_bounds_worker_count_without_bounding_positive_indices():
    environment = {
        "LAB_ARENA_WEBSHARE_PROXY_%d" % (index * 1000): _proxy(index)
        for index in range(1, 4)
    }

    with pytest.raises(ProxyWorkerConfigurationError, match="exceeds its bound"):
        proxy_workers_from_environment(environment, max_proxies=2)


def test_invalid_proxy_url_fails_without_exposing_credentials():
    secret_value = "https://user:very-secret@proxy.example.com:bad"

    with pytest.raises(ProxyWorkerConfigurationError) as raised:
        proxy_workers_from_environment(
            {"LAB_ARENA_WEBSHARE_PROXY_1": secret_value}
        )

    assert secret_value not in str(raised.value)
    assert "very-secret" not in repr(raised.value)


def test_preflight_verifies_connect_and_actual_distinct_exit_ips():
    inventory = proxy_workers_from_environment(_environment(3))
    transports = []
    exit_ips = {
        worker.proxy_url: "8.8.4.%d" % (worker.slot_index + 1)
        for worker in inventory.workers
    }

    verified = preflight_proxy_workers(
        inventory,
        transport_probe=lambda proxy_url, **kwargs: transports.append(
            (proxy_url, kwargs)
        ),
        exit_ip_probe=lambda proxy_url, **_kwargs: exit_ips[proxy_url],
        native_ip_probe=lambda **_kwargs: "1.1.1.1",
        timeout_seconds=2.5,
    )

    assert {item[0] for item in transports} == set(exit_ips)
    assert all(
        item[1]
        == {
            "destination_host": "openrouter.ai",
            "destination_port": 443,
            "timeout_seconds": 2.5,
        }
        for item in transports
    )
    assert [worker.exit_ip for worker in verified.workers] == [
        "8.8.4.2",
        "8.8.4.3",
        "8.8.4.4",
    ]
    assert verified.verified_workers[0].slot_index == 0
    assert verified.verified_workers[0].proxy_url is None
    assert verified.total_process_capacity == 4
    assert "secret" not in repr(verified)
    assert "8.8.4" not in repr(verified)


@pytest.mark.parametrize(
    ("worker_exits", "message"),
    (
        (("8.8.8.8", "8.8.8.8"), "matches proxy worker 1"),
        (("1.1.1.1", "8.8.8.8"), "matches the native coordinator"),
    ),
)
def test_preflight_rejects_duplicate_actual_exits(worker_exits, message):
    inventory = proxy_workers_from_environment(_environment(2))
    exit_ips = {
        worker.proxy_url: worker_exits[offset]
        for offset, worker in enumerate(inventory.workers)
    }

    with pytest.raises(ProxyWorkerPreflightError, match=message):
        preflight_proxy_workers(
            inventory,
            transport_probe=lambda *_args, **_kwargs: None,
            exit_ip_probe=lambda proxy_url, **_kwargs: exit_ips[proxy_url],
            native_ip_probe=lambda **_kwargs: "1.1.1.1",
        )


def test_preflight_failure_does_not_expose_proxy_secret():
    inventory = proxy_workers_from_environment(_environment(1))
    secret_url = inventory.workers[0].proxy_url

    def fail(proxy_url, **_kwargs):
        raise OSError("failed through " + proxy_url)

    with pytest.raises(ProxyWorkerPreflightError) as raised:
        preflight_proxy_workers(
            inventory,
            transport_probe=fail,
            exit_ip_probe=lambda *_args, **_kwargs: "8.8.8.8",
            native_ip_probe=lambda **_kwargs: "1.1.1.1",
        )

    assert secret_url not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_https_exit_probe_uses_explicit_proxy_policy_and_bounded_json(monkeypatch):
    observed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            observed.append(("read", size))
            return b'{"ip":"8.8.8.8"}'

    class Opener:
        def open(self, request, timeout):
            observed.append(("open", request.full_url, timeout))
            return Response()

    def build(*handlers):
        observed.append(("handlers", tuple(type(item) for item in handlers)))
        return Opener()

    monkeypatch.setattr(module, "build_opener", build)

    assert probe_https_exit_ip(_proxy(1), timeout_seconds=3) == "8.8.8.8"
    assert observed == [
        ("handlers", (module.ProxyHandler, module.HTTPSHandler)),
        ("open", module.DEFAULT_EXIT_IP_URL, 3.0),
        ("read", module._MAX_EXIT_IP_RESPONSE_BYTES + 1),
    ]


def test_exit_probe_rejects_non_tls_endpoint_before_network(monkeypatch):
    monkeypatch.setattr(
        module,
        "build_opener",
        lambda *_handlers: pytest.fail("network opener must not be built"),
    )

    with pytest.raises(ProxyWorkerPreflightError, match="endpoint is invalid"):
        probe_https_exit_ip(None, endpoint="http://api.ipify.org?format=json")


def test_shared_proxy_endpoint_hides_decoded_authorization():
    endpoint = proxy_transport.parse_http_connect_proxy_url(
        "http://worker%20name:very-secret@proxy.example.com:6162"
    )

    assert endpoint.scheme == "http"
    assert endpoint.host == "proxy.example.com"
    assert endpoint.port == 6162
    assert endpoint.authorization_header == "Basic d29ya2VyIG5hbWU6dmVyeS1zZWNyZXQ="
    assert "secret" not in repr(endpoint)
    assert "Basic" not in repr(endpoint)


def test_shared_connect_tunnel_accepts_pinned_global_ip_and_adds_proxy_auth():
    class Stream:
        def __init__(self):
            self.sent = []
            self.timeout = None

        def settimeout(self, value):
            self.timeout = value

        def sendall(self, value):
            self.sent.append(value)

        def recv(self, _size):
            return b"HTTP/1.1 200 Connection Established\r\n\r\n"

    stream = Stream()
    connected = []
    result = proxy_transport.open_http_connect_tunnel(
        "http://worker-1:secret-1@proxy-1.example.com:6162",
        "8.8.8.8",
        443,
        connector=lambda host, port: connected.append((host, port)) or stream,
        timeout_seconds=2,
    )

    assert result is stream
    assert connected == [("proxy-1.example.com", 6162)]
    assert stream.timeout == 2.0
    request = stream.sent[0]
    assert request.startswith(b"CONNECT 8.8.8.8:443 HTTP/1.1\r\n")
    assert b"Proxy-Authorization: Basic " in request


def test_pool_holds_stable_exclusive_slots_for_live_attempt_retries():
    verified = _verified_inventory(2)
    pool = ProxyWorkerPool(verified)

    native = pool.acquire(timeout=0)
    first = pool.acquire(timeout=0)
    second = pool.acquire(timeout=0)
    assert native.slot_index == 0
    assert native.proxy_url is None
    assert first.slot_index != second.slot_index
    assert pool.available == 0
    assert pool.capacity == 3
    assert pool.webshare_worker_count == 2
    assert pool.total_process_capacity == 3
    retry_urls = [first.proxy_url, first.proxy_url, first.proxy_url]
    assert retry_urls == [first.proxy_url] * 3
    assert "secret" not in repr(first)

    with pytest.raises(ProxyWorkerPoolError, match="no proxy worker slot"):
        pool.acquire(timeout=0)

    released_slot = first.slot_index
    first.release()
    independent_retry = pool.acquire(timeout=0)
    assert independent_retry.slot_index == released_slot

    native.release()
    second.release()
    independent_retry.release()
    assert pool.available == 3


@pytest.mark.parametrize("proxy_count", (9, 19))
def test_pool_leases_all_native_plus_proxy_capacity(proxy_count):
    pool = ProxyWorkerPool(_verified_inventory(proxy_count))

    leases = [pool.acquire(timeout=0) for _index in range(proxy_count + 1)]

    assert [lease.slot_index for lease in leases] == list(range(proxy_count + 1))
    assert leases[0].proxy_url is None
    assert all(lease.proxy_url is not None for lease in leases[1:])
    assert pool.capacity == proxy_count + 1
    assert pool.available == 0
    with pytest.raises(ProxyWorkerPoolError, match="no proxy worker slot"):
        pool.acquire(timeout=0)
    for lease in leases:
        lease.release()
    assert pool.available == proxy_count + 1


def test_blocking_pool_acquire_wakes_after_release():
    pool = ProxyWorkerPool(_verified_inventory(1))
    native = pool.acquire(timeout=0)
    first = pool.acquire(timeout=0)
    acquired = []
    ready = threading.Event()

    def wait_for_slot():
        ready.set()
        with pool.acquire(timeout=1) as lease:
            acquired.append(lease.slot_index)

    waiter = threading.Thread(target=wait_for_slot)
    waiter.start()
    assert ready.wait(timeout=1)
    first.release()
    waiter.join(timeout=1)

    assert acquired == [1]
    assert not waiter.is_alive()
    native.release()
    assert pool.available == 2


def test_quarantine_permanently_removes_native_and_proxy_slots():
    pool = ProxyWorkerPool(_verified_inventory(1))
    native = pool.acquire(timeout=0)
    proxy = pool.acquire(timeout=0)

    native.quarantine()
    native.quarantine()
    native.release()
    assert pool.quarantined == 1
    assert pool.available == 0

    proxy.release()
    reacquired = pool.acquire(timeout=0)
    assert reacquired.slot_index == 1
    reacquired.quarantine()
    reacquired.release()

    assert pool.quarantined == 2
    assert pool.available == 0
    with pytest.raises(ProxyWorkerPoolError, match="all proxy worker slots"):
        pool.acquire()


def test_released_lease_cannot_quarantine_a_reassigned_slot():
    pool = ProxyWorkerPool(_verified_inventory(1))
    old = pool.acquire(timeout=0)
    old.release()
    other = pool.acquire(timeout=0)
    current = pool.acquire(timeout=0)

    with pytest.raises(ProxyWorkerPoolError, match="released proxy worker lease"):
        old.quarantine()

    assert current.slot_index == old.slot_index
    other.release()
    current.release()


def test_recreated_pool_excludes_process_quarantined_exit_fingerprints():
    inventory = _verified_inventory(1)
    original = ProxyWorkerPool(inventory)
    native = original.acquire(timeout=0)
    assert native.slot_index == 0
    native.quarantine()

    recreated = ProxyWorkerPool(inventory)

    assert recreated.capacity == 2
    assert recreated.quarantined == 1
    assert recreated.available == 1
    remaining = recreated.acquire(timeout=0)
    assert remaining.slot_index == 1
    remaining.quarantine()
    with pytest.raises(ProxyWorkerPoolError, match="all proxy worker slots"):
        ProxyWorkerPool(inventory).acquire()
