"""The PCR0 allowlist refresh must be single-flight with failure backoff.

The fetch used to run while holding the cache lock and, on failure with a
non-empty cache, never advanced last_fetch — so with GitHub slow or down
every PCR0 verification re-attempted the fetch, serialized behind a
10-second urlopen. These tests pin the fixed semantics: a failed refresh
backs off for a full TTL, concurrent callers keep the previous allowlist,
and a successful refresh installs the new values.
"""
import threading
import time

import pytest

from leadpoet_canonical import nitro


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(
        nitro,
        "_pcr0_cache",
        {
            "gateway_pcr0": ["aa" * 48],
            "validator_pcr0": ["bb" * 48],
            "last_fetch": 0,
            "fetch_error": None,
        },
    )


def test_failed_refresh_backs_off_and_keeps_previous_allowlist(monkeypatch):
    calls = []

    def failing_fetch():
        calls.append(1)
        raise RuntimeError("github down")

    monkeypatch.setattr(nitro, "_fetch_pcr0_allowlist_from_github", failing_fetch)
    nitro._refresh_pcr0_cache_if_needed()
    nitro._refresh_pcr0_cache_if_needed()
    nitro._refresh_pcr0_cache_if_needed()
    # One fetch attempt only: the failure reserved the TTL window (backoff),
    # and the previous allowlist survives for verification.
    assert len(calls) == 1
    assert nitro._pcr0_cache["gateway_pcr0"] == ["aa" * 48]
    assert nitro._pcr0_cache["fetch_error"] == "github down"


def test_successful_refresh_installs_new_values(monkeypatch):
    monkeypatch.setattr(
        nitro,
        "_fetch_pcr0_allowlist_from_github",
        lambda: {"gateway_pcr0": ["cc" * 48], "validator_pcr0": ["dd" * 48]},
    )
    nitro._refresh_pcr0_cache_if_needed()
    assert nitro._pcr0_cache["gateway_pcr0"] == ["cc" * 48]
    assert nitro._pcr0_cache["fetch_error"] is None
    assert nitro._pcr0_cache["last_fetch"] > 0


def test_slow_fetch_does_not_block_concurrent_readers(monkeypatch):
    release = threading.Event()

    def slow_fetch():
        release.wait(5)
        return {"gateway_pcr0": ["ee" * 48], "validator_pcr0": ["ff" * 48]}

    monkeypatch.setattr(nitro, "_fetch_pcr0_allowlist_from_github", slow_fetch)
    worker = threading.Thread(target=nitro._refresh_pcr0_cache_if_needed)
    worker.start()
    time.sleep(0.1)
    # While the fetch is in flight, a reader must get the previous allowlist
    # immediately instead of serializing behind the network call.
    start = time.monotonic()
    values = nitro.get_allowed_gateway_pcr0()
    elapsed = time.monotonic() - start
    assert values == ["aa" * 48]
    assert elapsed < 1.0
    release.set()
    worker.join(10)
    assert nitro._pcr0_cache["gateway_pcr0"] == ["ee" * 48]
