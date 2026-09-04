"""The metagraph warmer must not block request-path cache reads."""

import threading
from types import SimpleNamespace

from gateway.utils import registry


def test_warm_metagraph_cache_releases_cache_lock_during_network_fetch(
    monkeypatch,
):
    fetch_started = threading.Event()
    release_fetch = threading.Event()
    results: list[bool] = []

    class _Subtensor:
        def __init__(self, **_kwargs):
            pass

        def metagraph(self, **_kwargs):
            fetch_started.set()
            assert release_fetch.wait(timeout=2)
            return SimpleNamespace(hotkeys=["hk"])

    monkeypatch.setattr(registry.bt, "Subtensor", _Subtensor)
    monkeypatch.setattr(registry, "_warm_lock", threading.Lock())
    monkeypatch.setattr(registry, "_metagraph_cache", None)
    monkeypatch.setattr(registry, "_cache_epoch", None)
    monkeypatch.setattr(registry, "_cache_epoch_timestamp", None)

    worker = threading.Thread(
        target=lambda: results.append(registry.warm_metagraph_cache(101)),
        daemon=True,
    )
    worker.start()
    acquired = False
    try:
        assert fetch_started.wait(timeout=1)
        acquired = registry._cache_lock.acquire(timeout=0.2)
        assert acquired
        registry._metagraph_cache = SimpleNamespace(hotkeys=["newer-hk"])
        registry._cache_epoch = 102
        registry._cache_epoch_timestamp = 1.0
    finally:
        if acquired:
            registry._cache_lock.release()
        release_fetch.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert results == [True]
    assert registry._cache_epoch == 102
    assert registry._metagraph_cache.hotkeys == ["newer-hk"]
