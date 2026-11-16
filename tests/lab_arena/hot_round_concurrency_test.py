"""Concurrent runner reads must share the existing short-lived round cache."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from lab_arena import service as module


def service(read):
    value = object.__new__(module.ArenaService)
    value._hot_rounds = {}
    value._hot_round_lock = threading.Lock()
    value._require_round_ownership = lambda _: None
    value._require_round_mode = lambda row: row
    value._store = SimpleNamespace(get_round=read)
    return value


class ObservedLock:
    def __init__(self, expected):
        self.lock = threading.Lock()
        self.count_lock = threading.Lock()
        self.count = 0
        self.expected = expected
        self.all_entered = threading.Event()

    def __enter__(self):
        with self.count_lock:
            self.count += 1
            if self.count >= self.expected:
                self.all_entered.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("slow", [False, True])
def test_concurrent_miss_reads_large_round_once(monkeypatch, expired, slow):
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    release = threading.Event()
    reads = []
    row = {"round_id": "round-a", "configuration_doc": {"policy": "unchanged"}}

    def read(round_id):
        reads.append(round_id)
        assert release.wait(3)
        return row

    value = service(read)
    value._hot_round_lock = ObservedLock(8)
    if expired:
        value._hot_rounds["round-a"] = (0.0, {"old": True})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [pool.submit(value._hot_round, "round-a") for _ in range(8)]
        try:
            assert value._hot_round_lock.all_entered.wait(3)
        finally:
            if slow:
                now[0] += module.HOT_ROUND_TTL_SECONDS
            release.set()
        assert all(result.result(timeout=3) is row for result in results)
    assert reads == ["round-a"]


def test_ttl_and_explicit_invalidation_still_refresh(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    reads = []

    def read(round_id):
        row = {"round_id": round_id, "revision": len(reads)}
        reads.append(row)
        return row

    value = service(read)
    first = value._hot_round("round-a")
    now[0] = module.HOT_ROUND_TTL_SECONDS - 0.001
    assert value._hot_round("round-a") is first
    now[0] = module.HOT_ROUND_TTL_SECONDS
    assert value._hot_round("round-a") is not first
    value._invalidate_hot_round()
    assert value._hot_round("round-a")["revision"] == 2
    assert value._hot_round("round-b")["round_id"] == "round-b"


@pytest.mark.parametrize("failure", [None, RuntimeError("read unavailable")])
def test_missing_or_failed_read_does_not_poison_cache(failure):
    calls = []

    def read(round_id):
        calls.append(round_id)
        if len(calls) == 1:
            if failure:
                raise failure
            return None
        return {"round_id": round_id}

    value = service(read)
    if failure:
        with pytest.raises(RuntimeError, match="read unavailable"):
            value._hot_round("round-a")
    else:
        assert value._hot_round("round-a") is None
    assert value._hot_round("round-a") == {"round_id": "round-a"}
    assert len(calls) == 2


def test_scope_and_mode_checks_are_not_skipped():
    value = service(lambda _: {"round_id": "round-a"})
    value._require_round_mode = lambda _: (_ for _ in ()).throw(ValueError("wrong mode"))
    with pytest.raises(ValueError, match="wrong mode"):
        value._hot_round("round-a")
    assert not value._hot_rounds
    value._hot_rounds["round-a"] = (float("inf"), {"round_id": "round-a"})
    value._require_round_ownership = lambda _: (_ for _ in ()).throw(ValueError("wrong scope"))
    with pytest.raises(ValueError, match="wrong scope"):
        value._hot_round("round-a")


def test_invalidation_cannot_be_overwritten_by_inflight_refresh(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    started = threading.Event()
    release = threading.Event()
    reads = []

    def read(round_id):
        reads.append(round_id)
        if len(reads) == 1:
            started.set()
            assert release.wait(3)
        return {"round_id": round_id, "revision": len(reads)}

    value = service(read)
    value._hot_round_lock = ObservedLock(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh = pool.submit(value._hot_round, "round-a")
        assert started.wait(3)
        invalidation = pool.submit(value._invalidate_hot_round)
        try:
            assert value._hot_round_lock.all_entered.wait(3)
        finally:
            release.set()
        assert refresh.result(timeout=3)["revision"] == 1
        invalidation.result(timeout=3)
    assert value._hot_round("round-a")["revision"] == 2


def test_slow_read_does_not_extend_original_ttl(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    reads = []

    def read(round_id):
        reads.append(round_id)
        now[0] += module.HOT_ROUND_TTL_SECONDS
        return {"round_id": round_id, "revision": len(reads)}

    value = service(read)
    assert value._hot_round("round-a")["revision"] == 1
    assert value._hot_round("round-a")["revision"] == 2
