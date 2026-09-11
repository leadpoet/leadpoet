"""The miner-visible open pool must not inherit the database's tail latency.

`GET /fulfillment/requests/active` is the gateway's highest-traffic endpoint and
every miner polls the same pool. When the pool cache expires, exactly one caller
should refresh it; the rest must keep serving the pool they already have. If
they queue behind the refresh instead, one slow database query is paid by every
miner polling at that moment — which is what turned a 40ms endpoint into a 1.1s
endpoint for all concurrent pollers on 2026-08-24.
"""

import threading
import time

import pytest

from gateway.fulfillment import api


class _FakeQuery:
    """Minimal stand-in for the chained Supabase query builder."""

    def __init__(self, table):
        self._table = table

    def select(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    def gt(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        return self._table._execute()


class _SlowSupabase:
    """A Supabase client whose every query takes `delay` seconds."""

    def __init__(self, delay, rows):
        self.delay = delay
        self.rows = rows
        self.query_count = 0
        self.started = threading.Event()

    def table(self, _name):
        return _FakeQuery(self)

    def _execute(self):
        self.query_count += 1
        self.started.set()
        time.sleep(self.delay)
        return type("Resp", (), {"data": list(self.rows)})()


@pytest.fixture(autouse=True)
def _clear_cache():
    api._active_requests_cache["rows"] = None
    api._active_requests_cache["fetched_at_mono"] = 0.0
    yield
    api._active_requests_cache["rows"] = None
    api._active_requests_cache["fetched_at_mono"] = 0.0


def test_cold_cache_blocks_and_queries_once():
    """With nothing to serve, concurrent callers wait — but for ONE query."""
    supabase = _SlowSupabase(delay=0.2, rows=[{"request_id": "a"}])
    results = []

    def poll():
        results.append(api._fetch_active_request_rows(supabase, "cutoff"))

    threads = [threading.Thread(target=poll) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert supabase.query_count == 1
    assert all(r == [{"request_id": "a"}] for r in results)


def test_stale_pool_is_served_while_a_slow_refresh_runs(monkeypatch):
    """A slow refresh costs one request, not every concurrent poller."""
    monkeypatch.setattr(api, "_ACTIVE_REQUESTS_CACHE_TTL_SECONDS", 0.05)
    monkeypatch.setattr(api, "_ACTIVE_REQUESTS_STALE_CEILING_SECONDS", 30.0)

    supabase = _SlowSupabase(delay=1.0, rows=[{"request_id": "a"}])
    # Prime the cache, then let it go stale.
    api._fetch_active_request_rows(supabase, "cutoff")
    time.sleep(0.1)
    supabase.started.clear()

    refresher = threading.Thread(
        target=api._fetch_active_request_rows, args=(supabase, "cutoff")
    )
    refresher.start()
    assert supabase.started.wait(timeout=2.0)

    # A second poller arriving mid-refresh is served immediately.
    started = time.monotonic()
    rows = api._fetch_active_request_rows(supabase, "cutoff")
    elapsed = time.monotonic() - started

    refresher.join()

    assert rows == [{"request_id": "a"}]
    assert elapsed < 0.2, f"poller blocked on the in-flight refresh ({elapsed:.2f}s)"


def test_pool_older_than_the_ceiling_is_not_served(monkeypatch):
    """Past the stale ceiling a caller waits rather than serve a dead pool."""
    monkeypatch.setattr(api, "_ACTIVE_REQUESTS_CACHE_TTL_SECONDS", 0.05)
    monkeypatch.setattr(api, "_ACTIVE_REQUESTS_STALE_CEILING_SECONDS", 0.1)

    supabase = _SlowSupabase(delay=0.3, rows=[{"request_id": "a"}])
    api._fetch_active_request_rows(supabase, "cutoff")
    time.sleep(0.15)
    supabase.started.clear()

    refresher = threading.Thread(
        target=api._fetch_active_request_rows, args=(supabase, "cutoff")
    )
    refresher.start()
    assert supabase.started.wait(timeout=2.0)

    started = time.monotonic()
    rows = api._fetch_active_request_rows(supabase, "cutoff")
    elapsed = time.monotonic() - started
    refresher.join()

    assert rows == [{"request_id": "a"}]
    assert elapsed >= 0.1, "a pool past the ceiling was served without refreshing"
    assert supabase.query_count == 2, "the waiter re-used the refresh it waited for"


def test_fresh_cache_makes_no_query():
    supabase = _SlowSupabase(delay=0.0, rows=[{"request_id": "a"}])
    api._fetch_active_request_rows(supabase, "cutoff")
    api._fetch_active_request_rows(supabase, "cutoff")
    assert supabase.query_count == 1
