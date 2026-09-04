"""Weight-critical reads retry transient edge failures, not logic errors.

A single Cloudflare/Supabase-edge 5xx or a dropped connection during the
3-minute weight-submission window previously burned the whole 72-minute
epoch (observed at epoch 24085: PostgREST returned a Cloudflare 400
"JSON could not be generated" mid-window). These reads are idempotent, so a
bounded retry is safe. The classifier is an allowlist: genuine query errors
still fail closed with no extra attempts.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.research_lab import store


class _CloudflareEdgeError(Exception):
    """Shaped like postgrest APIError for a Cloudflare edge rejection."""

    def __init__(self):
        super().__init__(
            "{'message': 'JSON could not be generated', 'code': 400, "
            "'details': '<html><center>cloudflare</center></html>'}"
        )
        self.message = "JSON could not be generated"
        self.code = 400


class _PostgrestLogicError(Exception):
    """A genuine PostgREST/Postgres query error (SQLSTATE)."""

    def __init__(self):
        super().__init__("column foo does not exist")
        self.message = "column foo does not exist"
        self.code = "42703"


class _ConnectionResetError(Exception):
    pass


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(store.asyncio, "sleep", _no_sleep)


def test_transient_classifier_allowlist():
    assert store._is_transient_read_error(_CloudflareEdgeError()) is True
    assert store._is_transient_read_error(_ConnectionResetError()) is True
    assert store._is_transient_read_error(TimeoutError("timed out")) is True
    # Genuine logic errors and unknown errors must not be retried.
    assert store._is_transient_read_error(_PostgrestLogicError()) is False
    assert store._is_transient_read_error(ValueError("bad input")) is False


def test_retry_recovers_from_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _CloudflareEdgeError()
        return "ok"

    result = asyncio.run(
        store._execute_read_with_retry(flaky, label="test")
    )
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_and_raises_last_transient(monkeypatch):
    calls = {"n": 0}

    def always_transient():
        calls["n"] += 1
        raise _ConnectionResetError()

    with pytest.raises(_ConnectionResetError):
        asyncio.run(
            store._execute_read_with_retry(always_transient, label="test")
        )
    assert calls["n"] == store._TRANSIENT_READ_ATTEMPTS


def test_logic_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def logic_error():
        calls["n"] += 1
        raise _PostgrestLogicError()

    with pytest.raises(_PostgrestLogicError):
        asyncio.run(
            store._execute_read_with_retry(logic_error, label="test")
        )
    assert calls["n"] == 1


def test_select_one_retries_through_helper(monkeypatch):
    attempts = {"n": 0}

    class _Resp:
        data = [{"id": 7}]

    def fake_call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _CloudflareEdgeError()
        return _Resp()

    # Patch the write client so the inner _call raises/returns deterministically.
    class _Query:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return fake_call()

    class _Client:
        def table(self, *_a, **_k):
            return _Query()

    monkeypatch.setattr(store, "get_write_client", lambda: _Client())
    monkeypatch.setattr(store, "_apply_filters", lambda q, _f: q)

    row = asyncio.run(store.select_one("t", filters=(("id", 7),)))
    assert row == {"id": 7}
    assert attempts["n"] == 2


def test_select_one_rebuilds_retry_with_generator_filters(monkeypatch):
    attempts = {"n": 0}
    observed_filters = []

    class _Resp:
        data = [{"id": 7}]

    class _Query:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, field, value):
            observed_filters.append((attempts["n"], field, value))
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _CloudflareEdgeError()
            return _Resp()

    class _Client:
        def table(self, *_args, **_kwargs):
            return _Query()

    monkeypatch.setattr(store, "get_write_client", lambda: _Client())

    filters = ((field, value) for field, value in (("id", 7),))
    row = asyncio.run(store.select_one("t", filters=filters))

    assert row == {"id": 7}
    assert observed_filters == [(0, "id", 7), (1, "id", 7)]


def test_select_all_crosses_postgrest_page_with_identical_query(monkeypatch):
    rows = [{"id": index} for index in range(1_001)]
    ranges = []
    filters = []
    orders = []

    class _Response:
        def __init__(self, data):
            self.data = data

    class _Query:
        def __init__(self):
            self.start = 0
            self.end = -1

        def select(self, columns):
            assert columns == "id"
            return self

        def eq(self, field, value):
            filters.append((field, value))
            return self

        def order(self, field, *, desc):
            orders.append((field, desc))
            return self

        def range(self, start, end):
            self.start = start
            self.end = end
            ranges.append((start, end))
            return self

        def execute(self):
            return _Response(rows[self.start : self.end + 1])

    class _Client:
        def table(self, table):
            assert table == "events"
            return _Query()

    monkeypatch.setattr(store, "get_write_client", lambda: _Client())

    result = asyncio.run(
        store.select_all(
            "events",
            columns="id",
            filters=(("event_type", "promotion_checked"),),
            order_by=(("created_at", True),),
            batch_size=1_000,
            max_rows=2_000,
        )
    )

    assert result == rows
    assert ranges == [(0, 999), (1_000, 1_999)]
    assert filters == [
        ("event_type", "promotion_checked"),
        ("event_type", "promotion_checked"),
    ]
    assert orders == [("created_at", True), ("created_at", True)]


def test_json_null_filter_uses_postgrest_is_operator():
    calls = []

    class _Query:
        def is_(self, field, value):
            calls.append((field, value))
            return self

    query = _Query()
    assert (
        store._apply_filters(
            query,
            (
                (
                    "allocation_doc->>historical_compute_fallback_source_epoch",
                    "is",
                    "null",
                ),
            ),
        )
        is query
    )
    assert calls == [
        (
            "allocation_doc->>historical_compute_fallback_source_epoch",
            "null",
        )
    ]


def test_insert_rows_sends_one_nonempty_payload(monkeypatch):
    payloads = []

    class _Response:
        data = [{"id": 2}, {"id": 1}]

    class _Query:
        def insert(self, payload):
            payloads.append(payload)
            return self

        def execute(self):
            return _Response()

    class _Client:
        def table(self, table):
            assert table == "events"
            return _Query()

    monkeypatch.setattr(store, "get_write_client", lambda: _Client())

    result = asyncio.run(
        store.insert_rows("events", ({"id": value} for value in (1, 2)))
    )

    assert payloads == [[{"id": 1}, {"id": 2}]]
    assert result == [{"id": 2}, {"id": 1}]


def test_insert_rows_rejects_empty_payload_before_client_access(monkeypatch):
    monkeypatch.setattr(
        store,
        "get_write_client",
        lambda: pytest.fail("empty batches must not open a database client"),
    )

    with pytest.raises(ValueError, match="requires at least one row"):
        asyncio.run(store.insert_rows("events", ()))


def test_retry_survives_a_run_of_transient_drops():
    """A read that is dropped repeatedly still completes.

    The 2026-09-04 edge regime dropped a fraction of reads for hours. A
    bundle build makes hundreds of reads and fails whole if any one of them
    exhausts, so the per-read allowance has to absorb a run of drops, not
    just an isolated one.
    """

    calls = {"n": 0}

    def dropped_five_times():
        calls["n"] += 1
        if calls["n"] <= 5:
            raise _CloudflareEdgeError()
        return "ok"

    result = asyncio.run(
        store._execute_read_with_retry(dropped_five_times, label="test")
    )
    assert result == "ok"
    assert calls["n"] == 6


def test_exhausted_read_sleeps_a_bounded_total():
    """The widened allowance stays inside the preparation fetch budget."""

    capped = store._TRANSIENT_READ_BACKOFF_SECONDS[-1]
    total = sum(
        store._TRANSIENT_READ_BACKOFF_SECONDS[
            min(attempt, len(store._TRANSIENT_READ_BACKOFF_SECONDS) - 1)
        ]
        for attempt in range(store._TRANSIENT_READ_ATTEMPTS - 1)
    )
    assert store._TRANSIENT_READ_BACKOFF_SECONDS[-1] == capped
    assert total <= 10.0
