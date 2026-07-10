import asyncio

import pytest
from fastapi import HTTPException

from gateway.middleware.body_size import BodySizeLimitMiddleware
from gateway.middleware.priority import classify_path
from gateway.utils.circuit_breaker import CircuitBreaker
from gateway.utils.hotkey_bucket import HotkeyBuckets, enforce, observe


def test_priority_route_classification_is_explicit():
    assert classify_path("/weights/submit") == "validator"
    assert classify_path("/weights/submit/v2") == "validator"
    assert classify_path("/fulfillment/scoring") == "validator"
    assert classify_path("/fulfillment/rewards/active") == "validator"
    assert classify_path("/fulfillment/requests/active") == "miner"
    assert classify_path("/fulfillment/commit") == "miner"
    assert classify_path("/fulfillment/reveal") == "miner"
    assert classify_path("/fulfillment/results/abc") == "validator"
    assert classify_path("/health") == "other"
    assert classify_path("/not-weights/submit") == "other"


def test_body_size_middleware_rejects_large_content_length():
    sent = []
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = BodySizeLimitMiddleware(app, max_body_bytes=4)
    scope = {
        "type": "http",
        "headers": [(b"content-length", b"5")],
    }
    asyncio.run(middleware(scope, receive, send))

    assert called is False
    assert sent[0]["status"] == 413


def test_hotkey_bucket_enforce_and_observe_only():
    bucket = HotkeyBuckets("test", rate_per_min=0.001, burst=1)

    enforce(bucket, "5test")
    with pytest.raises(HTTPException) as exc:
        enforce(bucket, "5test")
    assert exc.value.status_code == 429

    # Observe-only mode records pressure but never raises. This is what
    # unauthenticated active-request polling uses to avoid third-party quota
    # griefing against a real miner hotkey.
    observe(bucket, "5test")
    assert bucket.snapshot()["observed_denied"] >= 1


def test_circuit_breaker_opens_and_recovers(monkeypatch):
    monkeypatch.setattr("gateway.utils.circuit_breaker.FAILURE_THRESHOLD", 2)
    monkeypatch.setattr("gateway.utils.circuit_breaker.OPEN_SECONDS", 0.001)
    breaker = CircuitBreaker()

    assert breaker.before_call() is True
    breaker.record_failure()
    assert breaker.before_call() is True
    breaker.record_failure()
    assert breaker.snapshot()["state"] == "open"
    assert breaker.before_call() is False

    import time

    time.sleep(0.002)
    assert breaker.before_call() is True
    breaker.record_success()
    assert breaker.snapshot()["state"] == "closed"


@pytest.mark.asyncio
async def test_db_executor_sheds_when_queue_is_saturated():
    import gateway.utils.db_executor as db_executor

    with db_executor._lock:
        original = db_executor._in_flight
        db_executor._in_flight = db_executor.DB_QUEUE_HIGH_WATER
    try:
        with pytest.raises(HTTPException) as exc:
            await db_executor.run_db(lambda: "unused")
        assert exc.value.status_code == 503
    finally:
        with db_executor._lock:
            db_executor._in_flight = original
