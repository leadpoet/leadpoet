"""Verify the stale loop-projection reconciler + terminal-projection retry/alert.

Harness: monkeypatch worker.py's imported store fns with fakes, drive the
methods on a bare ResearchLabHostedWorker (built via __new__ to skip __init__).

Run: python3 scripts/verify_research_lab_loop_reconciler.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.worker as worker  # noqa: E402


@contextlib.contextmanager
def patched(**overrides):
    sentinel = object()
    originals = {k: getattr(worker, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(worker, k, v)
        yield
    finally:
        for k, v in originals.items():
            if v is sentinel:
                delattr(worker, k)
            else:
                setattr(worker, k, v)


def _bare_worker():
    w = worker.ResearchLabHostedWorker.__new__(worker.ResearchLabHostedWorker)
    w.worker_ref = "test-worker"
    return w


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


def test_reconciler(errors):
    created = []

    async def fake_create(**kwargs):
        created.append(kwargs)
        return {"event_id": "e", "seq": 1, "anchored_hash": "sha256:x"}

    # loop running; queue completed -> should append loop_completed
    async def sm_running(table, **kwargs):
        if table == "research_lab_auto_research_loop_current":
            return [{"run_id": "run-1", "ticket_id": "t-1", "current_loop_status": "running"}]
        return []

    async def so_completed(table, **kwargs):
        return {"run_id": "run-1", "ticket_id": "t-1", "current_queue_status": "completed"}

    w = _bare_worker()
    with patched(select_many=sm_running, select_one=so_completed, create_auto_research_loop_event=fake_create):
        n = asyncio.run(w._reconcile_stale_loop_projections())
    check(errors, n == 1 and len(created) == 1, f"[reconcile/completed] expected 1 reconcile, got n={n} created={len(created)}")
    if created:
        check(errors, created[0]["event_type"] == "loop_completed" and created[0]["loop_status"] == "completed",
              "[reconcile/completed] wrong terminal event")
        check(errors, created[0]["event_doc"]["reason"] == "stale_loop_projection_reconciled",
              "[reconcile/completed] missing reason")

    # queue failed -> loop_failed
    created.clear()

    async def so_failed(table, **kwargs):
        return {"run_id": "run-1", "ticket_id": "t-1", "current_queue_status": "failed"}

    with patched(select_many=sm_running, select_one=so_failed, create_auto_research_loop_event=fake_create):
        asyncio.run(w._reconcile_stale_loop_projections())
    check(errors, created and created[0]["event_type"] == "loop_failed" and created[0]["loop_status"] == "failed",
          "[reconcile/failed] expected loop_failed")

    # queue still active (started) -> NO reconcile
    created.clear()

    async def so_started(table, **kwargs):
        return {"run_id": "run-1", "ticket_id": "t-1", "current_queue_status": "started"}

    with patched(select_many=sm_running, select_one=so_started, create_auto_research_loop_event=fake_create):
        n = asyncio.run(w._reconcile_stale_loop_projections())
    check(errors, n == 0 and len(created) == 0, "[reconcile/active] must not finalize an active run")

    # idempotency: no running loops -> no work
    created.clear()

    async def sm_empty(table, **kwargs):
        return []

    with patched(select_many=sm_empty, select_one=so_completed, create_auto_research_loop_event=fake_create):
        n = asyncio.run(w._reconcile_stale_loop_projections())
    check(errors, n == 0 and len(created) == 0, "[reconcile/idempotent] no running loops => no-op")


def test_terminal_projection_retry(errors):
    # _ensure_terminal_loop_projection should retry transient failures then succeed.
    attempts = {"n": 0}

    async def fake_create_flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise worker.RetryableHostedResearchLabWorkerError("transient store write")
        return {"event_id": "e", "seq": 2, "anchored_hash": "sha256:y"}

    async def so_running(table, **kwargs):
        return {"run_id": "run-9", "current_loop_status": "running"}

    class Ctx:
        run_id = "run-9"
        ticket_id = "t-9"
        receipt_id = None

    w = _bare_worker()
    # _provider_usage is called inside; stub it
    w._provider_usage = lambda ctx: []
    with patched(select_one=so_running, create_auto_research_loop_event=fake_create_flaky):
        asyncio.run(w._ensure_terminal_loop_projection(
            Ctx(), event_type="loop_failed", loop_status="failed", reason="test"))
    check(errors, attempts["n"] == 2, f"[terminal/retry] expected retry then success, attempts={attempts['n']}")

    # already-terminal short-circuit: no create
    attempts["n"] = 0

    async def so_done(table, **kwargs):
        return {"run_id": "run-9", "current_loop_status": "completed"}

    with patched(select_one=so_done, create_auto_research_loop_event=fake_create_flaky):
        asyncio.run(w._ensure_terminal_loop_projection(
            Ctx(), event_type="loop_completed", loop_status="completed", reason="test"))
    check(errors, attempts["n"] == 0, "[terminal/idempotent] already-terminal must not re-append")


def main() -> int:
    errors: list[str] = []
    for t in (test_reconciler, test_terminal_projection_retry):
        try:
            t(errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{t.__name__}] raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — loop reconciler verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — loop reconciler verification (reconcile completed/failed, skip active, idempotent, "
          "terminal-projection retry + already-terminal short-circuit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
