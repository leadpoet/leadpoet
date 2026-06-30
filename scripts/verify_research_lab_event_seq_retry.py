"""Verify atomic event-seq append: audit hash is preserved byte-for-byte vs the
legacy inline form, seq-collisions (23505) are retried, non-seq errors propagate.

Run: python3 scripts/verify_research_lab_event_seq_retry.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.store as store  # noqa: E402


@contextlib.contextmanager
def patched(**overrides):
    sentinel = object()
    originals = {k: getattr(store, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(store, k, v)
        yield
    finally:
        for k, v in originals.items():
            if v is sentinel:
                delattr(store, k)
            else:
                setattr(store, k, v)


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


def test_hash_preserved(errors):
    captured = {}

    async def fake_seq(table, key_field, key_value):
        return 7

    async def fake_insert(table, row):
        captured["table"] = table
        captured["row"] = row
        return row

    with patched(next_event_seq=fake_seq, insert_row=fake_insert):
        row = asyncio.run(store.create_queue_event(
            run_id="r1", ticket_id="t1", event_type="queued",
            queue_priority=0, reason="x", worker_ref="w", event_doc={"a": 1}))

    # Exact legacy payload — hash must match byte-for-byte.
    expected_payload = {
        "run_id": "r1", "ticket_id": "t1", "seq": 7, "event_type": "queued",
        "queue_priority": 0, "worker_ref": "w", "reason": "x", "event_doc": {"a": 1},
    }
    check(errors, row["anchored_hash"] == store.canonical_hash(expected_payload),
          "[hash] anchored_hash changed vs legacy inline form")
    check(errors, row["seq"] == 7 and row["schema_version"] == "1.0" and row.get("event_id"),
          "[hash] row shape changed")
    check(errors, captured["table"] == "research_loop_run_queue_events", "[hash] wrong table")


def test_retry_on_seq_conflict(errors):
    seqs = iter([3, 4, 5])
    attempts = {"n": 0}

    async def fake_seq(table, key_field, key_value):
        return next(seqs)

    async def fake_insert(table, row):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "duplicate key value violates unique constraint "
                '"research_loop_run_queue_events_run_seq_key" (23505)')
        return row

    with patched(next_event_seq=fake_seq, insert_row=fake_insert):
        row = asyncio.run(store.create_queue_event(
            run_id="r1", ticket_id="t1", event_type="queued", queue_priority=0, reason="x"))
    check(errors, attempts["n"] == 2, f"[retry] expected 2 attempts, got {attempts['n']}")
    check(errors, row["seq"] == 4, f"[retry] expected re-read seq 4, got {row.get('seq')}")


def test_non_seq_error_propagates(errors):
    async def fake_seq(table, key_field, key_value):
        return 1

    async def fake_insert(table, row):
        raise RuntimeError("connection refused")  # not a seq conflict

    with patched(next_event_seq=fake_seq, insert_row=fake_insert):
        try:
            asyncio.run(store.create_queue_event(
                run_id="r1", ticket_id="t1", event_type="queued", queue_priority=0, reason="x"))
            errors.append("[propagate] expected non-seq error to raise")
        except RuntimeError as exc:
            check(errors, "connection refused" in str(exc), "[propagate] wrong error surfaced")


def test_detector(errors):
    check(errors, store._is_seq_conflict(RuntimeError("... unique constraint foo_seq_key ... 23505")),
          "[detector] should flag seq unique violation")
    check(errors, not store._is_seq_conflict(RuntimeError("duplicate key ... candidate_artifact_hash")),
          "[detector] non-seq unique violation must NOT be retried")
    check(errors, not store._is_seq_conflict(RuntimeError("network timeout")),
          "[detector] non-unique error must NOT be retried")


def main() -> int:
    errors: list[str] = []
    for t in (test_hash_preserved, test_retry_on_seq_conflict, test_non_seq_error_propagates, test_detector):
        try:
            t(errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{t.__name__}] raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — event-seq retry verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — event-seq retry verification (hash byte-preserved, retry on 23505 seq conflict, "
          "non-seq errors propagate, detector scoped to seq constraint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
