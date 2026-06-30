"""Verify _alert_stuck_candidates emits alerts for stuck baseline / un-rebased
stale-parent candidates and never mutates or raises.

Run: python3 scripts/verify_research_lab_stuck_alerts.py  (exit 0 == pass)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway.research_lab.scoring_worker as sw  # noqa: E402


@contextlib.contextmanager
def patched(**overrides):
    sentinel = object()
    originals = {k: getattr(sw, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(sw, k, v)
        yield
    finally:
        for k, v in originals.items():
            if v is sentinel:
                delattr(sw, k)
            else:
                setattr(sw, k, v)


def _bare():
    w = sw.ResearchLabGatewayScoringWorker.__new__(sw.ResearchLabGatewayScoringWorker)
    w.worker_ref = "test-scorer"
    w.config = SimpleNamespace(scoring_worker_baseline_not_ready_retry_seconds=900)
    return w


def check(errors, cond, msg):
    if not cond:
        errors.append(msg)


def test_alerts(errors):
    async def sm(table, **kwargs):
        filters = dict((f[0], f[1]) for f in kwargs.get("filters", ()))
        if table == "research_lab_candidate_evaluation_current":
            if filters.get("current_reason") == "baseline_not_ready":
                return [{"candidate_id": "candidate:b1", "current_status_at": "2020-01-01T00:00:00+00:00"}]
            if filters.get("current_reason") == "stale_parent_needs_rescore":
                return [{"candidate_id": "candidate:s1", "current_status_at": "2020-01-01T00:00:00+00:00"}]
        if table == "research_lab_candidate_promotion_events":
            return []  # no rebase yet -> overdue
        return []

    # force "stale" True for old timestamps
    def stale(value, seconds):
        return True

    records: list[str] = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    handler = H()
    sw.logger.addHandler(handler)
    try:
        w = _bare()
        with patched(select_many=sm, _status_is_stale=stale):
            asyncio.run(w._alert_stuck_candidates())  # must not raise
    finally:
        sw.logger.removeHandler(handler)

    joined = " ".join(records)
    check(errors, "research_lab_candidates_stuck_baseline_not_ready" in joined,
          "[alerts] baseline-stuck alert not emitted")
    check(errors, "research_lab_stale_parent_candidates_overdue" in joined,
          "[alerts] stale-parent-overdue alert not emitted")

    # rebased ones (promotion event exists) -> NOT overdue
    records.clear()

    async def sm2(table, **kwargs):
        filters = dict((f[0], f[1]) for f in kwargs.get("filters", ()))
        if table == "research_lab_candidate_evaluation_current" and filters.get("current_reason") == "stale_parent_needs_rescore":
            return [{"candidate_id": "candidate:s2", "current_status_at": "2020-01-01T00:00:00+00:00"}]
        if table == "research_lab_candidate_promotion_events":
            return [{"promotion_event_id": "p1"}]  # already rebased
        return []

    sw.logger.addHandler(handler)
    try:
        with patched(select_many=sm2, _status_is_stale=stale):
            asyncio.run(_bare()._alert_stuck_candidates())
    finally:
        sw.logger.removeHandler(handler)
    check(errors, "research_lab_stale_parent_candidates_overdue" not in " ".join(records),
          "[alerts] already-rebased candidate wrongly flagged overdue")


def main() -> int:
    errors: list[str] = []
    try:
        test_alerts(errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"raised {type(exc).__name__}: {exc}")
    if errors:
        print("FAIL — stuck-candidate alerts verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — stuck-candidate alerts verification (baseline-stuck + stale-parent-overdue alerts, "
          "already-rebased excluded, no mutation/raise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
