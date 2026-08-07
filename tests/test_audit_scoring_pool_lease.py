"""RL3: the global scoring pool must not evict a still-live slot owner on age
alone. A candidate run can outlive the lease (no heartbeat); evicting it lets a
second process re-claim the slot and over-subscribe past `size`, defeating the
OOM guard."""
import os
from research_lab.eval.global_scoring_pool import GlobalScoringSlotPool

NOW = 1_000_000.0


def _pool(tmp_path, lease=100.0):
    return GlobalScoringSlotPool(str(tmp_path / "pool.json"), size=2, lease_seconds=lease)


def test_live_owner_kept_past_lease(tmp_path):
    p = _pool(tmp_path)  # lease 100, hard_cap 400
    slots = [{"token": "t1", "pid": os.getpid(), "ts": NOW - 300.0}]  # age 300 > lease
    assert len(p._live_slots(slots, NOW)) == 1  # was wrongly evicted before fix


def test_dead_owner_reclaimed_even_if_fresh(tmp_path):
    p = _pool(tmp_path)
    slots = [{"token": "t1", "pid": 2_000_000_000, "ts": NOW - 5.0}]  # dead pid, fresh
    assert p._live_slots(slots, NOW) == []


def test_live_owner_past_hard_cap_reclaimed(tmp_path):
    p = _pool(tmp_path)  # hard_cap 400
    slots = [{"token": "t1", "pid": os.getpid(), "ts": NOW - 5000.0}]  # age 5000 > cap
    assert p._live_slots(slots, NOW) == []


def test_no_pid_slot_uses_lease(tmp_path):
    p = _pool(tmp_path)
    assert len(p._live_slots([{"token": "a", "pid": 0, "ts": NOW - 50.0}], NOW)) == 1
    assert p._live_slots([{"token": "b", "pid": 0, "ts": NOW - 500.0}], NOW) == []
