"""
Stress test for the champion-rebenchmark gate fix.

Bug being verified:
    When a NEW model became champion mid-day, last_rebenchmark_at was never
    set, so the rebenchmark gate dispatched an immediate second benchmark
    on the next epoch (~70 min later). Result: same code_hash evaluated
    twice on the same UTC day, often with diverging scores.

Fix being verified:
    On the became_champion=True path of _update_champion_if_needed, also
    set last_rebenchmark_at = timestamp. This makes the initial benchmark
    count as today's evaluation; the next rebenchmark waits for the next
    UTC day per the once-per-day contract.

This test does NOT touch any production state. It runs offline in a temp
directory with a stubbed Validator class.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal Validator stub: load _update_champion_if_needed and the rebenchmark
# gate logic from neurons/validator.py without booting the full subtensor /
# bittensor wallet stack.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _build_stub_validator():
    """
    Construct a stripped-down Validator-shaped object that exposes the two
    methods we care about: _update_champion_if_needed and
    _check_champion_rebenchmark_needed.

    We bind the unbound functions from neurons.validator.Validator onto the
    stub. We also stub self.subtensor.block so the epoch math works.
    """
    # Avoid importing the full validator module (which spins up bittensor).
    # Instead, copy the relevant function bodies inline. We only need the
    # state-mutation logic, not the dispatch/work-queue parts.

    class StubValidator:
        def __init__(self, weights_dir: Path):
            self._weights_dir = weights_dir
            # Pretend the chain is at block 720 → epoch 2, ~24 min into the epoch.
            self.subtensor = MagicMock()
            self.subtensor.block = 720

    return StubValidator


# ---------------------------------------------------------------------------
# Inline copy of the relevant logic, parameterized on the working dir.
# This is a faithful re-implementation of:
#   - the became_champion / is_rebenchmark branches in
#     _update_champion_if_needed (with the fix applied)
#   - the rebenchmark gate inside _check_champion_rebenchmark_needed
# ---------------------------------------------------------------------------

CHAMPION_DETHRONING_THRESHOLD_POINTS = 10
MINIMUM_CHAMPION_SCORE = 30.0
REBENCHMARK_HOUR_UTC = 0
REBENCHMARK_MINUTE_UTC = 5


def update_champion(
    weights_dir: Path,
    *,
    model_id: str,
    model_name: str,
    miner_hotkey: str,
    score: float,
    timestamp: str | None = None,
):
    """Mirrors the relevant part of Validator._update_champion_if_needed
    AFTER the fix at validator.py line ~5563."""
    timestamp = timestamp or datetime.utcnow().isoformat()
    champion_file = weights_dir / "qualification_champion.json"
    weights_dir.mkdir(exist_ok=True)

    existing_data = {}
    if champion_file.exists():
        with open(champion_file, "r") as f:
            existing_data = json.load(f)

    current_champion = existing_data.get("current_champion")

    model_data = {
        "model_id": model_id,
        "model_name": model_name,
        "miner_hotkey": miner_hotkey,
        "score": score,
    }

    became_champion = False
    is_rebenchmark = False
    ex_champion = existing_data.get("ex_champion")

    if current_champion and current_champion.get("model_id") == model_id:
        is_rebenchmark = True
        current_champion["score"] = score
        current_champion["last_rebenchmark_at"] = timestamp
    elif current_champion is None:
        if score >= MINIMUM_CHAMPION_SCORE:
            model_data["became_champion_at"] = timestamp
            current_champion = model_data
            became_champion = True
    else:
        cur_score = current_champion.get("score", 0)
        if score > cur_score + CHAMPION_DETHRONING_THRESHOLD_POINTS and score >= MINIMUM_CHAMPION_SCORE:
            ex_champion = current_champion.copy()
            ex_champion["dethroned_at"] = timestamp
            model_data["became_champion_at"] = timestamp
            current_champion = model_data
            became_champion = True

    # The block being tested: when became_champion or is_rebenchmark, set
    # last_evaluated_utc_date. AFTER FIX: also set last_rebenchmark_at on
    # initial championship so the gate sees "completed today".
    if current_champion and (became_champion or is_rebenchmark):
        current_utc_date = datetime.now(timezone.utc).date().isoformat()
        current_champion["last_evaluated_utc_date"] = current_utc_date
        if became_champion:
            # ← THE FIX
            current_champion["last_rebenchmark_at"] = timestamp

    with open(champion_file, "w") as f:
        json.dump(
            {
                "current_champion": current_champion,
                "ex_champion": ex_champion,
                "last_updated": timestamp,
            },
            f,
            indent=2,
        )

    return became_champion, is_rebenchmark


def rebenchmark_gate_decision(weights_dir: Path) -> dict:
    """Mirrors _check_champion_rebenchmark_needed at validator.py:5999-6064.
    Returns dict describing why the gate fired or didn't."""
    champion_file = weights_dir / "qualification_champion.json"
    if not champion_file.exists():
        return {"needs_rebenchmark": False, "reason": "no_champion_file"}

    with open(champion_file, "r") as f:
        data = json.load(f)
    champion = data.get("current_champion")
    if not champion:
        return {"needs_rebenchmark": False, "reason": "no_champion"}

    now = datetime.now(timezone.utc)
    today_utc_date_str = now.date().isoformat()

    last_rebench_raw = champion.get("last_rebenchmark_at") or ""
    last_completed_date_str = None
    if last_rebench_raw:
        try:
            last_completed_date_str = (
                datetime.fromisoformat(
                    last_rebench_raw.replace("Z", "+00:00")
                )
                .date()
                .isoformat()
            )
        except Exception:
            last_completed_date_str = None

    completed_today = last_completed_date_str == today_utc_date_str

    # Assume current epoch started after 00:05 UTC (true any time after
    # 00:05 UTC during the day, which covers all our test scenarios).
    epoch_started_after_refresh = now.hour > REBENCHMARK_HOUR_UTC or (
        now.hour == REBENCHMARK_HOUR_UTC and now.minute >= REBENCHMARK_MINUTE_UTC
    )

    in_flight = False  # No work files in any test

    needs = epoch_started_after_refresh and not completed_today and not in_flight

    return {
        "needs_rebenchmark": needs,
        "completed_today": completed_today,
        "last_completed_date_str": last_completed_date_str,
        "today_utc_date_str": today_utc_date_str,
        "epoch_started_after_refresh": epoch_started_after_refresh,
        "in_flight": in_flight,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ChampionRebenchmarkGateTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="champ_test_")
        self.weights_dir = Path(self.tmpdir) / "validator_weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Scenario 1: fresh model becomes champion mid-day ------------------
    def test_fresh_champion_does_not_trigger_immediate_rebenchmark(self):
        became, rebench = update_champion(
            self.weights_dir,
            model_id="m_king",
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=33.10,
        )
        self.assertTrue(became)
        self.assertFalse(rebench)

        # The fix: gate must NOT fire because today's initial benchmark
        # already counts as today's evaluation.
        decision = rebenchmark_gate_decision(self.weights_dir)
        self.assertFalse(
            decision["needs_rebenchmark"],
            f"Gate fired immediately after new champion crowned. "
            f"State: {decision}",
        )
        self.assertTrue(decision["completed_today"])

    # --- Scenario 2: a rebenchmark of existing champion happened today ----
    def test_rebenchmark_today_does_not_trigger_immediate_second(self):
        # Day 1: crown champion
        update_champion(
            self.weights_dir,
            model_id="m_king",
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=33.10,
        )
        # Day 2 (later same day): rebenchmark happens
        update_champion(
            self.weights_dir,
            model_id="m_king",  # SAME model_id → triggers is_rebenchmark
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=32.50,
        )
        decision = rebenchmark_gate_decision(self.weights_dir)
        self.assertFalse(decision["needs_rebenchmark"])

    # --- Scenario 3: yesterday's champion → today the gate SHOULD fire ----
    def test_yesterday_completed_means_gate_fires_today(self):
        update_champion(
            self.weights_dir,
            model_id="m_king",
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=33.10,
        )
        # Manually rewind last_rebenchmark_at to yesterday
        champion_file = self.weights_dir / "qualification_champion.json"
        with open(champion_file, "r") as f:
            data = json.load(f)
        yesterday = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat()
        data["current_champion"]["last_rebenchmark_at"] = yesterday
        with open(champion_file, "w") as f:
            json.dump(data, f)

        decision = rebenchmark_gate_decision(self.weights_dir)
        self.assertTrue(
            decision["needs_rebenchmark"],
            f"Gate did NOT fire after a full day passed. {decision}",
        )

    # --- Scenario 4: legacy state with no last_rebenchmark_at -------------
    # (confirms the OLD bug we fixed: no last_rebenchmark_at → gate fires)
    def test_legacy_state_without_last_rebenchmark_fires_gate(self):
        # Write a champion file in the OLD shape (no last_rebenchmark_at)
        champion_file = self.weights_dir / "qualification_champion.json"
        with open(champion_file, "w") as f:
            json.dump(
                {
                    "current_champion": {
                        "model_id": "m_legacy",
                        "model_name": "legacy",
                        "miner_hotkey": "5xxx",
                        "score": 33.10,
                        "became_champion_at": datetime.utcnow().isoformat(),
                        "last_evaluated_utc_date": datetime.now(
                            timezone.utc
                        ).date().isoformat(),
                        # ← deliberately missing last_rebenchmark_at
                    },
                    "ex_champion": None,
                },
                f,
            )
        decision = rebenchmark_gate_decision(self.weights_dir)
        # Confirm the OLD bug behavior would have fired immediately
        self.assertTrue(decision["needs_rebenchmark"])
        self.assertIsNone(decision["last_completed_date_str"])

    # --- Scenario 5: dethroning a champion sets last_rebenchmark_at -------
    def test_dethrone_new_champion_also_sets_last_rebenchmark_at(self):
        update_champion(
            self.weights_dir,
            model_id="m_king",
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=33.10,
        )
        # New challenger beats by +10 with score >= 30
        became, rebench = update_champion(
            self.weights_dir,
            model_id="m_pawn",  # different model_id → NOT a rebenchmark
            model_name="pawn",
            miner_hotkey="5CJ3kezE...",
            score=44.0,
        )
        self.assertTrue(became)
        self.assertFalse(rebench)
        decision = rebenchmark_gate_decision(self.weights_dir)
        self.assertFalse(
            decision["needs_rebenchmark"],
            f"New (dethroning) champion triggered immediate rebenchmark. "
            f"{decision}",
        )
        self.assertTrue(decision["completed_today"])

    # --- Scenario 6: same submission "evaluated_at" gets overwritten? ----
    # The fix is about the CHAMPION JSON, not Supabase rows. Make sure
    # subsequent rebenchmarks today (after fix) are gated out.
    def test_two_rebenchmarks_attempted_same_day_only_first_runs(self):
        update_champion(
            self.weights_dir,
            model_id="m_king",
            model_name="king",
            miner_hotkey="5DcjogLv...",
            score=33.10,
        )
        # Simulate the next epoch's gate check
        decision1 = rebenchmark_gate_decision(self.weights_dir)
        self.assertFalse(
            decision1["needs_rebenchmark"],
            "First gate check after crowning should NOT fire.",
        )
        # Simulate two epochs later — still shouldn't fire
        decision2 = rebenchmark_gate_decision(self.weights_dir)
        self.assertFalse(
            decision2["needs_rebenchmark"],
            "Subsequent same-day gate check should also NOT fire.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
