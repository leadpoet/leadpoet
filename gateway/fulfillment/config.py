"""
Fulfillment system configuration.

All values are read from environment variables with sensible defaults.
"""

import os
import logging

# Commit-window duration in epochs (1 epoch = 360 blocks × 12s = 72 min on
# this subnet).  Miners have this long, from the moment the request is
# promoted from pending to open, to source leads and submit a commit hash.
# Set to 1 (down from 2) on 2026-04-22 — observed miner pipelines finish
# well under an epoch and holding requests open longer just means more
# queued requests backing up behind FULFILLMENT_MAX_PARALLEL_REQUESTS.
T_EPOCHS = int(os.getenv("FULFILLMENT_T_EPOCHS", "1"))
T_SECONDS_OVERRIDE = int(os.getenv("FULFILLMENT_T_SECONDS", "0"))
M_MINUTES = int(os.getenv("FULFILLMENT_M_MINUTES", "15"))
BLOCK_TIME_SECONDS = 12
# Per-winning-lead emission share, paid every epoch for L_EPOCHS (100) epochs
# after the lead is fulfilled.  Trajectory:
#   2026-04-30: 0.001  → 0.0015 (per-lead bump over pool-share rebalance,
#               L_EPOCHS still 30, total per lead = 30 × 0.15% = 4.5%)
#   2026-05-11: 0.0015 → 0.0005 AND L_EPOCHS 30 → 100.  Total per lead is
#               now 100 × 0.05% = 5.0% (slight bump, ~11% lift).  The real
#               change is the reward runway: 100 epochs × 72 min/epoch =
#               7200 min = 120 hours = 5 days, up from ~36 hours.  Goal is
#               de-reg protection — a single fulfilled lead now keeps a
#               miner earning emission for ~5 days, so miners with even
#               one win in a low-volume window don't get pushed off the
#               subnet by the daily de-reg sweep.  Co-founder call.
#   2026-05-22: 0.0005 → 0.001 (true 2× lift; per epoch 0.05% → 0.1%;
#               total per lead 5% → 10% over the 100-epoch runway).  At
#               any epoch the validator caps SUM(active reward_pct) at
#               fulfillment_pool (neurons/validator.py::
#               _get_fulfillment_emission_share lines 3134–3139), so this
#               2× bump only translates to higher miner payouts during
#               LOW-VOLUME periods — when only one or two leads are
#               paying out simultaneously and the raw_total is below the
#               pool ceiling.  In high-volume periods the proportional
#               normalization absorbs the bump and per-miner payouts
#               are unchanged.  Goal: stronger de-reg protection for
#               miners with sparse wins, without inflating payouts
#               during busy periods.
#   2026-06-23: 0.001 → 0.004 (4× lift; per epoch 0.1% → 0.4%; total per
#               lead 10% → 40% over the 100-epoch runway).  Same pool-cap
#               caveat as 2026-05-22: the validator caps SUM(active
#               reward_pct) at fulfillment_pool, so this mainly raises
#               per-miner payouts during low-volume periods.
# Existing reward rows stay at their original (reward_pct, reward_expires_epoch)
# until expiry; only newly-fulfilled leads use the new rate AND new runway,
# so the rollover is gradual.  Old rows pay at their original per-epoch
# claim size for the remainder of their 100-epoch window (~5 days), at
# which point all live rows are paying the new 0.004 rate.
Z_PERCENT = float(os.getenv("FULFILLMENT_Z_PERCENT", "0.004"))
L_EPOCHS = int(os.getenv("FULFILLMENT_L_EPOCHS", "100"))
FULFILLMENT_MAX_CONCURRENT_SOURCES = int(os.getenv("FULFILLMENT_MAX_CONCURRENT_SOURCES", "2"))
FULFILLMENT_OPENROUTER_API_KEY = os.getenv("FULFILLMENT_OPENROUTER_API_KEY", "")
FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS = int(os.getenv("FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS", "30"))
FULFILLMENT_MIN_VALIDATORS = int(os.getenv("FULFILLMENT_MIN_VALIDATORS", "1"))
# How long the gateway waits (after reveal_window_end) for validators to
# score a request before recycling it with reason=no_validators_timeout.
# Must exceed the validator's worst-case end-to-end scoring time, not
# just its polling cadence.  Scoring a single request with 40+ leads
# takes 90-120 minutes under real load (Stage 4 LinkedIn scrapes, Tier
# 3 LLM calls per signal, TrueList batch verification, per-lead rep
# score lookup), plus queue time when ff-workers are busy with other
# requests.  Historical tightenings have all been too aggressive:
#   5 min  → 2026-04-21: requests died before validator even started
#  15 min  → 2026-04-22: 669fd2b7 / 7c666b4e died mid-scoring
#  90 min  → 2026-04-23: e3586265 died 18 s before scores landed — 48
#           scores (21 passing, would have easily fulfilled 10 winners)
#           posted in a 2-second batch 14 minutes after the recycle.
# Bumping to 180 min to clear the observed 1h 44m scoring path with a
# comfortable margin for ff-worker queueing under high request load.
# Still well under the 72-min-epoch × 100-epoch reward runway, so there
# is no downstream incentive impact from the longer wait.
FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES = int(os.getenv("FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES", "180"))
FULFILLMENT_BANS_ENABLED = os.getenv("FULFILLMENT_BANS_ENABLED", "false").lower() == "true"

FULFILLMENT_MAX_PARALLEL_REQUESTS = int(os.getenv("FULFILLMENT_MAX_PARALLEL_REQUESTS", "5"))
# Visibility cutoff: /fulfillment/requests/active hides any open or
# continued_open request whose window_end is less than this many minutes
# from now.  The intent is to avoid handing miners work they can't
# realistically commit + reveal in time.  Lowered from 15 → 5 minutes on
# 2026-04-27 because synchronized cycling (all 5 active requests promoting
# together with identical 72-min windows, then closing together) was
# producing a ~15-min "/active returns 0" dead-window every cycle —
# miner-reported and dashboard-confirmed.  Five minutes still gives a
# committed miner enough time to hash-and-commit while shrinking the
# dead-window proportionally.  Override via env var if needed.
FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES = int(os.getenv("FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES", "5"))

# Per-miner submission cap is (request.num_leads * this multiplier), ceil'd.
# Default 1.5 so a miner can commit ~50% more leads than the request requires
# to absorb the real-time validation flakiness (TrueList + LinkedIn scrapes
# currently pass ~70-80% of legitimate leads).  Only the top num_leads by
# score actually win rewards — the surplus just protects the miner from
# having their whole batch discarded because a couple of leads lost the
# coin flip on a transient failure.
#
# Increase on days with low miner participation, decrease once pass-rate
# improves.  Must be >= 1.0 (can't commit fewer than num_leads or the
# quota gate can never be met from a single miner).
FULFILLMENT_MINER_SUBMISSION_MULTIPLIER = float(os.getenv("FULFILLMENT_MINER_SUBMISSION_MULTIPLIER", "1.5"))
if FULFILLMENT_MINER_SUBMISSION_MULTIPLIER < 1.0:
    logging.warning(
        f"FULFILLMENT_MINER_SUBMISSION_MULTIPLIER={FULFILLMENT_MINER_SUBMISSION_MULTIPLIER} < 1.0; "
        "clamping to 1.0 so miners can always commit at least num_leads."
    )
    FULFILLMENT_MINER_SUBMISSION_MULTIPLIER = 1.0

FULFILLMENT_MIN_INTENT_SCORE = float(os.getenv("FULFILLMENT_MIN_INTENT_SCORE", "5.0"))
FULFILLMENT_INTENT_QUALITY_FLOOR = float(os.getenv("FULFILLMENT_INTENT_QUALITY_FLOOR", "5.0"))
FULFILLMENT_INTENT_BREADTH_WEIGHT = float(os.getenv("FULFILLMENT_INTENT_BREADTH_WEIGHT", "0.10"))

if os.getenv("ENABLE_FULFILLMENT", "false").lower() == "true" and not FULFILLMENT_OPENROUTER_API_KEY:
    logging.warning(
        "ENABLE_FULFILLMENT=true but FULFILLMENT_OPENROUTER_API_KEY is not set. "
        "Fulfillment scoring will fail when the first request is processed. "
        "Set the env var or disable fulfillment with ENABLE_FULFILLMENT=false."
    )


def epochs_to_seconds(num_epochs: int, tempo: int = 360) -> int:
    """Convert an epoch count to wall-clock seconds using the subnet tempo."""
    return num_epochs * tempo * BLOCK_TIME_SECONDS


def get_fulfillment_api_key() -> str:
    """Lazy validation — only raises when fulfillment scoring is actually invoked."""
    if not FULFILLMENT_OPENROUTER_API_KEY:
        raise ValueError(
            "FULFILLMENT_OPENROUTER_API_KEY must be set when fulfillment scoring is active. "
            "Set the env var or disable fulfillment with ENABLE_FULFILLMENT=false."
        )
    return FULFILLMENT_OPENROUTER_API_KEY
