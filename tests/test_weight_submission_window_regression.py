"""Regression guards for the 2026-07-30/31 weight-submission outage.

After the 2026-07-30 18:25 UTC gateway redeploy, the attested weight-input
builds in the gateway coordinator enclave slowed from ~1s to 145-700s per
epoch and the measured end-to-end primary pipeline reached ~959s. With
submission starting at epoch block 300 of 360 (a 720s window), the extrinsic
staleness gate evaluated after the official epoch closed on every attempt:
epochs 24256-24270 published signed bundles durably in-epoch, yet no
set_weights reached the chain for 18+ hours and every watched validator went
stale together (auditors mirror the primary's published bundle).

These tests pin the submission window and the inputs-client retry budget to
the timings measured during the incident. If either shrinks below what
production demonstrated is required — or the pipeline constants regress —
this file fails before the chain does.
"""

from __future__ import annotations

import inspect

from leadpoet_canonical.constants import (
    ALLOCATION_PREPARATION_BLOCK,
    EPOCH_LENGTH,
    MAX_BLOCK_DRIFT,
    WEIGHT_SUBMISSION_BLOCK,
)
from validator_tee.host.gateway_weight_inputs_v2 import (
    _MAX_RETRY_DELAY_SECONDS,
    fetch_gateway_weight_inputs_v2,
)

# Finney block cadence.
SECONDS_PER_BLOCK = 12

# Stage timings measured for settlement epoch 24270 on 2026-07-31 (dockerd
# receive timestamps on the primary and Supabase row timestamps), the first
# fully-instrumented epoch after the frame-limit fix restored publication:
#   gateway /weights/inputs/v2 request -> response   11:52:21 -> 11:57:56
#   enclave signing + publication preparation        11:57:56 -> 11:59:37
#   durable Supabase row -> gateway submit response  11:59:37 -> 12:07:02
#   extrinsic + chain finalization (healthy 2026-07-30 baseline:
#   bundle 17:52:06 -> finalization 17:53:24)
MEASURED_INPUTS_SECONDS = 335
MEASURED_SIGN_AND_PREPARE_SECONDS = 101
MEASURED_PUBLICATION_ACK_SECONDS = 445
MEASURED_EXTRINSIC_FINALIZATION_SECONDS = 78

MEASURED_PIPELINE_SECONDS = (
    MEASURED_INPUTS_SECONDS
    + MEASURED_SIGN_AND_PREPARE_SECONDS
    + MEASURED_PUBLICATION_ACK_SECONDS
    + MEASURED_EXTRINSIC_FINALIZATION_SECONDS
)

# Slowest attested input build observed during the incident (epochs
# 24265-24266, receipt-span measurement from
# research_lab_attested_execution_receipts_v2).
SLOWEST_OBSERVED_BUILD_SECONDS = 700

# Headroom multiplier: the window must not merely equal the observed worst
# case, because every stage above degrades further under coordinator-enclave
# contention.
SAFETY_MARGIN = 1.2


def test_submission_window_covers_measured_degraded_pipeline():
    window_seconds = (EPOCH_LENGTH - WEIGHT_SUBMISSION_BLOCK) * SECONDS_PER_BLOCK
    assert window_seconds >= MEASURED_PIPELINE_SECONDS * SAFETY_MARGIN, (
        "The weight submission window no longer covers the end-to-end "
        "pipeline duration measured during the 2026-07-31 outage; every "
        "submission would reach the extrinsic staleness gate after the "
        "official epoch closes."
    )


def test_allocation_preparation_retains_head_start():
    # Allocation preparation must still complete before the submission
    # window opens; the preparation itself runs through the same degraded
    # coordinator enclave.
    assert ALLOCATION_PREPARATION_BLOCK + 30 <= WEIGHT_SUBMISSION_BLOCK


def test_submission_window_stays_inside_official_epoch():
    assert 0 < WEIGHT_SUBMISSION_BLOCK < EPOCH_LENGTH
    assert MAX_BLOCK_DRIFT < WEIGHT_SUBMISSION_BLOCK


def test_inputs_client_attempt_budget_covers_slowest_observed_build():
    signature = inspect.signature(fetch_gateway_weight_inputs_v2)
    timeout_seconds = signature.parameters["timeout_seconds"].default
    max_attempts = signature.parameters["max_attempts"].default
    retry_delay_seconds = signature.parameters["retry_delay_seconds"].default

    delays = sum(
        min(retry_delay_seconds * (2 ** (attempt - 1)), _MAX_RETRY_DELAY_SECONDS)
        for attempt in range(1, max_attempts)
    )
    attempt_coverage_seconds = max_attempts * timeout_seconds + delays

    assert (
        attempt_coverage_seconds
        >= SLOWEST_OBSERVED_BUILD_SECONDS * SAFETY_MARGIN
    ), (
        "The gateway inputs client exhausts its attempts before the slowest "
        "observed singleflight build completes; the caller then restarts the "
        "flow and forces a rebuild, which is how epochs 24265-24266 tripled "
        "their build work."
    )
    # Even the slowest observed build must leave room in the submission
    # window for signing, durable publication, and the chain extrinsic. The
    # client re-attaches to the singleflight, so its actual spend tracks the
    # build duration, not the full attempt budget.
    window_seconds = (EPOCH_LENGTH - WEIGHT_SUBMISSION_BLOCK) * SECONDS_PER_BLOCK
    non_input_pipeline_seconds = (
        MEASURED_SIGN_AND_PREPARE_SECONDS
        + MEASURED_PUBLICATION_ACK_SECONDS
        + MEASURED_EXTRINSIC_FINALIZATION_SECONDS
    )
    assert (
        SLOWEST_OBSERVED_BUILD_SECONDS + non_input_pipeline_seconds
        <= window_seconds
    )
