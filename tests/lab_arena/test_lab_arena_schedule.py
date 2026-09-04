"""A daily round fits inside its day, and the judge outlives its lease only by working."""

from __future__ import annotations

from lab_arena import contracts
from lab_arena import service as svc


def test_a_days_cycle_fits_inside_twenty_four_hours():
    """Submission window plus every stage window leaves room for the next round's cutoff."""

    stages = sum(svc.DEFAULT_STAGE_MINUTES.values())
    window = svc.RoundDefaults().min_submission_hours * 60
    assert stages + window <= 24 * 60, (stages, window)
    # The stage windows are the ones the schedule builder lays out end to end.
    assert set(svc.DEFAULT_STAGE_MINUTES) == {"benchmark", "stage_1", "stage_1_scoring", "stage_2", "final_scoring"}


def test_judge_wall_clock_exceeds_the_models_and_fits_inside_one_lease():
    """A judge run can be longer than a model run while active calls refresh its lease."""

    assert contracts.SCORING_WALL_CLOCK_SECONDS > contracts.ICP_WALL_CLOCK_SECONDS
    # Provider calls refresh a lease, so a judge that keeps calling stays leased; a judge silent for
    # its three provider timeouts still fits inside one lease.
    from lab_arena import operations

    longest_call = max(operation.timeout_seconds for operation in operations.OPERATIONS.values())
    assert 3 * longest_call <= contracts.LEASE_TTL_SECONDS
