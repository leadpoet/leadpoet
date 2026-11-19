"""One publication attempt gives the driver thread back when evidence reads overrun."""
import time

import pytest

from lab_arena.service import PublicationBudgetExceeded, _PublicationReads
from tests.lab_arena.publication_evidence_reads_test import case
from tests.lab_arena.scoring_validator_attribution_test import SUBMISSION_ID


def reads(service, row, runs, deadline):
    return _PublicationReads(service, row, runs, deadline=deadline)


def test_unbounded_by_default():
    service, row, runs, _, _, _, _ = case()
    assert reads(service, row, runs, None).qualified_counts(SUBMISSION_ID, (0, 1)) == {0: 1, 1: 1}


def test_an_overrun_attempt_stops_fetching():
    service, row, runs, _, _, _, _ = case()
    with pytest.raises(PublicationBudgetExceeded):
        reads(service, row, runs, time.monotonic() - 1).qualified_counts(SUBMISSION_ID, (0, 1))


def test_work_already_gathered_is_still_served_after_the_budget_runs_out():
    service, row, runs, _, _, _, counts = case()
    publication_reads = reads(service, row, runs, time.monotonic() + 3600)
    assert publication_reads.qualified_counts(SUBMISSION_ID, (0, 1)) == {0: 1, 1: 1}
    publication_reads.deadline = time.monotonic() - 1
    counts.clear()
    # A cached position never reaches the store, so it never trips the budget.
    assert publication_reads.qualified_counts(SUBMISSION_ID, (0, 1)) == {0: 1, 1: 1}
    assert counts['company_rows'] == 0


def test_publish_reports_an_overrun_as_a_retry_rather_than_an_error():
    service, row, runs, _, _, _, _ = case()
    round_id = str(row['round_id'])

    def overrun(_round_id):
        raise PublicationBudgetExceeded(round_id)

    service._publish_locked = overrun
    assert service.publish(round_id) == {
        'status': 'retry',
        'round_status': 'scored',
        'reason': 'publication_evidence_budget_exhausted',
    }
