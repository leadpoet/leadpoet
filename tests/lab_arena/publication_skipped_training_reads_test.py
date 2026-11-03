"""Skipped capture audits must not reload the large frozen round."""

from copy import deepcopy

import pytest

from lab_arena import contracts, integrity
from tests.lab_arena.publication_stale_exit_test import publication


def configure(publication, *, margin=100.0):
    publication.row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        promotion_margin=margin,
    )


@pytest.mark.parametrize("challengers", [0, 13, 50])
def test_noncontenders_do_not_reload_the_round(publication, challengers):
    configure(publication)
    participants = [publication.row["participants"][0]] + [
        {"submission_id": f"challenger-{i}", "miner_hotkey": f"hotkey-{i}", "is_king": False}
        for i in range(challengers)
    ]
    publication.row["participants"] = participants
    entries = [dict(submission_id=p["submission_id"], hotkey=p["miner_hotkey"],
                    is_king=p["is_king"], stage1_score=50.0, final_score=50.0)
               for p in participants]
    publication.service._score_entries_from_runs = lambda *_a, **_k: deepcopy(entries)
    assert publication.service.publish("old")["status"] == "ok"
    assert publication.reads == 1
    assert publication.assessed == []
    assert len(publication.transitions) == 1
    saved = publication.row["publication_doc"]
    assert saved["king_decision"]["outcome"] == "no_king"
    assert all(row["final_score"] == 50.0 for row in saved["final_ranking"])
    assert all(row["training_data_eligibility"] == {
        "status": "not_assessed", "eligible": None, "reasons": []
    } for row in saved["final_ranking"])


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_skipped_audits_still_obey_final_compare_and_swap(publication, status):
    configure(publication)
    transition = publication.service._store.transition_round

    def competing_transition(*args):
        publication.row.update(status=status, publication_doc={"owner": "other"})
        return transition(*args)

    publication.service._store.transition_round = competing_transition
    assert publication.service.publish("old")["status"] == "stale"
    assert publication.assessed == []
    assert publication.row["publication_doc"] == {"owner": "other"}


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_qualifier_keeps_fresh_check_before_expensive_audit(publication, status):
    configure(publication, margin=8.0)
    cost = publication.service._submission_cost_eligibility

    def other_publisher(*args, **kwargs):
        publication.row["status"] = status
        return cost(*args, **kwargs)

    publication.service._submission_cost_eligibility = other_publisher
    assert publication.service.publish("old")["status"] == "stale"
    assert publication.assessed == []
    assert publication.transitions == []
    assert publication.reads == 2
