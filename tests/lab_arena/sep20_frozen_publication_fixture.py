"""Publication inputs for the sealed Sep20 release SQL, not a service override."""

from lab_arena import contracts, verify


def record_frozen_failed_execution_zeros(service, round_id, stage):
    """Exercise the pinned failed-run zero writer, retaining every failure.

    Current scoring leaves infrastructure failures unscored. The sealed Sep20
    release predates that policy and records their old numeric zero through
    its original SQL API. This does not create accepted executions or judges.
    """
    row = service.store.get_round(round_id)
    assert row["evaluation_date"] == "2026-09-20"
    assert row["status"] == "stage%d_judged" % stage
    runs = service.store.list_runs(round_id, stage=stage, kind="execute")
    accepted = {run["assignment_id"] for run in runs if run["status"] == "accepted"}
    latest = {}
    for run in runs:
        key = run["assignment_id"]
        if key not in latest or run["attempt"] > latest[key]["attempt"]:
            latest[key] = run
    zeros = [
        {"run_id": run["run_id"], "per_icp_score": 0,
         "qualification_doc": {"companies": []}}
        for run in latest.values()
        if run["status"] == "failed" and run["per_icp_score"] is None
        and run["assignment_id"] not in accepted
        and run["terminal_cause"] in (contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed"})
    ]
    if zeros:
        recorded = service.store.record_run_scores(round_id, stage, zeros)
        assert recorded["status"] == "ok" and recorded["recorded"] == len(zeros)
        for item in zeros:
            saved = service.store.get_run(item["run_id"])
            assert saved["status"] == "failed" and saved["per_icp_score"] == 0
            assert saved["terminal_cause"] in (
                contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed"}
            )


def frozen_publication_document(service, round_id):
    """Retain the old numeric-zero authority while testing today's safe hold.

    These sealed rounds deliberately retain failed baseline executions. Current
    service publication must refuse them. Their release migrations still need
    the original document assembled from those unchanged, persisted facts.
    """
    assert service.publish(round_id) == {
        "status": "retry", "round_status": "scored", "reason": "baseline_incomplete",
    }
    row = service.store.get_round(round_id)
    assert row["status"] == "scored" and row["publication_doc"] is None
    configuration = row["configuration_doc"]
    assert row["evaluation_date"] == "2026-09-20"
    assert configuration.get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
    positions = range(contracts.benchmark_icp_count(configuration))
    runs = service.store.list_runs(round_id, kind="execute")
    entries = service._score_entries_from_runs(row, positions, "final_score")
    eligibility = {
        entry["submission_id"]: service._submission_cost_eligibility(
            row, entry["submission_id"], runs, positions=positions,
        )
        for entry in entries
    }
    baseline = next(entry for entry in entries if entry["is_king"])
    # The pinned 251/292 SQL ranks only the participants with all twenty
    # persisted scores. Today's publication also lists ungraded participants
    # with a null final score; that later shape is not this release contract.
    complete_entries = [entry for entry in entries if entry["final_score"] is not None]
    assert baseline in complete_entries
    final_ranking = verify.final_ranking(complete_entries)
    for entry in final_ranking:
        entry.update(eligibility[entry["submission_id"]])
    document = {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
        "round_id": round_id,
        "participants": [
            {"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"],
             "is_baseline": bool(p.get("is_king"))}
            for p in row["participants"]
        ],
        "stage1_ranking": verify.stage1_ranking(service._score_entries_from_runs(
            row, contracts.stage_positions(1, configuration), "stage1_score",
        )),
        "finalists": list(row["finalists"] or []),
        "final_ranking": final_ranking,
        "king_decision": verify.king_decision(
            [entry for entry in entries if not entry["is_king"]
             and eligibility[entry["submission_id"]]["eligible"]],
            baseline, configuration=configuration,
        ),
        "published_at": service.now().isoformat().replace("+00:00", "Z"),
    }
    contracts.check_strict_document(document, contracts.PUBLICATION_LIMITS)
    return document


def publish_frozen_document(service, round_id, document):
    """Use the pinned SQL API with all original trigger and seal checks active."""
    return service.store.transition_round(round_id, "scored", "published", {
        "publication_doc": document, "published_at": document["published_at"],
    })
