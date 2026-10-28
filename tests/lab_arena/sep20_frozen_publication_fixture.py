"""Frozen scoring and publication producers for Sep20 and Oct10 release SQL."""

from lab_arena import contracts, scoring, verify


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
        and run["terminal_cause"] == "provider_error"
    ]
    if zeros:
        recorded = service.store.record_run_scores(round_id, stage, zeros)
        assert recorded["status"] == "ok" and recorded["recorded"] == len(zeros)
        for item in zeros:
            saved = service.store.get_run(item["run_id"])
            assert saved["status"] == "failed" and saved["per_icp_score"] == 0
            assert saved["terminal_cause"] == "provider_error"


def frozen_publication_document(service, round_id):
    """Retain the old numeric-zero authority while testing today's safe hold.

    These sealed rounds deliberately retain failed baseline executions. Current
    service publication must refuse them. Their release migrations still need
    the original document assembled from those unchanged, persisted facts.
    """
    row = service.store.get_round(round_id)
    assert row["evaluation_date"] in ("2026-09-20", "2026-10-10")
    actual = service.publish(round_id)
    assert actual == {
        "status": "retry", "round_status": "scored", "reason": "baseline_incomplete",
    }, actual
    row = service.store.get_round(round_id)
    assert row["status"] == "scored" and row["publication_doc"] is None
    configuration = row["configuration_doc"]
    assert row["evaluation_date"] in ("2026-09-20", "2026-10-10")
    if row["evaluation_date"] == "2026-09-20":
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
    # Post423 SQL also preserves null-final participant entries.
    final_ranking = verify.final_ranking(
        complete_entries if row["evaluation_date"] == "2026-09-20" else entries)
    for entry in final_ranking:
        if entry["final_score"] is None:
            entry.update(cost_summary=None, eligible=False,
                         eligibility_reason="execution_incomplete")
        else:
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


def commit_oct10_frozen_scoring_plan(service, round_id, stage):
    """Produce the sealed Oct10 provider-zero plan through its real SQL API.

    First prove today's producer leaves saved provider failures incomplete.
    The archived release requires its original zero classification; execution
    receipts, failed status, ledger rows and SQL guards stay unchanged.
    """
    row = service.store.get_round(round_id)
    assert row["evaluation_date"] == "2026-10-10"
    assert row["status"] == "stage%d_closed" % stage
    service.benchmark_icps(round_id)
    admitted = {p["submission_id"] for p in row["participants"]}
    runs = [r for r in service.store.list_runs(round_id, stage=stage, kind="execute")
            if r["submission_id"] in admitted]
    current = scoring.build_scoring_plan(
        round_id=round_id, stage=stage, runs=runs,
        execution_sequence_policy=row["configuration_doc"].get("execution_sequence_policy"),
        configuration=row["configuration_doc"],
    )
    assert not any(item["cause"] == "provider_error" for item in current["zero_rows"])
    frozen_zeros = [item for item in current.get("incomplete_rows", [])
                    if item["cause"] == "provider_error"]
    for item in frozen_zeros:
        sources = [r for r in runs if r["submission_id"] == item["submission_id"]
                   and r["icp_position"] == item["icp_position"]]
        assert sources and not any(r["status"] == "accepted" for r in sources)
        latest = max(sources, key=lambda r: r["attempt"])
        assert latest["status"] == "failed" and latest["terminal_cause"] == "provider_error"
        assert latest["output_ref"] is None
    frozen = {**current, "zero_rows": current["zero_rows"] + frozen_zeros,
              "incomplete_rows": [item for item in current.get("incomplete_rows", [])
                                  if item["cause"] != "provider_error"]}
    result = service.store.transition_round(round_id, row["status"], row["status"],
                                            {"stage%d_scoring_plan_doc" % stage: frozen})
    return {"status": result["status"], "work_items": len(frozen["work_items"])}
