"""Recover an observed student error from a signed, completed standing experiment."""

from __future__ import annotations

import json
from uuid import UUID

from lab_arena import collection_judge as judge
from lab_arena.collection_journal import WorkflowJournal
from lab_arena.collection_pool import POOL_SCOPE, POOL_TERMS_HASH
from lab_arena.collection_workflow import read_artifact

SCHEMA = "collection.funded-student.v1"


def funded_source(database, objects, batch_id, run_id, prompt):
    """Reconstruct the source from original provider replies, ledger funding and per-turn state."""
    from lab_arena.collection_experiment import load_experiment, pool_verified_result
    from lab_arena.collection_trajectories import verified_grants
    import re

    if (
        not re.fullmatch(r"pool-[0-9a-f]{64}", str(batch_id))
        or str(UUID(run_id)) != run_id
    ):
        raise ValueError("invalid standing source job")
    jobs = database.rpc("collection_batch_jobs_v1", {"p_batch": batch_id})
    matches = [j for j in jobs if j["job_id"] == run_id]
    if len(matches) != 1 or matches[0]["status"] != "accepted":
        raise ValueError("expected an originally verified and settled source run")
    job = dict(matches[0], batch_id=batch_id)
    grants = verified_grants(
        database,
        batch_id,
        [job],
        mode="live",
        scope=POOL_SCOPE,
        digest=POOL_TERMS_HASH,
        subset=True,
    )
    manifest = read_artifact(
        objects,
        batch_id,
        run_id,
        "manifest",
        {"sha256": job["artifact_sha256"]},
        1 << 20,
    )
    spec = load_experiment(
        read_artifact(
            objects, batch_id, run_id, "settings", manifest["settings"], 2 << 20
        ),
        prompt,
    )
    _, evidence = pool_verified_result(objects, job, spec)
    if (
        not evidence["complete_verified"]
        or evidence["task_pass"]
        or grants[run_id]["miner_cost_microusd"] != evidence["cost_microusd"]
        or grants[run_id]["generator_calls"] != evidence["calls"]
        or manifest["experiment"]["role"] != "expert"
    ):
        raise ValueError(
            "a correction requires a complete original funded task failure"
        )
    row = read_artifact(
        objects, batch_id, run_id, "generator", manifest["generator"], 128 << 20
    )
    family = job["scenario_id"].split("-s")[0]
    if family == "P8":
        from lab_arena.collection_sales_evidence import _history

        packet = read_artifact(
            objects, batch_id, run_id, "judge-input", manifest["packet"], 128 << 20
        )
        _, failure = _history(objects, job, manifest, row, packet)
        if failure is None:
            raise ValueError("no recoverable observed evidence mistake")
        checkpoint, rejected, turn, reason = (
            failure[k] for k in ("checkpoint", "rejected", "turn", "reason")
        )
        fatal = reason == "incomplete_or_incorrect_final_decision"
    else:
        checkpoint, rejected, turn, reason = _locked_mistake(
            objects, job, manifest, row, family
        )
        fatal = True
    origin = {k: job[k] for k in ("batch_id", "job_id", "artifact_sha256")}
    return {
        "schema": SCHEMA,
        "origin": origin,
        "scenario_id": job["scenario_id"],
        "student_model": job["model"],
        "source_sha256": judge.digest(origin),
        "branch_turn": turn,
        "checkpoint": checkpoint,
        "checkpoint_sha256": judge.digest(checkpoint),
        "rejected": rejected,
        "rejected_fatal": fatal,
        "failure": {"reason": reason, "turn": turn},
        "split": row["split"],
    }


def _locked_mistake(objects, job, manifest, row, family):
    if family not in {"P7", "P9", "P10"}:
        raise ValueError("unknown irreversible decision schema")
    evidence = row["evidence"]
    grade = evidence["grade"]
    state_key = "posting" if family == "P7" else "plan"
    # With a correct posting but a wrong final answer, the irreversible mistake
    # is the final reply. Never mislabel the student's correct write as rejected.
    wrong_commit = (
        not all(grade["credits"].get(k) == 1 for k in ("sealed", "charter"))
        if family == "P7"
        else grade["score"] != 1 or bool(grade.get("hard_fail"))
    )
    visible = row["messages"]
    assistant = [i for i, m in enumerate(visible) if m["role"] == "assistant"]
    before = None
    for data, _ in WorkflowJournal(objects, job["batch_id"], job["job_id"]).read(
        manifest["events"]
    ):
        event = json.loads(data)
        kind = event["kind"]
        if kind not in ("before", "finish"):
            continue
        if before is not None:
            wrong_write = bool(
                wrong_commit
                and before["state"].get(state_key) is None
                and event["state"].get(state_key) is not None
            )
            if wrong_write or kind == "finish":
                turn = before["turn"]
                cut = assistant[turn]
                cp = {
                    "messages": visible[:cut],
                    "tools": row["tools"],
                    **{k: before[k] for k in ("state", "files", "log")},
                }
                return (
                    cp,
                    visible[cut],
                    turn,
                    (
                        "incorrect_irreversible_plan"
                        if wrong_write
                        else "incorrect_terminal_decision"
                    ),
                )
        before = event
    raise ValueError("no observed failed student step")
