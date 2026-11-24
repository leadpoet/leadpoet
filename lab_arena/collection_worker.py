"""Run a locked generator and its private judges; publish immutable evidence before completing the lease."""

from __future__ import annotations

from dataclasses import asdict

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import (
    comparison_revisions,
    play_workflow,
    store_artifact,
)


def process_job(
    *,
    job,
    assignment,
    policy,
    prompt,
    intake,
    broker,
    objects,
    sources,
    epoch,
    research=None,
    control_packet=None,
    claim_pack=None,
    persist=store_artifact,
    prepare_launch=False,
    split=None,
    claim_protocol="collection.fresh-claims.v1",
    capture_version=None,
):
    """The journal stores judge bytes separately. Only scores and tags reach the public statement."""
    from lab_arena.collection_journal import WorkflowJournal

    run_id, batch = job["job_id"], job["batch_id"]
    capture = (
        {"capture_version": capture_version} if capture_version is not None else {}
    )
    agent, builder = comparison_revisions(capture)
    journal = None
    if job.get("deepline_key_id") is None and job["scenario_id"].startswith("research-"):
        raise ValueError("live research is outside synthetic funding scope")
    if (
        job["harness_commit"] != agent
        or job["model"] not in MODELS
        or job["mode"] != ("pilot" if policy.pilot else "live")
        or batch != policy.batch_id
        or assignment["run_id"] != run_id
    ):
        raise ValueError("frozen collection assignment mismatch")
    judge.validate_prompt(policy, prompt)
    control_ref = None
    row_ref = None
    if control_packet is not None:
        packet = control_packet["packet"]
        if (
            judge.digest(control_packet["case"]) != control_packet["case_sha256"]
            or packet["run_id"] != run_id
            or packet["scenario_id"] != job["scenario_id"]
        ):
            raise ValueError("invalid blind control assignment")
        assignment = dict(assignment, overlap=True)
        control_ref = persist(
            objects, batch, run_id, "control-case", control_packet["case"]
        )
    else:
        if job["scenario_id"].startswith("research-"):
            if research is None:
                raise ValueError("missing trusted research runner")
            row, packet = research(job, broker)
        elif capture:
            if (
                getattr(sources[0], "source_revision", None),
                getattr(sources[1], "source_revision", None),
            ) != (agent, builder):
                raise ValueError("comparison worker loaded the wrong pinned sources")
            journal = WorkflowJournal(objects, batch, run_id)
            try:
                row, packet = play_workflow(
                    *sources, job, broker, record=journal.append
                )
            except Exception:
                journal.finish("interrupted")
                raise
            journal.finish("completed")
        else:
            row, packet = play_workflow(*sources, job, broker)
        if (
            row["run"] != run_id
            or row["teacher"] != job["model"]
            or packet["scenario_id"] != job["scenario_id"]
            or packet["run_id"] != run_id
        ):
            raise ValueError("collection artifact identity mismatch")
        if any(
            x is not None and (x["teacher"] != job["model"] or x["run"] != run_id)
            for x in row.get("research_rows", [])
        ):
            raise ValueError("incorrect research segment identity")
        row_ref = persist(objects, batch, run_id, "generator", row)
    if split is not None:
        if control_ref and packet.get("split") != split:
            raise ValueError("control source does not belong to the approved split")
        packet = {**packet, "split": split}
    if not control_ref and claim_pack is not None:
        from lab_arena.collection_claims import apply_pack

        packet = apply_pack(packet, batch_id=batch, pack=claim_pack)
    if not control_ref and not job["scenario_id"].startswith("research-"):
        from lab_arena.collection_claims import read_fresh
        packet = read_fresh(packet, policy=policy, generator=job["model"], broker=broker,
                            record_vote=lambda vote: persist(objects, batch, run_id, "claim-vote", vote), protocol=claim_protocol)
    packet_ref = persist(objects, batch, run_id, "judge-input", packet)
    assessment = judge.judge_run(
        policy, assignment, packet, broker.judge, prompt=prompt
    )
    judge_ref = persist(objects, batch, run_id, "judge-output", assessment)
    manifest = {
        "job_id": run_id,
        "batch_id": batch,
        "model": job["model"],
        "scenario_id": job["scenario_id"],
        "harness_commit": agent,
        "builder_commit": builder,
        **capture,
        **({"events": journal.hashes} if journal is not None else {}),
        "judge_policy": asdict(policy),
        "generator": row_ref,
        "packet": packet_ref,
        "assessment": judge_ref,
    }
    from lab_arena.collection_ranking import rank_pack_if_ready

    if control_ref:
        manifest["control"] = control_ref
    else:
        ranking = rank_pack_if_ready(
            job, packet, assessment, policy, prompt, broker, objects, broker.database
        )
        if ranking is not None:
            manifest["ranking"] = ranking
    receipt = persist(objects, batch, run_id, "manifest", manifest)
    public = judge.public_result(assessment)["judges"]
    summary = public[0] if public else {}
    summary = dict(summary, training_status=assessment["status"])
    checks = packet["checks"]
    complete_execution = all(
        checks.get(k) is True
        for k in ("answered", "shape", "privacy", "training_split")
    )
    status = "accepted" if complete_execution and not control_ref else "rejected"
    if status == "accepted" and prepare_launch:
        intake.complete(run_id, job["lease_token"], status, summary, receipt["sha256"], epoch(), prepare_launch=True)
    else:
        intake.complete(run_id, job["lease_token"], status, summary, receipt["sha256"], epoch())
    return manifest
