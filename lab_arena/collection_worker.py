"""Run a locked generator and its private judges; publish immutable evidence before completing the lease."""
from __future__ import annotations

from dataclasses import asdict

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import CHAMPION_REV, POET_REV, play_workflow, store_artifact


def process_job(*, job, assignment, policy, prompt, intake, broker, objects, sources, epoch,
                research=None, persist=store_artifact):
    """The journal stores judge bytes separately. Only scores and tags reach the public statement."""
    run_id, batch = job["job_id"], job["batch_id"]
    if (job["harness_commit"] != CHAMPION_REV or job["model"] not in MODELS or
            job["mode"] != ("pilot" if policy.pilot else "live") or
            batch != policy.batch_id or assignment["run_id"] != run_id):
        raise ValueError("frozen collection assignment mismatch")
    judge.validate_prompt(policy,prompt)
    if job["scenario_id"].startswith("research-"):
        if research is None:
            raise ValueError("missing trusted research runner")
        row, packet = research(job,broker)
    else:
        row, packet = play_workflow(*sources, job, broker)
    if row["run"] != run_id or row["teacher"] != job["model"] or packet["scenario_id"] != job["scenario_id"]:
        raise ValueError("collection artifact identity mismatch")
    row_ref = persist(objects,batch,run_id,"generator",row)
    packet_ref = persist(objects,batch,run_id,"judge-input",packet)
    assessment = judge.judge_run(policy,assignment,packet,broker.judge,prompt=prompt)
    judge_ref = persist(objects,batch,run_id,"judge-output",assessment)
    manifest = {"job_id":run_id,"batch_id":batch,"model":job["model"],"scenario_id":job["scenario_id"],
                "harness_commit":CHAMPION_REV,"builder_commit":POET_REV,
                "judge_policy":asdict(policy),"generator":row_ref,"packet":packet_ref,"assessment":judge_ref}
    receipt = persist(objects,batch,run_id,"manifest",manifest)
    public = judge.public_result(assessment)["judges"]
    summary = public[0] if public else {}
    status = {"accepted":"accepted", "rejected":"rejected", "code_rejected":"rejected",
              "salvaged":"rejected", "needs_review":"failed", "cannot_judge":"failed"}[assessment["status"]]
    intake.complete(run_id,job["lease_token"],status,summary,receipt["sha256"],epoch())
    return manifest
