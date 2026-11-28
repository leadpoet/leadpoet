"""Compile the operator's prospective payout schedule from verified pilot evidence."""

from __future__ import annotations

from dataclasses import asdict
import re

from lab_arena import collection_judge as judge
from lab_arena import collection_judge_review as review
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import comparison_revisions


def compile_pilot(
    transport,
    bundle,
    receipt,
    pinned_public_key_pem,
    objects,
    *,
    approval=None,
    approver_key=None,
    witnessed_batch=None,
):
    """The human receipt is trusted only against a deployment-pinned reviewer key.

    The returned document is suitable for the operator-only cutover INSERT. No
    reviewer identity, prompt, marks, credentials or private response is exposed.
    """
    policy = judge.BatchPolicy(**bundle["policy"])
    agent, builder = comparison_revisions(bundle)
    if not policy.pilot:
        raise ValueError("expected pilot review")
    rows = transport.rpc("collection_pilot_costs_v1", {"p_batch": policy.batch_id})
    if not isinstance(rows, list) or not rows or len(rows) > 2 * 24576:
        raise ValueError("invalid pilot job evidence")
    attempts = rows
    rows, failures = review.terminal_pilot_attempts(attempts)
    verified = review.registered_bundle(
        policy,
        bundle["assignments"],
        planted=bundle["planted"],
        drift=bundle["drift"],
        jobs=rows,
        failed_attempts=failures,
        objects=objects,
        expected_builder=builder,
        capture_version=bundle.get("capture_version"),
    )
    verified["split_manifest_sha256"] = bundle.get("split_manifest_sha256")
    if "claim_protocol" in bundle:
        verified["claim_protocol"]=bundle["claim_protocol"]
    if judge.digest(verified) != judge.digest(bundle):
        raise ValueError("submitted pilot review differs from stored artifacts")
    panel = {}
    if policy.qualification and approval is not None:
        approved = review.verify_qualification(
            verified, receipt, pinned_public_key_pem, approval, approver_key
        )
        if approved["allow_live"] is not True:
            raise ValueError("panel approval excludes compensated collection")
        panel["panel_qualification_sha256"] = judge.digest(approval)
    else:
        review.verify_review(verified, receipt, pinned_public_key_pem)
    if re.fullmatch(r"rq-[0-9a-f]{64}", policy.batch_id):
        from scripts.run_locked_collection import load_batch
        from lab_arena.collection_route_witness import verified_witness
        from lab_arena.collection_trajectories import verified_grants
        from lab_arena.collection_intake import TASK_SCOPE

        if not witnessed_batch or "panel_qualification_sha256" not in panel:
            raise ValueError("qualified witnessed pricing pilot required")
        raw, prompt = witnessed_batch
        if not raw.get("route_witness_sha256") or load_batch(raw, prompt).batch_id != policy.batch_id:
            raise ValueError("route witness differs from the reviewed pricing pilot")
        verified_witness(raw, objects)
        verified_grants(transport, policy.batch_id, attempts, mode="pilot", scope=TASK_SCOPE, digest=raw["terms_sha256"])
        panel["route_witness_sha256"] = raw["route_witness_sha256"]
    if set(a["generator"] for a in bundle["assignments"]) != set(MODELS):
        raise ValueError("pilot release checks failed")
    for row in rows:
        if row["harness_commit"] != agent:
            raise ValueError("pilot harness does not match release")
    controls = sorted(set(bundle["drift"]) | set(bundle["planted"]))
    calibration = transport.rpc(
        "collection_pilot_fees_v1", {"p_batch": policy.batch_id, "p_controls": controls}
    )
    return {
        "schema": "leadpoet.collection.approved_pilot.v1",
        "batch_id": policy.batch_id,
        "review_sha256": judge.digest(receipt),
        "review_bundle_sha256": judge.digest(verified),
        "judge_policy_sha256": judge.digest(asdict(policy)),
        "judge_prompt_sha256": policy.prompt_sha256,
        "harness_commit": agent,
        "builder_commit": builder,
        **(
            {"capture_version": bundle["capture_version"]}
            if "capture_version" in bundle
            else {}
        ),
        "control_ids": controls,
        "split_manifest_sha256": verified["split_manifest_sha256"],
        **(
            {"claim_protocol": bundle["claim_protocol"]}
            if "claim_protocol" in bundle
            else {}
        ),
        **panel,
        **calibration,
    }
