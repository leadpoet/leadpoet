"""Compile the operator's prospective payout schedule from verified pilot evidence."""

from __future__ import annotations

from dataclasses import asdict

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
    rows, failures = review.terminal_pilot_attempts(rows)
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
