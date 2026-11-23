"""Compile the operator's prospective payout schedule from verified pilot evidence."""

from __future__ import annotations

from dataclasses import asdict

from lab_arena import collection_judge as judge
from lab_arena import collection_judge_review as review
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import CHAMPION_REV, POET_REV


def compile_pilot(transport, bundle, receipt, pinned_public_key_pem, objects):
    """The human receipt is trusted only against a deployment-pinned reviewer key.

    The returned document is suitable for the operator-only cutover INSERT. No
    reviewer identity, prompt, marks, credentials or private response is exposed.
    """
    policy = judge.BatchPolicy(**bundle["policy"])
    if not policy.pilot:
        raise ValueError("expected pilot review")
    rows = transport.rpc("collection_pilot_costs_v1", {"p_batch": policy.batch_id})
    if not isinstance(rows, list) or not rows or len(rows) > 24576:
        raise ValueError("invalid pilot job evidence")
    verified = review.registered_bundle(
        policy,
        bundle["assignments"],
        planted=bundle["planted"],
        drift=bundle["drift"],
        jobs=rows,
        objects=objects,
    )
    if judge.digest(verified) != judge.digest(bundle):
        raise ValueError("submitted pilot review differs from stored artifacts")
    report = review.verify_review(verified, receipt, pinned_public_key_pem)
    if not report["quality_ready"] or set(
        a["generator"] for a in bundle["assignments"]
    ) != set(MODELS):
        raise ValueError("pilot release checks failed")
    for row in rows:
        if row["harness_commit"] != CHAMPION_REV:
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
        "harness_commit": CHAMPION_REV,
        "builder_commit": POET_REV,
        "control_ids": controls,
        **calibration,
    }
