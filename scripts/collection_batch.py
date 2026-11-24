"""Prepare blind review, compile pilot prices, or release training rows from registered evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from lab_arena import collection_judge as judge, collection_judge_review as review
from lab_arena.collection_calibration import compile_pilot
from lab_arena.collection_intake import MODELS, live_finney_arena
from lab_arena.collection_workflow import comparison_revisions
from lab_arena.service import S3ObjectStore
from lab_arena.store import PostgrestTransport
from scripts.collection_review import read_file, write_new
from scripts.run_locked_collection import load_batch, assignment


def source_bundle(raw, prompt, transport, objects):
    policy = load_batch(raw, prompt)
    agent, builder = comparison_revisions(raw)
    jobs = transport.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
    expected_count = len(raw["scenario_ids"]) * len(MODELS)
    if (
        not isinstance(jobs, list)
        or not expected_count <= len(jobs) <= 2 * expected_count
    ):
        raise ValueError("incomplete registered batch")
    if any(j["batch_id"] != policy.batch_id or j["mode"] != raw["mode"] for j in jobs):
        raise ValueError("job differs from frozen batch")
    failures = []
    if policy.pilot:
        jobs, failures = review.terminal_pilot_attempts(jobs)
    if len(jobs) != expected_count:
        raise ValueError("incomplete registered batch")
    slots = {}
    for job in jobs:
        index = job["slot"]
        if (
            type(index) is not int
            or not 0 <= index < expected_count
            or index in slots
            or job["mode"] != raw["mode"]
            or job["status"] not in {"accepted", "rejected", "failed"}
            or job["scenario_id"] != raw["scenario_ids"][index // len(MODELS)]
            or job["model"] != MODELS[index % len(MODELS)]
            or job["harness_commit"] != agent
            or (
                not job.get("artifact_sha256")
                and not (policy.pilot and job["status"] == "failed")
            )
        ):
            raise ValueError("job differs from frozen batch")
        slots[index] = job

    def keyed_by_run(name):
        mapping = raw[name]
        if not isinstance(mapping, dict) or any(
            not re.fullmatch(r"0|[1-9][0-9]*", k) or int(k) not in slots
            for k in mapping
        ):
            raise ValueError("invalid frozen control slot")
        return {slots[int(k)]["job_id"]: expected for k, expected in mapping.items()}

    bundle = review.registered_bundle(
        policy,
        [assignment(j) for j in jobs],
        jobs=jobs,
        failed_attempts=failures,
        objects=objects,
        expected_builder=builder,
        capture_version=raw.get("capture_version"),
        planted=keyed_by_run("control_slots"),
        drift=keyed_by_run("drift_slots"),
    )
    slot_for = {row["job_id"]: str(row["slot"]) for row in jobs}
    checked_controls = {
        slot_for[item["packet"]["run_id"]]: item["control_sha256"]
        for item in bundle["runs"]
        if "control_sha256" in item
    }
    expected_controls = {
        k: v
        for k, v in raw.get("control_artifacts", {}).items()
        if slots[int(k)]["status"] != "failed"
    }
    if checked_controls != expected_controls:
        raise ValueError("registered control differs from frozen private case")
    from lab_arena.collection_quality import verify_split
    if "split_manifest" not in raw:
        raise ValueError("review requires a frozen template, rule and tool split")
    for item in bundle["runs"]:
        verify_split(item["packet"].get("split"), raw["split_manifest"])
        if (item["packet"]["checks"].get("training_split") is not True
                or item["packet"]["checks"].get("privacy") is not True):
            raise ValueError("hidden or unchecked material in review batch")
    if raw.get("claim_protocol") == "collection.fresh-claims.v2":
        bundle["claim_protocol"] = raw["claim_protocol"]
    return {**bundle, "split_manifest_sha256": judge.digest(raw["split_manifest"])}


def read_artifact(objects, batch, run, kind, link, limit):
    digest = link.get("sha256")
    if type(digest) is not str or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("missing trusted artifact hash")
    ref = f"collection/{batch}/{run}/{kind}-{digest[7:]}.json"
    if link.get("key", ref) != ref:
        raise ValueError("cross-run artifact reference")
    data = objects.get_bounded(ref, limit)
    if hashlib.sha256(data).hexdigest() != digest[7:]:
        raise ValueError("stored evidence did not match review")
    return json.loads(data)


def write_exports(bundle, receipt, public_key, objects, output):
    """Legacy comparison enrollment covers funding and private review only."""
    raise ValueError(
        "comparison enrollment did not grant training release rights; "
        "use a separately consented correction experiment"
    )


def cutover_anchor(transport, *, launch_job=None):
    """Bind an operator's pricing document to the current finalized legacy epoch."""
    from lab_arena import chain, signing
    from lab_arena.store import ArenaStore
    from leadpoet_canonical.arena_weights import verify_accepted_weight_state_signature

    config = chain.ArenaChainConfig(
        os.environ["LAB_ARENA_CHAIN_ENDPOINT"], 71, "finney", 15
    )
    client = chain.ArenaChain(config, chain.connect_substrate(config))
    try:
        snapshot = chain.finalized_epoch_snapshot(client)
        epoch = snapshot.settlement_epoch_id(chain.load_arena_cutover())
        row = ArenaStore(transport).get_weight_state("finney", 71, epoch)
        if not row or row["state_doc"]["reward_basis"]["schema_version"].startswith(
            "leadpoet.collection."
        ):
            raise ValueError("current signed legacy state required")
        signer = signing.KmsSigner(
            os.environ["LAB_ARENA_SIGNING_KEY_ID"],
            region_name=os.environ.get("AWS_REGION"),
        )
        state = row["state_doc"]
        verify_accepted_weight_state_signature(
            state,
            public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
        if (
            state["epoch"] != epoch
            or state["genesis_hash"] != snapshot.network_genesis_hash.removeprefix("0x")
            or (state["network"], state["netuid"]) != ("finney", 71)
            or not state["valid_from_block"]
            <= snapshot.current_block
            <= state["valid_until_block"]
        ):
            raise ValueError("accepted state differs from finalized chain")
        timestamp = client.client.query(
            "Timestamp", "Now", block_hash=snapshot.block_hash
        )
        millis = getattr(timestamp, "value", timestamp)
        if type(millis) is not int:
            raise ValueError("finalized timestamp missing")
        miner = {}
        if launch_job is not None:
            hotkey = launch_job["miner_hotkey"]
            uid = client.client.query("SubtensorModule", "Uids", [71, hotkey], block_hash=snapshot.block_hash)
            owner = client.client.query("SubtensorModule", "Owner", [hotkey], block_hash=snapshot.block_hash)
            if (type(getattr(uid, "value", uid)) is not int or hotkey == state["burn_hotkey"]
                    or getattr(owner, "value", owner) != launch_job["owner_coldkey"]):
                raise ValueError("launch miner is no longer eligible")
            miner = {"miner_hotkey": hotkey}
        return {
            **miner,
            "epoch": epoch,
            "state_hash": state["state_hash"],
            "head_block": snapshot.current_block,
            "head_hash": snapshot.block_hash,
            "observed_at": millis // 1000,
        }
    finally:
        client.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("bundle", "prices", "export", "experiment-audit", "corrections", "training-feedback"))
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel-batch", type=Path)
    p.add_argument("--panel-prompt", type=Path)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--dataset-metadata", type=Path)
    p.add_argument("--evaluation", type=Path)
    args = p.parse_args(argv)
    if args.action in {"prices", "export", "corrections"} and args.receipt is None:
        p.error("--receipt is required")
    if not live_finney_arena():
        raise ValueError("expected Finney/71 operator configuration")
    db = PostgrestTransport(
        os.environ["LAB_ARENA_SUPABASE_URL"],
        service_key=os.environ["LAB_ARENA_SERVICE_KEY"],
    )
    objects = S3ObjectStore(
        os.environ["LAB_ARENA_BUCKET"],
        prefix=os.environ.get("LAB_ARENA_OBJECT_PREFIX", ""),
    )
    raw, prompt = (json.loads(read_file(path)) for path in (args.batch, args.prompt))
    if args.action in {"experiment-audit", "corrections", "training-feedback"}:
        from lab_arena.collection_experiment import load_experiment
        from lab_arena.collection_experiment_export import record_feedback, write_dataset

        spec = load_experiment(raw, prompt)
        if args.action == "training-feedback":
            if not args.dataset_metadata or not args.evaluation:
                p.error("supply the released dataset metadata and signed Poet evaluation")
            metadata = json.loads(read_file(args.dataset_metadata))
            if metadata["batch_id"] != spec.batch_id:
                raise ValueError("evaluation references a different experiment")
            ref = record_feedback(
                metadata, json.loads(read_file(args.evaluation)), objects,
                read_file(Path(os.environ["LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE"]), private=False),
            )
            write_new(args.output, ref)
            return 0
        qualification = None
        if args.action == "corrections":
            if not (args.panel_batch and args.panel_prompt and args.qualification):
                p.error("supply the registered qualification pilot and its independent approvals")
            qualification = {
                "raw":json.loads(read_file(args.panel_batch)), "prompt":json.loads(read_file(args.panel_prompt)),
                "receipt":json.loads(read_file(args.receipt)), "approval":json.loads(read_file(args.qualification)),
                "reviewer_key":read_file(Path(os.environ["LEADPOET_COLLECTION_REVIEW_PUBLIC_KEY_FILE"]), private=False),
                "approver_key":read_file(Path(os.environ["LEADPOET_COLLECTION_QUALIFICATION_PUBLIC_KEY_FILE"]), private=False),
            }
        result = write_dataset(spec, db, objects, args.output, qualification=qualification)
        print(json.dumps({"batch":spec.batch_id,"attempts":result["attempts"],"rows":result["rows"]}))
        return 0
    bundle = source_bundle(raw, prompt, db, objects)
    if args.action == "bundle":
        write_new(args.output, bundle)
    else:
        receipt = json.loads(read_file(args.receipt))
        # The deployment owns this setting independently of the receipt submitted for review.
        public_key = read_file(
            Path(os.environ["LEADPOET_COLLECTION_REVIEW_PUBLIC_KEY_FILE"]),
            maximum=8192,
            private=False,
        )
        if args.action == "prices":
            write_new(
                args.output,
                dict(
                    compile_pilot(
                        db, bundle, receipt, public_key, objects,
                        approval=json.loads(read_file(args.qualification)) if args.qualification else None,
                        approver_key=read_file(Path(os.environ["LEADPOET_COLLECTION_QUALIFICATION_PUBLIC_KEY_FILE"]),
                                               maximum=8192, private=False) if args.qualification else None,
                    ),
                    cutover_anchor=cutover_anchor(db),
                ),
            )
        else:
            write_exports(bundle, receipt, public_key, objects, args.output)
    print(
        json.dumps(
            {
                "batch": bundle["policy"]["batch_id"],
                "action": args.action,
                "completed": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(
            "collection batch operation failed: " + type(exc).__name__
        ) from None
