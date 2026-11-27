"""Prepare blind review, compile pilot prices, or release training rows from registered evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from lab_arena import collection_judge as judge, collection_judge_review as review
from lab_arena.collection_calibration import compile_pilot
from lab_arena.collection_intake import MODELS, live_finney_arena
from lab_arena.collection_workflow import comparison_revisions, read_artifact
from lab_arena.service import S3ObjectStore
from lab_arena.store import PostgrestTransport
from scripts.collection_review import read_file, write_new
from scripts.run_locked_collection import load_batch, assignment


def assigned_export_slots(capacity, first, completed, registered):
    """A successor can expose only its genuine terminal prefix at the original absolute indices."""
    unique = sorted(set(registered))
    if (
        not completed
        or not unique
        or min(unique) != first
        or max(unique) >= capacity
        or unique != list(range(first, max(unique) + 1))
        or sorted(completed) != list(range(first, first + len(completed)))
        or not set(completed) <= set(unique)
    ):
        raise ValueError("missing, fabricated or duplicate successor slots")
    return set(completed)


def source_bundle(raw, prompt, transport, objects):
    if raw.get("reference_successor"):
        raise ValueError(
            "a witnessed route comparison is private reference data only for qualification or pricing"
        )
    return _registered_projection(raw, prompt, transport, objects)


def _registered_projection(raw, prompt, transport, objects, *, export_successor=False):
    policy = load_batch(raw, prompt)
    agent, builder = comparison_revisions(raw)
    jobs = transport.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
    expected_count = len(raw["scenario_ids"]) * len(MODELS)
    registered = jobs
    successor = raw.get("reference_successor") if export_successor else None
    if successor:
        terminal = {"accepted", "rejected", "failed"}
        active_slots = {r["slot"] for r in jobs if r["status"] not in terminal}
        jobs = [
            r for r in jobs if r["status"] in terminal and r["slot"] not in active_slots
        ]
    if not isinstance(jobs, list) or (
        not successor and not expected_count <= len(jobs) <= 2 * expected_count
    ):
        raise ValueError("incomplete registered batch")
    if any(j["batch_id"] != policy.batch_id or j["mode"] != raw["mode"] for j in jobs):
        raise ValueError("job differs from frozen batch")
    failures = []
    if policy.pilot:
        jobs, failures = review.terminal_pilot_attempts(jobs)
    if successor:
        assigned_export_slots(
            expected_count,
            successor["next_slot"],
            [j["slot"] for j in jobs],
            [j["slot"] for j in registered],
        )
    elif len(jobs) != expected_count:
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
    if successor:
        # Deliberately non-qualifying. The registered projection was verified in memory above.
        bundle["schema"] = "collection.successor-export-sources.v1"
    return {**bundle, "split_manifest_sha256": judge.digest(raw["split_manifest"])}


def write_exports(raw, prompt, database, objects, output, *, qualification=None, only_jobs=None, publish=True):
    """Publish authorized, originally funded comparison traces against a separately qualified panel."""
    from lab_arena import collection_intake as intake, collection_trajectories as traces
    from lab_arena.collection_broker import batch_versions
    from lab_arena.collection_experiment_export import verified_panel
    from lab_arena.collection_quality import (
        require_release_partition,
        verify_split,
        PARTITIONS,
    )
    from lab_arena.collection_workflow import store_artifact

    from lab_arena.collection_worker import registered_native_result
    from scripts.run_locked_collection import verify_live_panel

    terms = raw.get("terms_sha256")
    native = "payment_rule" in raw
    if only_jobs is not None and not native:
        raise ValueError("incremental comparison release requires an approved native offer")
    expected_terms = intake.PAID_WORKFLOW_TERMS_HASH if native else intake.SYNTHETIC_TRAINING_TERMS_HASH
    if not terms or terms != expected_terms:
        raise ValueError("comparison enrollment did not grant training release rights")
    if not qualification and not native:
        raise ValueError("independent reviewer and qualification signatures required")
    policy = load_batch(raw, prompt)
    split = raw["split_manifest"]
    split_hash = "sha256:" + judge.digest(split)
    require_release_partition(
        {
            "cases": [{"scenario_id": sid} for sid in raw["scenario_ids"]],
            "split_sha256": split_hash,
        }
    )
    if native:
        schedule = database.rpc("collection_offer_schedule_v1", {"p_offer": policy.batch_id})
        approved = (schedule or {}).get("pilot_evidence") or {}
        verify_live_panel(raw, policy, approved)
    else:
        approved = verified_panel(database, objects, raw, policy, **qualification)
    witness = None
    successor = raw.get("reference_successor")
    if successor or raw.get("route_witness_sha256"):
        from lab_arena.collection_route_witness import (
            verified_witness,
            verify_matching_panel,
        )

        witness = verified_witness(raw, objects)
        verify_matching_panel(
            raw,
            policy,
            qualification["raw"],
            qualification["prompt"],
            database,
            successor=bool(successor),
        )
    if not raw.get("capture_version"):
        raise ValueError(
            "comparison training release requires a checkpointed transcript"
        )
    bundle = {} if native else _registered_projection(
        raw, prompt, database, objects, export_successor=bool(successor)
    )
    rows = database.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
    if only_jobs is not None:
        rows = [row for row in rows if row["job_id"] in only_jobs]
        if not only_jobs or len(rows) != len(only_jobs) or {row["job_id"] for row in rows} != set(only_jobs):
            raise ValueError("snapshot jobs differ from registered work")
    grants = traces.verified_grants(
        database,
        policy.batch_id,
        rows,
        mode=raw["mode"],
        scope=intake.TASK_SCOPE,
        digest=terms,
        allow_pending=native or bool(successor),
        **({"subset": True} if only_jobs is not None else {}),
    )
    by_id = {item["packet"]["run_id"]: item for item in bundle.get("runs", ())}
    failed = {item["run_id"] for item in bundle.get("failed_attempts", ())}
    if native:
        rows = [job for job in rows if job["status"] in {"accepted", "rejected", "failed"}
                and not grants[job["job_id"]]["unresolved"]]
        if not rows:
            raise ValueError("no settled native collection runs available to export")
        roster = [m for m in MODELS if m in raw["budgets_microusd"]]
        for job in rows:
            rid = job["job_id"]
            slot = job["slot"]
            if (job["batch_id"] != policy.batch_id or job["mode"] != raw["mode"]
                    or type(slot) is not int or not 0 <= slot < len(raw["scenario_ids"]) * len(roster)
                    or job["scenario_id"] != raw["scenario_ids"][slot // len(roster)]
                    or job["model"] != roster[slot % len(roster)] or job["harness_commit"] != comparison_revisions(raw)[0]):
                raise ValueError("registered native job differs from the frozen assignment")
            if job["status"] == "failed":
                failed.add(rid)
                continue
            manifest, checked = registered_native_result(job, objects, policy, approved)
            if (checked["eligible"] != (job["status"] == "accepted")
                    or checked["cost_microusd"] != grants[rid]["miner_cost_microusd"]
                    or checked["calls"] != grants[rid]["generator_calls"]):
                raise ValueError("verified native run differs from the funded outcome")
            packet = read_artifact(objects, policy.batch_id, rid, "judge-input", manifest["packet"], 16 * 1024 * 1024)
            by_id[rid] = {"packet": packet, "assessment": None,
                          "manifest_sha256": job["artifact_sha256"], "native": checked}

    if successor:
        allowed = set(by_id) | failed
        rows = [r for r in rows if r["job_id"] in allowed]
    if set(by_id) | failed != {r["job_id"] for r in rows}:
        raise ValueError("registered jobs changed during release")
    controls = set(bundle.get("planted", ())) | set(bundle.get("drift", ()))
    agent, builder = comparison_revisions(raw)
    revisions = batch_versions(raw["routes"])
    settings_hash = "sha256:" + judge.digest(raw)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        stage = Path(td) / "trajectories.zip"
        with ZipFile(stage, "x", compression=ZIP_DEFLATED) as z:
            full = traces.TrajectoryArchive(z)
            for job in sorted(rows, key=lambda r: (r["slot"], r["attempt"])):
                rid = job["job_id"]
                if (
                    rid in controls
                    or str(job["slot"]) in raw["control_slots"]
                    or str(job["slot"]) in raw["drift_slots"]
                ):
                    full.stats["excluded_controls"] += 1
                    continue
                auth = grants[rid]
                if auth["unresolved"]:
                    raise ValueError("unconfirmed miner charges for successor row")
                billed = dict(
                    job,
                    **{
                        k: auth[k]
                        for k in (
                            "miner_cost_microusd",
                            "generator_calls",
                            "unresolved",
                        )
                    },
                )
                packet = None
                assessment = None
                hold = None
                stamp = None
                if rid in by_id:
                    item = by_id[rid]
                    if item["manifest_sha256"] != job.get(
                        "pilot_review_sha256", job["artifact_sha256"]
                    ):
                        raise ValueError("registered generator manifest changed")
                    manifest = read_artifact(
                        objects,
                        policy.batch_id,
                        rid,
                        "manifest",
                        {"sha256": item["manifest_sha256"]},
                        1024 * 1024,
                    )
                    generator = read_artifact(
                        objects,
                        policy.batch_id,
                        rid,
                        "generator",
                        manifest["generator"],
                        128 * 1024 * 1024,
                    )
                    packet = item["packet"]
                    assessment = item["assessment"]
                    stamp = verify_split(packet["split"], split)
                    if generator.get("split") != {k: stamp[k] for k in PARTITIONS}:
                        raise ValueError("comparison transcript differs from its split")
                    calls = [
                        ev
                        for ev in (generator.get("evidence") or {}).get("events", [])
                        if ev["kind"] == "reply"
                    ]
                    if (
                        not billed["generator_calls"]
                        or not billed["miner_cost_microusd"]
                    ):
                        raise ValueError(
                            "complete generator has no verified funded call"
                        )
                else:
                    saved = f"collection/{policy.batch_id}/{rid}/saved/"
                    pointer = objects.get_optional_bounded(saved + "split.json", 8192)
                    if pointer:
                        stamp = verify_split(
                            read_artifact(
                                objects,
                                policy.batch_id,
                                rid,
                                "split",
                                json.loads(pointer),
                                1024 * 1024,
                            ),
                            split,
                        )
                    original = (
                        bundle.get("failed_journals", {}).get(rid, {}).get("events")
                    )
                    generator, calls, hold = traces.partial_generator(
                        objects, job, original
                    )
                    if not stamp:
                        hold = "missing_verified_split"
                    if (
                        objects.get_optional_bounded(saved + "quarantine.json", 8192)
                        is not None
                    ):
                        hold = "quarantine"
                trajectory = traces.trajectory_row(
                    job=billed,
                    generator=None if hold else generator,
                    packet=packet,
                    calls=calls,
                    settings_sha256=settings_hash,
                    harness=agent,
                    builder=builder,
                    revision=revisions[job["model"]],
                    route=raw["routes"][job["model"]],
                    split=stamp,
                    terms_sha256=terms,
                    assessment=assessment,
                    partial=rid in failed,
                    hold=hold,
                )
                if native:
                    trajectory["verification_method"] = raw["payment_rule"]
                    if packet and trajectory["final_turn_present"]:
                        trajectory["verified_task_pass"] = all(packet["checks"][k] is True for k in (
                            "answered", "shape", "code_grade", "no_hard_failure", "privacy", "training_split"
                        ))
                    checked = by_id.get(rid, {}).get("native") or {}
                    if "verified_decision" in checked:
                        # These are the replayed checks and model-visible observations. Private
                        # source worlds, reference answers and undiscovered documents stay sealed.
                        trajectory["sales_evidence"] = {
                            key: checked[key] for key in (
                                "verified_decision", "field_evidence", "searches", "opened_sources",
                                "first_failed_decision", "observed_repair",
                            )
                        }
                        if packet and trajectory["final_turn_present"]:
                            trajectory["verified_task_pass"] = (
                                trajectory["verified_task_pass"] and checked["verified_decision"]
                            )
                if successor:
                    trajectory["reference_successor"] = dict(successor)
                if witness and job["model"] == witness["model"]:
                    trajectory["route_witness"] = witness
                full.write(trajectory)
            result = full.finish()
            metadata = {
                "schema": traces.SCHEMA,
                "batch_id": policy.batch_id,
                "settings_sha256": settings_hash,
                "split_sha256": split_hash,
                ("price_schedule_sha256" if native else "qualification_sha256"): "sha256:" + judge.digest(approved),
                "rows": result.get("trajectory_rows", 0),
                **result,
            }
            if successor:
                metadata["reference_successor"] = dict(successor)
                metadata["actual_slots"] = sorted({j["slot"] for j in rows})
                metadata["complete_qualification_batch"] = False
            if witness:
                metadata["route_witness"] = witness
            z.writestr("metadata.json", judge._json(metadata))
        os.chmod(stage, 0o600)
        if publish:
            store_artifact(objects, policy.batch_id, "export", "split", split)
            store_artifact(objects, policy.batch_id, "export", "release", metadata)
        os.link(stage, output)
    return metadata


def paid_capture_counts(archive, job, batch):
    """Verify the accepted job in a private first-payment export."""
    from uuid import UUID
    from lab_arena.collection_worker import P7_PAYMENT_RULE

    if str(UUID(job)) != job:
        raise ValueError("invalid funded job identifier")
    path = f"trajectories/{job}.jsonl"
    with ZipFile(archive) as z:
        if z.namelist().count(path) != 1:
            raise ValueError("first paid trajectory is missing or duplicated")
        with z.open(path) as source:
            row = json.load(source)
    if (
        row["run_id"] != job
        or row["batch_id"] != batch
        or row["job_status"] != "accepted"
        or row["verified_task_pass"] is not True
        or row["verification_method"] != P7_PAYMENT_RULE
    ):
        raise ValueError("export does not prove the first accepted paid job")
    counts = row["length"]
    if any(
        type(counts.get(k)) is not int or counts[k] <= 0
        for k in ("reasoning_bytes", "tool_result_bytes")
    ):
        raise ValueError("first paid trajectory lacks reasoning or tool results")
    return counts


def cutover_anchor(transport, *, launch_job=None):
    """Bind pricing to the current signed, finalized epoch before paid collection."""
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
        if not row:
            raise ValueError("current signed state required")
        basis = row["state_doc"]["reward_basis"]
        if basis["schema_version"].startswith("leadpoet.collection."):
            # The prospective retirement can have signed many empty epochs.
            # Paid launch still needs its original signed approvals and live job.
            schedule = transport.rpc("collection_reward_schedule_v1", {})
            if (
                basis["schema_version"] != "leadpoet.collection.fixed_run_quotes.v1"
                or basis["job_count"] != 0
                or basis["quotes_microusd"]
                or basis["valuation"] is not None
                or (schedule and not schedule.get("staged"))
            ):
                raise ValueError("expected an unpaid collection epoch before launch")
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
    raw, prompt = (json.loads(read_file(path)) for path in (args.batch, args.prompt))
    native_export = args.action == "export" and "payment_rule" in raw
    if args.action in {"prices", "export", "corrections"} and not native_export and args.receipt is None:
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
    from lab_arena.collection_experiment import SCHEMA, load_experiment
    from lab_arena.collection_experiment_export import record_feedback, write_dataset

    experiment = raw.get("schema") == SCHEMA
    if args.action == "training-feedback":
        if not args.dataset_metadata or not args.evaluation:
            p.error("supply the released dataset metadata and signed Poet evaluation")
        metadata = json.loads(read_file(args.dataset_metadata))
        batch = (
            load_experiment(raw, prompt).batch_id
            if experiment
            else load_batch(raw, prompt).batch_id
        )
        if metadata["batch_id"] != batch:
            raise ValueError("evaluation references a different batch")
        ref = record_feedback(
            metadata,
            json.loads(read_file(args.evaluation)),
            objects,
            read_file(
                Path(os.environ["LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE"]),
                private=False,
            ),
        )
        write_new(args.output, ref)
        return 0
    if args.action in {"experiment-audit", "corrections", "export"}:
        qualification = None
        if args.action != "experiment-audit" and not native_export:
            if not (args.panel_batch and args.panel_prompt and args.qualification):
                p.error(
                    "supply the original qualification pilot and independent approvals"
                )
            qualification = {
                "raw": json.loads(read_file(args.panel_batch)),
                "prompt": json.loads(read_file(args.panel_prompt)),
                "receipt": json.loads(read_file(args.receipt)),
                "approval": json.loads(read_file(args.qualification)),
                "reviewer_key": read_file(
                    Path(os.environ["LEADPOET_COLLECTION_REVIEW_PUBLIC_KEY_FILE"]),
                    private=False,
                ),
                "approver_key": read_file(
                    Path(
                        os.environ["LEADPOET_COLLECTION_QUALIFICATION_PUBLIC_KEY_FILE"]
                    ),
                    private=False,
                ),
            }
        if experiment:
            result = write_dataset(
                load_experiment(raw, prompt),
                db,
                objects,
                args.output,
                qualification=qualification,
                full_trajectories=args.action == "export",
            )
        elif args.action == "export":
            result = write_exports(
                raw, prompt, db, objects, args.output, qualification=qualification
            )
        else:
            p.error("expected an experiment manifest")
        print(json.dumps({"batch": result["batch_id"], "rows": result["rows"]}))
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
        write_new(
            args.output,
            dict(
                compile_pilot(
                    db,
                    bundle,
                    receipt,
                    public_key,
                    objects,
                    witnessed_batch=(raw, prompt),
                    approval=(
                        json.loads(read_file(args.qualification))
                        if args.qualification
                        else None
                    ),
                    approver_key=(
                        read_file(
                            Path(
                                os.environ[
                                    "LEADPOET_COLLECTION_QUALIFICATION_PUBLIC_KEY_FILE"
                                ]
                            ),
                            maximum=8192,
                            private=False,
                        )
                        if args.qualification
                        else None
                    ),
                ),
                cutover_anchor=cutover_anchor(db),
            ),
        )

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
