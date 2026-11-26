"""Freeze a paid P7 batch and stage its measured quotes using the existing launch transaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from lab_arena import collection_judge as judge
from lab_arena.collection_generation import CAPTURE_V2
from lab_arena.collection_intake import PAID_WORKFLOW_TERMS_HASH, live_finney_arena
from lab_arena.collection_worker import P7_PAYMENT_RULE, native_payment_evidence
from lab_arena.collection_workflow import comparison_revisions
from scripts.collection_batch import cutover_anchor, read_artifact
from scripts.collection_review import read_file, write_new
from scripts.run_locked_collection import evidence_stores, load_batch


def compile_prices(source, prompt, jobs, scenarios, db, objects):
    """Replay exactly the selected, paid-for generator calls. Judges never set the price."""
    pilot = load_batch(source, prompt)
    if (
        not pilot.pilot
        or source.get("capture_version") != CAPTURE_V2
        or not source.get("split_manifest")
    ):
        raise ValueError("expected a frozen synthetic comparison with full capture")
    if not jobs or len(set(jobs)) != len(jobs):
        raise ValueError("distinct measured job IDs required")
    jobs = sorted(jobs)
    agent, builder = comparison_revisions(source)
    rows = {
        row["job_id"]: row
        for row in db.rpc("collection_pilot_costs_v1", {"p_batch": pilot.batch_id})
    }
    for job_id in jobs:
        row = rows[job_id]
        if (
            row["status"] != "accepted"
            or row["harness_commit"] != agent
            or row["unresolved"]
            or row["scenario_id"] not in source["scenario_ids"]
            or str(row["slot"]) in source["control_slots"]
            or str(row["slot"]) in source["drift_slots"]
        ):
            raise ValueError("source must be ordinary, complete and settled")
        manifest = read_artifact(
            objects,
            pilot.batch_id,
            job_id,
            "manifest",
            {"sha256": row["artifact_sha256"]},
            1024 * 1024,
        )
        outcome = native_payment_evidence(objects, row, manifest)
        if (
            outcome["eligible"] is not True
            or outcome["cost_microusd"] != row["miner_cost_microusd"]
            or outcome["calls"] != row["generator_calls"]
        ):
            raise ValueError(
                "P7 native outcome, funded billing and original model replies must agree"
            )
    measurement = db.rpc(
        "collection_verified_p7_fees_v1", {"p_batch": pilot.batch_id, "p_jobs": jobs}
    )
    paid = {
        "mode": "live",
        "name": "paid-p7",
        "seed": source["seed"],
        "payment_rule": P7_PAYMENT_RULE,
        "control_slots": {},
        "drift_slots": {},
        "routes": source["routes"],
        "scenario_ids": scenarios,
        "budgets_microusd": measurement["budgets"]["P"],
        "fixed_quotes_microusd": measurement["prices"]["P"],
        "terms_sha256": PAID_WORKFLOW_TERMS_HASH,
        "split_manifest": source["split_manifest"],
        "capture_version": CAPTURE_V2,
    }
    live = load_batch(paid, prompt)
    evidence = {
        **measurement,
        "payment_rule": P7_PAYMENT_RULE,
        "schema": "leadpoet.collection.verified_p7_prices.v1",
        "batch_id": pilot.batch_id,
        "live_batch_id": live.batch_id,
        "measurement_jobs": jobs,
        "terms_sha256": PAID_WORKFLOW_TERMS_HASH,
        "routes": {model: paid["routes"][model] for model in paid["budgets_microusd"]},
        "harness_commit": agent,
        "builder_commit": builder,
        "capture_version": CAPTURE_V2,
        "split_manifest_sha256": judge.digest(paid["split_manifest"]),
        "judge_prompt_sha256": judge.digest(prompt),
    }
    from scripts.run_locked_collection import verify_live_panel

    verify_live_panel(paid, live, evidence)
    return paid, evidence


def stage(connection, evidence):
    """The immutable signed predecessor is checked again by PostgreSQL at insert."""
    from psycopg2.extras import Json

    if evidence.get(
        "schema"
    ) != "leadpoet.collection.verified_p7_prices.v1" or not re.fullmatch(
        r"dv-[a-f0-9]{64}", str(evidence.get("live_batch_id"))
    ):
        raise ValueError("expected a verified P7 paid offer")
    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout='5s'")
        cursor.execute("SET LOCAL ROLE lab_arena_owner")
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0))"
        )
        cursor.execute("SELECT public.collection_reward_schedule_v1()")
        existing = cursor.fetchone()[0]
        if existing:
            previous = dict(existing["pilot_evidence"])
            previous["cutover_anchor"] = evidence["cutover_anchor"]
            if previous != evidence:
                raise ValueError("a different immutable price schedule exists")
            return "already_staged" if existing.get("staged") else "active"
        anchor = evidence["cutover_anchor"]
        cursor.execute(
            "INSERT INTO public.collection_reward_launch(network,netuid,first_epoch,pilot_evidence) "
            "VALUES ('finney',71,%s,%s)",
            (anchor["epoch"] + 1, Json(evidence)),
        )
        cursor.execute("SELECT public.collection_reward_schedule_v1()")
        saved = cursor.fetchone()[0]
        if saved.get("staged") is not True or saved["pilot_evidence"] != evidence:
            raise RuntimeError("staged price schedule differs")
    return "staged"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser(
        "compile",
        help="replay registered sources and write a paid manifest and operator proof",
    )
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--prompt", required=True, type=Path)
    build.add_argument("--job", required=True, action="append")
    build.add_argument("--first-seed", required=True, type=int)
    build.add_argument("--scenario-count", required=True, type=int)
    build.add_argument("--paid-batch", required=True, type=Path)
    build.add_argument("--evidence", required=True, type=Path)
    approve = actions.add_parser(
        "stage", help="authorize the measured prices for prospectively signed grants"
    )
    approve.add_argument("--evidence", required=True, type=Path)
    approve.add_argument("--local-port", required=True, type=int)
    approve.add_argument("--ca-file", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "stage":
        from scripts.arena_rds_migration import connect

        evidence = json.loads(read_file(args.evidence))
        conn = connect(args.local_port, args.ca_file)
        try:
            conn.autocommit = False
            status = stage(conn, evidence)
        finally:
            conn.close()
    else:
        if (
            not live_finney_arena()
            or not 1 <= args.scenario_count <= 4096
            or not (1 <= args.first_seed <= 10**8 - args.scenario_count)
        ):
            raise ValueError("Finney/71 and a bounded P7 scenario range required")
        source, prompt = (json.loads(read_file(p)) for p in (args.source, args.prompt))
        db, objects = evidence_stores()
        scenarios = [
            f"P7-s{seed}"
            for seed in range(args.first_seed, args.first_seed + args.scenario_count)
        ]
        paid, evidence = compile_prices(
            source, prompt, args.job, scenarios, db, objects
        )
        evidence["cutover_anchor"] = cutover_anchor(db)
        # Recompilation may refresh the chain anchor but must never overwrite a different batch.
        if args.paid_batch.exists():
            if json.loads(read_file(args.paid_batch)) != paid:
                raise ValueError("different paid manifest exists at destination")
        else:
            write_new(args.paid_batch, paid)
        write_new(args.evidence, evidence)
        status = "compiled"
    print(
        json.dumps(
            {
                "status": status,
                "paid_batch": evidence["live_batch_id"],
                "quotes_microusd": evidence["prices"]["P"],
            }
        )
    )


if __name__ == "__main__":
    main()
