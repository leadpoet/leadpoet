"""Publish a subsequent fixed offer from independently verifiable, comparable costs."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import re

from lab_arena import collection_judge as judge
from lab_arena.collection_generation import CAPTURE_V2
from lab_arena.collection_intake import (
    PAID_OFFER_ID,
    PAID_TASK_FAMILIES,
    PAID_WORKFLOW_TERMS_HASH,
    VERIFIED_TASK_PRICE_SCHEMA,
    live_finney_arena,
)
from lab_arena.collection_workflow import comparison_revisions, read_artifact
from scripts.collection_review import read_file, write_new


def mean_quotes(rows):
    grouped = defaultdict(list)
    for row in rows:
        cost, cap = row["cost_microusd"], row["budget_microusd"]
        if (
            type(cost) is not int
            or type(cap) is not int
            or not 0 < cost <= cap <= 10**10
        ):
            raise ValueError("invalid observed cost or spending cap")
        grouped[row["model"]].append(row)
    if not grouped:
        raise ValueError("no comparable accepted runs")
    prices, caps = {}, {}
    for model, observations in grouped.items():
        count = len(observations)
        total = sum(row["cost_microusd"] for row in observations)
        prices[model] = (5 * total + 4 * count - 1) // (4 * count)
        caps[model] = min(row["budget_microusd"] for row in observations)
    return prices, caps


def _native(objects, batch_id, job, *, receipts_out=None):
    from lab_arena.collection_worker import native_payment_evidence

    manifest = read_artifact(
        objects,
        batch_id,
        job["job_id"],
        "manifest",
        {"sha256": job["artifact_sha256"]},
        1024 * 1024,
    )
    evidence = native_payment_evidence(
        objects, dict(job, batch_id=batch_id), manifest, receipts_out=receipts_out
    )
    if not evidence["eligible"]:
        raise ValueError("source task or complete evidence failed native verification")
    return evidence, manifest


def owner_samples(source, objects, lookup):
    """A reference is an owner experiment, never a registration or a payable job."""
    from uuid import UUID

    batch_id = source["batch_id"]
    if (
        not re.fullmatch(r"owner-p8-[a-f0-9]{64}", batch_id)
        or not 0 < len(source["jobs"]) <= 4096
    ):
        raise ValueError("expected bounded owner P8 references")
    seen = set()
    rows = []
    for job in source["jobs"]:
        if set(job) != {
            "job_id",
            "model",
            "scenario_id",
            "harness_commit",
            "budget_microusd",
            "artifact_sha256",
        }:
            raise ValueError("unexpected owner price reference")
        run = job["job_id"]
        if str(UUID(run)) != run or not re.fullmatch(
            r"P8-s[1-9][0-9]{0,7}", job["scenario_id"]
        ):
            raise ValueError("invalid owner calibration task")
        originals = []
        outcome, _ = _native(objects, batch_id, job, receipts_out=originals)
        if outcome["payment_rule"] != "verified_sales_evidence_v1":
            raise ValueError("different task cannot price a sales research workflow")
        receipts = []
        for generation, original_cost, _wire_hash in originals:
            if (
                not isinstance(generation, str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,199}", generation)
                or generation in seen
            ):
                raise ValueError("missing or reused owner generation")
            seen.add(generation)
            cost, digest = lookup(generation)
            if (
                cost is None
                or not isinstance(digest, str)
                or not re.fullmatch("[a-f0-9]{64}", digest)
            ):
                raise ValueError("owner generation has no confirmed native bill")
            if cost.microusd != original_cost:
                raise ValueError(
                    "owner generation charges differ from archived replies"
                )
            receipts.append((generation, cost.microusd, digest))
        if (
            outcome["calls"] != len(receipts)
            or sum(r[1] for r in receipts) != outcome["cost_microusd"]
        ):
            raise ValueError("owner receipt set differs from verified run")
        rows.append(
            dict(
                job,
                cost_microusd=outcome["cost_microusd"],
                calls=len(receipts),
                provider_receipts_sha256=judge.digest(receipts),
            )
        )
    return rows


def production_samples(source, prompt, db, objects, through):
    from scripts.run_locked_collection import load_batch, verify_live_panel

    policy = load_batch(source, prompt)
    if source.get("mode") != "live" or not PAID_OFFER_ID.fullmatch(policy.batch_id):
        raise ValueError("expected a frozen paid production offer")
    schedule = db.rpc("collection_offer_schedule_v1", {"p_offer": policy.batch_id})
    verify_live_panel(source, policy, schedule["pilot_evidence"])
    if through is None:
        jobs = db.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
        terminal = {
            j["slot"] for j in jobs if j["status"] in {"accepted", "rejected", "failed"}
        }
        through = -1
        while through + 1 in terminal:
            through += 1
    rows = db.rpc(
        "collection_offer_samples_v1",
        {"p_source": policy.batch_id, "p_through_slot": through},
    )
    for row in rows:
        e, _ = _native(objects, policy.batch_id, row)
        if e["payment_rule"] != source["payment_rule"] or (
            row["cost_microusd"],
            row["calls"],
        ) != (e["cost_microusd"], e["calls"]):
            raise ValueError(
                "production observation differs from original funded evidence"
            )
    return policy.batch_id, through, rows


def build_offer(
    source, prompt, rows, scenarios, rule, source_id, *, through=None, historical=None
):
    from scripts.run_locked_collection import load_batch, verify_live_panel

    family = PAID_TASK_FAMILIES[rule]
    prices, caps = mean_quotes(rows)
    if len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source job")
    paid = {
        "name": "paid-sales-" + ("p7" if family == "P" else "p8"),
        "mode": "live",
        "seed": source["seed"],
        "payment_rule": rule,
        "routes": source["routes"],
        "scenario_ids": scenarios,
        "control_slots": {},
        "drift_slots": {},
        "fixed_quotes_microusd": prices,
        "budgets_microusd": caps,
        "terms_sha256": PAID_WORKFLOW_TERMS_HASH,
        "capture_version": CAPTURE_V2,
        "split_manifest": source["split_manifest"],
    }
    agent, builder = comparison_revisions(paid)
    if any(r["harness_commit"] != agent for r in rows):
        raise ValueError("price observations belong to another harness")
    policy = load_batch(paid, prompt)
    approval = {
        "schema": VERIFIED_TASK_PRICE_SCHEMA,
        "live_batch_id": policy.batch_id,
        "payment_rule": rule,
        "batch_id": source_id,
        "terms_sha256": PAID_WORKFLOW_TERMS_HASH,
        "builder_commit": builder,
        "harness_commit": agent,
        "routes": {m: paid["routes"][m] for m in prices},
        "capture_version": CAPTURE_V2,
        "judge_prompt_sha256": judge.digest(prompt),
        "split_manifest_sha256": judge.digest(paid["split_manifest"]),
        "prices": {family: prices},
        "budgets": {family: caps},
        "job_count": len(rows),
        "observations": rows,
        "measurement_kind": (
            "owner_provider_receipts" if through is None else "settled_production"
        ),
    }
    if through is not None:
        approval["through_slot"] = through
    if historical is not None:
        approval["historical_realization"] = historical
    verify_live_panel(paid, policy, approval)
    return paid, approval


def historical_realization(db, blocks):
    from lab_arena.collection_settlement import verify_closed_payment
    from scripts.collection_participation_watch import VALIDATORS

    epoch, quote_boundary, pay_boundary = blocks
    receipt = verify_closed_payment(
        os.environ,
        db,
        {"epoch": epoch, "boundary": quote_boundary},
        {"boundary": pay_boundary},
        VALIDATORS,
    )
    if (receipt["result"] != "collection_recipients_observed" or not receipt["amounts"]
            or any(r["actual_microusd_at_signed_price"] is None for r in receipt["amounts"].values())):
        raise ValueError("missing finalized comparable payout observation")
    return {
        "quote_epoch": epoch,
        "payment_block": pay_boundary,
        "block_hash": receipt["block_hash"],
        "quoted_microusd": sum(
            r["quoted_microusd"] for r in receipt["amounts"].values()
        ),
        "received_microusd_at_signed_price": sum(
            r["actual_microusd_at_signed_price"] for r in receipt["amounts"].values()
        ),
    }


def publish(conn, approval, previous):
    from psycopg2.extras import Json

    offer = approval["live_batch_id"]
    with conn, conn.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout='5s'")
        cursor.execute("SET LOCAL ROLE lab_arena_owner")
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0))"
        )
        cursor.execute(
            "SELECT previous_offer_id,approval FROM public.collection_paid_offers WHERE offer_id=%s",
            (offer,),
        )
        old = cursor.fetchone()
        if old:
            if old != (previous, approval):
                raise ValueError("offer already exists with different measurements")
            return "already_published"
        cursor.execute(
            "INSERT INTO public.collection_paid_offers(offer_id,previous_offer_id,approval) VALUES(%s,%s,%s)",
            (offer, previous, Json(approval)),
        )
        cursor.execute("SELECT public.collection_offer_schedule_v1(%s)", (offer,))
        if cursor.fetchone()[0]["pilot_evidence"] != {
            k: v for k, v in approval.items() if k != "observations"
        }:
            raise RuntimeError("published offer differs from approved measurements")
    return "published"


def main(argv=None):
    from scripts.run_locked_collection import evidence_stores

    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(required=True, dest="action")
    c = cmds.add_parser("compile-successor")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path)
    src.add_argument("--owner-references", type=Path)
    c.add_argument("--prompt", type=Path, required=True)
    c.add_argument("--first-seed", type=int, required=True)
    c.add_argument("--scenario-count", type=int, required=True)
    c.add_argument("--through-slot", type=int)
    c.add_argument(
        "--historical-payment",
        nargs=3,
        type=int,
        metavar=("EPOCH", "QUOTE_BLOCK", "PAID_BLOCK"),
    )
    c.add_argument("--paid-batch", type=Path, required=True)
    c.add_argument("--evidence", type=Path, required=True)
    stage = cmds.add_parser("publish-offer")
    stage.add_argument("--evidence", type=Path, required=True)
    stage.add_argument("--local-port", type=int, required=True)
    stage.add_argument("--ca-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "publish-offer":
        from scripts.arena_rds_migration import connect

        doc = json.loads(read_file(args.evidence))
        approval = doc["approval"]
        conn = connect(args.local_port, args.ca_file)
        try:
            conn.autocommit = False
            status = publish(conn, approval, doc["previous_offer_id"])
        finally:
            conn.close()
    else:
        if (
            not live_finney_arena()
            or not 1 <= args.scenario_count <= 4096
            or not 1 <= args.first_seed <= 10**8 - args.scenario_count
        ):
            raise ValueError("expected Finney/71 and a bounded training scenario list")
        db, objects = evidence_stores()
        source = json.loads(read_file(args.source or args.owner_references))
        prompt = json.loads(read_file(args.prompt))
        current = db.rpc("collection_offer_schedule_v1", {"p_offer": None})
        previous = current["pilot_evidence"]["live_batch_id"]
        if current.get("staged"):
            raise ValueError("initial paid cutover is not active")
        if args.source:
            source_id, through, rows = production_samples(
                source, prompt, db, objects, args.through_slot
            )
            rule = source["payment_rule"]
        else:
            from lab_arena.broker import HttpxProviderTransport
            from scripts.reconcile_collection_generation import exact_generation

            if args.through_slot is not None:
                raise ValueError("owner measurements have no production slots")
            owner_key = os.environ["LAB_ARENA_OPENROUTER_API_KEY"]
            http = HttpxProviderTransport()
            rows = owner_samples(
                source, objects, lambda g: exact_generation(http, owner_key, g)
            )
            source_id = source["batch_id"]
            through = None
            rule = "verified_sales_evidence_v1"
        family = "P7" if PAID_TASK_FAMILIES[rule] == "P" else "P8"
        scenarios = [
            f"{family}-s{s}"
            for s in range(args.first_seed, args.first_seed + args.scenario_count)
        ]
        historical = (
            historical_realization(db, args.historical_payment)
            if args.historical_payment
            else None
        )
        raw, approval = build_offer(
            source,
            prompt,
            rows,
            scenarios,
            rule,
            source_id,
            through=through,
            historical=historical,
        )
        if args.paid_batch.exists():
            if json.loads(read_file(args.paid_batch)) != raw:
                raise ValueError("different committed batch at destination")
        else:
            write_new(args.paid_batch, raw)
        write_new(args.evidence, {"previous_offer_id": previous, "approval": approval})
        status = "compiled"
    print(
        json.dumps(
            {
                "status": status,
                "offer_id": approval["live_batch_id"],
                "prices": approval["prices"],
            }
        )
    )


if __name__ == "__main__":
    main()
