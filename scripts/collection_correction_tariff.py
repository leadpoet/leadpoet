"""Prepare or explicitly sign a private fixed price. Never activates or pays."""

import argparse
import json
import os
from pathlib import Path
from lab_arena import collection_intake as funding, collection_judge as judge
from lab_arena.collection_correction_tariff import (
    SCHEMA,
    prepare,
    verify_pricing_signature,
)
from lab_arena.collection_experiment import load_experiment
from lab_arena.collection_workflow import workflow_sources
from scripts.collection_review import read_file, write_new
from scripts.prepare_collection_experiment import sign
from scripts.run_locked_collection import evidence_stores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("prepare")
    stage = sub.add_parser(
        "stage",
        help="Recompute and install a separately signed quote using the authorized RDS owner channel",
    )
    for action in (plan, stage):
        for path in (
            "pilot",
            "live",
            "prompt",
            "panel-batch",
            "panel-prompt",
            "review",
            "qualification",
            "inputs",
            "champion-repo",
            "poet-repo",
        ):
            action.add_argument("--" + path, type=Path, required=True)
        action.add_argument("--environment-file", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    stage.add_argument("--approval", type=Path, required=True)
    stage.add_argument("--local-port", type=int, required=True)
    stage.add_argument("--ca-file", type=Path, required=True)
    approve = sub.add_parser(
        "sign",
        help="pricing authority independently approves this exact bounded schedule",
    )
    for path in ("candidate", "private-key", "output"):
        approve.add_argument("--" + path, type=Path, required=True)
    args = p.parse_args(argv)

    def load(path):
        return json.loads(read_file(path))

    if args.action == "sign":
        body = load(args.candidate)
        if body.get("schema") != SCHEMA or "signature" in body:
            raise ValueError("expected an unsigned correction tariff")
        write_new(args.output, sign(body, args.private_key))
        return 0
    if args.environment_file:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
    if not funding.live_finney_arena():
        raise ValueError("expected Finney/71")
    funding.register_signatures()
    prompt = load(args.prompt)
    db, objects = evidence_stores()
    q = {
        "raw": load(args.panel_batch),
        "prompt": load(args.panel_prompt),
        "receipt": load(args.review),
        "approval": load(args.qualification),
        "reviewer_key": read_file(
            Path(os.environ["LEADPOET_COLLECTION_REVIEW_PUBLIC_KEY_FILE"]),
            maximum=8192,
            private=False,
        ),
        "approver_key": read_file(
            Path(os.environ["LEADPOET_COLLECTION_QUALIFICATION_PUBLIC_KEY_FILE"]),
            maximum=8192,
            private=False,
        ),
    }
    from lab_arena.collection_experiment import source_revisions
    live, pilot = load_experiment(load(args.live),prompt), load_experiment(load(args.pilot),prompt)
    if source_revisions(live.raw) != source_revisions(pilot.raw):
        raise ValueError("pilot and live builder revisions differ")
    with workflow_sources(
        args.champion_repo, args.poet_repo, corrections=True, builder_revision=source_revisions(live.raw)[1]
    ) as sources:
        candidate = prepare(
            live, pilot,
            args.inputs,
            sources,
            db,
            objects,
            q,
        )
    if args.action == "stage":
        from contextlib import closing
        from scripts.arena_rds_migration import connect

        approval = load(args.approval)
        if verify_pricing_signature(approval) != candidate:
            raise ValueError(
                "signed tariff differs from recomputed authorized evidence"
            )
        with closing(connect(args.local_port, args.ca_file)) as connection:
            connection.autocommit = False
            _install(connection, candidate, approval)
    else:
        write_new(args.output, candidate)
    print(
        json.dumps(
            {
                "batch": candidate["live_batch"],
                "measured_runs": candidate["measurement"]["completed"],
                "fixed_quote_microusd": candidate["measurement"]["quote_microusd"],
                "sha256": judge.digest(candidate),
            }
        )
    )
    return 0


def _install(connection, candidate, approval):
    from psycopg2.extras import Json

    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE lab_arena_owner")
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            ("collection_tariff:" + candidate["live_batch"],),
        )
        cursor.execute(
            "SELECT approval FROM public.collection_correction_tariffs WHERE batch_id=%s",
            (candidate["live_batch"],),
        )
        record = cursor.fetchone()
        if record is None:
            cursor.execute(
                "INSERT INTO public.collection_correction_tariffs(batch_id,pilot_id,approval) VALUES(%s,%s,%s) RETURNING approval",
                (candidate["live_batch"], candidate["pilot_batch"], Json(approval)),
            )
            record = cursor.fetchone()
        if record[0] != approval:
            raise ValueError("a different immutable tariff already exists")


if __name__ == "__main__":
    raise SystemExit(main())
