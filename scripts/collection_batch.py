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
from lab_arena.collection_intake import MODELS
from lab_arena.collection_research import export_research_rows
from lab_arena.collection_workflow import CHAMPION_REV, POET_REV, export_row
from lab_arena.service import S3ObjectStore
from lab_arena.store import PostgrestTransport
from scripts.collection_review import read_file, write_new
from scripts.run_locked_collection import load_batch, assignment


def source_bundle(raw, prompt, transport, objects):
    policy = load_batch(raw, prompt)
    jobs = transport.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
    expected_count = len(raw["scenario_ids"]) * len(MODELS)
    if not isinstance(jobs, list) or len(jobs) != expected_count:
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
            or job["harness_commit"] != CHAMPION_REV
            or not job.get("artifact_sha256")
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
        objects=objects,
        planted=keyed_by_run("control_slots"),
        drift=keyed_by_run("drift_slots"),
    )
    return bundle


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
    """Publish an exclusive private ZIP after verifying all members. Failure leaves no partial ZIP."""
    report = review.verify_review(bundle, receipt, public_key)
    policy = judge.BatchPolicy(**bundle["policy"])
    assignments = {a["run_id"]: a for a in bundle["assignments"]}
    controls = set(bundle["planted"]) | set(bundle["drift"])
    modes = ("decisions", "reasoning", "hard_steps")
    metadata = {
        "schema": "leadpoet.collection.training-release.v1",
        "batch": policy.batch_id,
        "review_sha256": judge.digest(receipt),
        "views": {},
        "preferred_runs": {},
    }
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary)
        paths = {m: root / (m + ".jsonl") for m in modes}
        import contextlib

        with contextlib.ExitStack() as stack:
            handles = {m: stack.enter_context(p.open("xb")) for m, p in paths.items()}
            hashes = {m: hashlib.sha256() for m in modes}
            counts = dict.fromkeys(modes, 0)
            allowed = {
                item["assessment"]["run_id"]
                for item in bundle["runs"]
                if item["assessment"]["run_id"] not in controls
                and judge.training_selection(
                    item["assessment"],
                    approved_claim_runs=report.get("approved_claim_runs", []),
                )
            }
            for item in bundle["runs"]:
                a = item["assessment"]
                run = a["run_id"]
                assignment_doc = assignments[run]
                sid = a["scenario_id"]
                manifest = read_artifact(
                    objects,
                    policy.batch_id,
                    run,
                    "manifest",
                    {"sha256": item["manifest_sha256"]},
                    1024 * 1024,
                )
                if (
                    manifest["job_id"] != run
                    or manifest["batch_id"] != policy.batch_id
                    or manifest["model"] != assignment_doc["generator"]
                    or manifest["model"] not in MODELS
                    or manifest["harness_commit"] != CHAMPION_REV
                    or manifest["builder_commit"] != POET_REV
                    or manifest["scenario_id"] != sid
                ):
                    raise ValueError("generator source authority differs")
                if manifest.get("ranking"):
                    ranking = read_artifact(
                        objects,
                        policy.batch_id,
                        run,
                        "ranking",
                        manifest["ranking"],
                        1024 * 1024,
                    )
                    if ranking is not None and ranking.get("status") == "ranked":
                        if (
                            ranking["batch_id"] != policy.batch_id
                            or ranking["scenario_id"] != sid
                            or ranking["policy_sha256"]
                            != judge.digest(bundle["policy"])
                        ):
                            raise ValueError("ranking binding differs")
                        first = ranking["labels"][ranking["record"]["order"][0]]
                        if first in allowed:
                            metadata["preferred_runs"][sid] = first
                if run not in allowed:
                    continue
                row = read_artifact(
                    objects,
                    policy.batch_id,
                    run,
                    "generator",
                    manifest["generator"],
                    128 * 1024 * 1024,
                )
                if row["run"] != run:
                    raise ValueError("generator run differs")
                if sid.startswith("research-"):
                    segments = row["research_rows"]
                else:
                    segments = [row]
                if not segments or any(
                    s["teacher"] != manifest["model"]
                    or s["task"] != sid
                    or s["run"] != run
                    for s in segments
                ):
                    raise ValueError("generator identity differs")
                for mode in modes:
                    views = (
                        export_research_rows(row, a, mode)
                        if sid.startswith("research-")
                        else [export_row(row, a, mode)]
                    )
                    if not views or any(v is None for v in views):
                        raise ValueError("selected run has no trainable prefix")
                    for view in views:
                        data = (
                            json.dumps(
                                view,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode()
                        hashes[mode].update(data)
                        handles[mode].write(data)
                        counts[mode] += 1
        for m in modes:
            metadata["views"][m] = {"rows": counts[m], "sha256": hashes[m].hexdigest()}
        if not counts["decisions"]:
            raise ValueError("no trainable rows")
        archive = root / "release.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as z:
            for m, path in paths.items():
                z.write(path, arcname=path.name)
            z.writestr("manifest.json", json.dumps(metadata, sort_keys=True))
        archive.chmod(0o600)
        os.link(archive, output)
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("bundle", "prices", "export"))
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.action != "bundle" and args.receipt is None:
        p.error("--receipt is required")
    if (
        os.environ.get("LAB_ARENA_NETWORK") != "finney"
        or os.environ.get("LAB_ARENA_NETUID") != "71"
    ):
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
                args.output, compile_pilot(db, bundle, receipt, public_key, objects)
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
