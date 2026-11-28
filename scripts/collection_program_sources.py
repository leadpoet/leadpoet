"""Sign native Poet runs on their producer, or install a separately signed sandbox admission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from cryptography.hazmat.primitives.serialization import load_pem_public_key
from lab_arena import collection_programs as programs
from lab_arena.collection_checkpoint import _native_files, prepare_student_source
from lab_arena.collection_node_runs import NODES, RESTATEMENTS, node_trajectory
from lab_arena.collection_experiment import NAME, _hash
from lab_arena.collection_workflow import workflow_sources
from scripts.collection_review import read_file
from scripts.prepare_collection_experiment import sign, store_private_json


def sign_nodes(args):
    items = json.loads(read_file(args.runs))
    entry = programs.sandbox(args.family)
    if not isinstance(items, list) or not 1 <= len(items) <= 10000:
        raise ValueError("invalid node volume")
    if not NAME.fullmatch(args.producer):
        raise ValueError("invalid producer name")
    runs = {}
    sources = {}
    seen = set()
    draws = set()
    out = args.output
    with workflow_sources(
        args.champion_repo,
        args.poet_repo,
        corrections=True,
        builder_revision=entry["builder"],
    ) as (h, b):
        for item in items:
            if set(item) != {
                "folder",
                "producer_run_id",
                "student_model",
                "model_revision",
                "repeat",
            }:
                raise ValueError("invalid native dossier index")
            folder = (args.runs.parent / item["folder"]).resolve()
            if not folder.is_relative_to(args.runs.parent.resolve()):
                raise ValueError("native dossier is outside its signed input root")
            rid = item["producer_run_id"]
            if not NAME.fullmatch(rid) or rid in seen:
                raise ValueError("duplicate or invalid producer run id")
            seen.add(rid)
            original = _native_files(folder)
            digest = _hash(original)
            if digest in runs:
                raise ValueError("a duplicate native dossier cannot count as a new run")
            row = node_trajectory(
                original,
                harness=h,
                builders=b,
                **{k: v for k, v in item.items() if k != "folder"},
            )
            draw = (row["scenario_id"], row["model_revision"], row["repeat"])
            if draw in draws:
                raise ValueError("duplicate student draw")
            draws.add(draw)
            if programs.identity(row["scenario_id"])[0] != args.family:
                raise ValueError("different sandbox in bulk input")
            store_private_json(out / "nodes" / (digest[7:] + ".json"), original)
            source_hash = None
            if (
                row["verified_task_pass"] is False
                and (row["first_failed_check"] or {}).get("kind") == "observed_branch"
            ):
                source = prepare_student_source(
                    h, b, folder, model=item["student_model"]
                )
                source_hash = _hash(source)
                store_private_json(out / "inputs" / (source_hash[7:] + ".json"), source)
                sources[source_hash] = dict(
                    role="untuned_student",
                    model=item["student_model"],
                    model_revision=item["model_revision"],
                    source_run_id=rid,
                    split="training",
                    rights="owned_synthetic_training_and_review",
                )
            runs[digest] = {
                **{k: v for k, v in item.items() if k != "folder"},
                "source_sha256": source_hash,
                "scenario_id": row["scenario_id"],
            }
    node = sign(
        {
            "schema": NODES,
            "producer": args.producer,
            "records": runs,
            "rights": "owned_synthetic_training_and_review",
        },
        args.private_key,
    )
    receipt = sign(
        {
            "schema": "collection.poet-student-sources.v2",
            "producer": args.producer,
            "split_sha256": entry["partition"],
            "sources": sources,
        },
        args.private_key,
    )
    store_private_json(out / "attestation.json", receipt)
    store_private_json(out / "nodes.json", node)
    return {
        "runs": len(runs),
        "correction_sources": len(sources),
        "nodes_sha256": _hash(node),
        "attestation_sha256": _hash(receipt),
    }


def sign_restatements(args):
    items = json.loads(read_file(args.runs))
    seen = set()
    records = {}
    if (
        not NAME.fullmatch(args.producer)
        or not isinstance(items, list)
        or not 1 <= len(items) <= 10000
    ):
        raise ValueError("invalid restatement batch")
    for ref in items:
        path = (args.runs.parent / ref).resolve()
        if not path.is_relative_to(args.runs.parent.resolve()):
            raise ValueError("restatement is outside its input root")
        row = json.loads(read_file(path))
        rid = row["producer_run_id"]
        if (
            row["schema"] != "poet.student-restatement.v1"
            or not NAME.fullmatch(rid)
            or rid in seen
            or str(row["student_model"]).lower() != programs.STUDENT
            or row["model_revision"] != programs.STUDENT_REVISION
        ):
            raise ValueError("unexpected or duplicate restatement")
        from lab_arena.collection_checkpoint import require_private_evidence

        require_private_evidence(row)
        seen.add(rid)
        digest = _hash(row)
        store_private_json(args.output / "restatements" / (digest[7:] + ".json"), row)
        records[digest] = rid
    envelope = sign(
        {
            "schema": RESTATEMENTS,
            "producer": args.producer,
            "records": records,
            "rights": "owned_synthetic_training_and_review",
        },
        args.private_key,
    )
    store_private_json(args.output / "restatements.json", envelope)
    return {"records": len(records), "attestation_sha256": _hash(envelope)}


def admit(args):
    doc = json.loads(read_file(args.record))
    key = load_pem_public_key(read_file(args.public_key, maximum=8192, private=False))
    entry = programs.verify_admission(doc, key)
    if entry["family"] in programs._builtins():
        raise ValueError("built-in historical sandbox cannot be overridden")
    ref = _hash(entry)
    f = args.directory / (entry["family"] + ".json")
    # The runtime requires both the signed file and independently curated database row.
    # Repeatable after a partial install; identical rows are the only accepted conflict.
    if entry["kind"] == "train":
        if not args.artifacts:
            raise ValueError("a trainable sandbox requires its reviewed gate artifacts")
        from hashlib import sha256

        for witness in entry["evidence"].values():
            digest = witness["artifact_sha256"]
            if (
                "sha256:"
                + sha256(
                    read_file(args.artifacts / (digest[7:] + ".json"), maximum=64 << 20)
                ).hexdigest()
                != digest
            ):
                raise ValueError(
                    "reviewed gate evidence differs from its signed digest"
                )
    store_private_json(f, doc)
    from scripts.arena_rds_migration import connect

    with connect(args.local_port, args.ca_file) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.collection_registered_sandboxes (family, kind, registry_sha256) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (entry["family"], entry["kind"], ref),
        )
        cur.execute(
            "SELECT kind,registry_sha256 FROM public.collection_registered_sandboxes WHERE family=%s",
            (entry["family"],),
        )
        if cur.fetchone() != (entry["kind"], ref):
            raise ValueError("a different sandbox registration already exists")
    return {"family": entry["family"], "kind": entry["kind"], "registry_sha256": ref}


def cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("sign-students")
    r = sub.add_parser("sign-restatements")
    a = sub.add_parser("admit-sandbox")
    for parser in (s, r):
        for k in ("runs", "private-key", "output"):
            parser.add_argument("--" + k, type=Path, required=True)
        parser.add_argument("--producer", required=True)
    s.add_argument("--family", required=True)
    for k in ("champion-repo", "poet-repo"):
        s.add_argument("--" + k, type=Path, required=True)
    for k in ("record", "public-key", "directory", "ca-file"):
        a.add_argument("--" + k, type=Path, required=True)
    a.add_argument("--local-port", type=int, required=True)
    a.add_argument("--artifacts", type=Path)
    args = p.parse_args(argv)
    f = {
        "sign-students": sign_nodes,
        "sign-restatements": sign_restatements,
        "admit-sandbox": admit,
    }[args.action]
    print(json.dumps(f(args)))
    return 0
