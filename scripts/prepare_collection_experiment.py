"""Prepare a private one-source correction canary without making provider calls."""

import argparse
import base64
import json
import os
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from lab_arena import collection_judge as judge
from lab_arena.collection_experiment import SCHEMA, NAME, SHA, _hash, load_experiment
from lab_arena.collection_intake import GENERATOR_ROUTES
from lab_arena.collection_quality import admit_split
from scripts.collection_review import read_file, write_new


def sign(value, key_path):
    key = load_pem_private_key(read_file(key_path, maximum=8192), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("expected Ed25519 producer signing key")
    return {
        **value,
        "signature": base64.b64encode(key.sign(judge._json(value).encode())).decode(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser(
        "prepare",
        help="Sign original evidence on the producing machine and stage private inputs",
    )
    gateway = sub.add_parser(
        "prepare-gateway",
        help="Verify a private gateway student and its original owner-funded provider charges",
    )
    for name in ("source", "private-key"):
        prepare.add_argument("--" + name, type=Path, required=True)
    for name in ("producer", "student-revision"):
        prepare.add_argument("--" + name, required=True)
    gateway.add_argument("--native-sha256", required=True)
    gateway.add_argument("--environment-file", type=Path)
    for name in ("champion-repo", "poet-repo"):
        gateway.add_argument("--" + name, type=Path, required=True)
    for action in (prepare, gateway):
        for name in ("split", "output"):
            action.add_argument("--" + name, type=Path, required=True)
        for name in (
            "experiment-id",
            "producer-run-id",
            "question",
            "pass-mark",
            "terms-sha256",
        ):
            action.add_argument("--" + name, required=True)
        action.add_argument("--model", choices=tuple(GENERATOR_ROUTES), required=True)
        action.add_argument("--cap-microusd", type=int, required=True)
        action.add_argument("--max-steps", type=int, required=True)
        action.add_argument("--max-generated-tokens", type=int, required=True)
        action.add_argument("--repeat", type=int, default=1)
        action.add_argument(
            "--min-pricing-runs",
            type=int,
            help="Predeclare the minimum genuine sample before this pilot can price one paid class",
        )
    feedback = sub.add_parser(
        "sign-feedback",
        help="Sign actual paired held-out results on the Poet evaluator's machine",
    )
    for name in ("results", "private-key", "output"):
        feedback.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "sign-feedback":
        result = json.loads(read_file(args.results))
        if (
            result.get("schema") != "collection.poet-evaluation.v1"
            or "signature" in result
        ):
            raise ValueError("expected unsigned Poet evaluation")
        write_new(args.output, sign(result, args.private_key))
        return 0
    if not NAME.fullmatch(args.producer_run_id) or not SHA.fullmatch(args.terms_sha256):
        raise ValueError("invalid source run or terms digest")
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(read_file(args.split))
    split_id = _hash(split)
    if args.action == "prepare-gateway":
        from contextlib import closing
        from lab_arena.broker import HttpxProviderTransport
        from lab_arena.collection_gateway_student import (
            native_scenario,
            object_store,
            stage_gateway_source,
        )
        from lab_arena.collection_workflow import experiment_revisions, workflow_sources

        if args.environment_file:
            from scripts.run_lab_arena_service import load_scoped_environment

            load_scoped_environment(args.environment_file)
        objects = object_store()
        sid = native_scenario(objects, args.producer_run_id, args.native_sha256)
        _, builder = experiment_revisions([sid])
        with workflow_sources(
            args.champion_repo,
            args.poet_repo,
            corrections=True,
            builder_revision=builder,
        ) as sources, closing(HttpxProviderTransport()) as transport:
            src, attestation = stage_gateway_source(
                objects,
                sources,
                run_id=args.producer_run_id,
                native_sha256=args.native_sha256,
                split=split,
                transport=transport,
                owner_key=os.environ["LAB_ARENA_OPENROUTER_API_KEY"],
            )
        src_id = _hash(src)
    else:
        if not NAME.fullmatch(args.producer):
            raise ValueError("invalid external producer")
        import re

        if not re.fullmatch(
            r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})", args.student_revision
        ):
            raise ValueError("use the immutable student weight revision")
        src = json.loads(read_file(args.source))
        if (
            src.get("schema") != "leadpoet.collection.student-source.v1"
            or src.get("rejected_fatal") is not True
        ):
            raise ValueError("expected imported irreversible student failure")
        admit_split(src["split"], split)
        src_id, split_id = _hash(src), _hash(split)
        attestation = sign(
            {
                "schema": "collection.poet-student-sources.v1",
                "producer": args.producer,
                "split_sha256": split_id,
                "sources": {
                    src_id: {
                        "model": src["student_model"],
                        "model_revision": args.student_revision,
                        "role": "untuned_student",
                        "source_run_id": args.producer_run_id,
                        "split": "training",
                        "rights": "owned_synthetic_training_and_review",
                    }
                },
            },
            args.private_key,
        )
    raw = {
        "schema": SCHEMA,
        "mode": "pilot",
        "claim_protocol": "collection.fresh-claims.v2",
        "id": args.experiment_id,
        "question": args.question,
        "pass_mark": args.pass_mark,
        "terms_sha256": args.terms_sha256,
        "split_sha256": split_id,
        "provenance_sha256": _hash(attestation),
        "budget_microusd": args.cap_microusd * args.repeat,
        "cases": [{"scenario_id": src["scenario_id"], "source_sha256": src_id}],
        "arms": [
            {
                "id": "correction",
                "model": args.model,
                "role": "continuation",
                "repeats": args.repeat,
                "reasoning_effort": "low",
                "capture_reasoning": True,
                "budget_microusd": args.cap_microusd,
                "max_steps": args.max_steps,
                "max_tokens": args.max_generated_tokens,
            }
        ],
    }
    if args.min_pricing_runs is not None:
        raw["min_pricing_runs"] = args.min_pricing_runs
    load_experiment(
        raw, {}
    )  # Validate caps and schedule; the operator's frozen rubric determines the final ID.
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        root = Path(work) / "pack"
        inputs = root / "correction-inputs"
        inputs.mkdir(mode=0o700, parents=True)
        root.chmod(0o700)
        for val in (src, split, attestation):
            write_new(inputs / (judge.digest(val) + ".json"), val)
        write_new(root / "correction-batch.json", raw)
        os.rename(root, args.output)
    print(
        json.dumps(
            {
                "source_digest": src_id,
                "slots": args.repeat,
                "maximum_microusd": raw["budget_microusd"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
