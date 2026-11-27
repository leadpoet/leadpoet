"""Prepare a synthetic experiment from a frozen task or an authenticated student checkpoint."""

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


def store_private_json(path, value):
    """Publish a complete immutable input even if two operator workers resume together."""
    encoded = judge._json(value).encode()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if json.loads(read_file(path)) != value:
                raise ValueError("a different frozen experiment already exists at this path")


def prepare_pool(args):
    from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
    from lab_arena.collection_experiment import MAX_EXPERIMENT_SLOTS
    from lab_arena.collection_generation import CAPTURE_V2
    from lab_arena.collection_pool import POOL_TERMS_HASH, POOL_PAYMENT_RULE
    from lab_arena.collection_workflow import workflow_sources, experiment_revisions

    if (not 300 <= args.first_seed <= 10**8 - args.scenario_count
            or not 1 <= args.scenario_count <= MAX_EXPERIMENT_SLOTS
            or len(set(args.model)) != len(args.model)):
        raise ValueError("invalid training seeds or model roster")
    cases = [{"scenario_id":f"{args.family}-s{i}"} for i in range(args.first_seed,args.first_seed+args.scenario_count)]
    agent,builder=experiment_revisions(c["scenario_id"] for c in cases)
    with workflow_sources(args.champion_repo,args.poet_repo,corrections=True,builder_revision=builder) as sources:
        split=sources[1].training_partition(args.family)
    routes = {m: REFERENCE_ROUTE if m == REFERENCE_MODEL else GENERATOR_ROUTES[m].tag for m in args.model}
    raw={"schema":SCHEMA,"id":args.experiment_id,"mode":"live","payment_rule":POOL_PAYMENT_RULE,
        "question":"How often does each model complete this sales task correctly across repeated fresh runs?",
        "pass_mark":"Full native grade and verified task-specific evidence.",
        "max_slots":len(cases)*args.repeat*len(args.model),"budget_microusd":args.cap_microusd*len(cases)*args.repeat*len(args.model),
        "capture_version":CAPTURE_V2,"terms_sha256":POOL_TERMS_HASH,"split_sha256":_hash(split),"routes":routes,"cases":cases,
        "arms":[{"id":f"expert-{n}","role":"expert","model":m,"repeats":args.repeat,"max_steps":args.max_steps,
                 "max_tokens":args.max_generated_tokens,"max_run_tokens":args.max_run_tokens,"reasoning_effort":"low",
                 "capture_reasoning":True,"budget_microusd":args.cap_microusd} for n,m in enumerate(args.model,1)]}
    if bool(args.source_batch) != bool(args.source_job):
        raise ValueError("supply both the original funded batch and job")
    if args.source_batch:
        from scripts.run_lab_arena_service import load_scoped_environment
        from scripts.run_locked_collection import evidence_stores
        from lab_arena.collection_pool_source import funded_source
        from lab_arena.collection_intake import register_signatures
        if args.renew or len(cases)!=1:
            raise ValueError("a correction needs one fixed source case")
        if args.environment_file:
            load_scoped_environment(args.environment_file)
        register_signatures()
        prompt=json.loads(read_file(args.prompt));source_spec=load_experiment(json.loads(read_file(args.source_batch)),prompt)
        db,objects=evidence_stores()
        source=funded_source(db,objects,source_spec.batch_id,args.source_job,prompt)
        if source["scenario_id"]!=cases[0]["scenario_id"] or source["student_model"] in args.model:
            raise ValueError("expert and scenario must match the observed student failure")
        source_hash=_hash(source)
        cases[0]["source_sha256"]=source_hash
        for a in raw["arms"]:a["role"]="continuation"
        raw["question"]="Can the expert recover from the recorded model error at its exact predecision checkpoint?"
        store_private_json(args.inputs/(source_hash[7:]+".json"),source)
    if args.renew:
        raw["successor_seeds"]=True
    spec=load_experiment(raw,json.loads(read_file(args.prompt)))
    store_private_json(args.inputs/(spec.raw["split_sha256"][7:]+".json"),split)
    store_private_json(args.directory/(spec.batch_id+".json"),raw)
    print(json.dumps({"batch":spec.batch_id,"slots":len(spec.slots),"builder":builder,"harness":agent}))
    return 0


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
    expert = sub.add_parser("prepare-expert", help="Run approved models from the initial state of each frozen task")
    expert.add_argument("--scenario-id", action="append", required=True)
    expert.add_argument("--model", action="append", choices=tuple(GENERATOR_ROUTES), required=True)
    for action in (prepare, gateway):
        action.add_argument("--producer-run-id", required=True)
        action.add_argument("--model", choices=tuple(GENERATOR_ROUTES), required=True)
    for name in ("source", "private-key"):
        prepare.add_argument("--" + name, type=Path, required=True)
    for name in ("producer", "student-revision"):
        prepare.add_argument("--" + name, required=True)
    gateway.add_argument("--native-sha256", required=True)
    gateway.add_argument("--environment-file", type=Path)
    for name in ("champion-repo", "poet-repo"):
        gateway.add_argument("--" + name, type=Path, required=True)
    for action in (prepare, gateway, expert):
        for name in ("split", "output"):
            action.add_argument("--" + name, type=Path, required=True)
        for name in (
            "experiment-id",
            "question",
            "pass-mark",
            "terms-sha256",
        ):
            action.add_argument("--" + name, required=True)
        action.add_argument("--max-slots", type=int, default=12)
        action.add_argument("--cap-microusd", type=int, required=True)
        action.add_argument("--max-steps", type=int, required=True)
        action.add_argument(
            "--max-generated-tokens",
            type=int,
            required=True,
            help="Maximum output tokens per provider call",
        )
        action.add_argument(
            "--max-run-tokens",
            type=int,
            default=65536,
            help="Maximum output tokens across one run",
        )
        action.add_argument("--repeat", type=int, default=1)
        action.add_argument(
            "--min-pricing-runs",
            type=int,
            help="Predeclare the minimum genuine sample before this pilot can price one paid class",
        )
    pool = sub.add_parser("prepare-pool", help="Stage an operator-owned standing experiment on fresh training cases")
    pool.add_argument("--family", choices=("P7", "P8", "P9", "P10"), required=True)
    pool.add_argument("--first-seed", type=int, required=True)
    pool.add_argument("--scenario-count", type=int, default=8)
    pool.add_argument("--model", action="append", choices=tuple(GENERATOR_ROUTES), required=True)
    pool.add_argument("--source-batch",type=Path,help="Original standing manifest for a real failed student")
    pool.add_argument("--source-job",help="Verified completed student job ID; changes the arm into a continuation")
    pool.add_argument("--environment-file",type=Path)
    pool.add_argument("--repeat", type=int, default=4)
    pool.add_argument("--cap-microusd", type=int, required=True)
    pool.add_argument("--max-steps", type=int, default=32)
    pool.add_argument("--max-generated-tokens", type=int, default=4096)
    pool.add_argument("--max-run-tokens", type=int, default=65536)
    pool.add_argument("--experiment-id", required=True)
    pool.add_argument("--renew", action="store_true", help="Advance completed packs to new deterministic training seeds")
    for field in ("directory", "inputs", "prompt", "champion-repo", "poet-repo"):
        pool.add_argument("--" + field, type=Path, required=True)
    feedback = sub.add_parser(
        "sign-feedback",
        help="Sign actual paired held-out results on the Poet evaluator's machine",
    )
    for name in ("results", "private-key", "output"):
        feedback.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare-pool":
        return prepare_pool(args)
    if args.action == "sign-feedback":
        result = json.loads(read_file(args.results))
        if (
            result.get("schema") != "collection.poet-evaluation.v1"
            or "signature" in result
        ):
            raise ValueError("expected unsigned Poet evaluation")
        write_new(args.output, sign(result, args.private_key))
        return 0
    full_task = args.action == "prepare-expert"
    if (not full_task and not NAME.fullmatch(args.producer_run_id)) or not SHA.fullmatch(args.terms_sha256):
        raise ValueError("invalid source run or terms digest")
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(read_file(args.split))
    split_id = _hash(split)
    src = attestation = src_id = None
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
    elif args.action == "prepare":
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
    if full_task:
        models = args.model
        if len(set(models)) != len(models):
            raise ValueError("use the repeat setting for repeated runs of one model")
        cases = [{"scenario_id": sid} for sid in args.scenario_id]
    else:
        models = [args.model]
        cases = [{"scenario_id": src["scenario_id"], "source_sha256": src_id}]
    raw = {
        "schema": SCHEMA,
        "mode": "pilot",
        "capture_version": "collection.workflow-events.v2",
        "claim_protocol": "collection.fresh-claims.v2",
        "id": args.experiment_id,
        "question": args.question,
        "pass_mark": args.pass_mark,
        "terms_sha256": args.terms_sha256,
        "split_sha256": split_id,
        "max_slots": args.max_slots,
        "budget_microusd": args.cap_microusd * args.repeat * len(cases) * len(models),
        "cases": cases,
        "arms": [
            {
                "id": f"expert-{i}" if full_task else "correction",
                "model": model,
                "role": "expert" if full_task else "continuation",
                "repeats": args.repeat,
                "reasoning_effort": "low",
                "capture_reasoning": True,
                "budget_microusd": args.cap_microusd,
                "max_steps": args.max_steps,
                "max_tokens": args.max_generated_tokens,
                "max_run_tokens": args.max_run_tokens,
            } for i, model in enumerate(models, start=1)
        ],
    }
    if attestation is not None:
        raw["provenance_sha256"] = _hash(attestation)
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
        for val in ([split] if full_task else [src, split, attestation]):
            write_new(inputs / (judge.digest(val) + ".json"), val)
        write_new(root / "correction-batch.json", raw)
        os.rename(root, args.output)
    print(
        json.dumps(
            {
                "source_digest": src_id,
                "slots": len(models) * len(cases) * args.repeat,
                "maximum_microusd": raw["budget_microusd"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
