"""Compile finite, repeatable V3 experimentation packs from the reviewed registry and observed outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from zipfile import ZipFile
from hashlib import sha256

from lab_arena import collection_programs as programs
from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
from lab_arena.collection_experiment import SCHEMA, NAME, _hash, load_experiment
from lab_arena.collection_generation import CAPTURE_V2
from lab_arena.collection_intake import GENERATOR_ROUTES
from lab_arena.collection_pool import POOL_PAYMENT_RULE, POOL_TERMS_HASH
from lab_arena.collection_workflow import workflow_sources
from scripts.collection_review import read_file


def _pack(doc, cases, models, cap, *, receipts=None):
    collection = doc["collection"]
    role = {"C3": "student", "C4": "continuation"}.get(collection, "expert")
    repeat = {"C3": 8, "C4": 2}.get(collection, 4)
    arms = [
        dict(
            id=f"{role}-{i}",
            role=role,
            model=m,
            repeats=repeat,
            max_steps=32,
            max_tokens=8192,
            max_run_tokens=65536,
            reasoning_effort="high" if collection == "C5" else "low",
            capture_reasoning=True,
            budget_microusd=cap[m],
        )
        for i, m in enumerate(models, 1)
    ]
    slots = len(cases) * repeat * len(models)
    raw = dict(
        schema=SCHEMA,
        id=f"v3-{collection.lower()}-{doc['sandbox'].lower()}-{doc['variant']}",
        mode="live",
        program=doc,
        payment_rule=POOL_PAYMENT_RULE,
        capture_version=CAPTURE_V2,
        terms_sha256=POOL_TERMS_HASH,
        split_sha256=programs.sandbox(doc["sandbox"])["partition"],
        arms=arms,
        cases=cases,
        max_slots=slots,
        budget_microusd=len(cases)
        * sum(a["budget_microusd"] * a["repeats"] for a in arms),
        question="Which observed sales decisions satisfy the frozen native checks?",
        pass_mark="Complete workflow and exact native task evidence.",
        routes={
            m: REFERENCE_ROUTE if m == REFERENCE_MODEL else GENERATOR_ROUTES[m].tag
            for m in models
        },
    )
    if receipts:
        raw["provenance_sha256"] = _hash(receipts)
    return raw


def slot_keys(raw):
    for case in raw["cases"]:
        for arm in raw["arms"]:
            for repeat in range(1, arm["repeats"] + 1):
                yield (
                    case["scenario_id"],
                    case.get("decision_sha256", case.get("source_sha256")),
                    arm["model"],
                    arm["role"],
                    arm["reasoning_effort"],
                    repeat,
                )


def publish(packs, prompt, directory, inputs):
    """Fail before publishing a changed purchase of a slot. Retries reuse the original immutable ID."""
    from scripts.prepare_collection_experiment import (
        lock_pool_directory,
        store_private_json,
    )

    staged = {}
    with lock_pool_directory(directory):
        occupied = {}
        for path in sorted(directory.glob("pool-*.json")):
            old = json.loads(read_file(path))
            for slot in slot_keys(old):
                occupied.setdefault(slot, path.stem)
        for raw in packs:
            spec = load_experiment(raw, prompt)
            bid = spec.batch_id
            overlap = {
                occupied[s]
                for s in slot_keys(raw)
                if s in occupied and occupied[s] != bid
            }
            if overlap:
                raise ValueError(
                    "an equivalent funded slot already exists in "
                    + ",".join(sorted(overlap))
                )
            for s in slot_keys(raw):
                occupied[s] = bid
            staged[bid] = spec
        required = set()
        for spec in staged.values():
            required.add(spec.raw["split_sha256"])
            required.update(
                c["source_sha256"] for c in spec.raw["cases"] if "source_sha256" in c
            )
            if spec.raw.get("provenance_sha256"):
                required.add(spec.raw["provenance_sha256"])
            if spec.raw["program"].get("selection_sha256"):
                required.add(spec.raw["program"]["selection_sha256"])
        for digest in sorted(required):
            path = inputs[0] / (digest[7:] + ".json")
            if digest in inputs[1]:
                if _hash(inputs[1][digest]) != digest:
                    raise ValueError("input does not match its address")
                store_private_json(path, inputs[1][digest])
            elif _hash(json.loads(read_file(path))) != digest:
                raise ValueError("existing experiment input changed")
        for bid, spec in staged.items():
            store_private_json(directory / (bid + ".json"), spec.raw)
    return [
        {
            "batch": bid,
            "slots": len(s.slots),
            "program": s.raw["program"],
            "max_microusd": s.raw["budget_microusd"],
        }
        for bid, s in staged.items()
    ]


def compile_program(args):
    from scripts.run_lab_arena_service import load_scoped_environment
    from lab_arena.collection_experiment import verify_provenance

    if args.environment_file:
        load_scoped_environment(args.environment_file)
    if (
        not 1 <= args.cap_microusd <= 10_000_000_000
        or args.start_pack < 0
        or (args.pack_count is not None and args.pack_count < 1)
    ):
        raise ValueError("invalid spending ceiling or pack range")
    cap = dict.fromkeys(GENERATOR_ROUTES, args.cap_microusd)
    overrides = set()
    for override in args.cap_for:
        model, sep, value = override.partition("=")
        if (
            model not in cap
            or not sep
            or not value.isdecimal()
            or not 1 <= int(value) <= 10_000_000_000
            or model in overrides
        ):
            raise ValueError("invalid or duplicate per-model spending cap")
        overrides.add(model)
        cap[model] = int(value)
    prompt = json.loads(read_file(args.prompt))
    objects = {}
    packs = []
    size = 25
    for family in args.family:
        entry = programs.sandbox(family)
        with workflow_sources(
            args.champion_repo,
            args.poet_repo,
            corrections=True,
            builder_revision=entry["builder"],
        ) as sources:
            split = programs.producer_partition(sources[1], family, programs.BAND)
            if _hash(split) != entry["partition"]:
                raise ValueError("registry differs from producer split")
            native = getattr(sources[1], "sandbox_v3", None)
            if native is None or any(
                native.tool_aliases(family, b) != entry["aliases"][v]
                for b, v in programs.VARIANTS.items()
            ):
                raise ValueError(
                    "registered tool schemas differ from the pinned producer"
                )
            objects[_hash(split)] = split
            if args.collection in ("C1", "C2", "C3"):
                if args.student_bundle or args.exports:
                    raise ValueError("fresh programs do not take source evidence")
                for band in ((2, 3) if args.collection == "C2" else (1,)):
                    ordinals = range(0, 200, 2 if args.collection == "C2" else 1)
                    doc = programs.program(
                        args.collection, family, programs.VARIANTS[band]
                    )
                    models = (
                        programs.TEACHERS
                        if args.collection == "C1"
                        else (programs.CLOSE, programs.FAR)
                    )
                    if args.collection == "C3":
                        programs.hosted_student_route()
                        models = (programs.STUDENT,)
                    cases = [
                        {"scenario_id": f"{family}-s{band*programs.BAND+n}"}
                        for n in ordinals
                    ]
                    # The first case is always a separate canary. Expanding the
                    # series keeps that exact manifest and never buys its slots twice.
                    packs.append(_pack(doc, cases[:1], models, cap))
                    for i in range(1, len(cases), size):
                        packs.append(_pack(doc, cases[i : i + size], models, cap))
                continue
            if args.collection == "C4" and args.student_bundle:
                receipt = json.loads(
                    read_file(args.student_bundle / "attestation.json")
                )
                if receipt["schema"] != "collection.poet-student-sources.v2":
                    raise ValueError("expected signed bulk student sources")
                candidates = []
                for ref in sorted(receipt["sources"]):
                    if not programs.SHA.fullmatch(ref):
                        raise ValueError("invalid source hash")
                    val = json.loads(
                        read_file(args.student_bundle / "inputs" / (ref[7:] + ".json"))
                    )
                    if _hash(val) != ref:
                        raise ValueError("student source bytes changed")
                    if programs.identity(val["scenario_id"])[0] != family:
                        continue
                    if programs.identity(val["scenario_id"])[2] in range(0, 200, 2):
                        candidates.append((ref, val))
                for variant in programs.VARIANTS.values():
                    distinct = {}
                    for ref, v in candidates:
                        if programs.identity(v["scenario_id"])[1] == variant:
                            decision = programs.source_case(v)["decision_sha256"]
                            distinct.setdefault(decision, (ref, v))
                    values = list(distinct.values())
                    for i in range(0, len(values), size):
                        group = values[i : i + size]
                        cases = [programs.source_case(v) for _, v in group]
                        raw = _pack(
                            programs.program(
                                "C4", family, variant, source_kind="node_student"
                            ),
                            cases,
                            (programs.CLOSE, programs.FAR),
                            cap,
                            receipts=receipt,
                        )
                        spec = load_experiment(raw, prompt)
                        verify_provenance(spec, receipt, dict(group))
                        from lab_arena.collection_checkpoint import (
                            validate_student_source,
                        )

                        for ref, v in group:
                            validate_student_source(*sources, v)
                        objects.update(group)
                        objects[_hash(receipt)] = receipt
                        packs.append(raw)
                continue
            raise ValueError(
                "funded C4 and C5 are compiled by reconcile-v3 from an opted-in C1 pack"
            )
    if getattr(args, "followups", False):
        if args.collection != "C1":
            raise ValueError("only a C1 program can opt into follow-ups")
        for raw in packs:
            raw["followups"] = ["C4", "C5"]
    kept = packs[
        args.start_pack : (
            args.start_pack + args.pack_count if args.pack_count is not None else None
        )
    ]
    return publish(kept, prompt, args.directory, (args.inputs, objects))


def cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", choices=("C1", "C2", "C3", "C4"))
    parser.add_argument("--family", action="append", required=True)
    for field in ("prompt", "directory", "inputs", "champion-repo", "poet-repo"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--cap-microusd", type=int, required=True)
    parser.add_argument(
        "--cap-for", action="append", default=[], metavar="MODEL=MICROUSD"
    )
    parser.add_argument("--start-pack", type=int, default=0)
    parser.add_argument("--pack-count", type=int)
    parser.add_argument("--followups", action="store_true")
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--student-bundle", type=Path)
    parser.add_argument("--environment-file", type=Path)
    args = parser.parse_args(argv)
    if len(set(args.family)) != len(args.family):
        raise ValueError("duplicate sandbox")
    result = compile_program(args)
    print(json.dumps({"batches": result, "slots": sum(b["slots"] for b in result)}))
    return 0


def stage_followups(
    *, directory, inputs, exports, prompt, database, objects, max_cases=1
):
    """Reconcile a bounded number of newly released C1 groups. This never executes a model."""
    from lab_arena.collection_pool_source import funded_source, NoCorrectionState
    from scripts.prepare_collection_experiment import store_private_json

    if type(max_cases) is not int or not 1 <= max_cases <= 100:
        raise ValueError("invalid follow-up scan bound")
    digest = json.loads(read_file(exports / "current.json"))["snapshot"]
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid release index")
    frozen = read_file(exports / "snapshots" / (digest + ".json"), maximum=128 << 20)
    if sha256(frozen).hexdigest() != digest:
        raise ValueError("release index changed")
    roster = json.loads(frozen)["offers"]
    manifests = {}
    for f in directory.glob("pool-*.json"):
        value = json.loads(read_file(f))
        if value.get("program"):
            manifests[f.stem] = value
    prior_corrections = defaultdict(set)
    prior_maps = defaultdict(set)
    for value in manifests.values():
        p = value["program"]
        if p["collection"] == "C4":
            for c in value["cases"]:
                prior_corrections[c["scenario_id"]].add(c["decision_sha256"])
        if p["collection"] == "C5":
            prior_maps[p["sandbox"]].update(c["scenario_id"] for c in value["cases"])
    result = []
    for batch, raw in sorted(manifests.items()):
        if raw.get("followups") != ["C4", "C5"] or raw["program"]["collection"] != "C1":
            continue
        spec = load_experiment(raw, prompt)
        if spec.batch_id != batch:
            raise ValueError("baseline source identity changed")
        caps = {a["model"]: a["budget_microusd"] for a in raw["arms"]}
        for sid, g in sorted(roster.get(batch, {}).items()):
            if not g.get("file"):
                continue
            mark = (
                inputs
                / "program-followups"
                / batch
                / (g["dataset_sha256"].removeprefix("sha256:") + ".json")
            )
            if mark.exists():
                continue
            if len(result) >= max_cases:
                return result
            family, var, ordn = programs.identity(sid)
            if var != "base":
                raise ValueError("unexpected calibration variant")
            filename = g["dataset_sha256"].removeprefix("sha256:") + ".zip"
            if (
                not re.fullmatch(r"[0-9a-f]{64}\.zip", filename)
                or g["file"] != filename
            ):
                raise ValueError("invalid baseline archive")
            path = exports / batch / filename
            if (
                sha256(read_file(path, maximum=256 << 20)).hexdigest()
                != g["archive_sha256"]
            ):
                raise ValueError("changed baseline archive")
            observations = []
            packs = []
            source_objects = {}
            with ZipFile(path) as archive:
                for rid in sorted(g["released"]):
                    data = archive.read("trajectories/" + rid + ".jsonl")
                    row = json.loads(data)
                    if (
                        not row["verified_complete"]
                        or type(row["verified_task_pass"]) is not bool
                    ):
                        continue
                    if (
                        row["scenario_id"] != sid
                        or row["run_id"] != rid
                        or row["experiment"]["program"] != raw["program"]
                        or row["experiment"]["role"] != "expert"
                        or row.get("capture_hold")
                    ):
                        raise ValueError("invalid baseline evidence")
                    observations.append(
                        {
                            "run_id": rid,
                            "batch": batch,
                            "model": row["model"],
                            "repeat": row["experiment"]["repeat"],
                            "pass": row["verified_task_pass"],
                            "row_sha256": "sha256:" + sha256(data).hexdigest(),
                        }
                    )
                    if (
                        row["verified_task_pass"]
                        or ordn not in range(0, 200, 2)
                        or len(prior_corrections[sid]) >= 7
                    ):
                        continue
                    try:
                        source = funded_source(
                            database,
                            objects,
                            batch,
                            rid,
                            prompt,
                            classify_grounding=True,
                        )
                    except NoCorrectionState:
                        continue
                    if not (
                        source["rejected_fatal"]
                        or source["failure"].get("unrepaired_grounding")
                    ):
                        continue
                    case = programs.source_case(source)
                    if case["decision_sha256"] in prior_corrections[sid]:
                        continue
                    teachers = tuple(
                        m
                        for m in (
                            programs.CLOSE,
                            programs.FAR,
                            "nvidia/nemotron-3.5-super-vl",
                        )
                        if m != source["student_model"]
                    )[:2]
                    packs.append(
                        _pack(
                            programs.program(
                                "C4", family, var, source_kind="funded_failure"
                            ),
                            [case],
                            teachers,
                            caps,
                        )
                    )
                    source_objects[case["source_sha256"]] = source
                    prior_corrections[sid].add(case["decision_sha256"])
            draws = {(r["model"], r["repeat"]) for r in observations}
            expected = {(m, i) for m in programs.TEACHERS for i in range(1, 5)}
            if (
                sid not in prior_maps[family]
                and len(prior_maps[family]) < 50
                and len(observations) == len(expected)
                and draws == expected
                and 0.25 <= sum(r["pass"] for r in observations) / len(expected) <= 0.75
            ):
                proof = {
                    "schema": "collection.pass-rate-selection.v1",
                    "snapshot": digest,
                    "scenario_id": sid,
                    "observations": observations,
                }
                ref = _hash(proof)
                source_objects[ref] = proof
                packs.append(
                    _pack(
                        programs.program("C5", family, var, selection=ref),
                        [{"scenario_id": sid}],
                        programs.TEACHERS,
                        caps,
                    )
                )
                prior_maps[family].add(sid)
            published = publish(packs, prompt, directory, (inputs, source_objects))
            store_private_json(
                mark,
                {
                    "batch": batch,
                    "scenario": sid,
                    "dataset_sha256": g["dataset_sha256"],
                },
            )
            result.append(
                {
                    "baseline": batch,
                    "scenario": sid,
                    "staged": published,
                    "observations": len(observations),
                }
            )
    return result


def reconcile_cli(argv):
    from scripts.run_lab_arena_service import load_scoped_environment
    from scripts.run_locked_collection import evidence_stores
    from lab_arena.collection_intake import register_signatures

    parser = argparse.ArgumentParser(description=stage_followups.__doc__)
    for k in ("directory", "inputs", "exports", "prompt", "environment-file"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=1)
    a = parser.parse_args(argv)
    load_scoped_environment(a.environment_file)
    register_signatures()
    db, objects = evidence_stores()
    result = stage_followups(
        directory=a.directory,
        inputs=a.inputs,
        exports=a.exports,
        prompt=json.loads(read_file(a.prompt)),
        database=db,
        objects=objects,
        max_cases=a.max_cases,
    )
    print(json.dumps({"reviewed": result}))
    return 0
