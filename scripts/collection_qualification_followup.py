"""Seal one operator-pinned P7 qualification and privately prepare a real blind sample."""

from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import tempfile

from lab_arena import (
    collection_controls as controls,
    collection_judge as judge,
    collection_judge_review as review,
)
from lab_arena.collection_intake import MODELS, TASK_SCOPE
from lab_arena.collection_trajectories import verified_grants
from lab_arena.collection_route_witness import verified_witness
from scripts.collection_batch import read_artifact, source_bundle
from scripts.collection_reference_followup import finished_slots
from scripts.collection_review import read_file, write_new, blind_files
from scripts.run_locked_collection import load_batch


def reviewed_plants(cases, decisions, bindings):
    if (
        set(decisions) != {"sample_sha256", "reviewer", "reviews"}
        or decisions["sample_sha256"] != judge.digest(cases)
        or not isinstance(decisions["reviewer"], str)
        or not decisions["reviewer"].strip()
        or set(cases) != set(bindings)
        or set(cases) != set(decisions["reviews"])
        or len(cases) != 2
    ):
        raise ValueError("incomplete reviewer identity or mismatched blind sample")
    complete = []
    for alias, content in cases.items():
        item = bindings[alias]
        source = item["source"]
        packet = item["packet"]
        projected, run = judge._packet(dict(packet, run_id="blind-projection"))
        if (
            content != {"scenario": projected, "run": run}
            or source["packet_sha256"] != judge.digest(packet)
            or item["source_sha256"] != "sha256:" + judge.digest(source)
            or judge.claim_verdict(packet) != "passed"
        ):
            raise ValueError("review content differs from original source")
        mark = decisions["reviews"][alias]
        if (
            set(mark) != {"accept", "rationale"}
            or type(mark["accept"]) is not bool
            or not isinstance(mark["rationale"], str)
            or not mark["rationale"].strip()
        ):
            raise ValueError("one or more genuine review decisions are incomplete")
        complete.append(
            controls.planted_case(
                packet,
                expected=mark["accept"],
                reviewer=decisions["reviewer"].strip(),
                rationale=mark["rationale"],
                source_sha256=item["source_sha256"],
                source_generator=source["source_generator"],
            )
        )
    if {c["expected"] for c in complete} != {True, False}:
        raise ValueError(
            "both answers remain valid human feedback; this pair does not furnish both required control classes"
        )
    return complete


def seal(pending, prompt, ref, plants, db, objects):
    raw = pending["manifest"]
    policy = load_batch(raw, prompt)
    scenarios = raw["scenario_ids"]
    if (
        pending["schema"] != "sn71.p7.qualification-draft.v1"
        or not policy.pilot
        or not policy.qualification
        or any(not scenario.startswith("P7-s") for scenario in scenarios)
        or any(raw[k] for k in ("control_artifacts", "control_slots", "drift_slots"))
    ):
        raise ValueError("expected unsealed P7 qualification settings")
    if (
        not isinstance(plants, list)
        or len(plants) != 2
        or {c["expected"] for c in plants} != {True, False}
        or any(c["kind"] != "planted" for c in plants)
    ):
        raise ValueError("both separately reviewed known-answer cases are required")
    panel = [policy.version(role) for role in policy.roles]
    available = {
        s: [
            i * len(MODELS) + k
            for k, model in enumerate(MODELS)
            if len(judge.independent_roles(policy, model)) == len(panel)
        ]
        for i, s in enumerate(scenarios)
    }
    case_map = {}
    packet_hashes = set()

    def allocate(case):
        controls.validate_case(case)
        controls.verify_control_family(case, panel)
        sid = case["packet"]["scenario_id"]
        packet_hash = judge.digest(case["packet"])
        if not available.get(sid) or packet_hash in packet_hashes:
            return False
        case_map[available[sid].pop(0)] = case
        packet_hashes.add(packet_hash)
        return True

    for case in plants:
        if not allocate(case):
            raise ValueError(
                "known-answer packets must be distinct and match scheduled cases"
            )
    eligible = []
    frozen = [
        "mode",
        "routes",
        "split_manifest",
        "capture_version",
        "claim_protocol",
        "budgets_microusd",
        "thresholds",
        "terms_sha256",
        "route_witness_sha256",
    ]
    if any(ref.get(k) != raw.get(k) for k in frozen) or any(
        ref[k] for k in ("drift_slots", "control_slots")
    ):
        raise ValueError("reference uses different settings or contains controls")
    baseline = load_batch(ref, prompt)
    if not baseline.batch_id.startswith("rq-") or baseline.batch_id == policy.batch_id:
        raise ValueError("qualification needs a distinct complete witnessed reference")
    verified_witness(ref, objects)
    costs = db.rpc("collection_pilot_costs_v1", {"p_batch": baseline.batch_id})
    verified_grants(
        db,
        baseline.batch_id,
        costs,
        mode="pilot",
        scope=TASK_SCOPE,
        digest=ref["terms_sha256"],
    )
    measured = {
        r["job_id"]
        for r in costs
        if r["status"] in ("accepted", "rejected")
        and r["generator_calls"] > 0
        and r["miner_cost_microusd"] > 0
    }
    bundle = source_bundle(ref, prompt, db, objects)
    review._read_bundle(bundle)
    seen_runs = set()
    for item in bundle["runs"]:
        assessment = item["assessment"]
        records = assessment["records"]
        packet = item["packet"]
        if (
            packet["run_id"] not in measured
            or not available.get(packet["scenario_id"])
            or assessment.get("verifier_unavailable")
            or {r["judge"] for r in records} != set(policy.roles)
            or any(r["cannot_judge"] for r in records)
        ):
            continue
        rid = packet["run_id"]
        if rid in seen_runs:
            raise ValueError("repeated reference judgment")
        seen_runs.add(rid)
        order = judge.digest([policy.seed, "drift-reference", baseline.batch_id, rid])
        eligible.append((order, baseline.batch_id, rid, item))

    for _, batch, rid, item in sorted(eligible, key=lambda row: row[0]):
        recorded_sha = item["manifest_sha256"]
        manifest = read_artifact(
            objects, batch, rid, "manifest", {"sha256": recorded_sha}, 1024 * 1024
        )
        if manifest.get("control") or not manifest.get("generator"):
            raise ValueError("a drift reference must be an original generation")
        case = controls.drift_case(
            item["packet"],
            item["assessment"],
            manifest,
            source_manifest_sha256=recorded_sha,
        )
        allocate(case)
        if len(case_map) == 32:
            break
    if len(case_map) != 32:
        raise ValueError(
            f"only {len(case_map)-2}/30 distinct measured paired anchors fit the frozen scenarios"
        )
    sealed = dict(raw)
    sealed["control_slots"] = {
        str(i): c["expected"] for i, c in case_map.items() if c["kind"] == "planted"
    }
    sealed["drift_slots"] = {
        str(i): c["expected"] for i, c in case_map.items() if c["kind"] == "drift"
    }
    sealed["control_artifacts"] = {str(i): judge.digest(c) for i, c in case_map.items()}
    pack = {
        "schema": controls.PACK_SCHEMA,
        "cases": {judge.digest(c): c for c in case_map.values()},
    }
    ordinary = [
        MODELS[i % len(MODELS)]
        for i in range(len(scenarios) * len(MODELS))
        if i not in case_map
    ]
    if len(ordinary) < 100 or set(ordinary) != set(MODELS):
        raise ValueError("cohort cannot cover 100 ordinary runs and six generators")
    controls.ControlPack(sealed, pack)
    return sealed, pack, load_batch(sealed, prompt).batch_id


def pinned_json(path, expected):
    data = read_file(path)
    if sha256(data).hexdigest() != expected:
        raise ValueError("frozen qualification source file changed")
    return json.loads(data)


def store_once(directory, files):
    """Publish a whole packet or nothing. Reconstruct and compare every existing document."""
    if not directory.exists():
        with tempfile.TemporaryDirectory(dir=directory.parent) as scratch:
            working = Path(scratch) / "new"
            working.mkdir(mode=0o700)
            for name, value in files.items():
                write_new(working / name, value)
            try:
                working.rename(directory)
            except OSError:
                if not directory.is_dir():
                    raise
    if directory.is_symlink() or set(p.name for p in directory.iterdir()) != set(files):
        raise ValueError("private qualification output directory changed")
    if any(
        json.loads(read_file(directory / name)) != value
        for name, value in files.items()
    ):
        raise ValueError("earlier frozen qualification evidence differs")


def qualification_for_poll(raw, prompt, directory, db, objects):
    config = directory / "qualification-handoff.json"
    if not config.exists():
        return raw, None
    cfg = json.loads(read_file(config))
    reference = load_batch(raw, prompt)
    if reference.batch_id != cfg["reference_id"]:
        return raw, None
    expected = {
        "schema",
        "reference_id",
        "draft_sha256",
        "draft_id",
        "reference_sha256",
        "cases_sha256",
        "bindings_sha256",
        "total_budget_microusd",
    }
    if (
        set(cfg) != expected
        or cfg["schema"] != "collection.p7-qualification-followup.v1"
    ):
        raise ValueError("unsupported qualification handoff")
    ref = pinned_json(directory / "complete-reference.json", cfg["reference_sha256"])
    pending = pinned_json(
        directory / "qualification-settings.pending.json", cfg["draft_sha256"]
    )
    draft = pending["manifest"]
    policy = load_batch(draft, prompt)
    if (
        raw != ref
        or not reference.batch_id.startswith("rq-")
        or not policy.batch_id.startswith("rq-")
        or cfg["draft_id"] != policy.batch_id
        or policy.batch_id == reference.batch_id
        or len(ref["scenario_ids"]) != 10
        or len(draft["scenario_ids"]) != 24
        or not set(ref["scenario_ids"]) <= set(draft["scenario_ids"])
        or type(cfg["total_budget_microusd"]) is not int
        or cfg["total_budget_microusd"] != 43_200_000
        or cfg["total_budget_microusd"] != 24 * sum(draft["budgets_microusd"].values())
        or not policy.qualification
        or not policy.pilot
    ):
        raise ValueError("draft differs from the authorized qualification scope")
    rows = db.rpc("collection_pilot_costs_v1", {"p_batch": reference.batch_id})
    returns = directory / "returned-plant-decisions.json"
    if not finished_slots(rows, 0, 60) or not returns.is_file():
        return raw, None
    cases = pinned_json(directory / "planted-cases.json", cfg["cases_sha256"])
    bindings = pinned_json(
        directory / "planted-source-bindings.json", cfg["bindings_sha256"]
    )
    plants = reviewed_plants(cases, json.loads(read_file(returns)), bindings)
    cohort, pack, bid = seal(pending, prompt, ref, plants, db, objects)
    if len(cohort["control_slots"]) != 2 or len(cohort["drift_slots"]) != 30:
        raise ValueError("incomplete independent review controls")
    # The output is content-addressed by the original sealed controls. Changing a
    # human decision after claims have begun must never open a second cohort.
    frozen = directory / "sealed-qualification"
    store_once(frozen, {"batch.json": cohort, "controls.json": pack})
    terminal = db.rpc("collection_pilot_costs_v1", {"p_batch": bid})
    if finished_slots(terminal, 0, 144):
        evidence = source_bundle(cohort, prompt, db, objects)
        _, jobs = review.prepare_review(evidence, for_qualification=True)
        if len(jobs) != 50:
            raise ValueError("wrong blind sample size")
        store_once(
            directory / "blind-ordinary-review",
            {
                "jobs.json": jobs,
                "marks.blank.json": blind_files(jobs, judge.digest(evidence)),
                "rubric.json": {
                    k: prompt[k]
                    for k in (
                        "rubric_version",
                        "rubric",
                        "examples",
                        "run_instructions",
                    )
                },
                "instructions.json": {
                    "score": "Use the frozen rubric. Mark every case independently on all five dimensions, from 0 to 3. Keep cases and results private.",
                    "identity": "Give the operator your public Ed25519 PEM through the private review channel. Independently confirm its SHA-256 fingerprint before returning the signed receipt. Keep your private key locally.",
                    "sign": "python -m scripts.collection_review sign --jobs jobs.json --marks completed.json --reviewer REVIEWER --private-key reviewer.pem --receipt review-receipt.json",
                    "return": "Return only your signed receipt privately. The operator verifies it against the separately pinned public key. A separate operator signs any qualification approval.",
                },
            },
        )
    return cohort, frozen / "controls.json"
