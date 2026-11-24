"""Shape immutable collection evidence into private audit or reviewed correction datasets."""

from __future__ import annotations

import base64
from collections import Counter
from hashlib import sha256
import json
import math
import os
import re
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lab_arena import collection_judge as judge
from lab_arena import collection_judge_review as review
from lab_arena.collection_experiment import (
    NAME,
    SCHEMA,
    SHA,
    _hash,
    load_experiment,
    source_revisions,
)
from lab_arena.collection_quality import verify_split
from lab_arena.collection_workflow import store_artifact

CORRECTIONS = "collection.correction-pairs.v1"
AUDIT = "collection.experiment-private-audit.v1"
FEEDBACK = "collection.poet-evaluation.v1"
LIMIT = 128 * 1024 * 1024


def verified_panel(
    database, objects, spec, raw, prompt, receipt, approval, reviewer_key, approver_key
):
    from scripts.collection_batch import source_bundle

    bundle = source_bundle(raw, prompt, database, objects)
    signed = review.verify_qualification(
        bundle, receipt, reviewer_key, approval, approver_key
    )
    if signed["prompt_sha256"] != spec.policy.prompt_sha256:
        raise ValueError("experiment and qualified panel use different policies")
    return signed


def _pair(row, packet, student):
    """Only the separately verified expert's action is a positive target."""
    link = row.get("correction")
    required_code = {
        "answered",
        "shape",
        "code_grade",
        "no_hard_failure",
        "privacy",
        "training_split",
    }
    if (
        not link
        or not required_code <= set(packet["checks"])
        or not all(v is True for v in packet["checks"].values())
        or packet.get("projection_error")
    ):
        return None
    if judge.claim_verdict(packet) not in {"passed", "not_applicable"}:
        return None
    checks = link.get("correction_step_checks", {})
    required = {
        "same_pre_state",
        "different_action",
        "student_action_fatal",
        "expert_step_recoverable",
    }
    if set(checks) != required or not all(v is True for v in checks.values()):
        return None
    if (
        link["student_model"] != student["student_model"]
        or link["source_sha256"] != student["source_sha256"]
        or link["rejected_fatal"] is not True
        or link["rejected"] != student["rejected"]
        or link["checkpoint_sha256"] != judge.digest(student["checkpoint"])
    ):
        raise ValueError("correction does not bind original student checkpoint")
    at = link["prefix_messages"]
    msgs = row["messages"]
    if (
        type(at) is not int
        or not 0 <= at < len(msgs)
        or msgs[at]["role"] != "assistant"
    ):
        raise ValueError("correction target has no matching prefix")
    if msgs[at] == link["rejected"]:
        return None
    # The archive retains private world state and grade details. Models may train
    # only on information visible to the runner before acting.
    return {
        "context": msgs[:at],
        "tools": row["tools"],
        "rejected": link["rejected"],
        "chosen": msgs[at],
        "step_checks": checks,
        "branch_turn": link["branch_turn"],
        "checkpoint_sha256": link["checkpoint_sha256"],
        "student_model": link["student_model"],
        "expert_model": row["teacher"],
        "target_is_final_turn": at == len(msgs) - 1,
        "final_turn_retained": packet["checks"]["answered"] is True
        and msgs[-1]["role"] == "assistant",
        "expert_final_outcome": {
            "final_turn": msgs[-1],
            "checks": packet["checks"],
            "claim_verdict": judge.claim_verdict(packet),
        },
    }


def write_dataset(spec, database, objects, dest, *, qualification=None):
    from scripts.collection_batch import read_artifact
    from lab_arena.collection_intake import correction_terms_hash, register_signatures

    register_signatures()

    batch = spec.batch_id
    trained = qualification is not None
    export = CORRECTIONS if trained else AUDIT
    if trained:
        if spec.raw["terms_sha256"] != correction_terms_hash(spec.raw["mode"]):
            raise ValueError("release requires active approved correction consent")
        # Qualification documents are reconstructed from authenticated DB + original immutable evidence.
        receipt = verified_panel(database, objects, spec, **qualification)
    costs_rpc = (
        "collection_experiment_costs_v1"
        if spec.raw["mode"] == "live"
        else "collection_pilot_costs_v1"
    )
    rows = database.rpc(costs_rpc, {"p_batch": batch})
    if not isinstance(rows, list) or len(rows) > 2 * len(spec.slots) or not rows:
        raise ValueError("missing registered experiment runs")
    if trained and (
        any(j["status"] == "running" or j["unresolved"] for j in rows)
        or {j["slot"] for j in rows} != set(range(len(spec.slots)))
    ):
        raise ValueError(
            "correction batch has missing runs or unconfirmed miner charges"
        )
    stats = Counter()
    class_costs = {}
    training_rows = sha256()
    source_ids = []
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "dataset.zip"
        with ZipFile(stage, "x", compression=ZIP_DEFLATED) as z:
            for job in sorted(rows, key=lambda j: (j["slot"], j["attempt"])):
                i = job["slot"]
                if type(i) is not int or not 0 <= i < len(spec.slots):
                    raise ValueError("unregistered experiment slot")
                case, arm, repeat = spec.slots[i]
                rid = job["job_id"]
                if (
                    job["batch_id"] != batch
                    or job["model"] != arm["model"]
                    or job["scenario_id"] != case["scenario_id"]
                    or job["harness_commit"] != source_revisions()[0]
                ):
                    raise ValueError("job differs from frozen experiment")
                stats["attempts"] += 1
                stats["miner_cost_microusd"] += job["miner_cost_microusd"]
                stats["unknown_miner_bills"] += job["unresolved"]
                source_ids.append([rid, job.get("artifact_sha256")])
                if not trained:
                    z.writestr(f"{rid}/job.json", judge._json(job))
                event_hashes = []
                if job.get("artifact_sha256") and job["status"] != "failed":
                    manifest = read_artifact(
                        objects,
                        batch,
                        rid,
                        "manifest",
                        {"sha256": job["artifact_sha256"]},
                        LIMIT,
                    )
                    event_hashes = manifest["events"]
                else:
                    ref = objects.get_optional_bounded(
                        f"collection/{batch}/{rid}/interruption.json", 8192
                    )
                    if ref:
                        failure = read_artifact(
                            objects, batch, rid, "interruption", json.loads(ref), LIMIT
                        )
                        if (
                            job.get("artifact_sha256")
                            and _hash(failure) != job["artifact_sha256"]
                        ):
                            raise ValueError(
                                "failed attempt receipt differs from stored interruption"
                            )
                        event_hashes = failure["events"]
                        if not trained:
                            z.writestr(f"{rid}/interruption.json", judge._json(failure))
                # A hard stop may leave a verified prefix, but no terminal event index.
                incomplete = not job.get("artifact_sha256") and not event_hashes
                limit = 4096 if incomplete else len(event_hashes)
                for idx in range(limit):
                    key = f"collection/{batch}/{rid}/events/{idx:05d}.json"
                    event = objects.get_optional_bounded(key, LIMIT)
                    if incomplete and event is None:
                        break
                    if (
                        event is None
                        or json.loads(event)["index"] != idx
                        or (
                            not incomplete
                            and "sha256:" + sha256(event).hexdigest()
                            != event_hashes[idx]
                        )
                    ):
                        raise ValueError("run journal is missing or changed")
                    if not trained:
                        z.writestr(f"{rid}/events/{idx:05d}.json", event)
                if job["status"] == "failed" or not job.get("artifact_sha256"):
                    if job["status"] not in {"failed", "running"}:
                        raise ValueError("completed experiment missing archive")
                    stats["partial_or_failed"] += 1
                    if not trained:
                        for kind in (
                            "student-source",
                            "settings",
                            "generator",
                            "generation-receipts",
                            "pre-claim-input",
                            "judge-input",
                        ):
                            link = objects.get_optional_bounded(
                                f"collection/{batch}/{rid}/saved/{kind}.json", 8192
                            )
                            if link:
                                doc = read_artifact(
                                    objects, batch, rid, kind, json.loads(link), LIMIT
                                )
                                z.writestr(f"{rid}/{kind}.json", judge._json(doc))
                    continue

                def read(kind, ref):
                    return read_artifact(objects, batch, rid, kind, ref, LIMIT)

                if (
                    manifest["schema"] != SCHEMA
                    or manifest["job_id"] != rid
                    or manifest["batch_id"] != batch
                    or manifest["model"] != job["model"]
                    or manifest["scenario_id"] != job["scenario_id"]
                    or (manifest["harness_commit"], manifest["builder_commit"])
                    != source_revisions()
                    or manifest["experiment"]
                    != {
                        "id": spec.raw["id"],
                        "arm": arm["id"],
                        "settings_sha256": spec.settings_hash,
                        "repeat": repeat,
                        "role": arm["role"],
                        "capture_reasoning": arm["capture_reasoning"],
                        "reasoning_effort": arm["reasoning_effort"],
                    }
                ):
                    raise ValueError("experiment manifest identity mismatch")
                row = read("generator", manifest["generator"])
                packet = read("judge-input", manifest["packet"])
                student = read("student-source", manifest["student_source"])
                split = read("split", manifest["split"])
                calls = read("generation-receipts", manifest["calls"])
                settings = read("settings", manifest["settings"])
                if (
                    row["run"] != rid
                    or row["teacher"] != job["model"]
                    or packet["run_id"] != rid
                    or packet["scenario_id"] != case["scenario_id"]
                    or _hash(student) != case["source_sha256"]
                    or _hash(split) != spec.raw["split_sha256"]
                    or settings != spec.raw
                ):
                    raise ValueError("experiment evidence differs from assignment")
                if trained:
                    from lab_arena.collection_experiment import verify_provenance

                    provenance = read(
                        "producer-attestation", manifest["producer_attestation"]
                    )
                    verify_provenance(
                        spec, provenance, {case["source_sha256"]: student}
                    )
                observed = {}
                for unit, proof in packet.get("fresh_claim_proof", {}).items():
                    for order, labels in enumerate(proof["votes"]):
                        request_id = proof["request_ids"][order]
                        observed[request_id] = (
                            unit,
                            order,
                            proof["reader"],
                            proof["rubric_sha256"],
                            labels,
                        )
                seen = set()
                for ref in manifest["claim_votes"]:
                    # The run-owned path and exact content digest commit the original vote.
                    match = re.fullmatch(
                        r".*/claim-vote-(collection-[0-9a-f]{64}-1)-[0-9a-f]{64}\.json",
                        ref["key"],
                    )
                    if not match:
                        raise ValueError("invalid claim vote reference")
                    request_id = match[1]
                    vote = read("claim-vote-" + request_id, ref)
                    if (
                        vote["request_id"] != request_id
                        or vote["run_id"] != rid
                        or vote["batch_id"] != batch
                        or request_id in seen
                        or observed.get(request_id)
                        != (
                            vote["unit_sha256"],
                            vote["vote_index"],
                            vote["model"],
                            vote["rubric_sha256"],
                            vote["labels"],
                        )
                    ):
                        raise ValueError("claim vote identity mismatch")
                    seen.add(request_id)
                    if not trained:
                        z.writestr(f"{rid}/votes/{request_id}.json", judge._json(vote))
                if seen != set(observed):
                    raise ValueError("claim vote count differs from source")
                verify_split(manifest["split_stamp"], split)
                if row.get("split") != packet.get("split") or row.get("split") != {
                    k: manifest["split_stamp"][k]
                    for k in ("template_id", "rule_pool", "tool_variant")
                }:
                    raise ValueError(
                        "recorded split differs from verified student source"
                    )
                if not trained:
                    for n, v in {
                        "manifest": manifest,
                        "generator": row,
                        "packet": packet,
                        "source": student,
                        "receipts": calls,
                    }.items():
                        z.writestr(f"{rid}/{n}.json", judge._json(v))
                if (
                    job["status"] == "accepted"
                    and not job["unresolved"]
                    and job["generator_calls"] > 0
                    and job["miner_cost_microusd"] > 0
                ):
                    key = (
                        arm["role"],
                        case["scenario_id"].split("-s")[0],
                        job["model"],
                    )
                    group = class_costs.setdefault(key, [0, 0])
                    group[0] += 1
                    group[1] += job["miner_cost_microusd"]
                stats["finished_runs"] += 1
                stats["code_passed"] += all(packet["checks"].values())
                if (
                    trained
                    and job["status"] == "accepted"
                    and arm["role"] == "continuation"
                ):
                    pair = _pair(row, packet, student)
                    if pair is None:
                        continue
                    if not pair["final_turn_retained"]:
                        continue
                    pair.update(
                        run_id=rid,
                        scenario_id=case["scenario_id"],
                        arm=arm["id"],
                        settings_sha256=spec.settings_hash,
                        source_sha256=case["source_sha256"],
                        split=manifest["split_stamp"],
                    )
                    data = (judge._json(pair) + "\n").encode()
                    z.writestr(f"pairs/{rid}.jsonl", data)
                    training_rows.update(data)
                    stats["rows"] += 1
                    stats["endings_kept"] += int(pair["final_turn_retained"])
                    stats["targets_including_final_turn"] += int(
                        pair["target_is_final_turn"]
                    )
            if trained and not stats["rows"]:
                raise ValueError("no verified correction pairs")
            metadata = {
                "schema": export,
                "batch_id": batch,
                "settings_sha256": spec.settings_hash,
                "experiment": spec.raw["id"],
                "runs": source_ids,
                "split_sha256": spec.raw["split_sha256"],
                **dict(stats),
                "rows": stats["rows"],
                "endings_kept": stats["endings_kept"],
                "cost_per_kept_row_microusd": (
                    stats["miner_cost_microusd"] / stats["rows"]
                    if stats["rows"]
                    else None
                ),
            }
            metadata["price_observations"] = [
                {
                    "role": role,
                    "task_class": family,
                    "model": model,
                    "measured_runs": n,
                    "miner_cost_microusd": cost,
                    "proposed_fixed_microusd": (5 * cost + 4 * n - 1) // (4 * n),
                }
                for (role, family, model), (n, cost) in sorted(class_costs.items())
            ]
            if trained:
                metadata.update(
                    dataset_sha256="sha256:" + training_rows.hexdigest(),
                    qualification_sha256=_hash(receipt),
                )
            z.writestr("metadata.json", judge._json(metadata))
        os.chmod(stage, 0o600)
        if trained:
            store_artifact(objects, batch, "export", "split", split)
            store_artifact(objects, batch, "export", "release", metadata)
        os.link(stage, dest)
    return metadata


def record_feedback(metadata, result, objects, public_key_pem):
    """An identified Poet evaluator signs paired same-server held-out outcomes."""
    if metadata["schema"] != CORRECTIONS or not SHA.fullmatch(
        metadata["dataset_sha256"]
    ):
        raise ValueError("expected released correction dataset")
    from scripts.collection_batch import read_artifact

    saved = read_artifact(
        objects,
        metadata["batch_id"],
        "export",
        "release",
        {"sha256": _hash(metadata)},
        LIMIT,
    )
    if saved != metadata:
        raise ValueError("unregistered training export")
    body = {k: v for k, v in result.items() if k != "signature"}
    if set(body) != {
        "schema",
        "dataset_sha256",
        "run_id",
        "base_revision",
        "tuned_revision",
        "own_runs_revision",
        "server_sha256",
        "heldout_sha256",
        "split_sha256",
        "poet_revision",
        "tasks",
    }:
        raise ValueError("incomplete training evaluation")
    key = load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("expected pinned Poet evaluator key")
    key.verify(
        base64.b64decode(result["signature"], validate=True), judge._json(body).encode()
    )
    if (
        body["schema"] != FEEDBACK
        or body["dataset_sha256"] != metadata["dataset_sha256"]
        or not NAME.fullmatch(body["run_id"])
        or not all(
            SHA.fullmatch(body[k])
            for k in ("server_sha256", "heldout_sha256", "split_sha256")
        )
        or body["split_sha256"] != metadata["split_sha256"]
        or not all(
            isinstance(body[k], str) and body[k].strip() and len(body[k]) <= 256
            for k in (
                "base_revision",
                "tuned_revision",
                "own_runs_revision",
                "poet_revision",
            )
        )
        or len(
            {body["base_revision"], body["tuned_revision"], body["own_runs_revision"]}
        )
        != 3
    ):
        raise ValueError(
            "evaluation does not identify the dataset or paired model runs"
        )
    split = read_artifact(
        objects,
        metadata["batch_id"],
        "export",
        "split",
        {"sha256": metadata["split_sha256"]},
        LIMIT,
    )
    allowed = {
        k: {"sha256:" + sha256(v.encode()).hexdigest() for v in values}
        for k, values in split["heldout"].items()
    }
    samples = body["tasks"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= 100000:
        raise ValueError("empty or oversized paired evaluation")
    if any(
        set(x) != {"id_sha256", "base", "tuned", "own_runs", "partition_sha256"}
        or not isinstance(x["partition_sha256"], dict)
        or set(x["partition_sha256"]) != set(allowed)
        or any(x["partition_sha256"][k] not in ids for k, ids in allowed.items())
        or not SHA.fullmatch(x["id_sha256"])
        or any(type(x[k]) is not bool for k in ("base", "tuned", "own_runs"))
        for x in samples
    ):
        raise ValueError("invalid paired task outcomes")
    if len({x["id_sha256"] for x in samples}) != len(samples):
        raise ValueError("repeated heldout task")
    # Paired differences lie in [-1,1]. A conservative two-sided Hoeffding
    # interval is distribution free for independent held-out task draws.
    radius = math.sqrt(2 * math.log(40) / len(samples))
    comparisons = {}
    for baseline in ("base", "own_runs"):
        lift = sum(int(x["tuned"]) - int(x[baseline]) for x in samples) / len(samples)
        comparisons[baseline] = {
            "lift": lift,
            "lower_95": max(-1, lift - radius),
            "upper_95": min(1, lift + radius),
        }
    value = {
        "evaluation": result,
        "paired_tasks": len(samples),
        "comparisons": comparisons,
    }
    return store_artifact(
        objects, metadata["batch_id"], body["run_id"], "training-feedback", value
    )
