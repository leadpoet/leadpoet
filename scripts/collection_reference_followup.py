"""Advance from one frozen successor to one full pilot reference after funded evidence, without scheduling a release."""

from __future__ import annotations

from collections import defaultdict
import json

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import (
    MODELS,
    TASK_SCOPE,
    SYNTHETIC_TRAINING_TERMS_HASH,
)
from lab_arena.collection_quality import verify_split
from lab_arena.collection_route_witness import verified_witness
from lab_arena.collection_trajectories import verified_grants
from lab_arena.collection_workflow import comparison_revisions
from scripts.collection_batch import read_artifact
from scripts.collection_review import read_file


def finished_slots(rows, first, count):
    """Every genuine successor slot has ended. Wait for settlements and affordable replacements."""
    grouped = defaultdict(list)
    for row in rows:
        if (
            type(row["slot"]) is not int
            or type(row["attempt"]) is not int
            or row["status"] not in {"accepted", "rejected", "failed"}
            or row["unresolved"]
        ):
            return False
        grouped[row["slot"]].append(row)
    if set(grouped) != set(range(first, count)):
        return False
    for group in grouped.values():
        attempts = sorted(group, key=lambda j: j["attempt"])
        if [r["attempt"] for r in attempts] not in ([1], [1, 2]):
            return False
        if len(attempts) == 2 and attempts[0]["status"] != "failed":
            return False
        last = attempts[-1]
        if (
            last["status"] == "failed"
            and len(attempts) == 1
            and last["artifact_sha256"] is None
            and last["miner_cost_microusd"] < last["budget_microusd"]
        ):
            return False
    return True


def reference_for_poll(raw, prompt, configuration, database, objects):
    from scripts.run_locked_collection import load_batch
    from scripts.run_locked_collection import assignment
    from lab_arena import collection_judge_review as reviews

    control = json.loads(read_file(configuration))
    if (
        set(control)
        != {
            "schema",
            "after_batch_id",
            "next_batch_id",
            "next_batch_file",
            "contribution_cap_microusd",
        }
        or control["schema"] != "collection.one-reference-followup.v1"
    ):
        raise ValueError("invalid private reference progression setting")
    if control["next_batch_file"] != "complete-reference.json":
        raise ValueError("follow-up must use the co-located immutable manifest")
    next_raw = json.loads(read_file(configuration.parent / control["next_batch_file"]))
    current = load_batch(raw, prompt)
    future = load_batch(next_raw, prompt)
    successor = raw.get("reference_successor") or {}
    capacity = sum(next_raw["budgets_microusd"].values()) * len(
        next_raw["scenario_ids"]
    )
    if (
        not current.batch_id.startswith("rs-")
        or not future.batch_id.startswith("rq-")
        or not current.pilot
        or not future.pilot
        or current.batch_id != control["after_batch_id"]
        or future.batch_id != control["next_batch_id"]
        or not successor
        or next_raw.get("route_witness_sha256") != successor["owner_witness_sha256"]
        or not next_raw.get("terms_sha256")
        == raw.get("terms_sha256")
        == SYNTHETIC_TRAINING_TERMS_HASH
        or len(next_raw["scenario_ids"]) != 10
        or next_raw["control_slots"]
        or next_raw["drift_slots"]
        or next_raw.get("control_artifacts")
        or type(control["contribution_cap_microusd"]) is not int
        or control["contribution_cap_microusd"] != capacity
        or next_raw.get("reference_successor")
        or comparison_revisions(raw) != comparison_revisions(next_raw)
        or any(
            raw.get(k) != next_raw.get(k)
            for k in (
                "routes",
                "budgets_microusd",
                "split_manifest",
                "claim_protocol",
                "capture_version",
                "thresholds",
            )
        )
    ):
        raise ValueError(
            "reference progression settings differ from their frozen cohorts"
        )
    # Ordinary polling never advances the qualification draft. The active-file choice and
    # correction priority stay with the operator; job, owner, cap and spending checks run again on claim.
    rows = database.rpc("collection_pilot_costs_v1", {"p_batch": current.batch_id})
    if not finished_slots(
        rows, successor["next_slot"], len(raw["scenario_ids"]) * len(MODELS)
    ):
        return raw
    grants = verified_grants(
        database,
        current.batch_id,
        rows,
        mode="pilot",
        scope=TASK_SCOPE,
        digest=raw["terms_sha256"],
    )
    _, builder = comparison_revisions(raw)
    for job in sorted(rows, key=lambda j: (j["slot"], j["attempt"])):
        rid = job["job_id"]
        if (
            job["status"] != "accepted"
            or not job["artifact_sha256"]
            or not job["generator_calls"]
            or not job["miner_cost_microusd"]
            or job["model"] != MODELS[job["slot"] % len(MODELS)]
            or job["scenario_id"] != raw["scenario_ids"][job["slot"] // len(MODELS)]
        ):
            continue

        def read(kind, ref):
            return read_artifact(
                objects, current.batch_id, rid, kind, ref, 128 * 1024 * 1024
            )

        if not (
            0
            < job["miner_cost_microusd"]
            <= job["budget_microusd"]
            <= raw["budgets_microusd"][job["model"]]
        ):
            raise ValueError("funded reference cost exceeds its slot cap")
        checked = reviews.registered_bundle(
            current,
            [assignment(job)],
            planted={},
            drift={},
            jobs=[dict(job, mode="pilot")],
            objects=objects,
            expected_builder=builder,
            capture_version=raw["capture_version"],
        )
        (run,) = checked["runs"]
        packet = run["packet"]
        assessment = run["assessment"]
        verify_split(packet["split"], raw["split_manifest"])
        if not (
            all(
                packet["checks"].get(k) is True
                for k in (
                    "answered",
                    "shape",
                    "code_grade",
                    "no_hard_failure",
                    "privacy",
                    "training_split",
                )
            )
            and judge.claim_verdict(packet) in ("passed", "not_applicable")
            and judge.training_selection(assessment)
        ):
            continue
        manifest = read("manifest", {"sha256": job["artifact_sha256"]})
        generator = read("generator", manifest["generator"])
        if grants[rid]["generator_calls"] != sum(
            e["kind"] == "reply" for e in generator["evidence"]["events"]
        ):
            raise ValueError("funded call count differs from completed reference")
        verified_witness(next_raw, objects)
        return next_raw
    return raw
