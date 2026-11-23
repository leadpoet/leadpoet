"""Private judge controls are independent of the current job's generated output."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pytest
from lab_arena import (
    collection_controls as controls,
    collection_judge as j,
    collection_judge_review as review,
)
from lab_arena.collection_workflow import store_artifact
from lab_arena.service import LocalObjectStore
from tests.lab_arena.collection_judge_test import POLICY, PROMPT, Caller, packet


def source_job(objects):
    source = replace(POLICY, batch_id="reference")
    rid = "measured-original"
    p = packet(rid)
    c = Caller()
    a = j.judge_run(
        source,
        {"run_id": rid, "primary": "fable", "overlap": True},
        p,
        c,
        prompt=PROMPT,
    )
    manifest = {
        "job_id": rid,
        "batch_id": source.batch_id,
        "scenario_id": p["scenario_id"],
        "judge_policy": asdict(source),
        "packet": store_artifact(objects, source.batch_id, rid, "judge-input", p),
        "assessment": store_artifact(objects, source.batch_id, rid, "judge-output", a),
    }
    m = store_artifact(objects, source.batch_id, rid, "manifest", manifest)
    return p, a, manifest, m["sha256"]


def test_controls_bind_to_held_case_and_cannot_leak_labels_to_judge(tmp_path):
    objects = LocalObjectStore(tmp_path)
    p, a, m, sh = source_job(objects)
    gold = controls.drift_case(p, a, m, source_manifest_sha256=sh)
    raw = {
        "scenario_ids": [p["scenario_id"]],
        "control_slots": {},
        "drift_slots": {"2": gold["expected"]},
        "control_artifacts": {"2": j.digest(gold)},
    }
    pack = controls.ControlPack(
        raw, {"schema": controls.PACK_SCHEMA, "cases": {j.digest(gold): gold}}
    )
    assert pack.bind({"slot": 0}) is None
    rid = "batch-anchor"
    job = {
        "job_id": rid,
        "batch_id": POLICY.batch_id,
        "model": "open-model",
        "miner_hotkey": "fund",
        "harness_commit": "fixed",
        "slot": 2,
        "scenario_id": p["scenario_id"],
    }
    bound = pack.bind(job)
    assert bound["packet"]["run_id"] == rid
    c = Caller()
    assignment = {
        "run_id": rid,
        "generator": "open-model",
        "miner": "fund",
        "harness": "fixed",
    }
    sched = j.assign_judges(POLICY, [assignment])[rid]
    a2 = j.judge_run(POLICY, sched, bound["packet"], c, prompt=PROMPT)
    assert all(
        x not in json.dumps(c.requests)
        for x in ["measured-original", "provenance", "source_manifest_sha256"]
    )
    m2 = {
        "job_id": rid,
        "batch_id": POLICY.batch_id,
        "model": "open-model",
        "harness_commit": "fixed",
        "scenario_id": p["scenario_id"],
        "judge_policy": asdict(POLICY),
        "packet": store_artifact(
            objects, POLICY.batch_id, rid, "judge-input", bound["packet"]
        ),
        "assessment": store_artifact(objects, POLICY.batch_id, rid, "judge-output", a2),
        "control": store_artifact(objects, POLICY.batch_id, rid, "control-case", gold),
    }
    job["artifact_sha256"] = store_artifact(
        objects, POLICY.batch_id, rid, "manifest", m2
    )["sha256"]
    kwargs = {
        "jobs": [job],
        "objects": objects,
        "planted": {},
        "drift": {rid: gold["expected"]},
    }
    bundle = review.registered_bundle(POLICY, [assignment], **kwargs)
    assert bundle["runs"][0]["control_sha256"] == j.digest(gold)
    with pytest.raises(ValueError, match="unexpected control"):
        review.registered_bundle(POLICY, [assignment], **dict(kwargs, drift={}))
    corrupt = deepcopy(gold)
    corrupt["expected"]["astra"]["safety"] = 2
    m2["control"] = store_artifact(
        objects, POLICY.batch_id, rid, "control-case", corrupt
    )
    job["artifact_sha256"] = store_artifact(
        objects, POLICY.batch_id, rid, "manifest", m2
    )["sha256"]
    with pytest.raises(ValueError, match="recorded baseline"):
        review.registered_bundle(
            POLICY, [assignment], **dict(kwargs, drift={rid: corrupt["expected"]})
        )


def test_plants_require_known_labels_and_reject_relabeling_or_reuse():
    p = packet("example")
    case = controls.planted_case(
        p,
        expected=False,
        reviewer="fixture",
        rationale="contradicts known invoice in synthetic fixture",
        source_sha256="sha256:" + j.digest("fixture-source"),
    )
    h = j.digest(case)
    raw = {
        "scenario_ids": [p["scenario_id"]],
        "control_slots": {"0": False},
        "drift_slots": {},
        "control_artifacts": {"0": h},
    }
    pack = {"schema": controls.PACK_SCHEMA, "cases": {h: case}}
    c = controls.ControlPack(raw, pack)
    assert (
        c.bind({"slot": 0, "scenario_id": p["scenario_id"], "job_id": "j"})["kind"]
        == "planted"
    )
    wrong = deepcopy(raw)
    wrong["control_slots"]["0"] = True
    with pytest.raises(ValueError, match="label"):
        controls.ControlPack(wrong, pack)
    dup = deepcopy(raw)
    dup["control_artifacts"]["1"] = h
    dup["control_slots"]["1"] = False
    with pytest.raises(ValueError, match="duplicate"):
        controls.ControlPack(dup, pack)
    with pytest.raises(ValueError, match="differs"):
        controls.ControlPack(dict(raw, control_artifacts={}), pack)
    unavailable = deepcopy(p)
    unavailable["checks"]["shape"] = False
    with pytest.raises(ValueError, match="deterministic"):
        controls.planted_case(
            unavailable,
            expected=False,
            reviewer="a",
            rationale="invalid",
            source_sha256="sha256:" + j.digest(1),
        )
