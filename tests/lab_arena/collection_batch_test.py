"""Replay a complete synthetic pilot and export only signed generator evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from zipfile import ZipFile

import pytest

from lab_arena import collection_judge as judge, collection_judge_review as review
from lab_arena.collection_calibration import compile_pilot
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import CHAMPION_REV, POET_REV, store_artifact
from lab_arena.service import LocalObjectStore
from scripts import collection_batch as release
from scripts.run_locked_collection import load_batch, assignment
from tests.lab_arena.collection_judge_test import PROMPT, Caller, fail_truth, packet
from tests.lab_arena.collection_judge_review_test import keys, complete_marks


class Database:
    def __init__(self):
        self.rows = []

    def rpc(self, name, parameters):
        if name in {"collection_batch_jobs_v1", "collection_pilot_costs_v1"}:
            return self.rows
        if name == "collection_pilot_fees_v1":
            assert len(parameters["p_controls"]) == 32
            return {
                "prices": {"C": dict.fromkeys(MODELS, 125)},
                "budgets": {"C": dict.fromkeys(MODELS, 1000000)},
                "job_count": 100,
                "measurement_sha256": "a" * 64,
            }
        raise AssertionError(name)


@pytest.fixture
def pilot(tmp_path):
    db = Database()
    objects = LocalObjectStore(tmp_path / "objects")
    raw = {
        "name": "test",
        "mode": "pilot",
        "seed": "test-seed",
        "routes": dict.fromkeys(
            [*MODELS, "anthropic/claude-fable-5.2", "openai/gpt-6.2-astra"], "test"
        ),
        "scenario_ids": [f"C01-s{2000+i}" for i in range(22)],
        "budgets_microusd": dict.fromkeys(MODELS, 1000000),
        "control_slots": {"130": True, "131": False},
        "drift_slots": {
            str(i): {v: dict.fromkeys(judge.PARTS, 3) for v in judge.JUDGES}
            for i in range(100, 130)
        },
    }
    policy = load_batch(raw, PROMPT)
    for i in range(132):
        rid = f"run-{i}"
        sid = raw["scenario_ids"][i // 6]
        model = MODELS[i % 6]
        job = dict(
            job_id=rid,
            batch_id=policy.batch_id,
            mode="pilot",
            slot=i,
            model=model,
            harness_commit=CHAMPION_REV,
            miner_hotkey=f"test-miner-{i}",
            scenario_id=sid,
            status="accepted",
        )
        p = dict(packet(rid), scenario_id=sid)
        assigned = judge.assign_judges(policy, [assignment(job)])[rid]
        a = judge.judge_run(
            policy,
            assigned,
            p,
            Caller((lambda v, n: fail_truth(v)) if i == 131 else None),
            prompt=PROMPT,
        )
        row = {
            "run": rid,
            "teacher": model,
            "task": sid,
            "messages": [
                {
                    "role": "assistant",
                    "content": f"decision {n}",
                    "reasoning_content": "generator-only",
                }
                for n in range(5)
            ],
            "label_turns": list(range(5)),
            "no_loss_turns": [],
            "tools": [],
            "steps": [{"seq": n + 1, "message": n} for n in range(5)],
        }
        manifest = {
            "batch_id": policy.batch_id,
            "job_id": rid,
            "model": model,
            "scenario_id": sid,
            "judge_policy": asdict(policy),
            "harness_commit": CHAMPION_REV,
            "builder_commit": POET_REV,
        }
        for field, kind, value in [
            ("packet", "judge-input", p),
            ("assessment", "judge-output", a),
            ("generator", "generator", row),
        ]:
            manifest[field] = store_artifact(objects, policy.batch_id, rid, kind, value)
        job["artifact_sha256"] = store_artifact(
            objects, policy.batch_id, rid, "manifest", manifest
        )["sha256"]
        db.rows.append(job)
    bundle = release.source_bundle(raw, PROMPT, db, objects)
    private, public = keys()
    receipt = review.sign_review(bundle, complete_marks(bundle), "test-human", private)
    return raw, db, objects, bundle, receipt, public


def test_reviewed_registered_generator_release_and_pricing(tmp_path, pilot):
    raw, db, objects, bundle, receipt, public = pilot
    output = tmp_path / "release.zip"
    meta = release.write_exports(bundle, receipt, public, objects, output)
    assert meta["views"]["decisions"]["rows"] == 100
    with ZipFile(output) as z:
        only = z.read("decisions.jsonl").decode()
        assert (
            "generator-only" not in only
            and "Grounded claim" not in only
            and "run-131" not in only
        )
        assert "generator-only" in z.read("reasoning.jsonl").decode()
        sampled = json.loads(z.read("hard_steps.jsonl").decode().splitlines()[0])
        assert [
            n + 1 for n, m in enumerate(sampled["messages"]) if "reasoning_content" in m
        ] == [1, 3]
    assert (
        compile_pilot(db, bundle, receipt, public, objects)["prices"]["C"][MODELS[0]]
        == 125
    )
    with pytest.raises(FileExistsError):
        release.write_exports(bundle, receipt, public, objects, output)

    with pytest.raises(ValueError, match="signature"):
        release.write_exports(
            bundle, receipt, keys()[1], objects, tmp_path / "invalid.zip"
        )
    assert not (tmp_path / "invalid.zip").exists()
    alien = deepcopy(raw)
    alien["control_slots"] = {"129": True, "131": False}
    with pytest.raises(ValueError):
        release.source_bundle(alien, PROMPT, db, objects)

    db.rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        release.source_bundle(raw, PROMPT, db, objects)
