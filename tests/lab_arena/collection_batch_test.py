"""Replay a complete synthetic pilot and export only signed generator evidence."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from lab_arena import (
    collection_controls as controls,
    collection_judge as judge,
    collection_judge_review as review,
)
from lab_arena.collection_calibration import compile_pilot
from lab_arena.collection_intake import GENERATOR_ROUTES, MODELS
from lab_arena.collection_workflow import CHAMPION_REV, POET_REV, store_artifact
from lab_arena.service import LocalObjectStore
from scripts import collection_batch as release
from scripts.run_locked_collection import load_batch, assignment
from tests.lab_arena.collection_judge_test import PROMPT, SPLIT, Caller, fail_truth, packet
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
def pilot(tmp_path, request):
    db = Database()
    objects = LocalObjectStore(tmp_path / "objects")
    variant = getattr(request, "param", 22)
    candidate = variant == "open-panel"
    checkpointed = variant == "checkpointed"
    models = {m: j.route for m,j in judge.QUALIFICATION_JUDGES.items()} if candidate else {
        "anthropic/claude-fable-5.2": "test", "openai/gpt-6.2-astra": "test"
    }
    roles = judge.QUALIFICATION_ROLES if candidate else judge.JUDGES
    raw = {
        "name": "test",
        "split_manifest": SPLIT,
        "mode": "pilot",
        "seed": "test-seed",
        "routes": {
            **{model: route.tag for model, route in GENERATOR_ROUTES.items()},
            **models,
        },
        "scenario_ids": [
            f"C01-s{2000+i}" for i in range(22 if candidate or checkpointed else variant)
        ],
        "budgets_microusd": dict.fromkeys(MODELS, 1000000),
        "control_slots": {"130": True, "131": False},
        "drift_slots": {
            str(i): {v: dict.fromkeys(judge.PARTS, 3) for v in roles}
            for i in range(100, 130)
        },
    }
    private_cases = {}
    source_policy = judge.BatchPolicy(
        "prior-drift-baseline", "earlier-seed", judge.digest(PROMPT), pilot=True,
        fable_version=tuple(models)[0], astra_version=tuple(models)[1],
    )
    for i in range(100, 132):
        src = f"reference-{i}"
        sid = raw["scenario_ids"][i // 6]
        original = dict(
            packet(src), scenario_id=sid, task=f"Check synthetic account {i}"
        )
        if i >= 130:
            case = controls.planted_case(
                original,
                expected=i == 130,
                reviewer="fixture",
                rationale="Known fixture control outcome",
                source_sha256="sha256:" + judge.digest(original),
                source_generator="human",
            )
        else:
            sa = {"run_id": src, "primary": roles[0], "overlap": True}
            verdict = judge.judge_run(
                source_policy, sa, original, Caller(), prompt=PROMPT
            )
            baseline = {
                "job_id": src,
                "model": MODELS[1],
                "scenario_id": sid,
                "batch_id": source_policy.batch_id,
                "judge_policy": asdict(source_policy),
                "packet": store_artifact(
                    objects, source_policy.batch_id, src, "judge-input", original
                ),
                "assessment": store_artifact(
                    objects, source_policy.batch_id, src, "judge-output", verdict
                ),
            }
            manifest_ref = store_artifact(
                objects, source_policy.batch_id, src, "manifest", baseline
            )
            case = controls.drift_case(
                original,
                verdict,
                baseline,
                source_manifest_sha256=manifest_ref["sha256"],
            )
        private_cases[i] = case
    raw["control_artifacts"] = {
        str(i): judge.digest(case) for i, case in private_cases.items()
    }
    from lab_arena.collection_workflow import comparison_revisions, COMPARISON_CAPTURE
    if checkpointed:
        raw["capture_version"]=COMPARISON_CAPTURE
    agent,builder=comparison_revisions(raw)
    policy = load_batch(raw, PROMPT)
    for i in range(len(raw["scenario_ids"]) * len(MODELS)):
        rid = f"run-{i}"
        sid = raw["scenario_ids"][i // 6]
        model = MODELS[i % 6]
        job = dict(
            job_id=rid,
            batch_id=policy.batch_id,
            mode="pilot",
            slot=i,
            model=model,
            harness_commit=agent,
            miner_hotkey=f"test-miner-{i}",
            scenario_id=sid,
            status="accepted",
        )
        p = (
            dict(private_cases[i]["packet"], run_id=rid)
            if i in private_cases
            else dict(packet(rid), scenario_id=sid)
        )
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
            "harness_commit": agent,
            "builder_commit": builder,
        }
        for field, kind, value in [
            ("packet", "judge-input", p),
            ("assessment", "judge-output", a),
            ("generator", "generator", row),
        ]:
            manifest[field] = store_artifact(objects, policy.batch_id, rid, kind, value)
        if checkpointed:
            from lab_arena.collection_journal import WorkflowJournal
            manifest["capture_version"] = raw["capture_version"]
            if i not in private_cases:
                journal=WorkflowJournal(objects,policy.batch_id,rid)
                journal.append({"index":0,"kind":"start","model":model,"messages":[],"tools":row["tools"]})
                for m in row["messages"]:
                    journal.append({"index":len(journal.hashes),"kind":"reply","message":m})
                journal.append({"index":len(journal.hashes),"kind":"finish"})
                journal.finish("completed")
                manifest["events"]=journal.hashes
        if i in private_cases:
            manifest.pop("generator")
            manifest["control"] = store_artifact(
                objects, policy.batch_id, rid, "control-case", private_cases[i]
            )
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
    with pytest.raises(ValueError, match="training release rights"):
        release.write_exports(bundle, receipt, public, objects, output)
    assert not output.exists()
    assert (
        compile_pilot(db, bundle, receipt, public, objects)["prices"]["C"][MODELS[0]]
        == 125
    )
    with pytest.raises(ValueError, match="signature"):
        compile_pilot(db, bundle, receipt, keys()[1], objects)
    alien = deepcopy(raw)
    alien["control_slots"] = {"129": True, "131": False}
    with pytest.raises(ValueError):
        release.source_bundle(alien, PROMPT, db, objects)

    db.rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        release.source_bundle(raw, PROMPT, db, objects)


def test_price_compilation_refuses_missing_generator_upload(pilot, monkeypatch):
    _, db, objects, bundle, receipt, public = pilot
    get = objects.get_bounded

    def missing(ref, size):
        if "/generator-" in ref:
            raise FileNotFoundError("generator unavailable")
        return get(ref, size)

    monkeypatch.setattr(objects, "get_bounded", missing)
    with pytest.raises(FileNotFoundError, match="generator"):
        compile_pilot(db, bundle, receipt, public, objects)


def test_export_binds_reviewed_judges_to_generator_manifest(pilot, tmp_path):
    _, _, objects, bundle, _, _ = pilot
    altered = deepcopy(bundle)
    item = next(i for i in altered["runs"] if "control_sha256" not in i)
    rid = item["packet"]["run_id"]
    bid = altered["policy"]["batch_id"]
    manifest = release.read_artifact(
        objects, bid, rid, "manifest", {"sha256": item["manifest_sha256"]}, 1024 * 1024
    )
    manifest["judge_policy"] = {
        **altered["policy"],
        "fable_version": "moonshotai/kimi-k3",
        "astra_version": "mistralai/mistral-large-2512",
    }
    item["manifest_sha256"] = store_artifact(objects, bid, rid, "manifest", manifest)[
        "sha256"
    ]
    test_private, test_public = keys()
    signed = review.sign_review(
        altered, complete_marks(altered), "unit-test", test_private
    )
    out = tmp_path / "mismatched.zip"
    with pytest.raises(ValueError, match="training release rights"):
        release.write_exports(altered, signed, test_public, objects, out)
    assert not out.exists()


def test_registered_failed_attempts_remain_visible_without_review_votes(pilot):
    raw, db, objects, old, _, _ = pilot
    completed = db.rows[1]
    failed = dict(
        completed,
        job_id="original-failed",
        status="failed",
        artifact_sha256=None,
        attempt=1,
    )
    completed["attempt"] = 2
    db.rows.append(failed)
    new = release.source_bundle(raw, PROMPT, db, objects)
    assert new["runs"] == old["runs"]
    assert len(new["failed_attempts"]) == 1
    assert new["failed_attempts"][0]["run_id"] == "original-failed"
    assert review.prepare_review(new)[1] == review.prepare_review(old)[1]
    failed["status"] = "accepted"
    with pytest.raises(ValueError):
        release.source_bundle(raw, PROMPT, db, objects)


def test_unreplaced_failure_preserves_reviews_and_enforces_real_pilot_minimum(pilot):
    raw, db, objects, _, _, _ = pilot
    failed = db.rows[0]
    failed.update(status="failed", artifact_sha256=None)
    bundle = release.source_bundle(raw, PROMPT, db, objects)
    assert not any(i["packet"]["run_id"] == failed["job_id"] for i in bundle["runs"])
    private, _ = keys()
    report = review.sign_review(bundle, complete_marks(bundle), "test-human", private)[
        "report"
    ]
    assert report["blind_human_marks"] == 50
    assert "pilot_judged_runs" in report["holds"]
    assert "incomplete_batch" not in report["holds"]


@pytest.mark.parametrize("pilot", [23], indirect=True)
def test_exhausted_noncontrol_slot_does_not_erase_successes_but_failed_control_holds_export(
    pilot, tmp_path
):
    raw, db, objects, _, _, _ = pilot
    first = db.rows[0]
    first.update(status="failed", artifact_sha256=None)
    db.rows.append(dict(first, attempt=2, job_id="exhausted"))
    b = release.source_bundle(raw, PROMPT, db, objects)
    private, public = keys()
    signed = review.sign_review(b, complete_marks(b), "test-human", private)
    assert (
        signed["report"]["quality_ready"]
        and signed["report"]["terminal_failed_attempts"] == 2
    )
    assert len(b["runs"]) - len(b["drift"]) - len(b["planted"]) == 105
    assert review.verify_review(b, signed, public)["quality_ready"]
    db.rows[130].update(status="failed", artifact_sha256=None)
    b = release.source_bundle(raw, PROMPT, db, objects)
    signed = review.sign_review(b, complete_marks(b), "test-human", private)
    assert "unjudged_control" in signed["report"]["holds"]
    with pytest.raises(ValueError, match="did not pass review"):
        review.verify_review(b, signed, public)

@pytest.mark.parametrize("pilot", ["open-panel"], indirect=True)
def test_signed_live_panel_price_compilation_requires_separate_economic_approval(pilot):
    _, db, objects, bundle, _, _ = pilot
    reviewer, review_key = keys()
    operator, operator_key = keys()
    receipt = review.sign_review(bundle, complete_marks(bundle), "blind-reviewer", reviewer)
    assert receipt["report"]["holds"] == ["judge_qualification"]
    only_corrections = review.approve_qualification(bundle, receipt, review_key, "team-owner", operator)
    with pytest.raises(ValueError, match="excludes compensated"):
        compile_pilot(db, bundle, receipt, review_key, objects, approval=only_corrections, approver_key=operator_key)
    both = review.approve_qualification(bundle, receipt, review_key, "team-owner", operator, allow_live=True)
    result = compile_pilot(db, bundle, receipt, review_key, objects, approval=both, approver_key=operator_key)
    assert result["panel_qualification_sha256"] == judge.digest(both)
    assert result["split_manifest_sha256"] == judge.digest(SPLIT)
    assert set(result["prices"]["C"]) == set(MODELS)


def test_legacy_comparison_bundle_refuses_hidden_or_unpartitioned_review(pilot):
    raw, db, objects, bundle, _, _ = pilot
    from copy import deepcopy
    for state in ("missing", "hidden"):
        forged = deepcopy(bundle)
        if state == "missing":
            forged.pop("split_manifest_sha256")
        else:
            forged["runs"][0]["packet"]["checks"]["training_split"] = False
        with pytest.raises(ValueError):
            review.prepare_review(forged)

@pytest.mark.parametrize("pilot", ["checkpointed"], indirect=True)
def test_signed_checkpointed_prices_bind_journals_and_live_source(pilot):
    from scripts.run_locked_collection import verify_live_panel
    from lab_arena.collection_workflow import comparison_revisions
    import json

    raw, db, objects, bundle, receipt, public = pilot
    approved = compile_pilot(db,bundle,receipt,public,objects)
    assert (approved["harness_commit"],approved["builder_commit"]) == comparison_revisions(raw)
    assert approved["capture_version"] == raw["capture_version"] and approved["prices"]["C"][MODELS[0]]==125
    live=dict(raw,mode="live")
    verify_live_panel(live,load_batch(live,PROMPT),approved)
    legacy=dict(live)
    legacy.pop("capture_version")
    with pytest.raises(ValueError,match="differs"):
        verify_live_panel(legacy,load_batch(legacy,PROMPT),approved)
    # Changing a single paid run's original journal invalidates the price source.
    path=objects._root/f"collection/{bundle['policy']['batch_id']}/run-0/events/00001.json"
    doc=json.loads(path.read_bytes());doc["message"]["content"]="changed";path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="journal is missing or changed"):
        compile_pilot(db,bundle,receipt,public,objects)
