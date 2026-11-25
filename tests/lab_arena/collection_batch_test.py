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
        if name == "collection_training_consents_v1":
            return self.grants
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


def original_journal(objects, *, batch, rid, model, row):
    from lab_arena.collection_journal import WorkflowJournal

    j = WorkflowJournal(objects, batch, rid)
    j.append(
        {
            "index": 0,
            "kind": "start",
            "model": model,
            "messages": [],
            "tools": row["tools"],
        }
    )
    for i, msg in enumerate(row["messages"], 1):
        j.append({"index": i, "kind": "reply", "message": msg})
    j.append({"index": len(row["messages"]) + 1, "kind": "finish"})
    j.finish("completed")
    return j.hashes


@pytest.fixture
def pilot(tmp_path, request):
    db = Database()
    objects = LocalObjectStore(tmp_path / "objects")
    variant = getattr(request, "param", 22)
    witnessed = variant in {"witnessed-complete", "reference-60"}
    exportable = variant == "training-pack" or witnessed
    candidate = variant in {
        "open-panel",
        "training-pack",
        "witnessed-complete",
        "reference-60",
    }
    checkpointed = variant == "checkpointed" or exportable
    split = SPLIT
    if exportable:
        import json, os
        from pathlib import Path
        from bittensor_wallet import Keypair
        from lab_arena import collection_intake as intake, contracts
        from lab_arena.collection_quality import admit_split

        poet = os.environ.get("COLLECTION_TEST_POET_REPO")
        if not poet:
            pytest.skip("requires the pinned P7 partition")
        split = json.loads(
            (
                Path(poet) / "harness/gtm-tasks/partner_credit_partition.json"
            ).read_bytes()
        )["split"]
        identity = {k: v[0] for k, v in split["training"].items()}
        stamp = admit_split(identity, split)
        wallet = Keypair.create_from_uri("//TrajectoryTest")

        def signed_consent():
            return contracts.build_signed_request(
                scope=intake.SCOPE["register"],
                round_id=intake.scope_id("pilot"),
                hotkey=wallet.ss58_address,
                body={
                    "task_scope": intake.TASK_SCOPE,
                    "terms_sha256": intake.SYNTHETIC_TRAINING_TERMS_HASH,
                    "credentials_sha256": "sha256:" + "e" * 64,
                    "daily_cap_microusd": 300000,
                },
                timestamp=100,
                sign_message=lambda msg: wallet.sign(msg.encode()).hex(),
            )

        auth = signed_consent()
        db.grants = []

    models = {m: j.route for m,j in judge.QUALIFICATION_JUDGES.items()} if candidate else {
        "anthropic/claude-fable-5.2": "test", "openai/gpt-6.2-astra": "test"
    }
    roles = judge.QUALIFICATION_ROLES if candidate else judge.JUDGES
    judge_families = {judge.model_family(m) for m in models}
    drift_slots = [i for i in range(130) if judge.model_family(MODELS[i % len(MODELS)]) not in judge_families][-30:]
    raw = {
        "name": "test",
        "split_manifest": split,
        "mode": "pilot",
        "seed": "test-seed",
        "routes": {
            **{model: route.tag for model, route in GENERATOR_ROUTES.items()},
            **models,
        },
        "scenario_ids": [
            f"{'P7' if exportable else 'C01'}-s{2000+i}"
            for i in range(22 if candidate or checkpointed else variant)
        ],
        "budgets_microusd": dict.fromkeys(MODELS, 1000000),
        "control_slots": {"130": True, "131": False},
        "drift_slots": {
            str(i): {v: dict.fromkeys(judge.PARTS, 3) for v in roles}
            for i in drift_slots
        },
    }
    if exportable:
        raw["terms_sha256"] = intake.SYNTHETIC_TRAINING_TERMS_HASH
    if variant == "reference-60":
        raw["scenario_ids"] = [f"P7-s{n+2000}" for n in range(10)]
        raw["control_slots"] = {}
        raw["drift_slots"] = {}
        drift_slots = []
    private_cases = {}
    source_policy = judge.BatchPolicy(
        "prior-drift-baseline", "earlier-seed", judge.digest(PROMPT), pilot=True,
        fable_version=tuple(models)[0], astra_version=tuple(models)[1],
    )
    for i in ([] if variant == "reference-60" else [*drift_slots, 130, 131]):
        src = f"reference-{i}"
        sid = raw["scenario_ids"][i // 6]
        original = dict(
            packet(src), scenario_id=sid, task=f"Check synthetic account {i}"
        )
        if exportable:
            original["split"] = stamp
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
        raw["capture_version"] = COMPARISON_CAPTURE
    if witnessed:
        from lab_arena.collection_route_witness import WITNESS
        from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
        from lab_arena.collection_generation import CAPTURE_V2

        raw.update(route_witness_sha256=WITNESS, capture_version=CAPTURE_V2)
        raw["routes"][REFERENCE_MODEL] = REFERENCE_ROUTE

    agent, builder = comparison_revisions(raw)
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
        if exportable:
            from uuid import UUID

            job.update(attempt=1, miner_hotkey=wallet.ss58_address)
            p["split"] = stamp
            if variant == "reference-60":
                p["final_output"] = f"Original model output for slot {i}"
            billing = {
                "unresolved": 0,
                "generator_calls": 0 if i in private_cases else 5,
                "miner_cost_microusd": 0 if i in private_cases else 50,
            }
            job.update(billing)
            db.grants.append(
                dict(
                    job_id=rid,
                    miner_hotkey=wallet.ss58_address,
                    mode="pilot",
                    registration_id=str(UUID(hex=auth["request_id"])),
                    consent=deepcopy(auth),
                    **billing,
                )
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
        if exportable:
            row["split"] = identity
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
            manifest["capture_version"] = raw["capture_version"]
            if i not in private_cases:
                manifest["events"] = original_journal(
                    objects, batch=policy.batch_id, rid=rid, model=model, row=row
                )
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
    if variant == "reference-60":
        return raw, db, objects, bundle, None, None
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
@pytest.mark.parametrize("blind_only", [False, True])
def test_signed_live_panel_price_compilation_requires_separate_economic_approval(pilot, blind_only):
    _, db, objects, bundle, _, _ = pilot
    reviewer, review_key = keys()
    operator, operator_key = keys()
    if blind_only:
        _, jobs = review.prepare_review(bundle, for_qualification=True)
        receipt = review.sign_blind_review(jobs, complete_marks(bundle), judge.digest(bundle), "blind-reviewer", reviewer)
    else:
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
    approved = compile_pilot(db, bundle, receipt, public, objects)
    assert (
        approved["harness_commit"],
        approved["builder_commit"],
    ) == comparison_revisions(raw)
    assert (
        approved["capture_version"] == raw["capture_version"]
        and approved["prices"]["C"][MODELS[0]] == 125
    )
    live = dict(raw, mode="live")
    verify_live_panel(live, load_batch(live, PROMPT), approved)
    legacy = dict(live)
    legacy.pop("capture_version")
    with pytest.raises(ValueError, match="differs"):
        verify_live_panel(legacy, load_batch(legacy, PROMPT), approved)
    # Changing a single paid run's original journal invalidates the price source.
    path = (
        objects._root
        / f"collection/{bundle['policy']['batch_id']}/run-0/events/00001.json"
    )
    doc = json.loads(path.read_bytes())
    doc["message"]["content"] = "changed"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="journal is missing or changed"):
        compile_pilot(db, bundle, receipt, public, objects)


@pytest.mark.parametrize("pilot", ["training-pack"], indirect=True)
def test_qualified_six_model_export_checks_every_signed_job(pilot, tmp_path):
    import json
    from zipfile import ZipFile

    raw, db, objects, bundle, _, _ = pilot
    reviewer, review_key = keys()
    approver, approval_key = keys()
    receipt = review.sign_review(
        bundle, complete_marks(bundle), "human-fixture", reviewer
    )
    approval = review.approve_qualification(
        bundle, receipt, review_key, "independent-fixture", approver
    )
    qual = dict(
        raw=raw,
        prompt=PROMPT,
        receipt=receipt,
        approval=approval,
        reviewer_key=review_key,
        approver_key=approval_key,
    )
    out = tmp_path / "rows.zip"
    result = release.write_exports(raw, PROMPT, db, objects, out, qualification=qual)
    assert result["trajectory_rows"] == 100 and result["excluded_controls"] == 32
    with ZipFile(out) as z:
        rows = [
            json.loads(z.read(n)) for n in z.namelist() if n.startswith("trajectories/")
        ]
        assert {r["model"] for r in rows} == set(MODELS)
        assert all(
            r["miner_cost_microusd"] == 50 and "reference_answer" not in r for r in rows
        )
        assert all(r["manifest_sha256"] and r["messages"] for r in rows)
    db.grants[0]["consent"]["body"]["daily_cap_microusd"] += 1
    with pytest.raises(ValueError, match="signature"):
        release.write_exports(
            raw, PROMPT, db, objects, tmp_path / "tampered.zip", qualification=qual
        )
    assert not (tmp_path / "tampered.zip").exists()


@pytest.mark.parametrize("pilot", ["witnessed-complete"], indirect=True)
def test_signed_successor_export_keeps_genuine_absolute_slots_and_matching_route(
    pilot, tmp_path, monkeypatch
):
    import json
    from zipfile import ZipFile
    from lab_arena import collection_route_witness as witness
    from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
    from lab_arena.collection_intake import MODELS, GENERATOR_ROUTES

    qualraw, reference, objects, bundle, _, _ = pilot
    reviewer, rkey = keys()
    approver, akey = keys()
    receipt = review.sign_review(
        bundle, complete_marks(bundle), "independent-human", reviewer
    )
    approval = review.approve_qualification(
        bundle, receipt, rkey, "operator-test", approver
    )
    qualification = dict(
        raw=qualraw,
        prompt=PROMPT,
        receipt=receipt,
        reviewer_key=rkey,
        approval=approval,
        approver_key=akey,
    )
    initial = deepcopy(qualraw)
    initial.pop("route_witness_sha256")
    initial.update(
        name="new-case",
        scenario_ids=qualraw["scenario_ids"][:1],
        control_slots={},
        drift_slots={},
        control_artifacts={},
    )
    initial["routes"][REFERENCE_MODEL] = GENERATOR_ROUTES[REFERENCE_MODEL].tag
    parent = load_batch(initial, PROMPT)
    raw = deepcopy(initial)
    raw["routes"][REFERENCE_MODEL] = REFERENCE_ROUTE
    raw["reference_successor"] = {
        "parent": parent.batch_id,
        "next_slot": 1,
        "owner_witness_sha256": witness.WITNESS,
    }
    target = load_batch(raw, PROMPT)
    key = next(r for r in reference.rows if r["slot"] == 1)
    real = dict(
        key,
        batch_id=target.batch_id,
        job_id="new-successor-only",
        budget_microusd=raw["budgets_microusd"][REFERENCE_MODEL],
    )
    original = next(x for x in bundle["runs"] if x["packet"]["run_id"] == key["job_id"])
    rid = real["job_id"]
    reader = release.read_artifact
    previous = reader(
        objects,
        key["batch_id"],
        key["job_id"],
        "manifest",
        {"sha256": key["artifact_sha256"]},
        1 << 20,
    )
    generator = reader(
        objects,
        key["batch_id"],
        key["job_id"],
        "generator",
        previous["generator"],
        128 << 20,
    )
    generator.update(
        run=rid, evidence={"events": [{"kind": "reply"} for _ in range(5)]}
    )
    pkt = dict(original["packet"], run_id=rid)
    pkt["checks"] = dict(
        pkt["checks"], answered=True, shape=True, code_grade=True, no_hard_failure=True
    )
    a = judge.assign_judges(target, [assignment(real)])[rid]
    assessment = judge.judge_run(target, a, pkt, Caller(), prompt=PROMPT)
    manifest = dict(
        previous, batch_id=target.batch_id, job_id=rid, judge_policy=asdict(target)
    )
    for k, n, val in [
        ("packet", "judge-input", pkt),
        ("assessment", "judge-output", assessment),
        ("generator", "generator", generator),
    ]:
        manifest[k] = store_artifact(objects, target.batch_id, rid, n, val)
    manifest["events"] = original_journal(
        objects, batch=target.batch_id, rid=rid, model=real["model"], row=generator
    )
    real["artifact_sha256"] = store_artifact(
        objects, target.batch_id, rid, "manifest", manifest
    )["sha256"]
    target_data = Database()
    target_data.rows = [real]
    target_data.grants = [dict(reference.grants[1], job_id=rid)]

    class TwoBatches:
        def rpc(self, n, params):
            return (
                reference
                if params["p_batch"] == bundle["policy"]["batch_id"]
                else target_data
            ).rpc(n, params)

    db = TwoBatches()
    # The owner witness verifier is tested separately and exercised with its real
    # production S3 originals. The fixture keeps real qualification and miner signatures.
    monkeypatch.setattr(
        witness,
        "verified_witness",
        lambda raw, objects: {
            "owner_witness_sha256": witness.WITNESS,
            "model": REFERENCE_MODEL,
        },
    )
    dest = tmp_path / "successor.zip"
    meta = release.write_exports(
        raw, PROMPT, db, objects, dest, qualification=qualification
    )
    assert (
        meta["rows"] == 1
        and meta["actual_slots"] == [1]
        and meta["complete_qualification_batch"] is False
    )
    with ZipFile(dest) as z:
        row = json.loads(z.read("trajectories/" + rid + ".jsonl"))
        assert row["slot"] == 1 and row["model"] == REFERENCE_MODEL
        assert (
            row["provider_route"] == REFERENCE_ROUTE
            and row["reference_successor"] == raw["reference_successor"]
        )
        assert not any(key["job_id"] in name for name in z.namelist())
    with pytest.raises(ValueError, match="private reference"):
        release.source_bundle(raw, PROMPT, db, objects)
    bad = deepcopy(raw)
    bad["reference_successor"]["owner_witness_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        release.write_exports(
            bad, PROMPT, db, objects, tmp_path / "bad.zip", qualification=qualification
        )
    alt = deepcopy(qualraw)
    alt["budgets_microusd"][REFERENCE_MODEL] += 1
    with pytest.raises(ValueError, match="matches the witnessed route"):
        witness.verify_matching_panel(raw, target, alt, PROMPT, db, successor=True)
    with pytest.raises(ValueError, match="independent reviewer"):
        release.write_exports(raw, PROMPT, db, objects, tmp_path / "no-review.zip")

    from scripts import collection_reference_followup as progression
    from scripts.collection_review import write_new

    monkeypatch.setattr(progression, "verified_witness", witness.verified_witness)
    independently_frozen = deepcopy(qualraw)
    independently_frozen.update(
        name="full-reference",
        scenario_ids=qualraw["scenario_ids"][:10],
        control_slots={},
        drift_slots={},
        control_artifacts={},
    )
    future = load_batch(independently_frozen, PROMPT)
    setting = {
        "schema": "collection.one-reference-followup.v1",
        "after_batch_id": target.batch_id,
        "next_batch_id": future.batch_id,
        "next_batch_file": "complete-reference.json",
        "contribution_cap_microusd": sum(
            independently_frozen["budgets_microusd"].values()
        )
        * 10,
    }
    cfg = tmp_path / "handoff.json"
    write_new(cfg, setting)
    write_new(tmp_path / "complete-reference.json", independently_frozen)
    assert progression.reference_for_poll(raw, PROMPT, cfg, db, objects) == raw
    # Settled failed attempts whose entire original budget was consumed cannot be retried.
    # The independent passing, reviewed authentic slot above still supplies the quality evidence.
    for i in range(2, 6):
        budget = raw["budgets_microusd"][MODELS[i]]
        item = dict(
            real,
            model=MODELS[i],
            slot=i,
            status="failed",
            artifact_sha256=None,
            job_id=f"terminal-{i}",
            budget_microusd=budget,
            miner_cost_microusd=budget,
        )
        target_data.rows.append(item)
        target_data.grants.append(
            dict(
                target_data.grants[0], job_id=item["job_id"], miner_cost_microusd=budget
            )
        )
    assert (
        progression.reference_for_poll(raw, PROMPT, cfg, db, objects)
        == independently_frozen
    )
    target_data.rows[-1]["unresolved"] = 1
    assert progression.reference_for_poll(raw, PROMPT, cfg, db, objects) == raw
    wrong = tmp_path / "wrong.json"
    write_new(wrong, dict(setting, contribution_cap_microusd=1))
    with pytest.raises(ValueError, match="settings differ"):
        progression.reference_for_poll(raw, PROMPT, wrong, db, objects)


@pytest.mark.parametrize("pilot", ["reference-60"], indirect=True)
def test_original_sealer_uses_30_actual_distinct_paired_packets(pilot, monkeypatch):
    from scripts import collection_qualification_followup as h

    ref, db, objects, bundle, _, _ = pilot
    # The CoreWeave owner artifact is exercised against real S3 and in a dedicated test.
    monkeypatch.setattr(h, "verified_witness", lambda *a: {"checked": True})
    raw = dict(
        deepcopy(ref),
        name="independent-24",
        scenario_ids=[f"P7-s{2000+i}" for i in range(24)],
        seed="freeze-before",
    )
    pending = {"schema": "sn71.p7.qualification-draft.v1", "manifest": raw}
    case = bundle["runs"][0]["packet"]
    plants = []
    for i, v in enumerate((False, True)):
        public = dict(case, task=f"Reviewed planted statement {i}")
        plants.append(
            controls.planted_case(
                public,
                expected=v,
                reviewer="fixture",
                rationale="Compared to original ledger",
                source_sha256="sha256:" + judge.digest(public),
                source_generator="human",
            )
        )
    cohort, pack, bid = h.seal(pending, PROMPT, ref, plants, db, objects)
    assert (
        len(cohort["control_slots"]) == 2
        and len(cohort["drift_slots"]) == 30
        and len(pack["cases"]) == 32
    )
    measured = [c["provenance"] for c in pack["cases"].values() if c["kind"] == "drift"]
    assert len({m["source_run_id"] for m in measured}) == 30
    assert all(
        m["source_batch_id"] == bundle["policy"]["batch_id"]
        and m["source_manifest_sha256"]
        for m in measured
    )
    assert (
        bid == load_batch(cohort, PROMPT).batch_id
        and bid != bundle["policy"]["batch_id"]
    )
    for i in range(40):
        db.rows[i]["generator_calls"] = db.grants[i]["generator_calls"] = 0
    with pytest.raises(ValueError, match="30 distinct measured"):
        h.seal(pending, PROMPT, ref, plants, db, objects)
