"""Growing approved training packs remain readable across retries and offer changes."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import collection_export_snapshots as snapshots


FIRST = "dv-" + "a" * 64
SECOND = "dv-" + "b" * 64


def private(path, doc):
    path.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
    path.write_text(json.dumps(doc))
    path.chmod(0o600)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    out = tmp_path / "exports"
    private(root / "prompt.json", {"policy": "fixed"})
    private(root / "paid/batch.json", {"id": FIRST, "split_manifest": {}})
    jobs = {FIRST: [], SECOND: []}
    original = {}
    missing = set()
    denied = set()
    held = set()
    reads = []
    builds = []

    def add(batch, rid, case="P7-s1", status="accepted", unsettled=0):
        jobs[batch].append(
            {
                "job_id": rid,
                "slot": len(jobs[batch]),
                "batch_id": batch,
                "scenario_id": case,
                "status": status,
                "artifact_sha256": "original-" + rid,
            }
        )
        original[rid] = {"job_id": rid, "unresolved": unsettled, "generator_calls": 1}

    def rpc(name, args):
        if name == "collection_offer_schedule_v1":
            key = args["p_offer"]
            reads.append(key)
            return {} if key in denied else {"pilot_evidence": {"id": key}}
        key = args["p_batch"]
        if name == "collection_batch_jobs_v1":
            return list(jobs[key])
        assert name == "collection_training_consents_v1"
        return [original[r["job_id"]] for r in jobs[key]]

    def build(raw, prompt, db, objects, path, *, only_jobs, publish):
        assert not publish
        assert (
            db.rpc("collection_offer_schedule_v1", {"p_offer": raw["id"]})["pilot_evidence"]["id"]
            == raw["id"]
        )
        builds.append(sorted(only_jobs))
        if missing & only_jobs:
            raise FileNotFoundError("missing immutable artifact")
        rows = {j["job_id"]: j for j in jobs[raw["id"]]}
        digest = sha256(json.dumps([(i, i in held) for i in sorted(only_jobs)]).encode()).hexdigest()
        pairs = int({rows[i]["status"] for i in only_jobs} >= {"accepted", "rejected"})
        meta = dict(rows=len(only_jobs), preference_pairs=pairs, dataset_sha256="sha256:" + digest)
        with ZipFile(path, "x") as archive:
            for i in sorted(only_jobs):
                row = dict(
                    run_id=i,
                    scenario_id=rows[i]["scenario_id"],
                    messages=[] if i in held else [{"role": "assistant"}],
                )
                if i in held:
                    row["capture_hold"] = "quarantine"
                archive.writestr("trajectories/" + i + ".jsonl", json.dumps(row))
            archive.writestr("metadata.json", json.dumps(meta))
        path.chmod(0o600)
        return meta

    monkeypatch.setattr(snapshots, "load_batch", lambda raw, _: SimpleNamespace(batch_id=raw["id"]))

    def verify(raw, policy, approval):
        assert raw["id"] == policy.batch_id == approval["id"]

    monkeypatch.setattr(snapshots, "verify_live_panel", verify)
    monkeypatch.setattr(snapshots, "write_exports", build)
    monkeypatch.setattr(snapshots, "store_artifact", lambda *args: None)

    def tick(prev=None):
        return snapshots.observe(root, out, SimpleNamespace(rpc=rpc), None, prev, max_groups=20)

    def index():
        link = json.loads((out / "current.json").read_bytes())["snapshot"]
        blob = (out / "snapshots" / (link + ".json")).read_bytes()
        assert sha256(blob).hexdigest() == link
        return json.loads(blob)["offers"]

    return SimpleNamespace(**locals())


def test_growth_keeps_prior_zip_and_scenario_preference(lab):
    lab.add(FIRST, "a")
    s = lab.tick()
    first = lab.index()[FIRST]["P7-s1"]
    prior_file = lab.out / FIRST / first["file"]
    original_bytes = prior_file.read_bytes()
    assert s["registered"] == s["released"] == 1 and s["held"] == 0
    builds = len(lab.builds)
    assert lab.tick(s)["snapshot"] == s["snapshot"] and len(lab.builds) == builds
    lab.add(FIRST, "b", status="rejected")
    lab.add(FIRST, "c", case="P7-s2", status="running")
    lab.add(FIRST, "d", case="P7-s2", unsettled=1)
    s2 = lab.tick(s)
    assert (s2["registered"], s2["released"], s2["pending"]) == (4, 2, 2)
    latest = lab.index()[FIRST]["P7-s1"]
    assert latest["preference_pairs"] == 1 and latest["file"] != first["file"]
    assert prior_file.read_bytes() == original_bytes
    lab.original["d"]["unresolved"] = 0
    assert lab.tick(s2)["released"] == 3


def test_missing_upload_and_quarantine_do_not_hide_siblings_or_leak(lab):
    for i in ("a", "b", "c"):
        lab.add(FIRST, i, status="failed" if i == "c" else "accepted")
    lab.missing.add("a")
    lab.held.add("c")
    s = lab.tick()
    group = lab.index()[FIRST]["P7-s1"]
    assert s["held"] == 2 and group["released"] == ["b"]
    assert group["held"] == {"a": "missing_artifact", "c": "quarantine"}
    with ZipFile(lab.out / FIRST / group["file"]) as z:
        assert "trajectories/a.jsonl" not in z.namelist()
        assert json.loads(z.read("trajectories/c.jsonl"))["messages"] == []
    lab.missing.clear()
    assert lab.tick(s)["released"] == 2
    assert lab.index()[FIRST]["P7-s1"]["held"] == {"c": "quarantine"}


def test_later_offer_keeps_the_original_approval_and_denies_an_unknown_offer(lab):
    lab.add(FIRST, "a")
    s = lab.tick()
    private(lab.root / "paid/offers" / (SECOND + ".json"), {"id": SECOND, "split_manifest": {}})
    lab.add(SECOND, "b", "P8-s12")
    s2 = lab.tick(s)
    assert set(lab.index()) == {FIRST, SECOND} and s2["released"] == 2
    assert set(lab.reads) == {FIRST, SECOND}
    lab.denied.add(FIRST)
    with pytest.raises(ValueError, match="missing historical"):
        lab.tick(s2)
    assert json.loads((lab.out / "current.json").read_bytes())["snapshot"] == s2["snapshot"]


def test_exported_sales_memo_replays_sources_and_keeps_revision_separate(tmp_path, monkeypatch):
    import os
    from lab_arena import collection_intake, collection_judge, collection_trajectories
    from lab_arena.collection_broker import VERSIONS
    from lab_arena.collection_workflow import (
        COMPARISON_CAPTURE,
        CORRECTION_CHAMPION_REV,
        SALES_EVIDENCE_POET_REV,
        SALES_EVIDENCE_RULE,
        workflow_sources,
    )
    from lab_arena.service import LocalObjectStore
    from scripts import collection_batch
    from tests.lab_arena.collection_judge_test import PROMPT
    from tests.lab_arena.collection_sales_evidence_test import execute, script, bad_write

    agent = os.getenv("COLLECTION_TEST_AGENT_REPO")
    poet = os.getenv("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("requires the pinned public harness and approved synthetic evidence source")
    objects = LocalObjectStore(tmp_path)
    model = collection_intake.MODELS[1]
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=SALES_EVIDENCE_POET_REV
    ) as sources:
        task, reference, steps = script(sources[1])
        # The model first posts an unsupported citation, then observes and fixes it.
        job, manifest, checked, outcomes, _, policy = execute(
            sources,
            objects,
            steps[:-2] + [bad_write(reference), steps[-2], steps[-1]],
            mode="live",
        )
        raw = dict(
            mode="live",
            name="test-sales",
            seed="fixed",
            scenario_ids=[job["scenario_id"]],
            capture_version=COMPARISON_CAPTURE,
            payment_rule=SALES_EVIDENCE_RULE,
            terms_sha256=collection_intake.PAID_WORKFLOW_TERMS_HASH,
            control_slots={},
            drift_slots={},
            budgets_microusd={model: 100000},
            fixed_quotes_microusd={model: 5000},
            split_manifest=sources[1].training_partition("P8"),
            routes={
                m: (
                    collection_intake.GENERATOR_ROUTES[m].tag
                    if m in collection_intake.GENERATOR_ROUTES
                    else "test"
                )
                for m in VERSIONS
            },
        )
    approved = dict(
        schema=collection_intake.VERIFIED_TASK_PRICE_SCHEMA,
        payment_rule=SALES_EVIDENCE_RULE,
        live_batch_id=policy.batch_id,
        terms_sha256=raw["terms_sha256"],
        routes={model: raw["routes"][model]},
        prices={"P8": raw["fixed_quotes_microusd"]},
        budgets={"P8": raw["budgets_microusd"]},
        capture_version=COMPARISON_CAPTURE,
        split_manifest_sha256=collection_judge.digest(raw["split_manifest"]),
        judge_prompt_sha256=policy.prompt_sha256,
        harness_commit=CORRECTION_CHAMPION_REV,
        builder_commit=SALES_EVIDENCE_POET_REV,
    )
    job.update(slot=0, attempt=1, status=outcomes[0][0][2])

    def rpc(name, params):
        if name == "collection_offer_schedule_v1":
            assert params["p_offer"] == policy.batch_id
            return {"pilot_evidence": approved}
        assert name == "collection_batch_jobs_v1"
        return [job]

    def grants(*args, **kwargs):
        return {
            job["job_id"]: dict(
                unresolved=0,
                miner_cost_microusd=checked["cost_microusd"],
                generator_calls=checked["calls"],
            )
        }

    monkeypatch.setattr(collection_batch, "load_batch", lambda *args: policy)
    monkeypatch.setattr(collection_trajectories, "verified_grants", grants)
    zip_path = tmp_path / "approved.zip"
    meta = collection_batch.write_exports(
        raw, PROMPT, SimpleNamespace(rpc=rpc), objects, zip_path, only_jobs={job["job_id"]}
    )
    with ZipFile(zip_path) as z:
        row = json.loads(z.read(f'trajectories/{job["job_id"]}.jsonl'))
    assert meta["preference_pairs"] == 0 and row["verified_task_pass"]
    assert row["sales_evidence"]["first_failed_decision"]["checkpoint_sha256"]
    assert row["sales_evidence"]["observed_repair"]["kind"] == "observed_revision"
    assert row["sales_evidence"]["opened_sources"] == checked["opened_sources"]
    assert all(row["sales_evidence"]["field_evidence"]["citation_checks"].values())
    assert not any(key in row for key in ("reference_answer", "initial_state", "documents"))


def test_watcher_exports_standing_experiments_without_legacy_offer_approval(lab, monkeypatch):
    """Pool discovery goes through native experiment release and publicly approved training partition."""
    from lab_arena import collection_workflow
    batch = "pool-" + "e" * 64
    raw = {"id": "safety", "payment_rule": snapshots.POOL_PAYMENT_RULE, "split_sha256": "sha256:" + "b" * 64}
    private(lab.root / "paid/offers" / (batch + ".json"), raw)
    spec = SimpleNamespace(raw=raw, batch_id=batch)
    monkeypatch.setattr(snapshots, "load_experiment", lambda doc, prompt: spec if doc == raw else None)
    built = []
    def dataset(experiment, db, objs, output, *, only_jobs, full_trajectories, publish, prompt):
        assert experiment == spec and full_trajectories and not publish
        built.append(sorted(only_jobs))
        result = dict(rows=len(only_jobs), preference_pairs=0,
                      dataset_sha256="sha256:" + sha256(json.dumps(sorted(only_jobs)).encode()).hexdigest())
        with ZipFile(output, "x") as archive:
            archive.writestr("metadata.json", json.dumps(result))
            for run in only_jobs:
                archive.writestr(f"trajectories/{run}.jsonl",
                                 json.dumps({"run_id":run, "scenario_id":"P9-s1","messages":[]}))
        output.chmod(0o600)
        return result
    monkeypatch.setattr(snapshots, "write_dataset", dataset)
    monkeypatch.setattr(collection_workflow, "read_artifact", lambda *_: {"split": "registered"})
    lab.jobs[batch] = []
    lab.add(batch, "good", case="P9-s1")
    lab.add(batch, "pending", case="P9-s1", unsettled=1)
    first = lab.tick()
    assert (first["released"], first["pending"]) == (1, 1) and built == [["good"]]
    assert batch not in lab.reads
    lab.original["pending"]["unresolved"] = 0
    last = lab.tick(first)
    assert (last["released"], last["pending"]) == (2, 0)
    assert lab.index()[batch]["P9-s1"]["released"] == ["good", "pending"]


def test_new_paid_case_precedes_rebuilding_a_valid_old_archive(lab):
    lab.add(FIRST, "old", case="P7-s1")
    old = lab.tick()
    prior = lab.index()
    prior[FIRST]["P7-s1"].pop("export_version")
    snapshots._save(lab.out, {"schema": "collection.private-export-index.v1", "offers": prior})
    lab.add(FIRST, "new", case="P7-s2")
    snapshots.observe(lab.root, lab.out, SimpleNamespace(rpc=lab.rpc), None, {**old,"last_group":""}, max_groups=1)
    assert lab.builds[-1] == ["new"]


def test_funded_tick_can_export_three_groups_and_resume_after_its_alarm(lab, monkeypatch):
    for index in range(4):
        lab.add(FIRST, f"run{index}", case=f"P8-s{index+1}")
    original = snapshots._pack
    attempts = []

    def limited(*args):
        attempts.append(args[5])
        if len(attempts) == 3:
            raise snapshots.ExportWindowExpired()
        return original(*args)

    monkeypatch.setattr(snapshots, "_pack", limited)
    db = SimpleNamespace(rpc=lab.rpc)
    first = snapshots.observe(lab.root, lab.out, db, None)
    assert first["released"] == 2 and first["pending"] == 2
    assert first["last_group"] == FIRST + "/P8-s2"
    assert lab.index()[FIRST]["P8-s3"]["queued"]
    assert lab.index()[FIRST]["P8-s1"]["released"] == ["run0"]
    next_tick = snapshots.observe(lab.root, lab.out, db, None, first)
    assert next_tick["released"] == 4 and next_tick["pending"] == 0
    assert attempts == ["P8-s1", "P8-s2", "P8-s3", "P8-s3", "P8-s4"]


def test_a_first_group_timeout_remains_visible(lab, monkeypatch):
    lab.add(FIRST, "run")
    def fail(*args):
        raise snapshots.ExportWindowExpired()
    monkeypatch.setattr(snapshots, "_pack", fail)
    with pytest.raises(snapshots.ExportWindowExpired):
        snapshots.observe(lab.root, lab.out, SimpleNamespace(rpc=lab.rpc), None)


def test_a_slow_zip_leaves_the_next_group_queued(lab, monkeypatch):
    lab.add(FIRST, "first", case="P8-s1")
    lab.add(FIRST, "second", case="P8-s2")
    times = iter([0, 15, 15])
    monkeypatch.setattr(snapshots, "time", SimpleNamespace(monotonic=lambda: next(times)))
    result = snapshots.observe(lab.root, lab.out, SimpleNamespace(rpc=lab.rpc), None, deadline=27)
    assert result["released"] == result["pending"] == 1
    assert lab.index()[FIRST]["P8-s2"]["queued"]


def test_funded_tick_does_not_hide_backend_timeouts(lab, monkeypatch):
    lab.add(FIRST, "first", case="P8-s1")
    lab.add(FIRST, "second", case="P8-s2")
    original = snapshots._pack

    def unavailable(*args):
        if args[5] == "P8-s2":
            raise TimeoutError("object storage timeout")
        return original(*args)

    monkeypatch.setattr(snapshots, "_pack", unavailable)
    with pytest.raises(TimeoutError, match="object storage timeout"):
        snapshots.observe(lab.root, lab.out, SimpleNamespace(rpc=lab.rpc), None)


def test_unreleased_bill_change_keeps_funded_siblings_visible_during_queue(lab):
    lab.add(FIRST, "done")
    lab.add(FIRST, "waiting", status="failed", unsettled=1)
    initial = lab.tick()
    original = lab.index()[FIRST]["P7-s1"]
    archive = lab.out / FIRST / original["file"]
    data = archive.read_bytes()

    # A different group wins this tick, while the previously unpublished attempt settles.
    lab.add(FIRST, "new", case="P7-s0")
    lab.original["waiting"]["unresolved"] = 0
    db = SimpleNamespace(rpc=lab.rpc)
    queued = snapshots.observe(lab.root, lab.out, db, None, {**initial, "last_group": ""}, max_groups=1)
    group = lab.index()[FIRST]["P7-s1"]
    assert group["queued"] and group["released"] == ["done"] and group["file"] == original["file"]
    assert archive.read_bytes() == data
    published = lab.tick(queued)
    assert lab.index()[FIRST]["P7-s1"]["released"] == ["done", "waiting"]

    # A funding change to an already published member must invalidate the current link.
    lab.add(FIRST, "other", case="P7-s00")
    lab.original["waiting"]["unresolved"] = 1
    snapshots.observe(lab.root, lab.out, db, None, {**published, "last_group": "", "pass": 0}, max_groups=1)
    changed = lab.index()[FIRST]["P7-s1"]
    assert changed["queued"] and not changed.get("file") and not changed.get("released")
    assert archive.read_bytes() == data
