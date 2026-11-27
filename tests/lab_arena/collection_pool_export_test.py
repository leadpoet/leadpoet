"""Use actual repeated decisions and the original signed grant through the private exporter."""

from hashlib import sha256
import json
from zipfile import ZipFile

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_experiment_export import write_dataset
from lab_arena.collection_pool_source import funded_source
from scripts.collection_export_snapshots import observe
from scripts.collection_review import write_new
from tests.lab_arena.collection_judge_test import PROMPT
from tests.lab_arena.collection_pool_worker_test import EXPERT, pool_fixture, pool_raw, scripts_for, simulation


def test_pool_siblings_repeats_and_actual_counterfactual_export(pool_fixture, tmp_path):
    env = pool_fixture
    raw = pool_raw()
    yes, no = scripts_for(env, "P9")
    spec = simulation(env, raw, [yes, no, yes, no])
    root = tmp_path / "private"
    offers = root / "paid" / "offers"
    offers.mkdir(parents=True)
    write_new(root / "prompt.json", PROMPT)
    write_new(offers / (spec.batch_id + ".json"), raw)
    output = tmp_path / "export"
    case = raw["cases"][0]["scenario_id"]

    def tick(prior=None):
        state = observe(root, output, env.db, env.objects, prior)
        index = json.loads((output / "snapshots" / (state["snapshot"] + ".json")).read_bytes())
        group = index["offers"][spec.batch_id][case]
        return state, output / spec.batch_id / group["file"]

    # All four calls were made. Delayed confirmation for just one holds its own row.
    student_job = env.jobs[-1]
    env.grants[student_job["job_id"]]["unresolved"] = 1
    prior, first = tick()
    assert (prior["released"], prior["pending"]) == (3, 1)
    original = first.read_bytes()
    env.grants[student_job["job_id"]]["unresolved"] = 0
    state, corpus = tick(prior)
    assert state["registered"] == state["released"] == 4 and state["pending"] == state["held"] == 0
    assert first != corpus and first.read_bytes() == original
    with ZipFile(corpus) as z:
        refs = [json.loads(s) for s in z.read("preferences.jsonl").splitlines()]
        assert len(refs) == 2
        for pair in refs:
            rows = []
            for side in ("chosen", "rejected"):
                ref = pair[side]
                encoded = z.read(ref["path"])
                assert ref["sha256"] == "sha256:" + sha256(encoded).hexdigest()
                row = json.loads(encoded)
                assert row["verified_complete"] and row["payment_eligible"]
                assert row["earned_target_microusd"] == (5 * row["miner_cost_microusd"] + 3) // 4
                assert row["label_turns"] and row["messages"] and any(m["role"] == "tool" for m in row["messages"])
                rows.append(row)
            good, bad = rows
            assert good["model_revision"] == bad["model_revision"] and good["repeat_group_id"] == bad["repeat_group_id"]
            assert good["experiment"]["repeat"] != bad["experiment"]["repeat"]
            assert good["verified_task_pass"] and bad["verified_task_pass"] is False

    # Independently replay the signed student run and resume a different expert at its exact mistake.
    source = funded_source(env.db, env.objects, spec.batch_id, student_job["job_id"], PROMPT)
    source_hash = "sha256:" + judge.digest(source)
    write_new(env.inputs / (source_hash[7:] + ".json"), source)
    correction = pool_raw("P9", EXPERT, 1)
    correction.pop("successor_seeds")
    correction["arms"][0]["role"] = "continuation"
    correction["cases"][0]["source_sha256"] = source_hash
    resumed = simulation(env, correction, [yes[-2:]])
    out = tmp_path / "correction.zip"
    meta = write_dataset(resumed, env.db, env.objects, out, full_trajectories=True, prompt=PROMPT)
    assert meta["correction_pairs"] == meta["trajectory_rows"] == 1
    rid = env.jobs[-1]["job_id"]
    with ZipFile(out) as z:
        pair = json.loads(z.read(f"pairs/{rid}.jsonl"))
        assert pair["context"] == source["checkpoint"]["messages"] and pair["rejected"] == source["rejected"]
        assert pair["chosen"] != pair["rejected"] and pair["final_turn_retained"]
        assert pair["expert_model"] != pair["student_model"]


def test_interrupted_pool_prefix_exports_before_its_final_bill(pool_fixture, tmp_path, monkeypatch):
    from tests.lab_arena.collection_sales_evidence_test import OriginalReply

    env = pool_fixture
    raw = pool_raw(repeats=2)
    actions, _ = scripts_for(env, "P9")
    spec = simulation(env, raw, [actions])
    original_call = OriginalReply.model_call

    def disconnect(self, request, params):
        if request == "generate-2":
            raise ConnectionError("provider unavailable")
        return original_call(self, request, params)

    monkeypatch.setattr(OriginalReply, "model_call", disconnect)
    with pytest.raises(RuntimeError, match="experiment interrupted"):
        simulation(env, raw, [actions])
    failed = env.jobs[-1]
    assert failed["status"] == "failed"
    rid = failed["job_id"]
    grant = env.grants[rid]
    grant["unresolved"] = 1
    grant["miner_cost_microusd"] = 10
    grant["generator_calls"] = 1

    root = tmp_path / "private"
    offers = root / "paid" / "offers"
    offers.mkdir(parents=True)
    write_new(root / "prompt.json", PROMPT)
    write_new(offers / (spec.batch_id + ".json"), raw)
    output = tmp_path / "exports"
    case = raw["cases"][0]["scenario_id"]

    def tick(prior=None):
        state = observe(root, output, env.db, env.objects, prior)
        index = json.loads((output / "snapshots" / (state["snapshot"] + ".json")).read_bytes())
        group = index["offers"][spec.batch_id][case]
        assert rid in group["released"]
        path = output / spec.batch_id / group["file"]
        with ZipFile(path) as z:
            row = json.loads(z.read(f"trajectories/{rid}.jsonl"))
            assert not any(name.startswith("pairs/") or name == "preferences.jsonl" for name in z.namelist())
        assert group["members"][rid]["job_status"] == "failed"
        assert row["partial"] and row["verified_task_pass"] is None and not row["verified_complete"]
        assert row["payment_eligible"] is False and row["earned_target_microusd"] == 0
        assert set(row["label_turns"]) == set(row["no_loss_turns"])
        assert row["no_loss_turns"] == [i for i, m in enumerate(row["messages"]) if m["role"] == "assistant"]
        assert any(m.get("reasoning") for m in row["messages"]) and any(m["role"] == "tool" for m in row["messages"])
        return state, path, row

    before, old, partial = tick()
    prior_bytes = old.read_bytes()
    assert partial["billing_status"] == "pending" and partial["unresolved_miner_calls"] == 1
    assert partial["miner_cost_microusd"] is None and partial["settled_miner_cost_microusd"] == 10
    with ZipFile(old) as archive:
        meta = json.loads(archive.read("metadata.json"))
        assert meta["unknown_miner_bills"] == 1 and meta["settled_miner_cost_microusd"] >= 10
        assert meta["miner_cost_microusd"] is None and meta["cost_per_kept_row_microusd"] is None
    # Resolving one of multiple open reservations can change the known bill
    # while the job remains partial_billing_pending. Refresh that revision too.
    grant["unresolved"] = 2
    waiting, two_open, two = tick(before)
    assert two["unresolved_miner_calls"] == 2 and two_open != old
    grant["unresolved"] = 1
    grant["miner_cost_microusd"] = 11
    waiting, one_open, one = tick(waiting)
    assert one["settled_miner_cost_microusd"] == 11 and one_open not in (old, two_open)
    grant["unresolved"] = 0
    grant["miner_cost_microusd"] = 12
    after, updated, settled = tick(waiting)
    assert before["snapshot"] != after["snapshot"] and updated != old and old.read_bytes() == prior_bytes
    assert settled["miner_cost_microusd"] == 12 and "unresolved_miner_calls" not in settled
    assert partial["messages"] == settled["messages"]

    # A signed grant alone cannot authorize invented provider output.
    grant["unresolved"] = 1
    read_original = env.objects.get_bounded

    def missing_wire(key, maximum):
        if key.startswith(f"collection/{spec.batch_id}/{rid}/wire-generate-1-"):
            raise FileNotFoundError(key)
        return read_original(key, maximum)

    monkeypatch.setattr(env.objects, "get_bounded", missing_wire)
    damaged = observe(root, output, env.db, env.objects, after)
    snapshot = json.loads((output / "snapshots" / (damaged["snapshot"] + ".json")).read_bytes())
    group = snapshot["offers"][spec.batch_id][case]
    assert rid not in group["released"] and group["held"][rid] == "missing_artifact"
