"""Use actual repeated decisions and the original signed grant through the private exporter."""

from hashlib import sha256
import json
from zipfile import ZipFile

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
