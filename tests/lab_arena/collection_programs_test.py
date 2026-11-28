"""Production compiler, real pinned producer/harness, native verifier and private release path."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import json, os
import pytest

from lab_arena import collection_programs as v3, collection_experiment as xp
from lab_arena.collection_experiment_export import write_dataset
from scripts.collection_program_schedule import _pack, compile_program
from scripts.collection_review import write_new
from tests.lab_arena.collection_pool_worker_test import pool_fixture, simulation
from tests.lab_arena.collection_judge_test import PROMPT


@pytest.mark.parametrize("pool_fixture", ["v3"], indirect=True)
@pytest.mark.parametrize("family", ("P7", "P8", "P9", "P10"))
@pytest.mark.parametrize("band", (1, 2, 3))
def test_all_funded_variants_replay_from_native_wires(
    pool_fixture, tmp_path, family, band
):
    env = pool_fixture
    case = f"{family}-s{band*v3.BAND}"
    models = v3.TEACHERS if band == 1 else (v3.CLOSE, v3.FAR)
    raw = _pack(
        v3.program("C1" if band == 1 else "C2", family, v3.VARIANTS[band]),
        [{"scenario_id": case}],
        models,
        dict.fromkeys(models, 100000),
    )
    producer = env.sources[1]
    task = producer.build(family, band * v3.BAND)
    ref = producer.perfect(task)
    actions = (
        list(ref["calls"])
        + [
            ("write_file", {"path": k, "content": v})
            for k, v in ref.get("files", {}).items()
        ]
        + [ref["answer"]]
    )
    spec = simulation(env, raw, [actions])
    (job,) = env.jobs
    _, native = xp.pool_verified_result(env.objects, job, spec)
    assert native["task_pass"] and native["eligible"]
    out = tmp_path / "data.zip"
    m = write_dataset(
        spec, env.db, env.objects, out, full_trajectories=True, prompt=PROMPT
    )
    with ZipFile(out) as z:
        row = json.loads(z.read("trajectories/" + job["job_id"] + ".jsonl"))
        assert (
            row["experiment"]["program"] == raw["program"]
            and row["experiment"]["ordinal"] == row["experiment"]["case_pattern"] == 0
        )
        assert (
            row["verified_task_pass"]
            and row["label_turns"]
            and m["trajectory_rows"] == 1
        )
        if family == "P8":
            assert row["sales_evidence"]["opened_sources"]


def test_finite_compiler_balances_cases_and_resumes_original_canary(tmp_path):
    agent = os.getenv("COLLECTION_TEST_AGENT_REPO")
    poet = os.getenv("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("pinned producer checkout unavailable")
    prompt = tmp_path / "prompt.json"
    write_new(prompt, PROMPT)
    d = tmp_path / "offers"
    i = tmp_path / "inputs"
    args = SimpleNamespace(
        champion_repo=agent,
        poet_repo=poet,
        prompt=prompt,
        directory=d,
        inputs=i,
        student_bundle=None,
        exports=None,
        environment_file=None,
        collection="C1",
        family=["P9"],
        start_pack=0,
        pack_count=1,
        cap_microusd=95000,
        cap_for=["deepseek/deepseek-v4.1-flash-1017=20000"],
    )
    (canary,) = compile_program(args)
    assert canary["slots"] == 24
    args.pack_count = None
    full = compile_program(args)
    assert (
        full[0] == canary and len(full) == 9 and sum(g["slots"] for g in full) == 4800
    )
    args.collection = "C2"
    variants = compile_program(args)
    assert sum(g["slots"] for g in variants) == 1600
    packs = [json.loads(p.read_bytes()) for p in d.glob("*.json")]
    by = {}
    for p in packs:
        v = p["program"]["variant"]
        by.setdefault(v, []).extend(
            v3.identity(c["scenario_id"])[2] for c in p["cases"]
        )
        if v == "base":
            assert (
                next(a for a in p["arms"] if "flash" in a["model"])["budget_microusd"]
                == 20000
            )
    assert sorted(by["base"]) == list(range(200)) and sorted(by["brief"]) == sorted(
        by["review"]
    ) == list(range(0, 200, 2))
    args.collection = "C1"
    args.cap_microusd += 1
    with pytest.raises(ValueError, match="equivalent funded slot"):
        compile_program(args)
    raw = packs[0]
    bad = deepcopy(raw)
    bad["cases"][0]["scenario_id"] = "P9-s40000000"
    with pytest.raises(ValueError):
        xp.load_experiment(bad, PROMPT)
    with pytest.raises(ValueError, match="exact untuned"):
        v3.hosted_student_route()


def test_signed_student_pass_failure_import_and_bulk_correction(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from lab_arena.collection_node_runs import verified_node_runs
    from lab_arena.collection_workflow import workflow_sources
    from scripts.collection_program_sources import sign_nodes

    agent = os.getenv("COLLECTION_TEST_AGENT_REPO")
    poet = os.getenv("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("pinned checkout unavailable")
    private = Ed25519PrivateKey.generate()
    pk = tmp_path / "node.pem"
    pub = tmp_path / "trusted.pem"
    pk.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pk.chmod(0o600)
    pub.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    monkeypatch.setenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", str(pub))
    inputs = []
    rev = v3.sandbox("P9")["builder"]
    with workflow_sources(agent, poet, corrections=True, builder_revision=rev) as (
        h,
        b,
    ):
        t = b.build("P9", 10000000)
        good = b.perfect(t)
        wrong = deepcopy(good)
        for item in wrong["calls"][-1][1]["plan"]:
            item["disposition"] = "release"
        for rep, plan in enumerate((good, wrong)):
            folder = tmp_path / f"run-{rep}"
            b.branch.fake_source(folder, ("P9", 10000000, b.PLAIN), t, plan)
            _, tools = b.branch.fresh(t, tmp_path / f"initial-{rep}")
            lines = list(
                map(json.loads, (folder / "transcript.jsonl").read_text().splitlines())
            )
            lines[0]["content"] = b.branch.run_any.SYSTEM.format(
                names=", ".join(t["function"]["name"] for t in tools),
                today=b.branch.run_any.TODAY_TEXT,
            )
            (folder / "transcript.jsonl").write_text(
                "".join(json.dumps(v) + "\n" for v in lines)
            )
            meta = json.loads((folder / "meta.json").read_bytes())
            meta.update(model="Qwen/Qwen3.5-4B", local=True, run=rep)
            (folder / "meta.json").write_text(json.dumps(meta))
            inputs.append(
                dict(
                    folder=folder.name,
                    producer_run_id=f"node-{rep}",
                    student_model=meta["model"],
                    model_revision=v3.STUDENT_REVISION,
                    repeat=rep,
                )
            )
    idx = tmp_path / "index.json"
    write_new(idx, inputs)
    bundle = tmp_path / "signed"
    prompt = tmp_path / "prompt.json"
    write_new(prompt, PROMPT)
    signed = sign_nodes(
        SimpleNamespace(
            runs=idx,
            family="P9",
            producer="poet-node",
            private_key=pk,
            output=bundle,
            champion_repo=agent,
            poet_repo=poet,
        )
    )
    assert signed["runs"] == 2 and signed["correction_sources"] == 1
    with workflow_sources(agent, poet, corrections=True, builder_revision=rev) as (
        h,
        b,
    ):
        rows = sorted(
            verified_node_runs(bundle, harness=h, builders=b, public_key=pub),
            key=lambda r: r["run_id"],
        )
        assert (
            [r["verified_task_pass"] for r in rows] == [True, False]
            and rows[1]["source_sha256"]
            and rows[1]["first_failed_check"]
        )
        assert rows[1]["state_history"]["states"] and rows[0]["model"] == v3.STUDENT
    target = tmp_path / "offers"
    args = SimpleNamespace(
        collection="C4",
        family=["P9"],
        cap_microusd=100000,
        cap_for=[],
        directory=target,
        inputs=tmp_path / "gateway",
        prompt=prompt,
        student_bundle=bundle,
        exports=None,
        environment_file=None,
        champion_repo=agent,
        poet_repo=poet,
        start_pack=0,
        pack_count=None,
    )
    (one,) = compile_program(args)
    assert one["slots"] == 4 and one["program"]["source_kind"] == "node_student"
    raw = json.loads((target / (one["batch"] + ".json")).read_bytes())
    spec = xp.load_experiment(raw, PROMPT)
    with workflow_sources(
        agent, poet, corrections=True, builder_revision=rev
    ) as sources:
        result = xp.read_inputs(spec, args.inputs, sources, prompt=PROMPT)
        assert list(result[0]) == [rows[1]["source_sha256"]]
    sig = json.loads((bundle / "nodes.json").read_bytes())
    sig["producer"] = "substitute"
    (bundle / "nodes.json").write_text(json.dumps(sig))
    with pytest.raises(ValueError, match="signature"):
        list(verified_node_runs(bundle, harness=h, builders=b, public_key=pub))


@pytest.mark.parametrize("pool_fixture", ["v3"], indirect=True)
def test_verified_followups_recover_after_interruption_and_map_only_exports(
    pool_fixture, tmp_path, monkeypatch
):
    from scripts import collection_program_schedule as compiler
    from scripts import prepare_collection_experiment as prep
    from scripts.collection_export_snapshots import observe
    from lab_arena.collection_node_runs import (
        restatement_verifier,
        verified_restatements,
        RESTATEMENTS,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    env = pool_fixture
    folder = tmp_path / "pilot"
    offers = folder / "paid/offers"
    offers.mkdir(parents=True)
    exports = tmp_path / "archives"
    monkeypatch.setenv("LAB_ARENA_COLLECTION_EXPORT_DIRECTORY", str(exports))
    case = "P9-s10000000"
    raw = _pack(
        v3.program("C1", "P9", "base"),
        [{"scenario_id": case}],
        v3.TEACHERS,
        dict.fromkeys(v3.TEACHERS, 100000),
    )
    raw["followups"] = ["C4", "C5"]
    spec = xp.load_experiment(raw, PROMPT)
    write_new(offers / (spec.batch_id + ".json"), raw)
    write_new(folder / "prompt.json", PROMPT)
    producer = env.sources[1]
    ref = producer.perfect(producer.build("P9", v3.BAND))
    yes = list(ref["calls"]) + [ref["answer"]]
    bad = deepcopy(yes)
    for r in bad[-2][1]["plan"]:
        r["disposition"] = "release"
    simulation(env, raw, [bad] * 6 + [yes] * 6 + [bad] * 6)
    observe(folder, exports, env.db, env.objects)
    args = dict(
        directory=offers,
        inputs=env.inputs,
        exports=exports,
        prompt=PROMPT,
        database=env.db,
        objects=env.objects,
    )
    unfinished = compiler.stage_followups(**args)
    assert all(
        p["program"]["collection"] != "C5" for d in unfinished for p in d["staged"]
    )
    simulation(env, raw, [yes] * 6)
    observe(folder, exports, env.db, env.objects)
    original = prep.store_private_json

    def fail_mark(path, value):
        if "program-followups" in path.parts:
            raise RuntimeError("interrupted after publish")
        original(path, value)

    with monkeypatch.context() as mp:
        mp.setattr(prep, "store_private_json", fail_mark)
        with pytest.raises(RuntimeError, match="interrupted after publish"):
            compiler.stage_followups(**args)
    staged = {p.stem: json.loads(p.read_bytes()) for p in offers.glob("pool-*.json")}
    resumed = compiler.stage_followups(**args)
    assert resumed and resumed[0]["staged"] == []
    assert compiler.stage_followups(**args) == [] and set(staged) == {
        p.stem for p in offers.glob("pool-*.json")
    }
    c4 = [r for r in staged.values() if r["program"]["collection"] == "C4"]
    assert 1 <= len(c4) <= 7 and all(
        r["max_slots"] == 4 and len(r["arms"]) == 2 for r in c4
    )
    (c5,) = [r for r in staged.values() if r["program"]["collection"] == "C5"]
    spec5 = simulation(env, c5, [yes, bad])
    out = tmp_path / "map.zip"
    m = write_dataset(
        spec5, env.db, env.objects, out, full_trajectories=True, prompt=PROMPT
    )
    with ZipFile(out) as z:
        assert not m["preference_pairs"] and not m.get("correction_pairs", 0)
        for j in env.jobs[-2:]:
            r = json.loads(z.read("trajectories/" + j["job_id"] + ".jsonl"))
            assert (
                r["experiment"]["reasoning_effort"] == "high"
                and r["experiment"]["program"]["use"] == "pass_rate_only"
            )
            assert r["payment_eligible"] and r["no_loss_turns"] == r["label_turns"]
    c4spec = simulation(env, c4[0], [yes[-2:]])
    destination = tmp_path / "recover.zip"
    md = write_dataset(
        c4spec, env.db, env.objects, destination, full_trajectories=True, prompt=PROMPT
    )
    assert md["correction_pairs"] == 1
    # The pinned independent node verifier rewinds a real released teacher turn.
    idx = json.loads((exports / "current.json").read_bytes())["snapshot"]
    group = json.loads((exports / "snapshots" / (idx + ".json")).read_bytes())[
        "offers"
    ][spec.batch_id][case]
    with ZipFile(exports / spec.batch_id / group["file"]) as z:
        teacher = next(
            r
            for r in (
                json.loads(z.read("trajectories/" + i + ".jsonl"))
                for i in group["released"]
            )
            if r["verified_task_pass"]
        )
    ver = restatement_verifier(producer)
    at = teacher["label_turns"][0]
    source = ver._teacher_state(teacher, at)
    bundle = tmp_path / "restatement"
    records = {}
    key = Ed25519PrivateKey.generate()
    keyfile = tmp_path / "signer.pem"
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    keyfile.chmod(0o600)
    for n in range(8):
        decision = deepcopy(teacher["messages"][at])
        rid = ver.run_id(
            "C6",
            source["checkpoint_sha256"] + "/" + source["teacher_trajectory_sha256"],
            n,
        )
        if n == 2:
            decision["tool_calls"][0]["function"]["name"] = "wrong_tool"
        reply = {
            "id": "fixture-" + rid,
            "model": ver.MODEL,
            "choices": [{"message": decision, "finish_reason": "tool_calls"}],
        }
        record = {
            **source,
            "schema": "poet.student-restatement.v1",
            "student_model": ver.MODEL,
            "model_revision": ver.REVISION,
            "repeat": n,
            "producer_run_id": rid,
            "provider_response": reply,
            "student_action": decision,
        }
        ref = xp._hash(record)
        records[ref] = rid
        prep.store_private_json(bundle / "restatements" / (ref[7:] + ".json"), record)

    def signed_index(m):
        value = prep.sign(
            {"schema": RESTATEMENTS, "producer": "training-node", "records": m}, keyfile
        )
        path = bundle / "restatements.json"
        path.write_text(json.dumps(value))
        path.chmod(0o600)

    signed_index(records)
    restored = list(
        verified_restatements(
            bundle,
            builders=producer,
            public_key=key.public_key(),
            teacher_by_id=lambda _: teacher,
        )
    )
    assert len(restored) == 7 and not any(r["repeat"] == 2 for r in restored)
    signed_index(dict(list(records.items())[:-1]))
    with pytest.raises(ValueError, match="incomplete eight-draw"):
        list(
            verified_restatements(
                bundle,
                builders=producer,
                public_key=key.public_key(),
                teacher_by_id=lambda _: teacher,
            )
        )


def test_registry_verifies_gate_signer_results_and_refuses_evaluation(
    monkeypatch, tmp_path
):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from scripts.prepare_collection_experiment import sign

    key = Ed25519PrivateKey.generate()
    private = tmp_path / "private"
    public = tmp_path / "pub"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    private.chmod(0o600)
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    entry = deepcopy(v3.sandbox("P9"))
    entry["family"] = "P77"
    item = {"artifact_sha256": "sha256:" + "a" * 64}
    entry["evidence"] = {
        "independent_solver": dict(item, tested=200, matched=200),
        "teacher": dict(item, tested=50, passed=45, misses_reviewed=5),
        "lazy_policies": dict(
            item, failures=["suppress_all", "release_all", "copy_first", "cite_nothing"]
        ),
        "patterns": dict(item, distinct=200),
        "student": dict(item, model=v3.STUDENT, tested=100, passed=23),
        "partition": dict(item, split_sha256=entry["partition"]),
    }
    doc = sign(entry, private)
    assert v3.verify_admission(doc, key.public_key()) == entry
    with pytest.raises(ValueError, match="signature"):
        v3.verify_admission({**doc, "kind": "evaluation"}, key.public_key())
    entry["evidence"]["teacher"]["passes"] = 45
    entry["evidence"]["teacher"]["misses_reviewed"] = 0
    with pytest.raises(ValueError, match="gates"):
        v3.verify_admission(sign(entry, private), key.public_key())
    entry.update(
        family="P88",
        kind="evaluation",
        allowed_variants=["heldout"],
        aliases={"heldout": {}},
        evidence={"partition": {"split_sha256": entry["partition"]}},
    )
    index = tmp_path / "registry"
    index.mkdir()
    write_new(index / "P88.json", sign(entry, private))
    monkeypatch.setenv("LAB_ARENA_SANDBOX_REGISTRY", str(index))
    monkeypatch.setenv("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE", str(public))
    assert v3.sandbox("P88")["kind"] == "evaluation"
    with pytest.raises(ValueError, match="evaluation"):
        v3.program("C1", "P88", "base")
