import json
from pathlib import Path
from zipfile import ZipFile
import pytest
from scripts import freeze_sales_sandbox as f


def row(
    run, model, rep, passed, *, family="P9", variant="base", ordinal=18, student=False
):
    band = {"base": 1, "brief": 2, "review": 3}.get(variant, 4)
    scenario = f"{family}-s{band*10_000_000+ordinal}"
    from lab_arena import collection_programs as programs

    entry = programs.sandbox(family)
    messages = [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": scenario},
        {
            "role": "assistant",
            "content": "hold",
            "reasoning": "The recipient withdrew consent.",
        },
    ]
    return {
        "schema": "collection.trajectory.v1",
        "run_id": run,
        "batch_id": "pool-" + "f" * 64,
        "scenario_id": scenario,
        "split": {
            "template_id": family,
            "rule_pool": "fixture",
            "tool_variant": variant,
            "manifest_sha256": entry["partition"].removeprefix("sha256:"),
        },
        "builder_commit": entry["builder"],
        "harness_commit": "2" * 40,
        "manifest_sha256": "sha256:" + "3" * 64,
        "terms_sha256": "sha256:" + "4" * 64,
        "model": model,
        "model_revision": f.STUDENT_REV if student else model + "-pinned",
        "messages": messages,
        "tools": [],
        "label_turns": [2],
        "no_loss_turns": [],
        "verified_complete": True,
        "verified_task_pass": passed,
        "partial": False,
        "final_turn_present": True,
        "checks": {"training_split": True},
        "native_grade": {"score": int(passed)},
        "first_failed_check": (
            None
            if passed
            else {"check": "suppressed_contact_released", "turn": 0, "kind": "native"}
        ),
        "repeat_group_id": f"{model}-{family}-{variant}-{ordinal}",
        "experiment": {
            "role": "student" if student else "expert",
            "repeat": rep,
            "reasoning_effort": "low",
            "ordinal": ordinal,
            "case_pattern": ordinal % 200,
            "program": {
                "schema": "collection.sales-program.v3",
                "registry_sha256": "sha256:" + f.digest(entry),
                "collection": "C3" if student else "C1",
                "sandbox": family,
                "variant": variant,
                "use": "training",
            },
        },
    }


def test_shared_rows_materialize_genuine_training_objectives(tmp_path, monkeypatch):
    teachers = [
        row("c", f.CLOSE, 1, True),
        row("f", f.FAR, 1, True),
        row("expert-negative", f.FAR, 2, False),
    ]
    for run in ("correction", "student-correction"):
        item = row(run, f.FAR, 1, True)
        item["experiment"]["role"] = "continuation"
        item["experiment"]["program"].update(
            collection="C4",
            source_kind=(
                "node_student" if run.startswith("student") else "funded_failure"
            ),
        )
        teachers.append(item)

    def stream(*_):
        for teacher in teachers:
            if teacher["experiment"]["role"] == "continuation":
                prior = (
                    f.STUDENT if teacher["run_id"].startswith("student") else f.CLOSE
                )
                yield teacher, {
                    "run_id": teacher["run_id"],
                    "scenario_id": teacher["scenario_id"],
                    "split": teacher["split"],
                    "context": teacher["messages"][:2],
                    "chosen": teacher["messages"][2],
                    "tools": [],
                    "rejected": {"role": "assistant", "content": "release"},
                    "student_model": prior,
                    "source_sha256": "aa",
                    "checkpoint_sha256": "bb",
                    "step_checks": {"same_pre_state": True},
                }, "arc"
            else:
                yield teacher, None, "arc"

    monkeypatch.setattr(f, "packs", stream)
    students = [
        row(
            f"s-{family}-{variant}-{i}",
            f.STUDENT,
            i,
            i < 4,
            family=family,
            variant=variant,
            student=True,
        )
        for family, variant in (
            ("P9", "base"),
            ("P9", "brief"),
            ("P9", "review"),
            ("P10", "base"),
        )
        for i in range(8)
    ]
    dest = tmp_path / "out"
    restate = row("restate", f.STUDENT, 0, True, student=True)
    m = f.freeze(
        tmp_path,
        dest,
        snapshot="1" * 64,
        node_rows=[(s, "node") for s in students],
        restated=[restate],
    )
    for arm, entry in m["arm_counts"].items():
        assert entry["examples"] > 0, arm
        out = tmp_path / f"{arm}.jsonl"
        assert f.materialize(dest, arm, out) == entry["examples"]
    a5 = json.loads((tmp_path / "A5.jsonl").read_text())
    a6 = json.loads((tmp_path / "A6.jsonl").read_text())
    assert a5["run_id"] == a6["run_id"] == "c"
    assert (
        a5["messages"][-1]["reasoning_content"] == ""
        and a6["messages"][-1]["reasoning_content"]
    )
    a8 = json.loads((tmp_path / "A8.jsonl").read_text())
    assert a8["chosen"]["messages"][:-1] == a8["rejected"]["messages"][:-1]
    assert a8["chosen"]["messages"][-1]["content"] == "hold" and a8["chosen"][
        "label_turns"
    ] == [2]
    a10 = [json.loads(l) for l in (tmp_path / "A10.jsonl").read_text().splitlines()]
    bad = next(x for x in a10 if "first failed" in x["messages"][-2].get("content", ""))
    assert (
        bad["label_turns"] == [4]
        and "suppressed_contact" in bad["messages"][-1]["content"]
    )
    assert all(
        len(json.loads(l)["members"]) == 8
        for l in (tmp_path / "A14.jsonl").read_text().splitlines()
    )
    assert (
        f.materialize(dest, "A12", tmp_path / "control", control=True)
        == m["arm_counts"]["A12"]["examples"]
    )
    with ZipFile(dest / "sources.zip") as z:
        assert json.loads(z.read("runs/c.jsonl"))["messages"] == teachers[0]["messages"]
    (dest / "arms/A5.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="recipe hash"):
        f.materialize(dest, "A5", tmp_path / "tampered")


def test_no_heldout_high_effort_or_counterfactual_leakage():
    for variant, seed in (("base", 101), ("heldout", 40_000_018), ("base", 40_000_018)):
        r = row("r", f.FAR, 1, True)
        r["scenario_id"] = f"P9-s{seed}"
        r["experiment"]["program"]["variant"] = variant
        with pytest.raises(ValueError):
            f.observation(r, 256)
    r = row("r", f.FAR, 1, True)
    r["experiment"].update(reasoning_effort="high")
    with pytest.raises(ValueError, match="high-effort"):
        f.observation(r, 256)
    r["experiment"]["program"].update(collection="C5", use="pass_rate_only")
    assert not any(
        f.make_arms([f.observation(r, 256)], {}, set(), "P9", "P10").values()
    )
    r = row("wrong", f.STUDENT, 0, True, student=True)
    r["model_revision"] = "untested"
    with pytest.raises(ValueError, match="approved untuned"):
        f.observation(r, 256, node=True)


def test_private_index_rechecks_archive_and_excludes_quarantine(tmp_path):
    from hashlib import sha256
    from zipfile import ZipFile

    root = tmp_path / "exports"
    batch = "pool-" + "f" * 64
    row1 = row("good", f.FAR, 1, True)
    bad = row("held", f.FAR, 2, False)
    (root / "snapshots").mkdir(parents=True)
    folder = root / batch
    folder.mkdir()
    archive = folder / ("a" * 64 + ".zip")
    with ZipFile(archive, "x") as z:
        z.writestr("metadata.json", f.encoded({"dataset_sha256": "sha256:" + "a" * 64}))
        z.writestr("trajectories/good.jsonl", f.encoded(row1))
        z.writestr("trajectories/held.jsonl", f.encoded(bad))
    group = {
        "file": archive.name,
        "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
        "dataset_sha256": "sha256:" + "a" * 64,
        "released": ["good"],
        "held": {"held": "quarantine"},
    }
    idx = f.encoded(
        {
            "schema": "collection.private-export-index.v1",
            "offers": {batch: {row1["scenario_id"]: group}},
        }
    )
    ident = sha256(idx).hexdigest()
    index = root / "snapshots" / (ident + ".json")
    index.write_bytes(idx)
    index.chmod(0o600)
    assert [record[0]["run_id"] for record in f.packs(root, ident)] == ["good"]
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="archive hash"):
        list(f.packs(root, ident))
