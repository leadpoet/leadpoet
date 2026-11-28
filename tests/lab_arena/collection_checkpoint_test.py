"""Exercise the real, pinned Poet replay and public checkpoint harness."""

import copy
import json
import os
from pathlib import Path

import pytest

from lab_arena.collection_checkpoint import digest
from lab_arena.collection_workflow import (
    correction_scenario,
    prepare_student_source,
    play_workflow,
    validate_student_source,
    workflow_sources,
)


@pytest.fixture(scope="module")
def runtime():
    agent, poet = (
        os.environ.get(k)
        for k in ("COLLECTION_TEST_AGENT_REPO", "COLLECTION_TEST_POET_REPO")
    )
    if not agent or not poet:
        pytest.skip("requires both pinned training source repositories")
    with workflow_sources(agent, poet, corrections=True) as sources:
        yield sources


def test_real_native_replay_accepts_only_the_verified_pre_turn_state(runtime, tmp_path):
    harness, builders = runtime
    br = builders.branch
    src = tmp_path / "fixture"
    task = builders.build("C06", 3013)
    job = ("C06", 3013, builders.PLAIN)
    bad = next(x[1] for x in br.broken_of(task) if isinstance(x[1], dict))
    br.fake_source(src, job, task, bad)
    original_check = json.loads((src / "check.json").read_text())
    assert original_check["score"] < 1 or original_check.get("hard_fail")
    # The private Poet selftest supplies scripted, local actions. Use the same
    # initial prompt that its native student runner writes on actual model runs.
    _, tools = br.fresh(task, tmp_path / "initial")
    lines = [json.loads(s) for s in (src / "transcript.jsonl").read_text().splitlines()]
    lines[0]["content"] = br.run_any.SYSTEM.format(
        names=", ".join(t["function"]["name"] for t in tools),
        today=br.run_any.TODAY_TEXT,
    )
    (src / "transcript.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    meta = json.loads((src / "meta.json").read_text())
    meta["local"] = True
    (src / "meta.json").write_text(json.dumps(meta))
    source = prepare_student_source(harness, builders, src, model=meta["model"])
    assert validate_student_source(harness, builders, source) == source
    assert source["checkpoint_sha256"] == digest(source["checkpoint"])
    assert source["split"]["rule_pool"] == "poet:train_rules:chain"
    assert source["split"]["tool_variant"] == "contacts,functions,plain"
    tampered = copy.deepcopy(source)
    tampered["checkpoint"]["state"]["forged"] = True
    tampered["checkpoint_sha256"] = digest(tampered["checkpoint"])
    with pytest.raises(ValueError, match="match its replay"):
        validate_student_source(harness, builders, tampered)
    fresh = correction_scenario(builders, source)
    assert fresh == task
    before = source["checkpoint"]
    good = builders.perfect(task)
    todo = br.still_to_do(
        br.reference_plan(task, good, tmp_path / "good"), before["log"]
    )
    ask = br.scripted_expert(todo, good.get("answer") or "Done.")
    events = []

    class Expert:
        def model_call(self, rid, body):
            response = ask(None, None, body, None, None)
            # Provider's original id is per-generation, even when its mock
            # helper happens to reuse reference tool ids across continuations.
            choice = response["choices"][0]
            for c in choice["message"].get("tool_calls", []):
                c["id"] = "expert-" + rid
            return dict(response, model=body["model"], id=rid)

    row, p = play_workflow(
        harness,
        builders,
        {
            "job_id": "test-source-branch",
            "model": "fixture/expert",
            "scenario_id": source["scenario_id"],
        },
        Expert(),
        continuation=source,
        record=events.append,
    )
    assert p["checks"]["code_grade"]
    assert row["correction"]["correction_step_checks"]["same_pre_state"]
    assert (
        events[1]["state"] == before["state"] and events[1]["files"] == before["files"]
    )
    assert all(
        i in row["no_loss_turns"]
        for i in row["label_turns"]
        if i < len(before["messages"])
    )
    assert sorted(e["index"] for e in events) == list(range(len(events)))
    assert row["evidence"]["out"]["answer"] == (good.get("answer") or "Done.")
    assert p["split"] == source["split"]

    # The private importer never accepts an altered tool return even if the
    # alleged final answer/grade and reviewer-provided checkpoint are unchanged.
    tool = next(line for line in lines if line["role"] == "tool")
    tool["content"] += "altered"
    (src / "transcript.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    with pytest.raises(ValueError, match="no replayable failed"):
        prepare_student_source(harness, builders, src, model=meta["model"])


def test_provider_optional_tool_index_preserves_decision_bytes():
    from lab_arena.collection_checkpoint import _ordered_calls

    native = {
        "id": "call-a",
        "type": "function",
        "function": {"name": "read", "arguments": '{"a": 1}'},
    }
    provider = dict(native, index=0)
    assert _ordered_calls({"tool_calls": [provider]}) == [native]
    assert provider["index"] == 0
    assert _ordered_calls({"tool_calls": [dict(provider, unknown="extension")]}) != [
        native
    ]
    for wrong in (1, False, "0"):
        with pytest.raises(ValueError, match="order"):
            _ordered_calls({"tool_calls": [dict(provider, index=wrong)]})


def test_pinned_private_subtree_needs_no_other_project_blobs(tmp_path):
    import subprocess
    from lab_arena.collection_workflow import _materialize

    full = tmp_path / "full"
    partial = tmp_path / "partial.git"
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["git", "init", "-q", str(full)], check=True)
    for directory in ("harness/gtm-tasks", "unrelated"):
        (full / directory).mkdir(parents=True)
        (full / directory / "sample").write_text(directory)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(full), *args])
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.org",
        "commit", "-qm", "fixture")
    rev = git("rev-parse", "HEAD").decode().strip()
    objs = [rev + suffix for suffix in ("", "^{tree}", ":harness", ":harness/gtm-tasks", ":harness/gtm-tasks/sample")]
    objects = b"".join(git("rev-parse", obj) for obj in objs)
    pack = subprocess.check_output(["git", "-C", str(full), "pack-objects", "--stdout"], input=objects)
    subprocess.run(["git", "init", "--bare", "-q", str(partial)], check=True)
    subprocess.run(["git", "-C", str(partial), "index-pack", "--stdin"], input=pack, check=True, stdout=subprocess.DEVNULL)
    _materialize(partial, rev, out, "harness/gtm-tasks")
    assert (out / "harness/gtm-tasks/sample").read_text() == "harness/gtm-tasks"
    assert not (out / "unrelated").exists()
