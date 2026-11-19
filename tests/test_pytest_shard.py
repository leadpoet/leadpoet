"""The required CI aggregate rejects missing, partial, and failed test shards."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import pytest_shard
from scripts.pytest_shard import (
    assign, collection_hash, discover_modules, hashed_shard, load_timings, module_for,
    plan_modules, verify_manifests,
)
from scripts.update_pytest_shard_timings import median_timings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def use_timings(tmp_path, monkeypatch):
    def install(timings):
        path = tmp_path / "timings.json"
        path.write_text(json.dumps(timings))
        monkeypatch.setattr(pytest_shard, "TIMINGS_PATH", path)
    yield install


def write_manifests(root, count=8, durations=0, nodes=None):
    if nodes is None:
        nodes = [f"tests/test_module_{index}.py::test_case" for index in range(200)]
        nodes += [f"{module}::test_{case}" for module in sorted(load_timings())[:100] for case in range(3)]
    shards = assign(nodes, count, modules=discover_modules(ROOT))
    (root / "collection.json").write_text(json.dumps({
        "schema_version": 2, "collection_count": len(nodes),
        "collection_hash": collection_hash(nodes), "exitstatus": 0,
    }))
    for index in range(count):
        selected = sorted(node for node in nodes if shards[node] == index)
        (root / f"shard-{index}.json").write_text(json.dumps({
            "schema_version": 2, "index": index, "count": count,
            "selected_nodeids": selected, "executed_nodeids": selected,
            "results": {"passed": len(selected)} if selected else {},
            "collect_only": False, "exitstatus": 0,
            "module_durations_ms": {node.split("::")[0]: durations for node in selected},
        }))
    return nodes


def test_timed_modules_are_balanced_slowest_first_and_stay_whole(use_timings):
    use_timings({"a.py": 50, "b.py": 40, "c.py": 30, "d.py": 20, "e.py": 10})
    nodes = ["a.py::test_one", "b.py::test_one", "c.py::test_one", "c.py::Group::test_two",
             "d.py::test_one", "e.py::test_one"]
    assert assign(nodes, 2) == {
        "a.py::test_one": 0, "b.py::test_one": 1, "c.py::test_one": 1,
        "c.py::Group::test_two": 1, "d.py::test_one": 0, "e.py::test_one": 0,
    }


def test_module_slower_than_a_share_is_cut_into_sorted_contiguous_chunks(use_timings):
    use_timings({"big.py": 90, "small.py": 10})
    nodes = [f"big.py::test_{case}" for case in "hgfedcba"] + ["small.py::test_one", "small.py::test_two"]
    shards = assign(nodes, 4)
    chunks = [shards[f"big.py::test_{case}"] for case in "abcdefgh"]
    assert chunks == [0, 0, 1, 1, 2, 2, 3, 3]
    assert shards["small.py::test_one"] == shards["small.py::test_two"]
    assert assign(list(reversed(nodes)), 4) == shards


def test_exclusive_modules_run_only_in_the_last_shards(use_timings, monkeypatch, tmp_path):
    use_timings({"a.py": 50, "b.py": 40, "slow.py": 300})
    listed = tmp_path / "exclusive.txt"
    listed.write_text("# budgets\nslow.py\n")
    monkeypatch.setattr(pytest_shard, "EXCLUSIVE_PATH", listed)
    nodes = ["a.py::test_one", "b.py::test_one", "slow.py::test_one", "slow.py::test_two", "new.py::test_one"]
    shards = assign(nodes, 5, exclusive=2)
    assert shards["slow.py::test_one"] == shards["slow.py::test_two"] in {3, 4}
    assert {shards[node] for node in nodes if not node.startswith("slow")} <= {0, 1, 2}
    with pytest.raises(ValueError):
        assign(nodes, 1, exclusive=1)


def test_shard_that_collects_only_its_modules_selects_what_the_full_collection_would(use_timings):
    use_timings({"a.py": 50, "b.py": 40, "big.py": 90})
    nodes = [f"big.py::test_{case}" for case in "abcdef"] + ["a.py::test_one", "b.py::test_one"]
    modules = {"a.py", "b.py", "big.py", "empty_test.py"}
    full = assign(nodes, 4, modules=modules)
    plan = plan_modules(modules, 4)
    for index in range(4):
        own = {module for module, units in plan.items() if any(shard == index for shard, _, _ in units)}
        partial = assign([node for node in nodes if module_for(node) in own], 4, modules=modules)
        assert {n for n, s in partial.items() if s == index} == {n for n, s in full.items() if s == index}


def test_module_with_fewer_tests_than_pieces_leaves_a_shard_empty_and_still_verifies(use_timings, tmp_path):
    use_timings({"tests/big_test.py": 900, "tests/small_test.py": 10})
    nodes = ["tests/big_test.py::test_one", "tests/big_test.py::test_two", "tests/small_test.py::test_one"]
    write_manifests(tmp_path, count=4, nodes=nodes)
    assert any(not json.loads((tmp_path / f"shard-{i}.json").read_text())["selected_nodeids"] for i in range(4))
    assert verify_manifests(tmp_path, 4, modules=discover_modules(ROOT))["collected"] == 3


def test_committed_exclusive_modules_exist():
    assert all((ROOT / module).is_file() for module in pytest_shard.load_exclusive())


def test_new_module_uses_its_path_hash_and_moves_no_timed_module(use_timings):
    use_timings({"a.py": 50, "b.py": 40})
    timed = ["a.py::test_one", "b.py::test_one"]
    before = assign(timed, 8)
    digest = hashlib.sha256(b"tests/new_test.py").digest()[:8]
    after = assign(timed + ["tests/new_test.py::test_case"], 8)
    assert after["tests/new_test.py::test_case"] == hashed_shard("tests/new_test.py", 8)
    assert hashed_shard("tests/new_test.py", 8) == int.from_bytes(digest, "big") % 8
    assert {node: after[node] for node in timed} == before


@pytest.mark.parametrize("timings", [{}, [], {"a.py": -1}, {"a.py": 1.5}, {"a.py": True}])
def test_malformed_timings_fail_closed(use_timings, timings):
    use_timings(timings)
    with pytest.raises(ValueError):
        assign(["a.py::test_case"], 8)


def test_committed_timings_fill_all_shards_evenly():
    timings = load_timings()
    nodes = [f"{module}::test_{case:02}" for module in timings for case in range(40)]
    loads = [0.0] * 64
    for node, index in assign(nodes, 64).items():
        loads[index] += timings[node.split("::")[0]] / 40
    assert max(loads) - min(loads) <= 0.05 * max(loads)


def test_timing_refresh_takes_medians_of_verified_runs(tmp_path, monkeypatch):
    for run, duration in (("one", 10), ("two", 30), ("three", 20)):
        reports = tmp_path / run / "pytest-shard-0"
        reports.mkdir(parents=True)
        write_manifests(reports, durations=duration)
    monkeypatch.chdir(ROOT)
    timings = median_timings([tmp_path / "one", tmp_path / "two", tmp_path / "three"])
    assert timings["tests/test_module_0.py"] == 20

    (tmp_path / "one" / "pytest-shard-0" / "shard-3.json").unlink()
    with pytest.raises(ValueError):
        median_timings([tmp_path / "one"])


def test_full_union_is_verified_against_collection_and_execution(tmp_path):
    nodes = write_manifests(tmp_path)
    assert verify_manifests(tmp_path, 8, modules=discover_modules(ROOT)) == {
        "collected": len(nodes), "shards": 8, "results": {"passed": len(nodes)},
    }


@pytest.mark.parametrize("fault", [
    "missing", "extra", "failed", "exclusive", "collect_only", "unfinished", "duplicate",
    "wrong_shard", "different_collection", "reference_failed", "missing_reference",
    "missing_result", "invalid_result", "reported_failure",
])
def test_incomplete_or_failed_shards_cannot_pass(tmp_path, fault):
    write_manifests(tmp_path)
    path = tmp_path / "shard-0.json"
    state = json.loads(path.read_text())
    reference = tmp_path / "collection.json"
    if fault == "missing":
        path.unlink()
    elif fault == "missing_reference":
        reference.unlink()
    elif fault in {"different_collection", "reference_failed"}:
        collection = json.loads(reference.read_text())
        collection["collection_hash" if fault == "different_collection" else "exitstatus"] = (
            "0" * 64 if fault == "different_collection" else 2)
        reference.write_text(json.dumps(collection))
    elif fault == "extra":
        (tmp_path / "shard-8.json").write_text(json.dumps(state))
    else:
        if fault == "failed": state["exitstatus"] = 1
        elif fault == "exclusive": state["exclusive"] = 1
        elif fault == "collect_only": state["collect_only"] = True
        elif fault == "unfinished": state["executed_nodeids"] = state["executed_nodeids"][:-1]
        elif fault == "duplicate": state["selected_nodeids"] += state["selected_nodeids"][:1]
        elif fault == "wrong_shard": state["index"] = 1
        elif fault == "missing_result": state["results"]["passed"] -= 1
        elif fault == "invalid_result": state["results"] = {"invented": len(state["executed_nodeids"])}
        elif fault == "reported_failure":
            state["results"]["passed"] -= 1
            state["results"]["failed"] = 1
        path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8, modules=discover_modules(ROOT))


def run_shard(tmp_path, environment, index, *extra):
    return subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "scripts.pytest_shard",
        "--ci-shard-index", str(index), "--ci-shard-count", "8",
        "--ci-shard-manifest", str(tmp_path / f"shard-{index}.json"), *extra,
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)


def test_real_pytest_records_completed_pass_skip_and_failure(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    for index in range(64):
        (tmp_path / f"test_module_{index}.py").write_text(
            "import os, pytest\n"
            f"open(os.environ['IMPORT_LOG'], 'a').write('test_module_{index}.py\\n')\n"
            "def test_pass(): pass\n"
            "@pytest.mark.skip(reason='synthetic existing skip')\ndef test_skip(): pass\n"
        )
    modules = discover_modules(tmp_path)
    reference = subprocess.run([
        sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "scripts.pytest_shard",
        "--ci-shard-reference", str(tmp_path / "collection.json"),
    ], cwd=tmp_path, env={**environment, "IMPORT_LOG": str(tmp_path / "reference.log")},
        capture_output=True, text=True, timeout=30)
    assert reference.returncode == 0, reference.stdout + reference.stderr
    for index in range(8):
        log = tmp_path / f"import-{index}.log"
        result = run_shard(tmp_path, {**environment, "IMPORT_LOG": str(log)}, index)
        assert result.returncode == 0, result.stdout + result.stderr
        selected = json.loads((tmp_path / f"shard-{index}.json").read_text())["selected_nodeids"]
        # Each shard imports exactly the modules it runs.
        assert sorted(log.read_text().split()) == sorted({module_for(node) for node in selected})
    assert verify_manifests(tmp_path, 8, modules=modules)["results"] == {"passed": 64, "skipped": 64}
    recorded = {}
    for index in range(8):
        recorded.update(json.loads((tmp_path / f"shard-{index}.json").read_text())["module_durations_ms"])
    assert set(recorded) == {f"test_module_{index}.py" for index in range(64)}
    environment["IMPORT_LOG"] = os.devnull
    (tmp_path / "test_module_0.py").write_text("def test_failure(): assert False\n")
    index = hashed_shard("test_module_0.py", 8)
    result = run_shard(tmp_path, environment, index)
    assert result.returncode == 1
    state = json.loads((tmp_path / f"shard-{index}.json").read_text())
    assert state["results"]["failed"] == 1
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8, modules=modules)

    (tmp_path / "test_module_0.py").write_text(
        "import pytest\ndef test_exit(): pytest.exit('early exit', returncode=0)\n"
    )
    result = run_shard(tmp_path, environment, index)
    assert result.returncode == 1
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8, modules=modules)
