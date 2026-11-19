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
    collection_hash, load_timings, shard_for, timed_assignment, verify_manifests,
)
from scripts.update_pytest_shard_timings import median_timings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def use_timings(tmp_path, monkeypatch):
    def install(timings):
        path = tmp_path / "timings.json"
        path.write_text(json.dumps(timings))
        monkeypatch.setattr(pytest_shard, "TIMINGS_PATH", path)
        timed_assignment.cache_clear()
    yield install
    timed_assignment.cache_clear()


def write_manifests(root, count=8, durations=0):
    nodes = [f"tests/test_module_{index}.py::test_case" for index in range(200)]
    nodes += [f"{module}::test_case" for module in sorted(load_timings())[:100]]
    for index in range(count):
        selected = sorted(node for node in nodes if shard_for(node, count) == index)
        assert selected
        (root / f"shard-{index}.json").write_text(json.dumps({
            "schema_version": 1, "index": index, "count": count,
            "collection_count": len(nodes), "collection_hash": collection_hash(nodes),
            "selected_nodeids": selected, "executed_nodeids": selected,
            "results": {"passed": len(selected)}, "collect_only": False, "exitstatus": 0,
            "module_durations_ms": {node.split("::")[0]: durations for node in selected},
        }))
    return nodes


def test_timed_modules_are_balanced_slowest_first_and_stay_whole(use_timings):
    use_timings({"a.py": 50, "b.py": 40, "c.py": 30, "d.py": 20, "e.py": 10})
    assert timed_assignment(2) == {"a.py": 0, "b.py": 1, "c.py": 1, "d.py": 0, "e.py": 0}
    assert shard_for("c.py::test_one", 2) == shard_for("c.py::Group::test_two", 2) == 1


def test_new_module_uses_its_path_hash_and_moves_no_timed_module(use_timings):
    use_timings({"a.py": 50, "b.py": 40})
    before = timed_assignment(8)
    digest = hashlib.sha256(b"tests/new_test.py").digest()[:8]
    assert shard_for("tests/new_test.py::test_case", 8) == int.from_bytes(digest, "big") % 8
    assert timed_assignment(8) == before


@pytest.mark.parametrize("timings", [{}, [], {"a.py": -1}, {"a.py": 1.5}, {"a.py": True}])
def test_malformed_timings_fail_closed(use_timings, timings):
    use_timings(timings)
    with pytest.raises(ValueError):
        shard_for("a.py::test_case", 8)


def test_committed_timings_fill_all_shards_evenly():
    timings = load_timings()
    loads = [0] * 8
    for module, index in timed_assignment(8).items():
        loads[index] += timings[module]
    assert max(loads) - min(loads) <= 0.02 * max(loads)


def test_timing_refresh_takes_medians_of_verified_runs(tmp_path):
    for run, duration in (("one", 10), ("two", 30), ("three", 20)):
        reports = tmp_path / run / "pytest-shard-0"
        reports.mkdir(parents=True)
        write_manifests(reports, durations=duration)
    timings = median_timings([tmp_path / "one", tmp_path / "two", tmp_path / "three"])
    assert timings["tests/test_module_0.py"] == 20

    (tmp_path / "one" / "pytest-shard-0" / "shard-3.json").unlink()
    with pytest.raises(ValueError):
        median_timings([tmp_path / "one"])


def test_full_union_is_verified_against_collection_and_execution(tmp_path):
    nodes = write_manifests(tmp_path)
    assert verify_manifests(tmp_path, 8) == {
        "collected": len(nodes), "shards": 8, "results": {"passed": len(nodes)},
    }


@pytest.mark.parametrize("fault", [
    "missing", "extra", "failed", "collect_only", "unfinished", "duplicate",
    "wrong_shard", "different_collection", "missing_result", "invalid_result", "reported_failure",
])
def test_incomplete_or_failed_shards_cannot_pass(tmp_path, fault):
    write_manifests(tmp_path)
    path = tmp_path / "shard-0.json"
    state = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "extra":
        (tmp_path / "shard-8.json").write_text(json.dumps(state))
    else:
        if fault == "failed": state["exitstatus"] = 1
        elif fault == "collect_only": state["collect_only"] = True
        elif fault == "unfinished": state["executed_nodeids"] = state["executed_nodeids"][:-1]
        elif fault == "duplicate": state["selected_nodeids"] += state["selected_nodeids"][:1]
        elif fault == "wrong_shard": state["index"] = 1
        elif fault == "different_collection": state["collection_hash"] = "0" * 64
        elif fault == "missing_result": state["results"]["passed"] -= 1
        elif fault == "invalid_result": state["results"] = {"invented": len(state["executed_nodeids"])}
        elif fault == "reported_failure":
            state["results"]["passed"] -= 1
            state["results"]["failed"] = 1
        path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8)


def test_real_pytest_records_completed_pass_skip_and_failure(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    for index in range(64):
        (tmp_path / f"test_module_{index}.py").write_text(
            "import pytest\ndef test_pass(): pass\n"
            "@pytest.mark.skip(reason='synthetic existing skip')\ndef test_skip(): pass\n"
        )
    for index in range(8):
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-q", "-p", "scripts.pytest_shard",
            "--ci-shard-index", str(index), "--ci-shard-count", "8",
            "--ci-shard-manifest", str(tmp_path / f"shard-{index}.json"),
        ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    assert verify_manifests(tmp_path, 8)["results"] == {"passed": 64, "skipped": 64}
    recorded = {}
    for index in range(8):
        recorded.update(json.loads((tmp_path / f"shard-{index}.json").read_text())["module_durations_ms"])
    assert set(recorded) == {f"test_module_{index}.py" for index in range(64)}
    (tmp_path / "test_module_0.py").write_text("def test_failure(): assert False\n")
    index = shard_for("test_module_0.py::test_failure", 8)
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "scripts.pytest_shard",
        "--ci-shard-index", str(index), "--ci-shard-count", "8",
        "--ci-shard-manifest", str(tmp_path / f"shard-{index}.json"),
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    state = json.loads((tmp_path / f"shard-{index}.json").read_text())
    assert state["results"]["failed"] == 1
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8)

    (tmp_path / "test_module_0.py").write_text(
        "import pytest\ndef test_exit(): pytest.exit('early exit', returncode=0)\n"
    )
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "scripts.pytest_shard",
        "--ci-shard-index", str(index), "--ci-shard-count", "8",
        "--ci-shard-manifest", str(tmp_path / f"shard-{index}.json"),
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, 8)
