"""The required CI aggregate rejects missing, partial, and failed test shards."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.pytest_shard import collection_hash, shard_for, verify_manifests

ROOT = Path(__file__).resolve().parents[1]


def write_manifests(root, count=8):
    nodes = [f"tests/test_module_{index}.py::test_case" for index in range(200)]
    for index in range(count):
        selected = sorted(node for node in nodes if shard_for(node, count) == index)
        assert selected
        (root / f"shard-{index}.json").write_text(json.dumps({
            "schema_version": 1, "index": index, "count": count,
            "collection_count": len(nodes), "collection_hash": collection_hash(nodes),
            "selected_nodeids": selected, "executed_nodeids": selected,
            "results": {"passed": len(selected)}, "collect_only": False, "exitstatus": 0,
        }))
    return nodes


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
