"""Partition pytest by module and verify that every collected test finished."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest


def shard_for(nodeid: str, count: int) -> int:
    module = nodeid.split("::", 1)[0]
    return int.from_bytes(hashlib.sha256(module.encode()).digest()[:8], "big") % count


def collection_hash(nodeids) -> str:
    return hashlib.sha256(json.dumps(sorted(nodeids), separators=(",", ":")).encode()).hexdigest()


def pytest_addoption(parser):
    group = parser.getgroup("ci-shard")
    group.addoption("--ci-shard-index", type=int)
    group.addoption("--ci-shard-count", type=int)
    group.addoption("--ci-shard-manifest")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    path = config.getoption("ci_shard_manifest")
    if index is None or count is None or not path or not 0 <= index < count:
        raise pytest.UsageError("CI shard index, positive count, and manifest are required")
    nodeids = [item.nodeid for item in items]
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise pytest.UsageError("CI collection must be nonempty and have unique node IDs")
    selected = [item for item in items if shard_for(item.nodeid, count) == index]
    if not selected:
        raise pytest.UsageError("CI shard is empty")
    config._ci_shard = {
        "schema_version": 1, "index": index, "count": count,
        "collection_count": len(nodeids), "collection_hash": collection_hash(nodeids),
        "selected_nodeids": sorted(item.nodeid for item in selected),
        "executed_nodeids": [], "outcomes": {},
    }
    selected_ids = {item.nodeid for item in selected}
    config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in selected_ids])
    items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    state = getattr(item.config, "_ci_shard", None)
    if state is None:
        return
    if report.failed or report.when == "call" or report.skipped:
        state["outcomes"][item.nodeid] = report.outcome
    if report.when == "teardown":
        state["executed_nodeids"].append(item.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    state = getattr(session.config, "_ci_shard", None)
    if state is None:
        return
    state["executed_nodeids"].sort()
    state["collect_only"] = bool(session.config.option.collectonly)
    if (not state["collect_only"] and
            state["executed_nodeids"] != state["selected_nodeids"]):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    state["exitstatus"] = int(session.exitstatus)
    state["results"] = dict(Counter(state.pop("outcomes").values()))
    path = Path(session.config.getoption("ci_shard_manifest"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifests(directory: Path, count: int) -> dict:
    paths = sorted(directory.glob("shard-*.json"))
    if {path.name for path in paths} != {f"shard-{index}.json" for index in range(count)}:
        raise ValueError("missing or unexpected shard manifest")
    all_nodes, totals, hashes, indices, results = [], set(), set(), set(), Counter()
    for path in paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        index = state["index"]
        if (state["schema_version"] != 1 or state["count"] != count
                or index in indices or path.name != f"shard-{index}.json"
                or state["collect_only"] or state["exitstatus"] != 0):
            raise ValueError("shard did not complete successfully")
        selected, executed = state["selected_nodeids"], state["executed_nodeids"]
        if (not selected or selected != executed or len(selected) != len(set(selected))
                or any(shard_for(node, count) != index for node in selected)
                or not set(state["results"]).issubset({"passed", "failed", "skipped"})
                or any(type(value) is not int or value < 0 for value in state["results"].values())
                or sum(state["results"].values()) != len(executed)
                or state["results"].get("failed", 0)):
            raise ValueError("shard execution does not match its collection")
        indices.add(index)
        totals.add(state["collection_count"])
        hashes.add(state["collection_hash"])
        all_nodes.extend(selected)
        results.update(state["results"])
    if (len(totals) != 1 or len(hashes) != 1
            or len(all_nodes) != len(set(all_nodes))
            or len(all_nodes) != next(iter(totals))
            or collection_hash(all_nodes) != next(iter(hashes))):
        raise ValueError("shards do not cover the same full collection exactly once")
    return {"collected": len(all_nodes), "shards": count, "results": dict(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_manifests(args.directory, args.count), sort_keys=True))
