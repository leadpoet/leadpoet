"""Partition pytest by module and verify that every collected test finished.

Modules in the committed timing file are balanced across shards, slowest first.
A timed module slower than one shard's even share is cut into that many
contiguous chunks of its sorted node IDs, so no single file sets the floor.
The verifier repeats the split from the manifests alone.
Any other module is placed by a hash of its path, so adding a test file never
moves an existing module. CI runs several shards side by side on one runner;
modules in the exclusive list hold timing budgets that contention would break,
so they go only to the last shards, which CI runs alone. scripts/update_pytest_shard_timings.py refreshes the file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pytest

TIMINGS_PATH = Path(__file__).with_name("pytest_shard_timings.json")
EXCLUSIVE_PATH = Path(__file__).with_name("pytest_shard_exclusive.txt")


def module_for(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def load_timings(path: Path | None = None) -> dict[str, int]:
    timings = json.loads((path or TIMINGS_PATH).read_text(encoding="utf-8"))
    if (not isinstance(timings, dict) or not timings
            or any(type(value) is not int or value < 0 for value in timings.values())):
        raise ValueError("pytest shard timings must map modules to milliseconds")
    return timings


def load_exclusive() -> set[str]:
    lines = EXCLUSIVE_PATH.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def hashed_shard(module: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(module.encode()).digest()[:8], "big") % count


def _balance(modules: dict[str, list[str]], first: int, count: int, timings) -> dict[str, int]:
    share = sum(timings.get(module, 0) for module in modules) / count or 1
    units, assignment = [], {}
    for module, nodes in modules.items():
        if module not in timings:
            assignment.update(dict.fromkeys(nodes, first + hashed_shard(module, count)))
            continue
        pieces = max(1, min(len(nodes), math.ceil(timings[module] / share)))
        for piece in range(pieces):
            chunk = nodes[piece * len(nodes) // pieces:(piece + 1) * len(nodes) // pieces]
            units.append((timings[module] / pieces, f"{module}#{piece}", chunk))
    loads = [0.0] * count
    for weight, _, chunk in sorted(units, key=lambda unit: (-unit[0], unit[1])):
        index = min(range(count), key=lambda shard: (loads[shard], shard))
        assignment.update(dict.fromkeys(chunk, first + index))
        loads[index] += weight
    return assignment


def assign(nodeids, count: int, exclusive: int = 0) -> dict[str, int]:
    """Map each collected node ID to its shard; the last ``exclusive`` run alone."""
    if not 0 <= exclusive < count:
        raise ValueError("exclusive shards must leave at least one shared shard")
    timings = load_timings()
    alone = load_exclusive() if exclusive else set()
    shared: dict[str, list[str]] = {}
    isolated: dict[str, list[str]] = {}
    for nodeid in sorted(nodeids):
        module = module_for(nodeid)
        (isolated if module in alone else shared).setdefault(module, []).append(nodeid)
    assignment = _balance(shared, 0, count - exclusive, timings)
    if isolated:
        assignment.update(_balance(isolated, count - exclusive, exclusive, timings))
    return assignment


def collection_hash(nodeids) -> str:
    return hashlib.sha256(json.dumps(sorted(nodeids), separators=(",", ":")).encode()).hexdigest()


def pytest_addoption(parser):
    group = parser.getgroup("ci-shard")
    group.addoption("--ci-shard-index", type=int)
    group.addoption("--ci-shard-count", type=int)
    group.addoption("--ci-shard-manifest")
    group.addoption("--ci-shard-exclusive", type=int, default=0)


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
    exclusive = config.getoption("ci_shard_exclusive")
    shards = assign(nodeids, count, exclusive)
    selected = [item for item in items if shards[item.nodeid] == index]
    if not selected:
        raise pytest.UsageError("CI shard is empty")
    config._ci_shard = {
        "schema_version": 1, "index": index, "count": count, "exclusive": exclusive,
        "collection_count": len(nodeids), "collection_hash": collection_hash(nodeids),
        "selected_nodeids": sorted(item.nodeid for item in selected),
        "executed_nodeids": [], "outcomes": {}, "module_seconds": Counter(),
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
    state["module_seconds"][module_for(item.nodeid)] += report.duration
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
    state["module_durations_ms"] = {
        module: round(seconds * 1000) for module, seconds in state.pop("module_seconds").items()
    }
    path = Path(session.config.getoption("ci_shard_manifest"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifests(directory: Path, count: int, exclusive: int = 0) -> dict:
    paths = sorted(directory.glob("shard-*.json"))
    if {path.name for path in paths} != {f"shard-{index}.json" for index in range(count)}:
        raise ValueError("missing or unexpected shard manifest")
    states = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    shards = assign([node for state in states for node in state["selected_nodeids"]], count, exclusive)
    all_nodes, totals, hashes, indices, results = [], set(), set(), set(), Counter()
    for path, state in zip(paths, states):
        index = state["index"]
        if (state["schema_version"] != 1 or state["count"] != count
                or state.get("exclusive", 0) != exclusive
                or index in indices or path.name != f"shard-{index}.json"
                or state["collect_only"] or state["exitstatus"] != 0):
            raise ValueError("shard did not complete successfully")
        selected, executed = state["selected_nodeids"], state["executed_nodeids"]
        if (not selected or selected != executed or len(selected) != len(set(selected))
                or any(shards.get(node) != index for node in selected)
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
    parser.add_argument("--exclusive", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(verify_manifests(args.directory, args.count, args.exclusive), sort_keys=True))
