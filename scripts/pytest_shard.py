"""Partition pytest by module and verify that every collected test finished.

The plan is made from the test files on disk, so each shard process imports
only the files it runs. One reference process collects everything; the
verifier proves the shards ran exactly that collection.

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
import configparser
import hashlib
import json
from fnmatch import fnmatch
import math
import os
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


def discover_modules(rootdir: Path) -> set[str]:
    """Test files on disk per pytest.ini, as node ID module paths relative to the rootdir.

    A file pytest collects but this misses is collected by every shard and
    placed by hash, so a mismatch costs time, never coverage.
    """
    ini = configparser.ConfigParser()
    ini.read(rootdir / "pytest.ini")
    section = ini["pytest"] if ini.has_section("pytest") else {}
    testpaths = section.get("testpaths", "").split() or ["."]
    patterns = section.get("python_files", "").split() or ["test_*.py", "*_test.py"]
    modules = set()
    for testpath in testpaths:
        for directory, dirnames, filenames in os.walk(rootdir / testpath):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            modules.update(
                (Path(directory) / name).relative_to(rootdir).as_posix()
                for name in filenames if any(fnmatch(name, pattern) for pattern in patterns)
            )
    return modules


def _balance(modules, first: int, count: int, timings, split: bool) -> dict[str, list[tuple[int, int, int]]]:
    share = sum(timings.get(module, 0) for module in modules) / count or 1
    units, plan = [], {}
    for module in modules:
        if module not in timings:
            plan[module] = [(first + hashed_shard(module, count), 0, 1)]
            continue
        pieces = max(1, math.ceil(timings[module] / share)) if split else 1
        units.extend((timings[module] / pieces, module, piece, pieces) for piece in range(pieces))
    loads = [0.0] * count
    for weight, module, piece, pieces in sorted(units, key=lambda unit: (-unit[0], unit[1], unit[2])):
        index = min(range(count), key=lambda shard: (loads[shard], shard))
        plan.setdefault(module, []).append((first + index, piece, pieces))
        loads[index] += weight
    return plan


def plan_modules(modules, count: int, exclusive: int = 0) -> dict[str, list[tuple[int, int, int]]]:
    """Map each module to its (shard, piece, pieces) units; the last ``exclusive`` run alone.

    The plan depends only on module paths and the committed files, so a shard
    can skip importing every module that has no unit on it.
    """
    if not 0 <= exclusive < count:
        raise ValueError("exclusive shards must leave at least one shared shard")
    timings = load_timings()
    alone = load_exclusive() if exclusive else set()
    modules = sorted(set(modules))
    plan = _balance([m for m in modules if m not in alone], 0, count - exclusive, timings, split=True)
    # Exclusive modules stay whole: each split part would repeat their setup.
    if exclusive:
        plan.update(_balance([m for m in modules if m in alone], count - exclusive, exclusive, timings, split=False))
    return plan


def assign(nodeids, count: int, exclusive: int = 0, modules=()) -> dict[str, int]:
    """Map each node ID to its shard, planning over ``modules`` and the node IDs' own modules.

    A module is cut into contiguous chunks of its sorted node IDs.
    """
    by_module: dict[str, list[str]] = {}
    for nodeid in sorted(nodeids):
        by_module.setdefault(module_for(nodeid), []).append(nodeid)
    plan = plan_modules(set(modules) | set(by_module), count, exclusive)
    assignment = {}
    for module, nodes in by_module.items():
        for shard, piece, pieces in plan[module]:
            chunk = nodes[piece * len(nodes) // pieces:(piece + 1) * len(nodes) // pieces]
            assignment.update(dict.fromkeys(chunk, shard))
    return assignment


def collection_hash(nodeids) -> str:
    return hashlib.sha256(json.dumps(sorted(nodeids), separators=(",", ":")).encode()).hexdigest()


def pytest_addoption(parser):
    group = parser.getgroup("ci-shard")
    group.addoption("--ci-shard-index", type=int)
    group.addoption("--ci-shard-count", type=int)
    group.addoption("--ci-shard-manifest")
    group.addoption("--ci-shard-exclusive", type=int, default=0)
    group.addoption("--ci-shard-reference")


def pytest_configure(config):
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    if index is None or count is None or config.getoption("ci_shard_reference"):
        return
    config._ci_shard_modules = discover_modules(Path(config.rootpath))
    plan = plan_modules(config._ci_shard_modules, count, config.getoption("ci_shard_exclusive"))
    config._ci_shard_skip = {
        module for module, units in plan.items() if all(shard != index for shard, _, _ in units)
    }


def pytest_ignore_collect(collection_path, config):
    skip = getattr(config, "_ci_shard_skip", None)
    if skip is None or not collection_path.is_file():
        return None
    try:
        module = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    # None, not False: a planned module is skipped; anything else collects as usual.
    return True if module in skip else None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    reference = config.getoption("ci_shard_reference")
    if reference:
        nodeids = [item.nodeid for item in items]
        if not nodeids or len(nodeids) != len(set(nodeids)):
            raise pytest.UsageError("CI collection must be nonempty and have unique node IDs")
        config._ci_shard_reference = {
            "schema_version": 2, "collection_count": len(nodeids),
            "collection_hash": collection_hash(nodeids),
        }
        return
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    path = config.getoption("ci_shard_manifest")
    if index is None or count is None or not path or not 0 <= index < count:
        raise pytest.UsageError("CI shard index, positive count, and manifest are required")
    nodeids = [item.nodeid for item in items]
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise pytest.UsageError("CI collection must be nonempty and have unique node IDs")
    exclusive = config.getoption("ci_shard_exclusive")
    shards = assign(nodeids, count, exclusive, config._ci_shard_modules)
    # A shard can be empty when a slow module has fewer tests than pieces;
    # the verifier still proves the union of shards is the full collection.
    selected = [item for item in items if shards[item.nodeid] == index]
    config._ci_shard = {
        "schema_version": 2, "index": index, "count": count, "exclusive": exclusive,
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
    reference = getattr(session.config, "_ci_shard_reference", None)
    if reference is not None:
        reference["exitstatus"] = int(session.exitstatus)
        path = Path(session.config.getoption("ci_shard_reference"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reference, sort_keys=True) + "\n", encoding="utf-8")
        return
    state = getattr(session.config, "_ci_shard", None)
    if state is None:
        return
    state["executed_nodeids"].sort()
    if not state["selected_nodeids"] and session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
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


def verify_manifests(directory: Path, count: int, exclusive: int = 0, modules=None) -> dict:
    """Prove the shards ran the reference collection exactly once, per the plan.

    ``modules`` are the test files the shards planned over; by default the
    test files in the current checkout.
    """
    if modules is None:
        modules = discover_modules(Path.cwd())
    paths = sorted(directory.glob("shard-*.json"))
    if {path.name for path in paths} != {f"shard-{index}.json" for index in range(count)}:
        raise ValueError("missing or unexpected shard manifest")
    if not (directory / "collection.json").is_file():
        raise ValueError("missing reference collection")
    reference = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    if reference.get("schema_version") != 2 or reference.get("exitstatus") != 0:
        raise ValueError("reference collection did not complete")
    states = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    shards = assign([node for state in states for node in state["selected_nodeids"]], count, exclusive, modules)
    all_nodes, indices, results = [], set(), Counter()
    for path, state in zip(paths, states):
        index = state["index"]
        if (state["schema_version"] != 2 or state["count"] != count
                or state.get("exclusive", 0) != exclusive
                or index in indices or path.name != f"shard-{index}.json"
                or state["collect_only"] or state["exitstatus"] != 0):
            raise ValueError("shard did not complete successfully")
        selected, executed = state["selected_nodeids"], state["executed_nodeids"]
        if (selected != executed or len(selected) != len(set(selected))
                or any(shards.get(node) != index for node in selected)
                or not set(state["results"]).issubset({"passed", "failed", "skipped"})
                or any(type(value) is not int or value < 0 for value in state["results"].values())
                or sum(state["results"].values()) != len(executed)
                or state["results"].get("failed", 0)):
            raise ValueError("shard execution does not match its collection")
        indices.add(index)
        all_nodes.extend(selected)
        results.update(state["results"])
    if (len(all_nodes) != len(set(all_nodes))
            or len(all_nodes) != reference["collection_count"]
            or collection_hash(all_nodes) != reference["collection_hash"]):
        raise ValueError("shards do not cover the same full collection exactly once")
    return {"collected": len(all_nodes), "shards": count, "results": dict(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--exclusive", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(verify_manifests(args.directory, args.count, args.exclusive), sort_keys=True))
