"""Rewrite the committed pytest shard timings from passing Deploy Checks runs.

Download the shard reports of a few recent passing runs, one directory per run,
skipping nightly runs because coverage slows them:

    gh run download RUN_ID -R leadpoet/leadpoet -p 'pytest-shard-*' -D runs/RUN_ID
    python -m scripts.update_pytest_shard_timings runs/*

Each module gets its median duration across the runs that include it. Every run
must have used the timing file in this checkout: verification rejects any other,
so after a refresh merges, wait for new runs before refreshing again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from statistics import median
import tempfile

from scripts.pytest_shard import TIMINGS_PATH, verify_manifests


def run_durations(run: Path) -> dict[str, int]:
    durations = {}
    with tempfile.TemporaryDirectory() as directory:
        reports = list(run.rglob("shard-*.json"))
        for report in reports:
            shutil.copy(report, directory)
        verify_manifests(Path(directory), len(reports))
        for report in Path(directory).glob("shard-*.json"):
            durations.update(json.loads(report.read_text(encoding="utf-8"))["module_durations_ms"])
    return durations


def median_timings(runs: list[Path]) -> dict[str, int]:
    samples: dict[str, list[int]] = {}
    for run in runs:
        for module, duration in run_durations(run).items():
            samples.setdefault(module, []).append(duration)
    return {module: round(median(samples[module])) for module in sorted(samples)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    TIMINGS_PATH.write_text(
        json.dumps(median_timings(args.runs), indent=0, sort_keys=True) + "\n", encoding="utf-8"
    )
