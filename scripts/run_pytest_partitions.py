"""Run several CI pytest partitions side by side and print each log in turn.

    python -m scripts.run_pytest_partitions INDEX [INDEX ...] -- PYTEST_ARGS

Each partition writes ci-results/pytest-INDEX.log, its manifest and its own
coverage file. Python starts the children, not a shell's ``&``: a background
job in a non-interactive shell ignores SIGINT, and tests that send SIGINT to a
child would hang. Exits nonzero if any partition fails.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

RESULTS = Path("ci-results")


def main(argv: list[str]) -> int:
    split = argv.index("--")
    indices, extra = argv[:split], argv[split + 1:]
    if not indices or not all(index.isdigit() for index in indices):
        raise SystemExit("usage: run_pytest_partitions INDEX [INDEX ...] -- PYTEST_ARGS")
    RESULTS.mkdir(exist_ok=True)
    running = []
    for index in indices:
        log = (RESULTS / f"pytest-{index}.log").open("wb")
        env = {**os.environ, "COVERAGE_FILE": str(RESULTS / f".coverage.shard-{index}")}
        command = [sys.executable, "-m", "pytest", *extra, f"--ci-shard-index={index}",
                   f"--ci-shard-manifest={RESULTS / f'shard-{index}.json'}"]
        running.append((index, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)))
    failed = False
    for index, log, process in running:
        failed |= process.wait() != 0
        log.close()
    for index, _, process in running:
        text = (RESULTS / f"pytest-{index}.log").read_text(encoding="utf-8", errors="replace")
        print(f"::group::Partition {index} (exit {process.returncode})\n{text.rstrip()}\n::endgroup::")
        print("\n".join(text.splitlines()[-3:]), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
