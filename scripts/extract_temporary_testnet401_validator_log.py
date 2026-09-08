#!/usr/bin/env python3
"""Return public markers from the temporary validator's bounded private log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat
import sys

CONFIG = Path("/run/leadpoet-testnet401/config.json")
REPO = Path("/home/ec2-user/leadpoet/leadpoet")
PROCESS = "validator_application"
MAX_BYTES = 32 * 1024 * 1024
PATTERNS = (
    ("block", re.compile(rb"Block: ([0-9]{1,20}) \(block ([0-9]{1,20})/")),
    (
        "submission",
        re.compile(rb"gateway bundle persisted: (sha256:[0-9a-f]{13})\.\.\."),
    ),
    (
        "finalization",
        re.compile(rb"finalized chain state persisted: (sha256:[0-9a-f]{13})\.\.\."),
    ),
    ("completed", re.compile(rb"Successfully submitted weights to Bittensor chain")),
    (
        "tick",
        re.compile(
            rb'^\{"event": "automatic_weight_tick", "submitted_or_already_complete": true\}$'
        ),
    ),
)
START = re.compile(rb"SUBMITTING WEIGHTS FOR EPOCH ([0-9]{1,20})")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--epoch-id", required=True)
    args = parser.parse_args()
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha)), "candidate invalid"
    )
    require(bool(re.fullmatch(r"[0-9]{1,20}", args.epoch_id)), "epoch invalid")
    epoch = int(args.epoch_id)

    sys.path.insert(0, str(REPO))
    from scripts import bootstrap_temporary_testnet_weights_host as native

    config = native.load_config(CONFIG)
    require(
        (config["run_id"], config["candidate_sha"], config["expected_instance_id"])
        == (args.run_id, args.candidate_sha, args.instance_id),
        "host identity differs",
    )
    state = native._load_process_state(config)
    processes = [item for item in state["processes"] if item.get("name") == PROCESS]
    require(
        len(processes) == 1 and native._same_process(processes[0]), "process differs"
    )

    path = Path(config["runtime_root"]) / "logs" / f"{PROCESS}.log"
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and 0 < metadata.st_size <= MAX_BYTES,
        "log invalid",
    )
    matches = []
    current = None
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            start = START.search(line)
            if start:
                current = {"epoch_id": int(start.group(1)), "lines": [number]}
                continue
            if current is None or current["epoch_id"] != epoch:
                continue
            expected_name, pattern = PATTERNS[len(current["lines"]) - 1]
            found = pattern.search(line)
            if not found:
                continue
            current["lines"].append(number)
            if expected_name == "block":
                current.update(
                    block=int(found.group(1)), epoch_block=int(found.group(2))
                )
            elif expected_name in {"submission", "finalization"}:
                current[expected_name] = found.group(1).decode("ascii")
            if expected_name == "tick":
                matches.append(current)
                current = None
    identities = {(item["submission"], item["finalization"]) for item in matches}
    require(len(identities) == 1, "exact epoch markers unavailable")
    selected = matches[-1]
    print(
        json.dumps(
            {
                "status": "matched",
                "epoch_id": epoch,
                "process_name": PROCESS,
                "cmdline_hash": processes[0]["cmdline_hash"],
                "block": selected["block"],
                "epoch_block": selected["epoch_block"],
                "weight_submission_event_hash_prefix": selected["submission"],
                "weight_finalization_event_hash_prefix": selected["finalization"],
                "marker_line_numbers": selected["lines"],
                "raw_log_returned": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
