"""Prepare locked research cases from one private Poet9B Arena split inventory.

Read-only on the source records. Only train runs that produced an actual row,
with an exact stored ICP and verified native labels, can enter the new batch.
"""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path

from lab_arena import contracts, training_dataset
from lab_arena.collection_research import FrozenResearchCases

MAX_EXAMPLE_BYTES = 256 * 1024 * 1024


def _example(path):
    # Input must be an original native Arena export; slim/incomplete files
    # lose authority and are refused by example_exclusion's original hash.
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        raw = handle.read(MAX_EXAMPLE_BYTES + 1)
    if len(raw) > MAX_EXAMPLE_BYTES:
        raise ValueError("native export size exceeds limit")
    return json.loads(raw)


def _row_runs(path):
    result = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("source") == "arena" and row.get("run"):
                    result.add(row["run"])
    return result


def build_index(inventory, train_run_ids, load, limit=100):
    entries = inventory["runs"]
    if type(entries) is not list or not 1 <= limit <= 1024:
        raise ValueError("invalid training inventory")
    held = {r.get("icp_id") for r in entries if r.get("split") in ("dev", "test")}
    results = {}
    reasons = Counter()
    for item in sorted(entries, key=lambda x: x["run_id"] or ""):
        rid, icp_id = item.get("run_id"), item.get("icp_id")
        if not (
            item.get("source") == "arena"
            and item.get("used") is True
            and item.get("split") == "train"
            and icp_id
            and icp_id not in held
            and rid in train_run_ids
        ):
            reasons["outside_training_split"] += 1
            continue
        ref_id = (
            "research-"
            + contracts.document_hash([inventory["seed"], icp_id]).split(":")[1][:20]
        )
        if ref_id in results:
            continue
        ex = load(item)
        if ex.get("run_id") != rid or (ex.get("icp") or {}).get("icp_id") != icp_id:
            raise ValueError("training inventory and native export identities differ")
        fail = training_dataset.example_exclusion(ex, training_dataset.Rules())
        if fail:
            reasons[fail] += 1
            continue
        config = training_dataset.label_scorer_configuration(ex)
        record = {
            "run_id": rid,
            "split": "train",
            "example_hash": ex["example_hash"],
            "scorer_image_digest": config["scorer_image_digest"],
            "scorer_policy_hash": contracts.document_hash(config["scorer_policy"]),
        }
        cases = FrozenResearchCases(None, None, {ref_id: record})
        cases.fetch = lambda _rid: ex
        try:
            case = cases.load({"scenario_id": ref_id})
        except ValueError:
            reasons["unusable_research_authority"] += 1
            continue
        if not case["labels"]:
            reasons["no_evidenced_native_labels"] += 1
            continue
        results[ref_id] = record
        if len(results) >= limit:
            break
    return results, dict(reasons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="Private inventory.json from build_rows.py subnet",
    )
    parser.add_argument(
        "--train-rows",
        required=True,
        type=Path,
        help="Corresponding final train.jsonl or arena-train.jsonl",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    index, reasons = build_index(
        json.loads(args.inventory.read_text()),
        _row_runs(args.train_rows),
        lambda row: _example(row["file"]),
        args.limit,
    )
    if not index:
        print(json.dumps({"cases": 0, "excluded": reasons}))
        return 2
    with os.fdopen(
        os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(json.dumps({"cases": len(index), "excluded": reasons}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
