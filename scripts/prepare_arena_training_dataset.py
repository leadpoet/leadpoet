#!/usr/bin/env python3
"""Prepare company-qualification training records from Arena training exports.

Reads example files written by scripts/export_arena_training.py, read-only,
and writes a new private directory: train.jsonl, eval.jsonl, analysis.jsonl
and manifest.json. Train and eval hold only checked verdicts (qualified,
wrong fit, insufficient evidence) and never share a company cluster. The
analysis split keeps every other outcome for review. The manifest records
the sources, rules, counts and dataset version.

Exit 0 when at least one trainable record was written, 2 when none was, and
1 when the inputs or output could not be processed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_EXAMPLE_BYTES = 64 * 1024 * 1024


def _example_paths(paths):
    for path in paths:
        if path.is_dir():
            yield from sorted(item for item in path.glob("*.json") if item.is_file())
        else:
            yield path


def _examples(paths):
    for path in _example_paths(paths):
        if path.stat().st_size > MAX_EXAMPLE_BYTES:
            raise ValueError("example_too_large")
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError("example_not_object")
        yield document


def _write_private(path: Path, text: str) -> None:
    """Create a new mode-0600 file; never follow links or overwrite."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    from lab_arena import training_dataset
    from lab_arena.contracts import canonical_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, nargs="+", required=True,
                        help="exported example files, or directories whose top-level *.json files are exports")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the dataset")
    parser.add_argument("--eval-percent", type=int, default=training_dataset.Rules.eval_percent)
    parser.add_argument("--seed", default=training_dataset.Rules.seed)
    parser.add_argument("--include-nonproduction-rounds", action="store_true",
                        help="also use probe, replay and unpublished rounds")
    parser.add_argument("--scorer-image-digest", default=None, help="keep one scorer cohort only")
    parser.add_argument("--scorer-policy-hash", default=None, help="keep one exact scorer policy cohort only")
    parser.add_argument("--eval-mode", choices=("company", "company_and_icp"), default="company")
    parser.add_argument("--eval-after", help="strict YYYY-MM-DD cutoff; later records only in eval")
    parser.add_argument("--review-sample-size", type=int, default=0)
    args = parser.parse_args(argv)
    if args.review_sample_size < 0:
        parser.error("review-sample-size must be nonnegative")
    try:
        rules = training_dataset.Rules(
            eval_percent=args.eval_percent, seed=args.seed, eval_mode=args.eval_mode, eval_after=args.eval_after,
            production_rounds_only=not args.include_nonproduction_rounds,
            scorer_image_digest=args.scorer_image_digest,
            scorer_policy_hash=args.scorer_policy_hash,
        )
    except ValueError as exc:
        parser.error(str(exc))
    created = False
    try:
        splits, manifest = training_dataset.build(_examples(args.examples), rules)
        review = None
        if args.review_sample_size:
            review, manifest["review_sample"] = training_dataset.review_sample(splits, args.review_sample_size)
        args.output.mkdir(mode=0o700)
        created = True
        for name, records in splits.items():
            _write_private(args.output / (name + ".jsonl"),
                           "".join(canonical_json(record) + "\n" for record in records))
        if review is not None:
            _write_private(args.output / "review.jsonl", "".join(canonical_json(record) + "\n" for record in review))
        _write_private(args.output / "manifest.json", canonical_json(manifest) + "\n")
    except Exception as exc:
        if created:
            # Remove only what this run wrote, so a retry can use the path.
            with contextlib.suppress(OSError):
                for name in ("train.jsonl", "eval.jsonl", "analysis.jsonl", "manifest.json", "review.jsonl"):
                    (args.output / name).unlink(missing_ok=True)
                args.output.rmdir()
        # Inputs hold private evidence. The fixed cohort code is safe to show;
        # other exception messages may contain source text or private paths.
        reason = ("mixed_scorer_cohorts" if isinstance(exc, ValueError)
                  and str(exc) == "mixed_scorer_cohorts" else type(exc).__name__)
        print("Arena training dataset failed: " + reason, file=sys.stderr)
        return 1
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "records": {name: len(records) for name, records in splits.items()},
        "examples_used": manifest["counts"]["examples_used"],
        "examples_excluded": manifest["counts"]["examples_excluded"],
        "training_permission": manifest["training_permission"]["status"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if splits["train"] or splits["eval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
