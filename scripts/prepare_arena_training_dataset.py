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
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Conservative default; large native exports need an explicit byte budget.
# The bound applies after decompression. Parse only one export at a time.
MAX_EXAMPLE_BYTES = 256 * 1024 * 1024
MAX_REVIEW_BYTES = 16 * 1024 * 1024


def _example_paths(paths):
    for path in paths:
        if path.is_dir():
            yield from sorted(item for item in path.iterdir()
                              if item.is_file() and (item.name.endswith(".json") or item.name.endswith(".json.gz")))
        else:
            yield path


def _example_limit_argument(parser):
    parser.add_argument("--max-example-bytes", type=int, default=MAX_EXAMPLE_BYTES,
                        help="maximum decompressed bytes per example (default: 268435456); "
                             "raise explicitly for large native episodes; JSON parsing needs extra memory")


def _examples(paths, *, max_example_bytes=MAX_EXAMPLE_BYTES, max_total_bytes=None):
    if isinstance(max_example_bytes, bool) or not isinstance(max_example_bytes, int) or max_example_bytes <= 0:
        raise ValueError("invalid_max_example_bytes")
    total_bytes = 0
    for path in _example_paths(paths):
        compressed = path.name.endswith(".gz")
        with path.open("rb") as source:
            if not compressed and os.fstat(source.fileno()).st_size > max_example_bytes:
                raise ValueError("example_too_large")
            with contextlib.ExitStack() as stack:
                stream = stack.enter_context(gzip.GzipFile(fileobj=source)) if compressed else source
                remaining = max_example_bytes if max_total_bytes is None else min(max_example_bytes, max_total_bytes - total_bytes)
                if remaining < 0:
                    raise ValueError("input_batch_too_large")
                raw = stream.read(remaining + 1)
                if len(raw) > remaining:
                    raise ValueError("example_too_large")
        total_bytes += len(raw)
        document = json.loads(raw)
        del raw
        if not isinstance(document, dict):
            raise ValueError("example_not_object")
        yield document
        del document


def _write_private(path: Path, text: str) -> None:
    """Create a new mode-0600 file; never follow links or overwrite."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)


def _review_decisions(path: Path):
    if path.stat().st_size > MAX_REVIEW_BYTES:
        raise ValueError("review_decisions_too_large")
    with path.open("rb") as stream:
        raw = stream.read(MAX_REVIEW_BYTES + 1)
    if len(raw) > MAX_REVIEW_BYTES:
        raise ValueError("review_decisions_too_large")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows


def main(argv=None):
    from lab_arena import training_curation, training_dataset
    from lab_arena.contracts import canonical_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, nargs="+", required=True,
                        help="exported example files, or directories whose top-level *.json and *.json.gz files are exports")
    _example_limit_argument(parser)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the dataset")
    parser.add_argument("--eval-percent", type=int, default=training_dataset.Rules.eval_percent)
    parser.add_argument("--seed", default=training_dataset.Rules.seed)
    parser.add_argument("--include-nonproduction-rounds", action="store_true",
                        help="also use probe, replay and unpublished rounds")
    parser.add_argument("--scorer-image-digest", default=None, help="keep one scorer cohort only")
    parser.add_argument("--scorer-policy-hash", default=None, help="keep one exact scorer policy cohort only")
    parser.add_argument("--eval-mode", choices=("company", "company_and_icp"),
                        default=training_dataset.Rules.eval_mode)
    parser.add_argument("--eval-after", help="strict YYYY-MM-DD cutoff; later records only in eval")
    parser.add_argument("--review-sample-size", type=int, default=0)
    parser.add_argument("--review-decisions", type=Path,
                        help="exact-record review JSONL; write a separate reviewed dataset")
    args = parser.parse_args(argv)
    if args.max_example_bytes <= 0:
        parser.error("max-example-bytes must be positive")
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
        projections = {}
        splits, manifest = training_dataset.build(_examples(args.examples, max_example_bytes=args.max_example_bytes), rules,
                                                  curation_metadata=projections)
        source_counts = manifest["counts"]
        curation = training_curation.flag_manifest(splits, manifest["dataset_version"], projections)
        review = None
        if args.review_sample_size:
            review, review_summary = training_curation.review_sample(
                splits, args.review_sample_size, manifest, curation)
        if args.review_decisions:
            splits, manifest = training_curation.apply_review(
                splits, manifest, _review_decisions(args.review_decisions), curation)
        if args.review_sample_size:
            manifest["review_sample"] = review_summary
        args.output.mkdir(mode=0o700)
        created = True
        for name, records in splits.items():
            _write_private(args.output / (name + ".jsonl"),
                           "".join(canonical_json(record) + "\n" for record in records))
        if review is not None:
            _write_private(args.output / "review.jsonl", "".join(canonical_json(record) + "\n" for record in review))
        _write_private(args.output / "curation.json", canonical_json(curation) + "\n")
        _write_private(args.output / "manifest.json", canonical_json(manifest) + "\n")
    except Exception as exc:
        if created:
            # Remove only what this run wrote, so a retry can use the path.
            with contextlib.suppress(OSError):
                for name in ("train.jsonl", "eval.jsonl", "analysis.jsonl", "manifest.json", "review.jsonl", "curation.json"):
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
        "examples_used": source_counts["examples_used"],
        "examples_excluded": source_counts["examples_excluded"],
        "training_permission": manifest["training_permission"]["status"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if splits["train"] or splits["eval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
