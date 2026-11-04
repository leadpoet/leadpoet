#!/usr/bin/env python3
"""Write a new private observed agent-step dataset from immutable Arena exports."""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_arena_training_dataset import _example_limit_argument, _examples, _review_decisions, _write_private
from lab_arena import contracts, training_curation, training_steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-percent', type=int, default=20)
    parser.add_argument('--seed', default=training_steps.Rules.seed)
    parser.add_argument('--eval-mode', choices=('company', 'company_and_icp'),
                        default=training_steps.Rules.eval_mode)
    parser.add_argument('--eval-after')
    parser.add_argument('--include-nonproduction-rounds', action='store_true')
    parser.add_argument('--review-sample-size', type=int, default=0)
    parser.add_argument('--review-decisions', type=Path)
    _example_limit_argument(parser)
    args = parser.parse_args(argv)
    if args.max_example_bytes <= 0:
        parser.error("max-example-bytes must be positive")
    created = False
    try:
        rules = training_steps.Rules(
            eval_percent=args.eval_percent, seed=args.seed,
            production_rounds_only=not args.include_nonproduction_rounds,
            eval_mode=args.eval_mode, eval_after=args.eval_after)
        splits, manifest = training_steps.build(_examples(args.examples, max_example_bytes=args.max_example_bytes), rules)
        curation = training_curation.flag_manifest(splits, manifest['dataset_version'])
        review = None
        if args.review_sample_size:
            review, review_summary = training_curation.review_sample(
                splits, args.review_sample_size, manifest, curation)
        if args.review_decisions:
            splits, manifest = training_curation.apply_review(
                splits, manifest, _review_decisions(args.review_decisions), curation)
        if args.review_sample_size:
            manifest['review_sample'] = review_summary
        args.output.mkdir(mode=0o700)
        created = True
        for name, rows in splits.items():
            _write_private(args.output / (name + '.jsonl'), ''.join(contracts.canonical_json(row) + '\n' for row in rows))
        if review is not None:
            _write_private(args.output / 'review.jsonl', ''.join(contracts.canonical_json(row) + '\n' for row in review))
        _write_private(args.output / 'curation.json', contracts.canonical_json(curation) + '\n')
        _write_private(args.output / 'manifest.json', contracts.canonical_json(manifest) + '\n')
    except Exception:
        if created:
            with contextlib.suppress(OSError):
                for name in ('train.jsonl', 'eval.jsonl', 'analysis.jsonl', 'manifest.json', 'review.jsonl', 'curation.json'):
                    (args.output / name).unlink(missing_ok=True)
                args.output.rmdir()
        print('Arena agent dataset failed: invalid_input_or_output', file=sys.stderr)
        return 1
    print(json.dumps({'dataset_version': manifest['dataset_version'], 'counts': manifest['counts'],
                      'training_permission': manifest['training_permission']['status']}, sort_keys=True))
    return 0 if splits['train'] or splits['eval'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
