"""Build a private replay-verified source from a native Poet training run."""

import argparse
import json
from pathlib import Path

from lab_arena.collection_workflow import prepare_student_source, workflow_sources
from scripts.collection_review import read_file, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("champion-repo", "poet-repo", "run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--model", required=True, help="Exact student model in the producer record"
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        help="Private receipt pack for an approved owner-funded student",
    )
    args = parser.parse_args(argv)
    with workflow_sources(
        args.champion_repo, args.poet_repo, corrections=True
    ) as sources:
        provenance = (
            json.loads(read_file(args.provenance, maximum=32 * 1024 * 1024))
            if args.provenance
            else None
        )
        source = prepare_student_source(
            *sources, args.run, model=args.model, provenance=provenance
        )
        write_new(args.output, source)
    print(
        json.dumps(
            {
                "source_sha256": source["source_sha256"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "branch_turn": source["branch_turn"],
                "student_model": source["student_model"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
