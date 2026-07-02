#!/usr/bin/env python3
"""Write gateway/BUILD_INFO.json for a deployable gateway artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.build_info import create_build_info_document, write_build_info_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate non-secret gateway build provenance. Run this from the repo "
            "before rsync/docker/enclave build so prod can report the live commit."
        )
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "gateway" / "BUILD_INFO.json"),
        help="Path to write. Default: gateway/BUILD_INFO.json",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Git checkout to inspect. Default: repository root.",
    )
    parser.add_argument(
        "--service",
        default="leadpoet-gateway",
        help="Service name to record.",
    )
    parser.add_argument(
        "--build-id",
        default=None,
        help="Optional explicit build id. Defaults to a commit/timestamp id.",
    )
    parser.add_argument(
        "--require-git-commit",
        action="store_true",
        help="Exit non-zero if no commit can be recorded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = create_build_info_document(
        repo_root=args.repo_root,
        service=args.service,
        build_id=args.build_id,
    )
    if args.require_git_commit and document.get("git_commit") == "unknown":
        print("ERROR: no git commit could be recorded for gateway build info", file=sys.stderr)
        return 2

    output_path = write_build_info_file(args.output, document)
    print(f"Wrote {output_path}")
    print(f"  build_id: {document.get('build_id')}")
    print(f"  git_commit: {document.get('git_commit')}")
    print(f"  git_branch: {document.get('git_branch')}")
    print(f"  git_dirty: {document.get('git_dirty')}")
    print(f"  build_time_utc: {document.get('build_time_utc')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
