#!/usr/bin/env python3
"""Activate an already verified runtime composition through the existing store.

First run build_arena_reviewer_runtime.py --verify-image against the exact
published image. Run this operator command on the gateway's reviewed release.
It updates only the runtime selection, never a round or completed result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Operator commands must not write bytecode into the verified release tree.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import images
from lab_arena.store import PostgrestTransport
from scripts.run_lab_arena_service import load_scoped_environment


def registration_arguments(grading: str, runtime: str) -> dict[str, str]:
    original, repair = images.parse_reference(grading), images.parse_reference(runtime)
    if not original.digest or not repair.digest:
        raise ValueError("both images must be pinned by digest")
    if (original.registry, original.repository) != (repair.registry, repair.repository):
        raise ValueError("runtime must use the existing grader repository")
    return {
        "p_grading_image_digest": original.digest,
        "p_grading_image_reference": grading,
        "p_runtime_image_digest": repair.digest,
        "p_runtime_image_reference": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-image", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    arguments = registration_arguments(args.grading_image, args.runtime_image)
    if not args.apply:
        print(json.dumps({"action": "register_reviewer_runtime", "arguments": arguments,
                          "writes": False}, sort_keys=True))
        return 0
    if os.environ.get("LEADPOET_ARENA_REVIEWER_RUNTIME_APPLY") != "1":
        raise ValueError("runtime activation requires the operator apply environment")
    load_scoped_environment(args.environment_file)
    transport = PostgrestTransport(
        os.environ["LAB_ARENA_SUPABASE_URL"],
        service_key=os.environ["LAB_ARENA_SERVICE_KEY"],
    )
    try:
        result = transport.rpc("lab_arena_register_reviewer_runtime_v1", arguments)
    finally:
        transport.close()
    if not isinstance(result, dict) or any(
        result.get(key.removeprefix("p_")) != value for key, value in arguments.items()
    ):
        raise ValueError("runtime registration readback differs")
    print(json.dumps({key: result[key] for key in (
        "grading_image_digest", "grading_image_reference",
        "runtime_image_digest", "runtime_image_reference", "updated_at",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Store/credential failures may include URLs. Do not emit raw details.
        raise SystemExit("reviewer runtime registration failed: " + type(exc).__name__) from None
