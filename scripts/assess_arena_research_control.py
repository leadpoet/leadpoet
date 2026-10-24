#!/usr/bin/env python3
"""Prepare or read an offline research-control assessment; never admit a miner.

This command makes no network calls and does not execute submitted code.
--response validates a saved provider reply against the exact source and policy.
Without it, the output is a complete request for a separately budgeted reviewer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import code_review, contracts, research_control, source_bundle
from scripts.prepare_arena_training_dataset import _write_private

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _read(path, limit):
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > limit:
            raise ValueError("input_too_large")
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("input_too_large")
        return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="validated .tar.gz source bundle")
    parser.add_argument("--response", type=Path, help="complete saved OpenRouter JSON response")
    parser.add_argument("--output", type=Path, required=True, help="new private JSON file; never overwritten")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        prepared = research_control.prepare_request(
            _read(args.source, source_bundle.MAX_SOURCE_ARCHIVE_BYTES),
            max_output_tokens=args.max_output_tokens,
        )
        result = {
            "purpose": "offline_research_control_assessment",
            "production_eligibility": "not_established",
            "runtime_behavior": "not_verified",
            "policy_id": prepared.policy_id,
            "policy_hash": prepared.policy_hash,
            "source_archive_sha256": prepared.source_archive_sha256,
            "source_manifest_sha256": prepared.source_manifest_sha256,
            "request_sha256": prepared.request_sha256,
        }
        if args.response:
            raw = _read(args.response, MAX_RESPONSE_BYTES)
            # Reject duplicate top-level or nested JSON keys before validation.
            response = code_review._json_object_without_duplicates(raw.decode("utf-8"))
            assessment = research_control.parse_response(response, prepared)
            result.update(assessment=assessment.to_document(),
                          provider_response_sha256=contracts.hash_bytes(raw))
        else:
            result.update(parameters=prepared.parameters,
                          input_tokens_upper_bound=prepared.input_tokens_upper_bound)
        _write_private(args.output, contracts.canonical_json(result) + "\n")
    except (OSError, ValueError, TypeError, RuntimeError):
        # Source and provider errors may contain private submission contents.
        print("Research-control assessment failed: invalid_input_or_output", file=sys.stderr)
        return 1
    print(json.dumps({"policy_id": prepared.policy_id,
                      "classification": result.get("assessment", {}).get("classification", "not_run"),
                      "production_eligibility": "not_established"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
