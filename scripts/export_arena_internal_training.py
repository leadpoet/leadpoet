#!/usr/bin/env python3
"""Export private internal Arena calls from explicit refs or bounded S3 dates."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, internal_capture, reasoning_capture


def discover_s3(client, *, bucket: str, prefix: str, dates: list[str], max_events: int) -> tuple[list[str], dict]:
    """List only named date prefixes. A bounded partial inventory stays explicit."""
    from lab_arena.service import S3ObjectStore

    prefix = S3ObjectStore._validate_key_prefix(prefix)
    days = sorted(set(dates))
    if not days or len(days) > 31 or not 1 <= max_events <= internal_capture.MAX_EXPORT_EVENTS:
        raise ValueError("discovery_bounds_invalid")
    for day in days:
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("discovery_date_invalid")
    base = prefix + "/" if prefix else ""
    refs, scanned, ignored, truncated = [], 0, 0, False
    for day_index, day in enumerate(days):
        token = None
        while True:
            if scanned >= max_events:
                truncated = True
                break
            options = {"Bucket": bucket, "Prefix": base + internal_capture.OBJECT_PREFIX + day + "/",
                       "MaxKeys": min(1000, max_events - scanned)}
            if token is not None:
                options["ContinuationToken"] = token
            result = client.list_objects_v2(**options)
            contents = result.get("Contents") or []
            if not isinstance(contents, list) or len(contents) > options["MaxKeys"]:
                raise ValueError("listing_response_invalid")
            scanned += len(contents)
            for item in contents:
                key = item.get("Key") if isinstance(item, dict) else None
                if not isinstance(key, str) or not key.startswith(options["Prefix"]):
                    raise ValueError("listing_scope_invalid")
                ref = key[len(base):]
                try:
                    internal_capture.call_prefix(ref)
                except ValueError:
                    ignored += 1
                else:
                    refs.append(ref)
            if not result.get("IsTruncated"):
                break
            token = result.get("NextContinuationToken")
            if not isinstance(token, str) or not token or not contents:
                raise ValueError("listing_cursor_invalid")
        if truncated:
            break
        if scanned >= max_events and day_index + 1 < len(days):
            truncated = True
            break
    return refs, {"method": "bounded_s3_date_prefixes", "dates": days, "max_events": max_events,
                  "scanned_objects": scanned, "ignored_non_event_objects": ignored,
                  "truncated": truncated}


def _private_stream(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--refs", nargs="+", help="Explicit internal stage object refs; sibling stages are also read.")
    scope.add_argument("--dates", nargs="+", help="UTC dates (YYYY-MM-DD), at most 31; S3 listing only.")
    parser.add_argument("--object-root", type=Path, help="Existing local object-store root; requires --refs.")
    parser.add_argument("--bucket", default=os.environ.get("LAB_ARENA_BUCKET"))
    parser.add_argument("--prefix", default=os.environ.get("LAB_ARENA_OBJECT_PREFIX", ""))
    parser.add_argument("--max-events", type=int, default=3000)
    parser.add_argument("--reasoning-registry", type=Path, help="Reviewed route registry; defaults to the committed registry.")
    parser.add_argument("--output", type=Path, required=True, help="New private directory; existing paths are never overwritten.")
    args = parser.parse_args(argv)
    if not 1 <= args.max_events <= internal_capture.MAX_EXPORT_EVENTS:
        parser.error("max-events must be between 1 and 30000")
    if args.object_root and args.dates:
        parser.error("local reads require explicit --refs")
    created = False
    created_files = []
    try:
        registry = reasoning_capture.validate_registry(json.loads(args.reasoning_registry.read_text())) if args.reasoning_registry else reasoning_capture.load_registry()
        if args.refs:
            if len(args.refs) > args.max_events:
                raise ValueError("event_limit_exceeded")
            for ref in args.refs:
                internal_capture.call_prefix(ref)
            refs = args.refs
            discovery = {"method": "explicit_refs", "provided_refs": len(refs), "truncated": False}
        if args.object_root:
            from lab_arena.service import LocalObjectStore
            if not args.object_root.is_dir():
                raise ValueError("local_object_root_missing")
            objects = LocalObjectStore(args.object_root)
        else:
            import boto3
            from botocore.config import Config
            from lab_arena.service import S3ObjectStore
            if not args.bucket:
                raise ValueError("bucket_unavailable")
            client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"), config=Config(
                signature_version="s3v4", connect_timeout=0.5, read_timeout=0.5,
                retries={"total_max_attempts": 1, "mode": "standard"}))
            objects = S3ObjectStore(args.bucket, client=client, prefix=args.prefix)
            if args.dates:
                refs, discovery = discover_s3(client, bucket=args.bucket, prefix=args.prefix,
                                               dates=args.dates, max_events=args.max_events)
        args.output.mkdir(mode=0o700)
        created = True
        counts, gaps, purposes, exclusions, reasoning = Counter(), Counter(), Counter(), Counter(), Counter()
        with contextlib.ExitStack() as stack:
            streams = {}
            for name in ("raw.jsonl", "sft.jsonl"):
                path = args.output / name
                stream = _private_stream(path)
                created_files.append(path)
                streams[name] = stack.enter_context(stream)
            for raw, candidate in internal_capture.iter_exports(objects, refs, registry=registry):
                streams["raw.jsonl"].write(contracts.canonical_json(raw) + "\n")
                counts["calls"] += 1
                counts["complete"] += int(raw["completeness"]["complete"])
                purposes[str(raw.get("purpose"))] += 1
                gaps.update(raw["completeness"]["gaps"])
                exclusions.update(raw["sft_exclusions"])
                reasoning[str(raw["reasoning"]["status"].get("returned"))] += 1
                if candidate is not None:
                    streams["sft.jsonl"].write(contracts.canonical_json(candidate) + "\n")
                    counts["sft_candidates"] += 1
            manifest = {"schema_version": "leadpoet.lab_arena.internal_training_manifest.v1",
                        "actor_type": "arena_internal", "scope": "supplied_internal_event_refs_only",
                        "discovery": discovery, "counts": dict(counts), "purposes": dict(purposes),
                        "gap_counts": dict(gaps), "sft_exclusions": dict(exclusions),
                        "reasoning_status": dict(reasoning), "quality": "unreviewed",
                        "training_permission": {"status": "unconfirmed"},
                        "reasoning_registry_sha256": contracts.document_hash(registry),
                        "limits": ["request_event_does_not_prove_dispatch", "provider_returned_readable_text_only",
                                   "host_outcome_is_not_verified_ground_truth", "no_miner_eligibility_authority",
                                   "sft_is_an_unsplit_candidate_inventory"]}
            manifest["manifest_sha256"] = contracts.document_hash(manifest)
            path = args.output / "manifest.json"
            with _private_stream(path) as stream:
                created_files.append(path)
                stream.write(contracts.canonical_json(manifest) + "\n")
        print(json.dumps({"counts": manifest["counts"], "truncated": discovery["truncated"],
                          "artifact": str(args.output)}, sort_keys=True))
        return 0 if counts["calls"] and counts["complete"] == counts["calls"] and not discovery["truncated"] else 2
    except Exception as exc:
        if created:
            for path in created_files:
                with contextlib.suppress(OSError):
                    path.unlink()
            with contextlib.suppress(OSError):
                args.output.rmdir()
        print("Arena internal training export failed: " + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
