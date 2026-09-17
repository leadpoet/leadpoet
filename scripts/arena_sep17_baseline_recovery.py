#!/usr/bin/env python3
"""Prepare the sealed September 17 recovery; the normal Arena driver runs it."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, source_bundle
from lab_arena.service import S3ObjectStore

ROUND = "arena-2026-09-17"
BASELINE = "baseline-2026-09-17"
BANK_REF = f"arena/{ROUND}/benchmark.json"
BANK_SHA256 = "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871"
SOURCE_URL = "https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz"
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery278.tar.gz"
SOURCE_COMMIT = "4ceae936b902433432a195f77af3f94559d80378"
SOURCE_SHA256 = "6c9eeaac41386204a7817b00038e5a3f3545e1c205a2dc32de6889927ffe7245"
SOURCE_SIZE = 563030
TERMINAL_SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}.tar.gz"
TERMINAL_SOURCE_COMMIT = "ec4887f728b77002de570c4023415f8963995b79"
TERMINAL_SOURCE_SHA256 = "858ad5b2e68e3c6c354c0e4e358264186cec206b83220a68719a6e217273d057"
TERMINAL_SOURCE_SIZE = 547308
FORWARD_SCHEDULE = {
    "submission_open": "2026-09-16T00:00:00Z",
    "submission_cutoff": "2026-09-17T00:00:00Z",
    "benchmark_deadline": "2026-09-17T06:00:00Z",
    "stage_1_start": "2026-09-17T06:00:01Z",
    "stage_1_close": "2026-09-17T12:00:00Z",
    "stage_1_scoring_close": "2026-09-17T14:00:00Z",
    "stage_2_start": "2026-09-17T14:00:01Z",
    "stage_2_close": "2026-09-17T14:20:00Z",
    "final_scoring_close": "2026-09-17T16:20:00Z",
    "publication_deadline": "2026-09-17T16:50:00Z",
}


class RecoveryRefused(RuntimeError):
    pass


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_archive(payload: bytes, *, size: int, digest: str, commit: str) -> None:
    if len(payload) != size or _digest(payload) != digest:
        raise RecoveryRefused("source archive bytes differ from the sealed authority")
    source_bundle.validate_source_archive(payload, require_license=True)
    if source_bundle.source_archive_commit(payload) != commit:
        raise RecoveryRefused("source archive commit differs from the sealed authority")


def prepare(service: Any, *, dry_run: bool, now: datetime | None = None) -> dict[str, Any]:
    row = service.store.get_round(ROUND)
    baseline = service.store.get_submission(BASELINE)
    if not row or row.get("round_id") != ROUND or (
        row.get("evaluation_date") != "2026-09-17"
        or row.get("icp_set_date") != "2026-09-16"
        or row.get("benchmark_ref") != BANK_REF
    ):
        raise RecoveryRefused("current evaluation or frozen bank identity differs")
    if not baseline or baseline.get("round_id") != ROUND:
        raise RecoveryRefused("current baseline identity differs")
    replay = baseline.get("source_ref") == SOURCE_REF
    if not replay and (row.get("status") != "cancelled" or (
        baseline.get("source_ref") != TERMINAL_SOURCE_REF
    )):
        raise RecoveryRefused("round is neither the sealed terminal state nor its recovery")
    objects = service.config.object_store
    bank = json.loads(objects.get_bounded(BANK_REF, 2 * 1024 * 1024))
    if not isinstance(bank, dict) or set(bank) != {"schema_version", "round_id", "icps"} or (
        bank["schema_version"] != "leadpoet.lab_arena.benchmark.v1"
        or bank["round_id"] != ROUND or not isinstance(bank["icps"], list)
        or len(bank["icps"]) != 20
        or len({contracts.document_hash(item) for item in bank["icps"]}) != 20
        or _digest(contracts.canonical_json(bank["icps"]).encode()) != BANK_SHA256
    ):
        raise RecoveryRefused("frozen September 16 ICP bank differs")
    # Verify the immutable failed source even on replay. Recovery never rewrites it.
    terminal = bytes(objects.get_bounded(TERMINAL_SOURCE_REF, TERMINAL_SOURCE_SIZE))
    _verify_archive(terminal, size=TERMINAL_SOURCE_SIZE,
                    digest=TERMINAL_SOURCE_SHA256, commit=TERMINAL_SOURCE_COMMIT)
    config = copy.deepcopy(row.get("configuration_doc") or {})
    old_schedule = config.get("schedule") or {}
    if set(old_schedule) != set(FORWARD_SCHEDULE) or any(
        old_schedule.get(key) != FORWARD_SCHEDULE[key]
        for key in ("submission_open", "submission_cutoff")
    ):
        raise RecoveryRefused("frozen submission intake differs")
    config["schedule"] = dict(FORWARD_SCHEDULE)
    config.setdefault("call_quotas", {})["openrouter"] = 200
    contracts.validate_round_configuration(config)
    if not replay:
        now = now or datetime.now(timezone.utc)
        if now >= datetime.fromisoformat(FORWARD_SCHEDULE["benchmark_deadline"].replace("Z", "+00:00")):
            raise RecoveryRefused("sealed recovery admission window has closed")
    if replay:
        source = bytes(objects.get_bounded(SOURCE_REF, SOURCE_SIZE))
    else:
        fetcher = service.config.baseline_source_fetcher
        if fetcher is None:
            raise RecoveryRefused("normal public baseline source fetcher is unavailable")
        source = bytes(fetcher(SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    _verify_archive(source, size=SOURCE_SIZE, digest=SOURCE_SHA256, commit=SOURCE_COMMIT)
    result = {
        "status": "recovery278_preflight_ok", "round_id": ROUND,
        "evaluation_date": "2026-09-17", "icp_set_date": "2026-09-16",
        "bank_sha256": BANK_SHA256, "source_ref": SOURCE_REF,
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
        "source_size_bytes": SOURCE_SIZE, "execute_namespace": "rerun278",
        "openrouter_calls_per_icp": 200, "replay": replay,
    }
    if dry_run:
        return result
    if not isinstance(objects, S3ObjectStore):
        raise RecoveryRefused("live recovery requires the write-once Arena S3 store")
    # Existing S3 put uses If-None-Match and accepts an existing object only when
    # its bytes match. No catch-all read error is treated as a missing object.
    objects.put(SOURCE_REF, source)
    if bytes(objects.get_bounded(SOURCE_REF, SOURCE_SIZE)) != source:
        raise RecoveryRefused("staged source readback differs")
    acknowledged = service.store._transport.rpc(
        "lab_arena_prepare_sep17_baseline_recovery278_v1",
        {
            "p_source_size_bytes": SOURCE_SIZE, "p_source_sha256": SOURCE_SHA256,
            "p_source_commit": SOURCE_COMMIT, "p_bank_sha256": BANK_SHA256,
            "p_forward_schedule": dict(FORWARD_SCHEDULE),
        },
    )
    if not isinstance(acknowledged, dict) or acknowledged.get("status") not in (
        "prepared", "existing"
    ) or acknowledged.get("round_id") != ROUND or (
        acknowledged.get("baseline_execute_assignments") != 20
        or acknowledged.get("openrouter_calls_per_icp") != 200
        or acknowledged.get("execute_namespace") != "rerun278"
    ):
        raise RecoveryRefused("recovery RPC did not acknowledge the exact prepared cycle")
    return {**result, **acknowledged}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        from scripts.run_lab_arena_service import load_scoped_environment
        from lab_arena.wiring import build_service_from_environment
        load_scoped_environment(args.environment_file)
        if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
            raise RecoveryRefused("live Arena environment is required")
        service, _ = build_service_from_environment("live")
        print(json.dumps(prepare(service, dry_run=args.dry_run), sort_keys=True))
        return 0
    except RecoveryRefused as exc:
        print(f"Sep17 recovery refused: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("Sep17 recovery failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
