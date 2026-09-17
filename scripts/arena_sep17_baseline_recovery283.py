#!/usr/bin/env python3
"""Prepare sealed September 17 recovery283; the normal Arena driver runs it."""
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
SOURCE_URL = "https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz"
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery283.tar.gz"
SOURCE_COMMIT = "e40484afde214a417128d1eadacaff716755784f"
SOURCE_SHA256 = "7df255f5db5267b13a65c9c3476cb47fd84837e00ce503bd1f588585ee8a37ae"
SOURCE_SIZE = 581173
TERMINAL_SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery282.tar.gz"
TERMINAL_SOURCE_COMMIT = "5a89cee202416419ade064e85ff40499f51d34b0"
TERMINAL_SOURCE_SHA256 = "cfa4a9bf8093dd230e8643bec5508b675e6f4b0ded75ac3d0f4c2a23616b6e41"
TERMINAL_SOURCE_SIZE = 580050
FORWARD_SCHEDULE = {
    "submission_open": "2026-09-16T00:00:00Z",
    "submission_cutoff": "2026-09-17T00:00:00Z",
    "benchmark_deadline": "2026-09-17T14:00:00Z",
    "stage_1_start": "2026-09-17T14:00:01Z",
    "stage_1_close": "2026-09-17T20:00:00Z",
    "stage_1_scoring_close": "2026-09-17T22:00:00Z",
    "stage_2_start": "2026-09-17T22:00:01Z",
    "stage_2_close": "2026-09-17T22:20:00Z",
    "final_scoring_close": "2026-09-18T00:20:00Z",
    "publication_deadline": "2026-09-18T00:50:00Z",
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
    # Recovery283 reads recovery282 as immutable history and never rewrites it.
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
    if (config.get("call_quotas") or {}).get("openrouter") != 200:
        raise RecoveryRefused("frozen execution quota differs")
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
        "status": "recovery283_preflight_ok", "round_id": ROUND,
        "evaluation_date": "2026-09-17", "icp_set_date": "2026-09-16",
        "bank_sha256": BANK_SHA256, "source_ref": SOURCE_REF,
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
        "source_size_bytes": SOURCE_SIZE, "execute_namespace": "rerun283",
        "openrouter_calls_per_icp": 200, "replay": replay,
    }
    if dry_run:
        return result
    if not isinstance(objects, S3ObjectStore):
        raise RecoveryRefused("live recovery requires the write-once Arena S3 store")
    objects.put(SOURCE_REF, source)
    if bytes(objects.get_bounded(SOURCE_REF, SOURCE_SIZE)) != source:
        raise RecoveryRefused("staged source readback differs")
    acknowledged = service.store._transport.rpc(
        "lab_arena_prepare_sep17_baseline_recovery283_v1",
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
        or acknowledged.get("execute_namespace") != "rerun283"
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
        print(f"Sep17 recovery283 refused: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("Sep17 recovery283 failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
