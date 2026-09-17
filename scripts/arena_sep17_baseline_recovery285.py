#!/usr/bin/env python3
"""Prepare the minimal September 17 recovery285; the normal Arena driver runs it."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
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
SOURCE_URL = (
    "https://github.com/leadpoet/leadpoet-sales-agent/"
    "archive/refs/heads/lab.tar.gz"
)
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery285.tar.gz"
TERMINAL_SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery284.tar.gz"
TERMINAL_SOURCE_COMMIT = "91e8b95637c4cbc919f3150d6643985624dbb34a"
TERMINAL_SOURCE_SHA256 = "b5ca95d7c9c25650c5ecb04b05bde859cffa3fe9c6c7a6602e6a006d5e3dcd99"
TERMINAL_SOURCE_SIZE = 582191
FORWARD_SCHEDULE = {
    "submission_open": "2026-09-16T00:00:00Z",
    "submission_cutoff": "2026-09-17T00:00:00Z",
    "benchmark_deadline": "2026-09-17T18:30:00Z",
    "stage_1_start": "2026-09-17T18:30:01Z",
    "stage_1_close": "2026-09-17T22:00:00Z",
    "stage_1_scoring_close": "2026-09-18T00:00:00Z",
    "stage_2_start": "2026-09-18T00:00:01Z",
    "stage_2_close": "2026-09-18T00:20:00Z",
    "final_scoring_close": "2026-09-18T02:20:00Z",
    "publication_deadline": "2026-09-18T02:50:00Z",
}


class RecoveryRefused(RuntimeError):
    pass


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive_facts(payload: bytes) -> dict[str, Any]:
    source_bundle.validate_source_archive(payload, require_license=True)
    commit = source_bundle.source_archive_commit(payload)
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RecoveryRefused("source archive commit identity is invalid")
    return {
        "source_size_bytes": len(payload),
        "source_sha256": _digest(payload),
        "source_commit": commit,
    }


def _verify_terminal_archive(payload: bytes) -> None:
    if len(payload) != TERMINAL_SOURCE_SIZE or _digest(payload) != TERMINAL_SOURCE_SHA256:
        raise RecoveryRefused("recovery284 source archive bytes differ")
    facts = _archive_facts(payload)
    if facts["source_commit"] != TERMINAL_SOURCE_COMMIT:
        raise RecoveryRefused("recovery284 source archive commit differs")


def prepare(
    service: Any,
    *,
    dry_run: bool,
    expected_source_size_bytes: int,
    expected_source_sha256: str,
    expected_source_commit: str,
    now: datetime | None = None,
) -> dict[str, Any]:
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
    if not replay and (
        row.get("status") != "cancelled"
        or baseline.get("source_ref") != TERMINAL_SOURCE_REF
    ):
        raise RecoveryRefused("round is neither the terminal state nor recovery285")

    objects = service.config.object_store
    bank = json.loads(objects.get_bounded(BANK_REF, 2 * 1024 * 1024))
    if not isinstance(bank, dict) or set(bank) != {"schema_version", "round_id", "icps"} or (
        bank["schema_version"] != "leadpoet.lab_arena.benchmark.v1"
        or bank["round_id"] != ROUND
        or not isinstance(bank["icps"], list)
        or len(bank["icps"]) != 20
        or len({contracts.document_hash(item) for item in bank["icps"]}) != 20
        or _digest(contracts.canonical_json(bank["icps"]).encode()) != BANK_SHA256
    ):
        raise RecoveryRefused("frozen September 16 ICP bank differs")

    terminal = bytes(objects.get_bounded(TERMINAL_SOURCE_REF, TERMINAL_SOURCE_SIZE))
    _verify_terminal_archive(terminal)
    config = copy.deepcopy(row.get("configuration_doc") or {})
    old_schedule = config.get("schedule") or {}
    if set(old_schedule) != set(FORWARD_SCHEDULE) or any(
        old_schedule.get(key) != FORWARD_SCHEDULE[key]
        for key in ("submission_open", "submission_cutoff")
    ):
        raise RecoveryRefused("frozen submission intake differs")
    config["schedule"] = dict(FORWARD_SCHEDULE)
    if (config.get("call_quotas") or {}) != {
        "openrouter": 200, "deepline": 30, "scrapingdog": 30,
    }:
        raise RecoveryRefused("frozen execution quotas differ")
    contracts.validate_round_configuration(config)
    if not replay:
        now = now or datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(
            FORWARD_SCHEDULE["benchmark_deadline"].replace("Z", "+00:00")
        )
        if now >= deadline:
            raise RecoveryRefused("recovery285 admission window has closed")

    if replay:
        source = bytes(objects.get_bounded(SOURCE_REF, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    else:
        fetcher = service.config.baseline_source_fetcher
        if fetcher is None:
            raise RecoveryRefused("normal public baseline source fetcher is unavailable")
        source = bytes(fetcher(SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    facts = _archive_facts(source)
    if facts != {
        "source_size_bytes": expected_source_size_bytes,
        "source_sha256": expected_source_sha256,
        "source_commit": expected_source_commit,
    }:
        raise RecoveryRefused("lab source archive differs from reviewed identity")
    result = {
        "status": "recovery285_preflight_ok",
        "round_id": ROUND,
        "evaluation_date": "2026-09-17",
        "icp_set_date": "2026-09-16",
        "bank_sha256": BANK_SHA256,
        "source_ref": SOURCE_REF,
        **facts,
        "execute_namespace": "rerun285",
        "openrouter_calls_per_icp": 200,
        "replay": replay,
    }
    if dry_run:
        return result
    if not isinstance(objects, S3ObjectStore):
        raise RecoveryRefused("live recovery requires the write-once Arena S3 store")
    if not replay:
        objects.put(SOURCE_REF, source)
        if bytes(objects.get_bounded(
            SOURCE_REF, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        )) != source:
            raise RecoveryRefused("staged source readback differs")
    acknowledged = service.store._transport.rpc(
        "lab_arena_prepare_sep17_baseline_recovery285_v1",
        {
            "p_source_size_bytes": facts["source_size_bytes"],
            "p_source_sha256": facts["source_sha256"],
            "p_source_commit": facts["source_commit"],
            "p_bank_sha256": BANK_SHA256,
            "p_forward_schedule": dict(FORWARD_SCHEDULE),
        },
    )
    if not isinstance(acknowledged, dict) or acknowledged.get("status") not in (
        "prepared", "existing"
    ) or acknowledged.get("round_id") != ROUND or (
        acknowledged.get("baseline_execute_assignments") != 20
        or acknowledged.get("openrouter_calls_per_icp") != 200
        or acknowledged.get("execute_namespace") != "rerun285"
        or acknowledged.get("source_sha256") != facts["source_sha256"]
        or acknowledged.get("source_commit") != facts["source_commit"]
        or acknowledged.get("source_size_bytes") != facts["source_size_bytes"]
    ):
        raise RecoveryRefused("recovery RPC did not acknowledge the exact prepared cycle")
    return {**result, **acknowledged}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--expected-source-size-bytes", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        from scripts.run_lab_arena_service import load_scoped_environment
        from lab_arena.wiring import build_service_from_environment

        load_scoped_environment(args.environment_file)
        if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
            raise RecoveryRefused("live Arena environment is required")
        service, _ = build_service_from_environment("live")
        print(json.dumps(prepare(
            service,
            dry_run=args.dry_run,
            expected_source_size_bytes=args.expected_source_size_bytes,
            expected_source_sha256=args.expected_source_sha256,
            expected_source_commit=args.expected_source_commit,
        ), sort_keys=True))
        return 0
    except RecoveryRefused as exc:
        print(f"Sep17 recovery285 refused: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("Sep17 recovery285 failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
