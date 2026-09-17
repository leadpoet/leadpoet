#!/usr/bin/env python3
"""Seal and prepare the one-time Sep17 published-baseline rerun286."""
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
from typing import Any, Mapping

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
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery286.tar.gz"
TERMINAL_SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-recovery285.tar.gz"
PREFLIGHT_SCHEMA = "leadpoet.sep17_published_rerun286.preflight.v1"
READ_PAGE_SIZE = 500
MAX_SEALED_ROWS = 100_000


class RerunRefused(RuntimeError):
    pass


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive_facts(payload: bytes) -> dict[str, Any]:
    source_bundle.validate_source_archive(payload, require_license=True)
    commit = source_bundle.source_archive_commit(payload)
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RerunRefused("source archive commit identity is invalid")
    return {
        "source_ref": "",
        "source_size_bytes": len(payload),
        "source_sha256": _digest(payload),
        "source_commit": commit,
    }


def _source_facts(payload: bytes, source_ref: str) -> dict[str, Any]:
    return {**_archive_facts(payload), "source_ref": source_ref}


def _cost_state(service: Any, kind: str) -> dict[str, Any]:
    value = service.store.submission_costs(BASELINE)
    providers = value.get("providers") if isinstance(value, Mapping) else None
    if not isinstance(providers, list):
        raise RerunRefused("successful-call cost state is unavailable")
    selected = [item for item in providers if item.get("kind") == kind]
    if not selected or any(
        field not in item
        for item in selected
        for field in ("inflight_calls", "success_unresolved_calls")
    ):
        raise RerunRefused(f"Sep17 {kind} cost state is incomplete")
    return {
        field: sum(int(item.get(field, -1)) for item in selected)
        for field in ("inflight_calls", "success_unresolved_calls")
    }


def _select_all(
    service: Any,
    table: str,
    *,
    filters: Mapping[str, Any],
    order: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while len(rows) < MAX_SEALED_ROWS:
        page = service.store._transport.select(
            table,
            filters=dict(filters),
            order=order,
            limit=READ_PAGE_SIZE,
            offset=len(rows),
        )
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise RerunRefused(f"{table} sealed read is invalid")
        rows.extend(page)
        if len(page) < READ_PAGE_SIZE:
            return rows
    raise RerunRefused(f"{table} sealed read exceeds the safety bound")


def _sealed_rows(service: Any) -> dict[str, list[dict[str, Any]]]:
    submissions = _select_all(
        service, "lab_arena_submissions",
        filters={"round_id": ROUND}, order="submission_id",
    )
    runs = _select_all(
        service, "lab_arena_runs", filters={"round_id": ROUND}, order="run_id",
    )
    ledger = _select_all(
        service, "lab_arena_ledger", filters={"round_id": ROUND}, order="entry_id",
    )
    return {
        "baseline_runs": [item for item in runs if item.get("submission_id") == BASELINE],
        "baseline_ledger": [item for item in ledger if item.get("submission_id") == BASELINE],
        "nonbaseline_submissions": [
            item for item in submissions if item.get("submission_id") != BASELINE
        ],
        "nonbaseline_runs": [item for item in runs if item.get("submission_id") != BASELINE],
        "nonbaseline_ledger": [
            item for item in ledger if item.get("submission_id") != BASELINE
        ],
    }


def _require_quiescent(runs: list[Mapping[str, Any]], costs: Mapping[str, Any]) -> None:
    if any(run.get("status") in ("pending", "leased", "submitted") for run in runs):
        raise RerunRefused("Sep17 baseline still has an active claim")
    for kind in ("execute", "score"):
        state = costs.get(kind)
        if not isinstance(state, Mapping) or any(
            int(state.get(field, -1)) != 0
            for field in ("inflight_calls", "success_unresolved_calls")
        ):
            raise RerunRefused(f"Sep17 {kind} cost state is not quiescent")


def _bank(service: Any) -> dict[str, Any]:
    payload = json.loads(
        service.config.object_store.get_bounded(BANK_REF, 2 * 1024 * 1024)
    )
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "round_id", "icps"} or (
        payload["schema_version"] != "leadpoet.lab_arena.benchmark.v1"
        or payload["round_id"] != ROUND
        or not isinstance(payload["icps"], list)
        or len(payload["icps"]) != 20
        or len({contracts.document_hash(item) for item in payload["icps"]}) != 20
        or _digest(contracts.canonical_json(payload["icps"]).encode()) != BANK_SHA256
    ):
        raise RerunRefused("frozen September 16 ICP bank differs")
    return payload


def collect_preflight(service: Any) -> dict[str, Any]:
    """Collect exact private terminal rows after the canary; this performs no write."""
    row = service.store.get_round(ROUND)
    baseline = service.store.get_submission(BASELINE)
    if not row or row.get("status") != "published" or not row.get("reward_activated_at"):
        raise RerunRefused("Sep17 round is not an activated published terminal")
    if not baseline or baseline.get("round_id") != ROUND or baseline.get("source_ref") != TERMINAL_SOURCE_REF:
        raise RerunRefused("Sep17 recovery285 baseline identity differs")
    _bank(service)
    terminal_size = int(baseline.get("source_size_bytes") or 0)
    terminal = bytes(
        service.config.object_store.get_bounded(TERMINAL_SOURCE_REF, terminal_size)
    )
    terminal_facts = _source_facts(terminal, TERMINAL_SOURCE_REF)
    expected_doc = baseline.get("submission_doc") or {}
    if any(expected_doc.get(key) != terminal_facts[key] for key in (
        "source_ref", "source_size_bytes", "source_sha256", "source_commit"
    )):
        raise RerunRefused("Sep17 recovery285 submission source document differs")
    sealed = _sealed_rows(service)
    runs = sealed["baseline_runs"]
    ledger = sealed["baseline_ledger"]
    costs = {kind: _cost_state(service, kind) for kind in ("execute", "score")}
    _require_quiescent(runs, costs)
    nonbaseline_submissions = sealed["nonbaseline_submissions"]
    nonbaseline_runs = sealed["nonbaseline_runs"]
    nonbaseline_ledger = sealed["nonbaseline_ledger"]
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "read_only": True,
        "production_writes": 0,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "round": copy.deepcopy(row),
        "baseline": copy.deepcopy(baseline),
        "baseline_runs": copy.deepcopy(runs),
        "baseline_ledger": copy.deepcopy(ledger),
        "nonbaseline_submissions": copy.deepcopy(nonbaseline_submissions),
        "nonbaseline_runs": copy.deepcopy(nonbaseline_runs),
        "nonbaseline_ledger": copy.deepcopy(nonbaseline_ledger),
        "cost_state": copy.deepcopy(costs),
        "terminal_source": terminal_facts,
        "bank_sha256": BANK_SHA256,
    }


def _load_preflight(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise RerunRefused("sealed preflight hash differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != PREFLIGHT_SCHEMA or (
        value.get("read_only") is not True or value.get("production_writes") != 0
    ):
        raise RerunRefused("sealed preflight shape differs")
    return value


def _validate_schedule(value: Any) -> dict[str, str]:
    required = {
        "submission_open", "submission_cutoff", "benchmark_deadline",
        "stage_1_start", "stage_1_close", "stage_1_scoring_close",
        "stage_2_start", "stage_2_close", "final_scoring_close",
        "publication_deadline",
    }
    if not isinstance(value, dict) or set(value) != required or not all(
        isinstance(item, str) and item.endswith("Z") for item in value.values()
    ):
        raise RerunRefused("forward schedule shape differs")
    return dict(value)


def prepare(
    service: Any,
    *,
    preflight: Mapping[str, Any],
    forward_schedule: Mapping[str, Any],
    dry_run: bool,
    expected_source_size_bytes: int,
    expected_source_sha256: str,
    expected_source_commit: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    row = service.store.get_round(ROUND)
    baseline = service.store.get_submission(BASELINE)
    if not row or not baseline or baseline.get("round_id") != ROUND:
        raise RerunRefused("current Sep17 identity differs")
    replay = baseline.get("source_ref") == SOURCE_REF
    if not replay and (
        row != preflight.get("round") or baseline != preflight.get("baseline")
        or row.get("status") != "published"
        or baseline.get("source_ref") != TERMINAL_SOURCE_REF
    ):
        raise RerunRefused("current published terminal differs from sealed preflight")
    _bank(service)
    schedule = _validate_schedule(forward_schedule)
    config = copy.deepcopy(row.get("configuration_doc") or {})
    old_schedule = config.get("schedule") or {}
    if set(old_schedule) != set(schedule) or any(
        old_schedule.get(key) != schedule[key]
        for key in ("submission_open", "submission_cutoff")
    ):
        raise RerunRefused("frozen submission intake differs")
    config["schedule"] = schedule
    if (config.get("call_quotas") or {}) != {
        "openrouter": 200, "deepline": 30, "scrapingdog": 30,
    }:
        raise RerunRefused("frozen execution quotas differ")
    contracts.validate_round_configuration(config)
    if not replay:
        now = now or datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(schedule["benchmark_deadline"].replace("Z", "+00:00"))
        if now >= deadline:
            raise RerunRefused("rerun286 admission window has closed")
    terminal_facts = preflight.get("terminal_source")
    if not isinstance(terminal_facts, Mapping):
        raise RerunRefused("sealed terminal source facts are missing")
    terminal_size = int(terminal_facts.get("source_size_bytes") or 0)
    terminal = bytes(service.config.object_store.get_bounded(TERMINAL_SOURCE_REF, terminal_size))
    if _source_facts(terminal, TERMINAL_SOURCE_REF) != dict(terminal_facts):
        raise RerunRefused("recovery285 source archive differs from sealed preflight")
    sealed = _sealed_rows(service)
    current_runs = sealed["baseline_runs"]
    costs = {kind: _cost_state(service, kind) for kind in ("execute", "score")}
    if not replay:
        if current_runs != preflight.get("baseline_runs") or (
            sealed["baseline_ledger"] != preflight.get("baseline_ledger")
        ):
            raise RerunRefused("baseline run or ledger seal differs")
        if sealed["nonbaseline_submissions"] != preflight.get("nonbaseline_submissions") or (
            sealed["nonbaseline_runs"] != preflight.get("nonbaseline_runs")
        ) or sealed["nonbaseline_ledger"] != preflight.get("nonbaseline_ledger"):
            raise RerunRefused("nonbaseline evidence differs from sealed preflight")
        _require_quiescent(current_runs, costs)
    if replay:
        source = bytes(service.config.object_store.get_bounded(
            SOURCE_REF, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        ))
    else:
        fetcher = service.config.baseline_source_fetcher
        if fetcher is None:
            raise RerunRefused("normal public baseline source fetcher is unavailable")
        source = bytes(fetcher(SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    facts = _source_facts(source, SOURCE_REF)
    if facts != {
        "source_ref": SOURCE_REF,
        "source_size_bytes": expected_source_size_bytes,
        "source_sha256": expected_source_sha256,
        "source_commit": expected_source_commit,
    }:
        raise RerunRefused("lab source archive differs from reviewed identity")
    result = {
        "status": "rerun286_preflight_ok", "round_id": ROUND,
        "bank_sha256": BANK_SHA256, **facts,
        "execute_namespace": "rerun286", "score_namespace": "score:rerun286",
        "openrouter_calls_per_icp": 200, "replay": replay,
    }
    if dry_run:
        return result
    objects = service.config.object_store
    if not isinstance(objects, S3ObjectStore):
        raise RerunRefused("live rerun requires the write-once Arena S3 store")
    if not replay:
        objects.put(SOURCE_REF, source)
        if bytes(objects.get_bounded(SOURCE_REF, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)) != source:
            raise RerunRefused("staged source readback differs")
    acknowledged = service.store._transport.rpc(
        "lab_arena_prepare_sep17_published_rerun286_v1",
        {
            "p_source_size_bytes": facts["source_size_bytes"],
            "p_source_sha256": facts["source_sha256"],
            "p_source_commit": facts["source_commit"],
            "p_bank_sha256": BANK_SHA256,
            "p_forward_schedule": schedule,
        },
    )
    if not isinstance(acknowledged, dict) or acknowledged.get("status") not in ("prepared", "existing") or (
        acknowledged.get("round_id") != ROUND
        or acknowledged.get("baseline_execute_assignments") != 20
        or acknowledged.get("execute_namespace") != "rerun286"
        or acknowledged.get("score_namespace") != "score:rerun286"
        or any(acknowledged.get(key) != facts[key] for key in (
            "source_size_bytes", "source_sha256", "source_commit"
        ))
    ):
        raise RerunRefused("rerun286 RPC did not acknowledge the exact prepared cycle")
    return {**result, **acknowledged}


def _write_private(path: Path, value: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode())
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect-preflight")
    collect.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("prepare")
    run.add_argument("--preflight-file", type=Path, required=True)
    run.add_argument("--expected-preflight-sha256", required=True)
    run.add_argument("--forward-schedule-file", type=Path, required=True)
    run.add_argument("--expected-source-size-bytes", type=int, required=True)
    run.add_argument("--expected-source-sha256", required=True)
    run.add_argument("--expected-source-commit", required=True)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from scripts.run_lab_arena_service import load_scoped_environment
        from lab_arena.wiring import build_service_from_environment

        load_scoped_environment(args.environment_file)
        if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
            raise RerunRefused("live Arena environment is required")
        service, _ = build_service_from_environment("live")
        if args.command == "collect-preflight":
            value = collect_preflight(service)
            _write_private(args.output, value)
            print(json.dumps({
                "status": "preflight_collected", "path": str(args.output),
                "sha256": _digest(args.output.read_bytes()),
            }, sort_keys=True))
        else:
            preflight = _load_preflight(args.preflight_file, args.expected_preflight_sha256)
            schedule = json.loads(args.forward_schedule_file.read_text())
            value = prepare(
                service, preflight=preflight, forward_schedule=schedule,
                dry_run=args.dry_run,
                expected_source_size_bytes=args.expected_source_size_bytes,
                expected_source_sha256=args.expected_source_sha256,
                expected_source_commit=args.expected_source_commit,
            )
            print(json.dumps(value, sort_keys=True))
        return 0
    except RerunRefused as exc:
        print(f"Sep17 rerun286 refused: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("Sep17 rerun286 failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
