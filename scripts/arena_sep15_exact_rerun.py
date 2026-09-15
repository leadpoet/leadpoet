#!/usr/bin/env python3
"""Exact operator steps for the authorized Sep15 baseline rerun.

Use only after migrations 256/257 and the tested Tyche lab branch are live.
The normal Arena driver handles claims, attempts, stage close, scores, and
publication. This script only captures the new baseline source, prepares the
same published round, opens its baseline-only scoring, and adopts the already
open Sep16 configuration. It prints no ICP or miner identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEP15_ROUND = "arena-2026-09-15"
SEP16_ROUND = "arena-2026-09-16"
SEP15_BANK_HASH = "4e11123c9098bbba977fdae9e531419a6dea40cf76dacde1c4c83a82df0a95c2"
SEP15_SOURCE_REF = "arena/arena-2026-09-15/sources/baseline-2026-09-15-rerun256.tar.gz"
SEP15_BASIS_HASH = "sha256:502cd6a5234e5680cfdd761a49b5d825d2064ff3fb063fc8fe7d7f1755e96a18"


class ExactRerunRefused(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded(objects: Any, ref: str, limit: int) -> bytes:
    return bytes(objects.get_bounded(ref, limit))


def _bank_proof(service: Any, row: Mapping[str, Any]) -> str:
    from lab_arena import contracts

    if row.get("round_id") != SEP15_ROUND or row.get("benchmark_ref") != (
        "arena/arena-2026-09-15/benchmark.json"
    ):
        raise ExactRerunRefused("Sep15 benchmark identity differs")
    payload = _read_bounded(service.config.object_store, str(row["benchmark_ref"]), 2 * 1024 * 1024)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExactRerunRefused("Sep15 benchmark bytes are invalid") from exc
    if set(document) != {"schema_version", "round_id", "icps"} or (
        document["schema_version"] != "leadpoet.lab_arena.benchmark.v1"
        or document["round_id"] != SEP15_ROUND
        or len(document["icps"]) != 20
    ):
        raise ExactRerunRefused("Sep15 benchmark document differs")
    if len({contracts.document_hash(icp) for icp in document["icps"]}) != 20:
        raise ExactRerunRefused("Sep15 benchmark contains duplicate ICPs")
    digest = _sha256(contracts.canonical_json(document).encode("utf-8"))
    if digest != SEP15_BANK_HASH:
        raise ExactRerunRefused("Sep15 benchmark hash differs")
    return digest


def _source_proof(service: Any, expected_commit: str) -> tuple[bytes, str, str]:
    from lab_arena import source_bundle
    from lab_arena.service import DEFAULT_BASELINE_SOURCE_URL

    if len(expected_commit) != 40 or any(ch not in "0123456789abcdef" for ch in expected_commit):
        raise ExactRerunRefused("expected lab commit must be a full SHA-1")
    fetcher = service.config.baseline_source_fetcher
    if fetcher is None:
        raise ExactRerunRefused("public source fetcher is unavailable")
    payload = bytes(fetcher(DEFAULT_BASELINE_SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    source_bundle.validate_source_archive(payload, require_license=True)
    observed_commit = source_bundle.source_archive_commit(payload)
    if observed_commit != expected_commit:
        raise ExactRerunRefused("lab archive commit differs from the pushed commit")
    return payload, _sha256(payload), observed_commit


def _schedule_proof(row: Mapping[str, Any], path: Path,
                    verified_parallel_runner_slots: int) -> dict[str, Any]:
    from lab_arena import contracts

    try:
        schedule = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExactRerunRefused("forward schedule file is invalid") from exc
    config = dict(row.get("configuration_doc") or {})
    if not 1 <= verified_parallel_runner_slots <= 20:
        raise ExactRerunRefused("verified parallel runner slots must be 1..20")
    original = config.get("schedule") or {}
    if not isinstance(schedule, dict) or set(schedule) != set(original) or (
        schedule.get("submission_open") != original.get("submission_open")
        or schedule.get("submission_cutoff") != original.get("submission_cutoff")
    ):
        raise ExactRerunRefused("forward schedule must preserve the original intake windows")
    config.update(
        schedule=schedule,
        icp_wall_clock_seconds=2700,
        lease_ttl_seconds=3600,
        runner_slot_ceiling=20,
        parallel_twenty_icp_execution=True,
        checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY,
    )
    contracts.validate_round_configuration(config)
    now = datetime.now(timezone.utc)
    close = datetime.fromisoformat(schedule["stage_1_close"].replace("Z", "+00:00"))
    execution_waves = math.ceil(20 / verified_parallel_runner_slots)
    if (close - now).total_seconds() < execution_waves * (2700 + 60):
        raise ExactRerunRefused("Sep15 execution window cannot hold all twenty 45-minute runs")
    first_scoring_close = datetime.fromisoformat(
        schedule["stage_1_scoring_close"].replace("Z", "+00:00")
    )
    final_scoring_close = datetime.fromisoformat(
        schedule["final_scoring_close"].replace("Z", "+00:00")
    )
    stage2_close = datetime.fromisoformat(
        schedule["stage_2_close"].replace("Z", "+00:00")
    )
    scoring_waves = 2 * math.ceil(10 / verified_parallel_runner_slots)
    scoring_wave_seconds = int(config["scoring_wall_clock_seconds"]) + 60
    if (first_scoring_close - close).total_seconds() < scoring_waves * scoring_wave_seconds or (
        final_scoring_close - stage2_close
    ).total_seconds() < scoring_waves * scoring_wave_seconds:
        raise ExactRerunRefused("Sep15 baseline scoring windows need two attempt waves")
    return schedule


def _prepare(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = service.store.get_round(SEP15_ROUND)
    if row is None or row.get("status") != "published" or (
        row.get("reward_basis_hash") != SEP15_BASIS_HASH
    ):
        raise ExactRerunRefused("original Sep15 publication differs")
    bank_hash = _bank_proof(service, row)
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    schedule = _schedule_proof(row, args.forward_schedule_file,
                               args.verified_parallel_runner_slots)
    if args.dry_run:
        return {
            "status": "preflight_ok", "round_id": SEP15_ROUND,
            "bank_sha256": bank_hash, "source_sha256": source_hash,
            "source_commit": commit, "source_size_bytes": len(source),
            "verified_parallel_runner_slots": args.verified_parallel_runner_slots,
        }
    try:
        existing = _read_bounded(
            service.config.object_store, SEP15_SOURCE_REF, len(source) + 1
        )
    except Exception as exc:
        raise ExactRerunRefused("rerun source must be staged before release authority") from exc
    if existing != source:
        raise ExactRerunRefused("rerun source readback differs")
    result = service.store._transport.rpc("lab_arena_prepare_sep15_baseline_rerun_v1", {
        "p_source_size_bytes": len(source),
        "p_source_sha256": source_hash,
        "p_source_commit": commit,
        "p_bank_sha256": bank_hash,
        "p_forward_schedule": schedule,
    })
    if not isinstance(result, dict) or result.get("status") not in ("prepared", "existing"):
        raise ExactRerunRefused("Sep15 prepare RPC did not acknowledge the exact override")
    return dict(result)


def _stage_source(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = service.store.get_round(SEP15_ROUND)
    if row is None or row.get("status") != "published" or (
        row.get("reward_basis_hash") != SEP15_BASIS_HASH
    ):
        raise ExactRerunRefused("original Sep15 publication differs")
    bank_hash = _bank_proof(service, row)
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    _schedule_proof(row, args.forward_schedule_file,
                    args.verified_parallel_runner_slots)
    result = {
        "status": "source_preflight_ok" if args.dry_run else "source_staged",
        "round_id": SEP15_ROUND, "bank_sha256": bank_hash,
        "source_sha256": source_hash, "source_commit": commit,
        "source_size_bytes": len(source), "source_ref": SEP15_SOURCE_REF,
        "verified_parallel_runner_slots": args.verified_parallel_runner_slots,
    }
    if args.dry_run:
        return result
    try:
        existing = _read_bounded(service.config.object_store,
                                 SEP15_SOURCE_REF, len(source) + 1)
    except Exception:
        # This object key is new and delete-denied; a duplicate write fails.
        service.config.object_store.put(SEP15_SOURCE_REF, source)
    else:
        if existing != source:
            raise ExactRerunRefused("rerun source object holds different bytes")
    if _read_bounded(service.config.object_store,
                     SEP15_SOURCE_REF, len(source) + 1) != source:
        raise ExactRerunRefused("rerun source readback differs")
    return result


def _open_scoring(service: Any, stage: int) -> dict[str, Any]:
    if stage not in (1, 2):
        raise ExactRerunRefused("Sep15 scoring stage must be one or two")
    row = service.store.get_round(SEP15_ROUND)
    if row is None or row.get("status") != f"stage{stage}_closed" or (
        row.get("reward_basis_hash") != SEP15_BASIS_HASH
    ):
        raise ExactRerunRefused("Sep15 baseline scoring stage is not ready")
    original_open_scoring = service.store.open_scoring

    def exact_open_scoring(round_id: str, requested_stage: int, work_items: Any, *,
                           integrity_cache: bool = False,
                           company_quality_cache: bool = False) -> dict[str, Any]:
        if round_id != SEP15_ROUND or requested_stage != stage or (
            not integrity_cache or company_quality_cache
        ):
            raise ExactRerunRefused("normal scorer did not select Sep15 integrity mode")
        result = service.store._transport.rpc("lab_arena_open_sep15_baseline_scoring_v1", {
            "p_round_id": round_id, "p_stage": stage,
            "p_work_items": [dict(item) for item in work_items],
        })
        if not isinstance(result, dict):
            raise ExactRerunRefused("Sep15 scorer RPC returned no document")
        return result

    try:
        service.store.open_scoring = exact_open_scoring
        result = service.open_scoring(SEP15_ROUND, stage)
    finally:
        service.store.open_scoring = original_open_scoring
    if result.get("status") != "ok" or result.get("assignments") != 10:
        raise ExactRerunRefused("Sep15 baseline scoring did not open ten assignments")
    return dict(result)


def _audit(service: Any) -> dict[str, Any]:
    """Read only the exact public state needed after an uncertain RPC reply."""
    row = service.store.get_round(SEP15_ROUND)
    if row is None or row.get("reward_basis_hash") != SEP15_BASIS_HASH or (
        row.get("reward_activated_at") is None or row.get("king_outcome") != "no_king"
    ):
        raise ExactRerunRefused("Sep15 activated reward basis differs")
    original_runs = service.store.list_runs(SEP15_ROUND)
    archived_runs = service.store.list_runs("arena-2026-09-15-archive")
    archived_ledger = []
    for offset in range(0, 100000, 500):
        page = service.store._transport.select(
            "lab_arena_ledger",
            filters={"submission_id": "baseline-2026-09-15-archive"},
            columns="entry_id,entry_kind,amount_microusd",
            order="entry_id", limit=500, offset=offset,
        )
        archived_ledger.extend(page)
        if len(page) < 500:
            break
    else:
        raise ExactRerunRefused("archived ledger audit exceeds bounded read")
    baseline = service.store.get_submission("baseline-2026-09-15")
    new_source = bool(baseline and baseline.get("source_ref") == SEP15_SOURCE_REF)
    result = {
        "round_id": SEP15_ROUND,
        "round_status": row["status"],
        "old_activated_basis_hash": row["reward_basis_hash"],
        "checkpoint_deadline_policy":
            (row.get("configuration_doc") or {}).get("checkpoint_deadline_policy"),
        "new_baseline_source_selected": new_source,
        "archived_baseline_runs": sum(
            run.get("submission_id") == "baseline-2026-09-15-archive"
            for run in archived_runs
        ),
        "archived_actual_microusd": sum(
            int(entry.get("amount_microusd") or 0) for entry in archived_ledger
            if entry.get("entry_kind") in ("settlement", "uncertain")
        ),
        "preserved_challenger_runs": sum(
            run.get("submission_id") != "baseline-2026-09-15"
            for run in original_runs
        ),
    }
    for stage in (1, 2):
        result["stage%d_baseline_execute_assignments" % stage] = sum(
            run.get("submission_id") == "baseline-2026-09-15"
            and run.get("kind") == "execute" and int(run.get("stage") or 0) == stage
            and str(run.get("assignment_id") or "").endswith(":rerun256")
            for run in original_runs
        )
        result["stage%d_baseline_score_assignments" % stage] = sum(
            run.get("submission_id") == "baseline-2026-09-15"
            and run.get("kind") == "score" and int(run.get("stage") or 0) == stage
            and str(run.get("assignment_id") or "").endswith(":score:rerun256")
            for run in original_runs
        )
    return result


def _adopt_sep16(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    from lab_arena import capacity, contracts

    row = service.store.get_round(SEP16_ROUND)
    if row is None or row.get("status") != "open":
        raise ExactRerunRefused("Sep16 must still be open")
    config = dict(row.get("configuration_doc") or {})
    if config.get("parallel_twenty_icp_execution") is True:
        raise ExactRerunRefused("Sep16 already adopted; inspect audit before replay")
    try:
        forward_schedule = json.loads(args.forward_schedule_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExactRerunRefused("Sep16 forward schedule file is invalid") from exc
    original_schedule = config.get("schedule") or {}
    if not isinstance(forward_schedule, dict) or set(forward_schedule) != set(original_schedule) or any(
        forward_schedule.get(field) != original_schedule.get(field)
        for field in ("submission_open", "submission_cutoff", "benchmark_deadline", "stage_1_start")
    ):
        raise ExactRerunRefused("Sep16 forward schedule must preserve intake and stage-one start")
    verified_slots = args.verified_parallel_runner_slots
    if not 1 <= verified_slots <= 20 or len(config.get("runner_hotkeys") or []) != 1:
        raise ExactRerunRefused("Sep16 needs one frozen runner and 1..20 verified physical slots")
    config.update(
        schedule=forward_schedule,
        icp_wall_clock_seconds=2700,
        lease_ttl_seconds=3600,
        runner_slot_ceiling=20,
        parallel_twenty_icp_execution=True,
        checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY,
    )
    contracts.validate_round_configuration(config)
    accepted = len(service.store.list_submissions(
        SEP16_ROUND, status="accepted", columns="submission_id"
    ))
    modeled = dict(config, runner_slot_ceiling=verified_slots)
    configured_slots = verified_slots
    attempts = int(config["max_attempts_per_assignment"])
    first_phase_waves = attempts * math.ceil(20 * (accepted + 1) / configured_slots)
    scoring_waves = attempts * math.ceil(10 * (accepted + 1) / configured_slots)
    stage1_start = datetime.fromisoformat(
        config["schedule"]["stage_1_start"].replace("Z", "+00:00")
    )
    stage1_close = datetime.fromisoformat(
        config["schedule"]["stage_1_close"].replace("Z", "+00:00")
    )
    if (stage1_close - max(stage1_start, datetime.now(timezone.utc))).total_seconds() < first_phase_waves * 2760:
        raise ExactRerunRefused(
            "Sep16 accepted parallel workload exceeds the frozen stage1 window"
        )
    scoring_wave_seconds = int(config["scoring_wall_clock_seconds"]) + 60
    stage1_scoring_close = datetime.fromisoformat(
        config["schedule"]["stage_1_scoring_close"].replace("Z", "+00:00")
    )
    stage2_close = datetime.fromisoformat(
        config["schedule"]["stage_2_close"].replace("Z", "+00:00")
    )
    final_scoring_close = datetime.fromisoformat(
        config["schedule"]["final_scoring_close"].replace("Z", "+00:00")
    )
    if (stage1_scoring_close - stage1_close).total_seconds() < scoring_waves * scoring_wave_seconds or (
        final_scoring_close - stage2_close
    ).total_seconds() < scoring_waves * scoring_wave_seconds:
        raise ExactRerunRefused("Sep16 accepted scoring workload exceeds phase window")
    supported = capacity.daily_challenger_capacity(modeled)
    if supported < 1 or accepted > supported:
        raise ExactRerunRefused("Sep16 measured runner capacity is below accepted workload")
    proof = {
        "round_id": SEP16_ROUND,
        "validated_by": "arena_service_capacity_v1",
        "parallel_twenty_icp_execution": True,
        "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
        "icp_wall_clock_seconds": 2700,
        "runner_slot_ceiling": 20,
        "verified_parallel_runner_slots": verified_slots,
        "configured_challenger_capacity": supported,
        "already_accepted_challengers": accepted,
        "runner_hotkeys": config["runner_hotkeys"],
        "schedule": config["schedule"],
        "configuration_doc": config,
    }
    if args.dry_run:
        return {"status": "preflight_ok", "round_id": SEP16_ROUND,
                "configured_challenger_capacity": supported,
                "already_accepted_challengers": accepted,
                "verified_parallel_runner_slots": verified_slots,
                "max_challengers": config["max_challengers"],
                "future_admissions_may_exceed_measured_capacity":
                    int(config["max_challengers"]) > supported,
                "first_phase_retry_reserved_waves_at_verified_slots":
                    first_phase_waves,
                "scoring_retry_reserved_waves_per_stage": scoring_waves}
    transport = service.store._transport
    result = transport.rpc("lab_arena_adopt_sep16_open_config_v1", {
        "p_capacity_doc": proof,
    })
    if not isinstance(result, dict) or result.get("status") not in ("adopted", "existing"):
        raise ExactRerunRefused("Sep16 adoption RPC did not acknowledge the exact change")
    return dict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact Sep15 Arena baseline rerun")
    parser.add_argument("--environment-file", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("stage-source", "prepare"):
        source_step = commands.add_parser(command)
        source_step.add_argument("--expected-lab-commit", required=True)
        source_step.add_argument("--forward-schedule-file", type=Path, required=True)
        source_step.add_argument("--verified-parallel-runner-slots", type=int, required=True)
        source_step.add_argument("--dry-run", action="store_true")
    scoring = commands.add_parser("open-scoring")
    scoring.add_argument("--stage", type=int, choices=(1, 2), required=True)
    commands.add_parser("audit")
    sep16 = commands.add_parser("adopt-sep16")
    sep16.add_argument("--forward-schedule-file", type=Path, required=True)
    sep16.add_argument("--verified-parallel-runner-slots", type=int, required=True)
    sep16.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from scripts.run_lab_arena_service import load_scoped_environment
        from lab_arena.wiring import build_service_from_environment

        load_scoped_environment(args.environment_file)
        if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
            raise ExactRerunRefused("live Arena environment is required")
        service, _app = build_service_from_environment("live")
        if args.command == "stage-source":
            result = _stage_source(service, args)
        elif args.command == "prepare":
            result = _prepare(service, args)
        elif args.command == "open-scoring":
            result = _open_scoring(service, args.stage)
        elif args.command == "audit":
            result = _audit(service)
        else:
            result = _adopt_sep16(service, args)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except ExactRerunRefused as exc:
        print("exact Arena procedure refused: %s" % exc, file=sys.stderr)
        return 2
    except Exception:
        # Source/provider exceptions may retain private request data.
        print("exact Arena procedure failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
