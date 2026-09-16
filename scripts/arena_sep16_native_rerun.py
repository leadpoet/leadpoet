#!/usr/bin/env python3
"""Exact operator steps for the sealed Sep16 native TYCHE rerun."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.arena_sep15_exact_rerun import (  # noqa: E402
    ExactRerunRefused,
    _read_bounded,
    _sha256,
)

ROUND = "arena-2026-09-16"
ARCHIVE_ROUND = "arena-2026-09-16-archive"
RECOVERY_ARCHIVE_ROUND = "arena-2026-09-16-rerun265archive"
RECOVERY272_ARCHIVE_ROUND = "arena-2026-09-16-rerun269archive"
BASELINE = "baseline-2026-09-16"
BASIS_HASH = "sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f"
BANK_HASH = "42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390"
SOURCE_URL = "https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz"
SOURCE_REF = "arena/arena-2026-09-16/sources/baseline-2026-09-16-native-rerun268.tar.gz"
RECOVERY_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery269.tar.gz"
)
RECOVERY272_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery272.tar.gz"
)
VERIFIED_RUNNER_SLOTS = 10
RECOVERY_SUFFIX = ":rerun269"
RECOVERY272_SUFFIX = ":rerun272"


def _require_round(row: Mapping[str, Any] | None, status: str) -> Mapping[str, Any]:
    if row is None or row.get("round_id") != ROUND or row.get("status") != status or (
        row.get("reward_basis_hash") != BASIS_HASH
    ):
        raise ExactRerunRefused("sealed Sep16 round state differs")
    return row


def _bank_proof(service: Any, row: Mapping[str, Any]) -> str:
    from lab_arena import contracts

    if row.get("benchmark_ref") != "arena/arena-2026-09-16/benchmark.json":
        raise ExactRerunRefused("Sep16 benchmark identity differs")
    payload = _read_bounded(
        service.config.object_store, str(row["benchmark_ref"]), 2 * 1024 * 1024
    )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExactRerunRefused("Sep16 benchmark bytes are invalid") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "round_id", "icps"
    } or (
        document["schema_version"] != "leadpoet.lab_arena.benchmark.v1"
        or document["round_id"] != ROUND
        or not isinstance(document["icps"], list)
        or len(document["icps"]) != 20
        or len({contracts.document_hash(icp) for icp in document["icps"]}) != 20
    ):
        raise ExactRerunRefused("Sep16 benchmark document differs")
    digest = _sha256(contracts.canonical_json(document["icps"]).encode("utf-8"))
    if digest != BANK_HASH:
        raise ExactRerunRefused("Sep16 benchmark hash differs")
    return digest


def _source_proof(service: Any, expected_commit: str) -> tuple[bytes, str, str]:
    from lab_arena import source_bundle

    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ExactRerunRefused("expected champion_model lab commit must be a full SHA-1")
    fetcher = service.config.baseline_source_fetcher
    if fetcher is None:
        raise ExactRerunRefused("public source fetcher is unavailable")
    payload = bytes(fetcher(SOURCE_URL, source_bundle.MAX_SOURCE_ARCHIVE_BYTES))
    source_bundle.validate_source_archive(payload, require_license=True)
    observed_commit = source_bundle.source_archive_commit(payload)
    if observed_commit != expected_commit:
        raise ExactRerunRefused("champion_model lab archive commit differs")
    return payload, _sha256(payload), observed_commit


def _schedule_proof(row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    from lab_arena import contracts

    try:
        schedule = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExactRerunRefused("Sep16 forward schedule file is invalid") from exc
    config = dict(row.get("configuration_doc") or {})
    original = config.get("schedule") or {}
    if not isinstance(schedule, dict) or set(schedule) != set(original) or any(
        schedule.get(field) != original.get(field)
        for field in ("submission_open", "submission_cutoff")
    ):
        raise ExactRerunRefused("Sep16 forward schedule changes the frozen intake")
    config.update(
        schedule=schedule,
        baseline_source_url=SOURCE_URL,
        runner_slot_ceiling=VERIFIED_RUNNER_SLOTS,
    )
    contracts.validate_round_configuration(config)
    now = datetime.now(timezone.utc)
    stage1_close = datetime.fromisoformat(schedule["stage_1_close"].replace("Z", "+00:00"))
    stage1_scoring_close = datetime.fromisoformat(
        schedule["stage_1_scoring_close"].replace("Z", "+00:00")
    )
    stage2_start = datetime.fromisoformat(schedule["stage_2_start"].replace("Z", "+00:00"))
    stage2_close = datetime.fromisoformat(schedule["stage_2_close"].replace("Z", "+00:00"))
    final_scoring_close = datetime.fromisoformat(
        schedule["final_scoring_close"].replace("Z", "+00:00")
    )
    publication_deadline = datetime.fromisoformat(
        schedule["publication_deadline"].replace("Z", "+00:00")
    )
    execution_seconds = math.ceil(20 / VERIFIED_RUNNER_SLOTS) * 2760
    scoring_seconds = 2 * math.ceil(10 / VERIFIED_RUNNER_SLOTS) * 960
    if (stage1_close - now).total_seconds() < execution_seconds or (
        stage1_scoring_close - stage1_close
    ).total_seconds() < scoring_seconds or stage2_start <= stage1_scoring_close or (
        stage2_close <= stage2_start
    ) or (final_scoring_close - stage2_close).total_seconds() < scoring_seconds or (
        publication_deadline <= final_scoring_close
    ) or (publication_deadline - now).total_seconds() >= 24 * 3600:
        raise ExactRerunRefused("Sep16 forward schedule lacks the sealed execution windows")
    return schedule


def _stage_source(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = _require_round(service.store.get_round(ROUND), "published")
    bank_hash = _bank_proof(service, row)
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    _schedule_proof(row, args.forward_schedule_file)
    result = {
        "status": "source_preflight_ok" if args.dry_run else "source_staged",
        "round_id": ROUND,
        "bank_sha256": bank_hash,
        "source_sha256": source_hash,
        "source_commit": commit,
        "source_size_bytes": len(source),
        "source_ref": SOURCE_REF,
        "verified_parallel_runner_slots": VERIFIED_RUNNER_SLOTS,
    }
    if args.dry_run:
        return result
    try:
        existing = _read_bounded(service.config.object_store, SOURCE_REF, len(source) + 1)
    except Exception:
        service.config.object_store.put(SOURCE_REF, source)
    else:
        if existing != source:
            raise ExactRerunRefused("Sep16 native source object holds different bytes")
    if _read_bounded(service.config.object_store, SOURCE_REF, len(source) + 1) != source:
        raise ExactRerunRefused("Sep16 native source readback differs")
    return result


def _prepare(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = _require_round(service.store.get_round(ROUND), "published")
    bank_hash = _bank_proof(service, row)
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    schedule = _schedule_proof(row, args.forward_schedule_file)
    if args.dry_run:
        return {
            "status": "preflight_ok", "round_id": ROUND,
            "bank_sha256": bank_hash, "source_sha256": source_hash,
            "source_commit": commit, "source_size_bytes": len(source),
            "verified_parallel_runner_slots": VERIFIED_RUNNER_SLOTS,
        }
    try:
        existing = _read_bounded(service.config.object_store, SOURCE_REF, len(source) + 1)
    except Exception as exc:
        raise ExactRerunRefused("Sep16 native source must be staged before prepare") from exc
    if existing != source:
        raise ExactRerunRefused("Sep16 native source readback differs")
    result = service.store._transport.rpc(
        "lab_arena_prepare_sep16_baseline_rerun_v1",
        {
            "p_source_size_bytes": len(source), "p_source_sha256": source_hash,
            "p_source_commit": commit, "p_bank_sha256": bank_hash,
            "p_forward_schedule": schedule,
        },
    )
    if not isinstance(result, dict) or result.get("status") not in ("prepared", "existing"):
        raise ExactRerunRefused("Sep16 prepare RPC did not acknowledge the sealed rerun")
    return dict(result)


def _terminal_recovery_source_proof(
    service: Any, source_ref: str = SOURCE_REF,
) -> tuple[int, str, str]:
    from lab_arena import source_bundle

    submission = service.store.get_submission(BASELINE) or {}
    document = dict(submission.get("submission_doc") or {})
    if submission.get("source_ref") != source_ref or document.get("source_ref") != source_ref:
        raise ExactRerunRefused("Sep16 terminal recovery source reference differs")
    try:
        size = int(submission["source_size_bytes"])
        source_hash = str(document["source_sha256"])
        commit = str(document["source_commit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactRerunRefused("Sep16 terminal recovery source proof is incomplete") from exc
    if (
        not 0 < size <= source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ExactRerunRefused("Sep16 terminal recovery source proof is invalid")
    payload = _read_bounded(service.config.object_store, source_ref, size + 1)
    if len(payload) != size or _sha256(payload) != source_hash:
        raise ExactRerunRefused("Sep16 terminal recovery source readback differs")
    source_bundle.validate_source_archive(payload, require_license=True)
    if source_bundle.source_archive_commit(payload) != commit:
        raise ExactRerunRefused("Sep16 terminal recovery source commit differs")
    return size, source_hash, commit


def _recover(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = service.store.get_round(ROUND)
    if row is None or row.get("round_id") != ROUND or row.get("reward_basis_hash") != BASIS_HASH:
        raise ExactRerunRefused("sealed Sep16 round state differs")
    if row.get("status") != "cancelled":
        raise ExactRerunRefused("Sep16 recovery round status differs")
    bank_hash = _bank_proof(service, row)
    terminal_size, terminal_hash, terminal_commit = _terminal_recovery_source_proof(
        service
    )
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    schedule = _schedule_proof(row, args.forward_schedule_file)
    if args.dry_run:
        return {
            "status": "recovery_preflight_ok", "round_id": ROUND,
            "bank_sha256": bank_hash, "source_sha256": source_hash,
            "source_commit": commit, "source_size_bytes": len(source),
            "source_ref": RECOVERY_SOURCE_REF,
            "terminal_source_sha256": terminal_hash,
            "terminal_source_commit": terminal_commit,
            "terminal_source_size_bytes": terminal_size,
            "execute_namespace": RECOVERY_SUFFIX.removeprefix(":"),
            "verified_parallel_runner_slots": VERIFIED_RUNNER_SLOTS,
        }
    try:
        existing = _read_bounded(
            service.config.object_store, RECOVERY_SOURCE_REF, len(source) + 1
        )
    except Exception:
        service.config.object_store.put(RECOVERY_SOURCE_REF, source)
    else:
        if existing != source:
            raise ExactRerunRefused("Sep16 recovery source object holds different bytes")
    if _read_bounded(
        service.config.object_store, RECOVERY_SOURCE_REF, len(source) + 1
    ) != source:
        raise ExactRerunRefused("Sep16 recovery source readback differs")
    result = service.store._transport.rpc(
        "lab_arena_prepare_sep16_baseline_recovery_v1",
        {
            "p_source_size_bytes": len(source),
            "p_source_sha256": source_hash,
            "p_source_commit": commit,
            "p_forward_schedule": schedule,
        },
    )
    if not isinstance(result, dict) or result.get("status") not in ("prepared", "existing"):
        raise ExactRerunRefused("Sep16 recovery RPC did not acknowledge the sealed rerun")
    return dict(result)


def _recover272(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = service.store.get_round(ROUND)
    if row is None or row.get("round_id") != ROUND or row.get("reward_basis_hash") != BASIS_HASH:
        raise ExactRerunRefused("sealed Sep16 round state differs")
    if row.get("status") != "cancelled":
        raise ExactRerunRefused("Sep16 recovery272 round status differs")
    bank_hash = _bank_proof(service, row)
    terminal_size, terminal_hash, terminal_commit = _terminal_recovery_source_proof(
        service, RECOVERY_SOURCE_REF
    )
    source, source_hash, commit = _source_proof(service, args.expected_lab_commit)
    schedule = _schedule_proof(row, args.forward_schedule_file)
    if args.dry_run:
        return {
            "status": "recovery272_preflight_ok", "round_id": ROUND,
            "bank_sha256": bank_hash, "source_sha256": source_hash,
            "source_commit": commit, "source_size_bytes": len(source),
            "source_ref": RECOVERY272_SOURCE_REF,
            "terminal_source_sha256": terminal_hash,
            "terminal_source_commit": terminal_commit,
            "terminal_source_size_bytes": terminal_size,
            "execute_namespace": RECOVERY272_SUFFIX.removeprefix(":"),
            "verified_parallel_runner_slots": VERIFIED_RUNNER_SLOTS,
        }
    try:
        existing = _read_bounded(
            service.config.object_store, RECOVERY272_SOURCE_REF, len(source) + 1
        )
    except Exception:
        service.config.object_store.put(RECOVERY272_SOURCE_REF, source)
    else:
        if existing != source:
            raise ExactRerunRefused("Sep16 recovery272 source object holds different bytes")
    if _read_bounded(
        service.config.object_store, RECOVERY272_SOURCE_REF, len(source) + 1
    ) != source:
        raise ExactRerunRefused("Sep16 recovery272 source readback differs")
    result = service.store._transport.rpc(
        "lab_arena_prepare_sep16_baseline_recovery272_v1",
        {
            "p_source_size_bytes": len(source),
            "p_source_sha256": source_hash,
            "p_source_commit": commit,
            "p_forward_schedule": schedule,
        },
    )
    if not isinstance(result, dict) or result.get("status") not in ("prepared", "existing"):
        raise ExactRerunRefused("Sep16 recovery272 RPC did not acknowledge the sealed rerun")
    return dict(result)


def _open_scoring(service: Any, stage: int) -> dict[str, Any]:
    if stage not in (1, 2):
        raise ExactRerunRefused("Sep16 scoring stage must be one or two")
    _require_round(service.store.get_round(ROUND), f"stage{stage}_closed")
    original_open_scoring = service.store.open_scoring
    expected_assignments: int | None = None

    def exact_open_scoring(
        round_id: str, requested_stage: int, work_items: Any, *,
        integrity_cache: bool = False, company_quality_cache: bool = False,
    ) -> dict[str, Any]:
        nonlocal expected_assignments
        if round_id != ROUND or requested_stage != stage or (
            not integrity_cache or company_quality_cache
        ):
            raise ExactRerunRefused("normal scorer did not select Sep16 integrity mode")
        items = [dict(item) for item in work_items]
        expected_assignments = sum(item.get("submission_id") == BASELINE for item in items)
        if not 0 <= expected_assignments <= 10:
            raise ExactRerunRefused("Sep16 committed baseline scoring count differs")
        result = service.store._transport.rpc(
            "lab_arena_open_sep16_baseline_scoring_v1",
            {"p_round_id": round_id, "p_stage": stage, "p_work_items": items},
        )
        if not isinstance(result, dict):
            raise ExactRerunRefused("Sep16 scorer RPC returned no document")
        return result

    try:
        service.store.open_scoring = exact_open_scoring
        result = service.open_scoring(ROUND, stage)
    finally:
        service.store.open_scoring = original_open_scoring
    if result.get("status") not in ("ok", "existing") or (
        result.get("assignments") != expected_assignments
    ):
        raise ExactRerunRefused("Sep16 baseline scoring acknowledgement differs")
    return dict(result)


def _audit(service: Any) -> dict[str, Any]:
    row = service.store.get_round(ROUND)
    if row is None or row.get("reward_basis_hash") != BASIS_HASH:
        raise ExactRerunRefused("Sep16 activated reward basis differs")
    runs = service.store.list_runs(ROUND)
    archived = service.store.list_runs(ARCHIVE_ROUND)
    recovery_archived = service.store.list_runs(RECOVERY_ARCHIVE_ROUND)
    recovery272_archived = service.store.list_runs(RECOVERY272_ARCHIVE_ROUND)
    rerun265_execute = {
        run.get("assignment_id") for run in runs
        if run.get("submission_id") == BASELINE and run.get("kind") == "execute"
        and str(run.get("assignment_id") or "").endswith(":rerun265")
    }
    rerun269_execute = {
        run.get("assignment_id") for run in runs
        if run.get("submission_id") == BASELINE and run.get("kind") == "execute"
        and str(run.get("assignment_id") or "").endswith(RECOVERY_SUFFIX)
    }
    rerun272_execute = {
        run.get("assignment_id") for run in runs
        if run.get("submission_id") == BASELINE and run.get("kind") == "execute"
        and str(run.get("assignment_id") or "").endswith(RECOVERY272_SUFFIX)
    }
    active_namespace = (
        "rerun272" if rerun272_execute
        else "rerun269" if rerun269_execute
        else "rerun265" if rerun265_execute else None
    )
    active_score_suffix = (
        ":score:" + active_namespace if active_namespace is not None else None
    )
    return {
        "round_id": ROUND,
        "round_status": row.get("status"),
        "new_baseline_source_selected": bool(
            (service.store.get_submission(BASELINE) or {}).get("source_ref")
            == (
                RECOVERY272_SOURCE_REF if active_namespace == "rerun272"
                else RECOVERY_SOURCE_REF if active_namespace == "rerun269"
                else SOURCE_REF
            )
        ),
        "archived_baseline_runs": len(archived),
        "archived_failed_rerun_runs": len(recovery_archived),
        "archived_failed_recovery269_runs": len(recovery272_archived),
        "active_rerun_namespace": active_namespace,
        "rerun265_execute_assignments": len(rerun265_execute),
        "rerun269_execute_assignments": len(rerun269_execute),
        "rerun272_execute_assignments": len(rerun272_execute),
        "rerun_execute_assignments": len(
            rerun272_execute if active_namespace == "rerun272"
            else rerun269_execute if active_namespace == "rerun269"
            else rerun265_execute
        ),
        "rerun_score_assignments": len({
            run.get("assignment_id") for run in runs
            if run.get("submission_id") == BASELINE and run.get("kind") == "score"
            and active_score_suffix is not None
            and str(run.get("assignment_id") or "").endswith(active_score_suffix)
        }),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact Sep16 native TYCHE baseline rerun")
    parser.add_argument("--environment-file", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("stage-source", "prepare"):
        source_step = commands.add_parser(command)
        source_step.add_argument("--expected-lab-commit", required=True)
        source_step.add_argument("--forward-schedule-file", type=Path, required=True)
        source_step.add_argument("--dry-run", action="store_true")
    recovery = commands.add_parser("recover")
    recovery.add_argument("--expected-lab-commit", required=True)
    recovery.add_argument("--forward-schedule-file", type=Path, required=True)
    recovery.add_argument("--dry-run", action="store_true")
    recovery272 = commands.add_parser("recover272")
    recovery272.add_argument("--expected-lab-commit", required=True)
    recovery272.add_argument("--forward-schedule-file", type=Path, required=True)
    recovery272.add_argument("--dry-run", action="store_true")
    scoring = commands.add_parser("open-scoring")
    scoring.add_argument("--stage", type=int, choices=(1, 2), required=True)
    commands.add_parser("audit")
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
        elif args.command == "recover":
            result = _recover(service, args)
        elif args.command == "recover272":
            result = _recover272(service, args)
        elif args.command == "open-scoring":
            result = _open_scoring(service, args.stage)
        else:
            result = _audit(service)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except ExactRerunRefused as exc:
        print("exact Sep16 Arena procedure refused: %s" % exc, file=sys.stderr)
        return 2
    except Exception:
        print("exact Sep16 Arena procedure failed; inspect redacted service logs", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
