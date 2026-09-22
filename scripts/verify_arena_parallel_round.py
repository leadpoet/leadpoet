#!/usr/bin/env python3
"""Run one paid, reward-disabled production shadow round for all twenty ICPs.

The process owns one explicit round id.  It serves the normal Arena API on
loopback for a production validator and advances only that round.  It never
creates daily rounds, reviews submissions, promotes a baseline, or activates
rewards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARTIFACT_SCHEMA_VERSION = "leadpoet.lab_arena.parallel_round_verification.v1"
DEFAULT_PORT = 18792
DEFAULT_TICK_SECONDS = 5
TERMINAL_STATUSES = frozenset(("published", "cancelled"))
LEDGER_FIELDS = (
    "settled_microusd",
    "reserved_or_uncertain_microusd",
    "inflight_calls",
    "uncertain_calls",
    "refused_calls",
    "call_count",
    "successful_microusd",
    "successful_calls",
    "success_unresolved_microusd",
    "success_unresolved_calls",
)


class VerificationError(RuntimeError):
    """The requested process could affect the wrong round or cannot prove it."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _positive_port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be from 1 through 65535")
    return parsed


def _tick_seconds(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 60:
        raise argparse.ArgumentTypeError("tick seconds must be from 1 through 60")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--round-id", required=True)
        command.add_argument("--environment-file", type=Path, required=True)
        command.add_argument(
            "--status-file",
            type=Path,
            help="atomic JSON evidence path; defaults to /tmp/<round-id>.json",
        )

    serve = commands.add_parser("serve", help="create or resume, serve, and drive one round")
    common(serve)
    serve.add_argument("--port", type=_positive_port, default=DEFAULT_PORT)
    serve.add_argument(
        "--tick-seconds", type=_tick_seconds, default=DEFAULT_TICK_SECONDS
    )
    status = commands.add_parser("status", help="write and print current evidence")
    common(status)
    return parser


def _status_path(args: argparse.Namespace) -> Path:
    return args.status_file or Path("/tmp") / (args.round_id + ".json")


def _validate_round_id(round_id: str) -> str:
    from lab_arena import contracts

    if contracts.ROUND_ID_RE.fullmatch(str(round_id or "")) is None:
        raise VerificationError("round id is invalid")
    date_prefix = str(round_id)[:16]
    if not str(round_id).startswith(date_prefix + "-"):
        raise VerificationError("round id must have an explicit suffix")
    try:
        datetime.strptime(date_prefix, "arena-%Y-%m-%d")
    except ValueError as exc:
        raise VerificationError("round id date is invalid") from exc
    return str(round_id)


def _build_pinned_service(round_id: str):
    """Use production dependencies with a process-local shadow ownership gate."""

    source_mode = os.environ.get("LAB_ARENA_MODE", "").strip().lower()
    if source_mode != "live":
        raise VerificationError("the source environment must be the live gateway environment")
    from lab_arena.api import create_app
    from lab_arena import contracts
    from lab_arena.service import ArenaService, DEFAULT_BASELINE_SOURCE_URL
    from lab_arena.wiring import build_service_from_environment

    built, _unused_app = build_service_from_environment("shadow")
    defaults = replace(
        built.config.defaults,
        benchmark_icp_count=contracts.BENCHMARK_ICP_COUNT,
        promotion_margin=1.0,
        rewards_enabled=False,
        daily_cutoff_hour_utc=None,
        baseline_source_url=DEFAULT_BASELINE_SOURCE_URL,
    )
    config = replace(
        built.config,
        mode="shadow",
        pinned_round_id=round_id,
        defaults=defaults,
        reward_signer_factory=None,
        baseline_promoter_factory=None,
        code_reviewer=None,
    )
    service = ArenaService(config)
    return service, create_app(service)


def _validate_frozen_round(service: Any, row: Mapping[str, Any]) -> None:
    from lab_arena import contracts, icp_disclosure
    from lab_arena.service import DEFAULT_BASELINE_SOURCE_URL

    configuration = row.get("configuration_doc") or {}
    defaults = service.config.defaults
    expected = {
        "round_id": row.get("round_id"),
        "mode": "shadow",
        "network_name": service.config.network_name,
        "netuid": service.config.netuid,
        "rewards_enabled": False,
        "stage_1_icp_count": contracts.STAGE_1_ICP_COUNT,
        "stage_2_icp_count": contracts.STAGE_2_ICP_COUNT,
        "runner_slot_ceiling": contracts.RUNNER_SLOT_CEILING,
        "baseline_hotkey": defaults.baseline_hotkey,
        "baseline_source_url": DEFAULT_BASELINE_SOURCE_URL,
        "scorer_image_digest": defaults.scorer_image_digest,
        "scorer_image_reference": defaults.scorer_image_reference,
        "runner_hotkeys": list(defaults.runner_hotkeys),
    }
    mismatches = sorted(
        name for name, value in expected.items() if configuration.get(name) != value
    )
    sequence = configuration.get("execution_sequence_policy")
    if sequence == contracts.BASELINE_SCORED_FIRST_POLICY:
        if configuration.get("parallel_twenty_icp_execution"):
            mismatches.append("execution_sequence_policy")
    elif sequence is not None or configuration.get("parallel_twenty_icp_execution") is not True:
        mismatches.append("execution_sequence_policy")
    if configuration.get("benchmark_disclosure_policy") not in {
        icp_disclosure.DELAYED_DISCLOSURE_POLICY,
        icp_disclosure.CUTOFF_PUBLIC_POLICY,
    }:
        mismatches.append("benchmark_disclosure_policy")
    if row.get("round_id") != service.config.pinned_round_id:
        mismatches.append("pinned_round_id")
    if row.get("champion_submission_id") is not None:
        mismatches.append("champion_submission_id")
    if row.get("status") != "open" and row.get("champion_funding_frozen") is not True:
        mismatches.append("champion_funding_frozen")
    participants = list(row.get("participants") or [])
    if participants and (
        len(participants) != 1
        or participants[0].get("is_king") is not True
        or participants[0].get("miner_hotkey") != defaults.baseline_hotkey
    ):
        mismatches.append("baseline_only_participants")
    if mismatches:
        raise VerificationError(
            "existing round is incompatible: " + ",".join(sorted(set(mismatches)))
        )


def _create_or_resume(service: Any, round_id: str, *, now: datetime) -> str:
    from lab_arena.service import round_id_for_cutoff

    existing = service.store.get_round(round_id)
    if existing is not None:
        _validate_frozen_round(service, existing)
        submissions = service.store.list_submissions(round_id)
        if any(not row.get("is_king") for row in submissions):
            raise VerificationError("existing round contains a challenger submission")
        return "resumed"
    if not round_id.startswith(round_id_for_cutoff(now) + "-"):
        raise VerificationError("a new round id must use the current UTC date")
    configuration = service.create_round(now, round_id=round_id)
    if configuration.get("mode") != "shadow":
        raise VerificationError("created round is not shadow mode")
    created = service.store.get_round(round_id)
    if created is None:
        raise VerificationError("created round is unavailable")
    _validate_frozen_round(service, created)
    return "created"


def _ledger_totals(costs: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in costs.get("providers") or []:
        kind = str(row.get("kind") or "")
        aggregate = totals.setdefault(kind, {field: 0 for field in LEDGER_FIELDS})
        for field in LEDGER_FIELDS:
            aggregate[field] += int(row.get(field) or 0)
    return totals


def _run_counts(runs: list[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in runs if row.get("kind") == kind]
    assignments = {str(row.get("assignment_id") or "") for row in selected}
    accepted = [row for row in selected if row.get("status") == "accepted"]
    accepted_assignments = {
        str(row.get("assignment_id") or "") for row in accepted
    }
    accepted_output_refs = [
        str(row.get("output_ref") or "") for row in accepted
    ]
    by_submission: dict[str, list[int]] = {}
    for row in accepted:
        by_submission.setdefault(str(row.get("submission_id") or ""), []).append(
            int(row.get("icp_position"))
        )
    return {
        "attempt_rows": len(selected),
        "assignment_count": len(assignments),
        "accepted_attempts": len(accepted),
        "accepted_assignment_count": len(accepted_assignments),
        "accepted_output_ref_count": len(set(accepted_output_refs) - {""}),
        "accepted_result_count": sum(
            (row.get("result_doc") or {}).get("terminal_status") == "accepted"
            for row in accepted
        ),
        "scored_assignment_count": sum(
            row.get("per_icp_score") is not None for row in accepted
        ),
        "failed_attempts": sum(row.get("status") == "failed" for row in selected),
        "open_attempts": sum(
            row.get("status") in ("pending", "leased", "submitted")
            for row in selected
        ),
        "accepted_positions_by_submission": {
            submission_id: sorted(positions)
            for submission_id, positions in sorted(by_submission.items())
        },
    }


def _parse_result_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _interval_high_water(
    intervals: list[tuple[datetime, datetime]],
) -> int:
    """Return a conservative lower bound from second-resolution intervals."""

    events: list[tuple[datetime, int]] = []
    for started, finished in intervals:
        if finished > started:
            events.extend(((started, 1), (finished, -1)))
    active = 0
    high_water = 0
    # Process finishes before starts at the same timestamp. This avoids
    # claiming overlap that the one-second result timestamps cannot prove.
    for _timestamp, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        high_water = max(high_water, active)
    return high_water


def _execution_timing(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build conservative attempt-overlap evidence from immutable results.

    Timestamp overlap does not prove physical sandbox concurrency. The paid
    operator run must retain its independent sandbox PID observations.
    """

    from lab_arena import contracts

    accepted = [
        row
        for row in runs
        if row.get("kind") == "execute" and row.get("status") == "accepted"
    ]
    intervals: list[tuple[datetime, datetime]] = []
    by_position: dict[int, tuple[datetime, datetime]] = {}
    worker_slots: set[int] = set()
    exit_fingerprints: set[str] = set()
    runner_hotkeys: set[str] = set()
    invalid_results = 0
    for row in accepted:
        result = row.get("result_doc") or {}
        started = _parse_result_time(result.get("started_at"))
        finished = _parse_result_time(result.get("finished_at"))
        web_egress = (result.get("resource_summary") or {}).get("web_egress") or {}
        position = row.get("icp_position")
        slot = web_egress.get("worker_slot")
        fingerprint = str(web_egress.get("exit_fingerprint") or "")
        runner = str(row.get("runner_hotkey") or "")
        if (
            result.get("terminal_status") != "accepted"
            or started is None
            or finished is None
            or finished < started
            or isinstance(position, bool)
            or not isinstance(position, int)
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or not fingerprint
            or not runner
            or position in by_position
        ):
            invalid_results += 1
            continue
        intervals.append((started, finished))
        by_position[position] = (started, finished)
        worker_slots.add(slot)
        exit_fingerprints.add(fingerprint)
        runner_hotkeys.add(runner)

    attempt_intervals: list[tuple[int, datetime, datetime]] = [
        (position, interval[0], interval[1])
        for position, interval in by_position.items()
        if position in range(contracts.BENCHMARK_ICP_COUNT)
    ]
    failed_attempt_interval_count = 0
    invalid_failed_attempt_interval_count = 0
    reported_failures = contracts.MODEL_CAUSED_TERMINAL_CAUSES | {"provider_error"}
    for row in runs:
        if row.get("kind") != "execute" or row.get("status") != "failed":
            continue
        result = row.get("result_doc") or {}
        started = _parse_result_time(result.get("started_at"))
        finished = _parse_result_time(result.get("finished_at"))
        web_egress = (result.get("resource_summary") or {}).get("web_egress") or {}
        position = row.get("icp_position")
        slot = web_egress.get("worker_slot")
        fingerprint = str(web_egress.get("exit_fingerprint") or "")
        runner = str(row.get("runner_hotkey") or "")
        if (
            result.get("terminal_status") not in reported_failures
            or started is None
            or finished is None
            or finished < started
            or isinstance(position, bool)
            or not isinstance(position, int)
            or position not in range(contracts.BENCHMARK_ICP_COUNT)
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot not in worker_slots
            or fingerprint not in exit_fingerprints
            or runner not in runner_hotkeys
        ):
            invalid_failed_attempt_interval_count += 1
            continue
        attempt_intervals.append((position, started, finished))
        failed_attempt_interval_count += 1

    all_attempt_intervals = [
        (started, finished) for _position, started, finished in attempt_intervals
    ]
    first_batch_attempt_intervals = [
        (started, finished)
        for position, started, finished in attempt_intervals
        if position < 10
    ]
    second_batch_attempt_intervals = [
        (started, finished)
        for position, started, finished in attempt_intervals
        if position >= 10
    ]

    # The barrier opens only after every first-batch position has an accepted
    # result. Its other edge is the earliest valid attempt in the second batch,
    # including an attempt that failed before its accepted retry.
    first_batch = [by_position.get(position) for position in range(10)]
    first_finished = (
        max(interval[1] for interval in first_batch if interval is not None)
        if all(interval is not None for interval in first_batch)
        else None
    )
    second_started = (
        min(interval[0] for interval in second_batch_attempt_intervals)
        if second_batch_attempt_intervals
        else None
    )
    return {
        "accepted_interval_count": len(intervals),
        "invalid_result_count": invalid_results,
        "failed_attempt_interval_count": failed_attempt_interval_count,
        "invalid_failed_attempt_interval_count": (
            invalid_failed_attempt_interval_count
        ),
        "runner_hotkeys": sorted(runner_hotkeys),
        "worker_slots": sorted(worker_slots),
        "exit_fingerprint_count": len(exit_fingerprints),
        "concurrency_evidence": "overlapping_valid_attempt_intervals",
        "physical_concurrency_evidence": (
            "operator_sandbox_pid_observations_required"
        ),
        "concurrency_high_water_lower_bound": _interval_high_water(
            all_attempt_intervals
        ),
        "first_batch_concurrency_high_water_lower_bound": _interval_high_water(
            first_batch_attempt_intervals
        ),
        "second_batch_concurrency_high_water_lower_bound": _interval_high_water(
            second_batch_attempt_intervals
        ),
        "first_batch_latest_finish": (
            first_finished.isoformat().replace("+00:00", "Z")
            if first_finished is not None
            else None
        ),
        "second_batch_earliest_start": (
            second_started.isoformat().replace("+00:00", "Z")
            if second_started is not None
            else None
        ),
        # SQL independently requires terminal acceptance before the next wave.
        # Result timestamps prove execution separation, not commit timestamps.
        "second_batch_started_after_first_finished": (
            first_finished is not None
            and second_started is not None
            and second_started >= first_finished
        ),
    }


def _verify_host_funding(
    service: Any, round_id: str, verified_run_ids: set[str]
) -> None:
    """Verify each immutable run payer once before it can be accepted."""

    from lab_arena import contracts

    row = service.store.get_round(round_id)
    if row is None or row.get("champion_submission_id") is not None:
        raise VerificationError("shadow round selected champion funding")
    for run in service.store.list_runs(round_id):
        run_id = str(run.get("run_id") or "")
        if not run_id or run_id in verified_run_ids:
            continue
        for provider in contracts.PROVIDERS:
            funding = service.store.provider_funding(run_id, provider)
            if (
                funding.get("status") != "available"
                or funding.get("funding_source") != "host"
                or funding.get("champion_funding") is not False
                or funding.get("credential_submission_id") is not None
                or funding.get("credential_miner_hotkey") is not None
            ):
                raise VerificationError("run funding is not organizer-hosted")
        verified_run_ids.add(run_id)


def _ledger_funding(service: Any, runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    sources = set()
    entries = 0
    for run in runs:
        for row in service.store.list_ledger(run_id=str(run["run_id"])):
            entries += 1
            sources.add(str(row.get("funding_source") or ""))
    return {
        "entry_count": entries,
        "sources": sorted(sources),
        "all_host": not sources or sources == {"host"},
    }


def _durable_output_counts(
    service: Any, row: Mapping[str, Any], runs: list[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    """Read and validate accepted private objects without disclosing their data."""

    from lab_arena import contact_policy, contracts, integrity, scoring
    from lab_arena.output import MAX_OUTPUT_BYTES, validate_output_document

    configuration = row.get("configuration_doc") or {}
    accepted_identity = tuple(
        sorted(
            (
                str(run.get("kind") or ""),
                str(run.get("run_id") or ""),
                str(run.get("output_ref") or ""),
                str(run.get("output_hash") or ""),
                str(run.get("scored_run_id") or ""),
            )
            for run in runs
            if run.get("kind") in ("execute", "score")
            and run.get("status") == "accepted"
        )
    )
    cached = getattr(service, "_parallel_verification_durable_cache", None)
    if cached is not None and cached[0] == accepted_identity:
        return cached[1]
    counts = {
        "execute": {
            "accepted_count": 0,
            "verified_count": 0,
            "stored_hash_count": 0,
            "invalid_count": 0,
        },
        "score": {
            "accepted_count": 0,
            "verified_count": 0,
            "stored_hash_count": 0,
            "invalid_count": 0,
        },
    }
    for run in runs:
        kind = str(run.get("kind") or "")
        if kind not in counts or run.get("status") != "accepted":
            continue
        counts[kind]["accepted_count"] += 1
        try:
            output_ref = run.get("output_ref")
            if not isinstance(output_ref, str) or not output_ref:
                raise ValueError("accepted output reference is missing")
            maximum = (
                MAX_OUTPUT_BYTES
                if kind == "execute"
                else scoring.MAX_SCORING_OUTPUT_BYTES
            )
            raw = service._objects.get_bounded(output_ref, maximum)
            document = json.loads(raw.decode("utf-8"))
            stored_hash = run.get("output_hash")
            if isinstance(stored_hash, str) and stored_hash:
                counts[kind]["stored_hash_count"] += 1
                if contracts.document_hash(document) != stored_hash:
                    raise ValueError("accepted output hash does not match")
            if kind == "execute":
                validate_output_document(
                    document,
                    expected_schema_version=contact_policy.output_schema(configuration),
                    require_intent_dates=not integrity.enabled(configuration),
                )
            else:
                validated = scoring.validate_scoring_output_document(document)
                if (
                    "failure" in validated
                    or validated.get("scored_run_id") != run.get("scored_run_id")
                ):
                    raise ValueError("accepted score object does not match its run")
        except Exception:
            counts[kind]["invalid_count"] += 1
        else:
            counts[kind]["verified_count"] += 1
    service._parallel_verification_durable_cache = (accepted_identity, counts)
    return counts


def _evidence(service: Any, round_id: str) -> dict[str, Any]:
    from lab_arena import contact_policy, contracts, icp_disclosure

    row = service.store.get_round(round_id)
    if row is None:
        raise VerificationError("round does not exist")
    _validate_frozen_round(service, row)
    configuration = row.get("configuration_doc") or {}
    participants = list(row.get("participants") or [])
    runs = service.store.list_runs(round_id) if participants else []
    execution = _run_counts(runs, "execute")
    scoring_runs = _run_counts(runs, "score")
    execution_timing = _execution_timing(runs)
    durable_outputs = (
        _durable_output_counts(service, row, runs)
        if row.get("status") in TERMINAL_STATUSES
        else None
    )
    disclosure_metadata = icp_disclosure.disclosure_metadata(row)
    disclosure = icp_disclosure.baseline_disclosure(
        row, runs, now=service.now()
    )
    disclosure_evidence = {
        "status": "ready" if disclosure is not None else "pending",
        "public_at": (
            disclosure_metadata.get("public_at")
            if disclosure_metadata is not None
            else None
        ),
        "policy": (
            disclosure_metadata.get("disclosure_policy")
            if disclosure_metadata is not None
            else None
        ),
    }
    costs = []
    for participant in participants:
        aggregate = service.store.submission_costs(participant["submission_id"])
        costs.append(
            {
                "submission_id": participant["submission_id"],
                "providers": aggregate["providers"],
                "totals": _ledger_totals(aggregate),
            }
        )

    publication = row.get("publication_doc") or {}
    final_ranking = list(publication.get("final_ranking") or [])
    final_results = [
        {
            key: result.get(key)
            for key in (
                "submission_id",
                "miner_hotkey",
                "is_baseline",
                "rank",
                "final_score",
                "eligible",
                "eligibility_reason",
                "cost_summary",
            )
            if key in result
        }
        for result in final_ranking
    ]
    public_outputs = None
    ledger_funding = (
        _ledger_funding(service, runs)
        if row.get("status") in TERMINAL_STATUSES
        else {"entry_count": None, "sources": [], "all_host": None}
    )
    if row.get("status") == "published" and len(participants) == 1:
        public = service.public_results(round_id, participants[0]["submission_id"])
        public_outputs = {
            "output_count": len(public.get("outputs") or {}),
            "run_result_count": len(public.get("run_results") or []),
            "stage_1_score_count": len(
                (public.get("scores") or {}).get("stage_1") or []
            ),
            "stage_2_score_count": len(
                (public.get("scores") or {}).get("stage_2") or []
            ),
            "scored_positions": sorted(
                int(score["icp_position"])
                for score in (public.get("scores") or {}).get("stage_1", [])
                + (public.get("scores") or {}).get("stage_2", [])
            ),
            "public_icp_status": public.get("public_icp_status"),
            "public_icp_count": public.get("public_icp_count"),
            "contact_verifications_present": "contact_verifications" in public,
            "contact_verification_count": len(
                public.get("contact_verifications") or {}
            ),
            "submission_scores": public.get("submission_scores"),
        }

    errors = []
    if row.get("status") == "published":
        expected_positions = list(range(contracts.BENCHMARK_ICP_COUNT))
        expected_assignments = len(participants) * contracts.BENCHMARK_ICP_COUNT
        if len(participants) != 1:
            errors.append("participant_count")
        for name, counts in (("execute", execution), ("score", scoring_runs)):
            if counts["assignment_count"] != expected_assignments:
                errors.append(name + "_assignment_count")
            if counts["accepted_assignment_count"] != expected_assignments:
                errors.append(name + "_accepted_assignment_count")
            if counts["accepted_attempts"] != expected_assignments:
                errors.append(name + "_accepted_uniqueness")
            if counts["accepted_output_ref_count"] != expected_assignments:
                errors.append(name + "_accepted_output_refs")
            if counts["accepted_result_count"] != expected_assignments:
                errors.append(name + "_accepted_results")
            if counts["open_attempts"] != 0:
                errors.append(name + "_open_attempts")
            if len(participants) == 1 and counts[
                "accepted_positions_by_submission"
            ].get(participants[0]["submission_id"]) != expected_positions:
                errors.append(name + "_positions")
        if execution["scored_assignment_count"] != expected_assignments:
            errors.append("execute_scores")
        if execution_timing["accepted_interval_count"] != expected_assignments:
            errors.append("execution_timestamps")
        if execution_timing["invalid_result_count"]:
            errors.append("execution_result_evidence")
        configured_runners = set(
            (row.get("configuration_doc") or {}).get("runner_hotkeys") or []
        )
        if (
            len(execution_timing["runner_hotkeys"]) != 1
            or execution_timing["runner_hotkeys"][0] not in configured_runners
        ):
            errors.append("execution_runner_identity")
        if execution_timing["worker_slots"] != list(range(10)):
            errors.append("execution_worker_slots")
        if execution_timing["exit_fingerprint_count"] != 10:
            errors.append("execution_exit_fingerprints")
        if execution_timing["concurrency_high_water_lower_bound"] != 10:
            errors.append("execution_high_water")
        if configuration.get("parallel_twenty_icp_execution") is True:
            # The legacy policy requires two isolated ten-ICP waves. Current
            # baseline-first scheduling refills slots across all twenty ICPs.
            if execution_timing["first_batch_concurrency_high_water_lower_bound"] != 10:
                errors.append("execution_first_batch_high_water")
            if execution_timing["second_batch_concurrency_high_water_lower_bound"] != 10:
                errors.append("execution_second_batch_high_water")
            if not execution_timing["second_batch_started_after_first_finished"]:
                errors.append("execution_batch_barrier")
        if len(final_results) != 1 or final_results[0].get("final_score") is None:
            errors.append("final_aggregate")
        if ledger_funding["all_host"] is not True:
            errors.append("ledger_funding_source")
        if int(ledger_funding["entry_count"] or 0) < 1:
            errors.append("paid_ledger_empty")
        if disclosure_metadata is None:
            errors.append("disclosure_metadata")
        for kind in ("execute", "score"):
            durable = (durable_outputs or {}).get(kind) or {}
            if (
                durable.get("accepted_count") != expected_assignments
                or durable.get("verified_count") != expected_assignments
                or durable.get("invalid_count") != 0
            ):
                errors.append("durable_" + kind + "_outputs")
        if disclosure is not None:
            if (
                public_outputs is None
                or public_outputs["public_icp_status"] != "ready"
                or public_outputs["public_icp_count"]
                != contracts.BENCHMARK_ICP_COUNT
                or public_outputs["output_count"] != contracts.BENCHMARK_ICP_COUNT
                or public_outputs["scored_positions"] != expected_positions
            ):
                errors.append("public_outputs")
        elif (
            public_outputs is None
            or public_outputs["public_icp_status"] != "pending"
            or public_outputs["public_icp_count"] != 0
            or public_outputs["output_count"] != 0
            or public_outputs["run_result_count"] != 0
            or public_outputs["stage_1_score_count"] != 0
            or public_outputs["stage_2_score_count"] != 0
            or public_outputs["scored_positions"]
            or public_outputs["contact_verification_count"] != 0
            or (
                contact_policy.enabled(configuration)
                and not public_outputs["contact_verifications_present"]
            )
        ):
            errors.append("public_outputs_pending_contract")
        if len(costs) == 1 and len(final_results) == 1:
            raw = costs[0]["totals"]
            published_cost = final_results[0].get("cost_summary") or {}
            for kind, bucket in (("execute", "execution"), ("score", "judge")):
                if int((published_cost.get(bucket) or {}).get("settled_microusd") or 0) != int(
                    (raw.get(kind) or {}).get("settled_microusd") or 0
                ):
                    errors.append(kind + "_settled_cost")
                if int((raw.get(kind) or {}).get("call_count") or 0) < 1:
                    errors.append(kind + "_paid_calls")
                if int((raw.get(kind) or {}).get("settled_microusd") or 0) < 1:
                    errors.append(kind + "_paid_cost")
            if any(
                int((raw.get(kind) or {}).get(field) or 0)
                for kind in ("execute", "score")
                for field in (
                    "inflight_calls",
                    "uncertain_calls",
                    "reserved_or_uncertain_microusd",
                    "success_unresolved_calls",
                    "success_unresolved_microusd",
                )
            ):
                errors.append("open_costs")
        else:
            errors.append("cost_aggregate")

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "observed_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "round": {
            "round_id": round_id,
            "status": row.get("status"),
            "status_generation": row.get("status_generation"),
            "stage_generation": row.get("stage_generation"),
            "mode": configuration.get("mode"),
            "rewards_enabled": configuration.get("rewards_enabled"),
            "runner_slot_ceiling": configuration.get("runner_slot_ceiling"),
            "parallel_twenty_icp_execution": configuration.get(
                "parallel_twenty_icp_execution"
            ),
            "execution_sequence_policy": configuration.get("execution_sequence_policy"),
            "baseline_source_url": configuration.get("baseline_source_url"),
            "scorer_image_reference": configuration.get("scorer_image_reference"),
            "king_outcome": row.get("king_outcome"),
            "cancel_reason": row.get("cancel_reason"),
            "champion_funding_frozen": row.get("champion_funding_frozen"),
            "champion_submission_id": row.get("champion_submission_id"),
        },
        "participant_count": len(participants),
        "execution": execution,
        "execution_timing": execution_timing,
        "scoring": scoring_runs,
        "durable_outputs": durable_outputs,
        "disclosure": disclosure_evidence,
        "ledger": costs,
        "ledger_funding": ledger_funding,
        "final_ranking": final_results,
        "public_result": public_outputs,
        "proof": {
            "complete": row.get("status") == "published" and not errors,
            "errors": errors,
        },
    }


def _write_artifact(path: Path, document: Mapping[str, Any]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise VerificationError("status file parent does not exist")
    temporary = parent / ("." + path.name + ".tmp-%d" % os.getpid())
    encoded = (json.dumps(document, indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _tick(service: Any, round_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    advance = service.advance_round(round_id)
    reconciliation = service.reconcile_closed_provider_costs()
    return advance, reconciliation


def _serve(args: argparse.Namespace, service: Any, app: Any) -> int:
    import uvicorn

    state = _create_or_resume(service, args.round_id, now=_utc_now())
    status_path = _status_path(args)
    # Freeze the baseline before the loopback API becomes reachable.  The
    # cutoff is already current UTC, so this uses the normal open->committed
    # transition and leaves no submission interval on the verification host.
    current = service.store.get_round(args.round_id)
    if current is not None and current.get("status") == "open":
        committed = service.advance_round(args.round_id)
        if committed.get("status") not in ("ok", "existing"):
            raise VerificationError("baseline commit did not complete")
    current = service.store.get_round(args.round_id)
    if current is not None and current.get("status") == "committed":
        opened = service.advance_round(args.round_id)
        if opened.get("status") not in ("ok", "existing"):
            raise VerificationError("parallel execution did not open")
    verified_run_ids: set[str] = set()
    _verify_host_funding(service, args.round_id, verified_run_ids)
    initial = _evidence(service, args.round_id)
    _write_artifact(status_path, initial)
    print(
        json.dumps(
            {
                "action": state,
                "bind": "127.0.0.1",
                "port": args.port,
                "round_id": args.round_id,
                "status_file": str(status_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stop = threading.Event()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="info",
        )
    )
    final_document = initial
    driver_failure: list[str] = []

    def driver() -> None:
        nonlocal final_document
        while not stop.is_set() and not server.started:
            stop.wait(0.05)
        while not stop.is_set():
            try:
                advance, reconciliation = _tick(service, args.round_id)
                _verify_host_funding(service, args.round_id, verified_run_ids)
                final_document = _evidence(service, args.round_id)
                final_document["last_transition"] = advance
                final_document["last_reconciliation"] = reconciliation
                _write_artifact(status_path, final_document)
            except Exception as exc:
                driver_failure[:] = [type(exc).__name__]
                print(
                    "parallel round driver failed: %s" % type(exc).__name__,
                    file=sys.stderr,
                    flush=True,
                )
                stop.set()
                server.should_exit = True
                return
            round_status = final_document["round"]["status"]
            if round_status == "cancelled" or (
                round_status == "published"
                and final_document["proof"]["complete"]
            ):
                stop.set()
                server.should_exit = True
                return
            if round_status == "published" and reconciliation.get("status") == "none":
                driver_failure[:] = ["VerificationIncomplete"]
                stop.set()
                server.should_exit = True
                return
            stop.wait(args.tick_seconds)

    thread = threading.Thread(
        target=driver, name="arena-parallel-round-driver", daemon=True
    )
    thread.start()
    try:
        server.run()
    finally:
        stop.set()
        thread.join(timeout=max(5, args.tick_seconds + 1))
        try:
            final_document = _evidence(service, args.round_id)
            _write_artifact(status_path, final_document)
        except Exception as exc:
            driver_failure[:] = driver_failure or [type(exc).__name__]
    if driver_failure:
        return 1
    return 0 if final_document.get("proof", {}).get("complete") else 1


def run(args: argparse.Namespace, service: Any, app: Any) -> int:
    args.round_id = _validate_round_id(args.round_id)
    if args.command == "status":
        row = service.store.get_round(args.round_id)
        if row is None:
            raise VerificationError("round does not exist")
        _validate_frozen_round(service, row)
        document = _evidence(service, args.round_id)
        _write_artifact(_status_path(args), document)
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
        return 0
    return _serve(args, service, app)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
        args.round_id = _validate_round_id(args.round_id)
        service, app = _build_pinned_service(args.round_id)
        checks = service.startup_checks()
        print(
            json.dumps(
                {
                    "startup": {
                        key: value
                        for key, value in checks.items()
                        if key != "database_identity"
                    },
                    "database_role": (checks.get("database_identity") or {}).get(
                        "current_user"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return run(args, service, app)
    except VerificationError as exc:
        print("parallel round verification refused: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:
        # Provider and storage exceptions can retain request details.  Print
        # only the class at this outer boundary.
        print(
            "parallel round verification failed: %s" % type(exc).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
