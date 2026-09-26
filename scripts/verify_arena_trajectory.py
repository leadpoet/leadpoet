#!/usr/bin/env python3
"""Run one pinned Arena lease or prove four private trajectory executions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "leadpoet.lab_arena.trajectory_verification.v1"
ROLES = ("baseline", "miner")
PATHS = ("primary", "external-like")
PROVIDERS = ("openrouter", "deepline", "scrapingdog")
ROUND_RE = re.compile(r"^arena-\d{4}-\d{2}-\d{2}-trajectory(?:-[a-z0-9]+)+$")
IDENTITY_FIELDS = (
    "run_id", "round_id", "submission_id", "miner_hotkey", "runner_hotkey",
    "assignment_id", "stage", "icp_position", "attempt",
)


class VerificationError(RuntimeError):
    pass


def _round_id(value: str) -> str:
    if ROUND_RE.fullmatch(value or "") is None:
        raise argparse.ArgumentTypeError(
            "round id must be arena-YYYY-MM-DD-trajectory-<label>"
        )
    try:
        datetime.strptime(value[:16], "arena-%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("round id date is invalid") from exc
    return value


def _run_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if (
        len(parts) != 3 or parts[0] not in ROLES or parts[1] not in PATHS
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", parts[2]) is None
    ):
        raise argparse.ArgumentTypeError(
            "run must be baseline|miner:primary|external-like:run-id"
        )
    return parts[0], parts[1], parts[2]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--round-id", type=_round_id, required=True)
    run.add_argument("--environment-file", type=Path, required=True)
    run.add_argument("--expect-role", choices=ROLES, required=True)
    run.add_argument("--expect-kind", choices=("execute", "score"), default="execute")
    run.add_argument("--expect-submission-id", required=True)
    run.add_argument("--path-label", choices=PATHS, required=True)
    run.add_argument("--status-file", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--round-id", type=_round_id, required=True)
    verify.add_argument("--environment-file", type=Path, required=True)
    verify.add_argument("--run", dest="runs", type=_run_spec, action="append", required=True)
    verify.add_argument("--require-runtime-log", action="store_true")
    verify.add_argument("--status-file", type=Path, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--round-id", type=_round_id, required=True)
    serve.add_argument("--environment-file", type=Path, required=True)
    serve.add_argument("--published-icp-round", required=True)
    serve.add_argument("--public-miner-submission", required=True)
    serve.add_argument("--runner-hotkey", action="append", required=True)
    serve.add_argument("--icp-position", type=int, action="append")
    serve.add_argument("--cutoff-minutes", type=int, default=5)
    serve.add_argument("--tick-seconds", type=int, default=5)
    serve.add_argument("--port", type=int, default=19125)
    serve.add_argument("--status-file", type=Path, required=True)
    return root


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from scripts.verify_arena_parallel_round import _write_artifact
    _write_artifact(path, value)


def _load_runner_environment(path: Path) -> None:
    from scripts.run_arena_validator import _read_environment

    values, _raw = _read_environment(path)
    merged = {**os.environ, **values}
    forbidden = []
    for name, value in merged.items():
        upper = name.upper()
        database_secret = (
            "SUPABASE" in upper
            or upper in {
                "DATABASE_URL", "DB_PASSWORD", "PGPASSWORD",
                "POSTGRES_PASSWORD", "LAB_ARENA_SERVICE_KEY",
            }
        )
        provider_secret = (
            any(item in upper for item in ("OPENROUTER", "DEEPLINE", "SCRAPINGDOG"))
            and any(item in upper for item in ("KEY", "TOKEN", "SECRET", "CREDENTIAL"))
        )
        if value and (database_secret or provider_secret):
            forbidden.append(upper)
    if forbidden:
        raise VerificationError(
            "runner has service or provider credentials: " + ",".join(sorted(set(forbidden)))
        )
    os.environ.update(values)


def _public_participant(
    row: Mapping[str, Any], round_id: str, role: str, submission_id: str,
    kind: str = "execute",
) -> None:
    statuses = (
        ("stage1", "stage2")
        if kind == "execute" else ("stage1_scoring", "stage2_scoring")
    )
    if row.get("round_id") != round_id or row.get("status") not in statuses:
        raise VerificationError("round is not the pinned execution round")
    if row.get("benchmark_icp_count") != 2:
        raise VerificationError("probe round must contain exactly two ICPs")
    participants = row.get("participants")
    if not isinstance(participants, list) or len(participants) != 2:
        raise VerificationError("probe round must contain two participants")
    found = {
        "baseline" if item.get("is_baseline") is True else "miner": item.get("submission_id")
        for item in participants if isinstance(item, Mapping)
    }
    if set(found) != set(ROLES) or found[role] != submission_id:
        raise VerificationError("expected baseline/miner participant is absent")


def _run(args: argparse.Namespace) -> int:
    _load_runner_environment(args.environment_file)
    from lab_arena import validator
    from lab_arena.runner import HttpArenaApiClient
    from lab_arena.wiring import build_runner_from_environment

    options = validator._parser().parse_args(["--round-id", args.round_id, "--once"])
    options.poll_seconds = options.poll_seconds or int(
        os.environ.get("LAB_ARENA_VALIDATOR_POLL_SECONDS", "30")
    )
    api = HttpArenaApiClient(str(options.api_base_url).rstrip("/"))
    try:
        _public_participant(
            api.round(args.round_id), args.round_id, args.expect_role,
            args.expect_submission_id, args.expect_kind,
        )
    finally:
        api.close()
    keypair = validator.load_local_hotkey(options)
    runner = build_runner_from_environment(options, keypair=keypair)
    try:
        lease = runner.claim_one(args.round_id)
        if (
            lease.get("status") != "leased" or lease.get("kind") != args.expect_kind
            or lease.get("submission_id") != args.expect_submission_id
        ):
            raise VerificationError("claimed lease does not match the requested execution")
        envelope = runner._executor.execute(lease, str(lease["lease_token"]), lease["icp"])
        complete = runner._complete_with_retries(envelope)
        result = {
            "schema_version": SCHEMA, "round_id": args.round_id,
            "run_id": lease["run_id"], "submission_id": lease["submission_id"],
            "role": args.expect_role, "operator_path_label": args.path_label,
            "run_kind": args.expect_kind,
            "runner_hotkey": keypair.ss58_address,
            "transport_profile": "validator_gateway_only",
            "service_or_provider_credentials_present": False,
            "completion_status": complete.get("status"),
        }
        _write(args.status_file, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if complete.get("status") == "accepted" else 1
    finally:
        runner.close()


def _events(store: Any, run_id: str) -> list[Mapping[str, Any]]:
    method = getattr(store, "list_trajectory_events", None)
    if callable(method):
        return list(method(run_id))
    return list(store._transport.select(
        "lab_arena_trajectory_events", filters={"run_id": run_id},
        order="trajectory_id", limit=10_000,
    ))


def _event_provider(row: Mapping[str, Any]) -> str | None:
    from lab_arena import operations

    content = row.get("content") or {}
    call = content.get("call") if isinstance(content.get("call"), Mapping) else {}
    provider = content.get("provider") or call.get("provider")
    operation_id = content.get("operation_id") or call.get("operation_id")
    if provider in PROVIDERS:
        return provider
    operation = operations.OPERATIONS.get(operation_id)
    return operation.provider if operation is not None else None


def _one_proof(store: Any, round_id: str, spec: tuple[str, str, str], logs: bool) -> dict:
    role, path, run_id = spec
    run = store.get_run(run_id)
    if not isinstance(run, Mapping) or any((
        run.get("round_id") != round_id, run.get("kind") != "execute",
        run.get("status") != "accepted",
        (run.get("result_doc") or {}).get("terminal_status") != "accepted",
    )):
        raise VerificationError("run is not an accepted execution: " + run_id)
    submission = store.get_submission(run["submission_id"])
    actual_role = "baseline" if (submission or {}).get("is_king") is True else "miner"
    if actual_role != role or (submission or {}).get("miner_hotkey") != run.get("miner_hotkey"):
        raise VerificationError("run role is not canonical: " + run_id)
    events = _events(store, run_id)
    expected = {name: run.get(name) for name in IDENTITY_FIELDS}
    expected.update({
        "icp_identifier": "%s:icp:%s" % (round_id, run.get("icp_position")),
        "run_kind": "execute", "model_role": role,
    })
    if not events or any(
        any(event.get(name) != value for name, value in expected.items()) for event in events
    ):
        raise VerificationError("trajectory identity is incomplete: " + run_id)
    kinds = Counter(event.get("event_kind") for event in events)
    finished = [event for event in events if event.get("event_kind") == "runtime.finished"]
    log_count = kinds["runtime.stdout"] + kinds["runtime.stderr"]
    if (
        kinds["runtime.started"] != 1 or len(finished) != 1
        or (finished[0].get("content") or {}).get("status") != "accepted"
        or (logs and log_count < 1)
    ):
        raise VerificationError("runtime trajectory is incomplete: " + run_id)
    provider_events = defaultdict(Counter)
    for event in events:
        suffix = str(event.get("event_kind") or "").removeprefix("provider.")
        if suffix in ("request", "response", "error"):
            provider = _event_provider(event)
            if provider is None:
                raise VerificationError("provider event lacks provider identity")
            provider_events[provider][suffix] += 1
    ledger = list(store.list_ledger(run_id=run_id))
    provider_ledger = {}
    for provider in PROVIDERS:
        selected = [item for item in ledger if item.get("provider") == provider]
        if any(
            item.get("run_id") != run_id or item.get("round_id") != round_id
            or item.get("submission_id") != run.get("submission_id")
            or item.get("miner_hotkey") != run.get("miner_hotkey") for item in selected
        ):
            raise VerificationError("ledger identity is incomplete: " + run_id)
        by_call = defaultdict(set)
        for item in selected:
            by_call[item.get("call_identity")].add(item.get("entry_kind"))
        provider_ledger[provider] = {
            "dispatched": sum("dispatch" in value for value in by_call.values()),
            "terminal": sum(bool(value & {"settlement", "uncertain", "recovery"}) for value in by_call.values()),
        }
    return {
        "run_id": run_id, "role": role, "operator_path_label": path,
        "runner_hotkey": run.get("runner_hotkey"), "model_terminal_status": "accepted",
        "runtime_log_event_count": log_count,
        "provider_events": {provider: dict(provider_events[provider]) for provider in PROVIDERS},
        "provider_ledger": provider_ledger,
    }


def _proof(service: Any, round_id: str, specs: Sequence[tuple[str, str, str]], logs=False) -> dict:
    from lab_arena import contracts

    store = service.store
    row = store.get_round(round_id)
    config = (row or {}).get("configuration_doc") or {}
    participants = list((row or {}).get("participants") or [])
    if (
        config.get("mode") != "shadow" or config.get("rewards_enabled") is not False
        or (row or {}).get("reward_activated_at") is not None
        or (row or {}).get("status") in (None, "cancelled")
        or contracts.benchmark_icp_count(config) != 2
        or len(participants) != 2
        or sum(item.get("is_king") is True for item in participants) != 1
    ):
        raise VerificationError("round is not a private reward-disabled shadow")
    cells = {(role, path) for role in ROLES for path in PATHS}
    if len(specs) != 4 or {(role, path) for role, path, _ in specs} != cells:
        raise VerificationError("runs must cover baseline/miner on both validator paths")
    if len({run_id for _, _, run_id in specs}) != 4:
        raise VerificationError("run IDs must be unique")
    proofs = [_one_proof(store, round_id, spec, logs) for spec in specs]
    hotkeys = defaultdict(set)
    for proof in proofs:
        hotkeys[proof["operator_path_label"]].add(proof["runner_hotkey"])
    if any(len(value) != 1 for value in hotkeys.values()) or len(
        {next(iter(value)) for value in hotkeys.values()}
    ) != 2:
        raise VerificationError("validator labels do not map to two stable hotkeys")
    for provider in PROVIDERS:
        requests = sum(proof["provider_events"][provider].get("request", 0) for proof in proofs)
        terminals = sum(
            proof["provider_events"][provider].get("response", 0)
            + proof["provider_events"][provider].get("error", 0) for proof in proofs
        )
        dispatched = sum(proof["provider_ledger"][provider]["dispatched"] for proof in proofs)
        accounted = sum(proof["provider_ledger"][provider]["terminal"] for proof in proofs)
        if requests < 1 or terminals < requests or dispatched < 1 or accounted < 1:
            raise VerificationError(provider + " provider proof is incomplete")
    return {
        "schema_version": SCHEMA, "round_id": round_id, "round_status": row.get("status"),
        "runs": proofs,
        "validator_path_hotkeys": {path: next(iter(value)) for path, value in hotkeys.items()},
        "proof": {"complete": True, "errors": []},
    }


def _fixture_service(args: argparse.Namespace):
    """Build one pinned shadow and seed one clearly labelled miner fixture."""

    from lab_arena import contracts, source_bundle
    from lab_arena.api import create_app
    from lab_arena.owner_admission import resolve_finalized_owner
    from lab_arena.service import ArenaService, DEFAULT_BASELINE_SOURCE_URL
    from lab_arena.wiring import build_service_from_environment

    positions = tuple(args.icp_position or (0, 1))
    runners = tuple(args.runner_hotkey)
    if (
        len(positions) != 2 or len(set(positions)) != 2 or min(positions) < 0
        or len(runners) != 2 or len(set(runners)) != 2
        or not 1 <= args.cutoff_minutes <= 30 or not 1 <= args.tick_seconds <= 60
        or not 1 <= args.port <= 65535
    ):
        raise VerificationError("fixture bounds are invalid")
    for hotkey in runners:
        contracts.require_hotkey(hotkey)
    built, _app = build_service_from_environment("live")

    source_round = built.store.get_round(args.published_icp_round)
    if not source_round or source_round.get("status") != "published":
        raise VerificationError("ICP source round is not published")
    public = built.public_benchmark(args.published_icp_round)
    public_positions = {int(item["icp_position"]) for item in public.get("icps") or []}
    if not set(positions).issubset(public_positions):
        raise VerificationError("selected ICP positions are not public")
    source_icps = built.benchmark_icps(args.published_icp_round)
    selected_icps = [
        dict(source_icps[position], icp_id="%s:trajectory:%d" % (args.round_id, index))
        for index, position in enumerate(positions)
    ]

    old = built.store.get_submission(args.public_miner_submission)
    if (
        not old or old.get("status") not in ("accepted", "frozen")
        or old.get("is_king") is True
    ):
        raise VerificationError("miner source submission is not admitted")
    built.public_submission_code(args.public_miner_submission)
    payload = built._objects.get_bounded(
        old["source_ref"], source_bundle.MAX_SOURCE_ARCHIVE_BYTES
    )
    facts = source_bundle.validate_source_archive(payload, require_license=True)
    defaults = replace(
        built.config.defaults, benchmark_icp_count=2, rewards_enabled=False,
        daily_cutoff_hour_utc=None, runner_hotkeys=runners,
        baseline_source_url=DEFAULT_BASELINE_SOURCE_URL,
    )
    config = replace(
        built.config, mode="shadow", pinned_round_id=args.round_id,
        defaults=defaults, reward_signer_factory=None, baseline_promoter_factory=None,
        daily_icp_source=lambda *, set_id, active_at: {
            "status": "ready", "set_id": set_id, "icps": selected_icps,
        },
    )
    service = ArenaService(config)
    submission_id = "trajectory-" + hashlib.sha256(
        (args.round_id + ":" + args.public_miner_submission).encode()
    ).hexdigest()[:24]
    source_ref = "arena/%s/sources/%s.tar.gz" % (args.round_id, submission_id)
    checksum = base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")
    existing = service.store.get_round(args.round_id)
    if existing is not None:
        existing_config = existing.get("configuration_doc") or {}
        if (
            existing_config.get("mode") != "shadow"
            or existing_config.get("rewards_enabled") is not False
            or contracts.benchmark_icp_count(existing_config) != 2
            or set(existing_config.get("runner_hotkeys") or ()) != set(runners)
            or existing_config.get("baseline_source_url") != DEFAULT_BASELINE_SOURCE_URL
            or existing.get("reward_activated_at") is not None
        ):
            raise VerificationError("existing fixture round is incompatible")
        if existing.get("benchmark_ref"):
            benchmark = json.loads(service._objects.get(existing["benchmark_ref"]))
            if benchmark.get("icps") != selected_icps:
                raise VerificationError("existing fixture benchmark differs")
    if existing is None:
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=args.cutoff_minutes)
        service.create_round(cutoff, round_id=args.round_id)
        service._objects.put(source_ref, payload)
        owner = resolve_finalized_owner(service.config.chain, str(old["miner_hotkey"]))
        registered = service.store.register_submission(
            args.round_id, submission_id, str(old["miner_hotkey"]),
            {
                "source_ref": source_ref,
                "source_size_bytes": facts["source_size_bytes"],
                "source_content_md5": checksum,
                "consent": {"public_rerun": True},
                "fixture_origin": "public_source_operator_trajectory_probe",
            },
            owner_admission=owner,
        )
        if registered.get("status") != "registered":
            raise VerificationError("fixture miner registration failed")
        manager = service.config.credential_manager
        if manager is None:
            raise VerificationError("credential manager is unavailable")
        encrypted = {}
        for provider, name in (
            ("openrouter", "LAB_ARENA_OPENROUTER_API_KEY"),
            ("deepline", "LAB_ARENA_DEEPLINE_API_KEY"),
            ("scrapingdog", "LAB_ARENA_SCRAPINGDOG_API_KEY"),
        ):
            secret = os.environ.get(name, "")
            if not secret:
                raise VerificationError(provider + " organizer key is unavailable")
            encrypted[provider] = manager._encrypt(
                secret, submission_id=submission_id,
                miner_hotkey=str(old["miner_hotkey"]), provider=provider,
            )
            secret = ""
        accepted = service.store.accept_submission_with_credentials(
            args.round_id, submission_id, str(old["miner_hotkey"]), encrypted
        )
        if accepted.get("status") not in ("accepted", "existing"):
            raise VerificationError("fixture miner credential binding failed")
    fixture = service.store.get_submission(submission_id)
    if (
        not fixture or fixture.get("round_id") != args.round_id
        or fixture.get("miner_hotkey") != old.get("miner_hotkey")
        or fixture.get("source_ref") != source_ref
        or int(fixture.get("source_size_bytes") or 0) != len(payload)
        or (fixture.get("submission_doc") or {}).get("source_content_md5") != checksum
    ):
        raise VerificationError("fixture miner identity differs")
    if (
        fixture.get("code_review_status") != "passed"
        and (service.store.get_round(args.round_id) or {}).get("status") == "open"
    ):
        service.review_pending_submissions()
        fixture = service.store.get_submission(submission_id)
    if not fixture or fixture.get("code_review_status") != "passed":
        raise VerificationError("fixture miner code review did not pass")
    service._trajectory_fixture = {
        "fixture_kind": "operator_public_source_host_funded_miner",
        "submission_id": submission_id,
        "source_submission_id": args.public_miner_submission,
        "source_archive_hash": contracts.hash_bytes(payload),
        "published_icp_round": args.published_icp_round,
        "source_icp_positions": list(positions),
        "runner_hotkeys": list(runners),
    }
    return service, create_app(service)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn
    from scripts.run_lab_arena_service import load_scoped_environment

    load_scoped_environment(args.environment_file)
    if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
        raise VerificationError("fixture requires the live gateway environment")
    service, app = _fixture_service(args)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=args.port, log_level="info"
    ))
    stop = threading.Event()
    failure = []

    def status(last=None):
        row = service.store.get_round(args.round_id) or {}
        value = {
            "schema_version": SCHEMA, "round_id": args.round_id,
            "bind": "127.0.0.1", "port": args.port,
            "round_status": row.get("status"),
            "rewards_enabled": (row.get("configuration_doc") or {}).get("rewards_enabled"),
            "reward_activated": row.get("reward_activated_at") is not None,
            "fixture": service._trajectory_fixture,
        }
        if last is not None:
            value["last_transition"] = last
        _write(args.status_file, value)
        return value

    status()

    def driver():
        while not stop.is_set() and not server.started:
            stop.wait(0.05)
        while not stop.is_set():
            try:
                last = service.advance_round(args.round_id)
                service.reconcile_closed_provider_costs()
                current = status(last)
            except Exception as exc:
                failure.append(type(exc).__name__)
                server.should_exit = True
                return
            if current["round_status"] in ("published", "cancelled"):
                server.should_exit = True
                return
            stop.wait(args.tick_seconds)

    thread = threading.Thread(target=driver, name="arena-trajectory-driver", daemon=True)
    thread.start()
    print(json.dumps({
        "bind": "127.0.0.1", "port": args.port, "round_id": args.round_id,
        "status_file": str(args.status_file), "fixture": service._trajectory_fixture,
    }, sort_keys=True), flush=True)
    try:
        server.run()
    finally:
        stop.set()
        thread.join(timeout=args.tick_seconds + 2)
        final = status()
    return 0 if not failure and final["round_status"] == "published" else 1


def _verify(args: argparse.Namespace) -> int:
    from scripts.run_lab_arena_service import load_scoped_environment
    load_scoped_environment(args.environment_file)
    if os.environ.get("LAB_ARENA_MODE", "").strip().lower() != "live":
        raise VerificationError("central proof requires the live gateway environment")
    from lab_arena.wiring import build_service_from_environment
    service, _app = build_service_from_environment("shadow")
    result = _proof(service, args.round_id, args.runs, args.require_runtime_log)
    _write(args.status_file, result)
    print(json.dumps({"complete": True, "status_file": str(args.status_file)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "serve":
            return _serve(args)
        return _verify(args)
    except VerificationError as exc:
        print("Arena trajectory verification failed: %s" % exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            "Arena trajectory verification failed: %s" % type(exc).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
