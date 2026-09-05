#!/usr/bin/env python3
"""Small operator commands for Arena rounds.

    python3 scripts/lab_arena_admin.py status
    python3 scripts/lab_arena_admin.py advance --round <round-id> [--dry-run]
    python3 scripts/lab_arena_admin.py cancel --round <round-id> --reason <rule> [--dry-run]

Commands print only public operating state. ``LAB_ARENA_MODE=off`` refuses
every command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_ROUND_FIELDS = ("round_id", "status", "status_generation", "stage_generation", "reward_basis_hash", "king_outcome", "king_hotkey", "effective_reward_epoch", "cancel_reason")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Lab Arena operator commands")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="print the current round's public state")
    advance = commands.add_parser("advance", help="apply one idempotent round transition")
    advance.add_argument("--round", required=True)
    advance.add_argument("--expect-status", default=None, help="refuse unless the round is in this status")
    advance.add_argument("--dry-run", action="store_true")
    create = commands.add_parser("create", help="create the round whose submission cutoff is the given UTC instant")
    create.add_argument("--cutoff", required=True, help="ISO 8601 UTC instant, e.g. 2026-09-05T00:00:00Z")
    create.add_argument(
        "--round-id",
        help="explicit date-matched suffixed round id for an isolated manual round",
    )
    create.add_argument("--dry-run", action="store_true")
    cancel = commands.add_parser("cancel", help="cancel a round with a supported reason")
    cancel.add_argument("--round", required=True)
    cancel.add_argument("--reason", required=True)
    cancel.add_argument("--expect-status", default=None)
    cancel.add_argument("--dry-run", action="store_true")
    return parser


def public_view(row):
    return {key: row.get(key) for key in PUBLIC_ROUND_FIELDS}


def run(args, service) -> int:
    from lab_arena.service import CANCEL_REASONS, ServiceError

    if args.command == "create":
        from datetime import datetime, timezone

        from lab_arena import contracts
        from lab_arena.service import round_id_for_cutoff

        try:
            cutoff = datetime.strptime(args.cutoff, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            print("cutoff must look like 2026-09-05T00:00:00Z", file=sys.stderr)
            return 2
        default_round_id = round_id_for_cutoff(cutoff)
        round_id = args.round_id or default_round_id
        if args.round_id is not None and (
            contracts.ROUND_ID_RE.fullmatch(round_id) is None
            or not round_id.startswith(default_round_id + "-")
        ):
            print(
                "explicit round id must match the cutoff date and include a suffix",
                file=sys.stderr,
            )
            return 2
        open_round = service.open_round()
        if open_round is not None:
            # Rounds overlap: a running round never blocks the next one; only an open submission window does.
            print("a round is already open for submissions: %s (%s)" % (open_round["round_id"], open_round["status"]), file=sys.stderr)
            return 3
        if service.store.get_round(round_id) is not None:
            print("round %s already exists" % round_id, file=sys.stderr)
            return 3
        if args.dry_run:
            print(json.dumps({"dry_run": True, "would_create": round_id, "cutoff": args.cutoff}, indent=2, sort_keys=True))
            return 0
        created = service.create_round(cutoff, round_id=round_id)
        print(json.dumps({"created": created["round_id"], "schedule": created.get("schedule")}, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "status":
        current = service.current_round()
        latest = service.latest_published_round()
        print(json.dumps({"current": public_view(current) if current else None, "latest_published": {k: latest.get(k) for k in ("round_id", "king_outcome", "king_hotkey", "effective_reward_epoch", "reward_basis_hash")} if latest else None, "identity": {k: v for k, v in service.startup_checks().items() if k != "database_identity"}}, indent=2, sort_keys=True))
        return 0
    row = service.store.get_round(args.round)
    if row is None:
        print("unknown round", file=sys.stderr)
        return 2
    if args.expect_status and row["status"] != args.expect_status:
        print("round is %s, expected %s" % (row["status"], args.expect_status), file=sys.stderr)
        return 3
    if args.command == "advance":
        if args.dry_run:
            print(json.dumps({"dry_run": True, "round": public_view(row)}, indent=2, sort_keys=True))
            return 0
        result = service.advance_round(args.round)
        print(json.dumps({"result": result, "round": public_view(service.store.get_round(args.round))}, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "cancel":
        if args.reason not in CANCEL_REASONS.values():
            print("reason must be one of: %s" % ", ".join(sorted(CANCEL_REASONS.values())), file=sys.stderr)
            return 2
        if args.dry_run:
            print(json.dumps({"dry_run": True, "would_cancel": public_view(row), "reason": args.reason}, indent=2, sort_keys=True))
            return 0
        try:
            result = service.cancel(args.round, args.reason)
        except ServiceError as exc:
            print("cancel refused: %s" % exc.code, file=sys.stderr)
            return 4
        print(json.dumps({"result": result, "round": public_view(service.store.get_round(args.round))}, indent=2, sort_keys=True, default=str))
        return 0
    return 2


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    mode = os.environ.get("LAB_ARENA_MODE", "off").strip().lower()
    if mode == "off":
        print("LAB_ARENA_MODE=off: operator commands are disabled", file=sys.stderr)
        return 1
    from lab_arena.wiring import build_service_from_environment

    service, _app = build_service_from_environment(mode)
    return run(args, service)


if __name__ == "__main__":
    raise SystemExit(main())
