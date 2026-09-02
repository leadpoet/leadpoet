#!/usr/bin/env python3
"""Lab Arena operator commands (labarena.md section 14.4).

    python3 scripts/lab_arena_admin.py status
    python3 scripts/lab_arena_admin.py advance --round <round-id> [--dry-run]
    python3 scripts/lab_arena_admin.py cancel --round <round-id> --reason <rule> [--dry-run]

Every command requires the exact expected state and prints hashes rather than
private data. ``LAB_ARENA_MODE=off`` refuses every command.
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

PUBLIC_ROUND_FIELDS = ("round_id", "status", "status_generation", "stage_generation", "configuration_hash", "commitment_hash", "stage1_scoring_plan_hash", "stage2_scoring_plan_hash", "stage1_score_bundle_hash", "final_score_bundle_hash", "result_bundle_hash", "reward_basis_hash", "king_outcome", "king_hotkey", "effective_reward_epoch", "cancel_reason")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Lab Arena operator commands")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="print the current round's state hashes")
    advance = commands.add_parser("advance", help="apply one idempotent round transition")
    advance.add_argument("--round", required=True)
    advance.add_argument("--expect-status", default=None, help="refuse unless the round is in this status")
    advance.add_argument("--dry-run", action="store_true")
    cancel = commands.add_parser("cancel", help="cancel a round for an objective section 17 rule")
    cancel.add_argument("--round", required=True)
    cancel.add_argument("--reason", required=True)
    cancel.add_argument("--expect-status", default=None)
    cancel.add_argument("--dry-run", action="store_true")
    return parser


def public_view(row):
    return {key: row.get(key) for key in PUBLIC_ROUND_FIELDS}


def run(args, service) -> int:
    from lab_arena.service import CANCEL_REASONS, ServiceError

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
