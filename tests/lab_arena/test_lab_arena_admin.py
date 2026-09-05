"""Operator commands: expected state, dry runs, and public status output."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADMIN = runpy.run_path(str(ROOT / "scripts/lab_arena_admin.py"), run_name="lab_arena_admin_module")


class FakeStore:
    def __init__(self, row):
        self.row = row

    def get_round(self, round_id):
        return dict(self.row) if round_id == self.row["round_id"] else None


class FakeService:
    def __init__(self, status="stage1"):
        self.row = {"round_id": "arena-2026-09-02", "status": status, "status_generation": 3, "stage_generation": 1, "configuration_hash": "sha256:" + "a" * 64, "commitment_hash": "sha256:" + "b" * 64, "configuration_doc": {"secret_looking": "x"}, "journal": [{"private": True}]}
        self.store = FakeStore(self.row)
        self.advanced = 0
        self.cancelled = []

    def current_round(self):
        return dict(self.row)

    def open_round(self):
        return dict(self.row) if self.row["status"] == "open" else None

    def latest_published_round(self):
        return None

    def startup_checks(self):
        return {"database_identity": {"current_user": "lab_arena_service"}, "signing_public_key_hash": "sha256:" + "c" * 64}

    def advance_round(self, round_id):
        self.advanced += 1
        return {"status": "ok", "round_status": "stage1_closed"}

    def cancel(self, round_id, reason):
        self.cancelled.append((round_id, reason))
        return {"status": "cancelled"}


def run(args, service, capsys):
    parser = ADMIN["build_parser"]()
    code = ADMIN["run"](parser.parse_args(args), service)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_status_prints_plain_state_not_private_columns_or_old_hashes(capsys):
    service = FakeService()
    code, out, _ = run(["status"], service, capsys)
    assert code == 0
    document = json.loads(out)
    assert document["current"]["status"] == "stage1"
    assert "configuration_hash" not in out and "commitment_hash" not in out
    assert "configuration_doc" not in out and "journal" not in out and "secret_looking" not in out


def test_advance_requires_expected_state_and_supports_dry_run(capsys):
    service = FakeService()
    code, out, err = run(["advance", "--round", "arena-2026-09-02", "--expect-status", "stage2"], service, capsys)
    assert code == 3 and service.advanced == 0 and "expected stage2" in err
    code, out, _ = run(["advance", "--round", "arena-2026-09-02", "--expect-status", "stage1", "--dry-run"], service, capsys)
    assert code == 0 and service.advanced == 0 and json.loads(out)["dry_run"] is True
    code, out, _ = run(["advance", "--round", "arena-2026-09-02", "--expect-status", "stage1"], service, capsys)
    assert code == 0 and service.advanced == 1 and json.loads(out)["result"]["status"] == "ok"
    code, _, err = run(["advance", "--round", "arena-2026-01-01"], service, capsys)
    assert code == 2 and "unknown round" in err


def test_cancel_accepts_only_published_rules(capsys):
    service = FakeService()
    code, _, err = run(["cancel", "--round", "arena-2026-09-02", "--reason", "because"], service, capsys)
    assert code == 2 and service.cancelled == [] and "reason must be one of" in err
    code, out, _ = run(["cancel", "--round", "arena-2026-09-02", "--reason", "runner_capacity", "--dry-run"], service, capsys)
    assert code == 0 and service.cancelled == [] and json.loads(out)["would_cancel"]["status"] == "stage1"
    code, out, _ = run(["cancel", "--round", "arena-2026-09-02", "--reason", "runner_capacity"], service, capsys)
    assert code == 0 and service.cancelled == [("arena-2026-09-02", "runner_capacity")]


class IdleService(FakeService):
    """No round is open or running; the operator may create the next one."""

    def __init__(self):
        super().__init__()
        self.store = FakeStore({"round_id": "arena-2026-09-02", "status": "published"})
        self.created = []

    def current_round(self):
        return None

    def open_round(self):
        return None

    def create_round(self, cutoff, *, round_id=None):
        self.created.append((cutoff, round_id))
        return {"round_id": round_id or "arena-%s" % cutoff.strftime("%Y-%m-%d"), "schedule": {"submission_cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}}


def test_create_makes_the_round_for_a_cutoff_and_refuses_open_rounds_and_existing_dates(capsys):
    service = IdleService()
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-05T00:00:00Z", "--dry-run"]), service) == 0
    assert json.loads(capsys.readouterr().out)["would_create"] == "arena-2026-09-05" and not service.created
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-05T00:00:00Z"]), service) == 0
    assert json.loads(capsys.readouterr().out)["created"] == "arena-2026-09-05" and len(service.created) == 1
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-02T00:00:00Z"]), service) == 3  # that date's round exists
    assert "already exists" in capsys.readouterr().err
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-05"]), service) == 2  # not a UTC instant
    capsys.readouterr()
    busy = FakeService(status="open")
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-06T00:00:00Z"]), busy) == 3
    assert "already open for submissions" in capsys.readouterr().err
    # Rounds overlap: a running round never blocks the next one.
    running = FakeService(status="stage1")
    running.created = []
    running.create_round = lambda cutoff, *, round_id=None: IdleService.create_round(running, cutoff, round_id=round_id)
    assert ADMIN["run"](ADMIN["build_parser"]().parse_args(["create", "--cutoff", "2026-09-06T00:00:00Z"]), running) == 0
    assert json.loads(capsys.readouterr().out)["created"] == "arena-2026-09-06" and len(running.created) == 1


def test_create_accepts_only_a_date_matched_suffixed_explicit_round_id(capsys):
    service = IdleService()
    args = [
        "create",
        "--cutoff",
        "2026-09-05T08:00:00Z",
        "--round-id",
        "arena-2026-09-05-hosted1",
    ]
    code, out, err = run(args + ["--dry-run"], service, capsys)
    assert code == 0 and not err
    assert json.loads(out)["would_create"] == "arena-2026-09-05-hosted1"
    code, out, err = run(args, service, capsys)
    assert code == 0 and not err
    assert json.loads(out)["created"] == "arena-2026-09-05-hosted1"
    assert service.created[-1][1] == "arena-2026-09-05-hosted1"

    for invalid in (
        "arena-2026-09-05",
        "arena-2026-09-06-hosted1",
        "arena-2026-09-05-HOSTED",
    ):
        code, _, err = run(
            [
                "create",
                "--cutoff",
                "2026-09-05T08:00:00Z",
                "--round-id",
                invalid,
            ],
            IdleService(),
            capsys,
        )
        assert code == 2 and "match the cutoff date" in err
