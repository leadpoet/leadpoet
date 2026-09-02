"""Operator commands: exact expected state, dry runs, closed cancel reasons, hashes only."""

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


def test_status_prints_hashes_not_private_columns(capsys):
    service = FakeService()
    code, out, _ = run(["status"], service, capsys)
    assert code == 0
    document = json.loads(out)
    assert document["current"]["configuration_hash"].startswith("sha256:")
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
