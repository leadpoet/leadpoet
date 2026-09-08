"""Focused tests for the restart claim guard transport and fail-closed drain."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "lab_arena_restart_claim_guard.py"
SPEC = importlib.util.spec_from_file_location("lab_arena_restart_claim_guard", SCRIPT)
guard_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guard_cli)

CANDIDATE = "a" * 40
INVOCATION = "restart-unit"


def _args(**updates):
    values = {
        "candidate": CANDIDATE,
        "invocation": INVOCATION,
        "generation": 0,
        "scope": "all",
        "lease_seconds": 600,
        "timeout_seconds": 0.0,
        "poll_seconds": 0.001,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _drain_state(*, phase="draining", preserved=False):
    return {
        "schema_version": "leadpoet.lab_arena.restart_drain_state.v1",
        "captured_count": 1,
        "accepted_receipt_count": int(preserved),
        "reported_terminal_receipt_count": 0,
        "still_leased_count": 0 if preserved else 1,
        "lost_or_mutated_count": 0,
        "current_leased_count": 0 if preserved else 1,
        "pending_retry_count": 0,
        "snapshot_commitment": "sha256:" + "1" * 64,
        "outcome_commitment": "sha256:" + "2" * 64,
        "preserved": preserved,
        "guard_active": True,
        "guard_generation": 1,
        "restart_scope": "all",
        "restart_phase": phase,
    }


def _guard_state(*, phase="draining", present=True):
    guard, owner = guard_cli._identity(CANDIDATE, INVOCATION)
    guard_commitment, owner_commitment = guard_cli._commitments(guard, owner)
    return {
        "schema_version": "leadpoet.lab_arena.restart_guard_state.v1",
        "paused": present,
        "operator_paused": False,
        "guard_present": present,
        "guard_active": present,
        "guard_commitment": guard_commitment if present else "",
        "owner_commitment": owner_commitment if present else "",
        "guard_generation": 1 if present else 0,
        "guard_expires_at": "2099-01-01T00:00:00Z" if present else None,
        "candidate_commit": CANDIDATE if present else "",
        "restart_scope": "all" if present else "",
        "restart_phase": phase if present else "",
        "drain": _drain_state(phase=phase),
    }


def _state_for(candidate, *, phase, generation):
    guard, owner = guard_cli._identity(candidate, INVOCATION)
    guard_commitment, owner_commitment = guard_cli._commitments(guard, owner)
    value = _guard_state(phase=phase)
    value.update({
        "guard_commitment": guard_commitment,
        "owner_commitment": owner_commitment,
        "guard_generation": generation,
        "candidate_commit": candidate,
    })
    return value


class _Response:
    status = 200

    def read(self, _limit):
        return json.dumps({"ok": True}).encode()


class _Connection:
    created = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.request_args = None
        self.__class__.created.append(self)

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)

    def getresponse(self):
        return _Response()

    def close(self):
        pass


def test_transport_uses_exact_arena_database_and_scoped_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://wrong-primary.example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "primary-secret")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://arena.example")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_ANON_KEY", "arena-anon")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "sb_secret_arena")
    monkeypatch.setenv("LAB_ARENA_SERVICE_JWT", "cached.legacy.jwt")
    monkeypatch.setattr(guard_cli.http.client, "HTTPSConnection", _Connection)
    _Connection.created.clear()

    assert guard_cli._request("safe_rpc", {}) == {"ok": True}
    connection = _Connection.created[-1]
    assert connection.host == "arena.example"
    headers = connection.request_args[1]["headers"]
    assert headers["apikey"] == "sb_secret_arena"
    assert "Authorization" not in headers
    assert "primary-secret" not in repr(connection.request_args)


@pytest.mark.parametrize(
    "document",
    [
        (
            'LAB_ARENA_MODE="live"\n'
            'LAB_ARENA_SUPABASE_URL="https://arena.example/path with space"\n'
            'UNRELATED=$(touch should-never-run)\n'
        ),
        json.dumps({
            "LAB_ARENA_MODE": "shadow",
            "LAB_ARENA_SUPABASE_URL": "https://arena.example",
            "UNRELATED": "ignored",
        }),
    ],
)
def test_environment_parser_matches_service_scope_without_execution(
    monkeypatch, tmp_path, document,
):
    from scripts.run_lab_arena_service import load_scoped_environment

    environment_file = tmp_path / "arena.env"
    environment_file.write_text(document)
    marker = tmp_path / "should-never-run"
    monkeypatch.chdir(tmp_path)
    names = ("LAB_ARENA_MODE", "LAB_ARENA_SUPABASE_URL")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    load_scoped_environment(environment_file)
    expected = {name: os.environ[name] for name in names}
    for name in names:
        monkeypatch.delenv(name, raising=False)

    observed = guard_cli._read_scoped_environment(environment_file)
    assert {name: observed[name] for name in names} == expected
    assert "UNRELATED" not in observed
    assert not marker.exists()


def test_environment_parser_rejects_symlink(tmp_path):
    target = tmp_path / "target.env"
    target.write_text("LAB_ARENA_MODE=live\n")
    link = tmp_path / "link.env"
    link.symlink_to(target)
    with pytest.raises(guard_cli.GuardError, match="unavailable"):
        guard_cli._read_scoped_environment(link)


def test_environment_parser_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "arena.fifo"
    os.mkfifo(fifo)
    started = __import__("time").monotonic()
    with pytest.raises(guard_cli.GuardError, match="unavailable"):
        guard_cli._read_scoped_environment(fifo)
    assert __import__("time").monotonic() - started < 1


def test_transport_uses_legacy_arena_jwt_without_secret_diagnostics(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://arena.example")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_ANON_KEY", "arena-anon")
    monkeypatch.delenv("LAB_ARENA_SERVICE_KEY", raising=False)
    monkeypatch.setenv("LAB_ARENA_SERVICE_JWT", "header.payload.signature")
    monkeypatch.setattr(guard_cli.http.client, "HTTPSConnection", _Connection)
    _Connection.created.clear()

    guard_cli._request("safe_rpc", {})
    headers = _Connection.created[-1].request_args[1]["headers"]
    assert headers["apikey"] == "arena-anon"
    assert headers["Authorization"] == "Bearer header.payload.signature"

    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "invalid-key")
    with pytest.raises(guard_cli.GuardError) as error:
        guard_cli._request("safe_rpc", {})
    assert "header.payload.signature" not in str(error.value)
    assert "invalid-key" not in str(error.value)


def test_malformed_header_value_never_appears_in_cli_diagnostics(monkeypatch, capsys):
    marker = "private-marker"
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://arena.example")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_ANON_KEY", "arena-anon")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", f"sb_secret_{marker}\r\nInjected: yes")
    monkeypatch.delenv("LAB_ARENA_SERVICE_JWT", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        [str(SCRIPT), "state", "--candidate", CANDIDATE, "--invocation", INVOCATION],
    )
    assert guard_cli.main() == 1
    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err
    assert "database authority is unavailable" in captured.err


@pytest.mark.parametrize(
    "phase",
    [
        "draining", "gateway_destructive", "gateway_ready",
        "validator_destructive", "validator_ready",
    ],
)
def test_paired_drain_renews_every_legal_recovery_phase(monkeypatch, phase):
    calls = []

    def request(function, _payload):
        calls.append(function)
        if function == "lab_arena_restart_guard_state_v1":
            return _guard_state(phase=phase)
        if function == "lab_arena_acquire_restart_guard_v1":
            return _guard_state(phase=phase)
        if function == "lab_arena_restart_quiescence_v1":
            return {
                **_drain_state(phase=phase, preserved=True),
                "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
            }
        raise AssertionError(function)

    monkeypatch.setattr(guard_cli, "_request", request)
    result = guard_cli._drain(_args())
    assert result["preserved"] is True
    assert "lab_arena_abort_restart_guard_v1" not in calls


def test_timeout_aborts_only_owned_pre_destructive_guard(monkeypatch):
    calls = []

    def request(function, _payload):
        calls.append(function)
        if function == "lab_arena_restart_guard_state_v1":
            return _guard_state()
        if function == "lab_arena_acquire_restart_guard_v1":
            return _guard_state()
        if function == "lab_arena_restart_quiescence_v1":
            return {
                **_drain_state(),
                "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
            }
        if function == "lab_arena_abort_restart_guard_v1":
            return _guard_state(present=False)
        raise AssertionError(function)

    monkeypatch.setattr(guard_cli, "_request", request)
    with pytest.raises(guard_cli.GuardError, match="did not drain"):
        guard_cli._drain(_args())
    assert calls[-1] == "lab_arena_abort_restart_guard_v1"


def test_post_destructive_drain_failure_keeps_guard_for_canonical_retry(monkeypatch):
    calls = []

    def request(function, _payload):
        calls.append(function)
        if function == "lab_arena_restart_guard_state_v1":
            return _guard_state(phase="gateway_destructive")
        if function == "lab_arena_acquire_restart_guard_v1":
            return _guard_state(phase="gateway_destructive")
        if function == "lab_arena_restart_quiescence_v1":
            return {
                **_drain_state(phase="gateway_destructive"),
                "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
                "guard_active": False,
            }
        raise AssertionError(function)

    monkeypatch.setattr(guard_cli, "_request", request)
    with pytest.raises(guard_cli.GuardError, match="changed during drain"):
        guard_cli._drain(_args())
    assert "lab_arena_abort_restart_guard_v1" not in calls


def test_drain_retargets_exact_owner_after_candidate_advance(monkeypatch):
    old_candidate = "f" * 40
    phase = "gateway_ready"
    candidate = old_candidate
    generation = 1
    calls = []

    def request(function, payload):
        nonlocal candidate, generation
        calls.append(function)
        if function == "lab_arena_restart_guard_state_v1":
            return _state_for(candidate, phase=phase, generation=generation)
        if function == "lab_arena_retarget_restart_guard_v1":
            assert payload["p_new_candidate_commit"] == CANDIDATE
            assert payload["p_expected_generation"] == 1
            candidate = CANDIDATE
            generation = 2
            return _state_for(candidate, phase=phase, generation=generation)
        if function == "lab_arena_acquire_restart_guard_v1":
            return _state_for(candidate, phase=phase, generation=generation)
        if function == "lab_arena_restart_quiescence_v1":
            return {
                **_drain_state(phase=phase, preserved=True),
                "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
                "guard_generation": generation,
            }
        raise AssertionError(function)

    monkeypatch.setattr(guard_cli, "_request", request)
    result = guard_cli._drain(_args())
    assert result["preserved"] is True
    assert candidate == CANDIDATE and generation == 2
    assert calls.count("lab_arena_retarget_restart_guard_v1") == 1


def test_cli_paired_path_drains_both_components_then_releases(monkeypatch, capsys):
    phase = "draining"
    present = True

    def state():
        value = _guard_state(phase=phase, present=present)
        value["drain"] = _drain_state(phase=phase, preserved=True)
        return value

    def request(function, payload):
        nonlocal phase, present
        if function == "lab_arena_restart_guard_state_v1":
            return state()
        if function == "lab_arena_acquire_restart_guard_v1":
            return state()
        if function == "lab_arena_restart_quiescence_v1":
            return {
                **_drain_state(phase=phase, preserved=True),
                "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
            }
        if function == "lab_arena_authorize_restart_phase_v1":
            requested = payload["p_phase"]
            assert (phase, requested) in {
                ("draining", "gateway_destructive"),
                ("gateway_ready", "validator_destructive"),
            }
            phase = requested
            return state()
        if function == "lab_arena_mark_restart_ready_v1":
            requested = payload["p_phase"]
            assert (phase, requested) in {
                ("gateway_destructive", "gateway_ready"),
                ("validator_destructive", "validator_ready"),
            }
            phase = requested
            return state()
        if function == "lab_arena_release_restart_guard_v1":
            assert phase == "validator_ready"
            present = False
            phase = ""
            return state()
        raise AssertionError(function)

    monkeypatch.setattr(guard_cli, "_request", request)
    common = ["--candidate", CANDIDATE, "--invocation", INVOCATION]
    actions = [
        ["drain", *common, "--scope", "all"],
        ["authorize", *common, "--phase", "gateway_destructive"],
        ["ready", *common, "--phase", "gateway_ready"],
        ["drain", *common, "--scope", "all"],
        ["authorize", *common, "--phase", "validator_destructive"],
        ["ready", *common, "--phase", "validator_ready"],
        ["release", *common],
    ]
    for argv in actions:
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *argv])
        assert guard_cli.main() == 0
    assert present is False
    assert len(capsys.readouterr().out.splitlines()) == len(actions)
