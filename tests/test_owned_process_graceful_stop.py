"""Paid-work drain never escalates to a group kill."""

from __future__ import annotations

import signal
from argparse import Namespace

import pytest

from scripts import manage_owned_process_group as owned


def test_no_kill_stop_keeps_state_when_paid_work_exceeds_deadline(tmp_path, monkeypatch):
    path = tmp_path / "driver-state.json"
    path.write_text("owned")
    state = {"pid": 123, "pgid": 123}
    sent = []
    monkeypatch.setattr(owned, "_read_process", lambda _pid: {"state": "S"})
    monkeypatch.setattr(owned, "_same_process", lambda *_args: True)
    monkeypatch.setattr(owned, "_wait_for_group_exit", lambda *_args: False)
    monkeypatch.setattr(owned.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(owned.os, "killpg", lambda *_args: pytest.fail("group signal"))

    with pytest.raises(owned.OwnershipError, match="graceful deadline"):
        owned._stop_state(
            path, state, 1, 1, no_kill=True, signal_leader_only=True,
        )
    assert sent == [(123, signal.SIGTERM)]
    assert path.exists()


def test_recovery_owner_read_uses_exact_process_receipt_and_bounded_keys(
    tmp_path, monkeypatch, capsys,
):
    root = str(tmp_path / "release")
    command = ["python3", "-B", "-u", "scripts/run_arena_proxy_edge.py"]
    state = {"pid": 123, "pgid": 123, "start_time_ticks": 5, "uid": 1,
             "cwd": root, "argv": command}
    process = {**state, "state": "S", "session": 123}
    monkeypatch.setattr(owned, "_read_verify_state", lambda *_args: (b"", state))
    monkeypatch.setattr(owned, "_read_process", lambda _pid: process)
    monkeypatch.setattr(owned, "_matching_processes", lambda **_kwargs: [process])
    monkeypatch.setattr(owned.os, "open", lambda *_args: 42)
    monkeypatch.setattr(owned.os, "close", lambda _fd: None)
    raw = b"\0".join((
        b"PROVIDER_SECRET=never-log",
        b"GATEWAY_RESTART_INVOCATION_ID=gateway-1792000000-123",
        b"LEADPOET_RESTART_INVOCATION_ID=gateway-1792000000-123",
        b"GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=gateway-1791999999-100",
        f"LAB_ARENA_RELEASE_ROOT={root}".encode(),
        f"LAB_ARENA_RELEASE_COMMIT={'a' * 40}".encode(),
    )) + b"\0"
    monkeypatch.setattr(owned.os, "read", lambda _fd, _limit: raw)
    args = Namespace(
        release_sha="a" * 40, process_argv=command, cwd=root, uid=1,
        state_file=tmp_path / "edge.json",
    )
    owned._verify_invocation(args)
    output = capsys.readouterr().out
    assert output == "gateway-1791999999-100\n"
    assert "never-log" not in output

    monkeypatch.setattr(owned.os, "read", lambda _fd, _limit: raw.replace(
        b"LEADPOET_RESTART_INVOCATION_ID=gateway-1792000000-123",
        b"LEADPOET_RESTART_INVOCATION_ID=gateway-1792000001-123",
    ))
    with pytest.raises(owned.OwnershipError, match="does not match"):
        owned._verify_invocation(args)
    assert capsys.readouterr().out == ""
