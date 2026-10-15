"""The public and direct listeners use the same Arena security edge."""

from __future__ import annotations

import pytest

import subprocess

from scripts import arena_ingress_config as ingress
from scripts.arena_ingress_config import activate, commit_state, inspect, render


BASELINE = b"gateway.subnet71.com {\n\treverse_proxy localhost:8000\n}\n"


def test_domain_stage_keeps_legacy_gateway_but_moves_arena_to_edge():
    config = render(arena_edge_port=8801, backend="legacy", direct=False)
    assert config.count("reverse_proxy 127.0.0.1:8801") == 1
    assert config.count("reverse_proxy 127.0.0.1:8000") == 1
    assert ":8000 {" not in config
    assert "@arena path /arena /arena/* /testnet/arena /testnet/arena/*" in config


def test_dual_listener_holds_gateway_closed_until_verified():
    closed = render(arena_edge_port=8801, backend="closed", direct=True)
    assert closed.count("reverse_proxy 127.0.0.1:8801") == 2
    assert closed.count('respond "gateway temporarily unavailable" 503') == 2
    assert "reverse_proxy 127.0.0.1:8001" not in closed
    ready = render(arena_edge_port=8800, backend="ready", direct=True)
    assert ready.count("reverse_proxy 127.0.0.1:8800") == 2
    assert ready.count("reverse_proxy 127.0.0.1:8001") == 2


@pytest.mark.parametrize(
    ("edge_port", "backend", "direct"),
    [(8792, "ready", True), (8800, "legacy", True), (8800, "unknown", False)],
)
def test_invalid_ingress_target_fails_closed(edge_port, backend, direct):
    with pytest.raises(ValueError):
        render(arena_edge_port=edge_port, backend=backend, direct=direct)


def test_exact_legacy_config_is_the_only_unmanaged_start(tmp_path):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)
    assert inspect(state, caddy)["mode"] == "legacy"
    caddy.write_bytes(BASELINE + b"\n")
    with pytest.raises(ValueError, match="differs"):
        inspect(state, caddy)


def test_committed_ingress_must_match_exact_config(tmp_path):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    sha = "a" * 40
    caddy.write_text(render(arena_edge_port=8801, backend="ready", direct=True))
    commit_state(state, caddy, mode="dual", slot=1, sha=sha, backend="ready")
    assert inspect(state, caddy) == {
        "version": 1, "mode": "dual", "slot": 1, "sha": sha, "backend": "ready",
    }
    caddy.write_text(render(arena_edge_port=8801, backend="closed", direct=True))
    with pytest.raises(ValueError, match="differs"):
        inspect(state, caddy)
    with pytest.raises(ValueError, match="does not match"):
        commit_state(state, caddy, mode="dual", slot=1, sha=sha, backend="ready")


def test_symlink_state_and_wrong_slot_are_rejected(tmp_path):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)
    target = tmp_path / "unrelated"
    target.write_text("{}")
    state.symlink_to(target)
    with pytest.raises(ValueError, match="owned private"):
        inspect(state, caddy)
    with pytest.raises(ValueError, match="approved loopback"):
        render(arena_edge_port=8000, backend="ready", direct=True)


def test_exact_caddy_transition_is_validated_reloaded_and_recorded(tmp_path, monkeypatch):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_kw: calls.append(command[1]))
    activate(
        state, caddy, expected_mode="legacy", expected_slot=0,
        expected_sha="-", expected_backend="legacy",
        mode="domain", slot=1, sha="a" * 40, backend="legacy",
    )
    assert calls == ["validate", "reload"]
    assert inspect(state, caddy)["sha"] == "a" * 40
    activate(
        state, caddy, expected_mode="domain", expected_slot=1,
        expected_sha="a" * 40, expected_backend="legacy",
        mode="legacy", slot=0, sha="-", backend="legacy",
    )
    assert caddy.read_bytes() == BASELINE
    assert not state.exists()
    assert inspect(state, caddy)["mode"] == "legacy"
    with pytest.raises(ValueError, match="changed"):
        activate(
            state, caddy, expected_mode="domain", expected_slot=1,
            expected_sha="a" * 40, expected_backend="legacy",
            mode="domain", slot=1, sha="a" * 40, backend="legacy",
        )


def test_failed_caddy_reload_restores_exact_legacy_config(tmp_path, monkeypatch):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)
    calls = []

    def run(command, **_kw):
        calls.append(command[1])
        if calls == ["validate", "reload"]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        activate(
            state, caddy, expected_mode="legacy", expected_slot=0,
            expected_sha="-", expected_backend="legacy",
            mode="domain", slot=1, sha="a" * 40, backend="legacy",
        )
    assert calls == ["validate", "reload", "reload"]
    assert caddy.read_bytes() == BASELINE
    assert inspect(state, caddy)["mode"] == "legacy"


def test_failed_state_commit_restores_config_and_previous_metadata(tmp_path, monkeypatch):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kw: None)
    original = ingress.commit_state

    def committed_then_failed(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("state sync failed")

    monkeypatch.setattr(ingress, "commit_state", committed_then_failed)
    with pytest.raises(RuntimeError, match="state sync failed"):
        activate(
            state, caddy, expected_mode="legacy", expected_slot=0,
            expected_sha="-", expected_backend="legacy",
            mode="domain", slot=1, sha="a" * 40, backend="legacy",
        )
    assert caddy.read_bytes() == BASELINE
    assert not state.exists()


def test_reload_failure_never_overwrites_concurrent_caddy_edit(tmp_path, monkeypatch):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    caddy.write_bytes(BASELINE)

    def run(command, **_kw):
        if command[1] == "reload":
            caddy.write_text("unrelated site change\n")
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="changed during rollback"):
        activate(
            state, caddy, expected_mode="legacy", expected_slot=0,
            expected_sha="-", expected_backend="legacy",
            mode="domain", slot=1, sha="a" * 40, backend="legacy",
        )
    assert caddy.read_text() == "unrelated site change\n"
