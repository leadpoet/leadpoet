"""The public and direct listeners use the same Arena security edge."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import subprocess

from scripts import arena_ingress_config as ingress
from scripts.arena_ingress_config import activate, commit_state, inspect, render


BASELINE = b"gateway.subnet71.com {\n\treverse_proxy localhost:8000\n}\n"
DATABASE_HANDLER = (
    "    @database path /rest/v1/*\n"
    "    handle @database {\n"
    "        reverse_proxy 127.0.0.1:8081\n"
    "    }\n"
)


def _write_v1(state, caddy, *, slot=0, backend="ready", mode="dual"):
    """Freeze the original renderer independently of the new route/version."""
    def site(name):
        handler = ('respond "gateway temporarily unavailable" 503' if backend == "closed"
                   else f"reverse_proxy 127.0.0.1:{8000 if backend == 'legacy' else 8001}")
        return (
            f"{name} {{\n"
            "    @arena path /arena /arena/* /testnet/arena /testnet/arena/*\n"
            "    handle @arena {\n"
            f"        reverse_proxy 127.0.0.1:{ingress.slot_ports(slot)[1]}\n"
            "    }\n"
            "    handle {\n"
            f"        {handler}\n"
            "    }\n"
            "}\n"
        )

    caddy.write_text(site("gateway.subnet71.com") + ("\n" + site(":8000") if mode == "dual" else ""))
    state.write_text(json.dumps({"version": 1, "mode": mode, "slot": slot,
                                "sha": "a" * 40, "backend": backend}, indent=2) + "\n")
    state.chmod(0o600)


@pytest.mark.parametrize("edge", ingress.ARENA_EDGE_PORTS)
@pytest.mark.parametrize("backend,direct", [
    ("legacy", False), ("ready", False), ("closed", False),
    ("ready", True), ("closed", True),
])
def test_database_route_stays_on_https_for_every_arena_slot_and_backend(edge, backend, direct):
    config = render(arena_edge_port=edge, backend=backend, direct=direct)
    secure, _, plaintext = config.partition("\n:8000 {\n")
    assert secure.startswith("gateway.subnet71.com {\n")
    assert secure.count(DATABASE_HANDLER) == 1
    assert config.count("reverse_proxy 127.0.0.1:8081") == 1
    assert "handle_path" not in config and "strip_prefix" not in config
    assert "/rest/v1" not in plaintext and "8081" not in plaintext
    assert bool(plaintext) is direct


@pytest.mark.parametrize("slot,backend,mode", [
    (0, "legacy", "domain"), (1, "legacy", "domain"),
    (0, "ready", "dual"), (1, "ready", "dual"),
    (0, "closed", "dual"), (1, "closed", "dual"),
])
def test_first_exact_v1_transition_upgrades_to_v2_and_keeps_database_route(
    tmp_path, monkeypatch, slot, backend, mode,
):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy, slot=slot, backend=backend, mode=mode)
    assert inspect(state, caddy)["version"] == 1
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kw: None)
    activate(state, caddy, expected_mode=mode, expected_slot=slot,
             expected_sha="a" * 40, expected_backend=backend,
             mode="dual", slot=1-slot, sha="b" * 40, backend="closed")
    assert inspect(state, caddy) == {
        "version": 2, "mode": "dual", "slot": 1-slot,
        "sha": "b" * 40, "backend": "closed",
    }
    activate(state, caddy, expected_mode="dual", expected_slot=1-slot,
             expected_sha="b" * 40, expected_backend="closed",
             mode="dual", slot=slot, sha="c" * 40, backend="ready")
    assert inspect(state, caddy)["version"] == 2
    assert caddy.read_text().count(DATABASE_HANDLER) == 1


@pytest.mark.parametrize("failure", ["reload", "commit-before-write", "commit-after-write"])
def test_failed_first_v1_transition_restores_exact_config_and_state(tmp_path, monkeypatch, failure):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy)
    old_config, old_state = caddy.read_bytes(), state.read_bytes()
    calls = []

    def run(command, **_kwargs):
        calls.append(command[1])
        if failure == "reload" and calls == ["validate", "reload"]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    original = ingress.commit_state
    if failure != "reload":
        def fail_commit(*args, **kwargs):
            if failure == "commit-after-write":
                original(*args, **kwargs)
            raise RuntimeError("state sync failed")
        monkeypatch.setattr(ingress, "commit_state", fail_commit)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        activate(state, caddy, expected_mode="dual", expected_slot=0,
                 expected_sha="a" * 40, expected_backend="ready",
                 mode="dual", slot=1, sha="b" * 40, backend="closed")
    assert calls == ["validate", "reload", "reload"]
    assert caddy.read_bytes() == old_config
    assert state.read_bytes() == old_state
    assert inspect(state, caddy)["version"] == 1


@pytest.mark.parametrize("version", [1, 2])
def test_state_version_cannot_hide_missing_or_unrecorded_database_route(tmp_path, version):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy)
    record = json.loads(state.read_text())
    record["version"] = version
    state.write_text(json.dumps(record))
    if version == 1:
        caddy.write_text(caddy.read_text().replace("    @arena path", DATABASE_HANDLER + "    @arena path", 1))
    with pytest.raises(ValueError, match="differs"):
        inspect(state, caddy)


@pytest.mark.parametrize("version", [True, 1.0, 0, 3, "2"])
def test_unknown_or_noninteger_state_version_is_rejected(tmp_path, version):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy)
    record = json.loads(state.read_text())
    record["version"] = version
    state.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="state is invalid"):
        inspect(state, caddy)


@pytest.mark.parametrize("edit", ["whitespace", "wrong-slot", "wrong-backend"])
def test_v1_mismatch_is_rejected_before_validation_or_reload(tmp_path, monkeypatch, edit):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy)
    content = caddy.read_text()
    if edit == "whitespace":
        content += "\n"
    elif edit == "wrong-slot":
        content = content.replace("127.0.0.1:8890", "127.0.0.1:8891")
    else:
        content = content.replace("127.0.0.1:8001", "127.0.0.1:8081")
    caddy.write_text(content)
    old_state = state.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kw: pytest.fail("unverified route was used"))
    with pytest.raises(ValueError, match="differs"):
        activate(state, caddy, expected_mode="dual", expected_slot=0,
                 expected_sha="a" * 40, expected_backend="ready",
                 mode="dual", slot=1, sha="b" * 40, backend="closed")
    assert caddy.read_text() == content and state.read_bytes() == old_state


@pytest.mark.parametrize("version", [1, 2])
def test_inspect_cli_keeps_existing_controller_fields(tmp_path, monkeypatch, capsys, version):
    state, caddy = tmp_path / "state.json", tmp_path / "Caddyfile"
    _write_v1(state, caddy)
    if version == 2:
        caddy.write_text(render(arena_edge_port=8890, backend="ready", direct=True))
        commit_state(state, caddy, mode="dual", slot=0, sha="a" * 40, backend="ready")
    monkeypatch.setattr(sys, "argv", ["arena_ingress_config", "--inspect",
                                    "--state-file", str(state), "--caddy-file", str(caddy)])
    assert ingress.main() == 0
    assert capsys.readouterr().out == f"dual 0 {'a' * 40} ready\n"


def test_domain_stage_keeps_legacy_gateway_but_moves_arena_to_edge():
    config = render(arena_edge_port=8891, backend="legacy", direct=False)
    assert config.count("reverse_proxy 127.0.0.1:8891") == 1
    assert config.count("reverse_proxy 127.0.0.1:8000") == 1
    assert ":8000 {" not in config
    assert "@arena path /arena /arena/* /testnet/arena /testnet/arena/*" in config


def test_dual_listener_holds_gateway_closed_until_verified():
    closed = render(arena_edge_port=8891, backend="closed", direct=True)
    assert closed.count("reverse_proxy 127.0.0.1:8891") == 2
    assert closed.count('respond "gateway temporarily unavailable" 503') == 2
    assert "reverse_proxy 127.0.0.1:8001" not in closed
    ready = render(arena_edge_port=8890, backend="ready", direct=True)
    assert ready.count("reverse_proxy 127.0.0.1:8890") == 2
    assert ready.count("reverse_proxy 127.0.0.1:8001") == 2


@pytest.mark.parametrize(
    ("edge_port", "backend", "direct"),
    [(8792, "ready", True), (8890, "legacy", True), (8890, "unknown", False),
     (8800, "ready", True), (8801, "ready", True)],
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
    caddy.write_text(render(arena_edge_port=8891, backend="ready", direct=True))
    commit_state(state, caddy, mode="dual", slot=1, sha=sha, backend="ready")
    assert inspect(state, caddy) == {
        "version": 2, "mode": "dual", "slot": 1, "sha": sha, "backend": "ready",
    }
    caddy.write_text(render(arena_edge_port=8891, backend="closed", direct=True))
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


@pytest.mark.parametrize("slot,api_port,edge_port", [(0, 8792, 8890), (1, 8794, 8891)])
def test_caddy_and_controller_share_the_same_fixed_slot_pair(slot, api_port, edge_port):
    source = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text()
    function = source[source.index("arena_slot_ports() {"):].split("\n}\n", 1)[0] + "\n}\n"
    result = subprocess.run(
        ["bash", "-c", function + '\narena_slot_ports "$1"; printf "%s %s" "$ARENA_API_PORT" "$ARENA_EDGE_PORT"',
         "slot-parity", str(slot)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == f"{api_port} {edge_port}"
    assert ingress.slot_ports(slot) == (api_port, edge_port)
    assert f"reverse_proxy 127.0.0.1:{edge_port}" in render(
        arena_edge_port=edge_port, backend="ready", direct=True,
    )
