"""Retained-slot discovery never adopts or signals a foreign process."""
import argparse
import json
import os
from pathlib import Path

import pytest

from scripts import manage_owned_process_group as owned


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    tree = root / ("slot0-" + "6" * 40)
    tree.mkdir(parents=True)
    args = argparse.Namespace(runtime_root=root, slot=0, uid=os.getuid(), python_bin="/fixture/python",
                              environment_file="/fixture/gateway.env", owner="gateway-1000-2000")
    states = {}
    for number, component in enumerate(("driver", "edge", "api"), 10):
        command = [args.python_bin, "-B", "-u", "scripts/run_lab_arena_service.py", "--environment-file", args.environment_file]
        if component == "driver":
            command += ["--driver-only"]
        elif component == "api":
            command += ["--host", "127.0.0.1", "--port", "8792", "--no-driver"]
        else:
            command[3] = "scripts/run_arena_proxy_edge.py"
            command += ["--port", "8890", "--sidecar-port", "8792"]
        state = dict(version=1, pid=number, pgid=number, start_time_ticks=100 + number,
                     uid=os.getuid(), cwd=str(tree), argv=command)
        path = root / f"{component}-slot0-process.json"
        path.write_text(json.dumps(state))
        path.chmod(0o600)
        states[number] = state
    live = {pid: dict(state, state="S", session=pid) for pid, state in states.items()}
    monkeypatch.setattr(owned, "_read_process", lambda pid: live.get(pid))
    monkeypatch.setattr(owned, "_verified_invocation", lambda *args: "gateway-500-600")
    monkeypatch.setattr(owned, "_active_group_members", lambda pid: [live[pid]] if pid in live else [])
    monkeypatch.setattr(owned, "_matching_processes", lambda **kw: [p for p in live.values()
                          if p["argv"] == kw["argv"] and p["cwd"] == kw["cwd"]])
    monkeypatch.setattr(owned.os, "kill", lambda *_: pytest.fail("inspection signaled a process"))
    monkeypatch.setattr(owned.os, "killpg", lambda *_: pytest.fail("inspection signaled a group"))
    return args, states, live


@pytest.mark.parametrize("driver_alive", [True, False])
def test_exact_retained_slot_and_natural_driver_exit(tmp_path, monkeypatch, driver_alive):
    args, states, live = fixture(tmp_path, monkeypatch)
    if not driver_alive:
        live.pop(10)
    before = {p: p.read_bytes() for p in args.runtime_root.glob("*.json")}
    result = owned._inspect_retained_slot(args)
    assert result["sha"] == "6" * 40 and result["tree"] == states[10]["cwd"]
    assert result["owner"] == "gateway-500-600"
    assert set(result["components"]) == {"driver", "edge", "api"}
    assert all(len(digest) == 64 for digest in result["components"].values())
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("damage", ["pid", "start", "cwd", "argv", "uid", "source", "mixed_source", "symlink", "mode", "owner", "descendant"])
def test_foreign_or_changed_slot_refuses_before_any_stop(tmp_path, monkeypatch, damage):
    args, states, live = fixture(tmp_path, monkeypatch)
    path = args.runtime_root / "driver-slot0-process.json"
    if damage in {"pid", "start", "cwd", "argv", "uid"}:
        key = {"start": "start_time_ticks"}.get(damage, damage)
        live[10][key] = 999 if key in {"pid", "uid", "start_time_ticks"} else "/foreign" if key == "cwd" else ["foreign"]
    elif damage in {"source", "mixed_source"}:
        state = json.loads(path.read_text())
        state["cwd"] = str(tmp_path / "foreign") if damage == "source" else str(args.runtime_root / ("slot0-" + "7" * 40))
        Path(state["cwd"]).mkdir()
        path.write_text(json.dumps(state))
    elif damage == "symlink":
        original = path.with_suffix(".saved")
        path.rename(original)
        path.symlink_to(original)
    elif damage == "mode":
        path.chmod(0o644)
    elif damage == "owner":
        monkeypatch.setattr(owned, "_verified_invocation", lambda _args, state: f"gateway-{state['pid']}-600")
    else:
        live.pop(10)
        monkeypatch.setattr(owned, "_active_group_members", lambda pid: [{"pid": 999}] if pid == 10 else [live[pid]])
    with pytest.raises(owned.OwnershipError):
        owned._inspect_retained_slot(args)
