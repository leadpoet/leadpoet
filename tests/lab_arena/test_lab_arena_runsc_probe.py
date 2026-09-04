"""The Lab Arena runsc probe: dry-run plan here, live execution on a root
Linux x86_64 lane (labarena.md 18.4)."""

from __future__ import annotations

import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts/_lab_arena_runsc_probe_ci.py"


def test_dry_run_validates_the_sandbox_plan(capsys):
    module = runpy.run_path(str(PROBE), run_name="lab_arena_runsc_probe_module")
    assert module["main"](["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LAB_ARENA_RUNSC_PROBE_DRY_RUN_OK" in out
    assert out.count("PLAN ") == 3 and "--network=none" in out and "--rootless=false" in out


def test_live_probe_refuses_without_root_or_on_the_wrong_host():
    if os.geteuid() == 0 and platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64"):
        pytest.skip("live probe host: run scripts/_lab_arena_runsc_probe_ci.py in the sandbox lane")
    result = subprocess.run([sys.executable, str(PROBE)], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode != 0
    assert "must execute as root" in result.stderr or "requires" in result.stderr or "Linux" in result.stderr
