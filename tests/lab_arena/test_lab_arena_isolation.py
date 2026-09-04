"""Section 18.9: with LAB_ARENA_MODE=off no Arena process runs, no Arena
database request occurs, and the operator commands refuse to act."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "HOME": str(ROOT), "LAB_ARENA_MODE": "off", "SUPABASE_URL": "https://test.invalid", "SUPABASE_SERVICE_ROLE_KEY": "x"}


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    program = (
        "import sys, runpy\n"
        "sys.argv = [%r] + %r\n"
        "try:\n"
        "    runpy.run_path(%r, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = exc.code\n"
        "else:\n"
        "    code = 0\n"
        "bad = sorted(name for name in sys.modules if name in ('lab_arena.wiring', 'lab_arena.store', 'lab_arena.service', 'httpx', 'boto3'))\n"
        "print('MODULES', bad)\n"
        "raise SystemExit(code)\n" % (script, list(args), str(ROOT / script))
    )
    return subprocess.run([sys.executable, "-s", "-B", "-c", program], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=120, check=False)


def test_service_off_starts_nothing_and_touches_no_database_module():
    result = _run("scripts/run_lab_arena_service.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "nothing starts and nothing is served" in result.stdout
    assert "MODULES []" in result.stdout


def test_admin_off_refuses_every_command():
    for args in (["status"], ["advance", "--round", "arena-2026-09-02"], ["cancel", "--round", "arena-2026-09-02", "--reason", "operator"]):
        result = _run("scripts/lab_arena_admin.py", *args)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "operator commands are disabled" in result.stderr
        assert "MODULES []" in result.stdout


def test_runner_requires_explicit_api_and_round_and_imports_nothing_heavy():
    result = _run("scripts/run_lab_arena_runner.py")
    assert result.returncode == 2 and "required" in result.stderr
    assert "MODULES []" in result.stdout
