"""The three boundary tests of labarena.md section 3.1 (and 18.5).

1. The runtime import closure of every Arena module and entrypoint, in a
   fresh interpreter, contains no ``gateway.tee`` and no ``gateway.db``.
2. The enclave staging allowlists contain no ``lab_arena`` path.
3. No measured package imports ``lab_arena`` (the closure builder would not
   flag it and the enclave would fail at import time).
"""

from __future__ import annotations

import ast
import os
import re
import site
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "lab_arena"
MEASURED_PACKAGES = (
    "gateway",
    "leadpoet_canonical",
    "leadpoet_verifier",
    "research_lab",
    "qualification",
    "validator_models",
    "schemas",
    "Leadpoet",
)
ENTRYPOINT_SCRIPTS = (
    "scripts/run_lab_arena_service.py",
    "scripts/run_lab_arena_runner.py",
    "scripts/lab_arena_admin.py",
)


def arena_modules() -> list[str]:
    names = []
    for path in sorted(ARENA.glob("*.py")):
        if path.name == "__init__.py":
            names.append("lab_arena")
        else:
            names.append("lab_arena." + path.stem)
    return names


@pytest.mark.parametrize("module", arena_modules())
def test_arena_module_import_closure_has_no_enclave_or_database_module(module):
    program = (
        "import sys, importlib\n"
        "sys.path.insert(0, %r)\n" % str(ROOT) +
        "importlib.import_module(%r)\n"
        "bad = sorted(name for name in sys.modules if name == 'gateway.tee' or name.startswith('gateway.tee.') "
        "or name == 'gateway.db' or name.startswith('gateway.db.'))\n"
        "assert not bad, bad\n"
        "print('clean')\n" % module
    )
    result = subprocess.run(
        [sys.executable, "-s", "-B", "-c", program],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": os.pathsep.join((str(ROOT), site.getusersitepackages())), "SUPABASE_URL": "https://test.invalid", "SUPABASE_SERVICE_ROLE_KEY": "x", "HOME": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


@pytest.mark.parametrize("script", ENTRYPOINT_SCRIPTS)
def test_entrypoint_import_closure_has_no_enclave_or_database_module(script):
    path = ROOT / script
    if not path.exists():
        pytest.skip("entrypoint not built yet: %s" % script)
    program = (
        "import sys, runpy\n"
        "sys.path.insert(0, %r)\n" % str(ROOT) +
        "sys.argv = [%r, '--help']\n"
        "try:\n"
        "    runpy.run_path(%r, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "bad = sorted(name for name in sys.modules if name == 'gateway.tee' or name.startswith('gateway.tee.') "
        "or name == 'gateway.db' or name.startswith('gateway.db.'))\n"
        "assert not bad, bad\n"
        "print('clean')\n" % (str(path), str(path))
    )
    result = subprocess.run(
        [sys.executable, "-s", "-B", "-c", program],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "SUPABASE_URL": "https://test.invalid", "SUPABASE_SERVICE_ROLE_KEY": "x", "HOME": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_enclave_staging_allowlists_exclude_lab_arena():
    staging = (ROOT / "gateway/tee/stage_attested_runtime.sh").read_text(encoding="utf-8")
    packages_block = re.search(r"PACKAGES=\((.*?)\)", staging, re.S).group(1)
    assert "lab_arena" not in packages_block
    assert "lab_arena" not in staging
    code_hash = (ROOT / "gateway/tee/code_hash.py").read_text(encoding="utf-8")
    assert "lab_arena" not in code_hash
    dockerfile = (ROOT / "validator_tee/Dockerfile.enclave").read_text(encoding="utf-8")
    copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert copy_lines and not any("lab_arena" in line for line in copy_lines)
    gateway_dockerfile = (ROOT / "gateway/tee/Dockerfile.enclave").read_text(encoding="utf-8")
    assert "lab_arena" not in gateway_dockerfile
    closure = (ROOT / "gateway/tee/scoring_import_closure.py").read_text(encoding="utf-8")
    assert "lab_arena" not in closure


def test_lab_arena_is_in_the_distribution_without_a_runtime_lock():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "'lab_arena', 'lab_arena.*'" in setup
    assert "runtime.lock.json" not in setup


def _imports_lab_arena(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "lab_arena" or alias.name.startswith("lab_arena.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "lab_arena" or module.startswith("lab_arena.")):
                return True
    return False


def test_no_measured_package_imports_lab_arena():
    offenders = []
    for package in MEASURED_PACKAGES:
        base = ROOT / package
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if _imports_lab_arena(path):
                offenders.append(str(path.relative_to(ROOT)))
    for extra in (ROOT / "neurons", ROOT / "validator_tee"):
        for path in extra.rglob("*.py"):
            if _imports_lab_arena(path):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    # And the dynamic-import string form, which AST cannot see.
    text_offenders = []
    for package in MEASURED_PACKAGES:
        base = ROOT / package
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if re.search(r"import_module\(\s*['\"]lab_arena", path.read_text(encoding="utf-8", errors="replace")):
                text_offenders.append(str(path.relative_to(ROOT)))
    assert text_offenders == []
