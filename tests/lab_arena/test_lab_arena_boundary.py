"""The three boundary tests of labarena.md section 3.1 (and 18.5).

1. The runtime import closure of every Arena module and entrypoint, in a
   fresh interpreter, contains no ``gateway.tee`` and no ``gateway.db``.
2. Gateway enclave staging excludes Arena.
3. Measured code cannot import the Arena service or scoring runtime.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
import site
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "lab_arena"
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
    gateway_dockerfile = (ROOT / "gateway/tee/Dockerfile.enclave").read_text(encoding="utf-8")
    assert "lab_arena" not in gateway_dockerfile
    closure = (ROOT / "gateway/tee/scoring_import_closure.py").read_text(encoding="utf-8")
    assert "lab_arena" not in closure


def test_lab_arena_is_in_the_distribution_without_a_runtime_lock():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "'lab_arena', 'lab_arena.*'" in setup
    assert "runtime.lock.json" not in setup


def _enclave_python_sources(root: Path) -> set[Path]:
    from gateway.tee.scoring_import_closure import (
        DYNAMIC_IMPORT_MODULES,
        build_module_index,
        discover_scoring_modules,
    )

    assert not any(name == "lab_arena" or name.startswith("lab_arena.") for name in DYNAMIC_IMPORT_MODULES)
    index = build_module_index(gateway_root=root / "gateway", source_root=root)
    paths = {index[name] for name in discover_scoring_modules(index)}
    staging = (root / "gateway/tee/stage_attested_runtime.sh").read_text(encoding="utf-8")
    packages_block = re.search(r"^PACKAGES=\((.*?)\)", staging, re.S | re.M)
    assert packages_block is not None, "staging package allowlist is missing"
    packages = shlex.split(packages_block.group(1), comments=True)
    assert packages, "staging package allowlist is empty"
    for package in packages:
        assert re.fullmatch(r"[A-Za-z_]\w*", package), package
        assert package != "lab_arena", package
        base = root / package
        assert base.is_dir(), package
        paths.update(base.rglob("*.py"))
    # Preserve the separate validator-enclave guard if that tree is present.
    paths.update((root / "validator_tee/enclave").rglob("*.py"))
    return paths


def _assert_no_arena_runtime_imports(path: Path, root: Path) -> int:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    # Retain the literal-prefix guard even for concatenated import arguments.
    assert not re.search(r"(?:import_module|__import__)\(\s*['\"]lab_arena(?:\.|['\"])", source), path
    contracts_imports = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name == "lab_arena" or alias.name.startswith("lab_arena.") for alias in node.names), path
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "lab_arena" or module.startswith("lab_arena.")):
                # The signing validator may use only the canonical request
                # contracts, never the Arena service, scorer or store.
                assert path.relative_to(root).as_posix() == "leadpoet_canonical/arena_weights.py", path
                assert module == "lab_arena" and len(node.names) == 1, path
                assert node.names[0].name == "contracts" and node.names[0].asname is None, path
                contracts_imports += 1
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            if name in {"import_module", "__import__"}:
                arguments = list(node.args) + [keyword.value for keyword in node.keywords if keyword.arg == "name"]
                for argument in arguments:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        assert not (argument.value == "lab_arena" or argument.value.startswith("lab_arena.")), path
    assert contracts_imports <= 1, path
    return contracts_imports


def test_no_measured_package_imports_lab_arena():
    paths = _enclave_python_sources(ROOT)
    assert sum(_assert_no_arena_runtime_imports(path, ROOT) for path in sorted(paths)) == 1


def test_boundary_follows_enclave_dependencies_and_staged_packages(tmp_path):
    files = {
        "gateway/__init__.py": "",
        "gateway/tee/__init__.py": "",
        "gateway/tee/tee_service.py": "from gateway.tee import dependency\n",
        "gateway/tee/dependency.py": "",
        "gateway/tee/protected_workflows.py": "",
        "gateway/host_only.py": "from lab_arena import service\n",
        "leadpoet_canonical/__init__.py": "",
        "leadpoet_canonical/attested_v2.py": "",
        "leadpoet_canonical/staged_only.py": "",
        "gateway/tee/stage_attested_runtime.sh": 'PACKAGES=("leadpoet_canonical")\n',
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    paths = _enclave_python_sources(tmp_path)
    assert tmp_path / "gateway/tee/dependency.py" in paths
    assert tmp_path / "leadpoet_canonical/staged_only.py" in paths
    assert tmp_path / "gateway/host_only.py" not in paths
    for path in paths:
        _assert_no_arena_runtime_imports(path, tmp_path)
    # A newly reachable dependency and a non-entrypoint staged module are both
    # inside the boundary, even though the unrelated host module is outside it.
    for name in ("gateway/tee/dependency.py", "leadpoet_canonical/staged_only.py"):
        path = tmp_path / name
        path.write_text("from lab_arena import service\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            _assert_no_arena_runtime_imports(path, tmp_path)


@pytest.mark.parametrize("source", [
    "import lab_arena.scoring",
    "from lab_arena import contracts",
    "importlib.import_module('lab_arena.service')",
    "importlib.import_module('lab_arena' + '.service')",
    "import_module(name='lab_arena.scoring')",
    "__import__('lab_arena')",
])
def test_boundary_rejects_static_and_literal_dynamic_arena_imports(tmp_path, source):
    path = tmp_path / "measured.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_no_arena_runtime_imports(path, tmp_path)


@pytest.mark.parametrize("source", [
    "from lab_arena import service",
    "from lab_arena import contracts, scoring",
    "from lab_arena.contracts import signed_request_message",
    "from lab_arena import contracts as service",
    "from lab_arena import contracts\nfrom lab_arena import contracts",
    "importlib.import_module('lab_arena.contracts')",
])
def test_contracts_exception_cannot_import_arena_runtime(tmp_path, source):
    path = tmp_path / "leadpoet_canonical/arena_weights.py"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_no_arena_runtime_imports(path, tmp_path)


def test_boundary_does_not_ignore_unparseable_source(tmp_path):
    path = tmp_path / "measured.py"
    path.write_text("from lab_arena import (", encoding="utf-8")
    with pytest.raises(SyntaxError):
        _assert_no_arena_runtime_imports(path, tmp_path)
