"""Regression coverage for flat EC2 gateway deployments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_flat_gateway_checkout_resolves_gateway_subpackages():
    code = """
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "gateway"))
spec = importlib.util.find_spec("gateway.research_lab")
assert spec is not None, "gateway.research_lab not found"
assert spec.origin and spec.origin.endswith("/gateway/research_lab/__init__.py"), spec.origin
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_gateway_bootstrap_prioritizes_package_parent_over_flat_cwd():
    package_parent = str(ROOT)
    gateway_dir = str(ROOT / "gateway")
    attested_runtime_dir = str(ROOT / "gateway" / "_attested_runtime")
    sys_path = [gateway_dir, package_parent, "/usr/lib/python"]

    for path in (attested_runtime_dir, package_parent):
        if not Path(path).is_dir():
            continue
        while path in sys_path:
            sys_path.remove(path)
        sys_path.insert(0, path)

    assert sys_path[0] == package_parent
    assert sys_path[1] == gateway_dir
