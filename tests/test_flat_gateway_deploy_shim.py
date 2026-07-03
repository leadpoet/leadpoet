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
