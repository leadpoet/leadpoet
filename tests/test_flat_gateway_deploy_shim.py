"""Regression coverage for gateway bootstrap import ordering."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
