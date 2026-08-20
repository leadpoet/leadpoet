from __future__ import annotations

import os
from pathlib import Path

from tests.readiness_test_venv import build_dependency_complete_readiness_venv


def test_readiness_venv_removes_collaborative_umask_write_bits(
    tmp_path: Path,
) -> None:
    previous_umask = os.umask(0o002)
    try:
        root = tmp_path / "readiness-venv"
        python = build_dependency_complete_readiness_venv(root)
    finally:
        os.umask(previous_umask)

    assert python.is_file()
    assert not python.is_symlink()
    assert python.resolve(strict=True).is_relative_to(root.resolve(strict=True))
    assert all(
        path.is_symlink() or path.stat().st_mode & 0o022 == 0
        for path in (root, *root.rglob("*"))
    )
