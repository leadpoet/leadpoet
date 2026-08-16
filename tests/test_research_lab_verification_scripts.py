"""Regression coverage for the offline Research Lab workflow verifiers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name,success_marker",
    (
        (
            "verify_research_lab_scoring_worker_readiness.py",
            "measured provider preflight blocks claims until healthy",
        ),
        (
            "verify_research_lab_hosted_worker.py",
            "Research Lab hosted worker contracts verified",
        ),
        (
            "verify_research_lab_private_model_runtime.py",
            "Research Lab private model runtime bridge verified",
        ),
    ),
)
def test_research_lab_workflow_verifier_runs_without_production_credentials(
    script_name: str,
    success_marker: str,
    tmp_path: Path,
) -> None:
    env = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        # HOME is intentionally isolated, so retain only installed dependency
        # roots.  Passing the interpreter's stdlib entries through PYTHONPATH
        # can poison the fake Docker subprocess on macOS.
        "PYTHONPATH": os.pathsep.join(
            path for path in sys.path if path and "site-packages" in Path(path).parts
        ),
        "TMPDIR": str(tmp_path),
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(
            tmp_path / "docker-operation-v2.lock"
        ),
        "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE": str(
            tmp_path / "docker-operation-v2.lock.admission"
        ),
    }

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert success_marker in completed.stdout
