from pathlib import Path
import subprocess

from leadpoet_canonical.production_parity import sha256_bytes
from scripts.build_production_parity_contract import (
    ALWAYS_COMMITTED_PATHS,
    _source_commitments,
)


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_STAGING_PATH = "scripts/run_physical_v2_staging.py"


def test_physical_staging_is_bound_as_an_exact_candidate_git_blob() -> None:
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate_blob = subprocess.run(
        ["git", "show", f"{candidate_sha}:{PHYSICAL_STAGING_PATH}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout

    assert PHYSICAL_STAGING_PATH in ALWAYS_COMMITTED_PATHS
    assert _source_commitments(
        ROOT,
        candidate_sha,
        [PHYSICAL_STAGING_PATH],
    ) == [
        {
            "path": PHYSICAL_STAGING_PATH,
            "sha256": sha256_bytes(candidate_blob),
        }
    ]
