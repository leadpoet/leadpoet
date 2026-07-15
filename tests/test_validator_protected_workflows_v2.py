from pathlib import Path
import re
import subprocess

import pytest

from validator_tee.host.protected_workflows_v2 import (
    PROTECTED_SYMBOLS,
    ValidatorProtectedWorkflowError,
    build_manifest,
    load_manifest,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "validator_tee" / "enclave" / "protected_workflows_v2.json"
)


def test_committed_validator_protected_manifest_matches_source():
    manifest = load_manifest(MANIFEST_PATH)
    verify_manifest(ROOT, manifest)
    assert manifest["baseline_commit"] == (
        "7c9766b71d4c08b0059f6e3230dbe742b1d58e79"
    )
    protected_source = manifest["protected_source_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", protected_source)
    subprocess.run(
        ["git", "cat-file", "-e", protected_source + "^{commit}"],
        cwd=ROOT,
        check=True,
    )
    assert len(manifest["entries"]) == sum(
        len(items) for items in PROTECTED_SYMBOLS.values()
    )


def test_validator_protected_manifest_detects_weight_logic_change(tmp_path: Path):
    manifest = build_manifest(
        ROOT,
        baseline_commit="7c9766b71d4c08b0059f6e3230dbe742b1d58e79",
        protected_source_commit="178ad8652744f2171a3cf6775767f55c5e59bbae",
    )
    copied_root = tmp_path / "repo"
    for relative_path in PROTECTED_SYMBOLS:
        source = ROOT / relative_path
        destination = copied_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    target = copied_root / "leadpoet_canonical" / "weight_computation.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "def compute_final_weights(snapshot: Mapping[str, Any]) -> Dict[str, Any]:",
            "def compute_final_weights(snapshot: Mapping[str, Any]) -> Dict[str, Any]:\n"
            "    assert snapshot is not None",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidatorProtectedWorkflowError, match="compute_final_weights"):
        verify_manifest(copied_root, manifest)


def test_validator_build_runs_protected_workflow_gate():
    script = (ROOT / "validator_tee" / "scripts" / "build_enclave.sh").read_text(
        encoding="utf-8"
    )
    assert "validator_tee.host.protected_workflows_v2" in script
    assert "protected_workflows_v2.json" in script
