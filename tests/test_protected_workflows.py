from pathlib import Path
import re
import subprocess

import pytest

from gateway.tee.protected_workflows import (
    PROTECTED_SYMBOLS,
    ProtectedWorkflowError,
    build_manifest,
    load_manifest,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "gateway" / "tee" / "protected_workflows.json"


def test_committed_protected_workflow_manifest_matches_source():
    manifest = load_manifest(MANIFEST_PATH)
    verify_manifest(ROOT, manifest)
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["baseline_commit"])
    protected_source = manifest["protected_source_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", protected_source)
    subprocess.run(
        ["git", "cat-file", "-e", protected_source + "^{commit}"],
        cwd=ROOT,
        check=True,
    )
    assert len(manifest["entries"]) == sum(len(items) for items in PROTECTED_SYMBOLS.values())


def test_protected_manifest_detects_logic_change(tmp_path: Path):
    committed = load_manifest(MANIFEST_PATH)
    manifest = build_manifest(
        ROOT,
        baseline_commit=committed["baseline_commit"],
        protected_source_commit=committed["protected_source_commit"],
    )
    copied_root = tmp_path / "repo"
    for relative_path in PROTECTED_SYMBOLS:
        source = ROOT / relative_path
        destination = copied_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    target = copied_root / "research_lab" / "eval" / "promotion_metric.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "return PromotionGateDecision(",
            "assert threshold_points >= 0\n    return PromotionGateDecision(",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtectedWorkflowError, match="promotion_metric.py"):
        verify_manifest(copied_root, manifest)
