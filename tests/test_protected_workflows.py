from pathlib import Path

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
    assert manifest["baseline_commit"] == "7c9766b71d4c08b0059f6e3230dbe742b1d58e79"
    assert manifest["protected_source_commit"] == "322f9f653e44fc291710fc754d69c79ed23f9fa1"
    assert len(manifest["entries"]) == sum(len(items) for items in PROTECTED_SYMBOLS.values())


def test_protected_manifest_detects_logic_change(tmp_path: Path):
    manifest = build_manifest(
        ROOT,
        baseline_commit="7c9766b71d4c08b0059f6e3230dbe742b1d58e79",
        protected_source_commit="6a4deb788431acff9dd42aaf306cab7297e45053",
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
