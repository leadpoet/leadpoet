from __future__ import annotations

from pathlib import Path

import pytest

from gateway.research_lab.tee_protocol import (
    V2_PROTOCOL,
    ResearchLabTeeProtocolError,
    legacy_v1_enabled,
    normalize_tee_protocol,
    research_lab_tee_protocol,
    v2_enabled,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_protocol_defaults_to_v2_and_normalizes_explicit_aliases(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LAB_TEE_PROTOCOL", raising=False)
    assert research_lab_tee_protocol() == V2_PROTOCOL
    assert v2_enabled() is True
    assert legacy_v1_enabled() is False
    assert normalize_tee_protocol("authoritative_v2") == V2_PROTOCOL
    with pytest.raises(ResearchLabTeeProtocolError, match="V1 authority is retired"):
        normalize_tee_protocol("LEGACY_V1_COMPAT")
    with pytest.raises(ResearchLabTeeProtocolError, match="RESEARCH_LAB_TEE_PROTOCOL"):
        normalize_tee_protocol("automatic")


def test_gateway_restart_is_v2_only_and_keeps_the_release_gate() -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert '${RESEARCH_LAB_TEE_PROTOCOL:-v2}' in restart
    assert "gateway.tee.restart_preflight_v2" in restart
    assert "prepare_offline_artifacts_v2.sh" in restart
    assert "build_role_enclaves.sh" in restart
    assert "legacy_v1" not in restart
    assert "prepare_legacy_enclave_artifacts.sh" not in restart
    assert "GATEWAY_DEPLOY_COMMIT" not in restart
    assert "07c81c7f" not in restart


def test_runtime_annotations_evaluate_on_python_39():
    """PEP 604 unions in runtime-evaluated annotations crash Python 3.9.

    Module-level def annotations evaluate at import; nested-def annotations
    evaluate on every call of the enclosing function (weights.py's
    _chain_snapshot 500'd every /weights/submit during epoch 23918).
    Files opting into `from __future__ import annotations` are exempt.
    """
    import ast
    import itertools
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    scanned_trees = (
        "gateway",
        "research_lab",
        "qualification",
        "validator_models",
        "neurons",
        "leadpoet_verifier",
        "Leadpoet",
        "miner_models",
        "leadpoet_canonical",
        "leadpoet_audit",
    )
    offenders = []
    for path in itertools.chain.from_iterable(
        sorted((repo_root / tree).rglob("*.py")) for tree in scanned_trees
    ):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        has_future = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if has_future:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            annotations = [
                arg.annotation
                for arg in list(node.args.args) + list(node.args.kwonlyargs)
                if arg.annotation
            ]
            if node.returns:
                annotations.append(node.returns)
            for annotation in annotations:
                if any(
                    isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)
                    for sub in ast.walk(annotation)
                ):
                    offenders.append(f"{path}:{node.lineno} {node.name}")
                    break
    assert offenders == []
