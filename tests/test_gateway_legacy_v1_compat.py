from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.research_lab import allocations
from gateway.research_lab.tee_protocol import (
    LEGACY_V1_PROTOCOL,
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
    assert normalize_tee_protocol("LEGACY_V1_COMPAT") == LEGACY_V1_PROTOCOL
    assert normalize_tee_protocol("authoritative_v2") == V2_PROTOCOL
    with pytest.raises(ResearchLabTeeProtocolError, match="RESEARCH_LAB_TEE_PROTOCOL"):
        normalize_tee_protocol("automatic")


def test_gateway_restart_keeps_v2_default_but_legacy_skips_six_build_gate() -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert '${RESEARCH_LAB_TEE_PROTOCOL:-v2}' in restart
    assert 'if [ "$RESEARCH_LAB_TEE_PROTOCOL" = "v2" ]; then' in restart
    assert "gateway.tee.restart_preflight_v2" in restart
    assert "prepare_offline_artifacts_v2.sh" in restart
    assert "Legacy V1 selected; skipping the six-build V2 release preflight" in restart
    assert "prepare_legacy_enclave_artifacts.sh" in restart
    assert "Building one legacy gateway EIF from the current deploy commit" in restart
    assert 'RESEARCH_LAB_EVIDENCE_PROXY_URL="http://172.17.0.1:8791"' in restart
    assert "build_role_enclaves.sh" in restart
    assert "GATEWAY_DEPLOY_COMMIT" not in restart
    assert "07c81c7f" not in restart


def test_legacy_gateway_builds_and_starts_one_current_commit_enclave() -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    start = (ROOT / "gateway" / "tee" / "start_enclave.sh").read_text(
        encoding="utf-8"
    )
    artifacts = (
        ROOT / "gateway" / "tee" / "prepare_legacy_enclave_artifacts.sh"
    ).read_text(encoding="utf-8")

    assert '-f "$GATEWAY_ROOT/tee/Dockerfile.enclave"' in restart
    assert '"$GATEWAY_ROOT/"' in restart
    assert '--output-file "$GATEWAY_TEE_EIF_ROOT/tee-enclave.eif"' in restart
    assert '--enclave-cid 16' in start
    assert 'EIF_PATH="$EIF_ROOT/tee-enclave.eif"' in start
    assert "Legacy gateway enclave RPC is healthy" in start
    assert "requirements-scoring-py39.lock" in artifacts
    assert "runsc-runtime.lock.json" in artifacts
    assert artifacts.count(
        'PYTHONPATH="$REPO_ROOT" python3 '
        '"$SCRIPT_DIR/sandbox_runtime_artifact.py" verify'
    ) == 3
    assert "validator" not in artifacts.lower()
    assert "release-manifest" not in artifacts
    assert "GATEWAY_DEPLOY_COMMIT" not in artifacts


@pytest.mark.asyncio
async def test_legacy_allocation_preserves_host_reward_kernel(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")
    calls = []

    async def alpha_price(**_kwargs):
        return {"source": "test"}

    async def reimbursements(_epoch, *, policy):
        calls.append(("reimbursements", policy["policy_id"]))
        return [], []

    async def champions(_epoch, *, netuid):
        calls.append(("champions", netuid))
        return [], []

    async def source_add(_epoch, *, netuid):
        calls.append(("source_add", netuid))
        return [], []

    async def forbidden_v2(**_kwargs):
        raise AssertionError("legacy allocation must not call V2 authority")

    def allocate(epoch, policy, reimbursements, champions, **kwargs):
        calls.append(("allocate", epoch))
        assert policy["policy_id"] == "policy:test"
        assert reimbursements == []
        assert champions == []
        assert kwargs["active_source_add_obligations"] == []
        return {"allocation_hash": "sha256:" + "a" * 64}

    monkeypatch.setattr(allocations, "resolve_epoch_alpha_price_valuation", alpha_price)
    monkeypatch.setattr(
        allocations,
        "inject_alpha_price_valuation",
        lambda policy, _valuation: dict(policy),
    )
    monkeypatch.setattr(allocations, "_active_reimbursement_obligations", reimbursements)
    monkeypatch.setattr(allocations, "_active_champion_obligations", champions)
    monkeypatch.setattr(allocations, "_active_source_add_obligations", source_add)
    monkeypatch.setattr(allocations, "allocate_research_lab_epoch", allocate)
    monkeypatch.setattr(allocations, "build_allocation_v2", forbidden_v2)
    config = SimpleNamespace(
        reimbursement_policy_doc=lambda enabled: {
            "policy_id": "policy:test",
            "enabled": enabled,
        },
        reimbursement_dynamic_alpha_price_enabled=False,
        reimbursement_require_live_alpha_price=False,
        reimbursement_miner_alpha_per_epoch=1.0,
        reimbursement_usd_per_0_1_percent_epoch=1.0,
        reimbursements_enabled=True,
        weight_mutation_enabled=True,
        production_writes_enabled=False,
    )
    attestation = {}
    bundle = await allocations.build_research_lab_allocation_bundle(
        config=config,
        epoch=300,
        netuid=71,
        attestation_out=attestation,
    )
    assert bundle["submission_allowed"] is True
    assert attestation == {"status": "off", "protocol": "legacy_v1"}
    assert calls == [
        ("reimbursements", "policy:test"),
        ("champions", 71),
        ("source_add", 71),
        ("allocate", 300),
    ]


def test_gateway_annotations_evaluate_on_python_39():
    """PEP 604 unions in runtime-evaluated annotations crash Python 3.9.

    Module-level def annotations evaluate at import; nested-def annotations
    evaluate on every call of the enclosing function (weights.py's
    _chain_snapshot 500'd every /weights/submit during epoch 23918).
    Files opting into `from __future__ import annotations` are exempt.
    """
    import ast
    from pathlib import Path

    gateway_root = Path(__file__).resolve().parents[1] / "gateway"
    offenders = []
    for path in gateway_root.rglob("*.py"):
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
