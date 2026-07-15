"""Static authority gates for the primary and audit validator signing paths."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


def test_primary_validator_uses_only_the_measured_application_signing_helpers():
    called = _called_names(ROOT / "neurons" / "validator.py")
    expected = {
        "push_miner_curation_request",
        "fetch_miner_curation_result",
        "gateway_get_epoch_leads",
        "gateway_submit_validation",
        "gateway_get_fulfillment_reveals",
        "gateway_submit_fulfillment_scores",
    }
    assert expected <= called

    source = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")
    assert source.count("self.wallet.hotkey.sign(") == 1
    assert 'message = str(timestamp).encode()' in source


def test_primary_validator_does_not_reach_deprecated_generic_signers():
    called = _called_names(ROOT / "neurons" / "validator.py")
    assert {
        "save_leads_to_cloud",
        "gateway_get_presigned_url",
        "gateway_verify_submission",
        "gateway_submit_fulfillment_commit",
        "gateway_reveal_fulfillment",
        "calculate_weights",
        "TokenManager",
    }.isdisjoint(called)


def test_auditor_verification_failure_cannot_submit_a_host_fallback_vector():
    source = (ROOT / "neurons" / "auditor_validator.py").read_text(
        encoding="utf-8"
    )
    verification_branch = source.split(
        "if weights_data is None:", 1
    )[1].split("self.trust_level =", 1)[0]
    assert "submit_burn_weights_to_uid0" not in verification_branch
    assert "no fallback vector will be submitted" in verification_branch
